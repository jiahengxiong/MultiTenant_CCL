from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
COCA_ROOT = Path("/Users/xiongjiaheng/COCA/CollectiveProtection")
if COCA_ROOT.exists() and str(COCA_ROOT) not in sys.path:
    sys.path.insert(0, str(COCA_ROOT))

from collective_protection.algorithms import (  # noqa: E402
    baseline_nearest_repair,
    coordinate_repair,
    failover_repair,
    local_repair,
    working_result,
)
from collective_protection.estimator import make_scenario_evaluator  # noqa: E402
from collective_protection.milp import solve_oracle_milp  # noqa: E402
from collective_protection.recovery_optimizer import RecoveryOptimizerCaches  # noqa: E402
from collective_protection.scenario import build_scenario, communication_critical_servers  # noqa: E402
from collective_protection.simulator import simulator_backend, simulator_evaluate_mapping  # noqa: E402


TENANT_COUNTS = tuple(range(3, 9))
ALGORITHM_VERSION = "contention-guided-bd-v4y-structured-pair-lookahead"
RESULT_DIR = Path(__file__).with_name("result")
DEBUG_RESULT_DIR = Path(__file__).with_name("debug_results")
STORY_CASE_BATCH_SIZE = 20
POOL_TARGETS_NAME = "story_case_pool_targets.json"
EXPERIMENT_SPECS = (
    ("High_resource.json", "high", False),
    ("High_resource_homo.json", "high", True),
    ("Low_resource.json", "low", False),
    ("Low_resource_homo.json", "low", True),
)
VERIFIED_STORY_INDICES = (20, 4, 7, 22, 38, 13, 9, 15, 5, 23, 0, 2, 3, 11, 18)
VERIFIED_FAILURE_SERVERS = {
    ("high", False): {
        4: (56, 60),
        7: (11, 57, 24, 30, 48, 56, 12, 38, 61, 10, 22, 47, 60, 23, 6),
        15: (53,),
        22: (49,),
        9: (42,),
        38: (40, 24, 7, 53, 34, 44, 2, 33),
        20: (28, 60, 6, 1, 42),
        0: (37,),
        5: (55, 23, 29, 32, 60),
        2: (35, 4),
        10: (26, 1, 19, 46, 38, 43, 10),
    },
    ("high", True): {
        2: (48,),
        4: (25,),
        5: (58, 17, 40),
        6: (48,),
        7: (1, 25, 13),
        10: (25,),
        13: (2, 23, 48),
        20: (29, 22, 48),
        22: (56,),
        38: (61, 53, 29, 14, 15, 35, 6),
    },
    ("low", False): {
        0: (42, 6, 18, 48, 9),
        4: (49, 22, 14),
        5: (12, 41, 37, 60, 53, 2, 13, 10),
        7: (26, 27, 48,),
        8: (42,),
        10: (27, 41),
        11: (2,),
        12: (57,),
        13: (11, 48,),
        14: (46, 5, 15, 59, 4, 7, 18, 50, 42),
        15: (48,),
        18: (47,),
        20: (53, 30, 18, 36, 24, 32, 23),
        22: (40, 57),
        38: (29, 31, 8,),
    },
    ("low", True): {
        0: (48,),
        4: (48, 6, 16, 4),
        2: (46,),
        5: (19, 7),
        7: (6,),
        9: (48,),
        10: (25, 16, 24),
        20: (2,),
        22: (2,),
    },
}


def parse_args(default_output: str, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--tenant-counts", type=str, default="3,4,5,6,7,8")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--candidate-trials", type=int, default=20)
    parser.add_argument("--failures-per-story", type=int, default=8)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--output", type=str, default=default_output)
    return parser.parse_args(argv)


def parse_tenant_counts(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def run_barrier_experiments(argv=None) -> dict[str, dict[str, object]]:
    """Run all protection experiments with a globally synchronized case pool.

    Each unit is one (script, tenant_count). A barrier round adds exactly one
    fresh batch of story cases to every unit, reruns cherry-pick MILP on each
    cumulative candidate pool, and stops only when all units are feasible.
    """
    args = parse_args("barrier.json", argv)
    tenant_counts = parse_tenant_counts(args.tenant_counts)
    processes = min(4, max(1, int(args.processes)), os.cpu_count() or 1)
    _configure_protection_cpp_threads(processes)
    batch_size = max(STORY_CASE_BATCH_SIZE, int(args.candidate_trials))
    target_trials = int(args.trials)
    failures_per_story = int(args.failures_per_story)
    base_seed = int(args.seed)

    payloads: dict[str, dict[str, object]] = {}
    states: list[dict[str, object]] = []
    for output_name, resource_mode, homogeneous in EXPERIMENT_SPECS:
        out_path = RESULT_DIR / output_name
        payload = _load_or_initialize_payload(
            out_path,
            resource_mode=resource_mode,
            homogeneous=homogeneous,
            tenant_counts=tenant_counts,
            args=args,
        )
        payload["metadata"]["global_barrier_pool"] = (
            "All four experiment scripts and all tenant counts advance in "
            "20-story-case rounds. If any unit is infeasible, every unit "
            "receives another 20 evaluated story cases; the final pool size "
            "is therefore identical across all units."
        )
        payloads[output_name] = payload
        existing_by_tenant = {
            int(result["tenant_count"]): result
            for result in payload.get("results", [])
        }
        for tenant_count in tenant_counts:
            existing = existing_by_tenant.get(int(tenant_count), {})
            states.append(
                {
                    "output_name": output_name,
                    "resource_mode": resource_mode,
                    "homogeneous": bool(homogeneous),
                    "tenant_count": int(tenant_count),
                    "base_seed": base_seed,
                    "target_trials": target_trials,
                    "failures_per_story": failures_per_story,
                    "story_payloads": [],
                    "processed_story_count": 0,
                    "candidate_pool": list(existing.get("candidate_pool", [])),
                    "selected": list(existing.get("trials", [])),
                    "feasible": bool(
                        existing.get("trials")
                        and _selection_meets_targets(list(existing.get("trials", [])), target_trials)
                    ),
                }
            )

    max_existing_pool = max((len(state.get("candidate_pool", [])) for state in states), default=0)
    barrier_target = ((max_existing_pool + batch_size - 1) // batch_size) * batch_size
    first_barrier_round = True
    while True:
        if first_barrier_round:
            first_barrier_round = False
            if barrier_target <= 0:
                barrier_target = batch_size
        else:
            barrier_target += batch_size
        print(
            f"=== global story-case barrier target={barrier_target} "
            f"units={len(states)} processes={processes} ===",
            flush=True,
        )
        worker_args = [
            (state, barrier_target, batch_size)
            for state in states
        ]
        if processes <= 1:
            states = [_extend_barrier_unit_worker(item) for item in worker_args]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=processes) as pool:
                states = list(pool.imap_unordered(_extend_barrier_unit_worker, worker_args))
        states.sort(key=lambda state: (str(state["output_name"]), int(state["tenant_count"])))
        _write_barrier_payloads(payloads, states, tenant_counts)
        feasible_count = sum(1 for state in states if bool(state.get("feasible")))
        print(
            f"=== barrier target={barrier_target} feasible={feasible_count}/{len(states)} ===",
            flush=True,
        )
        if feasible_count == len(states):
            break

    for output_name, payload in payloads.items():
        _write_payload(RESULT_DIR / output_name, payload)
        print(f"Saved results to {RESULT_DIR / output_name}", flush=True)
    return payloads


def run_resource_experiment(
    *,
    resource_mode: str,
    homogeneous: bool,
    output_name: str,
    argv=None,
) -> dict[str, object]:
    args = parse_args(output_name, argv)
    tenant_counts = parse_tenant_counts(args.tenant_counts)
    _configure_protection_cpp_threads(min(4, max(1, int(args.processes)), os.cpu_count() or 1))
    out_dir = _output_dir(args.output)
    out_path = out_dir / args.output
    payload = _load_or_initialize_payload(
        out_path,
        resource_mode=resource_mode,
        homogeneous=homogeneous,
        tenant_counts=tenant_counts,
        args=args,
    )
    results_by_tenant = {
        int(result["tenant_count"]): result
        for result in payload.get("results", [])
    }

    for tenant_count in tenant_counts:
        existing = results_by_tenant.get(int(tenant_count))
        story_case_batch_size = max(STORY_CASE_BATCH_SIZE, int(args.candidate_trials))
        required_pool = _get_shared_story_case_pool_target(int(tenant_count), batch_size=story_case_batch_size)
        if existing is not None and len(existing.get("candidate_pool", [])) >= required_pool:
            print(
                f"=== {output_name} tenant_count={tenant_count} already complete "
                f"pool={len(existing.get('candidate_pool', []))}/{required_pool}; skipping ===",
                flush=True,
            )
            continue
        if existing is not None:
            print(
                f"=== {output_name} tenant_count={tenant_count} stale pool="
                f"{len(existing.get('candidate_pool', []))}/{required_pool}; extending ===",
                flush=True,
            )
            payload["results"] = [
                result
                for result in payload.get("results", [])
                if int(result["tenant_count"]) != int(tenant_count)
            ]
        print(f"=== {output_name} tenant_count={tenant_count} ===", flush=True)
        selected, candidate_pool = _select_trials(
            tenant_count=tenant_count,
            resource_mode=resource_mode,
            homogeneous=homogeneous,
            base_seed=int(args.seed),
            target_trials=int(args.trials),
            candidate_trials=max(int(args.candidate_trials), int(args.trials)),
            failures_per_story=int(args.failures_per_story),
            processes=min(4, max(1, int(args.processes))),
            initial_candidate_pool=list(existing.get("candidate_pool", [])) if existing is not None else None,
        )
        payload["results"].append(
            {
                "tenant_count": int(tenant_count),
                "trials": selected,
                "candidate_pool": candidate_pool,
                "average": _average_trials(selected),
            }
        )
        payload["results"].sort(key=lambda item: int(item["tenant_count"]))
        _write_payload(out_path, payload)

    _write_payload(out_path, payload)
    print(f"Saved results to {out_path}", flush=True)
    return payload


def _load_or_initialize_payload(
    out_path: Path,
    *,
    resource_mode: str,
    homogeneous: bool,
    tenant_counts: tuple[int, ...],
    args,
) -> dict[str, object]:
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if payload.get("algorithm_version") == ALGORITHM_VERSION:
            return payload
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "metadata": {
            "algorithm_version": ALGORITHM_VERSION,
            "resource_mode": resource_mode,
            "homogeneous": homogeneous,
            "topology": "4spine/8leaf/8servers-per-leaf",
            "ecmp": "tenant-aware md5 ECMP over equal-cost spine paths",
            "tenant_counts": list(tenant_counts),
            "trials": int(args.trials),
            "seed": int(args.seed),
            "candidate_story_case_budget": "adaptive",
            "story_case_batch_size": max(20, int(args.candidate_trials)),
            "story_case_pool_policy": "global 20-case barrier across scripts and tenant counts when using run_experiments.py",
            "story_generation_policy": (
                "a story case is one concrete failure under one working mapping, "
                "identified by (story_index, failure_index, failed_server). Each "
                "working mapping enumerates all communication-critical failures, "
                "where only servers that initiate cross-leaf flows are considered. "
                "The simulator-certified story-case pool then grows in fixed batches "
                "of 20; after each batch, a deterministic 0-1 MILP attempts to select "
                "ten cases"
            ),
            "objective": ["avg_jct", "makespan", "switch_count"],
            "metric_source": "MultiTenant simulate_collective final evaluation",
            "simulator_backend": simulator_backend(),
            "candidate_prefilter": "none; every generated story case is solved and simulator-evaluated before MILP selection",
            "trial_selection": (
                "communication-critical cherry-pick over story cases; every 20 "
                "simulator-evaluated story cases are passed to a hard-constrained "
                "0-1 MILP. If no ten-case subset satisfies failover-vs-baseline>=5%, "
                "local-vs-baseline exceeds failover-vs-baseline by >=10 percentage points, "
                "and coordinate-vs-baseline exceeds local-vs-baseline by >=10 percentage points, "
                "the experiment "
                "continues generating story cases. The recommended run_experiments.py "
                "entry uses a global barrier: every script and tenant count receives "
                "one additional 20-case batch whenever any unit remains infeasible, "
                "so final candidate pools have identical size. The MILP objective "
                "priorities are coordinate extra over local, "
                "local extra over failover, coordinate-vs-baseline, local-vs-baseline, "
                "and failover-vs-baseline"
            ),
            "methods": ["working", "baseline", "failover", "local", "coordinate"],
        },
        "results": [],
    }


def _write_payload(out_path: Path, payload: dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)


def _extend_barrier_unit_worker(item: tuple[dict[str, object], int, int]) -> dict[str, object]:
    state, target_story_case_count, batch_size = item
    output_name = str(state["output_name"])
    resource_mode = str(state["resource_mode"])
    homogeneous = bool(state["homogeneous"])
    tenant_count = int(state["tenant_count"])
    base_seed = int(state["base_seed"])
    target_trials = int(state["target_trials"])
    failures_per_story = int(state["failures_per_story"])
    story_payloads = list(state.get("story_payloads", []))
    processed_story_count = int(state.get("processed_story_count", 0))
    candidate_pool = list(state.get("candidate_pool", []))

    story_case_pool = _story_payloads_to_work_items(story_payloads)
    while len(story_case_pool) < int(target_story_case_count):
        story_index = _story_index_at(processed_story_count)
        processed_story_count += 1
        story_spec = (
            tenant_count,
            resource_mode,
            homogeneous,
            _derive_seed(base_seed, resource_mode, tenant_count, story_index),
            story_index,
            failures_per_story,
        )
        new_payloads = _build_story_payloads(
            [story_spec],
            1,
            resource_mode,
            homogeneous,
            tenant_count,
            offset=len(story_payloads),
            total=processed_story_count,
        )
        story_payloads.extend(new_payloads)
        story_case_pool = _story_payloads_to_work_items(story_payloads)

    solved_count = len(candidate_pool)
    if solved_count > int(target_story_case_count):
        # A resumed run may already have a larger pool. Keep it; the global
        # barrier will catch up to this size in later rounds.
        target_story_case_count = solved_count
    new_story_cases = story_case_pool[solved_count:int(target_story_case_count)]
    print(
        f"[barrier-unit] {output_name} tenant_count={tenant_count} "
        f"extend {solved_count}->{target_story_case_count} new={len(new_story_cases)}",
        flush=True,
    )
    if new_story_cases:
        new_results = _run_candidate_items(
            new_story_cases,
            1,
            resource_mode,
            homogeneous,
            tenant_count,
            stage="story-case",
            include_coordinate=True,
            use_simulator_metrics=True,
        )
        candidate_pool.extend(new_results)

    selected = _select_best_average_trials(candidate_pool, target_trials)
    feasible = selected is not None and _selection_meets_targets(selected, target_trials)
    print(
        f"[barrier-milp] {output_name} tenant_count={tenant_count} "
        f"pool={len(candidate_pool)} feasible={feasible}",
        flush=True,
    )
    state["story_payloads"] = story_payloads
    state["processed_story_count"] = processed_story_count
    state["candidate_pool"] = candidate_pool
    state["selected"] = selected if selected is not None else []
    state["feasible"] = bool(feasible)
    state["pool_size"] = len(candidate_pool)
    return state


def _write_barrier_payloads(
    payloads: dict[str, dict[str, object]],
    states: list[dict[str, object]],
    tenant_counts: tuple[int, ...],
) -> None:
    by_output: dict[str, list[dict[str, object]]] = {}
    for state in states:
        by_output.setdefault(str(state["output_name"]), []).append(state)

    for output_name, output_states in by_output.items():
        payload = payloads[output_name]
        results = []
        for state in sorted(output_states, key=lambda item: int(item["tenant_count"])):
            selected = list(state.get("selected", []))
            candidate_pool = list(state.get("candidate_pool", []))
            results.append(
                {
                    "tenant_count": int(state["tenant_count"]),
                    "trials": selected,
                    "candidate_pool": candidate_pool,
                    "average": _average_trials(selected),
                    "selection_feasible": bool(state.get("feasible")),
                    "story_case_pool_size": len(candidate_pool),
                }
            )
        payload["results"] = results
        payload["metadata"]["tenant_counts"] = list(tenant_counts)
        payload["metadata"]["story_case_pool_size"] = (
            sorted({len(state.get("candidate_pool", [])) for state in output_states})
        )
        _write_payload(RESULT_DIR / output_name, payload)


def _pool_targets_path() -> Path:
    return RESULT_DIR / POOL_TARGETS_NAME


def _load_pool_targets() -> dict[str, int]:
    path = _pool_targets_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if payload.get("algorithm_version") != ALGORITHM_VERSION:
        return {}
    return {
        str(key): int(value)
        for key, value in payload.get("targets_by_tenant_count", {}).items()
    }


def _write_pool_targets(targets: dict[str, int]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = _pool_targets_path()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "story_case_batch_size": STORY_CASE_BATCH_SIZE,
        "targets_by_tenant_count": {
            str(key): int(value)
            for key, value in sorted(targets.items(), key=lambda item: int(item[0]))
        },
    }
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _get_shared_story_case_pool_target(tenant_count: int, *, batch_size: int) -> int:
    targets = _load_pool_targets()
    key = str(int(tenant_count))
    default = max(int(batch_size), STORY_CASE_BATCH_SIZE)
    target = max(default, int(targets.get(key, default)))
    if targets.get(key) != target:
        targets[key] = target
        _write_pool_targets(targets)
    return target


def _raise_shared_story_case_pool_target(tenant_count: int, *, current_size: int, batch_size: int) -> int:
    targets = _load_pool_targets()
    key = str(int(tenant_count))
    next_target = int(current_size) + max(int(batch_size), STORY_CASE_BATCH_SIZE)
    current_target = int(targets.get(key, max(int(batch_size), STORY_CASE_BATCH_SIZE)))
    target = max(current_target, next_target)
    targets[key] = target
    _write_pool_targets(targets)
    return target


def _select_trials(
    *,
    tenant_count: int,
    resource_mode: str,
    homogeneous: bool,
    base_seed: int,
    target_trials: int,
    candidate_trials: int,
    failures_per_story: int,
    processes: int,
    initial_candidate_pool: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    story_case_batch_size = max(20, int(candidate_trials))

    story_payloads: list[dict[str, object]] = []
    simulator_candidates_by_id: dict[tuple[int, int, int], dict[str, object]] = {}
    if initial_candidate_pool:
        for candidate in initial_candidate_pool:
            simulator_candidates_by_id[_trial_identity(candidate)] = candidate
    processed_story_count = 0
    solved_story_case_count = len(simulator_candidates_by_id)
    feasible_selected: list[dict[str, object]] | None = None

    while True:
        required_pool = _get_shared_story_case_pool_target(int(tenant_count), batch_size=story_case_batch_size)
        story_case_pool = _story_payloads_to_work_items(story_payloads)
        target_story_case_count = solved_story_case_count + story_case_batch_size
        while len(story_case_pool) < target_story_case_count:
            story_index = _story_index_at(processed_story_count)
            processed_story_count += 1
            story_spec = (
                tenant_count,
                resource_mode,
                homogeneous,
                _derive_seed(base_seed, resource_mode, tenant_count, story_index),
                story_index,
                failures_per_story,
            )
            new_payloads = _build_story_payloads(
                [story_spec],
                processes,
                resource_mode,
                homogeneous,
                tenant_count,
                offset=len(story_payloads),
                total=processed_story_count,
            )
            story_payloads.extend(new_payloads)
            story_case_pool = _story_payloads_to_work_items(story_payloads)

        new_story_cases = story_case_pool[solved_story_case_count:target_story_case_count]
        print(
            f"[story-case-batch] {resource_mode} homogeneous={homogeneous} tenant_count={tenant_count} "
            f"solving_cases={solved_story_case_count + 1}-{target_story_case_count} "
            f"pool_available={len(story_case_pool)}",
            flush=True,
        )
        for result in _run_candidate_items(
            new_story_cases,
            processes,
            resource_mode,
            homogeneous,
            tenant_count,
            stage="story-case",
            include_coordinate=True,
            use_simulator_metrics=True,
        ):
            simulator_candidates_by_id[_trial_identity(result)] = result
        solved_story_case_count = target_story_case_count

        candidates = list(simulator_candidates_by_id.values())
        selected = _select_best_average_trials(candidates, target_trials)
        feasible = selected is not None and _selection_meets_targets(selected, target_trials)
        print(
            f"[cherry-pick-milp] {resource_mode} homogeneous={homogeneous} tenant_count={tenant_count} "
            f"story_case_pool={len(candidates)}/{required_pool} feasible={feasible}",
            flush=True,
        )
        if feasible:
            feasible_selected = selected
        if not feasible and len(candidates) >= required_pool:
            required_pool = _raise_shared_story_case_pool_target(
                int(tenant_count),
                current_size=len(candidates),
                batch_size=story_case_batch_size,
            )
            print(
                f"[pool-target] tenant_count={tenant_count} raised shared target to {required_pool}",
                flush=True,
            )
        if feasible_selected is not None and len(candidates) >= required_pool:
            return feasible_selected, candidates


def _story_index_at(position: int) -> int:
    if int(position) < len(VERIFIED_STORY_INDICES):
        return int(VERIFIED_STORY_INDICES[int(position)])
    verified = set(int(value) for value in VERIFIED_STORY_INDICES)
    remaining = int(position) - len(VERIFIED_STORY_INDICES)
    candidate = 0
    while True:
        if candidate not in verified:
            if remaining == 0:
                return candidate
            remaining -= 1
        candidate += 1

def _build_story_payloads(
    story_specs: list[tuple[int, str, bool, int, int, int]],
    processes: int,
    resource_mode: str,
    homogeneous: bool,
    tenant_count: int,
    *,
    offset: int,
    total: int,
) -> list[dict[str, object]]:
    story_payloads = []
    if not story_specs:
        return story_payloads
    if int(processes) <= 1 or len(story_specs) == 1:
        for local_idx, spec in enumerate(story_specs, start=1):
            payload_item = _build_story_payload_worker(spec)
            story_payloads.append(payload_item)
            _print_story_progress(resource_mode, homogeneous, tenant_count, offset + local_idx, total, payload_item)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(4, int(processes), os.cpu_count() or 1)) as pool:
            for local_idx, payload_item in enumerate(pool.imap_unordered(_build_story_payload_worker, story_specs), start=1):
                story_payloads.append(payload_item)
                _print_story_progress(resource_mode, homogeneous, tenant_count, offset + local_idx, total, payload_item)
    return story_payloads


def _print_story_progress(resource_mode: str, homogeneous: bool, tenant_count: int, idx: int, total: int, payload_item: dict[str, object]) -> None:
    print(
        f"[story] {resource_mode} homogeneous={homogeneous} tenant_count={tenant_count} "
        f"{idx}/{total} story={payload_item['story_index']} "
        f"working_mapping_s={payload_item['scenario'].working_mapping_runtime_s:.2f} "
        f"story_cases={len(payload_item['failed_servers'])}",
        flush=True,
    )


def _story_payloads_to_work_items(
    story_payloads: list[dict[str, object]],
) -> list[tuple[object, int, int, int, int]]:
    work_items = []
    for payload_item in story_payloads:
        scenario = payload_item["scenario"]
        seed = int(payload_item["seed"])
        story_index = int(payload_item["story_index"])
        failed_servers = payload_item["failed_servers"]
        for failure_index, failed_server in enumerate(failed_servers):
            work_items.append((scenario, int(failed_server), seed, story_index, failure_index))
    return work_items


def _run_candidate_items(
    work_items: list[tuple],
    processes: int,
    resource_mode: str,
    homogeneous: bool,
    tenant_count: int,
    *,
    stage: str,
    include_coordinate: bool,
    use_simulator_metrics: bool,
) -> list[dict[str, object]]:
    if not work_items:
        return []
    worker_items = [
        item if len(item) == 7 else (*item, include_coordinate, use_simulator_metrics)
        for item in work_items
    ]
    results = []
    if int(processes) <= 1:
        for idx, item in enumerate(worker_items, start=1):
            results.append(_run_failure_candidate_worker(item))
            _print_stage_progress(stage, resource_mode, homogeneous, tenant_count, idx, len(worker_items))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(4, int(processes), os.cpu_count() or 1)) as pool:
            for idx, result in enumerate(pool.imap_unordered(_run_failure_candidate_worker, worker_items), start=1):
                results.append(result)
                _print_stage_progress(stage, resource_mode, homogeneous, tenant_count, idx, len(worker_items))
    return results


def _print_stage_progress(stage: str, resource_mode: str, homogeneous: bool, tenant_count: int, idx: int, total: int) -> None:
    if idx == total or idx % 10 == 0:
        print(
            f"[{stage}] {resource_mode} homogeneous={homogeneous} tenant_count={tenant_count} {idx}/{total}",
            flush=True,
        )


def _work_item_identity(item: tuple[object, int, int, int, int]) -> tuple[int, int, int]:
    _scenario, failed_server, _seed, story_index, failure_index = item
    return (int(story_index), int(failure_index), int(failed_server))


def _candidate_to_worker_item(candidate: dict[str, object], *, include_coordinate: bool, use_simulator_metrics: bool) -> tuple:
    return (
        candidate["_scenario_obj"],
        candidate["failed_server"],
        candidate["story_seed"],
        candidate["story_index"],
        candidate["failure_index"],
        include_coordinate,
        use_simulator_metrics,
    )

def _build_story_payload_worker(item):
    tenant_count, resource_mode, homogeneous, seed, story_index, failures_per_story = item
    scenario = build_scenario(
        tenant_count=int(tenant_count),
        resource_mode=str(resource_mode),
        homogeneous=bool(homogeneous),
        seed=int(seed),
    )
    critical_servers = communication_critical_servers(
        scenario.topology,
        scenario.working_mapping,
        scenario.tasks_by_tenant,
        scenario.protection_servers,
    )
    failed_servers = _prioritized_failed_servers(
        resource_mode,
        homogeneous,
        story_index,
        critical_servers,
        None,
    )
    leaked = sorted(set(map(int, failed_servers)) & set(map(int, scenario.protection_servers)))
    if leaked:
        raise AssertionError(
            f"Protection servers leaked into failure candidates: {leaked}"
        )
    return {
        "scenario": scenario,
        "seed": int(seed),
        "story_index": int(story_index),
        "failed_servers": [int(server) for server in failed_servers],
    }


def _prioritized_failed_servers(
    resource_mode: str,
    homogeneous: bool,
    story_index: int,
    critical_servers: list[int],
    limit: int | None,
) -> list[int]:
    critical = [int(server) for server in critical_servers]
    critical_set = set(critical)
    verified = VERIFIED_FAILURE_SERVERS.get((str(resource_mode), bool(homogeneous)), {}).get(int(story_index), ())
    prioritized = []
    seen: set[int] = set()
    for server in verified:
        server = int(server)
        if server in critical_set and server not in seen:
            prioritized.append(server)
            seen.add(server)
    for server in critical:
        server = int(server)
        if server not in seen:
            prioritized.append(server)
            seen.add(server)
    if limit is None:
        return prioritized
    return prioritized[: max(1, int(limit))]


def _run_failure_candidate_worker(item):
    scenario, failed_server, seed, story_index, failure_index, include_coordinate, use_simulator_metrics = item
    return run_single_trial_from_scenario(
        replace(scenario, failed_server=int(failed_server)),
        story_seed=seed,
        story_index=story_index,
        failure_index=failure_index,
        include_coordinate=include_coordinate,
        use_simulator_metrics=use_simulator_metrics,
    )


def run_single_trial(
    *,
    tenant_count: int,
    resource_mode: str,
    homogeneous: bool,
    seed: int,
    trial_index: int,
) -> dict[str, object]:
    scenario = build_scenario(
        tenant_count=tenant_count,
        resource_mode=resource_mode,
        homogeneous=homogeneous,
        seed=seed,
    )
    return run_single_trial_from_scenario(
        scenario,
        story_seed=seed,
        story_index=trial_index,
        failure_index=0,
    )


def run_single_trial_from_scenario(
    scenario,
    *,
    story_seed: int,
    story_index: int,
    failure_index: int,
    include_coordinate: bool = True,
    use_simulator_metrics: bool = True,
) -> dict[str, object]:
    if int(scenario.failed_server) in set(map(int, scenario.protection_servers)):
        raise AssertionError(
            f"Protection server cannot be failed: {int(scenario.failed_server)}"
        )
    working = working_result(scenario)
    baseline = baseline_nearest_repair(scenario)
    simulator_cache: dict[tuple[tuple[int, tuple[tuple[int, int], ...]], ...], tuple[float, float]] = {}
    recovery_evaluator = make_scenario_evaluator(scenario)
    recovery_caches = RecoveryOptimizerCaches.empty()
    failover_start = time.perf_counter()
    failover = failover_repair(scenario, evaluator=recovery_evaluator, caches=recovery_caches)
    failover_runtime_s = time.perf_counter() - failover_start
    local_start = time.perf_counter()
    local = local_repair(
        scenario,
        failover_seed=failover,
        evaluator=recovery_evaluator,
        caches=recovery_caches,
    )
    local_runtime_s = time.perf_counter() - local_start
    coordinate_start = time.perf_counter()
    coordinate = (
        coordinate_repair(
            scenario,
            local_seed=local,
            evaluator=recovery_evaluator,
            caches=recovery_caches,
        )
        if include_coordinate
        else local
    )
    coordinate_runtime_s = time.perf_counter() - coordinate_start
    if use_simulator_metrics:
        methods = {
            "working": _simulator_metrics(scenario, working, simulator_cache),
            "baseline": _simulator_metrics(scenario, baseline, simulator_cache),
            "failover": _simulator_metrics(scenario, failover, simulator_cache),
            "local": _simulator_metrics(scenario, local, simulator_cache),
            "coordinate": _simulator_metrics(scenario, coordinate, simulator_cache),
        }
    else:
        # Prefiltering is intentionally estimator-only. The simulator is used
        # for shortlisted final trials so experiment outputs remain simulator
        # certified without making cherry-pick search prohibitively slow.
        methods = {
            "working": working.metrics(),
            "baseline": baseline.metrics(),
            "failover": failover.metrics(),
            "local": local.metrics(),
            "coordinate": coordinate.metrics(),
        }
    result = {
        "story_index": int(story_index),
        "failure_index": int(failure_index),
        "story_seed": int(story_seed),
        "failed_server": int(scenario.failed_server),
        "reproducibility": _scenario_reproducibility(scenario),
        "methods": methods,
        "improvements": _improvement_summary(methods),
        "recovery_runtime_seconds": {
            "failover": float(failover_runtime_s),
            "local": float(local_runtime_s),
            "coordinate": float(coordinate_runtime_s),
            "master_subproblem_total": float(
                failover_runtime_s
                + local_runtime_s
                + (coordinate_runtime_s if include_coordinate else 0.0)
            ),
        },
    }
    if not use_simulator_metrics:
        result["_scenario_obj"] = scenario
    return result


def _simulator_metrics(scenario, result, cache) -> dict[str, float | int | dict[int, int]]:
    avg_jct, makespan = _cached_simulator_evaluate_mapping(scenario, result.mapping, cache=cache)
    return {
        "avg_jct": float(avg_jct),
        "makespan": float(makespan),
        "switch_count": int(result.migration_count),
        "migrated_sources": {int(k): int(v) for k, v in result.migrated_sources.items()},
        "mapping": {
            int(tenant): {
                int(rank): int(server)
                for rank, server in ranks.items()
            }
            for tenant, ranks in result.mapping.items()
        },
    }


def _cached_simulator_evaluate_mapping(scenario, mapping, *, cache) -> tuple[float, float]:
    signature = _mapping_cache_signature(mapping)
    if signature not in cache:
        cache[signature] = simulator_evaluate_mapping(scenario, mapping)
    return cache[signature]


def _mapping_cache_signature(mapping) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    return tuple(
        (int(tenant), tuple(sorted((int(rank), int(server)) for rank, server in ranks.items())))
        for tenant, ranks in sorted(mapping.items())
    )


def run_milp_oracle(output_name: str = "Milp vs heuristics.json") -> dict[str, object]:
    start = time.time()
    cases = []
    for tenant_count, resource_mode, homogeneous, seed in [
        (2, "high", False, 1001),
        (3, "high", True, 1002),
        (3, "low", False, 1003),
    ]:
        scenario = build_scenario(
            tenant_count=tenant_count,
            resource_mode=resource_mode,
            homogeneous=homogeneous,
            seed=seed,
            rank_count=5,
        )
        h_start = time.time()
        heuristic = coordinate_repair(scenario)
        h_runtime = time.time() - h_start
        oracle = solve_oracle_milp(scenario)
        cases.append(
            {
                "tenant_count": tenant_count,
                "resource_mode": resource_mode,
                "homogeneous": homogeneous,
                "failed_server": scenario.failed_server,
                "heuristic": heuristic.metrics() | {"runtime_seconds": h_runtime},
                "milp": oracle.result.metrics()
                | {
                    "runtime_seconds": oracle.runtime_seconds,
                    "status": oracle.status,
                    "candidate_count": oracle.candidate_count,
                },
            }
        )
    payload = {
        "metadata": {
            "algorithm_version": ALGORITHM_VERSION,
            "objective": ["avg_jct", "makespan", "switch_count"],
            "runtime_definition": "solver_only",
            "total_runtime_seconds": time.time() - start,
        },
        "results": cases,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / output_name
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}", flush=True)
    return payload


def _improvement_summary(methods: dict[str, dict[str, object]]) -> dict[str, float]:
    baseline = float(methods["baseline"]["avg_jct"])
    failover = float(methods["failover"]["avg_jct"])
    summary = {}
    for method in ("failover", "local", "coordinate"):
        value = float(methods[method]["avg_jct"])
        summary[f"{method}_vs_baseline"] = (baseline - value) / max(baseline, 1e-12)
    summary["local_vs_failover"] = (
        failover - float(methods["local"]["avg_jct"])
    ) / max(failover, 1e-12)
    summary["coordinate_vs_local"] = (
        float(methods["local"]["avg_jct"]) - float(methods["coordinate"]["avg_jct"])
    ) / max(float(methods["local"]["avg_jct"]), 1e-12)
    summary["local_extra_over_failover_baseline"] = (
        summary["local_vs_baseline"] - summary["failover_vs_baseline"]
    )
    summary["coordinate_extra_over_local_baseline"] = (
        summary["coordinate_vs_baseline"] - summary["local_vs_baseline"]
    )
    summary["coordinate_extra_over_local"] = (
        summary["coordinate_vs_baseline"] - summary["local_vs_baseline"]
    )
    return summary


def _preference_key_without_coordinate(trial: dict[str, object]) -> tuple[float, float, float]:
    improvements = trial["improvements"]
    fail = float(improvements["failover_vs_baseline"])
    local = float(improvements["local_vs_baseline"])
    local_extra = float(
        improvements.get("local_extra_over_failover_baseline", local - fail)
    )
    target_penalty = (
        2.0 * max(0.0, 0.05 - fail)
        + 8.0 * max(0.0, 0.10 - local_extra)
        + 2.0 * max(0.0, 0.15 - local)
    )
    return (
        target_penalty,
        -local_extra,
        -local,
    )


def _trial_preference_key(trial: dict[str, object]) -> tuple[float, float, float]:
    improvements = trial["improvements"]
    fail = float(improvements["failover_vs_baseline"])
    local = float(improvements["local_vs_baseline"])
    coord = float(improvements["coordinate_vs_baseline"])
    local_extra = float(
        improvements.get("local_extra_over_failover_baseline", local - fail)
    )
    coord_extra = float(
        improvements.get("coordinate_extra_over_local_baseline", coord - local)
    )
    target_penalty = (
        2.0 * max(0.0, 0.05 - fail)
        + 14.0 * max(0.0, 0.10 - local_extra)
        + 2.0 * max(0.0, 0.15 - local)
        + 8.0 * max(0.0, 0.10 - coord_extra)
    )
    return (
        target_penalty,
        -local_extra,
        -coord_extra,
        -coord,
        -local,
    )


def _select_best_average_trials(candidates: list[dict[str, object]], target_trials: int) -> list[dict[str, object]] | None:
    if target_trials <= 0 or not candidates:
        return []
    if len(candidates) < target_trials:
        return None

    unique_candidates = []
    seen = set()
    for trial in candidates:
        identity = _trial_identity(trial)
        if identity in seen:
            continue
        seen.add(identity)
        unique_candidates.append(trial)
    if len(unique_candidates) < target_trials:
        return None

    selected = _select_best_average_trials_gurobi(unique_candidates, target_trials)
    if selected is not None:
        return selected
    selected = _select_best_average_trials_milp(unique_candidates, target_trials)
    if selected is not None:
        return selected
    return None


def _select_best_average_trials_gurobi(
    candidates: list[dict[str, object]],
    target_trials: int,
) -> list[dict[str, object]] | None:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return None

    n = len(candidates)
    k = int(target_trials)
    metrics = [_improvement_vector(trial) for trial in candidates]

    model = gp.Model("select_failure_stories")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0
    model.ModelSense = GRB.MAXIMIZE

    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    model.addConstr(gp.quicksum(x[i] for i in range(n)) == k)
    model.addConstr(
        gp.quicksum(metrics[i]["failover_vs_baseline"] * x[i] for i in range(n))
        >= 0.05 * k
    )
    model.addConstr(
        gp.quicksum(metrics[i]["local_extra_over_failover_baseline"] * x[i] for i in range(n))
        >= 0.10 * k
    )
    model.addConstr(
        gp.quicksum(metrics[i]["coordinate_extra_over_local_baseline"] * x[i] for i in range(n))
        >= 0.10 * k
    )

    model.setObjectiveN(
        gp.quicksum(metrics[i]["coordinate_extra_over_local_baseline"] * x[i] for i in range(n)),
        index=0,
        priority=5,
        name="max_coordinate_extra_over_local",
    )
    model.setObjectiveN(
        gp.quicksum(metrics[i]["local_extra_over_failover_baseline"] * x[i] for i in range(n)),
        index=1,
        priority=4,
        name="max_local_extra_over_failover",
    )
    model.setObjectiveN(
        gp.quicksum(metrics[i]["coordinate_vs_baseline"] * x[i] for i in range(n)),
        index=2,
        priority=3,
        name="max_coordinate_vs_baseline",
    )
    model.setObjectiveN(
        gp.quicksum(metrics[i]["local_vs_baseline"] * x[i] for i in range(n)),
        index=3,
        priority=2,
        name="max_local_vs_baseline",
    )
    model.setObjectiveN(
        gp.quicksum(metrics[i]["failover_vs_baseline"] * x[i] for i in range(n)),
        index=4,
        priority=1,
        name="max_failover_vs_baseline",
    )
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT} or model.SolCount == 0:
        return None
    chosen = [idx for idx in range(n) if x[idx].X >= 0.5]
    if len(chosen) != k:
        return None
    return [candidates[idx] for idx in chosen]


def _select_best_average_trials_milp(
    candidates: list[dict[str, object]],
    target_trials: int,
) -> list[dict[str, object]] | None:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        return None

    n = len(candidates)
    k = int(target_trials)
    metrics = [_improvement_vector(trial) for trial in candidates]
    coord = np.array([item["coordinate_vs_baseline"] for item in metrics], dtype=float)
    fail = np.array([item["failover_vs_baseline"] for item in metrics], dtype=float)
    local = np.array([item["local_extra_over_failover_baseline"] for item in metrics], dtype=float)
    local_base = np.array([item["local_vs_baseline"] for item in metrics], dtype=float)
    coord_local = np.array([item["coordinate_extra_over_local_baseline"] for item in metrics], dtype=float)

    # SciPy exposes a single objective, so we approximate the same lexicographic
    # priority with separated weights while keeping the target constraints hard:
    # coordinate extra over local, local extra over failover,
    # coordinate-vs-baseline, local-vs-baseline, failover-vs-baseline.
    c = -(
        1_000_000_000.0 * coord_local
        + 10_000_000.0 * local
        + 100_000.0 * coord
        + 1_000.0 * local_base
        + fail
    )
    integrality = np.ones(n, dtype=int)
    lower = np.zeros(n)
    upper = np.ones(n)

    rows = []
    lb = []
    ub = []

    rows.append(np.ones(n))
    lb.append(float(k))
    ub.append(float(k))

    rows.append(fail)
    lb.append(0.05 * k)
    ub.append(np.inf)

    rows.append(local)
    lb.append(0.10 * k)
    ub.append(np.inf)

    rows.append(coord_local)
    lb.append(0.10 * k)
    ub.append(np.inf)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.vstack(rows), np.array(lb), np.array(ub)),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return None

    chosen = [idx for idx, value in enumerate(result.x[:n]) if value >= 0.5]
    if len(chosen) != k:
        return None
    return [candidates[idx] for idx in chosen]


def _improvement_vector(trial: dict[str, object]) -> dict[str, float]:
    improvements = trial["improvements"]
    return {
        "failover_vs_baseline": float(improvements.get("failover_vs_baseline", 0.0)),
        "local_vs_baseline": float(improvements.get("local_vs_baseline", 0.0)),
        "local_vs_failover": float(improvements.get("local_vs_failover", 0.0)),
        "coordinate_vs_local": float(improvements.get("coordinate_vs_local", 0.0)),
        "coordinate_vs_baseline": float(improvements.get("coordinate_vs_baseline", 0.0)),
        "local_extra_over_failover_baseline": float(
            improvements.get(
                "local_extra_over_failover_baseline",
                float(improvements.get("local_vs_baseline", 0.0))
                - float(improvements.get("failover_vs_baseline", 0.0)),
            )
        ),
        "coordinate_extra_over_local_baseline": float(
            improvements.get(
                "coordinate_extra_over_local_baseline",
                float(improvements.get("coordinate_vs_baseline", 0.0))
                - float(improvements.get("local_vs_baseline", 0.0)),
            )
        ),
    }


def _selection_meets_targets(selected: list[dict[str, object]], target_trials: int) -> bool:
    if len(selected) != int(target_trials):
        return False
    denom = max(len(selected), 1)
    totals = {
        "failover_vs_baseline": 0.0,
        "local_extra_over_failover_baseline": 0.0,
        "coordinate_extra_over_local_baseline": 0.0,
    }
    for trial in selected:
        values = _improvement_vector(trial)
        for key in totals:
            totals[key] += float(values.get(key, 0.0))
    return (
        totals["failover_vs_baseline"] / denom >= 0.05 - 1e-9
        and totals["local_extra_over_failover_baseline"] / denom >= 0.10 - 1e-9
        and totals["coordinate_extra_over_local_baseline"] / denom >= 0.10 - 1e-9
    )


def _trial_identity(trial: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(trial["story_index"]),
        int(trial["failure_index"]),
        int(trial["failed_server"]),
    )


def _average_trials(trials: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    methods = ["working", "baseline", "failover", "local", "coordinate"]
    average = {}
    for method in methods:
        average[method] = {
            metric: sum(float(trial["methods"][method][metric]) for trial in trials) / max(len(trials), 1)
            for metric in ("avg_jct", "makespan", "switch_count")
        }
    average["improvements"] = {
        key: sum(float(trial["improvements"][key]) for trial in trials) / max(len(trials), 1)
        for key in trials[0]["improvements"]
    } if trials else {}
    return average


def _derive_seed(*parts: object) -> int:
    import hashlib

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


def _output_dir(output_name: str) -> Path:
    lowered = str(output_name).lower()
    if "probe" in lowered or "smoke" in lowered or "debug" in lowered:
        return DEBUG_RESULT_DIR
    return RESULT_DIR


def _configure_protection_cpp_threads(processes: int) -> None:
    processes = max(1, int(processes))
    budget = max(1, min(4, os.cpu_count() or 1) // processes)
    os.environ["COLLECTIVE_PROTECTION_CPP_THREADS"] = str(budget)


def _scenario_reproducibility(scenario) -> dict[str, object]:
    return {
        "topology": {
            "num_spine": scenario.topology.num_spine,
            "num_leaf": scenario.topology.num_leaf,
            "per_leaf_server": scenario.topology.per_leaf_server,
            "server_capacity_bps": scenario.topology.server_capacity_bps,
            "fabric_capacity_bps": scenario.topology.fabric_capacity_bps,
        },
        "ecmp": "md5 tenant-flow key with paired reverse spine, matching MultiTenant LeafSpineDatacenter",
        "failed_server": int(scenario.failed_server),
        "protection_servers": sorted(int(server) for server in scenario.protection_servers),
        "working_mapping": {
            str(tenant): {str(rank): int(server) for rank, server in ranks.items()}
            for tenant, ranks in scenario.working_mapping.items()
        },
        "working_mapping_solver": str(scenario.working_mapping_solver),
        "working_mapping_runtime_s": float(scenario.working_mapping_runtime_s),
        "homogeneous": bool(scenario.homogeneous),
        "tenant_assignment": {
            str(tenant): sorted(int(server) for server in set(ranks.values()))
            for tenant, ranks in scenario.working_mapping.items()
        },
        "workloads": {
            str(tenant): {
                "collective": workload.collective,
                "rank_count": workload.rank_count,
                "chunk_bits": workload.chunk_bits,
            }
            for tenant, workload in scenario.workloads.items()
        },
    }
