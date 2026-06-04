from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.baselines import build_leaf_local_mapping  # noqa: E402
from multitenant.simulator import simulate_collective  # noqa: E402
from multitenant.solvers import (  # noqa: E402
    MappingEstimatorBlackBoxOptimizer,
    MappingHybridHeuristicSolver,
)
from multitenant.topology import LeafSpineDatacenter  # noqa: E402


DEFAULT_INPUTS = (
    REPO_ROOT / "experiment" / "Low_contension.json",
    REPO_ROOT / "experiment" / "High_contension.json",
    REPO_ROOT / "experiment" / "Low_contension_homo.json",
    REPO_ROOT / "experiment" / "High_contension_homo.json",
)

Mapping = dict[int, dict[int, int]]
Specs = dict[int, dict[str, Any]]
Score = tuple[float, float]  # (avg_jct, makespan), simulator-facing order.


def parse_int_set(raw: str) -> set[int]:
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def denormalize_mapping(raw: dict[str, dict[str, int]]) -> Mapping:
    return {
        int(tenant): {
            int(rank): int(server)
            for rank, server in rank_to_server.items()
        }
        for tenant, rank_to_server in raw.items()
    }


def normalize_mapping(mapping: Mapping) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {
            str(rank): int(server)
            for rank, server in rank_to_server.items()
        }
        for tenant, rank_to_server in mapping.items()
    }


def denormalize_specs(raw: dict[str, dict[str, Any]]) -> Specs:
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def build_datacenter(metadata: dict[str, Any]) -> LeafSpineDatacenter:
    topology = metadata["topology"]
    return LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )


def evaluate_with_simulator(
    datacenter: LeafSpineDatacenter,
    mapping: Mapping,
    specs: Specs,
) -> Score:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return float(avg_jct), float(makespan)


def solver_path_table(datacenter: LeafSpineDatacenter, initial_mapping: Mapping):
    """Build the fixed tenant-aware ECMP table used during optimization.

    The values of ``initial_mapping`` are dictionaries, so the topology helper
    uses the tenant id as the fixed tenant port/flow key.  This matches the
    paper assumption: ports are tenant-specific and fixed during rank mapping.
    """

    return datacenter.build_tenant_ecmp_path_table(initial_mapping)


def run_solver_live(
    solver_cls,
    datacenter: LeafSpineDatacenter,
    initial_mapping: Mapping,
    specs: Specs,
    *,
    time_limit: float | None,
    label: str,
) -> tuple[Mapping, float, dict[str, Any]]:
    solver = solver_cls(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=solver_path_table(datacenter, initial_mapping),
        verbose=False,
    )
    print(f"    solving {label} (time_limit={time_limit})", flush=True)
    start = time.time()
    if time_limit is None:
        solver.solve()
    else:
        solver.solve(time_limit=float(time_limit))
    runtime = time.time() - start
    print(f"    finished {label} in {runtime:.2f}s", flush=True)
    mapping = solver.get_X_mapping()
    metadata = {
        "class": f"{solver_cls.__module__}.{solver_cls.__name__}",
        "surrogate_mode": getattr(solver, "surrogate_mode", None),
        "final_avg_jct": getattr(solver, "final_avg_jct", None),
        "final_makespan": getattr(solver, "final_makespan", None),
        "final_score_is_simulated": bool(getattr(solver, "final_score_is_simulated", False)),
        "search_rounds": getattr(solver, "search_rounds", None),
        "move_source_counts": dict(getattr(solver, "move_source_counts", {})),
        "line_search_evaluations": getattr(solver, "line_search_evaluations", None),
        "pass_evaluations": getattr(solver, "pass_evaluations", None),
        "runtime_seconds": float(runtime),
    }
    return mapping, float(runtime), metadata


def score_dict(score: Score) -> dict[str, float]:
    return {
        "avg_jct": float(score[0]),
        "makespan": float(score[1]),
    }


def relative_gap(candidate: Score, baseline: Score) -> dict[str, float]:
    return {
        "avg_jct": (float(candidate[0]) - float(baseline[0])) / max(float(baseline[0]), 1e-12),
        "makespan": (float(candidate[1]) - float(baseline[1])) / max(float(baseline[1]), 1e-12),
    }


def mapping_diff_by_tenant(left: Mapping, right: Mapping) -> dict[str, int]:
    return {
        str(tenant): sum(
            1
            for rank, server in left[int(tenant)].items()
            if int(right[int(tenant)][int(rank)]) != int(server)
        )
        for tenant in sorted(left)
    }


def iter_cases(paths: list[Path], tenant_counts: set[int], trial_indices: set[int]):
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        for group in payload["results"]:
            tenant_count = int(group["tenant_count"])
            if tenant_count not in tenant_counts:
                continue
            for trial in group["trial_results"]:
                trial_index = int(trial["trial_index"])
                if trial_index not in trial_indices:
                    continue
                yield {
                    "path": str(path),
                    "metadata": metadata,
                    "tenant_count": tenant_count,
                    "trial": trial,
                }


def run_case(task: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(task["path"]))
    metadata = dict(task["metadata"])
    tenant_count = int(task["tenant_count"])
    trial = dict(task["trial"])
    trial_index = int(trial["trial_index"])
    sensitivity = float(task.get("sensitivity", 0.02))

    print(
        f"[live-compare pid={os.getpid()}] {path.name} "
        f"scene={metadata.get('scene', path.stem)} "
        f"tenant_count={tenant_count} trial={trial_index + 1}",
        flush=True,
    )

    # JSON is only a scenario container.  We intentionally ignore stored
    # proposed_mapping/results because those may have been produced by older
    # code paths.
    datacenter = build_datacenter(metadata)
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    specs = denormalize_specs(trial["tenant_collective_specs"])
    locality_mapping = build_leaf_local_mapping(initial_mapping)

    default_score = evaluate_with_simulator(datacenter, initial_mapping, specs)
    locality_score = evaluate_with_simulator(datacenter, locality_mapping, specs)

    hybrid_mapping, hybrid_runtime, hybrid_meta = run_solver_live(
        MappingHybridHeuristicSolver,
        datacenter,
        initial_mapping,
        specs,
        time_limit=task.get("hybrid_time_limit"),
        label="collapsed-hybrid",
    )
    hybrid_score = evaluate_with_simulator(datacenter, hybrid_mapping, specs)

    time_expanded_mapping, time_expanded_runtime, time_expanded_meta = run_solver_live(
        MappingEstimatorBlackBoxOptimizer,
        datacenter,
        initial_mapping,
        specs,
        time_limit=task.get("time_expanded_time_limit"),
        label="time-expanded",
    )
    time_expanded_score = evaluate_with_simulator(datacenter, time_expanded_mapping, specs)

    gap_vs_hybrid = relative_gap(time_expanded_score, hybrid_score)
    pass_avg = gap_vs_hybrid["avg_jct"] <= sensitivity
    pass_lexicographic = (
        gap_vs_hybrid["avg_jct"] <= sensitivity
        and (
            gap_vs_hybrid["avg_jct"] < -1e-12
            or gap_vs_hybrid["makespan"] <= sensitivity
        )
    )

    print(
        "    default={:.6f}/{:.6f}, locality={:.6f}/{:.6f}, "
        "hybrid={:.6f}/{:.6f}, time-expanded={:.6f}/{:.6f}, "
        "avg_gap_vs_hybrid={:+.3%}, pass_avg={}".format(
            default_score[0],
            default_score[1],
            locality_score[0],
            locality_score[1],
            hybrid_score[0],
            hybrid_score[1],
            time_expanded_score[0],
            time_expanded_score[1],
            gap_vs_hybrid["avg_jct"],
            pass_avg,
        ),
        flush=True,
    )

    return {
        "input": str(path),
        "scene": metadata.get("scene", path.stem),
        "tenant_count": tenant_count,
        "trial_index": trial_index,
        "trial_seed": int(trial.get("trial_seed", -1)),
        "scores": {
            "default": score_dict(default_score),
            "locality": score_dict(locality_score),
            "hybrid_live": score_dict(hybrid_score),
            "time_expanded_live": score_dict(time_expanded_score),
        },
        "relative_gap_time_expanded_vs_hybrid": gap_vs_hybrid,
        "passed_avg_jct_sensitivity": bool(pass_avg),
        "passed_lexicographic_sensitivity": bool(pass_lexicographic),
        "same_mapping": bool(hybrid_mapping == time_expanded_mapping),
        "mapping_diff_by_tenant": mapping_diff_by_tenant(hybrid_mapping, time_expanded_mapping),
        "runtimes": {
            "hybrid_live": float(hybrid_runtime),
            "time_expanded_live": float(time_expanded_runtime),
        },
        "solver_metadata": {
            "hybrid_live": hybrid_meta,
            "time_expanded_live": time_expanded_meta,
        },
        "mappings": {
            "hybrid_live": normalize_mapping(hybrid_mapping),
            "time_expanded_live": normalize_mapping(time_expanded_mapping),
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"case_count": 0}
    avg_gaps = [
        float(row["relative_gap_time_expanded_vs_hybrid"]["avg_jct"])
        for row in rows
    ]
    mk_gaps = [
        float(row["relative_gap_time_expanded_vs_hybrid"]["makespan"])
        for row in rows
    ]
    failed = [row for row in rows if not bool(row["passed_avg_jct_sensitivity"])]
    return {
        "case_count": len(rows),
        "failed_avg_jct_sensitivity_count": len(failed),
        "max_avg_jct_gap": max(avg_gaps),
        "min_avg_jct_gap": min(avg_gaps),
        "max_makespan_gap": max(mk_gaps),
        "min_makespan_gap": min(mk_gaps),
        "failed_cases": [
            {
                "input": row["input"],
                "scene": row["scene"],
                "tenant_count": row["tenant_count"],
                "trial_index": row["trial_index"],
                "avg_jct_gap": row["relative_gap_time_expanded_vs_hybrid"]["avg_jct"],
                "makespan_gap": row["relative_gap_time_expanded_vs_hybrid"]["makespan"],
            }
            for row in failed
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live debug comparison between collapsed hybrid mapping and the "
            "standalone time-expanded-estimator optimizer.  Experiment JSON "
            "files are used only for topology/workload/initial-mapping inputs."
        )
    )
    parser.add_argument("--inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--tenant-counts", default="3")
    parser.add_argument("--trials", default="1", help="1-based trial indices, e.g. 1,2,3")
    parser.add_argument(
        "--hybrid-time-limit",
        type=float,
        default=None,
        help="Optional hybrid time limit. Default: no limit, matching the formal experiment entry.",
    )
    parser.add_argument("--time-expanded-time-limit", type=float, default=120.0)
    parser.add_argument("--sensitivity", type=float, default=0.02)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "debug" / "audits" / "live_hybrid_vs_time_expanded.json",
    )
    args = parser.parse_args()

    tenant_counts = parse_int_set(args.tenant_counts)
    trial_indices = {idx - 1 for idx in parse_int_set(args.trials)}
    tasks = list(iter_cases(args.inputs, tenant_counts, trial_indices))
    if not tasks:
        raise SystemExit("No matching cases found.")
    for task in tasks:
        task["hybrid_time_limit"] = args.hybrid_time_limit
        task["time_expanded_time_limit"] = args.time_expanded_time_limit
        task["sensitivity"] = args.sensitivity

    jobs = max(1, min(int(args.jobs), len(tasks), os.cpu_count() or 1))
    if jobs == 1:
        rows = [run_case(task) for task in tasks]
    else:
        rows = []
        with Pool(processes=jobs, maxtasksperchild=1) as pool:
            for row in pool.imap_unordered(run_case, tasks):
                rows.append(row)
        rows.sort(key=lambda row: (str(row["input"]), int(row["tenant_count"]), int(row["trial_index"])))

    output = {
        "metadata": {
            "inputs": [str(path) for path in args.inputs],
            "tenant_counts": sorted(tenant_counts),
            "trials_1_based": sorted(idx + 1 for idx in trial_indices),
            "hybrid_time_limit": args.hybrid_time_limit,
            "time_expanded_time_limit": args.time_expanded_time_limit,
            "sensitivity": float(args.sensitivity),
            "jobs": jobs,
            "note": (
                "Stored proposed mappings/results in experiment JSON files are "
                "ignored; both solvers are run live and evaluated by simulator."
            ),
        },
        "summary": summarize(rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved live comparison to {args.output}", flush=True)


if __name__ == "__main__":
    main()
