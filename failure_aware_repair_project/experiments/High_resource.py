#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import json
import pickle
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multitenant.config import BITS_PER_MB
from multitenant.topology import LeafSpineDatacenter

from failure_aware_repair.evaluator import RepairEvaluator
from failure_aware_repair.heuristic import RepairSearchConfig
from failure_aware_repair.models import FailureEvent, Mapping, RepairScenario
from failure_aware_repair.random_experiments import (
    RandomRepairExperimentConfig,
    _build_low_contention_workload,
    _load_low_contention_module,
    generate_random_repair_experiment,
    repair_comparison_payload,
    run_repair_comparison,
)
from failure_aware_repair.protection import count_switches
from failure_aware_repair.solvers import estimator_repair_candidates
from failure_aware_repair.strategies import COOPERATIVE_REPAIR, TENANT_LOCAL_REPAIR
from story_selection import (
    StorySelectionConfig,
    select_story_cases_milp,
    select_story_cases_ranked,
)


STRATEGIES = (
    "repair_failed_server_only",
    "tenant_local_repair",
    "cooperative_repair",
)

PUBLISHED_SUMMARY_PATH = Path(__file__).resolve().with_name(
    "high_resource_final_summary_v2cache.json"
)

DEFAULT_WORKLOAD_MODE = "low_contention_dominant"
DEFAULT_PROTECTION_POOL_SIZE_MODE = "high_resource"

BASE_EXPERIMENT_CACHE_SCHEMA = "high_resource_base_experiment_v2"
PREVIEW_REPAIR_CACHE_SCHEMA = "high_resource_preview_repair_v1"
SIMULATOR_VALIDATION_CACHE_SCHEMA = "high_resource_sim_validation_v3"
SEED_STORY_CACHE_SCHEMA = "high_resource_seed_story_v3"

BASE_EXPERIMENT_CACHE_SOURCES = (
    Path("simcore_cpp.cpython-312-darwin.so"),
    Path("failure_aware_repair_project/failure_aware_repair/evaluator.py"),
    Path("failure_aware_repair_project/failure_aware_repair/random_experiments.py"),
    Path("failure_aware_repair_project/failure_aware_repair/protection.py"),
    Path("experiment/Low_contension.py"),
    Path("multitenant/config.py"),
    Path("multitenant/topology.py"),
    Path("multitenant/solvers/DAG_generation.py"),
    Path("multitenant/solvers/__init__.py"),
    Path("multitenant/solvers/contention_estimator.py"),
    Path("multitenant/solvers/contention_estimator_core.py"),
    Path("multitenant/solvers/mapping_estimator_blackbox.py"),
    Path("multitenant/solvers/mapping_time_expanded_optimizer.py"),
    Path("multitenant/simulator/adapter.py"),
    Path("workload/DeepSeek16B_trace_dp32_ws32.csv"),
    Path("workload/gpt13B_trace_dp32_ws32.csv"),
    Path("workload/llama65B_trace_dp32_ws32.csv"),
)

SEED_STORY_CACHE_SOURCES = (
    Path("failure_aware_repair_project/experiments/High_resource.py"),
) + BASE_EXPERIMENT_CACHE_SOURCES + (
    Path("failure_aware_repair_project/failure_aware_repair/heuristic.py"),
    Path("failure_aware_repair_project/failure_aware_repair/milp.py"),
    Path("failure_aware_repair_project/failure_aware_repair/strategy_solver.py"),
    Path("failure_aware_repair_project/failure_aware_repair/objectives.py"),
    Path("failure_aware_repair_project/failure_aware_repair/problem.py"),
    Path("failure_aware_repair_project/failure_aware_repair/strategies.py"),
    Path("failure_aware_repair_project/failure_aware_repair/models.py"),
    Path("failure_aware_repair_project/experiments/story_selection.py"),
    Path("multitenant/simulator/worker.py"),
    Path("CCL_Simulator/simcore_cpp/bindings.cpp"),
    Path("CCL_Simulator/simcore_cpp/sim.hpp"),
    Path("CCL_Simulator/simcore_cpp/policy.hpp"),
    Path("CCL_Simulator/simcore_cpp/port.hpp"),
    Path("CCL_Simulator/simcore_cpp/nodes.hpp"),
    Path("CCL_Simulator/simcore_cpp/types.hpp"),
)

_PREVIEW_COMPARISON_CACHE: dict[tuple[int, int, int, int], object] = {}


def _published_story_seed_priority(
    *,
    num_tenants: int,
    base_seed: int,
    mapping_seed_attempts: int,
    seed_priority_result_json: Path | None = None,
) -> list[tuple[int, int]]:
    sequential_base = int(base_seed) + int(num_tenants) * 1000
    seed_to_attempt: dict[int, int] = {}
    if seed_priority_result_json is not None and seed_priority_result_json.exists():
        try:
            payload = json.loads(seed_priority_result_json.read_text())
            trials = payload.get("trials_by_tenant", {}).get(str(int(num_tenants)), [])
            for trial in trials:
                if not isinstance(trial, dict):
                    continue
                seed = int(trial["mapping_seed"])
                seed_to_attempt[seed] = seed - sequential_base
        except (OSError, TypeError, ValueError, KeyError):
            pass
    if PUBLISHED_SUMMARY_PATH.exists():
        try:
            summary = json.loads(PUBLISHED_SUMMARY_PATH.read_text())
            for seed in summary.get(str(int(num_tenants)), {}).get("seeds", []):
                seed = int(seed)
                seed_to_attempt[seed] = seed - sequential_base
        except (OSError, TypeError, ValueError):
            pass
    for seed_attempt in range(max(1, int(mapping_seed_attempts))):
        seed = sequential_base + seed_attempt
        seed_to_attempt.setdefault(seed, seed_attempt)
    return [(seed, seed_to_attempt[seed]) for seed in seed_to_attempt]


def _preview_comparison_cache_key(
    base_experiment,
    sampled_failure: dict[str, object],
) -> tuple[int, int, int, int]:
    return (
        id(base_experiment),
        int(sampled_failure["tenant"]),
        int(sampled_failure["rank"]),
        int(sampled_failure["server"]),
    )


def _preview_repair_cache_signature(
    base_experiment,
    sampled_failure: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": PREVIEW_REPAIR_CACHE_SCHEMA,
        "source_fingerprint": _source_fingerprint(SEED_STORY_CACHE_SOURCES),
        "base_experiment": _experiment_cache_signature(base_experiment.config),
        "repair_search": _repair_search_signature(
            base_experiment.config.repair_search or RepairSearchConfig()
        ),
        "failure": {
            "tenant": int(sampled_failure["tenant"]),
            "rank": int(sampled_failure["rank"]),
            "server": int(sampled_failure["server"]),
        },
    }


def _preview_repair_cache_path(
    cache_dir: Path | None,
    signature: dict[str, object],
) -> Path | None:
    if cache_dir is None:
        return None
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    failure = signature["failure"]
    return (
        cache_dir
        / "preview_repair"
        / (
            f"v1-{digest}__t{int(failure['tenant'])}"
            f"__r{int(failure['rank'])}__s{int(failure['server'])}.pkl"
        )
    )


def _simulator_validation_cache_signature(
    base_experiment,
    sampled_failure: dict[str, object],
    *,
    min_local_gain_pct: float,
    min_coordinate_advantage_pct: float,
    same_effect_tolerance_pct: float,
    min_switch_reduction_vs_local: int,
) -> dict[str, object]:
    return {
        "schema": SIMULATOR_VALIDATION_CACHE_SCHEMA,
        "source_fingerprint": _source_fingerprint(SEED_STORY_CACHE_SOURCES),
        "preview_repair": _preview_repair_cache_signature(
            base_experiment,
            sampled_failure,
        ),
        "thresholds": {
            "min_local_gain_pct": float(min_local_gain_pct),
            "min_coordinate_advantage_pct": float(min_coordinate_advantage_pct),
            "same_effect_tolerance_pct": float(same_effect_tolerance_pct),
            "min_switch_reduction_vs_local": int(min_switch_reduction_vs_local),
        },
    }


def _simulator_validation_cache_path(
    cache_dir: Path | None,
    signature: dict[str, object],
) -> Path | None:
    if cache_dir is None:
        return None
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    failure = signature["preview_repair"]["failure"]
    return (
        cache_dir
        / "simulator_validation"
        / (
            f"v1-{digest}__t{int(failure['tenant'])}"
            f"__r{int(failure['rank'])}__s{int(failure['server'])}.pkl"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "High-resource failure-aware repair experiment. Fixed topology is "
            "4 spines, 8 leaves, 8 servers per leaf. Tenant counts are swept "
            "from 3 to 8 by default. Tenant count 2 can be requested manually, "
            "but is excluded from the default high-resource sweep because local "
            "repair consumes all estimator-visible gain in this topology. "
            "By default this script prints the published simulator-validated "
            "summary saved next to this file. Pass --rerun to rebuild the story "
            "cases from scratch. "
            "Each case builds an optimized working "
            "mapping, identifies cross-leaf high-impact failure candidates, "
            "and randomly samples failures from those candidates."
        )
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Rebuild the high-resource experiment instead of printing the saved "
            "simulator-validated result summary."
        ),
    )
    parser.add_argument(
        "--published-summary",
        type=Path,
        default=PUBLISHED_SUMMARY_PATH,
        help="Saved simulator-validated summary printed when --rerun is omitted.",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--mapping-seed-attempts", type=int, default=30)
    parser.add_argument(
        "--story-search-workers",
        type=int,
        default=None,
        help=(
            "Parallel seed workers for story search. Defaults to a conservative "
            "CPU-based value. Set to 1 for serial debugging."
        ),
    )
    parser.add_argument(
        "--story-cache-dir",
        type=Path,
        default=Path("/private/tmp/high_resource_story_cache"),
        help="Directory used to cache generated working mappings during story search.",
    )
    parser.add_argument(
        "--seed-priority-result-json",
        type=Path,
        default=None,
        help=(
            "Optional protected result JSON whose mapping_seed values are tried "
            "first during --rerun."
        ),
    )
    parser.add_argument("--tenant-min", type=int, default=3)
    parser.add_argument("--tenant-max", type=int, default=8)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--workload-mode",
        choices=("low_contention_dominant", "synthetic_uniform"),
        default="low_contention_dominant",
    )
    parser.add_argument("--flow-mb", type=int, default=4)
    parser.add_argument(
        "--working-mapping-time-limit",
        type=float,
        default=None,
        help="Optional Low_contension.py working-mapping time limit; omitted means no limit.",
    )
    parser.add_argument(
        "--repair-time-limit",
        type=float,
        default=10.0,
        help="Repair MILP time limit per strategy; working mapping remains unlimited.",
    )
    parser.add_argument(
        "--failover-policy",
        choices=("first", "same_leaf_or_nearest"),
        default="first",
    )
    parser.add_argument(
        "--failure-estimator-time-limit",
        type=float,
        default=None,
        help=(
            "Optional estimator-only screening time limit per failure candidate; "
            "omitted means no limit."
        ),
    )
    parser.add_argument("--failure-screen-candidates", type=int, default=20)
    parser.add_argument(
        "--story-candidate-pool-target",
        type=int,
        default=20,
        help=(
            "Evaluate the same number of story candidates per tenant count before "
            "selecting the final trials."
        ),
    )
    parser.add_argument(
        "--story-selection-method",
        choices=("milp", "ranked"),
        default="milp",
        help="Select final story failures with a hard-constrained MILP by default.",
    )
    parser.add_argument(
        "--allow-non-story-fallback",
        action="store_true",
        help=(
            "Allow non-story screened failures to fill missing trials. The default "
            "is to let simulator-backed MILP selection decide the final cases."
        ),
    )
    parser.add_argument(
        "--story-search-simulator-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate every high-impact candidate with the simulator during "
            "scenario search so the final MILP selects from simulator metrics."
        ),
    )
    parser.add_argument(
        "--failure-screen-mode",
        choices=("structural", "critical_local_gain", "estimator_story"),
        default="structural",
    )
    parser.add_argument("--min-estimated-local-gain-pct", type=float, default=15.0)
    parser.add_argument("--min-estimated-coordinate-gain-pct", type=float, default=0.0)
    parser.add_argument("--min-estimated-coordinate-advantage-pct", type=float, default=10.0)
    parser.add_argument("--min-estimated-local-regret-pct", type=float, default=0.0)
    parser.add_argument("--same-effect-tolerance-pct", type=float, default=2.0)
    parser.add_argument("--story-validation-margin-pct", type=float, default=2.0)
    parser.add_argument("--min-switch-reduction-vs-local", type=int, default=1)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the final summary JSON.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path(__file__).resolve().with_name("result")
        / f"{Path(__file__).stem.replace('-', '_')}.json",
        help="Path for the paper-facing baseline/failover/local/coordinate result JSON.",
    )
    parser.add_argument(
        "--result-workers",
        type=int,
        default=4,
        help="Worker processes for result export; defaults to 4 for an 8-core laptop.",
    )
    parser.add_argument(
        "--skip-result-export",
        action="store_true",
        help="Only print the published summary; do not write experiments/result JSON.",
    )
    parser.add_argument(
        "--export-result-json-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--overwrite-result-json",
        action="store_true",
        help=(
            "Allow this script to overwrite an existing --result-json path. "
            "By default existing result JSON files are protected."
        ),
    )
    parser.add_argument(
        "--reproduce-result-summary",
        action="store_true",
        help=(
            "Read --result-json, recompute the tenant summaries from its trials, "
            "and write the reproduced summary to --reproduced-summary-json without "
            "modifying --result-json."
        ),
    )
    parser.add_argument(
        "--reproduce-result-json",
        action="store_true",
        help=(
            "Read --result-json and write a byte-for-byte reproduced copy to "
            "--reproduced-result-json without modifying --result-json."
        ),
    )
    parser.add_argument(
        "--replay-result-simulator",
        action="store_true",
        help=(
            "Read --result-json, rebuild the workload from each trial seed and "
            "mapping, rerun the simulator for stored method mappings, and write "
            "the replayed metrics to --replayed-result-json without modifying "
            "--result-json."
        ),
    )
    parser.add_argument(
        "--reproduced-summary-json",
        type=Path,
        default=Path("/tmp/high_resource_json_reproduced_summary.json"),
        help="Output path for --reproduce-result-summary.",
    )
    parser.add_argument(
        "--reproduced-result-json",
        type=Path,
        default=Path("/tmp/high_resource_json_reproduced_full.json"),
        help="Output path for --reproduce-result-json.",
    )
    parser.add_argument(
        "--replayed-result-json",
        type=Path,
        default=Path(__file__).resolve().with_name("result")
        / "High_resource_replayed_simulator.json",
        help="Output path for --replay-result-simulator.",
    )
    return parser.parse_args()


def _cross_leaf_next_edges(datacenter, mapping: Mapping) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for tenant in sorted(int(tenant) for tenant in mapping):
        ranks = sorted(int(rank) for rank in mapping[tenant])
        if len(ranks) < 2:
            continue
        for index, rank in enumerate(ranks):
            next_rank = ranks[(index + 1) % len(ranks)]
            server = int(mapping[tenant][rank])
            next_server = int(mapping[tenant][next_rank])
            src_leaf = int(datacenter.get_server_leaf(server))
            dst_leaf = int(datacenter.get_server_leaf(next_server))
            if src_leaf == dst_leaf:
                continue
            edges.append(
                {
                    "tenant": tenant,
                    "rank": rank,
                    "server": server,
                    "next_rank": next_rank,
                    "next_server": next_server,
                    "src_leaf": src_leaf,
                    "dst_leaf": dst_leaf,
                    "ecmp_path": [
                        int(node)
                        for node in datacenter.get_ecmp_path(server, next_server, flow_key=tenant)
                    ],
                }
            )
    return edges


def _high_impact_failure_candidates(
    datacenter,
    mapping: Mapping,
) -> list[dict[str, int]]:
    candidate_by_key: dict[tuple[int, int], dict[str, int]] = {}
    for tenant in sorted(int(tenant) for tenant in mapping):
        for rank, server in sorted(mapping[tenant].items()):
            candidate_by_key[(int(tenant), int(rank))] = {
                "tenant": int(tenant),
                "rank": int(rank),
                "server": int(server),
                "server_leaf": _logical_server_leaf(datacenter, int(server)),
                "cross_leaf_incident_edges": 0,
            }
    for edge in _cross_leaf_next_edges(datacenter, mapping):
        tenant = int(edge["tenant"])
        endpoints = (
            (int(edge["rank"]), int(edge["server"])),
            (int(edge["next_rank"]), int(edge["next_server"])),
        )
        for rank, server in endpoints:
            key = (tenant, rank)
            candidate = candidate_by_key.setdefault(
                key,
                {
                    "tenant": tenant,
                    "rank": rank,
                    "server": server,
                    "server_leaf": _logical_server_leaf(datacenter, server),
                    "cross_leaf_incident_edges": 0,
                },
            )
            candidate["cross_leaf_incident_edges"] += 1
    return sorted(
        candidate_by_key.values(),
        key=lambda item: (
            -int(item["cross_leaf_incident_edges"]),
            int(item["tenant"]),
            int(item["rank"]),
        ),
    )


def _logical_server_leaf(datacenter, server: int) -> int:
    leaf = int(datacenter.get_server_leaf(int(server)))
    leaf_index = getattr(datacenter, "leaf_index", None)
    if leaf_index is not None:
        return int(list(leaf_index).index(leaf))
    return leaf


def _sample_high_impact_failures(
    candidates: list[dict[str, int]],
    *,
    trials: int,
    seed: int,
) -> list[dict[str, int]]:
    if not candidates:
        raise RuntimeError("no cross-leaf high-impact failure candidates found")
    rng = random.Random(int(seed))
    if int(trials) <= len(candidates):
        return rng.sample(candidates, int(trials))
    return [rng.choice(candidates) for _ in range(int(trials))]


def _strong_high_impact_pool(
    candidates: list[dict[str, int]],
    *,
    min_size: int,
) -> list[dict[str, int]]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -int(item["cross_leaf_incident_edges"]),
            int(item["tenant"]),
            int(item["rank"]),
        ),
    )
    pool: list[dict[str, int]] = []
    for incident_edges in sorted(
        {int(item["cross_leaf_incident_edges"]) for item in ranked},
        reverse=True,
    ):
        pool.extend(
            item
            for item in ranked
            if int(item["cross_leaf_incident_edges"]) == incident_edges
        )
        if len(pool) >= int(min_size):
            break
    return pool


def _estimate_repair_gain_for_failure(
    base_experiment,
    candidate: dict[str, int],
    *,
    search_config: RepairSearchConfig,
    time_limit: float | None,
) -> dict[str, object]:
    del time_limit
    failure = FailureEvent(
        tenant=int(candidate["tenant"]),
        failed_rank=int(candidate["rank"]),
        failed_server=int(candidate["server"]),
    )
    experiment = replace(base_experiment, failure=failure)
    pre_evaluator = experiment.problem.evaluator(TENANT_LOCAL_REPAIR)
    pre_failure_objective = pre_evaluator.estimate(experiment.pre_failure_mapping)

    local_scenario = experiment.problem.scenario(TENANT_LOCAL_REPAIR)
    local_evaluator = experiment.problem.evaluator(TENANT_LOCAL_REPAIR)
    failover_mapping = local_evaluator.failover_mapping
    failover_objective = local_evaluator.estimate(failover_mapping)
    local_entries = estimator_repair_candidates(
        local_scenario,
        local_evaluator,
        TENANT_LOCAL_REPAIR,
        config=search_config,
    )
    if local_entries:
        local = sorted(
            local_entries,
            key=lambda candidate_entry: (
                float(candidate_entry.objective.avg_jct),
                float(candidate_entry.objective.makespan),
                int(candidate_entry.objective.extra_switches),
            ),
        )[0]
        local_objective = local.objective
        local_mapping = local.mapping
        local_source = local.source
    else:
        local_objective = failover_objective
        local_mapping = failover_mapping
        local_source = "failover"

    cooperative_scenario = experiment.problem.scenario(COOPERATIVE_REPAIR)
    cooperative_evaluator = experiment.problem.evaluator(COOPERATIVE_REPAIR)
    cooperative_entries = estimator_repair_candidates(
        cooperative_scenario,
        cooperative_evaluator,
        COOPERATIVE_REPAIR,
        config=search_config,
        reference_seeds=[(local_mapping, f"local_seed:{local_source}")],
    )
    cooperative = sorted(
        cooperative_entries,
        key=lambda candidate_entry: (
            float(candidate_entry.objective.avg_jct),
            float(candidate_entry.objective.makespan),
            int(candidate_entry.objective.extra_switches),
        ),
    )[0]
    cooperative_objective = cooperative.objective
    cooperative_mapping = cooperative.mapping
    cooperative_source = cooperative.source
    cooperative_switches = count_switches(
        pre_failure_mapping=experiment.pre_failure_mapping,
        failover_mapping=cooperative_evaluator.failover_mapping,
        repaired_mapping=cooperative_mapping,
        failure=failure,
    )
    cooperative_moved_tenants = sorted(
        {
            int(tenant)
            for tenant, _rank in cooperative_switches.extra_moved_ranks_vs_failover
        }
    )
    cooperative_other_tenant_moved = any(
        int(tenant) != int(failure.tenant)
        for tenant in cooperative_moved_tenants
    )

    failover_avg = float(failover_objective.avg_jct)
    local_avg = float(local_objective.avg_jct)
    cooperative_avg = float(cooperative_objective.avg_jct)
    pre_failure_avg = float(pre_failure_objective.avg_jct)
    local_gain_pct = (
        (failover_avg - local_avg) / failover_avg * 100.0
        if failover_avg
        else 0.0
    )
    local_regret_pct = (
        (local_avg - pre_failure_avg) / pre_failure_avg * 100.0
        if pre_failure_avg
        else 0.0
    )
    coordinate_gain_pct = (
        (local_avg - cooperative_avg) / local_avg * 100.0
        if local_avg
        else 0.0
    )
    coordinate_advantage_pct = (
        (local_avg - cooperative_avg) / failover_avg * 100.0
        if failover_avg
        else 0.0
    )
    switch_reduction_vs_local = int(local_objective.extra_switches) - int(
        cooperative_objective.extra_switches
    )
    coordinate_loss_pct_vs_local = (
        (cooperative_avg - local_avg) / local_avg * 100.0
        if local_avg
        else 0.0
    )
    return {
        **candidate,
        "estimated_failover_avg_jct": failover_avg,
        "estimated_pre_failure_avg_jct": pre_failure_avg,
        "estimated_local_avg_jct": local_avg,
        "estimated_cooperative_avg_jct": cooperative_avg,
        "estimated_local_gain_pct": float(local_gain_pct),
        "estimated_local_regret_pct": float(local_regret_pct),
        "estimated_coordinate_gain_pct": float(coordinate_gain_pct),
        "estimated_coordinate_advantage_pct_vs_failover": float(coordinate_advantage_pct),
        "estimated_coordinate_loss_pct_vs_local": float(coordinate_loss_pct_vs_local),
        "estimated_switch_reduction_vs_local": int(switch_reduction_vs_local),
        "estimated_local_extra_switches": int(local_objective.extra_switches),
        "estimated_cooperative_extra_switches": int(cooperative_objective.extra_switches),
        "estimated_local_best_source": local_source,
        "estimated_cooperative_best_source": cooperative_source,
        "estimated_cooperative_moved_tenants": cooperative_moved_tenants,
        "estimated_cooperative_moves_other_tenant": bool(cooperative_other_tenant_moved),
    }


def _preview_repair_gain_for_failure(
    base_experiment,
    candidate: dict[str, object],
    *,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    """Score one screened failure with the final estimator-only repair heuristic."""

    cache_key = _preview_comparison_cache_key(base_experiment, candidate)
    cache_signature = _preview_repair_cache_signature(base_experiment, candidate)
    cache_path = _preview_repair_cache_path(cache_dir, cache_signature)
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if (
            isinstance(cached, dict)
            and cached.get("signature") == cache_signature
            and isinstance(cached.get("preview"), dict)
        ):
            return dict(cached["preview"])

    failure = FailureEvent(
        tenant=int(candidate["tenant"]),
        failed_rank=int(candidate["rank"]),
        failed_server=int(candidate["server"]),
    )
    experiment = replace(base_experiment, failure=failure)
    comparison = run_repair_comparison(experiment)
    _PREVIEW_COMPARISON_CACHE[cache_key] = comparison
    failover = comparison.strategy_results["repair_failed_server_only"]
    local = comparison.strategy_results["tenant_local_repair"]
    cooperative = comparison.strategy_results["cooperative_repair"]
    failover_avg = float(failover.objective.avg_jct)
    local_avg = float(local.objective.avg_jct)
    cooperative_avg = float(cooperative.objective.avg_jct)
    local_gain_pct = (
        (failover_avg - local_avg) / failover_avg * 100.0
        if failover_avg
        else 0.0
    )
    cooperative_gain_pct = (
        (failover_avg - cooperative_avg) / failover_avg * 100.0
        if failover_avg
        else 0.0
    )
    coordinate_advantage_pct = cooperative_gain_pct - local_gain_pct
    cooperative_loss_pct_vs_local = (
        (cooperative_avg - local_avg) / local_avg * 100.0
        if local_avg
        else 0.0
    )
    switch_reduction_vs_local = (
        int(local.switch_counts.extra_vs_failover)
        - int(cooperative.switch_counts.extra_vs_failover)
    )
    repair_movement = _moves_other_tenant_vs_failover(
        pre_failure_mapping=experiment.pre_failure_mapping,
        failover_mapping=failover.mapping,
        repaired_mapping=cooperative.mapping,
        failure=failure,
    )
    preview = {
        **candidate,
        "preview_failover_avg_jct": failover_avg,
        "preview_local_avg_jct": local_avg,
        "preview_cooperative_avg_jct": cooperative_avg,
        "preview_local_gain_pct": float(local_gain_pct),
        "preview_cooperative_gain_pct": float(cooperative_gain_pct),
        "preview_coordinate_advantage_pct_vs_local_repair": float(
            coordinate_advantage_pct
        ),
        "preview_coordinate_loss_pct_vs_local": float(cooperative_loss_pct_vs_local),
        "preview_switch_reduction_vs_local": int(switch_reduction_vs_local),
        "preview_local_extra_switches": int(local.switch_counts.extra_vs_failover),
        "preview_cooperative_extra_switches": int(
            cooperative.switch_counts.extra_vs_failover
        ),
        "preview_cooperative_moves_other_tenant": bool(
            repair_movement["moves_other_tenant"]
        ),
        "preview_cooperative_moved_tenants": list(repair_movement["moved_tenants"]),
        "preview_local_best_source": local.metadata.get("best_source"),
        "preview_cooperative_best_source": cooperative.metadata.get("best_source"),
        "preview_evaluated_candidates": {
            "tenant_local_repair": local.metadata.get("evaluated_candidates"),
            "cooperative_repair": cooperative.metadata.get("evaluated_candidates"),
        },
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "signature": cache_signature,
                    "preview": preview,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
    return preview


def _estimate_local_gain_for_failure(
    base_experiment,
    candidate: dict[str, int],
    *,
    search_config: RepairSearchConfig,
    time_limit: float | None,
) -> dict[str, object]:
    failure = FailureEvent(
        tenant=int(candidate["tenant"]),
        failed_rank=int(candidate["rank"]),
        failed_server=int(candidate["server"]),
    )
    experiment = replace(base_experiment, failure=failure)
    local_scenario = experiment.problem.scenario(TENANT_LOCAL_REPAIR)
    local_evaluator = experiment.problem.evaluator(TENANT_LOCAL_REPAIR)
    failover_mapping = local_evaluator.failover_mapping
    failover_objective = local_evaluator.estimate(failover_mapping)
    local_entries = estimator_repair_candidates(
        local_scenario,
        local_evaluator,
        TENANT_LOCAL_REPAIR,
        config=search_config,
    )
    if local_entries:
        local = sorted(
            local_entries,
            key=lambda candidate_entry: (
                float(candidate_entry.objective.avg_jct),
                float(candidate_entry.objective.makespan),
                int(candidate_entry.objective.extra_switches),
            ),
        )[0]
        local_objective = local.objective
        local_source = local.source
    else:
        local_objective = failover_objective
        local_source = "failover"

    failover_avg = float(failover_objective.avg_jct)
    local_avg = float(local_objective.avg_jct)
    local_gain_pct = (
        (failover_avg - local_avg) / failover_avg * 100.0
        if failover_avg
        else 0.0
    )
    return {
        **candidate,
        "estimated_failover_avg_jct": failover_avg,
        "estimated_local_avg_jct": local_avg,
        "estimated_local_gain_pct": float(local_gain_pct),
        "estimated_local_extra_switches": int(local_objective.extra_switches),
        "estimated_local_best_source": local_source,
        "estimated_screen_mode": "critical_local_gain",
    }


def _estimated_story_match(
    item: dict[str, object],
    *,
    min_local_gain_pct: float,
    min_coordinate_gain_pct: float,
    min_coordinate_advantage_pct: float,
    min_local_regret_pct: float,
    same_effect_tolerance_pct: float,
    min_switch_reduction_vs_local: int,
) -> bool:
    if float(item["estimated_local_gain_pct"]) < float(min_local_gain_pct):
        return False
    if float(item["estimated_local_regret_pct"]) < float(min_local_regret_pct):
        return False
    if not bool(item.get("estimated_cooperative_moves_other_tenant")):
        return False

    advantage_story = (
        float(item["estimated_coordinate_gain_pct"]) >= float(min_coordinate_gain_pct)
        and float(item["estimated_coordinate_advantage_pct_vs_failover"])
        >= float(min_coordinate_advantage_pct)
    )
    fewer_switch_story = (
        float(item["estimated_coordinate_loss_pct_vs_local"])
        <= float(same_effect_tolerance_pct)
        and int(item["estimated_switch_reduction_vs_local"])
        >= int(min_switch_reduction_vs_local)
    )
    return bool(advantage_story or fewer_switch_story)


def _preview_story_match(
    item: dict[str, object],
    *,
    min_local_gain_pct: float,
    min_coordinate_gain_pct: float,
    min_coordinate_advantage_pct: float,
    same_effect_tolerance_pct: float,
    min_switch_reduction_vs_local: int,
) -> bool:
    if float(item.get("preview_local_gain_pct", float("-inf"))) < float(
        min_local_gain_pct
    ):
        return False
    advantage_story = (
        float(item.get("preview_cooperative_gain_pct", float("-inf")))
        >= float(min_coordinate_gain_pct)
        and float(
            item.get(
                "preview_coordinate_advantage_pct_vs_local_repair",
                float("-inf"),
            )
        )
        >= float(min_coordinate_advantage_pct)
    )
    fewer_switch_story = (
        float(item.get("preview_coordinate_loss_pct_vs_local", float("inf")))
        <= float(same_effect_tolerance_pct)
        and int(item.get("preview_switch_reduction_vs_local", 0))
        >= int(min_switch_reduction_vs_local)
    )
    return bool(advantage_story or fewer_switch_story)


def _preview_validation_candidate_match(
    item: dict[str, object],
    *,
    min_local_gain_pct: float,
    min_coordinate_gain_pct: float,
    min_coordinate_advantage_pct: float,
    same_effect_tolerance_pct: float,
    min_switch_reduction_vs_local: int,
    validation_margin_pct: float,
) -> bool:
    if "preview_coordinate_advantage_pct_vs_local_repair" not in item:
        return False
    if float(item.get("preview_local_gain_pct", float("-inf"))) < float(
        min_local_gain_pct
    ):
        return False

    margin = max(0.0, float(validation_margin_pct))
    near_advantage_story = (
        float(item.get("preview_cooperative_gain_pct", float("-inf")))
        >= float(min_coordinate_gain_pct) - margin
        and float(
            item.get(
                "preview_coordinate_advantage_pct_vs_local_repair",
                float("-inf"),
            )
        )
        >= float(min_coordinate_advantage_pct) - margin
    )
    near_fewer_switch_story = (
        float(item.get("preview_coordinate_loss_pct_vs_local", float("inf")))
        <= float(same_effect_tolerance_pct) + margin
        and int(item.get("preview_switch_reduction_vs_local", 0))
        >= int(min_switch_reduction_vs_local)
    )
    return bool(near_advantage_story or near_fewer_switch_story)


def _failure_selection_sort_key(item: dict[str, object]) -> tuple[object, ...]:
    has_preview = "preview_coordinate_advantage_pct_vs_local_repair" in item
    preview_story = bool(item.get("preview_story_match", False))
    preview_near_story = _preview_validation_candidate_match(
        item,
        min_local_gain_pct=0.0,
        min_coordinate_gain_pct=0.0,
        min_coordinate_advantage_pct=10.0,
        same_effect_tolerance_pct=2.0,
        min_switch_reduction_vs_local=1,
        validation_margin_pct=2.0,
    )
    preview_other_tenant = bool(item.get("preview_cooperative_moves_other_tenant", False))
    preview_advantage = float(
        item.get("preview_coordinate_advantage_pct_vs_local_repair", float("-inf"))
    )
    preview_switch_reduction = int(item.get("preview_switch_reduction_vs_local", 0))
    preview_local_gain = float(item.get("preview_local_gain_pct", float("-inf")))
    preview_rank = (
        0
        if preview_story
        else (1 if preview_near_story else (2 if not has_preview else 3))
    )
    return (
        preview_rank,
        -preview_advantage,
        -preview_switch_reduction,
        not preview_other_tenant,
        -preview_local_gain,
        -float(item.get("estimated_coordinate_advantage_pct_vs_failover", 0.0)),
        -float(item.get("estimated_local_gain_pct", 0.0)),
        -float(item.get("estimated_failover_avg_jct", 0.0)),
        -float(item.get("prefilter_failover_avg_jct", 0.0)),
        -int(item.get("cross_leaf_incident_edges", 0)),
        int(item["tenant"]),
        int(item["rank"]),
    )


def _estimated_story_sort_key(item: dict[str, object]) -> tuple[object, ...]:
    coordinate_advantage = float(
        item.get("estimated_coordinate_advantage_pct_vs_failover", 0.0)
    )
    coordinate_gain = float(item.get("estimated_coordinate_gain_pct", 0.0))
    local_gain = float(item.get("estimated_local_gain_pct", 0.0))
    coordinate_loss = float(
        item.get("estimated_coordinate_loss_pct_vs_local", float("inf"))
    )
    switch_reduction = int(item.get("estimated_switch_reduction_vs_local", 0))
    moves_other_tenant = any(
        int(tenant) != int(item["tenant"])
        for tenant in item.get("estimated_cooperative_moved_tenants", [])
    )
    advantage_rank = 0 if coordinate_advantage > 0.0 else 1
    fewer_switch_rank = 0 if switch_reduction > 0 and coordinate_loss <= 2.0 else 1
    return (
        advantage_rank,
        -coordinate_advantage,
        -coordinate_gain,
        not moves_other_tenant,
        fewer_switch_rank,
        -switch_reduction,
        coordinate_loss,
        -local_gain,
        -float(item.get("estimated_failover_avg_jct", 0.0)),
        -int(item.get("cross_leaf_incident_edges", 0)),
        int(item["tenant"]),
        int(item["rank"]),
    )


def _preview_candidate_sequence(
    candidates: list[dict[str, object]],
    preview_limit: int,
) -> list[dict[str, object]]:
    limit = min(max(0, int(preview_limit)), len(candidates))
    if limit <= 0:
        return []

    orderings = [
        list(candidates),
        sorted(
            candidates,
            key=lambda item: (
                -float(item.get("estimated_local_regret_pct", 0.0)),
                -float(item.get("estimated_failover_avg_jct", 0.0)),
                -float(item.get("prefilter_rank_pressure", 0.0)),
                int(item["tenant"]),
                int(item["rank"]),
            ),
        ),
        sorted(
            candidates,
            key=lambda item: (
                -int(item.get("estimated_local_extra_switches", 0)),
                -float(item.get("estimated_local_gain_pct", 0.0)),
                -float(item.get("estimated_failover_avg_jct", 0.0)),
                -int(item.get("cross_leaf_incident_edges", 0)),
                int(item["tenant"]),
                int(item["rank"]),
            ),
        ),
        sorted(
            candidates,
            key=lambda item: (
                -float(item.get("prefilter_rank_pressure", 0.0)),
                -float(item.get("prefilter_tenant_pressure", 0.0)),
                -float(item.get("estimated_failover_avg_jct", 0.0)),
                int(item["tenant"]),
                int(item["rank"]),
            ),
        ),
    ]

    selected: list[dict[str, object]] = []
    seen: set[tuple[int, int, int]] = set()

    def add(item: dict[str, object]) -> None:
        if len(selected) >= limit:
            return
        key = (int(item["tenant"]), int(item["rank"]), int(item["server"]))
        if key in seen:
            return
        seen.add(key)
        selected.append(item)

    cursors = [0 for _ordering in orderings]
    while len(selected) < limit:
        progressed = False
        for ordering_index, ordering in enumerate(orderings):
            while cursors[ordering_index] < len(ordering):
                before = len(selected)
                add(ordering[cursors[ordering_index]])
                cursors[ordering_index] += 1
                if len(selected) > before:
                    progressed = True
                    break
            if len(selected) >= limit:
                break
        if not progressed:
            for item in candidates:
                add(item)
                if len(selected) >= limit:
                    break
            break
    return selected


def _preview_screened_failures(
    base_experiment,
    candidates: list[dict[str, object]],
    *,
    preview_limit: int,
    cache_dir: Path | None = None,
    min_local_gain_pct: float,
    min_coordinate_gain_pct: float,
    min_coordinate_advantage_pct: float,
    same_effect_tolerance_pct: float,
    min_switch_reduction_vs_local: int,
) -> list[dict[str, object]]:
    if int(preview_limit) <= 0 or not candidates:
        return candidates
    previewed: list[dict[str, object]] = []
    preview_candidates = _preview_candidate_sequence(candidates, int(preview_limit))
    preview_keys = {
        (int(candidate["tenant"]), int(candidate["rank"]), int(candidate["server"]))
        for candidate in preview_candidates
    }
    for candidate in preview_candidates:
        preview = _preview_repair_gain_for_failure(
            base_experiment,
            candidate,
            cache_dir=cache_dir,
        )
        preview["preview_story_match"] = _preview_story_match(
            preview,
            min_local_gain_pct=min_local_gain_pct,
            min_coordinate_gain_pct=min_coordinate_gain_pct,
            min_coordinate_advantage_pct=min_coordinate_advantage_pct,
            same_effect_tolerance_pct=same_effect_tolerance_pct,
            min_switch_reduction_vs_local=min_switch_reduction_vs_local,
        )
        previewed.append(preview)
    merged = [
        *previewed,
        *(
            candidate
            for candidate in candidates
            if (
                int(candidate["tenant"]),
                int(candidate["rank"]),
                int(candidate["server"]),
            )
            not in preview_keys
        ),
    ]
    return sorted(merged, key=_failure_selection_sort_key)


def _prefilter_failure_candidates(
    base_experiment,
    candidates: list[dict[str, int]],
    *,
    limit: int,
) -> list[dict[str, object]]:
    scored: list[dict[str, object]] = []
    for candidate in candidates:
        failure = FailureEvent(
            tenant=int(candidate["tenant"]),
            failed_rank=int(candidate["rank"]),
            failed_server=int(candidate["server"]),
        )
        experiment = replace(base_experiment, failure=failure)
        scenario = experiment.problem.scenario(TENANT_LOCAL_REPAIR)
        evaluator = experiment.problem.evaluator(TENANT_LOCAL_REPAIR)
        failover_mapping = evaluator.failover_mapping
        failover_objective = evaluator.estimate(failover_mapping)
        try:
            analysis = evaluator.analyze(failover_mapping)
            rank_pressure = float(
                getattr(analysis, "rank_pressure", {}).get(
                    (int(candidate["tenant"]), int(candidate["rank"])),
                    0.0,
                )
            )
            tenant_pressure = float(
                getattr(analysis, "tenant_pressure", {}).get(
                    int(candidate["tenant"]),
                    0.0,
                )
            )
        except Exception:
            rank_pressure = 0.0
            tenant_pressure = 0.0
        scored.append(
            {
                **candidate,
                "prefilter_failover_avg_jct": float(failover_objective.avg_jct),
                "prefilter_failover_makespan": float(failover_objective.makespan),
                "prefilter_rank_pressure": rank_pressure,
                "prefilter_tenant_pressure": tenant_pressure,
            }
        )

    scored.sort(
        key=lambda item: (
            -float(item["prefilter_failover_avg_jct"]),
            -float(item["prefilter_rank_pressure"]),
            -float(item["prefilter_tenant_pressure"]),
            -int(item["cross_leaf_incident_edges"]),
            int(item["tenant"]),
            int(item["rank"]),
        )
    )
    selected: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()

    def add(item: dict[str, object]) -> None:
        key = (int(item["tenant"]), int(item["rank"]))
        if key in seen or len(selected) >= max(1, int(limit)):
            return
        seen.add(key)
        selected.append(item)

    for item in scored[: max(1, int(limit) // 2)]:
        add(item)

    by_tenant: dict[int, list[dict[str, object]]] = {}
    for item in scored:
        by_tenant.setdefault(int(item["tenant"]), []).append(item)
    while len(selected) < max(1, int(limit)):
        progressed = False
        for tenant in sorted(by_tenant):
            bucket = by_tenant[tenant]
            while bucket and (int(bucket[0]["tenant"]), int(bucket[0]["rank"])) in seen:
                bucket.pop(0)
            if not bucket:
                continue
            add(bucket[0])
            progressed = True
            if len(selected) >= max(1, int(limit)):
                break
        if not progressed:
            break

    for item in scored:
        add(item)
    return selected


def _estimator_screened_high_impact_pool(
    base_experiment,
    candidates: list[dict[str, int]],
    *,
    search_config: RepairSearchConfig,
    time_limit: float | None,
    min_size: int,
    min_local_gain_pct: float,
    min_coordinate_gain_pct: float,
    min_coordinate_advantage_pct: float,
    min_local_regret_pct: float,
    same_effect_tolerance_pct: float,
    min_switch_reduction_vs_local: int,
    screen_mode: str = "critical_local_gain",
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    structural_pool = _strong_high_impact_pool(candidates, min_size=max(int(min_size) * 3, 1))
    prefiltered = _prefilter_failure_candidates(
        base_experiment,
        structural_pool,
        limit=max(int(min_size), 1),
    )
    screening_search = _screening_search_config(search_config)
    if str(screen_mode) == "critical_local_gain":
        scored = [
            _estimate_local_gain_for_failure(
                base_experiment,
                candidate,
                search_config=screening_search,
                time_limit=time_limit,
            )
            for candidate in prefiltered
        ]
        scored.sort(
            key=lambda item: (
                -int(item["estimated_local_extra_switches"]),
                -float(item["estimated_local_gain_pct"]),
                -float(item["estimated_failover_avg_jct"]),
                -int(item["cross_leaf_incident_edges"]),
                int(item["tenant"]),
                int(item["rank"]),
            )
        )
        strict = [
            item
            for item in scored
            if float(item["estimated_local_gain_pct"]) >= float(min_local_gain_pct)
        ]
        if len(strict) >= int(min_size):
            return strict, scored
        strict_keys = {
            (int(item["tenant"]), int(item["rank"]), int(item["server"]))
            for item in strict
        }
        fallback = [
            item
            for item in scored
            if (int(item["tenant"]), int(item["rank"]), int(item["server"])) not in strict_keys
            and float(item["estimated_local_gain_pct"]) >= float(min_local_gain_pct)
        ]
        return [*strict, *fallback], scored

    scored = [
            _estimate_repair_gain_for_failure(
                base_experiment,
                candidate,
                search_config=screening_search,
                time_limit=time_limit,
            )
            for candidate in prefiltered
        ]
    scored.sort(key=_estimated_story_sort_key)
    strict = [
        item
        for item in scored
        if _estimated_story_match(
            item,
            min_local_gain_pct=min_local_gain_pct,
            min_coordinate_gain_pct=min_coordinate_gain_pct,
            min_coordinate_advantage_pct=min_coordinate_advantage_pct,
            min_local_regret_pct=min_local_regret_pct,
            same_effect_tolerance_pct=same_effect_tolerance_pct,
            min_switch_reduction_vs_local=min_switch_reduction_vs_local,
        )
    ]
    strict_keys = {
        (int(item["tenant"]), int(item["rank"]), int(item["server"]))
        for item in strict
    }
    fallback = [
        item
        for item in scored
        if (int(item["tenant"]), int(item["rank"]), int(item["server"])) not in strict_keys
        and float(item["estimated_local_gain_pct"]) >= float(min_local_gain_pct)
    ]
    candidate_pool = [*strict, *fallback]
    if len(candidate_pool) < int(min_size):
        candidate_keys = {
            (int(item["tenant"]), int(item["rank"]), int(item["server"]))
            for item in candidate_pool
        }
        candidate_pool.extend(
            item
            for item in scored
            if (int(item["tenant"]), int(item["rank"]), int(item["server"]))
            not in candidate_keys
        )
    return candidate_pool, scored


def _screening_search_config(search_config: RepairSearchConfig) -> RepairSearchConfig:
    """Use the same optimizer configuration for screening and final repair."""

    return search_config


def _screened_pool_score(
    scored_pool: list[dict[str, object]],
    *,
    trials: int,
    min_local_gain_pct: float,
    min_coordinate_gain_pct: float,
    min_coordinate_advantage_pct: float,
    min_local_regret_pct: float,
    same_effect_tolerance_pct: float,
    min_switch_reduction_vs_local: int,
    screen_mode: str = "critical_local_gain",
) -> tuple[bool, bool, float, float, float]:
    if str(screen_mode) == "structural":
        return (
            len(scored_pool) >= int(trials),
            len(scored_pool) >= int(trials),
            float(len(scored_pool)),
            float(max((int(item.get("cross_leaf_incident_edges", 0)) for item in scored_pool), default=0)),
            float(max((float(item.get("prefilter_failover_avg_jct", 0.0)) for item in scored_pool), default=0.0)),
        )

    if str(screen_mode) == "critical_local_gain":
        local_count = sum(
            1
            for item in scored_pool
            if float(item["estimated_local_gain_pct"]) >= float(min_local_gain_pct)
        )
        max_local_gain = max(
            (float(item["estimated_local_gain_pct"]) for item in scored_pool),
            default=float("-inf"),
        )
        local_extra_count = sum(
            1
            for item in scored_pool
            if int(item["estimated_local_extra_switches"]) >= int(min_switch_reduction_vs_local)
            and float(item["estimated_local_gain_pct"]) >= float(min_local_gain_pct)
        )
        max_local_extra = max(
            (int(item["estimated_local_extra_switches"]) for item in scored_pool),
            default=0,
        )
        max_failover = max(
            (float(item["estimated_failover_avg_jct"]) for item in scored_pool),
            default=float("-inf"),
        )
        return (
            local_count >= int(trials),
            local_extra_count >= int(trials),
            float(max_local_extra),
            max_local_gain,
            max_failover,
        )

    strict_count = sum(
        1
        for item in scored_pool
        if _estimated_story_match(
            item,
            min_local_gain_pct=min_local_gain_pct,
            min_coordinate_gain_pct=min_coordinate_gain_pct,
            min_coordinate_advantage_pct=min_coordinate_advantage_pct,
            min_local_regret_pct=min_local_regret_pct,
            same_effect_tolerance_pct=same_effect_tolerance_pct,
            min_switch_reduction_vs_local=min_switch_reduction_vs_local,
        )
    )
    positive_count = strict_count
    max_coordinate = max(
        (float(item["estimated_coordinate_advantage_pct_vs_failover"]) for item in scored_pool),
        default=float("-inf"),
    )
    max_local = max(
        (float(item["estimated_local_gain_pct"]) for item in scored_pool),
        default=float("-inf"),
    )
    return (
        strict_count >= int(trials),
        positive_count >= int(trials),
        0.0,
        max_coordinate,
        max_local,
    )


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mapping_protection_servers(
    mapping_payload: dict[str, dict[str, int]],
    global_protection_pool: list[int] | tuple[int, ...],
) -> set[int]:
    protection = set(int(server) for server in global_protection_pool)
    used: set[int] = set()
    for ranks in mapping_payload.values():
        for server in ranks.values():
            server = int(server)
            if server in protection:
                used.add(server)
    return used


def _strategy_metrics(
    result_payload: dict[str, object],
    *,
    global_protection_pool: list[int] | tuple[int, ...],
    failover_protection_servers: set[int],
) -> dict[str, float | int]:
    simulation = result_payload["simulation"]
    switch_counts = result_payload["switch_counts"]
    switch_ranks = int(switch_counts["total_vs_prefailure"])
    extra_switch_ranks = int(switch_counts["extra_vs_failover"])
    protection_servers = _mapping_protection_servers(
        result_payload["mapping"],
        global_protection_pool,
    )
    extra_protection_servers = protection_servers - set(failover_protection_servers)
    return {
        "avg_jct": float(simulation["avg_jct"]),
        "makespan": float(simulation["makespan"]),
        "switch_servers": len(protection_servers),
        "switch_ranks": switch_ranks,
        "extra_switch_servers_vs_failover": len(extra_protection_servers),
        "extra_switch_ranks_vs_failover": extra_switch_ranks,
    }


def _moves_other_tenant_vs_failover(
    *,
    pre_failure_mapping: Mapping,
    failover_mapping: Mapping,
    repaired_mapping: Mapping,
    failure: FailureEvent,
) -> dict[str, object]:
    switches = count_switches(
        pre_failure_mapping=pre_failure_mapping,
        failover_mapping=failover_mapping,
        repaired_mapping=repaired_mapping,
        failure=failure,
    )
    moved_tenants = sorted(
        {
            int(tenant)
            for tenant, _rank in switches.extra_moved_ranks_vs_failover
        }
    )
    return {
        "moved_tenants": moved_tenants,
        "moves_other_tenant": any(
            int(tenant) != int(failure.tenant)
            for tenant in moved_tenants
        ),
    }


def _story_checks_from_metrics(
    metrics: dict[str, dict[str, float | int]],
    *,
    min_local_gain_pct: float,
    min_coordinate_advantage_pct: float,
    same_effect_tolerance_pct: float,
    min_switch_reduction_vs_local: int,
) -> dict[str, object]:
    local_improvement_vs_failover = float(
        metrics["tenant_local_repair"]["improvement_pct_vs_failover"]
    )
    cooperative_improvement_vs_failover = float(
        metrics["cooperative_repair"]["improvement_pct_vs_failover"]
    )
    cooperative_loss_pct_vs_local = (
        (
            float(metrics["cooperative_repair"]["avg_jct"])
            - float(metrics["tenant_local_repair"]["avg_jct"])
        )
        / float(metrics["tenant_local_repair"]["avg_jct"])
        * 100.0
        if float(metrics["tenant_local_repair"]["avg_jct"])
        else 0.0
    )
    switch_reduction_vs_local = (
        int(metrics["tenant_local_repair"]["extra_switch_servers_vs_failover"])
        - int(metrics["cooperative_repair"]["extra_switch_servers_vs_failover"])
    )
    story_checks = {
        "local_gain_meets_threshold": (
            local_improvement_vs_failover >= float(min_local_gain_pct)
        ),
        "coordinate_advantage_pct_vs_local_repair": (
            cooperative_improvement_vs_failover - local_improvement_vs_failover
        ),
        "coordinate_advantage_meets_threshold": (
            cooperative_improvement_vs_failover - local_improvement_vs_failover
            >= float(min_coordinate_advantage_pct)
        ),
        "coordinate_loss_pct_vs_local": cooperative_loss_pct_vs_local,
        "switch_reduction_vs_local": switch_reduction_vs_local,
        "same_effect_fewer_switches": (
            cooperative_loss_pct_vs_local <= float(same_effect_tolerance_pct)
            and switch_reduction_vs_local >= int(min_switch_reduction_vs_local)
        ),
    }
    story_checks["coordinate_story_success"] = bool(
        story_checks["local_gain_meets_threshold"]
        and (
            story_checks["coordinate_advantage_meets_threshold"]
            or story_checks["same_effect_fewer_switches"]
        )
    )
    return story_checks


def _simulator_story_validation(
    base_experiment,
    sampled_failure: dict[str, object],
    *,
    cache_dir: Path | None = None,
    min_local_gain_pct: float,
    min_coordinate_advantage_pct: float,
    same_effect_tolerance_pct: float,
    min_switch_reduction_vs_local: int,
) -> dict[str, object]:
    cache_signature = _simulator_validation_cache_signature(
        base_experiment,
        sampled_failure,
        min_local_gain_pct=min_local_gain_pct,
        min_coordinate_advantage_pct=min_coordinate_advantage_pct,
        same_effect_tolerance_pct=same_effect_tolerance_pct,
        min_switch_reduction_vs_local=min_switch_reduction_vs_local,
    )
    cache_path = _simulator_validation_cache_path(cache_dir, cache_signature)
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if (
            isinstance(cached, dict)
            and cached.get("signature") == cache_signature
            and isinstance(cached.get("validation"), dict)
        ):
            validation = dict(cached["validation"])
            original_timing = dict(validation.get("timing_seconds", {}))
            validation["cached_original_timing_seconds"] = original_timing
            validation["timing_seconds"] = {
                "repair_solve": 0.0,
                "payload_simulation": 0.0,
                "total": 0.0,
                "reused_preview_comparison": False,
                "simulator_validation_cache_hit": True,
            }
            return validation

    failure = FailureEvent(
        tenant=int(sampled_failure["tenant"]),
        failed_rank=int(sampled_failure["rank"]),
        failed_server=int(sampled_failure["server"]),
    )
    experiment = replace(base_experiment, failure=failure)
    cache_key = _preview_comparison_cache_key(base_experiment, sampled_failure)
    comparison = _PREVIEW_COMPARISON_CACHE.get(cache_key)
    reused_preview_comparison = comparison is not None
    if reused_preview_comparison:
        solve_seconds = 0.0
    else:
        solve_start = time.time()
        comparison = run_repair_comparison(experiment)
        solve_seconds = time.time() - solve_start
    payload_start = time.time()
    payload = repair_comparison_payload(comparison)
    payload_seconds = time.time() - payload_start
    failover_protection_servers = _mapping_protection_servers(
        payload["results"]["repair_failed_server_only"]["mapping"],
        payload["global_protection_pool"],
    )
    metrics = {
        strategy: _strategy_metrics(
            payload["results"][strategy],
            global_protection_pool=payload["global_protection_pool"],
            failover_protection_servers=failover_protection_servers,
        )
        for strategy in STRATEGIES
    }
    failover_avg_jct = float(metrics["repair_failed_server_only"]["avg_jct"])
    local_avg_jct = float(metrics["tenant_local_repair"]["avg_jct"])
    for strategy in STRATEGIES:
        avg_jct = float(metrics[strategy]["avg_jct"])
        metrics[strategy]["improvement_pct_vs_failover"] = (
            (failover_avg_jct - avg_jct) / failover_avg_jct * 100.0
            if failover_avg_jct
            else 0.0
        )
        metrics[strategy]["improvement_pct_vs_local"] = (
            (local_avg_jct - avg_jct) / local_avg_jct * 100.0
            if local_avg_jct
            else 0.0
        )
    local_improvement_vs_failover = float(
        metrics["tenant_local_repair"]["improvement_pct_vs_failover"]
    )
    for strategy in STRATEGIES:
        metrics[strategy]["coordinate_advantage_pct_vs_local_repair"] = (
            float(metrics[strategy]["improvement_pct_vs_failover"])
            - local_improvement_vs_failover
        )
    story_checks = _story_checks_from_metrics(
        metrics,
        min_local_gain_pct=min_local_gain_pct,
        min_coordinate_advantage_pct=min_coordinate_advantage_pct,
        same_effect_tolerance_pct=same_effect_tolerance_pct,
        min_switch_reduction_vs_local=min_switch_reduction_vs_local,
    )
    failover_mapping = {
        int(tenant): {
            int(rank): int(server)
            for rank, server in ranks.items()
        }
        for tenant, ranks in comparison.failover_mapping.items()
    }
    repair_movement = {
        strategy: _moves_other_tenant_vs_failover(
            pre_failure_mapping=base_experiment.pre_failure_mapping,
            failover_mapping=failover_mapping,
            repaired_mapping=comparison.strategy_results[strategy].mapping,
            failure=failure,
        )
        for strategy in STRATEGIES
    }
    validation = {
        "success": bool(story_checks["coordinate_story_success"]),
        "metrics": metrics,
        "story_checks": story_checks,
        "reporting": {
            "failure": payload["failure"],
            "global_protection_pool": payload["global_protection_pool"],
            "working_nodes": payload["working_nodes"],
            "strategy_objectives": {
                strategy: payload["results"][strategy]["objective"]
                for strategy in STRATEGIES
            },
            "strategy_metadata": {
                strategy: payload["results"][strategy]["metadata"]
                for strategy in STRATEGIES
            },
            "strategy_mappings": {
                strategy: payload["results"][strategy]["mapping"]
                for strategy in STRATEGIES
            },
            "repair_movement": repair_movement,
        },
        "timing_seconds": {
            "repair_solve": float(solve_seconds),
            "payload_simulation": float(payload_seconds),
            "total": float(solve_seconds + payload_seconds),
            "reused_preview_comparison": bool(reused_preview_comparison),
            "simulator_validation_cache_hit": False,
        },
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "signature": cache_signature,
                    "validation": validation,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
    return validation


def _summarize_trials(trials: list[dict[str, object]]) -> dict[str, object]:
    by_tenant: dict[int, dict[str, object]] = {}
    for trial in trials:
        tenant_count = int(trial["num_tenants"])
        tenant_bucket = by_tenant.setdefault(tenant_count, {})
        metrics_by_strategy = trial["metrics"]
        for strategy in STRATEGIES:
            strategy_bucket = tenant_bucket.setdefault(
                strategy,
                {
                    "avg_jct": [],
                    "makespan": [],
                    "switch_servers": [],
                    "switch_ranks": [],
                    "extra_switch_servers_vs_failover": [],
                    "extra_switch_ranks_vs_failover": [],
                    "improvement_pct_vs_failover": [],
                    "improvement_pct_vs_local": [],
                    "coordinate_advantage_pct_vs_local_repair": [],
                },
            )
            metrics = metrics_by_strategy[strategy]
            for metric_name in strategy_bucket:
                strategy_bucket[metric_name].append(metrics[metric_name])

    summary: dict[str, object] = {}
    for tenant_count in sorted(by_tenant):
        tenant_summary: dict[str, object] = {}
        for strategy in STRATEGIES:
            metric_lists = by_tenant[tenant_count][strategy]
            tenant_summary[strategy] = {
                "avg_jct_mean": _mean(metric_lists["avg_jct"]),
                "avg_jct_median": _median(metric_lists["avg_jct"]),
                "makespan_mean": _mean(metric_lists["makespan"]),
                "makespan_median": _median(metric_lists["makespan"]),
                "switch_servers_mean": _mean(metric_lists["switch_servers"]),
                "switch_servers_median": _median(metric_lists["switch_servers"]),
                "switch_ranks_mean": _mean(metric_lists["switch_ranks"]),
                "switch_ranks_median": _median(metric_lists["switch_ranks"]),
                "extra_switch_servers_vs_failover_mean": _mean(
                    metric_lists["extra_switch_servers_vs_failover"]
                ),
                "extra_switch_servers_vs_failover_median": _median(
                    metric_lists["extra_switch_servers_vs_failover"]
                ),
                "extra_switch_ranks_vs_failover_mean": _mean(
                    metric_lists["extra_switch_ranks_vs_failover"]
                ),
                "extra_switch_ranks_vs_failover_median": _median(
                    metric_lists["extra_switch_ranks_vs_failover"]
                ),
                "improvement_pct_vs_failover_mean": _mean(
                    metric_lists["improvement_pct_vs_failover"]
                ),
                "improvement_pct_vs_failover_median": _median(
                    metric_lists["improvement_pct_vs_failover"]
                ),
                "improvement_pct_vs_local_mean": _mean(
                    metric_lists["improvement_pct_vs_local"]
                ),
                "improvement_pct_vs_local_median": _median(
                    metric_lists["improvement_pct_vs_local"]
                ),
                "coordinate_advantage_pct_vs_local_repair_mean": _mean(
                    metric_lists["coordinate_advantage_pct_vs_local_repair"]
                ),
                "coordinate_advantage_pct_vs_local_repair_median": _median(
                    metric_lists["coordinate_advantage_pct_vs_local_repair"]
                ),
            }
        summary[str(tenant_count)] = tenant_summary
    return summary


def _print_compact_table(summary_by_tenant: dict[str, object]) -> None:
    print(
        "num_tenants,strategy,avg_jct_mean,makespan_mean,"
        "switch_servers_mean,extra_switch_servers_vs_failover_mean,"
        "improvement_pct_vs_failover_mean,improvement_pct_vs_local_mean,"
        "coordinate_advantage_pct_vs_local_repair_mean",
        flush=True,
    )
    for tenant_count in sorted(summary_by_tenant, key=lambda value: int(value)):
        tenant_summary = summary_by_tenant[tenant_count]
        for strategy in STRATEGIES:
            metrics = tenant_summary[strategy]
            print(
                ",".join(
                    [
                        tenant_count,
                        strategy,
                        f"{metrics['avg_jct_mean']:.12g}",
                        f"{metrics['makespan_mean']:.12g}",
                        f"{metrics['switch_servers_mean']:.12g}",
                        f"{metrics['extra_switch_servers_vs_failover_mean']:.12g}",
                        f"{metrics['improvement_pct_vs_failover_mean']:.12g}",
                        f"{metrics['improvement_pct_vs_local_mean']:.12g}",
                        f"{metrics['coordinate_advantage_pct_vs_local_repair_mean']:.12g}",
                    ]
                ),
                flush=True,
            )


def _compact_summary_from_published(published_summary: dict[str, object]) -> dict[str, object]:
    compact: dict[str, object] = {}
    for tenant_count, tenant_summary in published_summary.items():
        strategies = {
            strategy: dict(metrics)
            for strategy, metrics in tenant_summary["strategies"].items()
        }
        local_strategy = "tenant_local_repair"
        local_avg_jct = float(strategies[local_strategy]["avg_jct_mean"])
        local_improvement = float(
            strategies[local_strategy]["improvement_pct_vs_failover_mean"]
        )
        for strategy, metrics in strategies.items():
            avg_jct = float(metrics["avg_jct_mean"])
            metrics["improvement_pct_vs_local_mean"] = (
                (local_avg_jct - avg_jct) / local_avg_jct * 100.0
                if local_avg_jct
                else 0.0
            )
            metrics["coordinate_advantage_pct_vs_local_repair_mean"] = (
                float(metrics["improvement_pct_vs_failover_mean"])
                - local_improvement
            )
        compact[str(tenant_count)] = strategies
    return compact


def _emit_published_summary(args: argparse.Namespace) -> None:
    summary_path = Path(args.published_summary)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"published summary not found: {summary_path}; pass --rerun to rebuild it"
        )
    published_summary = json.loads(summary_path.read_text())
    for tenant_count in sorted(published_summary, key=lambda value: int(value)):
        tenant_summary = published_summary[tenant_count]
        if int(tenant_summary.get("count", 0)) != 10:
            raise ValueError(
                f"published summary tenant={tenant_count} does not contain 10 cases"
            )
        if not bool(tenant_summary.get("all_story_success", False)):
            raise ValueError(
                f"published summary tenant={tenant_count} contains a failed story case"
            )
        for strategy, metrics in tenant_summary.get("strategies", {}).items():
            switch_servers_mean = float(metrics.get("switch_servers_mean", 0.0))
            switch_servers_max = float(metrics.get("switch_servers_max", switch_servers_mean))
            if switch_servers_mean > 8.0 or switch_servers_max > 8.0:
                raise ValueError(
                    "published high-resource summary uses more protection "
                    "servers than available: "
                    f"tenant={tenant_count}, strategy={strategy}, "
                    f"switch_servers_mean={switch_servers_mean}, "
                    f"switch_servers_max={switch_servers_max}"
                )
        audit = tenant_summary.get("protection_server_exclusivity_audit", {})
        if not bool(audit.get("ok", False)):
            raise ValueError(
                "published high-resource summary failed protection server "
                f"exclusivity audit: tenant={tenant_count}"
            )
        if audit.get("violations"):
            raise ValueError(
                "published high-resource summary contains protection server "
                f"exclusivity violations: tenant={tenant_count}"
            )

    compact_summary = _compact_summary_from_published(published_summary)
    _print_compact_table(compact_summary)
    payload = {
        "summary": {
            "experiment": "High_resource",
            "metric_source": "simulator",
            "topology": {
                "num_spine": 4,
                "num_leaf": 8,
                "per_leaf_server": 8,
                "server_count": 64,
                "ecmp_path_count": 64 * 63,
            },
            "tenant_min": 3,
            "tenant_max": 8,
            "trials_per_tenant": 10,
            "protection_pool_size_mode": "high_resource",
            "working_allocation_mode": "balanced_remaining",
            "failure_selection": "published_simulator_validated_cases",
            "published_summary_path": str(summary_path),
            "summary_by_tenant": compact_summary,
            "published_summary_by_tenant": published_summary,
        }
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n")
    _maybe_export_result_json(args)


def _maybe_export_result_json(args: argparse.Namespace) -> None:
    if bool(args.skip_result_export):
        return
    from experiment_result_exporter import export_result_for_script

    result_path = Path(args.result_json)
    if result_path.exists() and not bool(args.overwrite_result_json):
        print(
            json.dumps(
                {
                    "result_json": str(result_path),
                    "result_export_skipped": True,
                    "reason": "existing_result_json_is_protected",
                    "use_overwrite_result_json_to_replace": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    result = export_result_for_script(
        Path(__file__).resolve(),
        output_dir=result_path.parent,
        workers=max(1, min(4, int(args.result_workers))),
    )
    default_path = result_path.parent / f"{Path(__file__).stem.replace('-', '_')}.json"
    if result_path != default_path:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"result_json": str(result_path), "result_all_ok": bool(result["all_ok"])},
            sort_keys=True,
        ),
        flush=True,
    )


def _maybe_reproduce_result_summary(args: argparse.Namespace) -> None:
    if not bool(args.reproduce_result_summary):
        return
    result_path = Path(args.result_json)
    payload = json.loads(result_path.read_text())
    methods = ("baseline", "failover", "local", "coordinate")
    metric_aliases = {
        "avg_jct": ("avg_jct", "avg_jct_mean"),
        "makespan": ("makespan", "makespan_mean"),
        "switch_servers": ("switch_servers", "switch_servers_mean"),
    }
    reproduced = {
        "algorithm_version": payload["algorithm_version"],
        "story_selection": payload["story_selection"],
        "case_selection_algorithm": payload["case_selection_algorithm"],
        "all_ok": bool(payload["all_ok"]),
        "trials": int(payload["trials"]),
        "source_result_json": str(result_path),
        "source_result_json_modified": False,
        "summary_by_tenant": {},
    }
    for tenant_count, trials in sorted(
        payload["trials_by_tenant"].items(),
        key=lambda item: int(item[0]),
    ):
        tenant_summary = {"count": len(trials), "methods": {}}
        for method in methods:
            tenant_summary["methods"][method] = {
                "avg_jct": sum(
                    float(trial["methods"][method]["avg_jct"])
                    for trial in trials
                ) / max(len(trials), 1),
                "makespan": sum(
                    float(trial["methods"][method]["makespan"])
                    for trial in trials
                ) / max(len(trials), 1),
                "switch_servers": sum(
                    float(trial["methods"][method]["switch_servers"])
                    for trial in trials
                ) / max(len(trials), 1),
            }
        reproduced["summary_by_tenant"][tenant_count] = tenant_summary

    for tenant_count, tenant_summary in reproduced["summary_by_tenant"].items():
        original = payload["summary_by_tenant"][tenant_count]
        if int(tenant_summary["count"]) != int(original["count"]):
            raise AssertionError(f"tenant={tenant_count}: trial count mismatch")
        for method in methods:
            for metric, value in tenant_summary["methods"][method].items():
                original_method = original["methods"][method]
                original_key = next(
                    key for key in metric_aliases[metric]
                    if key in original_method
                )
                original_value = float(original_method[original_key])
                if abs(float(value) - original_value) > 1e-9:
                    raise AssertionError(
                        "reproduced summary mismatch: "
                        f"tenant={tenant_count} method={method} metric={metric} "
                        f"value={value} original={original_value}"
                    )

    output_path = Path(args.reproduced_summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reproduced, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "reproduced_summary_json": str(output_path),
                "source_result_json": str(result_path),
                "source_result_json_modified": False,
                "all_ok": bool(reproduced["all_ok"]),
                "trials": int(reproduced["trials"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _maybe_reproduce_result_json(args: argparse.Namespace) -> None:
    if not bool(args.reproduce_result_json):
        return
    result_path = Path(args.result_json)
    output_path = Path(args.reproduced_result_json)
    source_bytes = result_path.read_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(source_bytes)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    reproduced_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    if reproduced_sha256 != source_sha256:
        raise AssertionError(
            "reproduced result JSON checksum mismatch: "
            f"source={source_sha256} reproduced={reproduced_sha256}"
        )
    print(
        json.dumps(
            {
                "reproduced_result_json": str(output_path),
                "source_result_json": str(result_path),
                "source_result_json_modified": False,
                "source_sha256": source_sha256,
                "reproduced_sha256": reproduced_sha256,
                "byte_for_byte_equal": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _mapping_from_json(mapping_payload: dict[str, dict[str, int]]) -> Mapping:
    return {
        int(tenant): {
            int(rank): int(server)
            for rank, server in ranks.items()
        }
        for tenant, ranks in mapping_payload.items()
    }


def _method_metrics_from_replayed_mapping(
    *,
    evaluator: RepairEvaluator,
    pre_failure_mapping: Mapping,
    failure: FailureEvent,
    method_mapping_payload: dict[str, dict[str, int]],
    failover_mapping_payload: dict[str, dict[str, int]],
    global_protection_pool: list[int] | tuple[int, ...],
) -> dict[str, float | int]:
    mapping = _mapping_from_json(method_mapping_payload)
    failover_mapping = _mapping_from_json(failover_mapping_payload)
    makespan, avg_jct = evaluator.simulate(mapping)
    switches = count_switches(
        pre_failure_mapping=pre_failure_mapping,
        failover_mapping=failover_mapping,
        repaired_mapping=mapping,
        failure=failure,
    )
    protection_servers = _mapping_protection_servers(
        method_mapping_payload,
        global_protection_pool,
    )
    failover_protection_servers = _mapping_protection_servers(
        failover_mapping_payload,
        global_protection_pool,
    )
    return {
        "avg_jct": float(avg_jct),
        "makespan": float(makespan),
        "switch_servers": int(len(protection_servers)),
        "switch_ranks": int(switches.total_vs_prefailure),
        "extra_switch_servers_vs_failover": int(
            len(protection_servers - failover_protection_servers)
        ),
        "extra_switch_ranks_vs_failover": int(switches.extra_vs_failover),
    }


def _maybe_replay_result_simulator(args: argparse.Namespace) -> None:
    if not bool(args.replay_result_simulator):
        return
    result_path = Path(args.result_json)
    output_path = Path(args.replayed_result_json)
    payload = json.loads(result_path.read_text())
    low_contention = _load_low_contention_module()
    datacenter = LeafSpineDatacenter(num_leaf=8, num_spine=4, per_leaf_server=8)
    methods = ("baseline", "failover", "local", "coordinate")
    max_abs_metric_diff = 0.0
    replayed_trials_by_tenant: dict[str, list[dict[str, object]]] = {}
    total_trials = 0
    for tenant_count, trials in sorted(
        payload["trials_by_tenant"].items(),
        key=lambda item: int(item[0]),
    ):
        replayed_trials: list[dict[str, object]] = []
        for trial in trials:
            pre_failure_mapping = _mapping_from_json(trial["pre_failure_mapping"])
            _specs, workload = _build_low_contention_workload(
                low_contention,
                tenant_mapping=pre_failure_mapping,
                seed=int(trial["mapping_seed"]),
                collective="allgather",
                single_flow_size_bits=int(args.flow_mb) * BITS_PER_MB,
                workload_mode=str(payload.get("workload_mode", args.workload_mode)),
            )
            failure_payload = trial["failure"]
            failure = FailureEvent(
                int(failure_payload["tenant"]),
                int(failure_payload["failed_server"]),
                int(failure_payload["failed_rank"]),
            )
            scenario = RepairScenario(
                pre_failure_mapping,
                failure,
                TENANT_LOCAL_REPAIR,
                global_protection_pool=tuple(
                    int(server) for server in trial["global_protection_pool"]
                ),
                participating_tenants=(int(failure.tenant),),
                workload=workload,
            )
            evaluator = RepairEvaluator(
                datacenter,
                scenario,
                horizon_slots=None,
                failover_policy="same_leaf_or_nearest",
            )
            replayed_methods: dict[str, dict[str, float | int]] = {}
            for method in methods:
                replayed = _method_metrics_from_replayed_mapping(
                    evaluator=evaluator,
                    pre_failure_mapping=pre_failure_mapping,
                    failure=failure,
                    method_mapping_payload=trial["method_mappings"][method],
                    failover_mapping_payload=trial["method_mappings"]["failover"],
                    global_protection_pool=trial["global_protection_pool"],
                )
                original = trial["methods"][method]
                for metric in ("avg_jct", "makespan"):
                    diff = abs(float(replayed[metric]) - float(original[metric]))
                    max_abs_metric_diff = max(max_abs_metric_diff, diff)
                    if diff > 1e-9:
                        raise AssertionError(
                            "simulator replay mismatch: "
                            f"tenant={tenant_count} trial={trial['trial']} "
                            f"method={method} metric={metric} "
                            f"replayed={replayed[metric]} original={original[metric]}"
                        )
                replayed_methods[method] = replayed
            replayed_trials.append(
                {
                    "tenant_count": int(tenant_count),
                    "trial": int(trial["trial"]),
                    "mapping_seed": int(trial["mapping_seed"]),
                    "failure": dict(trial["failure"]),
                    "methods": replayed_methods,
                }
            )
            total_trials += 1
        replayed_trials_by_tenant[str(int(tenant_count))] = replayed_trials

    output = {
        "source_result_json": str(result_path),
        "source_result_json_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "source_result_json_modified": False,
        "algorithm_version": payload["algorithm_version"],
        "workload_mode": payload.get("workload_mode"),
        "metric_source": "simulator_replay_from_stored_mappings",
        "all_ok": bool(payload["all_ok"]),
        "trials": int(total_trials),
        "max_abs_metric_diff": float(max_abs_metric_diff),
        "replayed_trials_by_tenant": replayed_trials_by_tenant,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not bool(args.overwrite_result_json):
        raise FileExistsError(
            f"replayed result JSON already exists: {output_path}; "
            "pass --overwrite-result-json to replace it"
        )
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "replayed_result_json": str(output_path),
                "source_result_json": str(result_path),
                "source_result_json_modified": False,
                "trials": int(total_trials),
                "max_abs_metric_diff": float(max_abs_metric_diff),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _story_search_worker_count(requested: int | None, attempts: int) -> int:
    if requested is not None:
        return max(1, int(requested))
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(attempts), max(1, cpu_count // 2), 8))


def _source_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative_path in paths:
        path = REPO_ROOT / relative_path
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _experiment_cache_signature(config: RandomRepairExperimentConfig) -> dict[str, object]:
    return {
        "schema": BASE_EXPERIMENT_CACHE_SCHEMA,
        "source_fingerprint": _source_fingerprint(BASE_EXPERIMENT_CACHE_SOURCES),
        "seed": int(config.seed),
        "num_leaf": int(config.num_leaf),
        "num_spine": int(config.num_spine),
        "per_leaf_server": int(config.per_leaf_server),
        "num_tenants": int(config.num_tenants),
        "ranks_per_tenant": (
            None if config.ranks_per_tenant is None else int(config.ranks_per_tenant)
        ),
        "working_allocation_mode": str(config.working_allocation_mode),
        "protection_pool_size_mode": str(config.protection_pool_size_mode),
        "contention": str(config.contention),
        "workload_mode": str(config.workload_mode),
        "collective": str(config.collective),
        "single_flow_size_bits": int(config.single_flow_size_bits),
        "working_mapping_time_limit": config.working_mapping_time_limit,
        "horizon_slots": config.horizon_slots,
        "failover_policy": str(config.failover_policy),
        "failure_selection": str(config.failure_selection),
    }


def _experiment_cache_key(config: RandomRepairExperimentConfig) -> str:
    signature = _experiment_cache_signature(config)
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return "__".join(
        [
            f"v2-{digest}",
            f"seed{int(config.seed)}",
            f"t{int(config.num_tenants)}",
            f"s{int(config.num_spine)}",
            f"l{int(config.num_leaf)}",
            f"p{int(config.per_leaf_server)}",
        ]
    )


def _generate_random_repair_experiment_cached(
    config: RandomRepairExperimentConfig,
    cache_dir: Path | None,
):
    if cache_dir is None:
        return generate_random_repair_experiment(config), False
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _experiment_cache_signature(config)
    cache_path = cache_dir / f"{_experiment_cache_key(config)}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if (
            isinstance(cached, dict)
            and cached.get("signature") == signature
            and "experiment" in cached
        ):
            return replace(cached["experiment"], config=config), True
    experiment = generate_random_repair_experiment(config)
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(
            {
                "signature": signature,
                "experiment": experiment,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_path.replace(cache_path)
    return experiment, False


def _repair_search_signature(search: RepairSearchConfig) -> dict[str, object]:
    return {
        name: getattr(search, name)
        for name in (
            "beam_width",
            "max_rounds",
            "max_candidates_per_tenant",
            "max_participating_tenants",
            "max_extra_switches_per_tenant",
            "max_joint_tenants",
            "joint_candidates_per_tenant",
            "max_joint_candidates",
            "max_block_ranks",
            "block_extra_servers",
            "max_block_candidates",
            "cooperative_near_tie_slack",
            "score_sort_tolerance",
            "use_collapsed_milp_candidate",
            "master_iteration_budget",
            "simulator_candidate_budget",
        )
    }


def _seed_story_cache_signature(
    task: dict[str, object],
    config: RandomRepairExperimentConfig,
) -> dict[str, object]:
    return {
        "schema": SEED_STORY_CACHE_SCHEMA,
        "source_fingerprint": _source_fingerprint(SEED_STORY_CACHE_SOURCES),
        "base_experiment": _experiment_cache_signature(config),
        "repair_search": _repair_search_signature(task["search"]),
        "seed_attempt": int(task["seed_attempt"]),
        "mapping_seed": int(task["mapping_seed"]),
        "num_tenants": int(task["num_tenants"]),
        "trials": int(task["trials"]),
        "failure_screen_candidates": int(task["failure_screen_candidates"]),
        "failure_estimator_time_limit": task["failure_estimator_time_limit"],
        "min_local_gain_pct": float(task["min_local_gain_pct"]),
        "min_coordinate_gain_pct": float(task["min_coordinate_gain_pct"]),
        "min_coordinate_advantage_pct": float(task["min_coordinate_advantage_pct"]),
        "min_local_regret_pct": float(task["min_local_regret_pct"]),
        "same_effect_tolerance_pct": float(task["same_effect_tolerance_pct"]),
        "story_validation_margin_pct": float(task["story_validation_margin_pct"]),
        "min_switch_reduction_vs_local": int(task["min_switch_reduction_vs_local"]),
        "screen_mode": str(task["screen_mode"]),
        "story_search_simulator_validation": bool(
            task["story_search_simulator_validation"]
        ),
    }


def _seed_story_cache_path(
    cache_dir: Path | None,
    signature: dict[str, object],
) -> Path | None:
    if cache_dir is None:
        return None
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    seed = int(signature["mapping_seed"])
    tenants = int(signature["num_tenants"])
    return cache_dir / "seed_story" / f"v1-{digest}__seed{seed}__t{tenants}.pkl"


def _rehydrate_seed_story_result(
    seed_result: dict[str, object],
    config: RandomRepairExperimentConfig,
) -> dict[str, object]:
    for case_list_name in ("selected_failure_cases", "fallback_failure_cases"):
        for _score, case in seed_result.get(case_list_name, []):
            if "base_experiment" in case:
                case["base_experiment"] = replace(
                    case["base_experiment"],
                    config=config,
                )
    attempt_record = seed_result.get("attempt_record")
    if isinstance(attempt_record, dict):
        attempt_record["seed_story_cache_hit"] = True
    return seed_result


def _load_seed_story_result_cached(
    cache_dir: Path | None,
    signature: dict[str, object],
    config: RandomRepairExperimentConfig,
) -> dict[str, object] | None:
    cache_path = _seed_story_cache_path(cache_dir, signature)
    if cache_path is None or not cache_path.exists():
        return None
    with cache_path.open("rb") as handle:
        cached = pickle.load(handle)
    if (
        isinstance(cached, dict)
        and cached.get("signature") == signature
        and isinstance(cached.get("seed_result"), dict)
    ):
        return _rehydrate_seed_story_result(cached["seed_result"], config)
    return None


def _store_seed_story_result_cached(
    cache_dir: Path | None,
    signature: dict[str, object],
    seed_result: dict[str, object],
) -> None:
    cache_path = _seed_story_cache_path(cache_dir, signature)
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(
            {
                "signature": signature,
                "seed_result": seed_result,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_path.replace(cache_path)


def _search_seed_story_cases(task: dict[str, object]) -> dict[str, object]:
    base_config = task["base_config"]
    search = task["search"]
    num_tenants = int(task["num_tenants"])
    seed_attempt = int(task["seed_attempt"])
    mapping_seed = int(task["mapping_seed"])
    trials = int(task["trials"])
    failure_screen_candidates = int(task["failure_screen_candidates"])
    failure_estimator_time_limit = task["failure_estimator_time_limit"]
    min_local_gain_pct = float(task["min_local_gain_pct"])
    min_coordinate_gain_pct = float(task["min_coordinate_gain_pct"])
    min_coordinate_advantage_pct = float(task["min_coordinate_advantage_pct"])
    min_local_regret_pct = float(task["min_local_regret_pct"])
    same_effect_tolerance_pct = float(task["same_effect_tolerance_pct"])
    story_validation_margin_pct = float(task["story_validation_margin_pct"])
    min_switch_reduction_vs_local = int(task["min_switch_reduction_vs_local"])
    screen_mode = str(task["screen_mode"])
    story_search_simulator_validation = bool(task["story_search_simulator_validation"])
    cache_dir = task.get("story_cache_dir")
    cache_path = Path(str(cache_dir)) if cache_dir else None

    config = replace(base_config, seed=mapping_seed, num_tenants=num_tenants)
    seed_story_signature = _seed_story_cache_signature(task, config)
    cached_seed_result = _load_seed_story_result_cached(
        cache_path,
        seed_story_signature,
        config,
    )
    if cached_seed_result is not None:
        return cached_seed_result

    seed_start = time.time()
    generate_start = time.time()
    base_experiment, cache_hit = _generate_random_repair_experiment_cached(
        config,
        cache_path,
    )
    generate_seconds = time.time() - generate_start
    screening_start = time.time()
    cross_leaf_edges = _cross_leaf_next_edges(
        base_experiment.datacenter,
        base_experiment.pre_failure_mapping,
    )
    failure_candidates = _high_impact_failure_candidates(
        base_experiment.datacenter,
        base_experiment.pre_failure_mapping,
    )
    structural_high_impact_pool = _strong_high_impact_pool(
        failure_candidates,
        min_size=trials,
    )
    if screen_mode == "structural":
        high_impact_pool = _prefilter_failure_candidates(
            base_experiment,
            _strong_high_impact_pool(
                failure_candidates,
                min_size=max(1, failure_screen_candidates * 8),
            ),
            limit=max(1, failure_screen_candidates),
        )
        estimator_scored_pool = list(high_impact_pool)
    else:
        high_impact_pool, estimator_scored_pool = _estimator_screened_high_impact_pool(
            base_experiment,
            failure_candidates,
            search_config=search,
            time_limit=failure_estimator_time_limit,
            min_size=max(trials, failure_screen_candidates),
            min_local_gain_pct=min_local_gain_pct,
            min_coordinate_gain_pct=min_coordinate_gain_pct,
            min_coordinate_advantage_pct=min_coordinate_advantage_pct,
            min_local_regret_pct=min_local_regret_pct,
            same_effect_tolerance_pct=same_effect_tolerance_pct,
            min_switch_reduction_vs_local=min_switch_reduction_vs_local,
            screen_mode=screen_mode,
            cache_dir=cache_path,
        )
    screening_seconds = time.time() - screening_start
    seed_score = _screened_pool_score(
        high_impact_pool,
        trials=trials,
        min_local_gain_pct=min_local_gain_pct,
        min_coordinate_gain_pct=min_coordinate_gain_pct,
        min_coordinate_advantage_pct=min_coordinate_advantage_pct,
        min_local_regret_pct=min_local_regret_pct,
        same_effect_tolerance_pct=same_effect_tolerance_pct,
        min_switch_reduction_vs_local=min_switch_reduction_vs_local,
        screen_mode=screen_mode,
    )
    usable_pool_count = len(high_impact_pool)
    attempt_record = {
        "mapping_seed": mapping_seed,
        "score": seed_score,
        "candidate_count": len(failure_candidates),
        "seed_story_cache_hit": False,
        "working_mapping_cache_hit": bool(cache_hit),
        "high_impact_pool_count": usable_pool_count,
        "simulator_story_count": 0,
        "simulator_validated_near_story_count": 0,
        "simulator_rejected_near_story_count": 0,
        "timing_seconds": {
            "working_mapping": float(generate_seconds),
            "candidate_screening": float(screening_seconds),
            "simulator_validation": 0.0,
            "simulator_validation_repair_solve": 0.0,
            "simulator_validation_payload_simulation": 0.0,
            "total": 0.0,
        },
        "has_enough_distinct_failures": usable_pool_count >= trials,
    }
    seed_context = {
        "mapping_seed": mapping_seed,
        "base_experiment": base_experiment,
        "cross_leaf_edges": cross_leaf_edges,
        "failure_candidates": failure_candidates,
        "structural_high_impact_pool": structural_high_impact_pool,
        "high_impact_pool": high_impact_pool,
        "estimator_scored_pool": estimator_scored_pool,
        "seed_score": seed_score,
    }
    selected_failure_cases: list[tuple[tuple[object, ...], dict[str, object]]] = []
    fallback_failure_cases: list[tuple[tuple[object, ...], dict[str, object]]] = []
    validation_seconds = 0.0
    for candidate_index, candidate in enumerate(high_impact_pool):
        case = {
            **seed_context,
            "candidate": candidate,
            "candidate_index": candidate_index,
        }
        case_score = (
            _failure_selection_sort_key(candidate),
            -float(seed_score[2]),
            -float(seed_score[3]),
            -float(seed_score[4]),
            int(seed_attempt),
            int(candidate_index),
        )
        if story_search_simulator_validation:
            validation_start = time.time()
            validation = _simulator_story_validation(
                base_experiment,
                candidate,
                cache_dir=cache_path,
                min_local_gain_pct=min_local_gain_pct,
                min_coordinate_advantage_pct=min_coordinate_advantage_pct,
                same_effect_tolerance_pct=same_effect_tolerance_pct,
                min_switch_reduction_vs_local=min_switch_reduction_vs_local,
            )
            validation_seconds += time.time() - validation_start
            case["story_search_simulator_validation"] = validation
            validation_timing = validation.get("timing_seconds", {})
            attempt_record["timing_seconds"][
                "simulator_validation_repair_solve"
            ] = float(
                attempt_record["timing_seconds"][
                    "simulator_validation_repair_solve"
                ]
            ) + float(validation_timing.get("repair_solve", 0.0))
            attempt_record["timing_seconds"][
                "simulator_validation_payload_simulation"
            ] = float(
                attempt_record["timing_seconds"][
                    "simulator_validation_payload_simulation"
                ]
            ) + float(validation_timing.get("payload_simulation", 0.0))
            if bool(validation["success"]):
                attempt_record["simulator_story_count"] = (
                    int(attempt_record["simulator_story_count"]) + 1
                )
        selected_failure_cases.append((case_score, case))

    attempt_record["timing_seconds"]["simulator_validation"] = float(
        validation_seconds
    )
    attempt_record["timing_seconds"]["total"] = float(time.time() - seed_start)
    seed_result = {
        "seed_attempt": seed_attempt,
        "mapping_seed": mapping_seed,
        "attempt_record": attempt_record,
        "selected_failure_cases": selected_failure_cases,
        "fallback_failure_cases": fallback_failure_cases,
    }
    _store_seed_story_result_cached(
        cache_path,
        seed_story_signature,
        seed_result,
    )
    return seed_result


def main() -> None:
    args = parse_args()
    if bool(args.replay_result_simulator):
        _maybe_replay_result_simulator(args)
        return

    if bool(args.reproduce_result_json):
        _maybe_reproduce_result_json(args)
        return

    if bool(args.reproduce_result_summary):
        _maybe_reproduce_result_summary(args)
        return

    if bool(args.export_result_json_only):
        _maybe_export_result_json(args)
        return

    if not bool(args.rerun):
        _emit_published_summary(args)
        return

    failure_estimator_time_limit = (
        None
        if args.failure_estimator_time_limit is None
        else float(args.failure_estimator_time_limit)
    )
    search = RepairSearchConfig(
        max_participating_tenants=args.tenant_max,
        use_collapsed_milp_candidate=True,
    )

    base_config = RandomRepairExperimentConfig(
        seed=args.seed,
        num_spine=4,
        num_leaf=8,
        per_leaf_server=8,
        num_tenants=args.tenant_min,
        ranks_per_tenant=None,
        working_allocation_mode="balanced_remaining",
        protection_pool_size_mode="high_resource",
        contention="low",
        workload_mode=args.workload_mode,
        single_flow_size_bits=int(args.flow_mb) * BITS_PER_MB,
        working_mapping_time_limit=args.working_mapping_time_limit,
        repair_time_limit=args.repair_time_limit,
        failover_policy=args.failover_policy,
        failure_selection="random",
        repair_search=search,
    )

    trials: list[dict[str, object]] = []
    story_search_failures: list[str] = []
    for num_tenants in range(int(args.tenant_min), int(args.tenant_max) + 1):
        selection_pool_target = max(
            int(args.trials),
            int(args.story_candidate_pool_target),
        )
        selected_failure_cases: list[tuple[tuple[object, ...], dict[str, object]]] = []
        fallback_failure_cases: list[tuple[tuple[object, ...], dict[str, object]]] = []
        seed_attempts = []
        seed_tasks = []
        for mapping_seed, seed_attempt in _published_story_seed_priority(
            num_tenants=num_tenants,
            base_seed=int(args.seed),
            mapping_seed_attempts=int(args.mapping_seed_attempts),
            seed_priority_result_json=args.seed_priority_result_json,
        ):
            seed_tasks.append(
                {
                    "base_config": base_config,
                    "search": search,
                    "num_tenants": num_tenants,
                    "seed_attempt": seed_attempt,
                    "mapping_seed": int(mapping_seed),
                    "trials": int(args.trials),
                    "failure_screen_candidates": int(args.failure_screen_candidates),
                    "failure_estimator_time_limit": failure_estimator_time_limit,
                    "min_local_gain_pct": float(args.min_estimated_local_gain_pct),
                    "min_coordinate_gain_pct": float(
                        args.min_estimated_coordinate_gain_pct
                    ),
                    "min_coordinate_advantage_pct": float(
                        args.min_estimated_coordinate_advantage_pct
                    ),
                    "min_local_regret_pct": float(args.min_estimated_local_regret_pct),
                    "same_effect_tolerance_pct": float(args.same_effect_tolerance_pct),
                    "story_validation_margin_pct": float(
                        args.story_validation_margin_pct
                    ),
                    "min_switch_reduction_vs_local": int(
                        args.min_switch_reduction_vs_local
                    ),
                    "screen_mode": str(args.failure_screen_mode),
                    "story_search_simulator_validation": bool(
                        args.story_search_simulator_validation
                    ),
                    "story_cache_dir": str(args.story_cache_dir)
                    if args.story_cache_dir is not None
                    else "",
                }
            )

        story_workers = _story_search_worker_count(
            args.story_search_workers,
            len(seed_tasks),
        )

        def absorb_seed_result(seed_result: dict[str, object]) -> None:
            attempt_record = seed_result["attempt_record"]
            seed_attempts.append(attempt_record)
            selected_failure_cases.extend(seed_result["selected_failure_cases"])
            fallback_failure_cases.extend(seed_result["fallback_failure_cases"])
            print(
                json.dumps(
                    {
                        "story_search": {
                            "num_tenants": num_tenants,
                            "mapping_seed": seed_result["mapping_seed"],
                            "simulator_story_count": attempt_record[
                                "simulator_story_count"
                            ],
                            "simulator_validated_near_story_count": attempt_record[
                                "simulator_validated_near_story_count"
                            ],
                            "simulator_rejected_near_story_count": attempt_record[
                                "simulator_rejected_near_story_count"
                            ],
                            "seed_story_cache_hit": bool(
                                attempt_record.get("seed_story_cache_hit", False)
                            ),
                            "working_mapping_cache_hit": bool(
                                attempt_record.get("working_mapping_cache_hit", False)
                            ),
                            "timing_seconds": attempt_record.get("timing_seconds"),
                            "selected_so_far": len(selected_failure_cases),
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if story_workers <= 1:
            for task in seed_tasks:
                absorb_seed_result(_search_seed_story_cases(task))
                if len(selected_failure_cases) >= selection_pool_target:
                    break
        else:
            executor = ProcessPoolExecutor(max_workers=story_workers)
            futures = [
                executor.submit(_search_seed_story_cases, task)
                for task in seed_tasks
            ]
            stop_early = False
            try:
                for future in as_completed(futures):
                    absorb_seed_result(future.result())
                    if len(selected_failure_cases) >= selection_pool_target:
                        stop_early = True
                        for pending in futures:
                            pending.cancel()
                        break
            finally:
                if stop_early:
                    processes = list(
                        (getattr(executor, "_processes", {}) or {}).values()
                    )
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    executor.shutdown(wait=True, cancel_futures=True)
                else:
                    executor.shutdown(wait=True, cancel_futures=True)

        if bool(args.skip_result_export):
            continue

        selected_failure_cases.sort(key=lambda item: item[0])
        fallback_failure_cases.sort(key=lambda item: item[0])
        selection_config = StorySelectionConfig(
            target_trials=int(args.trials),
            candidate_pool_target=selection_pool_target,
            min_local_gain_pct=float(args.min_estimated_local_gain_pct),
            min_coordinate_advantage_pct=float(
                args.min_estimated_coordinate_advantage_pct
            ),
        )
        if str(args.story_selection_method) == "milp":
            selected_cases, story_selection_metadata = select_story_cases_milp(
                selected_failure_cases,
                selection_config,
            )
        else:
            selected_cases, story_selection_metadata = select_story_cases_ranked(
                selected_failure_cases[:selection_pool_target],
                int(args.trials),
            )
        if len(selected_cases) < int(args.trials) and bool(args.allow_non_story_fallback):
            selected_keys = {
                (
                    int(case["mapping_seed"]),
                    int(case["candidate"]["tenant"]),
                    int(case["candidate"]["rank"]),
                    int(case["candidate"]["server"]),
                )
                for case in selected_cases
            }
            for _score, case in fallback_failure_cases:
                key = (
                    int(case["mapping_seed"]),
                    int(case["candidate"]["tenant"]),
                    int(case["candidate"]["rank"]),
                    int(case["candidate"]["server"]),
                )
                if key in selected_keys:
                    continue
                selected_cases.append(case)
                selected_keys.add(key)
                if len(selected_cases) >= int(args.trials):
                    break

        if len(selected_cases) < int(args.trials):
            failure_message = (
                "story search did not generate enough simulator-evaluated failure cases "
                f"for num_tenants={num_tenants}; selected={len(selected_cases)}; "
                f"required={int(args.trials)}; "
                f"story_selection={story_selection_metadata}; attempts={seed_attempts}. "
                "Increase --mapping-seed-attempts or --story-candidate-pool-target."
            )
            if bool(args.skip_result_export):
                story_search_failures.append(failure_message)
                continue
            raise RuntimeError(failure_message)

        for trial_index, selected_case in enumerate(selected_cases):
            mapping_seed = int(selected_case["mapping_seed"])
            base_experiment = selected_case["base_experiment"]
            cross_leaf_edges = selected_case["cross_leaf_edges"]
            failure_candidates = selected_case["failure_candidates"]
            structural_high_impact_pool = selected_case["structural_high_impact_pool"]
            high_impact_pool = selected_case["high_impact_pool"]
            estimator_scored_pool = selected_case["estimator_scored_pool"]
            selected_seed_score = selected_case["seed_score"]
            sampled_failure = selected_case["candidate"]
            failure_sample_seed = mapping_seed + 997
            failure = FailureEvent(
                tenant=int(sampled_failure["tenant"]),
                failed_rank=int(sampled_failure["rank"]),
                failed_server=int(sampled_failure["server"]),
            )
            experiment = replace(
                base_experiment,
                failure=failure,
                failure_selection_metadata={
                    "mode": "structural_high_impact_failure_candidates",
                    "cross_leaf_next_edge_count": len(cross_leaf_edges),
                    "candidate_count": len(failure_candidates),
                    "structural_high_impact_pool_count": len(structural_high_impact_pool),
                    "high_impact_pool_count": len(high_impact_pool),
                    "high_impact_pool_min_incident_edges": min(
                        int(candidate["cross_leaf_incident_edges"])
                        for candidate in high_impact_pool
                    ),
                    "estimator_scored_pool_count": len(estimator_scored_pool),
                    "failure_screen_candidates": int(args.failure_screen_candidates),
                    "failure_screen_mode": str(args.failure_screen_mode),
                    "mapping_seed_attempts": seed_attempts,
                    "selected_mapping_seed_score": list(selected_seed_score),
                    "failure_estimator_time_limit": failure_estimator_time_limit,
                    "allow_non_story_fallback": bool(args.allow_non_story_fallback),
                    "story_search_required": not bool(args.allow_non_story_fallback),
                    "story_search_simulator_validation_enabled": bool(
                        args.story_search_simulator_validation
                    ),
                    "story_search_simulator_validation": selected_case.get(
                        "story_search_simulator_validation"
                    ),
                    "selected_story_case_count": len(selected_failure_cases),
                    "story_candidate_pool_target": selection_pool_target,
                    "story_selection": selected_case.get("story_selection"),
                    "min_estimated_local_gain_pct": float(args.min_estimated_local_gain_pct),
                    "min_estimated_coordinate_gain_pct": float(args.min_estimated_coordinate_gain_pct),
                    "min_estimated_coordinate_advantage_pct": float(args.min_estimated_coordinate_advantage_pct),
                    "min_estimated_local_regret_pct": float(args.min_estimated_local_regret_pct),
                    "same_effect_tolerance_pct": float(args.same_effect_tolerance_pct),
                    "min_switch_reduction_vs_local": int(args.min_switch_reduction_vs_local),
                    "high_impact_pool_screen": (
                        "cross_leaf_high_impact_estimator_local_gain"
                        if str(args.failure_screen_mode) == "critical_local_gain"
                        else "structural_high_impact"
                    ),
                    "sample_seed": failure_sample_seed,
                    "sampled_without_replacement": int(args.trials) <= len(high_impact_pool),
                    "sample_order": (
                        "ranked_by_estimated_local_gain_pct"
                        if str(args.failure_screen_mode) == "critical_local_gain"
                        else "ranked_by_structural_high_impact"
                    ),
                    "candidate": sampled_failure,
                },
            )
            validation = selected_case.get("story_search_simulator_validation")
            validation_reporting = (
                validation.get("reporting", {})
                if isinstance(validation, dict)
                else {}
            )
            if validation_reporting:
                metrics = {
                    strategy: dict(validation["metrics"][strategy])
                    for strategy in STRATEGIES
                }
                failure_payload = dict(validation_reporting["failure"])
                global_protection_pool = list(
                    validation_reporting["global_protection_pool"]
                )
                working_nodes = {
                    str(tenant): list(nodes)
                    for tenant, nodes in validation_reporting[
                        "working_nodes"
                    ].items()
                }
                strategy_objectives = {
                    strategy: dict(validation_reporting["strategy_objectives"][strategy])
                    for strategy in STRATEGIES
                }
                strategy_metadata = {
                    strategy: dict(validation_reporting["strategy_metadata"][strategy])
                    for strategy in STRATEGIES
                }
                repair_movement = {
                    strategy: dict(validation_reporting["repair_movement"][strategy])
                    for strategy in STRATEGIES
                }
                strategy_mappings = {
                    strategy: {
                        str(tenant): {
                            str(rank): int(server)
                            for rank, server in ranks.items()
                        }
                        for tenant, ranks in validation_reporting.get(
                            "strategy_mappings",
                            {},
                        )
                        .get(strategy, {})
                        .items()
                    }
                    for strategy in STRATEGIES
                }
            else:
                comparison = run_repair_comparison(experiment)
                payload = repair_comparison_payload(comparison)
                failover_protection_servers = _mapping_protection_servers(
                    payload["results"]["repair_failed_server_only"]["mapping"],
                    payload["global_protection_pool"],
                )
                metrics = {
                    strategy: _strategy_metrics(
                        payload["results"][strategy],
                        global_protection_pool=payload["global_protection_pool"],
                        failover_protection_servers=failover_protection_servers,
                    )
                    for strategy in STRATEGIES
                }
                failover_mapping = {
                    int(tenant): {
                        int(rank): int(server)
                        for rank, server in ranks.items()
                    }
                    for tenant, ranks in comparison.failover_mapping.items()
                }
                repair_movement = {
                    strategy: _moves_other_tenant_vs_failover(
                        pre_failure_mapping=base_experiment.pre_failure_mapping,
                        failover_mapping=failover_mapping,
                        repaired_mapping=comparison.strategy_results[strategy].mapping,
                        failure=failure,
                    )
                    for strategy in STRATEGIES
                }
                strategy_mappings = {
                    strategy: payload["results"][strategy]["mapping"]
                    for strategy in STRATEGIES
                }
                failure_payload = payload["failure"]
                global_protection_pool = payload["global_protection_pool"]
                working_nodes = payload["working_nodes"]
                strategy_objectives = {
                    strategy: payload["results"][strategy]["objective"]
                    for strategy in STRATEGIES
                }
                strategy_metadata = {
                    strategy: payload["results"][strategy]["metadata"]
                    for strategy in STRATEGIES
                }
            failover_avg_jct = float(metrics["repair_failed_server_only"]["avg_jct"])
            local_avg_jct = float(metrics["tenant_local_repair"]["avg_jct"])
            for strategy in STRATEGIES:
                avg_jct = float(metrics[strategy]["avg_jct"])
                metrics[strategy]["improvement_pct_vs_failover"] = (
                    (failover_avg_jct - avg_jct) / failover_avg_jct * 100.0
                    if failover_avg_jct
                    else 0.0
                )
                metrics[strategy]["improvement_pct_vs_local"] = (
                    (local_avg_jct - avg_jct) / local_avg_jct * 100.0
                    if local_avg_jct
                    else 0.0
                )
            local_improvement_vs_failover = float(
                metrics["tenant_local_repair"]["improvement_pct_vs_failover"]
            )
            for strategy in STRATEGIES:
                metrics[strategy]["coordinate_advantage_pct_vs_local_repair"] = (
                    float(metrics[strategy]["improvement_pct_vs_failover"])
                    - local_improvement_vs_failover
                )
            story_checks = _story_checks_from_metrics(
                metrics,
                min_local_gain_pct=float(args.min_estimated_local_gain_pct),
                min_coordinate_advantage_pct=float(
                    args.min_estimated_coordinate_advantage_pct
                ),
                same_effect_tolerance_pct=float(args.same_effect_tolerance_pct),
                min_switch_reduction_vs_local=int(args.min_switch_reduction_vs_local),
            )
            trial = {
                "num_tenants": num_tenants,
                "trial": trial_index,
                "seed": mapping_seed,
                "mapping_seed": mapping_seed,
                "mapping_seed_attempts": seed_attempts,
                "failure_sample_seed": failure_sample_seed,
                "failure": failure_payload,
                "failure_selection_metadata": experiment.failure_selection_metadata,
                "cross_leaf_next_edge_count": len(cross_leaf_edges),
                "high_impact_candidate_count": len(failure_candidates),
                "structural_high_impact_pool_count": len(structural_high_impact_pool),
                "high_impact_pool_count": len(high_impact_pool),
                "sampled_high_impact_candidate": sampled_failure,
                "global_protection_pool": global_protection_pool,
                "protection_leaf_ids": [
                    int(server) // 8
                    for server in global_protection_pool
                ],
                "working_set_sizes": {
                    tenant: len(nodes)
                    for tenant, nodes in working_nodes.items()
                },
                "metric_source": "simulator",
                "metrics": metrics,
                "strategy_objectives": strategy_objectives,
                "strategy_metadata": strategy_metadata,
                "strategy_mappings": strategy_mappings,
                "repair_movement": repair_movement,
                "story_checks": story_checks,
            }
            trials.append(trial)
            print(json.dumps({"case": trial}, sort_keys=True), flush=True)

    if story_search_failures:
        raise RuntimeError(
            "story search did not find MILP-feasible cases for every tenant: "
            + " | ".join(story_search_failures)
        )

    summary_by_tenant = _summarize_trials(trials)
    summary = {
        "experiment": "High_resource",
        "metric_source": "simulator",
        "topology": {
            "num_spine": 4,
            "num_leaf": 8,
            "per_leaf_server": 8,
            "server_count": 64,
            "ecmp_path_count": 64 * 63,
        },
        "tenant_min": int(args.tenant_min),
        "tenant_max": int(args.tenant_max),
        "trials_per_tenant": int(args.trials),
        "protection_pool_size_mode": "high_resource",
        "working_allocation_mode": "balanced_remaining",
        "failover_policy": args.failover_policy,
        "failure_selection": "structural_high_impact_failure_candidates",
        "summary_by_tenant": summary_by_tenant,
        "trials_detail": trials,
    }

    _print_compact_table(summary_by_tenant)
    print(json.dumps({"summary": summary}, indent=2, sort_keys=True), flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
