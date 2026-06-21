#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multitenant.config import BITS_PER_MB

from failure_aware_repair.heuristic import RepairSearchConfig
from failure_aware_repair.random_experiments import (
    RandomRepairExperimentConfig,
    generate_random_repair_experiment,
    repair_comparison_payload,
    run_repair_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated random low-contention failure-repair trials and report averages."
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--num-leaf", type=int, default=8)
    parser.add_argument("--num-spine", type=int, default=4)
    parser.add_argument("--per-leaf-server", type=int, default=8)
    parser.add_argument("--num-tenants", type=int, default=7)
    parser.add_argument(
        "--protection-pool-size-mode",
        choices=("high_resource", "low_resource"),
        default="high_resource",
    )
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
        help="Optional time limit for Low_contension.py working-set mapping; omitted means no limit.",
    )
    parser.add_argument(
        "--no-working-mapping-time-limit",
        action="store_true",
        help="Run the original Low_contension.py working-set mapper without a solver time limit.",
    )
    parser.add_argument("--repair-time-limit", type=float, default=10.0)
    parser.add_argument(
        "--failover-policy",
        choices=("first", "same_leaf_or_nearest"),
        default="first",
    )
    parser.add_argument(
        "--failure-selection",
        choices=("random", "critical_failover"),
        default="random",
    )
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-candidates-per-tenant", type=int, default=36)
    parser.add_argument("--max-joint-tenants", type=int, default=3)
    parser.add_argument("--joint-candidates-per-tenant", type=int, default=5)
    parser.add_argument("--max-joint-candidates", type=int, default=160)
    parser.add_argument("--max-block-ranks", type=int, default=4)
    parser.add_argument("--block-extra-servers", type=int, default=5)
    parser.add_argument("--max-block-candidates", type=int, default=128)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    search = RepairSearchConfig(
        beam_width=args.beam_width,
        max_rounds=args.max_rounds,
        max_candidates_per_tenant=args.max_candidates_per_tenant,
        max_participating_tenants=args.num_tenants,
        max_joint_tenants=args.max_joint_tenants,
        joint_candidates_per_tenant=args.joint_candidates_per_tenant,
        max_joint_candidates=args.max_joint_candidates,
        max_block_ranks=args.max_block_ranks,
        block_extra_servers=args.block_extra_servers,
        max_block_candidates=args.max_block_candidates,
    )
    working_mapping_time_limit = (
        None
        if args.no_working_mapping_time_limit
        else args.working_mapping_time_limit
    )
    base_config = RandomRepairExperimentConfig(
        seed=args.seed,
        num_leaf=args.num_leaf,
        num_spine=args.num_spine,
        per_leaf_server=args.per_leaf_server,
        num_tenants=args.num_tenants,
        ranks_per_tenant=None,
        working_allocation_mode="balanced_remaining",
        protection_pool_size_mode=args.protection_pool_size_mode,
        contention="low",
        workload_mode=args.workload_mode,
        single_flow_size_bits=int(args.flow_mb) * BITS_PER_MB,
        working_mapping_time_limit=working_mapping_time_limit,
        repair_time_limit=args.repair_time_limit,
        failover_policy=args.failover_policy,
        failure_selection=args.failure_selection,
        repair_search=search,
    )

    trials = []
    for offset in range(int(args.trials)):
        config = replace(base_config, seed=int(args.seed) + offset)
        comparison = run_repair_comparison(generate_random_repair_experiment(config))
        payload = repair_comparison_payload(comparison)
        failover_estimator = payload["results"]["repair_failed_server_only"]["objective"]["avg_jct"]
        tenant_local_estimator = payload["results"]["tenant_local_repair"]["objective"]["avg_jct"]
        cooperative_estimator = payload["results"]["cooperative_repair"]["objective"]["avg_jct"]
        failover = payload["results"]["repair_failed_server_only"]["simulation"]["avg_jct"]
        tenant_local = payload["results"]["tenant_local_repair"]["simulation"]["avg_jct"]
        cooperative = payload["results"]["cooperative_repair"]["simulation"]["avg_jct"]
        local_improvement = (failover - tenant_local) / failover * 100.0 if failover else 0.0
        cooperative_improvement = (failover - cooperative) / failover * 100.0 if failover else 0.0
        trial = {
            "seed": config.seed,
            "failure": payload["failure"],
            "failover_avg_jct": failover,
            "tenant_local_avg_jct": tenant_local,
            "cooperative_avg_jct": cooperative,
            "metric_source": "simulator",
            "estimator_avg_jct": {
                "failover": failover_estimator,
                "tenant_local": tenant_local_estimator,
                "cooperative": cooperative_estimator,
            },
            "tenant_local_improvement_pct": local_improvement,
            "cooperative_improvement_pct": cooperative_improvement,
        }
        trials.append(trial)
        print(json.dumps({"trial": offset, **trial}), flush=True)

    local_values = [trial["tenant_local_improvement_pct"] for trial in trials]
    cooperative_values = [trial["cooperative_improvement_pct"] for trial in trials]
    summary = {
        "seed": args.seed,
        "trials": int(args.trials),
        "topology": {
            "num_leaf": args.num_leaf,
            "num_spine": args.num_spine,
            "per_leaf_server": args.per_leaf_server,
        },
        "num_tenants": args.num_tenants,
        "working_allocation_mode": args.working_allocation_mode,
        "protection_pool_size_mode": args.protection_pool_size_mode,
        "failover_policy": args.failover_policy,
        "failure_selection": args.failure_selection,
        "metric_source": "simulator",
        "tenant_local_improvement_pct_mean": _mean(local_values),
        "tenant_local_improvement_pct_median": _median(local_values),
        "cooperative_improvement_pct_mean": _mean(cooperative_values),
        "cooperative_improvement_pct_median": _median(cooperative_values),
        "trials_detail": trials,
    }
    print(json.dumps({"summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
