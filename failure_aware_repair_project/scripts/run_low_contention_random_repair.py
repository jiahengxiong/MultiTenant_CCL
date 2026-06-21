#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
    find_random_repair_gain_case,
    generate_random_repair_experiment,
    repair_comparison_payload,
    run_repair_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Low-contention random repair experiment: random working sets, "
            "one global protection pool, "
            "Low_contension-style non-MILP mapping, random failure, and three repair modes."
        )
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--num-leaf", type=int, default=4)
    parser.add_argument("--num-spine", type=int, default=2)
    parser.add_argument("--per-leaf-server", type=int, default=6)
    parser.add_argument("--num-tenants", type=int, default=3)
    parser.add_argument(
        "--protection-pool-size-mode",
        choices=("high_resource", "low_resource"),
        default="high_resource",
        help=(
            "Protection resource mode: high_resource reserves one protection server "
            "per leaf; low_resource reserves one protection server per two leaves."
        ),
    )
    parser.add_argument(
        "--workload-mode",
        choices=("low_contention_dominant", "synthetic_uniform"),
        default="low_contention_dominant",
        help=(
            "Use original Low_contension.py dominant trace-derived tenant specs "
            "or a synthetic uniform collective for debugging."
        ),
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
        help=(
            "Traditional failover replacement policy. 'first' is a static "
            "traffic-oblivious backup; 'same_leaf_or_nearest' is a stronger "
            "topology-aware baseline."
        ),
    )
    parser.add_argument(
        "--failure-selection",
        choices=("random", "critical_failover"),
        default="random",
        help=(
            "Select a uniform random failed working server or the single "
            "failure whose traditional failover has the highest Avg JCT."
        ),
    )
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-candidates-per-tenant", type=int, default=32)
    parser.add_argument("--max-joint-tenants", type=int, default=2)
    parser.add_argument("--joint-candidates-per-tenant", type=int, default=4)
    parser.add_argument("--max-joint-candidates", type=int, default=48)
    parser.add_argument("--max-block-ranks", type=int, default=4)
    parser.add_argument("--block-extra-servers", type=int, default=4)
    parser.add_argument("--max-block-candidates", type=int, default=96)
    parser.add_argument(
        "--find-gain",
        action="store_true",
        help="Search consecutive seeds until failover > tenant-local > cooperative by Avg JCT.",
    )
    parser.add_argument("--max-trials", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    working_mapping_time_limit = (
        None
        if args.no_working_mapping_time_limit
        else args.working_mapping_time_limit
    )
    config = RandomRepairExperimentConfig(
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
        repair_search=RepairSearchConfig(
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
        ),
    )
    if args.find_gain:
        comparison = find_random_repair_gain_case(config, max_trials=args.max_trials)
    else:
        experiment = generate_random_repair_experiment(config)
        comparison = run_repair_comparison(experiment)
    print(json.dumps(repair_comparison_payload(comparison), indent=2))


if __name__ == "__main__":
    main()
