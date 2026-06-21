#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multitenant.config import BITS_PER_MB
from multitenant.topology import LeafSpineDatacenter

from failure_aware_repair.evaluator import RepairEvaluator
from failure_aware_repair.heuristic import CooperativeRepairHeuristic, RepairSearchConfig, TenantLocalRepairHeuristic
from failure_aware_repair.models import FailureEvent, RepairScenario
from failure_aware_repair.protection import build_failover_mapping, normalize_mapping_for_json


def build_audit_case():
    datacenter = LeafSpineDatacenter(num_leaf=4, num_spine=2, per_leaf_server=6)
    mapping = {
        0: {0: 14, 1: 0, 2: 17, 3: 7},
        1: {0: 6, 1: 11, 2: 22, 3: 21},
        2: {0: 10, 1: 4, 2: 2, 3: 20},
    }
    global_protection_pool = (8, 18)
    failure = FailureEvent(tenant=2, failed_server=10, failed_rank=0)
    return datacenter, mapping, global_protection_pool, failure


def main() -> None:
    datacenter, mapping, global_protection_pool, failure = build_audit_case()
    local_scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=failure,
        mode="tenant_local",
        global_protection_pool=global_protection_pool,
    )
    cooperative_scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=failure,
        mode="cooperative",
        global_protection_pool=global_protection_pool,
    )
    local_evaluator = RepairEvaluator(
        datacenter,
        local_scenario,
        single_flow_size_bits=4 * BITS_PER_MB,
        collective="allgather",
    )
    cooperative_evaluator = RepairEvaluator(
        datacenter,
        cooperative_scenario,
        single_flow_size_bits=4 * BITS_PER_MB,
        collective="allgather",
    )
    failover = build_failover_mapping(local_scenario, datacenter=datacenter)
    failover_objective = local_evaluator.estimate(failover)
    config = RepairSearchConfig(
        beam_width=5,
        max_rounds=3,
        max_candidates_per_tenant=32,
        max_participating_tenants=3,
    )
    local = TenantLocalRepairHeuristic(
        local_scenario,
        local_evaluator,
        config=config,
    ).solve(time_limit=10)
    cooperative = CooperativeRepairHeuristic(
        cooperative_scenario,
        cooperative_evaluator,
        config=config,
    ).solve(time_limit=10)

    payload = {
        "case": "fixed_shared_contention_repair_gain",
        "pre_failure_mapping": normalize_mapping_for_json(mapping),
        "failure": {
            "tenant": failure.tenant,
            "failed_rank": failure.failed_rank,
            "failed_server": failure.failed_server,
        },
        "global_protection_pool": list(global_protection_pool),
        "results": {
            "traditional_failover": {
                "mapping": normalize_mapping_for_json(failover),
                "objective": failover_objective.__dict__,
                "simulation": local_evaluator.simulate(failover),
            },
            local.name: {
                "mapping": normalize_mapping_for_json(local.mapping),
                "objective": local.objective.__dict__,
                "switch_counts": local.switch_counts.__dict__,
                "metadata": local.metadata,
                "simulation": local_evaluator.simulate(local.mapping),
                "avg_jct_reduction_vs_failover": (
                    failover_objective.avg_jct - local.objective.avg_jct
                ) / failover_objective.avg_jct,
            },
            cooperative.name: {
                "mapping": normalize_mapping_for_json(cooperative.mapping),
                "objective": cooperative.objective.__dict__,
                "switch_counts": cooperative.switch_counts.__dict__,
                "metadata": cooperative.metadata,
                "simulation": cooperative_evaluator.simulate(cooperative.mapping),
                "avg_jct_reduction_vs_failover": (
                    failover_objective.avg_jct - cooperative.objective.avg_jct
                ) / failover_objective.avg_jct,
            },
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
