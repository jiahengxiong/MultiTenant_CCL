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
from failure_aware_repair.milp import FailureAwareRepairTimeExpandedILPSolver
from failure_aware_repair.models import FailureEvent, RepairScenario
from failure_aware_repair.protection import normalize_mapping_for_json


def main() -> None:
    datacenter = LeafSpineDatacenter(num_leaf=2, num_spine=1, per_leaf_server=3)
    mapping = {0: {0: 0, 1: 1}, 1: {0: 2, 1: 3}}
    scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode="tenant_local",
        global_protection_pool=(4, 5),
    )
    evaluator = RepairEvaluator(
        datacenter,
        scenario,
        single_flow_size_bits=BITS_PER_MB // 8,
        collective="allgather",
        horizon_slots=8,
    )
    solver = FailureAwareRepairTimeExpandedILPSolver(
        scenario,
        evaluator,
        verbose=False,
        horizon_slots=8,
    )
    solver.solve(time_limit=10)
    result = solver.to_repair_result()
    print(
        json.dumps(
            {
                "mapping": normalize_mapping_for_json(result.mapping),
                "objective": result.objective.__dict__,
                "switch_counts": result.switch_counts.__dict__,
                "metadata": result.metadata,
                "simulation": evaluator.simulate(result.mapping),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
