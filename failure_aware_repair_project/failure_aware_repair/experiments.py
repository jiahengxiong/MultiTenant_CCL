from __future__ import annotations

import json
from pathlib import Path

from multitenant.config import BITS_PER_MB
from multitenant.topology import LeafSpineDatacenter

from .evaluator import RepairEvaluator
from .heuristic import CooperativeRepairHeuristic, RepairSearchConfig, TenantLocalRepairHeuristic
from .milp import ExactEnumerationConfig, ExactRepairEnumerator
from .models import FailureEvent, RepairScenario
from .protection import (
    build_failover_mapping,
    normalize_mapping_for_json,
)


def build_demo_mapping(datacenter: LeafSpineDatacenter) -> dict[int, dict[int, int]]:
    return {
        0: {0: 0, 1: 1, 2: 4, 3: 5},
        1: {0: 2, 1: 3, 2: 6, 3: 7},
    }


def run_debug_case(seed: int = 7) -> dict[str, object]:
    datacenter = LeafSpineDatacenter(num_leaf=4, num_spine=2, per_leaf_server=4)
    pre_mapping = build_demo_mapping(datacenter)
    global_pool = (14, 15)
    failure = FailureEvent(tenant=0, failed_server=pre_mapping[0][1], failed_rank=1)
    scenario = RepairScenario(
        pre_failure_mapping=pre_mapping,
        failure=failure,
        mode="cooperative",
        global_protection_pool=global_pool,
    )
    evaluator = RepairEvaluator(
        datacenter,
        scenario,
        single_flow_size_bits=2 * BITS_PER_MB,
        collective="allgather",
    )
    failover = build_failover_mapping(scenario, datacenter=datacenter)
    failover_obj = evaluator.estimate(failover)
    config = RepairSearchConfig(beam_width=4, max_rounds=2, max_candidates_per_tenant=16)
    local = TenantLocalRepairHeuristic(scenario, evaluator, config=config).solve(time_limit=10)
    coop = CooperativeRepairHeuristic(scenario, evaluator, config=config).solve(time_limit=10)
    exact = ExactRepairEnumerator(
        scenario,
        evaluator,
        config=ExactEnumerationConfig(max_assignments=10_000, time_limit_seconds=15),
    ).solve()

    payload = {
        "seed": seed,
        "pre_failure_mapping": normalize_mapping_for_json(pre_mapping),
        "global_protection_pool": list(global_pool),
        "failure": {
            "tenant": failure.tenant,
            "failed_rank": failure.failed_rank,
            "failed_server": failure.failed_server,
        },
        "results": {
            "failover": {
                "mapping": normalize_mapping_for_json(failover),
                "objective": failover_obj.__dict__,
                "simulation": evaluator.simulate(failover),
            },
            local.name: _result_payload(local, evaluator),
            coop.name: _result_payload(coop, evaluator),
            exact.name: _result_payload(exact, evaluator),
        },
    }
    return payload


def _result_payload(result, evaluator: RepairEvaluator) -> dict[str, object]:
    return {
        "mapping": normalize_mapping_for_json(result.mapping),
        "objective": result.objective.__dict__,
        "switch_counts": result.switch_counts.__dict__,
        "runtime_seconds": result.runtime_seconds,
        "metadata": result.metadata,
        "simulation": evaluator.simulate(result.mapping),
    }


def write_debug_case(path: str | Path, *, seed: int = 7) -> None:
    payload = run_debug_case(seed=seed)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
