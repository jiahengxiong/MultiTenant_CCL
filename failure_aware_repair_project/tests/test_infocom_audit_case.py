from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_infocom_repair_case import build_audit_case

from multitenant.config import BITS_PER_MB

from failure_aware_repair.evaluator import RepairEvaluator
from failure_aware_repair.heuristic import CooperativeRepairHeuristic, RepairSearchConfig, TenantLocalRepairHeuristic
from failure_aware_repair.milp import FailureAwareRepairTimeExpandedILPSolver
from failure_aware_repair.mapping_ilp_base import MappingILPSolver as CopiedMappingILPSolver
from failure_aware_repair.models import RepairScenario
from failure_aware_repair.protection import build_failover_mapping, check_repair_feasible


def test_fixed_infocom_case_has_repair_gain():
    datacenter, mapping, global_protection_pool, failure = build_audit_case()
    local_scenario = RepairScenario(mapping, failure, "tenant_local", global_protection_pool)
    cooperative_scenario = RepairScenario(mapping, failure, "cooperative", global_protection_pool)
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

    check_repair_feasible(local_scenario, local.mapping, local_evaluator.failover_mapping)
    check_repair_feasible(cooperative_scenario, cooperative.mapping, cooperative_evaluator.failover_mapping)
    assert local.objective.avg_jct < failover_objective.avg_jct
    assert cooperative.objective.avg_jct < failover_objective.avg_jct
    assert cooperative.objective.avg_jct <= local.objective.avg_jct
    assert local.switch_counts.extra_vs_failover >= 1


def test_time_expanded_repair_ilp_extends_copied_mapping_ilp():
    assert FailureAwareRepairTimeExpandedILPSolver.__mro__[1] is CopiedMappingILPSolver
    assert CopiedMappingILPSolver.__module__ == "failure_aware_repair.mapping_ilp_base"


def test_time_expanded_repair_ilp_solves_small_case():
    import gurobipy as gp
    from gurobipy import GRB
    from multitenant.topology import LeafSpineDatacenter
    from failure_aware_repair.models import FailureEvent

    _ = gp  # keeps the dependency explicit for skip/debug readability.
    datacenter = LeafSpineDatacenter(num_leaf=2, num_spine=1, per_leaf_server=3)
    mapping = {0: {0: 0, 1: 1}, 1: {0: 2, 1: 3}}
    scenario = RepairScenario(
        mapping,
        FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        "tenant_local",
        (4, 5),
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

    assert result.metadata["gurobi_status"] == GRB.OPTIMAL
    check_repair_feasible(scenario, result.mapping, evaluator.failover_mapping)
    assert result.objective.avg_jct >= 0.0
    assert result.switch_counts.extra_vs_failover == result.objective.extra_switches
