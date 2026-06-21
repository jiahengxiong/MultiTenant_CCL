from __future__ import annotations

import time

from multitenant.config import BITS_PER_MB
from multitenant.topology import LeafSpineDatacenter

from failure_aware_repair.evaluator import RepairEvaluator
from failure_aware_repair.heuristic import (
    CooperativeRepairHeuristic,
    RepairSearchConfig,
    TenantLocalRepairHeuristic,
)
from failure_aware_repair.milp import ExactEnumerationConfig, ExactRepairEnumerator
from failure_aware_repair.models import FailureEvent, RepairScenario
from failure_aware_repair.objectives import repair_sort_key
from failure_aware_repair.protection import (
    build_failover_mapping,
    check_repair_feasible,
)


def _small_case(mode: str = "cooperative"):
    datacenter = LeafSpineDatacenter(num_leaf=4, num_spine=2, per_leaf_server=4)
    mapping = {0: {0: 0, 1: 1}, 1: {0: 2, 1: 3}}
    scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode=mode,  # type: ignore[arg-type]
        global_protection_pool=(14, 15),
    )
    evaluator = RepairEvaluator(
        datacenter,
        scenario,
        single_flow_size_bits=BITS_PER_MB,
        collective="allgather",
    )
    return datacenter, scenario, evaluator


def test_evaluator_scores_failover_mapping_with_protection_node():
    datacenter, scenario, evaluator = _small_case("tenant_local")
    failover = build_failover_mapping(scenario, datacenter=datacenter)
    objective = evaluator.estimate(failover)
    sim_makespan, sim_avg = evaluator.simulate(failover)

    assert objective.avg_jct >= 0.0
    assert objective.makespan >= 0.0
    assert sim_makespan >= 0.0
    assert sim_avg >= 0.0


def test_tenant_local_repair_is_feasible_and_keeps_other_tenants_fixed():
    _datacenter, scenario, evaluator = _small_case("tenant_local")
    result = TenantLocalRepairHeuristic(
        scenario,
        evaluator,
        config=RepairSearchConfig(beam_width=3, max_rounds=1, max_candidates_per_tenant=8),
    ).solve(time_limit=5)

    check_repair_feasible(scenario, result.mapping, evaluator.failover_mapping)
    assert result.mapping[1] == evaluator.failover_mapping[1]
    assert result.objective.extra_switches == result.switch_counts.extra_vs_failover


def test_cooperative_repair_is_feasible():
    _datacenter, scenario, evaluator = _small_case("cooperative")
    result = CooperativeRepairHeuristic(
        scenario,
        evaluator,
        config=RepairSearchConfig(beam_width=3, max_rounds=1, max_candidates_per_tenant=8),
    ).solve(time_limit=5)

    check_repair_feasible(scenario, result.mapping, evaluator.failover_mapping)
    assert result.objective.extra_switches == result.switch_counts.extra_vs_failover


def test_exact_enumerator_is_no_worse_than_tenant_local_heuristic_on_small_case():
    _datacenter, scenario, evaluator = _small_case("tenant_local")
    heuristic = TenantLocalRepairHeuristic(
        scenario,
        evaluator,
        config=RepairSearchConfig(beam_width=3, max_rounds=1, max_candidates_per_tenant=6),
    ).solve(time_limit=5)
    exact = ExactRepairEnumerator(
        scenario,
        evaluator,
        config=ExactEnumerationConfig(max_assignments=1000, time_limit_seconds=10),
    ).solve()

    assert repair_sort_key(exact.objective) <= repair_sort_key(heuristic.objective)
    check_repair_feasible(scenario, exact.mapping, evaluator.failover_mapping)


def test_candidate_set_subproblem_moves_every_selected_rank_to_protection():
    _datacenter, scenario, evaluator = _small_case("tenant_local")
    search = TenantLocalRepairHeuristic(
        scenario,
        evaluator,
        config=RepairSearchConfig(beam_width=3, max_rounds=1, max_candidates_per_tenant=6),
    )
    analysis = search._safe_analysis(evaluator.failover_mapping)

    states = search._logic_benders_assignment_subproblem(
        scenario.pre_failure_mapping,
        analysis,
        time.time() + 5,
        protection=tuple(scenario.global_protection_pool),
        failed_tenant=0,
        failed_rank=1,
        failed_servers=list(scenario.global_protection_pool),
        healthy_selection=((0, 0),),
        is_coordinate=False,
        state_limit=1,
    )

    assert states
    _objective, mapping, moves, used_servers, moved_ranks = states[0]
    assert set(used_servers) <= set(scenario.global_protection_pool)
    assert moved_ranks == frozenset({(0, 0), (0, 1)})
    assert mapping[0][0] in scenario.global_protection_pool
    assert mapping[0][1] in scenario.global_protection_pool
    assert mapping[0][0] != mapping[0][1]
    assert sorted((tenant, rank) for tenant, rank, _server in moves) == [(0, 0), (0, 1)]
