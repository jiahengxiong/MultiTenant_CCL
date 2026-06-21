from __future__ import annotations

import random

import pytest

from multitenant.config import BITS_PER_MB
from multitenant.topology import LeafSpineDatacenter

from failure_aware_repair.heuristic import REPAIR_ALGORITHM_VERSION, RepairSearchConfig
from failure_aware_repair import random_experiments
from failure_aware_repair.random_experiments import (
    RandomRepairExperimentConfig,
    generate_random_repair_experiment,
    repair_comparison_payload,
    run_repair_comparison,
    sample_balanced_low_contention_node_sets,
    sample_random_tenant_node_sets,
    solve_best_working_set_mappings,
)
from failure_aware_repair.protection import check_repair_feasible
from failure_aware_repair.solvers.nearest_protection_baseline import (
    solve_nearest_protection_baseline,
)
from failure_aware_repair.models import FailureEvent, RepairScenario
from failure_aware_repair.strategy_solver import FailureAwareMappingStrategySolver
from failure_aware_repair.strategies import (
    COOPERATIVE_REPAIR,
    REPAIR_FAILED_SERVER_ONLY,
    TENANT_LOCAL_REPAIR,
)


def _fast_config(seed: int = 3) -> RandomRepairExperimentConfig:
    return RandomRepairExperimentConfig(
        seed=seed,
        num_leaf=4,
        num_spine=1,
        per_leaf_server=6,
        num_tenants=2,
        ranks_per_tenant=None,
        working_allocation_mode="balanced_remaining",
        contention="low",
        single_flow_size_bits=BITS_PER_MB // 8,
        working_mapping_time_limit=1,
        repair_time_limit=2,
        repair_search=RepairSearchConfig(
            beam_width=2,
            max_rounds=1,
            max_candidates_per_tenant=4,
            max_participating_tenants=2,
        ),
    )


def test_low_contention_uses_balanced_high_resource_protection_pool():
    import random

    first = sample_random_tenant_node_sets(
        range(24),
        num_tenants=2,
        ranks_per_tenant=None,
        rng=random.Random(1),
        contention="low",
        allocation_mode="balanced_remaining",
        num_leaf=4,
        per_leaf_server=6,
    )
    second = sample_random_tenant_node_sets(
        range(24),
        num_tenants=2,
        ranks_per_tenant=None,
        rng=random.Random(2),
        contention="low",
        allocation_mode="balanced_remaining",
        num_leaf=4,
        per_leaf_server=6,
    )
    assert first != second

    working, global_pool = first
    all_working = []
    for tenant in working:
        assert set(working[tenant]).isdisjoint(global_pool)
        all_working.extend(working[tenant])
    assert len(all_working) == len(set(all_working))
    assert set(all_working) | set(global_pool) == set(range(24))
    assert len(global_pool) == 4
    assert {server // 6 for server in global_pool} == {0, 1, 2, 3}


def test_low_contention_rejects_fixed_rank_allocation():
    with pytest.raises(ValueError, match="only supports balanced_remaining"):
        sample_random_tenant_node_sets(
            range(24),
            num_tenants=2,
            ranks_per_tenant=3,
            rng=random.Random(1),
            contention="low",
            allocation_mode="fixed_ranks_per_tenant",
            num_leaf=4,
            per_leaf_server=6,
        )


def test_balanced_low_contention_uses_all_non_protection_servers():
    import random

    working, global_pool = sample_balanced_low_contention_node_sets(
        range(64),
        num_tenants=7,
        rng=random.Random(5),
        num_leaf=8,
        per_leaf_server=8,
    )
    all_working = {
        int(server)
        for tenant_nodes in working.values()
        for server in tenant_nodes
    }
    assert len(all_working) == sum(len(nodes) for nodes in working.values())
    assert all_working.isdisjoint(global_pool)
    assert all_working | set(global_pool) == set(range(64))
    sizes = [len(working[tenant]) for tenant in sorted(working)]
    assert max(sizes) - min(sizes) <= 1


def test_high_resource_4spine_8leaf_8server_protection_has_one_per_leaf():
    import random

    working, global_pool = sample_balanced_low_contention_node_sets(
        range(64),
        num_tenants=7,
        rng=random.Random(10),
        protection_pool_size_mode="high_resource",
        num_leaf=8,
        per_leaf_server=8,
    )
    all_working = {
        int(server)
        for tenant_nodes in working.values()
        for server in tenant_nodes
    }

    assert len(global_pool) == 8
    assert {server // 8 for server in global_pool} == set(range(8))
    assert all_working.isdisjoint(global_pool)
    assert all_working | set(global_pool) == set(range(64))
    assert [len(working[tenant]) for tenant in sorted(working)] == [8] * 7


def test_balanced_low_contention_can_use_low_resource_protection():
    import random

    working, global_pool = sample_balanced_low_contention_node_sets(
        range(64),
        num_tenants=7,
        rng=random.Random(5),
        protection_pool_size_mode="low_resource",
        num_leaf=8,
        per_leaf_server=8,
    )
    all_working = {
        int(server)
        for tenant_nodes in working.values()
        for server in tenant_nodes
    }
    assert len(global_pool) == 4
    assert {server // 8 // 2 for server in global_pool} == {0, 1, 2, 3}
    assert all_working.isdisjoint(global_pool)
    assert all_working | set(global_pool) == set(range(64))


def test_full_server_partition_validation_rejects_idle_or_shared_servers():
    validator = random_experiments._validate_full_server_partition

    validator(range(6), {0: (0, 1), 1: (2, 3)}, (4, 5))

    with pytest.raises(ValueError, match="duplicate servers"):
        validator(range(6), {0: (0, 1), 1: (1, 2)}, (4, 5))
    with pytest.raises(ValueError, match="disjoint"):
        validator(range(6), {0: (0, 1), 1: (2, 3)}, (3, 5))
    with pytest.raises(ValueError, match="partition all servers"):
        validator(range(6), {0: (0, 1), 1: (2,)}, (4, 5))


def test_repair_feasibility_requires_failed_rank_on_one_protection_server():
    scenario = RepairScenario(
        pre_failure_mapping={0: {0: 0, 1: 1}, 1: {0: 2, 1: 3}},
        failure=FailureEvent(tenant=0, failed_rank=0, failed_server=0),
        mode="tenant_local",
        global_protection_pool=(4, 5),
    )
    mapping = {0: {0: 1, 1: 4}, 1: {0: 2, 1: 3}}

    with pytest.raises(ValueError, match="failed rank must occupy exactly one protection server"):
        check_repair_feasible(scenario, mapping)


def test_random_experiment_uses_low_contention_mapping_optimizer_not_milp():
    experiment = generate_random_repair_experiment(_fast_config())

    assert experiment.config.contention == "low"
    assert experiment.workload.tenant_collective_specs is not None
    assert experiment.workload.source == "experiment/Low_contension.py dominant trace-derived tenant_collective_specs"
    assert set(experiment.workload.tenant_collective_specs) == set(experiment.pre_failure_mapping)
    assert experiment.failure.tenant in experiment.pre_failure_mapping
    assert experiment.failure.failed_server in experiment.pre_failure_mapping[experiment.failure.tenant].values()

    for tenant, result in experiment.working_results.items():
        assert result.solver_name == "MappingEstimatorBlackBoxOptimizer"
        assert result.solver_source == "experiment/Low_contension.py::run_mapping"
        assert set(result.best_mapping.values()) == set(result.working_nodes)
        assert experiment.pre_failure_mapping[tenant] == result.best_mapping
        assert set(result.working_nodes).isdisjoint(experiment.global_protection_pool)
        assert "profile_name" in experiment.workload.tenant_collective_specs[tenant]
        assert "trace_path" in experiment.workload.tenant_collective_specs[tenant]
    assert len(experiment.global_protection_pool) == experiment.config.num_leaf


def test_working_mapping_calls_low_contention_once_with_all_tenants(monkeypatch):
    class FakeLowContention:
        MAPPING_TIME_LIMIT_SECONDS = None

        def __init__(self):
            self.seen_mapping = None
            self.seen_specs = None
            self.seen_time_limit = None

        def run_mapping(self, datacenter, tenant_mapping, tenant_collective_specs):
            del datacenter
            self.seen_time_limit = self.MAPPING_TIME_LIMIT_SECONDS
            self.seen_mapping = {
                int(tenant): dict(ranks)
                for tenant, ranks in tenant_mapping.items()
            }
            self.seen_specs = {
                int(tenant): dict(spec)
                for tenant, spec in tenant_collective_specs.items()
            }
            return tenant_mapping, 0.25

        def evaluate_collective(self, datacenter, mapping, tenant_collective_specs):
            del datacenter, mapping, tenant_collective_specs
            return 12.0, 3.0

    fake = FakeLowContention()
    monkeypatch.setattr(
        random_experiments,
        "_load_low_contention_module",
        lambda: fake,
    )
    initial_mapping = {
        0: {0: 0, 1: 1},
        1: {0: 2, 1: 3, 2: 4},
        2: {0: 5},
    }

    solved_mapping, results, workload = solve_best_working_set_mappings(
        LeafSpineDatacenter(num_leaf=2, num_spine=1, per_leaf_server=4),
        initial_mapping=initial_mapping,
        rng=random.Random(0),
        seed=123,
        collective="allgather",
        single_flow_size_bits=BITS_PER_MB,
        workload_mode="synthetic_uniform",
        time_limit=7,
    )

    assert fake.seen_time_limit == 7
    assert fake.seen_mapping == initial_mapping
    assert set(fake.seen_specs) == {0, 1, 2}
    assert solved_mapping == initial_mapping
    assert set(results) == {0, 1, 2}
    assert all(
        result.solver_source == "experiment/Low_contension.py::run_mapping"
        for result in results.values()
    )
    assert workload.source == "synthetic uniform collective specs"


def test_random_experiment_balanced_allocation_matches_low_contention_style():
    config = RandomRepairExperimentConfig(
        seed=6,
        num_leaf=4,
        num_spine=1,
        per_leaf_server=6,
        num_tenants=3,
        ranks_per_tenant=None,
        working_allocation_mode="balanced_remaining",
        contention="low",
        working_mapping_time_limit=1,
        repair_time_limit=1,
        repair_search=RepairSearchConfig(
            beam_width=2,
            max_rounds=1,
            max_candidates_per_tenant=4,
            max_participating_tenants=3,
        ),
    )
    experiment = generate_random_repair_experiment(config)
    all_working = {
        server
        for rank_to_server in experiment.pre_failure_mapping.values()
        for server in rank_to_server.values()
    }
    assert all_working.isdisjoint(experiment.global_protection_pool)
    assert all_working | set(experiment.global_protection_pool) == set(range(24))
    sizes = [len(experiment.pre_failure_mapping[tenant]) for tenant in sorted(experiment.pre_failure_mapping)]
    assert max(sizes) - min(sizes) <= 1


def test_random_failure_then_three_repair_modes_are_feasible():
    experiment = generate_random_repair_experiment(_fast_config())
    comparison = run_repair_comparison(experiment)

    assert comparison.failover_objective.extra_switches == 0
    assert comparison.tenant_local.name == "tenant_local_repair"
    assert comparison.cooperative.name == "cooperative_repair"
    assert set(comparison.strategy_results) == {
        "repair_failed_server_only",
        "tenant_local_repair",
        "cooperative_repair",
    }
    assert comparison.strategy_results["tenant_local_repair"].mapping == comparison.tenant_local.mapping
    assert comparison.strategy_results["cooperative_repair"].mapping == comparison.cooperative.mapping
    assert comparison.tenant_local.objective.avg_jct <= comparison.failover_objective.avg_jct + 1e-6
    assert comparison.cooperative.objective.avg_jct <= comparison.failover_objective.avg_jct + 1e-6
    assert comparison.tenant_local.metadata["algorithm_version"] == REPAIR_ALGORITHM_VERSION
    assert comparison.cooperative.metadata["algorithm_version"] == REPAIR_ALGORITHM_VERSION
    assert (
        comparison.strategy_results["repair_failed_server_only"].metadata["decomposition_algorithm"]
        == "contention_guided_logic_based_candidate_set_decomposition"
    )
    assert (
        comparison.tenant_local.metadata["decomposition_algorithm"]
        == "contention_guided_logic_based_candidate_set_decomposition"
    )
    assert (
        comparison.cooperative.metadata["decomposition_algorithm"]
        == "contention_guided_logic_based_candidate_set_decomposition"
    )
    assert comparison.tenant_local.metadata["proposal_order"] == "independent_protection_slot_assignment"
    assert comparison.cooperative.metadata["solver"]
    assert (
        comparison.tenant_local.metadata["best_source"].startswith("slot_assign_local:")
        or comparison.tenant_local.metadata["best_source"].startswith("failover")
    )
    assert (
        comparison.cooperative.metadata["best_source"].startswith("slot_assign_coordinate:")
        or comparison.cooperative.metadata["best_source"].startswith("failover")
    )
    assert "nested_local_seed" not in comparison.cooperative.metadata["best_source"]
    assert "tenant_local_result" not in comparison.cooperative.metadata["best_source"]
    payload = repair_comparison_payload(comparison)
    problem_payload = payload["failure_aware_mapping_problem"]
    assert problem_payload["failed_server"] == experiment.failure.failed_server
    assert problem_payload["global_protection_pool"] == list(experiment.global_protection_pool)
    problem = experiment.problem
    failover_scenario = problem.scenario(REPAIR_FAILED_SERVER_ONLY)
    local_scenario = problem.scenario(TENANT_LOCAL_REPAIR)
    cooperative_scenario = problem.scenario(COOPERATIVE_REPAIR)
    assert failover_scenario.workload is experiment.workload
    assert local_scenario.workload is experiment.workload
    assert cooperative_scenario.workload is experiment.workload
    assert local_scenario.participating_tenants == (experiment.failure.tenant,)
    assert set(cooperative_scenario.participating_tenants) == set(experiment.pre_failure_mapping)
    assert set(payload["strategies"]) == {
        "repair_failed_server_only",
        "tenant_local_repair",
        "cooperative_repair",
    }
    constraints = payload["strategy_constraints"]
    assert set(constraints) == {
        "repair_failed_server_only",
        "tenant_local_repair",
        "cooperative_repair",
    }
    failed_tenant = experiment.failure.tenant
    assert constraints["tenant_local_repair"]["participating_tenants"] == [failed_tenant]
    assert constraints["tenant_local_repair"]["strategy_spec"]["participant_scope"] == "failed_tenant"
    assert constraints["tenant_local_repair"]["strategy_spec"]["optimizes_mapping"] is True
    assert failed_tenant not in constraints["tenant_local_repair"]["fixed_tenants"]
    assert set(constraints["cooperative_repair"]["participating_tenants"]) == set(
        experiment.pre_failure_mapping
    )
    assert constraints["cooperative_repair"]["strategy_spec"]["participant_scope"] == "all_tenants"
    assert constraints["cooperative_repair"]["fixed_tenants"] == []
    assert constraints["repair_failed_server_only"]["only_failed_rank_replaced"] is True
    assert constraints["repair_failed_server_only"]["strategy_spec"]["optimizes_mapping"] is True
    assert constraints["repair_failed_server_only"]["healthy_ranks_fixed_to_prefailure"] is True
    assert failed_tenant not in constraints["repair_failed_server_only"]["fixed_tenants"]
    assert constraints["tenant_local_repair"]["failed_server_excluded_for_failed_tenant"] is True
    assert constraints["cooperative_repair"]["failed_server_excluded_for_failed_tenant"] is True
    assert (
        payload["results"]["repair_failed_server_only"]["metadata"]["solver"]
        == "unified_gurobi_master_subproblem_repair_optimizer"
    )
    assert payload["results"]["repair_failed_server_only"]["metadata"][
        "logic_benders_subproblem_solver"
    ] == "gurobi_unique_protection_slot_assignment_subproblem_with_contention_estimator"
    assert (
        payload["results"]["repair_failed_server_only"]["metadata"][
            "contains_nearest_baseline_candidate"
        ]
        is True
    )
    assert payload["results"]["tenant_local_repair"]["metadata"]["solver"] == "tenant_local_repair"
    assert (
        payload["results"]["tenant_local_repair"]["metadata"]["optimizer"]
        == "contention_guided_logic_based_repair_optimizer"
    )
    assert payload["results"]["cooperative_repair"]["metadata"]["solver"] == "cooperative_repair"
    assert (
        payload["results"]["cooperative_repair"]["metadata"]["optimizer"]
        == "contention_guided_logic_based_repair_optimizer"
    )

    # The feasibility checks are intentionally repeated here as regression
    # evidence for all three post-failure mappings.
    from failure_aware_repair.models import RepairScenario

    tenant_local = RepairScenario(
        experiment.pre_failure_mapping,
        experiment.failure,
        "tenant_local",
        global_protection_pool=experiment.global_protection_pool,
    )
    cooperative = RepairScenario(
        experiment.pre_failure_mapping,
        experiment.failure,
        "cooperative",
        global_protection_pool=experiment.global_protection_pool,
    )
    check_repair_feasible(tenant_local, comparison.failover_mapping, comparison.failover_mapping)
    check_repair_feasible(tenant_local, comparison.tenant_local.mapping, comparison.failover_mapping)
    check_repair_feasible(cooperative, comparison.cooperative.mapping, comparison.failover_mapping)


def test_random_low_contention_input_feeds_time_expanded_repair_model():
    from failure_aware_repair.evaluator import RepairEvaluator
    from failure_aware_repair.milp import FailureAwareRepairTimeExpandedILPSolver
    from failure_aware_repair.models import RepairScenario

    experiment = generate_random_repair_experiment(_fast_config(seed=5))
    scenario = RepairScenario(
        experiment.pre_failure_mapping,
        experiment.failure,
        "tenant_local",
        global_protection_pool=experiment.global_protection_pool,
        workload=experiment.workload,
    )
    evaluator = RepairEvaluator(
        experiment.datacenter,
        scenario,
        horizon_slots=8,
    )
    dag_summary = evaluator.dag_summary()
    assert dag_summary["source"] == "multitenant.solvers.DAG_generation.build_collective_dag_data"
    assert dag_summary["has_task_surrogate"] is True
    assert set(dag_summary["task_count_by_tenant"]) == {
        str(tenant) for tenant in experiment.pre_failure_mapping
    }
    assert all(count > 0 for count in dag_summary["task_count_by_tenant"].values())
    solver = FailureAwareRepairTimeExpandedILPSolver(
        scenario,
        evaluator,
        verbose=False,
        horizon_slots=8,
    )
    solver.model.update()

    assert solver.model.NumVars > 0
    assert solver.model.NumConstrs > 0


def test_collapsed_repair_milp_forbids_healthy_to_healthy_rank_moves():
    from failure_aware_repair.evaluator import RepairEvaluator
    from failure_aware_repair.milp import CollapsedRepairMILPSolver
    from failure_aware_repair.models import FailureEvent, RepairScenario
    from multitenant.topology import LeafSpineDatacenter

    datacenter = LeafSpineDatacenter(num_leaf=2, num_spine=2, per_leaf_server=3)
    mapping = {0: {0: 0, 1: 1}, 1: {0: 2, 1: 3}}
    scenario = RepairScenario(
        mapping,
        FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        "cooperative",
        global_protection_pool=(4, 5),
    )
    evaluator = RepairEvaluator(
        datacenter,
        scenario,
        single_flow_size_bits=BITS_PER_MB // 8,
        collective="allgather",
        horizon_slots=8,
    )

    solver = CollapsedRepairMILPSolver(scenario, evaluator)
    result = solver.solve(time_limit=5)

    check_repair_feasible(scenario, result.mapping, evaluator.failover_mapping)
    assert solver.model.getConstrByName("forbid_healthy_to_healthy_0_1_0") is not None
    assert solver.model.getConstrByName("forbid_healthy_to_healthy_1_0_3") is not None


def test_strategy_solver_exposes_original_mapping_style_api():
    experiment = generate_random_repair_experiment(_fast_config(seed=7))
    solver = FailureAwareMappingStrategySolver(
        experiment.problem,
        TENANT_LOCAL_REPAIR,
        search_config=experiment.config.repair_search,
    )
    result = solver.solve(time_limit=2)

    assert result.strategy.name == "tenant_local_repair"
    assert solver.get_X_mapping() == result.mapping
    assert solver.get_objective() == result.objective
    assert solver.to_strategy_result() == result
    assert result.metadata["solver"] == "tenant_local_repair"


def test_low_contention_random_seed_runs_balanced_repair_pipeline():
    config = RandomRepairExperimentConfig(
        seed=11,
        num_leaf=4,
        num_spine=2,
        per_leaf_server=6,
        num_tenants=3,
        ranks_per_tenant=None,
        working_allocation_mode="balanced_remaining",
        contention="low",
        workload_mode="synthetic_uniform",
        single_flow_size_bits=4 * BITS_PER_MB,
        working_mapping_time_limit=3,
        repair_time_limit=5,
        repair_search=RepairSearchConfig(
            beam_width=5,
            max_rounds=3,
            max_candidates_per_tenant=32,
            max_participating_tenants=3,
        ),
    )
    comparison = run_repair_comparison(generate_random_repair_experiment(config))

    assert comparison.failover_objective.avg_jct > 0
    assert comparison.tenant_local.objective.avg_jct > 0
    assert comparison.cooperative.objective.avg_jct > 0
    check_repair_feasible(
        comparison.tenant_local_scenario,
        comparison.tenant_local.mapping,
        comparison.failover_mapping,
    )
    check_repair_feasible(
        comparison.cooperative_scenario,
        comparison.cooperative.mapping,
        comparison.failover_mapping,
    )
    assert comparison.tenant_local.metadata["joint_guided_candidates"] == 0
    assert comparison.cooperative.metadata["solver"]


def test_repair_strategies_are_simulator_monotone_by_candidate_inclusion():
    experiment = generate_random_repair_experiment(_fast_config(seed=13))
    comparison = run_repair_comparison(experiment)
    payload = repair_comparison_payload(comparison)

    baseline = solve_nearest_protection_baseline(
        comparison.failover_scenario,
        datacenter=experiment.datacenter,
    )
    evaluator = comparison.strategy_results[REPAIR_FAILED_SERVER_ONLY.name].evaluator
    _baseline_makespan, baseline_avg_jct = evaluator.simulate(baseline.mapping)

    failover_avg_jct = payload["results"][REPAIR_FAILED_SERVER_ONLY.name]["simulation"]["avg_jct"]
    local_avg_jct = payload["results"][TENANT_LOCAL_REPAIR.name]["simulation"]["avg_jct"]
    cooperative_avg_jct = payload["results"][COOPERATIVE_REPAIR.name]["simulation"]["avg_jct"]

    assert failover_avg_jct <= baseline_avg_jct
    assert local_avg_jct <= failover_avg_jct
    assert cooperative_avg_jct <= local_avg_jct
    assert (
        comparison.cooperative.metadata["algorithm_version"]
        == comparison.tenant_local.metadata["algorithm_version"]
    )


def test_critical_failover_failure_selection_enumerates_working_servers():
    config = RandomRepairExperimentConfig(
        **{
            **_fast_config(seed=9).__dict__,
            "failure_selection": "critical_failover",
            "failover_policy": "same_leaf_or_nearest",
        }
    )
    experiment = generate_random_repair_experiment(config)
    comparison = run_repair_comparison(experiment)
    payload = repair_comparison_payload(comparison)

    working_total = sum(len(ranks) for ranks in experiment.pre_failure_mapping.values())
    metadata = payload["failure_selection_metadata"]
    assert payload["failure_selection"] == "critical_failover"
    assert metadata["mode"] == "critical_failover"
    assert metadata["evaluated_failures"] == working_total
    assert metadata["selected_failover_avg_jct"] == comparison.failover_objective.avg_jct
