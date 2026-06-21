from __future__ import annotations

import time
from dataclasses import dataclass

from .heuristic import REPAIR_ALGORITHM_VERSION, RepairSearchConfig
from .milp import CollapsedRepairMILPSolver
from .models import Mapping, RepairObjective, RepairScenario, SwitchCounts
from .objectives import repair_better
from .problem import FailureAwareMappingProblem
from .protection import check_repair_feasible, copy_mapping, infer_failed_rank
from .solvers import solve_contention_guided_repair
from .strategies import (
    COOPERATIVE_REPAIR,
    REPAIR_FAILED_SERVER_ONLY,
    TENANT_LOCAL_REPAIR,
    RepairStrategySpec,
)


@dataclass(frozen=True)
class StrategySolveResult:
    """Uniform result for one strategy on a fixed failure-aware mapping problem."""

    strategy: RepairStrategySpec
    scenario: RepairScenario
    evaluator: object
    mapping: Mapping
    objective: RepairObjective
    switch_counts: SwitchCounts
    runtime_seconds: float
    metadata: dict[str, object]


class FailureAwareMappingStrategySolver:
    """Original-mapping-style solver facade for one repair strategy."""

    name = "failure_aware_mapping_strategy_solver"

    def __init__(
        self,
        problem: FailureAwareMappingProblem,
        strategy: RepairStrategySpec,
        *,
        search_config: RepairSearchConfig | None = None,
        verbose: bool = False,
    ):
        self.problem = problem
        self.strategy = strategy
        self.search_config = search_config or RepairSearchConfig()
        self.verbose = bool(verbose)
        self.result: StrategySolveResult | None = None

    def solve(self, *, time_limit: float | None = None) -> StrategySolveResult:
        self.result = _solve_repair_strategy_impl(
            self.problem,
            self.strategy,
            search_config=self.search_config,
            time_limit=float("inf") if time_limit is None else float(time_limit),
        )
        return self.result

    def get_X_mapping(self) -> Mapping:
        if self.result is None:
            raise RuntimeError("solve() must be called before get_X_mapping()")
        return copy_mapping(self.result.mapping)

    def get_objective(self) -> RepairObjective:
        if self.result is None:
            raise RuntimeError("solve() must be called before get_objective()")
        return self.result.objective

    def to_strategy_result(self) -> StrategySolveResult:
        if self.result is None:
            raise RuntimeError("solve() must be called before to_strategy_result()")
        return self.result


def solve_repair_strategy(
    problem: FailureAwareMappingProblem,
    strategy: RepairStrategySpec,
    *,
    search_config: RepairSearchConfig,
    time_limit: float | None,
    failover_mapping: Mapping | None = None,
    reference_seeds: list[tuple[Mapping, str]] | None = None,
) -> StrategySolveResult:
    """Solve one strategy using the same problem/evaluator construction path."""

    return _solve_repair_strategy_impl(
        problem,
        strategy,
        search_config=search_config,
        time_limit=float("inf") if time_limit is None else float(time_limit),
        failover_mapping=failover_mapping,
        reference_seeds=reference_seeds,
    )


def _solve_repair_strategy_impl(
    problem: FailureAwareMappingProblem,
    strategy: RepairStrategySpec,
    *,
    search_config: RepairSearchConfig,
    time_limit: float | None,
    failover_mapping: Mapping | None = None,
    reference_seeds: list[tuple[Mapping, str]] | None = None,
) -> StrategySolveResult:
    start = time.time()
    scenario = problem.scenario(strategy)

    if strategy.name == REPAIR_FAILED_SERVER_ONLY.name:
        evaluator = problem.evaluator(strategy)
        result = solve_contention_guided_repair(
            scenario,
            evaluator,
            strategy,
            config=search_config,
            reference_seeds=reference_seeds,
            time_limit=time_limit,
        )
        mapping = result.mapping
        check_repair_feasible(scenario, mapping, evaluator.failover_mapping)
        failed_rank = infer_failed_rank(scenario.pre_failure_mapping, scenario.failure)
        replacement_server = int(mapping[int(scenario.failure.tenant)][int(failed_rank)])
        nearest_baseline_server = int(
            evaluator.failover_mapping[int(scenario.failure.tenant)][int(failed_rank)]
        )
        return StrategySolveResult(
            strategy=strategy,
            scenario=scenario,
            evaluator=evaluator,
            mapping=mapping,
            objective=result.objective,
            switch_counts=result.switch_counts,
            runtime_seconds=time.time() - start,
            metadata={
                "solver": "unified_gurobi_master_subproblem_repair_optimizer",
                "algorithm_version": REPAIR_ALGORITHM_VERSION,
                "decomposition_algorithm": "contention_guided_logic_based_candidate_set_decomposition",
                "logic_benders_master_solver": "fixed_failed_rank_candidate_set",
                "logic_benders_subproblem_solver": "gurobi_unique_protection_slot_assignment_subproblem_with_contention_estimator",
                "decomposition_iteration_policy": dict(
                    result.metadata.get("decomposition_iteration_policy", {})
                ),
                "lexicographic_objective_order": [
                    "avg_jct",
                    "makespan",
                    "extra_servers_switched_to_protection_set",
                ],
                "optimizes_mapping": True,
                "selection_metric": "simulator_avg_jct_then_makespan_then_switches",
                "search_space": "failed_rank_plus_0_healthy_ranks_to_unique_protection_slots",
                "movable_tenant_scope": "failed_tenant",
                "max_healthy_moves": 0,
                "evaluated_candidates": int(result.metadata.get("evaluated_candidates", 0)),
                "simulator_scored_candidates": int(result.metadata.get("simulator_scored_candidates", 0)),
                "replacement_server": int(replacement_server),
                "nearest_baseline_server": int(nearest_baseline_server),
                "contains_nearest_baseline_candidate": True,
            },
        )

    evaluator = problem.evaluator(strategy, failover_mapping=failover_mapping)
    if strategy.name == TENANT_LOCAL_REPAIR.name:
        result = solve_contention_guided_repair(
            scenario,
            evaluator,
            strategy,
            config=search_config,
            reference_seeds=reference_seeds,
            time_limit=time_limit,
        )
    elif strategy.name == COOPERATIVE_REPAIR.name:
        optimizer_result = solve_contention_guided_repair(
            scenario,
            evaluator,
            strategy,
            config=search_config,
            reference_seeds=reference_seeds,
            time_limit=time_limit,
        )
        if search_config.use_collapsed_milp_candidate:
            result = _best_with_collapsed_milp_candidate(
                scenario,
                evaluator,
                incumbent=optimizer_result,
                time_limit=time_limit,
            )
        else:
            optimizer_result.metadata.update(
                {
                    "collapsed_milp_attempted": False,
                    "collapsed_milp_reason": "disabled_for_large_scale_heuristic_experiment",
                }
            )
            result = optimizer_result
    else:
        raise ValueError(f"unknown repair strategy: {strategy.name}")

    check_repair_feasible(scenario, result.mapping, evaluator.failover_mapping)
    return StrategySolveResult(
        strategy=strategy,
        scenario=scenario,
        evaluator=evaluator,
        mapping=result.mapping,
        objective=result.objective,
        switch_counts=result.switch_counts,
        runtime_seconds=result.runtime_seconds,
        metadata={
            **result.metadata,
            "solver": strategy.name,
            "optimizer": result.metadata.get("optimizer", result.name),
            "optimizes_mapping": True,
        },
    )


def _best_with_collapsed_milp_candidate(
    scenario: RepairScenario,
    evaluator,
    *,
    incumbent,
    time_limit: float | None,
):
    """Add the collapsed repair MILP as an estimator-only candidate.

    The simulator is intentionally not used here.  This function only compares
    the heuristic and MILP candidate with the same estimator objective used by
    the repair algorithms, then leaves simulator validation to reporting code.
    """

    metadata = {
        "heuristic_solver": incumbent.name,
        "collapsed_milp_attempted": True,
    }
    try:
        milp_time_limit = None if time_limit == float("inf") else float(time_limit)
        milp_result = CollapsedRepairMILPSolver(scenario, evaluator).solve(
            time_limit=milp_time_limit,
        )
    except Exception as exc:
        incumbent.metadata.update(
            {
                **metadata,
                "collapsed_milp_selected": False,
                "collapsed_milp_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return incumbent

    if repair_better(milp_result.objective, incumbent.objective):
        milp_result.metadata.update(
            {
                **metadata,
                "collapsed_milp_selected": True,
                "heuristic_avg_jct": float(incumbent.objective.avg_jct),
                "heuristic_makespan": float(incumbent.objective.makespan),
                "heuristic_extra_switches": int(incumbent.objective.extra_switches),
            }
        )
        return milp_result

    incumbent.metadata.update(
        {
            **metadata,
            "collapsed_milp_selected": False,
            "collapsed_milp_avg_jct": float(milp_result.objective.avg_jct),
            "collapsed_milp_makespan": float(milp_result.objective.makespan),
            "collapsed_milp_extra_switches": int(milp_result.objective.extra_switches),
            "collapsed_milp_status": milp_result.metadata.get("gurobi_status"),
            "collapsed_milp_sol_count": milp_result.metadata.get("gurobi_sol_count"),
        }
    )
    return incumbent


def solve_repair_strategies(
    problem: FailureAwareMappingProblem,
    strategies: tuple[RepairStrategySpec, ...],
    *,
    search_config: RepairSearchConfig,
    time_limit: float | None,
) -> dict[str, StrategySolveResult]:
    results: dict[str, StrategySolveResult] = {}
    failover_reference: Mapping | None = None
    reference_seeds: list[tuple[Mapping, str]] = []
    for strategy in strategies:
        result = solve_repair_strategy(
            problem,
            strategy,
            search_config=search_config,
            time_limit=time_limit,
            failover_mapping=failover_reference,
            reference_seeds=reference_seeds,
        )
        results[strategy.name] = result
        if strategy.name == REPAIR_FAILED_SERVER_ONLY.name:
            failover_reference = copy_mapping(result.mapping)
            reference_seeds.append((copy_mapping(result.mapping), strategy.name))
        elif strategy.name == TENANT_LOCAL_REPAIR.name:
            reference_seeds.append((copy_mapping(result.mapping), strategy.name))
    return results
