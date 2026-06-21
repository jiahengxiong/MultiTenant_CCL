from __future__ import annotations

import itertools
import time
import os
from dataclasses import dataclass

from .heuristic import REPAIR_ALGORITHM_VERSION, RepairSearchConfig
from .milp import CollapsedRepairMILPSolver
from .models import Mapping, RepairObjective, RepairScenario, SwitchCounts
from .objectives import repair_better
from .problem import FailureAwareMappingProblem
from .protection import (
    check_repair_feasible,
    copy_mapping,
    count_switches,
    infer_failed_rank,
    participating_tenants,
)
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


def _collapsed_warm_rank_allowlist(
    scenario,
    evaluator,
    mapping: Mapping,
) -> set[tuple[int, int]]:
    failed_rank = scenario.failure.failed_rank
    if failed_rank is None:
        failed_rank = infer_failed_rank(
            scenario.pre_failure_mapping,
            scenario.failure,
        )
    failed_key = (int(scenario.failure.tenant), int(failed_rank))
    participants = set(participating_tenants(scenario))
    protection_slots = max(1, len(scenario.global_protection_pool))
    allowlist: set[tuple[int, int]] = {failed_key}
    combined_pressure: dict[tuple[int, int], float] = {}
    for pressure_mapping in (evaluator.failover_mapping, mapping):
        try:
            analysis = evaluator.analyze(pressure_mapping)
            rank_pressure = getattr(analysis, "rank_pressure", {}) or {}
        except Exception:
            rank_pressure = {}
        for key, pressure in rank_pressure.items():
            try:
                tenant, rank = key
                tenant = int(tenant)
                rank = int(rank)
            except Exception:
                continue
            combined_pressure[(tenant, rank)] = max(
                float(combined_pressure.get((tenant, rank), 0.0)),
                float(pressure),
            )

    scored: list[tuple[float, int, int]] = []
    for key, pressure in combined_pressure.items():
        try:
            tenant, rank = key
            tenant = int(tenant)
            rank = int(rank)
        except Exception:
            continue
        if int(tenant) not in participants:
            continue
        if (tenant, rank) == failed_key:
            continue
        if tenant not in scenario.pre_failure_mapping:
            continue
        if rank not in scenario.pre_failure_mapping[tenant]:
            continue
        scored.append((float(pressure), tenant, rank))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    per_tenant_quota = 2 if len(participants) > 1 else protection_slots
    for participant in sorted(participants):
        tenant_added = 0
        for _pressure, tenant, rank in scored:
            if tenant != int(participant):
                continue
            if (tenant, rank) in allowlist:
                continue
            allowlist.add((int(tenant), int(rank)))
            tenant_added += 1
            if len(allowlist) >= protection_slots or tenant_added >= per_tenant_quota:
                break
        if len(allowlist) >= protection_slots:
            break
    for _pressure, tenant, rank in scored:
        if (tenant, rank) in allowlist:
            continue
        allowlist.add((int(tenant), int(rank)))
        if len(allowlist) >= protection_slots:
            break
    return allowlist


def _slot_assignment_warm_start(
    scenario,
    evaluator,
    base_mapping: Mapping,
    rank_allowlist: set[tuple[int, int]],
    *,
    max_extra_moves: int = 2,
    deadline: float = float("inf"),
) -> tuple[Mapping, RepairObjective, dict[str, object]]:
    failed_rank = scenario.failure.failed_rank
    if failed_rank is None:
        failed_rank = infer_failed_rank(
            scenario.pre_failure_mapping,
            scenario.failure,
        )
    failed_key = (int(scenario.failure.tenant), int(failed_rank))
    protection = tuple(int(server) for server in scenario.global_protection_pool)
    optional_ranks = [
        (int(tenant), int(rank))
        for tenant, rank in sorted(rank_allowlist)
        if (int(tenant), int(rank)) != failed_key
    ]
    best_mapping = copy_mapping(base_mapping)
    best_objective = evaluator.estimate(best_mapping)
    evaluated = 1
    stopped = None
    max_extra_moves = max(0, min(int(max_extra_moves), len(optional_ranks), len(protection) - 1))

    for failed_server in protection:
        if time.time() >= deadline:
            stopped = "time_limit"
            break
        for extra_count in range(0, max_extra_moves + 1):
            for moved_ranks in itertools.combinations(optional_ranks, extra_count):
                if time.time() >= deadline:
                    stopped = "time_limit"
                    break
                remaining_slots = tuple(server for server in protection if int(server) != int(failed_server))
                for extra_servers in itertools.permutations(remaining_slots, extra_count):
                    if time.time() >= deadline:
                        stopped = "time_limit"
                        break
                    candidate = copy_mapping(base_mapping)
                    candidate[int(failed_key[0])][int(failed_key[1])] = int(failed_server)
                    for (tenant, rank), server in zip(moved_ranks, extra_servers):
                        candidate[int(tenant)][int(rank)] = int(server)
                    try:
                        check_repair_feasible(scenario, candidate, evaluator.failover_mapping)
                    except Exception:
                        continue
                    objective = evaluator.estimate(candidate)
                    evaluated += 1
                    if repair_better(objective, best_objective):
                        best_mapping = copy_mapping(candidate)
                        best_objective = objective
                if stopped:
                    break
            if stopped:
                break
        if stopped:
            break

    return best_mapping, best_objective, {
        "slot_assignment_warm_start_evaluated": int(evaluated),
        "slot_assignment_warm_start_stopped": stopped,
        "slot_assignment_warm_start_max_extra_moves": int(max_extra_moves),
        "slot_assignment_warm_start_avg_jct": float(best_objective.avg_jct),
        "slot_assignment_warm_start_makespan": float(best_objective.makespan),
        "slot_assignment_warm_start_extra_switches": int(best_objective.extra_switches),
    }


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

    if strategy.name == REPAIR_FAILED_SERVER_ONLY.name and not search_config.use_collapsed_milp_candidate:
        evaluator = problem.evaluator(strategy)
        mapping = copy_mapping(evaluator.failover_mapping)
        check_repair_feasible(scenario, mapping, evaluator.failover_mapping)
        objective = evaluator.estimate(mapping)
        failed_rank = infer_failed_rank(scenario.pre_failure_mapping, scenario.failure)
        replacement_server = int(mapping[int(scenario.failure.tenant)][int(failed_rank)])
        return StrategySolveResult(
            strategy=strategy,
            scenario=scenario,
            evaluator=evaluator,
            mapping=mapping,
            objective=objective,
            switch_counts=count_switches(
                pre_failure_mapping=scenario.pre_failure_mapping,
                failover_mapping=evaluator.failover_mapping,
                repaired_mapping=mapping,
                failure=scenario.failure,
            ),
            runtime_seconds=time.time() - start,
            metadata={
                "solver": "direct_failed_server_failover_mapping",
                "algorithm_version": REPAIR_ALGORITHM_VERSION,
                "replacement_server": int(replacement_server),
                "search_space": "failed_rank_to_failover_protection_slot",
                "optimizes_mapping": False,
            },
        )

    if search_config.use_collapsed_milp_candidate:
        evaluator = problem.evaluator(strategy, failover_mapping=failover_mapping)
        milp_time_limit = None if time_limit == float("inf") else float(time_limit)
        seed_mapping = copy_mapping(evaluator.failover_mapping)
        seed_source = "failover"
        seed_objective = evaluator.estimate(seed_mapping)
        for candidate_mapping, candidate_source in reference_seeds or []:
            try:
                check_repair_feasible(
                    scenario,
                    candidate_mapping,
                    evaluator.failover_mapping,
                )
            except Exception:
                continue
            candidate_objective = evaluator.estimate(candidate_mapping)
            if repair_better(candidate_objective, seed_objective):
                seed_mapping = copy_mapping(candidate_mapping)
                seed_source = str(candidate_source)
                seed_objective = candidate_objective
        warm_metadata: dict[str, object] = {}
        if strategy.name != REPAIR_FAILED_SERVER_ONLY.name:
            warm_base_mapping = (
                copy_mapping(evaluator.failover_mapping)
                if strategy.name == COOPERATIVE_REPAIR.name
                else copy_mapping(seed_mapping)
            )
            warm_rank_allowlist = _collapsed_warm_rank_allowlist(
                scenario,
                evaluator,
                warm_base_mapping,
            )
            if len(warm_rank_allowlist) > 1:
                slot_deadline = (
                    float("inf")
                    if milp_time_limit is None
                    else time.time() + max(1.0, min(3.0, float(milp_time_limit) * 0.3))
                )
                (
                    slot_mapping,
                    slot_objective,
                    slot_metadata,
                ) = _slot_assignment_warm_start(
                    scenario,
                    evaluator,
                    warm_base_mapping,
                    warm_rank_allowlist,
                    max_extra_moves=2,
                    deadline=slot_deadline,
                )
                if repair_better(slot_objective, seed_objective):
                    seed_mapping = copy_mapping(slot_mapping)
                    seed_source = "slot_assignment_warm_start"
                    seed_objective = slot_objective
                warm_limit = (
                    None
                    if milp_time_limit is None
                    else max(1.0, min(3.0, float(milp_time_limit) * 0.3))
                )
                warm_result = CollapsedRepairMILPSolver(
                    scenario,
                    evaluator,
                    movable_rank_allowlist=warm_rank_allowlist,
                ).solve(
                    time_limit=warm_limit,
                    verbose=os.environ.get("FAILURE_REPAIR_VERBOSE_MILP") == "1",
                    initial_mapping=warm_base_mapping,
                )
                if repair_better(warm_result.objective, seed_objective):
                    seed_mapping = copy_mapping(warm_result.mapping)
                    seed_source = "restricted_collapsed_milp_warm_start"
                    seed_objective = warm_result.objective
                warm_metadata = {
                    "restricted_warm_start_attempted": True,
                    "restricted_warm_start_rank_allowlist": [
                        {"tenant": int(tenant), "rank": int(rank)}
                        for tenant, rank in sorted(warm_rank_allowlist)
                    ],
                    "restricted_warm_start_status": warm_result.metadata.get("gurobi_status"),
                    "restricted_warm_start_sol_count": warm_result.metadata.get("gurobi_sol_count"),
                    "restricted_warm_start_avg_jct": float(warm_result.objective.avg_jct),
                    "restricted_warm_start_makespan": float(warm_result.objective.makespan),
                    "restricted_warm_start_extra_switches": int(warm_result.objective.extra_switches),
                    "restricted_warm_start_selected": seed_source == "restricted_collapsed_milp_warm_start",
                    **slot_metadata,
                    "slot_assignment_warm_start_selected": seed_source == "slot_assignment_warm_start",
                }
        result = CollapsedRepairMILPSolver(
            scenario,
            evaluator,
            failed_rank_only=strategy.name == REPAIR_FAILED_SERVER_ONLY.name,
        ).solve(
            time_limit=milp_time_limit,
            verbose=os.environ.get("FAILURE_REPAIR_VERBOSE_MILP") == "1",
            initial_mapping=seed_mapping,
        )
        if repair_better(seed_objective, result.objective):
            result = result.__class__(
                name=result.name,
                mapping=copy_mapping(seed_mapping),
                objective=seed_objective,
                switch_counts=count_switches(
                    pre_failure_mapping=scenario.pre_failure_mapping,
                    failover_mapping=evaluator.failover_mapping,
                    repaired_mapping=seed_mapping,
                    failure=scenario.failure,
                ),
                runtime_seconds=result.runtime_seconds,
                metadata={
                    **dict(result.metadata),
                    "incumbent_retained_after_full_milp": True,
                    "retained_incumbent_source": seed_source,
                },
            )
        check_repair_feasible(scenario, result.mapping, evaluator.failover_mapping)
        metadata = dict(result.metadata)
        metadata.update(
            {
                "solver": "collapsed_repair_milp",
                "algorithm_version": REPAIR_ALGORITHM_VERSION,
                "collapsed_milp_selected": True,
                "heuristic_bypass": True,
                "initial_solution_source": seed_source,
                "lexicographic_objective_order": [
                    "avg_jct",
                    "makespan",
                    "extra_servers_switched_to_protection_set",
                ],
                **warm_metadata,
            }
        )
        return StrategySolveResult(
            strategy=strategy,
            scenario=scenario,
            evaluator=evaluator,
            mapping=result.mapping,
            objective=result.objective,
            switch_counts=result.switch_counts,
            runtime_seconds=time.time() - start,
            metadata=metadata,
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
