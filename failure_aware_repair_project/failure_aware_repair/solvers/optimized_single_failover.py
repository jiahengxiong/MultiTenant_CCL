from __future__ import annotations

from dataclasses import dataclass

from ..models import Mapping, RepairObjective, RepairScenario, SwitchCounts
from ..objectives import repair_better
from ..protection import (
    check_repair_feasible,
    choose_failover_protection,
    copy_mapping,
    count_switches,
    infer_failed_rank,
    validate_global_protection_pool,
)


@dataclass(frozen=True)
class OptimizedSingleFailoverResult:
    """Simulator-selected failed-rank-only replacement over protection servers."""

    mapping: Mapping
    objective: RepairObjective
    switch_counts: SwitchCounts
    failed_rank: int
    replacement_server: int
    nearest_baseline_server: int
    evaluated_candidates: int
    simulator_scored_candidates: int


class OptimizedSingleFailoverSolver:
    """Search the failover-only space without moving healthy ranks."""

    name = "optimized_single_failed_rank_failover"

    def __init__(
        self,
        scenario: RepairScenario,
        evaluator,
        *,
        datacenter=None,
        simulator_candidate_limit: int = 3,
    ):
        self.scenario = scenario
        self.evaluator = evaluator
        self.datacenter = datacenter
        self.simulator_candidate_limit = max(1, int(simulator_candidate_limit))

    def solve(self) -> OptimizedSingleFailoverResult:
        validate_global_protection_pool(
            self.scenario.pre_failure_mapping,
            self.scenario.global_protection_pool,
        )
        failure = self.scenario.failure
        failed_rank = infer_failed_rank(self.scenario.pre_failure_mapping, failure)
        nearest = choose_failover_protection(
            self.scenario.global_protection_pool,
            failed_server=int(failure.failed_server),
            datacenter=self.datacenter,
            policy="same_leaf_or_nearest",
        )

        estimated_candidates = []
        for replacement in sorted(int(server) for server in self.scenario.global_protection_pool):
            mapping = copy_mapping(self.scenario.pre_failure_mapping)
            mapping[int(failure.tenant)][int(failed_rank)] = int(replacement)
            check_repair_feasible(self.scenario, mapping, mapping)
            estimated_candidates.append((self.evaluator.estimate(mapping), mapping, int(replacement)))

        if not estimated_candidates:
            raise RuntimeError("no feasible single-server failover candidate was evaluated")
        estimated_candidates.sort(key=lambda entry: (
            float(entry[0].avg_jct),
            float(entry[0].makespan),
            int(entry[0].extra_switches),
            int(entry[2]),
        ))

        simulator_candidates = []
        seen_replacements = set()

        def add_candidate(entry) -> None:
            replacement = int(entry[2])
            if replacement in seen_replacements:
                return
            seen_replacements.add(replacement)
            simulator_candidates.append(entry)

        for entry in estimated_candidates:
            if int(entry[2]) == int(nearest):
                add_candidate(entry)
                break
        for entry in estimated_candidates:
            add_candidate(entry)
            if len(simulator_candidates) >= self.simulator_candidate_limit:
                break

        best_mapping: Mapping | None = None
        best_objective = RepairObjective.infinity()
        best_replacement: int | None = None
        for _estimate, mapping, replacement in simulator_candidates:
            makespan, avg_jct = self.evaluator.simulate(mapping)
            candidate_objective = RepairObjective(
                avg_jct=float(avg_jct),
                makespan=float(makespan),
                extra_switches=0,
            )
            if repair_better(candidate_objective, best_objective):
                best_mapping = mapping
                best_objective = candidate_objective
                best_replacement = int(replacement)

        if best_mapping is None or best_replacement is None:
            raise RuntimeError("no feasible single-server failover simulator candidate was evaluated")

        best_switch_counts = count_switches(
            pre_failure_mapping=self.scenario.pre_failure_mapping,
            failover_mapping=best_mapping,
            repaired_mapping=best_mapping,
            failure=failure,
        )

        return OptimizedSingleFailoverResult(
            mapping=best_mapping,
            objective=best_objective,
            switch_counts=best_switch_counts,
            failed_rank=int(failed_rank),
            replacement_server=int(best_replacement),
            nearest_baseline_server=int(nearest),
            evaluated_candidates=int(len(estimated_candidates)),
            simulator_scored_candidates=int(len(simulator_candidates)),
        )


def solve_optimized_single_failover(
    scenario: RepairScenario,
    evaluator,
    *,
    datacenter=None,
    simulator_candidate_limit: int = 3,
) -> OptimizedSingleFailoverResult:
    return OptimizedSingleFailoverSolver(
        scenario,
        evaluator,
        datacenter=datacenter,
        simulator_candidate_limit=simulator_candidate_limit,
    ).solve()
