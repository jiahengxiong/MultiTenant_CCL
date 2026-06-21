from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from .evaluator import RepairEvaluator
from .models import Mapping, RepairResult, RepairScenario
from .objectives import repair_better
from .protection import (
    build_candidate_server_sets,
    check_repair_feasible,
    copy_mapping,
    count_switches,
    infer_failed_rank,
    participating_tenants,
)


REPAIR_ALGORITHM_VERSION = "heuristic_contention_guided_candidate_set_search_v48"


@dataclass
class RepairSearchConfig:
    beam_width: int = 6
    max_rounds: int = 5
    max_candidates_per_tenant: int = 48
    max_participating_tenants: int = 4
    max_extra_switches_per_tenant: int | None = None
    max_joint_tenants: int = 2
    joint_candidates_per_tenant: int = 4
    max_joint_candidates: int = 48
    max_block_ranks: int = 4
    block_extra_servers: int = 4
    max_block_candidates: int = 96
    cooperative_near_tie_slack: float = 0.0
    score_sort_tolerance: float = 1e-5
    use_collapsed_milp_candidate: bool = False
    master_iteration_budget: int | None = None
    simulator_candidate_budget: int | None = None


class FailureAwareRepairHeuristicBase:
    """Shared content-guided master/subproblem repair optimizer."""

    name = "failure_aware_repair"

    def __init__(
        self,
        scenario: RepairScenario,
        evaluator: RepairEvaluator,
        *,
        config: RepairSearchConfig | None = None,
        verbose: bool = False,
        reference_seeds: list[tuple[Mapping, str]] | None = None,
    ):
        self.scenario = scenario
        self.evaluator = evaluator
        self.config = config or RepairSearchConfig()
        self.verbose = bool(verbose)
        self.failover_mapping = copy_mapping(evaluator.failover_mapping)
        self.candidate_sets = build_candidate_server_sets(scenario)
        self.price_guided_candidates = 0
        self.price_lookup_fallbacks = 0
        self.joint_guided_candidates = 0
        self.initial_joint_seed_candidates = 0
        self.logic_benders_master_selections = 0
        self.logic_benders_subproblem_states = 0
        self.logic_benders_cuts_generated = 0
        self.logic_benders_cuts_applied = 0
        self.logic_benders_selection_candidates = 0
        self.contention_feedback_cuts_generated = 0
        self.contention_feedback_cuts_applied = 0
        self.contention_feedback_records: list[dict[str, object]] = []
        self.reference_seeds = [
            (copy_mapping(mapping), str(source))
            for mapping, source in (reference_seeds or [])
        ]

    def allowed_tenants(self) -> tuple[int, ...]:
        return participating_tenants(self.scenario)

    def solve(self, *, time_limit: float | None = None) -> RepairResult:
        start = time.time()
        deadline = float("inf") if time_limit is None else start + float(time_limit)

        failover = copy_mapping(self.failover_mapping)
        initial_entries = self._initial_seed_entries(deadline)
        if not initial_entries:
            initial_entries = [(self.evaluator.estimate(failover), failover, "failover")]

        best_obj, best_mapping, best_source = sorted(initial_entries, key=self._entry_sort_key)[0]
        all_entries = list(initial_entries)
        rounds = 0
        evaluated = len(initial_entries)

        beam = []
        while not self._use_independent_slot_search_only() and rounds < self.config.max_rounds and time.time() < deadline:
            if not beam:
                beam = self._select_beam(initial_entries)
            rounds += 1
            entries = list(beam)
            improved = False
            for _obj, mapping, _source in beam:
                if time.time() >= deadline:
                    break
                analysis = self._safe_analysis(mapping)
                tenant_candidate_cache = {}
                for tenant in self._tenant_order(mapping, analysis):
                    if time.time() >= deadline:
                        break
                    tenant_candidates = self._ranked_tenant_candidates(
                        mapping,
                        int(tenant),
                        analysis,
                        deadline,
                        limit=self.config.max_candidates_per_tenant,
                    )
                    tenant_candidate_cache[int(tenant)] = tenant_candidates
                    for candidate, source, _proposal_cost in tenant_candidates:
                        if time.time() >= deadline:
                            break
                        try:
                            check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                        except ValueError:
                            continue
                        obj = self.evaluator.estimate(candidate)
                        evaluated += 1
                        entry = (obj, candidate, source)
                        entries.append(entry)
                        all_entries.append(entry)
                        if self._candidate_better(
                            obj,
                            best_obj,
                            candidate,
                            source,
                            best_mapping,
                            best_source,
                        ):
                            best_obj, best_mapping, best_source = obj, candidate, source
                            improved = True

                for candidate, source in self._joint_candidates(
                    mapping,
                    tenant_candidate_cache,
                    deadline,
                ):
                    if time.time() >= deadline:
                        break
                    try:
                        check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                    except ValueError:
                        continue
                    obj = self.evaluator.estimate(candidate)
                    evaluated += 1
                    entry = (obj, candidate, source)
                    entries.append(entry)
                    all_entries.append(entry)
                    if self._candidate_better(
                        obj,
                        best_obj,
                        candidate,
                        source,
                        best_mapping,
                        best_source,
                    ):
                        best_obj, best_mapping, best_source = obj, candidate, source
                        improved = True

            next_beam = self._select_beam(entries)
            if not improved and self._beam_signature(next_beam) == self._beam_signature(beam):
                break
            beam = next_beam

        best_obj, best_mapping, best_source = self._postprocess_best_mapping(
            best_obj,
            best_mapping,
            best_source,
            deadline,
        )
        all_entries.append((best_obj, best_mapping, best_source))
        simulator_entries = self._simulator_selection_entries(
            all_entries,
            (best_obj, best_mapping, best_source),
        )
        best_obj, best_mapping, best_source, simulator_candidates = (
            self._select_simulator_best(simulator_entries, deadline)
        )
        best_obj, best_mapping, best_source, greedy_simulator_candidates = (
            self._postprocess_simulator_selected_mapping(
                best_obj,
                best_mapping,
                best_source,
                deadline,
            )
        )
        switches = self.evaluator.switch_counts(best_mapping)
        pipeline_makespan, pipeline_avg_jct = self.evaluator.pipeline_score(best_mapping)
        return RepairResult(
            name=self.name,
            mapping=copy_mapping(best_mapping),
            objective=best_obj,
            switch_counts=switches,
            runtime_seconds=time.time() - start,
            metadata={
                "rounds": rounds,
                "algorithm_version": REPAIR_ALGORITHM_VERSION,
                "evaluated_candidates": evaluated,
                "generated_candidate_entries": len(all_entries),
                "best_source": best_source,
                "proposal_order": "independent_protection_slot_assignment",
                "decomposition_algorithm": "contention_guided_logic_based_candidate_set_decomposition",
                "logic_benders_master_solver": "gurobi_contention_guided_recovery_candidate_set_master",
                "logic_benders_subproblem_solver": "gurobi_unique_protection_slot_assignment_subproblem_with_contention_estimator",
                "switch_candidate_set_size_bounds": {
                    "total_min_including_failed_rank": 1,
                    "total_max_including_failed_rank": len(self.scenario.global_protection_pool),
                    "healthy_optional_min": 0,
                    "healthy_optional_max": max(0, len(self.scenario.global_protection_pool) - 1),
                    "master_cardinality_strategy": "candidate_set_size_is_a_master_variable_between_bounds",
                },
                "decomposition_iteration_policy": {
                    "master_iteration_budget": self._master_iteration_budget(
                        is_coordinate=self.name == "cooperative_repair"
                    ),
                    "subproblem_solution_count_per_candidate_set": 1,
                    "simulator_candidate_budget": self._simulator_pool_cap(),
                    "legacy_beam_width_used": False,
                },
                "lexicographic_objective_order": [
                    "avg_jct",
                    "makespan",
                    "extra_servers_switched_to_protection_set",
                ],
                "price_guided_candidates": self.price_guided_candidates,
                "joint_guided_candidates": self.joint_guided_candidates,
                "initial_joint_seed_candidates": self.initial_joint_seed_candidates,
                "logic_benders_master_selections": self.logic_benders_master_selections,
                "logic_benders_subproblem_states": self.logic_benders_subproblem_states,
                "logic_benders_cuts_generated": self.logic_benders_cuts_generated,
                "logic_benders_cuts_applied": self.logic_benders_cuts_applied,
                "logic_benders_selection_candidates": self.logic_benders_selection_candidates,
                "contention_feedback_cuts_generated": self.contention_feedback_cuts_generated,
                "contention_feedback_cuts_applied": self.contention_feedback_cuts_applied,
                "contention_feedback_records": self.contention_feedback_records[:12],
                "price_lookup_fallbacks": self.price_lookup_fallbacks,
                "simulator_candidate_pool_size": len(simulator_entries),
                "simulator_scored_candidates": int(simulator_candidates),
                "greedy_simulator_scored_candidates": int(greedy_simulator_candidates),
                "selection_metric": "simulator_avg_jct_then_makespan_then_extra_switches",
                "pipeline_makespan": float(pipeline_makespan),
                "pipeline_avg_jct": float(pipeline_avg_jct),
            },
        )

    def _initial_seed_entries(self, deadline: float):
        seeds: list[tuple[Mapping, str]] = [(copy_mapping(self.failover_mapping), "failover")]
        seen = {self._signature(self.failover_mapping)}
        for mapping, source in self.reference_seeds:
            sig = self._signature(mapping)
            if sig in seen:
                continue
            seen.add(sig)
            seeds.append((copy_mapping(mapping), f"reference:{source}"))
        analysis = self._safe_analysis(self.failover_mapping)
        for candidate, source in self._slot_assignment_candidates(
            analysis,
            deadline,
            limit=self._candidate_output_budget(is_coordinate=False),
            source_prefix="slot_assign_local",
        ):
            if time.time() >= deadline:
                break
            sig = self._signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            seeds.append((candidate, source))

        entries = []
        for mapping, source in seeds:
            try:
                check_repair_feasible(self.scenario, mapping, self.failover_mapping)
            except ValueError:
                continue
            entries.append((self.evaluator.estimate(mapping), mapping, source))
        return entries

    def _use_independent_slot_search_only(self) -> bool:
        return True

    def _master_iteration_budget(self, *, is_coordinate: bool) -> int:
        if self.config.master_iteration_budget is not None:
            return max(1, int(self.config.master_iteration_budget))
        protection_slots = max(1, len(self.scenario.global_protection_pool))
        return max(
            protection_slots * (12 if is_coordinate else 8),
            96 if is_coordinate else 64,
        )

    def _candidate_output_budget(self, *, is_coordinate: bool) -> int:
        return max(
            self._simulator_pool_cap(),
            min(
                self._master_iteration_budget(is_coordinate=is_coordinate),
                80 if is_coordinate else 48,
            ),
        )

    def _slot_assignment_candidates(
        self,
        analysis,
        deadline: float,
        *,
        limit: int,
        source_prefix: str,
        participants_override: tuple[int, ...] | None = None,
    ):
        """Generate bounded joint assignments of ranks to unique protection slots.

        The state space is deliberately defined in repair semantics instead of
        implementation history: the failed rank is assigned to exactly one
        protection server, and each optional healthy move consumes one distinct
        protection server.  Healthy ranks never move to other working servers.
        """

        protection = tuple(sorted(int(server) for server in self.scenario.global_protection_pool))
        if not protection:
            return

        failed_tenant = int(self.scenario.failure.tenant)
        failed_rank = infer_failed_rank(self.scenario.pre_failure_mapping, self.scenario.failure)
        if participants_override is None:
            participants = tuple(int(tenant) for tenant in self.allowed_tenants())
        else:
            participants = tuple(int(tenant) for tenant in participants_override)
        if failed_tenant not in participants:
            return

        is_coordinate = "coordinate" in str(source_prefix)
        allow_healthy_moves = "failover" not in str(source_prefix)
        rank_limit_per_tenant = max(
            len(ranks)
            for tenant, ranks in self.failover_mapping.items()
            if int(tenant) in participants
        )
        ordered_tenants = [
            int(tenant)
            for tenant in self._tenant_order(self.failover_mapping, analysis)
            if int(tenant) in participants
        ]
        if failed_tenant not in ordered_tenants:
            ordered_tenants.insert(0, failed_tenant)

        movable_by_tenant: dict[int, list[tuple[int, int, float]]] = {}
        seen_rank_keys = {(failed_tenant, int(failed_rank))}
        for tenant in ordered_tenants:
            if time.time() >= deadline:
                return
            ranks = self._impact_rank_order(self.failover_mapping, int(tenant), analysis)
            if int(tenant) == failed_tenant and int(failed_rank) not in ranks:
                ranks = [int(failed_rank), *ranks]
            for rank in ranks:
                rank = int(rank)
                key = (int(tenant), rank)
                if key in seen_rank_keys:
                    continue
                original_server = int(self.scenario.pre_failure_mapping[int(tenant)][rank])
                if original_server in protection:
                    continue
                seen_rank_keys.add(key)
                movable_by_tenant.setdefault(int(tenant), []).append(
                    (
                        int(tenant),
                        rank,
                        self._rank_structural_impact(
                            self.failover_mapping,
                            int(tenant),
                            rank,
                            analysis,
                        ),
                    )
                )
                if len(movable_by_tenant[int(tenant)]) >= rank_limit_per_tenant:
                    break

        for tenant, rows in list(movable_by_tenant.items()):
            movable_by_tenant[int(tenant)] = sorted(
                rows,
                key=lambda item: (-float(item[2]), item[1]),
            )

        movable: list[tuple[int, int, float]] = []
        if is_coordinate:
            for depth in range(rank_limit_per_tenant):
                for tenant in ordered_tenants:
                    tenant_rows = movable_by_tenant.get(int(tenant), [])
                    if depth < len(tenant_rows):
                        movable.append(tenant_rows[depth])
        else:
            movable = list(movable_by_tenant.get(failed_tenant, []))

        state_budget = self._master_iteration_budget(is_coordinate=is_coordinate)
        output_budget = max(1, int(limit), self._candidate_output_budget(is_coordinate=is_coordinate))
        all_states: list[tuple[object, Mapping, tuple[tuple[int, int, int], ...], frozenset[int], frozenset[tuple[int, int]]]] = []
        base_template = copy_mapping(self.scenario.pre_failure_mapping)
        failed_servers = [
            int(server)
            for server in protection
            if int(server) in self.candidate_sets[int(failed_tenant)]
        ]
        all_states.extend(
            self._logic_benders_assignment_subproblem(
                base_template,
                analysis,
                deadline,
                protection=protection,
                failed_tenant=failed_tenant,
                failed_rank=int(failed_rank),
                failed_servers=failed_servers,
                healthy_selection=tuple(),
                is_coordinate=is_coordinate,
                state_limit=max(1, min(len(failed_servers), state_budget)),
            )
        )
        if allow_healthy_moves:
            self._add_logic_benders_selection_states(
                all_states,
                analysis,
                deadline,
                protection=protection,
                failed_tenant=failed_tenant,
                failed_rank=int(failed_rank),
                movable=movable,
                movable_by_tenant=movable_by_tenant,
                ordered_tenants=ordered_tenants,
                is_coordinate=is_coordinate,
                state_budget=state_budget,
            )

        yielded = 0
        yielded_signatures = set()
        scored_outputs = []
        for obj, mapping, moves, _used_servers, _moved_ranks in self._slot_assignment_output_order(
            all_states
        ):
            if time.time() >= deadline:
                break
            sig = self._signature(mapping)
            if sig in yielded_signatures:
                continue
            yielded_signatures.add(sig)
            source_moves = "+".join(
                f"t{tenant}:r{rank}:s{server}" for tenant, rank, server in moves
            )
            scored_outputs.append((
                self._entry_sort_key((obj, mapping, "slot_assign")),
                copy_mapping(mapping),
                f"{source_prefix}:prefailure:{source_moves}",
            ))
            if len(scored_outputs) >= output_budget:
                break

        for _key, mapping, source in sorted(scored_outputs, key=lambda item: item[0]):
            yield mapping, source
            yielded += 1
            if yielded >= output_budget:
                return

    def _add_logic_benders_selection_states(
        self,
        all_states: list,
        analysis,
        deadline: float,
        *,
        protection: tuple[int, ...],
        failed_tenant: int,
        failed_rank: int,
        movable: list[tuple[int, int, float]],
        movable_by_tenant: dict[int, list[tuple[int, int, float]]],
        ordered_tenants: list[int],
        is_coordinate: bool,
        state_budget: int,
    ) -> None:
        """Benders-inspired layer: choose ranks first, assign slots second.

        The master side emits small, high-impact rank-selection sets.  For each
        selected set, the logic subproblem solves a bounded unique assignment to
        protection slots.  This keeps the repair semantics explicit while still
        using the time-expanded estimator and simulator selection downstream.
        """

        if time.time() >= deadline:
            return
        max_optional_moves = min(len(protection) - 1, len(movable))
        if max_optional_moves <= 0:
            return

        optional_order = self._logic_benders_optional_rank_order(
            movable_by_tenant,
            ordered_tenants,
            failed_tenant=failed_tenant,
            is_coordinate=is_coordinate,
        )
        if not optional_order:
            return

        base_template = copy_mapping(self.scenario.pre_failure_mapping)
        seen = {self._signature(state[1]) for state in all_states}
        failed_servers = [
            int(server)
            for server in protection
            if int(server) in self.candidate_sets[int(failed_tenant)]
        ]
        if not failed_servers:
            return

        cuts: list[dict[str, object]] = []
        added = 0
        evaluated_selections: set[tuple[tuple[int, int], ...]] = set()
        while len(all_states) < state_budget and added < state_budget:
            if time.time() >= deadline:
                return
            master_sets = self._logic_benders_master_selections(
                optional_order,
                movable_by_tenant,
                ordered_tenants,
                failed_tenant=failed_tenant,
                max_optional_moves=max_optional_moves,
                is_coordinate=is_coordinate,
                cap=1,
                cuts=cuts,
            )
            if not master_sets:
                return
            selection = tuple(sorted(master_sets[0]))
            if selection in evaluated_selections:
                cuts.append({
                    "type": "exact_nogood_selection",
                    "selection": selection,
                    "reason": "master_repeated_evaluated_selection",
                })
                self.logic_benders_cuts_generated += 1
                continue

            evaluated_selections.add(selection)
            self.logic_benders_master_selections += 1
            states = self._logic_benders_assignment_subproblem(
                base_template,
                analysis,
                deadline,
                protection=protection,
                failed_tenant=int(failed_tenant),
                failed_rank=int(failed_rank),
                failed_servers=failed_servers,
                healthy_selection=selection,
                is_coordinate=is_coordinate,
                state_limit=self._logic_benders_subproblem_limit(
                    state_budget=state_budget,
                    added=added,
                    is_coordinate=is_coordinate,
                ),
            )
            self.logic_benders_subproblem_states += len(states)

            feedback_cuts = self._logic_benders_structural_cuts(
                selection,
                states,
                failed_tenant=int(failed_tenant),
                is_coordinate=is_coordinate,
            )
            feedback_cuts.append({
                "type": "exact_nogood_selection",
                "selection": selection,
                "reason": "evaluated_master_selection",
            })
            if any(cut.get("type") == "require_one_hotspot_rank" for cut in feedback_cuts):
                cuts = [
                    cut
                    for cut in cuts
                    if cut.get("type") != "require_one_hotspot_rank"
                ]
            cuts.extend(feedback_cuts)
            self.logic_benders_cuts_generated += len(feedback_cuts)

            for state in states:
                sig = self._signature(state[1])
                if sig in seen:
                    continue
                seen.add(sig)
                all_states.append(state)
                added += 1
                self.logic_benders_selection_candidates += 1
                if len(all_states) >= state_budget or added >= state_budget:
                    all_states[:] = sorted(
                        all_states,
                        key=lambda item: self._entry_sort_key((item[0], item[1], "logic_benders")),
                    )[:state_budget]
                    return
        if len(all_states) >= state_budget:
            all_states[:] = sorted(
                all_states,
                key=lambda item: self._entry_sort_key((item[0], item[1], "logic_benders")),
            )[:state_budget]

    def _logic_benders_structural_cuts(
        self,
        selection: tuple[tuple[int, int], ...],
        states: list,
        *,
        failed_tenant: int,
        is_coordinate: bool,
    ) -> list[dict[str, object]]:
        selected = tuple(sorted((int(tenant), int(rank)) for tenant, rank in selection))
        cuts: list[dict[str, object]] = []
        if not states:
            cuts.append({"type": "exact_nogood_selection", "selection": selected, "reason": "assignment_subproblem_infeasible"})
            return cuts

        ranked = sorted(
            states,
            key=lambda state: self._entry_sort_key((state[0], state[1], "logic_benders_cut")),
        )
        best_state = ranked[0]
        switches = self.evaluator.switch_counts(best_state[1])
        if not switches.extra_moved_ranks_vs_failover and selected:
            cuts.append({"type": "exact_nogood_selection", "selection": selected, "reason": "selection_collapses_to_failover"})

        if is_coordinate and selected:
            by_tenant: dict[int, int] = {}
            for tenant, _rank in selected:
                by_tenant[int(tenant)] = by_tenant.get(int(tenant), 0) + 1
            heavy_tenants = tuple(
                sorted(
                    tenant
                    for tenant, count in by_tenant.items()
                    if tenant != int(failed_tenant) and count >= 2
                )
            )
            if heavy_tenants:
                cuts.append({
                    "type": "limit_tenant_pressure",
                    "tenants": heavy_tenants,
                    "max_per_tenant": 1,
                    "reason": "multiple_nonfailed_tenant_ranks_selected_without_cross_tenant_diversity",
                })

        cuts.extend(
            self._contention_guided_feedback_cuts(
                selected,
                best_state[1],
                failed_tenant=int(failed_tenant),
                is_coordinate=is_coordinate,
            )
        )

        protection = set(int(server) for server in self.scenario.global_protection_pool)
        leaf_counts: dict[int, int] = {}
        for ranks in best_state[1].values():
            for server in ranks.values():
                server = int(server)
                if server not in protection:
                    continue
                leaf = self._server_leaf_sort_key(server)[0]
                leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1
        crowded_leaves = tuple(sorted(leaf for leaf, count in leaf_counts.items() if count >= 3))
        if crowded_leaves:
            cuts.append({
                "type": "limit_protection_leaf_crowding",
                "leaves": crowded_leaves,
                "max_per_leaf": 2,
                "reason": "assignment_subproblem_concentrated_on_few_protection_leaves",
            })
        return cuts

    def _contention_guided_feedback_cuts(
        self,
        selection: tuple[tuple[int, int], ...],
        mapping: Mapping,
        *,
        failed_tenant: int,
        is_coordinate: bool,
    ) -> list[dict[str, object]]:
        """Convert estimator hotspot structure into master-side logic guidance."""

        analysis = self._safe_analysis(mapping)
        selected = set((int(tenant), int(rank)) for tenant, rank in selection)
        failed_rank = infer_failed_rank(self.scenario.pre_failure_mapping, self.scenario.failure)
        fixed = {(int(failed_tenant), int(failed_rank)), *selected}

        hotspot_rank_scores: dict[tuple[int, int], float] = {}
        hotspot_tasks: list[tuple[int, int]] = []
        hotspot_resources = []
        hotspot_servers: set[int] = set()
        culprit_servers: set[int] = set()

        clusters = list(getattr(analysis, "contention_clusters", []) or [])
        for cluster in clusters[:8]:
            excess = float(cluster.get("excess", 0.0) or 0.0)
            resource = cluster.get("resource")
            if resource is not None:
                hotspot_resources.append(resource)
                hotspot_servers.update(self._servers_from_hotspot_resource(resource))
            for task_key in cluster.get("tasks", ()) or ():
                normalized_task = self._normalize_task_key(task_key)
                if normalized_task is not None:
                    hotspot_tasks.append(normalized_task)
            for rank_key in cluster.get("ranks", ()) or ():
                key = self._normalize_rank_key(rank_key)
                if key is None:
                    continue
                hotspot_rank_scores[key] = hotspot_rank_scores.get(key, 0.0) + max(excess, 1.0)

        if not hotspot_resources:
            hotspot_resources = self._hotspot_resources_from_slot_prices(analysis, limit=12)

        rank_pressure = getattr(analysis, "rank_pressure", {}) or {}
        for key, pressure in sorted(
            rank_pressure.items(),
            key=lambda item: (-float(item[1]), self._normalize_rank_key(item[0]) or (10**9, 10**9)),
        )[:12]:
            normalized = self._normalize_rank_key(key)
            if normalized is None:
                continue
            hotspot_rank_scores[normalized] = hotspot_rank_scores.get(normalized, 0.0) + float(pressure)

        critical_tasks = [
            task
            for task in (self._normalize_task_key(task) for task in getattr(analysis, "critical_tasks", set()))
            if task is not None
        ]
        task_pressure = getattr(analysis, "task_pressure", {}) or {}
        if not critical_tasks and task_pressure:
            critical_tasks = [
                task
                for task in (
                    self._normalize_task_key(key)
                    for key, _value in sorted(
                        task_pressure.items(),
                        key=lambda item: (-float(item[1]), self._normalize_task_key(item[0]) or (10**9, 10**9)),
                    )[:6]
                )
                if task is not None
            ]
        if not critical_tasks:
            critical_tasks = self._tasks_touching_ranks(
                mapping,
                [
                    key
                    for key, _score in sorted(
                        hotspot_rank_scores.items(),
                        key=lambda item: (-float(item[1]), item[0]),
                    )[:8]
                ],
                limit=12,
            )

        for tenant, rank in hotspot_rank_scores:
            if int(tenant) not in mapping or int(rank) not in mapping[int(tenant)]:
                continue
            server = int(mapping[int(tenant)][int(rank)])
            culprit_servers.add(server)
            hotspot_servers.add(server)

        eligible_ranks = []
        for key, score in sorted(
            hotspot_rank_scores.items(),
            key=lambda item: (-float(item[1]), item[0]),
        ):
            tenant, rank = key
            if key in fixed:
                continue
            if not is_coordinate and int(tenant) != int(failed_tenant):
                continue
            if int(tenant) not in self.scenario.pre_failure_mapping:
                continue
            if int(rank) not in self.scenario.pre_failure_mapping[int(tenant)]:
                continue
            original_server = int(self.scenario.pre_failure_mapping[int(tenant)][int(rank)])
            if original_server in set(int(server) for server in self.scenario.global_protection_pool):
                continue
            eligible_ranks.append((key, float(score)))
            if len(eligible_ranks) >= 8:
                break

        cuts: list[dict[str, object]] = []
        if eligible_ranks:
            max_score = max(float(score) for _key, score in eligible_ranks) or 1.0
            ranks = tuple(key for key, _score in eligible_ranks)
            weights = {
                f"{int(tenant)}:{int(rank)}": float(score) / float(max_score)
                for (tenant, rank), score in eligible_ranks
            }
            cuts.append({
                "type": "prioritize_hotspot_ranks",
                "ranks": ranks,
                "weights": weights,
                "reason": "subproblem_estimator_hotspot_culprit_ranks",
            })
            self.contention_feedback_cuts_generated += 1
            if len(eligible_ranks) >= 2:
                cuts.append({
                    "type": "require_one_hotspot_rank",
                    "ranks": ranks[: min(4, len(ranks))],
                    "reason": "avoid_repeating_candidate_sets_that_leave_top_hotspots_unaddressed",
                })
                self.contention_feedback_cuts_generated += 1

        if cuts:
            unique_hotspot_links = []
            seen_hotspot_links = set()
            for resource in hotspot_resources:
                if not self._is_hotspot_link_resource(resource):
                    continue
                jsonable = self._jsonable_resource(resource)
                key = repr(jsonable)
                if key in seen_hotspot_links:
                    continue
                seen_hotspot_links.add(key)
                unique_hotspot_links.append(jsonable)
            self.contention_feedback_records.append({
                "mode": "coordinate" if is_coordinate else "local",
                "selected_candidate_set": [
                    {"tenant": int(tenant), "rank": int(rank)}
                    for tenant, rank in sorted(selected)
                ],
                "hotspot_links": unique_hotspot_links[:8],
                "hotspot_servers": sorted(int(server) for server in hotspot_servers)[:12],
                "critical_path_tasks": [
                    {"tenant": int(tenant), "task": int(task)}
                    for tenant, task in sorted(set(critical_tasks or hotspot_tasks))[:12]
                ],
                "culprit_servers": sorted(int(server) for server in culprit_servers)[:12],
                "culprit_ranks": [
                    {"tenant": int(tenant), "rank": int(rank), "weight": float(weight)}
                    for (tenant, rank), weight in eligible_ranks[:8]
                ],
            })
        return cuts

    def _full_contention_analysis(self, mapping: Mapping):
        try:
            return self.evaluator.estimator.analyze(mapping)
        except Exception:
            return self._safe_analysis(mapping)

    def _hotspot_resources_from_slot_prices(self, analysis, *, limit: int) -> list[object]:
        resource_scores: dict[str, tuple[float, object]] = {}
        for price_state in getattr(analysis, "slot_prices", []) or []:
            for resource_type in ("edge", "sender", "receiver"):
                prices = price_state.get(resource_type, {}) if isinstance(price_state, dict) else {}
                for resource_id, price in prices.items():
                    resource = (str(resource_type), resource_id)
                    key = repr(self._jsonable_resource(resource))
                    old_score, _old_resource = resource_scores.get(key, (0.0, resource))
                    resource_scores[key] = (float(old_score) + max(0.0, float(price)), resource)
        return [
            resource
            for _score, resource in sorted(
                resource_scores.values(),
                key=lambda item: (-float(item[0]), repr(self._jsonable_resource(item[1]))),
            )[: max(1, int(limit))]
        ]

    def _tasks_touching_ranks(
        self,
        mapping: Mapping,
        ranks: list[tuple[int, int]],
        *,
        limit: int,
    ) -> list[tuple[int, int]]:
        rank_set = set((int(tenant), int(rank)) for tenant, rank in ranks)
        tasks = []
        seen = set()
        for tenant, rank in ranks:
            if int(tenant) not in mapping or int(rank) not in mapping[int(tenant)]:
                continue
            for task_id, src_rank, dst_rank, volume in self._task_infos(int(tenant)):
                if (int(tenant), int(src_rank)) not in rank_set and (int(tenant), int(dst_rank)) not in rank_set:
                    continue
                key = (int(tenant), int(task_id))
                if key in seen:
                    continue
                seen.add(key)
                tasks.append((float(volume), key))
        tasks.sort(key=lambda item: (-float(item[0]), item[1]))
        return [key for _volume, key in tasks[: max(1, int(limit))]]

    @staticmethod
    def _normalize_rank_key(value) -> tuple[int, int] | None:
        try:
            tenant, rank = value
            return (int(tenant), int(rank))
        except Exception:
            return None

    @staticmethod
    def _normalize_task_key(value) -> tuple[int, int] | None:
        try:
            tenant, task = value
            return (int(tenant), int(task))
        except Exception:
            return None

    @staticmethod
    def _is_hotspot_link_resource(resource) -> bool:
        try:
            resource_type, _resource_id = resource
            return str(resource_type) == "edge"
        except Exception:
            return False

    @staticmethod
    def _servers_from_hotspot_resource(resource) -> set[int]:
        try:
            resource_type, resource_id = resource
        except Exception:
            return set()
        if str(resource_type) in {"sender", "receiver"}:
            try:
                return {int(resource_id)}
            except Exception:
                return set()
        return set()

    def _jsonable_resource(self, resource):
        try:
            resource_type, resource_id = resource
        except Exception:
            return str(resource)
        return {
            "type": str(resource_type),
            "id": self._jsonable_resource_id(resource_id),
        }

    def _jsonable_resource_id(self, value):
        if isinstance(value, tuple):
            return [self._jsonable_resource_id(item) for item in value]
        if isinstance(value, list):
            return [self._jsonable_resource_id(item) for item in value]
        try:
            return int(value)
        except Exception:
            return str(value)

    def _logic_benders_master_cap(self, *, state_budget: int, is_coordinate: bool) -> int:
        return max(
            4 if is_coordinate else 3,
            min(
                int(state_budget),
                24 if is_coordinate else 10,
            ),
        )

    def _logic_benders_subproblem_limit(
        self,
        *,
        state_budget: int,
        added: int,
        is_coordinate: bool,
    ) -> int:
        del state_budget, added, is_coordinate
        return 1

    def _logic_benders_cut_rejects(
        self,
        selection: tuple[tuple[int, int], ...],
        cuts: list[dict[str, object]],
    ) -> bool:
        if not cuts:
            return False
        selected = tuple(sorted((int(tenant), int(rank)) for tenant, rank in selection))
        for cut in cuts:
            cut_type = cut.get("type")
            if cut_type == "exact_nogood_selection":
                forbidden = tuple(sorted(tuple(item) for item in cut.get("selection", ())))
                if forbidden and forbidden == selected:
                    self.logic_benders_cuts_applied += 1
                    return True
            elif cut_type == "limit_tenant_pressure":
                max_per_tenant = int(cut.get("max_per_tenant", 1))
                tenants = set(int(tenant) for tenant in cut.get("tenants", ()))
                counts: dict[int, int] = {}
                for tenant, _rank in selected:
                    if int(tenant) not in tenants:
                        continue
                    counts[int(tenant)] = counts.get(int(tenant), 0) + 1
                if any(count > max_per_tenant for count in counts.values()):
                    self.logic_benders_cuts_applied += 1
                    return True
            elif cut_type == "limit_protection_leaf_crowding":
                # This cut is assignment-side.  The master cannot know final
                # leaves, so it is recorded for metadata and future assignment
                # ranking, but it does not reject rank selections directly.
                continue
        return False

    def _logic_benders_assignment_subproblem(
        self,
        base_template: Mapping,
        analysis,
        deadline: float,
        *,
        protection: tuple[int, ...],
        failed_tenant: int,
        failed_rank: int,
        failed_servers: list[int],
        healthy_selection: tuple[tuple[int, int], ...],
        is_coordinate: bool,
        state_limit: int,
    ) -> list:
        selected_healthy = tuple((int(tenant), int(rank)) for tenant, rank in healthy_selection)
        if len(selected_healthy) + 1 > len(protection):
            return []
        rank_domains = (
            (int(failed_tenant), int(failed_rank), tuple(int(server) for server in failed_servers)),
            *(
                (
                    int(tenant),
                    int(rank),
                    tuple(
                        int(server)
                        for server in protection
                        if int(server) in self.candidate_sets[int(tenant)]
                    ),
                )
                for tenant, rank in selected_healthy
            ),
        )
        if any(not servers for _tenant, _rank, servers in rank_domains):
            return []
        states = self._solve_exact_assignment_milp(
            base_template,
            analysis,
            deadline,
            rank_domains=rank_domains,
            protection=protection,
            state_limit=max(1, int(state_limit)),
        )
        return sorted(
            states,
            key=lambda state: self._entry_sort_key((state[0], state[1], "logic_benders")),
        )[: max(1, int(state_limit))]

    def _solve_exact_assignment_milp(
        self,
        base_template: Mapping,
        analysis,
        deadline: float,
        *,
        rank_domains: tuple[tuple[int, int, tuple[int, ...]], ...],
        protection: tuple[int, ...],
        state_limit: int,
    ) -> list:
        """Solve the fixed-rank protection-slot assignment as a small MILP."""

        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return []

        if time.time() >= deadline:
            return []

        ranks = tuple((int(tenant), int(rank)) for tenant, rank, _servers in rank_domains)
        slots = tuple(int(server) for server in protection)
        slot_set = set(slots)
        costs: dict[tuple[int, int, int], float] = {}
        task_pair_prices_by_tenant: dict[int, object] = {}
        for tenant, rank, servers in rank_domains:
            tenant = int(tenant)
            rank = int(rank)
            task_pair_prices = task_pair_prices_by_tenant.setdefault(
                tenant,
                self._task_pair_prices(base_template, tenant, analysis),
            )
            for server in servers:
                server = int(server)
                if server not in slot_set:
                    continue
                candidate = copy_mapping(base_template)
                candidate[tenant][rank] = server
                price_cost = self._tenant_price_cost(
                    candidate,
                    tenant,
                    candidate[tenant],
                    task_pair_prices,
                    analysis,
                )
                original_server = int(self.scenario.pre_failure_mapping[tenant][rank])
                leaf_distance = abs(
                    self._server_leaf_sort_key(server)[0]
                    - self._server_leaf_sort_key(original_server)[0]
                )
                costs[(tenant, rank, server)] = (
                    float(price_cost)
                    + 1.0e-6 * float(leaf_distance)
                    + 1.0e-9 * float(server)
                )

        if any(
            not any((int(tenant), int(rank), int(server)) in costs for server in slots)
            for tenant, rank in ranks
        ):
            return []

        model = gp.Model("repair_assignment_subproblem")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        x = {
            key: model.addVar(vtype=GRB.BINARY, obj=float(cost), name=f"x_{key[0]}_{key[1]}_{key[2]}")
            for key, cost in costs.items()
        }
        for tenant, rank in ranks:
            model.addConstr(
                gp.quicksum(x[(int(tenant), int(rank), int(server))] for server in slots if (int(tenant), int(rank), int(server)) in x)
                == 1,
                name=f"assign_{tenant}_{rank}",
            )
        for server in slots:
            model.addConstr(
                gp.quicksum(x[(tenant, rank, int(server))] for tenant, rank in ranks if (tenant, rank, int(server)) in x)
                <= 1,
                name=f"slot_{server}",
            )
        model.ModelSense = GRB.MINIMIZE

        states = []
        for _iteration in range(max(1, int(state_limit))):
            if time.time() >= deadline:
                break
            model.optimize()
            if model.Status != GRB.OPTIMAL:
                break
            selected_keys = tuple(sorted(key for key, var in x.items() if float(var.X) > 0.5))
            if len(selected_keys) != len(ranks):
                break

            mapping = copy_mapping(base_template)
            moves = []
            used = set()
            moved = set()
            for tenant, rank, server in selected_keys:
                mapping[int(tenant)][int(rank)] = int(server)
                moves.append((int(tenant), int(rank), int(server)))
                used.add(int(server))
                moved.add((int(tenant), int(rank)))
            try:
                check_repair_feasible(self.scenario, mapping, self.failover_mapping)
            except ValueError:
                model.addConstr(
                    gp.quicksum(x[key] for key in selected_keys) <= len(selected_keys) - 1,
                    name=f"nogood_infeasible_{len(states)}",
                )
                continue
            states.append((
                self.evaluator.estimate(mapping),
                mapping,
                tuple(sorted(moves)),
                frozenset(used),
                frozenset(moved),
            ))

            model.addConstr(
                gp.quicksum(x[key] for key in selected_keys) <= len(selected_keys) - 1,
                name=f"nogood_solution_{len(states)}",
            )

        return states

    def _logic_benders_optional_rank_order(
        self,
        movable_by_tenant: dict[int, list[tuple[int, int, float]]],
        ordered_tenants: list[int],
        *,
        failed_tenant: int,
        is_coordinate: bool,
    ) -> list[tuple[int, int, float]]:
        if not is_coordinate:
            return list(movable_by_tenant.get(int(failed_tenant), ()))

        rows: list[tuple[int, int, float]] = []
        max_depth = max((len(movable_by_tenant.get(int(tenant), ())) for tenant in ordered_tenants), default=0)
        for depth in range(max_depth):
            for tenant in ordered_tenants:
                tenant_rows = movable_by_tenant.get(int(tenant), [])
                if depth < len(tenant_rows):
                    rows.append(tenant_rows[depth])
        return rows

    def _logic_benders_master_selections(
        self,
        optional_order: list[tuple[int, int, float]],
        movable_by_tenant: dict[int, list[tuple[int, int, float]]],
        ordered_tenants: list[int],
        *,
        failed_tenant: int,
        max_optional_moves: int,
        is_coordinate: bool,
        cap: int,
        cuts: list[dict[str, object]] | None = None,
    ) -> list[tuple[tuple[int, int], ...]]:
        milp_selected = self._logic_benders_master_milp_selections(
            optional_order,
            movable_by_tenant,
            ordered_tenants,
            failed_tenant=failed_tenant,
            max_optional_moves=max_optional_moves,
            is_coordinate=is_coordinate,
            cap=cap,
            cuts=cuts or [],
        )
        if milp_selected:
            return milp_selected

        selected: list[tuple[tuple[int, int], ...]] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        cuts = cuts or []

        def add(keys) -> None:
            normalized = []
            used = set()
            for tenant, rank in keys:
                key = (int(tenant), int(rank))
                if key in used:
                    continue
                used.add(key)
                normalized.append(key)
                if len(normalized) >= int(max_optional_moves):
                    break
            if not normalized:
                return
            item = tuple(normalized)
            if item in seen:
                return
            if self._logic_benders_cut_rejects(item, cuts):
                return
            seen.add(item)
            selected.append(item)

        for tenant, rank, _impact in optional_order[: max(4, int(cap))]:
            add(((tenant, rank),))
            if len(selected) >= int(cap):
                return selected

        for depth in range(2, max_optional_moves + 1):
            add((tenant_rank[:2] for tenant_rank in optional_order[:depth]))
            if len(selected) >= int(cap):
                return selected

        if is_coordinate:
            cover_tenants = [
                int(tenant)
                for tenant in ordered_tenants
                if movable_by_tenant.get(int(tenant))
            ]
            max_depth = max((len(movable_by_tenant.get(int(tenant), ())) for tenant in cover_tenants), default=0)
            for depth in range(max_depth):
                add(
                    (
                        (
                            int(movable_by_tenant[int(tenant)][min(depth, len(movable_by_tenant[int(tenant)]) - 1)][0]),
                            int(movable_by_tenant[int(tenant)][min(depth, len(movable_by_tenant[int(tenant)]) - 1)][1]),
                        )
                        for tenant in cover_tenants
                    )
                )
                if len(selected) >= int(cap):
                    return selected

        top_pool = [(int(tenant), int(rank)) for tenant, rank, _impact in optional_order[: min(10, len(optional_order))]]
        for width in (2, 3):
            if width > max_optional_moves:
                continue
            for combo in itertools.combinations(top_pool, width):
                add(combo)
                if len(selected) >= int(cap):
                    return selected
        return selected

    def _logic_benders_master_milp_selections(
        self,
        optional_order: list[tuple[int, int, float]],
        movable_by_tenant: dict[int, list[tuple[int, int, float]]],
        ordered_tenants: list[int],
        *,
        failed_tenant: int,
        max_optional_moves: int,
        is_coordinate: bool,
        cap: int,
        cuts: list[dict[str, object]],
    ) -> list[tuple[tuple[int, int], ...]]:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return []

        pool = []
        seen_keys = set()
        for tenant, rank, impact in optional_order:
            tenant = int(tenant)
            rank = int(rank)
            key = (tenant, rank)
            if key in seen_keys:
                continue
            if not is_coordinate and tenant != int(failed_tenant):
                continue
            if tenant not in movable_by_tenant:
                continue
            seen_keys.add(key)
            pool.append((tenant, rank, float(impact)))
        if not pool or int(max_optional_moves) <= 0:
            return []

        impact_scale = max(1.0, max(abs(float(impact)) for _tenant, _rank, impact in pool))
        pool_key_set = {(int(tenant), int(rank)) for tenant, rank, _impact in pool}
        priority_by_key: dict[tuple[int, int], float] = {}
        required_rank_groups: list[tuple[tuple[int, int], ...]] = []
        for cut in cuts:
            cut_type = cut.get("type")
            if cut_type == "prioritize_hotspot_ranks":
                matched = 0
                weights = cut.get("weights", {}) or {}
                for item_index, item in enumerate(cut.get("ranks", ()) or ()):
                    key = self._normalize_rank_key(item)
                    if key is None or key not in pool_key_set:
                        continue
                    weight_key = f"{int(key[0])}:{int(key[1])}"
                    weight = float(weights.get(weight_key, 1.0 / float(item_index + 1)))
                    priority_by_key[key] = priority_by_key.get(key, 0.0) + max(weight, 0.0)
                    matched += 1
                if matched:
                    self.contention_feedback_cuts_applied += 1
            elif cut_type == "require_one_hotspot_rank":
                keys = tuple(
                    key
                    for key in (self._normalize_rank_key(item) for item in cut.get("ranks", ()) or ())
                    if key is not None and key in pool_key_set
                )
                if keys:
                    required_rank_groups.append(tuple(sorted(set(keys))))
                    self.contention_feedback_cuts_applied += 1

        model = gp.Model("repair_master_rank_selection")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        y = {}
        for index, (tenant, rank, impact) in enumerate(pool):
            key = (int(tenant), int(rank))
            score = (
                float(impact) / impact_scale
                + 1.5 * float(priority_by_key.get(key, 0.0))
                - 1.0e-5 * index
                - (5.0e-4 if is_coordinate and int(tenant) == int(failed_tenant) else 0.0)
            )
            y[key] = model.addVar(
                vtype=GRB.BINARY,
                obj=float(score),
                name=f"y_{tenant}_{rank}",
            )

        model.addConstr(
            gp.quicksum(y.values()) <= min(int(max_optional_moves), len(pool)),
            name="capacity",
        )

        if not is_coordinate:
            for tenant, rank, _impact in pool:
                if int(tenant) == int(failed_tenant):
                    continue
                model.addConstr(y[(int(tenant), int(rank))] == 0, name=f"scope_{tenant}_{rank}")
        else:
            tenant_to_keys: dict[int, list[tuple[int, int]]] = {}
            for tenant, rank, _impact in pool:
                tenant_to_keys.setdefault(int(tenant), []).append((int(tenant), int(rank)))
            for tenant in ordered_tenants:
                keys = tenant_to_keys.get(int(tenant), [])
                if not keys:
                    continue
                model.addConstr(
                    gp.quicksum(y[key] for key in keys) <= min(len(keys), max(1, int(max_optional_moves))),
                    name=f"tenant_cap_{tenant}",
                )

        available_keys = set(y)
        exact_cut_key_sets: list[tuple[tuple[int, int], ...]] = []
        for cut in cuts:
            cut_type = cut.get("type")
            if cut_type == "exact_nogood_selection":
                keys = [
                    key
                    for key in (tuple(int(value) for value in item) for item in cut.get("selection", ()))
                    if key in available_keys
                ]
                if keys:
                    exact_cut_key_sets.append(tuple(sorted(keys)))
            elif cut_type == "limit_tenant_pressure":
                max_per_tenant = int(cut.get("max_per_tenant", 1))
                tenants = set(int(tenant) for tenant in cut.get("tenants", ()))
                for tenant in tenants:
                    keys = [key for key in available_keys if int(key[0]) == int(tenant)]
                    if not keys:
                        continue
                    model.addConstr(
                        gp.quicksum(y[key] for key in keys) <= float(max_per_tenant),
                        name=f"cut_tenant_pressure_{tenant}",
                    )
            elif cut_type == "prioritize_hotspot_ranks":
                continue
            elif cut_type == "require_one_hotspot_rank":
                continue

        for group_index, keys in enumerate(required_rank_groups):
            model.addConstr(
                gp.quicksum(y[key] for key in keys if key in y) >= 1,
                name=f"cut_require_hotspot_rank_{group_index}",
            )

        model.ModelSense = GRB.MAXIMIZE

        candidates: list[tuple[float, tuple[tuple[int, int], ...]]] = []
        seen_selection = set()
        max_cardinality = min(int(max_optional_moves), len(pool))
        for cardinality in range(1, max_cardinality + 1):
            cardinality_constraint = model.addConstr(
                gp.quicksum(y.values()) == int(cardinality),
                name=f"candidate_set_size_{cardinality}",
            )
            exact_constraints = []
            for keys in exact_cut_key_sets:
                if len(keys) != int(cardinality):
                    continue
                exact_constraints.append(
                    model.addConstr(
                        gp.quicksum(y[key] for key in keys) <= len(keys) - 1,
                        name=f"cut_exact_k{cardinality}_{len(exact_constraints)}",
                    )
                )
            per_size_seen: set[tuple[tuple[int, int], ...]] = set()
            generated_nogoods = []
            while len(per_size_seen) < max(1, int(cap)):
                model.optimize()
                if model.Status != GRB.OPTIMAL:
                    break
                chosen = tuple(sorted(key for key, var in y.items() if float(var.X) > 0.5))
                if len(chosen) != int(cardinality) or chosen in per_size_seen:
                    break
                per_size_seen.add(chosen)
                selection = tuple((int(tenant), int(rank)) for tenant, rank in chosen)
                if selection not in seen_selection and not self._logic_benders_cut_rejects(selection, cuts):
                    seen_selection.add(selection)
                    candidates.append((float(model.ObjVal), selection))
                generated_nogoods.append(
                    model.addConstr(
                        gp.quicksum(y[key] for key in chosen) <= len(chosen) - 1,
                        name=f"nogood_master_k{cardinality}_{len(per_size_seen)}",
                    )
                )
                if len(per_size_seen) >= max(1, int(cap)):
                    break
            model.remove([cardinality_constraint, *exact_constraints, *generated_nogoods])
            model.update()

        candidates.sort(
            key=lambda item: (
                -float(item[0]),
                len(item[1]),
                item[1],
            )
        )
        return [selection for _score, selection in candidates[: max(1, int(cap))]]

    def _add_slot_recombination_states(
        self,
        all_states: list,
        analysis,
        deadline: float,
        *,
        protection: tuple[int, ...],
        failed_tenant: int,
        failed_rank: int,
        movable: list[tuple[int, int, float]],
        movable_by_tenant: dict[int, list[tuple[int, int, float]]],
        ordered_tenants: list[int],
        is_coordinate: bool,
        state_budget: int,
    ) -> None:
        if time.time() >= deadline:
            return
        base_template = copy_mapping(self.scenario.pre_failure_mapping)
        seen = {self._signature(state[1]) for state in all_states}

        failed_servers = [
            int(server)
            for server in protection
            if int(server) in self.candidate_sets[int(failed_tenant)]
        ]
        if not failed_servers:
            return

        per_rank_servers = len(protection) if is_coordinate else max(2, min(len(protection), 4))
        rank_rows = []
        for tenant, rank, impact in movable:
            tenant = int(tenant)
            rank = int(rank)
            candidate_servers = [
                int(server)
                for server in protection
                if int(server) in self.candidate_sets[int(tenant)]
            ]
            if not candidate_servers:
                continue
            task_pair_prices = self._task_pair_prices(base_template, tenant, analysis)
            ranked_servers = self._ranked_available_servers(
                base_template,
                tenant,
                [rank],
                candidate_servers,
                task_pair_prices,
                analysis,
            )[:per_rank_servers]
            rank_rows.append((tenant, rank, float(impact), ranked_servers))

        # Favor cross-tenant coverage first for coordinate, while preserving
        # every rank as optional so the search is not a local/failover extension.
        if is_coordinate:
            rank_rows.sort(key=lambda row: (row[0] == int(failed_tenant), -row[2], row[0], row[1]))
        else:
            rank_rows.sort(key=lambda row: (-row[2], row[0], row[1]))

        beam = []
        for server in failed_servers:
            mapping = copy_mapping(base_template)
            mapping[int(failed_tenant)][int(failed_rank)] = int(server)
            moves = ((int(failed_tenant), int(failed_rank), int(server)),)
            used = frozenset({int(server)})
            moved = frozenset({(int(failed_tenant), int(failed_rank))})
            try:
                check_repair_feasible(self.scenario, mapping, self.failover_mapping)
            except ValueError:
                continue
            beam.append((self.evaluator.estimate(mapping), mapping, moves, used, moved))

        width = max(8 if is_coordinate else 4, int(self.config.beam_width) * (3 if is_coordinate else 1))
        for tenant, rank, _impact, ranked_servers in rank_rows:
            if time.time() >= deadline or not beam:
                break
            expanded = list(beam)
            rank_key = (int(tenant), int(rank))
            for _obj, mapping, moves, used, moved in beam:
                if rank_key in moved:
                    continue
                for server in ranked_servers:
                    if time.time() >= deadline:
                        break
                    server = int(server)
                    if server in used:
                        continue
                    candidate = copy_mapping(mapping)
                    candidate[int(tenant)][int(rank)] = server
                    next_moves = (*moves, (int(tenant), int(rank), server))
                    next_used = frozenset((*used, server))
                    next_moved = frozenset((*moved, rank_key))
                    try:
                        check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                    except ValueError:
                        continue
                    expanded.append((
                        self.evaluator.estimate(candidate),
                        candidate,
                        next_moves,
                        next_used,
                        next_moved,
                    ))
                    if len(expanded) >= max(state_budget, width * len(protection)):
                        break
            beam = self._select_slot_assignment_beam(expanded, width=width)

            for state in beam:
                sig = self._signature(state[1])
                if sig in seen:
                    continue
                seen.add(sig)
                all_states.append(state)
            if len(all_states) >= state_budget:
                all_states[:] = sorted(
                    all_states,
                    key=lambda state: self._entry_sort_key((state[0], state[1], "slot_assign")),
                )[:state_budget]

        if not is_coordinate or time.time() >= deadline:
            return

        cover_tenants = [
            int(tenant)
            for tenant in ordered_tenants
            if movable_by_tenant.get(int(tenant))
        ]
        if not cover_tenants:
            return

        max_depth = max(len(movable_by_tenant.get(int(tenant), ())) for tenant in cover_tenants)
        for failed_server in failed_servers:
            if time.time() >= deadline:
                return
            for depth in range(max_depth):
                mapping = copy_mapping(base_template)
                mapping[int(failed_tenant)][int(failed_rank)] = int(failed_server)
                moves = [(int(failed_tenant), int(failed_rank), int(failed_server))]
                used = {int(failed_server)}
                moved = {(int(failed_tenant), int(failed_rank))}

                for tenant in cover_tenants:
                    rows = movable_by_tenant.get(int(tenant), [])
                    if not rows:
                        continue
                    row = rows[min(depth, len(rows) - 1)]
                    tenant = int(row[0])
                    rank = int(row[1])
                    rank_key = (tenant, rank)
                    if rank_key in moved:
                        continue
                    available = [
                        int(server)
                        for server in protection
                        if int(server) not in used
                        and int(server) in self.candidate_sets[int(tenant)]
                    ]
                    if not available:
                        continue
                    task_pair_prices = self._task_pair_prices(mapping, tenant, analysis)
                    ranked_servers = self._ranked_available_servers(
                        mapping,
                        tenant,
                        [rank],
                        available,
                        task_pair_prices,
                        analysis,
                    )
                    if not ranked_servers:
                        continue
                    server = int(ranked_servers[0])
                    mapping[tenant][rank] = server
                    moves.append((tenant, rank, server))
                    used.add(server)
                    moved.add(rank_key)

                try:
                    check_repair_feasible(self.scenario, mapping, self.failover_mapping)
                except ValueError:
                    continue
                sig = self._signature(mapping)
                if sig in seen:
                    continue
                seen.add(sig)
                all_states.append((
                    self.evaluator.estimate(mapping),
                    mapping,
                    tuple(moves),
                    frozenset(used),
                    frozenset(moved),
                ))

    def _select_slot_assignment_beam(self, states, *, width: int | None = None):
        if not states:
            return []
        width = max(1, int(width or max(4, int(self.config.beam_width) * 2)))
        best_by_signature = {}
        for state in states:
            obj, mapping, _moves, _used_servers, _moved_ranks = state
            sig = self._signature(mapping)
            old = best_by_signature.get(sig)
            if old is None or repair_better(obj, old[0], tol=self.config.score_sort_tolerance):
                best_by_signature[sig] = state
        ranked = sorted(
            best_by_signature.values(),
            key=lambda state: self._entry_sort_key((state[0], state[1], "slot_assign")),
        )
        selected = []
        seen = set()

        def add(state) -> None:
            sig = self._signature(state[1])
            if sig in seen:
                return
            seen.add(sig)
            selected.append(state)

        for state in ranked[:width]:
            add(state)
        by_shape = {}
        for state in ranked:
            switches = self.evaluator.switch_counts(state[1])
            moved_tenants = tuple(
                sorted({int(tenant) for tenant, _rank in switches.extra_moved_ranks_vs_failover})
            )
            key = (len(moved_tenants), int(switches.extra_vs_failover), moved_tenants)
            by_shape.setdefault(key, state)
        for key in sorted(by_shape, reverse=True):
            if len(selected) >= width + max(4, width // 2):
                break
            add(by_shape[key])
        return sorted(
            selected,
            key=lambda state: self._entry_sort_key((state[0], state[1], "slot_assign")),
        )

    def _slot_assignment_output_order(self, states):
        best_by_signature = {}
        for state in states:
            sig = self._signature(state[1])
            old = best_by_signature.get(sig)
            if old is None or repair_better(state[0], old[0], tol=self.config.score_sort_tolerance):
                best_by_signature[sig] = state
        ranked = sorted(
            best_by_signature.values(),
            key=lambda state: self._entry_sort_key((state[0], state[1], "slot_assign")),
        )
        ordered = []
        seen = set()

        def add(state) -> None:
            sig = self._signature(state[1])
            if sig in seen:
                return
            seen.add(sig)
            ordered.append(state)

        for state in ranked[: max(8, int(self.config.beam_width) * 2)]:
            add(state)
        by_extra = {}
        by_tenants = {}
        by_rank = {}
        for state in ranked:
            switches = self.evaluator.switch_counts(state[1])
            extra = int(switches.extra_vs_failover)
            moved_tenants = tuple(
                sorted({int(tenant) for tenant, _rank in switches.extra_moved_ranks_vs_failover})
            )
            by_extra.setdefault(extra, state)
            by_tenants.setdefault((len(moved_tenants), moved_tenants, extra), state)
            for tenant, rank, _server in state[2]:
                by_rank.setdefault((int(tenant), int(rank)), state)
        for key in sorted(by_extra):
            add(by_extra[key])
        for key in sorted(by_tenants, reverse=True):
            add(by_tenants[key])
        for key in sorted(by_rank):
            add(by_rank[key])
        for state in ranked:
            add(state)
        return ordered

    def _slot_assignment_proxy_score(
        self,
        mapping: Mapping,
        moves: tuple[tuple[int, int, int], ...],
        analysis,
    ) -> float:
        moved_by_tenant: dict[int, list[int]] = {}
        for tenant, rank, _server in moves:
            moved_by_tenant.setdefault(int(tenant), []).append(int(rank))

        score = 0.0
        for tenant, ranks in moved_by_tenant.items():
            score -= 0.01 * sum(
                self._rank_structural_impact(mapping, int(tenant), int(rank), analysis)
                for rank in ranks
            )
            score += sum(
                abs(
                    self._server_leaf_sort_key(int(mapping[int(tenant)][int(rank)]))[0]
                    - self._server_leaf_sort_key(
                        int(self.scenario.pre_failure_mapping[int(tenant)][int(rank)])
                    )[0]
                )
                for rank in ranks
            )
        return float(score)

    def _ranked_available_servers_proxy(
        self,
        mapping: Mapping,
        tenant: int,
        rank: int,
        available: list[int],
    ) -> list[int]:
        original_server = int(self.scenario.pre_failure_mapping[int(tenant)][int(rank)])
        original_leaf = self._server_leaf_sort_key(original_server)[0]
        neighbor_leaf_counts: dict[int, int] = {}
        tenant_mapping = mapping[int(tenant)]
        for _task_id, src_rank, dst_rank, _volume in self._task_infos(int(tenant)):
            if int(src_rank) == int(rank):
                other = int(dst_rank)
            elif int(dst_rank) == int(rank):
                other = int(src_rank)
            else:
                continue
            if other not in tenant_mapping:
                continue
            other_leaf = self._server_leaf_sort_key(int(tenant_mapping[other]))[0]
            neighbor_leaf_counts[other_leaf] = neighbor_leaf_counts.get(other_leaf, 0) + 1

        return sorted(
            (int(server) for server in available),
            key=lambda server: (
                -neighbor_leaf_counts.get(self._server_leaf_sort_key(int(server))[0], 0),
                abs(self._server_leaf_sort_key(int(server))[0] - original_leaf),
                abs(int(server) - original_server),
                int(server),
            ),
        )

    def _postprocess_best_mapping(
        self,
        best_obj,
        best_mapping: Mapping,
        best_source: str,
        deadline: float,
    ):
        return best_obj, best_mapping, best_source

    def _simulator_selection_entries(self, entries, best_entry):
        selected = []
        selected_signatures = set()

        def add(entry, *, mandatory: bool = False) -> None:
            sig = self._signature(entry[1])
            if sig in selected_signatures:
                return
            if not mandatory and len(selected) >= self._simulator_pool_cap():
                return
            selected_signatures.add(sig)
            selected.append(entry)

        for entry in entries:
            source = str(entry[2])
            if source == "failover":
                add(entry, mandatory=True)

        add(best_entry, mandatory=True)

        sorted_entries = sorted(entries, key=self._entry_sort_key)
        if self.name == "cooperative_repair":
            for entry in sorted_entries:
                if str(entry[2]).startswith("slot_assign_coordinate:local_scope"):
                    add(entry, mandatory=True)
                    break

        for entry in sorted_entries:
            if str(entry[2]).startswith("reference:"):
                add(entry)
                break

        top_k = 16 if self.name == "cooperative_repair" else 4
        for entry in sorted_entries[:top_k]:
            add(entry)

        cooperative_prefixes = (
            "global_assign:",
            "joint:",
            "direct_joint:",
            "reassign_protection:",
            "impact_guided:",
            "slot_assign_",
        )
        cooperative_added = 0
        cooperative_limit = 16 if self.name == "cooperative_repair" else 4
        for entry in sorted_entries:
            if str(entry[2]).startswith(cooperative_prefixes):
                add(entry)
                cooperative_added += 1
                if cooperative_added >= cooperative_limit:
                    break

        if self.name == "cooperative_repair":
            seen_diversity_keys = set()
            for entry in sorted_entries:
                diversity_key = self._simulator_pool_diversity_key(entry)
                if diversity_key in seen_diversity_keys:
                    continue
                seen_diversity_keys.add(diversity_key)
                add(entry)
                if len(selected) >= self._simulator_pool_cap():
                    break
            if len(sorted_entries) > 1:
                sample_count = min(8, len(sorted_entries))
                for sample_index in range(sample_count):
                    offset = round(
                        sample_index * (len(sorted_entries) - 1) / max(1, sample_count - 1)
                    )
                    add(sorted_entries[int(offset)])
                    if len(selected) >= self._simulator_pool_cap():
                        break
            seen_rank_keys = set()
            for entry in sorted_entries:
                for rank_key in self._simulator_pool_rank_keys(entry):
                    if rank_key in seen_rank_keys:
                        continue
                    seen_rank_keys.add(rank_key)
                    add(entry)
                    break
                if len(selected) >= self._simulator_pool_cap():
                    break

        for entry in sorted_entries:
            add(entry)
            if len(selected) >= self._simulator_pool_cap():
                break
        return selected

    def _simulator_pool_cap(self) -> int:
        if self.config.simulator_candidate_budget is not None:
            return max(1, int(self.config.simulator_candidate_budget))
        if self.name == "repair_failed_server_only":
            return max(1, len(self.scenario.global_protection_pool))
        if self.name == "cooperative_repair":
            return 48
        return 32

    def _simulator_pool_diversity_key(self, entry) -> tuple[object, ...]:
        obj, mapping, source = entry
        switches = count_switches(
            pre_failure_mapping=self.scenario.pre_failure_mapping,
            failover_mapping=self.failover_mapping,
            repaired_mapping=mapping,
            failure=self.scenario.failure,
        )
        moved_tenants = tuple(sorted(
            int(tenant)
            for tenant, _rank in switches.extra_moved_ranks_vs_failover
        ))
        source_kind = str(source).split(":", 1)[0]
        return (
            source_kind,
            moved_tenants,
            int(obj.extra_switches),
        )

    def _simulator_pool_rank_keys(self, entry) -> tuple[tuple[int, int], ...]:
        _obj, mapping, _source = entry
        switches = count_switches(
            pre_failure_mapping=self.scenario.pre_failure_mapping,
            failover_mapping=self.failover_mapping,
            repaired_mapping=mapping,
            failure=self.scenario.failure,
        )
        return tuple(
            sorted((int(tenant), int(rank)) for tenant, rank in switches.extra_moved_ranks_vs_failover)
        )

    def _select_simulator_best(self, entries, deadline: float):
        best_entry = None
        seen = set()
        scored = 0
        for _estimate_obj, mapping, source in entries:
            if time.time() >= deadline and best_entry is not None:
                break
            sig = self._signature(mapping)
            if sig in seen:
                continue
            seen.add(sig)
            try:
                check_repair_feasible(self.scenario, mapping, self.failover_mapping)
            except ValueError:
                continue
            makespan, avg_jct = self.evaluator.simulate(mapping)
            switches = self.evaluator.switch_counts(mapping)
            obj = type(_estimate_obj)(
                avg_jct=float(avg_jct),
                makespan=float(makespan),
                extra_switches=int(switches.extra_vs_failover),
            )
            scored += 1
            entry = (obj, copy_mapping(mapping), f"{source}|simulator_selected")
            if best_entry is None or repair_better(
                obj,
                best_entry[0],
                tol=self.config.score_sort_tolerance,
            ):
                best_entry = entry
        if best_entry is None:
            raise RuntimeError("no feasible repair candidate survived simulator selection")
        best_obj, best_mapping, best_source = best_entry
        return best_obj, best_mapping, best_source, scored

    def _postprocess_simulator_selected_mapping(
        self,
        best_obj,
        best_mapping: Mapping,
        best_source: str,
        deadline: float,
    ):
        del deadline
        return best_obj, best_mapping, best_source, 0

    def _greedy_simulator_augmented_moves(
        self,
        mapping: Mapping,
        analysis,
        deadline: float,
        *,
        limit: int,
    ):
        protection = set(int(server) for server in self.scenario.global_protection_pool)
        active_servers = self._active_servers(mapping)
        free_protection = [
            int(server)
            for server in protection
            if int(server) not in active_servers
        ]
        if not free_protection:
            return []

        failed_key = (
            int(self.scenario.failure.tenant),
            infer_failed_rank(self.scenario.pre_failure_mapping, self.scenario.failure),
        )
        rows = []
        for tenant in participating_tenants(self.scenario):
            if time.time() >= deadline:
                break
            tenant = int(tenant)
            if tenant not in self.candidate_sets:
                continue
            ranks = self._impact_rank_order(mapping, tenant, analysis)[:8]
            task_pair_prices = self._task_pair_prices(mapping, tenant, analysis)
            available = [
                int(server)
                for server in free_protection
                if int(server) in self.candidate_sets[tenant]
            ]
            available = self._ranked_available_servers(
                mapping,
                tenant,
                ranks,
                available,
                task_pair_prices,
                analysis,
            )[:4]
            for rank in ranks:
                rank = int(rank)
                if (tenant, rank) == failed_key:
                    continue
                if int(mapping[tenant][rank]) in protection:
                    continue
                impact = self._rank_structural_impact(mapping, tenant, rank, analysis)
                for server in available:
                    if time.time() >= deadline:
                        break
                    candidate = copy_mapping(mapping)
                    candidate[tenant][rank] = int(server)
                    try:
                        check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                    except ValueError:
                        continue
                    estimate_obj = self.evaluator.estimate(candidate)
                    rows.append((
                        (
                            float(estimate_obj.avg_jct),
                            -float(impact),
                            float(estimate_obj.makespan),
                            int(estimate_obj.extra_switches),
                            tenant,
                            rank,
                            int(server),
                        ),
                        estimate_obj,
                        candidate,
                        f"impact_single:t{tenant}:r{rank}:s{int(server)}",
                    ))
        rows.sort(key=lambda item: item[0])
        selected = []
        seen = set()
        for _key, estimate_obj, candidate, source in rows:
            sig = self._signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            selected.append((estimate_obj, candidate, source))
            if len(selected) >= int(limit):
                break
        return selected

    def _candidate_better(
        self,
        candidate_obj,
        incumbent_obj,
        candidate_mapping: Mapping,
        candidate_source: str,
        incumbent_mapping: Mapping,
        incumbent_source: str,
    ) -> bool:
        del candidate_source, incumbent_source
        tol = float(self.config.score_sort_tolerance)
        if float(candidate_obj.avg_jct) < float(incumbent_obj.avg_jct) - tol:
            return True
        if float(candidate_obj.avg_jct) > float(incumbent_obj.avg_jct) + tol:
            return False
        if float(candidate_obj.makespan) < float(incumbent_obj.makespan) - tol:
            return True
        if float(candidate_obj.makespan) > float(incumbent_obj.makespan) + tol:
            return False
        if int(candidate_obj.extra_switches) < int(incumbent_obj.extra_switches):
            return True
        if int(candidate_obj.extra_switches) > int(incumbent_obj.extra_switches):
            return False

        return False

    def _entry_sort_key(self, entry):
        obj, _mapping, _source = entry
        tol = float(self.config.score_sort_tolerance)
        return (
            round(float(obj.avg_jct) / tol),
            round(float(obj.makespan) / tol),
            int(obj.extra_switches),
        )

    def _structural_seed_candidates(self, mapping: Mapping, tenant: int):
        analysis = self._safe_analysis(mapping)
        task_pair_prices = self._task_pair_prices(mapping, int(tenant), analysis)
        yield from self._protection_switch_candidates(mapping, int(tenant), analysis, task_pair_prices)

    def _tenant_order(self, mapping: Mapping, analysis) -> tuple[int, ...]:
        allowed = list(self.allowed_tenants())
        failed = int(self.scenario.failure.tenant)
        allowed.sort(
            key=lambda tenant: (
                tenant != failed,
                -float(getattr(analysis, "tenant_pressure", {}).get(int(tenant), 0.0)),
                -float(getattr(analysis, "tenant_peak_load", {}).get(int(tenant), 0.0)),
                int(tenant),
            )
        )
        return tuple(allowed)

    def _tenant_candidates(self, mapping: Mapping, tenant: int, analysis, deadline: float):
        for candidate, source, _proposal_cost in self._ranked_tenant_candidates(
            mapping,
            tenant,
            analysis,
            deadline,
            limit=self.config.max_candidates_per_tenant,
        ):
            yield candidate, source

    def _ranked_tenant_candidates(
        self,
        mapping: Mapping,
        tenant: int,
        analysis,
        deadline: float,
        *,
        limit: int,
    ) -> list[tuple[Mapping, str, float]]:
        seen = {self._signature(mapping)}
        task_pair_prices = self._task_pair_prices(mapping, tenant, analysis)
        proposals = []
        selected = []
        candidate_generators = (
            self._protection_switch_candidates,
            self._block_protection_switch_candidates,
        )
        for generator in candidate_generators:
            for candidate, source, proposal_cost in generator(mapping, tenant, analysis, task_pair_prices):
                if time.time() >= deadline:
                    return selected
                sig = self._signature(candidate)
                if sig in seen:
                    continue
                seen.add(sig)
                if not self._within_switch_budget(candidate, tenant):
                    continue
                proposals.append((float(proposal_cost), source, candidate))
        proposals.sort(key=lambda entry: (entry[0], entry[1]))
        self.price_guided_candidates += len(proposals)
        for proposal_cost, source, candidate in self._diverse_proposal_order(proposals, tenant):
            if time.time() >= deadline:
                break
            selected.append((candidate, source, float(proposal_cost)))
            if len(selected) >= int(limit):
                break
        return selected

    def _diverse_proposal_order(
        self,
        proposals: list[tuple[float, str, Mapping]],
        tenant: int,
    ) -> list[tuple[float, str, Mapping]]:
        if not proposals:
            return []

        ordered: list[tuple[float, str, Mapping]] = []
        seen = set()

        def add(entry) -> None:
            sig = self._signature(entry[2])
            if sig in seen:
                return
            seen.add(sig)
            ordered.append(entry)

        # Keep the best price candidates first, then inject structured
        # diversity so joint repair sees more than one local neighborhood.
        for entry in proposals[: max(1, min(4, len(proposals)))]:
            add(entry)

        buckets: dict[tuple[int, str], list[tuple[float, str, Mapping]]] = {}
        for entry in proposals:
            _cost, source, candidate = entry
            switch_count = self._tenant_extra_switch_count(candidate, int(tenant))
            source_kind = "block" if "block_protection_switch" in source else "single"
            buckets.setdefault((switch_count, source_kind), []).append(entry)

        for switch_count in sorted({key[0] for key in buckets}):
            for source_kind in ("single", "block"):
                bucket = buckets.get((switch_count, source_kind), [])
                for entry in bucket[:2]:
                    add(entry)

        for entry in proposals:
            add(entry)
        return ordered

    def _tenant_extra_switch_count(self, mapping: Mapping, tenant: int) -> int:
        switches = count_switches(
            pre_failure_mapping=self.scenario.pre_failure_mapping,
            failover_mapping=self.failover_mapping,
            repaired_mapping=mapping,
            failure=self.scenario.failure,
        )
        return sum(
            1
            for moved_tenant, _rank in switches.extra_moved_ranks_vs_failover
            if int(moved_tenant) == int(tenant)
        )

    def _joint_candidates(
        self,
        mapping: Mapping,
        tenant_candidate_cache: dict[int, list[tuple[Mapping, str, float]]],
        deadline: float,
    ):
        tenants = [
            int(tenant)
            for tenant, candidates in tenant_candidate_cache.items()
            if candidates
        ]
        max_joint_tenants = min(int(self.config.max_joint_tenants), len(tenants))
        if max_joint_tenants < 2:
            return

        per_tenant_limit = max(1, int(self.config.joint_candidates_per_tenant))
        max_joint_candidates = max(0, int(self.config.max_joint_candidates))
        if max_joint_candidates <= 0:
            return

        yielded = 0
        seen = {self._signature(mapping)}
        for width in range(2, max_joint_tenants + 1):
            for tenant_group in itertools.combinations(tenants, width):
                candidate_lists = [
                    tenant_candidate_cache[int(tenant)][:per_tenant_limit]
                    for tenant in tenant_group
                ]
                combinations = []
                for combo in itertools.product(*candidate_lists):
                    proposal_cost = sum(float(entry[2]) for entry in combo)
                    combinations.append((proposal_cost, combo))
                combinations.sort(key=lambda entry: entry[0])
                for _proposal_cost, combo in combinations:
                    if time.time() >= deadline:
                        return
                    candidate = copy_mapping(mapping)
                    source_parts = []
                    changed = False
                    for tenant, tenant_candidate in zip(tenant_group, combo):
                        tenant = int(tenant)
                        next_tenant_mapping = tenant_candidate[0][tenant]
                        if candidate[tenant] != next_tenant_mapping:
                            changed = True
                        candidate[tenant] = dict(next_tenant_mapping)
                        source_parts.append(tenant_candidate[1])
                    if not changed:
                        continue
                    sig = self._signature(candidate)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    self.joint_guided_candidates += 1
                    yielded += 1
                    yield candidate, "joint:" + "+".join(source_parts)
                    if yielded >= max_joint_candidates:
                        return

    def _protection_switch_candidates(self, mapping: Mapping, tenant: int, analysis, task_pair_prices):
        ranks = self._rank_order(tenant, analysis)
        active_servers = self._active_servers(mapping)
        protection = set(int(server) for server in self.scenario.global_protection_pool)
        available = [
            server
            for server in self.candidate_sets[int(tenant)]
            if int(server) not in active_servers and int(server) in protection
        ]
        available = sorted(available, key=self._server_leaf_sort_key)
        for rank in ranks:
            for server in available:
                if int(mapping[int(tenant)][int(rank)]) == int(server):
                    continue
                candidate = copy_mapping(mapping)
                candidate[int(tenant)][int(rank)] = int(server)
                price_cost = self._tenant_price_cost(
                    candidate,
                    int(tenant),
                    candidate[int(tenant)],
                    task_pair_prices,
                    analysis,
                )
                yield candidate, f"price_protection_switch:t{tenant}:r{rank}:s{server}", price_cost

    def _block_protection_switch_candidates(self, mapping: Mapping, tenant: int, analysis, task_pair_prices):
        rank_order = self._rank_order(tenant, analysis)
        if len(rank_order) <= 1:
            return
        active_servers = self._active_servers(mapping)
        protection = set(int(server) for server in self.scenario.global_protection_pool)
        available = [
            int(server)
            for server in self.candidate_sets[int(tenant)]
            if int(server) not in active_servers and int(server) in protection
        ]
        available = self._ranked_available_servers(
            mapping,
            int(tenant),
            rank_order[: max(1, int(self.config.max_block_ranks))],
            available,
            task_pair_prices,
            analysis,
        )
        groups = self._block_rank_groups(int(tenant), rank_order)
        yielded = 0
        seen = set()
        for group in groups:
            if yielded >= int(self.config.max_block_candidates):
                return
            group = tuple(int(rank) for rank in group)
            if group in seen:
                continue
            seen.add(group)
            group_released_protection = [
                int(mapping[int(tenant)][rank])
                for rank in group
                if int(mapping[int(tenant)][rank]) in protection
            ]
            server_options = list(dict.fromkeys(
                group_released_protection
                + available[: max(0, int(self.config.block_extra_servers))]
            ))
            if len(server_options) < len(group):
                continue
            for assignment in itertools.permutations(server_options, len(group)):
                if yielded >= int(self.config.max_block_candidates):
                    return
                candidate = copy_mapping(mapping)
                for rank, server in zip(group, assignment):
                    candidate[int(tenant)][int(rank)] = int(server)
                price_cost = self._tenant_price_cost(
                    candidate,
                    int(tenant),
                    candidate[int(tenant)],
                    task_pair_prices,
                    analysis,
                )
                yielded += 1
                yield (
                    candidate,
                    "price_block_protection_switch:"
                    f"t{tenant}:r{'-'.join(str(rank) for rank in group)}",
                    price_cost,
                )

    def _ranked_available_servers(
        self,
        mapping: Mapping,
        tenant: int,
        ranks: list[int],
        available: list[int],
        task_pair_prices,
        analysis,
    ) -> list[int]:
        scored = []
        for server in available:
            best = float("inf")
            for rank in ranks:
                candidate = copy_mapping(mapping)
                candidate[int(tenant)][int(rank)] = int(server)
                best = min(
                    best,
                    self._tenant_price_cost(
                        candidate,
                        int(tenant),
                        candidate[int(tenant)],
                        task_pair_prices,
                        analysis,
                    ),
                )
            scored.append((float(best), self._server_leaf_sort_key(int(server)), int(server)))
        return [
            int(server)
            for _score, _leaf_key, server in sorted(scored)
        ]

    @staticmethod
    def _active_servers(mapping: Mapping) -> set[int]:
        return {
            int(server)
            for ranks in mapping.values()
            for server in ranks.values()
        }

    def _block_rank_groups(self, tenant: int, rank_order: list[int]) -> list[tuple[int, ...]]:
        max_block = min(max(2, int(self.config.max_block_ranks)), len(rank_order))
        groups: list[tuple[int, ...]] = []
        hot = [int(rank) for rank in rank_order[: max_block + 2]]
        for size in range(2, max_block + 1):
            groups.append(tuple(hot[:size]))
            for combo in itertools.combinations(hot, size):
                groups.append(tuple(int(rank) for rank in combo))

        branch = [
            int(rank)
            for rank in self._branch_order(int(tenant))
            if int(rank) in set(rank_order)
        ]
        if branch:
            for size in range(2, max_block + 1):
                for start in range(0, max(1, min(len(branch) - size + 1, 4))):
                    groups.append(tuple(branch[start:start + size]))

        if int(tenant) == int(self.scenario.failure.tenant):
            failed_rank = infer_failed_rank(self.scenario.pre_failure_mapping, self.scenario.failure)
            if int(failed_rank) in set(rank_order):
                partners = [
                    int(rank)
                    for rank in rank_order
                    if int(rank) != int(failed_rank)
                ]
                for size in range(2, max_block + 1):
                    groups.append(tuple([int(failed_rank), *partners[: size - 1]]))

        deduped = []
        seen = set()
        for group in groups:
            key = tuple(int(rank) for rank in group)
            if len(set(key)) != len(key) or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _task_infos(self, tenant: int):
        task_meta = self.evaluator.estimator.task_dag.get(int(tenant), {})
        infos = []
        for task_id, task_tuple in task_meta.get("task_info", {}).items():
            infos.append(
                (
                    int(task_id),
                    int(task_tuple[1]),
                    int(task_tuple[2]),
                    float(task_tuple[3]),
                )
            )
        return infos

    def _task_pair_prices(self, mapping: Mapping, tenant: int, analysis):
        try:
            return self.evaluator.estimator.task_pair_price_lookup(
                mapping,
                int(tenant),
                self.candidate_sets[int(tenant)],
                analysis=analysis,
            )
        except Exception:
            self.price_lookup_fallbacks += 1
            return {}

    def _task_pair_price(
        self,
        mapping: Mapping,
        tenant: int,
        task_id: int,
        src_server: int,
        dst_server: int,
        task_pair_prices,
        analysis,
    ) -> float:
        if int(src_server) == int(dst_server):
            return 0.0
        key = (int(task_id), int(src_server), int(dst_server))
        if key in task_pair_prices:
            return float(task_pair_prices[key])
        return float(
            self.evaluator.estimator.task_pair_price(
                mapping,
                int(tenant),
                int(task_id),
                int(src_server),
                int(dst_server),
                analysis=analysis,
            )
        )

    def _tenant_price_cost(
        self,
        mapping: Mapping,
        tenant: int,
        tenant_mapping: dict[int, int],
        task_pair_prices,
        analysis,
    ) -> float:
        total = 0.0
        for task_id, src_rank, dst_rank, volume in self._task_infos(int(tenant)):
            src_server = int(tenant_mapping[int(src_rank)])
            dst_server = int(tenant_mapping[int(dst_rank)])
            total += float(volume) * self._task_pair_price(
                mapping,
                int(tenant),
                int(task_id),
                src_server,
                dst_server,
                task_pair_prices,
                analysis,
            )
        return float(total)

    def _branch_order(self, tenant: int) -> list[int]:
        data = self.evaluator.compiled_dag_data
        branch_order = (
            data.get("compiled_schedule", {})
            .get("per_tenant", {})
            .get(int(tenant), {})
            .get("branch_order", [])
        )
        return [int(rank) for rank in branch_order]

    def _rank_order(self, tenant: int, analysis) -> list[int]:
        ranks = sorted(self.failover_mapping[int(tenant)])
        pressure = getattr(analysis, "rank_pressure", {})
        branch_order = self._branch_order(int(tenant))
        branch_position = {int(rank): idx for idx, rank in enumerate(branch_order)}
        return sorted(
            ranks,
            key=lambda rank: (
                -float(pressure.get((int(tenant), int(rank)), 0.0)),
                branch_position.get(int(rank), len(branch_position)),
                int(rank),
            ),
        )

    def _impact_rank_order(self, mapping: Mapping, tenant: int, analysis) -> list[int]:
        ranks = list(self._rank_order(int(tenant), analysis))
        return sorted(
            ranks,
            key=lambda rank: (
                -self._rank_structural_impact(mapping, int(tenant), int(rank), analysis),
                ranks.index(int(rank)),
                int(rank),
            ),
        )

    def _rank_structural_impact(self, mapping: Mapping, tenant: int, rank: int, analysis) -> float:
        tenant_mapping = mapping[int(tenant)]
        rank = int(rank)
        if rank not in tenant_mapping:
            return 0.0
        rank_server = int(tenant_mapping[rank])
        rank_leaf = self._server_leaf_sort_key(rank_server)[0]
        impact = 0.0
        for _task_id, src_rank, dst_rank, volume in self._task_infos(int(tenant)):
            if int(src_rank) != rank and int(dst_rank) != rank:
                continue
            other_rank = int(dst_rank) if int(src_rank) == rank else int(src_rank)
            if other_rank not in tenant_mapping:
                continue
            other_leaf = self._server_leaf_sort_key(int(tenant_mapping[other_rank]))[0]
            leaf_weight = 2.0 if int(other_leaf) != int(rank_leaf) else 1.0
            impact += float(volume) * leaf_weight
        pressure = getattr(analysis, "rank_pressure", {})
        impact += float(pressure.get((int(tenant), rank), 0.0))
        return float(impact)

    def _within_switch_budget(self, mapping: Mapping, tenant: int) -> bool:
        if self.config.max_extra_switches_per_tenant is None:
            return True
        switches = count_switches(
            pre_failure_mapping=self.scenario.pre_failure_mapping,
            failover_mapping=self.failover_mapping,
            repaired_mapping=mapping,
            failure=self.scenario.failure,
        )
        tenant_extra = sum(1 for moved_tenant, _rank in switches.extra_moved_ranks_vs_failover if moved_tenant == int(tenant))
        return tenant_extra <= int(self.config.max_extra_switches_per_tenant)

    def _select_beam(self, entries):
        best_by_signature = {}
        for obj, mapping, source in entries:
            sig = self._signature(mapping)
            old = best_by_signature.get(sig)
            if old is None or repair_better(obj, old[0], tol=self.config.score_sort_tolerance):
                best_by_signature[sig] = (obj, mapping, source)

        ranked = sorted(best_by_signature.values(), key=self._entry_sort_key)
        selected = ranked[: self.config.beam_width]

        # Keep low-disruption alternatives alive even when their estimated score
        # is close but not identical to the current best.
        by_switch_count = {}
        for entry in ranked:
            by_switch_count.setdefault(entry[0].extra_switches, entry)
        for entry in by_switch_count.values():
            if len(selected) >= self.config.beam_width + 3:
                break
            if self._signature(entry[1]) not in {self._signature(item[1]) for item in selected}:
                selected.append(entry)
        return sorted(selected, key=self._entry_sort_key)

    def _safe_analysis(self, mapping: Mapping):
        try:
            return self.evaluator.analyze(mapping)
        except Exception:
            class EmptyAnalysis:
                tenant_pressure = {}
                tenant_peak_load = {}
                rank_pressure = {}
                slot_prices = []
                task_active_slots = {}
                task_ready_slots = {}

            return EmptyAnalysis()

    def _server_leaf_sort_key(self, server: int) -> tuple[int, int]:
        if hasattr(self.evaluator.datacenter, "get_server_leaf"):
            return (int(self.evaluator.datacenter.get_server_leaf(int(server))), int(server))
        return (int(server), int(server))

    @staticmethod
    def _signature(mapping: Mapping) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
        return tuple(
            (int(tenant), tuple(sorted((int(rank), int(server)) for rank, server in ranks.items())))
            for tenant, ranks in sorted(mapping.items())
        )

    def _beam_signature(self, beam) -> tuple:
        return tuple(self._signature(mapping) for _obj, mapping, _source in beam)


class FailoverOnlyRepairHeuristic(FailureAwareRepairHeuristicBase):
    name = "repair_failed_server_only"

    def allowed_tenants(self) -> tuple[int, ...]:
        return (int(self.scenario.failure.tenant),)

    def _initial_seed_entries(self, deadline: float):
        analysis = self._safe_analysis(self.failover_mapping)
        entries = []
        for candidate, source in self._slot_assignment_candidates(
            analysis,
            deadline,
            limit=max(1, len(self.scenario.global_protection_pool)),
            source_prefix="slot_assign_failover",
            participants_override=(int(self.scenario.failure.tenant),),
        ) or ():
            if time.time() >= deadline:
                break
            try:
                check_repair_feasible(self.scenario, candidate, self.failover_mapping)
            except ValueError:
                continue
            entries.append((self.evaluator.estimate(candidate), candidate, source))
        if not entries:
            failover = copy_mapping(self.failover_mapping)
            entries.append((self.evaluator.estimate(failover), failover, "failover"))
        return entries

    def _postprocess_simulator_selected_mapping(
        self,
        best_obj,
        best_mapping: Mapping,
        best_source: str,
        deadline: float,
    ):
        del deadline
        return best_obj, best_mapping, best_source, 0


class TenantLocalRepairHeuristic(FailureAwareRepairHeuristicBase):
    name = "tenant_local_repair"

    def allowed_tenants(self) -> tuple[int, ...]:
        return (int(self.scenario.failure.tenant),)


class CooperativeRepairHeuristic(FailureAwareRepairHeuristicBase):
    name = "cooperative_repair"

    def allowed_tenants(self) -> tuple[int, ...]:
        return participating_tenants(self.scenario)

    def _initial_seed_entries(self, deadline: float):
        failover = copy_mapping(self.failover_mapping)
        entries = [(self.evaluator.estimate(failover), failover, "failover")]
        seen = {self._signature(failover)}

        def add_seed(mapping: Mapping, source: str) -> None:
            if time.time() >= deadline:
                return
            sig = self._signature(mapping)
            if sig in seen:
                return
            try:
                check_repair_feasible(self.scenario, mapping, self.failover_mapping)
            except ValueError:
                return
            seen.add(sig)
            entries.append((self.evaluator.estimate(mapping), copy_mapping(mapping), source))

        for mapping, source in self.reference_seeds:
            add_seed(mapping, f"reference:{source}")

        analysis = self._safe_analysis(failover)
        for candidate, source in self._slot_assignment_candidates(
            analysis,
            deadline,
            limit=self._candidate_output_budget(is_coordinate=False),
            source_prefix="slot_assign_coordinate:local_scope",
            participants_override=(int(self.scenario.failure.tenant),),
        ) or ():
            add_seed(candidate, source)

        for candidate, source in self._slot_assignment_candidates(
            analysis,
            deadline,
            limit=self._candidate_output_budget(is_coordinate=True),
            source_prefix="slot_assign_coordinate",
        ) or ():
            add_seed(candidate, source)

        return entries

    def _add_initial_joint_seeds(
        self,
        entries,
        seen: set,
        base_mapping: Mapping,
        deadline: float,
    ) -> None:
        if time.time() >= deadline:
            return
        analysis = self._safe_analysis(base_mapping)
        candidate_budget = max(1, int(self.config.max_joint_candidates))
        global_yielded = 0
        for candidate, source in self._global_protection_assignment_candidates(
            base_mapping,
            analysis,
            deadline,
            limit=max(1, candidate_budget // 2),
        ) or ():
            if time.time() >= deadline:
                return
            sig = self._signature(candidate)
            if sig in seen:
                continue
            try:
                check_repair_feasible(self.scenario, candidate, self.failover_mapping)
            except ValueError:
                continue
            seen.add(sig)
            entries.append((self.evaluator.estimate(candidate), candidate, source))
            self.initial_joint_seed_candidates += 1
            global_yielded += 1

        single_yielded = 0
        for candidate, source in self._single_cross_tenant_protection_moves(
            base_mapping,
            analysis,
            deadline,
            limit=max(1, candidate_budget - global_yielded),
        ) or ():
            if time.time() >= deadline:
                return
            sig = self._signature(candidate)
            if sig in seen:
                continue
            try:
                check_repair_feasible(self.scenario, candidate, self.failover_mapping)
            except ValueError:
                continue
            seen.add(sig)
            entries.append((self.evaluator.estimate(candidate), candidate, source))
            self.initial_joint_seed_candidates += 1
            single_yielded += 1

        if time.time() >= deadline:
            return

        impact_yielded = 0
        for candidate, source in self._impact_guided_coordinate_moves(
            base_mapping,
            analysis,
            deadline,
            limit=max(1, min(candidate_budget, candidate_budget - global_yielded - single_yielded)),
        ) or ():
            if time.time() >= deadline:
                return
            sig = self._signature(candidate)
            if sig in seen:
                continue
            try:
                check_repair_feasible(self.scenario, candidate, self.failover_mapping)
            except ValueError:
                continue
            seen.add(sig)
            entries.append((self.evaluator.estimate(candidate), candidate, source))
            self.initial_joint_seed_candidates += 1
            impact_yielded += 1

        if time.time() >= deadline:
            return

        for candidate, source in self._direct_joint_protection_moves(
            base_mapping,
            analysis,
            deadline,
            limit=max(1, candidate_budget - global_yielded - single_yielded - impact_yielded),
        ) or ():
            if time.time() >= deadline:
                return
            sig = self._signature(candidate)
            if sig in seen:
                continue
            try:
                check_repair_feasible(self.scenario, candidate, self.failover_mapping)
            except ValueError:
                continue
            seen.add(sig)
            entries.append((self.evaluator.estimate(candidate), candidate, source))
            self.initial_joint_seed_candidates += 1

    def _impact_guided_coordinate_moves(
        self,
        mapping: Mapping,
        analysis,
        deadline: float,
        *,
        limit: int,
    ):
        if int(limit) <= 0:
            return

        protection = tuple(int(server) for server in self.scenario.global_protection_pool)
        active_servers = self._active_servers(mapping)
        free_protection = [
            int(server)
            for server in protection
            if int(server) not in active_servers
        ]
        if not free_protection:
            return

        failed_key = (
            int(self.scenario.failure.tenant),
            infer_failed_rank(self.scenario.pre_failure_mapping, self.scenario.failure),
        )
        per_tenant_rank_limit = max(2, min(6, int(self.config.max_block_ranks) + 3))
        per_rank_server_limit = max(2, min(4, int(self.config.block_extra_servers)))
        move_rows: list[tuple[float, int, int, int, Mapping]] = []

        tenants = list(participating_tenants(self.scenario))
        tenants.sort(
            key=lambda tenant: (
                -float(getattr(analysis, "tenant_pressure", {}).get(int(tenant), 0.0)),
                int(tenant),
            )
        )
        for tenant in tenants:
            if time.time() >= deadline:
                return
            tenant = int(tenant)
            if tenant not in self.candidate_sets:
                continue
            ranks = self._impact_rank_order(mapping, tenant, analysis)[:per_tenant_rank_limit]
            task_pair_prices = self._task_pair_prices(mapping, tenant, analysis)
            candidate_servers = [
                int(server)
                for server in free_protection
                if int(server) in self.candidate_sets[tenant]
            ]
            candidate_servers = self._ranked_available_servers(
                mapping,
                tenant,
                ranks,
                candidate_servers,
                task_pair_prices,
                analysis,
            )[:per_rank_server_limit]
            for rank in ranks:
                rank = int(rank)
                if (tenant, rank) == failed_key:
                    continue
                if int(mapping[tenant][rank]) in protection:
                    continue
                rank_impact = self._rank_structural_impact(mapping, tenant, rank, analysis)
                for server in candidate_servers:
                    if time.time() >= deadline:
                        return
                    if int(mapping[tenant][rank]) == int(server):
                        continue
                    candidate = copy_mapping(mapping)
                    candidate[tenant][rank] = int(server)
                    try:
                        check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                    except ValueError:
                        continue
                    cost = self._tenant_price_cost(
                        candidate,
                        tenant,
                        candidate[tenant],
                        task_pair_prices,
                        analysis,
                    )
                    score = (
                        -float(rank_impact),
                        float(cost),
                        self._server_leaf_sort_key(int(server))[0],
                        tenant,
                        rank,
                        int(server),
                    )
                    move_rows.append((score, tenant, rank, int(server), candidate))

        move_rows.sort(key=lambda item: item[0])
        yielded = 0
        seen = set()
        for _score, tenant, rank, server, candidate in move_rows:
            if time.time() >= deadline:
                return
            sig = self._signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            yielded += 1
            yield candidate, f"impact_guided:single:t{tenant}:r{rank}:s{server}"
            if yielded >= int(limit):
                return

        max_pair_rows = min(len(move_rows), max(8, int(limit) * 2))
        for left_index in range(max_pair_rows):
            _left_score, left_tenant, left_rank, left_server, _left_candidate = move_rows[left_index]
            for right_index in range(left_index + 1, max_pair_rows):
                if time.time() >= deadline:
                    return
                _right_score, right_tenant, right_rank, right_server, _right_candidate = move_rows[right_index]
                if int(left_tenant) == int(right_tenant):
                    continue
                if int(left_server) == int(right_server):
                    continue
                candidate = copy_mapping(mapping)
                candidate[int(left_tenant)][int(left_rank)] = int(left_server)
                candidate[int(right_tenant)][int(right_rank)] = int(right_server)
                try:
                    check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                except ValueError:
                    continue
                sig = self._signature(candidate)
                if sig in seen:
                    continue
                seen.add(sig)
                yielded += 1
                yield (
                    candidate,
                    "impact_guided:pair:"
                    f"t{left_tenant}:r{left_rank}:s{left_server}+"
                    f"t{right_tenant}:r{right_rank}:s{right_server}",
                )
                if yielded >= int(limit):
                    return

    def _impact_rank_order(self, mapping: Mapping, tenant: int, analysis) -> list[int]:
        ranks = list(self._rank_order(int(tenant), analysis))
        return sorted(
            ranks,
            key=lambda rank: (
                -self._rank_structural_impact(mapping, int(tenant), int(rank), analysis),
                ranks.index(int(rank)),
                int(rank),
            ),
        )

    def _rank_structural_impact(self, mapping: Mapping, tenant: int, rank: int, analysis) -> float:
        tenant_mapping = mapping[int(tenant)]
        rank = int(rank)
        if rank not in tenant_mapping:
            return 0.0
        rank_server = int(tenant_mapping[rank])
        rank_leaf = self._server_leaf_sort_key(rank_server)[0]
        impact = 0.0
        for _task_id, src_rank, dst_rank, volume in self._task_infos(int(tenant)):
            if int(src_rank) != rank and int(dst_rank) != rank:
                continue
            other_rank = int(dst_rank) if int(src_rank) == rank else int(src_rank)
            if other_rank not in tenant_mapping:
                continue
            other_leaf = self._server_leaf_sort_key(int(tenant_mapping[other_rank]))[0]
            leaf_weight = 2.0 if int(other_leaf) != int(rank_leaf) else 1.0
            impact += float(volume) * leaf_weight
        pressure = getattr(analysis, "rank_pressure", {})
        impact += float(pressure.get((int(tenant), rank), 0.0))
        return float(impact)

    def _single_cross_tenant_protection_moves(
        self,
        mapping: Mapping,
        analysis,
        deadline: float,
        *,
        limit: int,
    ):
        failed_tenant = int(self.scenario.failure.tenant)
        protection = set(int(server) for server in self.scenario.global_protection_pool)
        active_servers = self._active_servers(mapping)
        yielded = 0
        for tenant in self._tenant_order(mapping, analysis):
            tenant = int(tenant)
            if time.time() >= deadline:
                return
            if tenant == failed_tenant:
                continue
            ranks = self._rank_order(tenant, analysis)[
                : max(2, int(self.config.max_block_ranks) + 2)
            ]
            task_pair_prices = self._task_pair_prices(mapping, tenant, analysis)
            available = [
                int(server)
                for server in protection
                if int(server) in self.candidate_sets[tenant]
                and int(server) not in active_servers
            ]
            available = self._ranked_available_servers(
                mapping,
                tenant,
                ranks,
                available,
                task_pair_prices,
                analysis,
            )
            scored = []
            for rank in ranks:
                rank = int(rank)
                for server in available:
                    if int(mapping[tenant][rank]) == int(server):
                        continue
                    candidate = copy_mapping(mapping)
                    candidate[tenant][rank] = int(server)
                    cost = self._tenant_price_cost(
                        candidate,
                        tenant,
                        candidate[tenant],
                        task_pair_prices,
                        analysis,
                    )
                    scored.append((float(cost), tenant, rank, int(server), candidate))
            for _cost, move_tenant, rank, server, candidate in sorted(scored, key=lambda item: item[:4]):
                if time.time() >= deadline:
                    return
                yielded += 1
                yield candidate, f"coordinate_single:t{move_tenant}:r{rank}:s{server}"
                if yielded >= int(limit):
                    return

    def _candidate_better(
        self,
        candidate_obj,
        incumbent_obj,
        candidate_mapping: Mapping,
        candidate_source: str,
        incumbent_mapping: Mapping,
        incumbent_source: str,
    ) -> bool:
        if super()._candidate_better(
            candidate_obj,
            incumbent_obj,
            candidate_mapping,
            candidate_source,
            incumbent_mapping,
            incumbent_source,
        ):
            return True
        if not (
            str(candidate_source).startswith("joint:")
            or str(candidate_source).startswith("direct_joint:")
            or str(candidate_source).startswith("global_assign:")
            or str(candidate_source).startswith("impact_guided:")
            or str(candidate_source).startswith("slot_assign_")
        ):
            return False
        slack = max(0.0, float(self.config.cooperative_near_tie_slack))
        if slack <= 0.0:
            return False

        failover_avg_jct = float(self.evaluator.estimate(self.failover_mapping).avg_jct)
        if (
            int(candidate_obj.extra_switches) < int(incumbent_obj.extra_switches)
            and float(candidate_obj.avg_jct)
            <= float(incumbent_obj.avg_jct) * (1.0 + slack)
            and float(candidate_obj.makespan)
            <= float(incumbent_obj.makespan) * (1.0 + slack)
            and float(candidate_obj.avg_jct) <= failover_avg_jct
        ):
            return True

        candidate_shape = self._cooperative_move_shape(candidate_mapping)
        incumbent_shape = self._cooperative_move_shape(incumbent_mapping)
        if candidate_shape[0] <= incumbent_shape[0]:
            return False
        if candidate_obj.avg_jct > incumbent_obj.avg_jct * (1.0 + slack):
            return False
        if candidate_obj.makespan > incumbent_obj.makespan * (1.0 + slack):
            return False
        if candidate_obj.avg_jct > failover_avg_jct:
            return False
        return True

    def _postprocess_best_mapping(
        self,
        best_obj,
        best_mapping: Mapping,
        best_source: str,
        deadline: float,
    ):
        slack = max(0.0, float(self.config.cooperative_near_tie_slack))
        if slack <= 0.0:
            return best_obj, best_mapping, best_source

        current_obj = best_obj
        current_mapping = copy_mapping(best_mapping)
        current_source = best_source
        failover_avg_jct = float(self.evaluator.estimate(self.failover_mapping).avg_jct)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            switches = count_switches(
                pre_failure_mapping=self.scenario.pre_failure_mapping,
                failover_mapping=self.failover_mapping,
                repaired_mapping=current_mapping,
                failure=self.scenario.failure,
            )
            extra_moves = sorted(
                switches.extra_moved_ranks_vs_failover,
                key=lambda item: (int(item[0]), int(item[1])),
            )
            best_revert = None
            for tenant, rank in extra_moves:
                if time.time() >= deadline:
                    break
                candidate = copy_mapping(current_mapping)
                candidate[int(tenant)][int(rank)] = self.failover_mapping[int(tenant)][int(rank)]
                try:
                    check_repair_feasible(
                        self.scenario,
                        candidate,
                        self.failover_mapping,
                    )
                except ValueError:
                    continue
                obj = self.evaluator.estimate(candidate)
                if float(obj.avg_jct) > float(best_obj.avg_jct) * (1.0 + slack):
                    continue
                if float(obj.makespan) > float(best_obj.makespan) * (1.0 + slack):
                    continue
                if float(obj.avg_jct) > failover_avg_jct:
                    continue
                if int(obj.extra_switches) >= int(current_obj.extra_switches):
                    continue
                key = (
                    int(obj.extra_switches),
                    float(obj.avg_jct),
                    float(obj.makespan),
                    int(tenant),
                    int(rank),
                )
                if best_revert is None or key < best_revert[0]:
                    best_revert = (key, obj, candidate, tenant, rank)
            if best_revert is not None:
                _key, current_obj, current_mapping, tenant, rank = best_revert
                current_source = (
                    f"{current_source}|near_tie_switch_compress:t{int(tenant)}:r{int(rank)}"
                )
                improved = True

        return current_obj, current_mapping, current_source

    def _postprocess_simulator_selected_mapping(
        self,
        best_obj,
        best_mapping: Mapping,
        best_source: str,
        deadline: float,
    ):
        del deadline
        return best_obj, best_mapping, best_source, 0

    def _greedy_simulator_augmented_moves(
        self,
        mapping: Mapping,
        analysis,
        deadline: float,
        *,
        limit: int,
    ):
        protection = set(int(server) for server in self.scenario.global_protection_pool)
        active_servers = self._active_servers(mapping)
        free_protection = [
            int(server)
            for server in protection
            if int(server) not in active_servers
        ]
        if not free_protection:
            return []

        failed_key = (
            int(self.scenario.failure.tenant),
            infer_failed_rank(self.scenario.pre_failure_mapping, self.scenario.failure),
        )
        rows = []
        for tenant in participating_tenants(self.scenario):
            if time.time() >= deadline:
                break
            tenant = int(tenant)
            if tenant not in self.candidate_sets:
                continue
            ranks = self._impact_rank_order(mapping, tenant, analysis)[:8]
            task_pair_prices = self._task_pair_prices(mapping, tenant, analysis)
            available = [
                int(server)
                for server in free_protection
                if int(server) in self.candidate_sets[tenant]
            ]
            available = self._ranked_available_servers(
                mapping,
                tenant,
                ranks,
                available,
                task_pair_prices,
                analysis,
            )[:4]
            for rank in ranks:
                rank = int(rank)
                if (tenant, rank) == failed_key:
                    continue
                if int(mapping[tenant][rank]) in protection:
                    continue
                impact = self._rank_structural_impact(mapping, tenant, rank, analysis)
                for server in available:
                    if time.time() >= deadline:
                        break
                    candidate = copy_mapping(mapping)
                    candidate[tenant][rank] = int(server)
                    try:
                        check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                    except ValueError:
                        continue
                    estimate_obj = self.evaluator.estimate(candidate)
                    rows.append((
                        (
                            float(estimate_obj.avg_jct),
                            -float(impact),
                            float(estimate_obj.makespan),
                            int(estimate_obj.extra_switches),
                            tenant,
                            rank,
                            int(server),
                        ),
                        estimate_obj,
                        candidate,
                        f"impact_single:t{tenant}:r{rank}:s{int(server)}",
                    ))
        rows.sort(key=lambda item: item[0])
        selected = []
        seen = set()
        for _key, estimate_obj, candidate, source in rows:
            sig = self._signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            selected.append((estimate_obj, candidate, source))
            if len(selected) >= int(limit):
                break
        return selected

    def _joint_candidates(
        self,
        mapping: Mapping,
        tenant_candidate_cache: dict[int, list[tuple[Mapping, str, float]]],
        deadline: float,
    ):
        del tenant_candidate_cache
        max_joint_candidates = max(0, int(self.config.max_joint_candidates))
        if max_joint_candidates <= 0:
            return

        analysis = self._safe_analysis(self.failover_mapping)
        seen = {self._signature(mapping)}
        for candidate, source in self._slot_assignment_candidates(
            analysis,
            deadline,
            limit=max_joint_candidates,
            source_prefix="slot_assign_coordinate",
        ) or ():
            sig = self._signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            self.joint_guided_candidates += 1
            yield candidate, source

    def _global_protection_assignment_candidates(
        self,
        mapping: Mapping,
        analysis,
        deadline: float,
        *,
        limit: int,
    ):
        """Bounded global coordinate seed over unique protection slots."""

        del mapping
        if int(limit) <= 0:
            return
        protection = tuple(int(server) for server in self.scenario.global_protection_pool)
        if not protection:
            return

        failed_tenant = int(self.scenario.failure.tenant)
        failed_rank = infer_failed_rank(self.scenario.pre_failure_mapping, self.scenario.failure)
        base = copy_mapping(self.scenario.pre_failure_mapping)
        tenants = list(self._tenant_order(self.failover_mapping, analysis))
        per_tenant_ranks = max(1, int(self.config.joint_candidates_per_tenant))
        max_total_ranks = max(
            len(protection) + 2,
            int(self.config.max_joint_tenants) * per_tenant_ranks,
        )
        movable: list[tuple[int, int]] = []
        seen_ranks: set[tuple[int, int]] = set()
        for tenant in tenants:
            tenant = int(tenant)
            ranks = self._rank_order(tenant, analysis)[:per_tenant_ranks]
            if tenant == failed_tenant and int(failed_rank) not in ranks:
                ranks = [int(failed_rank), *ranks[: max(0, per_tenant_ranks - 1)]]
            for rank in ranks:
                key = (tenant, int(rank))
                if key == (failed_tenant, int(failed_rank)) or key in seen_ranks:
                    continue
                seen_ranks.add(key)
                movable.append(key)
                if len(movable) >= max_total_ranks:
                    break
            if len(movable) >= max_total_ranks:
                break

        price_cache = {
            int(tenant): self._task_pair_prices(self.failover_mapping, int(tenant), analysis)
            for tenant in tenants
            if int(tenant) in self.candidate_sets
        }
        max_extra = min(
            len(protection) - 1,
            len(movable),
            max(1, int(self.config.max_block_ranks)),
        )
        scored = []
        for failed_server in protection:
            if time.time() >= deadline:
                return
            remaining_servers = tuple(
                server for server in protection if int(server) != int(failed_server)
            )
            failed_base = copy_mapping(base)
            failed_base[failed_tenant][int(failed_rank)] = int(failed_server)
            single_moves = []
            for tenant, rank in movable:
                for server in remaining_servers:
                    if time.time() >= deadline:
                        return
                    candidate = copy_mapping(failed_base)
                    candidate[int(tenant)][int(rank)] = int(server)
                    try:
                        check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                    except ValueError:
                        continue
                    proxy_cost = self._tenant_price_cost(
                        candidate,
                        int(tenant),
                        candidate[int(tenant)],
                        price_cache.get(int(tenant), {}),
                        analysis,
                    )
                    single_moves.append(
                        (
                            float(proxy_cost),
                            int(tenant),
                            int(rank),
                            int(server),
                        )
                    )
            single_moves.sort(key=lambda item: item)
            move_window = single_moves[: max(8, min(len(single_moves), int(limit) * 2))]
            for width in range(1, max_extra + 1):
                for combo in itertools.combinations(move_window, width):
                    if time.time() >= deadline:
                        return
                    moved_ranks = [(int(tenant), int(rank)) for _cost, tenant, rank, _server in combo]
                    target_servers = [int(server) for _cost, _tenant, _rank, server in combo]
                    if len(set(moved_ranks)) != len(moved_ranks):
                        continue
                    if len(set(target_servers)) != len(target_servers):
                        continue
                    candidate = copy_mapping(failed_base)
                    source_parts = [
                        f"failed:t{failed_tenant}:r{int(failed_rank)}:s{int(failed_server)}"
                    ]
                    proxy_cost = 0.0
                    for move_cost, tenant, rank, server in combo:
                        candidate[int(tenant)][int(rank)] = int(server)
                        proxy_cost += float(move_cost)
                        source_parts.append(f"t{int(tenant)}:r{int(rank)}:s{int(server)}")
                    try:
                        check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                    except ValueError:
                        continue
                    scored.append(
                        (
                            float(proxy_cost),
                            len(combo),
                            source_parts,
                            candidate,
                        )
                    )

        yielded = 0
        seen = set()
        for _cost, width, source_parts, candidate in sorted(
            scored,
            key=lambda item: (item[0], -item[1], "+".join(item[2])),
        ):
            if time.time() >= deadline:
                return
            sig = self._signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            yielded += 1
            yield candidate, "global_assign:" + "+".join(source_parts)
            if yielded >= int(limit):
                return

    def _select_beam(self, entries):
        best_by_signature = {}
        for obj, mapping, source in entries:
            sig = self._signature(mapping)
            old = best_by_signature.get(sig)
            if old is None or repair_better(obj, old[0], tol=self.config.score_sort_tolerance):
                best_by_signature[sig] = (obj, mapping, source)

        ranked = sorted(best_by_signature.values(), key=self._entry_sort_key)
        selected = []
        selected_signatures = set()

        def add(entry) -> None:
            sig = self._signature(entry[1])
            if sig in selected_signatures:
                return
            selected_signatures.add(sig)
            selected.append(entry)

        base_width = int(self.config.beam_width)
        for entry in ranked[:base_width]:
            add(entry)

        # Cooperative repair needs enough live alternatives for later rounds to
        # combine moves across tenants.  Preserve a small frontier of joint and
        # multi-tenant candidates even when the switch-count tie-break would
        # otherwise discard them early.
        capacity = base_width + max(6, base_width)
        for entry in ranked:
            if len(selected) >= capacity:
                break
            if (
                str(entry[2]).startswith("joint:")
                or str(entry[2]).startswith("direct_joint:")
                or str(entry[2]).startswith("global_assign:")
                or str(entry[2]).startswith("impact_guided:")
                or str(entry[2]).startswith("slot_assign_")
            ):
                add(entry)

        by_participation = {}
        for entry in ranked:
            participating, extra_switches = self._cooperative_move_shape(entry[1])
            by_participation.setdefault((participating, extra_switches), entry)
        for key in sorted(by_participation, reverse=True):
            if len(selected) >= capacity:
                break
            add(by_participation[key])

        by_switch_count = {}
        for entry in ranked:
            by_switch_count.setdefault(entry[0].extra_switches, entry)
        for entry in by_switch_count.values():
            if len(selected) >= capacity:
                break
            add(entry)

        return sorted(selected, key=self._entry_sort_key)

    def _cooperative_move_shape(self, mapping: Mapping) -> tuple[int, int]:
        switches = count_switches(
            pre_failure_mapping=self.scenario.pre_failure_mapping,
            failover_mapping=self.failover_mapping,
            repaired_mapping=mapping,
            failure=self.scenario.failure,
        )
        moved_tenants = {
            int(tenant)
            for tenant, _rank in switches.extra_moved_ranks_vs_failover
        }
        return len(moved_tenants), int(switches.extra_vs_failover)

    def _direct_joint_protection_moves(
        self,
        mapping: Mapping,
        analysis,
        deadline: float,
        *,
        limit: int,
    ):
        tenants = list(self._tenant_order(mapping, analysis))
        if len(tenants) < 2:
            return

        protection = tuple(int(server) for server in self.scenario.global_protection_pool)
        failed_tenant = int(self.scenario.failure.tenant)
        failed_rank = infer_failed_rank(self.scenario.pre_failure_mapping, self.scenario.failure)
        moves_by_tenant: dict[int, list[tuple[int, int, float]]] = {}
        max_ranks = max(2, int(self.config.max_block_ranks) + 2)
        max_servers = max(2, int(self.config.block_extra_servers) + 2)

        for tenant in tenants:
            if time.time() >= deadline:
                return
            ranks = self._rank_order(int(tenant), analysis)[:max_ranks]
            if int(tenant) == failed_tenant and int(failed_rank) not in ranks:
                ranks = [int(failed_rank), *ranks[: max(0, max_ranks - 1)]]
            active_servers = self._active_servers(mapping)
            available = [
                int(server)
                for server in protection
                if int(server) in self.candidate_sets[int(tenant)]
                and int(server) not in active_servers
            ]
            available = self._ranked_available_servers(
                mapping,
                int(tenant),
                ranks,
                available,
                self._task_pair_prices(mapping, int(tenant), analysis),
                analysis,
            )[:max_servers]
            tenant_moves = []
            task_pair_prices = self._task_pair_prices(mapping, int(tenant), analysis)
            for rank in ranks:
                for server in available:
                    if int(mapping[int(tenant)][int(rank)]) == int(server):
                        continue
                    candidate = copy_mapping(mapping)
                    candidate[int(tenant)][int(rank)] = int(server)
                    cost = self._tenant_price_cost(
                        candidate,
                        int(tenant),
                        candidate[int(tenant)],
                        task_pair_prices,
                        analysis,
                    )
                    tenant_moves.append((int(rank), int(server), float(cost)))
            tenant_moves.sort(key=lambda item: (item[2], item[0], item[1]))
            if tenant_moves:
                moves_by_tenant[int(tenant)] = tenant_moves[: self.config.joint_candidates_per_tenant]

        active_tenants = sorted(moves_by_tenant)
        max_width = min(int(self.config.max_joint_tenants), len(active_tenants))
        yielded = 0
        for width in range(2, max_width + 1):
            for tenant_group in itertools.combinations(active_tenants, width):
                move_lists = [moves_by_tenant[int(tenant)] for tenant in tenant_group]
                combos = []
                for combo in itertools.product(*move_lists):
                    cost = sum(float(move[2]) for move in combo)
                    combos.append((cost, combo))
                combos.sort(key=lambda item: item[0])
                for _cost, combo in combos:
                    if time.time() >= deadline:
                        return
                    target_servers = [int(move[1]) for move in combo]
                    if len(set(target_servers)) != len(target_servers):
                        continue
                    candidate = copy_mapping(mapping)
                    source_parts = []
                    changed = False
                    for tenant, move in zip(tenant_group, combo):
                        rank, server, _move_cost = move
                        if int(candidate[int(tenant)][int(rank)]) != int(server):
                            changed = True
                        candidate[int(tenant)][int(rank)] = int(server)
                        source_parts.append(f"direct:t{tenant}:r{rank}:s{server}")
                    if not changed:
                        continue
                    yielded += 1
                    yield candidate, "direct_joint:" + "+".join(source_parts)
                    if yielded >= int(limit):
                        return

    def _protection_reassignment_moves(
        self,
        mapping: Mapping,
        analysis,
        deadline: float,
        *,
        limit: int,
    ):
        """Free a protection slot used by one extra move and reuse it elsewhere.

        In low-resource configurations the local seed can occupy nearly every
        protection server.  Cooperative repair then needs a structured way to
        trade a local extra move for a more valuable cross-tenant move without
        creating an intermediate infeasible mapping with duplicate active
        servers.
        """

        protection = set(int(server) for server in self.scenario.global_protection_pool)
        switches = count_switches(
            pre_failure_mapping=self.scenario.pre_failure_mapping,
            failover_mapping=self.failover_mapping,
            repaired_mapping=mapping,
            failure=self.scenario.failure,
        )
        occupied_extras = []
        for tenant, rank in switches.extra_moved_ranks_vs_failover:
            server = int(mapping[int(tenant)][int(rank)])
            if server in protection:
                occupied_extras.append((int(tenant), int(rank), int(server)))
        if not occupied_extras:
            return

        tenants = list(self._tenant_order(mapping, analysis))
        if len(tenants) < 2:
            return

        scored = []
        for release_tenant, release_rank, freed_server in occupied_extras:
            if time.time() >= deadline:
                return
            release_base = copy_mapping(mapping)
            release_base[int(release_tenant)][int(release_rank)] = int(
                self.failover_mapping[int(release_tenant)][int(release_rank)]
            )
            for target_tenant in tenants:
                target_tenant = int(target_tenant)
                if target_tenant == int(release_tenant):
                    continue
                if int(freed_server) not in self.candidate_sets[int(target_tenant)]:
                    continue
                ranks = self._rank_order(int(target_tenant), analysis)[
                    : max(2, int(self.config.max_block_ranks) + 2)
                ]
                task_pair_prices = self._task_pair_prices(
                    release_base,
                    int(target_tenant),
                    analysis,
                )
                for target_rank in ranks:
                    target_rank = int(target_rank)
                    if int(release_base[int(target_tenant)][int(target_rank)]) == int(freed_server):
                        continue
                    candidate = copy_mapping(release_base)
                    candidate[int(target_tenant)][int(target_rank)] = int(freed_server)
                    try:
                        check_repair_feasible(
                            self.scenario,
                            candidate,
                            self.failover_mapping,
                        )
                    except ValueError:
                        continue
                    cost = self._tenant_price_cost(
                        candidate,
                        int(target_tenant),
                        candidate[int(target_tenant)],
                        task_pair_prices,
                        analysis,
                    )
                    scored.append(
                        (
                            float(cost),
                            int(release_tenant),
                            int(release_rank),
                            int(target_tenant),
                            int(target_rank),
                            int(freed_server),
                            candidate,
                        )
                    )

        yielded = 0
        for (
            _cost,
            release_tenant,
            release_rank,
            target_tenant,
            target_rank,
            freed_server,
            candidate,
        ) in sorted(scored, key=lambda item: item[:6]):
            if time.time() >= deadline:
                return
            yielded += 1
            yield (
                candidate,
                "reassign_protection:"
                f"free:t{release_tenant}:r{release_rank}:"
                f"use:t{target_tenant}:r{target_rank}:s{freed_server}",
            )
            if yielded >= int(limit):
                return
