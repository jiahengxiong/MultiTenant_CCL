from __future__ import annotations

import time
from dataclasses import dataclass

from ..evaluator import RepairEvaluator
from ..heuristic import (
    FailureAwareRepairHeuristicBase,
    REPAIR_ALGORITHM_VERSION,
    RepairSearchConfig,
)
from ..models import Mapping, RepairObjective, RepairScenario
from ..protection import check_repair_feasible, copy_mapping, participating_tenants
from ..strategies import (
    COOPERATIVE_REPAIR,
    REPAIR_FAILED_SERVER_ONLY,
    TENANT_LOCAL_REPAIR,
    RepairStrategySpec,
)


@dataclass(frozen=True)
class EstimatorRepairCandidate:
    objective: RepairObjective
    mapping: Mapping
    source: str


class _UnifiedContentionGuidedRepairSearch(FailureAwareRepairHeuristicBase):
    """One master/subproblem search; strategy changes only candidate-set scope."""

    optimizer_name = "contention_guided_logic_based_repair_optimizer"

    def __init__(
        self,
        scenario: RepairScenario,
        evaluator: RepairEvaluator,
        strategy: RepairStrategySpec,
        *,
        config: RepairSearchConfig | None = None,
        reference_seeds: list[tuple[Mapping, str]] | None = None,
    ):
        super().__init__(
            scenario,
            evaluator,
            config=config,
            reference_seeds=reference_seeds,
        )
        self.strategy = strategy
        self.name = strategy.name

    def allowed_tenants(self) -> tuple[int, ...]:
        if self.strategy.name == REPAIR_FAILED_SERVER_ONLY.name:
            return (int(self.scenario.failure.tenant),)
        if self.strategy.name == TENANT_LOCAL_REPAIR.name:
            return (int(self.scenario.failure.tenant),)
        if self.strategy.name == COOPERATIVE_REPAIR.name:
            return participating_tenants(self.scenario)
        raise ValueError(f"unsupported repair strategy: {self.strategy.name}")

    def _candidate_scopes(self) -> tuple[tuple[str, tuple[int, ...] | None, bool], ...]:
        failed_tenant_scope = (int(self.scenario.failure.tenant),)
        if self.strategy.name == REPAIR_FAILED_SERVER_ONLY.name:
            return (("slot_assign_failover", failed_tenant_scope, False),)
        if self.strategy.name == TENANT_LOCAL_REPAIR.name:
            return (("slot_assign_local", failed_tenant_scope, False),)
        if self.strategy.name == COOPERATIVE_REPAIR.name:
            return (
                ("slot_assign_coordinate:local_scope", failed_tenant_scope, False),
                ("slot_assign_coordinate", None, True),
            )
        raise ValueError(f"unsupported repair strategy: {self.strategy.name}")

    def _initial_seed_entries(self, deadline: float):
        failover = copy_mapping(self.failover_mapping)
        seeds: list[tuple[Mapping, str]] = [(failover, "failover")]
        seen = {self._signature(failover)}

        def add_seed(mapping: Mapping, source: str) -> None:
            sig = self._signature(mapping)
            if sig in seen:
                return
            seen.add(sig)
            seeds.append((copy_mapping(mapping), source))

        for mapping, source in self.reference_seeds:
            if self.strategy.name == COOPERATIVE_REPAIR.name:
                add_seed(mapping, f"slot_assign_coordinate:reference:{source}")
            elif self.strategy.name == TENANT_LOCAL_REPAIR.name:
                add_seed(mapping, f"slot_assign_local:reference:{source}")
            else:
                add_seed(mapping, f"reference:{source}")

        analysis = self._safe_analysis(failover)

        if self.strategy.name == REPAIR_FAILED_SERVER_ONLY.name:
            for source_prefix, participants_override, is_coordinate in self._candidate_scopes():
                for candidate, source in self._slot_assignment_candidates(
                    analysis,
                    deadline,
                    limit=max(1, len(self.scenario.global_protection_pool)),
                    source_prefix=source_prefix,
                    participants_override=participants_override,
                ) or ():
                    if time.time() >= deadline:
                        break
                    add_seed(candidate, source)
        else:
            for candidate, source in self._policy_master_subproblem_candidates(
                analysis,
                deadline,
            ):
                if time.time() >= deadline:
                    break
                add_seed(candidate, source)

        entries = []
        for mapping, source in seeds:
            try:
                check_repair_feasible(self.scenario, mapping, self.failover_mapping)
            except ValueError:
                continue
            entries.append((self.evaluator.estimate(mapping), mapping, source))
        return entries

    def _policy_master_subproblem_candidates(self, analysis, deadline: float):
        protection = tuple(sorted(int(server) for server in self.scenario.global_protection_pool))
        failed_tenant = int(self.scenario.failure.tenant)
        failed_rank = int(
            self.scenario.failure.failed_rank
            if self.scenario.failure.failed_rank is not None
            else next(
                rank
                for rank, server in self.scenario.pre_failure_mapping[failed_tenant].items()
                if int(server) == int(self.scenario.failure.failed_server)
            )
        )
        failed_servers = [
            int(server)
            for server in protection
            if int(server) in self.candidate_sets[int(failed_tenant)]
        ]
        if not failed_servers:
            return

        participants = tuple(int(tenant) for tenant in self.allowed_tenants())
        base_template = copy_mapping(self.scenario.pre_failure_mapping)
        is_coordinate = self.strategy.name == COOPERATIVE_REPAIR.name
        max_optional = max(0, min(len(protection) - 1, len(self._policy_optional_rank_order(
            analysis,
            participants=participants,
            failed_tenant=failed_tenant,
            failed_rank=failed_rank,
        ))))
        beam_width = max(2, min(max(2, self.config.beam_width), max(2, len(protection))))
        action_width = max(
            beam_width,
            min(max(beam_width, self.config.max_candidates_per_tenant), max(1, max_optional)),
        )
        max_rounds = max(1, min(max_optional, max(1, int(self.config.max_rounds))))
        max_evaluations = self._master_iteration_budget(is_coordinate=is_coordinate)

        def evaluate(selection: tuple[tuple[int, int], ...], source: str):
            if time.time() >= deadline:
                return None
            normalized_selection = tuple(sorted((int(tenant), int(rank)) for tenant, rank in selection))
            sub_states = self._logic_benders_assignment_subproblem(
                base_template,
                analysis,
                deadline,
                protection=protection,
                failed_tenant=failed_tenant,
                failed_rank=failed_rank,
                failed_servers=failed_servers,
                healthy_selection=normalized_selection,
                is_coordinate=is_coordinate,
                state_limit=1,
            )
            self.logic_benders_master_selections += 1
            self.logic_benders_subproblem_states += len(sub_states)
            if not sub_states:
                return {
                    "selection": normalized_selection,
                    "state": None,
                    "reward": float("-inf"),
                    "source": source,
                    "analysis": analysis,
                }
            best_state = sorted(
                sub_states,
                key=lambda state: self._entry_sort_key((state[0], state[1], "candidate_set_search")),
            )[0]
            obj = best_state[0]
            reward = -(
                float(obj.avg_jct)
                + 0.05 * float(obj.makespan)
                + 0.001 * float(obj.extra_switches)
            )
            try:
                next_analysis = self._safe_analysis(best_state[1])
            except Exception:
                next_analysis = analysis
            feedback = self._candidate_set_feedback_from_analysis(
                next_analysis,
                best_state[1],
                normalized_selection,
                participants=participants,
                failed_tenant=failed_tenant,
                failed_rank=failed_rank,
                is_coordinate=is_coordinate,
            )
            return {
                "selection": normalized_selection,
                "state": best_state,
                "reward": float(reward),
                "source": source,
                "analysis": next_analysis,
                "mapping": best_state[1],
                "objective": obj,
                "feedback": feedback,
            }

        optional_order = self._policy_optional_rank_order(
            analysis,
            participants=participants,
            failed_tenant=failed_tenant,
            failed_rank=failed_rank,
        )
        initial_selections: list[tuple[tuple[int, int], ...]] = [tuple()]
        ranked_keys = [(int(tenant), int(rank)) for tenant, rank, _score in optional_order]
        for size in range(1, min(len(ranked_keys), len(protection) - 1, beam_width) + 1):
            initial_selections.append(tuple(ranked_keys[:size]))

        visited: set[tuple[tuple[int, int], ...]] = set()
        evaluated = []
        beam = []
        for selection in initial_selections:
            normalized = tuple(sorted(selection))
            if normalized in visited:
                continue
            visited.add(normalized)
            item = evaluate(normalized, "initial")
            if item is not None:
                evaluated.append(item)
                beam.append(item)

        for round_index in range(max_rounds):
            if time.time() >= deadline or not beam:
                break
            actions = []
            for item in sorted(beam, key=lambda row: -float(row["reward"]))[:beam_width]:
                actions.extend(
                    self._candidate_set_actions(
                        item["selection"],
                        item["analysis"],
                        item.get("feedback") or {},
                        participants=participants,
                        failed_tenant=failed_tenant,
                        failed_rank=failed_rank,
                        action_width=action_width,
                    )
                )
            next_items = []
            for action_name, selection in actions:
                if time.time() >= deadline:
                    break
                if len(evaluated) >= max_evaluations:
                    break
                normalized = tuple(sorted(selection))
                if len(normalized) > len(protection) - 1:
                    continue
                if normalized in visited:
                    continue
                visited.add(normalized)
                item = evaluate(normalized, f"round{round_index}:{action_name}")
                if item is None:
                    continue
                evaluated.append(item)
                next_items.append(item)
            beam = sorted(
                [*beam, *next_items],
                key=lambda row: (
                    -float(row["reward"]),
                    len(row["selection"]),
                    row["selection"],
                ),
            )[:beam_width]
            if len(evaluated) >= max_evaluations:
                break

        self.contention_feedback_records.extend(
            [
                {
                    "recovery_candidate_set": [
                        {"tenant": int(failed_tenant), "rank": int(failed_rank), "role": "failed"},
                        *(
                            {"tenant": int(tenant), "rank": int(rank), "role": "healthy"}
                            for tenant, rank in item["selection"]
                        ),
                    ],
                    "optional_healthy_candidate_set": [
                        {"tenant": int(tenant), "rank": int(rank)}
                        for tenant, rank in item["selection"]
                    ],
                    "reward": float(item["reward"]),
                    "source": str(item["source"]),
                    "avg_jct": (
                        None
                        if item.get("objective") is None
                        else float(item["objective"].avg_jct)
                    ),
                    "makespan": (
                        None
                        if item.get("objective") is None
                        else float(item["objective"].makespan)
                    ),
                    "switch_count": (
                        None
                        if item.get("objective") is None
                        else int(item["objective"].extra_switches)
                    ),
                    "hotspot_links": (item.get("feedback") or {}).get("hotspot_links", []),
                    "hotspot_servers": (item.get("feedback") or {}).get("hotspot_servers", []),
                    "critical_path_tasks": (item.get("feedback") or {}).get("critical_path_tasks", []),
                    "contention_prices": (item.get("feedback") or {}).get("contention_prices", []),
                    "subproblem_states": 0 if item["state"] is None else 1,
                }
                for item in sorted(evaluated, key=lambda row: -float(row["reward"]))[:12]
            ]
        )
        emitted = 0
        source_prefix = (
            "slot_assign_coordinate:policy_master_subproblem"
            if self.strategy.name == COOPERATIVE_REPAIR.name
            else "slot_assign_local:policy_master_subproblem"
        )
        output_items = [
            item
            for item in evaluated
            if item["state"] is not None
        ]
        for item in sorted(output_items, key=lambda row: -float(row["reward"])):
            obj, mapping, moves, _used, _moved = item["state"]
            source_moves = "+".join(
                f"t{tenant}:r{rank}:s{server}" for tenant, rank, server in moves
            )
            yield mapping, f"{source_prefix}:{item['source']}:{source_moves}"
            emitted += 1
            if emitted >= self._candidate_output_budget(
                is_coordinate=is_coordinate
            ):
                return

    def _candidate_set_actions(
        self,
        selection: tuple[tuple[int, int], ...],
        analysis,
        feedback: dict[str, object],
        *,
        participants: tuple[int, ...],
        failed_tenant: int,
        failed_rank: int,
        action_width: int,
    ) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
        selected = tuple(sorted((int(tenant), int(rank)) for tenant, rank in selection))
        selected_set = set(selected)
        hotspot_order = [
            (int(row["tenant"]), int(row["rank"]))
            for row in feedback.get("culprit_ranks", [])
            if int(row["tenant"]) in set(int(tenant) for tenant in participants)
        ]

        structural_order = [
            (int(tenant), int(rank))
            for tenant, rank, _score in self._policy_optional_rank_order(
                analysis,
                participants=participants,
                failed_tenant=failed_tenant,
                failed_rank=failed_rank,
            )
        ]
        optional = []
        for candidate in (*hotspot_order, *structural_order):
            if candidate in optional:
                continue
            optional.append(candidate)
            if len(optional) >= max(1, int(action_width)):
                break
        actions: list[tuple[str, tuple[tuple[int, int], ...]]] = []

        for candidate in optional:
            if candidate in selected_set:
                continue
            actions.append((f"Add(t{candidate[0]}r{candidate[1]})", tuple(sorted((*selected, candidate)))))

        for old in selected:
            reduced = tuple(item for item in selected if item != old)
            actions.append((f"Remove(t{old[0]}r{old[1]})", reduced))

        for old in selected:
            for candidate in optional:
                if candidate in selected_set:
                    continue
                replaced = tuple(sorted(candidate if item == old else item for item in selected))
                actions.append((
                    f"Replace(t{old[0]}r{old[1]},t{candidate[0]}r{candidate[1]})",
                    replaced,
                ))
                break

        deduped = []
        seen = set()
        for name, candidate_set in actions:
            normalized = tuple(sorted(candidate_set))
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append((name, normalized))
        return deduped[: max(1, int(action_width))]

    def _candidate_set_feedback_from_analysis(
        self,
        analysis,
        mapping: Mapping,
        selection: tuple[tuple[int, int], ...],
        *,
        participants: tuple[int, ...],
        failed_tenant: int,
        failed_rank: int,
        is_coordinate: bool,
    ) -> dict[str, object]:
        selected = set((int(tenant), int(rank)) for tenant, rank in selection)
        fixed = {(int(failed_tenant), int(failed_rank)), *selected}
        participant_set = set(int(tenant) for tenant in participants)
        protection = set(int(server) for server in self.scenario.global_protection_pool)

        hotspot_rank_scores: dict[tuple[int, int], float] = {}
        hotspot_tasks: list[tuple[int, int]] = []
        hotspot_resources = []
        hotspot_servers: set[int] = set()

        for cluster in list(getattr(analysis, "contention_clusters", []) or [])[:8]:
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
                if key is not None:
                    hotspot_rank_scores[key] = hotspot_rank_scores.get(key, 0.0) + max(excess, 1.0)

        rank_pressure = getattr(analysis, "rank_pressure", {}) or {}
        for key, pressure in sorted(
            rank_pressure.items(),
            key=lambda item: (-float(item[1]), self._normalize_rank_key(item[0]) or (10**9, 10**9)),
        )[:12]:
            normalized = self._normalize_rank_key(key)
            if normalized is not None:
                hotspot_rank_scores[normalized] = hotspot_rank_scores.get(normalized, 0.0) + float(pressure)

        culprit_ranks = []
        for (tenant, rank), score in sorted(
            hotspot_rank_scores.items(),
            key=lambda item: (-float(item[1]), item[0]),
        ):
            tenant = int(tenant)
            rank = int(rank)
            if (tenant, rank) in fixed:
                continue
            if tenant not in participant_set:
                continue
            if not is_coordinate and tenant != int(failed_tenant):
                continue
            if tenant not in mapping or rank not in mapping[tenant]:
                continue
            if int(mapping[tenant][rank]) in protection:
                continue
            hotspot_servers.add(int(mapping[tenant][rank]))
            culprit_ranks.append({"tenant": tenant, "rank": rank, "weight": float(score)})
            if len(culprit_ranks) >= 8:
                break

        critical_tasks = [
            {"tenant": int(tenant), "task": int(task)}
            for tenant, task in sorted(set(
                task
                for task in (
                    self._normalize_task_key(task)
                    for task in getattr(analysis, "critical_tasks", set())
                )
                if task is not None
            ))[:12]
        ]
        if not critical_tasks:
            critical_tasks = [
                {"tenant": int(tenant), "task": int(task)}
                for tenant, task in sorted(set(hotspot_tasks))[:12]
            ]

        hotspot_links = []
        seen_links = set()
        for resource in hotspot_resources or self._hotspot_resources_from_slot_prices(analysis, limit=8):
            if not self._is_hotspot_link_resource(resource):
                continue
            jsonable = self._jsonable_resource(resource)
            key = repr(jsonable)
            if key in seen_links:
                continue
            seen_links.add(key)
            hotspot_links.append(jsonable)
            if len(hotspot_links) >= 8:
                break

        contention_prices = []
        for price_state in list(getattr(analysis, "slot_prices", []) or [])[:2]:
            if not isinstance(price_state, dict):
                continue
            for resource_type in ("links", "senders", "receivers"):
                prices = price_state.get(resource_type, {}) or {}
                top_prices = sorted(prices.items(), key=lambda item: -float(item[1]))[:3]
                for resource_id, price in top_prices:
                    contention_prices.append({
                        "resource_type": resource_type,
                        "resource": self._jsonable_resource(resource_id),
                        "price": float(price),
                    })
                    if len(contention_prices) >= 8:
                        break
                if len(contention_prices) >= 8:
                    break
            if len(contention_prices) >= 8:
                break

        return {
            "culprit_ranks": culprit_ranks,
            "hotspot_links": hotspot_links,
            "hotspot_servers": sorted(int(server) for server in hotspot_servers)[:12],
            "critical_path_tasks": critical_tasks,
            "contention_prices": contention_prices,
        }

    def _policy_optional_rank_order(
        self,
        analysis,
        *,
        participants: tuple[int, ...],
        failed_tenant: int,
        failed_rank: int,
    ) -> list[tuple[int, int, float]]:
        protection = set(int(server) for server in self.scenario.global_protection_pool)
        rows: list[tuple[int, int, float]] = []
        for tenant in self._tenant_order(self.failover_mapping, analysis):
            if int(tenant) not in participants:
                continue
            for rank in self._impact_rank_order(self.failover_mapping, int(tenant), analysis):
                rank = int(rank)
                if int(tenant) == int(failed_tenant) and rank == int(failed_rank):
                    continue
                if int(self.failover_mapping[int(tenant)][rank]) in protection:
                    continue
                impact = self._rank_structural_impact(
                    self.failover_mapping,
                    int(tenant),
                    rank,
                    analysis,
                )
                rows.append((int(tenant), rank, float(impact)))
        if self.strategy.name == TENANT_LOCAL_REPAIR.name:
            rows = [row for row in rows if int(row[0]) == int(failed_tenant)]
        return sorted(rows, key=lambda row: (-float(row[2]), int(row[0]), int(row[1])))

    def _policy_candidate_sets(
        self,
        optional_order: list[tuple[int, int, float]],
    ) -> list[tuple[tuple[int, int], ...]]:
        protection_slots = len(self.scenario.global_protection_pool)
        max_optional = max(0, min(protection_slots - 1, len(optional_order)))
        if max_optional <= 0:
            return [tuple()]
        budget = self._candidate_output_budget(
            is_coordinate=self.strategy.name == COOPERATIVE_REPAIR.name
        )
        sets: list[tuple[tuple[int, int], ...]] = [tuple()]
        ranked_keys = [(int(tenant), int(rank)) for tenant, rank, _score in optional_order]
        for size in range(1, max_optional + 1):
            sets.append(tuple(ranked_keys[:size]))
            if len(sets) >= budget:
                return sets
        if self.strategy.name == COOPERATIVE_REPAIR.name:
            by_tenant: dict[int, list[tuple[int, int]]] = {}
            for tenant, rank in ranked_keys:
                by_tenant.setdefault(int(tenant), []).append((int(tenant), int(rank)))
            diversified: list[tuple[int, int]] = []
            depth = 0
            while len(diversified) < max_optional:
                progressed = False
                for tenant in sorted(by_tenant):
                    bucket = by_tenant[tenant]
                    if depth < len(bucket):
                        diversified.append(bucket[depth])
                        progressed = True
                        if len(diversified) >= max_optional:
                            break
                if not progressed:
                    break
                depth += 1
            for size in range(1, len(diversified) + 1):
                candidate = tuple(diversified[:size])
                if candidate not in sets:
                    sets.append(candidate)
                if len(sets) >= budget:
                    return sets
        return sets[:budget]

    def _direct_healthy_move_candidates(self, analysis, deadline: float):
        if self.strategy.name == REPAIR_FAILED_SERVER_ONLY.name:
            return
        protection = tuple(sorted(int(server) for server in self.scenario.global_protection_pool))
        base = copy_mapping(self.failover_mapping)
        active = self._active_servers(base)
        free_protection = [
            int(server)
            for server in protection
            if int(server) not in active
        ]
        if not free_protection:
            return

        failed_tenant = int(self.scenario.failure.tenant)
        failed_rank = int(
            self.scenario.failure.failed_rank
            if self.scenario.failure.failed_rank is not None
            else next(
                rank
                for rank, server in self.scenario.pre_failure_mapping[failed_tenant].items()
                if int(server) == int(self.scenario.failure.failed_server)
            )
        )
        participants = tuple(int(tenant) for tenant in self.allowed_tenants())
        rank_cap = 10 if self.strategy.name == TENANT_LOCAL_REPAIR.name else 16
        slot_cap = 4 if self.strategy.name == TENANT_LOCAL_REPAIR.name else 5
        candidate_rows: list[tuple[float, int, int]] = []
        for tenant in self._tenant_order(base, analysis):
            if int(tenant) not in participants:
                continue
            for rank in self._impact_rank_order(base, int(tenant), analysis):
                if time.time() >= deadline:
                    return
                rank = int(rank)
                if int(tenant) == failed_tenant and rank == failed_rank:
                    continue
                if int(base[int(tenant)][rank]) in protection:
                    continue
                impact = self._rank_structural_impact(base, int(tenant), rank, analysis)
                candidate_rows.append((float(impact), int(tenant), rank))
                if len(candidate_rows) >= rank_cap:
                    break
            if len(candidate_rows) >= rank_cap:
                break

        scored = []
        for _impact, tenant, rank in candidate_rows:
            if time.time() >= deadline:
                break
            task_pair_prices = self._task_pair_prices(base, int(tenant), analysis)
            ranked_slots = self._ranked_available_servers(
                base,
                int(tenant),
                [int(rank)],
                list(free_protection),
                task_pair_prices,
                analysis,
            )[:slot_cap]
            for server in ranked_slots:
                candidate = copy_mapping(base)
                candidate[int(tenant)][int(rank)] = int(server)
                try:
                    check_repair_feasible(self.scenario, candidate, self.failover_mapping)
                except ValueError:
                    continue
                obj = self.evaluator.estimate(candidate)
                scored.append((
                    self._entry_sort_key((obj, candidate, "direct_healthy_move")),
                    candidate,
                    f"direct_healthy_move:t{tenant}:r{rank}:s{int(server)}",
                ))

        prefix = (
            "slot_assign_coordinate:policy_direct"
            if self.strategy.name == COOPERATIVE_REPAIR.name
            else "slot_assign_local:policy_direct"
        )
        for _key, candidate, source in sorted(scored, key=lambda item: item[0])[: self._simulator_pool_cap()]:
            source = source.replace("direct_healthy_move:", f"{prefix}:")
            yield candidate, source

    def _postprocess_simulator_selected_mapping(
        self,
        best_obj,
        best_mapping: Mapping,
        best_source: str,
        deadline: float,
    ):
        del deadline
        return best_obj, best_mapping, best_source, 0


class ContentionGuidedRepairOptimizer:
    """Unified optimizer for failover, local, and coordinate repair modes.

    The strategy only changes the recovery-candidate-set scope.  The optimizer
    itself is the same contention-guided master/subproblem procedure.
    """

    name = "contention_guided_logic_based_repair_optimizer"

    def __init__(
        self,
        scenario: RepairScenario,
        evaluator: RepairEvaluator,
        strategy: RepairStrategySpec,
        *,
        config: RepairSearchConfig | None = None,
        reference_seeds: list[tuple[Mapping, str]] | None = None,
    ):
        self.scenario = scenario
        self.evaluator = evaluator
        self.strategy = strategy
        self.config = config or RepairSearchConfig()
        self.reference_seeds = reference_seeds or []

    def _search(self):
        return _UnifiedContentionGuidedRepairSearch(
            self.scenario,
            self.evaluator,
            self.strategy,
            config=self.config,
            reference_seeds=self.reference_seeds,
        )

    def solve(self, *, time_limit: float | None = None):
        result = self._search().solve(time_limit=time_limit)
        result.metadata.setdefault("optimizer", self.name)
        result.metadata.setdefault("optimizer_impl", _UnifiedContentionGuidedRepairSearch.optimizer_name)
        return result

    def estimator_candidates(self) -> list[EstimatorRepairCandidate]:
        entries = self._search()._initial_seed_entries(float("inf"))
        return [
            EstimatorRepairCandidate(
                objective=objective,
                mapping=mapping,
                source=str(source),
            )
            for objective, mapping, source in entries
        ]


def solve_contention_guided_repair(
    scenario: RepairScenario,
    evaluator: RepairEvaluator,
    strategy: RepairStrategySpec,
    *,
    config: RepairSearchConfig | None = None,
    reference_seeds: list[tuple[Mapping, str]] | None = None,
    time_limit: float | None = None,
):
    return ContentionGuidedRepairOptimizer(
        scenario,
        evaluator,
        strategy,
        config=config,
        reference_seeds=reference_seeds,
    ).solve(time_limit=time_limit)


def estimator_repair_candidates(
    scenario: RepairScenario,
    evaluator: RepairEvaluator,
    strategy: RepairStrategySpec,
    *,
    config: RepairSearchConfig | None = None,
    reference_seeds: list[tuple[Mapping, str]] | None = None,
) -> list[EstimatorRepairCandidate]:
    return ContentionGuidedRepairOptimizer(
        scenario,
        evaluator,
        strategy,
        config=config,
        reference_seeds=reference_seeds,
    ).estimator_candidates()


__all__ = [
    "ContentionGuidedRepairOptimizer",
    "EstimatorRepairCandidate",
    "REPAIR_ALGORITHM_VERSION",
    "RepairSearchConfig",
    "solve_contention_guided_repair",
    "estimator_repair_candidates",
]
