from __future__ import annotations

import math
import time

from .mapping_ilp import MappingHeuristicSolver
from .mapping_local_search import MappingLocalSearchHeuristicSolver


class MappingMultiNeighborhoodHeuristicSolver(MappingLocalSearchHeuristicSolver):
    """Surrogate-guided mapping with local and price-BnB neighborhoods.

    Each iteration computes congestion prices, asks multiple structured
    neighborhoods for candidate mappings, and accepts only the candidate that
    improves the whole-mapping surrogate objective. This keeps a single
    optimization objective while allowing both cheap local moves and stronger
    price-guided BnB jumps.
    """

    def __init__(
        self,
        *args,
        bnb_candidate_time_limit=1.0,
        max_bnb_tenants_per_round=2,
        max_joint_tenants_per_round=3,
        beam_width=4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bnb_candidate_time_limit = float(bnb_candidate_time_limit)
        requested_bnb_tenants = int(max_bnb_tenants_per_round)
        adaptive_bnb_tenants = max(3, int(math.ceil(0.5 * max(len(self.tenants), 1))))
        self.max_bnb_tenants_per_round = max(requested_bnb_tenants, adaptive_bnb_tenants)
        self.max_joint_tenants_per_round = max(3, int(max_joint_tenants_per_round))
        self.beam_width = max(1, int(beam_width))
        self.last_move_source = None
        self.move_source_counts = {"local": 0, "bnb": 0, "joint": 0}

    def _build_price_bnb_candidate(self, base_mapping, tenant, epoch_prices, deadline):
        if time.time() >= deadline:
            return base_mapping
        candidate_deadline = min(deadline, time.time() + self.bnb_candidate_time_limit)
        bnb_solver = MappingHeuristicSolver(
            self.datacenter,
            tenant_mapping=base_mapping,
            tenant_flows=self.tenant_flows,
            verbose=False,
            name=f"{self.model_name}_bnb_candidate",
            collective=self.collective,
            single_flow_size=self.single_flow_size,
            tenant_collective_specs=self.tenant_collective_specs,
            tenant_collective_programs=self.tenant_collective_programs,
            tenant_start_times=self.tenant_start_times,
            stage_flows=self.stage_flows,
            fairness_lambda=self.fairness_lambda,
            fairness_iterations=self.fairness_iterations,
            fairness_grouping=self.fairness_grouping,
            slot_duration=self.slot_duration_override,
            horizon_slots=self.horizon_slots_override,
            validate_with_simulator=False,
            path_table=self.path_table,
            extra_seed_mappings=self.extra_seed_mappings,
        )
        return bnb_solver._optimize_tenant_block_with_prices(
            base_mapping,
            tenant,
            epoch_prices,
            candidate_deadline,
        )[0]

    def _best_neighborhood_candidate(
        self,
        base_mapping,
        base_score,
        tenant,
        epoch_prices,
        deadline,
        *,
        include_local=True,
        allow_bnb=False,
    ):
        candidates = []

        if include_local:
            local_mapping, _ = super()._optimize_tenant_block_with_prices(
                base_mapping,
                tenant,
                epoch_prices,
                deadline,
            )
            candidates.append(("local", local_mapping))

        if allow_bnb:
            bnb_mapping = self._build_price_bnb_candidate(
                base_mapping,
                tenant,
                epoch_prices,
                deadline,
            )
            candidates.append(("bnb", bnb_mapping))

        best_source = None
        best_mapping = base_mapping
        best_score = base_score
        for source, candidate_mapping in candidates:
            if time.time() >= deadline:
                break
            if self._mapping_signature(candidate_mapping) == self._mapping_signature(base_mapping):
                continue
            candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
            if self._is_better_objective(candidate_score, best_score):
                best_source = source
                best_mapping = candidate_mapping
                best_score = candidate_score

        return best_source, best_mapping, best_score

    def _joint_pair_candidate(
        self,
        base_mapping,
        tenant_a,
        tenant_b,
        epoch_prices,
        deadline,
    ):
        candidates_a = []
        candidates_b = []

        local_mapping_a, _ = super()._optimize_tenant_block_with_prices(
            base_mapping,
            tenant_a,
            epoch_prices,
            deadline,
        )
        if self._mapping_signature(local_mapping_a) != self._mapping_signature(base_mapping):
            candidates_a.append(local_mapping_a)

        local_mapping_b, _ = super()._optimize_tenant_block_with_prices(
            base_mapping,
            tenant_b,
            epoch_prices,
            deadline,
        )
        if self._mapping_signature(local_mapping_b) != self._mapping_signature(base_mapping):
            candidates_b.append(local_mapping_b)

        if self.max_bnb_tenants_per_round > 0:
            bnb_mapping_a = self._build_price_bnb_candidate(
                base_mapping,
                tenant_a,
                epoch_prices,
                deadline,
            )
            if self._mapping_signature(bnb_mapping_a) != self._mapping_signature(base_mapping):
                candidates_a.append(bnb_mapping_a)

            bnb_mapping_b = self._build_price_bnb_candidate(
                base_mapping,
                tenant_b,
                epoch_prices,
                deadline,
            )
            if self._mapping_signature(bnb_mapping_b) != self._mapping_signature(base_mapping):
                candidates_b.append(bnb_mapping_b)

        if not candidates_a or not candidates_b:
            return None

        seen = set()
        joint_candidates = []
        for candidate_a in candidates_a:
            for candidate_b in candidates_b:
                joint_mapping = {
                    tenant: dict(rank_to_server)
                    for tenant, rank_to_server in base_mapping.items()
                }
                joint_mapping[tenant_a] = dict(candidate_a[tenant_a])
                joint_mapping[tenant_b] = dict(candidate_b[tenant_b])
                signature = self._mapping_signature(joint_mapping)
                if signature in seen:
                    continue
                seen.add(signature)
                joint_candidates.append(joint_mapping)

        return joint_candidates

    def _dedupe_ranked_candidates(self, candidates):
        ranked: list[tuple[tuple[float, float], dict[int, dict[int, int]], str | None]] = []
        seen = set()
        for score, mapping, source in candidates:
            signature = self._mapping_signature(mapping)
            if signature in seen:
                continue
            seen.add(signature)
            ranked.append((score, mapping, source))
        ranked.sort(key=lambda entry: self._objective_sort_key(entry[0]))
        return ranked[: self.beam_width]

    def _surrogate_local_repair(self, best_mapping, best_score, deadline, tenant_order=None, max_passes=2):
        """Greedily apply local remaps that improve the global surrogate.

        Beam search keeps only a small frontier. A mapping can become the best
        solution late in the search, leaving no full round to exploit its cheap
        one-tenant local improvements. This pass recomputes prices at the final
        incumbent and accepts only moves that improve the same global surrogate.
        """
        for _pass_idx in range(max_passes):
            if time.time() >= deadline:
                break

            improved = False
            _, _, epoch_prices = self._compute_epoch_link_prices(best_mapping)
            current_tenant_order = list(tenant_order) if tenant_order is not None else sorted(
                self.tenants,
                key=lambda tenant: (
                    -self.tenant_pressure.get(tenant, 0.0),
                    -self.tenant_peak_load.get(tenant, 0.0),
                    tenant,
                ),
            )

            for tenant in current_tenant_order:
                if time.time() >= deadline:
                    break
                source, candidate_mapping, candidate_score = self._best_neighborhood_candidate(
                    best_mapping,
                    best_score,
                    tenant,
                    epoch_prices,
                    deadline,
                    include_local=True,
                    allow_bnb=False,
                )
                if source is None:
                    continue

                best_mapping = candidate_mapping
                best_score = candidate_score
                improved = True
                self.last_move_source = "local_repair"
                self.move_source_counts["local"] = self.move_source_counts.get("local", 0) + 1
                self._register_surrogate_candidate(best_mapping, best_score)

                # Accepted moves change pressure, so refresh prices before the
                # next tenant instead of continuing with stale marginal costs.
                _, _, epoch_prices = self._compute_epoch_link_prices(best_mapping)

            if not improved:
                break

        return best_mapping, best_score

    def _build_initial_local_descent_seeds(self, deadline):
        """Build non-default seeds by locally improving the initial mapping."""
        base_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in self.initial_tenant_mapping.items()
        }
        base_score = self._evaluate_surrogate_mapping(base_mapping)
        _, _, _epoch_prices = self._compute_epoch_link_prices(base_mapping)
        pressure_desc = sorted(
            self.tenants,
            key=lambda tenant: (
                -self.tenant_pressure.get(tenant, 0.0),
                -self.tenant_peak_load.get(tenant, 0.0),
                tenant,
            ),
        )
        pressure_asc = list(reversed(pressure_desc))
        tenant_asc = list(self.tenants)
        tenant_desc = list(reversed(tenant_asc))

        if len(self.tenants) >= 6:
            tenant_orders = (pressure_asc, pressure_desc, tenant_asc, tenant_desc)
            max_passes = 3
        else:
            tenant_orders = (pressure_asc,)
            max_passes = 2

        seeds = []
        seen = {self._mapping_signature(base_mapping)}
        for tenant_order in tenant_orders:
            if time.time() >= deadline:
                break
            seed_mapping = {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in base_mapping.items()
            }
            seed_mapping, seed_score = self._surrogate_local_repair(
                seed_mapping,
                base_score,
                deadline,
                tenant_order=tenant_order,
                max_passes=max_passes,
            )
            signature = self._mapping_signature(seed_mapping)
            if signature in seen:
                continue
            seen.add(signature)
            seeds.append((seed_score, seed_mapping))
        return seeds

    def _build_initial_local_candidate_seeds(self, deadline):
        """Expose one-tenant local improvements from the initial mapping."""
        base_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in self.initial_tenant_mapping.items()
        }
        base_score = self._evaluate_surrogate_mapping(base_mapping)
        _, _, epoch_prices = self._compute_epoch_link_prices(base_mapping)
        tenant_order = sorted(
            self.tenants,
            key=lambda tenant: (
                -self.tenant_pressure.get(tenant, 0.0),
                -self.tenant_peak_load.get(tenant, 0.0),
                tenant,
            ),
        )

        candidates = []
        for tenant in tenant_order:
            if time.time() >= deadline:
                break
            source, candidate_mapping, candidate_score = self._best_neighborhood_candidate(
                base_mapping,
                base_score,
                tenant,
                epoch_prices,
                deadline,
                include_local=True,
                allow_bnb=False,
            )
            if source is None:
                continue
            candidates.append((candidate_score, candidate_mapping, "initial_local"))
        return candidates

    def _select_price_stable_candidate(self, best_mapping, best_score, deadline):
        """Use resource price as a secondary guard within estimator tolerance."""
        if self.extra_seed_mappings or not self._surrogate_candidates or time.time() >= deadline:
            return best_mapping, best_score

        best_makespan, best_avg_jct = best_score
        avg_tolerance = max(1e-9, abs(float(best_avg_jct)) * 0.02)
        makespan_tolerance = max(1e-9, abs(float(best_makespan)) * 0.05)

        selected_mapping = best_mapping
        selected_score = best_score
        selected_price = self._mapping_induced_price_cost(best_mapping)

        for candidate_score, _signature, candidate_mapping in self._surrogate_candidates:
            if time.time() >= deadline:
                break
            candidate_makespan, candidate_avg_jct = candidate_score
            if float(candidate_avg_jct) > float(best_avg_jct) + avg_tolerance:
                continue
            if float(candidate_makespan) > float(best_makespan) + makespan_tolerance:
                continue
            candidate_price = self._mapping_induced_price_cost(candidate_mapping)
            if candidate_price < selected_price - 1e-9:
                selected_mapping = {
                    tenant: dict(rank_to_server)
                    for tenant, rank_to_server in candidate_mapping.items()
                }
                selected_score = candidate_score
                selected_price = candidate_price

        return selected_mapping, selected_score

    def solve(self, time_limit=None):
        start_time = time.time()
        max_ranks_per_tenant = max((len(ranks) for ranks in self.rank_orders.values()), default=0)
        if max_ranks_per_tenant < 16:
            default_budget = 30.0
        elif len(self.tenants) < 6:
            default_budget = 60.0
        else:
            default_budget = 120.0
        total_budget = default_budget if time_limit is None else float(time_limit)
        deadline = start_time + total_budget
        repair_budget = min(max(total_budget * 0.15, 5.0), 20.0)
        search_deadline = max(start_time, deadline - repair_budget)
        final_selection_deadline = max(start_time, deadline - min(2.0, total_budget * 0.05))

        best_mapping = None
        best_score = (float("inf"), float("inf"))
        initial_candidates = []

        if time.time() < search_deadline:
            for candidate_score, seed_mapping in self._build_initial_local_descent_seeds(search_deadline):
                initial_candidates.append((candidate_score, seed_mapping, "initial_local"))
                self._register_surrogate_candidate(seed_mapping, candidate_score)
                if self._is_better_objective(candidate_score, best_score):
                    best_mapping = seed_mapping
                    best_score = candidate_score

        if time.time() < search_deadline:
            for candidate_score, seed_mapping, source in self._build_initial_local_candidate_seeds(search_deadline):
                initial_candidates.append((candidate_score, seed_mapping, source))
                self._register_surrogate_candidate(seed_mapping, candidate_score)
                if self._is_better_objective(candidate_score, best_score):
                    best_mapping = seed_mapping
                    best_score = candidate_score

        for seed_mapping in self._seed_mappings():
            if time.time() >= search_deadline:
                break
            candidate_score = self._evaluate_surrogate_mapping(seed_mapping)
            initial_candidates.append((candidate_score, seed_mapping, None))
            self._register_surrogate_candidate(seed_mapping, candidate_score)
            if self._is_better_objective(candidate_score, best_score):
                best_mapping = seed_mapping
                best_score = candidate_score

        if best_mapping is None:
            best_mapping = {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in self.initial_tenant_mapping.items()
            }
            best_score = self._evaluate_surrogate_mapping(best_mapping)
            initial_candidates = [(best_score, best_mapping, None)]

        beam = self._dedupe_ranked_candidates(initial_candidates)
        if not beam:
            beam = [(best_score, best_mapping, None)]

        round_idx = 0
        while round_idx < self.max_price_rounds and time.time() < search_deadline:
            round_idx += 1
            improved = False
            next_candidates = list(beam)
            previous_beam_signatures = [
                self._mapping_signature(mapping)
                for _score, mapping, _source in beam
            ]

            for beam_score, beam_mapping, _beam_source in beam:
                if time.time() >= search_deadline:
                    break

                _, _, epoch_prices = self._compute_epoch_link_prices(beam_mapping)
                tenant_order = sorted(
                    self.tenants,
                    key=lambda tenant: (
                        -self.tenant_pressure.get(tenant, 0.0),
                        -self.tenant_peak_load.get(tenant, 0.0),
                        tenant,
                    ),
                )

                for tenant in tenant_order:
                    if time.time() >= search_deadline:
                        break
                    source, candidate_mapping, candidate_score = self._best_neighborhood_candidate(
                        beam_mapping,
                        beam_score,
                        tenant,
                        epoch_prices,
                        search_deadline,
                        include_local=True,
                        allow_bnb=False,
                    )
                    if source is None:
                        continue
                    next_candidates.append((candidate_score, candidate_mapping, source))

                if self.max_bnb_tenants_per_round > 0 and time.time() < search_deadline:
                    _, _, epoch_prices = self._compute_epoch_link_prices(beam_mapping)
                    for tenant in tenant_order[: self.max_bnb_tenants_per_round]:
                        if time.time() >= search_deadline:
                            break
                        source, candidate_mapping, candidate_score = self._best_neighborhood_candidate(
                            beam_mapping,
                            beam_score,
                            tenant,
                            epoch_prices,
                            search_deadline,
                            include_local=False,
                            allow_bnb=True,
                        )
                        if source is None:
                            continue
                        next_candidates.append((candidate_score, candidate_mapping, source))

                if len(tenant_order) >= 2 and time.time() < search_deadline:
                    _, _, epoch_prices = self._compute_epoch_link_prices(beam_mapping)
                    joint_width = self.max_joint_tenants_per_round
                    joint_tenants = tenant_order[: min(len(tenant_order), joint_width)]
                    for idx, tenant_a in enumerate(joint_tenants):
                        if time.time() >= search_deadline:
                            break
                        for tenant_b in joint_tenants[idx + 1 :]:
                            if time.time() >= search_deadline:
                                break
                            joint_candidates = self._joint_pair_candidate(
                                beam_mapping,
                                tenant_a,
                                tenant_b,
                                epoch_prices,
                                search_deadline,
                            )
                            if not joint_candidates:
                                continue
                            for candidate_mapping in joint_candidates:
                                if time.time() >= search_deadline:
                                    break
                                candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
                                if self._is_better_objective(candidate_score, beam_score):
                                    next_candidates.append((candidate_score, candidate_mapping, "joint"))

            ranked_candidates = self._dedupe_ranked_candidates(next_candidates)
            if not ranked_candidates:
                break

            top_score, top_mapping, top_source = ranked_candidates[0]
            if self._is_better_objective(top_score, best_score):
                improved = True
                best_mapping = top_mapping
                best_score = top_score
                self.last_move_source = top_source
                if top_source is not None:
                    self.move_source_counts[top_source] = self.move_source_counts.get(top_source, 0) + 1
                self._register_surrogate_candidate(best_mapping, best_score)

            beam = ranked_candidates

            if not improved:
                current_beam_signatures = [
                    self._mapping_signature(mapping)
                    for _score, mapping, _source in beam
                ]
                if current_beam_signatures == previous_beam_signatures:
                    break

        if time.time() < deadline:
            best_mapping, best_score = self._select_price_stable_candidate(
                best_mapping,
                best_score,
                deadline,
            )

        if time.time() < final_selection_deadline:
            best_mapping, best_score = self._surrogate_local_repair(
                best_mapping,
                best_score,
                final_selection_deadline,
            )

        if time.time() < final_selection_deadline:
            best_mapping, best_score = self._select_price_stable_candidate(
                best_mapping,
                best_score,
                final_selection_deadline,
            )

        if time.time() < final_selection_deadline:
            best_mapping, best_score = self._surrogate_pair_swap_polish(
                best_mapping,
                best_score,
                final_selection_deadline,
            )

        if time.time() < deadline:
            best_mapping, best_score = self._select_price_stable_candidate(
                best_mapping,
                best_score,
                deadline,
            )

        self._register_surrogate_candidate(best_mapping, best_score)
        final_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in best_mapping.items()
        }
        final_score = best_score
        self.final_score_is_simulated = False
        if self.validate_with_simulator:
            simulated_best_mapping = final_mapping
            simulated_best_score = self._evaluate_mapping(final_mapping)
            for _surrogate_score, _signature, candidate_mapping in self._surrogate_candidates:
                if time.time() >= deadline:
                    break
                candidate_score = self._evaluate_mapping(candidate_mapping)
                if self._is_better_objective(candidate_score, simulated_best_score):
                    simulated_best_mapping = {
                        tenant: dict(rank_to_server)
                        for tenant, rank_to_server in candidate_mapping.items()
                    }
                    simulated_best_score = candidate_score
            final_mapping = simulated_best_mapping
            final_score = simulated_best_score
            self.final_score_is_simulated = True

        self.final_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in final_mapping.items()
        }
        self.final_obj = float(final_score[0])
        self.final_makespan = float(final_score[0])
        self.final_avg_jct = float(final_score[1])
        self.runtime_seconds = time.time() - start_time

        if self.verbose:
            print(
                f"Multi-neighborhood mapping solve complete: Makespan={final_score[0]:.12f}, "
                f"AvgJCT={final_score[1]:.12f}, "
                f"Score={'sim' if self.final_score_is_simulated else 'surrogate'}, "
                f"Moves={self.move_source_counts}, Runtime={self.runtime_seconds:.2f}s"
            )
        return self


MappingHybridHeuristicSolver = MappingMultiNeighborhoodHeuristicSolver
MappingPortfolioHeuristicSolver = MappingMultiNeighborhoodHeuristicSolver
