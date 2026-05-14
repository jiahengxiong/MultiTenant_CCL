from __future__ import annotations

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
        beam_width=2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bnb_candidate_time_limit = float(bnb_candidate_time_limit)
        self.max_bnb_tenants_per_round = int(max_bnb_tenants_per_round)
        self.beam_width = max(1, int(beam_width))
        self.last_move_source = None
        self.move_source_counts = {"local": 0, "bnb": 0}

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
            stage_flows=self.stage_flows,
            fairness_lambda=self.fairness_lambda,
            fairness_iterations=self.fairness_iterations,
            fairness_grouping=self.fairness_grouping,
            slot_duration=self.slot_duration_override,
            horizon_slots=self.horizon_slots_override,
            validate_with_simulator=False,
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

    def solve(self, time_limit=None):
        start_time = time.time()
        deadline = float("inf") if time_limit is None else start_time + float(time_limit)

        best_mapping = None
        best_score = (float("inf"), float("inf"))
        initial_candidates = []
        for seed_mapping in self._seed_mappings():
            if time.time() >= deadline:
                break
            candidate_score = self._evaluate_surrogate_mapping(seed_mapping)
            initial_candidates.append((candidate_score, seed_mapping, None))
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
        while round_idx < self.max_price_rounds and time.time() < deadline:
            round_idx += 1
            improved = False
            next_candidates = list(beam)

            for beam_score, beam_mapping, _beam_source in beam:
                if time.time() >= deadline:
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
                    if time.time() >= deadline:
                        break
                    source, candidate_mapping, candidate_score = self._best_neighborhood_candidate(
                        beam_mapping,
                        beam_score,
                        tenant,
                        epoch_prices,
                        deadline,
                        include_local=True,
                        allow_bnb=False,
                    )
                    if source is None:
                        continue
                    next_candidates.append((candidate_score, candidate_mapping, source))

                if self.max_bnb_tenants_per_round > 0 and time.time() < deadline:
                    _, _, epoch_prices = self._compute_epoch_link_prices(beam_mapping)
                    for tenant in tenant_order[: self.max_bnb_tenants_per_round]:
                        if time.time() >= deadline:
                            break
                        source, candidate_mapping, candidate_score = self._best_neighborhood_candidate(
                            beam_mapping,
                            beam_score,
                            tenant,
                            epoch_prices,
                            deadline,
                            include_local=False,
                            allow_bnb=True,
                        )
                        if source is None:
                            continue
                        next_candidates.append((candidate_score, candidate_mapping, source))

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
                break

        if time.time() < deadline:
            best_mapping, best_score = self._surrogate_pair_swap_polish(
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
