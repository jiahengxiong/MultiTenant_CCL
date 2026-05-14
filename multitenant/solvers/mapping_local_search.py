from __future__ import annotations

import time

from .mapping_ilp import MappingHeuristicSolver


class MappingLocalSearchHeuristicSolver(MappingHeuristicSolver):
    """Legacy price-guided local-search mapping heuristic.

    This keeps the same epoch-load surrogate and link-price model as the main
    MappingHeuristicSolver, but replaces tenant-wise BnB with the previous
    bounded swap/k-cycle local search. It is intended for ablation and regression
    comparisons against the BnB variant.
    """

    def _tenant_price_cost_lazy(self, tenant, mapping, epoch_prices):
        compiled_tenant = self.data["compiled_schedule"]["per_tenant"][tenant]
        total_cost = 0.0
        for epoch, src_rank, dst_rank, volume in compiled_tenant["all_flows"]:
            src_server = int(mapping[tenant][src_rank])
            dst_server = int(mapping[tenant][dst_rank])
            total_cost += float(volume) * self._path_epoch_price(
                epoch_prices,
                epoch,
                src_server,
                dst_server,
            )
        return float(total_cost)

    def _rank_swap_price_delta(self, tenant, mapping, epoch_prices, left_rank, right_rank):
        compiled_tenant = self.data["compiled_schedule"]["per_tenant"][tenant]
        affected_flows = {}
        for flow in compiled_tenant["rank_incidence"].get(left_rank, []):
            affected_flows[(int(flow[0]), int(flow[1]), int(flow[2]), float(flow[3]))] = flow
        for flow in compiled_tenant["rank_incidence"].get(right_rank, []):
            affected_flows[(int(flow[0]), int(flow[1]), int(flow[2]), float(flow[3]))] = flow

        current_tenant_mapping = mapping[tenant]
        left_server = int(current_tenant_mapping[left_rank])
        right_server = int(current_tenant_mapping[right_rank])
        delta = 0.0

        def swapped_server(rank):
            if int(rank) == int(left_rank):
                return right_server
            if int(rank) == int(right_rank):
                return left_server
            return int(current_tenant_mapping[rank])

        for epoch, src_rank, dst_rank, volume in affected_flows:
            old_src = int(current_tenant_mapping[src_rank])
            old_dst = int(current_tenant_mapping[dst_rank])
            new_src = swapped_server(src_rank)
            new_dst = swapped_server(dst_rank)
            old_cost = self._path_epoch_price(epoch_prices, epoch, old_src, old_dst)
            new_cost = self._path_epoch_price(epoch_prices, epoch, new_src, new_dst)
            delta += float(volume) * (new_cost - old_cost)
        return float(delta)

    def _rank_reassignment_price_delta(self, tenant, mapping, epoch_prices, reassignment):
        if not reassignment:
            return 0.0

        compiled_tenant = self.data["compiled_schedule"]["per_tenant"][tenant]
        affected_flows = {}
        touched_ranks = {int(rank) for rank in reassignment}
        for rank in touched_ranks:
            for flow in compiled_tenant["rank_incidence"].get(rank, []):
                affected_flows[(int(flow[0]), int(flow[1]), int(flow[2]), float(flow[3]))] = flow

        current_tenant_mapping = mapping[tenant]
        normalized_reassignment = {
            int(rank): int(server)
            for rank, server in reassignment.items()
        }
        delta = 0.0

        def reassigned_server(rank):
            return normalized_reassignment.get(int(rank), int(current_tenant_mapping[rank]))

        for epoch, src_rank, dst_rank, volume in affected_flows:
            old_src = int(current_tenant_mapping[src_rank])
            old_dst = int(current_tenant_mapping[dst_rank])
            new_src = reassigned_server(src_rank)
            new_dst = reassigned_server(dst_rank)
            old_cost = self._path_epoch_price(epoch_prices, epoch, old_src, old_dst)
            new_cost = self._path_epoch_price(epoch_prices, epoch, new_src, new_dst)
            delta += float(volume) * (new_cost - old_cost)
        return float(delta)

    def _block_move_rank_groups(self, branch_order, *, anchor_limit):
        prefix = [int(rank) for rank in branch_order[:anchor_limit]]
        groups: list[tuple[int, ...]] = []
        seen = set()

        def add_group(group):
            normalized = tuple(int(rank) for rank in group)
            if len(normalized) < 3 or normalized in seen:
                return
            seen.add(normalized)
            groups.append(normalized)

        for block_size in (3, 4, 5):
            if len(prefix) < block_size:
                continue
            total_windows = len(prefix) - block_size + 1
            max_windows = 12
            if total_windows <= max_windows:
                starts = range(total_windows)
            else:
                stride = max(1, total_windows // max_windows)
                starts = list(range(0, total_windows, stride))[:max_windows]
                if total_windows - 1 not in starts:
                    starts = list(starts) + [total_windows - 1]
            for start in starts:
                add_group(prefix[start:start + block_size])

        return groups

    def _optimize_tenant_block_with_prices(self, base_mapping, tenant, epoch_prices, deadline):
        ranks = [int(rank) for rank in self.rank_orders[tenant]]
        if len(ranks) <= 1:
            return base_mapping, self._tenant_price_cost_lazy(tenant, base_mapping, epoch_prices)

        best_mapping = {
            current_tenant: dict(rank_to_server)
            for current_tenant, rank_to_server in base_mapping.items()
        }
        best_cost = self._tenant_price_cost_lazy(tenant, best_mapping, epoch_prices)
        branch_order = [
            int(rank)
            for rank in self.data["compiled_schedule"]["per_tenant"][tenant]["branch_order"]
            if int(rank) in set(ranks)
        ]
        if not branch_order:
            branch_order = list(ranks)

        anchor_limit = min(len(branch_order), 32)
        partner_limit = min(len(ranks), 64)
        max_passes = 3
        block_groups = self._block_move_rank_groups(
            branch_order,
            anchor_limit=anchor_limit,
        )

        for _pass_idx in range(max_passes):
            if time.time() >= deadline:
                break

            best_move = None
            best_delta = 0.0
            anchors = branch_order[:anchor_limit]
            tried_pairs: set[tuple[int, int]] = set()

            for left_rank in anchors:
                if time.time() >= deadline:
                    break
                partner_candidates = [
                    rank for rank in branch_order[:partner_limit] if rank != left_rank
                ]
                if left_rank not in branch_order[:partner_limit]:
                    partner_candidates.extend(
                        rank for rank in branch_order[partner_limit:partner_limit + 8]
                        if rank != left_rank
                    )
                for right_rank in partner_candidates:
                    pair = tuple(sorted((int(left_rank), int(right_rank))))
                    if pair in tried_pairs:
                        continue
                    tried_pairs.add(pair)
                    if time.time() >= deadline:
                        break
                    delta = self._rank_swap_price_delta(
                        tenant,
                        best_mapping,
                        epoch_prices,
                        pair[0],
                        pair[1],
                    )
                    if delta < best_delta - 1e-12:
                        best_delta = float(delta)
                        best_move = {
                            "reassignment": {
                                int(pair[0]): int(best_mapping[tenant][pair[1]]),
                                int(pair[1]): int(best_mapping[tenant][pair[0]]),
                            },
                        }

            for rank_group in block_groups:
                if time.time() >= deadline:
                    break
                current_servers = [
                    int(best_mapping[tenant][rank])
                    for rank in rank_group
                ]
                for shift in range(1, len(rank_group)):
                    rotated_servers = current_servers[shift:] + current_servers[:shift]
                    reassignment = {
                        int(rank): int(server)
                        for rank, server in zip(rank_group, rotated_servers)
                    }
                    delta = self._rank_reassignment_price_delta(
                        tenant,
                        best_mapping,
                        epoch_prices,
                        reassignment,
                    )
                    if delta < best_delta - 1e-12:
                        best_delta = float(delta)
                        best_move = {"reassignment": reassignment}

            if best_move is None:
                break

            for rank, server in best_move["reassignment"].items():
                best_mapping[tenant][int(rank)] = int(server)
            best_cost += best_delta

        return self._canonicalize_ring_mapping(best_mapping), float(best_cost)

    def _surrogate_pair_swap_polish(self, best_mapping, best_score, deadline):
        max_passes = 3
        max_pairs_per_tenant = 512

        for _pass_idx in range(max_passes):
            if time.time() >= deadline:
                break
            improved = False
            self.search_rounds += 1

            for tenant in self.tenants:
                if time.time() >= deadline:
                    break

                ranks = [
                    int(rank)
                    for rank in self.data["compiled_schedule"]["per_tenant"][tenant]["branch_order"]
                    if int(rank) in self.rank_orders[tenant]
                ]
                if len(ranks) <= 1:
                    continue

                checked_pairs = 0
                accepted = False
                for left_idx, left_rank in enumerate(ranks):
                    if time.time() >= deadline or accepted:
                        break
                    for right_rank in ranks[left_idx + 1:]:
                        if time.time() >= deadline:
                            break
                        if checked_pairs >= max_pairs_per_tenant:
                            break
                        checked_pairs += 1
                        candidate_mapping = {
                            current_tenant: dict(rank_to_server)
                            for current_tenant, rank_to_server in best_mapping.items()
                        }
                        candidate_mapping[tenant][left_rank], candidate_mapping[tenant][right_rank] = (
                            candidate_mapping[tenant][right_rank],
                            candidate_mapping[tenant][left_rank],
                        )
                        candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
                        if self._is_better_objective(candidate_score, best_score):
                            best_mapping = candidate_mapping
                            best_score = candidate_score
                            improved = True
                            accepted = True
                            break
                    if checked_pairs >= max_pairs_per_tenant:
                        break

            if not improved:
                break

        return best_mapping, best_score

    def solve(self, time_limit=None):
        start_time = time.time()
        deadline = float("inf") if time_limit is None else start_time + float(time_limit)
        search_deadline = deadline

        best_mapping = None
        best_score = (float("inf"), float("inf"))
        seed_candidates = []

        for seed_mapping in self._seed_mappings():
            if time.time() >= search_deadline:
                break
            candidate_score = self._evaluate_surrogate_mapping(seed_mapping)
            seed_candidates.append(
                (
                    candidate_score,
                    {
                        tenant: dict(rank_to_server)
                        for tenant, rank_to_server in seed_mapping.items()
                    },
                )
            )
            if self._is_better_objective(candidate_score, best_score):
                best_mapping = seed_mapping
                best_score = candidate_score

        if best_mapping is None:
            best_mapping = {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in self.initial_tenant_mapping.items()
            }
            best_score = self._evaluate_surrogate_mapping(best_mapping)

        if time.time() < search_deadline and seed_candidates:
            seed_candidates.sort(key=lambda entry: self._objective_sort_key(entry[0]))
            initial_signature = self._mapping_signature(self.initial_tenant_mapping)
            polish_inputs = []
            seen_polish_inputs = set()

            def add_polish_input(score, mapping):
                signature = self._mapping_signature(mapping)
                if signature in seen_polish_inputs:
                    return
                seen_polish_inputs.add(signature)
                polish_inputs.append((score, mapping))

            initial_score = self._evaluate_surrogate_mapping(self.initial_tenant_mapping)
            add_polish_input(
                initial_score,
                {
                    tenant: dict(rank_to_server)
                    for tenant, rank_to_server in self.initial_tenant_mapping.items()
                },
            )
            for score, mapping in seed_candidates:
                if len(polish_inputs) >= 8:
                    break
                add_polish_input(score, mapping)
            if initial_signature not in seen_polish_inputs:
                add_polish_input(initial_score, self.initial_tenant_mapping)

            for seed_score, seed_mapping in polish_inputs:
                if time.time() >= search_deadline:
                    break
                polished_mapping, polished_score = self._surrogate_pair_swap_polish(
                    {
                        tenant: dict(rank_to_server)
                        for tenant, rank_to_server in seed_mapping.items()
                    },
                    seed_score,
                    search_deadline,
                )
                if self._is_better_objective(polished_score, best_score):
                    best_mapping = polished_mapping
                    best_score = polished_score

        working_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in best_mapping.items()
        }

        round_idx = 0
        while round_idx < self.max_price_rounds and time.time() < search_deadline:
            round_idx += 1
            _, _, epoch_prices = self._compute_epoch_link_prices(working_mapping)
            round_mapping = {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in working_mapping.items()
            }
            changed = False

            tenant_order = sorted(
                self.tenants,
                key=lambda tenant: (-self.tenant_pressure.get(tenant, 0.0), -self.tenant_peak_load.get(tenant, 0.0), tenant),
            )
            for tenant in tenant_order:
                if time.time() >= search_deadline:
                    break
                candidate_mapping, _ = self._optimize_tenant_block_with_prices(
                    round_mapping,
                    tenant,
                    epoch_prices,
                    search_deadline,
                )
                if self._mapping_signature(candidate_mapping) != self._mapping_signature(round_mapping):
                    round_mapping = candidate_mapping
                    changed = True

            round_score = self._evaluate_surrogate_mapping(round_mapping)
            if self._is_better_objective(round_score, best_score):
                best_mapping = {
                    tenant: dict(rank_to_server)
                    for tenant, rank_to_server in round_mapping.items()
                }
                best_score = round_score

            working_mapping = round_mapping
            if not changed:
                break

        if time.time() < search_deadline:
            best_mapping, best_score = self._surrogate_pair_swap_polish(
                best_mapping,
                best_score,
                search_deadline,
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
                f"Structured pure mapping solve complete: Makespan={final_score[0]:.12f}, "
                f"AvgJCT={final_score[1]:.12f}, "
                "Mode=link-price-local-search, "
                f"Score={'sim' if self.final_score_is_simulated else 'surrogate'}, "
                f"SurrogateEvalCount={len(self._surrogate_cache)}, "
                f"SimEvalCount={len(self._score_cache)}, Runtime={self.runtime_seconds:.2f}s"
            )

        return self


LegacyMappingHeuristicSolver = MappingLocalSearchHeuristicSolver
