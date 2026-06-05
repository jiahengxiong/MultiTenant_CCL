from __future__ import annotations

from collections import defaultdict
import gc
import itertools
import time

from multitenant.objectives import lexicographic_better

from .contention_estimator import TimeExpandedContentionEstimator

try:
    from . import _te_accel
except ImportError:  # pragma: no cover - optional C++ accelerator.
    _te_accel = None


class MappingTimeExpandedEstimatorOptimizer:
    """Standalone optimizer driven only by the time-expanded estimator.

    The class intentionally does not inherit from, wrap, or call the legacy
    collapsed-hybrid optimizer.  It keeps the same high-level search skeleton
    for debugging purposes: deterministic seeds, pressure-ordered tenants,
    local swaps, price-guided single-tenant remapping, joint tenant composition,
    and beam selection.  All scores and search signals come from
    :class:`TimeExpandedContentionEstimator`.
    """

    def __init__(
        self,
        datacenter,
        *,
        tenant_mapping,
        tenant_collective_specs=None,
        tenant_collective_programs=None,
        tenant_start_times=None,
        path_table=None,
        slot_duration=None,
        horizon_slots=None,
        extra_seed_mappings=None,
        verbose=True,
        name="time_expanded_mapping_optimizer",
        bnb_candidate_time_limit=5.0,
        max_bnb_tenants_per_round=2,
        beam_width=3,
        search_skeleton="diverse",
        **_ignored,
    ):
        self.datacenter = datacenter
        self.initial_tenant_mapping = self._copy_mapping(tenant_mapping)
        self.collective = _ignored.get("collective")
        self.tenant_collective_specs = tenant_collective_specs
        self.tenant_collective_programs = tenant_collective_programs
        self.verbose = bool(verbose)
        self.model_name = name
        self.surrogate_mode = "time_expanded"
        self.estimator = TimeExpandedContentionEstimator(
            datacenter,
            tenant_mapping=self.initial_tenant_mapping,
            tenant_collective_specs=tenant_collective_specs,
            tenant_collective_programs=tenant_collective_programs,
            tenant_start_times=tenant_start_times,
            path_table=path_table,
            slot_duration=slot_duration,
            horizon_slots=horizon_slots,
            name=f"{name}_estimator",
        )
        self.tenants = list(self.estimator.tenants)
        self.rank_orders = self.estimator.rank_orders
        self.server_sets = self.estimator.server_sets
        self.extra_seed_mappings = [
            self._copy_mapping(mapping)
            for mapping in (extra_seed_mappings or [])
        ]

        self.bnb_candidate_time_limit = float(bnb_candidate_time_limit)
        self.max_bnb_tenants_per_round = int(max_bnb_tenants_per_round)
        self.beam_width = max(1, int(beam_width))
        self.search_skeleton = str(search_skeleton).lower()
        server_count = int(
            getattr(
                self.datacenter,
                "num_server",
                sum(len(server_set) for server_set in self.server_sets.values()),
            )
        )
        self.max_price_rounds = max(1, min(server_count, 8))
        self.max_stagnation_rounds = 5
        # Time-expanded scores are produced by a fluid slot simulation. Tiny
        # sub-microsecond differences are numerical noise, not meaningful search
        # signal; keep candidate order stable inside this tolerance.
        self.score_sort_tolerance = 1e-8

        self._score_cache = {}
        self._score_component_cache = {}
        self._pipeline_score_cache = {}
        self._analysis_cache = {}
        self._search_state_cache = {}
        self._te_epoch_price_cache = {}
        self._coarse_epoch_price_cache = {}
        self._pair_epoch_price_cache = {}
        self._task_pair_price_cache = {}
        self._accelerated_move_cache = {}
        self._bnb_assignment_cache = {}
        self._task_info_cache = {}
        self._rank_incidence_cache = {}
        self._server_pair_edges_cache = {}
        self._default_edge_price_cache = None
        self._default_sender_price_cache = None
        self._default_receiver_price_cache = None
        self._cache_limits = {
            "score": 4096,
            "score_component": 1024,
            "pipeline_score": 1024,
            "analysis": 1,
            "search_state": 0,
            "te_epoch_price": 4,
            "coarse_epoch_price": 16,
            "pair_epoch_price": 8,
            "task_pair_price": 2,
            "accelerated_move": 512,
            "bnb_assignment": 16,
        }
        self.final_mapping = None
        self.final_obj = None
        self.final_makespan = None
        self.final_avg_jct = None
        self.final_score_is_simulated = False
        self.runtime_seconds = 0.0
        self.search_rounds = 0
        self.move_source_counts = {"local": 0, "bnb": 0, "joint": 0}
        self.last_move_source = None

    @staticmethod
    def _copy_mapping(mapping):
        return {
            int(tenant): {
                int(rank): int(server)
                for rank, server in rank_to_server.items()
            }
            for tenant, rank_to_server in mapping.items()
        }

    @staticmethod
    def _cache_put(cache, key, value, limit):
        if limit == 0:
            return value
        cache[key] = value
        if limit is None or limit <= 0:
            return value
        while len(cache) > int(limit):
            cache.pop(next(iter(cache)))
        return value

    def _clear_search_caches(self, *, keep_scores=True):
        """Release heavyweight time-expanded state between independent passes."""
        if not keep_scores:
            self._score_cache.clear()
            self._score_component_cache.clear()
            self._pipeline_score_cache.clear()
        self._analysis_cache.clear()
        self._search_state_cache.clear()
        self._te_epoch_price_cache.clear()
        self._coarse_epoch_price_cache.clear()
        self._pair_epoch_price_cache.clear()
        self._task_pair_price_cache.clear()
        self._accelerated_move_cache.clear()
        if not keep_scores:
            self._bnb_assignment_cache.clear()
        gc.collect()

    def _is_better_objective(self, candidate, incumbent, tol=None):
        if tol is None:
            tol = self.score_sort_tolerance
        candidate_makespan, candidate_avg_jct = candidate
        incumbent_makespan, incumbent_avg_jct = incumbent
        if float(candidate_avg_jct) < float(incumbent_avg_jct) - float(tol):
            return True
        if abs(float(candidate_avg_jct) - float(incumbent_avg_jct)) <= float(tol):
            return float(candidate_makespan) < float(incumbent_makespan) - float(tol)
        return False

    @staticmethod
    def _objective_sort_key(score):
        makespan, avg_jct = score
        return (float(avg_jct), float(makespan))

    def _score_components(self, mapping):
        normalized = self._normalize_mapping(mapping)
        signature = self._mapping_signature(normalized)
        cached = self._score_component_cache.get(signature)
        if cached is not None:
            return cached
        primary = self._score_cache.get(signature)
        if primary is None:
            estimate = self.estimator.evaluate(normalized)
            primary = (float(estimate.makespan), float(estimate.avg_jct))
            self._cache_put(self._score_cache, signature, primary, self._cache_limits["score"])
        components = {
            "primary": tuple(float(value) for value in primary),
            "pipeline": self._pipeline_score(normalized),
        }
        self._cache_put(
            self._score_component_cache,
            signature,
            components,
            self._cache_limits["score_component"],
        )
        return components

    def _pipeline_score(self, mapping):
        normalized = self._normalize_mapping(mapping)
        signature = self._mapping_signature(normalized)
        cached = self._pipeline_score_cache.get(signature)
        if cached is not None:
            return cached
        pipeline = self.estimator.pipeline_score(normalized)
        pipeline = tuple(float(value) for value in pipeline)
        self._cache_put(
            self._pipeline_score_cache,
            signature,
            pipeline,
            self._cache_limits["pipeline_score"],
        )
        return pipeline

    def _score_avg_bucket(self, score):
        _makespan, avg_jct = score
        return round(float(avg_jct) / self.score_sort_tolerance)

    def _score_makespan_bucket(self, score):
        makespan, _avg_jct = score
        return round(float(makespan) / self.score_sort_tolerance)

    def _candidate_sort_key(self, score, mapping):
        primary_makespan, primary_avg_jct = score
        pipeline_makespan, pipeline_avg_jct = self._pipeline_score(mapping)
        return (
            self._score_avg_bucket(score),
            float(pipeline_avg_jct),
            self._score_makespan_bucket(score),
            float(pipeline_makespan),
        )

    def _rank_entries_by_estimator(self, entries):
        """Sort candidates while evaluating pipeline scores only for near ties.

        The task-level score is the objective used to create primary buckets.
        The pipeline score is intentionally lazy: it is computed only inside a
        bucket whose task-level AvgJCT values are indistinguishable at the
        configured tolerance.
        """
        buckets = defaultdict(list)
        for entry in entries:
            buckets[self._score_avg_bucket(entry[0])].append(entry)

        ranked = []
        for bucket in sorted(buckets):
            group = buckets[bucket]
            if len(group) > 1:
                group.sort(key=lambda entry: self._candidate_sort_key(entry[0], entry[1]))
            else:
                group.sort(key=lambda entry: (
                    self._score_avg_bucket(entry[0]),
                    self._score_makespan_bucket(entry[0]),
                ))
            ranked.extend(group)
        return ranked

    def _is_better_mapping(self, candidate_score, candidate_mapping, incumbent_score, incumbent_mapping, tol=None):
        if tol is None:
            tol = self.score_sort_tolerance
        candidate_makespan, candidate_avg_jct = candidate_score
        incumbent_makespan, incumbent_avg_jct = incumbent_score
        if float(candidate_avg_jct) < float(incumbent_avg_jct) - float(tol):
            return True
        if abs(float(candidate_avg_jct) - float(incumbent_avg_jct)) <= float(tol):
            candidate_pipeline_makespan, candidate_pipeline_avg_jct = self._pipeline_score(candidate_mapping)
            incumbent_pipeline_makespan, incumbent_pipeline_avg_jct = self._pipeline_score(incumbent_mapping)
            if float(candidate_pipeline_avg_jct) < float(incumbent_pipeline_avg_jct) - 1e-9:
                return True
            if abs(float(candidate_pipeline_avg_jct) - float(incumbent_pipeline_avg_jct)) <= 1e-9:
                if float(candidate_makespan) < float(incumbent_makespan) - float(tol):
                    return True
                if abs(float(candidate_makespan) - float(incumbent_makespan)) <= float(tol):
                    return float(candidate_pipeline_makespan) < float(incumbent_pipeline_makespan) - 1e-9
        return False

    def _stable_objective_sort_key(self, score):
        makespan, avg_jct = score
        tol = self.score_sort_tolerance
        return (
            round(float(avg_jct) / tol),
            round(float(makespan) / tol),
        )

    def _normalize_mapping(self, mapping):
        return self._canonicalize_ring_mapping(mapping)

    def _tenant_has_ring_rotation_symmetry(self, tenant):
        ring_collectives = {"allgather", "reducescatter", "allreduce"}
        if self.tenant_collective_programs is not None:
            program = self.tenant_collective_programs.get(int(tenant), [])
            if not program:
                return False
            return all(str(op.get("collective", "")).lower() in ring_collectives for op in program)
        if self.tenant_collective_specs is not None:
            spec = self.tenant_collective_specs.get(int(tenant))
            if spec is None:
                spec = self.tenant_collective_specs.get(str(int(tenant)))
            if not spec:
                return False
            return str(spec.get("collective", "")).lower() in ring_collectives
        if self.collective is None:
            return False
        return str(self.collective).lower() in ring_collectives

    def _canonicalize_ring_mapping(self, mapping):
        canonical = self.estimator.normalize_mapping(mapping)
        for tenant in self.tenants:
            if not self._tenant_has_ring_rotation_symmetry(tenant):
                continue
            ranks = list(self.rank_orders[tenant])
            if len(ranks) <= 1:
                continue
            ordered_servers = [int(canonical[tenant][rank]) for rank in ranks]
            anchor_server = min(ordered_servers)
            anchor_idx = ordered_servers.index(anchor_server)
            if anchor_idx == 0:
                continue
            rotated = ordered_servers[anchor_idx:] + ordered_servers[:anchor_idx]
            canonical[tenant] = {
                int(rank): int(rotated[idx])
                for idx, rank in enumerate(ranks)
            }
        return canonical

    def _mapping_signature(self, mapping):
        return self.estimator.signature(mapping)

    def _evaluate_mapping(self, mapping):
        normalized = self._normalize_mapping(mapping)
        signature = self._mapping_signature(normalized)
        cached = self._score_cache.get(signature)
        if cached is not None:
            return cached
        estimate = self.estimator.evaluate(normalized)
        score = (float(estimate.makespan), float(estimate.avg_jct))
        self._cache_put(self._score_cache, signature, score, self._cache_limits["score"])
        return score

    def _analyze_mapping(self, mapping):
        normalized = self._normalize_mapping(mapping)
        signature = self._mapping_signature(normalized)
        cached = self._analysis_cache.get(signature)
        if cached is not None:
            return cached
        if hasattr(self.estimator, "analyze_for_search"):
            analysis = self.estimator.analyze_for_search(normalized)
        elif hasattr(self.estimator, "_analysis_from_state"):
            state = self._search_state(normalized)
            analysis = self.estimator._analysis_from_state(state, lightweight=True)
        else:
            analysis = self.estimator.analyze(normalized)
        self._cache_put(
            self._analysis_cache,
            signature,
            analysis,
            self._cache_limits["analysis"],
        )
        return analysis

    def _search_state(self, mapping):
        normalized = self._normalize_mapping(mapping)
        signature = self._mapping_signature(normalized)
        cached = self._search_state_cache.get(signature)
        if cached is not None:
            return cached
        if hasattr(self.estimator, "search_state"):
            state = self.estimator.search_state(normalized)
        else:
            state = self.estimator._backbone._compute_time_expanded_surrogate_state(
                normalized,
                collect_signals=True,
                include_hotspots=False,
            )
        self._cache_put(
            self._search_state_cache,
            signature,
            state,
            self._cache_limits["search_state"],
        )
        if "score" in state:
            if signature not in self._score_cache:
                self._cache_put(
                    self._score_cache,
                    signature,
                    tuple(float(value) for value in state["score"]),
                    self._cache_limits["score"],
                )
        return state

    def _time_expanded_epoch_prices(self, mapping):
        normalized = self._normalize_mapping(mapping)
        signature = self._mapping_signature(normalized)
        cached = self._te_epoch_price_cache.get(signature)
        if cached is not None:
            return cached
        state = self._search_state(normalized)
        value = self.estimator._backbone._time_expanded_epoch_prices_from_state(state)
        self._cache_put(
            self._te_epoch_price_cache,
            signature,
            value,
            self._cache_limits["te_epoch_price"],
        )
        return value

    def _server_leaf_sort_key(self, server):
        server_id = int(server)
        if hasattr(self.datacenter, "get_server_leaf"):
            return (int(self.datacenter.get_server_leaf(server_id)), server_id)
        topology = self.datacenter.topology
        for neighbor in topology.successors(server_id):
            if topology.nodes[neighbor].get("type") == "leaf":
                return (int(neighbor), server_id)
        return (server_id, server_id)

    def _apply_server_order(self, base_mapping, tenant, server_order):
        ranks = self.rank_orders[int(tenant)]
        candidate = self._copy_mapping(base_mapping)
        candidate[int(tenant)] = {
            int(rank): int(server_order[idx])
            for idx, rank in enumerate(ranks)
        }
        return self._normalize_mapping(candidate)

    def _seed_mappings(self):
        seeds = []
        seen = set()

        def add_seed(mapping):
            normalized = self._normalize_mapping(mapping)
            signature = self._mapping_signature(normalized)
            if signature in seen:
                return
            seen.add(signature)
            seeds.append(normalized)

        for mapping in self.extra_seed_mappings:
            add_seed(mapping)

        add_seed(self.initial_tenant_mapping)

        global_leaf_local = {
            tenant: self._apply_server_order(
                self.initial_tenant_mapping,
                tenant,
                tuple(
                    sorted(
                        [
                            self.initial_tenant_mapping[tenant][rank]
                            for rank in self.rank_orders[tenant]
                        ],
                        key=self._server_leaf_sort_key,
                    )
                ),
            )[tenant]
            for tenant in self.tenants
        }
        add_seed(global_leaf_local)

        global_leaf_local_rev = {
            tenant: self._apply_server_order(
                self.initial_tenant_mapping,
                tenant,
                tuple(
                    reversed(
                        sorted(
                            [
                                self.initial_tenant_mapping[tenant][rank]
                                for rank in self.rank_orders[tenant]
                            ],
                            key=self._server_leaf_sort_key,
                        )
                    )
                ),
            )[tenant]
            for tenant in self.tenants
        }
        add_seed(global_leaf_local_rev)

        for tenant in self.tenants:
            ranks = self.rank_orders[tenant]
            current_servers = [
                self.initial_tenant_mapping[tenant][rank]
                for rank in ranks
            ]
            if len(current_servers) <= 1:
                continue

            add_seed(self._apply_server_order(self.initial_tenant_mapping, tenant, tuple(reversed(current_servers))))
            leaf_local = tuple(sorted(current_servers, key=self._server_leaf_sort_key))
            add_seed(self._apply_server_order(self.initial_tenant_mapping, tenant, leaf_local))
            add_seed(self._apply_server_order(self.initial_tenant_mapping, tenant, tuple(reversed(leaf_local))))

            rotation_count = min(len(current_servers) - 1, 8)
            shifts = sorted(
                {
                    max(1, round(idx * len(current_servers) / (rotation_count + 1)))
                    for idx in range(1, rotation_count + 1)
                }
            )
            for shift in shifts:
                add_seed(self._apply_server_order(
                    self.initial_tenant_mapping,
                    tenant,
                    tuple(current_servers[shift:] + current_servers[:shift]),
                ))

        return seeds

    def _dedupe_ranked_candidates(self, candidates):
        seen = set()
        ranked = []
        for score, mapping, source in candidates:
            normalized = self._normalize_mapping(mapping)
            signature = self._mapping_signature(normalized)
            if signature in seen:
                continue
            seen.add(signature)
            ranked.append((score, normalized, source))
        ranked = self._rank_entries_by_estimator(ranked)
        selected = []
        selected_signatures = set()

        def add(entry):
            signature = self._mapping_signature(entry[1])
            if signature in selected_signatures:
                return
            selected_signatures.add(signature)
            selected.append(entry)

        for entry in ranked[: self.beam_width]:
            add(entry)

        # The estimator is the final objective, but its first-step ranking can
        # be nearly tied across tenants. Keep one representative from each
        # proposal source so a useful branch is not discarded before the next
        # full-estimator evaluation.
        seen_sources = set()
        max_selected = max(self.beam_width, min(len(ranked), self.beam_width + 3))
        for entry in ranked:
            source = entry[2]
            if source is None or source in seen_sources:
                continue
            seen_sources.add(source)
            add(entry)
            if len(selected) >= max_selected:
                break

        selected = self._rank_entries_by_estimator(selected)
        return selected

    def _dedupe_ranked_candidates_hybrid_order(self, candidates):
        """Match the legacy hybrid beam policy: dedupe, rank, keep top-B only."""
        seen = set()
        ranked = []
        for score, mapping, source in candidates:
            normalized = self._normalize_mapping(mapping)
            signature = self._mapping_signature(normalized)
            if signature in seen:
                continue
            seen.add(signature)
            ranked.append((score, normalized, source))
        ranked = self._rank_entries_by_estimator(ranked)
        return ranked[: self.beam_width]

    def _tenant_block_recombination_candidates(self, candidates, analysis, deadline):
        """Combine tenant-level slices from estimator-scored candidates.

        Local and BnB proposals change one tenant at a time, while some
        contention basins only improve when several tenant mappings change
        together.  This operator treats the estimator as a black-box objective:
        it builds a small set of tenant-level mapping variants from the current
        candidate pool, enumerates their pressure-ordered combinations, and
        keeps only complete mappings that improve the current best estimator
        score.  It never uses the collapsed hybrid solution as an incumbent.
        """
        if time.time() >= deadline or len(candidates) <= 1:
            return []
        ranked = sorted(
            (
                (tuple(float(value) for value in score), self._normalize_mapping(mapping), source)
                for score, mapping, source in candidates
            ),
            key=lambda entry: self._stable_objective_sort_key(entry[0]),
        )
        base_score, base_mapping, _base_source = ranked[0]
        base_signature = self._mapping_signature(base_mapping)
        tenant_order = self._tenant_order(analysis)
        selected_tenants = tenant_order[: min(10, len(tenant_order))]
        if len(selected_tenants) < 2:
            return []

        variants_by_tenant = []
        for tenant in selected_tenants:
            base_slice = tuple(
                int(base_mapping[int(tenant)][rank])
                for rank in self.rank_orders[int(tenant)]
            )
            variants = [(base_slice, dict(base_mapping[int(tenant)]))]
            seen_slices = {base_slice}
            for _score, mapping, _source in ranked:
                tenant_slice = tuple(
                    int(mapping[int(tenant)][rank])
                    for rank in self.rank_orders[int(tenant)]
                )
                if tenant_slice in seen_slices:
                    continue
                seen_slices.add(tenant_slice)
                variants.append((tenant_slice, dict(mapping[int(tenant)])))
                if len(variants) >= 2:
                    break
            variants_by_tenant.append((int(tenant), variants))

        if sum(len(variants) > 1 for _tenant, variants in variants_by_tenant) < 2:
            return []

        recombined = []
        seen = {base_signature}
        for choice in itertools.product(*(range(len(variants)) for _tenant, variants in variants_by_tenant)):
            if time.time() >= deadline:
                break
            if not any(choice):
                continue
            if sum(1 for idx in choice if idx != 0) < 2:
                continue
            mapping = self._copy_mapping(base_mapping)
            for (tenant, variants), variant_idx in zip(variants_by_tenant, choice):
                if variant_idx == 0:
                    continue
                mapping[int(tenant)] = dict(variants[variant_idx][1])
            mapping = self._normalize_mapping(mapping)
            signature = self._mapping_signature(mapping)
            if signature in seen:
                continue
            seen.add(signature)
            score = self._evaluate_mapping(mapping)
            if self._is_better_mapping(score, mapping, base_score, base_mapping):
                recombined.append((score, mapping, "recombine"))
        return recombined

    def _tenant_order(self, analysis):
        return sorted(
            self.tenants,
            key=lambda tenant: (
                -float(analysis.tenant_pressure.get(int(tenant), 0.0)),
                -float(analysis.tenant_peak_load.get(int(tenant), 0.0)),
                int(tenant),
            ),
        )

    def _rank_order(self, analysis, tenant):
        ranks = list(self.rank_orders[int(tenant)])
        task_meta = self.estimator.task_dag[int(tenant)]
        branch_order = [
            int(rank)
            for rank in task_meta.get("branch_order", ranks)
            if int(rank) in ranks
        ] or ranks
        branch_position = {int(rank): idx for idx, rank in enumerate(branch_order)}
        return sorted(
            ranks,
            key=lambda rank: (
                -float(analysis.rank_pressure.get((int(tenant), int(rank)), 0.0)),
                branch_position.get(int(rank), len(branch_position)),
                int(rank),
            ),
        )

    def _legacy_branch_order(self, tenant):
        ranks = list(self.rank_orders[int(tenant)])
        task_meta = self.estimator.task_dag[int(tenant)]
        branch_order = [
            int(rank)
            for rank in task_meta.get("branch_order", ranks)
            if int(rank) in ranks
        ]
        return branch_order or ranks

    def _task_infos(self, tenant):
        tenant = int(tenant)
        cached = self._task_info_cache.get(tenant)
        if cached is not None:
            return cached
        infos = []
        task_meta = self.estimator.task_dag[tenant]
        for task_id, task_tuple in task_meta["task_info"].items():
            infos.append(
                (
                    int(task_id),
                    int(task_tuple[1]),
                    int(task_tuple[2]),
                    float(task_tuple[3]),
                )
            )
        self._task_info_cache[tenant] = infos
        return infos

    def _rank_incidence(self, tenant):
        tenant = int(tenant)
        cached = self._rank_incidence_cache.get(tenant)
        if cached is not None:
            return cached
        incidence = defaultdict(list)
        for task_id, src_rank, dst_rank, volume in self._task_infos(tenant):
            incidence[int(src_rank)].append((task_id, src_rank, dst_rank, volume))
            incidence[int(dst_rank)].append((task_id, src_rank, dst_rank, volume))
        self._rank_incidence_cache[tenant] = incidence
        return incidence

    def _tenant_price_cost(self, base_mapping, tenant, tenant_mapping, task_pair_price):
        total = 0.0
        for task_id, src_rank, dst_rank, volume in self._task_infos(tenant):
            src_server = int(tenant_mapping[int(src_rank)])
            dst_server = int(tenant_mapping[int(dst_rank)])
            if src_server == dst_server:
                continue
            total += float(volume) * float(task_pair_price[(task_id, src_server, dst_server)])
        return float(total)

    def _coarse_epoch_prices(self, mapping):
        normalized = self._normalize_mapping(mapping)
        signature = self._mapping_signature(normalized)
        cached = self._coarse_epoch_price_cache.get(signature)
        if cached is not None:
            return cached
        backbone = self.estimator._backbone
        epoch_loads, epoch_sender_loads, epoch_receiver_loads, _epoch_maxima = (
            backbone._compute_epoch_resource_load_state(normalized)
        )
        prices = []
        server_send_capacity = backbone.data["server_send_capacity"]
        server_recv_capacity = backbone.data["server_recv_capacity"]
        for epoch_idx, epoch_load in enumerate(epoch_loads):
            prices.append(
                {
                    "edge": {
                        edge: backbone._edge_price_value(edge, normalized_load)
                        for edge, normalized_load in epoch_load.items()
                    },
                    "sender": {
                        server: backbone._resource_price_value(server_send_capacity[server], normalized_load)
                        for server, normalized_load in epoch_sender_loads[epoch_idx].items()
                    },
                    "receiver": {
                        server: backbone._resource_price_value(server_recv_capacity[server], normalized_load)
                        for server, normalized_load in epoch_receiver_loads[epoch_idx].items()
                    },
                }
            )
        self._cache_put(
            self._coarse_epoch_price_cache,
            signature,
            prices,
            self._cache_limits["coarse_epoch_price"],
        )
        return prices

    def _default_resource_prices(self):
        backbone = self.estimator._backbone
        if self._default_edge_price_cache is None:
            self._default_edge_price_cache = {
                edge: backbone._edge_price_value(edge, 0.0)
                for edge in backbone.data["edge_capacity"]
            }
        if self._default_sender_price_cache is None:
            self._default_sender_price_cache = {
                server: backbone._resource_price_value(capacity, 0.0)
                for server, capacity in backbone.data["server_send_capacity"].items()
            }
        if self._default_receiver_price_cache is None:
            self._default_receiver_price_cache = {
                server: backbone._resource_price_value(capacity, 0.0)
                for server, capacity in backbone.data["server_recv_capacity"].items()
            }
        return (
            self._default_edge_price_cache,
            self._default_sender_price_cache,
            self._default_receiver_price_cache,
        )

    def _server_pair_edges(self, tenant, servers):
        tenant = int(tenant)
        server_tuple = tuple(int(server) for server in servers)
        cache_key = (tenant, server_tuple)
        cached = self._server_pair_edges_cache.get(cache_key)
        if cached is not None:
            return cached
        backbone = self.estimator._backbone
        path_edges = backbone.data["path_edges"]
        lookup = {}
        for src_server in server_tuple:
            for dst_server in server_tuple:
                if src_server == dst_server:
                    continue
                lookup[(src_server, dst_server)] = tuple(
                    backbone._path_edges_for_pair(
                        path_edges,
                        tenant,
                        src_server,
                        dst_server,
                    )
                )
        self._cache_put(self._server_pair_edges_cache, cache_key, lookup, 128)
        return lookup

    def _coarse_path_price(self, epoch_prices, tenant, epoch, src_server, dst_server):
        backbone = self.estimator._backbone
        default_edge_price, default_sender_price, default_receiver_price = self._default_resource_prices()
        state = epoch_prices[int(epoch)]
        price = state["sender"].get(
            int(src_server),
            default_sender_price[int(src_server)],
        )
        for edge in self._server_pair_edges(int(tenant), (int(src_server), int(dst_server)))[(int(src_server), int(dst_server))]:
            price += state["edge"].get(edge, default_edge_price[edge])
        price += state["receiver"].get(
            int(dst_server),
            default_receiver_price[int(dst_server)],
        )
        return float(price)

    def _coarse_pair_epoch_prices(self, tenant, servers, epoch_prices):
        server_tuple = tuple(int(server) for server in servers)
        pair_edges = self._server_pair_edges(tenant, server_tuple)
        default_edge_price, default_sender_price, default_receiver_price = self._default_resource_prices()
        if _te_accel is None or not hasattr(_te_accel, "coarse_pair_epoch_prices"):
            raise RuntimeError("C++ coarse_pair_epoch_prices kernel is required")
        return _te_accel.coarse_pair_epoch_prices(
            int(tenant),
            list(server_tuple),
            list(epoch_prices),
            pair_edges,
            default_edge_price,
            default_sender_price,
            default_receiver_price,
        )

    def _cached_pair_epoch_prices(self, namespace, mapping, tenant, servers, epoch_prices):
        signature = self._mapping_signature(self._normalize_mapping(mapping))
        key = (str(namespace), signature, int(tenant), tuple(int(server) for server in servers))
        cached = self._pair_epoch_price_cache.get(key)
        if cached is not None:
            return cached
        lookup = self._coarse_pair_epoch_prices(tenant, servers, epoch_prices)
        self._cache_put(
            self._pair_epoch_price_cache,
            key,
            lookup,
            self._cache_limits["pair_epoch_price"],
        )
        return lookup

    def _coarse_rank_incidence(self, tenant):
        return self.estimator._backbone.data["compiled_schedule"]["per_tenant"][int(tenant)]["rank_incidence"]

    def _coarse_all_flows(self, tenant):
        return self.estimator._backbone.data["compiled_schedule"]["per_tenant"][int(tenant)]["all_flows"]

    def _coarse_reassignment_delta(self, mapping, tenant, reassignment, pair_epoch_price):
        affected = {}
        for rank in reassignment:
            for flow in self._coarse_rank_incidence(tenant).get(int(rank), []):
                affected[(int(flow[0]), int(flow[1]), int(flow[2]), float(flow[3]))] = flow

        current_mapping = mapping[int(tenant)]
        reassignment = {
            int(rank): int(server)
            for rank, server in reassignment.items()
        }

        def reassigned_server(rank):
            return reassignment.get(int(rank), int(current_mapping[int(rank)]))

        delta = 0.0
        for epoch, src_rank, dst_rank, volume in affected:
            old_src = int(current_mapping[int(src_rank)])
            old_dst = int(current_mapping[int(dst_rank)])
            new_src = reassigned_server(src_rank)
            new_dst = reassigned_server(dst_rank)
            delta += float(volume) * (
                pair_epoch_price[(int(epoch), new_src, new_dst)]
                - pair_epoch_price[(int(epoch), old_src, old_dst)]
            )
        return float(delta)

    def _accelerated_best_coarse_move(self, mapping, tenant, branch_order, anchor_limit, partner_limit, block_groups, pair_epoch_price):
        if _te_accel is None:
            return None
        cache_key = (
            "coarse-move",
            self._mapping_signature(mapping),
            int(tenant),
            tuple(int(rank) for rank in branch_order),
            int(anchor_limit),
            int(partner_limit),
            tuple(tuple(int(rank) for rank in group) for group in block_groups),
            id(pair_epoch_price),
        )
        cached = self._accelerated_move_cache.get(cache_key)
        if cached is not None:
            if cached is False:
                return None
            delta, reassignment = cached
            return float(delta), dict(reassignment)
        try:
            if hasattr(_te_accel, "best_coarse_move_delta"):
                result = _te_accel.best_coarse_move_delta(
                    list(self._coarse_all_flows(tenant)),
                    dict(mapping[int(tenant)]),
                    [int(rank) for rank in branch_order],
                    int(anchor_limit),
                    int(partner_limit),
                    [tuple(int(rank) for rank in group) for group in block_groups],
                    pair_epoch_price,
                )
            else:
                result = _te_accel.best_rank_swap_delta(
                    list(self._coarse_all_flows(tenant)),
                    dict(mapping[int(tenant)]),
                    [int(rank) for rank in branch_order],
                    int(anchor_limit),
                    int(partner_limit),
                    pair_epoch_price,
                )
                if result is not None:
                    delta, left_rank, right_rank = result
                    result = (
                        float(delta),
                        {
                            int(left_rank): int(mapping[int(tenant)][int(right_rank)]),
                            int(right_rank): int(mapping[int(tenant)][int(left_rank)]),
                        },
                    )
        except Exception:
            self._cache_put(
                self._accelerated_move_cache,
                cache_key,
                False,
                self._cache_limits["accelerated_move"],
            )
            return None
        if result is None:
            self._cache_put(
                self._accelerated_move_cache,
                cache_key,
                False,
                self._cache_limits["accelerated_move"],
            )
            return None
        delta, reassignment = result
        value = (
            float(delta),
            {
                int(rank): int(server)
                for rank, server in dict(reassignment).items()
            },
        )
        self._cache_put(
            self._accelerated_move_cache,
            cache_key,
            value,
            self._cache_limits["accelerated_move"],
        )
        return value[0], dict(value[1])

    def _accelerated_best_coarse_swap(self, mapping, tenant, branch_order, anchor_limit, partner_limit, pair_epoch_price):
        if _te_accel is None:
            return None
        cache_key = (
            "coarse-swap",
            self._mapping_signature(mapping),
            int(tenant),
            tuple(int(rank) for rank in branch_order),
            int(anchor_limit),
            int(partner_limit),
            id(pair_epoch_price),
        )
        cached = self._accelerated_move_cache.get(cache_key)
        if cached is not None:
            if cached is False:
                return None
            delta, reassignment = cached
            return float(delta), dict(reassignment)
        try:
            result = _te_accel.best_rank_swap_delta(
                list(self._coarse_all_flows(tenant)),
                dict(mapping[int(tenant)]),
                [int(rank) for rank in branch_order],
                int(anchor_limit),
                int(partner_limit),
                pair_epoch_price,
            )
        except Exception:
            self._cache_put(
                self._accelerated_move_cache,
                cache_key,
                False,
                self._cache_limits["accelerated_move"],
            )
            return None
        if result is None:
            self._cache_put(
                self._accelerated_move_cache,
                cache_key,
                False,
                self._cache_limits["accelerated_move"],
            )
            return None
        delta, left_rank, right_rank = result
        value = (
            float(delta),
            {
                int(left_rank): int(mapping[int(tenant)][int(right_rank)]),
                int(right_rank): int(mapping[int(tenant)][int(left_rank)]),
            },
        )
        self._cache_put(
            self._accelerated_move_cache,
            cache_key,
            value,
            self._cache_limits["accelerated_move"],
        )
        return value[0], dict(value[1])

    def _accelerated_coarse_local_descent(
        self,
        mapping,
        tenant,
        branch_order,
        anchor_limit,
        partner_limit,
        block_groups,
        pair_epoch_price,
        passes,
    ):
        if _te_accel is None or not hasattr(_te_accel, "coarse_local_descent"):
            return None
        cache_key = (
            "coarse-local-descent",
            self._mapping_signature(mapping),
            int(tenant),
            tuple(int(rank) for rank in branch_order),
            int(anchor_limit),
            int(partner_limit),
            tuple(tuple(int(rank) for rank in group) for group in block_groups),
            id(pair_epoch_price),
            int(passes),
        )
        cached = self._accelerated_move_cache.get(cache_key)
        if cached is not None:
            if cached is False:
                return None
            return dict(cached)
        try:
            flows = list(self._coarse_all_flows(tenant))
            rank_to_server = dict(mapping[int(tenant)])
            ordered_branch = [int(rank) for rank in branch_order]
            ordered_groups = [tuple(int(rank) for rank in group) for group in block_groups]
            result = _te_accel.coarse_local_descent(
                flows,
                rank_to_server,
                ordered_branch,
                int(anchor_limit),
                int(partner_limit),
                ordered_groups,
                pair_epoch_price,
                int(passes),
            )
        except Exception:
            self._cache_put(
                self._accelerated_move_cache,
                cache_key,
                False,
                self._cache_limits["accelerated_move"],
            )
            return None
        if result is None:
            self._cache_put(
                self._accelerated_move_cache,
                cache_key,
                False,
                self._cache_limits["accelerated_move"],
            )
            return None
        value = {
            int(rank): int(server)
            for rank, server in dict(result).items()
        }
        self._cache_put(
            self._accelerated_move_cache,
            cache_key,
            value,
            self._cache_limits["accelerated_move"],
        )
        return dict(value)

    def _coarse_local_candidate(self, base_mapping, base_score, tenant, deadline):
        ranks = list(self.rank_orders[int(tenant)])
        if len(ranks) <= 1 or time.time() >= deadline:
            return None, base_mapping, base_score

        mapping = self._copy_mapping(base_mapping)
        epoch_prices = self._coarse_epoch_prices(mapping)
        pair_epoch_price = self._cached_pair_epoch_prices(
            "coarse",
            mapping,
            tenant,
            self.server_sets[int(tenant)],
            epoch_prices,
        )
        branch_order = self._legacy_branch_order(tenant)
        anchor_limit = min(len(branch_order), 32)
        partner_limit = min(len(ranks), 64)
        block_groups = self._block_move_rank_groups(branch_order, anchor_limit=anchor_limit)

        accelerated_mapping = self._accelerated_coarse_local_descent(
            mapping,
            tenant,
            branch_order,
            anchor_limit,
            partner_limit,
            block_groups,
            pair_epoch_price,
            3,
        )
        if accelerated_mapping is not None:
            mapping[int(tenant)] = accelerated_mapping
        else:
            for _pass_idx in range(3):
                if time.time() >= deadline:
                    break
                best_delta = 0.0
                best_move = None
                tried_pairs = set()
                for left_rank in branch_order[:anchor_limit]:
                    if time.time() >= deadline:
                        break
                    partner_candidates = [rank for rank in branch_order[:partner_limit] if rank != left_rank]
                    if left_rank not in branch_order[:partner_limit]:
                        partner_candidates.extend(
                            rank
                            for rank in branch_order[partner_limit:partner_limit + 8]
                            if rank != left_rank
                        )
                    for right_rank in partner_candidates:
                        pair = tuple(sorted((int(left_rank), int(right_rank))))
                        if pair in tried_pairs:
                            continue
                        tried_pairs.add(pair)
                        reassignment = {
                            int(pair[0]): int(mapping[int(tenant)][pair[1]]),
                            int(pair[1]): int(mapping[int(tenant)][pair[0]]),
                        }
                        delta = self._coarse_reassignment_delta(
                            mapping,
                            tenant,
                            reassignment,
                            pair_epoch_price,
                        )
                        if delta < best_delta - 1e-12:
                            best_delta = float(delta)
                            best_move = reassignment

                for rank_group in block_groups:
                    if time.time() >= deadline:
                        break
                    current_servers = [int(mapping[int(tenant)][rank]) for rank in rank_group]
                    for shift in range(1, len(rank_group)):
                        reassignment = {
                            int(rank): int(server)
                            for rank, server in zip(rank_group, current_servers[shift:] + current_servers[:shift])
                        }
                        delta = self._coarse_reassignment_delta(
                            mapping,
                            tenant,
                            reassignment,
                            pair_epoch_price,
                        )
                        if delta < best_delta - 1e-12:
                            best_delta = float(delta)
                            best_move = reassignment

                if best_move is None:
                    break
                for rank, server in best_move.items():
                    mapping[int(tenant)][int(rank)] = int(server)

        mapping = self._normalize_mapping(mapping)
        if self._mapping_signature(mapping) == self._mapping_signature(base_mapping):
            return None, base_mapping, base_score
        score = self._evaluate_mapping(mapping)
        return "coarse-local", mapping, score

    def _epoch_price_local_candidate(self, base_mapping, base_score, tenant, deadline, *, analysis=None):
        """Generate one tenant-local proposal from time-expanded epoch prices.

        Full time-expanded scoring remains the objective.  This routine only
        coarsens the current slot prices into collective epochs to produce a
        more stable proposal signal for local remapping.
        """
        ranks = list(self.rank_orders[int(tenant)])
        if len(ranks) <= 1 or time.time() >= deadline:
            return None, base_mapping, base_score

        mapping = self._copy_mapping(base_mapping)
        if analysis is not None and hasattr(self.estimator, "epoch_prices_from_analysis"):
            _epoch_maxima, epoch_prices = self.estimator.epoch_prices_from_analysis(analysis)
        else:
            _epoch_maxima, epoch_prices = self._time_expanded_epoch_prices(mapping)
        pair_epoch_price = self._cached_pair_epoch_prices(
            "time-expanded",
            mapping,
            tenant,
            self.server_sets[int(tenant)],
            epoch_prices,
        )
        branch_order = self._legacy_branch_order(tenant)
        anchor_limit = min(len(branch_order), 32)
        partner_limit = min(len(ranks), 64)
        block_groups = self._block_move_rank_groups(branch_order, anchor_limit=anchor_limit)

        accelerated_mapping = self._accelerated_coarse_local_descent(
            mapping,
            tenant,
            branch_order,
            anchor_limit,
            partner_limit,
            block_groups,
            pair_epoch_price,
            3,
        )
        if accelerated_mapping is not None:
            mapping[int(tenant)] = accelerated_mapping
        else:
            for _pass_idx in range(3):
                if time.time() >= deadline:
                    break
                best_delta = 0.0
                best_move = None
                tried_pairs = set()
                for left_rank in branch_order[:anchor_limit]:
                    if time.time() >= deadline:
                        break
                    partner_candidates = [rank for rank in branch_order[:partner_limit] if rank != left_rank]
                    if left_rank not in branch_order[:partner_limit]:
                        partner_candidates.extend(
                            rank
                            for rank in branch_order[partner_limit:partner_limit + 8]
                            if rank != left_rank
                        )
                    for right_rank in partner_candidates:
                        pair = tuple(sorted((int(left_rank), int(right_rank))))
                        if pair in tried_pairs:
                            continue
                        tried_pairs.add(pair)
                        reassignment = {
                            int(pair[0]): int(mapping[int(tenant)][pair[1]]),
                            int(pair[1]): int(mapping[int(tenant)][pair[0]]),
                        }
                        delta = self._coarse_reassignment_delta(
                            mapping,
                            tenant,
                            reassignment,
                            pair_epoch_price,
                        )
                        if delta < best_delta - 1e-12:
                            best_delta = float(delta)
                            best_move = reassignment

                for rank_group in block_groups:
                    if time.time() >= deadline:
                        break
                    current_servers = [int(mapping[int(tenant)][rank]) for rank in rank_group]
                    for shift in range(1, len(rank_group)):
                        reassignment = {
                            int(rank): int(server)
                            for rank, server in zip(rank_group, current_servers[shift:] + current_servers[:shift])
                        }
                        delta = self._coarse_reassignment_delta(
                            mapping,
                            tenant,
                            reassignment,
                            pair_epoch_price,
                        )
                        if delta < best_delta - 1e-12:
                            best_delta = float(delta)
                            best_move = reassignment

                if best_move is None:
                    break
                for rank, server in best_move.items():
                    mapping[int(tenant)][int(rank)] = int(server)

        mapping = self._normalize_mapping(mapping)
        if self._mapping_signature(mapping) == self._mapping_signature(base_mapping):
            return None, base_mapping, base_score
        score = self._evaluate_mapping(mapping)
        return "epoch-local", mapping, score

    def _task_pair_prices(self, base_mapping, tenant, analysis=None):
        servers = tuple(int(base_mapping[int(tenant)][rank]) for rank in self.rank_orders[int(tenant)])
        signature = self._mapping_signature(self._normalize_mapping(base_mapping))
        key = (signature, int(tenant), servers)
        cached = self._task_pair_price_cache.get(key)
        if cached is not None:
            return cached
        lookup = self.estimator.task_pair_price_lookup(
            base_mapping,
            int(tenant),
            servers,
            analysis=analysis,
        )
        self._cache_put(
            self._task_pair_price_cache,
            key,
            lookup,
            self._cache_limits["task_pair_price"],
        )
        return lookup

    def _scored_swap_pairs(self, base_mapping, tenant, analysis, *, hot_limit=16, partner_limit=48):
        rank_order = self._rank_order(analysis, tenant)
        if len(rank_order) <= 1:
            return []
        hot_ranks = rank_order[: min(hot_limit, len(rank_order))]
        partner_ranks = rank_order[: min(partner_limit, len(rank_order))]
        for rank in hot_ranks:
            if rank not in partner_ranks:
                partner_ranks.append(rank)

        if _te_accel is None or not hasattr(_te_accel, "scored_swap_pairs_from_state"):
            raise RuntimeError("C++ scored swap-pair kernel is required")

        tenant = int(tenant)
        servers = tuple(int(base_mapping[tenant][rank]) for rank in self.rank_orders[tenant])
        server_send_capacity = self.estimator._backbone.data["server_send_capacity"]
        server_recv_capacity = self.estimator._backbone.data["server_recv_capacity"]
        edge_capacity = self.estimator._backbone.data["edge_capacity"]
        path_edges = self.estimator._backbone.data["path_edges"]
        server_count = self.estimator._dense_server_count(
            servers,
            server_send_capacity,
            server_recv_capacity,
        )
        edge_items, edge_to_idx, base_sender, base_receiver, base_edge = (
            self.estimator._dense_static_price_vectors(
                server_count,
                edge_capacity,
                server_send_capacity,
                server_recv_capacity,
            )
        )
        dense_entry_count = max(len(analysis.slot_prices), 1) * (server_count * 2 + len(edge_items))
        if dense_entry_count > 2_000_000:
            raise RuntimeError(
                f"scored swap-pair dense state too large: {dense_entry_count} entries"
            )
        path_edges_by_pair = self.estimator._dense_path_edges_by_pair(
            tenant,
            server_count,
            path_edges,
            edge_to_idx,
        )
        scored = _te_accel.scored_swap_pairs_from_state(
            [int(rank) for rank in hot_ranks],
            [int(rank) for rank in partner_ranks],
            list(self._task_infos(tenant)),
            {int(rank): int(server) for rank, server in base_mapping[tenant].items()},
            analysis.task_active_slots,
            analysis.task_ready_slots,
            tenant,
            analysis.slot_prices,
            list(base_sender),
            list(base_receiver),
            list(base_edge),
            edge_to_idx,
            path_edges_by_pair,
        )
        return [
            (float(delta), (int(pair[0]), int(pair[1])))
            for delta, pair in scored
        ]

    def _rank_reassignment_price_delta(
        self,
        base_mapping,
        tenant,
        reassignment,
        task_pair_price,
        rank_incidence=None,
    ):
        if not reassignment:
            return 0.0

        if rank_incidence is None:
            rank_incidence = self._rank_incidence(tenant)
        affected_tasks = {}
        for rank in reassignment:
            for task in rank_incidence.get(int(rank), []):
                affected_tasks[(task[0], task[1], task[2], task[3])] = task

        current_mapping = base_mapping[int(tenant)]
        reassignment = {
            int(rank): int(server)
            for rank, server in reassignment.items()
        }

        def reassigned_server(rank):
            return reassignment.get(int(rank), int(current_mapping[int(rank)]))

        delta = 0.0
        for task_id, src_rank, dst_rank, volume in affected_tasks.values():
            old_src = int(current_mapping[int(src_rank)])
            old_dst = int(current_mapping[int(dst_rank)])
            new_src = reassigned_server(src_rank)
            new_dst = reassigned_server(dst_rank)
            old_cost = task_pair_price.get((task_id, old_src, old_dst), 0.0)
            new_cost = task_pair_price.get((task_id, new_src, new_dst), 0.0)
            delta += float(volume) * (float(new_cost) - float(old_cost))
        return float(delta)

    @staticmethod
    def _block_move_rank_groups(branch_order, *, anchor_limit):
        prefix = [int(rank) for rank in branch_order[:anchor_limit]]
        groups = []
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

    def _local_block_candidate(self, base_mapping, tenant, analysis, deadline):
        ranks = list(self.rank_orders[int(tenant)])
        if len(ranks) <= 1 or time.time() >= deadline:
            return base_mapping

        best_mapping = self._copy_mapping(base_mapping)
        task_pair_price = self._task_pair_prices(base_mapping, tenant, analysis=analysis)
        rank_incidence = self._rank_incidence(tenant)
        branch_order = self._legacy_branch_order(tenant)
        anchor_limit = min(len(branch_order), 32)
        partner_limit = min(len(ranks), 64)
        block_groups = self._block_move_rank_groups(branch_order, anchor_limit=anchor_limit)
        max_passes = 3
        for _pass_idx in range(max_passes):
            if time.time() >= deadline:
                break
            best_move = None
            best_delta = 0.0
            anchors = branch_order[:anchor_limit]
            tried_pairs = set()

            for left_rank in anchors:
                if time.time() >= deadline:
                    break
                partner_candidates = [rank for rank in branch_order[:partner_limit] if rank != left_rank]
                if left_rank not in branch_order[:partner_limit]:
                    partner_candidates.extend(
                        rank
                        for rank in branch_order[partner_limit:partner_limit + 8]
                        if rank != left_rank
                    )
                for right_rank in partner_candidates:
                    pair = tuple(sorted((int(left_rank), int(right_rank))))
                    if pair in tried_pairs:
                        continue
                    tried_pairs.add(pair)
                    reassignment = {
                        int(pair[0]): int(best_mapping[int(tenant)][pair[1]]),
                        int(pair[1]): int(best_mapping[int(tenant)][pair[0]]),
                    }
                    delta = self._rank_reassignment_price_delta(
                        best_mapping,
                        tenant,
                        reassignment,
                        task_pair_price,
                        rank_incidence,
                    )
                    if delta < best_delta - 1e-12:
                        best_delta = float(delta)
                        best_move = reassignment

            for rank_group in block_groups:
                if time.time() >= deadline:
                    break
                current_servers = [int(best_mapping[int(tenant)][rank]) for rank in rank_group]
                for shift in range(1, len(rank_group)):
                    rotated_servers = current_servers[shift:] + current_servers[:shift]
                    reassignment = {
                        int(rank): int(server)
                        for rank, server in zip(rank_group, rotated_servers)
                    }
                    delta = self._rank_reassignment_price_delta(
                        best_mapping,
                        tenant,
                        reassignment,
                        task_pair_price,
                        rank_incidence,
                    )
                    if delta < best_delta - 1e-12:
                        best_delta = float(delta)
                        best_move = reassignment

            if best_move is None:
                break

            for rank, server in best_move.items():
                best_mapping[int(tenant)][int(rank)] = int(server)
        return self._normalize_mapping(best_mapping)

    def _local_price_candidate_pool(self, base_mapping, base_score, tenant, analysis, deadline):
        ranks = list(self.rank_orders[int(tenant)])
        if len(ranks) <= 1 or time.time() >= deadline:
            return []

        task_pair_price = self._task_pair_prices(base_mapping, tenant, analysis=analysis)
        rank_incidence = self._rank_incidence(tenant)
        branch_order = self._legacy_branch_order(tenant)
        anchor_limit = min(len(branch_order), 32)
        partner_limit = min(len(ranks), 64)
        block_groups = self._block_move_rank_groups(branch_order, anchor_limit=anchor_limit)

        moves = []
        tried_pairs = set()
        for left_rank in branch_order[:anchor_limit]:
            if time.time() >= deadline:
                break
            partner_candidates = [rank for rank in branch_order[:partner_limit] if rank != left_rank]
            if left_rank not in branch_order[:partner_limit]:
                partner_candidates.extend(
                    rank
                    for rank in branch_order[partner_limit:partner_limit + 8]
                    if rank != left_rank
                )
            for right_rank in partner_candidates:
                pair = tuple(sorted((int(left_rank), int(right_rank))))
                if pair in tried_pairs:
                    continue
                tried_pairs.add(pair)
                reassignment = {
                    int(pair[0]): int(base_mapping[int(tenant)][pair[1]]),
                    int(pair[1]): int(base_mapping[int(tenant)][pair[0]]),
                }
                delta = self._rank_reassignment_price_delta(
                    base_mapping,
                    tenant,
                    reassignment,
                    task_pair_price,
                    rank_incidence,
                )
                if delta < -1e-12:
                    moves.append((float(delta), reassignment))

        for rank_group in block_groups:
            if time.time() >= deadline:
                break
            current_servers = [int(base_mapping[int(tenant)][rank]) for rank in rank_group]
            for shift in range(1, len(rank_group)):
                reassignment = {
                    int(rank): int(server)
                    for rank, server in zip(rank_group, current_servers[shift:] + current_servers[:shift])
                }
                delta = self._rank_reassignment_price_delta(
                    base_mapping,
                    tenant,
                    reassignment,
                    task_pair_price,
                    rank_incidence,
                )
                if delta < -1e-12:
                    moves.append((float(delta), reassignment))

        moves.sort(key=lambda item: item[0])
        candidates = []
        seen = {self._mapping_signature(base_mapping)}
        first_move_limit = min(anchor_limit, len(moves))
        for _delta, reassignment in moves[:first_move_limit]:
            if time.time() >= deadline:
                break
            candidate = self._copy_mapping(base_mapping)
            for rank, server in reassignment.items():
                candidate[int(tenant)][int(rank)] = int(server)
            candidate = self._normalize_mapping(candidate)
            # One non-greedy first move is often only a bridge.  Continue with
            # the same local descent from that bridge before asking the
            # estimator to score the complete candidate.
            if time.time() < deadline:
                bridge_analysis = self._analyze_mapping(candidate)
                candidate = self._local_block_candidate(
                    candidate,
                    tenant,
                    bridge_analysis,
                    deadline,
                )
            signature = self._mapping_signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            score = self._evaluate_mapping(candidate)
            if self._is_better_objective(score, base_score):
                candidates.append((score, candidate, "local"))
        candidates.sort(key=lambda entry: self._stable_objective_sort_key(entry[0]))
        return candidates

    def _local_structural_moves(self, mapping, tenant, analysis, *, top_price_limit=8):
        task_pair_price = self._task_pair_prices(mapping, tenant, analysis=analysis)
        rank_incidence = self._rank_incidence(tenant)
        branch_order = self._legacy_branch_order(tenant)
        ranks = list(self.rank_orders[int(tenant)])
        anchor_limit = min(len(branch_order), 32)
        partner_limit = min(len(ranks), 64)
        moves = []
        tried_pairs = set()

        for left_rank in branch_order[:anchor_limit]:
            partner_candidates = [rank for rank in branch_order[:partner_limit] if rank != left_rank]
            if left_rank not in branch_order[:partner_limit]:
                partner_candidates.extend(
                    rank
                    for rank in branch_order[partner_limit:partner_limit + 8]
                    if rank != left_rank
                )
            best_for_anchor = None
            for right_rank in partner_candidates:
                pair = tuple(sorted((int(left_rank), int(right_rank))))
                if pair in tried_pairs:
                    continue
                tried_pairs.add(pair)
                reassignment = {
                    int(pair[0]): int(mapping[int(tenant)][pair[1]]),
                    int(pair[1]): int(mapping[int(tenant)][pair[0]]),
                }
                delta = self._rank_reassignment_price_delta(
                    mapping,
                    tenant,
                    reassignment,
                    task_pair_price,
                    rank_incidence,
                )
                if best_for_anchor is None or delta < best_for_anchor[0]:
                    best_for_anchor = (float(delta), reassignment, ("pair", pair))
            if best_for_anchor is not None and best_for_anchor[0] < -1e-12:
                moves.append(best_for_anchor)

        price_ranked_pairs = sorted(moves, key=lambda item: item[0])[:top_price_limit]
        structural_moves = list(price_ranked_pairs)
        seen_reassignments = {
            tuple(sorted(reassignment.items()))
            for _delta, reassignment, _kind in structural_moves
        }

        for rank_group in self._block_move_rank_groups(branch_order, anchor_limit=anchor_limit):
            current_servers = [int(mapping[int(tenant)][rank]) for rank in rank_group]
            for shift in range(1, len(rank_group)):
                reassignment = {
                    int(rank): int(server)
                    for rank, server in zip(rank_group, current_servers[shift:] + current_servers[:shift])
                }
                key = tuple(sorted(reassignment.items()))
                if key in seen_reassignments:
                    continue
                seen_reassignments.add(key)
                delta = self._rank_reassignment_price_delta(
                    mapping,
                    tenant,
                    reassignment,
                    task_pair_price,
                    rank_incidence,
                )
                structural_moves.append((float(delta), reassignment, ("block", tuple(rank_group), shift)))

        structural_moves.sort(key=lambda item: item[0])
        return structural_moves

    def _local_sequence_beam_candidates(self, base_mapping, base_score, tenant, analysis, deadline):
        beam = [(base_score, self._copy_mapping(base_mapping), None)]
        seen_global = {self._mapping_signature(base_mapping)}
        candidates = []
        max_depth = 3
        local_beam_width = 4

        for _depth in range(max_depth):
            expanded = []
            for _score, mapping, _kind in beam:
                if time.time() >= deadline:
                    break
                current_analysis = self._analyze_mapping(mapping)
                moves = self._local_structural_moves(mapping, tenant, current_analysis)
                for _delta, reassignment, kind in moves:
                    if time.time() >= deadline:
                        break
                    candidate = self._copy_mapping(mapping)
                    for rank, server in reassignment.items():
                        candidate[int(tenant)][int(rank)] = int(server)
                    candidate = self._normalize_mapping(candidate)
                    signature = self._mapping_signature(candidate)
                    if signature in seen_global:
                        continue
                    seen_global.add(signature)
                    score = self._evaluate_mapping(candidate)
                    expanded.append((score, candidate, kind))
                    if self._is_better_mapping(score, candidate, base_score, base_mapping):
                        candidates.append((score, candidate, "local"))

            if not expanded:
                break

            expanded = self._rank_entries_by_estimator(expanded)
            next_beam = []
            seen_beam = set()

            def add(entry):
                signature = self._mapping_signature(entry[1])
                if signature in seen_beam:
                    return
                seen_beam.add(signature)
                next_beam.append(entry)

            for entry in expanded:
                add(entry)
                if len(next_beam) >= local_beam_width:
                    break
            for entry in self._rank_entries_by_estimator(expanded):
                add(entry)
                break
            beam = next_beam

        candidates = self._rank_entries_by_estimator(candidates)
        return candidates[: self.beam_width]

    def _best_neighborhood_candidate(
        self,
        base_mapping,
        base_score,
        tenant,
        analysis,
        deadline,
        *,
        include_local=True,
        allow_bnb=False,
        allow_exploration=False,
    ):
        candidates = []
        if include_local:
            source, mapping, _score = self._coarse_local_candidate(
                base_mapping,
                base_score,
                tenant,
                deadline,
            )
            if source is not None:
                candidates.append((source, mapping))
            source, mapping, _score = self._epoch_price_local_candidate(
                base_mapping,
                base_score,
                tenant,
                deadline,
                analysis=analysis,
            )
            if source is not None:
                candidates.append((source, mapping))
        if allow_bnb:
            candidate_deadline = min(deadline, time.time() + self.bnb_candidate_time_limit)
            source, candidate_mapping, _candidate_score = self._price_guided_remap_pool(
                base_mapping,
                base_score,
                tenant,
                analysis,
                candidate_deadline,
            )
            if source is not None:
                candidates.append(("bnb", candidate_mapping))

        best_source = None
        best_mapping = base_mapping
        best_score = base_score
        exploration_source = None
        exploration_mapping = None
        exploration_score = None
        for source, candidate_mapping in candidates:
            if time.time() >= deadline:
                break
            if self._mapping_signature(candidate_mapping) == self._mapping_signature(base_mapping):
                continue
            score = self._evaluate_mapping(candidate_mapping)
            if self._is_better_mapping(score, candidate_mapping, best_score, best_mapping):
                best_source = source
                best_mapping = candidate_mapping
                best_score = score
            elif exploration_mapping is None:
                exploration_source = source
                exploration_mapping = candidate_mapping
                exploration_score = score
        if allow_exploration and best_source is None and exploration_mapping is not None:
            return exploration_source, exploration_mapping, exploration_score
        return best_source, best_mapping, best_score

    def _local_swap_candidates(self, base_mapping, base_score, tenant, analysis, deadline, *, max_candidates=16):
        candidates = []
        base_signature = self._mapping_signature(base_mapping)
        for delta, (left_rank, right_rank) in self._scored_swap_pairs(base_mapping, tenant, analysis)[:max_candidates]:
            if time.time() >= deadline:
                break
            if delta >= -1e-12:
                continue
            candidate = self._copy_mapping(base_mapping)
            candidate[int(tenant)][left_rank], candidate[int(tenant)][right_rank] = (
                candidate[int(tenant)][right_rank],
                candidate[int(tenant)][left_rank],
            )
            candidate = self._normalize_mapping(candidate)
            if self._mapping_signature(candidate) == base_signature:
                continue
            score = self._evaluate_mapping(candidate)
            if self._is_better_mapping(score, candidate, base_score, base_mapping):
                candidates.append((score, candidate, "local"))
        return candidates

    def _price_guided_remap_pool(
        self,
        base_mapping,
        base_score,
        tenant,
        analysis,
        deadline,
        *,
        return_pool=False,
        max_return=4,
    ):
        ranks = list(self.rank_orders[int(tenant)])
        if len(ranks) <= 1 or time.time() >= deadline:
            return [] if return_pool else (None, base_mapping, base_score)

        current_servers = tuple(int(base_mapping[int(tenant)][rank]) for rank in ranks)
        task_infos = self._task_infos(tenant)
        branch_ranks = self._rank_order(analysis, tenant)
        initial_assignment = {int(rank): int(base_mapping[int(tenant)][rank]) for rank in ranks}
        max_price_candidates = 48
        bnb_cache_key = (
            self._mapping_signature(base_mapping),
            int(tenant),
            tuple(int(rank) for rank in ranks),
            tuple(int(rank) for rank in branch_ranks),
        )
        cached_price_candidates = self._bnb_assignment_cache.get(bnb_cache_key)
        if cached_price_candidates is not None:
            pool = []
            best_mapping = base_mapping
            best_score = base_score
            for _candidate_cost, candidate_assignment in cached_price_candidates:
                if time.time() >= deadline:
                    break
                if candidate_assignment == initial_assignment:
                    continue
                candidate = self._copy_mapping(base_mapping)
                candidate[int(tenant)] = {
                    int(rank): int(candidate_assignment[int(rank)])
                    for rank in ranks
                }
                candidate = self._normalize_mapping(candidate)
                if return_pool:
                    pool.append(candidate)
                    if len(pool) >= max_return:
                        break
                    continue
                score = self._evaluate_mapping(candidate)
                if self._is_better_objective(score, best_score):
                    best_mapping = candidate
                    best_score = score
            if return_pool:
                return pool
            if self._mapping_signature(best_mapping) == self._mapping_signature(base_mapping):
                return None, base_mapping, base_score
            return "bnb", best_mapping, best_score
        if len(ranks) > 64:
            return [] if return_pool else (None, base_mapping, base_score)
        if time.time() >= deadline:
            return [] if return_pool else (None, base_mapping, base_score)

        raw_candidates = self.estimator.price_guided_remap_candidates_dense(
            base_mapping,
            int(tenant),
            ranks,
            current_servers,
            branch_ranks,
            task_infos,
            max_price_candidates,
            max(0.0, float(deadline - time.time())),
            analysis=analysis,
        )
        price_candidates = [
            (
                float(candidate_cost),
                {
                    int(rank): int(server)
                    for rank, server in dict(candidate_assignment).items()
                },
            )
            for candidate_cost, candidate_assignment in raw_candidates
        ]
        if not any(candidate_assignment != initial_assignment for _cost, candidate_assignment in price_candidates):
            return [] if return_pool else (None, base_mapping, base_score)
        self._cache_put(
            self._bnb_assignment_cache,
            bnb_cache_key,
            [
                (
                    float(candidate_cost),
                    {
                        int(rank): int(server)
                        for rank, server in candidate_assignment.items()
                    },
                )
                for candidate_cost, candidate_assignment in price_candidates
            ],
            self._cache_limits["bnb_assignment"],
        )
        pool = []
        best_mapping = base_mapping
        best_score = base_score
        for _candidate_cost, candidate_assignment in price_candidates:
            if time.time() >= deadline:
                break
            if candidate_assignment == initial_assignment:
                continue
            candidate = self._copy_mapping(base_mapping)
            candidate[int(tenant)] = {
                int(rank): int(candidate_assignment[int(rank)])
                for rank in ranks
            }
            candidate = self._normalize_mapping(candidate)
            if return_pool:
                pool.append(candidate)
                if len(pool) >= max_return:
                    break
                continue
            score = self._evaluate_mapping(candidate)
            if self._is_better_objective(score, best_score):
                best_mapping = candidate
                best_score = score
        if return_pool:
            return pool
        if self._mapping_signature(best_mapping) == self._mapping_signature(base_mapping):
            return None, base_mapping, base_score
        return "bnb", best_mapping, best_score

    def _projection_bnb_candidate(
        self,
        base_mapping,
        base_score,
        tenant,
        deadline,
        *,
        return_pool=False,
        max_return=4,
    ):
        """Full-tenant remap proposal from epoch-level contention prices.

        This is a proposal generator only: the final candidate is selected by
        the time-expanded estimator.  The search mirrors the MILP structure by
        fixing other tenants and reassigning one tenant's ranks under a
        bijection over its allocated servers.
        """
        raise RuntimeError(
            "Projection BnB is disabled in the time-expanded optimizer; "
            "use the C++ price-guided remap kernel instead."
        )

    def _cyclic_rank_order_candidate(
        self,
        base_mapping,
        base_score,
        tenant,
        analysis,
        deadline,
    ):
        ranks = list(self.rank_orders[int(tenant)])
        if len(ranks) <= 2 or time.time() >= deadline:
            return None, base_mapping, base_score

        current_order = tuple(int(base_mapping[int(tenant)][rank]) for rank in ranks)
        leaf_order = tuple(sorted(current_order, key=self._server_leaf_sort_key))
        source_orders = []
        seen_orders = set()
        for order in (current_order, tuple(reversed(current_order)), leaf_order, tuple(reversed(leaf_order))):
            if order in seen_orders:
                continue
            seen_orders.add(order)
            source_orders.append(order)

        n = len(ranks)
        small_window = min(8, n - 1)
        shifts = set(range(0, small_window + 1))
        shifts.update(n - shift for shift in range(1, small_window + 1))
        shifts.update(
            max(1, round(idx * n / 8))
            for idx in range(1, 8)
            if 0 < round(idx * n / 8) < n
        )

        task_pair_price = self._task_pair_prices(base_mapping, tenant, analysis=analysis)
        price_ranked = []
        seen_signatures = set()
        for order in source_orders:
            for shift in sorted(shifts):
                if time.time() >= deadline:
                    break
                rotated_order = order[shift:] + order[:shift]
                if rotated_order == current_order:
                    continue
                candidate = self._apply_server_order(base_mapping, tenant, rotated_order)
                signature = self._mapping_signature(candidate)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                price_cost = self._tenant_price_cost(
                    base_mapping,
                    tenant,
                    candidate[int(tenant)],
                    task_pair_price,
                )
                price_ranked.append((float(price_cost), candidate))

        if not price_ranked:
            return None, base_mapping, base_score

        price_ranked.sort(key=lambda item: item[0])
        best_mapping = base_mapping
        best_score = base_score
        for _price_cost, candidate in price_ranked[:4]:
            if time.time() >= deadline:
                break
            score = self._evaluate_mapping(candidate)
            if self._is_better_objective(score, best_score):
                best_mapping = candidate
                best_score = score

        if self._mapping_signature(best_mapping) == self._mapping_signature(base_mapping):
            return None, base_mapping, base_score
        return "time_expanded_cyclic", best_mapping, best_score

    def _cyclic_local_candidates(self, base_mapping, base_score, tenant, deadline):
        """Expose the ring-order remaps hidden in the legacy local move.

        The collapsed optimizer canonicalizes ring mappings after a local move;
        this can turn a small price-guided reassignment into a cyclic remap of
        almost the whole ring.  For the standalone time-expanded optimizer we
        make that candidate family explicit and score it directly with the
        estimator.
        """
        if not self._tenant_has_ring_rotation_symmetry(tenant):
            return []
        ranks = list(self.rank_orders[int(tenant)])
        if len(ranks) <= 2:
            return []

        current_order = tuple(int(base_mapping[int(tenant)][rank]) for rank in ranks)
        leaf_order = tuple(sorted(current_order, key=self._server_leaf_sort_key))
        source_orders = []
        seen_orders = set()
        for order in (
            current_order,
            tuple(reversed(current_order)),
            leaf_order,
            tuple(reversed(leaf_order)),
        ):
            if order in seen_orders:
                continue
            seen_orders.add(order)
            source_orders.append(order)

        candidates = []
        seen_signatures = {self._mapping_signature(base_mapping)}
        for order in source_orders:
            for shift in range(1, len(order)):
                if time.time() >= deadline:
                    break
                rotated = order[shift:] + order[:shift]
                candidate = self._apply_server_order(base_mapping, tenant, rotated)
                signature = self._mapping_signature(candidate)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                score = self._evaluate_mapping(candidate)
                if self._is_better_objective(score, base_score):
                    candidates.append((score, candidate, "local"))
        candidates.sort(key=lambda entry: self._stable_objective_sort_key(entry[0]))
        return candidates[:4]

    def _rank_order_matching_candidate(self, base_mapping, base_score, tenant, analysis, deadline):
        ranks = list(self.rank_orders[int(tenant)])
        if len(ranks) <= 2 or time.time() >= deadline:
            return None, base_mapping, base_score

        current_servers = tuple(int(base_mapping[int(tenant)][rank]) for rank in ranks)
        task_pair_price = self._task_pair_prices(base_mapping, tenant, analysis=analysis)
        rank_incidence = self._rank_incidence(tenant)
        anchor_server = min(current_servers)
        ordered_servers = tuple(sorted(current_servers, key=self._server_leaf_sort_key))
        seed_orders = [
            ordered_servers,
            tuple(reversed(ordered_servers)),
            (anchor_server,) + tuple(server for server in reversed(ordered_servers) if int(server) != int(anchor_server)),
        ]

        def incremental_cost(rank, server, assignment, remaining_servers):
            exact = 0.0
            optimistic = 0.0
            remaining_after = [candidate for candidate in remaining_servers if int(candidate) != int(server)]
            for task_id, src_rank, dst_rank, volume in rank_incidence.get(int(rank), []):
                other_rank = dst_rank if src_rank == int(rank) else src_rank
                if other_rank in assignment:
                    if src_rank == int(rank):
                        src_server = int(server)
                        dst_server = int(assignment[other_rank])
                    else:
                        src_server = int(assignment[other_rank])
                        dst_server = int(server)
                    exact += float(volume) * task_pair_price[(task_id, src_server, dst_server)]
                    continue
                if src_rank == int(rank):
                    options = [
                        task_pair_price[(task_id, int(server), int(dst_server))]
                        for dst_server in remaining_after
                        if int(dst_server) != int(server)
                    ]
                else:
                    options = [
                        task_pair_price[(task_id, int(src_server), int(server))]
                        for src_server in remaining_after
                        if int(src_server) != int(server)
                    ]
                if options:
                    optimistic += float(volume) * min(options)
            return float(exact + optimistic), float(exact)

        candidate_mappings = []
        for seed_order in seed_orders:
            if time.time() >= deadline:
                break
            assignment = {int(ranks[0]): int(seed_order[0])}
            used = {int(seed_order[0])}
            for rank in ranks[1:]:
                remaining = [int(server) for server in seed_order if int(server) not in used]
                if not remaining:
                    break
                scored = []
                for server in remaining:
                    score, exact = incremental_cost(rank, server, assignment, remaining)
                    scored.append((score, exact, int(server)))
                scored.sort(key=lambda item: (item[0], item[1], item[2]))
                _score, _exact, chosen = scored[0]
                assignment[int(rank)] = int(chosen)
                used.add(int(chosen))
            if len(assignment) != len(ranks):
                continue
            candidate = self._copy_mapping(base_mapping)
            candidate[int(tenant)] = {rank: int(assignment[int(rank)]) for rank in ranks}
            candidate = self._normalize_mapping(candidate)
            if self._mapping_signature(candidate) != self._mapping_signature(base_mapping):
                candidate_mappings.append(candidate)

        best_mapping = base_mapping
        best_score = base_score
        for candidate in candidate_mappings:
            if time.time() >= deadline:
                break
            score = self._evaluate_mapping(candidate)
            if self._is_better_objective(score, best_score):
                best_mapping = candidate
                best_score = score
        if self._mapping_signature(best_mapping) == self._mapping_signature(base_mapping):
            return None, base_mapping, base_score
        return "bnb", best_mapping, best_score

    def _effective_bnb_tenant_count(self, tenant_order):
        return min(len(tenant_order), max(0, self.max_bnb_tenants_per_round))

    def _effective_joint_tenants(self, tenant_order):
        return tenant_order[: max(2, self.max_bnb_tenants_per_round)]

    def _joint_candidates(self, base_mapping, base_score, tenant_order, analysis, deadline):
        joint_tenants = self._effective_joint_tenants(tenant_order)
        if len(joint_tenants) < 2:
            return []

        pools = []
        for tenant in joint_tenants:
            if time.time() >= deadline:
                break
            candidate_deadline = min(deadline, time.time() + self.bnb_candidate_time_limit)
            pool = [
                candidate
                for _score, candidate, _source in self._local_swap_candidates(
                    base_mapping,
                    base_score,
                    tenant,
                    analysis,
                    candidate_deadline,
                    max_candidates=3,
                )
            ]
            pool.extend(
                self._price_guided_remap_pool(
                    base_mapping,
                    base_score,
                    tenant,
                    analysis,
                    candidate_deadline,
                    return_pool=True,
                    max_return=3,
                )
            )
            if not pool:
                pool = [base_mapping]
            pools.append((tenant, pool))

        if len(pools) < 2:
            return []

        base_signature = self._mapping_signature(base_mapping)
        seen = set()
        candidates = []
        for combination in itertools.product(*(pool for _tenant, pool in pools)):
            if time.time() >= deadline:
                break
            joint_mapping = self._copy_mapping(base_mapping)
            changed = False
            for (tenant, _pool), candidate_mapping in zip(pools, combination):
                if self._mapping_signature(candidate_mapping) == base_signature:
                    continue
                joint_mapping[int(tenant)] = dict(candidate_mapping[int(tenant)])
                changed = True
            if not changed:
                continue
            joint_mapping = self._normalize_mapping(joint_mapping)
            signature = self._mapping_signature(joint_mapping)
            if signature == base_signature or signature in seen:
                continue
            seen.add(signature)
            score = self._evaluate_mapping(joint_mapping)
            if self._is_better_objective(score, base_score):
                candidates.append((score, joint_mapping, "joint"))
        return candidates

    def _joint_pair_candidate(self, base_mapping, tenant_a, tenant_b, analysis, deadline):
        candidates_a = []
        candidates_b = []

        source_a, local_a, _score_a = self._best_neighborhood_candidate(
            base_mapping,
            self._evaluate_mapping(base_mapping),
            tenant_a,
            analysis,
            deadline,
            include_local=True,
            allow_bnb=False,
        )
        if source_a is not None:
            candidates_a.append(local_a)

        source_b, local_b, _score_b = self._best_neighborhood_candidate(
            base_mapping,
            self._evaluate_mapping(base_mapping),
            tenant_b,
            analysis,
            deadline,
            include_local=True,
            allow_bnb=False,
        )
        if source_b is not None:
            candidates_b.append(local_b)

        if self.max_bnb_tenants_per_round > 0:
            source_a, bnb_a, _score_a = self._price_guided_remap_pool(
                base_mapping,
                self._evaluate_mapping(base_mapping),
                tenant_a,
                analysis,
                min(deadline, time.time() + self.bnb_candidate_time_limit),
            )
            if source_a is not None:
                candidates_a.append(bnb_a)

            source_b, bnb_b, _score_b = self._price_guided_remap_pool(
                base_mapping,
                self._evaluate_mapping(base_mapping),
                tenant_b,
                analysis,
                min(deadline, time.time() + self.bnb_candidate_time_limit),
            )
            if source_b is not None:
                candidates_b.append(bnb_b)

        if not candidates_a or not candidates_b:
            return []

        seen = set()
        joint_candidates = []
        for candidate_a in candidates_a:
            for candidate_b in candidates_b:
                joint_mapping = self._copy_mapping(base_mapping)
                joint_mapping[int(tenant_a)] = dict(candidate_a[int(tenant_a)])
                joint_mapping[int(tenant_b)] = dict(candidate_b[int(tenant_b)])
                joint_mapping = self._normalize_mapping(joint_mapping)
                signature = self._mapping_signature(joint_mapping)
                if signature in seen:
                    continue
                seen.add(signature)
                joint_candidates.append(joint_mapping)
        return joint_candidates

    def _time_expanded_pair_swap_polish(self, base_mapping, base_score, deadline):
        """Final deterministic descent over all same-tenant rank swaps."""
        best_mapping = self._copy_mapping(base_mapping)
        best_score = tuple(float(value) for value in base_score)
        while time.time() < deadline:
            round_best = None
            round_best_score = best_score
            for tenant in self.tenants:
                if time.time() >= deadline:
                    break
                ranks = list(self.rank_orders[int(tenant)])
                for idx, left_rank in enumerate(ranks):
                    if time.time() >= deadline:
                        break
                    for right_rank in ranks[idx + 1:]:
                        if time.time() >= deadline:
                            break
                        candidate = self._copy_mapping(best_mapping)
                        candidate[int(tenant)][int(left_rank)], candidate[int(tenant)][int(right_rank)] = (
                            candidate[int(tenant)][int(right_rank)],
                            candidate[int(tenant)][int(left_rank)],
                        )
                        candidate = self._normalize_mapping(candidate)
                        if self._mapping_signature(candidate) == self._mapping_signature(best_mapping):
                            continue
                        candidate_score = self._evaluate_mapping(candidate)
                        if self._is_better_mapping(candidate_score, candidate, round_best_score, best_mapping):
                            round_best = candidate
                            round_best_score = candidate_score

            if round_best is None:
                break
            best_mapping = round_best
            best_score = round_best_score
            self.move_source_counts["polish"] = self.move_source_counts.get("polish", 0) + 1

        return best_mapping, best_score

    def _tabu_swap_refinement(self, base_mapping, base_score, deadline):
        """Black-box refinement over estimator-scored rank swaps.

        The estimator supplies the objective and pressure ordering.  Tabu
        memory prevents immediate backtracking, while aspiration allows any
        move that improves the historical best.  This can cross shallow
        multi-tenant basins without importing the collapsed hybrid solution.
        """
        if time.time() >= deadline:
            return base_mapping, base_score
        current_mapping = self._copy_mapping(base_mapping)
        current_score = tuple(float(value) for value in base_score)
        best_mapping = self._copy_mapping(base_mapping)
        best_score = tuple(float(value) for value in base_score)
        tabu_until = {}
        iteration = 0
        stale_iterations = 0
        max_stale = 12
        tenure = max(4, min(12, 2 * len(self.tenants)))

        while time.time() < deadline and stale_iterations < max_stale:
            iteration += 1
            analysis = self._analyze_mapping(current_mapping)
            tenant_order = self._tenant_order(analysis)
            neighbor_pool = []
            for tenant in tenant_order:
                if time.time() >= deadline:
                    break
                for _delta, (left_rank, right_rank) in self._scored_swap_pairs(
                    current_mapping,
                    tenant,
                    analysis,
                    hot_limit=8,
                    partner_limit=24,
                )[:8]:
                    candidate = self._copy_mapping(current_mapping)
                    candidate[int(tenant)][int(left_rank)], candidate[int(tenant)][int(right_rank)] = (
                        candidate[int(tenant)][int(right_rank)],
                        candidate[int(tenant)][int(left_rank)],
                    )
                    candidate = self._normalize_mapping(candidate)
                    signature = self._mapping_signature(candidate)
                    move_key = (int(tenant), min(int(left_rank), int(right_rank)), max(int(left_rank), int(right_rank)))
                    score = self._evaluate_mapping(candidate)
                    is_aspiration = self._is_better_mapping(score, candidate, best_score, best_mapping)
                    if tabu_until.get(move_key, -1) > iteration and not is_aspiration:
                        continue
                    neighbor_pool.append((score, candidate, move_key))

            if not neighbor_pool:
                break

            neighbor_pool = self._rank_entries_by_estimator([
                (score, mapping, move_key)
                for score, mapping, move_key in neighbor_pool
            ])
            next_score, next_mapping, move_key = neighbor_pool[0]
            current_mapping = next_mapping
            current_score = next_score
            tabu_until[move_key] = iteration + tenure

            if self._is_better_mapping(current_score, current_mapping, best_score, best_mapping):
                best_mapping = self._copy_mapping(current_mapping)
                best_score = current_score
                stale_iterations = 0
                self.move_source_counts["tabu"] = self.move_source_counts.get("tabu", 0) + 1
            else:
                stale_iterations += 1

        return best_mapping, best_score

    def _solve_hybrid_skeleton(self, time_limit=None):
        start_time = time.time()
        deadline = float("inf") if time_limit is None else start_time + float(time_limit)
        self.move_source_counts = {"local": 0, "bnb": 0, "joint": 0}
        self.last_move_source = None

        best_mapping = None
        best_score = (float("inf"), float("inf"))
        initial_candidates = []
        for seed_mapping in self._seed_mappings():
            if time.time() >= deadline:
                break
            score = self._evaluate_mapping(seed_mapping)
            initial_candidates.append((score, seed_mapping, None))
            if self._is_better_objective(score, best_score):
                best_mapping = seed_mapping
                best_score = score

        if best_mapping is None:
            best_mapping = self._copy_mapping(self.initial_tenant_mapping)
            best_score = self._evaluate_mapping(best_mapping)
            initial_candidates = [(best_score, best_mapping, None)]

        # Keep the search skeleton identical to the collapsed hybrid beam
        # search.  The only intended difference is the evaluator/signals used
        # to score and order candidates.
        beam = self._dedupe_ranked_candidates_hybrid_order(initial_candidates)
        if not beam:
            beam = [(best_score, best_mapping, None)]
        best_score, best_mapping, _best_source = beam[0]

        round_idx = 0
        while round_idx < self.max_price_rounds and time.time() < deadline:
            round_idx += 1
            improved = False
            next_candidates = list(beam)

            for beam_score, beam_mapping, _beam_source in beam:
                if time.time() >= deadline:
                    break
                analysis = self._analyze_mapping(beam_mapping)
                tenant_order = self._tenant_order(analysis)

                for tenant in tenant_order:
                    if time.time() >= deadline:
                        break
                    source, candidate_mapping, candidate_score = self._best_neighborhood_candidate(
                        beam_mapping,
                        beam_score,
                        tenant,
                        analysis,
                        deadline,
                        include_local=True,
                        allow_bnb=False,
                    )
                    if source is None:
                        continue
                    next_candidates.append((candidate_score, candidate_mapping, f"{source}:t{tenant}"))

                if self.max_bnb_tenants_per_round > 0 and time.time() < deadline:
                    for tenant in tenant_order[: self._effective_bnb_tenant_count(tenant_order)]:
                        if time.time() >= deadline:
                            break
                        candidate_deadline = min(deadline, time.time() + self.bnb_candidate_time_limit)
                        source, candidate_mapping, candidate_score = self._price_guided_remap_pool(
                            beam_mapping,
                            beam_score,
                            tenant,
                            analysis,
                            candidate_deadline,
                        )
                        if source is None:
                            continue
                        next_candidates.append((candidate_score, candidate_mapping, f"{source}:t{tenant}"))

                if len(tenant_order) >= 2 and time.time() < deadline:
                    joint_tenants = self._effective_joint_tenants(tenant_order)
                    for idx, tenant_a in enumerate(joint_tenants):
                        if time.time() >= deadline:
                            break
                        for tenant_b in joint_tenants[idx + 1:]:
                            if time.time() >= deadline:
                                break
                            for candidate_mapping in self._joint_pair_candidate(
                                beam_mapping,
                                tenant_a,
                                tenant_b,
                                analysis,
                                deadline,
                            ):
                                if time.time() >= deadline:
                                    break
                                candidate_score = self._evaluate_mapping(candidate_mapping)
                                if self._is_better_mapping(candidate_score, candidate_mapping, beam_score, beam_mapping):
                                    next_candidates.append((candidate_score, candidate_mapping, f"joint:t{tenant_a}-{tenant_b}"))

            ranked = self._dedupe_ranked_candidates_hybrid_order(next_candidates)
            if not ranked:
                break

            top_score, top_mapping, top_source = ranked[0]
            if self._is_better_mapping(top_score, top_mapping, best_score, best_mapping):
                improved = True
                best_mapping = top_mapping
                best_score = top_score
                self.last_move_source = top_source
                if top_source is not None:
                    source_family = str(top_source).split(":", 1)[0]
                    self.move_source_counts[source_family] = self.move_source_counts.get(source_family, 0) + 1

            beam = ranked
            if not improved:
                break

        self.search_rounds = round_idx
        self.final_mapping = self._copy_mapping(best_mapping)
        self.final_obj = float(best_score[0])
        self.final_makespan = float(best_score[0])
        self.final_avg_jct = float(best_score[1])
        self.final_score_is_simulated = False
        self.runtime_seconds = time.time() - start_time

        if self.verbose:
            print(
                f"Time-expanded mapping solve complete: Makespan={best_score[0]:.12f}, "
                f"AvgJCT={best_score[1]:.12f}, "
                f"Moves={self.move_source_counts}, Runtime={self.runtime_seconds:.2f}s"
            )
        self._clear_search_caches(keep_scores=False)
        return self

    def _solve_diverse_beam(self, time_limit=None):
        start_time = time.time()
        deadline = float("inf") if time_limit is None else start_time + float(time_limit)
        self.move_source_counts = {"local": 0, "bnb": 0, "joint": 0}
        self.last_move_source = None
        if time_limit is None:
            search_deadline = deadline
        else:
            search_deadline = deadline

        best_mapping = None
        best_score = (float("inf"), float("inf"))
        initial_candidates = []
        for seed_mapping in self._seed_mappings():
            if time.time() >= search_deadline:
                break
            score = self._evaluate_mapping(seed_mapping)
            initial_candidates.append((score, seed_mapping, None))
            if self._is_better_objective(score, best_score):
                best_mapping = seed_mapping
                best_score = score

        if best_mapping is None:
            best_mapping = self._copy_mapping(self.initial_tenant_mapping)
            best_score = self._evaluate_mapping(best_mapping)
            initial_candidates = [(best_score, best_mapping, None)]

        beam = self._dedupe_ranked_candidates(initial_candidates)
        if not beam:
            beam = [(best_score, best_mapping, None)]
        initial_beam = []
        seen_initial_score_buckets = set()
        for entry in beam:
            bucket = self._stable_objective_sort_key(entry[0])
            if bucket in seen_initial_score_buckets:
                continue
            seen_initial_score_buckets.add(bucket)
            initial_beam.append(entry)
        if initial_beam:
            beam = initial_beam
        best_score, best_mapping, _best_source = beam[0]

        round_idx = 0
        stagnation_rounds = 0
        while round_idx < self.max_price_rounds and time.time() < search_deadline:
            round_idx += 1
            improved = False
            next_candidates = list(beam)
            previous_beam_signatures = {
                self._mapping_signature(mapping)
                for _score, mapping, _source in beam
            }

            for beam_score, beam_mapping, _beam_source in beam:
                if time.time() >= search_deadline:
                    break
                analysis = self._analyze_mapping(beam_mapping)
                tenant_order = self._tenant_order(analysis)
                local_improved = False
                bnb_improved = False

                for tenant in tenant_order:
                    if time.time() >= search_deadline:
                        break
                    source, candidate_mapping, candidate_score = self._best_neighborhood_candidate(
                        beam_mapping,
                        beam_score,
                        tenant,
                        analysis,
                        search_deadline,
                        include_local=True,
                        allow_bnb=False,
                        allow_exploration=True,
                    )
                    if source is not None:
                        if self._is_better_mapping(candidate_score, candidate_mapping, beam_score, beam_mapping):
                            local_improved = True
                        next_candidates.append((candidate_score, candidate_mapping, f"{source}:t{tenant}"))

                if (not local_improved) and self.max_bnb_tenants_per_round > 0 and time.time() < search_deadline:
                    for tenant in tenant_order[: self._effective_bnb_tenant_count(tenant_order)]:
                        if time.time() >= search_deadline:
                            break
                        candidate_deadline = min(search_deadline, time.time() + self.bnb_candidate_time_limit)
                        source, candidate_mapping, candidate_score = self._price_guided_remap_pool(
                            beam_mapping,
                            beam_score,
                            tenant,
                            analysis,
                            candidate_deadline,
                        )
                        if source is not None:
                            if self._is_better_mapping(candidate_score, candidate_mapping, beam_score, beam_mapping):
                                bnb_improved = True
                            next_candidates.append((candidate_score, candidate_mapping, f"{source}:t{tenant}"))

                if (not local_improved and not bnb_improved) and len(tenant_order) >= 2 and time.time() < search_deadline:
                    joint_tenants = self._effective_joint_tenants(tenant_order)
                    for idx, tenant_a in enumerate(joint_tenants):
                        if time.time() >= search_deadline:
                            break
                        for tenant_b in joint_tenants[idx + 1:]:
                            if time.time() >= search_deadline:
                                break
                            for candidate_mapping in self._joint_pair_candidate(
                                beam_mapping,
                                tenant_a,
                                tenant_b,
                                analysis,
                                search_deadline,
                            ):
                                if time.time() >= search_deadline:
                                    break
                                candidate_score = self._evaluate_mapping(candidate_mapping)
                                if self._is_better_mapping(candidate_score, candidate_mapping, beam_score, beam_mapping):
                                    next_candidates.append((candidate_score, candidate_mapping, f"joint:t{tenant_a}-{tenant_b}"))

            if time.time() < search_deadline and len(next_candidates) > len(beam):
                recombination_anchor = min(
                    next_candidates,
                    key=lambda entry: self._stable_objective_sort_key(entry[0]),
                )
                recombination_analysis = self._analyze_mapping(recombination_anchor[1])
                next_candidates.extend(
                    self._tenant_block_recombination_candidates(
                        next_candidates,
                        recombination_analysis,
                        search_deadline,
                    )
                )

            ranked = self._dedupe_ranked_candidates(next_candidates)
            if not ranked:
                break

            top_score, top_mapping, top_source = ranked[0]
            if self._is_better_mapping(top_score, top_mapping, best_score, best_mapping):
                improved = True
                best_mapping = top_mapping
                best_score = top_score
                self.last_move_source = top_source
                if top_source is not None:
                    source_family = str(top_source).split(":", 1)[0]
                    self.move_source_counts[source_family] = self.move_source_counts.get(source_family, 0) + 1

            beam = ranked
            new_beam_signatures = {
                self._mapping_signature(mapping)
                for _score, mapping, _source in beam
            }
            if improved:
                stagnation_rounds = 0
            elif new_beam_signatures - previous_beam_signatures:
                stagnation_rounds += 1
            else:
                break

            if not improved and stagnation_rounds > self.max_stagnation_rounds:
                break

        if time.time() < search_deadline:
            best_mapping, best_score = self._tabu_swap_refinement(
                best_mapping,
                best_score,
                search_deadline,
            )

        self.search_rounds = round_idx
        self.final_mapping = self._copy_mapping(best_mapping)
        self.final_obj = float(best_score[0])
        self.final_makespan = float(best_score[0])
        self.final_avg_jct = float(best_score[1])
        self.final_score_is_simulated = False
        self.runtime_seconds = time.time() - start_time

        if self.verbose:
            print(
                f"Time-expanded mapping solve complete: Makespan={best_score[0]:.12f}, "
                f"AvgJCT={best_score[1]:.12f}, "
                f"Moves={self.move_source_counts}, Runtime={self.runtime_seconds:.2f}s"
            )
        self._clear_search_caches(keep_scores=False)
        return self

    def _solution_snapshot(self):
        return {
            "mapping": self._copy_mapping(self.final_mapping),
            "score": (float(self.final_makespan), float(self.final_avg_jct)),
            "rounds": int(self.search_rounds),
            "moves": dict(self.move_source_counts),
            "last_move_source": self.last_move_source,
            "runtime": float(self.runtime_seconds),
            "tolerance": float(self.score_sort_tolerance),
        }

    def _restore_solution_snapshot(self, snapshot, total_runtime=None):
        self.final_mapping = self._copy_mapping(snapshot["mapping"])
        self.final_obj = float(snapshot["score"][0])
        self.final_makespan = float(snapshot["score"][0])
        self.final_avg_jct = float(snapshot["score"][1])
        self.final_score_is_simulated = False
        self.search_rounds = int(snapshot["rounds"])
        self.move_source_counts = dict(snapshot["moves"])
        self.last_move_source = snapshot.get("last_move_source")
        self.runtime_seconds = float(snapshot["runtime"] if total_runtime is None else total_runtime)
        return self

    def _solve_precision_portfolio(self, time_limit):
        start_time = time.time()
        original_tolerance = float(self.score_sort_tolerance)
        attempt_budget = max(1.0, float(time_limit) / 2.0)
        snapshots = []
        for tolerance in (1e-5, 1e-8):
            self.score_sort_tolerance = float(tolerance)
            self._solve_diverse_beam(time_limit=attempt_budget)
            snapshots.append(self._solution_snapshot())
            self._clear_search_caches(keep_scores=True)

        self.score_sort_tolerance = original_tolerance
        best = min(
            snapshots,
            key=lambda snapshot: (
                float(snapshot["score"][1]),
                float(snapshot["score"][0]),
            ),
        )
        result = self._restore_solution_snapshot(
            best,
            total_runtime=time.time() - start_time,
        )
        self._clear_search_caches(keep_scores=False)
        return result

    def solve(self, time_limit=None):
        if self.search_skeleton in {"hybrid", "legacy-hybrid", "collapsed-hybrid"}:
            return self._solve_hybrid_skeleton(time_limit=time_limit)
        if time_limit is not None and float(time_limit) >= 2.0:
            return self._solve_precision_portfolio(time_limit=time_limit)
        return self._solve_diverse_beam(time_limit=time_limit)

    def get_X_mapping(self):
        if self.final_mapping is None:
            return self._copy_mapping(self.initial_tenant_mapping)
        return self._copy_mapping(self.final_mapping)
