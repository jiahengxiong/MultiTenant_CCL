from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contention_estimator_core import TimeExpandedEstimatorCore

try:
    from . import _te_accel
except ImportError:  # pragma: no cover - optional C++ accelerator.
    _te_accel = None


Mapping = dict[int, dict[int, int]]
Objective = tuple[float, float]


@dataclass(frozen=True)
class ContentionEstimate:
    """Estimated completion objective for one rank mapping."""

    makespan: float
    avg_jct: float
    tenant_finish: dict[int, float]
    slot_duration: float

    @property
    def score(self) -> Objective:
        return (self.makespan, self.avg_jct)


@dataclass(frozen=True)
class ContentionAnalysis:
    """Time-expanded contention signals exposed to black-box optimizers."""

    estimate: ContentionEstimate
    task_state: dict[tuple[int, int], dict[str, Any]]
    task_active_slots: dict[tuple[int, int], set[int]]
    task_ready_slots: dict[tuple[int, int], set[int]]
    epoch_active_slots: list[set[int]]
    slot_ready_tasks: list[set[tuple[int, int]]]
    slot_prices: list[dict[str, dict[Any, float]]]
    slot_resource_pressure: list[dict[str, dict[Any, float]]]
    critical_tasks: set[tuple[int, int]]
    tenant_pressure: dict[int, float]
    tenant_peak_load: dict[int, float]
    task_pressure: dict[tuple[int, int], float]
    rank_pressure: dict[tuple[int, int], float]
    tenant_pair_interaction: dict[tuple[int, int], float]
    hotspots: list[dict[str, Any]]
    contention_clusters: list[dict[str, Any]]


class _EstimatorBackbone(TimeExpandedEstimatorCore):
    """Reuse the compiled DAG and MILP-derived estimator without solver policy."""

    def _canonicalize_ring_mapping(self, mapping):  # noqa: D401
        """Preserve the physical mapping exactly.

        Ring rotations are logically equivalent inside a CCL policy, but they
        are not equivalent once the logical policy is materialized on fixed
        server-level ECMP paths.  The estimator therefore must not collapse
        rotations.
        """
        return {
            int(tenant): {
                int(rank): int(server)
                for rank, server in rank_to_server.items()
            }
            for tenant, rank_to_server in mapping.items()
        }


class TimeExpandedContentionEstimator:
    """MILP-derived time-expanded contention estimator.

    The estimator is intentionally independent from any rank-mapping search
    policy.  It consumes a task DAG compiled from collective programs, a fixed
    topology, a tenant-aware ECMP path table, and a candidate rank mapping.  It
    then propagates task readiness through the DAG while sharing sender,
    receiver, and link capacities among simultaneously active tasks.
    """

    def __init__(
        self,
        datacenter,
        *,
        tenant_mapping: Mapping,
        tenant_collective_specs=None,
        tenant_collective_programs=None,
        tenant_start_times=None,
        path_table=None,
        slot_duration=None,
        horizon_slots=None,
        name="time_expanded_contention_estimator",
    ):
        self._backbone = _EstimatorBackbone(
            datacenter,
            tenant_mapping=tenant_mapping,
            tenant_collective_specs=tenant_collective_specs,
            tenant_collective_programs=tenant_collective_programs,
            tenant_start_times=tenant_start_times,
            path_table=path_table,
            slot_duration=slot_duration,
            horizon_slots=horizon_slots,
            surrogate_mode="time_expanded",
            validate_with_simulator=False,
            verbose=False,
            name=name,
        )
        self._dense_static_price_cache = {}
        self._dense_path_edges_cache = {}

    @property
    def tenants(self) -> list[int]:
        return list(self._backbone.tenants)

    @property
    def rank_orders(self) -> dict[int, list[int]]:
        return {
            int(tenant): [int(rank) for rank in ranks]
            for tenant, ranks in self._backbone.rank_orders.items()
        }

    @property
    def server_sets(self) -> dict[int, tuple[int, ...]]:
        return {
            int(tenant): tuple(int(server) for server in servers)
            for tenant, servers in self._backbone.server_sets.items()
        }

    @property
    def task_dag(self) -> dict[int, dict[str, Any]]:
        return self._backbone.data.get("task_surrogate", {})

    def normalize_mapping(self, mapping: Mapping) -> Mapping:
        return self._backbone._canonicalize_ring_mapping(mapping)

    def signature(self, mapping: Mapping):
        return self._backbone._mapping_signature(mapping)

    def evaluate(self, mapping: Mapping) -> ContentionEstimate:
        normalized = self.normalize_mapping(mapping)
        state = self._backbone._compute_time_expanded_surrogate_state(
            normalized,
            collect_signals=False,
        )
        makespan, avg_jct = state["score"]
        return ContentionEstimate(
            makespan=float(makespan),
            avg_jct=float(avg_jct),
            tenant_finish={
                int(tenant): float(value)
                for tenant, value in state.get("tenant_finish", {}).items()
            },
            slot_duration=float(state.get("slot_duration", 0.0)),
        )

    def analyze(self, mapping: Mapping) -> ContentionAnalysis:
        normalized = self.normalize_mapping(mapping)
        state = self._backbone._compute_time_expanded_surrogate_state(
            normalized,
            collect_signals=True,
        )
        return self._analysis_from_state(state)

    def analyze_for_search(self, mapping: Mapping) -> ContentionAnalysis:
        """Return optimizer-facing signals without retaining debug trajectories."""
        normalized = self.normalize_mapping(mapping)
        state = self._backbone._compute_time_expanded_surrogate_state(
            normalized,
            collect_signals=True,
            include_hotspots=False,
        )
        return self._analysis_from_state(state, lightweight=True)

    def search_state(self, mapping: Mapping) -> dict[str, Any]:
        """Return the lightweight state used by search-side price aggregation."""
        normalized = self.normalize_mapping(mapping)
        return self._backbone._compute_time_expanded_surrogate_state(
            normalized,
            collect_signals=True,
            include_hotspots=False,
        )

    def _analysis_from_state(
        self,
        state: dict[str, Any],
        *,
        lightweight: bool = False,
    ) -> ContentionAnalysis:
        makespan, avg_jct = state["score"]
        estimate = ContentionEstimate(
            makespan=float(makespan),
            avg_jct=float(avg_jct),
            tenant_finish={
                int(tenant): float(value)
                for tenant, value in state.get("tenant_finish", {}).items()
            },
            slot_duration=float(state.get("slot_duration", 0.0)),
        )
        if lightweight:
            # Search only needs exposure windows, slot prices, and aggregated
            # pressure signals.  Dropping full trajectories avoids retaining
            # multi-GB time-expanded states in optimizer caches.
            task_state = {}
            slot_ready_tasks = []
            slot_resource_pressure = []
            critical_tasks = set()
            task_pressure = {}
            hotspots = []
            contention_clusters = []
        else:
            task_state = state.get("task_state", {})
            slot_ready_tasks = list(state.get("slot_ready_tasks", []))
            slot_resource_pressure = state.get("slot_resource_pressure", [])
            critical_tasks = set(state.get("critical_tasks", set()))
            task_pressure = {
                (int(tenant), int(task_id)): float(value)
                for (tenant, task_id), value in state.get("task_pressure", {}).items()
            }
            hotspots = list(state.get("hotspots", []))
            contention_clusters = list(state.get("contention_clusters", []))
        return ContentionAnalysis(
            estimate=estimate,
            task_state=task_state,
            task_active_slots=state.get("task_active_slots", {}),
            task_ready_slots=state.get("task_ready_slots", {}),
            epoch_active_slots=[
                set(int(slot) for slot in slot_set)
                for slot_set in state.get("epoch_active_slots", [])
            ],
            slot_ready_tasks=slot_ready_tasks,
            slot_prices=state.get("slot_prices", []),
            slot_resource_pressure=slot_resource_pressure,
            critical_tasks=critical_tasks,
            tenant_pressure={
                int(tenant): float(value)
                for tenant, value in state.get("tenant_pressure", {}).items()
            },
            tenant_peak_load={
                int(tenant): float(value)
                for tenant, value in state.get("tenant_peak_load", {}).items()
            },
            task_pressure=task_pressure,
            rank_pressure={
                (int(tenant), int(rank)): float(value)
                for (tenant, rank), value in state.get("rank_pressure", {}).items()
            },
            tenant_pair_interaction={
                (int(left), int(right)): float(value)
                for (left, right), value in state.get("tenant_pair_interaction", {}).items()
            },
            hotspots=hotspots,
            contention_clusters=contention_clusters,
        )

    def epoch_prices_from_analysis(self, analysis: ContentionAnalysis):
        """Aggregate slot prices into collective-epoch prices without recomputing state."""
        state = {
            "slot_prices": analysis.slot_prices,
            "epoch_active_slots": analysis.epoch_active_slots,
            "task_active_slots": analysis.task_active_slots,
            "task_state": analysis.task_state,
        }
        return self._backbone._time_expanded_epoch_prices_from_state(state)

    def audit_release_gates(self) -> dict[int, dict[int, tuple[tuple[int, ...], float]]]:
        """Expose release gates for debug audits."""
        return {
            int(tenant): {
                int(task_id): (
                    tuple(int(prev) for prev in previous_tasks),
                    float(gap),
                )
                for task_id, (previous_tasks, gap) in meta.get("release_gates", {}).items()
            }
            for tenant, meta in self.task_dag.items()
        }

    def path_edges_for_pair(self, tenant: int, src_server: int, dst_server: int):
        """Return the fixed ECMP path edges for one tenant/server pair."""
        return tuple(
            self._backbone._path_edges_for_pair(
                self._backbone.data["path_edges"],
                int(tenant),
                int(src_server),
                int(dst_server),
            )
        )

    def task_pair_price_lookup(
        self,
        mapping: Mapping,
        tenant: int,
        candidate_servers,
        *,
        analysis: ContentionAnalysis | None = None,
    ) -> dict[tuple[int, int, int], float]:
        """Price candidate server pairs using current time-expanded slot prices.

        The optimizer uses this as a cheap proposal signal before calling the
        full estimator on a complete candidate mapping.
        """
        mapping = self.normalize_mapping(mapping)
        if analysis is None:
            analysis = self.analyze(mapping)

        task_active_slots = analysis.task_active_slots
        task_ready_slots = analysis.task_ready_slots
        slot_prices = analysis.slot_prices
        task_meta = self.task_dag[int(tenant)]
        path_edges = self._backbone.data["path_edges"]
        server_send_capacity = self._backbone.data["server_send_capacity"]
        server_recv_capacity = self._backbone.data["server_recv_capacity"]
        servers = tuple(int(server) for server in candidate_servers)

        return self._accelerated_task_pair_price_lookup(
            int(tenant),
            servers,
            task_meta,
            path_edges,
            server_send_capacity,
            server_recv_capacity,
            analysis,
        )

    def _accelerated_task_pair_price_lookup(
        self,
        tenant: int,
        servers: tuple[int, ...],
        task_meta: dict[str, Any],
        path_edges,
        server_send_capacity,
        server_recv_capacity,
        analysis: ContentionAnalysis,
    ) -> dict[tuple[int, int, int], float] | None:
        if _te_accel is None:
            raise RuntimeError("C++ task-pair price lookup kernel is required")

        edge_capacity = self._backbone.data["edge_capacity"]
        server_count = self._dense_server_count(
            servers,
            server_send_capacity,
            server_recv_capacity,
        )
        edge_items, edge_to_idx, base_sender, base_receiver, base_edge = (
            self._dense_static_price_vectors(
                server_count,
                edge_capacity,
                server_send_capacity,
                server_recv_capacity,
            )
        )

        slot_count = max(len(analysis.slot_prices), 1)
        dense_entry_count = int(slot_count) * (
            int(server_count) * 2 + len(edge_items)
        )
        if dense_entry_count > 2_000_000:
            raise RuntimeError(
                f"task-pair price dense state too large: {dense_entry_count} entries"
            )
        sender_prices = [list(base_sender) for _slot in range(slot_count)]
        receiver_prices = [list(base_receiver) for _slot in range(slot_count)]
        edge_prices = [list(base_edge) for _slot in range(slot_count)]
        for slot_idx, price_state in enumerate(analysis.slot_prices[:slot_count]):
            for server, price in price_state.get("sender", {}).items():
                if 0 <= int(server) < server_count:
                    sender_prices[slot_idx][int(server)] = float(price)
            for server, price in price_state.get("receiver", {}).items():
                if 0 <= int(server) < server_count:
                    receiver_prices[slot_idx][int(server)] = float(price)
            for edge, price in price_state.get("edge", {}).items():
                edge_idx = edge_to_idx.get(edge)
                if edge_idx is not None:
                    edge_prices[slot_idx][edge_idx] = float(price)

        path_edges_by_pair = self._dense_path_edges_by_pair(
            int(tenant),
            server_count,
            path_edges,
            edge_to_idx,
        )

        task_ids = []
        exposure_slots_by_task = []
        for task_id in task_meta["task_info"]:
            task_id = int(task_id)
            exposure_slots = sorted(
                set(int(slot) for slot in analysis.task_active_slots.get((int(tenant), task_id), set()))
                | set(int(slot) for slot in analysis.task_ready_slots.get((int(tenant), task_id), set()))
            )
            if not exposure_slots:
                exposure_slots = [0]
            task_ids.append(task_id)
            exposure_slots_by_task.append(exposure_slots)

        accelerated = _te_accel.task_pair_price_lookup_dense(
            task_ids,
            exposure_slots_by_task,
            list(servers),
            sender_prices,
            receiver_prices,
            edge_prices,
            path_edges_by_pair,
        )
        return {
            (int(task_id), int(src_server), int(dst_server)): float(price)
            for (task_id, src_server, dst_server), price in accelerated.items()
        }

    def price_guided_remap_candidates_dense(
        self,
        mapping: Mapping,
        tenant: int,
        ranks,
        current_servers,
        branch_ranks,
        task_infos,
        max_price_candidates: int,
        time_budget_seconds: float,
        *,
        analysis: ContentionAnalysis | None = None,
    ):
        """Generate price-guided tenant remap candidates without a Python price dict."""
        if _te_accel is None or not hasattr(_te_accel, "price_guided_remap_candidates_dense"):
            raise RuntimeError("C++ dense price-guided remap kernel is required")

        mapping = self.normalize_mapping(mapping)
        if analysis is None:
            analysis = self.analyze(mapping)

        tenant = int(tenant)
        servers = tuple(int(server) for server in current_servers)
        path_edges = self._backbone.data["path_edges"]
        server_send_capacity = self._backbone.data["server_send_capacity"]
        server_recv_capacity = self._backbone.data["server_recv_capacity"]
        edge_capacity = self._backbone.data["edge_capacity"]

        server_count = self._dense_server_count(
            servers,
            server_send_capacity,
            server_recv_capacity,
        )
        edge_items, edge_to_idx, base_sender, base_receiver, base_edge = (
            self._dense_static_price_vectors(
                server_count,
                edge_capacity,
                server_send_capacity,
                server_recv_capacity,
            )
        )

        slot_count = max(len(analysis.slot_prices), 1)
        dense_entry_count = int(slot_count) * (
            int(server_count) * 2 + len(edge_items)
        )
        if dense_entry_count > 2_000_000:
            raise RuntimeError(
                f"dense price-guided remap state too large: {dense_entry_count} entries"
            )

        sender_prices = [list(base_sender) for _slot in range(slot_count)]
        receiver_prices = [list(base_receiver) for _slot in range(slot_count)]
        edge_prices = [list(base_edge) for _slot in range(slot_count)]
        for slot_idx, price_state in enumerate(analysis.slot_prices[:slot_count]):
            for server, price in price_state.get("sender", {}).items():
                if 0 <= int(server) < server_count:
                    sender_prices[slot_idx][int(server)] = float(price)
            for server, price in price_state.get("receiver", {}).items():
                if 0 <= int(server) < server_count:
                    receiver_prices[slot_idx][int(server)] = float(price)
            for edge, price in price_state.get("edge", {}).items():
                edge_idx = edge_to_idx.get(edge)
                if edge_idx is not None:
                    edge_prices[slot_idx][edge_idx] = float(price)

        path_edges_by_pair = self._dense_path_edges_by_pair(
            tenant,
            server_count,
            path_edges,
            edge_to_idx,
        )

        exposure_slots_by_task = []
        for task_id, _src_rank, _dst_rank, _volume in task_infos:
            task_id = int(task_id)
            exposure_slots = sorted(
                set(int(slot) for slot in analysis.task_active_slots.get((tenant, task_id), set()))
                | set(int(slot) for slot in analysis.task_ready_slots.get((tenant, task_id), set()))
            )
            if not exposure_slots:
                exposure_slots = [0]
            exposure_slots_by_task.append(exposure_slots)

        return _te_accel.price_guided_remap_candidates_dense(
            [int(rank) for rank in ranks],
            [int(server) for server in servers],
            [int(rank) for rank in branch_ranks],
            list(task_infos),
            exposure_slots_by_task,
            sender_prices,
            receiver_prices,
            edge_prices,
            path_edges_by_pair,
            int(max_price_candidates),
            float(time_budget_seconds),
        )

    @staticmethod
    def _dense_server_count(servers, server_send_capacity, server_recv_capacity) -> int:
        max_server = 0
        for table in (server_send_capacity, server_recv_capacity):
            if table:
                max_server = max(max_server, max(int(server) for server in table))
        for server in servers:
            max_server = max(max_server, int(server))
        return max_server + 1

    def _dense_static_price_vectors(
        self,
        server_count: int,
        edge_capacity,
        server_send_capacity,
        server_recv_capacity,
    ):
        key = int(server_count)
        cached = self._dense_static_price_cache.get(key)
        if cached is not None:
            return cached
        edge_items = tuple(sorted(edge_capacity))
        edge_to_idx = {edge: idx for idx, edge in enumerate(edge_items)}
        base_sender = tuple(
            self._backbone._resource_price_value(
                server_send_capacity.get(server, 0.0),
                0.0,
            )
            for server in range(server_count)
        )
        base_receiver = tuple(
            self._backbone._resource_price_value(
                server_recv_capacity.get(server, 0.0),
                0.0,
            )
            for server in range(server_count)
        )
        base_edge = tuple(
            self._backbone._edge_price_value(edge, 0.0)
            for edge in edge_items
        )
        cached = (edge_items, edge_to_idx, base_sender, base_receiver, base_edge)
        self._dense_static_price_cache[key] = cached
        return cached

    def _dense_path_edges_by_pair(
        self,
        tenant: int,
        server_count: int,
        path_edges,
        edge_to_idx,
    ):
        key = (int(tenant), int(server_count))
        cached = self._dense_path_edges_cache.get(key)
        if cached is not None:
            return cached
        path_edges_by_pair = []
        for src_server in range(server_count):
            row = []
            for dst_server in range(server_count):
                if src_server == dst_server:
                    row.append([])
                else:
                    row.append([
                        edge_to_idx[edge]
                        for edge in self._backbone._path_edges_for_pair(
                            path_edges,
                            int(tenant),
                            int(src_server),
                            int(dst_server),
                        )
                        if edge in edge_to_idx
                    ])
            path_edges_by_pair.append(row)
        self._dense_path_edges_cache[key] = path_edges_by_pair
        return path_edges_by_pair

    def score_components(self, mapping: Mapping) -> dict[str, tuple[float, float]]:
        """Return task-level and pipeline-level scores for near-tie ranking.

        The task-level score remains the primary objective.  The pipeline score
        is a second view of the same time-expanded graph that accounts for
        per-hop draining and sender-port queueing, and is only used by the
        optimizer to order task-level near ties.
        """
        normalized = self.normalize_mapping(mapping)
        primary_state = self._backbone._compute_time_expanded_surrogate_state(
            normalized,
            collect_signals=False,
        )
        pipeline_state = self._backbone._compute_time_expanded_pipeline_state(
            normalized,
            collect_signals=False,
        )
        return {
            "primary": tuple(float(value) for value in primary_state["score"]),
            "pipeline": tuple(float(value) for value in pipeline_state["score"]),
        }

    def pipeline_score(self, mapping: Mapping) -> tuple[float, float]:
        """Return only the pipeline-level score used for near-tie ranking."""
        normalized = self.normalize_mapping(mapping)
        pipeline_state = self._backbone._compute_time_expanded_pipeline_state(
            normalized,
            collect_signals=False,
        )
        return tuple(float(value) for value in pipeline_state["score"])

    def task_pair_price(
        self,
        mapping: Mapping,
        tenant: int,
        task_id: int,
        src_server: int,
        dst_server: int,
        *,
        analysis: ContentionAnalysis | None = None,
    ) -> float:
        """Price one task/server-pair using the same rule as the lookup table."""
        if int(src_server) == int(dst_server):
            return 0.0
        mapping = self.normalize_mapping(mapping)
        if analysis is None:
            analysis = self.analyze(mapping)

        task_id = int(task_id)
        exposure_slots = sorted(
            int(slot)
            for slot in analysis.task_active_slots.get((int(tenant), task_id), set())
        )
        exposure_slots = sorted(
            set(exposure_slots)
            | {
                int(slot)
                for slot in analysis.task_ready_slots.get((int(tenant), task_id), set())
            }
        )
        if not exposure_slots:
            exposure_slots = [0]

        path_edges = self._backbone.data["path_edges"]
        server_send_capacity = self._backbone.data["server_send_capacity"]
        server_recv_capacity = self._backbone.data["server_recv_capacity"]
        price_sum = 0.0
        for slot_idx in exposure_slots:
            price_state = analysis.slot_prices[slot_idx] if 0 <= slot_idx < len(analysis.slot_prices) else {}
            resource_prices = [
                price_state.get("sender", {}).get(
                    int(src_server),
                    self._backbone._resource_price_value(server_send_capacity[int(src_server)], 0.0),
                ),
                price_state.get("receiver", {}).get(
                    int(dst_server),
                    self._backbone._resource_price_value(server_recv_capacity[int(dst_server)], 0.0),
                ),
            ]
            for edge in self._backbone._path_edges_for_pair(
                path_edges,
                int(tenant),
                int(src_server),
                int(dst_server),
            ):
                resource_prices.append(
                    price_state.get("edge", {}).get(
                        edge,
                        self._backbone._edge_price_value(edge, 0.0),
                    )
                )
            price_sum += max((float(value) for value in resource_prices), default=0.0)
        return float(price_sum / max(len(exposure_slots), 1))
