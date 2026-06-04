from __future__ import annotations

from collections import defaultdict
import math

from multitenant.collectives import has_collective_workload, normalize_collective_programs
from multitenant.objectives import lexicographic_better
from multitenant.simulator.adapter import simulate_collective_details

from .DAG_generation import build_collective_dag_data

try:
    from . import _te_accel
except ImportError:  # pragma: no cover - optional C++ accelerator.
    _te_accel = None


class TimeExpandedEstimatorCore:
    """Pure-mapping solver with task-centric surrogate evaluation.

    The canonical abstraction is task-level: a communication program is compiled
    into a resource-coupled task DAG. For structured staged collectives, the
    default evaluator uses a collapsed fast path that aggregates the task-level
    surrogate at the epoch/frontier granularity for efficiency.
    """

    def __init__(
        self,
        datacenter,
        tenant_mapping,
        tenant_flows=None,
        verbose=True,
        name="mapping_ilp",
        collective=None,
        single_flow_size=None,
        tenant_collective_specs=None,
        tenant_collective_programs=None,
        tenant_start_times=None,
        stage_flows=None,
        fairness_lambda=1.0,
        fairness_iterations=2,
        fairness_grouping="phase",
        slot_duration=None,
        horizon_slots=None,
        validate_with_simulator=None,
        surrogate_mode="collapsed",
        path_table=None,
        extra_seed_mappings=None,
    ):
        self.datacenter = datacenter
        self.tenant_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in tenant_mapping.items()
        }
        self.initial_tenant_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in tenant_mapping.items()
        }
        self.tenant_flows = tenant_flows
        self.collective = collective
        self.single_flow_size = single_flow_size
        self.tenant_collective_specs = tenant_collective_specs
        self.tenant_start_times = {
            int(tenant): float(start_time)
            for tenant, start_time in (tenant_start_times or {}).items()
        }
        self.tenant_collective_programs = normalize_collective_programs(
            self.tenant_mapping,
            collective=collective,
            single_flow_size=single_flow_size,
            tenant_collective_specs=tenant_collective_specs,
            tenant_collective_programs=tenant_collective_programs,
        )
        self.stage_flows = stage_flows
        self.verbose = verbose
        self.model_name = name
        self.slot_duration_override = slot_duration
        self.horizon_slots_override = horizon_slots
        self.validate_with_simulator = bool(validate_with_simulator)
        normalized_surrogate_mode = str(surrogate_mode).lower()
        if normalized_surrogate_mode == "epoch":
            normalized_surrogate_mode = "collapsed"
        if normalized_surrogate_mode in {"slot", "time", "time-expanded"}:
            normalized_surrogate_mode = "time_expanded"
        self.surrogate_mode = normalized_surrogate_mode
        self.fairness_lambda = float(fairness_lambda)
        self.fairness_iterations = max(1, int(fairness_iterations))
        self.fairness_grouping = fairness_grouping
        self.path_table = path_table or self.datacenter.build_tenant_ecmp_path_table(
            sorted(int(tenant) for tenant in tenant_mapping)
        )
        self.path_ordered_edges = self.datacenter.build_edge_table_from_paths(self.path_table)
        self.extra_seed_mappings = [
            {
                int(tenant): {
                    int(rank): int(server)
                    for rank, server in rank_to_server.items()
                }
                for tenant, rank_to_server in seed_mapping.items()
            }
            for seed_mapping in (extra_seed_mappings or [])
        ]

        self.model = None
        self.X = {}
        self.final_obj = None
        self.final_makespan = None
        self.final_avg_jct = None
        self.final_mapping = None
        self.final_score_is_simulated = False
        self.runtime_seconds = 0.0
        self.search_rounds = 0

        self.tenants = sorted(self.initial_tenant_mapping.keys())
        self.rank_orders = {
            tenant: sorted(self.initial_tenant_mapping[tenant].keys())
            for tenant in self.tenants
        }
        self.server_sets = {
            tenant: tuple(self.initial_tenant_mapping[tenant][rank] for rank in self.rank_orders[tenant])
            for tenant in self.tenants
        }
        self._score_cache: dict[tuple[tuple[int, tuple[int, ...]], ...], tuple[float, float]] = {}
        self._surrogate_cache: dict[tuple[tuple[int, tuple[int, ...]], ...], tuple[float, float]] = {}
        self._time_expanded_state_cache: dict[tuple[tuple[int, tuple[int, ...]], ...], dict[str, object]] = {}
        self._pair_epoch_price_cache: dict[tuple[int, int, tuple[int, ...]], dict[tuple[int, int, int], float]] = {}
        self._time_expanded_state_cache_limit = 4
        self._surrogate_candidate_limit = 32 if self.tenant_collective_programs is not None else 16
        self._surrogate_candidates: list[tuple[tuple[float, float], tuple[tuple[int, tuple[int, ...]], ...], dict[int, dict[int, int]]]] = []
        self.tenant_pressure: dict[int, float] = {}
        self.tenant_peak_load: dict[int, float] = {}
        self.rank_pressure: dict[tuple[int, int], float] = {}
        self.tenant_pair_interaction: dict[tuple[int, int], float] = {}
        self.link_price_beta = 4.0
        self.link_price_gamma = 2.0
        self.critical_path_price_beta = 2.0
        self._cpp_time_expanded_engine = None
        self._cpp_time_expanded_engine_error = None
        server_count = int(
            getattr(
                self.datacenter,
                "num_server",
                sum(len(server_set) for server_set in self.server_sets.values()),
            )
        )
        self.max_price_rounds = max(1, min(server_count, 8))
        self.data = self._build_data()

    def _get_cpp_time_expanded_engine(self):
        if _te_accel is None:
            return None
        if self._cpp_time_expanded_engine is not None:
            return self._cpp_time_expanded_engine
        if not self.data.get("task_surrogate"):
            return None

        max_server_id = 0
        for capacity_table in (
            self.data.get("server_send_capacity", {}),
            self.data.get("server_recv_capacity", {}),
        ):
            if capacity_table:
                max_server_id = max(max_server_id, max(int(server) for server in capacity_table))
        for _tenant, _src, _dst in self.data.get("path_edges", {}):
            max_server_id = max(max_server_id, int(_src), int(_dst))
        server_count = max_server_id + 1

        server_send_capacity = [
            float(self.data["server_send_capacity"].get(server, 0.0))
            for server in range(server_count)
        ]
        server_recv_capacity = [
            float(self.data["server_recv_capacity"].get(server, 0.0))
            for server in range(server_count)
        ]

        edge_items = sorted(self.data.get("edge_capacity", {}))
        edge_to_idx = {edge: idx for idx, edge in enumerate(edge_items)}
        edge_capacity = [
            float(self.data["edge_capacity"][edge])
            for edge in edge_items
        ]

        path_edges_by_tenant = []
        for tenant in self.tenants:
            tenant_paths = []
            for src_server in range(server_count):
                row = []
                for dst_server in range(server_count):
                    if src_server == dst_server:
                        row.append([])
                        continue
                    path = self._path_edges_for_pair(
                        self.data["path_edges"],
                        int(tenant),
                        int(src_server),
                        int(dst_server),
                    )
                    row.append([edge_to_idx[edge] for edge in path if edge in edge_to_idx])
                tenant_paths.append(row)
            path_edges_by_tenant.append(tenant_paths)

        task_index = {}
        task_entries = []
        for tenant_pos, tenant in enumerate(self.tenants):
            tenant_meta = self.data.get("task_surrogate", {}).get(int(tenant), {})
            rank_pos = {
                int(rank): idx
                for idx, rank in enumerate(self.rank_orders[int(tenant)])
            }
            task_order_position = {
                int(task_id): idx
                for idx, task_id in enumerate(tenant_meta.get("task_order", []))
            }
            for task_id, task_tuple in sorted(tenant_meta.get("task_info", {}).items()):
                task_id = int(task_id)
                _task_id, src_rank, dst_rank, volume = task_tuple
                task_index[(int(tenant), task_id)] = len(task_entries)
                task_entries.append([
                    int(tenant_pos),
                    task_id,
                    int(rank_pos[int(src_rank)]),
                    int(rank_pos[int(dst_rank)]),
                    float(self._volume_to_bits(volume)),
                    int(task_order_position.get(task_id, task_id)),
                    [],
                    [],
                    0.0,
                ])

        for tenant_pos, tenant in enumerate(self.tenants):
            tenant = int(tenant)
            tenant_meta = self.data.get("task_surrogate", {}).get(tenant, {})
            collective_preds = tenant_meta.get("collective_preds", tenant_meta.get("preds", {}))
            for task_id in tenant_meta.get("task_info", {}):
                entry_idx = task_index[(tenant, int(task_id))]
                predecessors = []
                for pred_task_id in collective_preds.get(int(task_id), []):
                    pred_key = (tenant, int(pred_task_id))
                    if pred_key in task_index:
                        predecessors.append(task_index[pred_key])
                release_prev_indices = []
                release_gate = tenant_meta.get("release_gates", {}).get(int(task_id))
                release_gap = 0.0
                if release_gate is not None:
                    previous_task_ids, release_gap = release_gate
                    for previous_task_id in previous_task_ids:
                        pred_key = (tenant, int(previous_task_id))
                        if pred_key in task_index:
                            pred_idx = task_index[pred_key]
                            release_prev_indices.append(pred_idx)
                            predecessors.append(pred_idx)
                task_entries[entry_idx][6] = sorted(set(predecessors))
                task_entries[entry_idx][7] = sorted(set(release_prev_indices))
                task_entries[entry_idx][8] = float(release_gap)

        try:
            self._cpp_time_expanded_engine = _te_accel.TimeExpandedScoreEngine(
                [int(tenant) for tenant in self.tenants],
                [float(self.tenant_start_times.get(int(tenant), 0.0)) for tenant in self.tenants],
                server_send_capacity,
                server_recv_capacity,
                edge_capacity,
                path_edges_by_tenant,
                task_entries,
                self.slot_duration_override,
                self.horizon_slots_override,
            )
        except Exception:
            import traceback
            self._cpp_time_expanded_engine_error = traceback.format_exc()
            self._cpp_time_expanded_engine = None
        return self._cpp_time_expanded_engine

    def _cpp_time_expanded_score(self, mapping):
        engine = self._get_cpp_time_expanded_engine()
        if engine is None:
            return None
        server_by_tenant_rank = [
            [
                int(mapping[int(tenant)][int(rank)])
                for rank in self.rank_orders[int(tenant)]
            ]
            for tenant in self.tenants
        ]
        result = engine.evaluate(server_by_tenant_rank)
        makespan, avg_jct, raw_makespan, raw_avg_jct, contention_potential = result[:5]
        slot_duration = float(result[5]) if len(result) > 5 else self._estimate_time_slot_duration(mapping)
        tenant_finish = {
            int(tenant): float("nan")
            for tenant in self.tenants
        }
        return {
            "score": (float(makespan), float(avg_jct)),
            "raw_score": (float(raw_makespan), float(raw_avg_jct)),
            "contention_potential": float(contention_potential),
            "tenant_finish": tenant_finish,
            "slot_duration": slot_duration,
        }

    def _cpp_time_expanded_pipeline_score(self, mapping):
        engine = self._get_cpp_time_expanded_engine()
        if engine is None:
            return None
        server_by_tenant_rank = [
            [
                int(mapping[int(tenant)][int(rank)])
                for rank in self.rank_orders[int(tenant)]
            ]
            for tenant in self.tenants
        ]
        result = engine.evaluate_pipeline(server_by_tenant_rank)
        makespan, avg_jct, raw_makespan, raw_avg_jct, contention_potential = result[:5]
        slot_duration = float(result[5]) if len(result) > 5 else self._estimate_time_slot_duration(mapping)
        return {
            "score": (float(makespan), float(avg_jct)),
            "raw_score": (float(raw_makespan), float(raw_avg_jct)),
            "contention_potential": float(contention_potential),
            "tenant_finish": {int(tenant): float("nan") for tenant in self.tenants},
            "slot_duration": slot_duration,
        }

    def _collective_mode(self) -> bool:
        return has_collective_workload(
            collective=self.collective,
            single_flow_size=self.single_flow_size,
            tenant_collective_specs=self.tenant_collective_specs,
            tenant_collective_programs=self.tenant_collective_programs,
        )

    def _program_mode(self) -> bool:
        return self.tenant_collective_programs is not None

    def _build_data(self):
        ranks = {
            tenant: list(self.rank_orders[tenant])
            for tenant in self.tenants
        }
        servers = {
            tenant: list(self.server_sets[tenant])
            for tenant in self.tenants
        }

        if self._collective_mode():
            return build_collective_dag_data(
                datacenter=self.datacenter,
                initial_tenant_mapping=self.initial_tenant_mapping,
                tenant_collective_programs=self.tenant_collective_programs,
                tenants=self.tenants,
                rank_orders=self.rank_orders,
                server_sets=self.server_sets,
                path_ordered_edges=self.path_ordered_edges,
                program_mode=self._program_mode(),
                tenant_pressure=self.tenant_pressure,
                tenant_peak_load=self.tenant_peak_load,
            )

        stage_flows = self.stage_flows
        if stage_flows is None and self.tenant_flows is not None:
            stage_flows = {
                tenant: [list(self.tenant_flows[tenant])]
                for tenant in self.tenants
            }

        return {
            "M": list(self.tenants),
            "R": ranks,
            "S": servers,
            "tasks": {tenant: [] for tenant in self.tenants},
            "stage_flows": stage_flows,
        }

    @staticmethod
    def _is_better_objective(candidate, incumbent, tol=1e-12):
        return lexicographic_better(candidate, incumbent, tol=tol)

    @staticmethod
    def _objective_sort_key(score):
        makespan, avg_jct = score
        return (float(avg_jct), float(makespan))

    def _mapping_signature(self, mapping):
        mapping = self._canonicalize_ring_mapping(mapping)
        return tuple(
            (tenant, tuple(mapping[tenant][rank] for rank in self.rank_orders[tenant]))
            for tenant in self.tenants
        )

    def _tenant_has_ring_rotation_symmetry(self, tenant):
        ring_collectives = {"allgather", "reducescatter", "allreduce"}
        if self.tenant_collective_programs is not None:
            program = self.tenant_collective_programs.get(int(tenant), [])
            if not program:
                return False
            return all(str(op.get("collective", "")).lower() in ring_collectives for op in program)
        if self.collective is None:
            return False
        return str(self.collective).lower() in ring_collectives

    def _canonicalize_ring_mapping(self, mapping):
        canonical = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in mapping.items()
        }
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
                rank: int(rotated[idx])
                for idx, rank in enumerate(ranks)
            }
        return canonical

    def _register_surrogate_candidate(self, mapping, score):
        mapping = self._canonicalize_ring_mapping(mapping)
        signature = self._mapping_signature(mapping)
        candidate_entry = (
            score,
            signature,
            {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in mapping.items()
            },
        )

        filtered = [
            entry
            for entry in self._surrogate_candidates
            if entry[1] != signature
        ]
        filtered.append(candidate_entry)
        filtered.sort(key=lambda entry: self._objective_sort_key(entry[0]))
        self._surrogate_candidates = filtered[: self._surrogate_candidate_limit]

    def _path_edges_for_pair(self, path_edges, tenant, src_server, dst_server):
        key = (int(tenant), int(src_server), int(dst_server))
        edges = path_edges.get(key)
        if edges is None:
            edges = tuple(
                tuple(int(node) for node in edge)
                for edge in self.path_ordered_edges[key]
            )
            path_edges[key] = edges
        return edges

    def _compute_epoch_resource_load_state(self, mapping):
        compiled_schedule = self.data["compiled_schedule"]["per_tenant"]
        global_max_epoch = int(self.data["compiled_schedule"]["global_max_epoch"])
        edge_capacity = self.data["edge_capacity"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        path_edges = self.data["path_edges"]

        epoch_loads: list[dict[tuple[int, int], float]] = [defaultdict(float) for _ in range(global_max_epoch + 1)]
        epoch_sender_loads: list[dict[int, float]] = [defaultdict(float) for _ in range(global_max_epoch + 1)]
        epoch_receiver_loads: list[dict[int, float]] = [defaultdict(float) for _ in range(global_max_epoch + 1)]
        epoch_maxima = [0.0 for _ in range(global_max_epoch + 1)]

        for epoch in range(global_max_epoch + 1):
            normalized_link_load = epoch_loads[epoch]
            normalized_sender_load = epoch_sender_loads[epoch]
            normalized_receiver_load = epoch_receiver_loads[epoch]
            for tenant in self.tenants:
                tenant_epochs = compiled_schedule[tenant]["epoch_flows"]
                for src_rank, dst_rank, volume in tenant_epochs.get(epoch, []):
                    src_server = int(mapping[tenant][src_rank])
                    dst_server = int(mapping[tenant][dst_rank])
                    normalized_sender_load[src_server] += float(volume) / server_send_capacity[src_server]
                    normalized_receiver_load[dst_server] += float(volume) / server_recv_capacity[dst_server]
                    for edge in self._path_edges_for_pair(path_edges, tenant, src_server, dst_server):
                        normalized_link_load[edge] += float(volume) / edge_capacity[edge]
            epoch_maxima[epoch] = max(
                max(normalized_link_load.values(), default=0.0),
                max(normalized_sender_load.values(), default=0.0),
                max(normalized_receiver_load.values(), default=0.0),
            )

        return epoch_loads, epoch_sender_loads, epoch_receiver_loads, epoch_maxima

    def _compute_tenant_epoch_resource_load_state(self, mapping):
        compiled_schedule = self.data["compiled_schedule"]["per_tenant"]
        global_max_epoch = int(self.data["compiled_schedule"]["global_max_epoch"])
        edge_capacity = self.data["edge_capacity"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        path_edges = self.data["path_edges"]

        epoch_loads: list[dict[tuple[int, int], float]] = [defaultdict(float) for _ in range(global_max_epoch + 1)]
        epoch_sender_loads: list[dict[int, float]] = [defaultdict(float) for _ in range(global_max_epoch + 1)]
        epoch_receiver_loads: list[dict[int, float]] = [defaultdict(float) for _ in range(global_max_epoch + 1)]
        tenant_epoch_edges: dict[int, list[set[tuple[int, int]]]] = {
            tenant: [set() for _ in range(global_max_epoch + 1)]
            for tenant in self.tenants
        }
        tenant_epoch_senders: dict[int, list[set[int]]] = {
            tenant: [set() for _ in range(global_max_epoch + 1)]
            for tenant in self.tenants
        }
        tenant_epoch_receivers: dict[int, list[set[int]]] = {
            tenant: [set() for _ in range(global_max_epoch + 1)]
            for tenant in self.tenants
        }

        for epoch in range(global_max_epoch + 1):
            normalized_link_load = epoch_loads[epoch]
            normalized_sender_load = epoch_sender_loads[epoch]
            normalized_receiver_load = epoch_receiver_loads[epoch]
            for tenant in self.tenants:
                tenant_epochs = compiled_schedule[tenant]["epoch_flows"]
                for src_rank, dst_rank, volume in tenant_epochs.get(epoch, []):
                    src_server = int(mapping[tenant][src_rank])
                    dst_server = int(mapping[tenant][dst_rank])
                    normalized_sender_load[src_server] += float(volume) / server_send_capacity[src_server]
                    normalized_receiver_load[dst_server] += float(volume) / server_recv_capacity[dst_server]
                    tenant_epoch_senders[tenant][epoch].add(src_server)
                    tenant_epoch_receivers[tenant][epoch].add(dst_server)
                    for edge in self._path_edges_for_pair(path_edges, tenant, src_server, dst_server):
                        normalized_link_load[edge] += float(volume) / edge_capacity[edge]
                        tenant_epoch_edges[tenant][epoch].add(edge)

        tenant_epoch_maxima: dict[int, list[float]] = {
            tenant: [0.0 for _ in range(global_max_epoch + 1)]
            for tenant in self.tenants
        }
        for tenant in self.tenants:
            for epoch in range(global_max_epoch + 1):
                tenant_epoch_maxima[tenant][epoch] = max(
                    max((epoch_loads[epoch][edge] for edge in tenant_epoch_edges[tenant][epoch]), default=0.0),
                    max((epoch_sender_loads[epoch][server] for server in tenant_epoch_senders[tenant][epoch]), default=0.0),
                    max((epoch_receiver_loads[epoch][server] for server in tenant_epoch_receivers[tenant][epoch]), default=0.0),
                )

        return epoch_loads, epoch_sender_loads, epoch_receiver_loads, tenant_epoch_maxima

    def _compute_epoch_link_load_state(self, mapping):
        epoch_loads, _epoch_sender_loads, _epoch_receiver_loads, epoch_maxima = self._compute_epoch_resource_load_state(mapping)
        return epoch_loads, epoch_maxima

    def _compute_tenant_epoch_link_load_state(self, mapping):
        epoch_loads, _epoch_sender_loads, _epoch_receiver_loads, tenant_epoch_maxima = self._compute_tenant_epoch_resource_load_state(mapping)
        return epoch_loads, tenant_epoch_maxima

    def _compute_task_level_resource_load_state(self, mapping):
        global_max_level = int(self.data.get("task_global_max_level", -1))
        edge_capacity = self.data["edge_capacity"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        path_edges = self.data["path_edges"]
        task_surrogate = self.data.get("task_surrogate", {})

        level_edge_loads = [defaultdict(float) for _ in range(global_max_level + 1)]
        level_sender_loads = [defaultdict(float) for _ in range(global_max_level + 1)]
        level_receiver_loads = [defaultdict(float) for _ in range(global_max_level + 1)]

        for tenant, tenant_meta in task_surrogate.items():
            for level, flows in tenant_meta["level_tasks"].items():
                edge_loads = level_edge_loads[int(level)]
                sender_loads = level_sender_loads[int(level)]
                receiver_loads = level_receiver_loads[int(level)]
                for _task_id, src_rank, dst_rank, volume in flows:
                    src_server = int(mapping[tenant][src_rank])
                    dst_server = int(mapping[tenant][dst_rank])
                    sender_loads[src_server] += float(volume) / server_send_capacity[src_server]
                    receiver_loads[dst_server] += float(volume) / server_recv_capacity[dst_server]
                    for edge in self._path_edges_for_pair(path_edges, tenant, src_server, dst_server):
                        edge_loads[edge] += float(volume) / edge_capacity[edge]

        return level_edge_loads, level_sender_loads, level_receiver_loads

    def _volume_to_bits(self, volume):
        # _build_data compiles collective tasks with scale=1.0, so volumes are bits.
        return float(volume)

    def _task_resources(self, mapping, tenant, task_tuple):
        _task_id, src_rank, dst_rank, _volume = task_tuple
        src_server = int(mapping[tenant][int(src_rank)])
        dst_server = int(mapping[tenant][int(dst_rank)])
        path = tuple(self._path_edges_for_pair(self.data["path_edges"], tenant, src_server, dst_server))
        return src_server, dst_server, path

    def _estimate_time_slot_duration(self, mapping):
        if self.slot_duration_override is not None:
            return max(float(self.slot_duration_override), 1e-12)

        edge_capacity = self.data["edge_capacity"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        samples = []
        for tenant, tenant_meta in self.data.get("task_surrogate", {}).items():
            for task_tuple in tenant_meta["task_info"].values():
                _task_id, _src_rank, _dst_rank, volume = task_tuple
                src_server, dst_server, path = self._task_resources(mapping, tenant, task_tuple)
                bottleneck = min(
                    [server_send_capacity[src_server], server_recv_capacity[dst_server]]
                    + [edge_capacity[edge] for edge in path if edge in edge_capacity],
                    default=0.0,
                )
                if bottleneck > 0.0:
                    samples.append(self._volume_to_bits(volume) / bottleneck)
        if not samples:
            return 1e-6

        samples.sort()
        p25 = samples[max(0, min(len(samples) - 1, int(0.25 * (len(samples) - 1))))]
        smallest = samples[0]
        return max(min(float(p25) / 2.0, float(smallest)), 1e-9)

    def _build_time_expanded_task_state(self, mapping):
        task_state = {}
        successors: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        predecessors: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        indegree: dict[tuple[int, int], int] = {}
        for tenant, tenant_meta in self.data.get("task_surrogate", {}).items():
            task_order_position = {
                int(task_id): idx
                for idx, task_id in enumerate(tenant_meta.get("task_order", []))
            }
            for task_id, task_tuple in tenant_meta["task_info"].items():
                task_key = (int(tenant), int(task_id))
                src_server, dst_server, path = self._task_resources(mapping, tenant, task_tuple)
                _task_id, src_rank, dst_rank, volume = task_tuple
                task_state[task_key] = {
                    "tenant": int(tenant),
                    "task_id": int(task_id),
                    "epoch": int(
                        self.data["compiled_schedule"]["per_tenant"][int(tenant)]
                        .get("task_levels", {})
                        .get(int(task_id), tenant_meta.get("task_levels", {}).get(int(task_id), 0))
                    ),
                    "src_rank": int(src_rank),
                    "dst_rank": int(dst_rank),
                    "src_server": int(src_server),
                    "dst_server": int(dst_server),
                    "path": path,
                    "task_order_pos": int(task_order_position.get(int(task_id), int(task_id))),
                    "resources": (
                        ("sender", int(src_server)),
                        ("receiver", int(dst_server)),
                        *[("edge", edge) for edge in path],
                    ),
                    "resource_capacities": (
                        self.data["server_send_capacity"][int(src_server)],
                        self.data["server_recv_capacity"][int(dst_server)],
                        *[self.data["edge_capacity"][edge] for edge in path],
                    ),
                    "volume_bits": self._volume_to_bits(volume),
                    "remaining_bits": self._volume_to_bits(volume),
                    "start_time": None,
                    "finish_time": None,
                }
                indegree[task_key] = 0

        edge_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        def add_edge(src_key, dst_key):
            if src_key == dst_key or (src_key, dst_key) in edge_set:
                return
            edge_set.add((src_key, dst_key))
            successors[src_key].append(dst_key)
            predecessors[dst_key].append(src_key)
            indegree[dst_key] = int(indegree.get(dst_key, 0)) + 1

        for tenant, tenant_meta in self.data.get("task_surrogate", {}).items():
            for task_id in tenant_meta.get("task_info", {}):
                task_key = (int(tenant), int(task_id))
                collective_preds = tenant_meta.get("collective_preds", tenant_meta.get("preds", {}))
                for pred_task_id in collective_preds.get(int(task_id), []):
                    add_edge((int(tenant), int(pred_task_id)), task_key)

            for task_id, release_gate in tenant_meta.get("release_gates", {}).items():
                task_key = (int(tenant), int(task_id))
                previous_task_ids, _gap_after = release_gate
                for previous_task_id in previous_task_ids:
                    add_edge((int(tenant), int(previous_task_id)), task_key)

        return task_state, successors, predecessors, indegree

    @staticmethod
    def _slot_aligned_time(value, slot_duration):
        if slot_duration is None:
            return float(value)
        slot_duration = max(float(slot_duration), 1e-12)
        return float(math.ceil(max(float(value), 0.0) / slot_duration - 1e-9) * slot_duration)

    def _time_expanded_release_time(self, task_key, task_state, finish_times, slot_duration=None):
        tenant, task_id = task_key
        tenant_meta = self.data.get("task_surrogate", {}).get(tenant, {})
        release_time = self._slot_aligned_time(
            self.tenant_start_times.get(tenant, 0.0),
            slot_duration,
        )
        release_gate = tenant_meta.get("release_gates", {}).get(int(task_id))
        if release_gate is not None:
            previous_task_ids, gap_after = release_gate
            if previous_task_ids:
                previous_keys = [(tenant, int(prev_task_id)) for prev_task_id in previous_task_ids]
                if any(previous_key not in finish_times for previous_key in previous_keys):
                    return float("inf")
                previous_release_time = max(
                    self._slot_aligned_time(finish_times[previous_key], slot_duration)
                    for previous_key in previous_keys
                )
                if slot_duration is None:
                    gate_time = previous_release_time + float(gap_after)
                else:
                    gap_slots = int(math.ceil(float(gap_after) / max(float(slot_duration), 1e-12) - 1e-9))
                    gate_time = previous_release_time + max(gap_slots, 0) * float(slot_duration)
                release_time = max(
                    release_time,
                    gate_time,
                )
            else:
                release_time = max(
                    release_time,
                    self._slot_aligned_time(float(gap_after), slot_duration),
                )
        return release_time

    def _time_expanded_slot_prices(
        self,
        edge_pressure,
        sender_pressure,
        receiver_pressure,
        *,
        active=None,
        service_rate_by_task=None,
        task_state=None,
        critical_resources=None,
    ):
        critical_resources = critical_resources or set()

        def critical_multiplier(resource):
            return 1.0 + self.critical_path_price_beta if resource in critical_resources else 1.0

        if active is None or service_rate_by_task is None or task_state is None:
            edge_prices = {
                edge: critical_multiplier(("edge", edge)) * self._edge_price_value(edge, pressure)
                for edge, pressure in edge_pressure.items()
            }
            server_send_capacity = self.data["server_send_capacity"]
            server_recv_capacity = self.data["server_recv_capacity"]
            sender_prices = {
                server: critical_multiplier(("sender", server))
                * self._resource_price_value(server_send_capacity[server], pressure)
                for server, pressure in sender_pressure.items()
            }
            receiver_prices = {
                server: critical_multiplier(("receiver", server))
                * self._resource_price_value(server_recv_capacity[server], pressure)
                for server, pressure in receiver_pressure.items()
            }
            return {"edge": edge_prices, "sender": sender_prices, "receiver": receiver_prices}

        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]

        edge_service: dict[tuple[int, int], float] = defaultdict(float)
        sender_service: dict[int, float] = defaultdict(float)
        receiver_service: dict[int, float] = defaultdict(float)
        for task_key in active:
            rate = float(service_rate_by_task.get(task_key, 0.0))
            if rate <= 0.0:
                continue
            state = task_state[task_key]
            src_server = int(state["src_server"])
            dst_server = int(state["dst_server"])
            sender_service[src_server] += rate
            receiver_service[dst_server] += rate
            for edge in state["path"]:
                edge_service[edge] += rate

        edge_prices = {}
        for edge, service_rate in edge_service.items():
            capacity = float(self.data["edge_capacity"][edge])
            utilization = max(0.0, min(1.0, float(service_rate) / max(capacity, 1e-12)))
            edge_prices[edge] = (
                critical_multiplier(("edge", edge))
                * utilization
                * self._edge_price_value(edge, edge_pressure.get(edge, 0.0))
            )

        sender_prices = {}
        for server, service_rate in sender_service.items():
            capacity = float(server_send_capacity[server])
            utilization = max(0.0, min(1.0, float(service_rate) / max(capacity, 1e-12)))
            sender_prices[server] = (
                critical_multiplier(("sender", server))
                * utilization
                * self._resource_price_value(capacity, sender_pressure.get(server, 0.0))
            )

        receiver_prices = {}
        for server, service_rate in receiver_service.items():
            capacity = float(server_recv_capacity[server])
            utilization = max(0.0, min(1.0, float(service_rate) / max(capacity, 1e-12)))
            receiver_prices[server] = (
                critical_multiplier(("receiver", server))
                * utilization
                * self._resource_price_value(capacity, receiver_pressure.get(server, 0.0))
            )

        return {"edge": edge_prices, "sender": sender_prices, "receiver": receiver_prices}

    def _time_expanded_slot_prices_batch(
        self,
        slot_resource_pressure,
        slot_active_tasks,
        slot_service_rates,
        task_state,
        critical_resources_by_slot,
    ):
        def dense_capacity_vector(capacities):
            if isinstance(capacities, dict):
                if not capacities:
                    return []
                max_key = max(int(key) for key in capacities)
                return [
                    float(capacities.get(idx, 0.0))
                    for idx in range(max_key + 1)
                ]
            return [float(value) for value in capacities]

        if _te_accel is not None and hasattr(_te_accel, "time_expanded_slot_prices_batch"):
            try:
                return list(
                    _te_accel.time_expanded_slot_prices_batch(
                        slot_resource_pressure,
                        slot_active_tasks,
                        slot_service_rates,
                        task_state,
                        critical_resources_by_slot,
                        self.data["edge_capacity"],
                        dense_capacity_vector(self.data["server_send_capacity"]),
                        dense_capacity_vector(self.data["server_recv_capacity"]),
                        float(self.link_price_beta),
                        float(self.link_price_gamma),
                        float(self.critical_path_price_beta),
                    )
                )
            except Exception:
                pass
        return [
            self._time_expanded_slot_prices(
                pressure_state.get("edge", {}),
                pressure_state.get("sender", {}),
                pressure_state.get("receiver", {}),
                active=slot_active_tasks[slot_idx],
                service_rate_by_task=slot_service_rates[slot_idx],
                task_state=task_state,
                critical_resources=critical_resources_by_slot[slot_idx],
            )
            for slot_idx, pressure_state in enumerate(slot_resource_pressure)
        ]

    def _time_expanded_task_resources(self, task_key, task_state):
        state = task_state[task_key]
        resources = state.get("resources")
        if resources is not None:
            return resources
        return (
            ("sender", int(state["src_server"])),
            ("receiver", int(state["dst_server"])),
            *[("edge", edge) for edge in state["path"]],
        )

    @staticmethod
    def _time_expanded_outgoing_port(task_key, task_state):
        state = task_state[task_key]
        path = tuple(state.get("path", ()))
        return path[0] if path else ("server", int(state["src_server"]))

    @classmethod
    def _time_expanded_queue_key(cls, task_key, task_state):
        state = task_state[task_key]
        return (
            int(state["tenant"]),
            cls._time_expanded_outgoing_port(task_key, task_state),
        )

    @classmethod
    def _time_expanded_service_frontier(cls, active, task_state):
        """Return tasks at the head of each tenant outgoing-port queue.

        The collective DAG controls readiness.  Sender-side communication
        serialization is modeled as a resource discipline: for the same tenant
        and the same outgoing port, packets drain FIFO; different tenants are
        independent queues and only interact through shared physical resources.
        """
        frontier: dict[tuple[int, object], tuple[tuple[float, int, int], tuple[int, int]]] = {}
        for task_key in active:
            state = task_state[task_key]
            queue_key = cls._time_expanded_queue_key(task_key, task_state)
            order_key = (
                float(state.get("start_time") or 0.0),
                int(state.get("task_order_pos", state["task_id"])),
                int(state["task_id"]),
            )
            current = frontier.get(queue_key)
            if current is None or order_key < current[0]:
                frontier[queue_key] = (order_key, task_key)
        return {task_key for _order_key, task_key in frontier.values()}

    def _extract_time_expanded_critical_tasks(self, task_state, predecessors):
        critical_tasks = set()
        for tenant in self.tenants:
            tenant_tasks = [
                task_key
                for task_key, state in task_state.items()
                if int(state["tenant"]) == int(tenant)
            ]
            if not tenant_tasks:
                continue
            terminal_finish = max(
                float(task_state[task_key].get("finish_time") or 0.0)
                for task_key in tenant_tasks
            )
            stack = [
                task_key
                for task_key in tenant_tasks
                if abs(float(task_state[task_key].get("finish_time") or 0.0) - terminal_finish) <= 1e-9
            ]
            while stack:
                task_key = stack.pop()
                if task_key in critical_tasks:
                    continue
                critical_tasks.add(task_key)
                pred_keys = [
                    pred_key
                    for pred_key in predecessors.get(task_key, [])
                    if pred_key in task_state
                ]
                if not pred_keys:
                    continue
                pred_finish = [
                    (float(task_state[pred_key].get("finish_time") or 0.0), pred_key)
                    for pred_key in pred_keys
                ]
                max_pred_finish = max(value for value, _pred_key in pred_finish)
                stack.extend(
                    pred_key
                    for value, pred_key in pred_finish
                    if abs(value - max_pred_finish) <= 1e-9
                )
        return critical_tasks

    def _critical_resources_by_slot(self, task_state, slot_active_tasks, critical_tasks):
        critical_resources_by_slot = []
        for active_tasks in slot_active_tasks:
            resources = set()
            for task_key in set(active_tasks) & critical_tasks:
                resources.update(self._time_expanded_task_resources(task_key, task_state))
            critical_resources_by_slot.append(resources)
        return critical_resources_by_slot

    def _time_expanded_search_signals(
        self,
        task_state,
        successors,
        slot_active_tasks,
        slot_ready_tasks,
        slot_service_rates,
        slot_prices,
        slot_resource_pressure,
        slot_duration,
        critical_tasks,
        *,
        include_hotspots=True,
    ):
        tenant_pressure: dict[int, float] = defaultdict(float)
        tenant_peak_load: dict[int, float] = defaultdict(float)
        task_pressure: dict[tuple[int, int], float] = defaultdict(float)
        rank_pressure: dict[tuple[int, int], float] = defaultdict(float)
        excess_task_pressure: dict[tuple[int, int], float] = defaultdict(float)
        excess_rank_pressure: dict[tuple[int, int], float] = defaultdict(float)
        tenant_pair_interaction: dict[tuple[int, int], float] = defaultdict(float)
        hotspots = []
        contention_clusters = []
        slot_duration = float(slot_duration)

        tenant_finish = {}
        for tenant in self.tenants:
            tenant_tasks = [
                task_key
                for task_key, state in task_state.items()
                if int(state["tenant"]) == int(tenant)
            ]
            tenant_finish[int(tenant)] = max(
                (float(task_state[task_key].get("finish_time") or 0.0) for task_key in tenant_tasks),
                default=0.0,
            )

        def realized_task_weight(task_key):
            state = task_state[task_key]
            critical_bonus = self.critical_path_price_beta if task_key in critical_tasks else 0.0
            return 1.0 + critical_bonus

        def pressure_for_resource(pressure_state, resource):
            resource_type, resource_id = resource
            return float(pressure_state.get(resource_type, {}).get(resource_id, 0.0))

        for slot_idx, active_tasks in enumerate(slot_active_tasks):
            ready_tasks = (
                slot_ready_tasks[slot_idx]
                if slot_idx < len(slot_ready_tasks)
                else set(active_tasks)
            )
            pressure_state = slot_resource_pressure[slot_idx] if slot_idx < len(slot_resource_pressure) else {}
            service_rates = slot_service_rates[slot_idx] if slot_idx < len(slot_service_rates) else {}
            resource_tenant_exposure: dict[tuple[str, object], dict[int, float]] = defaultdict(lambda: defaultdict(float))

            for task_key in ready_tasks:
                state = task_state[task_key]
                tenant = int(state["tenant"])
                src_server = int(state["src_server"])
                dst_server = int(state["dst_server"])
                realized_load = max(
                    float(pressure_state.get("sender", {}).get(src_server, 0.0)),
                    float(pressure_state.get("receiver", {}).get(dst_server, 0.0)),
                    max(
                        (
                            float(pressure_state.get("edge", {}).get(edge, 0.0))
                            for edge in state["path"]
                        ),
                        default=0.0,
                    ),
                )
                if realized_load <= 0.0:
                    continue
                weight = realized_load * realized_task_weight(task_key)
                task_pressure[task_key] += weight
                tenant_pressure[tenant] += weight
                tenant_peak_load[tenant] = max(float(tenant_peak_load.get(tenant, 0.0)), realized_load)
                for rank in (int(state["src_rank"]), int(state["dst_rank"])):
                    rank_pressure[(tenant, rank)] += 0.5 * weight

            if not include_hotspots:
                continue

            queue_members: dict[tuple[int, object], list[tuple[int, int]]] = defaultdict(list)
            for task_key in ready_tasks:
                queue_members[self._time_expanded_queue_key(task_key, task_state)].append(task_key)
            for members in queue_members.values():
                members.sort(
                    key=lambda key: (
                        float(task_state[key].get("start_time") or 0.0),
                        int(task_state[key].get("task_order_pos", task_state[key]["task_id"])),
                        int(task_state[key]["task_id"]),
                    )
                )

            resources_in_slot = set()
            for task_key in active_tasks:
                resources_in_slot.update(self._time_expanded_task_resources(task_key, task_state))

            for resource in resources_in_slot:
                normalized_load = pressure_for_resource(pressure_state, resource)
                excess = max(0.0, normalized_load - 1.0)
                if excess <= 0.0:
                    continue
                users = [
                    task_key
                    for task_key in active_tasks
                    if resource in self._time_expanded_task_resources(task_key, task_state)
                ]
                if not users:
                    continue
                service_by_task = {
                    task_key: max(float(service_rates.get(task_key, 0.0)) * slot_duration, 1.0)
                    for task_key in users
                }
                total_service = max(sum(service_by_task.values()), 1e-12)
                cluster_tasks = []
                cluster_service_tasks = []
                cluster_queued_tasks = []
                cluster_ranks = set()
                cluster_tenants = set()
                contribution_by_task: dict[tuple[int, int], float] = defaultdict(float)
                for task_key in users:
                    contribution = excess * service_by_task[task_key] / total_service
                    contribution_by_task[task_key] += contribution
                    cluster_service_tasks.append(task_key)
                    queue_key = self._time_expanded_queue_key(task_key, task_state)
                    waiters = [
                        queued_key
                        for queued_key in queue_members.get(queue_key, [])
                        if queued_key != task_key
                    ]
                    if waiters:
                        waiter_share = 0.5 * contribution / max(len(waiters), 1)
                        for queued_key in waiters:
                            contribution_by_task[queued_key] += waiter_share
                            cluster_queued_tasks.append(queued_key)

                for task_key, contribution in contribution_by_task.items():
                    state = task_state[task_key]
                    tenant = int(state["tenant"])
                    weighted_contribution = float(contribution) * realized_task_weight(task_key)
                    excess_task_pressure[task_key] += weighted_contribution
                    cluster_tasks.append(task_key)
                    cluster_tenants.add(tenant)
                    for rank in (int(state["src_rank"]), int(state["dst_rank"])):
                        excess_rank_pressure[(tenant, rank)] += 0.5 * weighted_contribution
                        cluster_ranks.add((tenant, rank))
                    resource_tenant_exposure[resource][tenant] += weighted_contribution
                hotspots.append(
                    {
                        "slot": int(slot_idx),
                        "resource": resource,
                        "load": float(normalized_load),
                        "excess": float(excess),
                    }
                )
                contention_clusters.append(
                    {
                        "slot": int(slot_idx),
                        "resource": resource,
                        "excess": float(excess),
                        "tasks": tuple(cluster_tasks),
                        "service_tasks": tuple(cluster_service_tasks),
                        "queued_tasks": tuple(sorted(set(cluster_queued_tasks))),
                        "ranks": tuple(sorted(cluster_ranks)),
                        "tenants": tuple(sorted(cluster_tenants)),
                    }
                )

            for task_key in active_tasks:
                state = task_state[task_key]
                tenant = int(state["tenant"])
                src_server = int(state["src_server"])
                dst_server = int(state["dst_server"])
                task_peak = max(
                    float(pressure_state.get("sender", {}).get(src_server, 0.0)),
                    float(pressure_state.get("receiver", {}).get(dst_server, 0.0)),
                    max(
                        (
                            float(pressure_state.get("edge", {}).get(edge, 0.0))
                            for edge in state["path"]
                        ),
                        default=0.0,
                    ),
                )
                tenant_peak_load[tenant] = max(float(tenant_peak_load.get(tenant, 0.0)), task_peak)

            for exposure_by_tenant in resource_tenant_exposure.values():
                items = sorted(exposure_by_tenant.items())
                for idx, (tenant_a, exposure_a) in enumerate(items):
                    for tenant_b, exposure_b in items[idx + 1:]:
                        pair = (int(tenant_a), int(tenant_b))
                        tenant_pair_interaction[pair] += min(float(exposure_a), float(exposure_b))

        hotspots.sort(key=lambda item: (-float(item["excess"]), int(item["slot"]), str(item["resource"])))
        contention_clusters.sort(key=lambda item: (-float(item["excess"]), int(item["slot"]), str(item["resource"])))
        return {
            "tenant_pressure": {int(tenant): float(value) for tenant, value in tenant_pressure.items()},
            "tenant_peak_load": {int(tenant): float(value) for tenant, value in tenant_peak_load.items()},
            "task_pressure": {key: float(value) for key, value in task_pressure.items()},
            "rank_pressure": {key: float(value) for key, value in rank_pressure.items()},
            "excess_task_pressure": {key: float(value) for key, value in excess_task_pressure.items()},
            "excess_rank_pressure": {key: float(value) for key, value in excess_rank_pressure.items()},
            "tenant_pair_interaction": {key: float(value) for key, value in tenant_pair_interaction.items()},
            "hotspots": hotspots,
            "contention_clusters": contention_clusters,
        }

    def _resource_capacity(self, resource):
        resource_type, resource_id = resource
        if resource_type == "sender":
            return float(self.data["server_send_capacity"][resource_id])
        if resource_type == "receiver":
            return float(self.data["server_recv_capacity"][resource_id])
        return float(self.data["edge_capacity"][resource_id])

    def _max_min_service_rates(self, active, task_state, slot_duration):
        """Approximate the MILP bottleneck-link sharing constraints in one slot."""
        if _te_accel is not None and active:
            ordered_tasks = list(active)
            resources_by_task = []
            capacities_by_task = []
            demand_rates = []
            for task_key in ordered_tasks:
                state = task_state[task_key]
                resources = tuple(self._time_expanded_task_resources(task_key, task_state))
                capacities = tuple(float(value) for value in state.get("resource_capacities", ()))
                resources_by_task.append(resources)
                capacities_by_task.append(capacities)
                demand_rates.append(
                    float(state["remaining_bits"]) / max(float(slot_duration), 1e-12)
                )
            accelerated_rates = _te_accel.max_min_rates(
                resources_by_task,
                capacities_by_task,
                demand_rates,
            )
            return {
                task_key: float(accelerated_rates[idx])
                for idx, task_key in enumerate(ordered_tasks)
            }

        remaining = set(active)
        rates = {task_key: 0.0 for task_key in active}
        residual_capacity = {}
        resource_users: dict[tuple[str, object], list[tuple[int, int]]] = defaultdict(list)
        active_count: dict[tuple[str, object], int] = defaultdict(int)
        task_resources: dict[tuple[int, int], tuple[tuple[str, object], ...]] = {}
        for task_key in active:
            state = task_state[task_key]
            resources = tuple(self._time_expanded_task_resources(task_key, task_state))
            task_resources[task_key] = resources
            capacities = state.get("resource_capacities", ())
            for resource, capacity in zip(resources, capacities):
                resource_users[resource].append(task_key)
                active_count[resource] += 1
                if resource not in residual_capacity:
                    residual_capacity[resource] = float(capacity)

        while remaining:
            limiting_share = float("inf")
            limiting_resources = []
            for resource, count in active_count.items():
                if count <= 0:
                    continue
                share = residual_capacity.get(resource, 0.0) / count
                if share < limiting_share - 1e-12:
                    limiting_share = share
                    limiting_resources = [resource]
                elif abs(share - limiting_share) <= 1e-12:
                    limiting_resources.append(resource)

            if not math.isfinite(limiting_share):
                break

            frozen = set()
            for resource in limiting_resources:
                frozen.update(
                    task_key
                    for task_key in resource_users[resource]
                    if task_key in remaining
                )
            if not frozen:
                frozen = set(remaining)

            for task_key in frozen:
                demand_rate = float(task_state[task_key]["remaining_bits"]) / max(slot_duration, 1e-12)
                rates[task_key] = max(0.0, min(float(limiting_share), demand_rate))

            for task_key in frozen:
                rate = rates[task_key]
                for resource in task_resources.get(task_key, ()):
                    residual_capacity[resource] = max(0.0, residual_capacity.get(resource, 0.0) - rate)
                    active_count[resource] -= 1
                remaining.remove(task_key)

        return rates

    def _compute_time_expanded_pipeline_state(self, mapping, *, collect_signals=True):
        if not collect_signals:
            cpp_state = self._cpp_time_expanded_pipeline_score(mapping)
            if cpp_state is not None:
                return cpp_state

        edge_capacity = self.data["edge_capacity"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        slot_duration = self._estimate_time_slot_duration(mapping)
        if self.horizon_slots_override is not None:
            max_slots = max(1, int(self.horizon_slots_override))
        else:
            task_count = sum(len(meta["task_info"]) for meta in self.data.get("task_surrogate", {}).values())
            max_slots = max(512, min(16384, 16 * max(task_count, 1)))

        task_state, successors, predecessors, remaining_preds = self._build_time_expanded_task_state(mapping)
        pending = set(task_state)
        active: set[tuple[int, int]] = set()
        finish_times: dict[tuple[int, int], float] = {}
        task_buffers: dict[tuple[int, int], list[float]] = {
            task_key: [0.0 for _edge in state["path"]]
            for task_key, state in task_state.items()
        }
        task_delivered: dict[tuple[int, int], float] = defaultdict(float)
        slot_prices = []
        slot_maxima = []
        slot_resource_pressure = []
        slot_active_tasks = []
        slot_ready_tasks = []
        slot_service_rates = []
        task_active_slots: dict[tuple[int, int], set[int]] = defaultdict(set)
        global_max_epoch = int(self.data["compiled_schedule"]["global_max_epoch"])
        epoch_active_slots = [set() for _ in range(global_max_epoch + 1)]
        contention_potential = 0.0

        def task_remaining(task_key):
            state = task_state[task_key]
            return max(0.0, float(state["volume_bits"]) - float(task_delivered.get(task_key, 0.0)))

        def item_resources(task_key, hop_idx):
            state = task_state[task_key]
            edge = state["path"][hop_idx]
            resources = [("edge", edge)]
            capacities = [float(edge_capacity[edge])]
            if int(hop_idx) == 0:
                resources.append(("sender", int(state["src_server"])))
                capacities.append(float(server_send_capacity[int(state["src_server"])]))
            if int(hop_idx) == len(state["path"]) - 1:
                resources.append(("receiver", int(state["dst_server"])))
                capacities.append(float(server_recv_capacity[int(state["dst_server"])]))
            return tuple(resources), tuple(capacities)

        def service_frontier(items):
            frontier = {}
            for item in items:
                task_key, hop_idx = item
                state = task_state[task_key]
                edge = state["path"][hop_idx]
                queue_key = (int(state["tenant"]), edge)
                order_key = (
                    float(state.get("start_time") or 0.0),
                    int(state.get("task_order_pos", state["task_id"])),
                    int(state["task_id"]),
                    int(hop_idx),
                )
                current = frontier.get(queue_key)
                if current is None or order_key < current[0]:
                    frontier[queue_key] = (order_key, item)
            return {item for _order, item in frontier.values()}

        def max_min_item_rates(service_items, item_available):
            if _te_accel is not None and service_items:
                ordered_items = list(service_items)
                resources_by_item = []
                capacities_by_item = []
                demand_rates = []
                for item in ordered_items:
                    task_key, hop_idx = item
                    resources, capacities = item_resources(task_key, hop_idx)
                    resources_by_item.append(resources)
                    capacities_by_item.append(capacities)
                    demand_rates.append(
                        float(item_available.get(item, 0.0)) / max(float(slot_duration), 1e-12)
                    )
                accelerated_rates = _te_accel.max_min_rates(
                    resources_by_item,
                    capacities_by_item,
                    demand_rates,
                )
                return {
                    item: float(accelerated_rates[idx])
                    for idx, item in enumerate(ordered_items)
                }

            remaining_items = set(service_items)
            rates = {item: 0.0 for item in service_items}
            residual_capacity = {}
            resource_users: dict[tuple[str, object], list[tuple[tuple[int, int], int]]] = defaultdict(list)
            active_count: dict[tuple[str, object], int] = defaultdict(int)
            resources_by_item = {}
            for item in service_items:
                task_key, hop_idx = item
                resources, capacities = item_resources(task_key, hop_idx)
                resources_by_item[item] = resources
                for resource, capacity in zip(resources, capacities):
                    resource_users[resource].append(item)
                    active_count[resource] += 1
                    residual_capacity.setdefault(resource, float(capacity))

            while remaining_items:
                limiting_share = float("inf")
                limiting_resources = []
                for resource, count in active_count.items():
                    if count <= 0:
                        continue
                    share = residual_capacity.get(resource, 0.0) / count
                    if share < limiting_share - 1e-12:
                        limiting_share = share
                        limiting_resources = [resource]
                    elif abs(share - limiting_share) <= 1e-12:
                        limiting_resources.append(resource)
                if not math.isfinite(limiting_share):
                    break

                frozen = set()
                for resource in limiting_resources:
                    frozen.update(item for item in resource_users[resource] if item in remaining_items)
                if not frozen:
                    frozen = set(remaining_items)

                for item in frozen:
                    demand_rate = float(item_available.get(item, 0.0)) / max(slot_duration, 1e-12)
                    rates[item] = max(0.0, min(float(limiting_share), demand_rate))

                for item in frozen:
                    rate = rates[item]
                    for resource in resources_by_item.get(item, ()):
                        residual_capacity[resource] = max(0.0, residual_capacity.get(resource, 0.0) - rate)
                        active_count[resource] -= 1
                    remaining_items.remove(item)
            return rates

        current_time = 0.0
        slot_idx = 0
        while (pending or active) and slot_idx < max_slots:
            newly_ready = []
            for task_key in list(pending):
                if remaining_preds.get(task_key, 0) != 0:
                    continue
                if self._time_expanded_release_time(task_key, task_state, finish_times, slot_duration) <= current_time + 1e-12:
                    newly_ready.append(task_key)
            for task_key in newly_ready:
                pending.remove(task_key)
                state = task_state[task_key]
                if state["start_time"] is None:
                    state["start_time"] = current_time
                if not state["path"]:
                    state["finish_time"] = current_time
                    finish_times[task_key] = current_time
                    for succ_key in successors.get(task_key, []):
                        remaining_preds[succ_key] = int(remaining_preds.get(succ_key, 0)) - 1
                    continue
                task_buffers[task_key][0] += float(state["volume_bits"])
                active.add(task_key)

            if not active:
                candidate_releases = [
                    self._time_expanded_release_time(task_key, task_state, finish_times, slot_duration)
                    for task_key in pending
                    if remaining_preds.get(task_key, 0) == 0
                ]
                finite_releases = [release_time for release_time in candidate_releases if math.isfinite(release_time)]
                next_release = min(finite_releases, default=current_time + slot_duration)
                if collect_signals:
                    slot_prices.append({"edge": {}, "sender": {}, "receiver": {}})
                    slot_resource_pressure.append({"edge": {}, "sender": {}, "receiver": {}})
                    slot_active_tasks.append(set())
                    slot_ready_tasks.append(set())
                    slot_service_rates.append({})
                    slot_maxima.append(0.0)
                current_time = max(current_time + slot_duration, float(next_release))
                slot_idx += 1
                continue

            ready_active = set(active)
            candidate_items = []
            item_available = {}
            for task_key in active:
                for hop_idx, amount in enumerate(task_buffers[task_key]):
                    if float(amount) <= 1e-9:
                        continue
                    item = (task_key, int(hop_idx))
                    candidate_items.append(item)
                    item_available[item] = float(amount)

            service_items = service_frontier(candidate_items)
            edge_pressure: dict[tuple[int, int], float] = defaultdict(float)
            sender_pressure: dict[int, float] = defaultdict(float)
            receiver_pressure: dict[int, float] = defaultdict(float)
            for item in service_items:
                task_key, hop_idx = item
                resources, capacities = item_resources(task_key, hop_idx)
                for resource, capacity in zip(resources, capacities):
                    normalized = min(float(item_available[item]), float(capacity) * slot_duration) / max(
                        float(capacity) * slot_duration,
                        1e-12,
                    )
                    resource_type, resource_id = resource
                    if resource_type == "edge":
                        edge_pressure[resource_id] += normalized
                    elif resource_type == "sender":
                        sender_pressure[resource_id] += normalized
                    else:
                        receiver_pressure[resource_id] += normalized

            for pressure_state in (edge_pressure, sender_pressure, receiver_pressure):
                for normalized_load in pressure_state.values():
                    excess = max(0.0, float(normalized_load) - 1.0)
                    contention_potential += excess * excess

            item_rates = max_min_item_rates(service_items, item_available)
            task_rate_by_slot: dict[tuple[int, int], float] = defaultdict(float)
            next_hop_add: dict[tuple[tuple[int, int], int], float] = defaultdict(float)
            delivered_this_slot: dict[tuple[int, int], float] = defaultdict(float)
            for item in list(service_items):
                task_key, hop_idx = item
                rate = float(item_rates.get(item, 0.0))
                if rate <= 0.0:
                    continue
                service_bits = min(float(task_buffers[task_key][hop_idx]), rate * slot_duration)
                if service_bits <= 0.0:
                    continue
                task_buffers[task_key][hop_idx] -= service_bits
                task_rate_by_slot[task_key] += service_bits / max(slot_duration, 1e-12)
                if collect_signals:
                    task_active_slots[task_key].add(slot_idx)
                    task_epoch = int(task_state[task_key].get("epoch", 0))
                    if 0 <= task_epoch <= global_max_epoch:
                        epoch_active_slots[task_epoch].add(slot_idx)
                if hop_idx + 1 < len(task_buffers[task_key]):
                    next_hop_add[(task_key, hop_idx + 1)] += service_bits
                else:
                    delivered_this_slot[task_key] += service_bits

            for (task_key, hop_idx), amount in next_hop_add.items():
                task_buffers[task_key][hop_idx] += float(amount)

            completed = []
            for task_key, amount in delivered_this_slot.items():
                task_delivered[task_key] += float(amount)
                state = task_state[task_key]
                state["remaining_bits"] = task_remaining(task_key)
                if task_remaining(task_key) <= 1e-6:
                    last_rate = max(float(task_rate_by_slot.get(task_key, 0.0)), 1e-12)
                    finish_time = current_time + min(slot_duration, float(amount) / last_rate)
                    state["finish_time"] = finish_time
                    finish_times[task_key] = finish_time
                    completed.append(task_key)

            for task_key in list(active):
                task_state[task_key]["remaining_bits"] = task_remaining(task_key)

            for task_key in completed:
                active.discard(task_key)
                for succ_key in successors.get(task_key, []):
                    remaining_preds[succ_key] = int(remaining_preds.get(succ_key, 0)) - 1

            if collect_signals:
                slot_prices.append({"edge": {}, "sender": {}, "receiver": {}})
                slot_resource_pressure.append({
                    "edge": dict(edge_pressure),
                    "sender": dict(sender_pressure),
                    "receiver": dict(receiver_pressure),
                })
                slot_active_tasks.append(set(task_key for task_key, _hop_idx in service_items))
                slot_ready_tasks.append(set(ready_active))
                slot_service_rates.append(dict(task_rate_by_slot))
                slot_maxima.append(
                    max(
                        max(edge_pressure.values(), default=0.0),
                        max(sender_pressure.values(), default=0.0),
                        max(receiver_pressure.values(), default=0.0),
                    )
                )
            current_time += slot_duration
            slot_idx += 1

        if pending or active:
            penalty_start = current_time
            for task_key in list(active) + list(pending):
                if task_key in finish_times:
                    continue
                state = task_state[task_key]
                src_server = int(state["src_server"])
                dst_server = int(state["dst_server"])
                bottleneck = min(
                    [server_send_capacity[src_server], server_recv_capacity[dst_server]]
                    + [edge_capacity[edge] for edge in state["path"]],
                    default=1.0,
                )
                penalty_start += task_remaining(task_key) / max(bottleneck, 1e-12)
                state["finish_time"] = penalty_start
                finish_times[task_key] = penalty_start

        tenant_finish = {}
        for tenant in self.tenants:
            tenant_task_keys = [key for key in task_state if key[0] == int(tenant)]
            tenant_finish[tenant] = max((finish_times.get(key, 0.0) for key in tenant_task_keys), default=0.0)

        raw_score = (
            float(max(tenant_finish.values(), default=0.0)),
            float(sum(tenant_finish.values()) / max(len(tenant_finish), 1)),
        )
        contention_tiebreak = 1e-9 * float(contention_potential)
        score = (
            float(raw_score[0] + contention_tiebreak),
            float(raw_score[1] + contention_tiebreak),
        )
        if not collect_signals:
            return {
                "score": score,
                "raw_score": raw_score,
                "contention_potential": float(contention_potential),
                "tenant_finish": tenant_finish,
                "slot_duration": slot_duration,
            }

        critical_tasks = self._extract_time_expanded_critical_tasks(task_state, predecessors)
        critical_resources_by_slot = self._critical_resources_by_slot(
            task_state,
            slot_active_tasks,
            critical_tasks,
        )
        slot_prices = self._time_expanded_slot_prices_batch(
            slot_resource_pressure,
            slot_active_tasks,
            slot_service_rates,
            task_state,
            critical_resources_by_slot,
        )
        search_signals = self._time_expanded_search_signals(
            task_state,
            successors,
            slot_active_tasks,
            slot_ready_tasks,
            slot_service_rates,
            slot_prices,
            slot_resource_pressure,
            slot_duration,
            critical_tasks,
        )
        tenant_pressure = search_signals["tenant_pressure"]
        tenant_peak_load = search_signals["tenant_peak_load"]

        return {
            "score": score,
            "raw_score": raw_score,
            "contention_potential": float(contention_potential),
            "slot_prices": slot_prices,
            "slot_maxima": slot_maxima,
            "slot_resource_pressure": slot_resource_pressure,
            "slot_active_tasks": slot_active_tasks,
            "slot_ready_tasks": slot_ready_tasks,
            "slot_service_rates": slot_service_rates,
            "task_active_slots": task_active_slots,
            "epoch_active_slots": epoch_active_slots,
            "task_state": task_state,
            "predecessors": predecessors,
            "critical_tasks": critical_tasks,
            "critical_resources_by_slot": critical_resources_by_slot,
            "tenant_finish": tenant_finish,
            "tenant_pressure": {int(tenant): float(value) for tenant, value in tenant_pressure.items()},
            "tenant_peak_load": {int(tenant): float(value) for tenant, value in tenant_peak_load.items()},
            "task_pressure": dict(search_signals["task_pressure"]),
            "rank_pressure": dict(search_signals["rank_pressure"]),
            "tenant_pair_interaction": dict(search_signals["tenant_pair_interaction"]),
            "hotspots": list(search_signals["hotspots"]),
            "contention_clusters": list(search_signals["contention_clusters"]),
            "slot_duration": slot_duration,
        }

    def _compute_time_expanded_surrogate_state(self, mapping, *, collect_signals=True, include_hotspots=True):
        if not collect_signals:
            cpp_state = self._cpp_time_expanded_score(mapping)
            if cpp_state is not None:
                return cpp_state

        edge_capacity = self.data["edge_capacity"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        slot_duration = self._estimate_time_slot_duration(mapping)
        if self.horizon_slots_override is not None:
            max_slots = max(1, int(self.horizon_slots_override))
        else:
            task_count = sum(len(meta["task_info"]) for meta in self.data.get("task_surrogate", {}).values())
            max_slots = max(256, min(8192, 8 * max(task_count, 1)))

        task_state, successors, predecessors, remaining_preds = self._build_time_expanded_task_state(mapping)
        pending = set(task_state)
        active: set[tuple[int, int]] = set()
        finish_times: dict[tuple[int, int], float] = {}
        slot_prices = []
        slot_maxima = []
        slot_resource_pressure = []
        slot_active_tasks = []
        slot_ready_tasks = []
        slot_service_rates = []
        task_active_slots: dict[tuple[int, int], set[int]] = defaultdict(set)
        task_ready_slots: dict[tuple[int, int], set[int]] = defaultdict(set)
        global_max_epoch = int(self.data["compiled_schedule"]["global_max_epoch"])
        epoch_active_slots = [set() for _ in range(global_max_epoch + 1)]
        tenant_pressure: dict[int, float] = defaultdict(float)
        tenant_peak_load: dict[int, float] = defaultdict(float)
        contention_potential = 0.0

        current_time = 0.0
        slot_idx = 0
        while (pending or active) and slot_idx < max_slots:
            newly_ready = []
            for task_key in list(pending):
                if remaining_preds.get(task_key, 0) != 0:
                    continue
                if self._time_expanded_release_time(task_key, task_state, finish_times, slot_duration) <= current_time + 1e-12:
                    newly_ready.append(task_key)
            for task_key in newly_ready:
                pending.remove(task_key)
                active.add(task_key)
                if task_state[task_key]["start_time"] is None:
                    task_state[task_key]["start_time"] = current_time

            if not active:
                candidate_releases = [
                    self._time_expanded_release_time(task_key, task_state, finish_times, slot_duration)
                    for task_key in pending
                    if remaining_preds.get(task_key, 0) == 0
                ]
                finite_releases = [release_time for release_time in candidate_releases if math.isfinite(release_time)]
                next_release = min(finite_releases, default=current_time + slot_duration)
                if collect_signals:
                    slot_prices.append({"edge": {}, "sender": {}, "receiver": {}})
                    slot_resource_pressure.append({"edge": {}, "sender": {}, "receiver": {}})
                    slot_active_tasks.append(set())
                    slot_ready_tasks.append(set())
                    slot_service_rates.append({})
                    slot_maxima.append(0.0)
                current_time = max(current_time + slot_duration, float(next_release))
                slot_idx += 1
                continue

            edge_pressure: dict[tuple[int, int], float] = defaultdict(float)
            sender_pressure: dict[int, float] = defaultdict(float)
            receiver_pressure: dict[int, float] = defaultdict(float)
            ready_active = set(active)
            service_active = self._time_expanded_service_frontier(active, task_state)

            for task_key in service_active:
                state = task_state[task_key]
                remaining_bits = float(state["remaining_bits"])
                src_server = int(state["src_server"])
                dst_server = int(state["dst_server"])
                sender_pressure[src_server] += min(remaining_bits, server_send_capacity[src_server] * slot_duration) / max(
                    server_send_capacity[src_server] * slot_duration,
                    1e-12,
                )
                receiver_pressure[dst_server] += min(remaining_bits, server_recv_capacity[dst_server] * slot_duration) / max(
                    server_recv_capacity[dst_server] * slot_duration,
                    1e-12,
                )
                for edge in state["path"]:
                    edge_pressure[edge] += min(remaining_bits, edge_capacity[edge] * slot_duration) / max(
                        edge_capacity[edge] * slot_duration,
                        1e-12,
                    )

            for pressure_state in (edge_pressure, sender_pressure, receiver_pressure):
                for normalized_load in pressure_state.values():
                    excess = max(0.0, float(normalized_load) - 1.0)
                    contention_potential += excess * excess

            service_rate_by_task = self._max_min_service_rates(service_active, task_state, slot_duration)

            if collect_signals:
                for task_key in service_active:
                    state = task_state[task_key]
                    tenant = int(state["tenant"])
                    src_server = int(state["src_server"])
                    dst_server = int(state["dst_server"])
                    task_pressure = max(
                        float(sender_pressure.get(src_server, 0.0)),
                        float(receiver_pressure.get(dst_server, 0.0)),
                        max((float(edge_pressure.get(edge, 0.0)) for edge in state["path"]), default=0.0),
                    )
                    tenant_pressure[tenant] += task_pressure
                    tenant_peak_load[tenant] = max(float(tenant_peak_load.get(tenant, 0.0)), task_pressure)
                for task_key in ready_active:
                    task_ready_slots[task_key].add(slot_idx)

            if collect_signals:
                slot_active = set(service_active)
                slot_service_rates.append(dict(service_rate_by_task))
            completed = []
            for task_key in list(service_active):
                rate = service_rate_by_task.get(task_key, 0.0)
                if rate <= 0.0:
                    continue
                state = task_state[task_key]
                service_bits = min(float(state["remaining_bits"]), rate * slot_duration)
                state["remaining_bits"] = float(state["remaining_bits"]) - service_bits
                if collect_signals:
                    task_active_slots[task_key].add(slot_idx)
                    task_epoch = int(task_state[task_key].get("epoch", 0))
                    if 0 <= task_epoch <= global_max_epoch:
                        epoch_active_slots[task_epoch].add(slot_idx)
                if state["remaining_bits"] <= 1e-9:
                    finish_time = current_time + service_bits / max(rate, 1e-12)
                    state["finish_time"] = finish_time
                    finish_times[task_key] = finish_time
                    completed.append(task_key)

            for task_key in completed:
                active.remove(task_key)
                for succ_key in successors.get(task_key, []):
                    remaining_preds[succ_key] = int(remaining_preds.get(succ_key, 0)) - 1

            if collect_signals:
                slot_prices.append({"edge": {}, "sender": {}, "receiver": {}})
                slot_resource_pressure.append({
                    "edge": dict(edge_pressure),
                    "sender": dict(sender_pressure),
                    "receiver": dict(receiver_pressure),
                })
                slot_active_tasks.append(set(slot_active))
                slot_ready_tasks.append(set(ready_active))
                slot_maxima.append(
                    max(
                        max(edge_pressure.values(), default=0.0),
                        max(sender_pressure.values(), default=0.0),
                        max(receiver_pressure.values(), default=0.0),
                    )
                )
            current_time += slot_duration
            slot_idx += 1

        if pending or active:
            penalty_start = current_time
            for task_key in list(active) + list(pending):
                if task_key in finish_times:
                    continue
                state = task_state[task_key]
                src_server = int(state["src_server"])
                dst_server = int(state["dst_server"])
                bottleneck = min(
                    [server_send_capacity[src_server], server_recv_capacity[dst_server]]
                    + [edge_capacity[edge] for edge in state["path"]],
                    default=1.0,
                )
                penalty_start += float(state["remaining_bits"]) / max(bottleneck, 1e-12)
                state["finish_time"] = penalty_start
                finish_times[task_key] = penalty_start

        tenant_finish = {}
        for tenant in self.tenants:
            tenant_task_keys = [key for key in task_state if key[0] == int(tenant)]
            tenant_finish[tenant] = max((finish_times.get(key, 0.0) for key in tenant_task_keys), default=0.0)

        raw_score = (
            float(max(tenant_finish.values(), default=0.0)),
            float(sum(tenant_finish.values()) / max(len(tenant_finish), 1)),
        )
        contention_tiebreak = 1e-9 * float(contention_potential)
        score = (
            float(raw_score[0] + contention_tiebreak),
            float(raw_score[1] + contention_tiebreak),
        )
        if not collect_signals:
            return {
                "score": score,
                "raw_score": raw_score,
                "contention_potential": float(contention_potential),
                "tenant_finish": tenant_finish,
                "slot_duration": slot_duration,
            }

        critical_tasks = self._extract_time_expanded_critical_tasks(task_state, predecessors)
        critical_resources_by_slot = self._critical_resources_by_slot(
            task_state,
            slot_active_tasks,
            critical_tasks,
        )
        slot_prices = self._time_expanded_slot_prices_batch(
            slot_resource_pressure,
            slot_active_tasks,
            slot_service_rates,
            task_state,
            critical_resources_by_slot,
        )
        search_signals = self._time_expanded_search_signals(
            task_state,
            successors,
            slot_active_tasks,
            slot_ready_tasks,
            slot_service_rates,
            slot_prices,
            slot_resource_pressure,
            slot_duration,
            critical_tasks,
            include_hotspots=include_hotspots,
        )
        tenant_pressure = search_signals["tenant_pressure"]
        tenant_peak_load = search_signals["tenant_peak_load"]

        return {
            "score": score,
            "raw_score": raw_score,
            "contention_potential": float(contention_potential),
            "slot_prices": slot_prices,
            "slot_maxima": slot_maxima,
            "slot_resource_pressure": slot_resource_pressure,
            "slot_active_tasks": slot_active_tasks,
            "slot_ready_tasks": slot_ready_tasks,
            "slot_service_rates": slot_service_rates,
            "task_active_slots": task_active_slots,
            "task_ready_slots": task_ready_slots,
            "epoch_active_slots": epoch_active_slots,
            "task_state": task_state,
            "predecessors": predecessors,
            "critical_tasks": critical_tasks,
            "critical_resources_by_slot": critical_resources_by_slot,
            "tenant_finish": tenant_finish,
            "tenant_pressure": {int(tenant): float(value) for tenant, value in tenant_pressure.items()},
            "tenant_peak_load": {int(tenant): float(value) for tenant, value in tenant_peak_load.items()},
            "task_pressure": dict(search_signals["task_pressure"]),
            "rank_pressure": dict(search_signals["rank_pressure"]),
            "tenant_pair_interaction": dict(search_signals["tenant_pair_interaction"]),
            "hotspots": list(search_signals["hotspots"]),
            "contention_clusters": list(search_signals["contention_clusters"]),
            "slot_duration": slot_duration,
        }

    def _get_time_expanded_surrogate_state(self, mapping):
        mapping = self._canonicalize_ring_mapping(mapping)
        signature = self._mapping_signature(mapping)
        cached = self._time_expanded_state_cache.get(signature)
        if cached is not None:
            return cached
        state = self._compute_time_expanded_surrogate_state(mapping)
        self._time_expanded_state_cache[signature] = state
        while len(self._time_expanded_state_cache) > int(self._time_expanded_state_cache_limit):
            self._time_expanded_state_cache.pop(next(iter(self._time_expanded_state_cache)))
        self._surrogate_cache[signature] = state["score"]
        return state

    def _time_expanded_epoch_prices_from_state(self, state):
        global_max_epoch = int(self.data["compiled_schedule"]["global_max_epoch"])
        if _te_accel is None or not hasattr(_te_accel, "aggregate_epoch_prices"):
            raise RuntimeError("C++ aggregate_epoch_prices kernel is required")
        if "slot_prices" not in state or "epoch_active_slots" not in state:
            raise RuntimeError("time-expanded state lacks slot_prices or epoch_active_slots")
        epoch_maxima, epoch_prices = _te_accel.aggregate_epoch_prices(
            state["slot_prices"],
            state["epoch_active_slots"],
            global_max_epoch,
        )
        return list(epoch_maxima), list(epoch_prices)

    def _resource_price_value(self, capacity, normalized_load):
        base_cost = 1.0 / max(capacity, 1e-12)
        return base_cost * (1.0 + self.link_price_beta * (float(normalized_load) ** self.link_price_gamma))

    def _edge_price_value(self, edge, normalized_load):
        return self._resource_price_value(self.data["edge_capacity"][edge], normalized_load)
