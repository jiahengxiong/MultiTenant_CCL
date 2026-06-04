from __future__ import annotations

from collections import defaultdict
import itertools
import math
import time

from multitenant.collectives import has_collective_workload, normalize_collective_programs
from multitenant.objectives import lexicographic_better
from multitenant.simulator import simulate_collective
from multitenant.simulator.adapter import simulate_collective_details

from .DAG_generation import build_collective_dag_data


class MappingHeuristicSolver:
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
        self._surrogate_candidate_limit = 32 if self.tenant_collective_programs is not None else 16
        self._surrogate_candidates: list[tuple[tuple[float, float], tuple[tuple[int, tuple[int, ...]], ...], dict[int, dict[int, int]]]] = []
        self.tenant_pressure: dict[int, float] = {}
        self.tenant_peak_load: dict[int, float] = {}
        self.rank_pressure: dict[tuple[int, int], float] = {}
        self.tenant_pair_interaction: dict[tuple[int, int], float] = {}
        self.link_price_beta = 4.0
        self.link_price_gamma = 2.0
        self.critical_path_price_beta = 2.0
        server_count = int(
            getattr(
                self.datacenter,
                "num_server",
                sum(len(server_set) for server_set in self.server_sets.values()),
            )
        )
        self.max_price_rounds = max(1, min(server_count, 8))
        self.data = self._build_data()

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

    @staticmethod
    def _time_expanded_service_frontier(active, task_state):
        """Return tasks at the head of each tenant outgoing-port queue.

        The collective DAG controls readiness.  Sender-side communication
        serialization is modeled as a resource discipline: for the same tenant
        and the same outgoing port, packets drain FIFO; different tenants are
        independent queues and only interact through shared physical resources.
        """
        frontier: dict[tuple[int, object], tuple[tuple[float, int, int], tuple[int, int]]] = {}
        for task_key in active:
            state = task_state[task_key]
            queue_key = MappingHeuristicSolver._time_expanded_queue_key(task_key, task_state)
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
        slot_prices = [
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

    def _compute_time_expanded_surrogate_state(self, mapping, *, collect_signals=True):
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
        slot_prices = [
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
            "task_ready_slots": task_ready_slots,
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
        self._surrogate_cache[signature] = state["score"]
        return state

    def _time_expanded_epoch_prices_from_state(self, state):
        global_max_epoch = int(self.data["compiled_schedule"]["global_max_epoch"])
        epoch_slot_sets = [set() for _ in range(global_max_epoch + 1)]
        task_state = state["task_state"]
        for task_key, active_slots in state["task_active_slots"].items():
            task_epoch = int(task_state[task_key].get("epoch", 0))
            if 0 <= task_epoch <= global_max_epoch:
                epoch_slot_sets[task_epoch].update(int(slot) for slot in active_slots)

        epoch_prices: list[dict[str, dict[object, float]]] = []
        epoch_maxima = []
        slot_prices = state["slot_prices"]
        for slot_set in epoch_slot_sets:
            edge_prices: dict[object, float] = defaultdict(float)
            sender_prices: dict[object, float] = defaultdict(float)
            receiver_prices: dict[object, float] = defaultdict(float)
            for slot_idx in slot_set:
                if slot_idx < 0 or slot_idx >= len(slot_prices):
                    continue
                price_state = slot_prices[slot_idx]
                for edge, price in price_state.get("edge", {}).items():
                    edge_prices[edge] = max(float(edge_prices[edge]), float(price))
                for server, price in price_state.get("sender", {}).items():
                    sender_prices[server] = max(float(sender_prices[server]), float(price))
                for server, price in price_state.get("receiver", {}).items():
                    receiver_prices[server] = max(float(receiver_prices[server]), float(price))
            epoch_prices.append({
                "edge": dict(edge_prices),
                "sender": dict(sender_prices),
                "receiver": dict(receiver_prices),
            })
            epoch_maxima.append(
                max(
                    max(edge_prices.values(), default=0.0),
                    max(sender_prices.values(), default=0.0),
                    max(receiver_prices.values(), default=0.0),
                )
            )
        return epoch_maxima, epoch_prices

    def _resource_price_value(self, capacity, normalized_load):
        base_cost = 1.0 / max(capacity, 1e-12)
        return base_cost * (1.0 + self.link_price_beta * (float(normalized_load) ** self.link_price_gamma))

    def _edge_price_value(self, edge, normalized_load):
        return self._resource_price_value(self.data["edge_capacity"][edge], normalized_load)

    def _compute_epoch_link_prices(self, mapping):
        self._pair_epoch_price_cache.clear()
        if self.surrogate_mode == "time_expanded":
            state = self._get_time_expanded_surrogate_state(mapping)
            epoch_maxima, epoch_prices = self._time_expanded_epoch_prices_from_state(state)
            self.tenant_pressure = dict(state.get("tenant_pressure", {}))
            self.tenant_peak_load = dict(state.get("tenant_peak_load", {}))
            self.rank_pressure = dict(state.get("rank_pressure", {}))
            self.tenant_pair_interaction = dict(state.get("tenant_pair_interaction", {}))
            return [], epoch_maxima, epoch_prices

        epoch_loads, epoch_sender_loads, epoch_receiver_loads, epoch_maxima = self._compute_epoch_resource_load_state(mapping)
        epoch_prices: list[dict[str, dict[object, float]]] = []
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        for epoch_idx, epoch_load in enumerate(epoch_loads):
            edge_prices = {}
            for edge, normalized_load in epoch_load.items():
                edge_prices[edge] = self._edge_price_value(edge, normalized_load)
            sender_prices = {}
            for server, normalized_load in epoch_sender_loads[epoch_idx].items():
                sender_prices[server] = self._resource_price_value(server_send_capacity[server], normalized_load)
            receiver_prices = {}
            for server, normalized_load in epoch_receiver_loads[epoch_idx].items():
                receiver_prices[server] = self._resource_price_value(server_recv_capacity[server], normalized_load)
            epoch_prices.append({
                "edge": edge_prices,
                "sender": sender_prices,
                "receiver": receiver_prices,
            })
        return epoch_loads, epoch_maxima, epoch_prices

    def _pair_epoch_price_lookup(self, tenant, candidate_servers, epoch_prices):
        cache_key = (
            id(epoch_prices),
            int(tenant),
            tuple(sorted(int(server) for server in candidate_servers)),
        )
        cached = self._pair_epoch_price_cache.get(cache_key)
        if cached is not None:
            return cached

        lookup: dict[tuple[int, int, int], float] = {}
        candidate_server_tuple = cache_key[2]
        for epoch, _edge_prices in enumerate(epoch_prices):
            for src_server in candidate_server_tuple:
                for dst_server in candidate_server_tuple:
                    if int(src_server) == int(dst_server):
                        continue
                    lookup[(epoch, int(src_server), int(dst_server))] = self._path_epoch_price(
                        epoch_prices,
                        tenant,
                        epoch,
                        int(src_server),
                        int(dst_server),
                    )
        self._pair_epoch_price_cache[cache_key] = lookup
        return lookup

    def _path_epoch_price(self, epoch_prices, tenant, epoch, src_server, dst_server):
        path_edges = self.data["path_edges"]
        epoch_price_state = epoch_prices[int(epoch)]
        edge_prices = epoch_price_state["edge"]
        sender_prices = epoch_price_state["sender"]
        receiver_prices = epoch_price_state["receiver"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        if self.surrogate_mode == "time_expanded":
            resource_prices = [
                sender_prices.get(src_server, self._resource_price_value(server_send_capacity[src_server], 0.0)),
                receiver_prices.get(dst_server, self._resource_price_value(server_recv_capacity[dst_server], 0.0)),
            ]
            resource_prices.extend(
                edge_prices.get(edge, self._edge_price_value(edge, 0.0))
                for edge in self._path_edges_for_pair(path_edges, tenant, src_server, dst_server)
            )
            # The time-expanded estimator serves each task as a fluid flow whose
            # progress is limited by the tightest resource on its server-to-server
            # path.  Expose the same bottleneck semantics to the proposal search:
            # a candidate pair is priced by its most expensive sender/receiver/link
            # resource, not by summing all pipelined links along the path.
            return float(max(resource_prices, default=0.0))
        path_cost = 0.0
        path_cost += sender_prices.get(src_server, self._resource_price_value(server_send_capacity[src_server], 0.0))
        for edge in self._path_edges_for_pair(path_edges, tenant, src_server, dst_server):
            path_cost += edge_prices.get(edge, self._edge_price_value(edge, 0.0))
        path_cost += receiver_prices.get(dst_server, self._resource_price_value(server_recv_capacity[dst_server], 0.0))
        return float(path_cost)

    def _tenant_price_cost(self, tenant, mapping, pair_epoch_price):
        compiled_tenant = self.data["compiled_schedule"]["per_tenant"][tenant]
        total_cost = 0.0
        for epoch, src_rank, dst_rank, volume in compiled_tenant["all_flows"]:
            src_server = int(mapping[tenant][src_rank])
            dst_server = int(mapping[tenant][dst_rank])
            total_cost += float(volume) * pair_epoch_price[(int(epoch), src_server, dst_server)]
        return float(total_cost)

    def _surrogate_score_from_epoch_max(self, epoch_maxima):
        epoch_prefix: list[float] = []
        running = 0.0
        for delta in epoch_maxima:
            running += float(delta)
            epoch_prefix.append(running)

        tenant_finish: list[float] = []
        compiled_schedule = self.data["compiled_schedule"]["per_tenant"]
        for tenant in self.tenants:
            max_epoch = int(compiled_schedule[tenant]["max_epoch"])
            if max_epoch < 0:
                tenant_finish.append(0.0)
            else:
                tenant_finish.append(
                    epoch_prefix[max_epoch]
                    + float(compiled_schedule[tenant].get("tenant_gap_time", 0.0))
                )

        return (
            float(max(tenant_finish, default=0.0)),
            float(sum(tenant_finish) / max(len(tenant_finish), 1)),
        )

    def _surrogate_score_from_tenant_epoch_max(self, tenant_epoch_maxima):
        tenant_finish: list[float] = []
        compiled_schedule = self.data["compiled_schedule"]["per_tenant"]
        for tenant in self.tenants:
            max_epoch = int(compiled_schedule[tenant]["max_epoch"])
            epoch_deltas = tenant_epoch_maxima.get(tenant, [])
            if max_epoch < 0:
                tenant_finish.append(0.0)
            else:
                tenant_finish.append(
                    sum(float(delta) for delta in epoch_deltas[: max_epoch + 1])
                    + float(compiled_schedule[tenant].get("tenant_gap_time", 0.0))
                )

        return (
            float(max(tenant_finish, default=0.0)),
            float(sum(tenant_finish) / max(len(tenant_finish), 1)),
        )

    def _evaluate_surrogate_mapping(self, mapping):
        mapping = self._canonicalize_ring_mapping(mapping)
        signature = self._mapping_signature(mapping)
        cached = self._surrogate_cache.get(signature)
        if cached is not None:
            return cached

        if not self._collective_mode():
            raise ValueError(
                "Structured pure mapping solver currently requires collective inputs via "
                "tenant_collective_programs, tenant_collective_specs, or collective/single_flow_size."
            )

        if self.surrogate_mode == "task":
            level_edge_loads, level_sender_loads, level_receiver_loads = self._compute_task_level_resource_load_state(mapping)
            tenant_finish = {}
            path_edges = self.data["path_edges"]
            compiled_schedule = self.data["compiled_schedule"]["per_tenant"]
            task_surrogate = self.data.get("task_surrogate", {})
            for tenant, tenant_meta in task_surrogate.items():
                finish_by_task = {}
                for task_id in tenant_meta["task_order"]:
                    task_level = int(tenant_meta["task_levels"][int(task_id)])
                    _task_id, src_rank, dst_rank, _volume = tenant_meta["task_info"][int(task_id)]
                    src_server = int(mapping[tenant][src_rank])
                    dst_server = int(mapping[tenant][dst_rank])
                    path = self._path_edges_for_pair(path_edges, tenant, src_server, dst_server)
                    task_cost = max(
                        max((level_edge_loads[task_level][edge] for edge in path), default=0.0),
                        float(level_sender_loads[task_level].get(src_server, 0.0)),
                        float(level_receiver_loads[task_level].get(dst_server, 0.0)),
                    )
                    ready_time = max(
                        (finish_by_task[int(pred)] for pred in tenant_meta["preds"].get(int(task_id), [])),
                        default=0.0,
                    )
                    finish_by_task[int(task_id)] = ready_time + float(task_cost)
                tenant_finish[tenant] = (
                    max(finish_by_task.values(), default=0.0)
                    + float(compiled_schedule[tenant].get("tenant_gap_time", 0.0))
                )
            score = (
                float(max(tenant_finish.values(), default=0.0)),
                float(sum(tenant_finish.values()) / max(len(tenant_finish), 1)),
            )
            self._surrogate_cache[signature] = score
            self._register_surrogate_candidate(mapping, score)
            return score

        if self.surrogate_mode == "time_expanded":
            state = self._compute_time_expanded_surrogate_state(mapping, collect_signals=False)
            score = state["score"]
            self._surrogate_cache[signature] = score
            self._register_surrogate_candidate(mapping, score)
            return score

        if self.surrogate_mode not in {"collapsed", "task", "time_expanded"}:
            raise ValueError(
                f"Unsupported surrogate_mode={self.surrogate_mode!r}; "
                "expected 'time_expanded', 'task', or 'collapsed' (legacy alias: 'epoch')."
            )

        _, global_epoch_deltas = self._compute_epoch_link_load_state(mapping)
        _, tenant_epoch_deltas = self._compute_tenant_epoch_link_load_state(mapping)

        global_score = self._surrogate_score_from_epoch_max(global_epoch_deltas)
        tenant_score = self._surrogate_score_from_tenant_epoch_max(tenant_epoch_deltas)
        score = (float(global_score[0]), float(tenant_score[1]))
        self._surrogate_cache[signature] = score
        self._register_surrogate_candidate(mapping, score)
        return score

    def _evaluate_mapping(self, mapping):
        mapping = self._canonicalize_ring_mapping(mapping)
        signature = self._mapping_signature(mapping)
        cached = self._score_cache.get(signature)
        if cached is not None:
            return cached

        if not self._collective_mode():
            raise ValueError(
                "Pure mapping solver currently requires collective inputs via "
                "tenant_collective_programs, tenant_collective_specs, or collective/single_flow_size."
            )

        makespan, avg_jct = simulate_collective(
            self.datacenter.topology,
            mapping,
            self.path_table,
            tenant_start_times=self.tenant_start_times,
            tenant_collective_programs=self.tenant_collective_programs,
        )
        score = (float(makespan), float(avg_jct))
        self._score_cache[signature] = score
        return score

    def _apply_server_order(self, base_mapping, tenant, server_order):
        ranks = self.rank_orders[tenant]
        candidate = {
            current_tenant: dict(rank_to_server)
            for current_tenant, rank_to_server in base_mapping.items()
        }
        candidate[tenant] = {
            rank: int(server_order[idx])
            for idx, rank in enumerate(ranks)
        }
        return candidate

    def _seed_mappings(self):
        seeds = []
        seen = set()

        def add_seed(mapping):
            mapping = self._canonicalize_ring_mapping(mapping)
            signature = self._mapping_signature(mapping)
            if signature in seen:
                return
            seen.add(signature)
            seeds.append(mapping)

        for seed_mapping in self.extra_seed_mappings:
            add_seed(seed_mapping)

        add_seed(self.initial_tenant_mapping)

        # A global leaf-local ordering is a deterministic baseline that should
        # always be in the candidate pool, not only per-tenant one-off variants.
        global_leaf_local = {
            tenant: self._apply_server_order(
                self.initial_tenant_mapping,
                tenant,
                tuple(sorted(
                    [self.initial_tenant_mapping[tenant][rank] for rank in self.rank_orders[tenant]],
                    key=self._server_leaf_sort_key,
                )),
            )[tenant]
            for tenant in self.tenants
        }
        add_seed(global_leaf_local)
        global_leaf_local_rev = {
            tenant: self._apply_server_order(
                self.initial_tenant_mapping,
                tenant,
                tuple(reversed(sorted(
                    [self.initial_tenant_mapping[tenant][rank] for rank in self.rank_orders[tenant]],
                    key=self._server_leaf_sort_key,
                ))),
            )[tenant]
            for tenant in self.tenants
        }
        add_seed(global_leaf_local_rev)

        for tenant in self.tenants:
            ranks = self.rank_orders[tenant]
            current_servers = [self.initial_tenant_mapping[tenant][rank] for rank in ranks]
            if len(current_servers) <= 1:
                continue

            add_seed(self._apply_server_order(self.initial_tenant_mapping, tenant, tuple(reversed(current_servers))))
            leaf_local = tuple(sorted(current_servers, key=self._server_leaf_sort_key))
            add_seed(self._apply_server_order(self.initial_tenant_mapping, tenant, leaf_local))
            add_seed(self._apply_server_order(self.initial_tenant_mapping, tenant, tuple(reversed(leaf_local))))

            rotation_count = min(len(current_servers) - 1, 8)
            rotation_shifts = sorted({
                max(1, round(idx * len(current_servers) / (rotation_count + 1)))
                for idx in range(1, rotation_count + 1)
            })
            for shift in rotation_shifts:
                rotated = current_servers[shift:] + current_servers[:shift]
                add_seed(self._apply_server_order(self.initial_tenant_mapping, tenant, tuple(rotated)))

        return seeds

    def _server_leaf_sort_key(self, server):
        server_id = int(server)
        if hasattr(self.datacenter, "get_server_leaf"):
            return (int(self.datacenter.get_server_leaf(server_id)), server_id)
        topology = self.datacenter.topology
        for neighbor in topology.successors(server_id):
            if topology.nodes[neighbor].get("type") == "leaf":
                return (int(neighbor), server_id)
        return (server_id, server_id)

    def _optimize_tenant_block_with_prices(self, base_mapping, tenant, epoch_prices, deadline):
        ranks = self.rank_orders[tenant]
        current_servers = tuple(base_mapping[tenant][rank] for rank in ranks)
        if len(current_servers) <= 1:
            return base_mapping, self._tenant_price_cost(
                tenant,
                base_mapping,
                self._pair_epoch_price_lookup(tenant, current_servers, epoch_prices),
            )

        pair_epoch_price = self._pair_epoch_price_lookup(tenant, current_servers, epoch_prices)
        initial_cost = self._tenant_price_cost(tenant, base_mapping, pair_epoch_price)

        compiled_tenant = self.data["compiled_schedule"]["per_tenant"][tenant]
        all_flows = compiled_tenant["all_flows"]
        branch_ranks = [int(rank) for rank in compiled_tenant["branch_order"] if int(rank) in ranks]
        branch_position = {int(rank): idx for idx, rank in enumerate(branch_ranks)}
        rank_pressure = getattr(self, "rank_pressure", {})
        branch_ranks.sort(
            key=lambda rank: (
                -float(rank_pressure.get((int(tenant), int(rank)), 0.0)),
                branch_position.get(int(rank), len(branch_position)),
                int(rank),
            )
        )
        rank_incidence = compiled_tenant["rank_incidence"]

        assignment: dict[int, int] = {}
        used_servers: set[int] = set()
        initial_assignment = {int(rank): int(base_mapping[tenant][rank]) for rank in ranks}
        max_price_candidates = 32
        price_candidates: list[tuple[float, dict[int, int]]] = [
            (float(initial_cost), dict(initial_assignment))
        ]

        def add_price_candidate(cost, candidate_assignment):
            normalized = {
                int(rank): int(server)
                for rank, server in candidate_assignment.items()
            }
            signature = tuple(normalized[rank] for rank in sorted(normalized))
            for existing_cost, existing_assignment in price_candidates:
                if signature == tuple(existing_assignment[rank] for rank in sorted(existing_assignment)):
                    return
            price_candidates.append((float(cost), normalized))
            price_candidates.sort(key=lambda item: item[0])
            del price_candidates[max_price_candidates:]

        def current_prune_cost():
            if len(price_candidates) < max_price_candidates:
                return float("inf")
            return float(price_candidates[-1][0])

        def lower_bound(current_partial_cost):
            remaining_servers = [int(server) for server in current_servers if int(server) not in used_servers]
            bound = float(current_partial_cost)
            for epoch, src_rank, dst_rank, volume in all_flows:
                src_rank = int(src_rank)
                dst_rank = int(dst_rank)
                src_assigned = src_rank in assignment
                dst_assigned = dst_rank in assignment
                if src_assigned and dst_assigned:
                    continue
                if src_assigned and not dst_assigned:
                    src_server = assignment[src_rank]
                    candidates = [
                        pair_epoch_price[(int(epoch), int(src_server), int(dst_server))]
                        for dst_server in remaining_servers
                        if int(dst_server) != int(src_server)
                    ]
                elif dst_assigned and not src_assigned:
                    dst_server = assignment[dst_rank]
                    candidates = [
                        pair_epoch_price[(int(epoch), int(src_server), int(dst_server))]
                        for src_server in remaining_servers
                        if int(src_server) != int(dst_server)
                    ]
                else:
                    candidates = [
                        pair_epoch_price[(int(epoch), int(src_server), int(dst_server))]
                        for src_server in remaining_servers
                        for dst_server in remaining_servers
                        if int(src_server) != int(dst_server)
                    ]
                if candidates:
                    bound += float(volume) * min(candidates)
            return float(bound)

        def rank_server_price(rank, server, partial_assignment, available_servers):
            exact_delta = 0.0
            optimistic_delta = 0.0
            remaining_after = [
                int(candidate_server)
                for candidate_server in available_servers
                if int(candidate_server) != int(server)
            ]
            for epoch, src_rank, dst_rank, volume in rank_incidence.get(int(rank), []):
                src_rank = int(src_rank)
                dst_rank = int(dst_rank)
                other_rank = dst_rank if src_rank == int(rank) else src_rank
                if other_rank in partial_assignment:
                    if src_rank == int(rank):
                        src_server = int(server)
                        dst_server = int(partial_assignment[other_rank])
                    else:
                        src_server = int(partial_assignment[other_rank])
                        dst_server = int(server)
                    exact_delta += float(volume) * pair_epoch_price[(int(epoch), src_server, dst_server)]
                    continue

                if src_rank == int(rank):
                    candidates = [
                        pair_epoch_price[(int(epoch), int(server), int(dst_server))]
                        for dst_server in remaining_after
                        if int(dst_server) != int(server)
                    ]
                else:
                    candidates = [
                        pair_epoch_price[(int(epoch), int(src_server), int(server))]
                        for src_server in remaining_after
                        if int(src_server) != int(server)
                    ]
                if candidates:
                    optimistic_delta += float(volume) * min(candidates)
            return float(exact_delta), float(optimistic_delta)

        def candidate_servers_for_rank(rank):
            available = [int(server) for server in current_servers if int(server) not in used_servers]
            preferred = int(base_mapping[tenant][rank])
            scored = []
            for server in available:
                exact_delta, optimistic_delta = rank_server_price(
                    rank,
                    server,
                    assignment,
                    available,
                )
                scored.append((
                    float(exact_delta + optimistic_delta),
                    0 if int(server) == preferred else 1,
                    int(server),
                ))
            scored.sort(key=lambda item: (item[0], item[1], item[2]))
            return [server for _, _, server in scored]

        def apply_rank(rank, server, current_partial_cost):
            assignment[int(rank)] = int(server)
            used_servers.add(int(server))
            delta = 0.0
            for epoch, src_rank, dst_rank, volume in rank_incidence.get(int(rank), []):
                src_rank = int(src_rank)
                dst_rank = int(dst_rank)
                other_rank = dst_rank if src_rank == int(rank) else src_rank
                if other_rank not in assignment:
                    continue
                if src_rank == int(rank):
                    src_server = int(server)
                    dst_server = int(assignment[other_rank])
                else:
                    src_server = int(assignment[other_rank])
                    dst_server = int(server)
                delta += float(volume) * pair_epoch_price[(int(epoch), src_server, dst_server)]
            return current_partial_cost + delta

        def rollback_rank(rank, server):
            used_servers.remove(int(server))
            del assignment[int(rank)]

        def greedy_price_assignment():
            greedy_assignment: dict[int, int] = {}
            greedy_used_servers: set[int] = set()
            greedy_cost = 0.0

            for rank in branch_ranks:
                available = [
                    int(server)
                    for server in current_servers
                    if int(server) not in greedy_used_servers
                ]
                preferred = int(base_mapping[tenant][rank])
                scored_servers = []
                for server in available:
                    exact_delta, optimistic_delta = rank_server_price(
                        rank,
                        server,
                        greedy_assignment,
                        available,
                    )
                    scored_servers.append((
                        float(exact_delta + optimistic_delta),
                        0 if int(server) == preferred else 1,
                        float(exact_delta),
                        int(server),
                    ))
                if not scored_servers:
                    continue
                scored_servers.sort(key=lambda item: (item[0], item[1], item[3]))
                _ranking_delta, _preferred_rank, exact_delta, chosen_server = scored_servers[0]
                greedy_assignment[int(rank)] = int(chosen_server)
                greedy_used_servers.add(int(chosen_server))
                greedy_cost += float(exact_delta)

            if len(greedy_assignment) == len(branch_ranks):
                add_price_candidate(greedy_cost, greedy_assignment)

        greedy_price_assignment()

        def dfs(depth, current_partial_cost):
            if time.time() >= deadline:
                return

            bound = lower_bound(current_partial_cost)
            if bound >= current_prune_cost() - 1e-12:
                return

            if depth >= len(branch_ranks):
                add_price_candidate(current_partial_cost, assignment)
                return

            rank = branch_ranks[depth]
            for server in candidate_servers_for_rank(rank):
                if time.time() >= deadline:
                    break
                next_cost = apply_rank(rank, server, current_partial_cost)
                dfs(depth + 1, next_cost)
                rollback_rank(rank, server)

        dfs(0, 0.0)

        best_mapping = base_mapping
        best_score = self._evaluate_surrogate_mapping(base_mapping)
        best_cost = float(initial_cost)

        for candidate_cost, candidate_assignment in price_candidates:
            if time.time() >= deadline:
                break
            if candidate_assignment == initial_assignment:
                continue
            candidate_mapping = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            candidate_mapping[tenant] = {
                rank: int(candidate_assignment[int(rank)])
                for rank in ranks
            }
            candidate_mapping = self._canonicalize_ring_mapping(candidate_mapping)
            candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
            if self._is_better_objective(candidate_score, best_score):
                best_mapping = candidate_mapping
                best_score = candidate_score
                best_cost = float(candidate_cost)
        return best_mapping, float(best_cost)

    def _surrogate_pair_swap_polish(self, best_mapping, best_score, deadline):
        improved = True
        while improved and time.time() < deadline:
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
                current_servers = [best_mapping[tenant][rank] for rank in ranks]
                for left_idx in range(len(current_servers)):
                    for right_idx in range(left_idx + 1, len(current_servers)):
                        if time.time() >= deadline:
                            break
                        swapped = list(current_servers)
                        swapped[left_idx], swapped[right_idx] = swapped[right_idx], swapped[left_idx]
                        candidate_mapping = self._apply_server_order(best_mapping, tenant, tuple(swapped))
                        candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
                        if self._is_better_objective(candidate_score, best_score):
                            best_mapping = candidate_mapping
                            best_score = candidate_score
                            improved = True
                            current_servers = [best_mapping[tenant][rank] for rank in ranks]
                            break
                    if improved or time.time() >= deadline:
                        break
                if improved or time.time() >= deadline:
                    break

        return best_mapping, best_score

    def _simulator_rerank_candidates(self, deadline):
        if not self._surrogate_candidates:
            return (
                {
                    tenant: dict(rank_to_server)
                    for tenant, rank_to_server in self.initial_tenant_mapping.items()
                },
                self._evaluate_mapping(self.initial_tenant_mapping),
            )

        best_mapping = None
        best_score = (float("inf"), float("inf"))
        for _surrogate_score, _signature, mapping in self._surrogate_candidates:
            if time.time() >= deadline:
                break
            candidate_score = self._evaluate_mapping(mapping)
            if self._is_better_objective(candidate_score, best_score):
                best_mapping = {
                    tenant: dict(rank_to_server)
                    for tenant, rank_to_server in mapping.items()
                }
                best_score = candidate_score

        if best_mapping is None:
            _, _, mapping = self._surrogate_candidates[0]
            best_mapping = {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in mapping.items()
            }
            best_score = self._evaluate_mapping(best_mapping)

        return best_mapping, best_score

    def _simulator_pair_swap_polish(self, best_mapping, best_score, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            self.search_rounds += 1

            for tenant in self.tenants:
                if time.time() >= deadline:
                    break
                ranks = self.rank_orders[tenant]
                current_servers = [best_mapping[tenant][rank] for rank in ranks]
                for left_idx in range(len(current_servers)):
                    for right_idx in range(left_idx + 1, len(current_servers)):
                        if time.time() >= deadline:
                            break
                        swapped = list(current_servers)
                        swapped[left_idx], swapped[right_idx] = swapped[right_idx], swapped[left_idx]
                        candidate_mapping = self._apply_server_order(best_mapping, tenant, tuple(swapped))
                        candidate_score = self._evaluate_mapping(candidate_mapping)
                        if self._is_better_objective(candidate_score, best_score):
                            best_mapping = candidate_mapping
                            best_score = candidate_score
                            improved = True
                            current_servers = [best_mapping[tenant][rank] for rank in ranks]
                            break
                    if improved or time.time() >= deadline:
                        break
                if improved or time.time() >= deadline:
                    break

        return best_mapping, best_score

    def solve(self, time_limit=None):
        start_time = time.time()
        total_budget = float(time_limit) if time_limit is not None else 30.0
        deadline = start_time + total_budget
        if self.validate_with_simulator:
            polish_budget = min(max(total_budget * 0.25, 3.0), 10.0)
            search_deadline = max(start_time, deadline - polish_budget)
        else:
            search_deadline = deadline

        best_mapping = None
        best_score = (float("inf"), float("inf"))

        for seed_mapping in self._seed_mappings():
            if time.time() >= search_deadline:
                break
            candidate_score = self._evaluate_surrogate_mapping(seed_mapping)
            if self._is_better_objective(candidate_score, best_score):
                best_mapping = seed_mapping
                best_score = candidate_score

        if best_mapping is None:
            best_mapping = {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in self.initial_tenant_mapping.items()
            }
            best_score = self._evaluate_surrogate_mapping(best_mapping)

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
            best_mapping, best_score = self._surrogate_pair_swap_polish(best_mapping, best_score, search_deadline)

        self._register_surrogate_candidate(best_mapping, best_score)
        final_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in best_mapping.items()
        }
        final_score = best_score
        self.final_score_is_simulated = False
        if self.validate_with_simulator:
            final_mapping, final_score = self._simulator_rerank_candidates(deadline)
            if time.time() < deadline:
                final_mapping, final_score = self._simulator_pair_swap_polish(
                    final_mapping,
                    final_score,
                    deadline,
                )
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
                "Mode=link-price-bnb, "
                f"Score={'sim' if self.final_score_is_simulated else 'surrogate'}, "
                f"SurrogateEvalCount={len(self._surrogate_cache)}, "
                f"SimEvalCount={len(self._score_cache)}, Runtime={self.runtime_seconds:.2f}s"
            )

        return self

    def get_X_mapping(self, thr=0.5):
        del thr
        if self.final_mapping is not None:
            return {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in self.final_mapping.items()
            }
        return {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in self.initial_tenant_mapping.items()
        }


PureMappingSolver = MappingHeuristicSolver
MappingSearchSolver = MappingHeuristicSolver

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
        beam_width=3,
        miqp_pricing_time_limit=5.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bnb_candidate_time_limit = float(bnb_candidate_time_limit)
        self.max_bnb_tenants_per_round = int(max_bnb_tenants_per_round)
        self.beam_width = max(1, int(beam_width))
        self.miqp_pricing_time_limit = max(0.0, float(miqp_pricing_time_limit))
        self.last_move_source = None
        self.move_source_counts = {"local": 0, "bnb": 0, "joint": 0}

    def _build_price_bnb_candidate(self, base_mapping, tenant, epoch_prices, deadline):
        if time.time() >= deadline:
            return base_mapping
        candidate_deadline = min(deadline, time.time() + self.bnb_candidate_time_limit)
        if self.surrogate_mode == "time_expanded":
            return self._build_time_expanded_bnb_candidate(
                base_mapping,
                tenant,
                candidate_deadline,
            )
        return MappingHeuristicSolver._optimize_tenant_block_with_prices(
            self,
            base_mapping,
            tenant,
            epoch_prices,
            candidate_deadline,
        )[0]

    def _time_expanded_task_pair_price_lookup(self, base_mapping, tenant, candidate_servers):
        state = self._get_time_expanded_surrogate_state(base_mapping)
        task_active_slots = state.get("task_active_slots", {})
        slot_prices = state.get("slot_prices", [])
        task_meta = self.data["task_surrogate"][tenant]
        path_edges = self.data["path_edges"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        lookup: dict[tuple[int, int, int], float] = {}

        for task_id, task_tuple in task_meta["task_info"].items():
            task_id = int(task_id)
            active_slots = sorted(int(slot) for slot in task_active_slots.get((int(tenant), task_id), []))
            if not active_slots:
                active_slots = [0]
            for src_server in candidate_servers:
                for dst_server in candidate_servers:
                    src_server = int(src_server)
                    dst_server = int(dst_server)
                    if src_server == dst_server:
                        continue
                    price_sum = 0.0
                    for slot_idx in active_slots:
                        price_state = slot_prices[slot_idx] if 0 <= slot_idx < len(slot_prices) else {}
                        sender_prices = price_state.get("sender", {})
                        receiver_prices = price_state.get("receiver", {})
                        edge_prices = price_state.get("edge", {})
                        resource_prices = [
                            sender_prices.get(
                                src_server,
                                self._resource_price_value(server_send_capacity[src_server], 0.0),
                            ),
                            receiver_prices.get(
                                dst_server,
                                self._resource_price_value(server_recv_capacity[dst_server], 0.0),
                            ),
                        ]
                        for edge in self._path_edges_for_pair(path_edges, tenant, src_server, dst_server):
                            resource_prices.append(edge_prices.get(edge, self._edge_price_value(edge, 0.0)))
                        price_sum += max((float(value) for value in resource_prices), default=0.0)
                    lookup[(task_id, src_server, dst_server)] = float(price_sum / max(len(active_slots), 1))
        return lookup

    def _build_time_expanded_bnb_candidate(
        self,
        base_mapping,
        tenant,
        deadline,
        *,
        return_pool=False,
        max_return=4,
    ):
        ranks = [int(rank) for rank in self.rank_orders[tenant]]
        if len(ranks) <= 1:
            return [] if return_pool else base_mapping

        current_servers = tuple(int(base_mapping[tenant][rank]) for rank in ranks)
        task_pair_price = self._time_expanded_task_pair_price_lookup(
            base_mapping,
            tenant,
            current_servers,
        )
        task_meta = self.data["task_surrogate"][tenant]
        task_infos = [
            (
                int(task_id),
                int(task_tuple[1]),
                int(task_tuple[2]),
                float(task_tuple[3]),
            )
            for task_id, task_tuple in task_meta["task_info"].items()
        ]
        rank_incidence: dict[int, list[tuple[int, int, int, float]]] = {
            int(rank): [] for rank in ranks
        }
        for task_id, src_rank, dst_rank, volume in task_infos:
            rank_incidence.setdefault(src_rank, []).append((task_id, src_rank, dst_rank, volume))
            rank_incidence.setdefault(dst_rank, []).append((task_id, src_rank, dst_rank, volume))

        branch_ranks = [
            int(rank)
            for rank in self.data["compiled_schedule"]["per_tenant"][tenant]["branch_order"]
            if int(rank) in set(ranks)
        ] or list(ranks)
        branch_position = {int(rank): idx for idx, rank in enumerate(branch_ranks)}
        rank_pressure = getattr(self, "rank_pressure", {})
        branch_ranks.sort(
            key=lambda rank: (
                -float(rank_pressure.get((int(tenant), int(rank)), 0.0)),
                branch_position.get(int(rank), len(branch_position)),
                int(rank),
            )
        )

        def assignment_cost(assignment):
            total = 0.0
            for task_id, src_rank, dst_rank, volume in task_infos:
                src_server = int(assignment[src_rank])
                dst_server = int(assignment[dst_rank])
                if src_server == dst_server:
                    continue
                total += float(volume) * task_pair_price[(task_id, src_server, dst_server)]
            return float(total)

        initial_assignment = {int(rank): int(base_mapping[tenant][rank]) for rank in ranks}
        price_candidates: list[tuple[float, dict[int, int]]] = [
            (assignment_cost(initial_assignment), dict(initial_assignment))
        ]
        max_price_candidates = 48
        assignment: dict[int, int] = {}
        used_servers: set[int] = set()

        def add_price_candidate(cost, candidate_assignment):
            normalized = {int(rank): int(server) for rank, server in candidate_assignment.items()}
            signature = tuple(normalized[rank] for rank in sorted(normalized))
            for _existing_cost, existing_assignment in price_candidates:
                if signature == tuple(existing_assignment[rank] for rank in sorted(existing_assignment)):
                    return
            price_candidates.append((float(cost), normalized))
            price_candidates.sort(key=lambda item: item[0])
            del price_candidates[max_price_candidates:]

        def current_prune_cost():
            if len(price_candidates) < max_price_candidates:
                return float("inf")
            return float(price_candidates[-1][0])

        def lower_bound(current_partial_cost):
            remaining_servers = [server for server in current_servers if server not in used_servers]
            bound = float(current_partial_cost)
            for task_id, src_rank, dst_rank, volume in task_infos:
                src_assigned = src_rank in assignment
                dst_assigned = dst_rank in assignment
                if src_assigned and dst_assigned:
                    continue
                if src_assigned:
                    src_server = assignment[src_rank]
                    candidates = [
                        task_pair_price[(task_id, src_server, dst_server)]
                        for dst_server in remaining_servers
                        if dst_server != src_server
                    ]
                elif dst_assigned:
                    dst_server = assignment[dst_rank]
                    candidates = [
                        task_pair_price[(task_id, src_server, dst_server)]
                        for src_server in remaining_servers
                        if src_server != dst_server
                    ]
                else:
                    candidates = [
                        task_pair_price[(task_id, src_server, dst_server)]
                        for src_server in remaining_servers
                        for dst_server in remaining_servers
                        if src_server != dst_server
                    ]
                if candidates:
                    bound += float(volume) * min(candidates)
            return float(bound)

        def incremental_rank_server_price(rank, server, partial_assignment, available_servers):
            exact_delta = 0.0
            optimistic_delta = 0.0
            remaining_after = [candidate for candidate in available_servers if candidate != int(server)]
            for task_id, src_rank, dst_rank, volume in rank_incidence.get(int(rank), []):
                other_rank = dst_rank if src_rank == int(rank) else src_rank
                if other_rank in partial_assignment:
                    if src_rank == int(rank):
                        src_server = int(server)
                        dst_server = int(partial_assignment[other_rank])
                    else:
                        src_server = int(partial_assignment[other_rank])
                        dst_server = int(server)
                    exact_delta += float(volume) * task_pair_price[(task_id, src_server, dst_server)]
                    continue
                if src_rank == int(rank):
                    candidates = [
                        task_pair_price[(task_id, int(server), dst_server)]
                        for dst_server in remaining_after
                        if dst_server != int(server)
                    ]
                else:
                    candidates = [
                        task_pair_price[(task_id, src_server, int(server))]
                        for src_server in remaining_after
                        if src_server != int(server)
                    ]
                if candidates:
                    optimistic_delta += float(volume) * min(candidates)
            return float(exact_delta), float(optimistic_delta)

        def candidate_servers_for_rank(rank):
            available = [server for server in current_servers if server not in used_servers]
            preferred = int(base_mapping[tenant][rank])
            scored = []
            for server in available:
                exact_delta, optimistic_delta = incremental_rank_server_price(
                    rank,
                    server,
                    assignment,
                    available,
                )
                scored.append((
                    float(exact_delta + optimistic_delta),
                    0 if int(server) == preferred else 1,
                    int(server),
                ))
            scored.sort(key=lambda item: (item[0], item[1], item[2]))
            return [server for _score, _preferred, server in scored]

        def apply_rank(rank, server, current_partial_cost):
            assignment[int(rank)] = int(server)
            used_servers.add(int(server))
            delta = 0.0
            for task_id, src_rank, dst_rank, volume in rank_incidence.get(int(rank), []):
                other_rank = dst_rank if src_rank == int(rank) else src_rank
                if other_rank not in assignment:
                    continue
                if src_rank == int(rank):
                    src_server = int(server)
                    dst_server = int(assignment[other_rank])
                else:
                    src_server = int(assignment[other_rank])
                    dst_server = int(server)
                delta += float(volume) * task_pair_price[(task_id, src_server, dst_server)]
            return current_partial_cost + delta

        def rollback_rank(rank, server):
            used_servers.remove(int(server))
            del assignment[int(rank)]

        def greedy_price_assignment():
            greedy_assignment: dict[int, int] = {}
            greedy_used: set[int] = set()
            greedy_cost = 0.0
            for rank in branch_ranks:
                available = [server for server in current_servers if server not in greedy_used]
                scored_servers = []
                for server in available:
                    exact_delta, optimistic_delta = incremental_rank_server_price(
                        rank,
                        server,
                        greedy_assignment,
                        available,
                    )
                    scored_servers.append((float(exact_delta + optimistic_delta), float(exact_delta), int(server)))
                if not scored_servers:
                    continue
                scored_servers.sort(key=lambda item: (item[0], item[2]))
                _score, exact_delta, server = scored_servers[0]
                greedy_assignment[int(rank)] = int(server)
                greedy_used.add(int(server))
                greedy_cost += float(exact_delta)
            if len(greedy_assignment) == len(branch_ranks):
                add_price_candidate(greedy_cost, greedy_assignment)

        greedy_price_assignment()

        def dfs(depth, current_partial_cost):
            if time.time() >= deadline:
                return
            if lower_bound(current_partial_cost) >= current_prune_cost() - 1e-12:
                return
            if depth >= len(branch_ranks):
                add_price_candidate(current_partial_cost, assignment)
                return
            rank = branch_ranks[depth]
            for server in candidate_servers_for_rank(rank):
                if time.time() >= deadline:
                    break
                next_cost = apply_rank(rank, server, current_partial_cost)
                dfs(depth + 1, next_cost)
                rollback_rank(rank, server)

        dfs(0, 0.0)

        candidate_mappings = []
        best_mapping = base_mapping
        best_score = self._evaluate_surrogate_mapping(base_mapping)
        for _candidate_cost, candidate_assignment in price_candidates:
            if time.time() >= deadline:
                break
            if candidate_assignment == initial_assignment:
                continue
            candidate_mapping = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            candidate_mapping[tenant] = {
                rank: int(candidate_assignment[int(rank)])
                for rank in ranks
            }
            candidate_mapping = self._canonicalize_ring_mapping(candidate_mapping)
            candidate_mappings.append(candidate_mapping)
            if return_pool and len(candidate_mappings) >= max_return:
                break
            if return_pool:
                continue
            candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
            if self._is_better_objective(candidate_score, best_score):
                best_mapping = candidate_mapping
                best_score = candidate_score
        if return_pool:
            return candidate_mappings
        return best_mapping

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

    def _time_expanded_direct_candidate_limit(self):
        return max(8, 2 * max(1, self.beam_width))

    def _time_expanded_hot_rank_order(self, tenant):
        ranks = [
            int(rank)
            for rank in self.data["compiled_schedule"]["per_tenant"][tenant]["branch_order"]
            if int(rank) in self.rank_orders[tenant]
        ] or [int(rank) for rank in self.rank_orders[tenant]]
        rank_pressure = getattr(self, "rank_pressure", {})
        branch_position = {int(rank): idx for idx, rank in enumerate(ranks)}
        return sorted(
            ranks,
            key=lambda rank: (
                -float(rank_pressure.get((int(tenant), int(rank)), 0.0)),
                branch_position.get(int(rank), len(branch_position)),
                int(rank),
            ),
        )

    def _anchored_reverse_leaf_order(self, server_order):
        servers = [int(server) for server in server_order]
        if len(servers) <= 1:
            return tuple(servers)
        anchor = min(servers)
        remaining = list(servers)
        remaining.remove(anchor)
        return (int(anchor),) + tuple(
            int(server)
            for server in reversed(sorted(remaining, key=self._server_leaf_sort_key))
        )

    def _time_expanded_graph_seed_mappings(self):
        seeds = []
        seen = set()

        def add_seed(mapping):
            mapping = self._canonicalize_ring_mapping(mapping)
            signature = self._mapping_signature(mapping)
            if signature in seen:
                return
            seen.add(signature)
            seeds.append(mapping)

        for seed_mapping in self._seed_mappings():
            add_seed(seed_mapping)

        global_anchor_reverse = {
            tenant: self._apply_server_order(
                self.initial_tenant_mapping,
                tenant,
                self._anchored_reverse_leaf_order(
                    [self.initial_tenant_mapping[tenant][rank] for rank in self.rank_orders[tenant]]
                ),
            )[tenant]
            for tenant in self.tenants
        }
        add_seed(global_anchor_reverse)

        for tenant in self.tenants:
            server_order = [
                self.initial_tenant_mapping[tenant][rank]
                for rank in self.rank_orders[tenant]
            ]
            add_seed(
                self._apply_server_order(
                    self.initial_tenant_mapping,
                    tenant,
                    self._anchored_reverse_leaf_order(server_order),
                )
            )

        return seeds

    def _time_expanded_order_candidates(self, base_mapping, tenant):
        ranks = list(self.rank_orders[tenant])
        if len(ranks) <= 2:
            return []

        current_order = tuple(int(base_mapping[tenant][rank]) for rank in ranks)
        leaf_order = tuple(sorted(current_order, key=self._server_leaf_sort_key))
        hot_rank_order = self._time_expanded_hot_rank_order(tenant)
        hot_first_ranks = tuple(hot_rank_order + [rank for rank in ranks if rank not in set(hot_rank_order)])
        high_capacity_servers = tuple(
            sorted(
                current_order,
                key=lambda server: (
                    -float(self.data["server_send_capacity"].get(int(server), 0.0)),
                    self._server_leaf_sort_key(server),
                ),
            )
        )

        server_orders = []
        seen_orders = set()

        def add_order(order):
            normalized = tuple(int(server) for server in order)
            if normalized == current_order or normalized in seen_orders:
                return
            seen_orders.add(normalized)
            server_orders.append(normalized)

        n = len(current_order)
        leaf_boundaries = {
            idx
            for idx in range(1, n)
            if self._server_leaf_sort_key(leaf_order[idx])[0]
            != self._server_leaf_sort_key(leaf_order[idx - 1])[0]
        }
        shifts = sorted({
            candidate_shift
            for boundary in leaf_boundaries
            for candidate_shift in (boundary - 1, boundary, boundary + 1)
            if 0 < candidate_shift < n
        })
        shift_count = min(3, max(0, n - 1))
        shifts.extend(
            max(1, round(idx * n / (shift_count + 1)))
            for idx in range(1, shift_count + 1)
            if 0 < round(idx * n / (shift_count + 1)) < n
        )
        shifts = sorted(set(shifts))

        for base_order in (
            leaf_order,
            tuple(reversed(leaf_order)),
            self._anchored_reverse_leaf_order(leaf_order),
            current_order,
        ):
            for shift in shifts:
                add_order(base_order[shift:] + base_order[:shift])

        add_order(tuple(reversed(current_order)))
        add_order(leaf_order)
        add_order(tuple(reversed(leaf_order)))
        add_order(self._anchored_reverse_leaf_order(current_order))
        add_order(self._anchored_reverse_leaf_order(leaf_order))
        add_order(high_capacity_servers)
        add_order(tuple(reversed(high_capacity_servers)))

        candidates = []
        base_signature = self._mapping_signature(base_mapping)
        for order in server_orders:
            candidate = self._apply_server_order(base_mapping, tenant, order)
            candidate = self._canonicalize_ring_mapping(candidate)
            if self._mapping_signature(candidate) != base_signature:
                candidates.append(candidate)

        # A second family maps the hottest logical ranks to the best current
        # server order.  This is still a complete rank-to-server mapping and is
        # evaluated by the time-expanded estimator before entering the beam.
        if hot_first_ranks and high_capacity_servers:
            candidate = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            for idx, rank in enumerate(hot_first_ranks):
                candidate[tenant][int(rank)] = int(high_capacity_servers[idx])
            candidate = self._canonicalize_ring_mapping(candidate)
            if self._mapping_signature(candidate) != base_signature:
                candidates.append(candidate)

        return candidates[:10]

    def _time_expanded_scored_swap_pairs(self, base_mapping, tenant):
        hot_order = self._time_expanded_hot_rank_order(tenant)
        if len(hot_order) <= 1:
            return []

        hot_limit = min(len(hot_order), 12)
        partner_limit = min(len(hot_order), 48)
        hot_ranks = hot_order[:hot_limit]
        partner_ranks = list(hot_order[:partner_limit])
        for rank in hot_ranks:
            if rank not in partner_ranks:
                partner_ranks.append(rank)

        current_servers = tuple(int(base_mapping[tenant][rank]) for rank in self.rank_orders[tenant])
        task_pair_price = self._time_expanded_task_pair_price_lookup(
            base_mapping,
            tenant,
            current_servers,
        )
        task_meta = self.data["task_surrogate"][tenant]
        rank_incidence: dict[int, list[tuple[int, int, int, float]]] = {
            int(rank): [] for rank in self.rank_orders[tenant]
        }
        for task_id, task_tuple in task_meta["task_info"].items():
            task_id = int(task_id)
            src_rank = int(task_tuple[1])
            dst_rank = int(task_tuple[2])
            volume = float(task_tuple[3])
            rank_incidence.setdefault(src_rank, []).append((task_id, src_rank, dst_rank, volume))
            rank_incidence.setdefault(dst_rank, []).append((task_id, src_rank, dst_rank, volume))

        def swap_delta(left_rank, right_rank):
            affected = {}
            for rank in (int(left_rank), int(right_rank)):
                for task in rank_incidence.get(rank, []):
                    affected[(task[0], task[1], task[2], task[3])] = task
            left_server = int(base_mapping[tenant][left_rank])
            right_server = int(base_mapping[tenant][right_rank])

            def swapped_server(rank):
                if int(rank) == int(left_rank):
                    return right_server
                if int(rank) == int(right_rank):
                    return left_server
                return int(base_mapping[tenant][rank])

            delta = 0.0
            for task_id, src_rank, dst_rank, volume in affected.values():
                old_src = int(base_mapping[tenant][src_rank])
                old_dst = int(base_mapping[tenant][dst_rank])
                new_src = swapped_server(src_rank)
                new_dst = swapped_server(dst_rank)
                old_cost = task_pair_price.get((task_id, old_src, old_dst), 0.0)
                new_cost = task_pair_price.get((task_id, new_src, new_dst), 0.0)
                delta += float(volume) * (float(new_cost) - float(old_cost))
            return float(delta)

        base_signature = self._mapping_signature(base_mapping)
        scored_pairs = []
        seen_pairs = set()
        for left_rank in hot_ranks:
            for right_rank in partner_ranks:
                if int(left_rank) == int(right_rank):
                    continue
                pair = tuple(sorted((int(left_rank), int(right_rank))))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                scored_pairs.append((swap_delta(pair[0], pair[1]), pair))

        scored_pairs.sort(key=lambda item: (item[0], item[1]))
        return scored_pairs

    def _time_expanded_swap_candidates(self, base_mapping, tenant, *, max_candidates=3):
        scored_pairs = self._time_expanded_scored_swap_pairs(base_mapping, tenant)
        if not scored_pairs:
            return []
        candidates = []
        base_signature = self._mapping_signature(base_mapping)
        for _delta, pair in scored_pairs[:max_candidates]:
            candidate = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            candidate[tenant][pair[0]], candidate[tenant][pair[1]] = (
                candidate[tenant][pair[1]],
                candidate[tenant][pair[0]],
            )
            candidate = self._canonicalize_ring_mapping(candidate)
            if self._mapping_signature(candidate) == base_signature:
                continue
            candidates.append(candidate)
        return candidates

    def _time_expanded_swap_matching_candidates(self, base_mapping, tenant):
        scored_pairs = self._time_expanded_scored_swap_pairs(base_mapping, tenant)
        if not scored_pairs:
            return []

        candidates = []
        base_signature = self._mapping_signature(base_mapping)
        for batch_size in (2, 4):
            used_ranks = set()
            selected_pairs = []
            for _delta, pair in scored_pairs[: min(len(scored_pairs), 48)]:
                if pair[0] in used_ranks or pair[1] in used_ranks:
                    continue
                selected_pairs.append(pair)
                used_ranks.update(pair)
                if len(selected_pairs) >= batch_size:
                    break
            if not selected_pairs:
                continue
            candidate = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            for left_rank, right_rank in selected_pairs:
                candidate[tenant][left_rank], candidate[tenant][right_rank] = (
                    candidate[tenant][right_rank],
                    candidate[tenant][left_rank],
                )
            candidate = self._canonicalize_ring_mapping(candidate)
            if self._mapping_signature(candidate) == base_signature:
                continue
            candidates.append(candidate)
        return candidates

    def _time_expanded_leaf_boundary_blocks(self, mapping, tenant):
        ranks = list(self.rank_orders[tenant])
        blocks = []
        current = []
        current_leaf = None
        for rank in ranks:
            server = int(mapping[tenant][rank])
            leaf = self._server_leaf_sort_key(server)[0]
            if current and leaf != current_leaf:
                blocks.append(tuple(current))
                current = []
            current.append(int(rank))
            current_leaf = leaf
        if current:
            blocks.append(tuple(current))
        return [block for block in blocks if len(block) >= 2]

    def _time_expanded_tenant_price_cost(
        self,
        base_mapping,
        tenant,
        tenant_mapping,
        critical_tasks,
        task_pair_price=None,
    ):
        ranks = list(self.rank_orders[tenant])
        if task_pair_price is None:
            candidate_servers = tuple(int(tenant_mapping[rank]) for rank in ranks)
            task_pair_price = self._time_expanded_task_pair_price_lookup(
                base_mapping,
                tenant,
                candidate_servers,
            )
        total = 0.0
        for task_id, task_tuple in self.data["task_surrogate"][tenant]["task_info"].items():
            task_id = int(task_id)
            _task_id, src_rank, dst_rank, volume = task_tuple
            src_server = int(tenant_mapping[int(src_rank)])
            dst_server = int(tenant_mapping[int(dst_rank)])
            if src_server == dst_server:
                continue
            critical_weight = 1.0 + (
                self.critical_path_price_beta
                if (int(tenant), task_id) in critical_tasks
                else 0.0
            )
            total += critical_weight * float(volume) * task_pair_price[(task_id, src_server, dst_server)]
        return float(total)

    def _time_expanded_boundary_remap_candidates(
        self,
        base_mapping,
        tenant,
        state,
        *,
        max_candidates=3,
        block_option_limit=4,
        beam_limit=8,
    ):
        blocks = self._time_expanded_leaf_boundary_blocks(base_mapping, tenant)
        if not blocks:
            return []

        critical_tasks = set(state.get("critical_tasks", set()))
        base_tenant_mapping = {
            int(rank): int(server)
            for rank, server in base_mapping[tenant].items()
        }
        task_active_slots = state.get("task_active_slots", {})
        slot_prices = state.get("slot_prices", [])
        path_edges = self.data["path_edges"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        task_pair_price_cache: dict[tuple[int, int, int], float] = {}

        def task_pair_price(task_id, src_server, dst_server):
            src_server = int(src_server)
            dst_server = int(dst_server)
            if src_server == dst_server:
                return 0.0
            cache_key = (int(task_id), src_server, dst_server)
            cached = task_pair_price_cache.get(cache_key)
            if cached is not None:
                return cached
            active_slots = sorted(
                int(slot)
                for slot in task_active_slots.get((int(tenant), int(task_id)), [])
            )
            if not active_slots:
                active_slots = [0]
            price_sum = 0.0
            for slot_idx in active_slots:
                price_state = slot_prices[slot_idx] if 0 <= slot_idx < len(slot_prices) else {}
                sender_prices = price_state.get("sender", {})
                receiver_prices = price_state.get("receiver", {})
                edge_prices = price_state.get("edge", {})
                price_sum += sender_prices.get(
                    src_server,
                    self._resource_price_value(server_send_capacity[src_server], 0.0),
                )
                price_sum += receiver_prices.get(
                    dst_server,
                    self._resource_price_value(server_recv_capacity[dst_server], 0.0),
                )
                for edge in self._path_edges_for_pair(path_edges, tenant, src_server, dst_server):
                    price_sum += edge_prices.get(edge, self._edge_price_value(edge, 0.0))
            value = float(price_sum / max(len(active_slots), 1))
            task_pair_price_cache[cache_key] = value
            return value
        task_records = []
        rank_incidence: dict[int, set[int]] = {
            int(rank): set()
            for rank in self.rank_orders[tenant]
        }
        for task_id, task_tuple in self.data["task_surrogate"][tenant]["task_info"].items():
            task_id = int(task_id)
            _task_id, src_rank, dst_rank, volume = task_tuple
            task_idx = len(task_records)
            critical_weight = 1.0 + (
                self.critical_path_price_beta
                if (int(tenant), task_id) in critical_tasks
                else 0.0
            )
            task_records.append((
                task_id,
                int(src_rank),
                int(dst_rank),
                float(volume) * critical_weight,
            ))
            rank_incidence.setdefault(int(src_rank), set()).add(task_idx)
            rank_incidence.setdefault(int(dst_rank), set()).add(task_idx)

        def task_cost(record, tenant_mapping):
            task_id, src_rank, dst_rank, weighted_volume = record
            src_server = int(tenant_mapping[int(src_rank)])
            dst_server = int(tenant_mapping[int(dst_rank)])
            if src_server == dst_server:
                return 0.0
            return float(weighted_volume) * task_pair_price(task_id, src_server, dst_server)

        def affected_task_indices(touched_ranks):
            affected = set()
            for rank in touched_ranks:
                affected.update(rank_incidence.get(int(rank), set()))
            return affected

        def remap_delta(old_mapping, new_mapping, touched_ranks):
            delta = 0.0
            for task_idx in affected_task_indices(touched_ranks):
                record = task_records[task_idx]
                delta += task_cost(record, new_mapping) - task_cost(record, old_mapping)
            return float(delta)

        base_cost = sum(task_cost(record, base_tenant_mapping) for record in task_records)

        def block_options(block):
            block = tuple(int(rank) for rank in block)
            if len(block) < 2:
                return []
            first_rank = block[0]
            last_rank = block[-1]
            block_servers = [int(base_tenant_mapping[rank]) for rank in block]
            options = []
            seen = set()
            for first_server in block_servers:
                for last_server in block_servers:
                    if len(block) > 1 and int(first_server) == int(last_server):
                        continue
                    remaining_servers = list(block_servers)
                    if int(first_server) not in remaining_servers or int(last_server) not in remaining_servers:
                        continue
                    remaining_servers.remove(int(first_server))
                    remaining_servers.remove(int(last_server))
                    reassignment = dict(base_tenant_mapping)
                    reassignment[first_rank] = int(first_server)
                    reassignment[last_rank] = int(last_server)
                    internal_ranks = list(block[1:-1])
                    for rank, server in zip(internal_ranks, remaining_servers):
                        reassignment[int(rank)] = int(server)
                    signature = tuple(reassignment[rank] for rank in block)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    cost = base_cost + remap_delta(base_tenant_mapping, reassignment, block)
                    options.append((float(cost), reassignment))
            options.sort(key=lambda item: item[0])
            return options[:block_option_limit]

        tenant_beam = [(float(base_cost), base_tenant_mapping)]
        for block in blocks:
            options = block_options(block)
            if not options:
                continue
            next_beam = []
            for _current_cost, current_mapping in tenant_beam:
                for _option_cost, option_mapping in options:
                    merged = dict(current_mapping)
                    for rank in block:
                        merged[int(rank)] = int(option_mapping[int(rank)])
                    cost = float(_current_cost) + remap_delta(current_mapping, merged, block)
                    next_beam.append((float(cost), merged))
            next_beam.sort(key=lambda item: item[0])
            deduped = []
            seen = set()
            for cost, mapping in next_beam:
                signature = tuple(mapping[rank] for rank in self.rank_orders[tenant])
                if signature in seen:
                    continue
                seen.add(signature)
                deduped.append((cost, mapping))
                if len(deduped) >= beam_limit:
                    break
            tenant_beam = deduped

        base_signature = self._mapping_signature(base_mapping)
        candidates = []
        for _cost, tenant_mapping in tenant_beam[:max_candidates]:
            candidate = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            candidate[tenant] = {
                int(rank): int(tenant_mapping[int(rank)])
                for rank in self.rank_orders[tenant]
            }
            candidate = self._canonicalize_ring_mapping(candidate)
            if self._mapping_signature(candidate) == base_signature:
                continue
            candidates.append(candidate)
        return candidates

    def _time_expanded_miqp_pricing_candidate(self, base_mapping, tenant, state, deadline):
        if self.miqp_pricing_time_limit <= 0.0 or time.time() >= deadline:
            return None
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:
            return None

        ranks = list(self.rank_orders[tenant])
        servers = list(self.server_sets[tenant])
        if len(ranks) <= 1 or len(ranks) != len(servers):
            return None

        critical_tasks = set(state.get("critical_tasks", set()))
        task_active_slots = state.get("task_active_slots", {})
        slot_prices = state.get("slot_prices", [])
        path_edges = self.data["path_edges"]
        server_send_capacity = self.data["server_send_capacity"]
        server_recv_capacity = self.data["server_recv_capacity"]
        task_pair_price_cache: dict[tuple[int, int, int], float] = {}

        def task_pair_price(task_id, src_server, dst_server):
            src_server = int(src_server)
            dst_server = int(dst_server)
            if src_server == dst_server:
                return 0.0
            cache_key = (int(task_id), src_server, dst_server)
            cached = task_pair_price_cache.get(cache_key)
            if cached is not None:
                return cached
            active_slots = sorted(
                int(slot)
                for slot in task_active_slots.get((int(tenant), int(task_id)), [])
            )
            if not active_slots:
                active_slots = [0]
            price_sum = 0.0
            for slot_idx in active_slots:
                price_state = slot_prices[slot_idx] if 0 <= slot_idx < len(slot_prices) else {}
                sender_prices = price_state.get("sender", {})
                receiver_prices = price_state.get("receiver", {})
                edge_prices = price_state.get("edge", {})
                price_sum += sender_prices.get(
                    src_server,
                    self._resource_price_value(server_send_capacity[src_server], 0.0),
                )
                price_sum += receiver_prices.get(
                    dst_server,
                    self._resource_price_value(server_recv_capacity[dst_server], 0.0),
                )
                for edge in self._path_edges_for_pair(path_edges, tenant, src_server, dst_server):
                    price_sum += edge_prices.get(edge, self._edge_price_value(edge, 0.0))
            value = float(price_sum / max(len(active_slots), 1))
            task_pair_price_cache[cache_key] = value
            return value

        rank_idx = {int(rank): idx for idx, rank in enumerate(ranks)}
        server_idx = {int(server): idx for idx, server in enumerate(servers)}
        coefficients: dict[tuple[int, int, int, int], float] = {}
        for task_id, task_tuple in self.data["task_surrogate"][tenant]["task_info"].items():
            task_id = int(task_id)
            _task_id, src_rank, dst_rank, volume = task_tuple
            src_rank = int(src_rank)
            dst_rank = int(dst_rank)
            critical_weight = 1.0 + (
                self.critical_path_price_beta
                if (int(tenant), task_id) in critical_tasks
                else 0.0
            )
            weighted_volume = float(volume) * critical_weight
            for src_server in servers:
                for dst_server in servers:
                    if int(src_server) == int(dst_server):
                        continue
                    value = weighted_volume * task_pair_price(task_id, src_server, dst_server)
                    if value == 0.0:
                        continue
                    key = (src_rank, dst_rank, int(src_server), int(dst_server))
                    coefficients[key] = float(coefficients.get(key, 0.0) + value)

        try:
            model = gp.Model()
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = max(
                0.1,
                min(self.miqp_pricing_time_limit, max(0.1, deadline - time.time())),
            )
            model.Params.MIPFocus = 1
            y = model.addVars(len(ranks), len(servers), vtype=GRB.BINARY, name="y")
            for rank_pos in range(len(ranks)):
                model.addConstr(gp.quicksum(y[rank_pos, server_pos] for server_pos in range(len(servers))) == 1)
            for server_pos in range(len(servers)):
                model.addConstr(gp.quicksum(y[rank_pos, server_pos] for rank_pos in range(len(ranks))) == 1)

            objective = gp.QuadExpr()
            for (src_rank, dst_rank, src_server, dst_server), coeff in coefficients.items():
                objective.add(
                    coeff
                    * y[rank_idx[src_rank], server_idx[src_server]]
                    * y[rank_idx[dst_rank], server_idx[dst_server]]
                )
            model.setObjective(objective, GRB.MINIMIZE)
            for rank in ranks:
                current_server = int(base_mapping[tenant][rank])
                if current_server in server_idx:
                    y[rank_idx[int(rank)], server_idx[current_server]].Start = 1.0
            model.optimize()
        except Exception:
            return None

        if getattr(model, "SolCount", 0) <= 0:
            return None

        candidate = {
            current_tenant: dict(rank_to_server)
            for current_tenant, rank_to_server in base_mapping.items()
        }
        for rank in ranks:
            rank_pos = rank_idx[int(rank)]
            assigned_server = max(
                servers,
                key=lambda server: float(y[rank_pos, server_idx[int(server)]].X),
            )
            candidate[tenant][int(rank)] = int(assigned_server)
        candidate = self._canonicalize_ring_mapping(candidate)
        if self._mapping_signature(candidate) == self._mapping_signature(base_mapping):
            return None
        return candidate

    def _time_expanded_batch_candidates(self, base_mapping, ranked_by_tenant):
        active_tenants = [
            tenant
            for tenant, tenant_candidates in ranked_by_tenant
            if tenant_candidates
        ][: min(3, len(ranked_by_tenant))]
        if len(active_tenants) < 2:
            return []

        pool_by_tenant = {
            tenant: candidates[: min(4, len(candidates))]
            for tenant, candidates in ranked_by_tenant
            if tenant in set(active_tenants)
        }
        base_signature = self._mapping_signature(base_mapping)
        batch_candidates = []
        seen = set()
        for combination in itertools.product(*(pool_by_tenant[tenant] for tenant in active_tenants)):
            candidate = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            for tenant, tenant_candidate in zip(active_tenants, combination):
                candidate[tenant] = dict(tenant_candidate[tenant])
            candidate = self._canonicalize_ring_mapping(candidate)
            signature = self._mapping_signature(candidate)
            if signature == base_signature or signature in seen:
                continue
            seen.add(signature)
            batch_candidates.append(candidate)
        return batch_candidates

    def _time_expanded_graph_candidates(self, base_mapping, base_score, deadline):
        if time.time() >= deadline:
            return []

        state = self._get_time_expanded_surrogate_state(base_mapping)
        self.tenant_pressure = dict(state.get("tenant_pressure", {}))
        self.tenant_peak_load = dict(state.get("tenant_peak_load", {}))
        self.rank_pressure = dict(state.get("rank_pressure", {}))
        self.tenant_pair_interaction = dict(state.get("tenant_pair_interaction", {}))

        tenant_order = sorted(
            self.tenants,
            key=lambda tenant: (
                -float(self.tenant_pressure.get(tenant, 0.0)),
                -float(self.tenant_peak_load.get(tenant, 0.0)),
                int(tenant),
            ),
        )

        scored_candidates: list[tuple[tuple[float, float], dict[int, dict[int, int]], str | None]] = []
        seen = {self._mapping_signature(base_mapping)}
        ranked_by_tenant = []

        def evaluate_candidate(source, candidate):
            if time.time() >= deadline:
                return None
            candidate = self._canonicalize_ring_mapping(candidate)
            signature = self._mapping_signature(candidate)
            if signature in seen:
                return None
            seen.add(signature)
            score = self._evaluate_surrogate_mapping(candidate)
            scored_candidates.append((score, candidate, source))
            return score, candidate

        active_tenant_count = min(len(tenant_order), max(3, (len(tenant_order) + 1) // 2))
        for tenant in tenant_order[:active_tenant_count]:
            if time.time() >= deadline:
                break
            tenant_scored = []
            raw_candidates = self._time_expanded_boundary_remap_candidates(
                base_mapping,
                tenant,
                state,
            ) + self._time_expanded_order_candidates(base_mapping, tenant)
            miqp_candidate = self._time_expanded_miqp_pricing_candidate(
                base_mapping,
                tenant,
                state,
                deadline,
            )
            if miqp_candidate is not None:
                raw_candidates.insert(0, miqp_candidate)
            for candidate in raw_candidates:
                result = evaluate_candidate("time_expanded_graph", candidate)
                if result is None:
                    continue
                tenant_scored.append(result)
            ranked_by_tenant.append((tenant, raw_candidates[: min(4, len(raw_candidates))]))

        for candidate in self._time_expanded_batch_candidates(base_mapping, ranked_by_tenant):
            if time.time() >= deadline:
                break
            evaluate_candidate("time_expanded_batch", candidate)

        scored_candidates.sort(key=lambda entry: self._objective_sort_key(entry[0]))
        return scored_candidates[: self._time_expanded_direct_candidate_limit()]

    def _solve_time_expanded_graph_search(self, time_limit=None):
        start_time = time.time()
        deadline = float("inf") if time_limit is None else start_time + float(time_limit)

        initial_candidates = []
        best_mapping = None
        best_score = (float("inf"), float("inf"))
        for seed_mapping in self._time_expanded_graph_seed_mappings():
            if time.time() >= deadline:
                break
            seed_score = self._evaluate_surrogate_mapping(seed_mapping)
            initial_candidates.append((seed_score, seed_mapping, "seed"))
            if self._is_better_objective(seed_score, best_score):
                best_mapping = seed_mapping
                best_score = seed_score

        if best_mapping is None:
            best_mapping = {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in self.initial_tenant_mapping.items()
            }
            best_score = self._evaluate_surrogate_mapping(best_mapping)
            initial_candidates = [(best_score, best_mapping, "seed")]

        direct_beam_width = max(self.beam_width, 2)
        original_beam_width = self.beam_width
        self.beam_width = direct_beam_width
        try:
            beam = self._dedupe_ranked_candidates(initial_candidates)
        finally:
            self.beam_width = original_beam_width
        if not beam:
            beam = [(best_score, best_mapping, "seed")]

        stagnation_rounds = 0
        max_rounds = 2
        for _round_idx in range(max_rounds):
            if time.time() >= deadline:
                break
            next_candidates = list(beam)
            for beam_score, beam_mapping, _source in beam:
                if time.time() >= deadline:
                    break
                next_candidates.extend(
                    self._time_expanded_graph_candidates(
                        beam_mapping,
                        beam_score,
                        deadline,
                    )
                )

            original_beam_width = self.beam_width
            self.beam_width = direct_beam_width
            try:
                ranked = self._dedupe_ranked_candidates(next_candidates)
            finally:
                self.beam_width = original_beam_width
            if not ranked:
                break

            top_score, top_mapping, top_source = ranked[0]
            improved = self._is_better_objective(top_score, best_score)
            if improved:
                best_score = top_score
                best_mapping = top_mapping
                self.last_move_source = top_source
                if top_source is not None:
                    self.move_source_counts[top_source] = self.move_source_counts.get(top_source, 0) + 1
                self._register_surrogate_candidate(best_mapping, best_score)
                stagnation_rounds = 0
            else:
                stagnation_rounds += 1

            previous_signatures = {self._mapping_signature(mapping) for _score, mapping, _source in beam}
            next_signatures = {self._mapping_signature(mapping) for _score, mapping, _source in ranked}
            beam = ranked
            if previous_signatures == next_signatures and stagnation_rounds >= 2:
                break

        self._register_surrogate_candidate(best_mapping, best_score)
        final_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in best_mapping.items()
        }
        final_score = best_score
        self.final_score_is_simulated = False
        self.final_mapping = final_mapping
        self.final_obj = float(final_score[0])
        self.final_makespan = float(final_score[0])
        self.final_avg_jct = float(final_score[1])
        self.runtime_seconds = time.time() - start_time

        if self.verbose:
            print(
                f"Time-expanded graph search complete: Makespan={final_score[0]:.12f}, "
                f"AvgJCT={final_score[1]:.12f}, Runtime={self.runtime_seconds:.2f}s"
            )
        return self

    def _cyclic_rank_order_candidate(
        self,
        base_mapping,
        base_score,
        tenant,
        epoch_prices,
        deadline,
    ):
        """Generate rank-order rotations using time-expanded resource prices."""
        ranks = list(self.rank_orders[tenant])
        if len(ranks) <= 2 or time.time() >= deadline:
            return None, base_mapping, base_score

        current_order = tuple(int(base_mapping[tenant][rank]) for rank in ranks)
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

        n = len(ranks)
        small_window = min(8, n - 1)
        shifts = set(range(0, small_window + 1))
        shifts.update(n - shift for shift in range(1, small_window + 1))
        shifts.update(
            max(1, round(idx * n / 8))
            for idx in range(1, 8)
            if 0 < round(idx * n / 8) < n
        )

        pair_epoch_price = self._pair_epoch_price_lookup(
            tenant,
            self.server_sets[tenant],
            epoch_prices,
        )
        price_ranked = []
        seen_signatures = set()
        for order in source_orders:
            for shift in sorted(shifts):
                if time.time() >= deadline:
                    break
                rotated_order = order[shift:] + order[:shift]
                if rotated_order == current_order:
                    continue
                candidate_mapping = self._apply_server_order(
                    base_mapping,
                    tenant,
                    rotated_order,
                )
                candidate_mapping = self._canonicalize_ring_mapping(candidate_mapping)
                signature = self._mapping_signature(candidate_mapping)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                price_cost = self._tenant_price_cost(
                    tenant,
                    candidate_mapping,
                    pair_epoch_price,
                )
                price_ranked.append((float(price_cost), candidate_mapping))

        if not price_ranked:
            return None, base_mapping, base_score

        price_ranked.sort(key=lambda item: item[0])
        best_mapping = base_mapping
        best_score = base_score
        for _price_cost, candidate_mapping in price_ranked[:4]:
            if time.time() >= deadline:
                break
            candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
            if self._is_better_objective(candidate_score, best_score):
                best_mapping = candidate_mapping
                best_score = candidate_score

        if self._mapping_signature(best_mapping) == self._mapping_signature(base_mapping):
            return None, base_mapping, base_score
        return "time_expanded_cyclic", best_mapping, best_score

    def _generate_time_expanded_candidate_frontier(self, deadline):
        candidates: list[tuple[str, dict[int, dict[int, int]]]] = []
        seen = set()

        def add_candidate(source, mapping):
            signature = self._mapping_signature(mapping)
            if signature in seen:
                return
            seen.add(signature)
            candidates.append((
                source,
                {
                    tenant: dict(rank_to_server)
                    for tenant, rank_to_server in mapping.items()
                },
            ))

        seed_frontier: list[tuple[tuple[float, float], dict[int, dict[int, int]]]] = []
        for seed_mapping in self._seed_mappings():
            if time.time() >= deadline:
                break
            add_candidate("seed", seed_mapping)
            seed_score = self._evaluate_surrogate_mapping(seed_mapping)
            seed_frontier.append((seed_score, seed_mapping))

        seed_frontier.sort(key=lambda entry: self._objective_sort_key(entry[0]))
        for start_score, start_mapping in seed_frontier[: self.beam_width]:
            if time.time() >= deadline:
                break
            beam_score = start_score
            beam_mapping = {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in start_mapping.items()
            }

            frontier_rounds = max(2, min(self.max_price_rounds, len(self.tenants) // 2))
            for _round_idx in range(frontier_rounds):
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

                best_source = None
                best_mapping = beam_mapping
                best_score = beam_score
                cyclic_tenant_count = max(1, min(2, len(tenant_order)))
                for tenant_idx, tenant in enumerate(tenant_order):
                    if time.time() >= deadline:
                        break
                    if tenant_idx < cyclic_tenant_count:
                        cyclic_source, cyclic_mapping, cyclic_score = self._cyclic_rank_order_candidate(
                            beam_mapping,
                            beam_score,
                            tenant,
                            epoch_prices,
                            deadline,
                        )
                        if cyclic_source is not None:
                            add_candidate(cyclic_source, cyclic_mapping)
                            if self._is_better_objective(cyclic_score, best_score):
                                best_source = cyclic_source
                                best_mapping = cyclic_mapping
                                best_score = cyclic_score

                    local_mapping, _ = super()._optimize_tenant_block_with_prices(
                        beam_mapping,
                        tenant,
                        epoch_prices,
                        deadline,
                    )
                    if self._mapping_signature(local_mapping) == self._mapping_signature(beam_mapping):
                        continue
                    local_score = self._evaluate_surrogate_mapping(local_mapping)
                    add_candidate("time_expanded_local", local_mapping)
                    if self._is_better_objective(local_score, best_score):
                        best_source = "time_expanded_local"
                        best_mapping = local_mapping
                        best_score = local_score

                if best_source is None:
                    break

                beam_mapping = {
                    tenant: dict(rank_to_server)
                    for tenant, rank_to_server in best_mapping.items()
                }
                beam_score = best_score
                add_candidate(best_source, beam_mapping)

        return candidates

    def _effective_bnb_tenant_count(self, tenant_order):
        if self.surrogate_mode != "time_expanded":
            return min(len(tenant_order), max(0, self.max_bnb_tenants_per_round))
        return min(
            len(tenant_order),
            max(
                self.max_bnb_tenants_per_round,
                3,
                (len(tenant_order) + 1) // 2,
            ),
        )

    def _effective_joint_tenants(self, tenant_order):
        if self.surrogate_mode != "time_expanded":
            return tenant_order[: max(2, self.max_bnb_tenants_per_round)]
        return tenant_order[: min(3, len(tenant_order))]

    def _time_expanded_rank_swap_candidates(
        self,
        base_mapping,
        tenant,
        epoch_prices,
        deadline,
        *,
        max_candidates=16,
    ):
        ranks = [
            int(rank)
            for rank in self.data["compiled_schedule"]["per_tenant"][tenant]["branch_order"]
            if int(rank) in self.rank_orders[tenant]
        ]
        if len(ranks) <= 1 or time.time() >= deadline:
            return []

        rank_pressure = getattr(self, "rank_pressure", {})
        branch_position = {int(rank): idx for idx, rank in enumerate(ranks)}
        hot_ranks = sorted(
            ranks,
            key=lambda rank: (
                -float(rank_pressure.get((int(tenant), int(rank)), 0.0)),
                branch_position[int(rank)],
                int(rank),
            ),
        )[: min(16, len(ranks))]
        partner_ranks = ranks[: min(48, len(ranks))]
        for rank in hot_ranks:
            if rank not in partner_ranks:
                partner_ranks.append(rank)

        pair_epoch_price = self._pair_epoch_price_lookup(
            tenant,
            self.server_sets[tenant],
            epoch_prices,
        )
        scored_pairs = []
        seen_pairs = set()
        for left_rank in hot_ranks:
            if time.time() >= deadline:
                break
            for right_rank in partner_ranks:
                if int(left_rank) == int(right_rank):
                    continue
                pair = tuple(sorted((int(left_rank), int(right_rank))))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                delta = self._rank_swap_price_delta(
                    tenant,
                    base_mapping,
                    pair_epoch_price,
                    pair[0],
                    pair[1],
                )
                if delta < -1e-12:
                    scored_pairs.append((float(delta), pair))

        scored_pairs.sort(key=lambda item: item[0])
        candidates = []
        base_signature = self._mapping_signature(base_mapping)
        for _delta, (left_rank, right_rank) in scored_pairs[:max_candidates]:
            if time.time() >= deadline:
                break
            candidate_mapping = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            candidate_mapping[tenant][left_rank], candidate_mapping[tenant][right_rank] = (
                candidate_mapping[tenant][right_rank],
                candidate_mapping[tenant][left_rank],
            )
            candidate_mapping = self._canonicalize_ring_mapping(candidate_mapping)
            if self._mapping_signature(candidate_mapping) == base_signature:
                continue
            candidates.append(candidate_mapping)
        return candidates

    def _time_expanded_joint_bnb_candidates(
        self,
        base_mapping,
        tenant_order,
        deadline,
        *,
        tenants_per_joint=3,
        pool_per_tenant=3,
    ):
        joint_tenants = list(tenant_order[: min(tenants_per_joint, len(tenant_order))])
        if len(joint_tenants) < 2 or time.time() >= deadline:
            return []

        pools = []
        for tenant in joint_tenants:
            if time.time() >= deadline:
                break
            candidate_deadline = min(deadline, time.time() + self.bnb_candidate_time_limit)
            pool = []
            descent_candidate = self._time_expanded_price_descent_candidate(
                base_mapping,
                tenant,
                candidate_deadline,
            )
            if self._mapping_signature(descent_candidate) != self._mapping_signature(base_mapping):
                pool.append(descent_candidate)
            pool = self._build_time_expanded_bnb_candidate(
                base_mapping,
                tenant,
                candidate_deadline,
                return_pool=True,
                max_return=pool_per_tenant,
            )
            if self._mapping_signature(descent_candidate) != self._mapping_signature(base_mapping):
                pool = [descent_candidate] + pool
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
            joint_mapping = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            changed = False
            for (tenant, _pool), candidate_mapping in zip(pools, combination):
                if self._mapping_signature(candidate_mapping) == base_signature:
                    continue
                joint_mapping[tenant] = dict(candidate_mapping[tenant])
                changed = True
            if not changed:
                continue
            joint_mapping = self._canonicalize_ring_mapping(joint_mapping)
            signature = self._mapping_signature(joint_mapping)
            if signature in seen or signature == base_signature:
                continue
            seen.add(signature)
            candidates.append(joint_mapping)
        return candidates

    def _time_expanded_price_descent_candidate(self, base_mapping, tenant, deadline):
        ranks = [int(rank) for rank in self.rank_orders[tenant]]
        if len(ranks) <= 1 or time.time() >= deadline:
            return base_mapping

        task_pair_price = self._time_expanded_task_pair_price_lookup(
            base_mapping,
            tenant,
            tuple(int(base_mapping[tenant][rank]) for rank in ranks),
        )
        task_infos = []
        rank_incidence: dict[int, list[tuple[int, int, int, float]]] = {
            int(rank): [] for rank in ranks
        }
        for task_id, task_tuple in self.data["task_surrogate"][tenant]["task_info"].items():
            task_id = int(task_id)
            src_rank = int(task_tuple[1])
            dst_rank = int(task_tuple[2])
            volume = float(task_tuple[3])
            task_infos.append((task_id, src_rank, dst_rank, volume))
            rank_incidence.setdefault(src_rank, []).append((task_id, src_rank, dst_rank, volume))
            rank_incidence.setdefault(dst_rank, []).append((task_id, src_rank, dst_rank, volume))

        working_mapping = {
            current_tenant: dict(rank_to_server)
            for current_tenant, rank_to_server in base_mapping.items()
        }
        branch_order = [
            int(rank)
            for rank in self.data["compiled_schedule"]["per_tenant"][tenant]["branch_order"]
            if int(rank) in set(ranks)
        ] or list(ranks)
        rank_pressure = getattr(self, "rank_pressure", {})
        branch_position = {int(rank): idx for idx, rank in enumerate(branch_order)}
        ordered_ranks = sorted(
            branch_order,
            key=lambda rank: (
                -float(rank_pressure.get((int(tenant), int(rank)), 0.0)),
                branch_position[int(rank)],
                int(rank),
            ),
        )
        active_ranks = ordered_ranks[: min(len(ordered_ranks), 64)]

        def swap_delta(left_rank, right_rank):
            affected = {}
            for rank in (int(left_rank), int(right_rank)):
                for task in rank_incidence.get(rank, []):
                    affected[(task[0], task[1], task[2], task[3])] = task
            left_server = int(working_mapping[tenant][left_rank])
            right_server = int(working_mapping[tenant][right_rank])

            def swapped_server(rank):
                if int(rank) == int(left_rank):
                    return right_server
                if int(rank) == int(right_rank):
                    return left_server
                return int(working_mapping[tenant][rank])

            delta = 0.0
            for task_id, src_rank, dst_rank, volume in affected.values():
                old_src = int(working_mapping[tenant][src_rank])
                old_dst = int(working_mapping[tenant][dst_rank])
                new_src = swapped_server(src_rank)
                new_dst = swapped_server(dst_rank)
                old_cost = task_pair_price.get((task_id, old_src, old_dst), 0.0)
                new_cost = task_pair_price.get((task_id, new_src, new_dst), 0.0)
                delta += float(volume) * (new_cost - old_cost)
            return float(delta)

        max_passes = 6
        for _pass_idx in range(max_passes):
            if time.time() >= deadline:
                break
            best_pair = None
            best_delta = 0.0
            for left_idx, left_rank in enumerate(active_ranks):
                if time.time() >= deadline:
                    break
                for right_rank in active_ranks[left_idx + 1:]:
                    delta = swap_delta(left_rank, right_rank)
                    if delta < best_delta - 1e-12:
                        best_delta = float(delta)
                        best_pair = (int(left_rank), int(right_rank))
            if best_pair is None:
                break
            left_rank, right_rank = best_pair
            working_mapping[tenant][left_rank], working_mapping[tenant][right_rank] = (
                working_mapping[tenant][right_rank],
                working_mapping[tenant][left_rank],
            )

        return self._canonicalize_ring_mapping(working_mapping)

    def _time_expanded_rank_order_matching_candidate(
        self,
        base_mapping,
        base_score,
        tenant,
        deadline,
    ):
        ranks = [int(rank) for rank in self.rank_orders[tenant]]
        if len(ranks) <= 2 or time.time() >= deadline:
            return None, base_mapping, base_score

        current_servers = tuple(int(base_mapping[tenant][rank]) for rank in ranks)
        task_pair_price = self._time_expanded_task_pair_price_lookup(
            base_mapping,
            tenant,
            current_servers,
        )
        task_meta = self.data["task_surrogate"][tenant]
        rank_incidence: dict[int, list[tuple[int, int, int, float]]] = {
            int(rank): [] for rank in ranks
        }
        for task_id, task_tuple in task_meta["task_info"].items():
            task_id = int(task_id)
            src_rank = int(task_tuple[1])
            dst_rank = int(task_tuple[2])
            volume = float(task_tuple[3])
            rank_incidence.setdefault(src_rank, []).append((task_id, src_rank, dst_rank, volume))
            rank_incidence.setdefault(dst_rank, []).append((task_id, src_rank, dst_rank, volume))

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
                    candidates = [
                        task_pair_price[(task_id, int(server), int(dst_server))]
                        for dst_server in remaining_after
                        if int(dst_server) != int(server)
                    ]
                else:
                    candidates = [
                        task_pair_price[(task_id, int(src_server), int(server))]
                        for src_server in remaining_after
                        if int(src_server) != int(server)
                    ]
                if candidates:
                    optimistic += float(volume) * min(candidates)
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
            candidate_mapping = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            candidate_mapping[tenant] = {
                rank: int(assignment[int(rank)])
                for rank in ranks
            }
            candidate_mapping = self._canonicalize_ring_mapping(candidate_mapping)
            if self._mapping_signature(candidate_mapping) != self._mapping_signature(base_mapping):
                candidate_mappings.append(candidate_mapping)

        best_mapping = base_mapping
        best_score = base_score
        for candidate_mapping in candidate_mappings:
            if time.time() >= deadline:
                break
            candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
            if self._is_better_objective(candidate_score, best_score):
                best_mapping = candidate_mapping
                best_score = candidate_score
        if self._mapping_signature(best_mapping) == self._mapping_signature(base_mapping):
            return None, base_mapping, base_score
        return "bnb", best_mapping, best_score


    def _solve_neighborhood_search(self, time_limit=None):
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

                if self.surrogate_mode == "time_expanded":
                    cyclic_tenant_count = min(2, len(tenant_order))
                    for tenant in tenant_order[:cyclic_tenant_count]:
                        if time.time() >= deadline:
                            break
                        source, candidate_mapping, candidate_score = self._cyclic_rank_order_candidate(
                            beam_mapping,
                            beam_score,
                            tenant,
                            epoch_prices,
                            deadline,
                        )
                        if source is None:
                            continue
                        next_candidates.append((candidate_score, candidate_mapping, source))

                    for tenant in tenant_order:
                        if time.time() >= deadline:
                            break
                        source, candidate_mapping, candidate_score = self._time_expanded_rank_order_matching_candidate(
                            beam_mapping,
                            beam_score,
                            tenant,
                            deadline,
                        )
                        if source is None:
                            continue
                        next_candidates.append((candidate_score, candidate_mapping, source))

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

                if self.surrogate_mode == "time_expanded" and time.time() < deadline:
                    _, _, epoch_prices = self._compute_epoch_link_prices(beam_mapping)
                    for tenant in tenant_order:
                        if time.time() >= deadline:
                            break
                        for candidate_mapping in self._time_expanded_rank_swap_candidates(
                            beam_mapping,
                            tenant,
                            epoch_prices,
                            deadline,
                        ):
                            if time.time() >= deadline:
                                break
                            candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
                            if self._is_better_objective(candidate_score, beam_score):
                                next_candidates.append((candidate_score, candidate_mapping, "local"))

                if self.max_bnb_tenants_per_round > 0 and time.time() < deadline:
                    _, _, epoch_prices = self._compute_epoch_link_prices(beam_mapping)
                    for tenant in tenant_order[: self._effective_bnb_tenant_count(tenant_order)]:
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

                if self.surrogate_mode == "time_expanded" and len(tenant_order) >= 2 and time.time() < deadline:
                    for candidate_mapping in self._time_expanded_joint_bnb_candidates(
                        beam_mapping,
                        tenant_order,
                        deadline,
                    ):
                        if time.time() >= deadline:
                            break
                        candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
                        if self._is_better_objective(candidate_score, beam_score):
                            next_candidates.append((candidate_score, candidate_mapping, "joint"))

                if len(tenant_order) >= 2 and time.time() < deadline:
                    _, _, epoch_prices = self._compute_epoch_link_prices(beam_mapping)
                    joint_tenants = self._effective_joint_tenants(tenant_order)
                    for idx, tenant_a in enumerate(joint_tenants):
                        if time.time() >= deadline:
                            break
                        for tenant_b in joint_tenants[idx + 1 :]:
                            if time.time() >= deadline:
                                break
                            joint_candidates = self._joint_pair_candidate(
                                beam_mapping,
                                tenant_a,
                                tenant_b,
                                epoch_prices,
                                deadline,
                            )
                            if not joint_candidates:
                                continue
                            for candidate_mapping in joint_candidates:
                                if time.time() >= deadline:
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
                break

        if time.time() < deadline and self.surrogate_mode != "time_expanded":
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
        if self.validate_with_simulator and self.surrogate_mode != "time_expanded":
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

    def solve(self, time_limit=None):
        if self.surrogate_mode == "time_expanded":
            return self._solve_time_expanded_graph_search(time_limit=time_limit)
        return self._solve_neighborhood_search(time_limit=time_limit)


MappingHybridHeuristicSolver = MappingMultiNeighborhoodHeuristicSolver
MappingPortfolioHeuristicSolver = MappingMultiNeighborhoodHeuristicSolver
