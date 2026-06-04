from __future__ import annotations

from collections import defaultdict
import itertools
import math

import gurobipy as gp
from gurobipy import GRB

from multitenant.collectives import has_collective_workload, normalize_collective_programs
from multitenant.simulator.adapter import simulate_collective_details
from multitenant.workloads import build_collective_program_schedule


class MappingILPSolver:
    """Task-DAG mapping ILP with time slots, directed links, and bottleneck max-min sharing."""

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
        build_model=True,
        compact_task_windows=False,
        compact_window_mapping=None,
        lp_method=1,
        node_method=1,
        presolve=None,
        prepasses=None,
        numeric_focus=None,
        enable_heuristic_warm_start=False,
        enable_full_mip_start=False,
        enable_fixed_mapping_subproblem_start=False,
        path_table=None,
    ):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.collective = collective
        self.single_flow_size = single_flow_size
        self.tenant_collective_specs = tenant_collective_specs
        self.tenant_start_times = {
            int(tenant): float(start_time)
            for tenant, start_time in (tenant_start_times or {}).items()
        }
        self.tenant_collective_programs = normalize_collective_programs(
            tenant_mapping,
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
        self.compact_task_windows = bool(compact_task_windows)
        self.compact_window_mapping = compact_window_mapping
        self.lp_method = lp_method
        self.node_method = node_method
        self.presolve = presolve
        self.prepasses = prepasses
        self.numeric_focus = numeric_focus
        self.volume_scale = (
            1e3
            if has_collective_workload(
                collective=collective,
                single_flow_size=single_flow_size,
                tenant_collective_specs=tenant_collective_specs,
                tenant_collective_programs=self.tenant_collective_programs,
            )
            else 1e9
        )

        # Kept for API compatibility with earlier solver variants.
        self.fairness_lambda = float(fairness_lambda)
        self.fairness_iterations = max(1, int(fairness_iterations))
        self.fairness_grouping = fairness_grouping
        # Deprecated compatibility flag. External search-based warm starts are
        # intentionally kept out of this ILP module.
        self.enable_external_warm_start = bool(enable_heuristic_warm_start)
        self.enable_full_mip_start = bool(enable_full_mip_start)
        self.enable_fixed_mapping_subproblem_start = bool(enable_fixed_mapping_subproblem_start)
        self.path_table = path_table or self.datacenter.build_tenant_ecmp_path_table(
            sorted(int(tenant) for tenant in tenant_mapping)
        )
        self.path_ordered_edges = self.datacenter.build_edge_table_from_paths(self.path_table)
        self.path_edge_set = {
            key: set(edges)
            for key, edges in self.path_ordered_edges.items()
        }

        self.data = None
        self.model = None

        self.X = {}
        self.Y = {}
        self.ring_patterns = {}
        self.U = {}
        self.W = {}
        self.LinkMinRate = {}
        self.R = {}
        self.Z = {}
        self.S_active = {}
        self.P_active = {}
        self.D_full = {}
        self.Q_bottleneck = {}
        self.F = {}
        self.C = {}
        self.task_finish = {}
        self.tenant_finish = {}
        self.T_max = None
        self.final_obj = None
        self.final_makespan = None
        self.final_avg_jct = None
        self.final_mapping = None
        self.warm_start_mapping = None
        self.warm_start_schedule_horizon_bound = None
        self.warm_start_runtime = 0.0
        if build_model:
            self._build(name=name)
        else:
            self.data = self._build_data()

    def _derive_tasks(self):
        if self.tenant_collective_programs is not None:
            schedule = build_collective_program_schedule(
                self.tenant_mapping,
                self.tenant_collective_programs,
                scale=self.volume_scale,
            )
            return (
                {
                    tenant: list(tenant_schedule.get("tasks", []))
                    for tenant, tenant_schedule in schedule.items()
                },
                {
                    tenant: {
                        "task_order": list(tenant_schedule.get("task_order", [])),
                        "sender_order": {
                            int(sender): list(task_ids)
                            for sender, task_ids in tenant_schedule.get("sender_order", {}).items()
                        },
                        "collective_edges": list(tenant_schedule.get("collective_edges", [])),
                        "sender_order_edges": list(tenant_schedule.get("sender_order_edges", [])),
                        "collective_program": list(tenant_schedule.get("collective_program", [])),
                    }
                    for tenant, tenant_schedule in schedule.items()
                },
            )

        stage_flows = self.stage_flows
        if stage_flows is None:
            if self.tenant_flows is None:
                raise ValueError(
                    "tenant_flows is required when collective/stage_flows are not provided"
                )
            stage_flows = {tenant: [self.tenant_flows[tenant]] for tenant in self.tenant_mapping}
        else:
            stage_flows = {
                tenant: [
                    [tuple(flow) for flow in stage]
                    for stage in tenant_stages
                ]
                for tenant, tenant_stages in stage_flows.items()
            }

        tasks_by_tenant = {}
        schedule_by_tenant = {}
        for tenant, tenant_stages in stage_flows.items():
            tasks = []
            next_task_id = 0
            previous_stage_task_ids = []
            for stage_idx, stage in enumerate(tenant_stages):
                current_stage_task_ids = []
                for flow_idx, (src, dst, volume) in enumerate(stage):
                    task_id = next_task_id
                    next_task_id += 1
                    tasks.append(
                        {
                            "task_id": task_id,
                            "phase": f"stage_{stage_idx}",
                            "chunk": flow_idx,
                            "step": stage_idx,
                            "u": int(src),
                            "v": int(dst),
                            "V": float(volume),
                            "preds": list(previous_stage_task_ids),
                            "src_rank": int(src),
                            "dst_rank": int(dst),
                        }
                    )
                    current_stage_task_ids.append(task_id)
                previous_stage_task_ids = current_stage_task_ids
            tasks_by_tenant[tenant] = tasks
            schedule_by_tenant[tenant] = {
                "task_order": [task["task_id"] for task in tasks],
                "sender_order": {
                    rank: [task["task_id"] for task in tasks if task["src_rank"] == rank]
                    for rank in sorted(self.tenant_mapping[tenant].keys())
                },
                "collective_edges": [
                    {
                        "src_task_id": pred_task_id,
                        "dst_task_id": task["task_id"],
                        "type": "collective",
                    }
                    for task in tasks
                    for pred_task_id in task["preds"]
                ],
                "sender_order_edges": [],
            }
        return tasks_by_tenant, schedule_by_tenant

    @staticmethod
    def _longest_task_chain(tasks):
        preds_by_task = {task["task_id"]: list(task["preds"]) for task in tasks}
        memo = {}

        def depth(task_id):
            if task_id in memo:
                return memo[task_id]
            preds = preds_by_task.get(task_id, [])
            if not preds:
                memo[task_id] = 1
            else:
                memo[task_id] = 1 + max(depth(pred) for pred in preds)
            return memo[task_id]

        return max((depth(task["task_id"]) for task in tasks), default=0)

    @staticmethod
    def _longest_weighted_task_chain(tasks, task_weights):
        preds_by_task = {task["task_id"]: list(task["preds"]) for task in tasks}
        memo = {}

        def depth(task_id):
            if task_id in memo:
                return memo[task_id]
            preds = preds_by_task.get(task_id, [])
            own_weight = task_weights.get(task_id, 1)
            if not preds:
                memo[task_id] = own_weight
            else:
                memo[task_id] = own_weight + max(depth(pred) for pred in preds)
            return memo[task_id]

        return max((depth(task["task_id"]) for task in tasks), default=0)

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return None
        sorted_values = sorted(float(value) for value in values)
        if len(sorted_values) == 1:
            return sorted_values[0]
        position = (len(sorted_values) - 1) * float(percentile)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return sorted_values[lower]
        weight = position - lower
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    def _estimate_task_isolated_times(self, tasks_by_tenant, servers, capacities):
        isolated_times = []
        for tenant, tasks in tasks_by_tenant.items():
            candidate_servers = servers[tenant]
            for task in tasks:
                task_times = []
                volume = float(task["V"])
                for src_server in candidate_servers:
                    for dst_server in candidate_servers:
                        if src_server == dst_server:
                            continue
                        edges = self.path_edge_set.get((int(tenant), int(src_server), int(dst_server)))
                        if not edges:
                            continue
                        bottleneck = min((capacities[edge] for edge in edges if edge in capacities), default=None)
                        if bottleneck and bottleneck > 0.0:
                            task_times.append(volume / bottleneck)
                if task_times:
                    isolated_times.append(min(task_times))
        return isolated_times

    def _estimate_positive_gaps(self, schedule_by_tenant):
        gaps = []
        for tenant_schedule in schedule_by_tenant.values():
            for op in tenant_schedule.get("collective_program", []):
                gap = float(op.get("gap_after", 0.0))
                if gap > 0.0:
                    gaps.append(gap)
        return gaps

    def _estimate_horizon_time_budget(
        self,
        total_task_volume,
        min_capacity,
        dependency_lower_bound,
        isolated_task_times,
        positive_gaps,
    ):
        median_task_time = self._percentile(isolated_task_times, 0.5) or (total_task_volume / max(min_capacity, 1e-12))
        critical_path_time = float(dependency_lower_bound) * median_task_time
        work_time = total_task_volume / max(min_capacity, 1e-12)
        gap_time = sum(positive_gaps)
        return max(critical_path_time + gap_time, work_time + gap_time, median_task_time)

    def _build_data(self):
        tenants = sorted(self.tenant_mapping.keys())
        ranks = {tenant: sorted(self.tenant_mapping[tenant].keys()) for tenant in tenants}
        servers = {tenant: sorted(self.tenant_mapping[tenant].values()) for tenant in tenants}
        tasks_by_tenant, schedule_by_tenant = self._derive_tasks()

        links = list(self.datacenter.topology.edges())
        capacities = {
            (src, dst): float(self.datacenter.topology[src][dst].get("capacity", 0.0) / self.volume_scale)
            for (src, dst) in links
        }

        min_capacity = min(capacities.values()) if capacities else 1.0
        task_volume_upper = {}
        task_volume_representative = {}
        positive_volumes = []
        total_task_volume = 0.0
        for tenant in tenants:
            for task in tasks_by_tenant[tenant]:
                task_key = (tenant, task["task_id"])
                representative_volume = float(task["V"])
                upper_volume = float(task["V"])
                positive_volume = float(task["V"])
                task_volume_representative[task_key] = representative_volume
                task_volume_upper[task_key] = upper_volume
                total_task_volume += representative_volume
                if positive_volume > 0.0:
                    positive_volumes.append(positive_volume)
        min_task_volume = min(positive_volumes, default=1.0)
        total_task_count = sum(len(tasks_by_tenant[tenant]) for tenant in tenants)
        task_hop_lb = {}
        for tenant in tenants:
            candidate_servers = servers[tenant]
            for task in tasks_by_tenant[tenant]:
                max_hops = 1
                for src_server in candidate_servers:
                    for dst_server in candidate_servers:
                        if src_server == dst_server:
                            continue
                        edges = self.path_edge_set.get((int(tenant), int(src_server), int(dst_server)))
                        if edges:
                            max_hops = max(max_hops, len(edges))
                task_hop_lb[(tenant, task["task_id"])] = max_hops
        dependency_lower_bound = max(
            (
                max(
                    self._longest_weighted_task_chain(
                        tasks_by_tenant[tenant],
                        {task["task_id"]: task_hop_lb[(tenant, task["task_id"])] for task in tasks_by_tenant[tenant]},
                    ),
                    max(
                        (
                            sum(task_hop_lb[(tenant, task["task_id"])] for task in tasks_by_tenant[tenant] if task["src_rank"] == rank)
                            for rank in ranks[tenant]
                        ),
                        default=0,
                    ),
                )
                for tenant in tenants
            ),
            default=1,
        )
        target_horizon_slots = max(12, min(20, dependency_lower_bound + 4))
        isolated_task_times = self._estimate_task_isolated_times(
            tasks_by_tenant,
            servers,
            capacities,
        )
        positive_gaps = self._estimate_positive_gaps(schedule_by_tenant)

        if self.slot_duration_override is not None:
            slot_duration = float(self.slot_duration_override)
        else:
            p50_task_time = self._percentile(isolated_task_times, 0.5)
            p25_task_time = self._percentile(isolated_task_times, 0.25)
            flow_reference_time = p25_task_time or p50_task_time or (min_task_volume / min_capacity)
            accuracy_slot = flow_reference_time / 2.0
            if positive_gaps:
                accuracy_slot = min(accuracy_slot, min(positive_gaps) / 2.0)

            target_slot_budget = 48
            if total_task_count > 64:
                target_slot_budget = 40
            if total_task_count > 128:
                target_slot_budget = 32
            horizon_time_budget = self._estimate_horizon_time_budget(
                total_task_volume,
                min_capacity,
                dependency_lower_bound,
                isolated_task_times,
                positive_gaps,
            )
            budget_slot = horizon_time_budget / max(target_slot_budget, 1)
            slot_duration = max(accuracy_slot, budget_slot, 1e-6)
            # The task-time model assumes an active unfinished task has at least one
            # bottleneck link saturated in each active slot. If the slot is coarser
            # than the smallest task's isolated transmission time, a small task could
            # not possibly saturate any link while active, which makes otherwise legal
            # heterogeneous-payload instances infeasible. Cap the slot accordingly.
            smallest_task_time = min_task_volume / max(min_capacity, 1e-12)
            slot_duration = min(slot_duration, max(smallest_task_time, 1e-6))

        compact_op_earliest_slot = {}
        compact_op_latest_finish_slot = {}
        compact_op_duration_lb = {}
        if self.compact_task_windows:
            for tenant in tenants:
                program = sorted(
                    list(schedule_by_tenant.get(tenant, {}).get("collective_program", [])),
                    key=lambda item: int(item.get("op_idx", 0)),
                )
                if not program:
                    continue

                tasks_by_op = defaultdict(list)
                for task in tasks_by_tenant[tenant]:
                    tasks_by_op[int(task.get("op_idx", 0))].append(task)

                op_duration_lb = {}
                for op in program:
                    op_idx = int(op["op_idx"])
                    op_tasks = tasks_by_op.get(op_idx, [])
                    if not op_tasks:
                        op_duration_lb[op_idx] = 0
                        continue
                    chain_lb = self._longest_task_chain(op_tasks)
                    sender_lb = max(
                        (
                            sum(1 for task in op_tasks if int(task["src_rank"]) == rank)
                            for rank in ranks[tenant]
                        ),
                        default=0,
                    )
                    op_duration_lb[op_idx] = max(1, int(chain_lb), int(sender_lb))
                    compact_op_duration_lb[(tenant, op_idx)] = op_duration_lb[op_idx]

                elapsed = 0
                for op in program:
                    op_idx = int(op["op_idx"])
                    compact_op_earliest_slot[(tenant, op_idx)] = elapsed
                    gap_slots = int(math.ceil(float(op.get("gap_after", 0.0)) / slot_duration - 1e-9))
                    elapsed += op_duration_lb.get(op_idx, 0) + max(gap_slots, 0)

                suffix_after = 0
                for op in reversed(program):
                    op_idx = int(op["op_idx"])
                    gap_slots = int(math.ceil(float(op.get("gap_after", 0.0)) / slot_duration - 1e-9))
                    compact_op_latest_finish_slot[(tenant, op_idx)] = max(gap_slots, 0) + suffix_after
                    suffix_after += op_duration_lb.get(op_idx, 0) + max(gap_slots, 0)

        release_gates = {}
        total_gap_slots_by_tenant = {}
        task_earliest_slot = {}
        for tenant in tenants:
            release_gates[tenant] = {}
            total_gap_slots_by_tenant[tenant] = 0
            program = list(schedule_by_tenant.get(tenant, {}).get("collective_program", []))
            if self.compact_task_windows:
                for task in tasks_by_tenant[tenant]:
                    op_idx = int(task.get("op_idx", 0))
                    task_earliest_slot[(tenant, int(task["task_id"]))] = max(
                        0,
                        compact_op_earliest_slot.get((tenant, op_idx), 0),
                    )
            if not program:
                continue
            op_by_idx = {int(op["op_idx"]): op for op in program}
            for op in program:
                op_idx = int(op["op_idx"])
                if op_idx == 0:
                    continue
                previous_op = op_by_idx[op_idx - 1]
                gap_after_s = float(previous_op.get("gap_after", 0.0))
                gap_slots = int(math.ceil(gap_after_s / slot_duration - 1e-9))
                total_gap_slots_by_tenant[tenant] += gap_slots
                previous_task_ids = [int(task_id) for task_id in previous_op.get("task_ids", [])]
                for task_id in op.get("initial_task_ids", []):
                    release_gates[tenant][int(task_id)] = {
                        "previous_task_ids": previous_task_ids,
                        "gap_after_s": gap_after_s,
                        "gap_slots": gap_slots,
                    }

        if self.horizon_slots_override is not None:
            horizon_slots = int(self.horizon_slots_override)
        else:
            work_conserving_slots = int(math.ceil(total_task_volume / min_capacity / slot_duration))
            extra_dependency_slack = 2 if total_task_count > 64 else 4
            max_gap_slots = max(total_gap_slots_by_tenant.values(), default=0)
            horizon_slots = max(
                dependency_lower_bound + max_gap_slots + extra_dependency_slack,
                work_conserving_slots + max_gap_slots + 2,
                4,
            )
            horizon_slots = max(horizon_slots, 4)

        task_latest_finish_slot = {}
        if self.compact_task_windows:
            seed_op_latest_finish_slot = {}
            if self.compact_window_mapping:
                simulator_result = simulate_collective_details(
                    self.datacenter.topology,
                    self.compact_window_mapping,
                    self.path_table,
                    self.single_flow_size,
                    self.collective,
                    tenant_start_times=self.tenant_start_times,
                    tenant_collective_specs=self.tenant_collective_specs,
                    tenant_collective_programs=self.tenant_collective_programs,
                )
                task_by_name = {}
                for tenant in tenants:
                    for task in tasks_by_tenant[tenant]:
                        task_name = task.get("name")
                        if task_name is not None:
                            task_by_name[str(task_name)] = (tenant, int(task.get("op_idx", 0)))
                for tx_id, finish_time in simulator_result.get("tx_complete_time", {}).items():
                    flow_id = self._simulator_flow_id(tx_id).split("-Q", 1)[0]
                    if flow_id not in task_by_name:
                        continue
                    tenant, op_idx = task_by_name[flow_id]
                    finish_slot = int(math.ceil(float(finish_time) / slot_duration - 1e-9)) + 1
                    seed_op_latest_finish_slot[(tenant, op_idx)] = max(
                        seed_op_latest_finish_slot.get((tenant, op_idx), 0),
                        finish_slot,
                    )
                for tenant in tenants:
                    program = sorted(
                        list(schedule_by_tenant.get(tenant, {}).get("collective_program", [])),
                        key=lambda item: int(item.get("op_idx", 0)),
                    )
                    previous_latest = None
                    previous_gap_slots = 0
                    for op in program:
                        op_idx = int(op.get("op_idx", 0))
                        current_latest = seed_op_latest_finish_slot.get((tenant, op_idx))
                        if current_latest is None:
                            current_latest = (
                                compact_op_earliest_slot.get((tenant, op_idx), 0)
                                + compact_op_duration_lb.get((tenant, op_idx), 1)
                            )
                        if previous_latest is not None:
                            # The next collective is released only after the previous
                            # collective completes plus its fixed gap. Keep simulator
                            # seed windows closed under that discrete release rule.
                            current_latest = max(
                                current_latest,
                                previous_latest
                                + previous_gap_slots
                                + compact_op_duration_lb.get((tenant, op_idx), 1)
                                + 1,
                            )
                        seed_op_latest_finish_slot[(tenant, op_idx)] = current_latest
                        previous_latest = current_latest
                        previous_gap_slots = int(
                            math.ceil(float(op.get("gap_after", 0.0)) / slot_duration - 1e-9)
                        )
            for tenant in tenants:
                for task in tasks_by_tenant[tenant]:
                    task_id = int(task["task_id"])
                    op_idx = int(task.get("op_idx", 0))
                    latest_finish = horizon_slots - compact_op_latest_finish_slot.get((tenant, op_idx), 0)
                    if (tenant, op_idx) in seed_op_latest_finish_slot:
                        seed_latest_finish = max(
                            seed_op_latest_finish_slot[(tenant, op_idx)],
                            task_earliest_slot.get((tenant, task_id), 0)
                            + compact_op_duration_lb.get((tenant, op_idx), 1),
                        )
                        latest_finish = min(latest_finish, seed_latest_finish)
                    latest_finish = min(
                        horizon_slots,
                        max(task_earliest_slot.get((tenant, task_id), 0) + 1, latest_finish),
                    )
                    task_latest_finish_slot[(tenant, task_id)] = latest_finish

        task_total_volume = dict(task_volume_upper)
        min_send_unit = min_task_volume / max(horizon_slots * 1024.0, 1.0)
        sender_task_ids = {
            tenant: {
                rank: list(schedule_by_tenant.get(tenant, {}).get("sender_order", {}).get(rank, []))
                for rank in ranks[tenant]
            }
            for tenant in tenants
        }

        return {
            "M": tenants,
            "R": ranks,
            "S": servers,
            "tasks": tasks_by_tenant,
            "schedule": schedule_by_tenant,
            "L": links,
            "cap": capacities,
            "path_edges": self.path_edge_set,
            "path_ordered_edges": self.path_ordered_edges,
            "slot_duration": slot_duration,
            "Horizon": list(range(horizon_slots)),
            "num_slots": horizon_slots,
            "max_task_count": max((len(tasks_by_tenant[tenant]) for tenant in tenants), default=0),
            "task_total_volume": task_total_volume,
            "task_volume_representative": task_volume_representative,
            "min_send_unit": min_send_unit,
            "sender_tasks": sender_task_ids,
            "release_gates": release_gates,
            "task_earliest_slot": task_earliest_slot,
            "task_latest_finish_slot": task_latest_finish_slot,
        }

    def _build(self, name="mapping_ilp"):
        if self.model is not None:
            self.model.dispose()

        self.X = {}
        self.Y = {}
        self.ring_patterns = {}
        self.U = {}
        self.W = {}
        self.LinkMinRate = {}
        self.R = {}
        self.Z = {}
        self.S_active = {}
        self.P_active = {}
        self.D_full = {}
        self.Q_bottleneck = {}
        self.F = {}
        self.C = {}
        self.task_finish = {}
        self.tenant_finish = {}
        self.T_max = None
        self.warm_start_schedule_horizon_bound = None

        self.data = self._build_data()

        self.model = gp.Model(name)
        self.model.Params.OutputFlag = 1 if self.verbose else 0

        self._add_X()
        self._add_perm_constraints()
        self._add_U_and_endpoint_constraints()
        self._add_task_time_model()
        self._set_lexicographic_objective()
        self.model.update()

    def _add_X(self):
        tenants, ranks, servers = self.data["M"], self.data["R"], self.data["S"]
        for tenant in tenants:
            use_patterns = self._use_ring_pattern_formulation(tenant)
            for rank in ranks[tenant]:
                for server in servers[tenant]:
                    if use_patterns:
                        self.X[(tenant, rank, server)] = self.model.addVar(
                            vtype=GRB.CONTINUOUS,
                            lb=0.0,
                            ub=1.0,
                            name=f"X_{tenant}_{rank}_{server}",
                        )
                    else:
                        self.X[(tenant, rank, server)] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"X_{tenant}_{rank}_{server}",
                        )

    def _use_ring_pattern_formulation(self, tenant):
        return self.compact_task_windows and self._tenant_has_ring_rotation_symmetry(tenant)

    def _canonical_ring_patterns(self, tenant):
        tenant_ranks = list(self.data["R"][tenant])
        tenant_servers = sorted(self.data["S"][tenant])
        if not tenant_ranks or not tenant_servers:
            return []

        anchor_server = min(tenant_servers)
        other_servers = [server for server in tenant_servers if server != anchor_server]
        if len(tenant_ranks) <= 2:
            sequences = [(anchor_server, *other_servers)]
        else:
            sequences = []
            for tail in itertools.permutations(other_servers):
                sequence = (anchor_server, *tail)
                reversed_sequence = (anchor_server, *reversed(tail))
                if sequence <= reversed_sequence:
                    sequences.append(sequence)

        return [
            {
                rank: int(sequence[idx])
                for idx, rank in enumerate(tenant_ranks)
            }
            for sequence in sequences
        ]

    def _tenant_has_ring_rotation_symmetry(self, tenant):
        ring_collectives = {"allgather", "reducescatter", "allreduce"}
        if self.tenant_collective_programs is not None:
            program = list(self.data["schedule"].get(tenant, {}).get("collective_program", []))
            if not program:
                return False
            return all(str(op.get("collective", "")).lower() in ring_collectives for op in program)
        return str(self.collective or "").lower() in ring_collectives

    def _add_perm_constraints(self):
        tenants, ranks, servers = self.data["M"], self.data["R"], self.data["S"]

        for tenant in tenants:
            if self._use_ring_pattern_formulation(tenant):
                patterns = self._canonical_ring_patterns(tenant)
                self.ring_patterns[tenant] = patterns
                for pattern_idx, _pattern in enumerate(patterns):
                    self.Y[(tenant, pattern_idx)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"Y_ring_{tenant}_{pattern_idx}",
                    )
                self.model.addConstr(
                    gp.quicksum(
                        self.Y[(tenant, pattern_idx)]
                        for pattern_idx in range(len(patterns))
                    )
                    == 1,
                    name=f"ring_pattern_one_{tenant}",
                )
                for rank in ranks[tenant]:
                    for server in servers[tenant]:
                        self.model.addConstr(
                            self.X[(tenant, rank, server)]
                            == gp.quicksum(
                                self.Y[(tenant, pattern_idx)]
                                for pattern_idx, pattern in enumerate(patterns)
                                if pattern[rank] == server
                            ),
                            name=f"X_from_ring_pattern_{tenant}_{rank}_{server}",
                )
                continue

            for rank in ranks[tenant]:
                self.model.addConstr(
                    gp.quicksum(self.X[(tenant, rank, server)] for server in servers[tenant]) == 1,
                    name=f"rank_one_{tenant}_{rank}",
                )

        for tenant in tenants:
            if self._use_ring_pattern_formulation(tenant):
                continue
            for server in servers[tenant]:
                self.model.addConstr(
                    gp.quicksum(self.X[(tenant, rank, server)] for rank in ranks[tenant]) == 1,
                    name=f"srv_one_{tenant}_{server}",
                )

        # Ring-style collectives are invariant to a cyclic relabeling of logical ranks, and
        # often also to reversal of the cycle under symmetric equal-size communication. Fixing
        # one anchor server to rank 0 and ordering the two neighbors removes this purely
        # combinational symmetry without changing the physical solutions we can represent.
        for tenant in tenants:
            tenant_ranks = list(ranks[tenant])
            tenant_servers = list(servers[tenant])
            if self._use_ring_pattern_formulation(tenant):
                continue
            if not self._tenant_has_ring_rotation_symmetry(tenant):
                continue
            if not tenant_ranks or not tenant_servers:
                continue
            anchor_server = min(tenant_servers)
            anchor_rank = min(tenant_ranks)
            self.model.addConstr(
                self.X[(tenant, anchor_rank, anchor_server)] == 1,
                name=f"ring_anchor_{tenant}",
            )
            if self.compact_task_windows and len(tenant_ranks) > 2:
                first_neighbor_rank = tenant_ranks[1]
                last_neighbor_rank = tenant_ranks[-1]
                self.model.addConstr(
                    gp.quicksum(
                        server * self.X[(tenant, first_neighbor_rank, server)]
                        for server in tenant_servers
                    )
                    <= gp.quicksum(
                        server * self.X[(tenant, last_neighbor_rank, server)]
                        for server in tenant_servers
                    ),
                    name=f"ring_reversal_{tenant}",
                )

    def _add_U_and_endpoint_constraints(self):
        tenants, servers = self.data["M"], self.data["S"]
        tasks_by_tenant = self.data["tasks"]

        for tenant in tenants:
            candidate_servers = servers[tenant]
            patterns = self.ring_patterns.get(tenant)
            for task in tasks_by_tenant[tenant]:
                task_id = task["task_id"]
                logical_src = task["u"]
                logical_dst = task["v"]
                flow_vars = []

                for src_server in candidate_servers:
                    for dst_server in candidate_servers:
                        if src_server == dst_server:
                            continue

                        key = (tenant, task_id, src_server, dst_server)
                        if patterns is not None:
                            self.U[key] = self.model.addVar(
                                vtype=GRB.CONTINUOUS,
                                lb=0.0,
                                ub=1.0,
                                name=f"U_{tenant}_{task_id}_{src_server}_{dst_server}",
                            )
                        else:
                            self.U[key] = self.model.addVar(
                                vtype=GRB.BINARY,
                                name=f"U_{tenant}_{task_id}_{src_server}_{dst_server}",
                            )
                        flow_vars.append(self.U[key])

                        if patterns is None:
                            self.model.addConstr(
                                self.U[key] <= self.X[(tenant, logical_src, src_server)],
                                name=f"U_le_Xsrc_{tenant}_{task_id}_{src_server}_{dst_server}",
                            )
                            self.model.addConstr(
                                self.U[key] <= self.X[(tenant, logical_dst, dst_server)],
                                name=f"U_le_Xdst_{tenant}_{task_id}_{src_server}_{dst_server}",
                            )
                            self.model.addConstr(
                                self.U[key]
                                >= self.X[(tenant, logical_src, src_server)]
                                + self.X[(tenant, logical_dst, dst_server)]
                                - 1,
                                name=f"U_ge_AND_{tenant}_{task_id}_{src_server}_{dst_server}",
                            )
                        else:
                            self.model.addConstr(
                                self.U[key]
                                == gp.quicksum(
                                    self.Y[(tenant, pattern_idx)]
                                    for pattern_idx, pattern in enumerate(patterns)
                                    if (
                                        pattern[int(logical_src)] == src_server
                                        and pattern[int(logical_dst)] == dst_server
                                    )
                                ),
                                name=f"U_from_ring_pattern_{tenant}_{task_id}_{src_server}_{dst_server}",
                            )

                if patterns is None:
                    self.model.addConstr(
                        gp.quicksum(flow_vars) == 1,
                        name=f"U_onepair_{tenant}_{task_id}",
                    )

    def _add_task_time_model(self):
        tenants = self.data["M"]
        servers = self.data["S"]
        links = self.data["L"]
        capacities = self.data["cap"]
        tasks_by_tenant = self.data["tasks"]
        path_edges = self.data["path_edges"]
        path_ordered_edges = self.data["path_ordered_edges"]
        horizon = self.data["Horizon"]
        num_slots = self.data["num_slots"]
        slot_duration = self.data["slot_duration"]
        min_send_unit = self.data["min_send_unit"]
        release_gates = self.data.get("release_gates", {})
        task_earliest_slot = self.data.get("task_earliest_slot", {})
        task_latest_finish_slot = self.data.get("task_latest_finish_slot", {})
        required_load_terms = {
            tenant: {
                task["task_id"]: {link: [] for link in links}
                for task in tasks_by_tenant[tenant]
            }
            for tenant in tenants
        }
        usage_u_vars = {
            tenant: {
                task["task_id"]: {link: [] for link in links}
                for task in tasks_by_tenant[tenant]
            }
            for tenant in tenants
        }
        first_hop_u_vars = {
            tenant: {
                task["task_id"]: defaultdict(list)
                for task in tasks_by_tenant[tenant]
            }
            for tenant in tenants
        }

        for tenant in tenants:
            candidate_servers = servers[tenant]
            for task in tasks_by_tenant[tenant]:
                task_id = task["task_id"]
                logical_src = int(task["src_rank"])
                logical_dst = int(task["dst_rank"])
                volume = float(task["V"])
                for src_server in candidate_servers:
                    for dst_server in candidate_servers:
                        if src_server == dst_server:
                            continue

                        u_var = self.U[(tenant, task_id, src_server, dst_server)]
                        path_key = (int(tenant), int(src_server), int(dst_server))
                        edges = path_edges.get(path_key)
                        if edges is None:
                            continue
                        ordered_edges = path_ordered_edges.get(path_key, [])
                        if ordered_edges:
                            first_hop_u_vars[tenant][task_id][ordered_edges[0]].append(u_var)

                        for link in edges:
                            if link not in capacities:
                                continue
                            required_load_terms[tenant][task_id][link].append(volume * u_var)
                            usage_u_vars[tenant][task_id][link].append(u_var)

        task_lookup = {
            tenant: {task["task_id"]: task for task in tasks_by_tenant[tenant]}
            for tenant in tenants
        }
        def earliest_slot(tenant, task_id):
            return min(
                max(int(task_earliest_slot.get((tenant, task_id), 0)), 0),
                max(num_slots - 1, 0),
            )

        def latest_finish_slot(tenant, task_id):
            return min(
                max(
                    int(task_latest_finish_slot.get((tenant, task_id), num_slots)),
                    earliest_slot(tenant, task_id) + 1,
                ),
                num_slots,
            )

        def task_horizon_slots(tenant, task_id):
            return range(earliest_slot(tenant, task_id), latest_finish_slot(tenant, task_id))

        def c_expr(tenant, task_id, t):
            if t < earliest_slot(tenant, task_id):
                return 0.0
            if t >= latest_finish_slot(tenant, task_id):
                return 1.0
            return self.C[(tenant, task_id, t)]

        def z_expr(tenant, task_id, t):
            if t < earliest_slot(tenant, task_id) or t >= latest_finish_slot(tenant, task_id):
                return 0.0
            return self.Z[(tenant, task_id, t)]

        for tenant in tenants:
            task_ids = [task["task_id"] for task in tasks_by_tenant[tenant]]
            for task_id in task_ids:
                task = task_lookup[tenant][task_id]
                preds = list(task["preds"])
                active_horizon = list(task_horizon_slots(tenant, task_id))
                task_slot_rate_ub = self.data["task_total_volume"][(tenant, task_id)]
                if self.compact_task_windows:
                    task_slot_rate_ub = min(
                        task_slot_rate_ub,
                        max(
                            (
                                min(
                                    (
                                        capacities[edge] * slot_duration
                                        for edge in path_edges.get((int(tenant), int(src_server), int(dst_server)), ())
                                        if edge in capacities
                                    ),
                                    default=0.0,
                                )
                                for src_server in servers[tenant]
                                for dst_server in servers[tenant]
                                if src_server != dst_server
                            ),
                            default=task_slot_rate_ub,
                        ),
                    )
                for t in active_horizon:
                    self.R[(tenant, task_id, t)] = self.model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=0.0,
                        name=f"R_{tenant}_{task_id}_{t}",
                    )
                    self.Z[(tenant, task_id, t)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"Z_{tenant}_{task_id}_{t}",
                    )
                    self.S_active[(tenant, task_id, t)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"S_{tenant}_{task_id}_{t}",
                    )
                    self.D_full[(tenant, task_id, t)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"D_full_{tenant}_{task_id}_{t}",
                    )
                    self.C[(tenant, task_id, t)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"C_{tenant}_{task_id}_{t}",
                    )

                task_first_hop_exprs = {
                    port: gp.quicksum(u_vars)
                    for port, u_vars in first_hop_u_vars[tenant][task_id].items()
                }
                for port, port_expr in task_first_hop_exprs.items():
                    for t in active_horizon:
                        self.P_active[(tenant, task_id, port, t)] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"P_active_{tenant}_{task_id}_{port[0]}_{port[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.P_active[(tenant, task_id, port, t)]
                            <= self.S_active[(tenant, task_id, t)],
                            name=f"P_le_S_{tenant}_{task_id}_{port[0]}_{port[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.P_active[(tenant, task_id, port, t)] <= port_expr,
                            name=f"P_le_port_{tenant}_{task_id}_{port[0]}_{port[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.P_active[(tenant, task_id, port, t)]
                            >= self.S_active[(tenant, task_id, t)] + port_expr - 1,
                            name=f"P_ge_S_port_{tenant}_{task_id}_{port[0]}_{port[1]}_{t}",
                        )

                for link in links:
                    u_vars = usage_u_vars[tenant][task_id][link]
                    if not u_vars:
                        continue

                    self.W[(tenant, task_id, link)] = gp.quicksum(u_vars)

                    load_expr = gp.quicksum(required_load_terms[tenant][task_id][link])
                    big_m = self.data["task_total_volume"][(tenant, task_id)]

                    for t in active_horizon:
                        self.F[(tenant, task_id, link, t)] = self.model.addVar(
                            vtype=GRB.CONTINUOUS,
                            lb=0.0,
                            name=f"F_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
                        self.Q_bottleneck[(tenant, task_id, link, t)] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"Q_bn_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )

                        self.model.addConstr(
                            self.Q_bottleneck[(tenant, task_id, link, t)]
                            <= self.W[(tenant, task_id, link)],
                            name=f"Q_le_W_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )

                        self.model.addConstr(
                            self.F[(tenant, task_id, link, t)]
                            <= capacities[link] * slot_duration * self.W[(tenant, task_id, link)],
                            name=f"F_gate_W_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.F[(tenant, task_id, link, t)]
                            <= capacities[link] * slot_duration * self.S_active[(tenant, task_id, t)],
                            name=f"F_gate_S_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.F[(tenant, task_id, link, t)] <= self.R[(tenant, task_id, t)],
                            name=f"F_le_R_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.F[(tenant, task_id, link, t)]
                            >= self.R[(tenant, task_id, t)]
                            - task_slot_rate_ub * (1 - self.W[(tenant, task_id, link)])
                            - task_slot_rate_ub * (1 - self.S_active[(tenant, task_id, t)]),
                            name=f"F_ge_R_if_WS_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )

                    self.model.addConstr(
                        gp.quicksum(self.F[(tenant, task_id, link, t)] for t in active_horizon) == load_expr,
                        name=f"load_balance_{tenant}_{task_id}_{link[0]}_{link[1]}",
                    )
                    cumulative = gp.LinExpr()
                    for t in active_horizon:
                        cumulative += self.F[(tenant, task_id, link, t)]
                        self.model.addConstr(
                            cumulative >= load_expr - big_m * (1 - self.C[(tenant, task_id, t)]),
                            name=f"complete_if_sent_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )

                for t in active_horizon:
                    self.model.addConstr(
                        gp.quicksum(
                            self.Q_bottleneck[(tenant, task_id, link, t)]
                            for link in links
                            if (tenant, task_id, link, t) in self.Q_bottleneck
                        )
                        == self.D_full[(tenant, task_id, t)],
                        name=f"Q_one_bottleneck_{tenant}_{task_id}_{t}",
                    )

                for t in range(earliest_slot(tenant, task_id), latest_finish_slot(tenant, task_id) - 1):
                    self.model.addConstr(
                        self.C[(tenant, task_id, t)] <= self.C[(tenant, task_id, t + 1)],
                        name=f"C_mono_{tenant}_{task_id}_{t}",
                    )

                self.model.addConstr(
                    self.C[(tenant, task_id, latest_finish_slot(tenant, task_id) - 1)] == 1,
                    name=f"C_terminal_{tenant}_{task_id}",
                )
                if self.compact_task_windows:
                    for t in active_horizon:
                        if preds:
                            for pred_task_id in preds:
                                self.model.addConstr(
                                    self.C[(tenant, task_id, t)]
                                    <= c_expr(tenant, pred_task_id, t - 1),
                                    name=f"C_after_pred_{tenant}_{task_id}_{pred_task_id}_{t}",
                                )
                        elif task_id in release_gates.get(tenant, {}):
                            release_gate = release_gates[tenant][task_id]
                            gap_slots = int(release_gate.get("gap_slots", 0))
                            release_check_t = t - gap_slots - 1
                            if release_check_t < 0:
                                self.model.addConstr(
                                    self.C[(tenant, task_id, t)] == 0,
                                    name=f"C_before_program_release_{tenant}_{task_id}_{t}",
                                )
                            else:
                                for previous_task_id in release_gate["previous_task_ids"]:
                                    self.model.addConstr(
                                        self.C[(tenant, task_id, t)]
                                        <= c_expr(tenant, previous_task_id, release_check_t),
                                        name=(
                                            f"C_after_program_release_{tenant}_{task_id}_"
                                            f"{previous_task_id}_{t}"
                                        ),
                                    )

                first_slot = earliest_slot(tenant, task_id)
                if first_slot == 0 and preds:
                    self.model.addConstr(
                        self.Z[(tenant, task_id, 0)] == 0,
                        name=f"Z_wait_preds_{tenant}_{task_id}",
                    )
                elif first_slot == 0 and task_id in release_gates.get(tenant, {}):
                    self.model.addConstr(
                        self.Z[(tenant, task_id, 0)] == 0,
                        name=f"Z_wait_program_release_{tenant}_{task_id}",
                    )
                elif first_slot == 0:
                    self.model.addConstr(
                        self.Z[(tenant, task_id, 0)] == 1,
                        name=f"Z_ready_init_{tenant}_{task_id}",
                    )
                if first_slot == 0:
                    self.model.addConstr(
                        self.S_active[(tenant, task_id, 0)] <= self.Z[(tenant, task_id, 0)],
                        name=f"S_init_le_Z_{tenant}_{task_id}",
                    )

                for t in active_horizon:
                    if t == 0:
                        continue
                    self.model.addConstr(
                        self.Z[(tenant, task_id, t)] <= 1 - c_expr(tenant, task_id, t - 1),
                        name=f"Z_after_completion_{tenant}_{task_id}_{t}",
                    )
                    if preds:
                        for pred_task_id in preds:
                            self.model.addConstr(
                                self.Z[(tenant, task_id, t)] <= c_expr(tenant, pred_task_id, t - 1),
                                name=f"pred_ready_{tenant}_{task_id}_{pred_task_id}_{t}",
                            )
                        self.model.addConstr(
                            self.Z[(tenant, task_id, t)]
                            >= 1
                            - c_expr(tenant, task_id, t - 1)
                            + gp.quicksum(c_expr(tenant, pred_task_id, t - 1) for pred_task_id in preds)
                            - len(preds),
                            name=f"Z_ready_exact_{tenant}_{task_id}_{t}",
                        )
                    elif task_id in release_gates.get(tenant, {}):
                        release_gate = release_gates[tenant][task_id]
                        previous_task_ids = list(release_gate["previous_task_ids"])
                        gap_slots = int(release_gate.get("gap_slots", 0))
                        release_check_t = t - gap_slots - 1
                        if release_check_t < 0:
                            self.model.addConstr(
                                self.Z[(tenant, task_id, t)] == 0,
                                name=f"Z_before_program_release_{tenant}_{task_id}_{t}",
                            )
                        else:
                            for previous_task_id in previous_task_ids:
                                self.model.addConstr(
                                    self.Z[(tenant, task_id, t)]
                                    <= c_expr(tenant, previous_task_id, release_check_t),
                                    name=f"program_release_ready_{tenant}_{task_id}_{previous_task_id}_{t}",
                                )
                            self.model.addConstr(
                                self.Z[(tenant, task_id, t)]
                                >= 1
                                - c_expr(tenant, task_id, t - 1)
                                + gp.quicksum(
                                    c_expr(tenant, previous_task_id, release_check_t)
                                    for previous_task_id in previous_task_ids
                                )
                                - len(previous_task_ids),
                                name=f"Z_program_release_exact_{tenant}_{task_id}_{t}",
                            )
                    else:
                        self.model.addConstr(
                            self.Z[(tenant, task_id, t)] == 1 - c_expr(tenant, task_id, t - 1),
                            name=f"Z_no_pred_exact_{tenant}_{task_id}_{t}",
                        )

                    self.model.addConstr(
                        self.S_active[(tenant, task_id, t)] <= self.Z[(tenant, task_id, t)],
                        name=f"S_le_Z_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(
                        self.S_active[(tenant, task_id, t)]
                        >= self.S_active.get((tenant, task_id, t - 1), 0.0) - c_expr(tenant, task_id, t - 1),
                        name=f"S_nonpreempt_{tenant}_{task_id}_{t}",
                    )

                for t in active_horizon:
                    self.model.addConstr(
                        self.D_full[(tenant, task_id, t)] <= self.S_active[(tenant, task_id, t)],
                        name=f"D_full_le_S_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(
                        self.D_full[(tenant, task_id, t)] <= 1 - self.C[(tenant, task_id, t)],
                        name=f"D_full_le_not_done_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(
                        self.D_full[(tenant, task_id, t)]
                        >= self.S_active[(tenant, task_id, t)] - self.C[(tenant, task_id, t)],
                        name=f"D_full_ge_active_not_done_{tenant}_{task_id}_{t}",
                    )

                for t in active_horizon:
                    self.model.addConstr(
                        self.R[(tenant, task_id, t)]
                        <= task_slot_rate_ub * self.S_active[(tenant, task_id, t)],
                        name=f"R_gate_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(
                        self.R[(tenant, task_id, t)]
                        >= min_send_unit * self.S_active[(tenant, task_id, t)],
                        name=f"R_active_lb_{tenant}_{task_id}_{t}",
                    )

                self.task_finish[(tenant, task_id)] = 1 + gp.quicksum(
                    1 - c_expr(tenant, task_id, t) for t in horizon
                )

            for sender_rank, sender_task_ids in self.data["sender_tasks"][tenant].items():
                if not sender_task_ids:
                    continue
                sender_task_ids = list(sender_task_ids)
                sender_ports = sorted(
                    {
                        port
                        for task_id in sender_task_ids
                        for port in first_hop_u_vars[tenant][task_id]
                    }
                )
                for t in horizon:
                    for port in sender_ports:
                        port_active_terms = [
                            self.P_active[(tenant, task_id, port, t)]
                            for task_id in sender_task_ids
                            if (tenant, task_id, port, t) in self.P_active
                        ]
                        if not port_active_terms:
                            continue
                        port_active_sum = gp.quicksum(port_active_terms)
                        self.model.addConstr(
                            port_active_sum <= 1,
                            name=f"port_service_ub_{tenant}_{sender_rank}_{port[0]}_{port[1]}_{t}",
                        )
                        for task_id in sender_task_ids:
                            if port not in first_hop_u_vars[tenant][task_id]:
                                continue
                            if (tenant, task_id, t) not in self.Z:
                                continue
                            port_selected = gp.quicksum(first_hop_u_vars[tenant][task_id][port])
                            self.model.addConstr(
                                port_active_sum
                                >= self.Z[(tenant, task_id, t)] + port_selected - 1,
                                name=f"port_service_lb_{tenant}_{sender_rank}_{task_id}_{port[0]}_{port[1]}_{t}",
                            )
                    for port in sender_ports:
                        for idx, task_id in enumerate(sender_task_ids):
                            if (tenant, task_id, port, t) not in self.P_active:
                                continue
                            if t == 0:
                                start_expr = self.P_active[(tenant, task_id, port, 0)]
                            else:
                                previous_p = self.P_active.get((tenant, task_id, port, t - 1), 0.0)
                                start_expr = (
                                    self.P_active[(tenant, task_id, port, t)]
                                    - previous_p
                                )
                            for earlier_task_id in sender_task_ids[:idx]:
                                if port not in first_hop_u_vars[tenant][earlier_task_id]:
                                    continue
                                earlier_port_selected = gp.quicksum(
                                    first_hop_u_vars[tenant][earlier_task_id][port]
                                )
                                self.model.addConstr(
                                    start_expr
                                    <= 2
                                    - z_expr(tenant, earlier_task_id, t)
                                    - earlier_port_selected,
                                    name=(
                                        f"port_fifo_start_{tenant}_{sender_rank}_{port[0]}_{port[1]}_"
                                        f"{earlier_task_id}_{task_id}_{t}"
                                    ),
                                )

            self.tenant_finish[tenant] = self.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"T_tenant_slot_{tenant}",
            )
            for task_id in task_ids:
                self.model.addConstr(
                    self.tenant_finish[tenant] >= self.task_finish[(tenant, task_id)],
                    name=f"T_tenant_ge_task_{tenant}_{task_id}",
                )
        max_rate_big_m = max(
            max(self.data["task_total_volume"].values(), default=1.0),
            max((capacities[link] * slot_duration for link in links), default=1.0),
        )

        for link in links:
            capacity_per_slot = capacities[link] * slot_duration
            for t in horizon:
                self.LinkMinRate[(link, t)] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=capacity_per_slot if self.compact_task_windows else GRB.INFINITY,
                    name=f"Rmin_{link[0]}_{link[1]}_{t}",
                )
                link_flow_expr = gp.quicksum(
                    self.F[(tenant, task["task_id"], link, t)]
                    for tenant in tenants
                    for task in tasks_by_tenant[tenant]
                    if (tenant, task["task_id"], link, t) in self.F
                )
                self.model.addConstr(
                    link_flow_expr <= capacity_per_slot,
                    name=f"link_cap_{link[0]}_{link[1]}_{t}",
                )

                for tenant in tenants:
                    for task in tasks_by_tenant[tenant]:
                        task_id = task["task_id"]
                        w_key = (tenant, task_id, link)
                        if w_key in self.W and (tenant, task_id, t) in self.R:
                            bn_big_m = max_rate_big_m
                            if self.compact_task_windows:
                                bn_big_m = min(
                                    max_rate_big_m,
                                    min(
                                        self.data["task_total_volume"][(tenant, task_id)],
                                        capacity_per_slot,
                                    ),
                                )
                            self.model.addConstr(
                                self.R[(tenant, task_id, t)]
                                >= self.LinkMinRate[(link, t)]
                                - bn_big_m * (1 - self.W[w_key])
                                - bn_big_m * (1 - self.S_active[(tenant, task_id, t)]),
                                name=f"bn_minrate_lb_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                            )

                        q_key = (tenant, task_id, link, t)
                        if q_key not in self.Q_bottleneck:
                            continue

                        q_var = self.Q_bottleneck[q_key]
                        self.model.addConstr(
                            link_flow_expr
                            >= capacity_per_slot - capacity_per_slot * (1 - q_var),
                            name=f"bn_saturates_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.R[(tenant, task_id, t)]
                            <= self.LinkMinRate[(link, t)] + (
                                min(max_rate_big_m, capacity_per_slot)
                                if self.compact_task_windows
                                else max_rate_big_m
                            ) * (1 - q_var),
                            name=f"bn_minrate_eq_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )

    def _set_lexicographic_objective(self):
        tenant_count = max(len(self.data["M"]), 1)
        avg_completion = gp.quicksum(self.tenant_finish[tenant] for tenant in self.data["M"]) / tenant_count
        self.T_max = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="T_max",
        )
        for tenant in self.data["M"]:
            self.model.addConstr(
                self.T_max >= self.tenant_finish[tenant],
                name=f"T_max_ge_{tenant}",
        )

        self.model.ModelSense = GRB.MINIMIZE
        self.model.setObjectiveN(avg_completion, index=0, priority=2, name="avg_jct")
        self.model.setObjectiveN(self.T_max, index=1, priority=1, name="makespan")

    def _apply_mapping_warm_start(self, mapping):
        if not mapping:
            return
        mapping = self._canonicalize_ring_mapping(mapping)
        self.warm_start_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in mapping.items()
        }
        for (tenant, rank, server), var in self.X.items():
            var.Start = 1.0 if mapping.get(tenant, {}).get(rank) == server else 0.0
        for (tenant, pattern_idx), var in self.Y.items():
            pattern = self.ring_patterns.get(tenant, [])[pattern_idx]
            var.Start = 1.0 if all(
                mapping.get(tenant, {}).get(rank) == server
                for rank, server in pattern.items()
            ) else 0.0
        for key, var in self.U.items():
            tenant, task_id, src_server, dst_server = key
            task = next(
                task
                for task in self.data["tasks"][tenant]
                if int(task["task_id"]) == int(task_id)
            )
            logical_src = int(task["src_rank"])
            logical_dst = int(task["dst_rank"])
            expected_src = mapping.get(tenant, {}).get(logical_src)
            expected_dst = mapping.get(tenant, {}).get(logical_dst)
            var.Start = 1.0 if (expected_src == src_server and expected_dst == dst_server) else 0.0
        self.model.update()

    def _register_mapping_seed(self, mapping):
        if not mapping:
            return
        mapping = self._canonicalize_ring_mapping(mapping)
        self.warm_start_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in mapping.items()
        }

    def _canonicalize_ring_mapping(self, mapping):
        canonical = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in mapping.items()
        }
        for tenant, rank_to_server in canonical.items():
            if not self._tenant_has_ring_rotation_symmetry(tenant):
                continue
            tenant_ranks = sorted(self.data["R"][tenant])
            if not tenant_ranks:
                continue
            anchor_rank = tenant_ranks[0]
            candidate_servers = sorted(self.data["S"][tenant])
            if not candidate_servers:
                continue
            anchor_server = candidate_servers[0]
            ordered_servers = [rank_to_server[rank] for rank in tenant_ranks]
            if anchor_server not in ordered_servers:
                continue
            anchor_index = ordered_servers.index(anchor_server)
            rotated = ordered_servers[anchor_index:] + ordered_servers[:anchor_index]
            if self.compact_task_windows and len(rotated) > 2:
                reversed_rotated = [rotated[0]] + list(reversed(rotated[1:]))
                if tuple(reversed_rotated) < tuple(rotated):
                    rotated = reversed_rotated
            canonical[tenant] = {
                rank: rotated[idx]
                for idx, rank in enumerate(tenant_ranks)
            }
        return canonical

    def _serial_slot_schedule_from_mapping(self, mapping):
        if not mapping:
            return None

        slot_duration = float(self.data["slot_duration"])
        capacities = self.data["cap"]
        path_edges = self.data["path_edges"]
        tasks_by_tenant = self.data["tasks"]
        sender_tasks = self.data["sender_tasks"]
        release_gates = self.data.get("release_gates", {})

        predecessors = {}
        sender_prev = {}
        for tenant in self.data["M"]:
            predecessors[tenant] = {int(task["task_id"]): set(int(pred) for pred in task["preds"]) for task in tasks_by_tenant[tenant]}
            sender_prev[tenant] = {}
            for sender_rank, sender_task_ids in sender_tasks[tenant].items():
                previous = None
                for task_id in sender_task_ids:
                    task_id = int(task_id)
                    if previous is not None:
                        predecessors[tenant][task_id].add(previous)
                        sender_prev[tenant][task_id] = previous
                    previous = task_id

        ready_finish_slot = {}
        task_start_slot = {}
        task_finish_slot = {}
        task_send_per_slot = {}

        cursor = 0
        remaining = {
            tenant: set(int(task["task_id"]) for task in tasks_by_tenant[tenant])
            for tenant in self.data["M"]
        }
        task_lookup = {
            tenant: {int(task["task_id"]): task for task in tasks_by_tenant[tenant]}
            for tenant in self.data["M"]
        }

        while any(remaining.values()):
            ready_candidates = []
            for tenant in self.data["M"]:
                for task_id in sorted(remaining[tenant]):
                    if all((tenant, pred) in ready_finish_slot for pred in predecessors[tenant][task_id]):
                        ready_candidates.append((tenant, task_id))
            if not ready_candidates:
                break
            tenant, task_id = min(ready_candidates)
            task = task_lookup[tenant][task_id]
            src_rank = int(task["src_rank"])
            dst_rank = int(task["dst_rank"])
            src_server = int(mapping[tenant][src_rank])
            dst_server = int(mapping[tenant][dst_rank])
            volume = float(task["V"])
            edges = tuple(path_edges[(src_server, dst_server)])
            bottleneck = min(float(capacities[edge]) for edge in edges)
            full_slot_capacity = bottleneck * slot_duration
            active_slots = max(1, int(math.ceil(volume / max(full_slot_capacity, 1e-12))))
            pred_ready = 0
            if predecessors[tenant][task_id]:
                pred_ready = max(ready_finish_slot[(tenant, pred)] for pred in predecessors[tenant][task_id])
            start_slot = max(cursor, pred_ready)
            sends = []
            remaining_volume = volume
            for _ in range(active_slots):
                send = min(full_slot_capacity, remaining_volume)
                sends.append(send)
                remaining_volume -= send
            finish_index = start_slot + active_slots - 1
            finish_slot = finish_index + 1

            task_start_slot[(tenant, task_id)] = start_slot
            task_finish_slot[(tenant, task_id)] = finish_slot
            task_send_per_slot[(tenant, task_id)] = sends
            ready_finish_slot[(tenant, task_id)] = finish_slot
            remaining[tenant].remove(task_id)
            cursor = finish_slot

        task_ready_slot = {}
        for tenant in self.data["M"]:
            for task in tasks_by_tenant[tenant]:
                task_id = int(task["task_id"])
                ready_slot = 0
                preds = predecessors[tenant].get(task_id, set())
                if preds:
                    ready_slot = max(ready_finish_slot[(tenant, pred)] for pred in preds)
                elif task_id in release_gates.get(tenant, {}):
                    gate = release_gates[tenant][task_id]
                    previous_task_ids = [int(prev) for prev in gate.get("previous_task_ids", [])]
                    gap_slots = int(gate.get("gap_slots", 0))
                    ready_slot = max((ready_finish_slot.get((tenant, prev), 0) for prev in previous_task_ids), default=0) + gap_slots
                task_ready_slot[(tenant, task_id)] = ready_slot

        return task_ready_slot, task_start_slot, task_finish_slot, task_send_per_slot

    def _apply_serial_schedule_warm_start(self, mapping):
        schedule = self._serial_slot_schedule_from_mapping(mapping)
        if schedule is None:
            return
        task_ready_slot, task_start_slot, task_finish_slot, task_send_per_slot = schedule
        self._apply_slot_schedule_warm_start(
            mapping,
            task_ready_slot=task_ready_slot,
            task_start_slot=task_start_slot,
            task_finish_slot=task_finish_slot,
            task_send_per_slot=task_send_per_slot,
        )

    @staticmethod
    def _simulator_flow_id(tx_id):
        if isinstance(tx_id, tuple) and tx_id:
            return str(tx_id[0])
        return str(tx_id)

    def _simulator_priority_slots_from_mapping(self, mapping, simulator_result=None):
        if not mapping:
            return None, simulator_result

        mapping = self._canonicalize_ring_mapping(mapping)

        if simulator_result is None:
            simulator_result = simulate_collective_details(
                self.datacenter.topology,
                mapping,
                self.path_table,
                self.single_flow_size,
                self.collective,
                tenant_start_times=self.tenant_start_times,
                tenant_collective_specs=self.tenant_collective_specs,
                tenant_collective_programs=self.tenant_collective_programs,
            )

        slot_duration = float(self.data["slot_duration"])
        num_slots = int(self.data["num_slots"])
        tasks_by_tenant = self.data["tasks"]

        task_by_name = {}
        for tenant in self.data["M"]:
            for task in tasks_by_tenant[tenant]:
                task_name = task.get("name")
                if task_name is not None:
                    task_by_name[str(task_name)] = (tenant, int(task["task_id"]))

        task_ready_time = {}
        task_start_time = {}
        task_finish_time = {}

        for tx_id, ready_time in simulator_result.get("chunk_ready_time", {}).items():
            flow_id = self._simulator_flow_id(tx_id).split("-Q", 1)[0]
            if flow_id in task_by_name:
                key = task_by_name[flow_id]
                task_ready_time[key] = min(task_ready_time.get(key, float("inf")), float(ready_time))

        for tx_id, start_time in simulator_result.get("tx_service_start_time", {}).items():
            flow_id = self._simulator_flow_id(tx_id).split("-Q", 1)[0]
            if flow_id in task_by_name:
                key = task_by_name[flow_id]
                task_start_time[key] = min(task_start_time.get(key, float("inf")), float(start_time))

        for tx_id, finish_time in simulator_result.get("tx_complete_time", {}).items():
            flow_id = self._simulator_flow_id(tx_id).split("-Q", 1)[0]
            if flow_id in task_by_name:
                key = task_by_name[flow_id]
                task_finish_time[key] = max(task_finish_time.get(key, 0.0), float(finish_time))

        expected_task_keys = {
            (tenant, int(task["task_id"]))
            for tenant in self.data["M"]
            for task in tasks_by_tenant[tenant]
        }
        missing_task_keys = sorted(expected_task_keys - set(task_finish_time))
        if missing_task_keys:
            if self.verbose:
                print(
                    "[simulator-warm-start] skipped schedule start; "
                    f"missing simulator timings for {len(missing_task_keys)} tasks"
                )
            return None, simulator_result

        def ceil_slot(value):
            return int(math.ceil(max(float(value), 0.0) / slot_duration - 1e-9))

        task_priority_slot = {}
        for tenant in self.data["M"]:
            for task in tasks_by_tenant[tenant]:
                task_id = int(task["task_id"])
                key = (tenant, task_id)
                if key not in task_finish_time:
                    continue

                ready_time = task_ready_time.get(key, task_start_time.get(key, 0.0))
                start_time = task_start_time.get(key, ready_time)
                finish_time = max(task_finish_time[key], start_time)
                ready_slot = min(max(ceil_slot(ready_time), 0), max(num_slots - 1, 0))
                start_slot = min(max(ceil_slot(start_time), ready_slot), max(num_slots - 1, 0))
                finish_slot = min(max(ceil_slot(finish_time), start_slot + 1), num_slots)

                task_priority_slot[key] = start_slot
        return task_priority_slot, simulator_result

    def _apply_simulator_schedule_warm_start(self, mapping, simulator_result=None):
        if not mapping:
            return simulator_result

        mapping = self._canonicalize_ring_mapping(mapping)

        task_priority_slot, simulator_result = self._simulator_priority_slots_from_mapping(
            mapping,
            simulator_result=simulator_result,
        )
        if task_priority_slot is None:
            return simulator_result

        shifted_schedule = self._shift_slot_schedule_for_ilp_start(
            mapping,
            task_priority_slot,
        )
        if shifted_schedule is not None:
            task_ready_slot, task_start_slot, task_finish_slot, task_send_per_slot = shifted_schedule
        else:
            return simulator_result

        self._apply_mapping_warm_start(mapping)
        self._apply_slot_schedule_warm_start(
            mapping,
            task_ready_slot=task_ready_slot,
            task_start_slot=task_start_slot,
            task_finish_slot=task_finish_slot,
            task_send_per_slot=task_send_per_slot,
        )
        return simulator_result

    def _shift_slot_schedule_for_ilp_start(self, mapping, task_priority_slot):
        slot_duration = float(self.data["slot_duration"])
        num_slots = int(self.data["num_slots"])
        capacities = self.data["cap"]
        path_edges = self.data["path_edges"]
        path_ordered_edges = self.data["path_ordered_edges"]
        tasks_by_tenant = self.data["tasks"]
        release_gates = self.data.get("release_gates", {})

        task_lookup = {
            tenant: {int(task["task_id"]): task for task in tasks_by_tenant[tenant]}
            for tenant in self.data["M"]
        }
        scheduling_predecessors = {
            tenant: {
                int(task["task_id"]): set(int(pred) for pred in task.get("preds", []))
                for task in tasks_by_tenant[tenant]
            }
            for tenant in self.data["M"]
        }
        sender_previous = {tenant: {} for tenant in self.data["M"]}
        for tenant in self.data["M"]:
            for sender_rank, sender_task_ids in self.data["sender_tasks"][tenant].items():
                previous = None
                for task_id in sender_task_ids:
                    task_id = int(task_id)
                    if previous is not None:
                        scheduling_predecessors[tenant][task_id].add(previous)
                        sender_previous[tenant][task_id] = previous
                    previous = task_id

        task_path = {}
        task_first_hop_port = {}
        task_bottleneck = {}
        for tenant in self.data["M"]:
            for task in tasks_by_tenant[tenant]:
                task_id = int(task["task_id"])
                src_server = int(mapping[tenant][int(task["src_rank"])])
                dst_server = int(mapping[tenant][int(task["dst_rank"])])
                path_key = (int(tenant), int(src_server), int(dst_server))
                edges = tuple(path_edges.get(path_key, ()))
                ordered_edges = tuple(path_ordered_edges.get(path_key, ()))
                bottleneck = min((float(capacities[edge]) for edge in edges if edge in capacities), default=0.0)
                task_path[(tenant, task_id)] = edges
                task_first_hop_port[(tenant, task_id)] = ordered_edges[0] if ordered_edges else None
                task_bottleneck[(tenant, task_id)] = bottleneck

        remaining = {
            (tenant, int(task["task_id"]))
            for tenant in self.data["M"]
            for task in tasks_by_tenant[tenant]
        }
        task_ready_slot = {}
        task_start_slot = {}
        task_finish_slot = {}
        task_send_per_slot = {}
        link_load_by_slot = defaultdict(float)
        sender_busy_slots = defaultdict(set)

        def original_ready_slot(tenant, task_id):
            task = task_lookup[tenant][task_id]
            preds = [int(pred) for pred in task.get("preds", [])]
            if preds:
                return max(task_finish_slot[(tenant, pred)] for pred in preds)
            if task_id in release_gates.get(tenant, {}):
                gate = release_gates[tenant][task_id]
                previous_task_ids = [int(prev) for prev in gate.get("previous_task_ids", [])]
                gap_slots = int(math.ceil(float(gate.get("gap_after_s", 0.0)) / slot_duration - 1e-9))
                return max((task_finish_slot.get((tenant, prev), 0) for prev in previous_task_ids), default=0) + gap_slots
            return 0

        def scheduling_ready_slot(tenant, task_id):
            ready = original_ready_slot(tenant, task_id)
            for pred in scheduling_predecessors[tenant].get(task_id, set()):
                ready = max(ready, task_finish_slot[(tenant, pred)])
            return ready

        def task_sends(tenant, task_id):
            task = task_lookup[tenant][task_id]
            bottleneck = max(task_bottleneck[(tenant, task_id)], 1e-12)
            full_slot_capacity = bottleneck * slot_duration
            remaining_volume = float(task["V"])
            sends = []
            while remaining_volume > 1e-12:
                send = min(full_slot_capacity, remaining_volume)
                sends.append(send)
                remaining_volume -= send
            return sends or [0.0]

        def can_place(tenant, task_id, start_slot, sends):
            task = task_lookup[tenant][task_id]
            sender_rank = int(task["src_rank"])
            edges = task_path[(tenant, task_id)]
            first_hop_port = task_first_hop_port[(tenant, task_id)]
            if start_slot < 0 or start_slot + len(sends) > num_slots:
                return False
            for offset, send in enumerate(sends):
                t = start_slot + offset
                if t in sender_busy_slots[(tenant, sender_rank, first_hop_port)]:
                    return False
                for edge in edges:
                    capacity_per_slot = float(capacities.get(edge, 0.0)) * slot_duration
                    if link_load_by_slot[(edge, t)] + send > capacity_per_slot + 1e-9:
                        return False
            return True

        while remaining:
            ready_candidates = []
            for tenant, task_id in sorted(remaining):
                if all((tenant, pred) not in remaining for pred in scheduling_predecessors[tenant].get(task_id, set())):
                    ready_candidates.append((tenant, task_id))
            if not ready_candidates:
                if self.verbose:
                    print("[simulator-warm-start] skipped schedule start; cyclic scheduling dependencies")
                return None

            tenant, task_id = min(
                ready_candidates,
                key=lambda key: (
                    scheduling_ready_slot(*key),
                    task_priority_slot.get(key, num_slots + 1),
                    key[0],
                    key[1],
                ),
            )
            sends = task_sends(tenant, task_id)
            start_slot = scheduling_ready_slot(tenant, task_id)

            while not can_place(tenant, task_id, start_slot, sends):
                start_slot += 1
                if start_slot + len(sends) > num_slots:
                    if self.verbose:
                        print(
                            "[simulator-warm-start] skipped schedule start; "
                            f"could not place task {(tenant, task_id)} within horizon"
                        )
                    return None

            ready_for_sender = scheduling_ready_slot(tenant, task_id)
            sender_rank = int(task_lookup[tenant][task_id]["src_rank"])
            first_hop_port = task_first_hop_port[(tenant, task_id)]
            if any(
                t not in sender_busy_slots[(tenant, sender_rank, first_hop_port)]
                for t in range(ready_for_sender, start_slot)
            ):
                if self.verbose:
                    print(
                        "[simulator-warm-start] skipped schedule start; "
                        f"placing task {(tenant, task_id)} would leave sender {sender_rank} "
                        f"port {first_hop_port} idle after readiness"
                    )
                return None

            finish_slot = start_slot + len(sends)
            task = task_lookup[tenant][task_id]
            if not task.get("preds", []) and task_id in release_gates.get(tenant, {}):
                task_ready_slot[(tenant, task_id)] = start_slot
            else:
                task_ready_slot[(tenant, task_id)] = original_ready_slot(tenant, task_id)
            task_start_slot[(tenant, task_id)] = start_slot
            task_finish_slot[(tenant, task_id)] = finish_slot
            task_send_per_slot[(tenant, task_id)] = sends

            for offset, send in enumerate(sends):
                t = start_slot + offset
                sender_busy_slots[(tenant, sender_rank, first_hop_port)].add(t)
                for edge in task_path[(tenant, task_id)]:
                    link_load_by_slot[(edge, t)] += send
            remaining.remove((tenant, task_id))

        return task_ready_slot, task_start_slot, task_finish_slot, task_send_per_slot

    def _apply_slot_schedule_warm_start(
        self,
        mapping,
        *,
        task_ready_slot,
        task_start_slot,
        task_finish_slot,
        task_send_per_slot,
    ):
        slot_duration = float(self.data["slot_duration"])
        num_slots = int(self.data["num_slots"])
        capacities = self.data["cap"]
        path_edges = self.data["path_edges"]
        path_ordered_edges = self.data["path_ordered_edges"]
        tasks_by_tenant = self.data["tasks"]

        task_path = {}
        task_first_hop_port = {}
        task_bottleneck_links = {}
        for tenant in self.data["M"]:
            for task in tasks_by_tenant[tenant]:
                task_id = int(task["task_id"])
                src_rank = int(task["src_rank"])
                dst_rank = int(task["dst_rank"])
                src_server = int(mapping[tenant][src_rank])
                dst_server = int(mapping[tenant][dst_rank])
                path_key = (int(tenant), int(src_server), int(dst_server))
                edges = tuple(path_edges.get(path_key, ()))
                ordered_edges = tuple(path_ordered_edges.get(path_key, ()))
                bottleneck = min((float(capacities[edge]) for edge in edges if edge in capacities), default=0.0)
                bottleneck_links = tuple(edge for edge in edges if abs(float(capacities.get(edge, 0.0)) - bottleneck) <= 1e-12)
                task_path[(tenant, task_id)] = edges
                task_first_hop_port[(tenant, task_id)] = ordered_edges[0] if ordered_edges else None
                task_bottleneck_links[(tenant, task_id)] = bottleneck_links
                task_bottleneck_links[(tenant, task_id, "selected")] = bottleneck_links[0] if bottleneck_links else None

                if (tenant, task_id) not in task_send_per_slot and (tenant, task_id) in task_start_slot:
                    start_slot = int(task_start_slot[(tenant, task_id)])
                    finish_slot = int(task_finish_slot.get((tenant, task_id), start_slot + 1))
                    active_slots = max(finish_slot - start_slot, 1)
                    volume = float(task["V"])
                    full_slot_capacity = max(bottleneck * slot_duration, 1e-12)
                    sends = []
                    remaining = volume
                    for _ in range(active_slots):
                        send = min(full_slot_capacity, remaining)
                        sends.append(send)
                        remaining -= send
                    task_send_per_slot[(tenant, task_id)] = sends

        tenant_finish_slots = {}
        p_active_by_task_time = defaultdict(list)
        for (tenant, task_id, port, t), var in self.P_active.items():
            p_active_by_task_time[(tenant, task_id, t)].append((port, var))

        for tenant in self.data["M"]:
            for task in tasks_by_tenant[tenant]:
                task_id = int(task["task_id"])
                start_slot = int(task_start_slot.get((tenant, task_id), num_slots + 1))
                finish_slot = int(task_finish_slot.get((tenant, task_id), num_slots + 1))
                finish_index = finish_slot - 1
                sends = list(task_send_per_slot.get((tenant, task_id), []))
                used_links = set(task_path.get((tenant, task_id), ()))
                selected_bottleneck_link = task_bottleneck_links.get((tenant, task_id, "selected"))
                ready_slot = int(task_ready_slot.get((tenant, task_id), start_slot))

                if 0 <= finish_slot <= num_slots:
                    tenant_finish_slots[tenant] = max(tenant_finish_slots.get(tenant, 0), finish_slot)

                for t in range(num_slots):
                    active = 1.0 if start_slot <= t <= finish_index else 0.0
                    completed = 1.0 if t >= finish_index else 0.0
                    ready = 1.0 if (ready_slot <= t <= finish_index) else 0.0
                    unfinished = bool(active and t < finish_index)
                    send_amount = sends[t - start_slot] if 0 <= (t - start_slot) < len(sends) else 0.0

                    if (tenant, task_id, t) in self.S_active:
                        self.S_active[(tenant, task_id, t)].Start = active
                    first_hop_port = task_first_hop_port.get((tenant, task_id))
                    for port, var in p_active_by_task_time.get((tenant, task_id, t), []):
                        var.Start = active if port == first_hop_port else 0.0
                    if (tenant, task_id, t) in self.C:
                        self.C[(tenant, task_id, t)].Start = completed
                    if (tenant, task_id, t) in self.Z:
                        self.Z[(tenant, task_id, t)].Start = ready
                    if (tenant, task_id, t) in self.D_full:
                        self.D_full[(tenant, task_id, t)].Start = 1.0 if unfinished else 0.0
                    if (tenant, task_id, t) in self.R:
                        self.R[(tenant, task_id, t)].Start = send_amount

                    for link in self.data["L"]:
                        f_key = (tenant, task_id, link, t)
                        q_key = (tenant, task_id, link, t)
                        if f_key in self.F:
                            self.F[f_key].Start = send_amount if link in used_links else 0.0
                        if q_key in self.Q_bottleneck:
                            self.Q_bottleneck[q_key].Start = (
                                1.0 if (unfinished and link == selected_bottleneck_link) else 0.0
                            )

        for (link, t), var in self.LinkMinRate.items():
            active_rates = []
            for tenant in self.data["M"]:
                for task in tasks_by_tenant[tenant]:
                    task_id = int(task["task_id"])
                    if link not in set(task_path.get((tenant, task_id), ())):
                        continue
                    start_slot = int(task_start_slot.get((tenant, task_id), num_slots + 1))
                    finish_slot = int(task_finish_slot.get((tenant, task_id), -1))
                    if not (start_slot <= t < finish_slot):
                        continue
                    sends = task_send_per_slot.get((tenant, task_id), [])
                    send_amount = sends[t - start_slot] if 0 <= (t - start_slot) < len(sends) else 0.0
                    if send_amount > 0.0:
                        active_rates.append(send_amount)
            var.Start = min(active_rates) if active_rates else 0.0

        for tenant, var in self.tenant_finish.items():
            var.Start = float(tenant_finish_slots.get(tenant, 0))
        if self.T_max is not None:
            self.T_max.Start = float(max(tenant_finish_slots.values(), default=0))
        self.warm_start_schedule_horizon_bound = float(max(tenant_finish_slots.values(), default=0))

        self.model.update()

    @staticmethod
    def _defined_start_value(var):
        try:
            start_value = var.Start
        except Exception:
            return None
        if start_value == GRB.UNDEFINED or abs(float(start_value)) >= 1e100:
            return None
        return float(start_value)

    def _mapping_horizon_bound(self, mapping):
        if not mapping:
            return None
        mapping = self._canonicalize_ring_mapping(mapping)

        simulator_result = simulate_collective_details(
            self.datacenter.topology,
            mapping,
            self.path_table,
            self.single_flow_size,
            self.collective,
            tenant_start_times=self.tenant_start_times,
            tenant_collective_specs=self.tenant_collective_specs,
            tenant_collective_programs=self.tenant_collective_programs,
        )
        tasks_by_tenant = self.data["tasks"]
        task_by_name = {}
        for tenant in self.data["M"]:
            for task in tasks_by_tenant[tenant]:
                task_name = task.get("name")
                if task_name is not None:
                    task_by_name[str(task_name)] = (tenant, int(task["task_id"]))

        task_finish_time = {}
        for tx_id, finish_time in simulator_result.get("tx_complete_time", {}).items():
            flow_id = self._simulator_flow_id(tx_id).split("-Q", 1)[0]
            if flow_id in task_by_name:
                key = task_by_name[flow_id]
                task_finish_time[key] = max(task_finish_time.get(key, 0.0), float(finish_time))

        expected_task_keys = {
            (tenant, int(task["task_id"]))
            for tenant in self.data["M"]
            for task in tasks_by_tenant[tenant]
        }
        missing_task_keys = sorted(expected_task_keys - set(task_finish_time))
        if missing_task_keys:
            if self.verbose:
                print(
                    "[mapping-horizon] missing simulator timings for "
                    f"{len(missing_task_keys)} tasks"
                )
            return None

        slot_duration = float(self.data["slot_duration"])
        max_finish_time = max(task_finish_time.values(), default=0.0)
        # Program gaps are enforced with discrete release gates. A simulator finish
        # time rounded to slots can otherwise be one slot too tight per positive gap.
        release_alignment_slots = max(
            (
                sum(
                    1
                    for op in self.data["schedule"].get(tenant, {}).get("collective_program", [])
                    if float(op.get("gap_after", 0.0)) > 0.0
                )
                for tenant in self.data["M"]
            ),
            default=0,
        )
        return int(math.ceil(max_finish_time / slot_duration - 1e-9)) + int(release_alignment_slots)

    def _apply_mapping_horizon_bound(self, mapping, margin_slots=0):
        bound = self._mapping_horizon_bound(mapping)
        if bound is None:
            raise RuntimeError("Could not construct a default-mapping warm-start horizon bound.")
        bound = int(math.ceil(float(bound) + float(margin_slots)))
        if bound <= 0:
            raise RuntimeError(f"Invalid default-mapping warm-start horizon bound: {bound}")

        num_slots = int(self.data["num_slots"])
        if bound >= num_slots:
            raise RuntimeError(
                "Default-mapping warm-start horizon bound is not tighter than the model horizon: "
                f"bound={bound}, num_slots={num_slots}"
            )

        self.T_max.UB = min(float(self.T_max.UB), float(bound))
        for tenant, var in self.tenant_finish.items():
            var.UB = min(float(var.UB), float(bound))
        for (tenant, task_id), finish_expr in self.task_finish.items():
            self.model.addConstr(
                finish_expr <= bound,
                name=f"mapping_horizon_task_{tenant}_{task_id}",
            )
        self.model.update()
        return float(bound)

    def _apply_warm_start_horizon_bound(self, margin_slots=0):
        if self.T_max is None:
            return None
        start_value = self._defined_start_value(self.T_max)
        if start_value is None:
            return None

        bound = int(math.ceil(start_value + float(margin_slots)))
        if bound <= 0:
            return None

        self.T_max.UB = min(float(self.T_max.UB), float(bound))
        for tenant, var in self.tenant_finish.items():
            var.UB = min(float(var.UB), float(bound))
        for (tenant, task_id), finish_expr in self.task_finish.items():
            self.model.addConstr(
                finish_expr <= bound,
                name=f"warm_start_horizon_task_{tenant}_{task_id}",
            )
        self.model.update()
        return float(bound)

    def _copy_full_start_from_solver(self, other):
        for key, var in self.X.items():
            if key in other.X:
                var.Start = other.X[key].X
        for key, var in self.U.items():
            if key in other.U:
                var.Start = other.U[key].X
        for key, var in self.R.items():
            if key in other.R:
                var.Start = other.R[key].X
        for key, var in self.Z.items():
            if key in other.Z:
                var.Start = other.Z[key].X
        for key, var in self.S_active.items():
            if key in other.S_active:
                var.Start = other.S_active[key].X
        for key, var in self.P_active.items():
            if key in other.P_active:
                var.Start = other.P_active[key].X
        for key, var in self.D_full.items():
            if key in other.D_full:
                var.Start = other.D_full[key].X
        for key, var in self.Q_bottleneck.items():
            if key in other.Q_bottleneck:
                var.Start = other.Q_bottleneck[key].X
        for key, var in self.F.items():
            if key in other.F:
                var.Start = other.F[key].X
        for key, var in self.C.items():
            if key in other.C:
                var.Start = other.C[key].X
        for key, var in self.LinkMinRate.items():
            if key in other.LinkMinRate:
                var.Start = other.LinkMinRate[key].X
        for key, var in self.tenant_finish.items():
            if key in other.tenant_finish:
                var.Start = other.tenant_finish[key].X
        if self.T_max is not None and other.T_max is not None:
            self.T_max.Start = other.T_max.X
        self.model.update()

    def _maybe_seed_from_fixed_mapping_subproblem(self, time_limit):
        return

    def _maybe_seed_from_external_warm_start(self, time_limit):
        del time_limit
        return

    def _maybe_seed_full_mip_start(self, time_limit):
        if not self.enable_full_mip_start:
            return
        if self.warm_start_mapping is None:
            return
        if time_limit is not None and float(time_limit) <= 5.0:
            return

        seed_budget = 15.0 if time_limit is None else min(15.0, max(float(time_limit) * 0.5, 5.0))
        fixed_solver = MappingMILPSolver(
            self.datacenter,
            self.tenant_mapping,
            tenant_flows=self.tenant_flows,
            verbose=False,
            name=f"{self.model_name}_seed",
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
            compact_task_windows=self.compact_task_windows,
            compact_window_mapping=self.compact_window_mapping,
            lp_method=self.lp_method,
            node_method=self.node_method,
            presolve=self.presolve,
            prepasses=self.prepasses,
            numeric_focus=self.numeric_focus,
            enable_heuristic_warm_start=False,
            enable_full_mip_start=False,
            path_table=self.path_table,
        )
        for (tenant, rank, server), var in fixed_solver.X.items():
            target = 1.0 if self.warm_start_mapping.get(tenant, {}).get(rank) == server else 0.0
            fixed_solver.model.addConstr(var == target)

        try:
            fixed_solver.solve(time_limit=seed_budget)
        except Exception:
            return

        seed_values = {
            var.VarName: var.X
            for var in fixed_solver.model.getVars()
        }
        for var in self.model.getVars():
            if var.VarName in seed_values:
                var.Start = seed_values[var.VarName]
        self.model.update()

    def solve(self, time_limit=None):
        self.model.update()
        if time_limit is not None:
            self.model.Params.TimeLimit = float(time_limit)

        self._maybe_seed_from_fixed_mapping_subproblem(time_limit)
        self._maybe_seed_from_external_warm_start(time_limit)
        self._maybe_seed_full_mip_start(time_limit)
        self.model.Params.OptimalityTol = 1e-9
        self.model.Params.MIPGap = 0.0
        self.model.Params.MIPGapAbs = 1e-9
        self.model.Params.IntegralityFocus = 1
        if self.lp_method is not None:
            self.model.Params.Method = int(self.lp_method)
        if self.node_method is not None:
            self.model.Params.NodeMethod = int(self.node_method)
        if self.presolve is not None:
            self.model.Params.Presolve = int(self.presolve)
        if self.prepasses is not None:
            self.model.Params.PrePasses = int(self.prepasses)
        if self.numeric_focus is not None:
            self.model.Params.NumericFocus = int(self.numeric_focus)
        self.model.optimize()

        status = self.model.Status
        if self.verbose:
            print("\n=== GUROBI STATUS REPORT ===")
            print("Status =", status)
            print(
                "StatusStr =",
                {
                    GRB.OPTIMAL: "OPTIMAL",
                    GRB.SUBOPTIMAL: "SUBOPTIMAL",
                    GRB.INFEASIBLE: "INFEASIBLE",
                    GRB.UNBOUNDED: "UNBOUNDED",
                    GRB.TIME_LIMIT: "TIME_LIMIT",
                }.get(status, "OTHER"),
            )
            print("SolCount =", self.model.SolCount)
            print("ObjVal =", getattr(self.model, "ObjVal", None))
            print("ObjBound =", getattr(self.model, "ObjBound", None))
            print("MIPGap =", getattr(self.model, "MIPGap", None))
            print("Runtime =", self.model.Runtime)

        if self.model.SolCount == 0 or status in (GRB.INFEASIBLE, GRB.UNBOUNDED):
            raise RuntimeError(
                f"Model not solved to a valid solution. Status={status}, SolCount={self.model.SolCount}"
            )
        self.final_mapping = self.get_X_mapping()
        slot_duration = float(self.data["slot_duration"])
        tenant_finishes = [self.tenant_finish[tenant].X for tenant in self.data["M"]]
        self.final_makespan = (max(tenant_finishes) * slot_duration) if tenant_finishes else 0.0
        self.final_avg_jct = (
            sum(tenant_finishes) / len(tenant_finishes) * slot_duration
            if tenant_finishes
            else 0.0
        )
        self.final_obj = self.final_makespan
        if self.verbose and self.warm_start_mapping is not None:
            print(
                f"Warm-start mapping seeded in {self.warm_start_runtime:.2f}s: "
                f"{self.warm_start_mapping}"
            )
        return self

    def get_X_mapping(self, thr=0.5):
        if self.final_mapping is not None:
            return {
                tenant: dict(rank_to_server)
                for tenant, rank_to_server in self.final_mapping.items()
            }
        tenants, ranks, servers = self.data["M"], self.data["R"], self.data["S"]
        mapping = {tenant: {} for tenant in tenants}
        for tenant in tenants:
            for rank in ranks[tenant]:
                for server in servers[tenant]:
                    if self.X[(tenant, rank, server)].X > thr:
                        mapping[tenant][rank] = server
                        break

        if self.verbose:
            print("X mapping:", mapping)
        return mapping


MappingMILPSolver = MappingILPSolver
LegacyTimeSlotMappingILPSolver = MappingMILPSolver
