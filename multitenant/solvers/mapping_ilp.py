from __future__ import annotations

from collections import defaultdict
import itertools
import math
import time

import gurobipy as gp
from gurobipy import GRB

from multitenant.simulator import simulate_collective
from multitenant.workloads import build_collective_schedule


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
        stage_flows=None,
        fairness_lambda=1.0,
        fairness_iterations=2,
        fairness_grouping="phase",
        slot_duration=None,
        horizon_slots=None,
        enable_heuristic_warm_start=True,
        enable_full_mip_start=True,
    ):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.collective = collective
        self.single_flow_size = single_flow_size
        self.tenant_collective_specs = tenant_collective_specs
        self.stage_flows = stage_flows
        self.verbose = verbose
        self.model_name = name
        self.slot_duration_override = slot_duration
        self.horizon_slots_override = horizon_slots
        self.volume_scale = (
            1e3
            if tenant_collective_specs is not None
            or (
                collective in {"allgather", "reducescatter", "allreduce", "alltoall"}
                and single_flow_size is not None
            )
            else 1e9
        )

        # Kept for API compatibility with earlier solver variants.
        self.fairness_lambda = float(fairness_lambda)
        self.fairness_iterations = max(1, int(fairness_iterations))
        self.fairness_grouping = fairness_grouping
        self.enable_heuristic_warm_start = bool(enable_heuristic_warm_start)
        self.enable_full_mip_start = bool(enable_full_mip_start)

        self.data = None
        self.model = None

        self.X = {}
        self.U = {}
        self.W = {}
        self.Y = {}
        self.N = {}
        self.LinkLevel = {}
        self.LinkMinRate = {}
        self.R = {}
        self.Z = {}
        self.S_active = {}
        self.G_start = {}
        self.D_full = {}
        self.Q_bottleneck = {}
        self.F = {}
        self.C = {}
        self.H = {}
        self.task_finish = {}
        self.tenant_finish = {}
        self.T_m = {}
        self.T_stage = {}
        self.T_max = None
        self.final_obj = None
        self.final_makespan = None
        self.final_avg_jct = None
        self.final_mapping = None
        self.warm_start_mapping = None
        self.warm_start_runtime = 0.0

        self._build(name=name)

    def _derive_tasks(self):
        if self.tenant_collective_specs is not None or (
            self.collective in {"allgather", "reducescatter", "allreduce", "alltoall"}
            and self.single_flow_size is not None
        ):
            schedule = build_collective_schedule(
                self.tenant_mapping,
                None if self.single_flow_size is None else int(self.single_flow_size),
                self.collective,
                scale=self.volume_scale,
                tenant_collective_specs=self.tenant_collective_specs,
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
        min_task_volume = min(
            (
                float(task["V"])
                for tenant in tenants
                for task in tasks_by_tenant[tenant]
                if float(task["V"]) > 0.0
            ),
            default=1.0,
        )
        total_task_volume = sum(
            float(task["V"])
            for tenant in tenants
            for task in tasks_by_tenant[tenant]
        )
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
                        edges = self.datacenter.ECMP_edge_set.get((src_server, dst_server))
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

        if self.slot_duration_override is not None:
            slot_duration = float(self.slot_duration_override)
        else:
            refinement_factor = 1
            if (
                (self.tenant_collective_specs is not None or self.collective in {"allgather", "reducescatter", "allreduce", "alltoall"})
                and total_task_count <= 64
            ):
                refinement_factor = 2
            isolated_task_time = (min_task_volume / min_capacity) / refinement_factor
            coarse_target_time = total_task_volume / min_capacity / max(target_horizon_slots, 1)
            slot_duration = max(isolated_task_time, coarse_target_time / refinement_factor, 1e-6)

        if self.horizon_slots_override is not None:
            horizon_slots = int(self.horizon_slots_override)
        else:
            work_conserving_slots = int(math.ceil(total_task_volume / min_capacity / slot_duration))
            extra_dependency_slack = 2 if total_task_count > 64 else 4
            horizon_slots = max(dependency_lower_bound + extra_dependency_slack, work_conserving_slots + 2, 4)
            horizon_slots = max(horizon_slots, 4)

        task_total_volume = {
            (tenant, task["task_id"]): float(task["V"])
            for tenant in tenants
            for task in tasks_by_tenant[tenant]
        }
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
            "path_edges": self.datacenter.ECMP_edge_set,
            "slot_duration": slot_duration,
            "Horizon": list(range(horizon_slots)),
            "num_slots": horizon_slots,
            "max_task_count": max((len(tasks_by_tenant[tenant]) for tenant in tenants), default=0),
            "task_total_volume": task_total_volume,
            "min_send_unit": min_send_unit,
            "sender_tasks": sender_task_ids,
        }

    def _build(self, name="mapping_ilp"):
        if self.model is not None:
            self.model.dispose()

        self.X = {}
        self.U = {}
        self.W = {}
        self.Y = {}
        self.N = {}
        self.LinkLevel = {}
        self.LinkMinRate = {}
        self.R = {}
        self.Z = {}
        self.S_active = {}
        self.G_start = {}
        self.D_full = {}
        self.Q_bottleneck = {}
        self.F = {}
        self.C = {}
        self.H = {}
        self.task_finish = {}
        self.tenant_finish = {}
        self.T_m = {}
        self.T_stage = {}
        self.T_max = None

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
            for rank in ranks[tenant]:
                for server in servers[tenant]:
                    self.X[(tenant, rank, server)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"X_{tenant}_{rank}_{server}",
                    )

    def _add_perm_constraints(self):
        tenants, ranks, servers = self.data["M"], self.data["R"], self.data["S"]

        for tenant in tenants:
            for rank in ranks[tenant]:
                self.model.addConstr(
                    gp.quicksum(self.X[(tenant, rank, server)] for server in servers[tenant]) == 1,
                    name=f"rank_one_{tenant}_{rank}",
                )

        for tenant in tenants:
            for server in servers[tenant]:
                self.model.addConstr(
                    gp.quicksum(self.X[(tenant, rank, server)] for rank in ranks[tenant]) == 1,
                    name=f"srv_one_{tenant}_{server}",
                )

    def _add_U_and_endpoint_constraints(self):
        tenants, servers = self.data["M"], self.data["S"]
        tasks_by_tenant = self.data["tasks"]

        for tenant in tenants:
            candidate_servers = servers[tenant]
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
                        self.U[key] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"U_{tenant}_{task_id}_{src_server}_{dst_server}",
                        )
                        flow_vars.append(self.U[key])

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
        horizon = self.data["Horizon"]
        num_slots = self.data["num_slots"]
        slot_duration = self.data["slot_duration"]
        min_send_unit = self.data["min_send_unit"]
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

        for tenant in tenants:
            candidate_servers = servers[tenant]
            for task in tasks_by_tenant[tenant]:
                task_id = task["task_id"]
                volume = float(task["V"])
                for src_server in candidate_servers:
                    for dst_server in candidate_servers:
                        if src_server == dst_server:
                            continue

                        u_var = self.U[(tenant, task_id, src_server, dst_server)]
                        edges = path_edges.get((src_server, dst_server))
                        if edges is None:
                            continue

                        for link in edges:
                            if link not in capacities:
                                continue
                            required_load_terms[tenant][task_id][link].append(volume * u_var)
                            usage_u_vars[tenant][task_id][link].append(u_var)

        task_lookup = {
            tenant: {task["task_id"]: task for task in tasks_by_tenant[tenant]}
            for tenant in tenants
        }
        for tenant in tenants:
            task_ids = [task["task_id"] for task in tasks_by_tenant[tenant]]
            for task_id in task_ids:
                task = task_lookup[tenant][task_id]
                preds = list(task["preds"])
                for t in horizon:
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
                    self.G_start[(tenant, task_id, t)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"G_start_{tenant}_{task_id}_{t}",
                    )
                    self.D_full[(tenant, task_id, t)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"D_full_{tenant}_{task_id}_{t}",
                    )
                    self.C[(tenant, task_id, t)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"C_{tenant}_{task_id}_{t}",
                    )

                for link in links:
                    u_vars = usage_u_vars[tenant][task_id][link]
                    if not u_vars:
                        continue

                    self.W[(tenant, task_id, link)] = gp.quicksum(u_vars)

                    load_expr = gp.quicksum(required_load_terms[tenant][task_id][link])
                    big_m = self.data["task_total_volume"][(tenant, task_id)]

                    for t in horizon:
                        self.Y[(tenant, task_id, link, t)] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"Y_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
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
                            self.Y[(tenant, task_id, link, t)] <= self.W[(tenant, task_id, link)],
                            name=f"Y_le_W_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.Y[(tenant, task_id, link, t)] <= self.S_active[(tenant, task_id, t)],
                            name=f"Y_le_S_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.Y[(tenant, task_id, link, t)]
                            >= self.W[(tenant, task_id, link)] + self.S_active[(tenant, task_id, t)] - 1,
                            name=f"Y_ge_WS_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.Q_bottleneck[(tenant, task_id, link, t)]
                            <= self.Y[(tenant, task_id, link, t)],
                            name=f"Q_le_Y_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.Q_bottleneck[(tenant, task_id, link, t)]
                            <= self.D_full[(tenant, task_id, t)],
                            name=f"Q_le_D_full_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
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
                            - self.data["task_total_volume"][(tenant, task_id)] * (1 - self.W[(tenant, task_id, link)])
                            - self.data["task_total_volume"][(tenant, task_id)] * (1 - self.S_active[(tenant, task_id, t)]),
                            name=f"F_ge_R_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )

                    self.model.addConstr(
                        gp.quicksum(self.F[(tenant, task_id, link, t)] for t in horizon) == load_expr,
                        name=f"load_balance_{tenant}_{task_id}_{link[0]}_{link[1]}",
                    )
                    cumulative = gp.LinExpr()
                    for t in horizon:
                        cumulative += self.F[(tenant, task_id, link, t)]
                        self.model.addConstr(
                            cumulative >= load_expr - big_m * (1 - self.C[(tenant, task_id, t)]),
                            name=f"complete_if_sent_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )

                for t in horizon:
                    self.model.addConstr(
                        gp.quicksum(
                            self.Q_bottleneck[(tenant, task_id, link, t)]
                            for link in links
                            if (tenant, task_id, link, t) in self.Q_bottleneck
                        )
                        == self.D_full[(tenant, task_id, t)],
                        name=f"Q_one_bottleneck_{tenant}_{task_id}_{t}",
                    )

                for t in range(num_slots - 1):
                    self.model.addConstr(
                        self.C[(tenant, task_id, t)] <= self.C[(tenant, task_id, t + 1)],
                        name=f"C_mono_{tenant}_{task_id}_{t}",
                    )

                self.model.addConstr(
                    self.C[(tenant, task_id, num_slots - 1)] == 1,
                    name=f"C_terminal_{tenant}_{task_id}",
                )

                if preds:
                    self.model.addConstr(
                        self.Z[(tenant, task_id, 0)] == 0,
                        name=f"Z_wait_preds_{tenant}_{task_id}",
                    )
                else:
                    self.model.addConstr(
                        self.Z[(tenant, task_id, 0)] == 1,
                        name=f"Z_ready_init_{tenant}_{task_id}",
                    )
                self.model.addConstr(
                    self.S_active[(tenant, task_id, 0)] <= self.Z[(tenant, task_id, 0)],
                    name=f"S_init_le_Z_{tenant}_{task_id}",
                )
                self.model.addConstr(
                    self.G_start[(tenant, task_id, 0)] == self.S_active[(tenant, task_id, 0)],
                    name=f"G_start_init_{tenant}_{task_id}",
                )

                for t in range(1, num_slots):
                    self.model.addConstr(
                        self.Z[(tenant, task_id, t)] <= 1 - self.C[(tenant, task_id, t - 1)],
                        name=f"Z_after_completion_{tenant}_{task_id}_{t}",
                    )
                    if preds:
                        for pred_task_id in preds:
                            self.model.addConstr(
                                self.Z[(tenant, task_id, t)] <= self.C[(tenant, pred_task_id, t - 1)],
                                name=f"pred_ready_{tenant}_{task_id}_{pred_task_id}_{t}",
                            )
                        self.model.addConstr(
                            self.Z[(tenant, task_id, t)]
                            >= 1
                            - self.C[(tenant, task_id, t - 1)]
                            + gp.quicksum(self.C[(tenant, pred_task_id, t - 1)] for pred_task_id in preds)
                            - len(preds),
                            name=f"Z_ready_exact_{tenant}_{task_id}_{t}",
                        )
                    else:
                        self.model.addConstr(
                            self.Z[(tenant, task_id, t)] == 1 - self.C[(tenant, task_id, t - 1)],
                            name=f"Z_no_pred_exact_{tenant}_{task_id}_{t}",
                        )

                    self.model.addConstr(
                        self.S_active[(tenant, task_id, t)] <= self.Z[(tenant, task_id, t)],
                        name=f"S_le_Z_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(
                        self.S_active[(tenant, task_id, t)]
                        >= self.S_active[(tenant, task_id, t - 1)] - self.C[(tenant, task_id, t - 1)],
                        name=f"S_nonpreempt_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(
                        self.G_start[(tenant, task_id, t)] <= self.S_active[(tenant, task_id, t)],
                        name=f"G_start_le_S_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(
                        self.G_start[(tenant, task_id, t)] <= 1 - self.S_active[(tenant, task_id, t - 1)],
                        name=f"G_start_le_not_prev_S_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(
                        self.G_start[(tenant, task_id, t)]
                        >= self.S_active[(tenant, task_id, t)] - self.S_active[(tenant, task_id, t - 1)],
                        name=f"G_start_ge_S_rise_{tenant}_{task_id}_{t}",
                    )

                for t in horizon:
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

                self.model.addConstr(
                    gp.quicksum(self.G_start[(tenant, task_id, t)] for t in horizon) <= 1,
                    name=f"G_start_once_{tenant}_{task_id}",
                )

                for t in horizon:
                    self.model.addConstr(
                        self.R[(tenant, task_id, t)]
                        <= self.data["task_total_volume"][(tenant, task_id)] * self.S_active[(tenant, task_id, t)],
                        name=f"R_gate_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(
                        self.R[(tenant, task_id, t)]
                        >= min_send_unit * self.S_active[(tenant, task_id, t)],
                        name=f"R_active_lb_{tenant}_{task_id}_{t}",
                    )

                self.task_finish[(tenant, task_id)] = 1 + gp.quicksum(
                    1 - self.C[(tenant, task_id, t)] for t in horizon
                )
                self.T_stage[(tenant, task_id)] = self.task_finish[(tenant, task_id)]

            for sender_rank, sender_task_ids in self.data["sender_tasks"][tenant].items():
                if not sender_task_ids:
                    continue
                sender_task_ids = list(sender_task_ids)
                for t in horizon:
                    sender_active_sum = gp.quicksum(
                        self.S_active[(tenant, task_id, t)] for task_id in sender_task_ids
                    )
                    for task_id in sender_task_ids:
                        self.model.addConstr(
                            sender_active_sum >= self.Z[(tenant, task_id, t)],
                            name=f"sender_service_lb_{tenant}_{sender_rank}_{task_id}_{t}",
                        )
                    self.model.addConstr(
                        sender_active_sum <= 1,
                        name=f"sender_service_ub_{tenant}_{sender_rank}_{t}",
                    )
                    for idx, task_id in enumerate(sender_task_ids):
                        for earlier_task_id in sender_task_ids[:idx]:
                            self.model.addConstr(
                                self.G_start[(tenant, task_id, t)]
                                <= 1 - self.Z[(tenant, earlier_task_id, t)],
                                name=f"sender_fifo_start_{tenant}_{sender_rank}_{earlier_task_id}_{task_id}_{t}",
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
            self.T_m[tenant] = self.tenant_finish[tenant]

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
                    name=f"Rmin_{link[0]}_{link[1]}_{t}",
                )
                active_terms = [
                    self.Y[(tenant, task["task_id"], link, t)]
                    for tenant in tenants
                    for task in tasks_by_tenant[tenant]
                    if (tenant, task["task_id"], link, t) in self.Y
                ]
                active_count_expr = gp.quicksum(active_terms)

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
                        y_key = (tenant, task_id, link, t)
                        if y_key in self.Y:
                            self.model.addConstr(
                                self.R[(tenant, task_id, t)]
                                >= self.LinkMinRate[(link, t)]
                                - max_rate_big_m * (1 - self.Y[y_key]),
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
                            <= self.LinkMinRate[(link, t)] + max_rate_big_m * (1 - q_var),
                            name=f"bn_minrate_eq_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )

                self.model.addConstr(
                    gp.quicksum(
                        self.Q_bottleneck[(tenant, task["task_id"], link, t)]
                        for tenant in tenants
                        for task in tasks_by_tenant[tenant]
                        if (tenant, task["task_id"], link, t) in self.Q_bottleneck
                    )
                    <= active_count_expr,
                    name=f"bn_count_le_active_{link[0]}_{link[1]}_{t}",
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
        self.model.setObjectiveN(self.T_max, index=0, priority=2, name="makespan")
        self.model.setObjectiveN(avg_completion, index=1, priority=1, name="avg_jct")

    def _apply_mapping_warm_start(self, mapping):
        if not mapping:
            return
        self.warm_start_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in mapping.items()
        }
        for (tenant, rank, server), var in self.X.items():
            var.Start = 1.0 if mapping.get(tenant, {}).get(rank) == server else 0.0
        self.model.update()

    def _maybe_seed_from_heuristic(self, time_limit):
        if not self.enable_heuristic_warm_start:
            return
        if self.warm_start_mapping is not None:
            return
        if self.tenant_collective_specs is None and not (
            self.collective in {"allgather", "reducescatter", "allreduce", "alltoall"}
            and self.single_flow_size is not None
        ):
            return

        heuristic_budget = 10.0
        if time_limit is not None:
            heuristic_budget = min(15.0, max(float(time_limit) * 0.25, 10.0), float(time_limit))

        start_time = time.time()
        heuristic = MappingHeuristicSolver(
            self.datacenter,
            self.tenant_mapping,
            tenant_flows=self.tenant_flows,
            verbose=False,
            name=f"{self.model_name}_warm",
            collective=self.collective,
            single_flow_size=self.single_flow_size,
            tenant_collective_specs=self.tenant_collective_specs,
            stage_flows=self.stage_flows,
            fairness_lambda=self.fairness_lambda,
            fairness_iterations=self.fairness_iterations,
            fairness_grouping=self.fairness_grouping,
            slot_duration=self.slot_duration_override,
            horizon_slots=self.horizon_slots_override,
        )
        heuristic.solve(time_limit=heuristic_budget)
        self.warm_start_runtime = time.time() - start_time
        self._apply_mapping_warm_start(heuristic.get_X_mapping())

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
            stage_flows=self.stage_flows,
            fairness_lambda=self.fairness_lambda,
            fairness_iterations=self.fairness_iterations,
            fairness_grouping=self.fairness_grouping,
            slot_duration=self.slot_duration_override,
            horizon_slots=self.horizon_slots_override,
            enable_heuristic_warm_start=False,
            enable_full_mip_start=False,
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

        self._maybe_seed_from_heuristic(time_limit)
        self._maybe_seed_full_mip_start(time_limit)
        self.model.Params.FeasibilityTol = 1e-9
        self.model.Params.OptimalityTol = 1e-9
        self.model.Params.MIPGap = 0.0
        self.model.Params.MIPGapAbs = 1e-9
        self.model.Params.IntegralityFocus = 1
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


class MappingHeuristicSolver:
    """Pure-mapping solver with deterministic collective execution evaluation."""

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
        stage_flows=None,
        fairness_lambda=1.0,
        fairness_iterations=2,
        fairness_grouping="phase",
        slot_duration=None,
        horizon_slots=None,
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
        self.stage_flows = stage_flows
        self.verbose = verbose
        self.model_name = name
        self.slot_duration_override = slot_duration
        self.horizon_slots_override = horizon_slots
        self.fairness_lambda = float(fairness_lambda)
        self.fairness_iterations = max(1, int(fairness_iterations))
        self.fairness_grouping = fairness_grouping

        self.model = None
        self.X = {}
        self.final_obj = None
        self.final_makespan = None
        self.final_mapping = None
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
        self._surrogate_candidate_limit = 8
        self._surrogate_candidates: list[tuple[tuple[float, float], tuple[tuple[int, tuple[int, ...]], ...], dict[int, dict[int, int]]]] = []
        self.tenant_pressure: dict[int, float] = {}
        self.tenant_peak_load: dict[int, float] = {}
        self.link_price_beta = 4.0
        self.link_price_gamma = 2.0
        self.max_price_rounds = 6
        self.data = self._build_data()

    def _collective_mode(self) -> bool:
        return self.tenant_collective_specs is not None or (
            self.collective in {"allgather", "reducescatter", "allreduce", "alltoall"}
            and self.single_flow_size is not None
        )

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
            schedule = build_collective_schedule(
                self.initial_tenant_mapping,
                None if self.single_flow_size is None else int(self.single_flow_size),
                self.collective,
                scale=1e3,
                tenant_collective_specs=self.tenant_collective_specs,
            )
            tasks = {
                tenant: list(schedule.get(tenant, {}).get("tasks", []))
                for tenant in self.tenants
            }
            edge_capacity = {
                (int(src), int(dst)): float(attrs["capacity"])
                for src, dst, attrs in self.datacenter.topology.edges(data=True)
            }
            path_edges = {}
            for tenant in self.tenants:
                for src_server in servers[tenant]:
                    for dst_server in servers[tenant]:
                        if src_server == dst_server:
                            continue
                        path_edges[(int(src_server), int(dst_server))] = tuple(
                            tuple(int(node) for node in edge)
                            for edge in self.datacenter.ECMP_table[(int(src_server), int(dst_server))]
                        )

            compiled_schedule = self._compile_epoch_schedule(schedule)
            for tenant in self.tenants:
                aggregate_pressure = 0.0
                peak_load = 0.0
                for epoch_flows in compiled_schedule["per_tenant"][tenant]["epoch_flows"].values():
                    epoch_edge_loads: dict[tuple[int, int], float] = defaultdict(float)
                    for src_rank, dst_rank, volume in epoch_flows:
                        src_server = int(self.initial_tenant_mapping[tenant][src_rank])
                        dst_server = int(self.initial_tenant_mapping[tenant][dst_rank])
                        for edge in path_edges[(src_server, dst_server)]:
                            epoch_edge_loads[edge] += float(volume) / edge_capacity[edge]
                    aggregate_pressure += sum(epoch_edge_loads.values())
                    peak_load = max(peak_load, max(epoch_edge_loads.values(), default=0.0))
                self.tenant_pressure[tenant] = float(aggregate_pressure)
                self.tenant_peak_load[tenant] = float(peak_load)
            return {
                "M": list(self.tenants),
                "R": ranks,
                "S": servers,
                "tasks": tasks,
                "schedule": schedule,
                "compiled_schedule": compiled_schedule,
                "edge_capacity": edge_capacity,
                "path_edges": path_edges,
            }

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
        cand_max, cand_avg = candidate
        inc_max, inc_avg = incumbent
        if cand_max < inc_max - tol:
            return True
        if cand_max > inc_max + tol:
            return False
        return cand_avg < inc_avg - tol

    @staticmethod
    def _lower_bound_can_improve(lower_bound, incumbent, tol=1e-12):
        lb_max, lb_avg = lower_bound
        inc_max, inc_avg = incumbent
        if lb_max > inc_max + tol:
            return False
        if lb_max < inc_max - tol:
            return True
        return lb_avg < inc_avg - tol

    def _mapping_signature(self, mapping):
        return tuple(
            (tenant, tuple(mapping[tenant][rank] for rank in self.rank_orders[tenant]))
            for tenant in self.tenants
        )

    @staticmethod
    def _task_levels(tasks, edges):
        task_ids = [int(task["task_id"]) for task in tasks]
        preds_by_task: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
        for edge in edges:
            preds_by_task[int(edge["dst_task_id"])].append(int(edge["src_task_id"]))

        memo: dict[int, int] = {}

        def level(task_id: int) -> int:
            if task_id in memo:
                return memo[task_id]
            preds = preds_by_task[task_id]
            if not preds:
                memo[task_id] = 0
            else:
                memo[task_id] = 1 + max(level(pred_task_id) for pred_task_id in preds)
            return memo[task_id]

        return {task_id: level(task_id) for task_id in task_ids}

    def _compile_epoch_schedule(self, schedule):
        compiled: dict[int, dict[str, object]] = {}
        global_max_epoch = -1
        for tenant in self.tenants:
            tenant_schedule = schedule.get(tenant, {})
            tasks = list(tenant_schedule.get("tasks", []))
            edges = list(tenant_schedule.get("edges", []))
            levels = self._task_levels(tasks, edges)
            epoch_pair_volume: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
            rank_weight: dict[int, float] = defaultdict(float)
            rank_incidence: dict[int, list[tuple[int, int, int, float]]] = defaultdict(list)

            for task in tasks:
                task_id = int(task["task_id"])
                epoch = int(levels[task_id])
                src_rank = int(task["src_rank"])
                dst_rank = int(task["dst_rank"])
                volume = float(task["V"])
                pair = (src_rank, dst_rank)
                epoch_pair_volume[epoch][pair] = epoch_pair_volume[epoch].get(pair, 0.0) + volume
                rank_weight[src_rank] += volume
                rank_weight[dst_rank] += volume
                rank_incidence[src_rank].append((epoch, src_rank, dst_rank, volume))
                rank_incidence[dst_rank].append((epoch, src_rank, dst_rank, volume))

            epoch_flows = {
                int(epoch): [
                    (int(src_rank), int(dst_rank), float(volume))
                    for (src_rank, dst_rank), volume in sorted(pair_volume.items())
                ]
                for epoch, pair_volume in epoch_pair_volume.items()
            }
            max_epoch = max(epoch_flows.keys(), default=-1)
            global_max_epoch = max(global_max_epoch, max_epoch)
            branch_order = sorted(
                self.rank_orders[tenant],
                key=lambda rank: (-rank_weight.get(rank, 0.0), int(rank)),
            )
            compiled[tenant] = {
                "task_levels": levels,
                "epoch_flows": epoch_flows,
                "all_flows": [
                    (int(epoch), int(src_rank), int(dst_rank), float(volume))
                    for epoch, flows in epoch_flows.items()
                    for src_rank, dst_rank, volume in flows
                ],
                "max_epoch": int(max_epoch),
                "branch_order": branch_order,
                "rank_incidence": {
                    int(rank): list(entries)
                    for rank, entries in rank_incidence.items()
                },
            }

        return {
            "per_tenant": compiled,
            "global_max_epoch": int(global_max_epoch),
        }

    def _register_surrogate_candidate(self, mapping, score):
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
        filtered.sort(key=lambda entry: (entry[0][0], entry[0][1]))
        self._surrogate_candidates = filtered[: self._surrogate_candidate_limit]

    def _compute_epoch_link_load_state(self, mapping):
        compiled_schedule = self.data["compiled_schedule"]["per_tenant"]
        global_max_epoch = int(self.data["compiled_schedule"]["global_max_epoch"])
        edge_capacity = self.data["edge_capacity"]
        path_edges = self.data["path_edges"]

        epoch_loads: list[dict[tuple[int, int], float]] = [defaultdict(float) for _ in range(global_max_epoch + 1)]
        epoch_maxima = [0.0 for _ in range(global_max_epoch + 1)]

        for epoch in range(global_max_epoch + 1):
            normalized_link_load = epoch_loads[epoch]
            for tenant in self.tenants:
                tenant_epochs = compiled_schedule[tenant]["epoch_flows"]
                for src_rank, dst_rank, volume in tenant_epochs.get(epoch, []):
                    src_server = int(mapping[tenant][src_rank])
                    dst_server = int(mapping[tenant][dst_rank])
                    for edge in path_edges[(src_server, dst_server)]:
                        normalized_link_load[edge] += float(volume) / edge_capacity[edge]
            epoch_maxima[epoch] = max(normalized_link_load.values(), default=0.0)

        return epoch_loads, epoch_maxima

    def _edge_price_value(self, edge, normalized_load):
        capacity = self.data["edge_capacity"][edge]
        base_cost = 1.0 / max(capacity, 1e-12)
        return base_cost * (1.0 + self.link_price_beta * (float(normalized_load) ** self.link_price_gamma))

    def _compute_epoch_link_prices(self, mapping):
        epoch_loads, epoch_maxima = self._compute_epoch_link_load_state(mapping)
        epoch_prices: list[dict[tuple[int, int], float]] = []
        for epoch_load in epoch_loads:
            edge_prices = {}
            for edge, normalized_load in epoch_load.items():
                edge_prices[edge] = self._edge_price_value(edge, normalized_load)
            epoch_prices.append(edge_prices)
        return epoch_loads, epoch_maxima, epoch_prices

    def _pair_epoch_price_lookup(self, candidate_servers, epoch_prices):
        path_edges = self.data["path_edges"]
        lookup: dict[tuple[int, int, int], float] = {}
        for epoch, edge_prices in enumerate(epoch_prices):
            for src_server in candidate_servers:
                for dst_server in candidate_servers:
                    if int(src_server) == int(dst_server):
                        continue
                    path_cost = 0.0
                    for edge in path_edges[(int(src_server), int(dst_server))]:
                        path_cost += edge_prices.get(edge, self._edge_price_value(edge, 0.0))
                    lookup[(epoch, int(src_server), int(dst_server))] = path_cost
        return lookup

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
                tenant_finish.append(epoch_prefix[max_epoch])

        return (
            float(max(tenant_finish, default=0.0)),
            float(sum(tenant_finish) / max(len(tenant_finish), 1)),
        )

    def _evaluate_surrogate_mapping(self, mapping):
        signature = self._mapping_signature(mapping)
        cached = self._surrogate_cache.get(signature)
        if cached is not None:
            return cached

        if not self._collective_mode():
            raise ValueError(
                "Structured pure mapping solver currently requires collective inputs via "
                "tenant_collective_specs or collective/single_flow_size."
            )

        compiled_schedule = self.data["compiled_schedule"]["per_tenant"]
        global_max_epoch = int(self.data["compiled_schedule"]["global_max_epoch"])
        edge_capacity = self.data["edge_capacity"]
        path_edges = self.data["path_edges"]

        del global_max_epoch, edge_capacity, path_edges
        _, epoch_deltas = self._compute_epoch_link_load_state(mapping)

        score = self._surrogate_score_from_epoch_max(epoch_deltas)
        self._surrogate_cache[signature] = score
        self._register_surrogate_candidate(mapping, score)
        return score

    def _evaluate_mapping(self, mapping):
        signature = self._mapping_signature(mapping)
        cached = self._score_cache.get(signature)
        if cached is not None:
            return cached

        if not self._collective_mode():
            raise ValueError(
                "Pure mapping solver currently requires collective inputs via "
                "tenant_collective_specs or collective/single_flow_size."
            )

        makespan, avg_jct = simulate_collective(
            self.datacenter.topology,
            mapping,
            self.datacenter.paths,
            None if self.single_flow_size is None else int(self.single_flow_size),
            self.collective,
            tenant_collective_specs=self.tenant_collective_specs,
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
            signature = self._mapping_signature(mapping)
            if signature in seen:
                return
            seen.add(signature)
            seeds.append(mapping)

        add_seed(self.initial_tenant_mapping)

        for tenant in self.tenants:
            ranks = self.rank_orders[tenant]
            current_servers = [self.initial_tenant_mapping[tenant][rank] for rank in ranks]
            if len(current_servers) <= 1:
                continue

            add_seed(self._apply_server_order(self.initial_tenant_mapping, tenant, tuple(reversed(current_servers))))

            for shift in range(1, len(current_servers)):
                rotated = current_servers[shift:] + current_servers[:shift]
                add_seed(self._apply_server_order(self.initial_tenant_mapping, tenant, tuple(rotated)))

        return seeds

    def _optimize_tenant_block(self, base_mapping, tenant, incumbent_score, deadline):
        ranks = self.rank_orders[tenant]
        current_servers = tuple(base_mapping[tenant][rank] for rank in ranks)
        if len(current_servers) <= 1:
            return base_mapping, incumbent_score
        return self._branch_and_bound_tenant_block(
            base_mapping,
            tenant,
            incumbent_score,
            deadline,
        )

    def _branch_and_bound_tenant_block(self, base_mapping, tenant, incumbent_score, deadline):
        compiled_tenant = self.data["compiled_schedule"]["per_tenant"][tenant]
        global_max_epoch = int(self.data["compiled_schedule"]["global_max_epoch"])
        edge_capacity = self.data["edge_capacity"]
        path_edges = self.data["path_edges"]
        rank_incidence = compiled_tenant["rank_incidence"]
        branch_ranks = [
            int(rank)
            for rank in compiled_tenant["branch_order"]
            if rank in self.rank_orders[tenant]
        ]
        available_servers = tuple(base_mapping[tenant][rank] for rank in self.rank_orders[tenant])

        epoch_loads: list[dict[tuple[int, int], float]] = [defaultdict(float) for _ in range(global_max_epoch + 1)]
        epoch_maxima = [0.0 for _ in range(global_max_epoch + 1)]

        for other_tenant in self.tenants:
            if other_tenant == tenant:
                continue
            for epoch, flows in self.data["compiled_schedule"]["per_tenant"][other_tenant]["epoch_flows"].items():
                for src_rank, dst_rank, volume in flows:
                    src_server = int(base_mapping[other_tenant][src_rank])
                    dst_server = int(base_mapping[other_tenant][dst_rank])
                    for edge in path_edges[(src_server, dst_server)]:
                        epoch_loads[epoch][edge] += float(volume) / edge_capacity[edge]
                        if epoch_loads[epoch][edge] > epoch_maxima[epoch]:
                            epoch_maxima[epoch] = epoch_loads[epoch][edge]

        best_mapping = base_mapping
        best_score = incumbent_score
        assignment: dict[int, int] = {}
        used_servers: set[int] = set()

        def apply_rank(rank, server):
            assignment[rank] = int(server)
            used_servers.add(int(server))
            changes: list[tuple[int, tuple[int, int], float]] = []
            old_maxima: dict[int, float] = {}
            for epoch, src_rank, dst_rank, volume in rank_incidence.get(rank, []):
                other_rank = dst_rank if src_rank == rank else src_rank
                if other_rank not in assignment:
                    continue
                src_server = assignment[src_rank]
                dst_server = assignment[dst_rank]
                for edge in path_edges[(src_server, dst_server)]:
                    if epoch not in old_maxima:
                        old_maxima[epoch] = epoch_maxima[epoch]
                    increment = float(volume) / edge_capacity[edge]
                    epoch_loads[epoch][edge] += increment
                    if epoch_loads[epoch][edge] > epoch_maxima[epoch]:
                        epoch_maxima[epoch] = epoch_loads[epoch][edge]
                    changes.append((epoch, edge, increment))
            return changes, old_maxima

        def rollback_rank(rank, server, changes, old_maxima):
            for epoch, edge, increment in reversed(changes):
                epoch_loads[epoch][edge] -= increment
                if abs(epoch_loads[epoch][edge]) <= 1e-15:
                    del epoch_loads[epoch][edge]
            for epoch, value in old_maxima.items():
                epoch_maxima[epoch] = value
            used_servers.remove(int(server))
            del assignment[rank]

        def candidate_servers_for_rank(rank):
            candidates = [int(server) for server in available_servers if int(server) not in used_servers]
            if len(candidates) <= 1:
                return candidates

            scored_candidates = []
            preferred = int(base_mapping[tenant][rank])
            for server in candidates:
                changes, old_maxima = apply_rank(rank, server)
                lower_bound = self._surrogate_score_from_epoch_max(epoch_maxima)
                rollback_rank(rank, server, changes, old_maxima)
                scored_candidates.append((lower_bound, 0 if server == preferred else 1, server))

            scored_candidates.sort(key=lambda item: (item[0][0], item[0][1], item[1], item[2]))
            return [server for _, _, server in scored_candidates]

        def finalize_mapping():
            candidate = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            candidate[tenant] = {
                rank: int(assignment[rank])
                for rank in self.rank_orders[tenant]
            }
            return candidate

        def dfs(depth):
            nonlocal best_mapping, best_score
            if time.time() >= deadline:
                return

            lower_bound = self._surrogate_score_from_epoch_max(epoch_maxima)
            if not self._lower_bound_can_improve(lower_bound, best_score):
                return

            if depth >= len(branch_ranks):
                candidate_mapping = finalize_mapping()
                candidate_score = self._evaluate_surrogate_mapping(candidate_mapping)
                if self._is_better_objective(candidate_score, best_score):
                    best_mapping = candidate_mapping
                    best_score = candidate_score
                return

            rank = branch_ranks[depth]
            for server in candidate_servers_for_rank(rank):
                if time.time() >= deadline:
                    break
                changes, old_maxima = apply_rank(rank, server)
                dfs(depth + 1)
                rollback_rank(rank, server, changes, old_maxima)

        dfs(0)
        return best_mapping, best_score

    def _optimize_tenant_block_with_prices(self, base_mapping, tenant, epoch_prices, deadline):
        ranks = self.rank_orders[tenant]
        current_servers = tuple(base_mapping[tenant][rank] for rank in ranks)
        if len(current_servers) <= 1:
            return base_mapping, self._tenant_price_cost(
                tenant,
                base_mapping,
                self._pair_epoch_price_lookup(current_servers, epoch_prices),
            )

        pair_epoch_price = self._pair_epoch_price_lookup(current_servers, epoch_prices)
        best_mapping = base_mapping
        best_cost = self._tenant_price_cost(tenant, base_mapping, pair_epoch_price)

        compiled_tenant = self.data["compiled_schedule"]["per_tenant"][tenant]
        all_flows = compiled_tenant["all_flows"]
        branch_ranks = [int(rank) for rank in compiled_tenant["branch_order"] if rank in ranks]
        rank_incidence = compiled_tenant["rank_incidence"]

        assignment: dict[int, int] = {}
        used_servers: set[int] = set()
        best_assignment = {int(rank): int(base_mapping[tenant][rank]) for rank in ranks}
        partial_cost = 0.0

        def lower_bound(current_partial_cost):
            remaining_servers = [int(server) for server in current_servers if int(server) not in used_servers]
            bound = float(current_partial_cost)
            for epoch, src_rank, dst_rank, volume in all_flows:
                src_assigned = src_rank in assignment
                dst_assigned = dst_rank in assignment
                if src_assigned and dst_assigned:
                    continue
                if src_assigned and not dst_assigned:
                    src_server = assignment[src_rank]
                    best_pair = min(
                        pair_epoch_price[(int(epoch), int(src_server), int(dst_server))]
                        for dst_server in remaining_servers
                        if int(dst_server) != int(src_server)
                    )
                    bound += float(volume) * best_pair
                elif dst_assigned and not src_assigned:
                    dst_server = assignment[dst_rank]
                    best_pair = min(
                        pair_epoch_price[(int(epoch), int(src_server), int(dst_server))]
                        for src_server in remaining_servers
                        if int(src_server) != int(dst_server)
                    )
                    bound += float(volume) * best_pair
                else:
                    if len(remaining_servers) < 2:
                        continue
                    best_pair = min(
                        pair_epoch_price[(int(epoch), int(src_server), int(dst_server))]
                        for src_server in remaining_servers
                        for dst_server in remaining_servers
                        if int(src_server) != int(dst_server)
                    )
                    bound += float(volume) * best_pair
            return float(bound)

        def candidate_servers_for_rank(rank):
            available = [int(server) for server in current_servers if int(server) not in used_servers]
            preferred = int(base_mapping[tenant][rank])
            scored = []
            for server in available:
                delta = 0.0
                for epoch, src_rank, dst_rank, volume in rank_incidence.get(rank, []):
                    other_rank = dst_rank if src_rank == rank else src_rank
                    if other_rank not in assignment:
                        continue
                    if src_rank == rank:
                        src_server = int(server)
                        dst_server = int(assignment[other_rank])
                    else:
                        src_server = int(assignment[other_rank])
                        dst_server = int(server)
                    delta += float(volume) * pair_epoch_price[(int(epoch), src_server, dst_server)]
                scored.append((float(delta), 0 if int(server) == preferred else 1, int(server)))
            scored.sort(key=lambda item: (item[0], item[1], item[2]))
            return [server for _, _, server in scored]

        def apply_rank(rank, server, current_partial_cost):
            assignment[rank] = int(server)
            used_servers.add(int(server))
            delta = 0.0
            for epoch, src_rank, dst_rank, volume in rank_incidence.get(rank, []):
                other_rank = dst_rank if src_rank == rank else src_rank
                if other_rank not in assignment:
                    continue
                if src_rank == rank:
                    src_server = int(server)
                    dst_server = int(assignment[other_rank])
                else:
                    src_server = int(assignment[other_rank])
                    dst_server = int(server)
                delta += float(volume) * pair_epoch_price[(int(epoch), src_server, dst_server)]
            return current_partial_cost + delta

        def rollback_rank(rank, server):
            used_servers.remove(int(server))
            del assignment[rank]

        def finalize_mapping():
            candidate = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            candidate[tenant] = {
                rank: int(best_assignment[rank])
                for rank in ranks
            }
            return candidate

        def dfs(depth, current_partial_cost):
            nonlocal best_cost, best_assignment, partial_cost
            if time.time() >= deadline:
                return

            bound = lower_bound(current_partial_cost)
            if bound >= best_cost - 1e-12:
                return

            if depth >= len(branch_ranks):
                if current_partial_cost < best_cost - 1e-12:
                    best_cost = float(current_partial_cost)
                    best_assignment = {int(rank): int(server) for rank, server in assignment.items()}
                return

            rank = branch_ranks[depth]
            for server in candidate_servers_for_rank(rank):
                if time.time() >= deadline:
                    break
                next_cost = apply_rank(rank, server, current_partial_cost)
                dfs(depth + 1, next_cost)
                rollback_rank(rank, server)

        partial_cost = 0.0
        dfs(0, partial_cost)
        if best_assignment != {int(rank): int(base_mapping[tenant][rank]) for rank in ranks}:
            best_mapping = {
                current_tenant: dict(rank_to_server)
                for current_tenant, rank_to_server in base_mapping.items()
            }
            best_mapping[tenant] = {
                rank: int(best_assignment[rank])
                for rank in ranks
            }
        return best_mapping, float(best_cost)

    def _coordinate_surrogate_descent(self, best_mapping, best_score, deadline):
        if not self.tenants:
            return best_mapping, best_score

        improved = True
        while improved and time.time() < deadline:
            improved = False
            self.search_rounds += 1

            for tenant in self.tenants:
                if time.time() >= deadline:
                    break
                tenant_mapping, tenant_score = self._optimize_tenant_block(
                    best_mapping,
                    tenant,
                    best_score,
                    deadline,
                )
                if tenant_mapping is not best_mapping:
                    best_mapping = tenant_mapping
                    best_score = tenant_score
                    improved = True

        return best_mapping, best_score

    def _surrogate_pair_swap_polish(self, best_mapping, best_score, deadline):
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
        for _, _, mapping in self._surrogate_candidates:
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
        polish_budget = min(max(total_budget * 0.25, 3.0), 10.0)
        search_deadline = max(start_time, deadline - polish_budget)

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
        working_score = best_score

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
            working_score = round_score
            if not changed:
                break

        if time.time() < search_deadline:
            best_mapping, best_score = self._coordinate_surrogate_descent(best_mapping, best_score, search_deadline)
        if time.time() < search_deadline:
            best_mapping, best_score = self._surrogate_pair_swap_polish(best_mapping, best_score, search_deadline)

        self._register_surrogate_candidate(best_mapping, best_score)
        final_mapping, final_score = self._simulator_rerank_candidates(deadline)
        if time.time() < deadline:
            final_mapping, final_score = self._simulator_pair_swap_polish(
                final_mapping,
                final_score,
                deadline,
            )

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
MappingILPSolver = MappingMILPSolver
