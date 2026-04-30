from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from multitenant.workloads import build_collective_schedule, build_collective_stage_flows


def _path_edges(path_table, src, dst):
    path = path_table.get((src, dst))
    if not path:
        return []
    return list(zip(path[:-1], path[1:]))


def _build_stage_metadata(
    datacenter,
    tenant_mapping,
    tenant_flows,
    path_table,
    collective=None,
    single_flow_size=None,
    tenant_collective_specs=None,
):
    capacities = {
        (src, dst): float(datacenter.topology[src][dst]["capacity"])
        for src, dst in datacenter.topology.edges()
    }

    if tenant_collective_specs is not None or (
        collective in {"allgather", "reducescatter", "allreduce", "alltoall"}
        and single_flow_size is not None
    ):
        stage_flows_by_tenant = build_collective_stage_flows(
            tenant_mapping,
            None if single_flow_size is None else int(single_flow_size),
            collective,
            tenant_collective_specs=tenant_collective_specs,
        )
    else:
        if tenant_flows is None:
            raise ValueError(
                "tenant_flows is required when collective/single_flow_size are not provided"
            )
        stage_flows_by_tenant = {tenant: [tenant_flows[tenant]] for tenant in tenant_mapping}

    stage_data = {}
    base_durations = {}
    aggregate_demands = {}
    bottleneck_caps = {}

    for tenant in sorted(tenant_mapping.keys()):
        tenant_stages = []
        total_duration = 0.0
        aggregate_link_vol_bits = {}
        tenant_bottleneck_caps = []

        for stage_flows in stage_flows_by_tenant.get(tenant, []):
            link_vol_bits = {}
            for logical_src, logical_dst, volume in stage_flows:
                volume_bits = float(volume) * 1e9
                physical_src = tenant_mapping[tenant][logical_src]
                physical_dst = tenant_mapping[tenant][logical_dst]
                if physical_src == physical_dst:
                    continue

                for edge in _path_edges(path_table, physical_src, physical_dst):
                    link_vol_bits[edge] = link_vol_bits.get(edge, 0.0) + volume_bits
                    aggregate_link_vol_bits[edge] = aggregate_link_vol_bits.get(edge, 0.0) + volume_bits

            stage_duration = 0.0
            stage_bottleneck_cap = 0.0
            for edge, volume_bits in link_vol_bits.items():
                capacity = capacities.get(edge, 0.0)
                if capacity <= 0:
                    continue
                edge_duration = volume_bits / capacity
                if edge_duration > stage_duration:
                    stage_duration = edge_duration
                    stage_bottleneck_cap = capacity

            stage_duration = max(stage_duration, 1e-9)
            stage_demand_gbps = {}
            stage_load_fraction = {}
            stage_link_volume_bits = {}
            for edge, volume_bits in link_vol_bits.items():
                if volume_bits <= 0:
                    continue
                capacity = capacities.get(edge, 0.0)
                stage_link_volume_bits[edge] = volume_bits
                stage_demand_gbps[edge] = (volume_bits / stage_duration) / 1e9
                if capacity > 0:
                    stage_load_fraction[edge] = volume_bits / (stage_duration * capacity)

            tenant_stages.append(
                {
                    "base_duration": stage_duration,
                    "demand_gbps": stage_demand_gbps,
                    "load_fraction": stage_load_fraction,
                    "link_volume_bits": stage_link_volume_bits,
                }
            )
            total_duration += stage_duration
            tenant_bottleneck_caps.append(stage_bottleneck_cap)

        if not tenant_stages:
            tenant_stages.append(
                {
                    "base_duration": 1e-9,
                    "demand_gbps": {},
                    "load_fraction": {},
                    "link_volume_bits": {},
                }
            )
            total_duration = 1e-9
            tenant_bottleneck_caps.append(0.0)

        aggregate_demand = {}
        for edge, volume_bits in aggregate_link_vol_bits.items():
            if volume_bits > 0:
                aggregate_demand[edge] = (volume_bits / total_duration) / 1e9

        stage_data[tenant] = tenant_stages
        base_durations[tenant] = max(total_duration, 1e-9)
        aggregate_demands[tenant] = aggregate_demand
        bottleneck_caps[tenant] = tenant_bottleneck_caps

    return capacities, stage_data, base_durations, aggregate_demands, bottleneck_caps


def _build_task_metadata(
    datacenter,
    tenant_mapping,
    tenant_flows,
    path_table,
    collective=None,
    single_flow_size=None,
    tenant_collective_specs=None,
):
    capacities = {
        (src, dst): float(datacenter.topology[src][dst]["capacity"])
        for src, dst in datacenter.topology.edges()
    }

    if tenant_collective_specs is not None or (
        collective in {"allgather", "reducescatter", "allreduce", "alltoall"}
        and single_flow_size is not None
    ):
        schedule_by_tenant = build_collective_schedule(
            tenant_mapping,
            None if single_flow_size is None else int(single_flow_size),
            collective,
            tenant_collective_specs=tenant_collective_specs,
        )
    else:
        if tenant_flows is None:
            raise ValueError(
                "tenant_flows is required when collective/single_flow_size are not provided"
            )
        schedule_by_tenant = {}
        for tenant, flows in tenant_flows.items():
            tasks = []
            sender_order = {}
            for task_id, (src, dst, volume) in enumerate(flows):
                tasks.append(
                    {
                        "task_id": int(task_id),
                        "name": f"{tenant}-flow-{task_id}",
                        "src_rank": int(src),
                        "dst_rank": int(dst),
                        "V": float(volume),
                        "preds": [],
                    }
                )
                sender_order.setdefault(int(src), []).append(int(task_id))
            schedule_by_tenant[tenant] = {
                "tasks": tasks,
                "sender_order": sender_order,
                "task_order": [int(task["task_id"]) for task in tasks],
            }

    task_data = {}
    base_durations = {}
    aggregate_demands = {}

    for tenant in sorted(tenant_mapping.keys()):
        mapping = tenant_mapping[tenant]
        tenant_schedule = schedule_by_tenant.get(tenant, {})
        tasks = tenant_schedule.get("tasks", [])
        sender_order = {
            int(sender): [int(task_id) for task_id in task_ids]
            for sender, task_ids in tenant_schedule.get("sender_order", {}).items()
        }

        tenant_tasks = []
        aggregate_link_bits = {}
        min_task_isolated = None
        total_isolated = 0.0

        for task in tasks:
            task_id = int(task["task_id"])
            src_phys = mapping[int(task["src_rank"])]
            dst_phys = mapping[int(task["dst_rank"])]
            path = path_table.get((src_phys, dst_phys))
            if not path:
                continue

            path_edges = _path_edges(path_table, src_phys, dst_phys)
            volume_bits = float(task["V"]) * 1e9
            isolated_duration = 0.0
            if path_edges:
                isolated_duration = max(
                    (volume_bits / capacities[edge] for edge in path_edges if capacities.get(edge, 0.0) > 0.0),
                    default=0.0,
                )
            min_task_isolated = isolated_duration if min_task_isolated is None else min(min_task_isolated, isolated_duration)
            total_isolated += isolated_duration

            for edge in path_edges:
                aggregate_link_bits[edge] = aggregate_link_bits.get(edge, 0.0) + volume_bits

            tenant_tasks.append(
                {
                    "task_id": task_id,
                    "name": str(task["name"]),
                    "sender": int(task["src_rank"]),
                    "preds": [int(pred) for pred in task.get("preds", [])],
                    "volume_bits": volume_bits,
                    "path_edges": path_edges,
                }
            )

        task_lookup = {task["task_id"]: task for task in tenant_tasks}
        aggregate_demand = {
            edge: (bits / max(total_isolated, 1e-9)) / 1e9
            for edge, bits in aggregate_link_bits.items()
            if bits > 0.0
        }
        task_data[tenant] = {
            "tasks": tenant_tasks,
            "task_lookup": task_lookup,
            "sender_order": sender_order,
            "aggregate_link_bits": aggregate_link_bits,
            "min_task_isolated": max(min_task_isolated or 1e-9, 1e-9),
        }
        base_durations[tenant] = max(total_isolated, 1e-9)
        aggregate_demands[tenant] = aggregate_demand

    return capacities, task_data, base_durations, aggregate_demands


class HarmonicsBaselineILP:
    """Harmonics scheduling baseline with stage-aware collective dependencies."""

    def __init__(
        self,
        datacenter,
        tenant_mapping,
        tenant_flows=None,
        path_table=None,
        single_flow_size=None,
        collective=None,
        tenant_collective_specs=None,
        verbose=True,
        rmin=0.1,
        makespan_weight=0.0,
        estimation=0,
    ):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.path_table = path_table
        self.single_flow_size = single_flow_size
        self.collective = collective
        self.tenant_collective_specs = tenant_collective_specs
        self.verbose = verbose
        self.rmin = rmin
        self.makespan_weight = makespan_weight
        self.estimation = estimation

        self.model = None
        self.schedule = {}
        self.M = sorted(self.tenant_mapping.keys())
        self.L = list(self.datacenter.topology.edges())

        (
            self.cap,
            self.stage_data,
            self.base_dur,
            self.demands,
            self.stage_bottleneck_caps,
        ) = _build_stage_metadata(
            self.datacenter,
            self.tenant_mapping,
            self.tenant_flows,
            self.path_table,
            collective=self.collective,
            single_flow_size=self.single_flow_size,
            tenant_collective_specs=self.tenant_collective_specs,
        )

        self._build_model()

    def _build_model(self):
        self.model = gp.Model("HarmonicsBaseline")
        self.model.Params.OutputFlag = 1 if self.verbose else 0

        min_stage_duration = min(
            stage["base_duration"]
            for tenant in self.M
            for stage in self.stage_data[tenant]
        )
        min_stage_duration = max(min_stage_duration, 1e-9)
        self.dt = min_stage_duration / 5.0
        horizon_estimate = max(self.estimation, max(self.base_dur.values(), default=1.0))
        self.H = int((horizon_estimate * 2) / self.dt) + 5
        self.T_slots = range(self.H)

        if self.verbose:
            print(f"Building Harmonics baseline: dt={self.dt}s, Horizon={self.H}")

        self.rates = [round(rate, 2) for rate in np.linspace(1.0, self.rmin, 4)]
        self.K_levels = range(len(self.rates))

        self.stage_dur_slots = {}
        self.stage_offsets = {}
        self.total_dur_slots = {}
        for tenant in self.M:
            self.stage_dur_slots[tenant] = {}
            self.stage_offsets[tenant] = {}
            self.total_dur_slots[tenant] = {}
            for level in self.K_levels:
                rate_ratio = self.rates[level]
                offsets = []
                durations = []
                current_offset = 0
                for stage in self.stage_data[tenant]:
                    slots_needed = int(stage["base_duration"] / (rate_ratio * self.dt)) + 1
                    durations.append(slots_needed)
                    offsets.append(current_offset)
                    current_offset += slots_needed
                self.stage_dur_slots[tenant][level] = durations
                self.stage_offsets[tenant][level] = offsets
                self.total_dur_slots[tenant][level] = current_offset

        self.x = {}
        for tenant in self.M:
            for start_slot in self.T_slots:
                for level in self.K_levels:
                    if start_slot + self.total_dur_slots[tenant][level] > self.H:
                        continue
                    self.x[(tenant, start_slot, level)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{tenant}_{start_slot}_{level}",
                    )

        for tenant in self.M:
            self.model.addConstr(
                gp.quicksum(
                    self.x[(tenant, start_slot, level)]
                    for start_slot in self.T_slots
                    for level in self.K_levels
                    if (tenant, start_slot, level) in self.x
                )
                == 1,
                name=f"StartOnce_{tenant}",
            )

        relevant_edges = {
            edge
            for tenant in self.M
            for stage in self.stage_data[tenant]
            for edge in stage["demand_gbps"]
        }
        for edge in relevant_edges:
            capacity_gbps = self.cap[edge] / 1e9
            for current_slot in self.T_slots:
                terms = []
                for tenant in self.M:
                    for level in self.K_levels:
                        rate_ratio = self.rates[level]
                        for stage_idx, stage in enumerate(self.stage_data[tenant]):
                            load_gbps = stage["demand_gbps"].get(edge, 0.0)
                            if load_gbps <= 0:
                                continue
                            duration_slots = self.stage_dur_slots[tenant][level][stage_idx]
                            offset_slots = self.stage_offsets[tenant][level][stage_idx]
                            start_min = max(0, current_slot - offset_slots - duration_slots + 1)
                            start_max = current_slot - offset_slots
                            if start_max < start_min:
                                continue
                            for start_slot in range(start_min, start_max + 1):
                                if (tenant, start_slot, level) in self.x:
                                    terms.append(load_gbps * rate_ratio * self.x[(tenant, start_slot, level)])

                if terms:
                    self.model.addConstr(
                        gp.quicksum(terms) <= capacity_gbps,
                        name=f"Cap_{edge}_{current_slot}",
                    )

        self.C = {}
        makespan = self.model.addVar(vtype=GRB.CONTINUOUS, name="makespan")
        objective_terms = []
        for tenant in self.M:
            completion_time = gp.quicksum(
                self.x[(tenant, start_slot, level)]
                * (start_slot + self.total_dur_slots[tenant][level])
                * self.dt
                for start_slot in self.T_slots
                for level in self.K_levels
                if (tenant, start_slot, level) in self.x
            )
            self.C[tenant] = completion_time
            objective_terms.append(completion_time)
            self.model.addConstr(makespan >= completion_time)

        tenant_count = max(len(self.M), 1)
        self.model.ModelSense = GRB.MINIMIZE
        self.model.setObjectiveN(
            makespan,
            index=0,
            priority=2,
            name="makespan",
        )
        self.model.setObjectiveN(
            gp.quicksum(objective_terms) / tenant_count,
            index=1,
            priority=1,
            name="avg_jct",
        )

    def solve(self, timelimit=30, mipgap=0.05):
        self.model.setParam("TimeLimit", timelimit)
        self.model.setParam("MIPGap", mipgap)
        self.model.optimize()

        if self.model.SolCount == 0:
            if self.verbose:
                print("[Harmonics] No solution found.")
            return None

        schedule = {}
        for tenant in self.M:
            for start_slot in self.T_slots:
                for level in self.K_levels:
                    if (tenant, start_slot, level) not in self.x:
                        continue
                    if self.x[(tenant, start_slot, level)].X <= 0.5:
                        continue

                    start_time = start_slot * self.dt
                    effective_capacity = min(self.stage_bottleneck_caps[tenant]) if self.stage_bottleneck_caps[tenant] else 0.0
                    physical_rate = self.rates[level] * effective_capacity
                    schedule[tenant] = (start_time * 0.98, physical_rate)
                    if self.verbose:
                        end_time = start_time + self.total_dur_slots[tenant][level] * self.dt
                        print(
                            f"Tenant {tenant}: Start={start_time:.4f} End={end_time:.4f} "
                            f"Rate={physical_rate / 1e9:.2f} Gbps (Ratio={self.rates[level]:.2f})"
                        )
                    break
                if tenant in schedule:
                    break

        return schedule


class HarmonicsBaselineHeuristic:
    @staticmethod
    def _better_objective(candidate, incumbent, tol=1e-9):
        cand_max, cand_avg = candidate
        inc_max, inc_avg = incumbent
        if cand_max < inc_max - tol:
            return True
        if cand_max > inc_max + tol:
            return False
        return cand_avg < inc_avg - tol

    def __init__(
        self,
        datacenter,
        tenant_mapping,
        tenant_flows=None,
        path_table=None,
        single_flow_size=None,
        collective="allreduce",
        tenant_collective_specs=None,
        verbose=True,
    ):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.path_table = path_table
        self.single_flow_size = single_flow_size
        self.collective = collective
        self.tenant_collective_specs = tenant_collective_specs
        self.verbose = verbose

        self.M_tenants = sorted(self.tenant_mapping.keys())
        self.use_task_dag = self.tenant_collective_specs is not None or (
            self.collective in {"allgather", "reducescatter", "allreduce", "alltoall"}
            and self.single_flow_size is not None
        )

        self.cap = {}
        self.stage_data = {}
        self.task_data = {}
        self.durations = {}
        self.dt = 1e-4
        self.tenant_profiles = {}
        self.tenant_pressure = {}
        self.tenant_peak_load = {}
        self._max_isolated_slots = 64

        if self.use_task_dag:
            (
                self.cap,
                self.task_data,
                self.durations,
                _aggregate_demands,
            ) = _build_task_metadata(
                self.datacenter,
                self.tenant_mapping,
                self.tenant_flows,
                self.path_table,
                collective=self.collective,
                single_flow_size=self.single_flow_size,
                tenant_collective_specs=self.tenant_collective_specs,
            )

            min_task_duration = min(
                (
                    task_info["min_task_isolated"]
                    for task_info in self.task_data.values()
                    if task_info["tasks"]
                ),
                default=1e-9,
            )
            self.dt = max(min_task_duration / 4.0, 1e-4)

            for tenant in self.M_tenants:
                aggregate_link_bits = self.task_data[tenant]["aggregate_link_bits"]
                pressure = 0.0
                peak_load = 0.0
                for edge, bits in aggregate_link_bits.items():
                    capacity = self.cap.get(edge, 0.0)
                    if capacity <= 0.0:
                        continue
                    occupancy = bits / capacity
                    pressure += occupancy
                    peak_load = max(peak_load, occupancy)
                total_slots = int(np.ceil(max(self.durations.get(tenant, self.dt), self.dt) / self.dt))
                self.tenant_profiles[tenant] = {
                    "stage_slots": [max(1, total_slots)],
                    "stage_offsets": [0],
                    "total_slots": max(1, total_slots),
                }
                self.tenant_pressure[tenant] = pressure
                self.tenant_peak_load[tenant] = peak_load

            self._max_isolated_slots = int(
                sum(self.tenant_profiles[tenant]["total_slots"] for tenant in self.M_tenants)
            ) + 64
        else:
            (
                self.cap,
                self.stage_data,
                self.durations,
                _aggregate_demands,
                _bottleneck_caps,
            ) = _build_stage_metadata(
                self.datacenter,
                self.tenant_mapping,
                self.tenant_flows,
                self.path_table,
                collective=self.collective,
                single_flow_size=self.single_flow_size,
                tenant_collective_specs=self.tenant_collective_specs,
            )
            min_stage_duration = min(
                stage["base_duration"]
                for tenant in self.M_tenants
                for stage in self.stage_data[tenant]
            )
            self.dt = max(min_stage_duration / 4.0, 1e-4)
            for tenant in self.M_tenants:
                stage_slots = [max(1, int(np.ceil(stage["base_duration"] / self.dt))) for stage in self.stage_data[tenant]]
                stage_offsets = []
                current_offset = 0
                pressure = 0.0
                peak_load = 0.0
                for stage_idx, duration_slots in enumerate(stage_slots):
                    stage_offsets.append(current_offset)
                    current_offset += duration_slots
                    load_sum = sum(self.stage_data[tenant][stage_idx]["load_fraction"].values())
                    pressure += load_sum * duration_slots
                    peak_load = max(
                        peak_load,
                        max(self.stage_data[tenant][stage_idx]["load_fraction"].values(), default=0.0),
                    )
                self.tenant_profiles[tenant] = {
                    "stage_slots": stage_slots,
                    "stage_offsets": stage_offsets,
                    "total_slots": current_offset,
                }
                self.tenant_pressure[tenant] = pressure
                self.tenant_peak_load[tenant] = peak_load
            self._max_isolated_slots = int(
                sum(self.tenant_profiles[tenant]["total_slots"] for tenant in self.M_tenants)
            ) + 64

        if self.verbose:
            print(f"Harmonics baseline prep: {len(self.M_tenants)} tenants.")
            print(f"Durations (max): {self.durations}")

    @staticmethod
    def _objective(schedule):
        if not schedule:
            return (float("inf"), float("inf"))
        total_completion = sum(end_time for _, end_time in schedule.values())
        avg_completion = total_completion / max(len(schedule), 1)
        makespan = max(end_time for _, end_time in schedule.values())
        return (makespan, avg_completion)

    def _candidate_orders(self):
        duration_order = sorted(self.M_tenants, key=lambda tenant: (self.durations[tenant], self.tenant_pressure[tenant]))
        pressure_order = sorted(
            self.M_tenants,
            key=lambda tenant: (-self.tenant_pressure[tenant], -self.tenant_peak_load[tenant], self.durations[tenant]),
        )
        hybrid_order = sorted(
            self.M_tenants,
            key=lambda tenant: (
                -(self.tenant_pressure[tenant] * max(self.durations[tenant], self.dt)),
                self.durations[tenant],
            ),
        )

        orders = []
        for order in (duration_order, pressure_order, hybrid_order):
            if order not in orders:
                orders.append(order)
        return orders

    def _ensure_ready_stage(self, tenant, slot, current_stage, remaining_bits, stage_finish_slots):
        while current_stage[tenant] < len(self.stage_data[tenant]):
            stage_idx = current_stage[tenant]
            stage_bits = {
                edge: bits
                for edge, bits in self.stage_data[tenant][stage_idx]["link_volume_bits"].items()
                if bits > 1e-12
            }
            if stage_bits:
                remaining_bits[tenant] = stage_bits
                return
            stage_finish_slots[tenant].append(slot)
            current_stage[tenant] += 1
        remaining_bits.pop(tenant, None)

    def _simulate_schedule(self, release_slots):
        tenants = sorted(release_slots.keys())
        if not tenants:
            return {}, {}

        normalized_release_slots = {tenant: max(0, int(slot)) for tenant, slot in release_slots.items()}
        current_stage = {tenant: 0 for tenant in tenants}
        remaining_bits = {}
        stage_finish_slots = {tenant: [] for tenant in tenants}
        final_schedule = {}

        horizon_slots = max(normalized_release_slots.values(), default=0) + self._max_isolated_slots * 4

        for slot in range(horizon_slots):
            for tenant in tenants:
                if tenant in final_schedule or slot < normalized_release_slots[tenant]:
                    continue
                if tenant not in remaining_bits:
                    self._ensure_ready_stage(tenant, slot, current_stage, remaining_bits, stage_finish_slots)
                    if current_stage[tenant] >= len(self.stage_data[tenant]):
                        final_schedule[tenant] = (
                            normalized_release_slots[tenant] * self.dt,
                            slot * self.dt,
                        )

            active_edges = {}
            active_stages = []
            for tenant in tenants:
                if tenant in final_schedule or tenant not in remaining_bits:
                    continue
                stage_bits = remaining_bits[tenant]
                if not stage_bits:
                    continue
                active_stages.append((tenant, stage_bits))
                for edge, bits in stage_bits.items():
                    if bits > 1e-12:
                        active_edges[edge] = active_edges.get(edge, 0) + 1

            if not active_stages:
                if len(final_schedule) == len(tenants):
                    break
                continue

            for tenant, stage_bits in active_stages:
                for edge, bits in list(stage_bits.items()):
                    if bits <= 1e-12:
                        continue
                    occupancy = active_edges.get(edge, 0)
                    if occupancy <= 0:
                        continue
                    service_bits = (self.cap[edge] * self.dt) / occupancy
                    stage_bits[edge] = max(0.0, bits - service_bits)

            for tenant, stage_bits in active_stages:
                if any(bits > 1e-9 for bits in stage_bits.values()):
                    continue
                stage_finish_slots[tenant].append(slot + 1)
                current_stage[tenant] += 1
                remaining_bits.pop(tenant, None)
                self._ensure_ready_stage(tenant, slot + 1, current_stage, remaining_bits, stage_finish_slots)
                if current_stage[tenant] >= len(self.stage_data[tenant]):
                    final_schedule[tenant] = (
                        normalized_release_slots[tenant] * self.dt,
                        (slot + 1) * self.dt,
                    )

        for tenant in tenants:
            if tenant not in final_schedule:
                fallback_end = horizon_slots * self.dt
                final_schedule[tenant] = (
                    normalized_release_slots[tenant] * self.dt,
                    fallback_end,
                )

        return final_schedule, stage_finish_slots

    def _simulate_task_schedule(self, release_slots):
        tenants = sorted(release_slots.keys())
        if not tenants:
            return {}, {}

        normalized_release_slots = {
            tenant: max(0, int(slot))
            for tenant, slot in release_slots.items()
        }
        release_times = {
            tenant: normalized_release_slots[tenant] * self.dt
            for tenant in tenants
        }

        remaining_bits = {
            tenant: {
                task["task_id"]: float(task["volume_bits"])
                for task in self.task_data[tenant]["tasks"]
            }
            for tenant in tenants
        }
        completed = {tenant: set() for tenant in tenants}
        active_by_sender = {}
        task_finish_times = {}
        final_schedule = {}

        horizon_slots = max(normalized_release_slots.values(), default=0) + self._max_isolated_slots * 4

        for slot in range(horizon_slots):
            current_time = slot * self.dt

            active_tasks = []
            for tenant in tenants:
                if tenant in final_schedule and len(completed[tenant]) == len(self.task_data[tenant]["tasks"]):
                    continue
                if slot < normalized_release_slots[tenant]:
                    continue

                sender_order = self.task_data[tenant]["sender_order"]
                task_lookup = self.task_data[tenant]["task_lookup"]
                for sender, ordered_task_ids in sender_order.items():
                    current_task_id = active_by_sender.get((tenant, sender))
                    if current_task_id is not None and remaining_bits[tenant][current_task_id] > 1e-9:
                        active_tasks.append((tenant, current_task_id))
                        continue

                    active_by_sender.pop((tenant, sender), None)
                    for task_id in ordered_task_ids:
                        if remaining_bits[tenant].get(task_id, 0.0) <= 1e-9:
                            continue
                        task = task_lookup.get(task_id)
                        if task is None:
                            continue
                        if any(pred_task_id not in completed[tenant] for pred_task_id in task["preds"]):
                            continue
                        active_by_sender[(tenant, sender)] = task_id
                        active_tasks.append((tenant, task_id))
                        break

            if not active_tasks:
                if all(len(completed[tenant]) == len(self.task_data[tenant]["tasks"]) for tenant in tenants):
                    break
                continue

            link_to_active = {}
            for tenant, task_id in active_tasks:
                task = self.task_data[tenant]["task_lookup"][task_id]
                for edge in task["path_edges"]:
                    link_to_active[edge] = link_to_active.get(edge, 0) + 1

            completed_this_slot = []
            for tenant, task_id in active_tasks:
                task = self.task_data[tenant]["task_lookup"][task_id]
                path_edges = task["path_edges"]
                if not path_edges:
                    service_bits = remaining_bits[tenant][task_id]
                else:
                    service_bits = min(
                        (self.cap[edge] * self.dt) / max(link_to_active.get(edge, 1), 1)
                        for edge in path_edges
                    )

                remaining_bits[tenant][task_id] = max(0.0, remaining_bits[tenant][task_id] - service_bits)
                if remaining_bits[tenant][task_id] <= 1e-9:
                    completed_this_slot.append((tenant, task_id))
                    task_finish_times[str(task["name"])] = (slot + 1) * self.dt
                    active_by_sender.pop((tenant, task["sender"]), None)

            for tenant, task_id in completed_this_slot:
                completed[tenant].add(task_id)
                if len(completed[tenant]) == len(self.task_data[tenant]["tasks"]) and tenant not in final_schedule:
                    final_schedule[tenant] = (
                        release_times[tenant],
                        (slot + 1) * self.dt,
                    )

        for tenant in tenants:
            if tenant not in final_schedule:
                fallback_end = horizon_slots * self.dt
                final_schedule[tenant] = (
                    release_times[tenant],
                    fallback_end,
                )

        return final_schedule, task_finish_times

    def _candidate_release_slots(self, release_slots):
        if not release_slots:
            return [0]
        if self.use_task_dag:
            schedule, task_finish_times = self._simulate_task_schedule(release_slots)
            event_slots = {0}
            event_slots.update(max(0, int(np.ceil(start / self.dt))) for start, _ in schedule.values())
            event_slots.update(max(0, int(np.ceil(end / self.dt))) for _, end in schedule.values())
            event_slots.update(max(0, int(np.ceil(finish / self.dt))) for finish in task_finish_times.values())
            latest_slot = max(event_slots)
            event_slots.add(latest_slot + 1)
            return sorted(event_slots)

        schedule, stage_finish_slots = self._simulate_schedule(release_slots)
        event_slots = {0}
        event_slots.update(max(0, int(start / self.dt)) for start, _ in schedule.values())
        event_slots.update(max(0, int(np.ceil(end / self.dt))) for _, end in schedule.values())
        for slots in stage_finish_slots.values():
            event_slots.update(max(0, int(slot)) for slot in slots)
        latest_slot = max(event_slots)
        event_slots.add(latest_slot + 1)
        return sorted(event_slots)

    def _build_schedule_for_order(self, tenant_order):
        release_slots = {}

        for tenant in tenant_order:
            best_slot = 0
            best_schedule = None
            best_score = (float("inf"), float("inf"))
            for candidate_slot in self._candidate_release_slots(release_slots):
                candidate_release = dict(release_slots)
                candidate_release[tenant] = candidate_slot
                if self.use_task_dag:
                    candidate_schedule, _ = self._simulate_task_schedule(candidate_release)
                else:
                    candidate_schedule, _ = self._simulate_schedule(candidate_release)
                candidate_score = self._objective(candidate_schedule)
                if self._better_objective(candidate_score, best_score):
                    best_slot = candidate_slot
                    best_schedule = candidate_schedule
                    best_score = candidate_score
            release_slots[tenant] = best_slot

        if self.use_task_dag:
            final_schedule, _ = self._simulate_task_schedule(release_slots)
        else:
            final_schedule, _ = self._simulate_schedule(release_slots)
        return final_schedule

    def _improve_order(self, tenant_order, schedule):
        best_order = list(tenant_order)
        best_schedule = schedule
        best_score = self._objective(schedule)

        improved = True
        while improved:
            improved = False

            for idx in range(len(best_order) - 1):
                candidate_order = best_order.copy()
                candidate_order[idx], candidate_order[idx + 1] = candidate_order[idx + 1], candidate_order[idx]
                candidate_schedule = self._build_schedule_for_order(candidate_order)
                candidate_score = self._objective(candidate_schedule)
                if self._better_objective(candidate_score, best_score):
                    best_order = candidate_order
                    best_schedule = candidate_schedule
                    best_score = candidate_score
                    improved = True
                    break

            if improved:
                continue

            for src in range(len(best_order)):
                tenant = best_order[src]
                for dst in range(len(best_order)):
                    if src == dst:
                        continue
                    candidate_order = best_order.copy()
                    candidate_order.pop(src)
                    candidate_order.insert(dst, tenant)
                    candidate_schedule = self._build_schedule_for_order(candidate_order)
                    candidate_score = self._objective(candidate_schedule)
                    if self._better_objective(candidate_score, best_score):
                        best_order = candidate_order
                        best_schedule = candidate_schedule
                        best_score = candidate_score
                        improved = True
                        break
                if improved:
                    break

        return best_order, best_schedule

    def solve(self):
        best_order = None
        best_schedule = None
        best_score = (float("inf"), float("inf"))

        for tenant_order in self._candidate_orders():
            candidate_schedule = self._build_schedule_for_order(tenant_order)
            candidate_score = self._objective(candidate_schedule)
            if self._better_objective(candidate_score, best_score):
                best_order = tenant_order
                best_schedule = candidate_schedule
                best_score = candidate_score

        assert best_order is not None and best_schedule is not None
        best_order, best_schedule = self._improve_order(best_order, best_schedule)

        if self.M_tenants and self.verbose:
            avg_jct = sum(completion for _, completion in best_schedule.values()) / len(self.M_tenants)
            makespan = max(completion for _, completion in best_schedule.values())
            print(f"Harmonics heuristic order: {best_order}")
            print(f"Harmonics baseline solved: AvgJCT={avg_jct:.4f}, Makespan={makespan:.4f}")

        return best_schedule
