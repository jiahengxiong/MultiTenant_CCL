from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from multitenant.workloads import build_collective_stage_flows


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
):
    capacities = {
        (src, dst): float(datacenter.topology[src][dst]["capacity"])
        for src, dst in datacenter.topology.edges()
    }

    if collective in {"allgather", "allreduce"} and single_flow_size is not None:
        stage_flows_by_tenant = build_collective_stage_flows(
            tenant_mapping,
            int(single_flow_size),
            collective,
        )
    else:
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
            for edge, volume_bits in link_vol_bits.items():
                if volume_bits <= 0:
                    continue
                capacity = capacities.get(edge, 0.0)
                stage_demand_gbps[edge] = (volume_bits / stage_duration) / 1e9
                if capacity > 0:
                    stage_load_fraction[edge] = volume_bits / (stage_duration * capacity)

            tenant_stages.append(
                {
                    "base_duration": stage_duration,
                    "demand_gbps": stage_demand_gbps,
                    "load_fraction": stage_load_fraction,
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


class HarmonicsBaselineILP:
    """Harmonics scheduling baseline with stage-aware collective dependencies."""

    def __init__(
        self,
        datacenter,
        tenant_mapping,
        tenant_flows,
        path_table,
        single_flow_size=None,
        collective=None,
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

        self.model.setObjective(
            gp.quicksum(objective_terms) + self.makespan_weight * makespan,
            GRB.MINIMIZE,
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
    def __init__(
        self,
        datacenter,
        tenant_mapping,
        tenant_flows,
        path_table,
        single_flow_size,
        collective="allreduce",
        verbose=True,
    ):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.path_table = path_table
        self.single_flow_size = single_flow_size
        self.collective = collective
        self.verbose = verbose

        self.M_tenants = sorted(self.tenant_mapping.keys())
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
        )

        if self.verbose:
            print(f"Harmonics baseline prep: {len(self.M_tenants)} tenants.")
            print(f"Durations (max): {self.durations}")

    def solve(self):
        sorted_tenants = sorted(self.M_tenants, key=lambda tenant: self.durations[tenant])
        min_stage_duration = min(
            stage["base_duration"]
            for tenant in self.M_tenants
            for stage in self.stage_data[tenant]
        )
        dt = max(min_stage_duration / 5.0, 1e-4)
        horizon_slots = int(sum(self.durations.values()) / dt) + 2000
        link_timeline = {}
        final_schedule = {}

        if self.verbose:
            print(f"Harmonics bandwidth-aware order: {sorted_tenants}")

        for tenant in sorted_tenants:
            stage_slots = [int(stage["base_duration"] / dt) + 1 for stage in self.stage_data[tenant]]
            stage_offsets = []
            current_offset = 0
            for duration_slots in stage_slots:
                stage_offsets.append(current_offset)
                current_offset += duration_slots

            start_slot = 0
            while True:
                fits = True
                for stage_idx, stage in enumerate(self.stage_data[tenant]):
                    offset = stage_offsets[stage_idx]
                    duration_slots = stage_slots[stage_idx]
                    for edge, demand in stage["load_fraction"].items():
                        if edge not in link_timeline:
                            link_timeline[edge] = [0.0] * horizon_slots

                        required_len = start_slot + offset + duration_slots
                        if required_len > len(link_timeline[edge]):
                            link_timeline[edge].extend([0.0] * (required_len - len(link_timeline[edge]) + 1000))

                        for slot in range(start_slot + offset, start_slot + offset + duration_slots):
                            if link_timeline[edge][slot] + demand > 1.0 + 1e-9:
                                fits = False
                                break
                        if not fits:
                            break
                    if not fits:
                        break

                if fits:
                    break
                start_slot += 1

            for stage_idx, stage in enumerate(self.stage_data[tenant]):
                offset = stage_offsets[stage_idx]
                duration_slots = stage_slots[stage_idx]
                for edge, demand in stage["load_fraction"].items():
                    for slot in range(start_slot + offset, start_slot + offset + duration_slots):
                        link_timeline[edge][slot] += demand

            start_time = start_slot * dt
            end_time = (start_slot + current_offset) * dt
            final_schedule[tenant] = (start_time, end_time)

        if self.M_tenants and self.verbose:
            avg_jct = sum(completion for _, completion in final_schedule.values()) / len(self.M_tenants)
            makespan = max(completion for _, completion in final_schedule.values())
            print(f"Harmonics baseline solved: AvgJCT={avg_jct:.4f}, Makespan={makespan:.4f}")

        return final_schedule
