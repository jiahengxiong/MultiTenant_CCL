from __future__ import annotations

import gurobipy as gp
import numpy as np
from gurobipy import GRB


class HarmonicsBaselineILP:
    """
    Harmonics scheduling baseline.

    This is the baseline scheduler for tenant launch time and rate control.
    """

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
        self.cap = {}
        for src, dst in self.datacenter.topology.edges():
            capacity = float(self.datacenter.topology[src][dst]["capacity"])
            self.cap[(src, dst)] = capacity
            self.cap[(dst, src)] = capacity

        self.base_dur = {}
        self.demands = {}

        self._prepare_data()
        self._build_model()

    def _path_edges(self, src, dst):
        path = self.path_table.get((src, dst))
        if not path:
            return []
        return list(zip(path[:-1], path[1:]))

    def _prepare_data(self):
        for tenant in self.M:
            link_vol_bits = {}

            for logical_src, logical_dst, volume in self.tenant_flows[tenant]:
                volume_bits = volume * 1e9
                physical_src = self.tenant_mapping[tenant][logical_src]
                physical_dst = self.tenant_mapping[tenant][logical_dst]
                if physical_src == physical_dst:
                    continue

                for edge in self._path_edges(physical_src, physical_dst):
                    link_vol_bits[edge] = link_vol_bits.get(edge, 0.0) + float(volume_bits)

            base_duration = 0.0
            for edge, volume_bits in link_vol_bits.items():
                capacity = self.cap.get(edge, 0.0)
                if capacity > 0:
                    edge_duration = volume_bits / capacity
                    if edge_duration > base_duration:
                        base_duration = edge_duration

            base_duration = max(base_duration, 1e-9)
            self.base_dur[tenant] = base_duration

            demand = {}
            for edge, volume_bits in link_vol_bits.items():
                if volume_bits <= 0:
                    continue
                demand[edge] = (volume_bits / base_duration) / 1e9
            self.demands[tenant] = demand

        if self.verbose:
            print("[HarmonicsBaselineILP] base_dur:", self.base_dur)

    def _build_model(self):
        self.model = gp.Model("HarmonicsBaseline")
        self.model.Params.OutputFlag = 1 if self.verbose else 0

        min_base_duration = min(self.base_dur.values()) if self.base_dur else 1.0
        self.dt = min_base_duration / 20.0
        self.H = int((self.estimation * 5) / self.dt) + 10
        self.T_slots = range(self.H)

        if self.verbose:
            print(f"Building Harmonics baseline: dt={self.dt}s, Horizon={self.H}")

        self.rates = [round(rate, 2) for rate in np.linspace(1.0, self.rmin, 10)]
        self.K_levels = range(len(self.rates))

        self.dur_slots = {}
        for tenant in self.M:
            self.dur_slots[tenant] = {}
            for level in self.K_levels:
                rate_ratio = self.rates[level]
                slots_needed = int(self.base_dur[tenant] / (rate_ratio * self.dt)) + 1
                self.dur_slots[tenant][level] = slots_needed

        self.x = {}
        for tenant in self.M:
            for start_slot in self.T_slots:
                for level in self.K_levels:
                    if start_slot + self.dur_slots[tenant][level] > self.H:
                        continue
                    self.x[(tenant, start_slot, level)] = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{tenant}_{start_slot}_{level}",
                    )

        for tenant in self.M:
            lhs = gp.quicksum(
                self.x[(tenant, start_slot, level)]
                for start_slot in self.T_slots
                for level in self.K_levels
                if (tenant, start_slot, level) in self.x
            )
            self.model.addConstr(lhs == 1, name=f"StartOnce_{tenant}")

        link_tenants = {}
        for tenant in self.M:
            for edge in self.demands[tenant]:
                link_tenants.setdefault(edge, []).append(tenant)

        for edge, tenants in link_tenants.items():
            if len(tenants) < 2:
                continue

            capacity_gbps = self.cap[edge] / 1e9
            for current_slot in self.T_slots:
                terms = []
                for tenant in tenants:
                    load_gbps = self.demands[tenant][edge]
                    for level in self.K_levels:
                        actual_gbps = load_gbps * self.rates[level]
                        duration_slots = self.dur_slots[tenant][level]
                        start_min = max(0, current_slot - duration_slots + 1)
                        for start_slot in range(start_min, current_slot + 1):
                            if (tenant, start_slot, level) in self.x:
                                terms.append(actual_gbps * self.x[(tenant, start_slot, level)])

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
                * (start_slot + self.dur_slots[tenant][level])
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

    def solve(self, timelimit=120, mipgap=0.01):
        self.model.setParam("TimeLimit", timelimit)
        self.model.setParam("MIPGap", mipgap)
        self.model.optimize()

        if self.model.SolCount == 0:
            if self.verbose:
                print("[Harmonics] No solution found.")
            return None

        schedule = {}
        for tenant in self.M:
            link_vol_bits = {}
            for logical_src, logical_dst, volume in self.tenant_flows[tenant]:
                volume_bits = volume * 1e9
                physical_src = self.tenant_mapping[tenant][logical_src]
                physical_dst = self.tenant_mapping[tenant][logical_dst]
                if physical_src == physical_dst:
                    continue
                for edge in self._path_edges(physical_src, physical_dst):
                    link_vol_bits[edge] = link_vol_bits.get(edge, 0.0) + float(volume_bits)

            bottleneck_capacity = 1.0
            max_duration = -1.0
            for edge, volume_bits in link_vol_bits.items():
                capacity = self.cap.get(edge, 0.0)
                if capacity > 0:
                    duration = volume_bits / capacity
                    if duration > max_duration:
                        max_duration = duration
                        bottleneck_capacity = capacity

            for start_slot in self.T_slots:
                for level in self.K_levels:
                    if (tenant, start_slot, level) not in self.x:
                        continue
                    if self.x[(tenant, start_slot, level)].X <= 0.5:
                        continue

                    start_time = start_slot * self.dt
                    physical_rate = self.rates[level] * bottleneck_capacity
                    schedule[tenant] = (start_time * 0.98, physical_rate)
                    if self.verbose:
                        end_time = start_time + self.dur_slots[tenant][level] * self.dt
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

        self.durations = {}
        self.durations_per_link = {}
        self.link_users = {}

        self._prepare_data()

    def _prepare_data(self):
        self.M_tenants = sorted(self.tenant_mapping.keys())
        self.L_links = list(self.datacenter.topology.edges())
        self.cap = {
            edge: self.datacenter.topology[edge[0]][edge[1]]["capacity"] for edge in self.L_links
        }

        link_users = {edge: set() for edge in self.L_links}
        self.durations_per_link = {}

        for tenant in self.M_tenants:
            self.durations_per_link[tenant] = {}
            link_load = {}
            for logical_src, logical_dst, volume in self.tenant_flows[tenant]:
                physical_src = self.tenant_mapping[tenant][logical_src]
                physical_dst = self.tenant_mapping[tenant][logical_dst]
                if physical_src == physical_dst:
                    continue
                path = self.path_table.get((physical_src, physical_dst))
                if not path:
                    continue

                for idx in range(len(path) - 1):
                    edge = (path[idx], path[idx + 1])
                    link_load[edge] = link_load.get(edge, 0.0) + (volume / 8.0)
                    link_users[edge].add(tenant)

            max_duration = 0.0
            for edge, load in link_load.items():
                if edge in self.cap and self.cap[edge] > 0:
                    duration = (load * 1e9 * 8.0) / self.cap[edge]
                    self.durations_per_link[tenant][edge] = duration
                    if duration > max_duration:
                        max_duration = duration

            self.durations[tenant] = max(max_duration, 1e-6)

        self.link_users = link_users

        if self.verbose:
            print(f"Harmonics baseline prep: {len(self.M_tenants)} tenants.")
            print(f"Durations (max): {self.durations}")

    def solve(self):
        sorted_tenants = sorted(self.M_tenants, key=lambda tenant: self.durations[tenant])
        dt = 0.001
        horizon_slots = int(sum(self.durations.values()) / dt) + 2000
        link_timeline = {}
        final_schedule = {}

        if self.verbose:
            print(f"Harmonics bandwidth-aware order: {sorted_tenants}")

        for tenant in sorted_tenants:
            demands = {}
            tenant_links = set()

            for logical_src, logical_dst, _volume in self.tenant_flows[tenant]:
                physical_src = self.tenant_mapping[tenant][logical_src]
                physical_dst = self.tenant_mapping[tenant][logical_dst]
                if physical_src == physical_dst:
                    continue
                path = self.path_table.get((physical_src, physical_dst))
                if not path:
                    continue
                for idx in range(len(path) - 1):
                    tenant_links.add((path[idx], path[idx + 1]))

            duration = self.durations[tenant]
            for edge in tenant_links:
                edge_duration = self.durations_per_link[tenant].get(edge, 0.0)
                if duration > 1e-9:
                    demand = edge_duration / duration
                    if demand > 1e-5:
                        demands[edge] = demand

            duration_slots = int(duration / dt) + 1
            start_slot = 0

            while True:
                fits = True
                for edge, demand in demands.items():
                    if edge not in link_timeline:
                        link_timeline[edge] = [0.0] * horizon_slots

                    required_len = start_slot + duration_slots
                    if required_len > len(link_timeline[edge]):
                        link_timeline[edge].extend([0.0] * (required_len - len(link_timeline[edge]) + 1000))

                    for slot in range(start_slot, start_slot + duration_slots):
                        if link_timeline[edge][slot] + demand > 1.0:
                            fits = False
                            break
                    if not fits:
                        break

                if fits:
                    break
                start_slot += 1

            for edge, demand in demands.items():
                for slot in range(start_slot, start_slot + duration_slots):
                    link_timeline[edge][slot] += demand

            start_time = start_slot * dt
            end_time = (start_slot + duration_slots) * dt
            final_schedule[tenant] = (start_time, end_time)

        if self.M_tenants and self.verbose:
            avg_jct = sum(completion for _, completion in final_schedule.values()) / len(self.M_tenants)
            makespan = max(completion for _, completion in final_schedule.values())
            print(f"Harmonics baseline solved: AvgJCT={avg_jct:.4f}, Makespan={makespan:.4f}")

        return final_schedule
