from __future__ import annotations

import random
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from multitenant.workloads import build_collective_stage_flows


class MappingCGSolver:
    """Column-generation mapping solver used for larger-scale experiments."""

    def __init__(
        self,
        datacenter,
        tenant_mapping,
        tenant_flows=None,
        verbose=False,
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
        self.tenant_mapping = tenant_mapping
        self.initial_tenant_mapping = {
            tenant: dict(mapping) for tenant, mapping in tenant_mapping.items()
        }
        self.tenant_flows = tenant_flows
        self.collective = collective
        self.single_flow_size = single_flow_size
        self.tenant_collective_specs = tenant_collective_specs
        self.stage_flows = stage_flows or self._derive_stage_flows()
        self.verbose = verbose
        self.fairness_lambda = float(fairness_lambda)
        self.fairness_iterations = max(1, int(fairness_iterations))
        self.fairness_grouping = fairness_grouping
        self.slot_duration_override = slot_duration
        self.horizon_slots_override = horizon_slots

        self.BigM = 1e9
        self.scale = 1e9

        self.M = sorted(tenant_mapping.keys())
        self.L = list(datacenter.topology.edges())
        self.link_to_idx = {link: idx for idx, link in enumerate(self.L)}
        self.num_links = len(self.L)
        self.cap = {
            link: datacenter.topology[src][dst]["capacity"] / self.scale
            for (src, dst), link in zip(self.L, self.L)
        }

        self.all_servers = self.datacenter.get_all_servers()
        self.server_to_idx = {server: idx for idx, server in enumerate(self.all_servers)}
        self.num_servers = len(self.all_servers)
        self.max_stage_count = max((len(self.stage_flows[tenant]) for tenant in self.M), default=0)

        self.stage_flows_by_tenant = {}
        self.stage_flow_matrices = {}

        for tenant, stage_list in self.stage_flows.items():
            rank_count = len(self.tenant_mapping[tenant])
            formatted_stages = []
            stage_matrices = []
            for stage in stage_list:
                formatted_stage = [(src, dst, volume) for (src, dst, volume) in stage]
                formatted_stages.append(formatted_stage)

                flow_matrix = np.zeros((rank_count, rank_count))
                for src, dst, volume in formatted_stage:
                    if src < rank_count and dst < rank_count:
                        flow_matrix[src, dst] += volume
                stage_matrices.append(flow_matrix)

            self.stage_flows_by_tenant[tenant] = formatted_stages
            self.stage_flow_matrices[tenant] = stage_matrices

        self.path_edges_indices = [[[] for _ in range(self.num_servers)] for _ in range(self.num_servers)]
        rows = []
        cols = []
        data = []

        for src in self.all_servers:
            src_idx = self.server_to_idx[src]
            for dst in self.all_servers:
                dst_idx = self.server_to_idx[dst]
                if src == dst:
                    continue

                path = self.datacenter.paths.get((src, dst))
                if not path:
                    continue

                edges = self.datacenter.path_to_edges(path)
                indices = [self.link_to_idx[edge] for edge in edges if edge in self.link_to_idx]
                self.path_edges_indices[src_idx][dst_idx] = indices

                row_idx = src_idx * self.num_servers + dst_idx
                for edge_idx in indices:
                    rows.append(row_idx)
                    cols.append(edge_idx)
                    data.append(1.0)

        self.path_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_servers * self.num_servers, self.num_links),
        )

        self.patterns = {tenant: [] for tenant in self.M}
        self.added_patterns_hashes = {tenant: set() for tenant in self.M}

        self.rmp = None
        self.lambdas = {}
        self.T_stage = {}
        self.T_m = {}
        self.constr_convex = {}
        self.constr_link = {}
        self.constr_server = {}
        self.final_obj = None
        self.final_makespan = None
        self.fairness_weights = self._compute_fairness_weights(self.tenant_mapping)

    def _derive_stage_flows(self):
        if self.tenant_collective_specs is not None or (
            self.collective in {"allgather", "reducescatter", "allreduce", "alltoall"}
            and self.single_flow_size is not None
        ):
            return build_collective_stage_flows(
                self.tenant_mapping,
                None if self.single_flow_size is None else int(self.single_flow_size),
                self.collective,
                tenant_collective_specs=self.tenant_collective_specs,
            )
        if self.tenant_flows is None:
            raise ValueError(
                "tenant_flows is required when collective/single_flow_size are not provided"
            )
        return {tenant: [self.tenant_flows[tenant]] for tenant in self.tenant_mapping}

    def _slot_model_config(self):
        min_capacity = min(self.cap.values()) if self.cap else 1.0
        min_stage_volume = min(
            (
                float(volume)
                for tenant in self.M
                for stage in self.stage_flows_by_tenant[tenant]
                for _src, _dst, volume in stage
                if float(volume) > 0.0
            ),
            default=1.0,
        )

        if self.slot_duration_override is not None:
            slot_duration = float(self.slot_duration_override)
        else:
            slot_duration = max(min_stage_volume / min_capacity, 1e-6)

        if self.horizon_slots_override is not None:
            horizon_slots = int(self.horizon_slots_override)
        else:
            total_stage_volume = sum(
                float(volume)
                for tenant in self.M
                for stage in self.stage_flows_by_tenant[tenant]
                for _src, _dst, volume in stage
            )
            stage_slack = sum(len(self.stage_flows_by_tenant[tenant]) for tenant in self.M) + len(self.M) + 2
            horizon_slots = int(np.ceil(total_stage_volume / min_capacity / slot_duration)) + stage_slack
            horizon_slots = max(horizon_slots, 4)

        return slot_duration, horizon_slots

    def _mapping_to_stage_link_loads(self, mapping_by_tenant):
        loads = {}
        for tenant in self.M:
            tenant_stage_loads = []
            mapping = mapping_by_tenant[tenant]
            for stage in self.stage_flows_by_tenant[tenant]:
                link_loads = {}
                for src, dst, volume in stage:
                    mapped_src = mapping[src]
                    mapped_dst = mapping[dst]
                    if mapped_src == mapped_dst:
                        continue
                    src_idx = self.server_to_idx[mapped_src]
                    dst_idx = self.server_to_idx[mapped_dst]
                    for edge_idx in self.path_edges_indices[src_idx][dst_idx]:
                        link_loads[edge_idx] = link_loads.get(edge_idx, 0.0) + float(volume)
                tenant_stage_loads.append(link_loads)
            loads[tenant] = tenant_stage_loads
        return loads

    @staticmethod
    def _is_better_objective(candidate, incumbent, tol=1e-9):
        cand_max, cand_avg = candidate
        inc_max, inc_avg = incumbent
        if cand_max < inc_max - tol:
            return True
        if cand_max > inc_max + tol:
            return False
        return cand_avg < inc_avg - tol

    def _evaluate_time_slot_objective(self, mapping_by_tenant):
        slot_duration, horizon_slots = self._slot_model_config()
        stage_link_loads = self._mapping_to_stage_link_loads(mapping_by_tenant)
        remaining = {
            tenant: [dict(link_loads) for link_loads in stage_link_loads[tenant]]
            for tenant in self.M
        }
        current_stage = {tenant: 0 for tenant in self.M}
        finish_times = {}
        pressure_by_stage = {
            stage_idx: np.zeros(self.num_links)
            for stage_idx in range(self.max_stage_count)
        }

        for slot in range(horizon_slots):
            active = []
            link_to_active = {}

            for tenant in self.M:
                stage_idx = current_stage[tenant]
                if stage_idx >= len(remaining[tenant]):
                    continue

                link_loads = remaining[tenant][stage_idx]
                active_links = [edge_idx for edge_idx, volume in link_loads.items() if volume > 1e-12]
                if not active_links:
                    current_stage[tenant] += 1
                    if current_stage[tenant] >= len(remaining[tenant]):
                        finish_times.setdefault(tenant, slot * slot_duration)
                    continue

                active.append((tenant, stage_idx, active_links))
                for edge_idx in active_links:
                    link_to_active.setdefault(edge_idx, []).append((tenant, stage_idx))

            if not active:
                if len(finish_times) == len(self.M):
                    break
                continue

            for edge_idx, users in link_to_active.items():
                occupancy = len(users)
                for _tenant, user_stage in users:
                    pressure_by_stage[user_stage][edge_idx] += occupancy

            for edge_idx, users in link_to_active.items():
                share = self.cap[self.L[edge_idx]] * slot_duration / max(len(users), 1)
                for tenant, stage_idx in users:
                    remaining_load = remaining[tenant][stage_idx][edge_idx]
                    remaining[tenant][stage_idx][edge_idx] = max(0.0, remaining_load - share)

            for tenant, stage_idx, _active_links in active:
                if all(volume <= 1e-12 for volume in remaining[tenant][stage_idx].values()):
                    current_stage[tenant] += 1
                    if current_stage[tenant] >= len(remaining[tenant]):
                        finish_times[tenant] = (slot + 1) * slot_duration

            if len(finish_times) == len(self.M):
                break

        if len(finish_times) != len(self.M):
            return float("inf"), float("inf"), finish_times, pressure_by_stage

        avg_finish = float(sum(finish_times.values()) / len(self.M))
        makespan = float(max(finish_times.values(), default=0.0))
        return makespan, avg_finish, finish_times, pressure_by_stage

    def _generate_time_slot_candidates(self, tenant, incumbent_mapping, pressure_by_stage):
        rank_count = len(self.tenant_mapping[tenant])
        fixed_servers = sorted(self.initial_tenant_mapping[tenant].values())
        fixed_server_indices = [self.server_to_idx[server] for server in fixed_servers]

        path_cost_by_stage = {}
        for stage_idx in range(len(self.stage_flows_by_tenant[tenant])):
            weights = pressure_by_stage.get(stage_idx, np.zeros(self.num_links))
            path_cost_flat = self.path_matrix.dot(weights)
            path_cost_by_stage[stage_idx] = path_cost_flat.reshape((self.num_servers, self.num_servers))

        incumbent_locations = np.array(
            [self.server_to_idx[incumbent_mapping[rank]] for rank in range(rank_count)]
        )

        num_starts = 8
        start_mappings = [incumbent_locations]
        for _ in range(num_starts - 1):
            shuffled = fixed_server_indices.copy()
            random.shuffle(shuffled)
            start_mappings.append(np.array(shuffled))

        candidates = {tuple(int(x) for x in incumbent_locations)}

        for current_locations in start_mappings:
            current_locations = current_locations.copy()
            for _ in range(8):
                traffic_costs = np.zeros((rank_count, rank_count))
                for stage_idx, flow_matrix in enumerate(self.stage_flow_matrices[tenant]):
                    distance_subset = path_cost_by_stage[stage_idx][fixed_server_indices, :][:, current_locations]
                    traffic_costs += flow_matrix @ distance_subset.T

                _row_ind, col_ind = linear_sum_assignment(traffic_costs)
                new_locations = np.array([fixed_server_indices[col] for col in col_ind])
                if np.array_equal(new_locations, current_locations):
                    break
                current_locations = new_locations

            def score(locations):
                total = 0.0
                for stage_idx, stage in enumerate(self.stage_flows_by_tenant[tenant]):
                    for src, dst, volume in stage:
                        total += float(volume) * path_cost_by_stage[stage_idx][locations[src], locations[dst]]
                return total

            current_score = score(current_locations)
            improved = True
            while improved:
                improved = False
                for _ in range(max(20, rank_count * 2)):
                    idx1, idx2 = random.sample(range(rank_count), 2)
                    test_locations = current_locations.copy()
                    test_locations[idx1], test_locations[idx2] = test_locations[idx2], test_locations[idx1]
                    test_score = score(test_locations)
                    if test_score < current_score - 1e-9:
                        current_locations = test_locations
                        current_score = test_score
                        improved = True
                        break

            candidates.add(tuple(int(x) for x in current_locations))

        mapping_candidates = []
        for locations in candidates:
            mapping = {rank: self.all_servers[server_idx] for rank, server_idx in enumerate(locations)}
            mapping_candidates.append(mapping)

        return mapping_candidates

    def _stage_group_key(self, tenant, stage_idx):
        if self.fairness_grouping == "stage":
            return f"stage_{stage_idx}"

        if self.collective == "allreduce":
            stage_count = len(self.stage_flows_by_tenant[tenant])
            midpoint = max(stage_count // 2, 1)
            return "RS" if stage_idx < midpoint else "AG"
        if self.collective == "allgather":
            return "AG"
        return "GENERIC"

    def _compute_fairness_weights(self, mapping):
        counts = {}

        for tenant, stage_list in self.stage_flows_by_tenant.items():
            physical_mapping = mapping[tenant]
            for stage_idx, stage in enumerate(stage_list):
                group_key = self._stage_group_key(tenant, stage_idx)
                used_edge_indices = set()
                for src, dst, _volume in stage:
                    mapped_src = physical_mapping[src]
                    mapped_dst = physical_mapping[dst]
                    if mapped_src == mapped_dst:
                        continue

                    src_idx = self.server_to_idx[mapped_src]
                    dst_idx = self.server_to_idx[mapped_dst]
                    used_edge_indices.update(self.path_edges_indices[src_idx][dst_idx])

                for edge_idx in used_edge_indices:
                    link = self.L[edge_idx]
                    counts[(group_key, link)] = counts.get((group_key, link), 0) + 1

        weights = {}
        for key, count in counts.items():
            weights[key] = 1.0 + self.fairness_lambda * max(0, count - 1)
        return weights

    @staticmethod
    def _fairness_weights_close(lhs, rhs, tol=1e-9):
        all_keys = set(lhs) | set(rhs)
        for key in all_keys:
            if abs(lhs.get(key, 1.0) - rhs.get(key, 1.0)) > tol:
                return False
        return True

    def _reset_master_state(self):
        if self.rmp is not None:
            self.rmp.dispose()
        self.patterns = {tenant: [] for tenant in self.M}
        self.added_patterns_hashes = {tenant: set() for tenant in self.M}
        self.rmp = None
        self.lambdas = {}
        self.T_stage = {}
        self.T_m = {}
        self.constr_convex = {}
        self.constr_link = {}
        self.constr_server = {}
        self.final_obj = None

    def initialize_columns(self, seed_patterns=None):
        seed_patterns = seed_patterns or {}
        for tenant in self.M:
            self.add_pattern_to_list(tenant, self.initial_tenant_mapping[tenant].copy())

            current_mapping = self.tenant_mapping[tenant].copy()
            if current_mapping != self.initial_tenant_mapping[tenant]:
                self.add_pattern_to_list(tenant, current_mapping)

            for mapping in seed_patterns.get(tenant, []):
                self.add_pattern_to_list(tenant, dict(mapping))

    def add_pattern_to_list(self, tenant, mapping):
        stage_traffic = {}
        stage_traffic_indices = {}

        for stage_idx, stage in enumerate(self.stage_flows_by_tenant[tenant]):
            traffic = {}
            for src, dst, volume in stage:
                mapped_src, mapped_dst = mapping[src], mapping[dst]
                if mapped_src == mapped_dst:
                    continue

                src_idx = self.server_to_idx[mapped_src]
                dst_idx = self.server_to_idx[mapped_dst]
                edge_indices = self.path_edges_indices[src_idx][dst_idx]

                for edge_idx in edge_indices:
                    traffic[edge_idx] = traffic.get(edge_idx, 0.0) + volume

            stage_traffic_indices[stage_idx] = traffic
            stage_traffic[stage_idx] = {self.L[idx]: volume for idx, volume in traffic.items()}

        servers_used = {server: 0 for server in self.all_servers}
        for server in mapping.values():
            servers_used[server] = 1

        pattern_data = {
            "mapping": mapping,
            "stage_traffic": stage_traffic,
            "stage_traffic_indices": stage_traffic_indices,
            "servers": servers_used,
        }

        mapping_hash = tuple(sorted(mapping.items()))
        if mapping_hash in self.added_patterns_hashes[tenant]:
            return -1

        self.patterns[tenant].append(pattern_data)
        self.added_patterns_hashes[tenant].add(mapping_hash)
        return len(self.patterns[tenant]) - 1

    def initialize_rmp(self):
        self.rmp = gp.Model("MappingCG")
        self.rmp.Params.OutputFlag = 0
        self.rmp.Params.Method = 1

        for tenant in self.M:
            self.T_m[tenant] = gp.LinExpr()
            for stage_idx in range(len(self.stage_flows_by_tenant[tenant])):
                var = self.rmp.addVar(lb=0.0, name=f"T_{tenant}_{stage_idx}")
                self.T_stage[(tenant, stage_idx)] = var
                self.T_m[tenant] += var

        for tenant in self.M:
            for pattern_idx, _pattern in enumerate(self.patterns[tenant]):
                self.lambdas[(tenant, pattern_idx)] = self.rmp.addVar(
                    lb=0.0,
                    ub=1.0,
                    name=f"lam_{tenant}_{pattern_idx}",
                )

        self.rmp.update()
        tenant_count = max(len(self.M), 1)
        self.T_cg_max = self.rmp.addVar(lb=0.0, name="T_cg_max")
        for tenant in self.M:
            self.rmp.addConstr(self.T_cg_max >= self.T_m[tenant], name=f"T_cg_max_ge_{tenant}")
        self.rmp.ModelSense = GRB.MINIMIZE
        self.rmp.setObjectiveN(self.T_cg_max, index=0, priority=2, name="makespan")
        self.rmp.setObjectiveN(
            gp.quicksum(self.T_stage.values()) / tenant_count,
            index=1,
            priority=1,
            name="avg_jct",
        )

        for tenant in self.M:
            expr = gp.quicksum(
                self.lambdas[(tenant, pattern_idx)] for pattern_idx in range(len(self.patterns[tenant]))
            )
            self.constr_convex[tenant] = self.rmp.addConstr(expr == 1.0, name=f"convex_{tenant}")

        for tenant in self.M:
            for stage_idx in range(len(self.stage_flows_by_tenant[tenant])):
                group_key = self._stage_group_key(tenant, stage_idx)
                for link in self.L:
                    fairness_weight = self.fairness_weights.get((group_key, link), 1.0)
                    own_load_expr = gp.LinExpr()
                    for pattern_idx, pattern in enumerate(self.patterns[tenant]):
                        volume = pattern["stage_traffic"].get(stage_idx, {}).get(link, 0.0)
                        if volume > 1e-9:
                            own_load_expr.addTerms(
                                fairness_weight * volume,
                                self.lambdas[(tenant, pattern_idx)],
                            )

                    if own_load_expr.size() == 0:
                        continue

                    self.constr_link[(tenant, stage_idx, link)] = self.rmp.addConstr(
                        own_load_expr <= self.cap[link] * self.T_stage[(tenant, stage_idx)],
                        name=f"link_{stage_idx}_{link}_{tenant}",
                    )

        for server in self.all_servers:
            expr = gp.LinExpr()
            for tenant in self.M:
                for pattern_idx, pattern in enumerate(self.patterns[tenant]):
                    if pattern["servers"].get(server, 0) > 0.5:
                        expr.addTerms(1.0, self.lambdas[(tenant, pattern_idx)])
            self.constr_server[server] = self.rmp.addConstr(expr <= 1.0, name=f"server_{server}")

    def add_column_to_rmp(self, tenant, pattern_idx):
        pattern = self.patterns[tenant][pattern_idx]
        column = gp.Column()

        column.addTerms(1.0, self.constr_convex[tenant])

        for stage_idx, traffic in pattern["stage_traffic_indices"].items():
            group_key = self._stage_group_key(tenant, stage_idx)
            for edge_idx, volume in traffic.items():
                if volume <= 1e-9:
                    continue
                link = self.L[edge_idx]
                fairness_weight = self.fairness_weights.get((group_key, link), 1.0)

                if (tenant, stage_idx, link) in self.constr_link:
                    column.addTerms(
                        fairness_weight * volume,
                        self.constr_link[(tenant, stage_idx, link)],
                    )

        for server, used in pattern["servers"].items():
            if used > 0.5 and server in self.constr_server:
                column.addTerms(1.0, self.constr_server[server])

        self.lambdas[(tenant, pattern_idx)] = self.rmp.addVar(
            obj=0.0,
            column=column,
            lb=0.0,
            ub=1.0,
            name=f"lam_{tenant}_{pattern_idx}",
        )

    def solve_pricing(self, duals_convex, duals_link, duals_server):
        new_columns_count = 0

        server_weights = np.array([-duals_server.get(server, 0.0) for server in self.all_servers])

        tenant_order = list(self.M)
        random.shuffle(tenant_order)

        improvements_found = 0
        max_improvements_per_iter = max(5, len(self.M) // 2)

        for tenant in tenant_order:
            if improvements_found >= max_improvements_per_iter:
                break

            rank_count = len(self.tenant_mapping[tenant])
            fixed_servers = list(self.tenant_mapping[tenant].values())
            fixed_server_indices = [self.server_to_idx[server] for server in fixed_servers]

            total_path_cost_by_stage = {}
            for stage_idx in range(len(self.stage_flows_by_tenant[tenant])):
                weighted_duals = np.zeros(self.num_links)
                group_key = self._stage_group_key(tenant, stage_idx)
                for link in self.L:
                    link_idx = self.link_to_idx[link]
                    fairness_weight = self.fairness_weights.get((group_key, link), 1.0)
                    weighted_duals[link_idx] = (
                        -fairness_weight * duals_link.get((tenant, stage_idx, link), 0.0)
                    )
                path_cost_flat = self.path_matrix.dot(weighted_duals)
                total_path_cost_by_stage[stage_idx] = path_cost_flat.reshape(
                    (self.num_servers, self.num_servers)
                )

            num_starts = 10
            start_mappings = [np.array(fixed_server_indices)]
            for _ in range(num_starts - 1):
                shuffled = fixed_server_indices.copy()
                random.shuffle(shuffled)
                start_mappings.append(np.array(shuffled))

            candidates = []
            for current_locations in start_mappings:
                for _ in range(10):
                    traffic_costs = np.zeros((rank_count, rank_count))
                    for stage_idx, flow_matrix in enumerate(self.stage_flow_matrices[tenant]):
                        distance_subset = total_path_cost_by_stage[stage_idx][fixed_server_indices, :][:, current_locations]
                        traffic_costs += flow_matrix @ distance_subset.T

                    server_costs = server_weights[fixed_server_indices]
                    cost_matrix = traffic_costs + server_costs

                    _row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    new_locations = np.array([fixed_server_indices[col] for col in col_ind])

                    if np.array_equal(new_locations, current_locations):
                        break
                    current_locations = new_locations

                def calc_rc(locations):
                    rc = 0.0
                    rc += server_weights[locations].sum()

                    for stage_idx, stage in enumerate(self.stage_flows_by_tenant[tenant]):
                        for src, dst, volume in stage:
                            src_idx = locations[src]
                            dst_idx = locations[dst]
                            if src_idx == dst_idx:
                                continue

                            rc += volume * total_path_cost_by_stage[stage_idx][src_idx, dst_idx]

                    rc -= duals_convex[tenant]
                    return rc

                current_rc = calc_rc(current_locations)
                improved = True
                while improved:
                    improved = False
                    for _ in range(max(50, rank_count * 2)):
                        idx1, idx2 = random.sample(range(rank_count), 2)
                        test_locations = current_locations.copy()
                        test_locations[idx1], test_locations[idx2] = (
                            test_locations[idx2],
                            test_locations[idx1],
                        )
                        test_rc = calc_rc(test_locations)
                        if test_rc < current_rc - 1e-9:
                            current_rc = test_rc
                            current_locations = test_locations
                            improved = True
                            break

                if current_rc < -1e-9:
                    mapping_dict = tuple(
                        (rank, self.all_servers[server_idx]) for rank, server_idx in enumerate(current_locations)
                    )
                    candidates.append((current_rc, mapping_dict))

            candidates.sort(key=lambda item: item[0])
            added_count = 0
            seen_mappings = set()

            for reduced_cost, mapping_tuple in candidates:
                if added_count >= 3:
                    break
                if mapping_tuple in seen_mappings:
                    continue

                mapping = dict(mapping_tuple)
                pattern_idx = self.add_pattern_to_list(tenant, mapping)
                if pattern_idx == -1:
                    continue

                self.add_column_to_rmp(tenant, pattern_idx)
                new_columns_count += 1
                added_count += 1
                seen_mappings.add(mapping_tuple)

            if added_count > 0:
                improvements_found += 1

        return new_columns_count

    def _solve_with_current_weights(self, max_iter, seed_patterns=None):
        self.initialize_columns(seed_patterns=seed_patterns)
        self.initialize_rmp()

        iteration = 0
        start_time = time.time()
        obj_history = []

        while iteration < max_iter:
            iteration += 1
            self.rmp.optimize()
            if self.rmp.Status != GRB.OPTIMAL:
                if self.verbose:
                    print(f"Status {self.rmp.Status}")
                break

            obj_val = self.rmp.ObjVal
            if self.verbose:
                print(f"Iter {iteration}: {obj_val:.6f}")

            obj_history.append(obj_val)
            if iteration > 20 and len(obj_history) >= 8:
                improvement = (obj_history[-8] - obj_history[-1]) / abs(obj_history[-8])
                if improvement < 0.0005:
                    if self.verbose:
                        print("Proposed Mapping (CG): makespan objective converged due to stagnation.")
                    break

            duals_convex = {tenant: self.constr_convex[tenant].Pi for tenant in self.M}
            duals_link = {
                (tenant, stage_idx, link): constr.Pi
                for (tenant, stage_idx, link), constr in self.constr_link.items()
            }
            duals_server = {server: self.constr_server[server].Pi for server in self.all_servers}

            new_columns = self.solve_pricing(duals_convex, duals_link, duals_server)
            if new_columns == 0:
                break

        if self.verbose:
            print(
                f"Proposed Mapping (CG): master loop finished in {time.time() - start_time:.2f}s "
                f"after {iteration} iterations."
            )

        for variable in self.rmp.getVars():
            if variable.VarName.startswith("lam_"):
                variable.VType = GRB.BINARY
        self.rmp.update()
        self.rmp.Params.MIPGap = 0.05
        self.rmp.Params.MIPFocus = 1
        self.rmp.Params.TimeLimit = 3.0
        self.rmp.Params.OutputFlag = 0
        self.rmp.optimize()

        if self.rmp.SolCount > 0:
            self.final_obj = self.rmp.ObjVal
            if self.verbose:
                print(
                    f"Proposed Mapping (CG): final makespan objective = {self.final_obj:.6f} "
                    f"(status {self.rmp.Status})."
                )
            return self.extract_solution()

        if self.verbose:
            print(f"Proposed Mapping (CG): integer solve failed (status {self.rmp.Status}).")
        self.final_obj = None
        self.final_makespan = None
        return None

    def solve(self, max_iter=100):
        current_mapping = {
            tenant: dict(self.initial_tenant_mapping[tenant]) for tenant in self.M
        }
        best_makespan, best_avg, _finish_times, pressure_by_stage = self._evaluate_time_slot_objective(current_mapping)
        best_objective = (best_makespan, best_avg)
        self.final_obj = best_makespan if np.isfinite(best_makespan) else None
        self.final_makespan = best_makespan if np.isfinite(best_makespan) else None

        if self.verbose:
            print(
                "Proposed Mapping (CG): initial lexicographic objective = "
                f"(makespan={best_makespan:.6f}, avg_jct={best_avg:.6f})"
            )

        for iteration in range(max_iter):
            improved = False
            tenant_order = list(self.M)
            random.shuffle(tenant_order)

            for tenant in tenant_order:
                candidates = self._generate_time_slot_candidates(
                    tenant,
                    current_mapping[tenant],
                    pressure_by_stage,
                )

                best_tenant_mapping = current_mapping[tenant]
                best_tenant_objective = best_objective
                best_tenant_pressure = pressure_by_stage

                for candidate in candidates:
                    if candidate == current_mapping[tenant]:
                        continue

                    trial_mapping = {
                        other_tenant: (
                            dict(candidate) if other_tenant == tenant else dict(current_mapping[other_tenant])
                        )
                        for other_tenant in self.M
                    }
                    trial_makespan, trial_avg, _trial_finish, trial_pressure = self._evaluate_time_slot_objective(trial_mapping)
                    trial_objective = (trial_makespan, trial_avg)
                    if self._is_better_objective(trial_objective, best_tenant_objective):
                        best_tenant_objective = trial_objective
                        best_tenant_mapping = candidate
                        best_tenant_pressure = trial_pressure

                if self._is_better_objective(best_tenant_objective, best_objective):
                    current_mapping[tenant] = dict(best_tenant_mapping)
                    best_objective = best_tenant_objective
                    pressure_by_stage = best_tenant_pressure
                    improved = True
                    if self.verbose:
                        print(
                            f"Proposed Mapping (CG): iter {iteration + 1}, tenant {tenant} improved "
                            "lexicographic objective to "
                            f"(makespan={best_objective[0]:.6f}, avg_jct={best_objective[1]:.6f})"
                        )

            if not improved:
                break

        self.final_obj = best_objective[0] if np.isfinite(best_objective[0]) else None
        self.final_makespan = best_objective[0] if np.isfinite(best_objective[0]) else None
        return current_mapping

    def extract_solution(self):
        solution = {}
        for tenant in self.M:
            for pattern_idx, pattern in enumerate(self.patterns[tenant]):
                variable = self.rmp.getVarByName(f"lam_{tenant}_{pattern_idx}")
                if variable and variable.X > 0.5:
                    solution[tenant] = pattern["mapping"]
                    break
        return solution
