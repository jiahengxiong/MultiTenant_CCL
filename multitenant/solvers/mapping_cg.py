from __future__ import annotations

import random
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix


class MappingCGSolver:
    """Column-generation mapping solver used for larger-scale experiments."""

    def __init__(self, datacenter, tenant_mapping, tenant_flows, verbose=False):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.verbose = verbose

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

        self.flows = {}
        self.flow_matrices = {}

        for tenant, flow_list in tenant_flows.items():
            self.flows[tenant] = [(src, dst, volume) for (src, dst, volume) in flow_list]

            rank_count = len(self.tenant_mapping[tenant])
            flow_matrix = np.zeros((rank_count, rank_count))
            for src, dst, volume in self.flows[tenant]:
                if src < rank_count and dst < rank_count:
                    flow_matrix[src, dst] += volume
            self.flow_matrices[tenant] = flow_matrix

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
        self.T_m = {}
        self.constr_convex = {}
        self.constr_link = {}
        self.constr_server = {}
        self.final_obj = None

    def initialize_columns(self):
        for tenant in self.M:
            self.add_pattern_to_list(tenant, self.tenant_mapping[tenant].copy())

    def add_pattern_to_list(self, tenant, mapping):
        traffic = {}

        for src, dst, volume in self.flows[tenant]:
            mapped_src, mapped_dst = mapping[src], mapping[dst]
            if mapped_src == mapped_dst:
                continue

            src_idx = self.server_to_idx[mapped_src]
            dst_idx = self.server_to_idx[mapped_dst]
            edge_indices = self.path_edges_indices[src_idx][dst_idx]

            for edge_idx in edge_indices:
                traffic[edge_idx] = traffic.get(edge_idx, 0.0) + volume

        traffic_obj = {self.L[idx]: volume for idx, volume in traffic.items()}
        servers_used = {server: 0 for server in self.all_servers}
        for server in mapping.values():
            servers_used[server] = 1

        pattern_data = {
            "mapping": mapping,
            "traffic": traffic_obj,
            "servers": servers_used,
            "traffic_indices": traffic,
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

        self.T_m = {tenant: self.rmp.addVar(lb=0.0, name=f"T_{tenant}") for tenant in self.M}

        for tenant in self.M:
            for pattern_idx, _pattern in enumerate(self.patterns[tenant]):
                self.lambdas[(tenant, pattern_idx)] = self.rmp.addVar(
                    lb=0.0,
                    ub=1.0,
                    name=f"lam_{tenant}_{pattern_idx}",
                )

        self.rmp.update()
        self.rmp.setObjective(gp.quicksum(self.T_m.values()), GRB.MINIMIZE)

        for tenant in self.M:
            expr = gp.quicksum(
                self.lambdas[(tenant, pattern_idx)] for pattern_idx in range(len(self.patterns[tenant]))
            )
            self.constr_convex[tenant] = self.rmp.addConstr(expr == 1.0, name=f"convex_{tenant}")

        for link in self.L:
            total_load_expr = gp.LinExpr()
            for other_tenant in self.M:
                for pattern_idx, pattern in enumerate(self.patterns[other_tenant]):
                    volume = pattern["traffic"].get(link, 0.0)
                    if volume > 1e-9:
                        total_load_expr.addTerms(volume, self.lambdas[(other_tenant, pattern_idx)])

            for tenant in self.M:
                is_used_expr = gp.LinExpr()
                for pattern_idx, pattern in enumerate(self.patterns[tenant]):
                    if pattern["traffic"].get(link, 0.0) > 1e-9:
                        is_used_expr.addTerms(1.0, self.lambdas[(tenant, pattern_idx)])

                self.constr_link[(tenant, link)] = self.rmp.addConstr(
                    total_load_expr + self.BigM * is_used_expr - self.cap[link] * self.T_m[tenant] <= self.BigM,
                    name=f"link_{link}_{tenant}",
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

        for edge_idx, volume in pattern["traffic_indices"].items():
            if volume <= 1e-9:
                continue
            link = self.L[edge_idx]

            if (tenant, link) in self.constr_link:
                column.addTerms(volume + self.BigM, self.constr_link[(tenant, link)])

            for other_tenant in self.M:
                if other_tenant == tenant:
                    continue
                if (other_tenant, link) in self.constr_link:
                    column.addTerms(volume, self.constr_link[(other_tenant, link)])

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

        sum_mu = np.zeros(self.num_links)
        for (_tenant, link), value in duals_link.items():
            link_idx = self.link_to_idx[link]
            sum_mu[link_idx] += value

        traffic_weights = -sum_mu
        server_weights = np.array([-duals_server.get(server, 0.0) for server in self.all_servers])

        path_traf_cost_flat = self.path_matrix.dot(traffic_weights)
        path_traf_cost = path_traf_cost_flat.reshape((self.num_servers, self.num_servers))

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

            fixed_weights = np.zeros(self.num_links)
            for link in self.L:
                link_idx = self.link_to_idx[link]
                fixed_weights[link_idx] = -self.BigM * duals_link.get((tenant, link), 0.0)

            path_fixed_cost_flat = self.path_matrix.dot(fixed_weights)
            path_fixed_cost = path_fixed_cost_flat.reshape((self.num_servers, self.num_servers))
            total_path_cost = path_traf_cost + path_fixed_cost

            flow_matrix = self.flow_matrices[tenant]
            num_starts = 10
            start_mappings = [np.array(fixed_server_indices)]
            for _ in range(num_starts - 1):
                shuffled = fixed_server_indices.copy()
                random.shuffle(shuffled)
                start_mappings.append(np.array(shuffled))

            candidates = []
            for current_locations in start_mappings:
                for _ in range(10):
                    server_costs = server_weights[fixed_server_indices]
                    distance_subset = total_path_cost[fixed_server_indices, :][:, current_locations]
                    traffic_costs = flow_matrix @ distance_subset.T
                    cost_matrix = traffic_costs + server_costs

                    _row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    new_locations = np.array([fixed_server_indices[col] for col in col_ind])

                    if np.array_equal(new_locations, current_locations):
                        break
                    current_locations = new_locations

                def calc_rc(locations):
                    rc = 0.0
                    rc += server_weights[locations].sum()

                    used_edges = set()
                    for src, dst, volume in self.flows[tenant]:
                        src_idx = locations[src]
                        dst_idx = locations[dst]
                        if src_idx == dst_idx:
                            continue

                        rc += volume * path_traf_cost[src_idx, dst_idx]
                        for edge_idx in self.path_edges_indices[src_idx][dst_idx]:
                            used_edges.add(edge_idx)

                    for edge_idx in used_edges:
                        rc += fixed_weights[edge_idx]

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

    def solve(self, max_iter=100):
        self.initialize_columns()
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
                        print("Proposed Mapping (CG): converged due to stagnation.")
                    break

            duals_convex = {tenant: self.constr_convex[tenant].Pi for tenant in self.M}
            duals_link = {(tenant, link): constr.Pi for (tenant, link), constr in self.constr_link.items()}
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
        self.rmp.optimize()

        if self.rmp.SolCount > 0:
            self.final_obj = self.rmp.ObjVal
            if self.verbose:
                print(
                    f"Proposed Mapping (CG): final integer objective = {self.final_obj:.6f} "
                    f"(status {self.rmp.Status})."
                )
            return self.extract_solution()

        if self.verbose:
            print(f"Proposed Mapping (CG): integer solve failed (status {self.rmp.Status}).")
        self.final_obj = None
        return None

    def extract_solution(self):
        solution = {}
        for tenant in self.M:
            for pattern_idx, pattern in enumerate(self.patterns[tenant]):
                variable = self.rmp.getVarByName(f"lam_{tenant}_{pattern_idx}")
                if variable and variable.X > 0.5:
                    solution[tenant] = pattern["mapping"]
                    break
        return solution
