from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB


class MappingILPSolver:
    """Exact mapping ILP used for small-scale validation."""

    def __init__(self, datacenter, tenant_mapping, tenant_flows, verbose=True, name="mapping_ilp"):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.verbose = verbose

        self.data = None
        self.model = None

        self.X = {}
        self.U = {}
        self.T_m = {}
        self.T_max = None

        self._build(name=name)

    def set_objective_sum_tm(self):
        for tenant in self.data["M"]:
            self.T_m[tenant] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"T_{tenant}")

        self.model.setObjective(gp.quicksum(self.T_m.values()), GRB.MINIMIZE)
        return self

    def _build(self, name="mapping_ilp"):
        self.data = self._build_data()

        self.model = gp.Model(name)
        self.model.Params.OutputFlag = 1 if self.verbose else 0

        self.set_objective_sum_tm()
        self._add_X()
        self._add_perm_constraints()
        self._add_U_and_endpoint_constraints()
        self._add_load_constraints()

    def _build_data(self):
        tenant_mapping = self.tenant_mapping
        tenant_flows = self.tenant_flows
        datacenter = self.datacenter

        tenants = sorted(tenant_mapping.keys())
        ranks = {tenant: sorted(tenant_mapping[tenant].keys()) for tenant in tenants}
        servers = {tenant: sorted(tenant_mapping[tenant].values()) for tenant in tenants}

        flows = {
            tenant: [{"u": int(src), "v": int(dst), "V": float(volume)} for (src, dst, volume) in tenant_flows[tenant]]
            for tenant in tenants
        }

        links = list(datacenter.topology.edges())
        capacities = {
            (src, dst): float(datacenter.topology[src][dst].get("capacity", 0.0) / 1e9)
            for (src, dst) in links
        }

        return {
            "M": tenants,
            "R": ranks,
            "S": servers,
            "flows": flows,
            "L": links,
            "cap": capacities,
            "path_edges": datacenter.ECMP_edge_set,
            "Bmax": max(capacities.values()) if capacities else 0.0,
        }

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
        flows = self.data["flows"]

        for tenant in tenants:
            candidate_servers = servers[tenant]
            for flow_idx, flow in enumerate(flows[tenant]):
                logical_src = flow["u"]
                logical_dst = flow["v"]

                flow_vars = []
                for src_server in candidate_servers:
                    for dst_server in candidate_servers:
                        if src_server == dst_server:
                            continue

                        key = (tenant, flow_idx, src_server, dst_server)
                        self.U[key] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"U_{tenant}_{flow_idx}_{src_server}_{dst_server}",
                        )
                        flow_vars.append(self.U[key])

                        self.model.addConstr(
                            self.U[key] <= self.X[(tenant, logical_src, src_server)],
                            name=f"U_le_Xsrc_{tenant}_{flow_idx}_{src_server}_{dst_server}",
                        )
                        self.model.addConstr(
                            self.U[key] <= self.X[(tenant, logical_dst, dst_server)],
                            name=f"U_le_Xdst_{tenant}_{flow_idx}_{src_server}_{dst_server}",
                        )
                        self.model.addConstr(
                            self.U[key]
                            >= self.X[(tenant, logical_src, src_server)]
                            + self.X[(tenant, logical_dst, dst_server)]
                            - 1,
                            name=f"U_ge_AND_{tenant}_{flow_idx}_{src_server}_{dst_server}",
                        )

                self.model.addConstr(gp.quicksum(flow_vars) == 1, name=f"U_onepair_{tenant}_{flow_idx}")

    def _add_load_constraints(self):
        tenants = self.data["M"]
        servers = self.data["S"]
        links = self.data["L"]
        capacities = self.data["cap"]
        flows = self.data["flows"]
        path_edges = self.data["path_edges"]

        big_m = 100.0
        link_load_terms = {link: [] for link in links}
        self.Y = {}
        usage_u_vars = {tenant: {link: [] for link in links} for tenant in tenants}

        for tenant in tenants:
            candidate_servers = servers[tenant]
            for flow_idx, flow in enumerate(flows[tenant]):
                volume = float(flow["V"])
                for src_server in candidate_servers:
                    for dst_server in candidate_servers:
                        if src_server == dst_server:
                            continue

                        u_var = self.U[(tenant, flow_idx, src_server, dst_server)]
                        edges = path_edges.get((src_server, dst_server))
                        if edges is None:
                            continue

                        for link in edges:
                            if link not in capacities:
                                continue
                            link_load_terms[link].append(volume * u_var)
                            usage_u_vars[tenant][link].append(u_var)

        for tenant in tenants:
            for link in links:
                if not usage_u_vars[tenant][link]:
                    continue

                self.Y[(tenant, link)] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"Y_{tenant}_{link[0]}_{link[1]}",
                )
                u_vars = usage_u_vars[tenant][link]
                self.model.addConstr(
                    gp.quicksum(u_vars) <= len(u_vars) * self.Y[(tenant, link)],
                    name=f"Y_def_{tenant}_{link[0]}_{link[1]}",
                )

        for link in links:
            total_load_expr = gp.quicksum(link_load_terms[link])
            for tenant in tenants:
                if (tenant, link) not in self.Y:
                    continue

                self.model.addConstr(
                    total_load_expr
                    <= capacities[link] * self.T_m[tenant] + big_m * (1 - self.Y[(tenant, link)]),
                    name=f"load_cap_{link[0]}_{link[1]}_{tenant}",
                )

    def solve(self, time_limit=None):
        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit

        self.model.Params.MIPGap = 0.0
        self.model.Params.MIPGapAbs = 1e-9
        self.model.Params.IntegralityFocus = 1
        self.model.optimize()

        if not self.verbose:
            return self

        status = self.model.Status
        print("\n=== GUROBI STATUS REPORT ===")
        print("Status =", status)
        print(
            "StatusStr =",
            {
                GRB.OPTIMAL: "OPTIMAL",
                GRB.SUBOPTIMAL: "SUBOPTIMAL",
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.UNBOUNDED: "UNBOUNDED",
            }.get(status, "OTHER"),
        )
        print("SolCount =", self.model.SolCount)
        print("ObjVal =", getattr(self.model, "ObjVal", None))
        print("ObjBound =", getattr(self.model, "ObjBound", None))
        print("MIPGap =", getattr(self.model, "MIPGap", None))
        print("Runtime =", self.model.Runtime)

        if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL) or self.model.SolCount == 0:
            raise RuntimeError(
                f"Model not solved to a valid solution. Status={status}, SolCount={self.model.SolCount}"
            )
        return self

    def get_X_mapping(self, thr=0.5):
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
