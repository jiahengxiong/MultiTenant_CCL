import gurobipy as gp
from gurobipy import GRB
import math


class MultiTenantILP:
    def __init__(self, datacenter, tenant_mapping, tenant_flows,
                 verbose=True, name="multi_tenant_ilp"):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.verbose = verbose

        self.data = None
        self.model = None

        # variables
        self.X = {}
        self.U = {}
        # self.b, self.bhat, self.T_flow, self.T_tenant removed for Load-Based MILP
        
        self.T_max = None

        self._build(name=name)

    # -------------------------
    # Objective
    # -------------------------
    def set_objective_makespan(self):
        self.T_max = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="T_max")
        self.model.setObjective(self.T_max, GRB.MINIMIZE)
        return self

    # -------------------------
    # Build all
    # -------------------------
    def _build(self, name="multi_tenant_ilp"):
        self.data = self._build_data()

        self.model = gp.Model(name)
        self.model.Params.OutputFlag = 1 if self.verbose else 0

        # Initialize T_max early so it can be used in load constraints
        self.set_objective_makespan()

        self._add_X()
        self._add_perm_constraints()
        self._add_U_and_endpoint_constraints()
        self._add_load_constraints()

    def _build_data(self):
        tm = self.tenant_mapping
        tf = self.tenant_flows
        dc = self.datacenter

        M = sorted(tm.keys())
        R = {m: sorted(tm[m].keys()) for m in M}
        S = {m: sorted(tm[m].values()) for m in M}

        flows = {
            m: [{"u": int(u), "v": int(v), "V": float(V)} for (u, v, V) in tf[m]]
            for m in M
        }

        L = list(dc.topology.edges())
        cap = {(a, b): float(dc.topology[a][b].get("capacity", 0.0)/1e9)for (a, b) in L}

        data = {
            "M": M, "R": R, "S": S,
            "flows": flows,
            "L": L, "cap": cap,
            "path_edges": dc.ECMP_edge_set,   # dict[(s,t)] -> set((u,v),...)
            "Bmax": max(cap.values())/1e9 if cap else 0.0,
        }
        return data

    def _add_X(self):
        M, R, S = self.data["M"], self.data["R"], self.data["S"]
        for m in M:
            for r in R[m]:
                for s in S[m]:
                    self.X[(m, r, s)] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"X_{m}_{r}_{s}"
                    )

    def _add_perm_constraints(self):
        M, R, S = self.data["M"], self.data["R"], self.data["S"]

        for m in M:
            for r in R[m]:
                self.model.addConstr(
                    gp.quicksum(self.X[(m, r, s)] for s in S[m]) == 1,
                    name=f"rank_one_{m}_{r}",
                )

        for m in M:
            for s in S[m]:
                self.model.addConstr(
                    gp.quicksum(self.X[(m, r, s)] for r in R[m]) == 1,
                    name=f"srv_one_{m}_{s}",
                )

    def _add_U_and_endpoint_constraints(self):
        M, S = self.data["M"], self.data["S"]
        flows = self.data["flows"]

        if self.verbose:
            print(f"Flows: {flows}")

        for m in M:
            Sm = S[m]
            for j, fj in enumerate(flows[m]):
                u = fj["u"]
                v = fj["v"]

                U_vars = []
                for s in Sm:
                    for t in Sm:
                        if s == t:
                            continue
                        key = (m, j, s, t)
                        self.U[key] = self.model.addVar(vtype=GRB.BINARY, name=f"U_{m}_{j}_{s}_{t}")
                        U_vars.append(self.U[key])

                        self.model.addConstr(self.U[key] <= self.X[(m, u, s)],
                                             name=f"U_le_Xsrc_{m}_{j}_{s}_{t}")
                        self.model.addConstr(self.U[key] <= self.X[(m, v, t)],
                                             name=f"U_le_Xdst_{m}_{j}_{s}_{t}")

                        self.model.addConstr(self.U[key] >= self.X[(m, u, s)] + self.X[(m, v, t)] - 1,
                                             name=f"U_ge_AND_{m}_{j}_{s}_{t}")

                self.model.addConstr(gp.quicksum(U_vars) == 1, name=f"U_onepair_{m}_{j}")

    # -------------------------
    # Load constraints (Linearized)
    # -------------------------
    def _add_load_constraints(self):
        """
        Load-Based MILP formulation:
        For each link ell:
            sum_{m,j,s,t: ell in path(s,t)} (U[m,j,s,t] * V[j]) <= cap[ell] * T_max
        """
        M = self.data["M"]
        S = self.data["S"]
        L = self.data["L"]
        cap = self.data["cap"]
        flows = self.data["flows"]
        path_edges = self.data["path_edges"]

        link_load_terms = {ell: [] for ell in L}

        for m in M:
            Sm = S[m]
            for j, fj in enumerate(flows[m]):
                V = float(fj["V"])
                for s in Sm:
                    for t in Sm:
                        if s == t:
                            continue
                        
                        U_mjst = self.U[(m, j, s, t)]
                        pe = path_edges.get((s, t), None)
                        if pe is None:
                            raise KeyError(f"path_edges missing key {(s,t)}")

                        for ell in pe:
                            # ell must be a directed edge in L
                            if ell not in cap:
                                raise KeyError(f"Edge {ell} from path_edges[(s,t)] not in topology edges/cap")

                            # Load term: V * U_mjst
                            link_load_terms[ell].append(V * U_mjst)

        # Capacity constraints: Load <= Cap * T_max
        for ell in L:
            if link_load_terms[ell]:
                self.model.addConstr(
                    gp.quicksum(link_load_terms[ell]) <= cap[ell] * self.T_max,
                    name=f"load_cap_{ell[0]}_{ell[1]}"
                )

    # _add_time_constraints_soc removed

    # -------------------------
    # Solve + extract
    # -------------------------
    def solve(self, time_limit=None):
        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit
        
        # Set tighter tolerances to avoid premature termination
        self.model.Params.MIPGap = 0.0
        self.model.Params.MIPGapAbs = 1e-9
        self.model.Params.IntegralityFocus = 1
        
        self.model.optimize()

        if not self.verbose:
            return self

        st = self.model.Status
        print("\n=== GUROBI STATUS REPORT ===")
        print("Status =", st)
        print("StatusStr =", {GRB.OPTIMAL: "OPTIMAL", GRB.SUBOPTIMAL: "SUBOPTIMAL",
                             GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED"}.get(st, "OTHER"))
        print("SolCount =", self.model.SolCount)
        print("ObjVal =", getattr(self.model, "ObjVal", None))
        print("ObjBound =", getattr(self.model, "ObjBound", None))
        print("MIPGap =", getattr(self.model, "MIPGap", None))
        print("Runtime =", self.model.Runtime)

        if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL) and self.model.SolCount > 0:
            print("T_max.X =", self.T_max.X if self.T_max is not None else None)
        else:
            raise RuntimeError(f"Model not solved to a valid solution. Status={st}, SolCount={self.model.SolCount}")

        return self

    def get_X_mapping(self, thr=0.5):
        M, R, S = self.data["M"], self.data["R"], self.data["S"]
        out = {m: {} for m in M}
        for m in M:
            for r in R[m]:
                for s in S[m]:
                    if self.X[(m, r, s)].X > thr:
                        out[m][r] = s
                        break
        if self.verbose:
            print("X mapping:", out)
        return out
