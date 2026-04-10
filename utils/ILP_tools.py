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
    def set_objective_sum_tm(self):
        self.T_m = {}
        for m in self.data["M"]:
            self.T_m[m] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"T_{m}")
        
        self.model.setObjective(gp.quicksum(self.T_m.values()), GRB.MINIMIZE)
        return self

    # -------------------------
    # Build all
    # -------------------------
    def _build(self, name="multi_tenant_ilp"):
        self.data = self._build_data()

        self.model = gp.Model(name)
        self.model.Params.OutputFlag = 1 if self.verbose else 0

        # Initialize T_m early so it can be used in load constraints
        self.set_objective_sum_tm()

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
        Load-Based MILP formulation for Min Sum T_m:
        For each link ell and each tenant m:
            TotalLoad(ell) <= cap[ell] * T_m[m] + BigM * (1 - Y_{m,ell})
        where Y_{m,ell} = 1 if m uses ell.
        """
        M_tenants = self.data["M"]
        S = self.data["S"]
        L = self.data["L"]
        cap = self.data["cap"]
        flows = self.data["flows"]
        path_edges = self.data["path_edges"]
        
        BigM = 100.0

        # Precompute load terms and usage indicators
        link_load_terms = {ell: [] for ell in L}
        
        # We need Y variables: Y[m, ell]
        self.Y = {}

        # To track which U variables contribute to Y constraint
        # usage_U_vars[m][ell] = list of U_{m,j,s,t} that use ell
        usage_U_vars = {m: {ell: [] for ell in L} for m in M_tenants}

        for m in M_tenants:
            Sm = S[m]
            for j, fj in enumerate(flows[m]):
                V = float(fj["V"])
                for s in Sm:
                    for t in Sm:
                        if s == t: continue
                        
                        U_mjst = self.U[(m, j, s, t)]
                        pe = path_edges.get((s, t), None)
                        if pe is None: continue

                        for ell in pe:
                            if ell not in cap: continue

                            # Load term: V * U_mjst
                            link_load_terms[ell].append(V * U_mjst)
                            
                            # Usage tracking
                            usage_U_vars[m][ell].append(U_mjst)

        # Create Y variables and constraints
        for m in M_tenants:
            for ell in L:
                if not usage_U_vars[m][ell]:
                    # m never uses ell (based on topology/flows)
                    # No Y needed, treat as Y=0.
                    # Constraint: TotalLoad <= Cap * Tm + BigM
                    # This is always loose if BigM is large.
                    # Optimization: Don't add constraint if m cannot use ell?
                    # No, TotalLoad might be high due to OTHERS.
                    # If m doesn't use ell, m doesn't care about ell's load.
                    # So constraint is trivial (satisfied by BigM).
                    pass
                else:
                    # m might use ell
                    self.Y[(m, ell)] = self.model.addVar(vtype=GRB.BINARY, name=f"Y_{m}_{ell[0]}_{ell[1]}")
                    
                    # Y >= U for all U in usage
                    # To minimize constraints, we can say Y >= sum(U)/Count? No.
                    # Y >= U is needed.
                    # Actually, we can just say: Y * Count >= sum(U) ?
                    # If sum(U) > 0, Y must be 1 (if Y binary).
                    # Yes: sum(U) <= Count * Y  => Y >= sum(U)/Count.
                    # If sum(U) is 0, Y can be 0.
                    # If sum(U) >= 1, Y must be >= 1/Count -> Y=1.
                    # This is sufficient.
                    
                    u_list = usage_U_vars[m][ell]
                    self.model.addConstr(
                        gp.quicksum(u_list) <= len(u_list) * self.Y[(m, ell)],
                        name=f"Y_def_{m}_{ell[0]}_{ell[1]}"
                    )

        # Capacity constraints: Load <= Cap * T_m + BigM * (1 - Y)
        for ell in L:
            total_load_expr = gp.quicksum(link_load_terms[ell])
            
            for m in M_tenants:
                if (m, ell) in self.Y:
                    # Active constraint
                    self.model.addConstr(
                        total_load_expr <= cap[ell] * self.T_m[m] + BigM * (1 - self.Y[(m, ell)]),
                        name=f"load_cap_{ell[0]}_{ell[1]}_{m}"
                    )
                else:
                    # m doesn't use ell. Constraint is vacuous if BigM is large.
                    # 0 <= Cap * Tm + BigM. Always true since Load >= 0 and Tm >= 0.
                    pass

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
