import gurobipy as gp
from gurobipy import GRB
import itertools
import random
import time
import math

class ColumnGenerationSolver:
    def __init__(self, datacenter, tenant_mapping, tenant_flows, verbose=False):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.verbose = verbose
        
        # Scaling factor to avoid numerical issues (e.g., 1e9 for Gbits)
        self.scale = 1e9
        
        # Extract basic data
        self.M = sorted(tenant_mapping.keys())
        self.L = list(datacenter.topology.edges())
        # Scale capacity: e.g., 25,000,000,000 -> 25.0
        self.cap = {(u,v): datacenter.topology[u][v]['capacity'] / self.scale for (u,v) in self.L}
        
        # Logical flows per tenant: list of (u, v, V)
        # Input tenant_flows are already scaled by 1e9 in main.py?
        # Let's check magnitude. If V < 1, assume scaled.
        # But to be safe and consistent with main.py which divides by 1e9:
        # We assume tenant_flows V is in Gbits (if main.py did V/1e9).
        # We assume topology capacity is in bits/sec (raw).
        
        # If we want consistent units (Gbits), we scale capacity by 1e9, 
        # and keep flows as is (if they are already Gbits).
        
        self.flows = {}
        for m, flist in tenant_flows.items():
            # Check first flow to guess scaling? No, dangerous.
            # Based on current main.py, flows are scaled.
            # self.flows[m] = [(u, v, V / self.scale) for (u, v, V) in flist] # REMOVED double scaling
            self.flows[m] = [(u, v, V) for (u, v, V) in flist]
        
        # Patterns: self.patterns[m] = list of patterns
        # A pattern is a dict: {logical_rank: physical_server}
        self.patterns = {m: [] for m in self.M}
        
        # RMP model
        self.rmp = None
        self.lambdas = {} # (m, p_idx) -> Var
        self.T_max = None
        
        # Constraints
        self.constr_convex = {} # m -> Constr
        self.constr_link = {}   # ell -> Constr
        self.constr_server = {} # s -> Constr

    def initialize_columns(self):
        """
        Generate initial columns (patterns).
        We use the initial random mapping provided in tenant_mapping as the first column.
        """
        for m in self.M:
            initial_pat = self.tenant_mapping[m].copy()
            self.add_pattern(m, initial_pat)

    def add_pattern(self, m, mapping):
        """
        Add a new pattern for tenant m.
        mapping: {rank: server}
        """
        # Calculate traffic footprint for this pattern
        # Traffic_mp[ell] = sum of Volume of flows of tenant m that pass through ell
        
        traffic = {ell: 0.0 for ell in self.L}
        
        # For each flow in tenant m
        for (u, v, V) in self.flows[m]:
            # logical u -> phys s, logical v -> phys t
            s = mapping[u]
            t = mapping[v]
            if s == t: continue 
            
            # Get path
            path = self.datacenter.paths.get((s,t))
            if not path: continue
            
            edges = self.datacenter.path_to_edges(path)
            for ell in edges:
                if ell in traffic:
                    traffic[ell] += V
        
        # Server usage: {server: 1 if used}
        servers_used = {s: 0 for s in self.datacenter.get_all_servers()}
        for s in mapping.values():
            servers_used[s] = 1
            
        pat_data = {
            "mapping": mapping,
            "traffic": traffic,
            "servers": servers_used
        }
        
        self.patterns[m].append(pat_data)

    def build_rmp(self):
        """
        Build Restricted Master Problem
        """
        self.rmp = gp.Model("MultiTenantCG")
        self.rmp.Params.OutputFlag = 0
        
        # Variable: T_max (minimize)
        self.T_max = self.rmp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="T_max")
        
        # Variables: Lambda_m_p
        self.lambdas = {}
        for m in self.M:
            for p_idx, pat in enumerate(self.patterns[m]):
                self.lambdas[(m, p_idx)] = self.rmp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"lam_{m}_{p_idx}")
        
        self.rmp.update()
        
        # Objective: Minimize T_max
        self.rmp.setObjective(self.T_max, GRB.MINIMIZE)
        
        # Constraint 1: Convexity (Each tenant chooses exactly one pattern)
        # sum_{p} lambda_{m,p} = 1
        for m in self.M:
            expr = gp.quicksum(self.lambdas[(m, p_idx)] for p_idx in range(len(self.patterns[m])))
            self.constr_convex[m] = self.rmp.addConstr(expr == 1.0, name=f"convex_{m}")
            
        # Constraint 2: Link Capacity (Load <= Cap * T_max)
        # sum_{m,p} lambda_{m,p} * Traffic_{m,p}^ell - Cap_ell * T_max <= 0
        for ell in self.L:
            expr = gp.quicksum(
                self.lambdas[(m, p_idx)] * self.patterns[m][p_idx]["traffic"][ell]
                for m in self.M
                for p_idx in range(len(self.patterns[m]))
            )
            self.constr_link[ell] = self.rmp.addConstr(expr - self.cap[ell] * self.T_max <= 0, name=f"link_{ell}")
            
        # Constraint 3: Server Exclusivity (Each server used at most once)
        # sum_{m,p} lambda_{m,p} * IsUsed_{m,p}^s <= 1
        all_servers = self.datacenter.get_all_servers()
        for s in all_servers:
            expr = gp.quicksum(
                self.lambdas[(m, p_idx)] * self.patterns[m][p_idx]["servers"].get(s, 0)
                for m in self.M
                for p_idx in range(len(self.patterns[m]))
            )
            self.constr_server[s] = self.rmp.addConstr(expr <= 1.0, name=f"server_{s}")

    def solve_pricing(self, duals_convex, duals_link, duals_server):
        """
        Solve Pricing Problem for each tenant.
        Find a pattern p for tenant m that has negative Reduced Cost.
        
        RC = 0 - [ pi_m * 1 + sum(mu_ell * Traffic_ell) + sum(sigma_s * IsUsed_s) ]
        RC = -pi_m + sum(-mu_ell * Traffic_ell) + sum(-sigma_s * IsUsed_s)
        
        Let w_ell = -mu_ell >= 0 (since mu_ell <= 0)
        Let v_s = -sigma_s >= 0 (since sigma_s <= 0)
        
        We want to Minimize: Cost(p) = sum(w_ell * Traffic_ell) + sum(v_s * IsUsed_s)
        If MinCost < pi_m, then RC < 0.
        """
        
        new_columns_count = 0
        
        # Pre-calculate weights (positive costs)
        # Note: Duals for <= constraints are non-positive in Gurobi (usually).
        # But wait, Gurobi's dual sign depends on optimization direction (Min) and constraint type.
        # Min c'x, s.t. Ax >= b -> Dual >= 0
        # Min c'x, s.t. Ax <= b -> Dual <= 0
        # Here we have <= 0 constraints, so duals should be <= 0.
        # So -duals should be >= 0.
        
        w_link = {ell: -duals_link[ell] for ell in self.L}
        v_server = {s: -duals_server.get(s, 0.0) for s in self.datacenter.get_all_servers()}
        
        # Optimization: Filter out zero weights to speed up calculation? 
        # No, iterating all links is slow.
        # Traffic is sum of V * edges_in_path.
        # Cost = sum(V * sum(w_ell for ell in path)) + sum(v_s)
        # Let PathCost(s,t) = sum(w_ell for ell in path_s_t)
        # This can be pre-calculated for all s,t pairs!
        
        # Pre-calculate Path Costs
        path_costs = {} # (s,t) -> cost
        all_servers = self.datacenter.get_all_servers()
        # We only need pairs that might communicate. 
        # But calculating all pairs is O(N^2 * PathLen), feasible for N=100.
        
        for s in all_servers:
            for t in all_servers:
                if s == t:
                    path_costs[(s,t)] = 0.0
                    continue
                path = self.datacenter.paths.get((s,t))
                if path:
                    edges = self.datacenter.path_to_edges(path)
                    cost = sum(w_link.get(e, 0.0) for e in edges)
                    path_costs[(s,t)] = cost
                else:
                    path_costs[(s,t)] = float('inf')

        for m in self.M:
            k = len(self.tenant_mapping[m])
            ranks = list(range(k))
            
            # Pricing Problem: Assign k ranks to k distinct servers to minimize total cost
            # Total Cost = Sum_{flows} (V * PathCost(map(u), map(v))) + Sum_{servers} v_server(s)
            
            best_val = float('inf')
            best_mapping = None
            
            # Simple heuristic / brute force depending on size
            # For small N (e.g. 12) and small k (e.g. 4), permutations is fine.
            # itertools.permutations(all_servers, k)
            
            # Heuristic pruning:
            # Sort servers by v_server cost? 
            # But link costs dominate usually.
            
            # Let's try full permutation for now. 
            # If N > 20, we might need to sample or use a better heuristic.
            
            # Optimization: If N is large, random sampling might be needed for "Simulated Annealing" or similar
            # But for the given problem size (Leaf=3, PerLeaf=4 => 12 servers), it's small.
            
            # To avoid exploring 12*11*10*9 ~ 11k iterations per tenant per CG iter (which is fast actually),
            # we just do it.
            
            # If N is large (e.g. 100), 100*99*98*97 ~ 100M, too slow.
            # Assuming small scale for now based on main.py config.
            
            # Constraint: we must pick distinct servers.
            # RESTRICTION: The set of servers is FIXED to the initial assignment (per user request).
            # We only optimize the permutation (mapping of ranks to these specific servers).
            
            # Retrieve the fixed servers for this tenant
            fixed_servers = list(self.tenant_mapping[m].values())
            
            # For permutation, k must equal len(fixed_servers). 
            # If for some reason they differ (e.g. overprovisioning?), we might need combinations.
            # But based on main.py, k = len(assigned_servers).
            
            # Exact search over permutations of the FIXED set of servers
            # Store all valid patterns with negative Reduced Cost
            candidates = []
            
            for server_perm in itertools.permutations(fixed_servers, k):
                current_mapping = {ranks[i]: server_perm[i] for i in range(k)}
                current_cost = 0.0
                for s in server_perm:
                    current_cost += v_server.get(s, 0.0)
                
                for (u, v, V) in self.flows[m]:
                    s_node = current_mapping[u]
                    t_node = current_mapping[v]
                    current_cost += V * path_costs.get((s_node, t_node), float('inf'))
                
                candidates.append((current_cost, current_mapping))
            
            # Sort by cost (lowest first)
            candidates.sort(key=lambda x: x[0])
            
            # Check Reduced Cost
            # RC = -pi_m + best_val
            pi_m = duals_convex[m]
            
            # Add up to Top-K patterns with negative RC
            # Increase K to ensure we have a rich pool for the integer phase
            top_k = 50
            added_count = 0
            
            for cost, mapping in candidates:
                # Tighten tolerance to 1e-9 to capture subtle improvements
                if cost < pi_m - 1e-9:
                    self.add_pattern(m, mapping)
                    new_columns_count += 1
                    added_count += 1
                    if added_count >= top_k:
                        break
            
        return new_columns_count
                
        return new_columns_count

    def solve(self, max_iter=100):
        self.initialize_columns()
        self.build_rmp()
        
        iter_count = 0
        start_time = time.time()
        
        while iter_count < max_iter:
            iter_count += 1
            self.rmp.optimize()
            
            if self.rmp.Status != GRB.OPTIMAL:
                if self.verbose: print(f"RMP Status {self.rmp.Status} in iter {iter_count}")
                break
                
            obj_val = self.rmp.ObjVal
            if self.verbose: print(f"CG Iter {iter_count}: LP Obj = {obj_val:.6f}")
            
            # Get Duals
            duals_convex = {m: self.constr_convex[m].Pi for m in self.M}
            duals_link = {ell: self.constr_link[ell].Pi for ell in self.L}
            duals_server = {s: self.constr_server[s].Pi for s in self.datacenter.get_all_servers()}
            
            # Solve Pricing
            new_cols = self.solve_pricing(duals_convex, duals_link, duals_server)
            
            if new_cols == 0:
                if self.verbose: print("No new columns found. LP Optimal reached.")
                break
            
            # Rebuild RMP (simplest way to add columns for this prototype)
            # For efficiency, we should use Column object, but rebuilding is safer for correctness now.
            self.build_rmp()
            
        print(f"CG Loop finished in {time.time() - start_time:.2f}s, {iter_count} iters.")
            
        # Final Solve (Integer)
        # Convert all lambda variables to Binary
        # We need to rebuild or modify the existing model
        
        # To avoid "modification" issues, let's just set vtype
        for v in self.rmp.getVars():
            if v.VarName.startswith("lam_"):
                v.VType = GRB.BINARY
        
        self.rmp.update()
        self.rmp.optimize()
        
        if self.rmp.Status == GRB.OPTIMAL:
            self.final_obj = self.rmp.ObjVal
            print(f"CG Final Integer Obj: {self.rmp.ObjVal:.6f}")
            return self.extract_solution()
        else:
            self.final_obj = None
            print("CG Integer solution not found")
            return None

    def extract_solution(self):
        # Return mapping {m: {rank: server}}
        solution = {}
        for m in self.M:
            for p_idx, pat in enumerate(self.patterns[m]):
                var = self.rmp.getVarByName(f"lam_{m}_{p_idx}")
                # Use a small epsilon for float comparison if continuous, 
                # but we solved as binary, so > 0.5 is safe.
                if var and var.X > 0.5:
                    solution[m] = pat["mapping"]
                    break
        return solution
