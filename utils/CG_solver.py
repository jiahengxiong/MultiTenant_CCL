import gurobipy as gp
from gurobipy import GRB
import itertools
import random
import time
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

class ColumnGenerationSolver:
    def __init__(self, datacenter, tenant_mapping, tenant_flows, verbose=False):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.verbose = verbose
        
        # 1. Fix: Increase BigM
        self.BigM = 1e9 
        self.scale = 1e9
        
        # Basic Data
        self.M = sorted(tenant_mapping.keys())
        self.L = list(datacenter.topology.edges())
        self.link_to_idx = {ell: i for i, ell in enumerate(self.L)}
        self.num_links = len(self.L)
        
        self.cap = {ell: datacenter.topology[u][v]['capacity'] / self.scale for (u,v), ell in zip(self.L, self.L)}
        
        # Server Indexing
        self.all_servers = self.datacenter.get_all_servers()
        self.server_to_idx = {s: i for i, s in enumerate(self.all_servers)}
        self.num_servers = len(self.all_servers)
        
        # Flows & Precomputed Flow Matrices
        self.flows = {}
        self.flow_matrices = {} # m -> (k, k) array
        self.flow_outgoing = {} # m -> list of (u, v, vol)
        
        for m, flist in tenant_flows.items():
            # Store scaled flows if needed, but we keep original logic (V/scale was handled in main?)
            # Assuming input V is consistent with logic.
            self.flows[m] = [(u, v, V) for (u, v, V) in flist]
            
            k = len(self.tenant_mapping[m])
            mat = np.zeros((k, k))
            # Also need efficient adjacency for pricing?
            # Matrix is good for vectorization.
            for (u, v, V) in self.flows[m]:
                if u < k and v < k:
                    mat[u, v] += V
            self.flow_matrices[m] = mat

        # Precompute Path Edge Indices for Vectorized Cost Calculation
        # path_edges_indices[s_idx][t_idx] = list of edge_indices
        self.path_edges_indices = [[[] for _ in range(self.num_servers)] for _ in range(self.num_servers)]
        
        # Also build Sparse Matrix P: (num_servers^2, num_links)
        # Row i corresponds to path between (i // num_servers) and (i % num_servers)
        # P[row, edge_idx] = 1 if edge in path
        rows = []
        cols = []
        data = []
        
        for s in self.all_servers:
            s_idx = self.server_to_idx[s]
            for t in self.all_servers:
                t_idx = self.server_to_idx[t]
                if s == t: continue
                
                path = self.datacenter.paths.get((s,t))
                if path:
                    edges = self.datacenter.path_to_edges(path)
                    indices = [self.link_to_idx[e] for e in edges if e in self.link_to_idx]
                    self.path_edges_indices[s_idx][t_idx] = indices
                    
                    # Sparse Matrix entries
                    row_idx = s_idx * self.num_servers + t_idx
                    for e_idx in indices:
                        rows.append(row_idx)
                        cols.append(e_idx)
                        data.append(1.0)
                        
        self.path_matrix = csr_matrix((data, (rows, cols)), shape=(self.num_servers * self.num_servers, self.num_links))
        
        # Patterns
        self.patterns = {m: [] for m in self.M}
        self.added_patterns_hashes = {m: set() for m in self.M}
        
        # RMP
        self.rmp = None
        self.lambdas = {} 
        self.T_m = {}
        self.constr_convex = {}
        self.constr_link = {}
        self.constr_server = {}

    def initialize_columns(self):
        """Generate initial columns from input mapping."""
        for m in self.M:
            self.add_pattern_to_list(m, self.tenant_mapping[m].copy())

    def add_pattern_to_list(self, m, mapping):
        """Helper to compute pattern data and store it."""
        # Calculate traffic
        traffic = {} # Sparse dict: idx -> vol
        
        for (u, v, V) in self.flows[m]:
            s, t = mapping[u], mapping[v]
            if s == t: continue
            
            s_idx, t_idx = self.server_to_idx[s], self.server_to_idx[t]
            edge_indices = self.path_edges_indices[s_idx][t_idx]
            
            for e_idx in edge_indices:
                traffic[e_idx] = traffic.get(e_idx, 0.0) + V
                
        # Convert traffic to ell object keys for compatibility if needed, 
        # but for internal RMP building we can use indices or map back.
        # Let's map back to ell objects to match existing constraint dict keys.
        traffic_obj = {self.L[i]: v for i, v in traffic.items()}
        
        servers_used = {s: 0 for s in self.all_servers}
        for s in mapping.values(): servers_used[s] = 1
            
        pat_data = {
            "mapping": mapping,
            "traffic": traffic_obj,
            "servers": servers_used,
            "traffic_indices": traffic # Keep indices for fast RMP add
        }
        
        # Check duplicate?
        # Hash mapping
        map_hash = tuple(sorted(mapping.items()))
        if map_hash in self.added_patterns_hashes[m]:
            return -1
            
        self.patterns[m].append(pat_data)
        self.added_patterns_hashes[m].add(map_hash)
        return len(self.patterns[m]) - 1

    def initialize_rmp(self):
        """Build RMP once."""
        self.rmp = gp.Model("MultiTenantCG")
        self.rmp.Params.OutputFlag = 0
        # 3. Strategy: Force Dual Simplex for efficient re-optimization
        self.rmp.Params.Method = 1  # Dual Simplex
        # self.rmp.Params.HotStart = 1 # Removed invalid param
        
        # Variables
        self.T_m = {m: self.rmp.addVar(lb=0.0, name=f"T_{m}") for m in self.M}
        
        for m in self.M:
            for p_idx, pat in enumerate(self.patterns[m]):
                self.lambdas[(m, p_idx)] = self.rmp.addVar(lb=0.0, ub=1.0, name=f"lam_{m}_{p_idx}")
        
        self.rmp.update()
        self.rmp.setObjective(gp.quicksum(self.T_m.values()), GRB.MINIMIZE)
        
        # Constraints
        
        # 1. Convexity
        for m in self.M:
            expr = gp.quicksum(self.lambdas[(m, p_idx)] for p_idx in range(len(self.patterns[m])))
            self.constr_convex[m] = self.rmp.addConstr(expr == 1.0, name=f"convex_{m}")
            
        # 2. Link Capacity
        # We need constraints for ALL (m, ell)
        for ell in self.L:
            # Global Load Term
            # It's expensive to iterate all patterns here, but done only once.
            total_load_expr = gp.LinExpr()
            for k in self.M:
                for p_idx, pat in enumerate(self.patterns[k]):
                    vol = pat["traffic"].get(ell, 0.0)
                    if vol > 1e-9:
                        total_load_expr.addTerms(vol, self.lambdas[(k, p_idx)])
            
            for m in self.M:
                # IsUsed Term
                is_used_expr = gp.LinExpr()
                for p_idx, pat in enumerate(self.patterns[m]):
                    if pat["traffic"].get(ell, 0.0) > 1e-9:
                        is_used_expr.addTerms(1.0, self.lambdas[(m, p_idx)])
                
                self.constr_link[(m, ell)] = self.rmp.addConstr(
                    total_load_expr + self.BigM * is_used_expr - self.cap[ell] * self.T_m[m] <= self.BigM,
                    name=f"link_{ell}_{m}"
                )
                
        # 3. Server Exclusivity
        for s in self.all_servers:
            expr = gp.LinExpr()
            for m in self.M:
                for p_idx, pat in enumerate(self.patterns[m]):
                    if pat["servers"].get(s, 0) > 0.5:
                        expr.addTerms(1.0, self.lambdas[(m, p_idx)])
            self.constr_server[s] = self.rmp.addConstr(expr <= 1.0, name=f"server_{s}")

    def add_column_to_rmp(self, m, pat_idx):
        """Add a new column dynamically."""
        pat = self.patterns[m][pat_idx]
        col = gp.Column()
        
        # 1. Convexity
        col.addTerms(1.0, self.constr_convex[m])
        
        # 2. Link Constraints
        # Used links
        used_link_indices = pat["traffic_indices"] # dict idx->vol
        
        for l_idx, vol in used_link_indices.items():
            if vol <= 1e-9: continue
            ell = self.L[l_idx]
            
            # For (m, ell): coeff = vol + BigM
            if (m, ell) in self.constr_link:
                col.addTerms(vol + self.BigM, self.constr_link[(m, ell)])
            
            # For (k, ell) where k != m: coeff = vol
            for k in self.M:
                if k == m: continue
                if (k, ell) in self.constr_link:
                    col.addTerms(vol, self.constr_link[(k, ell)])
                    
        # 3. Server Constraints
        for s, used in pat["servers"].items():
            if used > 0.5:
                if s in self.constr_server:
                    col.addTerms(1.0, self.constr_server[s])
                    
        # Add variable
        self.lambdas[(m, pat_idx)] = self.rmp.addVar(obj=0.0, column=col, lb=0.0, ub=1.0, name=f"lam_{m}_{pat_idx}")

    def solve_pricing(self, duals_convex, duals_link, duals_server):
        new_columns_count = 0
        
        # 1. Vectorized Precomputation
        
        # W_traf vector (size |L|)
        sum_mu = np.zeros(self.num_links)
        for (k, ell), val in duals_link.items():
            l_idx = self.link_to_idx[ell]
            sum_mu[l_idx] += val
            
        w_traf = -sum_mu # Array of size |L|
        
        # V_server vector (size |S|)
        v_server = np.array([ -duals_server.get(s, 0.0) for s in self.all_servers ])
        
        # Path Cost Matrix ( |S| x |S| ) - TRAFFIC ONLY
        # Using Sparse Matrix P: P @ w_traf -> (num_servers * num_servers, )
        path_traf_cost_flat = self.path_matrix.dot(w_traf)
        path_traf_cost = path_traf_cost_flat.reshape((self.num_servers, self.num_servers))
        
        # 2. Solve for each tenant (Partial Pricing with Shuffle)
        tenant_order = list(self.M)
        random.shuffle(tenant_order)
        
        improvements_found = 0
        max_improvements_per_iter = max(5, len(self.M) // 2) # Slightly relaxed limit
        
        for m in tenant_order:
            if improvements_found >= max_improvements_per_iter:
                break
                
            k = len(self.tenant_mapping[m])
            fixed_servers_obj = list(self.tenant_mapping[m].values())
            fixed_servers_indices = [self.server_to_idx[s] for s in fixed_servers_obj]
            
            # W_fixed for this tenant
            w_fixed = np.zeros(self.num_links)
            for ell in self.L:
                l_idx = self.link_to_idx[ell]
                w_fixed[l_idx] = -self.BigM * duals_link.get((m, ell), 0.0)
            
            # Compute Path Fixed Costs using Matrix
            # P @ w_fixed
            path_fixed_cost_flat = self.path_matrix.dot(w_fixed)
            path_fixed_cost = path_fixed_cost_flat.reshape((self.num_servers, self.num_servers))
            
            # Total Path Cost for LAP (Traffic + Fixed)
            # This is the CRITICAL FIX: LAP now sees the fixed costs!
            total_path_cost = path_traf_cost + path_fixed_cost
            
            # Flow Matrix
            F = self.flow_matrices[m] # (k, k)
            
            # Multi-Start
            num_starts = 10 # Increase exploration
            start_mappings_indices = []
            
            # 1. Current best (initial)
            start_mappings_indices.append(np.array(fixed_servers_indices))
            
            # 2. Random perms
            for _ in range(num_starts - 1):
                shuffled = fixed_servers_indices.copy()
                random.shuffle(shuffled)
                start_mappings_indices.append(np.array(shuffled))
                
            candidates = [] # List of (rc, mapping_dict)
            
            for current_locs in start_mappings_indices: # current_locs is array of server indices for ranks 0..k-1
                
                # Iterative LAP
                for _ in range(10): # Max 10 iters
                    
                    # Build Cost Matrix (k x k)
                    # Cost[u, i] = v_server[s_i] + sum_v F[u, v] * total_path_cost[s_i, loc(v)]
                    
                    # 1. Server Dual Cost (Broadcast)
                    server_costs = v_server[fixed_servers_indices] # shape (k,)
                    
                    # 2. Path Cost (Traffic + Fixed)
                    D_subset = total_path_cost[fixed_servers_indices, :][:, current_locs] # Shape (k, k)
                    
                    # Result = F @ D_subset.T
                    traf_costs = F @ D_subset.T # Shape (k, k). Row u, Col i.
                    
                    cost_matrix = traf_costs + server_costs # Broadcast add to each row
                    
                    # Solve LAP
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    # New locations
                    new_locs = np.array([fixed_servers_indices[c] for c in col_ind])
                    
                    if np.array_equal(new_locs, current_locs):
                        break
                    current_locs = new_locs
                    
                # Exact Evaluation & Local Search
                
                # Function to calc exact RC (uses original logic to be safe, but total_path_cost is good proxy)
                def calc_rc(locs):
                    # locs: array of server indices
                    rc = 0.0
                    rc += v_server[locs].sum()
                    
                    used_edges = set()
                    possible = True
                    # Vectorized flow cost calculation? No, simple loop is fine for RC check
                    for (u, v, V) in self.flows[m]:
                        s_idx, t_idx = locs[u], locs[v]
                        if s_idx == t_idx: continue
                        
                        cost = path_traf_cost[s_idx, t_idx] # Only traffic part
                        if cost >= 1e17: 
                            possible = False; break
                        rc += V * cost
                        
                        for e_idx in self.path_edges_indices[s_idx][t_idx]:
                            used_edges.add(e_idx)
                            
                    if not possible: return float('inf')
                    
                    for e_idx in used_edges:
                        rc += w_fixed[e_idx]
                        
                    rc -= duals_convex[m]
                    return rc

                current_rc = calc_rc(current_locs)
                
                # Random Swap Local Search
                improved = True
                while improved:
                    improved = False
                    n_swaps = max(50, k * 2)
                    for _ in range(n_swaps):
                        idx1, idx2 = random.sample(range(k), 2)
                        test_locs = current_locs.copy()
                        test_locs[idx1], test_locs[idx2] = test_locs[idx2], test_locs[idx1]
                        
                        test_rc = calc_rc(test_locs)
                        if test_rc < current_rc - 1e-9:
                            current_rc = test_rc
                            current_locs = test_locs
                            improved = True
                            break
                            
                if current_rc < -1e-9:
                    mapping_dict = tuple((r, self.all_servers[idx]) for r, idx in enumerate(current_locs))
                    candidates.append((current_rc, mapping_dict))
            
            # Sort candidates by RC and add Top-K unique
            candidates.sort(key=lambda x: x[0])
            
            added_count = 0
            seen_mappings = set()
            
            # Add up to 3 best unique columns
            for rc, map_tuple in candidates:
                if added_count >= 3: break
                
                if map_tuple not in seen_mappings:
                    mapping = dict(map_tuple)
                    idx = self.add_pattern_to_list(m, mapping)
                    if idx != -1:
                        self.add_column_to_rmp(m, idx)
                        new_columns_count += 1
                        added_count += 1
                        seen_mappings.add(map_tuple)
            
            if added_count > 0:
                improvements_found += 1
                    
        return new_columns_count

    def solve(self, max_iter=100):
        self.initialize_columns()
        self.initialize_rmp()
        
        iter_count = 0
        start_time = time.time()
        obj_history = []
        
        while iter_count < max_iter:
            iter_count += 1
            self.rmp.optimize()
            if self.rmp.Status != GRB.OPTIMAL:
                if self.verbose: print(f"Status {self.rmp.Status}")
                break
                
            obj_val = self.rmp.ObjVal
            if self.verbose: print(f"Iter {iter_count}: {obj_val:.6f}")
            
            # Strategy 2: Early Termination (Stagnation detection)
            # DELAYED CHECK: Don't check until iter 20
            obj_history.append(obj_val)
            if iter_count > 20 and len(obj_history) >= 8:
                improvement = (obj_history[-8] - obj_history[-1]) / abs(obj_history[-8])
                if improvement < 0.0005:
                    if self.verbose: print("Converged (Stagnation)")
                    print(f"  > CG Stagnation detected at iter {iter_count}, stopping early.")
                    break
            
            # Get Duals
            duals_convex = {m: self.constr_convex[m].Pi for m in self.M}
            duals_link = {}
            for (m, ell), constr in self.constr_link.items():
                duals_link[(m, ell)] = constr.Pi
            duals_server = {s: self.constr_server[s].Pi for s in self.all_servers}
            
            # Pricing
            new_cols = self.solve_pricing(duals_convex, duals_link, duals_server)
            
            if new_cols == 0:
                break
                
        print(f"CG Loop finished in {time.time() - start_time:.2f}s, {iter_count} iters.")
        
        # Integer Solve
        for v in self.rmp.getVars():
            if v.VarName.startswith("lam_"):
                v.VType = GRB.BINARY
        self.rmp.update()
        # Optimize MIP settings for speed
        self.rmp.Params.MIPGap = 0.05 
        self.rmp.Params.MIPFocus = 1 # Focus on finding feasible solutions quickly
        self.rmp.Params.TimeLimit = 3.0 # Set a strict time limit for the integer solve
        self.rmp.optimize()
        
        if self.rmp.SolCount > 0:
            self.final_obj = self.rmp.ObjVal
            print(f"CG Final Integer Obj: {self.final_obj:.6f} (Status: {self.rmp.Status})")
            return self.extract_solution()
        else:
            print(f"CG Integer Solve Failed (Status: {self.rmp.Status})")
            self.final_obj = None
            return None

    def extract_solution(self):
        solution = {}
        for m in self.M:
            for p_idx, pat in enumerate(self.patterns[m]):
                var = self.rmp.getVarByName(f"lam_{m}_{p_idx}")
                if var and var.X > 0.5:
                    solution[m] = pat["mapping"]
                    break
        return solution
