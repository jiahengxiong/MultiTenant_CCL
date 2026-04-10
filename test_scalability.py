
from utils.tools import Datacenter
from utils.simulation import simulate
import random
import time
from utils.ILP_tools import MultiTenantILP
from utils.CG_solver import ColumnGenerationSolver
import sys

def run_experiment(num_tenants, per_tenant_size=4):
    # Scale datacenter to fit tenants
    # We need num_tenants * per_tenant_size servers
    total_servers_needed = num_tenants * per_tenant_size
    
    # Heuristic to config datacenter
    per_leaf = 8
    num_leaf = (total_servers_needed + per_leaf - 1) // per_leaf
    if num_leaf < 2: num_leaf = 2
    num_spine = max(2, num_leaf // 2)
    
    print(f"\n[Scalability Test] Tenants: {num_tenants} | Total Servers: {total_servers_needed} | Topology: {num_leaf} Leaves, {num_spine} Spines")
    
    datacenter = Datacenter(num_leaf, num_spine, per_leaf)
    
    # Check actual capacity
    real_total = len(datacenter.get_all_servers())
    if real_total < total_servers_needed:
        # adjust per_leaf
        per_leaf = (total_servers_needed + num_leaf - 1) // num_leaf
        datacenter = Datacenter(num_leaf, num_spine, per_leaf)
        
    # Generate Scenario
    physical_server_index = list(range(total_servers_needed)) # Use first N servers
    # Map to actual server IDs
    all_servers = datacenter.get_all_servers()
    # random.shuffle(all_servers) # Keep topology locality somewhat? Or random.
    # Let's shuffle to make it interesting
    used_servers = all_servers[:total_servers_needed]
    random.shuffle(used_servers)
    
    tenant_mapping = {}
    idx = 0
    for t in range(num_tenants):
        t_map = {}
        for r in range(per_tenant_size):
            t_map[r] = used_servers[idx]
            idx += 1
        tenant_mapping[t] = t_map

    # Flows
    single_flow_size = 8 * 1024 * 1024 * 8 # 8MB
    size_factor = 2 # allreduce
    tenant_flows = {}
    scale = 1e9
    for t in tenant_mapping:
        tenant_flows[t] = []
        k = len(tenant_mapping[t])
        ranks = list(range(k))
        for i in range(k):
            u = ranks[i]
            v = ranks[(i+1)%k]
            V = (k-1)*single_flow_size*size_factor
            tenant_flows[t].append((u, v, V/scale))

    # Run ILP
    print("  > Running ILP...", end="", flush=True)
    start_t = time.time()
    ilp = MultiTenantILP(datacenter, tenant_mapping, tenant_flows, verbose=False)
    ilp.model.Params.TimeLimit = 60  # User requested timeout
    ilp.solve()
    ilp_dur = time.time() - start_t
    
    if ilp.model.Status == 2: # Optimal
        ilp_obj = ilp.model.ObjVal
    elif ilp.model.SolCount > 0: # Found solution but maybe TimeLimit
        ilp_obj = ilp.model.ObjVal
        print(f" (Status {ilp.model.Status})", end="")
    else:
        ilp_obj = -1.0
        
    print(f" Done. Time: {ilp_dur:.4f}s | Obj: {ilp_obj:.4f}")

    # Run CG
    print("  > Running CG... ", end="", flush=True)
    start_t = time.time()
    cg = ColumnGenerationSolver(datacenter, tenant_mapping, tenant_flows, verbose=False)
    cg.solve(max_iter=50)
    cg_dur = time.time() - start_t
    cg_obj = cg.final_obj if cg.final_obj else -1.0
    print(f" Done. Time: {cg_dur:.4f}s | Obj: {cg_obj:.4f}")
    
    return ilp_dur, cg_dur

def main():
    tenant_counts = [4, 8, 16, 32]
    results = []
    
    print("=== SCALABILITY BENCHMARK: ILP vs CG ===")
    
    for n in tenant_counts:
        try:
            i_time, c_time = run_experiment(n)
            results.append((n, i_time, c_time))
        except Exception as e:
            print(f"Error in experiment {n}: {e}")
            
    print("\n=== SUMMARY ===")
    print(f"{'Tenants':<10} | {'ILP Time (s)':<15} | {'CG Time (s)':<15} | {'Ratio (ILP/CG)':<15}")
    print("-" * 60)
    for n, i_t, c_t in results:
        ratio = i_time / c_time if c_time > 0 else 0
        print(f"{n:<10} | {i_t:<15.4f} | {c_t:<15.4f} | {i_t/c_t:<15.2f}")

if __name__ == "__main__":
    main()
