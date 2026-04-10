
from utils.tools import Datacenter
from utils.simulation import simulate
import random
import time
from utils.ILP_tools import MultiTenantILP
from utils.CG_solver import ColumnGenerationSolver

def test_cg_vs_ilp():
    num_tenants = 4
    number_leaf = 3
    num_spine = 2
    per_leaf_servers = 4
    single_flow_size = 8 * 1024 * 1024 * 8 # 8MB
    collective = "allreduce"
    num_experiments = 10
    
    size_factor = 1
    if collective == "allreduce":
        size_factor = 2 

    # Metrics
    ilp_objectives = []
    cg_objectives = []
    
    ilp_times = []
    cg_times = []
    
    ilp_makespans = []
    cg_makespans = []
    
    ilp_avg_jcts = []
    cg_avg_jcts = []
    
    gap_percentages = []

    print(f"Running {num_experiments} experiments to compare ILP vs CG...")
    print(f"Settings: {num_tenants} tenants, {number_leaf} leaves, {num_spine} spines.")

    for exp in range(num_experiments):
        print(f"\n--- Experiment {exp+1}/{num_experiments} ---")
        
        # 1. Setup Environment
        num_server = per_leaf_servers * number_leaf
        datacenter = Datacenter(number_leaf, num_spine, per_leaf_servers)
        
        # Random initial assignment
        servers_per_tenant = num_server // num_tenants
        physical_server_index = list(range(num_server))
        random.shuffle(physical_server_index)
        
        tenant_mapping = {}
        for tenant in range(num_tenants):
            start = tenant * servers_per_tenant
            end = (tenant + 1) * servers_per_tenant
            assigned_servers = physical_server_index[start:end]
            tenant_mapping[tenant] = {rank: phys_srv for rank, phys_srv in enumerate(assigned_servers)}
            
        # Build logical flows
        tenant_flows = {}
        scale = 1e9
        for tenant in tenant_mapping.keys():
            tenant_flows[tenant] = []
            k = len(tenant_mapping[tenant])
            ranks = list(range(k))
            for i in range(k):
                u = ranks[i]
                v = ranks[(i + 1) % k]
                V = (k - 1) * single_flow_size * size_factor
                tenant_flows[tenant].append((u, v, V/scale))

        # 2. Run ILP (The Baseline)
        print("Running ILP...")
        start_t = time.time()
        ilp = MultiTenantILP(datacenter, tenant_mapping, tenant_flows, verbose=False)
        ilp.solve()
        ilp_dur = time.time() - start_t
        
        if ilp.model.Status == 2: # Optimal
            ilp_obj = ilp.model.ObjVal
            ilp_mapping = ilp.get_X_mapping()
            ilp_mks, ilp_avg = simulate(datacenter.topology, ilp_mapping, datacenter.paths, single_flow_size, collective)
        else:
            print("ILP Failed/Infeasible")
            ilp_obj = float('inf')
            ilp_mks, ilp_avg = float('inf'), float('inf')

        # 3. Run CG (The Heuristic/Approximation)
        print("Running CG...")
        start_t = time.time()
        cg = ColumnGenerationSolver(datacenter, tenant_mapping, tenant_flows, verbose=False)
        cg_mapping = cg.solve(max_iter=50) # Limit iters for speed
        cg_dur = time.time() - start_t
        
        if cg.final_obj is not None:
            cg_obj = cg.final_obj
            cg_mks, cg_avg = simulate(datacenter.topology, cg_mapping, datacenter.paths, single_flow_size, collective)
        else:
            print("CG Failed")
            cg_obj = float('inf')
            cg_mks, cg_avg = float('inf'), float('inf')

        # 4. Record Data
        ilp_objectives.append(ilp_obj)
        cg_objectives.append(cg_obj)
        
        ilp_times.append(ilp_dur)
        cg_times.append(cg_dur)
        
        ilp_makespans.append(ilp_mks)
        cg_makespans.append(cg_mks)
        
        ilp_avg_jcts.append(ilp_avg)
        cg_avg_jcts.append(cg_avg)
        
        # Gap = (CG - ILP) / ILP * 100
        if ilp_obj > 1e-6:
            gap = (cg_obj - ilp_obj) / ilp_obj * 100.0
        else:
            gap = 0.0
        gap_percentages.append(gap)
        
        print(f"  [Objective] ILP: {ilp_obj:.6f} | CG: {cg_obj:.6f} | Gap: {gap:.2f}%")
        print(f"  [Sim Makespan] ILP: {ilp_mks:.6f}s | CG: {cg_mks:.6f}s")
        print(f"  [Sim Avg JCT]  ILP: {ilp_avg:.6f}s | CG: {cg_avg:.6f}s")

    # Final Summary
    print("\n=== FINAL COMPARISON SUMMARY ===")
    print(f"Avg ILP Objective: {sum(ilp_objectives)/num_experiments:.6f}")
    print(f"Avg CG  Objective: {sum(cg_objectives)/num_experiments:.6f}")
    print(f"Avg Optimality Gap: {sum(gap_percentages)/num_experiments:.2f}%")
    print("-" * 30)
    print(f"Avg ILP Time: {sum(ilp_times)/num_experiments:.4f} s")
    print(f"Avg CG  Time: {sum(cg_times)/num_experiments:.4f} s")
    print("-" * 30)
    print(f"Avg ILP Makespan: {sum(ilp_makespans)/num_experiments:.6f} s")
    print(f"Avg CG  Makespan: {sum(cg_makespans)/num_experiments:.6f} s")
    print(f"Avg ILP Avg JCT:  {sum(ilp_avg_jcts)/num_experiments:.6f} s")
    print(f"Avg CG  Avg JCT:  {sum(cg_avg_jcts)/num_experiments:.6f} s")

if __name__ == "__main__":
    test_cg_vs_ilp()
