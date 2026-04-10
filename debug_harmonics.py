from CCL_Simulator.simcore import  Sim, PolicyEntry
from utils.tools import Datacenter
import simpy
import  time
import random
from utils.draw import draw_datacenter_topology
from utils.simulation import simulate
import random
import networkx as nx
from utils.ILP_tools import MultiTenantILP
from utils.CG_solver import ColumnGenerationSolver
from utils.Harmonics_tools import HarmonicsILP, HarmonicsHeuristic

MB = 1024 * 1024 * 8 # 1MB

def main():
    num_tenants = 4
    number_leaf = 3
    num_spine = 2
    per_leaf_servers = 4
    single_flow_size = 8 * MB
    collective = "allreduce"
    num_experiments = 3
    
    # 4 Groups of Results (Makespan)
    Random_results = []           # 1. Random Placement + No Sched
    Harmonics_Random_results = [] # 2. Random Placement + Harmonics Sched
    CG_results = []               # 3. CG Placement + No Sched
    Combined_results = []         # 4. CG Placement + Harmonics Sched
    
    # 4 Groups of Results (Average JCT)
    Random_avg_results = []
    Harmonics_Random_avg_results = []
    CG_avg_results = []
    Combined_avg_results = []
    
    size_factor = 1
    if collective == "allreduce":
        size_factor = 2 

    for exp in range(num_experiments):
        num_server = per_leaf_servers * number_leaf
        datacenter = Datacenter(number_leaf, num_spine, per_leaf_servers)
        ECMP_routing_table = datacenter.ECMP_table

        # Random initial tenant->servers assignment
        servers_per_tenant = num_server // num_tenants
        physical_server_index = list(range(num_server))
        random.shuffle(physical_server_index)
        tenant_mapping = {}
        for tenant in range(num_tenants):
            start = tenant * servers_per_tenant
            end = (tenant + 1) * servers_per_tenant
            assigned_servers = physical_server_index[start:end]
            tenant_mapping[tenant] = {rank: phys_srv for rank, phys_srv in enumerate(assigned_servers)}

        # Build ring flows (logical)
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
        
        # We need raw flows (unscaled V) for ILP if ILP expects raw bytes?
        # Actually HarmonicsILP expects V/scale if we follow the pattern, BUT inside HarmonicsILP we saw it uses V directly?
        # In HarmonicsILP, we did `link_load[e] += vol`. 
        # And `d = (val * 1e9 * 8.0) / self.cap[e]`.
        # So it expects `vol` to be `V/scale` (Gigabytes). 
        # Yes, `tenant_flows` passed here has `V/scale`. Correct.
        
        # We also need tenant_flows with raw bytes for ILP/CG? 
        # ILP/CG usually take the same `tenant_flows` structure.
        # Let's check `MultiTenantILP` or `CG`...
        # In `main.py` original, it passed `tenant_flows` (scaled).
        # So we are consistent.

        print(f"\n--- Experiment {exp+1}/{num_experiments} ---")

        # Group 1: Random + Default
        print("[1] Random + Default...")
        makespan_1, avg_1 = simulate(datacenter.topology, tenant_mapping, datacenter.paths, single_flow_size, collective)
        Random_results.append(makespan_1)
        Random_avg_results.append(avg_1)
        print(f"    Makespan: {makespan_1:.6f} s, Avg JCT: {avg_1:.6f} s")

        # Group 2: Random + Harmonics
        print("[2] Random + Harmonics...")
        harmonics_r = HarmonicsHeuristic(datacenter, tenant_mapping, tenant_flows, datacenter.paths, 
                                   single_flow_size, collective, verbose=True)
        sched_r = harmonics_r.solve()
        start_times_r = {t: sched_r[t][0] for t in sched_r} if sched_r else None
        makespan_2, avg_2 = simulate(datacenter.topology, tenant_mapping, datacenter.paths, single_flow_size, collective,
                                     tenant_start_times=start_times_r)
        Harmonics_Random_results.append(makespan_2)
        Harmonics_Random_avg_results.append(avg_2)
        print(f"    Makespan: {makespan_2:.6f} s, Avg JCT: {avg_2:.6f} s")

        # Solve Placement with CG
        print("Solving Placement with CG...")
        cg = ColumnGenerationSolver(datacenter, tenant_mapping, tenant_flows, verbose=False)
        cg_mapping = cg.solve()
        
        if cg_mapping:
            # Group 3: CG + Default
            print("[3] CG + Default...")
            makespan_3, avg_3 = simulate(datacenter.topology, cg_mapping, datacenter.paths, single_flow_size, collective)
            CG_results.append(makespan_3)
            CG_avg_results.append(avg_3)
            print(f"    Makespan: {makespan_3:.6f} s, Avg JCT: {avg_3:.6f} s")
            
            # Group 4: CG + Harmonics
            print("[4] CG + Harmonics...")
            harmonics_cg = HarmonicsILP(datacenter, cg_mapping, tenant_flows, datacenter.paths, 
                                        single_flow_size, collective, verbose=False)
            sched_cg = harmonics_cg.solve()
            start_times_cg = {t: sched_cg[t][0] for t in sched_cg} if sched_cg else None
            makespan_4, avg_4 = simulate(datacenter.topology, cg_mapping, datacenter.paths, single_flow_size, collective,
                                         tenant_start_times=start_times_cg)
            Combined_results.append(makespan_4)
            Combined_avg_results.append(avg_4)
            print(f"    Makespan: {makespan_4:.6f} s, Avg JCT: {avg_4:.6f} s")
        else:
            print("CG failed to find integer solution. Skipping CG groups for this run.")
            # Pad with baseline values to keep lists aligned or just skip?
            # Better to append NaN or the baseline values to not skew averages?
            # Let's append Baseline values so code doesn't break, but note it.
            CG_results.append(makespan_1) 
            CG_avg_results.append(avg_1)
            Combined_results.append(makespan_2)
            Combined_avg_results.append(avg_2)

    # Summary
    print("\n=== Final Results Summary ===")
    def avg(lst): return sum(lst)/len(lst) if lst else 0.0
    
    # Makespan Stats
    m_1 = avg(Random_results)
    m_2 = avg(Harmonics_Random_results)
    m_3 = avg(CG_results)
    m_4 = avg(Combined_results)
    
    # JCT Stats
    j_1 = avg(Random_avg_results)
    j_2 = avg(Harmonics_Random_avg_results)
    j_3 = avg(CG_avg_results)
    j_4 = avg(Combined_avg_results)
    
    print("--- Makespan ---")
    print(f"1. Random + Default:   {m_1:.6f} s")
    print(f"2. Random + Harmonics: {m_2:.6f} s (Improv: {m_1/m_2 if m_2>0 else 1:.2f}x)")
    print(f"3. CG + Default:       {m_3:.6f} s (Improv: {m_1/m_3 if m_3>0 else 1:.2f}x)")
    print(f"4. CG + Harmonics:     {m_4:.6f} s (Improv: {m_1/m_4 if m_4>0 else 1:.2f}x)")
    
    print("\n--- Average JCT ---")
    print(f"1. Random + Default:   {j_1:.6f} s")
    print(f"2. Random + Harmonics: {j_2:.6f} s (Improv: {j_1/j_2 if j_2>0 else 1:.2f}x)")
    print(f"3. CG + Default:       {j_3:.6f} s (Improv: {j_1/j_3 if j_3>0 else 1:.2f}x)")
    print(f"4. CG + Harmonics:     {j_4:.6f} s (Improv: {j_1/j_4 if j_4>0 else 1:.2f}x)")

if __name__ == "__main__":
    main()
