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
from utils.Harmonics_tools import HarmonicsHeuristic, HarmonicsILP

MB = 1024 * 1024 * 8 # 1MB




def main(num_tenants=3, num_experiments=10):
    number_leaf = 3
    num_spine = 2
    per_leaf_servers = 4
    single_flow_size = 8 * MB
    collective = "allgather"
    
    # 4 Groups of Results (Makespan)
    Random_results = []           # 1. Random Placement + No Sched
    Harmonics_Random_results = [] # 2. Random Placement + Harmonics Sched
    CG_results = []               # 3. CG Placement + No Sched
    Combined_results = []         # 4. CG Placement + Harmonics Sched
    
    # 4 Groups of Results (Avg JCT)
    Random_avg_results = []
    Harmonics_Random_avg_results = []
    CG_avg_results = []
    Combined_avg_results = []
    
    ILP_results = []              # (Optional) ILP Placement + No Sched
    
    size_factor = 1
    if collective == "allreduce":
        size_factor = 2 
    num_server = per_leaf_servers * number_leaf
    datacenter = Datacenter(number_leaf, num_spine, per_leaf_servers)

    # 如果你后面要用 ECMP（你现在先打印看看）
    ECMP_routing_table = datacenter.ECMP_table

    for exp in range(num_experiments):
        

        

        # -------- Random initial tenant->servers assignment --------
        servers_per_tenant = num_server // num_tenants
        physical_server_index = list(range(num_server))
        random.shuffle(physical_server_index)

        tenant_mapping = {}  # {tenant: {rank: physical_server}}
        for tenant in range(num_tenants):
            start = tenant * servers_per_tenant
            end = (tenant + 1) * servers_per_tenant
            assigned_servers = physical_server_index[start:end]
            tenant_mapping[tenant] = {rank: phys_srv for rank, phys_srv in enumerate(assigned_servers)}

        # simulate(datacenter.topology, tenant_mapping, datacenter.paths, single_flow_size,collective)

        # -------- Build ring flows (logical) --------
        tenant_flows = {}
        scale = 1e9
        for tenant in tenant_mapping.keys():
            tenant_flows[tenant] = []
            k = len(tenant_mapping[tenant])  # number of ranks/servers for this tenant
            ranks = list(range(k))           # logical ranks: 0..k-1
            for i in range(k):
                u = ranks[i]
                v = ranks[(i + 1) % k]
                V = (k - 1) * single_flow_size * size_factor
                tenant_flows[tenant].append((u, v, V/scale))
        

        # -------- Collect ILP inputs --------
        # print("Solving with ILP...")
        # ilp = MultiTenantILP(datacenter, tenant_mapping, tenant_flows, verbose=False)
        # ilp.solve()
        # if ilp.model.Status == 2:
        #     ILP_makespan, ILP_avg = simulate(datacenter.topology, ilp.get_X_mapping(), datacenter.paths, single_flow_size,collective)
        #     print(f"ILP ObjVal = {ilp.model.ObjVal:.6f}")
        #     print(f"ILP makespan = {ILP_makespan:.6f} s, Avg tenant = {ILP_avg:.6f} s")
        #     ILP_results.append(ILP_makespan)
        
        # -------- Solve with Column Generation --------
        if 'CG_results' not in locals(): CG_results = []
        print("Solving with Column Generation...")
        cg = ColumnGenerationSolver(datacenter, tenant_mapping, tenant_flows, verbose=False)
        cg_mapping = cg.solve()
        
        if cg_mapping:
             print("CG Solution Found")
             print(f"CG ObjVal = {cg.final_obj:.6f}")
             CG_makespan, CG_avg = simulate(datacenter.topology, cg_mapping, datacenter.paths, single_flow_size, collective)
             print(f"CG makespan = {CG_makespan:.6f} s, Avg tenant = {CG_avg:.6f} s")
             CG_results.append(CG_makespan)
             CG_avg_results.append(CG_avg)
             
             # Harmonics + CG
             print("Solving Harmonics (CG)...")
             harmonics_cg = HarmonicsILP(datacenter, cg_mapping, tenant_flows, datacenter.paths, single_flow_size, collective, estimation=CG_makespan)
             sched_cg = harmonics_cg.solve()
             start_times_cg = {t: sched_cg[t][0] for t in sched_cg} if sched_cg else None
             hcg_makespan, hcg_avg = simulate(datacenter.topology, cg_mapping, datacenter.paths, single_flow_size, collective, tenant_start_times=start_times_cg)
             print(f"Harmonics+CG makespan = {hcg_makespan:.6f} s, Avg tenant = {hcg_avg:.6f} s")
             Combined_results.append(hcg_makespan)
             Combined_avg_results.append(hcg_avg)
        else:
             print("CG failed to find integer solution")
             # Pad with baselines
             CG_results.append(makespan)
             CG_avg_results.append(rand_avg)
             Combined_results.append(hm_makespan)
             Combined_avg_results.append(hm_avg)

        # -------- Debug prints (minimal) --------
        print("tenant_mapping:", tenant_mapping)
        makespan, rand_avg = simulate(datacenter.topology, tenant_mapping, datacenter.paths, single_flow_size,collective)
        print(f"Random makespan = {makespan:.6f} s, Avg tenant = {rand_avg:.6f} s")
        Random_results.append(makespan)
        Random_avg_results.append(rand_avg)

        # Harmonics + Random
        print("Solving Harmonics (Random)...")
        harmonics_r = HarmonicsILP(datacenter, tenant_mapping, tenant_flows, datacenter.paths, single_flow_size, collective, estimation=makespan)
        sched_r = harmonics_r.solve()
        start_times_r = {t: sched_r[t][0] for t in sched_r} if sched_r else None
        hm_makespan, hm_avg = simulate(datacenter.topology, tenant_mapping, datacenter.paths, single_flow_size, collective, tenant_start_times=start_times_r)
        print(f"Harmonics+Random makespan = {hm_makespan:.6f} s, Avg tenant = {hm_avg:.6f} s")
        Harmonics_Random_results.append(hm_makespan)
        Harmonics_Random_avg_results.append(hm_avg)
        # print("b:", {(m,j): ilp.b[(m,j)].X for (m,j) in ilp.b})
        # print("Tflow:", {(m,j): ilp.T_flow[(m,j)].X for (m,j) in ilp.T_flow})
        # print("Tmax:", ilp.T_max.X)
        # dc = datacenter
        # print("server->leaf cap:", [dc.topology[u][v]["capacity"] for (u,v) in dc.topology.edges() if dc.topology.nodes[u]["type"]=="server"])
        # print("leaf->spine cap sample:", [(u,v,dc.topology[u][v]["capacity"]) for (u,v) in dc.topology.edges()
        #                                 if dc.topology.nodes[u]["type"]=="leaf" and dc.topology.nodes[v]["type"]=="spine"][:5])
        # print("min cap:", min(nx.get_edge_attributes(dc.topology, "capacity").values()))
        # print("max cap:", max(nx.get_edge_attributes(dc.topology, "capacity").values()))
        # from collections import defaultdict

        # load = defaultdict(float)
        # count = defaultdict(int)

        # for (m,j,s,t,ell), var in ilp.bhat.items():
        #     if var.X > 1e-9:
        #         load[ell] += var.X
        #         count[ell] += 1

        # util = []
        # for ell, ld in load.items():
        #     cap = datacenter.topology[ell[0]][ell[1]]["capacity"]
        #     util.append((ld/cap, ell, ld, cap, count[ell]))

        # util.sort(reverse=True, key=lambda x: x[0])

        # print("=== Top bottlenecks by utilization ===")
        # for u, ell, ld, cap, cnt in util[:10]:
        #     print(f"ell {ell} util {u:.3f} load {ld:.3e} cap {cap:.3e} count {cnt}")
    
    print("\n=== Final Summary ===")
    random_makespan = sum(Random_results)/num_experiments
    harmonics_random_makespan = sum(Harmonics_Random_results)/num_experiments
    
    print(f"Random Makespan: {random_makespan:.6f} s")
    print(f"Harmonics+Random Makespan: {harmonics_random_makespan:.6f} s")
    
    cg_makespan = 0
    harmonics_cg_makespan = 0
    if CG_results:
        cg_makespan = sum(CG_results)/len(CG_results)
        harmonics_cg_makespan = sum(Combined_results)/len(Combined_results)
        print(f"CG Makespan: {cg_makespan:.6f} s")
        print(f"Harmonics+CG Makespan: {harmonics_cg_makespan:.6f} s")

    random_avg = sum(Random_avg_results)/num_experiments
    harmonics_random_avg = sum(Harmonics_Random_avg_results)/num_experiments
    print(f"\nRandom Avg JCT: {random_avg:.6f} s")
    print(f"Harmonics+Random Avg JCT: {harmonics_random_avg:.6f} s")
    
    cg_avg = 0
    harmonics_cg_avg = 0
    if CG_avg_results:
        cg_avg = sum(CG_avg_results)/len(CG_avg_results)
        harmonics_cg_avg = sum(Combined_avg_results)/len(Combined_avg_results)
        print(f"CG Avg JCT: {cg_avg:.6f} s")
        print(f"Harmonics+CG Avg JCT: {harmonics_cg_avg:.6f} s")

    return {
        "Default": random_makespan,
        "Harmonics+Default": harmonics_random_makespan,
        "CG": cg_makespan,
        "Harmonics+CG": harmonics_cg_makespan,
        "Default_avg": random_avg,
        "Harmonics+Default_avg": harmonics_random_avg,
        "CG_avg": cg_avg,
        "Harmonics+CG_avg": harmonics_cg_avg
    }


if __name__ == "__main__":
    ilp_data = main()
