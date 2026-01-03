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

MB = 1024 * 1024 * 8 # 1MB




def main():
    num_tenants = 4
    number_leaf = 3
    num_spine = 2
    per_leaf_servers = 4
    single_flow_size = 8 * MB
    collective = "allreduce"
    num_experiments = 20
    Random_results = []
    ILP_results = []
    size_factor = 1
    if collective == "allreduce":
        size_factor = 2 

    for exp in range(num_experiments):
        

        num_server = per_leaf_servers * number_leaf
        datacenter = Datacenter(number_leaf, num_spine, per_leaf_servers)

        # 如果你后面要用 ECMP（你现在先打印看看）
        ECMP_routing_table = datacenter.ECMP_table

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
        print("Solving with ILP...")
        ilp = MultiTenantILP(datacenter, tenant_mapping, tenant_flows, verbose=False)
        ilp.solve()
        if ilp.model.Status == 2:
            ILP_makespan, ILP_avg = simulate(datacenter.topology, ilp.get_X_mapping(), datacenter.paths, single_flow_size,collective)
            print(f"ILP ObjVal = {ilp.model.ObjVal:.6f}")
            print(f"ILP makespan = {ILP_makespan:.6f} s, Avg tenant = {ILP_avg:.6f} s")
            ILP_results.append(ILP_makespan)
        
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
        else:
             print("CG failed to find integer solution")

        # # -------- Debug prints (minimal) --------
        print("tenant_mapping:", tenant_mapping)
        makespan, rand_avg = simulate(datacenter.topology, tenant_mapping, datacenter.paths, single_flow_size,collective)
        print(f"Random makespan = {makespan:.6f} s, Avg tenant = {rand_avg:.6f} s")
        Random_results.append(makespan)
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
    print(f"Random makespan average = {sum(Random_results)/num_experiments:.6f} s")
    if CG_results:
        print(f"CG makespan average = {sum(CG_results)/len(CG_results):.6f} s")
        print(f"Speedup (CG vs Random) = {sum(Random_results)/sum(CG_results) * (len(CG_results)/num_experiments):.6f}")
    
    if ILP_results:
        print(f"ILP makespan average = {sum(ILP_results)/len(ILP_results):.6f} s")
        print(f"Speedup (ILP vs Random) = {sum(Random_results)/sum(ILP_results) * (len(ILP_results)/num_experiments):.6f}")

    print("CG makespan list:", CG_results)
    print("ILP makespan list:", ILP_results)
    print("Random makespan list:", Random_results)
    # print("b values:", {(m,j): ilp.b[(m,j)].X for (m,j) in ilp.b})
        # print("flows example tenant0:", ilp_data["flows"][1])
        # print("num links:", len(ilp_data["L"]), "Bmax:", ilp_data["Bmax"])
        # print("ECMP_routing_table:", ECMP_routing_table)
        # print('----------------------')
        # for key, value in ilp_data.items():
        #     print(key, value)
        #
        # return ilp_data


if __name__ == "__main__":
    ilp_data = main()







