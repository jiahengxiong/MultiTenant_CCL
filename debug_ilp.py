
from utils.tools import Datacenter
from utils.ILP_tools import MultiTenantILP
import random

def debug_ilp():
    print("--- Debugging ILP ---")
    
    # 1. Setup Datacenter
    # 3 leaves, 2 spines, 2 servers per leaf. Total 6 servers.
    num_leaf = 3
    num_spine = 2
    per_leaf = 2
    dc = Datacenter(num_leaf, num_spine, per_leaf)
    
    # 2. Setup Tenants
    num_tenants = 2
    servers_per_tenant = 3
    # Initial mapping (Random)
    # T0: [0, 1, 2], T1: [3, 4, 5] (Example)
    # Servers 0,1 on Leaf 0. Server 2,3 on Leaf 1. Server 4,5 on Leaf 2.
    
    # Let's force a mapping that might cause contention
    # T0: {0:0, 1:2, 2:4} (Spread across L0, L1, L2)
    # T1: {0:1, 1:3, 2:5} (Spread across L0, L1, L2)
    
    # Physical servers:
    # L0: 0, 1
    # L1: 2, 3
    # L2: 4, 5
    
    tenant_mapping = {
        0: {0: 0, 1: 2, 2: 4},
        1: {0: 1, 1: 3, 2: 5}
    }
    
    # 3. Setup Flows (Ring)
    MB = 1024 * 1024 * 8 # User's value (8 million)
    single_flow_size = 8 * MB # 64 million
    
    tenant_flows = {}
    for tenant in tenant_mapping:
        tenant_flows[tenant] = []
        k = len(tenant_mapping[tenant])
        ranks = list(range(k))
        for i in range(k):
            u = ranks[i]
            v = ranks[(i + 1) % k]
            V = (k - 1) * single_flow_size
            tenant_flows[tenant].append((u, v, V))
            
    print(f"Flow Volume V = {tenant_flows[0][0][2]}")
    
    # Scale units to avoid numeric issues
    scale = 1e9
    print(f"\n[DEBUG] Scaling units by 1e9. Capacities 25G -> 25. V -> V/1e9.")
    
    # Modify datacenter capacities
    for u,v in dc.topology.edges():
        dc.topology[u][v]['capacity'] /= scale
        
    # Modify flow volumes
    for m in tenant_flows:
        for i in range(len(tenant_flows[m])):
            u, v, V = tenant_flows[m][i]
            tenant_flows[m][i] = (u, v, V / scale)
            
    print(f"Scaled Flow Volume V = {tenant_flows[0][0][2]}")
    
    # 4. Build and Solve ILP
    ilp = MultiTenantILP(dc, tenant_mapping, tenant_flows, verbose=True)
    
    # DEBUG: Force b to be large to find bottleneck
    # print("\n[DEBUG] Forcing b[(0,0)] >= 10e9 to find bottleneck...")
    # ilp.model.addConstr(ilp.b[(0,0)] >= 10e9, name="Force_b_large")
    
    ilp.solve()
    
    print(f"BigM = {ilp.data['Bmax']}")
    
    # Inspect bhat vs b
    print("\n--- bhat vs b check ---")
    for m in ilp.data["M"]:
        for j in range(len(ilp.data["flows"][m])):
            b_val = ilp.b[(m,j)].X
            # Find active U
            for s in ilp.data["S"][m]:
                for t in ilp.data["S"][m]:
                    if s == t: continue
                    u_val = ilp.U[(m,j,s,t)].X
                    if u_val > 0.5:
                         # Check bhat for this path
                         pe = ilp.data["path_edges"][(s,t)]
                         for ell in pe:
                             bh_key = (m, j, s, t, ell)
                             bhat_val = ilp.bhat[bh_key].X
                             print(f"Flow {m},{j} Path {s}->{t} Link {ell}: b={b_val:.4e} bhat={bhat_val:.4e} U={u_val}")
                             if abs(bhat_val - b_val) > 1e-4:
                                 print("  MISMATCH WARNING!")

    
    if ilp.model.Status == 3: # Infeasible
        print("Model is Infeasible! Computing IIS...")
        ilp.model.computeIIS()
        ilp.model.write("model.ilp")
        print("IIS written to model.ilp")
        for c in ilp.model.getConstrs():
            if c.IISConstr:
                print(f"Constraint {c.ConstrName} is in IIS")
        for q in ilp.model.getQConstrs():
             if q.IISQConstr:
                 print(f"QConstraint {q.QCName} is in IIS")
        return

    print(f"\nObjVal (T_max) = {ilp.model.ObjVal}")
    
    # 5. Inspect Solution
    print("\n--- Rates (b) ---")
    for k, v in ilp.b.items():
        print(f"Flow {k}: {v.X:.4f} Gbps")
        
    print("\n--- Mapping (X) ---")
    mapping = ilp.get_X_mapping()
    print(mapping)
    
    print("\n--- Link Loads ---")
    # Check loads manually
    # We need to see which links are congested
    
    # Iterate over all links and sum bhat
    cap = ilp.data["cap"]
    for ell in ilp.data["L"]:
        load = 0
        contributors = []
        for m in ilp.data["M"]:
            for j in range(len(ilp.data["flows"][m])):
                # Check if this flow uses this link
                # We need to find the active path
                # active path depends on U
                
                # Find active s,t
                active_s = -1
                active_t = -1
                for s in ilp.data["S"][m]:
                    for t in ilp.data["S"][m]:
                        if s == t: continue
                        if ilp.U[(m, j, s, t)].X > 0.5:
                            active_s = s
                            active_t = t
                            break
                    if active_s != -1: break
                
                if active_s != -1:
                    path = dc.paths[(active_s, active_t)]
                    edges = dc.path_to_edges(path)
                    if ell in edges:
                        b_val = ilp.b[(m,j)].X
                        load += b_val
                        contributors.append(f"T{m}_F{j}({b_val:.2f}G)")
        
        if load > 1e-9:
            print(f"Link {ell}: Load={load:.4f}G / Cap={cap[ell]:.1f}G  [{', '.join(contributors)}]")

if __name__ == "__main__":
    debug_ilp()
