import pickle
import sys
import simpy
import networkx as nx
from CCL_Simulator.simcore import Sim, PolicyEntry

def run_simulation_worker():
    # Read pickled data from stdin
    try:
        data = pickle.load(sys.stdin.buffer)
    except EOFError:
        return

    topology_graph = data['topology']
    policy_entries = data['policy']
    
    # Rebuild topology (ensure correct types)
    topo = nx.DiGraph()
    for node, attrs in topology_graph.nodes(data=True):
        topo.add_node(node, **attrs)
    for u, v, attrs in topology_graph.edges(data=True):
        topo.add_edge(u, v, **attrs)

    # Setup SimPy
    env = simpy.Environment()
    sim = Sim(env, topo)
    sim.load_policy(policy_entries)
    
    sim.start()
    sim.run()
    
    # Process results
    tenant_makespans = {}
    for tx_id, t in sim.tx_complete_time.items():
        flow_id = tx_id
        if isinstance(tx_id, tuple):
             flow_id = tx_id[0]
        
        if isinstance(flow_id, str):
            tenant = flow_id.split('-')[0]
            try:
                # Handle cases where tenant ID might be complex string
                # But usually it's "0-..."
                if tenant not in tenant_makespans:
                    tenant_makespans[tenant] = 0.0
                if t > tenant_makespans[tenant]:
                    tenant_makespans[tenant] = t
            except:
                pass

    global_makespan = max(sim.tx_complete_time.values()) if sim.tx_complete_time else 0.0
    avg_tenant_makespan = 0.0
    if tenant_makespans:
        avg_tenant_makespan = sum(tenant_makespans.values()) / len(tenant_makespans)

    # Write results to stdout (pickled)
    result = {
        'global_makespan': global_makespan,
        'avg_tenant_makespan': avg_tenant_makespan,
        'tenant_makespans': tenant_makespans
    }
    pickle.dump(result, sys.stdout.buffer)

if __name__ == "__main__":
    run_simulation_worker()
