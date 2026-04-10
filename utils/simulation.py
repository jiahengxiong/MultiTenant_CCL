from CCL_Simulator.simcore import  Sim, PolicyEntry
from utils.tools import Datacenter
import simpy
import  time
import random
from utils.draw import draw_datacenter_topology
import networkx as nx

def build_topology(G):
    topo = nx.DiGraph()
    for node in G.nodes():
        if G.nodes[node]['type'] == 'server':
            topo.add_node(node, type="gpu", num_qps=8, quantum_packets=1, tx_proc_delay=0.0, gpu_store_delay=0.0)
        else:
            topo.add_node(node, type="switch", num_qps=8, quantum_packets=1, tx_proc_delay=0.0, gpu_store_delay=0.0)

    for (u, v) in G.edges():
        topo.add_edge(u, v,  link_rate_bps=G.edges[u, v]['capacity'], prop_delay=0.0)
        topo.add_edge(v, u, link_rate_bps=G.edges[u, v]['capacity'], prop_delay=0.0)

    return topo


def allgather_policy(tenant_servers, path_table, single_flow_size, tenant_start_times=None, tenant_rates=None):
    if tenant_start_times is None:
        tenant_start_times = {}
    if tenant_rates is None:
        tenant_rates = {}
    flow_list = []
    policy = []
    for tenant in tenant_servers:
        start_time = tenant_start_times.get(tenant, 0.0)
        rank_list = list(tenant_servers[tenant].keys())
        num_flows = len(rank_list) - 1
        for rank in rank_list:
            start_index = rank_list.index(rank)
            for i in range(num_flows):
                src = rank_list[(start_index + i) % len(rank_list)]
                dst = rank_list[(start_index + i + 1) % len(rank_list)]
                physical_src = tenant_servers[tenant][src]
                physical_dst = tenant_servers[tenant][dst]
                flow_list.append([f"{tenant}-{rank}", physical_src, physical_dst, path_table[(physical_src, physical_dst)], start_time])
    for flow_id, src, dst, path, start_time in flow_list:
        tenant = flow_id.split('-')[0]
        # Use tenant-specific rate if available (must be in bps), otherwise 'Max'
        # The simulator expects 'Max' or a number (bps).
        rate = tenant_rates.get(int(tenant), 'Max')
        policy.append(PolicyEntry(flow_id, src, dst, 0, rate, single_flow_size, path, time=start_time))

    # print("Flow list:", flow_list)

    return policy

def allreduce_policy(tenant_servers, path_table, chunk_size, tenant_start_times=None, tenant_rates=None):
    if tenant_start_times is None:
        tenant_start_times = {}
    if tenant_rates is None:
        tenant_rates = {}
    policy = []
    for tenant, mapping in tenant_servers.items():
        start_time = tenant_start_times.get(tenant, 0.0)
        rate = tenant_rates.get(tenant, 'Max')
        
        # Get logical ranks sorted
        ranks = sorted(mapping.keys())
        P = len(ranks)
        if P < 2:
            continue
        
        # Map logical rank to physical server
        phys_ranks = [mapping[r] for r in ranks]
        
        # We process P chunks (one originating from each rank)
        # ----------------------------------------------------
        # Phase 1: Reduce-Scatter
        # ----------------------------------------------------
        # For each chunk c (originating at rank c), it travels P-1 hops.
        # Hop s (0 to P-2):
        #   src = (c + s) % P
        #   dst = (c + s + 1) % P
        #   Dependency: Previous hop for this chunk must have arrived at src.
        
        for c in range(P): # For each chunk ID c
            for s in range(P - 1): # P-1 steps
                src_idx = (c + s) % P
                dst_idx = (c + s + 1) % P
                
                src_phys = phys_ranks[src_idx]
                dst_phys = phys_ranks[dst_idx]
                
                # Unique ID for this specific transfer
                chunk_id = f"{tenant}-RS-C{c}-S{s}"
                
                # Path lookup
                path = path_table.get((src_phys, dst_phys))
                if not path:
                    # Fallback or error? usually ECMP path exists
                    # Assuming full mesh reachability in simulator
                    continue
                
                # Dependencies
                deps = []
                if s > 0:
                    # Depend on the previous step of RS for this chunk
                    prev_chunk_id = f"{tenant}-RS-C{c}-S{s-1}"
                    deps.append(prev_chunk_id)
                
                # Create PolicyEntry
                # Note: rate='Max', time=0.0
                pe = PolicyEntry(
                    chunk_id=chunk_id,
                    src=src_phys,
                    dst=dst_phys,
                    qpid=0,
                    rate=rate,
                    chunk_size_bytes=chunk_size,
                    path=path,
                    time=start_time,
                    dependency=deps
                )
                policy.append(pe)

        # ----------------------------------------------------
        # Phase 2: All-Gather
        # ----------------------------------------------------
        # Starts after Reduce-Scatter.
        # At the end of RS, the fully reduced chunk c resides at node (c + P - 1) % P.
        # Let's call this root_node = (c - 1) % P.
        # From here, we broadcast this fully reduced chunk to everyone else along the ring.
        
        for c in range(P):
            # The node holding the fully reduced chunk c
            root_idx = (c + P - 1) % P
            
            for s in range(P - 1): # P-1 steps
                src_idx = (root_idx + s) % P
                dst_idx = (root_idx + s + 1) % P
                
                src_phys = phys_ranks[src_idx]
                dst_phys = phys_ranks[dst_idx]
                
                chunk_id = f"{tenant}-AG-C{c}-S{s}"
                
                path = path_table.get((src_phys, dst_phys))
                if not path: continue
                
                deps = []
                if s == 0:
                    # Depend on the LAST step of RS for this chunk c
                    # Last RS step was s = P - 2
                    prev_chunk_id = f"{tenant}-RS-C{c}-S{P-2}"
                    deps.append(prev_chunk_id)
                else:
                    # Depend on previous AG step
                    prev_chunk_id = f"{tenant}-AG-C{c}-S{s-1}"
                    deps.append(prev_chunk_id)
                
                pe = PolicyEntry(
                    chunk_id=chunk_id,
                    src=src_phys,
                    dst=dst_phys,
                    qpid=0,
                    rate=rate,
                    chunk_size_bytes=chunk_size,
                    path=path,
                    time=start_time,
                    dependency=deps
                )
                policy.append(pe)
                
    return policy

import pickle
import subprocess
import sys
import os

def simulate(topology, tenant_servers, path_table, single_flow_size, collective, tenant_start_times=None, tenant_rates=None):
    # Prepare policy (this runs in CPython)
    # Note: build_topology logic is now just for creating the graph object to pass
    # We construct a NetworkX graph that carries all necessary attributes
    
    # Reconstruct graph with attributes expected by simcore
    # The original 'topology' is a networkx graph from tools.py (Datacenter)
    # But simcore expects specific node/edge attributes.
    
    # We reuse build_topology logic but return the graph object
    sim_topo = build_topology(topology)
    
    policy = []
    if collective == 'allgather':
        policy = allgather_policy(tenant_servers, path_table, int(single_flow_size/8.0), tenant_start_times, tenant_rates)
    elif collective == 'allreduce':
        policy = allreduce_policy(tenant_servers, path_table, int(single_flow_size/8.0), tenant_start_times, tenant_rates)

    # Package data
    data = {
        'topology': sim_topo,
        'policy': policy
    }
    
    # Serialize
    input_bytes = pickle.dumps(data)
    
    # Determine python executable (prefer pypy3 if available, else python3)
    # You can hardcode this if needed
    pypy_exec = "pypy3"
    
    # Fallback to sys.executable if pypy3 not found (for safety during dev)
    # Check if pypy3 exists
    from shutil import which
    if which("pypy3") is None:
        print("[Warning] pypy3 not found, falling back to current python (slow).")
        pypy_exec = sys.executable

    # Run subprocess
    # We run 'run_simulation.py' which must be in the same root or accessible path
    # Assuming run_simulation.py is in the project root
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_simulation.py')
    
    process = subprocess.Popen(
        [pypy_exec, script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr  # Pass stderr through for debugging
    )
    
    try:
        stdout_data, _ = process.communicate(input=input_bytes)
        if process.returncode != 0:
            raise RuntimeError(f"Simulation subprocess failed with code {process.returncode}")
            
        result = pickle.loads(stdout_data)
        return result['global_makespan'], result['avg_tenant_makespan']
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        # Fallback to local execution if IPC fails? 
        # Or just re-raise
        raise e

