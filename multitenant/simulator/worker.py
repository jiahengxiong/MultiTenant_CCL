from __future__ import annotations

import pickle
import sys

# Add the project root to the path so CCL_Simulator can be imported
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import networkx as nx
import simpy
from typing import Dict, List, Tuple, Union

try:
    import simcore_cpp
    USE_CPP = True
except ImportError:
    USE_CPP = False
    from CCL_Simulator.simcore import Sim


def simulation_worker_main():
    try:
        data = pickle.load(sys.stdin.buffer)
    except EOFError:
        return

    topology_graph = data["topology"]
    policy_entries = data["policy"]

    # Limit packets per chunk to 1024, with a minimum packet size of 1500 bytes
    max_chunk_size = max((e.chunk_size_bytes for e in policy_entries), default=0)
    packet_size_bytes = max(1500, (max_chunk_size + 1023) // 1024)

    if USE_CPP:
        sim = simcore_cpp.Sim(packet_size_bytes=packet_size_bytes, header_size_bytes=0)
        
        for node, attrs in topology_graph.nodes(data=True):
            ntype = attrs.get("type", None)
            num_qps = int(attrs.get("num_qps", 1))
            quantum_packets = int(attrs.get("quantum_packets", 1))
            tx_proc_delay = float(attrs.get("tx_proc_delay", 0.0))
            sw_proc_delay = float(attrs.get("sw_proc_delay", 0.0))
            gpu_store_delay = float(attrs.get("gpu_store_delay", 0.0))
            
            if ntype == "gpu":
                sim.add_gpu(str(node), num_qps, quantum_packets, tx_proc_delay, gpu_store_delay)
            elif ntype == "switch":
                sim.add_switch(str(node), num_qps, quantum_packets, tx_proc_delay, sw_proc_delay)
                
        for src, dst, attrs in topology_graph.edges(data=True):
            rate = float(attrs.get("link_rate_bps", 0.0))
            delay = float(attrs.get("prop_delay", 0.0))
            sim.add_link(str(src), str(dst), rate, delay)
            
        cpp_policy = []
        for e in policy_entries:
            pe = simcore_cpp.PolicyEntry()
            pe.chunk_id = str(e.chunk_id)
            pe.src = str(e.src)
            pe.dst = str(e.dst)
            pe.qpid = int(e.qpid)
            
            if isinstance(e.rate, str) and e.rate.strip().lower() == "max":
                pe.rate = 0.0
                pe.use_max_rate = True
            else:
                pe.rate = float(e.rate)
                pe.use_max_rate = False
                
            pe.chunk_size_bytes = int(e.chunk_size_bytes)
            pe.path = [str(x) for x in e.path]
            pe.time = float(e.time)
            pe.dependency = [str(x) for x in e.dependency] if e.dependency else []
            cpp_policy.append(pe)
            
        sim.load_policy(cpp_policy)
        sim.start()
        sim.run()
        
        tx_complete_time = sim.tx_complete_time
    else:
        topology = nx.DiGraph()
        for node, attrs in topology_graph.nodes(data=True):
            topology.add_node(node, **attrs)
        for src, dst, attrs in topology_graph.edges(data=True):
            topology.add_edge(src, dst, **attrs)
    
        env = simpy.Environment()
        sim = Sim(env, topology, packet_size_bytes=packet_size_bytes, header_size_bytes=0)
        sim.load_policy(policy_entries)
        sim.start()
        sim.run()
        tx_complete_time = sim.tx_complete_time

    tenant_makespans = {}
    for tx_id, completion_time in tx_complete_time.items():
        flow_id = tx_id[0] if isinstance(tx_id, tuple) else tx_id
        if not isinstance(flow_id, str):
            continue

        tenant = flow_id.split("-")[0]
        tenant_makespans[tenant] = max(tenant_makespans.get(tenant, 0.0), completion_time)

    result = {
        "global_makespan": max(tx_complete_time.values()) if tx_complete_time else 0.0,
        "avg_tenant_makespan": (
            sum(tenant_makespans.values()) / len(tenant_makespans) if tenant_makespans else 0.0
        ),
        "tenant_makespans": tenant_makespans,
    }
    pickle.dump(result, sys.stdout.buffer)


if __name__ == "__main__":
    simulation_worker_main()
