from __future__ import annotations

import pickle
import sys

import networkx as nx
import simpy

from CCL_Simulator.simcore import Sim


def simulation_worker_main():
    try:
        data = pickle.load(sys.stdin.buffer)
    except EOFError:
        return

    topology_graph = data["topology"]
    policy_entries = data["policy"]

    topology = nx.DiGraph()
    for node, attrs in topology_graph.nodes(data=True):
        topology.add_node(node, **attrs)
    for src, dst, attrs in topology_graph.edges(data=True):
        topology.add_edge(src, dst, **attrs)

    env = simpy.Environment()
    sim = Sim(env, topology)
    sim.load_policy(policy_entries)
    sim.start()
    sim.run()

    tenant_makespans = {}
    for tx_id, completion_time in sim.tx_complete_time.items():
        flow_id = tx_id[0] if isinstance(tx_id, tuple) else tx_id
        if not isinstance(flow_id, str):
            continue

        tenant = flow_id.split("-")[0]
        tenant_makespans[tenant] = max(tenant_makespans.get(tenant, 0.0), completion_time)

    result = {
        "global_makespan": max(sim.tx_complete_time.values()) if sim.tx_complete_time else 0.0,
        "avg_tenant_makespan": (
            sum(tenant_makespans.values()) / len(tenant_makespans) if tenant_makespans else 0.0
        ),
        "tenant_makespans": tenant_makespans,
    }
    pickle.dump(result, sys.stdout.buffer)


if __name__ == "__main__":
    simulation_worker_main()
