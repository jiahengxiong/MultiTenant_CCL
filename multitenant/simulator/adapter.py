from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path
from shutil import which

import networkx as nx

from CCL_Simulator.simcore import PolicyEntry


def build_simulator_topology(graph: nx.DiGraph) -> nx.DiGraph:
    topo = nx.DiGraph()
    for node, attrs in graph.nodes(data=True):
        if attrs["type"] == "server":
            topo.add_node(
                node,
                type="gpu",
                num_qps=8,
                quantum_packets=1,
                tx_proc_delay=0.0,
                gpu_store_delay=0.0,
            )
        else:
            topo.add_node(
                node,
                type="switch",
                num_qps=8,
                quantum_packets=1,
                tx_proc_delay=0.0,
                gpu_store_delay=0.0,
            )

    for src, dst, attrs in graph.edges(data=True):
        topo.add_edge(src, dst, link_rate_bps=attrs["capacity"], prop_delay=0.0)

    return topo


def allgather_policy(
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int], list[int]],
    single_flow_size_bytes: int,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
) -> list[PolicyEntry]:
    tenant_start_times = tenant_start_times or {}
    tenant_rates = tenant_rates or {}
    policy: list[PolicyEntry] = []

    for tenant, mapping in tenant_servers.items():
        start_time = tenant_start_times.get(tenant, 0.0)
        rate = tenant_rates.get(tenant, "Max")
        ranks = sorted(mapping.keys())
        participant_count = len(ranks)
        if participant_count < 2:
            continue

        physical_ranks = [mapping[rank] for rank in ranks]
        for chunk_idx in range(participant_count):
            for step in range(participant_count - 1):
                src_idx = (chunk_idx + step) % participant_count
                dst_idx = (chunk_idx + step + 1) % participant_count
                src_phys = physical_ranks[src_idx]
                dst_phys = physical_ranks[dst_idx]
                path = path_table.get((src_phys, dst_phys))
                if not path:
                    continue

                deps = []
                if step > 0:
                    deps.append(f"{tenant}-AG-C{chunk_idx}-S{step - 1}")

                policy.append(
                    PolicyEntry(
                        chunk_id=f"{tenant}-AG-C{chunk_idx}-S{step}",
                        src=src_phys,
                        dst=dst_phys,
                        qpid=0,
                        rate=rate,
                        chunk_size_bytes=single_flow_size_bytes,
                        path=path,
                        time=start_time,
                        dependency=deps,
                    )
                )
    return policy


def allreduce_policy(
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int], list[int]],
    chunk_size_bytes: int,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
) -> list[PolicyEntry]:
    tenant_start_times = tenant_start_times or {}
    tenant_rates = tenant_rates or {}
    policy: list[PolicyEntry] = []

    for tenant, mapping in tenant_servers.items():
        start_time = tenant_start_times.get(tenant, 0.0)
        rate = tenant_rates.get(tenant, "Max")
        ranks = sorted(mapping.keys())
        participant_count = len(ranks)
        if participant_count < 2:
            continue

        physical_ranks = [mapping[rank] for rank in ranks]

        for chunk_idx in range(participant_count):
            for step in range(participant_count - 1):
                src_idx = (chunk_idx + step) % participant_count
                dst_idx = (chunk_idx + step + 1) % participant_count
                src_phys = physical_ranks[src_idx]
                dst_phys = physical_ranks[dst_idx]
                path = path_table.get((src_phys, dst_phys))
                if not path:
                    continue

                deps: list[str] = []
                if step > 0:
                    deps.append(f"{tenant}-RS-C{chunk_idx}-S{step - 1}")

                policy.append(
                    PolicyEntry(
                        chunk_id=f"{tenant}-RS-C{chunk_idx}-S{step}",
                        src=src_phys,
                        dst=dst_phys,
                        qpid=0,
                        rate=rate,
                        chunk_size_bytes=chunk_size_bytes,
                        path=path,
                        time=start_time,
                        dependency=deps,
                    )
                )

        for chunk_idx in range(participant_count):
            root_idx = (chunk_idx + participant_count - 1) % participant_count
            for step in range(participant_count - 1):
                src_idx = (root_idx + step) % participant_count
                dst_idx = (root_idx + step + 1) % participant_count
                src_phys = physical_ranks[src_idx]
                dst_phys = physical_ranks[dst_idx]
                path = path_table.get((src_phys, dst_phys))
                if not path:
                    continue

                deps = [f"{tenant}-RS-C{chunk_idx}-S{participant_count - 2}"]
                if step > 0:
                    deps = [f"{tenant}-AG-C{chunk_idx}-S{step - 1}"]

                policy.append(
                    PolicyEntry(
                        chunk_id=f"{tenant}-AG-C{chunk_idx}-S{step}",
                        src=src_phys,
                        dst=dst_phys,
                        qpid=0,
                        rate=rate,
                        chunk_size_bytes=chunk_size_bytes,
                        path=path,
                        time=start_time,
                        dependency=deps,
                    )
                )

    return policy


def _simulation_worker_path() -> str:
    return str(Path(__file__).with_name("worker.py"))


def simulate_collective(
    topology: nx.DiGraph,
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int], list[int]],
    single_flow_size_bits: int,
    collective: str,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
) -> tuple[float, float]:
    sim_topology = build_simulator_topology(topology)

    if collective == "allgather":
        policy = allgather_policy(
            tenant_servers,
            path_table,
            int(single_flow_size_bits / 8.0),
            tenant_start_times,
            tenant_rates,
        )
    elif collective == "allreduce":
        policy = allreduce_policy(
            tenant_servers,
            path_table,
            int(single_flow_size_bits / 8.0),
            tenant_start_times,
            tenant_rates,
        )
    else:
        raise ValueError(f"Unsupported collective: {collective}")

    payload = pickle.dumps({"topology": sim_topology, "policy": policy})
    
    # Force use of python3 instead of pypy3, as pypy3 is taking too long/hanging
    python_exec = sys.executable

    process = subprocess.Popen(
        [python_exec, _simulation_worker_path()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
    )

    stdout_data, _ = process.communicate(input=payload)
    if process.returncode != 0:
        raise RuntimeError(f"Simulation subprocess failed with code {process.returncode}")

    result = pickle.loads(stdout_data)
    return result["global_makespan"], result["avg_tenant_makespan"]


simulate = simulate_collective
