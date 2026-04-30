from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path
from shutil import which

import networkx as nx

from CCL_Simulator.simcore import PolicyEntry
from multitenant.workloads import build_collective_schedule


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


def _policy_from_schedule(
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int], list[int]],
    schedule: dict[int, dict[str, object]],
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
) -> list[PolicyEntry]:
    tenant_start_times = tenant_start_times or {}
    tenant_rates = tenant_rates or {}
    policy: list[PolicyEntry] = []

    for tenant, mapping in tenant_servers.items():
        start_time = tenant_start_times.get(tenant, 0.0)
        rate = tenant_rates.get(tenant, "Max")
        tenant_schedule = schedule.get(tenant, {})
        tasks = tenant_schedule.get("tasks", [])
        task_lookup = {int(task["task_id"]): task for task in tasks}

        for task_id in tenant_schedule.get("task_order", []):
            task = task_lookup[int(task_id)]
            src_phys = mapping[int(task["src_rank"])]
            dst_phys = mapping[int(task["dst_rank"])]
            path = path_table.get((src_phys, dst_phys))
            if not path:
                continue

            deps = [str(task_lookup[int(pred_task_id)]["name"]) for pred_task_id in task["preds"]]
            chunk_size_bytes = int(round(float(task["V"]) * 1e9 / 8.0))
            policy.append(
                PolicyEntry(
                    chunk_id=str(task["name"]),
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


def allgather_policy(
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int], list[int]],
    single_flow_size_bytes: int,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> list[PolicyEntry]:
    schedule = build_collective_schedule(
        tenant_servers,
        int(single_flow_size_bytes * 8),
        "allgather",
        tenant_collective_specs=tenant_collective_specs,
    )
    return _policy_from_schedule(tenant_servers, path_table, schedule, tenant_start_times, tenant_rates)


def reducescatter_policy(
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int], list[int]],
    chunk_size_bytes: int,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> list[PolicyEntry]:
    schedule = build_collective_schedule(
        tenant_servers,
        int(chunk_size_bytes * 8),
        "reducescatter",
        tenant_collective_specs=tenant_collective_specs,
    )
    return _policy_from_schedule(tenant_servers, path_table, schedule, tenant_start_times, tenant_rates)


def alltoall_policy(
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int], list[int]],
    chunk_size_bytes: int,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> list[PolicyEntry]:
    schedule = build_collective_schedule(
        tenant_servers,
        int(chunk_size_bytes * 8),
        "alltoall",
        tenant_collective_specs=tenant_collective_specs,
    )
    return _policy_from_schedule(tenant_servers, path_table, schedule, tenant_start_times, tenant_rates)


def allreduce_policy(
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int], list[int]],
    chunk_size_bytes: int,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> list[PolicyEntry]:
    schedule = build_collective_schedule(
        tenant_servers,
        int(chunk_size_bytes * 8),
        "allreduce",
        tenant_collective_specs=tenant_collective_specs,
    )
    return _policy_from_schedule(tenant_servers, path_table, schedule, tenant_start_times, tenant_rates)


def _simulation_worker_path() -> str:
    return str(Path(__file__).with_name("worker.py"))


def simulate_collective(
    topology: nx.DiGraph,
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int], list[int]],
    single_flow_size_bits: int | None = None,
    collective: str | None = None,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> tuple[float, float]:
    sim_topology = build_simulator_topology(topology)

    if tenant_collective_specs is not None:
        schedule = build_collective_schedule(
            tenant_servers,
            single_flow_size_bits,
            collective,
            tenant_collective_specs=tenant_collective_specs,
        )
        policy = _policy_from_schedule(
            tenant_servers,
            path_table,
            schedule,
            tenant_start_times,
            tenant_rates,
        )
    elif collective == "allgather":
        policy = allgather_policy(
            tenant_servers,
            path_table,
            int(single_flow_size_bits / 8.0),
            tenant_start_times,
            tenant_rates,
        )
    elif collective == "reducescatter":
        policy = reducescatter_policy(
            tenant_servers,
            path_table,
            int(single_flow_size_bits / 8.0),
            tenant_start_times,
            tenant_rates,
        )
    elif collective == "alltoall":
        policy = alltoall_policy(
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
