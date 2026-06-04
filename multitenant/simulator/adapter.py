from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path
from shutil import which

import networkx as nx
import numpy as np

from CCL_Simulator.simcore import PolicyEntry
from multitenant.collectives import normalize_collective_programs
from multitenant.workloads import build_collective_program_schedule, build_collective_schedule


def _lookup_path(path_table, tenant, src, dst):
    return path_table.get((int(tenant), int(src), int(dst)))


def _path_edges(path_table, tenant, src, dst):
    path = _lookup_path(path_table, tenant, src, dst)
    if not path:
        return []
    return list(zip(path[:-1], path[1:]))


def _resolve_dependency_names(dependency_names, dependency_chunk_lookup):
    return [
        str(dependency_chunk_lookup.get(str(dependency_name), str(dependency_name)))
        for dependency_name in dependency_names
    ]


def _build_quota_segment_specs(
    task_name,
    chunk_size_bytes,
    task_time,
    dependency_names,
    dependency_scope,
    dependency_delay,
    path_bottleneck_bps,
    rate_windows,
):
    if not rate_windows:
        return [
            {
                "chunk_id": str(task_name),
                "chunk_size_bytes": int(chunk_size_bytes),
                "time": float(task_time),
                "dependency": list(dependency_names),
                "dependency_scope": str(dependency_scope),
                "dependency_delay": float(dependency_delay),
                "rate": "Max",
            }
        ]

    remaining_bytes = int(chunk_size_bytes)
    segment_specs = []
    sorted_windows = sorted(
        (
            {
                "start": float(window["start"])
                if "start" in window
                else float(task_time) + float(window.get("offset", 0.0)),
                "duration": max(0.0, float(window.get("duration", 0.0))),
                "scale": float(window.get("scale", 1.0)),
            }
            for window in rate_windows
        ),
        key=lambda window: (window["start"], window["duration"]),
    )

    previous_chunk_id = None
    for segment_idx, window in enumerate(sorted_windows):
        if remaining_bytes <= 0:
            break
        if path_bottleneck_bps <= 0.0 or window["duration"] <= 0.0:
            continue
        scale = max(0.0, float(window["scale"]))
        window_rate_bps = path_bottleneck_bps * scale
        if window_rate_bps <= 0.0:
            continue
        budget_bytes = max(1, int(np.floor((window_rate_bps * window["duration"]) / 8.0)))
        segment_bytes = min(remaining_bytes, budget_bytes)
        segment_start = float(window["start"])
        if previous_chunk_id is None:
            segment_dependency = list(dependency_names)
            segment_dependency_scope = str(dependency_scope)
            segment_dependency_delay = float(dependency_delay)
            if segment_dependency_scope == "global":
                # Program-level dependencies already express "previous op finished
                # plus gap/offset"; the absolute quota window is an earliest-time
                # bound, not another relative delay after the previous op.
                segment_time = max(float(task_time), segment_start)
            else:
                segment_time = max(float(task_time), segment_start)
        else:
            segment_dependency = [str(previous_chunk_id)]
            segment_dependency_scope = "node"
            segment_dependency_delay = 0.0
            segment_time = max(float(task_time), segment_start)

        chunk_id = f"{task_name}-Q{segment_idx}"
        segment_specs.append(
            {
                "chunk_id": chunk_id,
                "chunk_size_bytes": int(segment_bytes),
                "time": segment_time,
                "dependency": segment_dependency,
                "dependency_scope": segment_dependency_scope,
                "dependency_delay": segment_dependency_delay,
                # The segment byte budget enforces the quota over this window.
                # Sending the budgeted segment at Max avoids double-throttling
                # on packet-serial links while preserving the per-window average.
                "rate": "Max",
            }
        )
        previous_chunk_id = chunk_id
        remaining_bytes -= int(segment_bytes)

    if remaining_bytes > 0:
        tail_start = max(
            (window["start"] + window["duration"] for window in sorted_windows),
            default=float(task_time),
        )
        if previous_chunk_id is None:
            tail_dependency = list(dependency_names)
            tail_dependency_scope = str(dependency_scope)
            tail_dependency_delay = float(dependency_delay)
            if tail_dependency_scope == "global":
                # Same semantics as the first quota segment: the window start is
                # absolute, while dependency_delay remains relative to deps.
                tail_time = max(float(task_time), tail_start)
            else:
                tail_time = max(float(task_time), tail_start)
        else:
            tail_dependency = [str(previous_chunk_id)]
            tail_dependency_scope = "node"
            tail_dependency_delay = 0.0
            tail_time = max(float(task_time), tail_start)
        segment_specs.append(
            {
                "chunk_id": f"{task_name}-Qtail",
                "chunk_size_bytes": int(remaining_bytes),
                "time": tail_time,
                "dependency": tail_dependency,
                "dependency_scope": tail_dependency_scope,
                "dependency_delay": tail_dependency_delay,
                "rate": "Max",
            }
        )

    return segment_specs


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
    path_table: dict[tuple[int, int, int], list[int]],
    schedule: dict[int, dict[str, object]],
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    collective_start_times: dict[int, dict[int, float]] | None = None,
    collective_rate_scales: dict[int, dict[int, float]] | None = None,
    collective_rate_schedule: dict[int, dict[int, list[dict[str, float]]]] | None = None,
    task_rate_schedule: dict[int, dict[str, list[dict[str, float]]]] | None = None,
    edge_capacities: dict[tuple[int, int], float] | None = None,
    policy_qpid_mode: str = "tenant",
) -> list[PolicyEntry]:
    tenant_start_times = tenant_start_times or {}
    tenant_rates = tenant_rates or {}
    collective_start_times = collective_start_times or {}
    collective_rate_scales = collective_rate_scales or {}
    collective_rate_schedule = collective_rate_schedule or {}
    task_rate_schedule = task_rate_schedule or {}
    edge_capacities = edge_capacities or {}
    policy: list[PolicyEntry] = []
    if policy_qpid_mode not in {"tenant", "single"}:
        raise ValueError("policy_qpid_mode must be either 'tenant' or 'single'")

    for tenant, mapping in tenant_servers.items():
        tenant_qpid = 0 if policy_qpid_mode == "single" else int(tenant)
        start_time = tenant_start_times.get(tenant, 0.0)
        rate = tenant_rates.get(tenant, "Max")
        tenant_collective_offsets = collective_start_times.get(tenant, {})
        tenant_collective_rate_scales = collective_rate_scales.get(tenant, {})
        tenant_collective_rate_schedule = collective_rate_schedule.get(tenant, {})
        tenant_task_rate_schedule = task_rate_schedule.get(tenant, {})
        tenant_schedule = schedule.get(tenant, {})
        tasks = tenant_schedule.get("tasks", [])
        task_lookup = {int(task["task_id"]): task for task in tasks}
        incoming_task_ids: dict[int, list[int]] = {
            int(task["task_id"]): [int(pred_task_id) for pred_task_id in task.get("preds", [])]
            for task in tasks
        }
        dependency_chunk_lookup: dict[str, str] = {}
        program_releases: dict[int, tuple[list[str], float, float | None]] = {}
        program = tenant_schedule.get("collective_program", [])
        if program:
            for op_idx, op_meta in enumerate(program):
                if op_idx == 0:
                    continue
                previous_op = program[op_idx - 1]
                gap_after = float(previous_op.get("gap_after", 0.0))
                collective_offset = float(tenant_collective_offsets.get(op_idx, 0.0))
                for task_id in op_meta.get("initial_task_ids", []):
                    program_releases[int(task_id)] = (
                        [str(task_lookup[int(previous_task_id)]["name"]) for previous_task_id in previous_op.get("task_ids", [])],
                        gap_after + collective_offset,
                        None,
                    )

        for task_id in tenant_schedule.get("task_order", []):
            task = task_lookup[int(task_id)]
            src_phys = mapping[int(task["src_rank"])]
            dst_phys = mapping[int(task["dst_rank"])]
            path = _lookup_path(path_table, tenant, src_phys, dst_phys)
            if not path:
                continue

            dependency_scope = "node"
            dependency_delay = 0.0
            if int(task_id) in program_releases:
                release_dependency_names, dependency_delay, scheduled_abs = program_releases[int(task_id)]
                deps = _resolve_dependency_names(
                    release_dependency_names,
                    dependency_chunk_lookup,
                )
                dependency_scope = "global"
            else:
                scheduled_abs = None
                deps = _resolve_dependency_names(
                    [str(task_lookup[int(pred_task_id)]["name"]) for pred_task_id in incoming_task_ids.get(int(task_id), [])],
                    dependency_chunk_lookup,
                )
            chunk_size_bytes = int(round(float(task["V"]) * 1e9 / 8.0))
            task_time = start_time
            if "op_idx" in task:
                op_idx = int(task["op_idx"])
                collective_offset = float(tenant_collective_offsets.get(op_idx, 0.0))
                if op_idx == 0:
                    task_time = start_time + collective_offset
            if scheduled_abs is not None:
                task_time = max(task_time, float(scheduled_abs))
            task_rate = rate
            if "op_idx" in task and int(task["op_idx"]) in tenant_collective_rate_scales:
                rate_scale = float(tenant_collective_rate_scales[int(task["op_idx"])])
                path_edges = _path_edges(path_table, tenant, src_phys, dst_phys)
                path_bottleneck_bps = min(
                    (edge_capacities[edge] for edge in path_edges if edge in edge_capacities),
                    default=0.0,
                )
                if path_bottleneck_bps > 0.0 and rate_scale < 1.0 - 1e-12:
                    task_rate = rate_scale * path_bottleneck_bps
            rate_windows = None
            path_edges = _path_edges(path_table, tenant, src_phys, dst_phys)
            path_bottleneck_bps = min(
                (edge_capacities[edge] for edge in path_edges if edge in edge_capacities),
                default=0.0,
            )
            if str(task["name"]) in tenant_task_rate_schedule:
                rate_windows = tenant_task_rate_schedule.get(str(task["name"]))
            elif "op_idx" in task:
                rate_windows = tenant_collective_rate_schedule.get(int(task["op_idx"]))

            if rate_windows:
                segment_specs = _build_quota_segment_specs(
                    str(task["name"]),
                    chunk_size_bytes,
                    task_time,
                    deps,
                    dependency_scope,
                    dependency_delay,
                    path_bottleneck_bps,
                    rate_windows,
                )
                final_chunk_id = str(task["name"])
                for segment_spec in segment_specs:
                    policy.append(
                        PolicyEntry(
                            chunk_id=str(segment_spec["chunk_id"]),
                            src=src_phys,
                            dst=dst_phys,
                            qpid=tenant_qpid,
                            rate=segment_spec["rate"],
                            chunk_size_bytes=int(segment_spec["chunk_size_bytes"]),
                            path=path,
                            time=float(segment_spec["time"]),
                            dependency=list(segment_spec["dependency"]),
                            dependency_scope=str(segment_spec["dependency_scope"]),
                            dependency_delay=float(segment_spec["dependency_delay"]),
                        )
                    )
                    final_chunk_id = str(segment_spec["chunk_id"])
                dependency_chunk_lookup[str(task["name"])] = final_chunk_id
            else:
                policy.append(
                    PolicyEntry(
                        chunk_id=str(task["name"]),
                        src=src_phys,
                        dst=dst_phys,
                        qpid=tenant_qpid,
                        rate=task_rate,
                        chunk_size_bytes=chunk_size_bytes,
                        path=path,
                        time=task_time,
                        dependency=deps,
                        dependency_scope=dependency_scope,
                        dependency_delay=dependency_delay,
                    )
                )
                dependency_chunk_lookup[str(task["name"])] = str(task["name"])
    return policy


def allgather_policy(
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int, int], list[int]],
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
    path_table: dict[tuple[int, int, int], list[int]],
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
    path_table: dict[tuple[int, int, int], list[int]],
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
    path_table: dict[tuple[int, int, int], list[int]],
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


def _run_simulation_worker_result(sim_topology: nx.DiGraph, policy: list[PolicyEntry]) -> dict[str, object]:
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
    return result


def _run_simulation_worker(sim_topology: nx.DiGraph, policy: list[PolicyEntry]) -> tuple[float, float]:
    result = _run_simulation_worker_result(sim_topology, policy)
    return result["global_makespan"], result["avg_tenant_makespan"]


def simulate_collective(
    topology: nx.DiGraph,
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int, int], list[int]],
    single_flow_size_bits: int | None = None,
    collective: str | None = None,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
    tenant_collective_programs: dict[int, list[dict[str, object]]] | None = None,
    collective_start_times: dict[int, dict[int, float]] | None = None,
    collective_rate_scales: dict[int, dict[int, float]] | None = None,
    collective_rate_schedule: dict[int, dict[int, list[dict[str, float]]]] | None = None,
    task_rate_schedule: dict[int, dict[str, list[dict[str, float]]]] | None = None,
    policy_qpid_mode: str = "tenant",
) -> tuple[float, float]:
    sim_topology = build_simulator_topology(topology)
    edge_capacities = {
        (int(src), int(dst)): float(attrs["capacity"])
        for src, dst, attrs in topology.edges(data=True)
    }

    tenant_collective_programs = normalize_collective_programs(
        tenant_servers,
        collective=collective,
        single_flow_size=single_flow_size_bits,
        tenant_collective_specs=tenant_collective_specs,
        tenant_collective_programs=tenant_collective_programs,
    )
    if tenant_collective_programs is not None:
        schedule = build_collective_program_schedule(
            tenant_servers,
            tenant_collective_programs,
        )
        policy = _policy_from_schedule(
            tenant_servers,
            path_table,
            schedule,
            tenant_start_times=tenant_start_times,
            tenant_rates=tenant_rates,
            collective_start_times=collective_start_times,
            collective_rate_scales=collective_rate_scales,
            collective_rate_schedule=collective_rate_schedule,
            task_rate_schedule=task_rate_schedule,
            edge_capacities=edge_capacities,
            policy_qpid_mode=policy_qpid_mode,
        )
    else:
        raise ValueError(f"Unsupported collective: {collective}")

    return _run_simulation_worker(sim_topology, policy)


def simulate_collective_details(
    topology: nx.DiGraph,
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int, int], list[int]],
    single_flow_size_bits: int | None = None,
    collective: str | None = None,
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
    tenant_collective_programs: dict[int, list[dict[str, object]]] | None = None,
    collective_start_times: dict[int, dict[int, float]] | None = None,
    collective_rate_scales: dict[int, dict[int, float]] | None = None,
    collective_rate_schedule: dict[int, dict[int, list[dict[str, float]]]] | None = None,
    task_rate_schedule: dict[int, dict[str, list[dict[str, float]]]] | None = None,
    policy_qpid_mode: str = "tenant",
) -> dict[str, object]:
    sim_topology = build_simulator_topology(topology)
    edge_capacities = {
        (int(src), int(dst)): float(attrs["capacity"])
        for src, dst, attrs in topology.edges(data=True)
    }

    tenant_collective_programs = normalize_collective_programs(
        tenant_servers,
        collective=collective,
        single_flow_size=single_flow_size_bits,
        tenant_collective_specs=tenant_collective_specs,
        tenant_collective_programs=tenant_collective_programs,
    )
    if tenant_collective_programs is not None:
        schedule = build_collective_program_schedule(
            tenant_servers,
            tenant_collective_programs,
        )
        policy = _policy_from_schedule(
            tenant_servers,
            path_table,
            schedule,
            tenant_start_times=tenant_start_times,
            tenant_rates=tenant_rates,
            collective_start_times=collective_start_times,
            collective_rate_scales=collective_rate_scales,
            collective_rate_schedule=collective_rate_schedule,
            task_rate_schedule=task_rate_schedule,
            edge_capacities=edge_capacities,
            policy_qpid_mode=policy_qpid_mode,
        )
    else:
        raise ValueError(f"Unsupported collective: {collective}")

    return _run_simulation_worker_result(sim_topology, policy)


def collective_program_policy(
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int, int], list[int]],
    tenant_collective_programs: dict[int, list[dict[str, object]]],
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    collective_start_times: dict[int, dict[int, float]] | None = None,
    collective_rate_scales: dict[int, dict[int, float]] | None = None,
    collective_rate_schedule: dict[int, dict[int, list[dict[str, float]]]] | None = None,
    task_rate_schedule: dict[int, dict[str, list[dict[str, float]]]] | None = None,
    edge_capacities: dict[tuple[int, int], float] | None = None,
    policy_qpid_mode: str = "tenant",
) -> list[PolicyEntry]:
    schedule = build_collective_program_schedule(tenant_servers, tenant_collective_programs)
    return _policy_from_schedule(
        tenant_servers,
        path_table,
        schedule,
        tenant_start_times=tenant_start_times,
        tenant_rates=tenant_rates,
        collective_start_times=collective_start_times,
        collective_rate_scales=collective_rate_scales,
        collective_rate_schedule=collective_rate_schedule,
        task_rate_schedule=task_rate_schedule,
        edge_capacities=edge_capacities,
        policy_qpid_mode=policy_qpid_mode,
    )


def simulate_collective_program(
    topology: nx.DiGraph,
    tenant_servers: dict[int, dict[int, int]],
    path_table: dict[tuple[int, int, int], list[int]],
    tenant_collective_programs: dict[int, list[dict[str, object]]],
    tenant_start_times: dict[int, float] | None = None,
    tenant_rates: dict[int, float] | None = None,
    collective_start_times: dict[int, dict[int, float]] | None = None,
    collective_rate_scales: dict[int, dict[int, float]] | None = None,
    collective_rate_schedule: dict[int, dict[int, list[dict[str, float]]]] | None = None,
    task_rate_schedule: dict[int, dict[str, list[dict[str, float]]]] | None = None,
    policy_qpid_mode: str = "tenant",
) -> tuple[float, float]:
    return simulate_collective(
        topology,
        tenant_servers,
        path_table,
        tenant_collective_programs=tenant_collective_programs,
        tenant_start_times=tenant_start_times,
        tenant_rates=tenant_rates,
        collective_start_times=collective_start_times,
        collective_rate_scales=collective_rate_scales,
        collective_rate_schedule=collective_rate_schedule,
        task_rate_schedule=task_rate_schedule,
        policy_qpid_mode=policy_qpid_mode,
    )


simulate = simulate_collective
