from __future__ import annotations

from collections import defaultdict

from multitenant.schedule_compiler import compile_epoch_schedule, task_levels
from multitenant.workloads import build_collective_program_schedule


def build_task_release_gates(tenant_schedule):
    """Release initial tasks of op i+1 after op i finishes plus gap_after."""
    program = list(tenant_schedule.get("collective_program", []))
    release_gates: dict[int, tuple[tuple[int, ...], float]] = {}
    for prev_op, next_op in zip(program, program[1:]):
        previous_task_ids = tuple(int(task_id) for task_id in prev_op.get("task_ids", []))
        gap_after = float(prev_op.get("gap_after", 0.0))
        if not previous_task_ids and gap_after <= 0.0:
            continue
        for task_id in next_op.get("initial_task_ids", []):
            release_gates[int(task_id)] = (previous_task_ids, gap_after)
    return release_gates


def path_edges_for_pair(path_edges, path_ordered_edges, tenant, src_server, dst_server):
    """Return cached directed ECMP path edges for one tenant-aware server pair."""
    key = (int(tenant), int(src_server), int(dst_server))
    edges = path_edges.get(key)
    if edges is None:
        edges = tuple(
            tuple(int(node) for node in edge)
            for edge in path_ordered_edges[key]
        )
        path_edges[key] = edges
    return edges


def build_task_dag_surrogate(schedule, tenants):
    """Compile collective schedules into the task-level DAG used by estimators."""
    task_surrogate = {}
    task_global_max_level = -1

    for tenant in tenants:
        tenant_schedule = schedule.get(tenant, {})
        tenant_tasks = list(tenant_schedule.get("tasks", []))
        unified_edges = list(tenant_schedule.get("edges", []))
        if not tenant_tasks:
            task_surrogate[tenant] = {
                "preds": {},
                "execution_preds": {},
                "collective_preds": {},
                "task_order": [],
                "task_levels": {},
                "task_info": {},
                "level_tasks": {},
                "sender_order": {},
                "release_gates": {},
            }
            continue

        base_levels = task_levels(tenant_tasks, unified_edges)
        collective_preds_by_task = {int(task["task_id"]): [] for task in tenant_tasks}
        for edge in tenant_schedule.get("collective_edges", []):
            collective_preds_by_task[int(edge["dst_task_id"])].append(int(edge["src_task_id"]))

        execution_preds_by_task = {int(task["task_id"]): [] for task in tenant_tasks}
        for edge in unified_edges:
            execution_preds_by_task[int(edge["dst_task_id"])].append(int(edge["src_task_id"]))

        topo_order = sorted(
            [int(task["task_id"]) for task in tenant_tasks],
            key=lambda task_id: (int(base_levels.get(int(task_id), 0)), int(task_id)),
        )
        topo_pos = {int(task_id): idx for idx, task_id in enumerate(topo_order)}
        receiver_order = defaultdict(list)
        for task in tenant_tasks:
            receiver_order[int(task["dst_rank"])].append(int(task["task_id"]))

        receiver_edges = []
        for dst_rank, task_ids in receiver_order.items():
            task_ids.sort(key=lambda task_id: topo_pos[int(task_id)])
            for earlier_task_id, later_task_id in zip(task_ids, task_ids[1:]):
                receiver_edges.append(
                    {
                        "src_task_id": int(earlier_task_id),
                        "dst_task_id": int(later_task_id),
                        "type": "receiver_order",
                        "receiver": int(dst_rank),
                    }
                )

        augmented_edges = unified_edges + receiver_edges
        levels = task_levels(tenant_tasks, augmented_edges)
        preds_by_task = {int(task["task_id"]): [] for task in tenant_tasks}
        for edge in augmented_edges:
            preds_by_task[int(edge["dst_task_id"])].append(int(edge["src_task_id"]))

        ordered_task_ids = sorted(
            preds_by_task.keys(),
            key=lambda task_id: (int(levels.get(int(task_id), 0)), int(task_id)),
        )
        task_info = {}
        level_tasks = defaultdict(list)
        for task in tenant_tasks:
            task_id = int(task["task_id"])
            task_tuple = (
                task_id,
                int(task["src_rank"]),
                int(task["dst_rank"]),
                float(task["V"]),
            )
            task_info[task_id] = task_tuple
            level_tasks[int(levels[task_id])].append(task_tuple)

        task_surrogate[tenant] = {
            "preds": {int(task_id): list(preds) for task_id, preds in preds_by_task.items()},
            "execution_preds": {
                int(task_id): list(preds)
                for task_id, preds in execution_preds_by_task.items()
            },
            "collective_preds": {
                int(task_id): list(preds)
                for task_id, preds in collective_preds_by_task.items()
            },
            "task_order": [int(task_id) for task_id in ordered_task_ids],
            "task_levels": {int(task_id): int(level) for task_id, level in levels.items()},
            "task_info": task_info,
            "level_tasks": {int(level): list(flows) for level, flows in level_tasks.items()},
            "sender_order": {
                int(sender): [int(task_id) for task_id in task_ids]
                for sender, task_ids in tenant_schedule.get("sender_order", {}).items()
            },
            "release_gates": build_task_release_gates(tenant_schedule),
        }
        task_global_max_level = max(
            task_global_max_level,
            max((int(level) for level in level_tasks.keys()), default=-1),
        )

    return task_surrogate, int(task_global_max_level)


def build_collective_dag_data(
    *,
    datacenter,
    initial_tenant_mapping,
    tenant_collective_programs,
    tenants,
    rank_orders,
    server_sets,
    path_ordered_edges,
    program_mode,
    tenant_pressure=None,
    tenant_peak_load=None,
):
    """Compile collective programs into solver-ready DAG and resource data."""
    ranks = {
        tenant: list(rank_orders[tenant])
        for tenant in tenants
    }
    servers = {
        tenant: list(server_sets[tenant])
        for tenant in tenants
    }
    schedule = build_collective_program_schedule(
        initial_tenant_mapping,
        tenant_collective_programs,
        scale=1.0,
    )
    tasks = {
        tenant: list(schedule.get(tenant, {}).get("tasks", []))
        for tenant in tenants
    }
    edge_capacity = {
        (int(src), int(dst)): float(attrs["capacity"])
        for src, dst, attrs in datacenter.topology.edges(data=True)
    }
    server_send_capacity = {}
    server_recv_capacity = {}
    for server in datacenter.get_all_servers():
        leaf = int(datacenter.get_server_leaf(server))
        server_send_capacity[int(server)] = float(edge_capacity[(int(server), leaf)])
        server_recv_capacity[int(server)] = float(edge_capacity[(leaf, int(server))])

    path_edges = {}
    compiled_schedule = compile_epoch_schedule(
        schedule,
        tenants,
        rank_orders,
        program_mode=program_mode,
    )
    task_surrogate, task_global_max_level = build_task_dag_surrogate(schedule, tenants)

    if tenant_pressure is not None:
        tenant_pressure.clear()
    if tenant_peak_load is not None:
        tenant_peak_load.clear()
    for tenant in tenants:
        aggregate_pressure = 0.0
        peak_load = 0.0
        for epoch_flows in compiled_schedule["per_tenant"][tenant]["epoch_flows"].values():
            epoch_edge_loads: dict[tuple[int, int], float] = defaultdict(float)
            epoch_sender_loads: dict[int, float] = defaultdict(float)
            epoch_receiver_loads: dict[int, float] = defaultdict(float)
            for src_rank, dst_rank, volume in epoch_flows:
                src_server = int(initial_tenant_mapping[tenant][src_rank])
                dst_server = int(initial_tenant_mapping[tenant][dst_rank])
                epoch_sender_loads[src_server] += float(volume) / server_send_capacity[src_server]
                epoch_receiver_loads[dst_server] += float(volume) / server_recv_capacity[dst_server]
                for edge in path_edges_for_pair(
                    path_edges,
                    path_ordered_edges,
                    tenant,
                    src_server,
                    dst_server,
                ):
                    epoch_edge_loads[edge] += float(volume) / edge_capacity[edge]
            aggregate_pressure += (
                sum(epoch_edge_loads.values())
                + sum(epoch_sender_loads.values())
                + sum(epoch_receiver_loads.values())
            )
            peak_load = max(
                peak_load,
                max(epoch_edge_loads.values(), default=0.0),
                max(epoch_sender_loads.values(), default=0.0),
                max(epoch_receiver_loads.values(), default=0.0),
            )
        if tenant_pressure is not None:
            tenant_pressure[tenant] = float(aggregate_pressure)
        if tenant_peak_load is not None:
            tenant_peak_load[tenant] = float(peak_load)

    return {
        "M": list(tenants),
        "R": ranks,
        "S": servers,
        "tasks": tasks,
        "schedule": schedule,
        "compiled_schedule": compiled_schedule,
        "task_surrogate": task_surrogate,
        "task_global_max_level": int(task_global_max_level),
        "edge_capacity": edge_capacity,
        "server_send_capacity": server_send_capacity,
        "server_recv_capacity": server_recv_capacity,
        "path_edges": path_edges,
    }
