from __future__ import annotations

import random
import heapq
from typing import Sequence


def _normalize_tenant_collective_specs(
    tenant_mapping: dict[int, dict[int, int]],
    single_flow_size_bits: int | None = None,
    collective: str | None = None,
    *,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> dict[int, dict[str, object]]:
    tenant_specs: dict[int, dict[str, object]] = {}

    if tenant_collective_specs is not None:
        for tenant in tenant_mapping:
            if tenant not in tenant_collective_specs:
                raise ValueError(f"Missing collective spec for tenant {tenant}")
            spec = dict(tenant_collective_specs[tenant])
            tenant_collective = spec.get("collective")
            tenant_flow_size_bits = spec.get("single_flow_size_bits")
            if tenant_collective is None:
                raise ValueError(f"Missing collective in tenant spec for tenant {tenant}")
            if tenant_flow_size_bits is None:
                raise ValueError(f"Missing single_flow_size_bits in tenant spec for tenant {tenant}")
            tenant_specs[tenant] = {
                "collective": str(tenant_collective),
                "single_flow_size_bits": int(tenant_flow_size_bits),
            }
        return tenant_specs

    if collective is None or single_flow_size_bits is None:
        raise ValueError(
            "Either tenant_collective_specs or both collective and single_flow_size_bits must be provided"
        )

    for tenant in tenant_mapping:
        tenant_specs[tenant] = {
            "collective": str(collective),
            "single_flow_size_bits": int(single_flow_size_bits),
        }
    return tenant_specs


def _collective_size_factor(collective: str) -> int:
    return 2 if collective == "allreduce" else 1


def build_random_tenant_mapping(
    all_servers: Sequence[int],
    num_tenants: int,
    *,
    rng: random.Random | None = None,
    servers_per_tenant: int | None = None,
) -> dict[int, dict[int, int]]:
    if num_tenants <= 0:
        raise ValueError("num_tenants must be positive")

    rng = rng or random.Random()
    candidates = list(all_servers)
    if not candidates:
        raise ValueError("all_servers cannot be empty")

    if servers_per_tenant is None:
        servers_per_tenant = len(candidates) // num_tenants

    required_servers = num_tenants * servers_per_tenant
    if required_servers > len(candidates):
        raise ValueError("Not enough servers to place all tenants")

    selected = candidates[:required_servers]
    rng.shuffle(selected)

    tenant_mapping: dict[int, dict[int, int]] = {}
    for tenant in range(num_tenants):
        start = tenant * servers_per_tenant
        end = (tenant + 1) * servers_per_tenant
        assigned_servers = selected[start:end]
        tenant_mapping[tenant] = {
            rank: physical_server for rank, physical_server in enumerate(assigned_servers)
        }
    return tenant_mapping


def build_ring_flows(
    tenant_mapping: dict[int, dict[int, int]],
    single_flow_size_bits: int,
    collective: str,
    *,
    scale: float = 1e9,
) -> dict[int, list[tuple[int, int, float]]]:
    size_factor = _collective_size_factor(collective)
    tenant_flows: dict[int, list[tuple[int, int, float]]] = {}

    for tenant, mapping in tenant_mapping.items():
        tenant_flows[tenant] = []
        ranks = sorted(mapping.keys())
        for idx, src in enumerate(ranks):
            dst = ranks[(idx + 1) % len(ranks)]
            volume_bits = (len(ranks) - 1) * single_flow_size_bits * size_factor
            tenant_flows[tenant].append((src, dst, volume_bits / scale))

    return tenant_flows


def _build_ring_stage_sequence(
    ranks: list[int],
    stage_count: int,
    single_flow_size_bits: int,
    *,
    scale: float,
) -> list[list[tuple[int, int, float]]]:
    stage_flows: list[list[tuple[int, int, float]]] = []
    for step in range(stage_count):
        stage = []
        for chunk_idx in range(len(ranks)):
            src = ranks[(chunk_idx + step) % len(ranks)]
            dst = ranks[(chunk_idx + step + 1) % len(ranks)]
            stage.append((src, dst, single_flow_size_bits / scale))
        stage_flows.append(stage)
    return stage_flows


def _build_allreduce_allgather_sequence(
    ranks: list[int],
    single_flow_size_bits: int,
    *,
    scale: float,
) -> list[list[tuple[int, int, float]]]:
    stage_flows: list[list[tuple[int, int, float]]] = []
    participant_count = len(ranks)

    for step in range(participant_count - 1):
        stage = []
        for chunk_idx in range(participant_count):
            root_idx = (chunk_idx + participant_count - 1) % participant_count
            src = ranks[(root_idx + step) % participant_count]
            dst = ranks[(root_idx + step + 1) % participant_count]
            stage.append((src, dst, single_flow_size_bits / scale))
        stage_flows.append(stage)

    return stage_flows


def _build_alltoall_stage_sequence(
    ranks: list[int],
    single_flow_size_bits: int,
    *,
    scale: float,
) -> list[list[tuple[int, int, float]]]:
    stage_flows: list[list[tuple[int, int, float]]] = []
    participant_count = len(ranks)

    for offset in range(1, participant_count):
        stage = []
        for src_idx, src in enumerate(ranks):
            dst = ranks[(src_idx + offset) % participant_count]
            stage.append((src, dst, single_flow_size_bits / scale))
        stage_flows.append(stage)

    return stage_flows


def build_collective_stage_flows(
    tenant_mapping: dict[int, dict[int, int]],
    single_flow_size_bits: int | None = None,
    collective: str | None = None,
    *,
    scale: float = 1e9,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> dict[int, list[list[tuple[int, int, float]]]]:
    """Build stage-by-stage communication rounds for supported collectives.

    Each stage contains the logical source/destination pairs active in that round,
    with the transmitted volume expressed in Gbit to match the existing flow model.
    """

    tenant_specs = _normalize_tenant_collective_specs(
        tenant_mapping,
        single_flow_size_bits,
        collective,
        tenant_collective_specs=tenant_collective_specs,
    )

    tenant_stage_flows: dict[int, list[list[tuple[int, int, float]]]] = {}
    for tenant, mapping in tenant_mapping.items():
        ranks = sorted(mapping.keys())
        participant_count = len(ranks)
        tenant_collective = str(tenant_specs[tenant]["collective"])
        tenant_flow_size_bits = int(tenant_specs[tenant]["single_flow_size_bits"])

        if participant_count <= 1:
            tenant_stage_flows[tenant] = []
            continue

        if tenant_collective == "allgather":
            tenant_stage_flows[tenant] = _build_ring_stage_sequence(
                ranks,
                participant_count - 1,
                tenant_flow_size_bits,
                scale=scale,
            )
        elif tenant_collective == "reducescatter":
            tenant_stage_flows[tenant] = _build_ring_stage_sequence(
                ranks,
                participant_count - 1,
                tenant_flow_size_bits,
                scale=scale,
            )
        elif tenant_collective == "allreduce":
            reduce_scatter = _build_ring_stage_sequence(
                ranks,
                participant_count - 1,
                tenant_flow_size_bits,
                scale=scale,
            )
            allgather = _build_allreduce_allgather_sequence(
                ranks,
                tenant_flow_size_bits,
                scale=scale,
            )
            tenant_stage_flows[tenant] = reduce_scatter + allgather
        elif tenant_collective == "alltoall":
            tenant_stage_flows[tenant] = _build_alltoall_stage_sequence(
                ranks,
                tenant_flow_size_bits,
                scale=scale,
            )
        else:
            tenant_stage_flows[tenant] = [build_ring_flows(
                {tenant: mapping},
                tenant_flow_size_bits,
                tenant_collective,
                scale=scale,
            )[tenant]]

    return tenant_stage_flows


def build_collective_tasks(
    tenant_mapping: dict[int, dict[int, int]],
    single_flow_size_bits: int | None = None,
    collective: str | None = None,
    *,
    scale: float = 1e9,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> dict[int, list[dict[str, object]]]:
    """Build task-level chunk/step DAGs for supported collectives."""

    tenant_specs = _normalize_tenant_collective_specs(
        tenant_mapping,
        single_flow_size_bits,
        collective,
        tenant_collective_specs=tenant_collective_specs,
    )

    tenant_tasks: dict[int, list[dict[str, object]]] = {}

    def task_name(tenant_id: int, phase: str, chunk_idx: int, step: int) -> str:
        return f"{tenant_id}-{phase}-C{chunk_idx}-S{step}"

    def alltoall_task_name(tenant_id: int, src_rank: int, dst_rank: int) -> str:
        return f"{tenant_id}-A2A-{src_rank}TO{dst_rank}"

    for tenant, mapping in tenant_mapping.items():
        ranks = sorted(mapping.keys())
        participant_count = len(ranks)
        tenant_collective = str(tenant_specs[tenant]["collective"])
        tenant_flow_size_bits = int(tenant_specs[tenant]["single_flow_size_bits"])
        task_volume = tenant_flow_size_bits / scale
        tasks: list[dict[str, object]] = []
        next_task_id = 0

        if participant_count <= 1:
            tenant_tasks[tenant] = tasks
            continue

        if tenant_collective == "allgather":
            for chunk_idx in range(participant_count):
                previous_task_id = None
                for step in range(participant_count - 1):
                    src = ranks[(chunk_idx + step) % participant_count]
                    dst = ranks[(chunk_idx + step + 1) % participant_count]
                    task_id = next_task_id
                    next_task_id += 1
                    tasks.append(
                        {
                            "task_id": task_id,
                            "name": task_name(tenant, "AG", chunk_idx, step),
                            "phase": "AG",
                            "chunk": chunk_idx,
                            "step": step,
                            "u": src,
                            "v": dst,
                            "V": task_volume,
                            "preds": [] if previous_task_id is None else [previous_task_id],
                            "src_rank": src,
                            "dst_rank": dst,
                        }
                    )
                    previous_task_id = task_id

        elif tenant_collective == "reducescatter":
            for chunk_idx in range(participant_count):
                previous_task_id = None
                for step in range(participant_count - 1):
                    src = ranks[(chunk_idx + step) % participant_count]
                    dst = ranks[(chunk_idx + step + 1) % participant_count]
                    task_id = next_task_id
                    next_task_id += 1
                    tasks.append(
                        {
                            "task_id": task_id,
                            "name": task_name(tenant, "RS", chunk_idx, step),
                            "phase": "RS",
                            "chunk": chunk_idx,
                            "step": step,
                            "u": src,
                            "v": dst,
                            "V": task_volume,
                            "preds": [] if previous_task_id is None else [previous_task_id],
                            "src_rank": src,
                            "dst_rank": dst,
                        }
                    )
                    previous_task_id = task_id

        elif tenant_collective == "allreduce":
            last_reduce_scatter_task: dict[int, int] = {}

            for chunk_idx in range(participant_count):
                previous_task_id = None
                for step in range(participant_count - 1):
                    src = ranks[(chunk_idx + step) % participant_count]
                    dst = ranks[(chunk_idx + step + 1) % participant_count]
                    task_id = next_task_id
                    next_task_id += 1
                    tasks.append(
                        {
                            "task_id": task_id,
                            "name": task_name(tenant, "RS", chunk_idx, step),
                            "phase": "RS",
                            "chunk": chunk_idx,
                            "step": step,
                            "u": src,
                            "v": dst,
                            "V": task_volume,
                            "preds": [] if previous_task_id is None else [previous_task_id],
                            "src_rank": src,
                            "dst_rank": dst,
                        }
                    )
                    previous_task_id = task_id
                last_reduce_scatter_task[chunk_idx] = previous_task_id

            for chunk_idx in range(participant_count):
                previous_task_id = last_reduce_scatter_task[chunk_idx]
                root_idx = (chunk_idx + participant_count - 1) % participant_count
                for step in range(participant_count - 1):
                    src = ranks[(root_idx + step) % participant_count]
                    dst = ranks[(root_idx + step + 1) % participant_count]
                    task_id = next_task_id
                    next_task_id += 1
                    tasks.append(
                        {
                            "task_id": task_id,
                            "name": task_name(tenant, "AG", chunk_idx, step),
                            "phase": "AG",
                            "chunk": chunk_idx,
                            "step": step,
                            "u": src,
                            "v": dst,
                            "V": task_volume,
                            "preds": [] if previous_task_id is None else [previous_task_id],
                            "src_rank": src,
                            "dst_rank": dst,
                        }
                    )
                    previous_task_id = task_id

        elif tenant_collective == "alltoall":
            for round_idx in range(1, participant_count):
                for src_idx, src in enumerate(ranks):
                    dst = ranks[(src_idx + round_idx) % participant_count]
                    task_id = next_task_id
                    next_task_id += 1
                    tasks.append(
                        {
                            "task_id": task_id,
                            "name": alltoall_task_name(tenant, src, dst),
                            "phase": "A2A",
                            "chunk": src_idx,
                            "step": round_idx - 1,
                            "u": src,
                            "v": dst,
                            "V": task_volume,
                            "preds": [],
                            "src_rank": src,
                            "dst_rank": dst,
                            "round": round_idx - 1,
                        }
                    )

        else:
            for task_idx, (src, dst, volume) in enumerate(
                build_ring_flows(
                    {tenant: mapping},
                    tenant_flow_size_bits,
                    tenant_collective,
                    scale=scale,
                )[tenant]
            ):
                tasks.append(
                    {
                        "task_id": task_idx,
                        "name": task_name(tenant, "GENERIC", task_idx, 0),
                        "phase": "GENERIC",
                        "chunk": task_idx,
                        "step": 0,
                        "u": src,
                        "v": dst,
                        "V": float(volume),
                        "preds": [],
                        "src_rank": src,
                        "dst_rank": dst,
                    }
                )

        tenant_tasks[tenant] = tasks

    return tenant_tasks


def build_collective_dag(
    tenant_mapping: dict[int, dict[int, int]],
    single_flow_size_bits: int | None = None,
    collective: str | None = None,
    *,
    scale: float = 1e9,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> dict[int, dict[str, object]]:
    """Build the fixed collective dependency DAG used by the simulator.

    The node naming matches simulator ``PolicyEntry.chunk_id`` values.
    Only algorithmic dependency edges are included here; sender arbitration is
    represented separately through ``sender_order`` in the schedule structure.
    """

    tasks_by_tenant = build_collective_tasks(
        tenant_mapping,
        single_flow_size_bits,
        collective,
        scale=scale,
        tenant_collective_specs=tenant_collective_specs,
    )

    dag: dict[int, dict[str, object]] = {}
    for tenant, tasks in tasks_by_tenant.items():
        task_lookup = {int(task["task_id"]): task for task in tasks}
        edges: list[dict[str, object]] = []
        for task in tasks:
            dst_task_id = int(task["task_id"])
            dst_name = str(task["name"])
            for pred_task_id in task["preds"]:
                pred_task = task_lookup[int(pred_task_id)]
                edges.append(
                    {
                        "src_task_id": int(pred_task_id),
                        "dst_task_id": dst_task_id,
                        "src": str(pred_task["name"]),
                        "dst": dst_name,
                        "type": "collective",
                    }
                )

        dag[tenant] = {
            "nodes": tasks,
            "edges": edges,
        }

    return dag


def _stable_topological_task_order(
    tasks: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> list[int]:
    """Return a deterministic topological order over task ids.

    Ties are broken by task id so that sender-local orders derived from this
    sequence stay stable across runs.
    """

    task_ids = [int(task["task_id"]) for task in tasks]
    adjacency: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
    indegree: dict[int, int] = {task_id: 0 for task_id in task_ids}

    for edge in edges:
        src_task_id = int(edge["src_task_id"])
        dst_task_id = int(edge["dst_task_id"])
        adjacency[src_task_id].append(dst_task_id)
        indegree[dst_task_id] += 1

    ready_heap = [task_id for task_id in task_ids if indegree[task_id] == 0]
    heapq.heapify(ready_heap)
    topo_order: list[int] = []

    while ready_heap:
        task_id = heapq.heappop(ready_heap)
        topo_order.append(task_id)
        for dst_task_id in adjacency[task_id]:
            indegree[dst_task_id] -= 1
            if indegree[dst_task_id] == 0:
                heapq.heappush(ready_heap, dst_task_id)

    if len(topo_order) != len(task_ids):
        raise ValueError("Collective task graph contains a cycle")

    return topo_order


def _task_ready_levels(
    tasks: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> dict[int, int]:
    """Return the earliest-ready level induced by the collective DAG."""

    task_ids = [int(task["task_id"]) for task in tasks]
    preds_by_task: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
    for edge in edges:
        preds_by_task[int(edge["dst_task_id"])].append(int(edge["src_task_id"]))

    memo: dict[int, int] = {}

    def level(task_id: int) -> int:
        if task_id in memo:
            return memo[task_id]
        preds = preds_by_task[task_id]
        if not preds:
            memo[task_id] = 0
        else:
            memo[task_id] = 1 + max(level(pred_task_id) for pred_task_id in preds)
        return memo[task_id]

    return {task_id: level(task_id) for task_id in task_ids}


def build_collective_schedule(
    tenant_mapping: dict[int, dict[int, int]],
    single_flow_size_bits: int | None = None,
    collective: str | None = None,
    *,
    scale: float = 1e9,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
) -> dict[int, dict[str, object]]:
    """Build a fixed collective schedule description for mapping-only models.

    The returned schedule contains:
    - ``tasks``: transmission tasks
    - ``edges``: a unified DAG including collective and sender-order edges
    - ``sender_order``: the per-sender order used to derive sender-order edges
    - ``task_order``: a deterministic global append order consistent with the
      collective DAG levels

    This is intended to be treated as model input rather than a decision.
    """

    tasks_by_tenant = build_collective_tasks(
        tenant_mapping,
        single_flow_size_bits,
        collective,
        scale=scale,
        tenant_collective_specs=tenant_collective_specs,
    )
    dag_by_tenant = build_collective_dag(
        tenant_mapping,
        single_flow_size_bits,
        collective,
        scale=scale,
        tenant_collective_specs=tenant_collective_specs,
    )

    schedule: dict[int, dict[str, object]] = {}
    for tenant, tasks in tasks_by_tenant.items():
        task_lookup = {int(task["task_id"]): task for task in tasks}
        collective_edges = list(dag_by_tenant[tenant]["edges"])
        ready_levels = _task_ready_levels(tasks, collective_edges)
        task_order = sorted(
            [int(task["task_id"]) for task in tasks],
            key=lambda task_id: (ready_levels[int(task_id)], int(task_id)),
        )
        sender_order: dict[int, list[int]] = {}
        for task in tasks:
            sender_order.setdefault(int(task["src_rank"]), []).append(int(task["task_id"]))
        for sender_rank, task_ids in sender_order.items():
            task_ids.sort(
                key=lambda task_id: (
                    ready_levels[int(task_id)],
                    int(task_id),
                )
            )

        sender_order_edges: list[dict[str, object]] = []
        for src_rank, task_ids in sender_order.items():
            for earlier_task_id, later_task_id in zip(task_ids, task_ids[1:]):
                earlier_task = task_lookup[int(earlier_task_id)]
                later_task = task_lookup[int(later_task_id)]
                sender_order_edges.append(
                    {
                        "src_task_id": int(earlier_task_id),
                        "dst_task_id": int(later_task_id),
                        "src": str(earlier_task["name"]),
                        "dst": str(later_task["name"]),
                        "type": "sender_order",
                        "sender": int(src_rank),
                    }
                )

        unified_edges = collective_edges + sender_order_edges

        schedule[tenant] = {
            "tasks": tasks,
            "nodes": dag_by_tenant[tenant]["nodes"],
            "edges": unified_edges,
            "collective_edges": collective_edges,
            "sender_order_edges": sender_order_edges,
            "dag_nodes": dag_by_tenant[tenant]["nodes"],
            "dag_edges": unified_edges,
            "sender_order": sender_order,
            "ready_levels": ready_levels,
            "task_order": task_order,
        }

    return schedule
