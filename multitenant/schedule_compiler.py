from __future__ import annotations

from collections import defaultdict
import math


def task_levels(
    tasks: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> dict[int, int]:
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


def compile_epoch_schedule(
    schedule: dict[int, dict[str, object]],
    tenants: list[int],
    rank_orders: dict[int, list[int]],
    *,
    program_mode: bool = False,
) -> dict[str, object]:
    if program_mode:
        return _compile_program_epoch_schedule(schedule, tenants, rank_orders)
    return _compile_collective_epoch_schedule(schedule, tenants, rank_orders)


def _compile_collective_epoch_schedule(
    schedule: dict[int, dict[str, object]],
    tenants: list[int],
    rank_orders: dict[int, list[int]],
) -> dict[str, object]:
    compiled: dict[int, dict[str, object]] = {}
    global_max_epoch = -1
    for tenant in tenants:
        tenant_schedule = schedule.get(tenant, {})
        tasks = list(tenant_schedule.get("tasks", []))
        edges = list(tenant_schedule.get("edges", []))
        levels = task_levels(tasks, edges)
        epoch_pair_volume: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
        rank_weight: dict[int, float] = defaultdict(float)
        rank_incidence: dict[int, list[tuple[int, int, int, float]]] = defaultdict(list)

        for task in tasks:
            task_id = int(task["task_id"])
            epoch = int(levels[task_id])
            src_rank = int(task["src_rank"])
            dst_rank = int(task["dst_rank"])
            volume = float(task["V"])
            pair = (src_rank, dst_rank)
            epoch_pair_volume[epoch][pair] = epoch_pair_volume[epoch].get(pair, 0.0) + volume
            rank_weight[src_rank] += volume
            rank_weight[dst_rank] += volume
            rank_incidence[src_rank].append((epoch, src_rank, dst_rank, volume))
            rank_incidence[dst_rank].append((epoch, src_rank, dst_rank, volume))

        epoch_flows = _format_epoch_flows(epoch_pair_volume)
        max_epoch = max(epoch_flows.keys(), default=-1)
        global_max_epoch = max(global_max_epoch, max_epoch)
        compiled[tenant] = {
            "task_levels": levels,
            "epoch_flows": epoch_flows,
            "all_flows": _flatten_epoch_flows(epoch_flows),
            "max_epoch": int(max_epoch),
            "branch_order": _rank_branch_order(rank_orders[tenant], rank_weight),
            "rank_incidence": {
                int(rank): list(entries)
                for rank, entries in rank_incidence.items()
            },
        }

    return {
        "per_tenant": compiled,
        "global_max_epoch": int(global_max_epoch),
    }


def _compile_program_epoch_schedule(
    schedule: dict[int, dict[str, object]],
    tenants: list[int],
    rank_orders: dict[int, list[int]],
) -> dict[str, object]:
    compiled: dict[int, dict[str, object]] = {}
    global_max_epoch = -1
    positive_gaps = [
        float(op_meta.get("gap_after", 0.0))
        for tenant in tenants
        for op_meta in schedule.get(tenant, {}).get("collective_program", [])
        if float(op_meta.get("gap_after", 0.0)) > 0.0
    ]
    gap_epoch_unit = min(positive_gaps) if positive_gaps else None

    for tenant in tenants:
        tenant_schedule = schedule.get(tenant, {})
        tasks = list(tenant_schedule.get("tasks", []))
        task_lookup = {int(task["task_id"]): task for task in tasks}
        edges = list(tenant_schedule.get("edges", []))
        program = list(tenant_schedule.get("collective_program", []))
        epoch_pair_volume: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
        levels: dict[int, int] = {}
        rank_weight: dict[int, float] = defaultdict(float)
        rank_incidence: dict[int, list[tuple[int, int, int, float]]] = defaultdict(list)
        op_epoch_ranges: list[tuple[int, int, int]] = []
        epoch_offset = 0
        tenant_gap_time = 0.0

        for op_position, op_meta in enumerate(program):
            op_idx = int(op_meta.get("op_idx", op_position))
            op_task_ids = [int(task_id) for task_id in op_meta.get("task_ids", [])]
            op_task_id_set = set(op_task_ids)
            op_tasks = [task_lookup[task_id] for task_id in op_task_ids]
            op_edges = [
                edge
                for edge in edges
                if int(edge["src_task_id"]) in op_task_id_set
                and int(edge["dst_task_id"]) in op_task_id_set
            ]
            op_levels = task_levels(op_tasks, op_edges)

            for task_id, local_level in op_levels.items():
                levels[int(task_id)] = int(epoch_offset + local_level)

            for task_id in op_task_ids:
                task = task_lookup[task_id]
                epoch = int(levels[task_id])
                src_rank = int(task["src_rank"])
                dst_rank = int(task["dst_rank"])
                volume = float(task["V"])
                pair = (src_rank, dst_rank)
                epoch_pair_volume[epoch][pair] = epoch_pair_volume[epoch].get(pair, 0.0) + volume
                rank_weight[src_rank] += volume
                rank_weight[dst_rank] += volume
                rank_incidence[src_rank].append((epoch, src_rank, dst_rank, volume))
                rank_incidence[dst_rank].append((epoch, src_rank, dst_rank, volume))

            op_max_local_epoch = max(op_levels.values(), default=-1)
            op_start_epoch = int(epoch_offset)
            op_end_epoch = (
                int(epoch_offset + op_max_local_epoch)
                if op_max_local_epoch >= 0
                else int(epoch_offset - 1)
            )
            op_epoch_ranges.append((op_idx, op_start_epoch, op_end_epoch))

            if op_position < len(program) - 1:
                gap_after = float(op_meta.get("gap_after", 0.0))
                tenant_gap_time += gap_after
                gap_epochs = 0
                if gap_after > 0.0 and gap_epoch_unit is not None:
                    gap_epochs = max(1, int(math.ceil(gap_after / gap_epoch_unit)))
                epoch_offset = op_end_epoch + 1 + gap_epochs

        epoch_flows = _format_epoch_flows(epoch_pair_volume)
        max_epoch = max(epoch_flows.keys(), default=-1)
        global_max_epoch = max(global_max_epoch, max_epoch)
        compiled[tenant] = {
            "task_levels": levels,
            "epoch_flows": epoch_flows,
            "all_flows": _flatten_epoch_flows(epoch_flows),
            "max_epoch": int(max_epoch),
            "branch_order": _rank_branch_order(rank_orders[tenant], rank_weight),
            "rank_incidence": {
                int(rank): list(entries)
                for rank, entries in rank_incidence.items()
            },
            "op_epoch_ranges": op_epoch_ranges,
            "tenant_gap_time": float(tenant_gap_time),
        }

    return {
        "per_tenant": compiled,
        "global_max_epoch": int(global_max_epoch),
        "gap_epoch_unit": gap_epoch_unit,
    }


def _format_epoch_flows(
    epoch_pair_volume: dict[int, dict[tuple[int, int], float]],
) -> dict[int, list[tuple[int, int, float]]]:
    return {
        int(epoch): [
            (int(src_rank), int(dst_rank), float(volume))
            for (src_rank, dst_rank), volume in sorted(pair_volume.items())
        ]
        for epoch, pair_volume in epoch_pair_volume.items()
    }


def _flatten_epoch_flows(
    epoch_flows: dict[int, list[tuple[int, int, float]]],
) -> list[tuple[int, int, int, float]]:
    return [
        (int(epoch), int(src_rank), int(dst_rank), float(volume))
        for epoch, flows in epoch_flows.items()
        for src_rank, dst_rank, volume in flows
    ]


def _rank_branch_order(
    rank_order: list[int],
    rank_weight: dict[int, float],
) -> list[int]:
    return sorted(
        rank_order,
        key=lambda rank: (-rank_weight.get(rank, 0.0), int(rank)),
    )
