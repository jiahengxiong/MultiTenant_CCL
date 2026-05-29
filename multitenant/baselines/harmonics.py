from __future__ import annotations

import math
from itertools import product

import gurobipy as gp
from gurobipy import GRB

from multitenant.collectives import has_collective_workload, normalize_collective_programs
from multitenant.solvers.mapping_ilp import MappingILPSolver
from multitenant.simulator.adapter import simulate_collective_details
from multitenant.workloads import build_collective_program_schedule, build_collective_schedule


def _path_edges(path_table, tenant, src, dst):
    path = path_table.get((int(tenant), int(src), int(dst)))
    if not path:
        return []
    return list(zip(path[:-1], path[1:]))


def _is_interrack_edge(datacenter, edge):
    src, dst = edge
    src_type = datacenter.topology.nodes[src].get("type")
    dst_type = datacenter.topology.nodes[dst].get("type")
    return src_type != "server" and dst_type != "server"


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _build_schedule(
    tenant_mapping,
    tenant_flows,
    collective,
    single_flow_size,
    tenant_collective_specs,
    tenant_collective_programs,
):
    if has_collective_workload(
        collective=collective,
        single_flow_size=single_flow_size,
        tenant_collective_specs=tenant_collective_specs,
        tenant_collective_programs=tenant_collective_programs,
    ):
        if tenant_collective_programs is not None:
            return build_collective_program_schedule(
                tenant_mapping,
                tenant_collective_programs,
            )
        return build_collective_schedule(
            tenant_mapping,
            None if single_flow_size is None else int(single_flow_size),
            collective,
            tenant_collective_specs=tenant_collective_specs,
        )

    if tenant_flows is None:
        raise ValueError(
            "tenant_flows is required when collective/single_flow_size are not provided"
        )

    schedule_by_tenant = {}
    for tenant, flows in tenant_flows.items():
        tasks = []
        sender_order = {}
        for task_id, (src, dst, volume) in enumerate(flows):
            tasks.append(
                {
                    "task_id": int(task_id),
                    "name": f"{tenant}-flow-{task_id}",
                    "src_rank": int(src),
                    "dst_rank": int(dst),
                    "V": float(volume),
                    "preds": [],
                    "op_idx": 0,
                }
            )
            sender_order.setdefault(int(src), []).append(int(task_id))
        schedule_by_tenant[tenant] = {
            "tasks": tasks,
            "sender_order": sender_order,
            "task_order": [int(task["task_id"]) for task in tasks],
            "collective_program": [
                {
                    "op_idx": 0,
                    "collective": "custom",
                    "single_flow_size_bits": 0,
                    "gap_after": 0.0,
                    "task_ids": [int(task["task_id"]) for task in tasks],
                    "initial_task_ids": [int(task["task_id"]) for task in tasks],
                }
            ],
        }
    return schedule_by_tenant


def _build_program_metadata(
    datacenter,
    tenant_mapping,
    tenant_flows,
    path_table,
    collective=None,
    single_flow_size=None,
    tenant_collective_specs=None,
    tenant_collective_programs=None,
    slot_duration_override=None,
):
    capacities = {
        (src, dst): float(datacenter.topology[src][dst]["capacity"])
        for src, dst in datacenter.topology.edges()
    }
    interrack_capacities = {
        edge: cap
        for edge, cap in capacities.items()
        if _is_interrack_edge(datacenter, edge)
    }
    schedule_by_tenant = _build_schedule(
        tenant_mapping,
        tenant_flows,
        collective,
        single_flow_size,
        tenant_collective_specs,
        tenant_collective_programs,
    )

    op_metadata = {}
    program_by_tenant = {}
    task_durations = []
    op_durations = []
    positive_gaps = []

    for tenant in sorted(tenant_mapping):
        mapping = tenant_mapping[tenant]
        tenant_schedule = schedule_by_tenant.get(tenant, {})
        tasks = tenant_schedule.get("tasks", [])
        task_lookup = {}
        for task in tasks:
            task_id = int(task["task_id"])
            src_phys = mapping[int(task["src_rank"])]
            dst_phys = mapping[int(task["dst_rank"])]
            path_edges = _path_edges(path_table, tenant, src_phys, dst_phys)
            volume_bits = float(task["V"]) * 1e9
            path_bottleneck_bps = min(
                (
                    capacities[edge]
                    for edge in path_edges
                    if capacities.get(edge, 0.0) > 0.0
                ),
                default=0.0,
            )
            isolated_seconds = (
                volume_bits / path_bottleneck_bps
                if path_bottleneck_bps > 0.0
                else 0.0
            )
            task_durations.append(max(isolated_seconds, 1e-9))
            task_lookup[task_id] = {
                "task_id": task_id,
                "name": str(task["name"]),
                "op_idx": int(task.get("op_idx", 0)),
                "sender": int(task["src_rank"]),
                "preds": [int(pred) for pred in task.get("preds", [])],
                "volume_bits": volume_bits,
                "interrack_edges": [
                    edge for edge in path_edges if edge in interrack_capacities
                ],
                "isolated_seconds": max(isolated_seconds, 1e-9),
            }

        sender_order = {
            int(sender): [int(task_id) for task_id in task_ids]
            for sender, task_ids in tenant_schedule.get("sender_order", {}).items()
        }
        tenant_program = list(tenant_schedule.get("collective_program", []))
        program_by_tenant[tenant] = tenant_program
        for op_position, op_meta in enumerate(tenant_program):
            op_idx = int(op_meta.get("op_idx", op_position))
            op_task_ids = [int(task_id) for task_id in op_meta.get("task_ids", [])]
            op_task_set = set(op_task_ids)
            op_preds = {task_id: set() for task_id in op_task_ids}
            op_succs = {task_id: set() for task_id in op_task_ids}
            edge_bits = {}

            for task_id in op_task_ids:
                task = task_lookup.get(task_id)
                if task is None:
                    continue
                for pred in task["preds"]:
                    if pred in op_task_set:
                        op_preds[task_id].add(pred)
                        op_succs[pred].add(task_id)
                for edge in task["interrack_edges"]:
                    edge_bits[edge] = edge_bits.get(edge, 0.0) + float(task["volume_bits"])

            for task_ids in sender_order.values():
                for prev_task_id, next_task_id in zip(task_ids, task_ids[1:]):
                    prev_task_id = int(prev_task_id)
                    next_task_id = int(next_task_id)
                    if prev_task_id in op_task_set and next_task_id in op_task_set:
                        op_preds[next_task_id].add(prev_task_id)
                        op_succs[prev_task_id].add(next_task_id)

            remaining_preds = {task_id: set(preds) for task_id, preds in op_preds.items()}
            ready = sorted(task_id for task_id in op_task_ids if not remaining_preds[task_id])
            stages = []
            assigned = set()
            while ready:
                current_stage = list(ready)
                assigned.update(current_stage)
                stage_duration = max(
                    (float(task_lookup[task_id]["isolated_seconds"]) for task_id in current_stage),
                    default=0.0,
                )
                stage_edge_bits = {}
                for task_id in current_stage:
                    for edge in task_lookup[task_id]["interrack_edges"]:
                        stage_edge_bits[edge] = stage_edge_bits.get(edge, 0.0) + float(
                            task_lookup[task_id]["volume_bits"]
                        )
                stages.append(
                    {
                        "duration_s": max(stage_duration, 1e-9),
                        "edge_bits": stage_edge_bits,
                        "task_ids": list(current_stage),
                    }
                )
                next_ready = []
                for task_id in current_stage:
                    for succ in sorted(op_succs[task_id]):
                        remaining_preds[succ].discard(task_id)
                        if not remaining_preds[succ]:
                            next_ready.append(succ)
                ready = sorted(dict.fromkeys(next_ready))

            if len(assigned) != len(op_task_ids):
                raise ValueError(
                    f"Failed to build op stages for tenant={tenant} op={op_idx}"
                )

            isolated_seconds = sum(float(stage["duration_s"]) for stage in stages)
            gap_after = float(op_meta.get("gap_after", 0.0))
            if gap_after > 0.0:
                positive_gaps.append(gap_after)
            op_metadata[(tenant, op_idx)] = {
                "tenant": int(tenant),
                "op_idx": int(op_idx),
                "collective": str(op_meta.get("collective", "")),
                "gap_after": gap_after,
                "duration_s": max(isolated_seconds, 1e-9),
                "edge_bits": edge_bits,
                "stages": stages,
            }
            op_durations.append(max(isolated_seconds, 1e-9))

    if slot_duration_override is not None:
        slot_duration = float(slot_duration_override)
    else:
        min_op_duration = min(op_durations, default=1e-4)
        slot_duration = max(float(min_op_duration), 1e-6)

    for op_key, meta in op_metadata.items():
        duration_slots = max(1, int(math.ceil(float(meta["duration_s"]) / slot_duration)))
        gap_slots = max(0, int(math.ceil(float(meta["gap_after"]) / slot_duration)))
        meta["duration_slots"] = duration_slots
        meta["gap_slots"] = gap_slots
        meta["edge_load_gbits_per_slot"] = {
            edge: (float(bits) / 1e9) / duration_slots
            for edge, bits in meta["edge_bits"].items()
            if duration_slots > 0
        }
        stage_offset_slots = 0
        stage_slot_profile = []
        for stage in meta.get("stages", []):
            stage_duration_slots = max(
                1, int(math.ceil(float(stage["duration_s"]) / slot_duration))
            )
            stage_slot_profile.append(
                {
                    "start_offset_slots": stage_offset_slots,
                    "end_offset_slots": stage_offset_slots + stage_duration_slots,
                    "duration_slots": stage_duration_slots,
                    "edge_load_gbits_per_slot": {
                        edge: (float(bits) / 1e9) / stage_duration_slots
                        for edge, bits in stage.get("edge_bits", {}).items()
                        if stage_duration_slots > 0
                    },
                }
            )
            stage_offset_slots += stage_duration_slots
        if stage_slot_profile:
            meta["stages"] = stage_slot_profile
            meta["duration_slots"] = stage_slot_profile[-1]["end_offset_slots"]
        else:
            meta["stages"] = [
                {
                    "start_offset_slots": 0,
                    "end_offset_slots": duration_slots,
                    "duration_slots": duration_slots,
                    "edge_load_gbits_per_slot": dict(meta["edge_load_gbits_per_slot"]),
                }
            ]

    return {
        "capacities": capacities,
        "collective_program": program_by_tenant,
        "op_metadata": op_metadata,
        "slot_duration": slot_duration,
        "min_task_duration": min(task_durations, default=min(op_durations, default=1e-6)),
    }


class HarmonicsProgramILP:
    """Collective-level start-choice MILP.

    Each collective chooses one start slot. Fixed duration and fixed average
    link load per slot are derived from the given mapping and collective DAG.
    Overlap is allowed only if all shared links stay within capacity.
    """

    def __init__(
        self,
        datacenter,
        tenant_mapping,
        tenant_flows=None,
        path_table=None,
        single_flow_size=None,
        collective="allreduce",
        tenant_collective_specs=None,
        tenant_collective_programs=None,
        verbose=True,
        slot_duration=None,
        horizon_slots=None,
        **_,
    ):
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.path_table = path_table or self.datacenter.build_tenant_ecmp_path_table(
            sorted(int(tenant) for tenant in tenant_mapping)
        )
        self.single_flow_size = single_flow_size
        self.collective = collective
        self.tenant_collective_specs = tenant_collective_specs
        self.tenant_collective_programs = normalize_collective_programs(
            self.tenant_mapping,
            collective=self.collective,
            single_flow_size=self.single_flow_size,
            tenant_collective_specs=self.tenant_collective_specs,
            tenant_collective_programs=tenant_collective_programs,
        )
        self.verbose = bool(verbose)
        self.slot_duration_override = slot_duration
        self.horizon_slots_override = horizon_slots

        self.M = sorted(self.tenant_mapping)
        self.data = _build_program_metadata(
            self.datacenter,
            self.tenant_mapping,
            self.tenant_flows,
            self.path_table,
            collective=self.collective,
            single_flow_size=self.single_flow_size,
            tenant_collective_specs=self.tenant_collective_specs,
            tenant_collective_programs=self.tenant_collective_programs,
            slot_duration_override=self.slot_duration_override,
        )
        self.model = gp.Model("HarmonicsCollectiveMILP")
        self.model.Params.OutputFlag = 1 if self.verbose else 0

        self.start_choice = {}
        self.start_slot = {}
        self.op_release_slot = self.start_slot
        self.offset_slot = {}
        self.finish_slot = {}
        self.op_finish_slot = self.finish_slot
        self.tenant_finish = {}
        self.avg_completion_expr = None
        self.T_max = None

        self.final_makespan = 0.0
        self.final_avg_jct = 0.0
        self.final_tenant_avg_jct = 0.0
        self.tenant_finish_values = {int(tenant): 0.0 for tenant in self.M}
        self.collective_start_times = {int(tenant): {} for tenant in self.M}
        self.collective_rate_scales = {int(tenant): {} for tenant in self.M}
        self.collective_rate_schedule = {int(tenant): {} for tenant in self.M}
        self.task_rate_schedule = {int(tenant): {} for tenant in self.M}

        self._build_model()

    def _all_op_keys(self):
        return [
            (int(tenant), int(op_meta.get("op_idx", 0)))
            for tenant in self.M
            for op_meta in self.data["collective_program"].get(tenant, [])
        ]

    def _build_zero_offset_solution(self):
        slot_duration = float(self.data["slot_duration"])
        starts = {int(tenant): {} for tenant in self.M}
        tenant_finishes = {}
        for tenant in self.M:
            current_release_slots = 0
            for op_meta in self.data["collective_program"].get(tenant, []):
                op_idx = int(op_meta.get("op_idx", 0))
                op_key = (tenant, op_idx)
                starts[int(tenant)][int(op_idx)] = 0.0
                finish_slots = current_release_slots + int(self.data["op_metadata"][op_key]["duration_slots"])
                current_release_slots = finish_slots + int(self.data["op_metadata"][op_key]["gap_slots"])
            tenant_finishes[int(tenant)] = current_release_slots * slot_duration
        self.collective_start_times = starts
        self.collective_rate_scales = {int(tenant): {} for tenant in self.M}
        self.collective_rate_schedule = {int(tenant): {} for tenant in self.M}
        self.task_rate_schedule = {int(tenant): {} for tenant in self.M}
        self.tenant_finish_values = tenant_finishes
        self.final_tenant_avg_jct = (
            sum(tenant_finishes.values()) / len(tenant_finishes) if tenant_finishes else 0.0
        )
        self.final_avg_jct = self.final_tenant_avg_jct
        self.final_makespan = max(tenant_finishes.values(), default=0.0)
        return starts

    def _zero_offset_score(self):
        slot_duration = float(self.data["slot_duration"])
        tenant_finishes = []
        for tenant in self.M:
            current_release_slots = 0
            for op_meta in self.data["collective_program"].get(tenant, []):
                op_idx = int(op_meta.get("op_idx", 0))
                op_key = (tenant, op_idx)
                finish_slots = current_release_slots + int(self.data["op_metadata"][op_key]["duration_slots"])
                current_release_slots = finish_slots + int(self.data["op_metadata"][op_key]["gap_slots"])
            tenant_finishes.append(current_release_slots * slot_duration)
        avg = sum(tenant_finishes) / len(tenant_finishes) if tenant_finishes else 0.0
        makespan = max(tenant_finishes, default=0.0)
        return avg, makespan

    def _simulator_score(self, collective_start_times):
        if self.path_table is None or self.tenant_collective_programs is None:
            return None
        details = simulate_collective_details(
            self.datacenter.topology,
            self.tenant_mapping,
            self.path_table,
            tenant_collective_programs=self.tenant_collective_programs,
            collective_start_times=collective_start_times,
            collective_rate_scales={int(tenant): {} for tenant in self.M},
            collective_rate_schedule={int(tenant): {} for tenant in self.M},
            task_rate_schedule={int(tenant): {} for tenant in self.M},
        )
        return (
            float(details["avg_tenant_makespan"]),
            float(details["global_makespan"]),
        )

    def _apply_zero_offset_warm_start(self):
        for tenant in self.M:
            current_release = 0
            for op_meta in self.data["collective_program"].get(tenant, []):
                op_idx = int(op_meta.get("op_idx", 0))
                op_key = (tenant, op_idx)
                duration = int(self.data["op_metadata"][op_key]["duration_slots"])
                self.offset_slot[op_key].Start = 0.0
                for start_t in self._candidate_starts(op_key):
                    self.start_choice[(op_key, start_t)].Start = 1.0 if start_t == current_release else 0.0
                self.start_slot[op_key].Start = float(current_release)
                self.finish_slot[op_key].Start = float(current_release + duration)
                current_release += duration + int(self.data["op_metadata"][op_key]["gap_slots"])
            self.tenant_finish[tenant].Start = current_release * float(self.data["slot_duration"])
        self.model.update()

    def _candidate_starts(self, op_key):
        min_start = int(self.data["op_metadata"][op_key]["earliest_start_slot"])
        max_start = int(self.data["op_metadata"][op_key]["latest_start_slot"])
        return range(min_start, max_start + 1)

    def _active_expr(self, op_key, slot_t):
        duration = int(self.data["op_metadata"][op_key]["duration_slots"])
        earliest_start = int(self.data["op_metadata"][op_key]["earliest_start_slot"])
        latest_start = int(self.data["op_metadata"][op_key]["latest_start_slot"])
        terms = []
        start_lo = max(earliest_start, slot_t - duration + 1)
        start_hi = min(slot_t, latest_start)
        for start_t in range(start_lo, start_hi + 1):
            terms.append(self.start_choice[(op_key, start_t)])
        if not terms:
            return gp.LinExpr(0.0)
        return gp.quicksum(terms)

    def _stage_active_expr(self, op_key, stage_idx, slot_t):
        meta = self.data["op_metadata"][op_key]
        stage = meta["stages"][stage_idx]
        earliest_start = int(meta["earliest_start_slot"])
        latest_start = int(meta["latest_start_slot"])
        stage_start = int(stage["start_offset_slots"])
        stage_end = int(stage["end_offset_slots"])
        terms = []
        start_lo = max(earliest_start, slot_t - stage_end + 1)
        start_hi = min(latest_start, slot_t - stage_start)
        for start_t in range(start_lo, start_hi + 1):
            terms.append(self.start_choice[(op_key, start_t)])
        if not terms:
            return gp.LinExpr(0.0)
        return gp.quicksum(terms)

    def _build_model(self):
        all_op_keys = self._all_op_keys()
        capacities = self.data["capacities"]
        slot_duration = float(self.data["slot_duration"])
        delay_budget_per_collective = max(1, int(self.data.get("max_offset_slots_per_collective", max(2, len(self.M)))))
        tenant_horizon_slots = {}
        for tenant in self.M:
            current_release = 0
            for op_position, op_meta in enumerate(self.data["collective_program"].get(tenant, [])):
                op_idx = int(op_meta.get("op_idx", op_position))
                op_key = (tenant, op_idx)
                duration = int(self.data["op_metadata"][op_key]["duration_slots"])
                gap_slots = int(self.data["op_metadata"][op_key]["gap_slots"])
                earliest_start = current_release
                latest_start = earliest_start + (op_position + 1) * delay_budget_per_collective
                self.data["op_metadata"][op_key]["earliest_start_slot"] = earliest_start
                self.data["op_metadata"][op_key]["latest_start_slot"] = latest_start
                current_release = earliest_start + duration + gap_slots
            tenant_horizon_slots[int(tenant)] = current_release + len(self.data["collective_program"].get(tenant, [])) * delay_budget_per_collective

        default_horizon = max(8, max(tenant_horizon_slots.values(), default=0) + delay_budget_per_collective + 2)
        self.data["num_slots"] = (
            int(self.horizon_slots_override)
            if self.horizon_slots_override is not None
            else default_horizon
        )
        self.data["Horizon"] = list(range(self.data["num_slots"]))

        for op_key in all_op_keys:
            tenant, op_idx = op_key
            duration = int(self.data["op_metadata"][op_key]["duration_slots"])
            earliest_start = int(self.data["op_metadata"][op_key]["earliest_start_slot"])
            latest_start = max(
                earliest_start,
                min(
                int(self.data["op_metadata"][op_key]["latest_start_slot"]),
                self.data["num_slots"] - duration,
                ),
            )
            self.data["op_metadata"][op_key]["latest_start_slot"] = latest_start
            offset_ub = max(0, latest_start - earliest_start)
            self.offset_slot[op_key] = self.model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=offset_ub,
                name=f"Offset_{tenant}_{op_idx}",
            )
            self.start_slot[op_key] = self.model.addVar(
                vtype=GRB.INTEGER,
                lb=earliest_start,
                ub=latest_start,
                name=f"Start_{tenant}_{op_idx}",
            )
            self.finish_slot[op_key] = self.model.addVar(
                vtype=GRB.INTEGER,
                lb=duration,
                ub=self.data["num_slots"],
                name=f"Finish_{tenant}_{op_idx}",
            )
            choice_vars = []
            for start_t in self._candidate_starts(op_key):
                var = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"Y_{tenant}_{op_idx}_{start_t}",
                )
                self.start_choice[(op_key, start_t)] = var
                choice_vars.append(var)
            self.model.addConstr(
                gp.quicksum(choice_vars) == 1,
                name=f"Choose_{tenant}_{op_idx}",
            )
            self.model.addConstr(
                self.start_slot[op_key]
                == gp.quicksum(start_t * self.start_choice[(op_key, start_t)] for start_t in self._candidate_starts(op_key)),
                name=f"LinkStart_{tenant}_{op_idx}",
            )
            self.model.addConstr(
                self.finish_slot[op_key] == self.start_slot[op_key] + duration,
                name=f"LinkFinish_{tenant}_{op_idx}",
            )

        for tenant in self.M:
            program = list(self.data["collective_program"].get(tenant, []))
            self.tenant_finish[tenant] = self.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"T_{tenant}",
            )
            for op_position, op_meta in enumerate(program):
                op_idx = int(op_meta.get("op_idx", op_position))
                op_key = (tenant, op_idx)
                if op_position > 0:
                    prev_meta = program[op_position - 1]
                    prev_key = (tenant, int(prev_meta.get("op_idx", op_position - 1)))
                    prev_gap = int(self.data["op_metadata"][prev_key]["gap_slots"])
                    self.model.addConstr(
                        self.start_slot[op_key] == self.finish_slot[prev_key] + prev_gap + self.offset_slot[op_key],
                        name=f"Chain_{tenant}_{op_idx}",
                    )
                else:
                    earliest_start = int(self.data["op_metadata"][op_key]["earliest_start_slot"])
                    self.model.addConstr(
                        self.start_slot[op_key] == earliest_start + self.offset_slot[op_key],
                        name=f"FirstChain_{tenant}_{op_idx}",
                    )
                self.model.addConstr(
                    self.tenant_finish[tenant] >= self.finish_slot[op_key] * slot_duration,
                    name=f"TenantFinishGe_{tenant}_{op_idx}",
                )

        for link, capacity_bps in capacities.items():
            slot_capacity_gbits = float(capacity_bps) * slot_duration / 1e9
            for slot_t in self.data["Horizon"]:
                link_terms = []
                for op_key in all_op_keys:
                    for stage_idx, stage in enumerate(self.data["op_metadata"][op_key]["stages"]):
                        load = float(stage["edge_load_gbits_per_slot"].get(link, 0.0))
                        if load <= 0.0:
                            continue
                        link_terms.append(load * self._stage_active_expr(op_key, stage_idx, slot_t))
                if link_terms:
                    self.model.addConstr(
                        gp.quicksum(link_terms) <= slot_capacity_gbits,
                        name=f"LinkCap_{link[0]}_{link[1]}_{slot_t}",
                    )

        self.T_max = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="T_max")
        for tenant in self.M:
            self.model.addConstr(self.T_max >= self.tenant_finish[tenant], name=f"TmaxGe_{tenant}")
        if self.M:
            self.avg_completion_expr = gp.quicksum(self.tenant_finish[tenant] for tenant in self.M) / len(self.M)
        else:
            self.avg_completion_expr = gp.LinExpr(0.0)
        self.model.ModelSense = GRB.MINIMIZE
        self.model.setObjectiveN(self.avg_completion_expr, index=0, priority=2, name="avg_jct")
        self.model.setObjectiveN(self.T_max, index=1, priority=1, name="makespan")
        self.model.update()
        self._apply_zero_offset_warm_start()

    def solve(self, timelimit=30, mipgap=0.0):
        self.model.update()
        if timelimit is not None:
            self.model.setParam("TimeLimit", float(timelimit))
        self.model.setParam("MIPGap", float(mipgap))
        self.model.optimize()
        if self.model.SolCount == 0:
            return self._build_zero_offset_solution()

        slot_duration = float(self.data["slot_duration"])
        tenant_finishes = []
        starts = {int(tenant): {} for tenant in self.M}
        for tenant in self.M:
            prev_finish_slot = None
            prev_gap_slots = 0
            for op_meta in self.data["collective_program"].get(tenant, []):
                op_idx = int(op_meta.get("op_idx", 0))
                op_key = (tenant, op_idx)
                release_slot = float(self.start_slot[op_key].X)
                finish_slot = float(self.finish_slot[op_key].X)
                if prev_finish_slot is None:
                    offset_slots = max(0.0, release_slot)
                else:
                    offset_slots = max(0.0, release_slot - prev_finish_slot - prev_gap_slots)
                starts[int(tenant)][int(op_idx)] = offset_slots * slot_duration
                prev_finish_slot = finish_slot
                prev_gap_slots = int(self.data["op_metadata"][op_key]["gap_slots"])
            tenant_finish_s = float(self.tenant_finish[tenant].X)
            self.tenant_finish_values[int(tenant)] = tenant_finish_s
            tenant_finishes.append(tenant_finish_s)

        self.collective_start_times = starts
        self.final_tenant_avg_jct = (
            sum(tenant_finishes) / len(tenant_finishes) if tenant_finishes else 0.0
        )
        self.final_avg_jct = self.final_tenant_avg_jct
        self.final_makespan = max(tenant_finishes, default=0.0)
        zero_model_avg, zero_model_makespan = self._zero_offset_score()
        candidate_score = self._simulator_score(starts)
        zero_starts = {
            int(tenant): {
                int(op_meta.get("op_idx", 0)): 0.0
                for op_meta in self.data["collective_program"].get(tenant, [])
            }
            for tenant in self.M
        }
        zero_score = self._simulator_score(zero_starts)
        if candidate_score is not None and zero_score is not None:
            if (
                candidate_score[0] > zero_score[0] + 1e-12
                or (
                    abs(candidate_score[0] - zero_score[0]) <= 1e-12
                    and candidate_score[1] > zero_score[1] + 1e-12
                )
            ):
                return self._build_zero_offset_solution()
            self.final_avg_jct = candidate_score[0]
            self.final_tenant_avg_jct = candidate_score[0]
            self.final_makespan = candidate_score[1]
        self.final_avg_jct = min(self.final_avg_jct, zero_model_avg)
        self.final_tenant_avg_jct = min(self.final_tenant_avg_jct, zero_model_avg)
        if abs(self.final_avg_jct - zero_model_avg) <= 1e-12:
            self.final_makespan = min(self.final_makespan, zero_model_makespan)
        return starts

    def get_collective_start_times(self):
        return self.collective_start_times

    def get_collective_rate_scales(self):
        return self.collective_rate_scales

    def get_collective_rate_schedule(self):
        return self.collective_rate_schedule

    def get_collective_scheduled_times(self):
        return self.collective_start_times

    def get_task_rate_schedule(self):
        return self.task_rate_schedule


_StageProfileHarmonicsProgramILP = HarmonicsProgramILP


class HarmonicsProgramILP(MappingILPSolver):
    """Fixed-mapping + collective-offset MILP built directly from mapping ILP."""

    def __init__(
        self,
        datacenter,
        tenant_mapping,
        tenant_flows=None,
        path_table=None,
        single_flow_size=None,
        collective="allreduce",
        tenant_collective_specs=None,
        tenant_collective_programs=None,
        verbose=True,
        slot_duration=None,
        horizon_slots=None,
        enable_offset_scan_fallback=True,
        **kwargs,
    ):
        self.path_table = path_table or datacenter.build_tenant_ecmp_path_table(
            sorted(int(tenant) for tenant in tenant_mapping)
        )
        self._enable_offset_scan_fallback = bool(enable_offset_scan_fallback)
        self._spawn_args = dict(
            datacenter=datacenter,
            tenant_mapping=tenant_mapping,
            tenant_flows=tenant_flows,
            path_table=self.path_table,
            single_flow_size=single_flow_size,
            collective=collective,
            tenant_collective_specs=tenant_collective_specs,
            tenant_collective_programs=tenant_collective_programs,
            verbose=verbose,
            slot_duration=slot_duration,
            horizon_slots=horizon_slots,
            enable_offset_scan_fallback=False,
        )
        kwargs.pop("timelimit", None)
        kwargs.pop("mipgap", None)
        self._program_metadata = _build_program_metadata(
            datacenter,
            tenant_mapping,
            tenant_flows,
            self.path_table,
            collective=collective,
            single_flow_size=single_flow_size,
            tenant_collective_specs=tenant_collective_specs,
            tenant_collective_programs=tenant_collective_programs,
            slot_duration_override=None,
        )
        if slot_duration is None:
            # Use a finer task-time grid than the offset quantum so the model can
            # distinguish nearby offset choices that fall into the same coarse
            # completion slot under a 1-task-per-slot discretization.
            slot_duration = max(float(self._program_metadata.get("min_task_duration", 1e-6)) / 2.0, 1e-6)
        offset_quantum_s = max(float(self._program_metadata.get("min_task_duration", slot_duration)), 1e-6)
        self._offset_quantum_slots = max(
            1,
            int(
                math.ceil(
                    float(offset_quantum_s) / max(float(slot_duration), 1e-12)
                )
            ),
        )
        self.collective_rate_scales = {int(tenant): {} for tenant in tenant_mapping}
        self.collective_rate_schedule = {int(tenant): {} for tenant in tenant_mapping}
        self.task_rate_schedule = {int(tenant): {} for tenant in tenant_mapping}
        self.collective_start_times = {int(tenant): {} for tenant in tenant_mapping}
        self.tenant_finish_values = {int(tenant): 0.0 for tenant in tenant_mapping}
        self.final_tenant_avg_jct = 0.0
        super().__init__(
            datacenter=datacenter,
            tenant_mapping=tenant_mapping,
            tenant_flows=tenant_flows,
            verbose=verbose,
            name="harmonics_fixed_mapping_timing",
            collective=collective,
            single_flow_size=single_flow_size,
            tenant_collective_specs=tenant_collective_specs,
            tenant_collective_programs=tenant_collective_programs,
            slot_duration=slot_duration,
            horizon_slots=horizon_slots,
            enable_heuristic_warm_start=False,
            enable_full_mip_start=False,
            path_table=self.path_table,
            **kwargs,
        )

    def _build(self, name="harmonics_fixed_mapping_timing"):
        if self.model is not None:
            self.model.dispose()

        self.op_release_slot = {}
        self.op_offset_step = {}
        self.op_offset_slot = {}
        self.op_finish_slot = {}
        self.op_release_choice = {}
        self.W = {}
        self.R = {}
        self.Z = {}
        self.S_active = {}
        self.G_start = {}
        self.D_full = {}
        self.Q_bottleneck = {}
        self.F = {}
        self.C = {}
        self.LinkMinRate = {}
        self.task_finish = {}
        self.tenant_finish = {}
        self.T_m = {}
        self.T_stage = {}
        self.T_max = None

        self.data = self._build_data()
        self._augment_release_data()

        self.model = gp.Model(name)
        self.model.Params.OutputFlag = 1 if self.verbose else 0

        self._add_release_variables()
        self._add_fixed_task_time_model()
        self._set_lexicographic_objective()
        self.model.update()
        self._apply_zero_offset_warm_start()

    def _augment_release_data(self):
        delay_budget_steps = max(8, len(self.data["M"]) * 4)
        self.data["max_offset_steps_per_collective"] = delay_budget_steps
        op_task_ids = {}
        op_initial_tasks = {}
        task_to_op = {}
        op_release_windows = {}
        for tenant in self.data["M"]:
            program = list(self.data["schedule"].get(tenant, {}).get("collective_program", []))
            baseline_release = 0
            for op_pos, op_meta in enumerate(program):
                op_idx = int(op_meta.get("op_idx", op_pos))
                key = (tenant, op_idx)
                task_ids = [int(task_id) for task_id in op_meta.get("task_ids", [])]
                initial_task_ids = [int(task_id) for task_id in op_meta.get("initial_task_ids", [])]
                op_task_ids[key] = task_ids
                op_initial_tasks[key] = set(initial_task_ids)
                for task_id in task_ids:
                    task_to_op[(tenant, task_id)] = key
                latest_release = baseline_release + (op_pos + 1) * delay_budget_steps * self._offset_quantum_slots
                op_release_windows[key] = (baseline_release, latest_release)
                op_duration_slots = int(self._program_metadata["op_metadata"].get(key, {}).get("duration_slots", 1))
                gap_slots = 0
                bootstrap_meta = self._program_metadata["op_metadata"].get(key)
                if bootstrap_meta is not None:
                    gap_slots = int(bootstrap_meta.get("gap_slots", 0))
                baseline_release += op_duration_slots + gap_slots
        self.data["op_task_ids"] = op_task_ids
        self.data["op_initial_tasks"] = op_initial_tasks
        self.data["task_to_op"] = task_to_op
        self.data["op_release_windows"] = op_release_windows
        tenant_horizon = []
        for tenant in self.data["M"]:
            total_slots = 0
            for op_meta in self.data["schedule"].get(tenant, {}).get("collective_program", []):
                op_idx = int(op_meta.get("op_idx", 0))
                key = (tenant, op_idx)
                bootstrap_meta = self._program_metadata["op_metadata"].get(key)
                if bootstrap_meta is None:
                    continue
                total_slots += int(bootstrap_meta.get("duration_slots", 1))
                total_slots += int(bootstrap_meta.get("gap_slots", 0))
            total_slots += len(self.data["schedule"].get(tenant, {}).get("collective_program", [])) * delay_budget_steps * self._offset_quantum_slots
            tenant_horizon.append(total_slots)
        self.data["num_slots"] = max(
            8,
            max(tenant_horizon, default=0) + 2,
            sum(tenant_horizon) + 2,
        )
        self.data["Horizon"] = list(range(self.data["num_slots"]))

    def _candidate_release_slots(self, tenant, op_idx):
        lo, hi = self.data["op_release_windows"][(tenant, op_idx)]
        hi = min(int(hi), self.data["num_slots"] - 1)
        lo = max(0, min(int(lo), hi))
        step = max(1, int(getattr(self, "_offset_quantum_slots", 1)))
        candidates = list(range(lo, hi + 1, step))
        if not candidates or candidates[-1] != hi:
            candidates.append(hi)
        return candidates

    def _release_ready_expr(self, tenant, op_idx, slot_t):
        terms = [
            self.op_release_choice[(tenant, op_idx, start_t)]
            for start_t in range(0, slot_t + 1)
            if (tenant, op_idx, start_t) in self.op_release_choice
        ]
        if not terms:
            return gp.LinExpr(0.0)
        return gp.quicksum(terms)

    def _task_path_edges(self, tenant, task):
        src_server = self.tenant_mapping[tenant][int(task["src_rank"])]
        dst_server = self.tenant_mapping[tenant][int(task["dst_rank"])]
        return list(self.data["path_edges"].get((int(tenant), int(src_server), int(dst_server)), []))

    def _add_release_variables(self):
        schedule = self.data["schedule"]
        for tenant in self.data["M"]:
            for op_pos, op_meta in enumerate(schedule.get(tenant, {}).get("collective_program", [])):
                op_idx = int(op_meta.get("op_idx", op_pos))
                key = (tenant, op_idx)
                self.op_release_slot[key] = self.model.addVar(
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=self.data["num_slots"] - 1,
                    name=f"OpRelease_{tenant}_{op_idx}",
                )
                self.op_offset_step[key] = self.model.addVar(
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=self.data["max_offset_steps_per_collective"],
                    name=f"OpOffsetStep_{tenant}_{op_idx}",
                )
                self.op_offset_slot[key] = self.model.addVar(
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=self.data["max_offset_steps_per_collective"] * self._offset_quantum_slots,
                    name=f"OpOffset_{tenant}_{op_idx}",
                )
                self.op_finish_slot[key] = self.model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    name=f"OpFinish_{tenant}_{op_idx}",
                )
                choices = []
                for start_t in self._candidate_release_slots(tenant, op_idx):
                    var = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"OpReleaseChoice_{tenant}_{op_idx}_{start_t}",
                    )
                    self.op_release_choice[(tenant, op_idx, start_t)] = var
                    choices.append(var)
                self.model.addConstr(gp.quicksum(choices) == 1, name=f"OpReleaseOne_{tenant}_{op_idx}")
                self.model.addConstr(
                    self.op_release_slot[key]
                    == gp.quicksum(
                        start_t * self.op_release_choice[(tenant, op_idx, start_t)]
                        for start_t in self._candidate_release_slots(tenant, op_idx)
                    ),
                    name=f"OpReleaseLink_{tenant}_{op_idx}",
                )
                self.model.addConstr(
                    self.op_offset_slot[key] == self.op_offset_step[key] * self._offset_quantum_slots,
                    name=f"OpOffsetQuantum_{tenant}_{op_idx}",
                )

        for tenant in self.data["M"]:
            program = list(schedule.get(tenant, {}).get("collective_program", []))
            for op_pos, op_meta in enumerate(program):
                op_idx = int(op_meta.get("op_idx", op_pos))
                key = (tenant, op_idx)
                if op_pos == 0:
                    self.model.addConstr(
                        self.op_release_slot[key] == self.op_offset_slot[key],
                        name=f"OpReleaseFirst_{tenant}_{op_idx}",
                    )
                else:
                    prev_meta = program[op_pos - 1]
                    prev_idx = int(prev_meta.get("op_idx", op_pos - 1))
                    prev_key = (tenant, prev_idx)
                    gap_slots = int(math.ceil(float(prev_meta.get("gap_after", 0.0)) / float(self.data["slot_duration"])))
                    self.model.addConstr(
                        self.op_release_slot[key]
                        == self.op_finish_slot[prev_key] + gap_slots + self.op_offset_slot[key],
                        name=f"OpReleaseChain_{tenant}_{op_idx}",
                    )

    def _add_fixed_task_time_model(self):
        tenants = self.data["M"]
        links = self.data["L"]
        capacities = self.data["cap"]
        tasks_by_tenant = self.data["tasks"]
        horizon = self.data["Horizon"]
        num_slots = self.data["num_slots"]
        slot_duration = self.data["slot_duration"]
        min_send_unit = self.data["min_send_unit"]
        task_lookup = {
            tenant: {task["task_id"]: task for task in tasks_by_tenant[tenant]}
            for tenant in tenants
        }
        task_path_edges = {
            (tenant, task["task_id"]): [edge for edge in self._task_path_edges(tenant, task) if edge in capacities]
            for tenant in tenants
            for task in tasks_by_tenant[tenant]
        }

        for tenant in tenants:
            task_ids = [task["task_id"] for task in tasks_by_tenant[tenant]]
            for task_id in task_ids:
                task = task_lookup[tenant][task_id]
                preds = list(task["preds"])
                task_op_key = self.data["task_to_op"][(tenant, task_id)]
                is_initial = task_id in self.data["op_initial_tasks"].get(task_op_key, set())
                for t in horizon:
                    self.R[(tenant, task_id, t)] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"R_{tenant}_{task_id}_{t}")
                    self.Z[(tenant, task_id, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"Z_{tenant}_{task_id}_{t}")
                    self.S_active[(tenant, task_id, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"S_{tenant}_{task_id}_{t}")
                    self.G_start[(tenant, task_id, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"G_start_{tenant}_{task_id}_{t}")
                    self.D_full[(tenant, task_id, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"D_full_{tenant}_{task_id}_{t}")
                    self.C[(tenant, task_id, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"C_{tenant}_{task_id}_{t}")

                for link in task_path_edges[(tenant, task_id)]:
                    self.W[(tenant, task_id, link)] = 1.0
                    volume = self.data["task_total_volume"][(tenant, task_id)]
                    for t in horizon:
                        self.F[(tenant, task_id, link, t)] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"F_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}")
                        self.Q_bottleneck[(tenant, task_id, link, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"Q_bn_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}")
                        self.model.addConstr(self.F[(tenant, task_id, link, t)] <= capacities[link] * slot_duration * self.S_active[(tenant, task_id, t)], name=f"F_gate_S_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}")
                        self.model.addConstr(self.F[(tenant, task_id, link, t)] <= self.R[(tenant, task_id, t)], name=f"F_le_R_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}")
                    self.model.addConstr(
                        gp.quicksum(self.F[(tenant, task_id, link, t)] for t in horizon) == volume,
                        name=f"load_balance_{tenant}_{task_id}_{link[0]}_{link[1]}",
                    )
                    cumulative = gp.LinExpr()
                    for t in horizon:
                        cumulative += self.F[(tenant, task_id, link, t)]
                        self.model.addConstr(
                            cumulative >= volume - volume * (1 - self.C[(tenant, task_id, t)]),
                            name=f"complete_if_sent_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                        )

                for t in horizon:
                    bn_terms = [
                        self.Q_bottleneck[(tenant, task_id, link, t)]
                        for link in task_path_edges[(tenant, task_id)]
                        if (tenant, task_id, link, t) in self.Q_bottleneck
                    ]
                    self.model.addConstr(
                        gp.quicksum(bn_terms) == self.D_full[(tenant, task_id, t)],
                        name=f"Q_one_bottleneck_{tenant}_{task_id}_{t}",
                    )

                for t in range(num_slots - 1):
                    self.model.addConstr(self.C[(tenant, task_id, t)] <= self.C[(tenant, task_id, t + 1)], name=f"C_mono_{tenant}_{task_id}_{t}")
                self.model.addConstr(self.C[(tenant, task_id, num_slots - 1)] == 1, name=f"C_terminal_{tenant}_{task_id}")

                release0 = self._release_ready_expr(task_op_key[0], task_op_key[1], 0) if is_initial else gp.LinExpr(1.0)
                if preds:
                    self.model.addConstr(self.Z[(tenant, task_id, 0)] == 0, name=f"Z_wait_preds_{tenant}_{task_id}")
                else:
                    self.model.addConstr(self.Z[(tenant, task_id, 0)] == release0, name=f"Z_init_{tenant}_{task_id}")
                self.model.addConstr(self.S_active[(tenant, task_id, 0)] <= self.Z[(tenant, task_id, 0)], name=f"S_init_le_Z_{tenant}_{task_id}")
                self.model.addConstr(self.G_start[(tenant, task_id, 0)] == self.S_active[(tenant, task_id, 0)], name=f"G_start_init_{tenant}_{task_id}")

                for t in range(1, num_slots):
                    conditions = [1 - self.C[(tenant, task_id, t - 1)]]
                    self.model.addConstr(self.Z[(tenant, task_id, t)] <= 1 - self.C[(tenant, task_id, t - 1)], name=f"Z_after_completion_{tenant}_{task_id}_{t}")
                    for pred_task_id in preds:
                        self.model.addConstr(self.Z[(tenant, task_id, t)] <= self.C[(tenant, pred_task_id, t - 1)], name=f"pred_ready_{tenant}_{task_id}_{pred_task_id}_{t}")
                        conditions.append(self.C[(tenant, pred_task_id, t - 1)])
                    if is_initial:
                        release_expr = self._release_ready_expr(task_op_key[0], task_op_key[1], t)
                        self.model.addConstr(self.Z[(tenant, task_id, t)] <= release_expr, name=f"release_ready_{tenant}_{task_id}_{t}")
                        conditions.append(release_expr)
                    self.model.addConstr(
                        self.Z[(tenant, task_id, t)] >= gp.quicksum(conditions) - (len(conditions) - 1),
                        name=f"Z_ready_exact_{tenant}_{task_id}_{t}",
                    )
                    self.model.addConstr(self.S_active[(tenant, task_id, t)] <= self.Z[(tenant, task_id, t)], name=f"S_le_Z_{tenant}_{task_id}_{t}")
                    self.model.addConstr(self.S_active[(tenant, task_id, t)] >= self.S_active[(tenant, task_id, t - 1)] - self.C[(tenant, task_id, t - 1)], name=f"S_nonpreempt_{tenant}_{task_id}_{t}")
                    self.model.addConstr(self.G_start[(tenant, task_id, t)] <= self.S_active[(tenant, task_id, t)], name=f"G_start_le_S_{tenant}_{task_id}_{t}")
                    self.model.addConstr(self.G_start[(tenant, task_id, t)] <= 1 - self.S_active[(tenant, task_id, t - 1)], name=f"G_start_le_not_prev_S_{tenant}_{task_id}_{t}")
                    self.model.addConstr(self.G_start[(tenant, task_id, t)] >= self.S_active[(tenant, task_id, t)] - self.S_active[(tenant, task_id, t - 1)], name=f"G_start_ge_S_rise_{tenant}_{task_id}_{t}")

                for t in horizon:
                    self.model.addConstr(self.D_full[(tenant, task_id, t)] <= self.S_active[(tenant, task_id, t)], name=f"D_full_le_S_{tenant}_{task_id}_{t}")
                    self.model.addConstr(self.D_full[(tenant, task_id, t)] <= 1 - self.C[(tenant, task_id, t)], name=f"D_full_le_not_done_{tenant}_{task_id}_{t}")
                    self.model.addConstr(self.D_full[(tenant, task_id, t)] >= self.S_active[(tenant, task_id, t)] - self.C[(tenant, task_id, t)], name=f"D_full_ge_active_not_done_{tenant}_{task_id}_{t}")
                self.model.addConstr(gp.quicksum(self.G_start[(tenant, task_id, t)] for t in horizon) <= 1, name=f"G_start_once_{tenant}_{task_id}")

                for t in horizon:
                    self.model.addConstr(self.R[(tenant, task_id, t)] <= self.data["task_total_volume"][(tenant, task_id)] * self.S_active[(tenant, task_id, t)], name=f"R_gate_{tenant}_{task_id}_{t}")
                    self.model.addConstr(self.R[(tenant, task_id, t)] >= min_send_unit * self.S_active[(tenant, task_id, t)], name=f"R_active_lb_{tenant}_{task_id}_{t}")

                self.task_finish[(tenant, task_id)] = 1 + gp.quicksum(1 - self.C[(tenant, task_id, t)] for t in horizon)
                self.T_stage[(tenant, task_id)] = self.task_finish[(tenant, task_id)]

            for sender_rank, sender_task_ids in self.data["sender_tasks"][tenant].items():
                if not sender_task_ids:
                    continue
                sender_task_ids = list(sender_task_ids)
                for t in horizon:
                    sender_active_sum = gp.quicksum(self.S_active[(tenant, task_id, t)] for task_id in sender_task_ids)
                    for task_id in sender_task_ids:
                        self.model.addConstr(sender_active_sum >= self.Z[(tenant, task_id, t)], name=f"sender_service_lb_{tenant}_{sender_rank}_{task_id}_{t}")
                    self.model.addConstr(sender_active_sum <= 1, name=f"sender_service_ub_{tenant}_{sender_rank}_{t}")
                    for idx, task_id in enumerate(sender_task_ids):
                        for earlier_task_id in sender_task_ids[:idx]:
                            self.model.addConstr(
                                self.G_start[(tenant, task_id, t)] <= 1 - self.Z[(tenant, earlier_task_id, t)],
                                name=f"sender_fifo_start_{tenant}_{sender_rank}_{earlier_task_id}_{task_id}_{t}",
                            )

            self.tenant_finish[tenant] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"T_tenant_slot_{tenant}")
            for op_pos, op_meta in enumerate(self.data["schedule"].get(tenant, {}).get("collective_program", [])):
                op_idx = int(op_meta.get("op_idx", op_pos))
                op_key = (tenant, op_idx)
                for task_id in self.data["op_task_ids"].get(op_key, []):
                    self.model.addConstr(
                        self.op_finish_slot[op_key] >= self.task_finish[(tenant, task_id)],
                        name=f"OpFinishGe_{tenant}_{op_idx}_{task_id}",
                    )
                self.model.addConstr(
                    self.tenant_finish[tenant] >= self.op_finish_slot[op_key],
                    name=f"T_tenant_ge_op_{tenant}_{op_idx}",
                )
            self.T_m[tenant] = self.tenant_finish[tenant]

        max_rate_big_m = max(
            max(self.data["task_total_volume"].values(), default=1.0),
            max((capacities[link] * slot_duration for link in links), default=1.0),
        )
        for link in links:
            capacity_per_slot = capacities[link] * slot_duration
            link_users = [
                (tenant, task["task_id"])
                for tenant in tenants
                for task in tasks_by_tenant[tenant]
                if (tenant, task["task_id"], link) in self.W
            ]
            for t in horizon:
                self.LinkMinRate[(link, t)] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"Rmin_{link[0]}_{link[1]}_{t}")
                link_flow_expr = gp.quicksum(
                    self.F[(tenant, task["task_id"], link, t)]
                    for tenant in tenants
                    for task in tasks_by_tenant[tenant]
                    if (tenant, task["task_id"], link, t) in self.F
                )
                self.model.addConstr(link_flow_expr <= capacity_per_slot, name=f"link_cap_{link[0]}_{link[1]}_{t}")
                for tenant in tenants:
                    for task in tasks_by_tenant[tenant]:
                        task_id = task["task_id"]
                        if (tenant, task_id, link) in self.W:
                            self.model.addConstr(
                                self.R[(tenant, task_id, t)]
                                >= self.LinkMinRate[(link, t)]
                                - max_rate_big_m * (1 - self.S_active[(tenant, task_id, t)]),
                                name=f"bn_minrate_lb_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}",
                            )
                        q_key = (tenant, task_id, link, t)
                        if q_key not in self.Q_bottleneck:
                            continue
                        q_var = self.Q_bottleneck[q_key]
                        self.model.addConstr(link_flow_expr >= capacity_per_slot - capacity_per_slot * (1 - q_var), name=f"bn_saturates_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}")
                        self.model.addConstr(self.R[(tenant, task_id, t)] <= self.LinkMinRate[(link, t)] + max_rate_big_m * (1 - q_var), name=f"bn_minrate_eq_{tenant}_{task_id}_{link[0]}_{link[1]}_{t}")
                # Harmonics should only optimize collective start times. When
                # multiple tenants are simultaneously active on the same
                # directed link, do not let the MILP create a fictitious
                # per-link priority order that the simulator/runtime would not
                # have. Enforce equal per-link share among concurrent users.
                for idx, (tenant_i, task_i) in enumerate(link_users):
                    for tenant_j, task_j in link_users[idx + 1 :]:
                        if tenant_i == tenant_j:
                            continue
                        self.model.addConstr(
                            self.F[(tenant_i, task_i, link, t)] - self.F[(tenant_j, task_j, link, t)]
                            <= capacity_per_slot
                            * (
                                2
                                - self.S_active[(tenant_i, task_i, t)]
                                - self.S_active[(tenant_j, task_j, t)]
                            ),
                            name=f"link_equal_share_ub_{tenant_i}_{task_i}_{tenant_j}_{task_j}_{link[0]}_{link[1]}_{t}",
                        )
                        self.model.addConstr(
                            self.F[(tenant_j, task_j, link, t)] - self.F[(tenant_i, task_i, link, t)]
                            <= capacity_per_slot
                            * (
                                2
                                - self.S_active[(tenant_i, task_i, t)]
                                - self.S_active[(tenant_j, task_j, t)]
                            ),
                            name=f"link_equal_share_lb_{tenant_i}_{task_i}_{tenant_j}_{task_j}_{link[0]}_{link[1]}_{t}",
                        )

    def _apply_zero_offset_warm_start(self):
        # A partial MIP start is expensive here, but zero-offset variable hints
        # are still useful to help Gurobi find a first incumbent quickly in the
        # highly symmetric cases where all tenants start together.
        for key, var in self.op_offset_step.items():
            var.VarHintVal = 0.0
        for key, var in self.op_offset_slot.items():
            var.VarHintVal = 0.0
        for key, var in self.op_release_slot.items():
            var.VarHintVal = 0.0
        for key, var in self.op_finish_slot.items():
            var.VarHintVal = 0.0
        for (tenant, op_idx, start_t), var in self.op_release_choice.items():
            var.VarHintVal = 1.0 if start_t == 0 else 0.0

    def solve(self, timelimit=30, mipgap=0.0):
        self.model.update()
        if timelimit is not None:
            self.model.setParam("TimeLimit", float(timelimit))
        self.model.setParam("MIPGap", float(mipgap))
        self.model.optimize()
        if self.model.SolCount == 0:
            fallback_starts = None
            if self._enable_offset_scan_fallback:
                fallback_starts = self._fallback_scan_single_collective_offsets(
                    timelimit=max(float(timelimit or 30.0), 30.0),
                    mipgap=mipgap,
                )
                if fallback_starts is None:
                    fallback_starts = self._fallback_per_op_single_collective_offsets(
                        timelimit=max(float(timelimit or 30.0), 30.0),
                        mipgap=mipgap,
                    )
            if fallback_starts is not None:
                return fallback_starts
            zero_starts = {int(tenant): {} for tenant in self.data["M"]}
            for tenant in self.data["M"]:
                for op_pos, op_meta in enumerate(self.data["schedule"].get(tenant, {}).get("collective_program", [])):
                    op_idx = int(op_meta.get("op_idx", op_pos))
                    zero_starts[int(tenant)][int(op_idx)] = 0.0
            self.collective_start_times = zero_starts
            self.tenant_finish_values = {int(tenant): 0.0 for tenant in self.data["M"]}
            self.final_avg_jct = 0.0
            self.final_tenant_avg_jct = 0.0
            self.final_makespan = 0.0
            return self.collective_start_times

        slot_duration = float(self.data["slot_duration"])
        starts = {int(tenant): {} for tenant in self.data["M"]}
        tenant_finishes = []
        for tenant in self.data["M"]:
            for op_pos, op_meta in enumerate(self.data["schedule"].get(tenant, {}).get("collective_program", [])):
                op_idx = int(op_meta.get("op_idx", op_pos))
                starts[int(tenant)][int(op_idx)] = float(self.op_offset_slot[(tenant, op_idx)].X) * slot_duration
            finish_s = float(self.tenant_finish[tenant].X) * slot_duration
            self.tenant_finish_values[int(tenant)] = finish_s
            tenant_finishes.append(finish_s)
        self.collective_start_times = starts
        self.final_makespan = max(tenant_finishes, default=0.0)
        self.final_avg_jct = sum(tenant_finishes) / len(tenant_finishes) if tenant_finishes else 0.0
        self.final_tenant_avg_jct = self.final_avg_jct
        return starts

    def _fallback_scan_single_collective_offsets(self, timelimit=30.0, mipgap=0.0):
        programs = {
            int(tenant): list(self.data["schedule"].get(tenant, {}).get("collective_program", []))
            for tenant in self.data["M"]
        }
        if not programs or any(len(program) != 1 for program in programs.values()):
            return None

        max_steps = int(self.data.get("max_offset_steps_per_collective", 0))
        if max_steps <= 0:
            return None

        best = None
        candidate_steps = sorted(
            {
                0,
                max_steps,
                max(0, int(round(max_steps * 0.25))),
                max(0, int(round(max_steps * 0.50))),
                max(0, int(round(max_steps * 0.75))),
            }
        )
        per_try_limit = max(10.0, min(float(timelimit), 20.0))

        for delayed_tenant in sorted(self.data["M"]):
            for forced_step in candidate_steps:
                child = HarmonicsProgramILP(**self._spawn_args)
                for tenant in child.data["M"]:
                    fixed = forced_step if int(tenant) == int(delayed_tenant) else 0
                    child.model.addConstr(
                        child.op_offset_step[(tenant, 0)] == fixed,
                        name=f"scan_fix_offset_{tenant}_{fixed}",
                    )
                starts = child.solve(timelimit=per_try_limit, mipgap=mipgap)
                if child.model.SolCount == 0:
                    continue
                score = (float(child.final_avg_jct), float(child.final_makespan))
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "starts": starts,
                        "tenant_finish_values": dict(child.tenant_finish_values),
                    }

        if best is None:
            return None

        self.collective_start_times = best["starts"]
        self.tenant_finish_values = best["tenant_finish_values"]
        self.final_avg_jct = best["score"][0]
        self.final_tenant_avg_jct = best["score"][0]
        self.final_makespan = best["score"][1]
        return self.collective_start_times

    def _fallback_per_op_single_collective_offsets(self, timelimit=30.0, mipgap=0.0):
        programs = {
            int(tenant): list(self.data["schedule"].get(tenant, {}).get("collective_program", []))
            for tenant in self.data["M"]
        }
        max_ops = max((len(program) for program in programs.values()), default=0)
        if max_ops <= 1:
            return None

        starts = {int(tenant): {} for tenant in self.data["M"]}
        per_op_limit = max(10.0, min(float(timelimit) / max(max_ops, 1), 20.0))

        for op_pos in range(max_ops):
            sub_programs = {}
            for tenant, program in programs.items():
                if op_pos >= len(program):
                    continue
                op = dict(program[op_pos])
                op["gap_after"] = 0.0
                sub_programs[int(tenant)] = [op]
            if len(sub_programs) <= 1:
                for tenant in sub_programs:
                    starts[int(tenant)][int(op_pos)] = 0.0
                continue
            max_steps = max(1, int(self.data.get("max_offset_steps_per_collective", 0)))
            candidate_steps = sorted(
                {
                    0,
                    max_steps,
                    max(0, int(round(max_steps * 0.25))),
                    max(0, int(round(max_steps * 0.50))),
                    max(0, int(round(max_steps * 0.75))),
                }
            )
            best_step_by_tenant = {int(tenant): 0.0 for tenant in self.data["M"]}
            best_score = None
            for delayed_tenant in sorted(sub_programs):
                for forced_step in candidate_steps:
                    child = HarmonicsProgramILP(
                        datacenter=self._spawn_args["datacenter"],
                        tenant_mapping=self._spawn_args["tenant_mapping"],
                        tenant_flows=self._spawn_args["tenant_flows"],
                        path_table=self._spawn_args["path_table"],
                        single_flow_size=self._spawn_args["single_flow_size"],
                        collective=self._spawn_args["collective"],
                        tenant_collective_specs=self._spawn_args["tenant_collective_specs"],
                        tenant_collective_programs=sub_programs,
                        verbose=self._spawn_args["verbose"],
                        slot_duration=self._spawn_args["slot_duration"],
                        horizon_slots=self._spawn_args["horizon_slots"],
                        enable_offset_scan_fallback=False,
                    )
                    for tenant in child.data["M"]:
                        fixed = forced_step if int(tenant) == int(delayed_tenant) else 0
                        child.model.addConstr(
                            child.op_offset_step[(tenant, 0)] == fixed,
                            name=f"scan_fix_offset_{tenant}_{fixed}",
                        )
                    child.solve(timelimit=max(3.0, min(per_op_limit, 8.0)), mipgap=mipgap)
                    if child.model.SolCount == 0:
                        continue
                    score = (float(child.final_avg_jct), float(child.final_makespan))
                    if best_score is None or score < best_score:
                        best_score = score
                        best_step_by_tenant = {
                            int(tenant): float(child.collective_start_times.get(int(tenant), {}).get(0, 0.0))
                            for tenant in self.data["M"]
                        }
            for tenant in self.data["M"]:
                starts[int(tenant)][int(op_pos)] = float(best_step_by_tenant.get(int(tenant), 0.0))

        self.collective_start_times = starts
        return self.collective_start_times

    def get_collective_start_times(self):
        return self.collective_start_times

    def get_collective_rate_scales(self):
        return self.collective_rate_scales

    def get_collective_rate_schedule(self):
        return self.collective_rate_schedule

    def get_collective_scheduled_times(self):
        return self.collective_start_times

    def get_task_rate_schedule(self):
        return self.task_rate_schedule


class HarmonicsBaselineILP:
    def __init__(self, *args, **kwargs):
        self._solver = HarmonicsProgramILP(*args, **kwargs)

    def solve(self, timelimit=30, mipgap=0.0):
        starts = self._solver.solve(timelimit=timelimit, mipgap=mipgap)
        if starts is None:
            return None
        return {
            int(tenant): (0.0, float(self._solver.tenant_finish_values[int(tenant)]))
            for tenant in self._solver.M
        }

    def __getattr__(self, name):
        return getattr(self._solver, name)


class HarmonicsBaselineHeuristic:
    def __init__(
        self,
        datacenter,
        tenant_mapping,
        tenant_flows=None,
        path_table=None,
        single_flow_size=None,
        collective="allreduce",
        tenant_collective_specs=None,
        tenant_collective_programs=None,
        verbose=True,
        program_time_limit_s=30.0,
        slot_duration=None,
        **kwargs,
    ):
        self.verbose = bool(verbose)
        self.program_time_limit_s = float(program_time_limit_s)
        self.datacenter = datacenter
        self.tenant_mapping = tenant_mapping
        self.tenant_flows = tenant_flows
        self.path_table = path_table or datacenter.build_tenant_ecmp_path_table(
            sorted(int(tenant) for tenant in tenant_mapping)
        )
        self.single_flow_size = single_flow_size
        self.collective = collective
        self.tenant_collective_specs = tenant_collective_specs
        self.tenant_collective_programs = normalize_collective_programs(
            tenant_mapping,
            collective=collective,
            single_flow_size=single_flow_size,
            tenant_collective_specs=tenant_collective_specs,
            tenant_collective_programs=tenant_collective_programs,
        )
        self.slot_duration_override = slot_duration
        self.quantum_s = max(float(slot_duration or 0.0), 0.0)
        self.collective_start_times = {int(t): {} for t in tenant_mapping}
        self.collective_rate_scales = {int(t): {} for t in tenant_mapping}
        self.collective_rate_schedule = {int(t): {} for t in tenant_mapping}
        self.task_rate_schedule = {int(t): {} for t in tenant_mapping}
        self.tenant_finish_values = {int(t): 0.0 for t in tenant_mapping}
        self.final_avg_jct = 0.0
        self.final_tenant_avg_jct = 0.0
        self.final_makespan = 0.0
        self._metadata = _build_program_metadata(
            datacenter,
            tenant_mapping,
            tenant_flows,
            self.path_table,
            collective=collective,
            single_flow_size=single_flow_size,
            tenant_collective_specs=tenant_collective_specs,
            tenant_collective_programs=self.tenant_collective_programs,
            slot_duration_override=slot_duration,
        )
        self._solver = None

    def solve(self):
        starts = self._solve_single_collective_joint_offsets()
        if starts is None:
            starts = self._solve_price_guided_offsets()
        return {
            int(tenant): (0.0, float(self.tenant_finish_values[int(tenant)]))
            for tenant in self.tenant_mapping
        }

    def _is_single_collective_program(self):
        programs = self._metadata["collective_program"]
        return all(len(programs.get(int(tenant), [])) == 1 for tenant in self.tenant_mapping)

    def _candidate_delays(self):
        return [0.0]

    def _candidate_delays_for_op(self, op_meta, reference_durations=None):
        base = list(self._candidate_delays())
        durations = []
        own_duration = float(op_meta.get("duration_s", 0.0))
        if own_duration > 0.0:
            durations.append(own_duration)
        for duration in reference_durations or []:
            duration = float(duration)
            if duration > 0.0:
                durations.append(duration)

        unique_durations = sorted(
            {
                round(float(duration), 12)
                for duration in durations
                if float(duration) > 0.0
            }
        )
        for duration in unique_durations:
            base.extend(
                [
                    0.25 * duration,
                    0.5 * duration,
                    0.75 * duration,
                    duration,
                ]
            )
        cleaned = sorted(
            {
                round(float(delay), 12)
                for delay in base
                if float(delay) >= 0.0
            }
        )
        return [float(delay) for delay in cleaned]

    def _edge_capacity_gbps(self, edge):
        return float(self._metadata["capacities"].get(edge, 0.0)) / 1e9

    def _stage_rate_profile(self, op_meta):
        slot_duration = float(self._metadata["slot_duration"])
        profile = []
        for stage in op_meta.get("stages", []):
            start_s = float(stage["start_offset_slots"]) * slot_duration
            end_s = float(stage["end_offset_slots"]) * slot_duration
            edge_rates = {
                edge: float(load_gbits_per_slot) / max(slot_duration, 1e-12)
                for edge, load_gbits_per_slot in stage.get("edge_load_gbits_per_slot", {}).items()
            }
            profile.append((start_s, end_s, edge_rates))
        return profile

    def _schedule_penalty(self, candidate_stages, committed):
        penalty = 0.0
        for cand_start, cand_end, cand_rates in candidate_stages:
            for other_start, other_end, other_rates in committed:
                overlap = min(cand_end, other_end) - max(cand_start, other_start)
                if overlap <= 0.0:
                    continue
                shared_edges = set(cand_rates).intersection(other_rates)
                for edge in shared_edges:
                    cap = self._edge_capacity_gbps(edge)
                    shared_load = min(cand_rates[edge], other_rates[edge])
                    overflow = cand_rates[edge] + other_rates[edge] - cap
                    penalty += overlap * shared_load
                    if overflow > 0.0:
                        penalty += overlap * overflow * 4.0
        return float(penalty)

    def _pairwise_stage_delay_penalty(self, stages_a, stages_b):
        return float(self._schedule_penalty(stages_a, stages_b))

    def _result_tenant_makespan(self, result, tenant):
        tenant_values = result.get("tenant_makespans", {})
        if str(int(tenant)) in tenant_values:
            return float(tenant_values[str(int(tenant))])
        if int(tenant) in tenant_values:
            return float(tenant_values[int(tenant)])
        return float(result.get("avg_tenant_makespan", 0.0))

    def _single_collective_program_subset(self, tenants):
        programs = self.tenant_collective_programs or self._metadata["collective_program"]
        return {
            int(tenant): [dict(programs[int(tenant)][0])]
            for tenant in tenants
        }

    def _simulate_single_collective_subset(self, tenants):
        subset_tenants = [int(tenant) for tenant in tenants]
        subset_mapping = {
            int(tenant): self.tenant_mapping[int(tenant)]
            for tenant in subset_tenants
        }
        subset_program = self._single_collective_program_subset(subset_tenants)
        return simulate_collective_details(
            self.datacenter.topology,
            subset_mapping,
            self.path_table,
            tenant_collective_programs=subset_program,
            policy_qpid_mode="tenant",
        )

    def _standalone_single_collective_makespans(self, tenants):
        standalone = {}
        for tenant in tenants:
            result = self._simulate_single_collective_subset([int(tenant)])
            standalone[int(tenant)] = self._result_tenant_makespan(result, int(tenant))
        return standalone

    def _single_collective_hotspot_signatures(self, tenants):
        signatures = {}
        for tenant in tenants:
            op_meta = self._metadata["op_metadata"][(int(tenant), 0)]
            edge_weight = {}
            for stage in op_meta.get("stages", []):
                start_slot = int(stage.get("start_offset_slots", 0))
                end_slot = int(stage.get("end_offset_slots", start_slot))
                stage_span = max(1, end_slot - start_slot)
                for edge, load_gbits_per_slot in stage.get("edge_load_gbits_per_slot", {}).items():
                    edge_weight[edge] = edge_weight.get(edge, 0.0) + float(load_gbits_per_slot) * float(stage_span)
            signatures[int(tenant)] = edge_weight
        return signatures

    def _hotspot_overlap_score(self, signature_a, signature_b):
        if not signature_a or not signature_b:
            return 0.0
        shared_edges = set(signature_a).intersection(signature_b)
        if not shared_edges:
            return 0.0
        shared = sum(
            min(float(signature_a[edge]), float(signature_b[edge]))
            for edge in shared_edges
        )
        total_a = sum(float(v) for v in signature_a.values())
        total_b = sum(float(v) for v in signature_b.values())
        denom = max(min(total_a, total_b), 1e-12)
        return float(shared / denom)

    def _pairwise_simulated_contention_penalties(self, tenants, standalone=None):
        standalone = standalone or self._standalone_single_collective_makespans(tenants)
        hotspot_signatures = self._single_collective_hotspot_signatures(tenants)
        pair_penalties = {}
        for idx_a, tenant_a in enumerate(tenants):
            for tenant_b in tenants[idx_a + 1 :]:
                tenant_a = int(tenant_a)
                tenant_b = int(tenant_b)
                result = self._simulate_single_collective_subset([tenant_a, tenant_b])
                joint_a = self._result_tenant_makespan(result, tenant_a)
                joint_b = self._result_tenant_makespan(result, tenant_b)
                base_a = max(float(standalone[tenant_a]), 1e-12)
                base_b = max(float(standalone[tenant_b]), 1e-12)
                ratio_a = max(0.0, float(joint_a - standalone[tenant_a])) / base_a
                ratio_b = max(0.0, float(joint_b - standalone[tenant_b])) / base_b
                contention_score = max(ratio_a, ratio_b)
                corun_avg = 0.5 * (float(joint_a) + float(joint_b))
                serial_ab_avg = float(standalone[tenant_a]) + 0.5 * float(standalone[tenant_b])
                serial_ba_avg = float(standalone[tenant_b]) + 0.5 * float(standalone[tenant_a])
                best_serial_avg = min(serial_ab_avg, serial_ba_avg)
                best_serial_order = (
                    (tenant_a, tenant_b)
                    if serial_ab_avg <= serial_ba_avg
                    else (tenant_b, tenant_a)
                )
                hotspot_overlap = self._hotspot_overlap_score(
                    hotspot_signatures.get(tenant_a, {}),
                    hotspot_signatures.get(tenant_b, {}),
                )
                effective_score = float(contention_score) * float(hotspot_overlap)
                pair_penalties[(tenant_a, tenant_b)] = {
                    "score": float(contention_score),
                    "effective_score": float(effective_score),
                    "hotspot_overlap": float(hotspot_overlap),
                    "slowdown_a": float(ratio_a),
                    "slowdown_b": float(ratio_b),
                    "corun_avg": float(corun_avg),
                    "best_serial_avg": float(best_serial_avg),
                    "serial_improvement": float(corun_avg - best_serial_avg),
                    "best_serial_order": tuple(int(x) for x in best_serial_order),
                }
        return pair_penalties

    def _color_single_collective_conflict_graph(self, tenants, durations, pair_penalties):
        neighbors = {int(tenant): set() for tenant in tenants}
        weighted_degree = {int(tenant): 0.0 for tenant in tenants}
        soft_neighbors = {int(tenant): set() for tenant in tenants}
        hard_threshold = 0.05
        unilateral_hard_threshold = 0.20
        hotspot_hard_threshold = 0.25
        for (tenant_a, tenant_b), metrics in pair_penalties.items():
            score = float(metrics["score"])
            effective_score = float(metrics.get("effective_score", score))
            hotspot_overlap = float(metrics.get("hotspot_overlap", 0.0))
            slowdown_a = float(metrics["slowdown_a"])
            slowdown_b = float(metrics["slowdown_b"])
            if score <= 1e-12:
                continue
            tenant_a = int(tenant_a)
            tenant_b = int(tenant_b)
            weighted_degree[tenant_a] += effective_score
            weighted_degree[tenant_b] += effective_score
            serial_improvement = float(metrics.get("serial_improvement", 0.0))
            is_hard = (
                serial_improvement > 1e-12
                and hotspot_overlap > hotspot_hard_threshold
                and (
                    (slowdown_a > hard_threshold and slowdown_b > hard_threshold)
                    or effective_score > unilateral_hard_threshold
                )
            )
            if is_hard:
                neighbors[tenant_a].add(tenant_b)
                neighbors[tenant_b].add(tenant_a)
            else:
                soft_neighbors[tenant_a].add(tenant_b)
                soft_neighbors[tenant_b].add(tenant_a)

        coloring_order = sorted(
            [int(tenant) for tenant in tenants],
            key=lambda tenant: (
                -float(weighted_degree[int(tenant)]),
                float(durations[int(tenant)]),
                int(tenant),
            ),
        )

        colors = {}
        for tenant in coloring_order:
            forbidden = {
                int(colors[neighbor])
                for neighbor in neighbors[int(tenant)]
                if int(neighbor) in colors
            }
            color = 0
            while color in forbidden:
                color += 1
            colors[int(tenant)] = int(color)

        color_groups = {}
        for tenant, color in colors.items():
            color_groups.setdefault(int(color), []).append(int(tenant))

        # Use the shortest member as the color priority proxy so mixed-size
        # color groups do not get dragged back by one long tenant.
        color_representative = {
            int(color): min(float(durations[int(tenant)]) for tenant in members)
            for color, members in color_groups.items()
        }
        ordered_colors = sorted(
            color_groups,
            key=lambda color: (float(color_representative[int(color)]), int(color)),
        )
        color_rank = {int(color): idx for idx, color in enumerate(ordered_colors)}

        predecessors = {int(tenant): [] for tenant in tenants}
        for tenant_a in tenants:
            for tenant_b in neighbors[int(tenant_a)]:
                if int(tenant_a) >= int(tenant_b):
                    continue
                color_a = int(colors[int(tenant_a)])
                color_b = int(colors[int(tenant_b)])
                rank_a = int(color_rank[color_a])
                rank_b = int(color_rank[color_b])
                if rank_a < rank_b:
                    predecessors[int(tenant_b)].append(int(tenant_a))
                elif rank_b < rank_a:
                    predecessors[int(tenant_a)].append(int(tenant_b))

        execution_order = sorted(
            [int(tenant) for tenant in tenants],
            key=lambda tenant: (
                int(color_rank[int(colors[int(tenant)])]),
                float(durations[int(tenant)]),
                int(tenant),
            ),
        )
        return {
            "colors": colors,
            "neighbors": neighbors,
            "soft_neighbors": soft_neighbors,
            "weighted_degree": weighted_degree,
            "color_groups": color_groups,
            "ordered_colors": ordered_colors,
            "color_rank": color_rank,
            "predecessors": predecessors,
            "execution_order": execution_order,
        }

    def _op_epoch_link_loads(self, op_meta, start_slot_offset=0):
        slot_duration = float(self._metadata["slot_duration"])
        capacities = self._metadata["capacities"]
        slot_loads = {}
        for stage in op_meta.get("stages", []):
            start_slot = int(stage.get("start_offset_slots", 0)) + int(start_slot_offset)
            end_slot = int(stage.get("end_offset_slots", int(stage.get("start_offset_slots", 0)))) + int(start_slot_offset)
            edge_loads = stage.get("edge_load_gbits_per_slot", {})
            for slot in range(start_slot, end_slot):
                slot_state = slot_loads.setdefault(int(slot), {})
                for edge, load_gbits_per_slot in edge_loads.items():
                    cap_gbits_per_slot = (float(capacities.get(edge, 0.0)) / 1e9) * slot_duration
                    if cap_gbits_per_slot <= 1e-12:
                        continue
                    slot_state[edge] = slot_state.get(edge, 0.0) + (
                        float(load_gbits_per_slot) / cap_gbits_per_slot
                    )
        return slot_loads

    def _pairwise_epoch_contention_penalty(self, epoch_loads_a, epoch_loads_b):
        penalty = 0.0
        common_slots = set(epoch_loads_a).intersection(epoch_loads_b)
        for slot in common_slots:
            loads_a = epoch_loads_a[slot]
            loads_b = epoch_loads_b[slot]
            for edge in set(loads_a).intersection(loads_b):
                shared = min(float(loads_a[edge]), float(loads_b[edge]))
                overflow = max(0.0, float(loads_a[edge]) + float(loads_b[edge]) - 1.0)
                penalty += shared + 4.0 * overflow
        return float(penalty)

    def _pairwise_surrogate_contention_penalties(self, tenants, epoch_loads):
        pair_penalties = {}
        for idx_a, tenant_a in enumerate(tenants):
            for tenant_b in tenants[idx_a + 1 :]:
                tenant_a = int(tenant_a)
                tenant_b = int(tenant_b)
                pair_penalties[(tenant_a, tenant_b)] = self._pairwise_epoch_contention_penalty(
                    epoch_loads[tenant_a],
                    epoch_loads[tenant_b],
                )
        return pair_penalties

    def _contention_score_thresholds(self, tenants, pair_penalties):
        positive_scores = [
            float(score)
            for score in pair_penalties.values()
            if float(score) > 1e-12
        ]
        global_cutoff = 0.0
        if positive_scores:
            global_cutoff = max(1e-12, float(_percentile(positive_scores, 0.75) or 0.0))

        per_tenant_cutoff = {}
        for tenant in tenants:
            incident = [
                float(score)
                for (left, right), score in pair_penalties.items()
                if int(tenant) in (int(left), int(right)) and float(score) > 1e-12
            ]
            if not incident:
                per_tenant_cutoff[int(tenant)] = float("inf")
                continue
            peak = max(incident)
            per_tenant_cutoff[int(tenant)] = max(global_cutoff, 0.5 * float(peak))
        return global_cutoff, per_tenant_cutoff

    def _op_stage_windows(self, op_meta, start_time_s):
        windows = []
        for stage_start, stage_end, edge_rates in self._stage_rate_profile(op_meta):
            windows.append((start_time_s + stage_start, start_time_s + stage_end, edge_rates))
        return windows

    def _pairwise_hotspot_delay(self, predecessor_stages, successor_stages):
        best = None
        for pred_start, pred_end, pred_rates in predecessor_stages:
            for succ_start, succ_end, succ_rates in successor_stages:
                overlap = min(pred_end, succ_end) - max(pred_start, succ_start)
                if overlap <= 0.0:
                    continue
                shared_edges = set(pred_rates).intersection(succ_rates)
                if not shared_edges:
                    continue
                score = 0.0
                for edge in shared_edges:
                    cap = self._edge_capacity_gbps(edge)
                    shared_rate = min(float(pred_rates[edge]), float(succ_rates[edge]))
                    overflow = max(
                        0.0,
                        float(pred_rates[edge]) + float(succ_rates[edge]) - cap,
                    )
                    score += overlap * (shared_rate + 4.0 * overflow)
                if score <= 0.0:
                    continue
                candidate = (
                    float(score),
                    max(0.0, float(pred_end) - float(succ_start)),
                )
                if best is None or candidate > best:
                    best = candidate
        return 0.0 if best is None else float(best[1])

    def _hotspot_stage_window(self, stage_windows, shared_edges=None):
        if not stage_windows:
            return None
        best = None
        for start_s, end_s, edge_rates in stage_windows:
            if shared_edges is None:
                weight = sum(float(v) for v in edge_rates.values())
            else:
                weight = sum(float(edge_rates.get(edge, 0.0)) for edge in shared_edges)
            score = (weight, end_s - start_s)
            if best is None or score > best[0]:
                best = (score, (float(start_s), float(end_s), edge_rates))
        return None if best is None else best[1]

    def _shared_edges_with_others(self, tenant, zero_stage_profiles):
        own_edges = set()
        for _, _, edge_rates in zero_stage_profiles[int(tenant)]:
            own_edges.update(edge_rates.keys())
        other_edges = set()
        for other_tenant, windows in zero_stage_profiles.items():
            if int(other_tenant) == int(tenant):
                continue
            for _, _, edge_rates in windows:
                other_edges.update(edge_rates.keys())
        return own_edges.intersection(other_edges)

    def _contention_affected_volume_score(self, tenant, op_meta, candidate_infos, tenants):
        own_zero = candidate_infos[int(tenant)][0]["stages"]
        score = 0.0
        for other in tenants:
            if int(other) == int(tenant):
                continue
            other_zero = candidate_infos[int(other)][0]["stages"]
            for start_a, end_a, rates_a in own_zero:
                for start_b, end_b, rates_b in other_zero:
                    overlap = min(end_a, end_b) - max(start_a, start_b)
                    if overlap <= 0.0:
                        continue
                    for edge in set(rates_a).intersection(rates_b):
                        shared_rate = min(float(rates_a[edge]), float(rates_b[edge]))
                        score += overlap * shared_rate
        return float(score)

    def _single_collective_benefit_graph(self, tenants, durations, pair_penalties):
        neighbors = {int(tenant): set() for tenant in tenants}
        weights = {}
        weighted_degree = {int(tenant): 0.0 for tenant in tenants}
        priority = {}

        for (tenant_a, tenant_b), metrics in pair_penalties.items():
            benefit = max(0.0, float(metrics.get("serial_improvement", 0.0)))
            hotspot = max(0.0, float(metrics.get("hotspot_overlap", 0.0)))
            weight = float(benefit * max(hotspot, 0.0))
            if weight <= 1e-12:
                continue
            tenant_a = int(tenant_a)
            tenant_b = int(tenant_b)
            neighbors[tenant_a].add(tenant_b)
            neighbors[tenant_b].add(tenant_a)
            weights[(min(tenant_a, tenant_b), max(tenant_a, tenant_b))] = weight
            weighted_degree[tenant_a] += weight
            weighted_degree[tenant_b] += weight

        for tenant in tenants:
            tenant = int(tenant)
            priority[tenant] = float(weighted_degree[tenant] / max(float(durations[tenant]), 1e-12))

        remaining = set(int(tenant) for tenant in tenants)
        components = []
        while remaining:
            root = remaining.pop()
            stack = [root]
            component = {root}
            while stack:
                node = stack.pop()
                for neighbor in neighbors[node]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component))

        components.sort(
            key=lambda comp: (
                -sum(priority[int(tenant)] for tenant in comp),
                min(float(durations[int(tenant)]) for tenant in comp),
                len(comp),
            ),
        )
        return {
            "neighbors": neighbors,
            "weights": weights,
            "weighted_degree": weighted_degree,
            "priority": priority,
            "components": components,
        }

    def _single_collective_edge_weight(self, tenant_a, tenant_b, graph):
        key = (min(int(tenant_a), int(tenant_b)), max(int(tenant_a), int(tenant_b)))
        return float(graph["weights"].get(key, 0.0))

    def _single_collective_component_candidates(self, tenant, starts, placed, durations, graph):
        tenant = int(tenant)
        anchors = {0.0, float(starts[int(tenant)][0])}
        for neighbor in graph["neighbors"].get(tenant, set()):
            if int(neighbor) not in placed:
                continue
            neighbor_start = float(starts[int(neighbor)][0])
            neighbor_duration = float(durations[int(neighbor)])
            anchors.add(neighbor_start)
            anchors.add(neighbor_start + 0.5 * neighbor_duration)
            anchors.add(neighbor_start + neighbor_duration)
        return sorted(
            {
                round(float(anchor), 12)
                for anchor in anchors
                if float(anchor) >= 0.0
            }
        )

    def _single_collective_component_penalty(self, tenant, delay, starts, placed, durations, graph):
        tenant = int(tenant)
        own_finish = float(delay) + float(durations[int(tenant)])
        penalty = 0.0
        for other in placed:
            other = int(other)
            weight = self._single_collective_edge_weight(tenant, other, graph)
            if weight <= 1e-12:
                continue
            other_start = float(starts[int(other)][0])
            other_finish = other_start + float(durations[int(other)])
            overlap = min(own_finish, other_finish) - max(float(delay), other_start)
            if overlap <= 0.0:
                continue
            norm = overlap / max(min(float(durations[int(tenant)]), float(durations[int(other)])), 1e-12)
            penalty += weight * norm
        return float(penalty)

    def _build_single_collective_structured_schedule(self, tenants, starts, durations, graph):
        for component in graph["components"]:
            if len(component) <= 1:
                continue
            order = sorted(
                [int(tenant) for tenant in component],
                key=lambda tenant: (
                    -float(graph["priority"].get(int(tenant), 0.0)),
                    float(durations[int(tenant)]),
                    int(tenant),
                ),
            )
            placed = []
            current_component_makespan = 0.0
            for tenant in order:
                anchors = self._single_collective_component_candidates(
                    tenant,
                    starts,
                    set(placed),
                    durations,
                    graph,
                )
                best = None
                for delay in anchors:
                    finish_time = float(delay) + float(durations[int(tenant)])
                    overlap_penalty = self._single_collective_component_penalty(
                        tenant,
                        delay,
                        starts,
                        placed,
                        durations,
                        graph,
                    )
                    score = (
                        float(overlap_penalty),
                        max(float(current_component_makespan), finish_time),
                        finish_time,
                        float(delay),
                    )
                    if best is None or score < best[0]:
                        best = (score, float(delay))
                starts[int(tenant)][0] = 0.0 if best is None else float(best[1])
                placed.append(int(tenant))
                current_component_makespan = max(
                    float(current_component_makespan),
                    float(starts[int(tenant)][0]) + float(durations[int(tenant)]),
                )
        return starts

    def _refine_single_collective_structured_schedule(self, tenants, starts, durations, graph, current_score):
        candidates = sorted(
            [int(tenant) for tenant in tenants if graph["neighbors"].get(int(tenant))],
            key=lambda tenant: (
                -float(graph["priority"].get(int(tenant), 0.0)),
                float(durations[int(tenant)]),
                int(tenant),
            ),
        )
        if not candidates:
            return starts, current_score

        improved = True
        rounds = 0
        max_rounds = max(2, len(candidates))
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            best_move = None
            best_score = current_score
            for tenant in candidates:
                anchors = self._single_collective_component_candidates(
                    tenant,
                    starts,
                    set(int(t) for t in tenants if int(t) != int(tenant)),
                    durations,
                    graph,
                )
                ranked = []
                current_delay = float(starts[int(tenant)][0])
                for delay in anchors:
                    delay = float(delay)
                    if abs(delay - current_delay) <= 1e-15:
                        continue
                    trial = {
                        int(t): {0: float(starts[int(t)][0])}
                        for t in tenants
                    }
                    trial[int(tenant)][0] = delay
                    ranked.append((self._surrogate_single_collective_score(trial), trial))
                ranked.sort(key=lambda item: item[0])
                for _, trial in ranked[:3]:
                    trial_score = self._simulator_single_collective_score(trial)
                    if trial_score is None:
                        continue
                    if (trial_score[0], trial_score[1]) < (best_score[0], best_score[1]):
                        best_score = trial_score
                        best_move = trial
            if best_move is not None:
                starts = best_move
                current_score = best_score
                improved = True
        return starts, current_score

    def _solve_single_collective_joint_offsets(self):
        if not self._is_single_collective_program():
            return None

        tenants = sorted(self.tenant_mapping)
        durations = self._standalone_single_collective_makespans(tenants)
        pair_penalties = self._pairwise_simulated_contention_penalties(tenants, durations)
        coloring = self._color_single_collective_conflict_graph(
            tenants,
            durations,
            pair_penalties,
        )
        weighted_degree = coloring["weighted_degree"]

        reference_durations = [float(durations[int(tenant)]) for tenant in tenants]
        candidate_delays = {
            int(tenant): self._candidate_delays_for_op(
                self._metadata["op_metadata"][(int(tenant), 0)],
                reference_durations,
            )
            for tenant in tenants
        }
        candidate_tenants = sorted(
            tenants,
            key=lambda tenant: (
                -float(weighted_degree.get(int(tenant), 0.0)),
                float(durations[int(tenant)]),
                int(tenant),
            ),
        )

        starts = {int(tenant): {0: 0.0} for tenant in tenants}
        score = self._simulator_single_collective_score(starts)
        if score is None:
            return None

        improved = True
        max_rounds = max(2, len(tenants))
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            best_move = None
            best_score = score

            for tenant in candidate_tenants:
                tenant = int(tenant)
                current_delay = float(starts[tenant][0])
                ranked_candidates = []
                for delay in candidate_delays[tenant]:
                    delay = float(delay)
                    if abs(delay - current_delay) <= 1e-15:
                        continue
                    trial = {
                        int(t): {0: float(starts[int(t)][0])}
                        for t in tenants
                    }
                    trial[tenant][0] = delay
                    surrogate = self._surrogate_single_collective_score(trial)
                    ranked_candidates.append((surrogate, delay, trial))

                ranked_candidates.sort(key=lambda item: item[0])
                for _, delay, trial in ranked_candidates[:3]:
                    trial_score = self._simulator_single_collective_score(trial)
                    if trial_score is None:
                        continue
                    if (trial_score[0], trial_score[1]) < (best_score[0], best_score[1]):
                        best_score = trial_score
                        best_move = (tenant, delay, trial)

            if best_move is not None:
                _, _, starts = best_move
                score = best_score
                improved = True

        self.collective_start_times = starts
        self.tenant_finish_values = score[2]
        self.final_avg_jct = score[0]
        self.final_tenant_avg_jct = score[0]
        self.final_makespan = score[1]
        return self.collective_start_times

    def _single_collective_shortlist(self, candidate_infos, pair_penalty, tenants):
        shortlist = {}
        zero_choice = {int(t): 0 for t in tenants}
        for tenant in tenants:
            scored = []
            for cand_idx, cand in enumerate(candidate_infos[int(tenant)]):
                if cand_idx == 0:
                    continue
                total_penalty = 0.0
                for other in tenants:
                    if other == tenant:
                        continue
                    if int(tenant) < int(other):
                        key = (int(tenant), int(cand_idx), int(other), 0)
                    else:
                        key = (int(other), 0, int(tenant), int(cand_idx))
                    total_penalty += pair_penalty[key]
                score = (total_penalty, float(cand["finish"]))
                scored.append((score, int(cand_idx)))
            scored.sort()
            shortlist[int(tenant)] = [0] + [idx for _, idx in scored[:2]]
        return shortlist

    def _evaluate_single_collective_choice(self, choice_tuple, candidate_infos, pair_penalty, tenants):
        finishes = []
        total_penalty = 0.0
        starts = {int(t): {} for t in tenants}
        for tenant, choice_idx in zip(tenants, choice_tuple):
            cand = candidate_infos[int(tenant)][int(choice_idx)]
            finishes.append(float(cand["finish"]))
            starts[int(tenant)][0] = float(cand["delay"])
        for idx_a, tenant_a in enumerate(tenants):
            for idx_b in range(idx_a + 1, len(tenants)):
                tenant_b = tenants[idx_b]
                total_penalty += pair_penalty[
                    (
                        int(tenant_a),
                        int(choice_tuple[idx_a]),
                        int(tenant_b),
                        int(choice_tuple[idx_b]),
                    )
                ]
        avg_finish = sum(finishes) / max(len(finishes), 1)
        makespan = max(finishes, default=0.0)
        score = (total_penalty, avg_finish, makespan)
        return (score, starts, finishes)

    def _simulator_single_collective_score(self, starts):
        if self.path_table is None or self.tenant_collective_programs is None:
            return None
        details = simulate_collective_details(
            self.datacenter.topology,
            self.tenant_mapping,
            self.path_table,
            tenant_collective_programs=self.tenant_collective_programs,
            collective_start_times=starts,
            collective_rate_scales={int(tenant): {} for tenant in starts},
            collective_rate_schedule={int(tenant): {} for tenant in starts},
            task_rate_schedule={int(tenant): {} for tenant in starts},
        )
        return (
            float(details["avg_tenant_makespan"]),
            float(details["global_makespan"]),
            {int(k): float(v) for k, v in details.get("tenant_makespans", {}).items()},
        )

    def _refine_single_collective_offsets_with_simulator(self, initial_starts):
        score = self._simulator_single_collective_score(initial_starts)
        if score is None:
            return None

        tenants = sorted(initial_starts)
        current = {int(tenant): {0: float(initial_starts[int(tenant)][0])} for tenant in tenants}
        zero = {int(tenant): {0: 0.0} for tenant in tenants}
        zero_score = self._simulator_single_collective_score(zero)
        current_score = score
        reference_durations = [
            float(self._metadata["op_metadata"][(tenant, 0)].get("duration_s", 0.0))
            for tenant in tenants
        ]
        if zero_score is not None and (zero_score[0], zero_score[1]) < (current_score[0], current_score[1]):
            current = zero
            current_score = zero_score

        improved = True
        shortlist = {
            int(tenant): self._candidate_delays_for_op(
                self._metadata["op_metadata"][(tenant, 0)],
                reference_durations,
            )
            for tenant in tenants
        }
        while improved:
            improved = False
            for tenant in tenants:
                best_local = current
                best_score = current_score
                delays = shortlist[int(tenant)]
                ranked = []
                for delay in delays:
                    if abs(float(delay) - float(current[int(tenant)][0])) <= 1e-15:
                        continue
                    candidate = {
                        int(t): {0: float(current[int(t)][0])}
                        for t in tenants
                    }
                    candidate[int(tenant)][0] = float(delay)
                    cand_score = self._surrogate_single_collective_score(candidate)
                    ranked.append((cand_score, candidate))
                ranked.sort(key=lambda item: item[0])
                for _, candidate in ranked[:3]:
                    cand_score = self._simulator_single_collective_score(candidate)
                    if cand_score is None:
                        continue
                    if (cand_score[0], cand_score[1]) < (best_score[0], best_score[1]):
                        best_local = candidate
                        best_score = cand_score
                if best_local is not current:
                    current = best_local
                    current_score = best_score
                    improved = True

        self.tenant_finish_values = current_score[2]
        self.final_avg_jct = current_score[0]
        self.final_tenant_avg_jct = current_score[0]
        self.final_makespan = current_score[1]
        return current

    def _surrogate_single_collective_score(self, starts):
        tenants = sorted(starts)
        all_stages = []
        finishes = []
        total_penalty = 0.0
        for tenant in tenants:
            op_meta = self._metadata["op_metadata"][(tenant, 0)]
            delay = float(starts[int(tenant)][0])
            stage_windows = self._op_stage_windows(op_meta, delay)
            finish_time = delay + float(op_meta["duration_s"])
            finishes.append(finish_time)
            total_penalty += self._schedule_penalty(stage_windows, all_stages)
            all_stages.extend(stage_windows)
        avg_finish = sum(finishes) / max(len(finishes), 1)
        makespan = max(finishes, default=0.0)
        return (total_penalty, avg_finish, makespan)

    def _solve_price_guided_offsets(self):
        programs = self._metadata["collective_program"]
        tenants = sorted(self.tenant_mapping)
        max_ops = max((len(programs.get(t, [])) for t in tenants), default=0)
        committed_stages = []
        prev_finish = {int(t): 0.0 for t in tenants}

        for op_pos in range(max_ops):
            current_ops = []
            for tenant in tenants:
                program = list(programs.get(tenant, []))
                if op_pos >= len(program):
                    continue
                op_meta = self._metadata["op_metadata"][(tenant, op_pos)]
                current_ops.append((tenant, op_meta))

            current_ops.sort(
                key=lambda item: (
                    -sum(sum(stage[2].values()) for stage in self._op_stage_windows(item[1], 0.0)),
                    -float(item[1]["duration_s"]),
                    int(item[0]),
                )
            )

            current_round_stages = []
            for tenant, op_meta in current_ops:
                gap_after_prev = 0.0
                if op_pos > 0:
                    prev_op = programs[tenant][op_pos - 1]
                    gap_after_prev = float(prev_op.get("gap_after", 0.0))
                base_release = prev_finish[int(tenant)] + gap_after_prev

                best = None
                for delay in self._candidate_delays():
                    start_time = base_release + delay
                    stage_windows = self._op_stage_windows(op_meta, start_time)
                    penalty = self._schedule_penalty(stage_windows, committed_stages + current_round_stages)
                    finish_time = start_time + float(op_meta["duration_s"])
                    score = (
                        finish_time + 0.2 * penalty,
                        penalty,
                        finish_time,
                    )
                    if best is None or score < best[0]:
                        best = (score, delay, finish_time, stage_windows)

                _, delay, finish_time, stage_windows = best
                self.collective_start_times[int(tenant)][int(op_pos)] = float(delay)
                prev_finish[int(tenant)] = float(finish_time)
                current_round_stages.extend(stage_windows)

            committed_stages.extend(current_round_stages)

        tenant_finishes = [float(prev_finish[int(t)]) for t in tenants]
        self.tenant_finish_values = {int(t): float(prev_finish[int(t)]) for t in tenants}
        self.final_makespan = max(tenant_finishes, default=0.0)
        self.final_avg_jct = sum(tenant_finishes) / max(len(tenant_finishes), 1)
        self.final_tenant_avg_jct = self.final_avg_jct
        return self.collective_start_times

    def get_collective_start_times(self):
        return self.collective_start_times

    def get_collective_admission_order(self):
        return [
            (int(tenant), int(op_meta.get("op_idx", op_position)))
            for tenant in self.tenant_mapping
            for op_position, op_meta in enumerate(
                self._metadata["collective_program"].get(tenant, [])
            )
        ]

    def get_collective_rate_scales(self):
        return self.collective_rate_scales

    def get_collective_rate_schedule(self):
        return self.collective_rate_schedule

    def get_collective_scheduled_times(self):
        return self.collective_start_times

    def get_task_rate_schedule(self):
        return self.task_rate_schedule

    def __getattr__(self, name):
        raise AttributeError(name)


class FrontierRuntimeOracleBaseline(HarmonicsBaselineHeuristic):
    pass
