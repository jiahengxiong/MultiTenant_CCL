from __future__ import annotations

import sys
from pathlib import Path
import math

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from multitenant.baselines import HarmonicsProgramILP
from multitenant.simulator import simulate_collective
from multitenant.topology import LeafSpineDatacenter


def _pct_change(new_value: float, old_value: float) -> float:
    if abs(old_value) <= 1e-18:
        return 0.0
    return (new_value - old_value) / old_value * 100.0


def _run_case(name: str, mapping, program, *, slot_duration: float, timelimit: float):
    dc = LeafSpineDatacenter(3, 2, 2)
    path_table = dc.build_tenant_ecmp_path_table(mapping)

    default_ms, default_avg = simulate_collective(
        dc.topology,
        mapping,
        path_table,
        tenant_collective_programs=program,
    )

    solver = HarmonicsProgramILP(
        dc,
        mapping,
        None,
        path_table,
        tenant_collective_programs=program,
        verbose=False,
        slot_duration=slot_duration,
    )
    starts = solver.solve(timelimit=timelimit, mipgap=0.0)
    if starts is None:
        raise RuntimeError(f"{name}: MILP failed to produce a schedule")

    harmonics_ms, harmonics_avg = simulate_collective(
        dc.topology,
        mapping,
        path_table,
        tenant_collective_programs=program,
        collective_start_times=starts,
        collective_rate_scales=solver.get_collective_rate_scales(),
        collective_rate_schedule=solver.get_collective_rate_schedule(),
        task_rate_schedule=solver.get_task_rate_schedule(),
    )

    ms_delta_pct = _pct_change(harmonics_ms, default_ms)
    avg_delta_pct = _pct_change(harmonics_avg, default_avg)

    return {
        "name": name,
        "default": (default_ms, default_avg),
        "starts": starts,
        "milp": (solver.final_makespan, solver.final_avg_jct),
        "sim": (harmonics_ms, harmonics_avg),
        "delta_pct": (ms_delta_pct, avg_delta_pct),
    }


def _run_zero_offset_case(name: str, mapping, program, *, slot_duration: float, timelimit: float):
    dc = LeafSpineDatacenter(3, 2, 2)
    path_table = dc.build_tenant_ecmp_path_table(mapping)
    solver = HarmonicsProgramILP(
        dc,
        mapping,
        None,
        path_table,
        tenant_collective_programs=program,
        verbose=False,
        slot_duration=slot_duration,
        enable_full_mip_warm_start=False,
    )

    for tenant, ops in solver.data["collective_program"].items():
        op_by_idx = {int(op["op_idx"]): op for op in ops}
        solver.model.addConstr(solver.op_release_slot[(tenant, 0)] == 0)
        for op in ops:
            op_idx = int(op["op_idx"])
            if op_idx == 0:
                continue
            prev_op = op_by_idx[op_idx - 1]
            gap_slots = int(math.ceil(float(prev_op.get("gap_after", 0.0)) / solver.data["slot_duration"]))
            solver.model.addConstr(
                solver.op_release_slot[(tenant, op_idx)]
                == solver.op_finish_slot[(tenant, op_idx - 1)] + gap_slots
            )
    solver.model.update()
    starts = solver.solve(timelimit=timelimit, mipgap=0.0)
    if starts is None:
        raise RuntimeError(f"{name}: zero-offset MILP failed to produce a schedule")
    return {
        "starts": starts,
        "milp": (solver.final_makespan, solver.final_avg_jct),
    }


def _validate_against_zero_offset(name: str, optimized_result, zero_offset_result, tol: float = 1e-9):
    opt_ms, opt_avg = optimized_result["milp"]
    zero_ms, zero_avg = zero_offset_result["milp"]
    if opt_avg > zero_avg + tol:
        raise AssertionError(
            f"{name}: optimized MILP avg_jct {opt_avg:.9f} is worse than zero-offset baseline {zero_avg:.9f}"
        )
    if abs(opt_avg - zero_avg) <= tol and opt_ms > zero_ms + tol:
        raise AssertionError(
            f"{name}: optimized MILP makespan {opt_ms:.9f} is worse than zero-offset baseline {zero_ms:.9f} under same avg_jct"
        )


def main():
    cases = [
        {
            "name": "single_collective_positive_case",
            "mapping": {
                0: {0: 2, 1: 1},
                1: {0: 0, 1: 3},
                2: {0: 5, 1: 4},
            },
            "program": {
                tenant: [
                    {
                        "collective": "allreduce",
                        "single_flow_size_bits": 8 * 8 * 1024 * 1024,
                        "gap_after": 0.0,
                    }
                ]
                for tenant in range(3)
            },
            "slot_duration": 0.001,
            "timelimit": 120.0,
        },
        {
            "name": "multi_collective_positive_case",
            "mapping": {
                0: {0: 0, 1: 3},
                1: {0: 4, 1: 5},
                2: {0: 1, 1: 2},
            },
            "program": {
                tenant: [
                    {
                        "collective": "allreduce",
                        "single_flow_size_bits": 8 * 8 * 1024 * 1024,
                        "gap_after": 1e-3,
                    },
                    {
                        "collective": "allgather",
                        "single_flow_size_bits": 4 * 8 * 1024 * 1024,
                        "gap_after": 0.0,
                    },
                ]
                for tenant in range(3)
            },
            "slot_duration": 0.001,
            "timelimit": 180.0,
        },
        {
            "name": "multi_collective_recovered_case",
            "mapping": {
                0: {0: 2, 1: 0},
                1: {0: 5, 1: 3},
                2: {0: 4, 1: 1},
            },
            "program": {
                tenant: [
                    {
                        "collective": "allreduce",
                        "single_flow_size_bits": 8 * 8 * 1024 * 1024,
                        "gap_after": 1e-3,
                    },
                    {
                        "collective": "allgather",
                        "single_flow_size_bits": 4 * 8 * 1024 * 1024,
                        "gap_after": 0.0,
                    },
                ]
                for tenant in range(3)
            },
            "slot_duration": 0.001,
            "timelimit": 180.0,
        },
    ]

    print("=== Harmonics MILP Regression ===")
    for case in cases:
        result = _run_case(
            case["name"],
            case["mapping"],
            case["program"],
            slot_duration=case["slot_duration"],
            timelimit=case["timelimit"],
        )
        zero_offset_result = _run_zero_offset_case(
            case["name"],
            case["mapping"],
            case["program"],
            slot_duration=case["slot_duration"],
            timelimit=case["timelimit"],
        )
        _validate_against_zero_offset(case["name"], result, zero_offset_result)

        default_ms, default_avg = result["default"]
        milp_ms, milp_avg = result["milp"]
        sim_ms, sim_avg = result["sim"]
        ms_delta_pct, avg_delta_pct = result["delta_pct"]
        print()
        print(result["name"])
        print(f"  starts={result['starts']}")
        print(f"  default      makespan={default_ms:.9f}  avg_jct={default_avg:.9f}")
        print(f"  milp         makespan={milp_ms:.9f}  avg_jct={milp_avg:.9f}")
        print(
            "  zero-offset  makespan="
            f"{zero_offset_result['milp'][0]:.9f}  avg_jct={zero_offset_result['milp'][1]:.9f}"
        )
        print(f"  simulator    makespan={sim_ms:.9f}  avg_jct={sim_avg:.9f}")
        print(f"  delta        makespan={ms_delta_pct:+.2f}%  avg_jct={avg_delta_pct:+.2f}%")

    print()
    print("All Harmonics MILP regression checks passed.")


if __name__ == "__main__":
    main()
