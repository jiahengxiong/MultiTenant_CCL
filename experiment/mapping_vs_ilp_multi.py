from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.config import BITS_PER_MB
from multitenant.simulator import simulate_collective_program
from multitenant.solvers import MappingHybridHeuristicSolver, MappingILPSolver
from multitenant.topology import LeafSpineDatacenter


TOPOLOGY = {
    "num_spine": 2,
    "num_leaf": 3,
    "per_leaf_server": 4,
}
TENANT_COUNTS = (2, 3, 4)
COLLECTIVE = "allgather"
SINGLE_FLOW_SIZE_BITS = 8 * BITS_PER_MB
NUM_COLLECTIVES = 3
GAP_AFTER_SECONDS = 0.1


def build_random_server_disjoint_mapping(
    datacenter: LeafSpineDatacenter,
    tenant_count: int,
    seed: int,
) -> dict[int, dict[int, int]]:
    all_servers = list(datacenter.get_all_servers())
    total_servers = len(all_servers)
    base = total_servers // tenant_count
    rem = total_servers % tenant_count
    per_tenant_sizes = [base + (1 if idx < rem else 0) for idx in range(tenant_count)]

    rng = random.Random(seed + tenant_count)
    rng.shuffle(all_servers)

    mapping: dict[int, dict[int, int]] = {}
    next_server_idx = 0
    for tenant, size in enumerate(per_tenant_sizes):
        mapping[tenant] = {}
        for rank in range(size):
            mapping[tenant][rank] = int(all_servers[next_server_idx])
            next_server_idx += 1
    return mapping


def build_programs(tenant_mapping: dict[int, dict[int, int]]) -> dict[int, list[dict[str, object]]]:
    programs: dict[int, list[dict[str, object]]] = {}
    for tenant in tenant_mapping:
        program = []
        for op_idx in range(NUM_COLLECTIVES):
            program.append(
                {
                    "collective": COLLECTIVE,
                    "single_flow_size_bits": SINGLE_FLOW_SIZE_BITS,
                    "gap_after": GAP_AFTER_SECONDS if op_idx < NUM_COLLECTIVES - 1 else 0.0,
                }
            )
        programs[int(tenant)] = program
    return programs


def evaluate_mapping(
    datacenter: LeafSpineDatacenter,
    mapping: dict[int, dict[int, int]],
    tenant_collective_programs: dict[int, list[dict[str, object]]],
) -> tuple[float, float]:
    makespan, avg_jct = simulate_collective_program(
        datacenter.topology,
        mapping,
        datacenter.paths,
        tenant_collective_programs=tenant_collective_programs,
    )
    return float(makespan), float(avg_jct)


def solve_with_heuristic(
    datacenter: LeafSpineDatacenter,
    tenant_mapping: dict[int, dict[int, int]],
    tenant_collective_programs: dict[int, list[dict[str, object]]],
    time_limit: float,
) -> tuple[dict[int, dict[int, int]], float]:
    solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=tenant_mapping,
        tenant_collective_programs=tenant_collective_programs,
        verbose=False,
    )
    start_time = time.time()
    solver.solve(time_limit=time_limit)
    runtime_seconds = time.time() - start_time
    return solver.get_X_mapping(), float(runtime_seconds)


def solve_with_ilp(
    datacenter: LeafSpineDatacenter,
    tenant_mapping: dict[int, dict[int, int]],
    tenant_collective_programs: dict[int, list[dict[str, object]]],
    time_limit: float | None,
    verbose: bool,
    warm_start_mode: str,
    warm_start_mapping: dict[int, dict[int, int]] | None = None,
) -> tuple[dict[int, dict[int, int]] | None, float, str, float | None]:
    seed_mapping = warm_start_mapping if warm_start_mapping is not None else tenant_mapping
    horizon_probe = MappingILPSolver(
        datacenter,
        tenant_mapping=tenant_mapping,
        verbose=False,
        tenant_collective_programs=tenant_collective_programs,
        build_model=False,
        enable_heuristic_warm_start=False,
        enable_full_mip_start=False,
        enable_fixed_mapping_subproblem_start=False,
    )
    default_mapping_horizon_bound = horizon_probe._mapping_horizon_bound(tenant_mapping)
    if default_mapping_horizon_bound is None:
        raise RuntimeError("Could not construct a default-mapping warm-start horizon bound.")
    default_mapping_horizon_bound = int(default_mapping_horizon_bound)
    if default_mapping_horizon_bound <= 0:
        raise RuntimeError(f"Invalid ILP warm-start horizon bound: {default_mapping_horizon_bound}")

    solver = MappingILPSolver(
        datacenter,
        tenant_mapping=tenant_mapping,
        verbose=verbose,
        tenant_collective_programs=tenant_collective_programs,
        horizon_slots=default_mapping_horizon_bound,
        compact_task_windows=True,
        compact_window_mapping=tenant_mapping,
        enable_heuristic_warm_start=False,
        enable_full_mip_start=(warm_start_mode == "fixed"),
        enable_fixed_mapping_subproblem_start=False,
    )
    solver.T_max.UB = min(float(solver.T_max.UB), float(default_mapping_horizon_bound))
    for var in solver.tenant_finish.values():
        var.UB = min(float(var.UB), float(default_mapping_horizon_bound))
    solver.model.update()
    if warm_start_mode == "fixed":
        solver._register_mapping_seed(seed_mapping)
    elif warm_start_mode == "mapping":
        solver._apply_mapping_warm_start(seed_mapping)
    elif warm_start_mode == "simulator":
        solver._apply_simulator_schedule_warm_start(seed_mapping)
    if warm_start_mode != "none":
        solver.model.Params.StartNodeLimit = 0
    start_time = time.time()
    try:
        solver.solve(time_limit=time_limit)
        runtime_seconds = time.time() - start_time
        status_code = int(solver.model.Status)
        if status_code == 2:
            status = "ok"
        elif status_code == 9 and solver.model.SolCount > 0:
            status = "time_limit_with_solution"
        else:
            status = f"status_{status_code}_with_solution"
        return solver.get_X_mapping(), float(runtime_seconds), status, default_mapping_horizon_bound
    except Exception as exc:  # pragma: no cover - experiment harness
        runtime_seconds = time.time() - start_time
        return None, float(runtime_seconds), f"error: {exc}", default_mapping_horizon_bound


def normalize_mapping(mapping: dict[int, dict[int, int]] | None) -> dict[str, dict[str, int]] | None:
    if mapping is None:
        return None
    return {
        str(tenant): {str(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in mapping.items()
    }


def summarize_case(
    tenant_count: int,
    seed: int,
    heuristic_time_limit: float,
    ilp_time_limit: float | None,
    ilp_verbose: bool,
    ilp_warm_start_mode: str,
) -> dict[str, object]:
    datacenter = LeafSpineDatacenter(
        num_leaf=TOPOLOGY["num_leaf"],
        num_spine=TOPOLOGY["num_spine"],
        per_leaf_server=TOPOLOGY["per_leaf_server"],
    )
    default_mapping = build_random_server_disjoint_mapping(datacenter, tenant_count, seed)
    tenant_collective_programs = build_programs(default_mapping)

    heuristic_mapping, heuristic_runtime = solve_with_heuristic(
        datacenter,
        default_mapping,
        tenant_collective_programs,
        time_limit=heuristic_time_limit,
    )
    heuristic_mk, heuristic_avg = evaluate_mapping(datacenter, heuristic_mapping, tenant_collective_programs)

    ilp_mapping, ilp_runtime, ilp_status, ilp_default_mapping_horizon_bound = solve_with_ilp(
        datacenter,
        default_mapping,
        tenant_collective_programs,
        time_limit=ilp_time_limit,
        verbose=ilp_verbose,
        warm_start_mode=ilp_warm_start_mode,
        warm_start_mapping=heuristic_mapping,
    )
    if ilp_mapping is not None:
        ilp_mk, ilp_avg = evaluate_mapping(datacenter, ilp_mapping, tenant_collective_programs)
    else:
        ilp_mk, ilp_avg = None, None

    return {
        "tenant_count": tenant_count,
        "tenant_collective_programs": tenant_collective_programs,
        "initial_mapping": normalize_mapping(default_mapping),
        "heuristic": {
            "runtime_seconds": heuristic_runtime,
            "runtime_definition": "solver_only",
            "mapping": normalize_mapping(heuristic_mapping),
            "avg_jct": heuristic_avg,
            "makespan": heuristic_mk,
        },
        "ilp": {
            "runtime_seconds": ilp_runtime,
            "runtime_definition": "solver_only",
            "status": ilp_status,
            "warm_start_mode": ilp_warm_start_mode,
            "default_mapping_horizon_bound": ilp_default_mapping_horizon_bound,
            "mapping": normalize_mapping(ilp_mapping),
            "avg_jct": ilp_avg,
            "makespan": ilp_mk,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare heuristic mapping and pure ILP on multi-collective small cases.",
    )
    parser.add_argument("--seed", type=int, default=20250511)
    parser.add_argument(
        "--heuristic-time-limit",
        type=float,
        default=10.0,
        help="Time limit in seconds for the heuristic solver.",
    )
    parser.add_argument(
        "--ilp-time-limit",
        type=float,
        default=120.0,
        help="Optional time limit in seconds for the pure MILP solver. Default: 120s.",
    )
    parser.add_argument(
        "--ilp-verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Gurobi solver logs for the ILP runs.",
    )
    parser.add_argument(
        "--ilp-warm-start-mode",
        choices=("none", "mapping", "fixed", "simulator"),
        default="fixed",
        help=(
            "ILP warm-start mode: none disables starts, mapping seeds only the mapping, "
            "fixed solves a fixed-mapping subproblem for a complete start, and simulator "
            "seeds mapping plus a simulator-derived full schedule."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/xiongjiaheng/COCA/MultiTenant/experiment/mapping_vs_ilp_multi.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for tenant_count in TENANT_COUNTS:
        print(f"=== tenant_count={tenant_count} ===")
        case_result = summarize_case(
            tenant_count=tenant_count,
            seed=args.seed,
            heuristic_time_limit=args.heuristic_time_limit,
            ilp_time_limit=args.ilp_time_limit,
            ilp_verbose=args.ilp_verbose,
            ilp_warm_start_mode=args.ilp_warm_start_mode,
        )
        results.append(case_result)

        heuristic_avg = case_result["heuristic"]["avg_jct"]
        heuristic_rt = case_result["heuristic"]["runtime_seconds"]
        ilp_avg = case_result["ilp"]["avg_jct"]
        ilp_rt = case_result["ilp"]["runtime_seconds"]
        ilp_status = case_result["ilp"]["status"]
        print(
            f"heuristic avg={heuristic_avg:.12f}, rt={heuristic_rt:.2f}s | "
            f"ilp avg={ilp_avg if ilp_avg is not None else 'N/A'}, rt={ilp_rt:.2f}s, status={ilp_status}"
        )

    payload = {
        "metadata": {
            "topology": TOPOLOGY,
            "collective": COLLECTIVE,
            "single_flow_size_bits": SINGLE_FLOW_SIZE_BITS,
            "num_collectives": NUM_COLLECTIVES,
            "gap_after_seconds": GAP_AFTER_SECONDS,
            "tenant_counts": list(TENANT_COUNTS),
            "seed": args.seed,
            "heuristic_time_limit": args.heuristic_time_limit,
            "ilp_time_limit": args.ilp_time_limit,
            "ilp_warm_start_mode": args.ilp_warm_start_mode,
            "ilp_default_mapping_horizon_bound": "mandatory",
            "ilp_pure": True,
            "runtime_definition": "solver_only",
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
