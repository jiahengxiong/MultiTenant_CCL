from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.config import BITS_PER_MB
from multitenant.simulator import simulate_collective
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


def derive_seed(base_seed: int, *components: object) -> int:
    payload = "|".join([str(base_seed), *(str(component) for component in components)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


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

    rng = random.Random(seed)
    rng.shuffle(all_servers)

    mapping: dict[int, dict[int, int]] = {}
    next_server_idx = 0
    for tenant, size in enumerate(per_tenant_sizes):
        mapping[tenant] = {}
        for rank in range(size):
            mapping[tenant][rank] = int(all_servers[next_server_idx])
            next_server_idx += 1
    return mapping


def evaluate_mapping(
    datacenter: LeafSpineDatacenter,
    mapping: dict[int, dict[int, int]],
) -> tuple[float, float]:
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        datacenter.paths,
        SINGLE_FLOW_SIZE_BITS,
        COLLECTIVE,
    )
    return float(makespan), float(avg_jct)


def solve_with_heuristic(
    datacenter: LeafSpineDatacenter,
    tenant_mapping: dict[int, dict[int, int]],
) -> tuple[dict[int, dict[int, int]], float]:
    solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=tenant_mapping,
        collective=COLLECTIVE,
        single_flow_size=SINGLE_FLOW_SIZE_BITS,
        verbose=False,
    )
    start_time = time.time()
    solver.solve()
    runtime_seconds = time.time() - start_time
    return solver.get_X_mapping(), float(runtime_seconds)


def solve_with_ilp(
    datacenter: LeafSpineDatacenter,
    tenant_mapping: dict[int, dict[int, int]],
    time_limit: float | None,
    verbose: bool,
    warm_start_mode: str,
) -> tuple[dict[int, dict[int, int]] | None, float, str]:
    solver = MappingILPSolver(
        datacenter,
        tenant_mapping=tenant_mapping,
        verbose=verbose,
        collective=COLLECTIVE,
        single_flow_size=SINGLE_FLOW_SIZE_BITS,
        enable_heuristic_warm_start=False,
        enable_full_mip_start=False,
        enable_fixed_mapping_subproblem_start=False,
    )
    if warm_start_mode == "mapping":
        solver._apply_mapping_warm_start(tenant_mapping)
    elif warm_start_mode == "simulator":
        solver._apply_mapping_warm_start(tenant_mapping)
        solver._apply_simulator_schedule_warm_start(tenant_mapping)
    start_time = time.time()
    try:
        solver.solve(time_limit=time_limit)
        runtime_seconds = time.time() - start_time
        return solver.get_X_mapping(), float(runtime_seconds), "ok"
    except Exception as exc:  # pragma: no cover - experiment harness
        runtime_seconds = time.time() - start_time
        return None, float(runtime_seconds), f"error: {exc}"


def normalize_mapping(mapping: dict[int, dict[int, int]] | None) -> dict[str, dict[str, int]] | None:
    if mapping is None:
        return None
    return {
        str(tenant): {str(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in mapping.items()
    }


def denormalize_mapping(mapping: dict[str, dict[str, int]] | None) -> dict[int, dict[int, int]] | None:
    if mapping is None:
        return None
    return {
        int(tenant): {int(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in mapping.items()
    }


def load_replay_initial_mappings(path: Path) -> dict[int, dict[int, dict[int, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    replay_cases: dict[int, dict[int, dict[int, int]]] = {}
    for case in payload.get("results", []):
        tenant_count = int(case["tenant_count"])
        replay_mapping = denormalize_mapping(case.get("initial_mapping"))
        if replay_mapping is None:
            continue
        replay_cases[tenant_count] = replay_mapping
    return replay_cases


def summarize_case(
    tenant_count: int,
    seed: int,
    ilp_time_limit: float | None,
    ilp_verbose: bool,
    ilp_warm_start_mode: str,
    initial_mapping_override: dict[int, dict[int, int]] | None = None,
) -> dict[str, object]:
    datacenter = LeafSpineDatacenter(
        num_leaf=TOPOLOGY["num_leaf"],
        num_spine=TOPOLOGY["num_spine"],
        per_leaf_server=TOPOLOGY["per_leaf_server"],
    )
    default_mapping = (
        {tenant: dict(rank_to_server) for tenant, rank_to_server in initial_mapping_override.items()}
        if initial_mapping_override is not None
        else build_random_server_disjoint_mapping(datacenter, tenant_count, seed)
    )

    heuristic_mapping, heuristic_runtime = solve_with_heuristic(
        datacenter,
        default_mapping,
    )
    heuristic_mk, heuristic_avg = evaluate_mapping(datacenter, heuristic_mapping)

    ilp_mapping, ilp_runtime, ilp_status = solve_with_ilp(
        datacenter,
        default_mapping,
        time_limit=ilp_time_limit,
        verbose=ilp_verbose,
        warm_start_mode=ilp_warm_start_mode,
    )
    if ilp_mapping is not None:
        ilp_mk, ilp_avg = evaluate_mapping(datacenter, ilp_mapping)
    else:
        ilp_mk, ilp_avg = None, None

    return {
        "tenant_count": tenant_count,
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
            "mapping": normalize_mapping(ilp_mapping),
            "avg_jct": ilp_avg,
            "makespan": ilp_mk,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare heuristic mapping and pure ILP on single-collective small cases.",
    )
    parser.add_argument("--seed", type=int, default=20250511)
    parser.add_argument(
        "--ilp-time-limit",
        type=float,
        default=None,
        help="Optional time limit in seconds for the pure MILP solver. Default: no time limit.",
    )
    parser.add_argument(
        "--ilp-verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Gurobi solver logs for the ILP runs.",
    )
    parser.add_argument(
        "--ilp-warm-start-mode",
        choices=("none", "mapping", "simulator"),
        default="mapping",
        help="ILP warm-start mode: none, mapping-only partial start, or simulator-based full start.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/xiongjiaheng/COCA/MultiTenant/experiment/mapping_vs_ilp_single.json"),
    )
    parser.add_argument(
        "--replay-input",
        type=Path,
        default=None,
        help="Optional JSON results file to replay exact initial_mapping cases from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_cases = load_replay_initial_mappings(args.replay_input) if args.replay_input is not None else {}
    results = []
    for tenant_count in TENANT_COUNTS:
        print(f"=== tenant_count={tenant_count} ===")
        case_result = summarize_case(
            tenant_count=tenant_count,
            seed=derive_seed(args.seed, "mapping_vs_ilp_single", tenant_count),
            ilp_time_limit=args.ilp_time_limit,
            ilp_verbose=args.ilp_verbose,
            ilp_warm_start_mode=args.ilp_warm_start_mode,
            initial_mapping_override=replay_cases.get(tenant_count),
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
            "tenant_counts": list(TENANT_COUNTS),
            "seed": args.seed,
            "ilp_time_limit": args.ilp_time_limit,
            "ilp_warm_start_mode": args.ilp_warm_start_mode,
            "replay_input": str(args.replay_input) if args.replay_input is not None else None,
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
