from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Type

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.config import BITS_PER_MB
from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingHeuristicSolver, MappingLocalSearchHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


def build_random_server_disjoint_mapping(
    datacenter: LeafSpineDatacenter,
    tenant_count: int,
    seed: int,
) -> dict[int, dict[int, int]]:
    servers = list(datacenter.get_all_servers())
    rng = random.Random(seed + tenant_count)
    rng.shuffle(servers)

    base = len(servers) // tenant_count
    rem = len(servers) % tenant_count
    mapping: dict[int, dict[int, int]] = {}
    next_server = 0
    for tenant in range(tenant_count):
        size = base + (1 if tenant < rem else 0)
        mapping[tenant] = {
            rank: int(servers[next_server + rank])
            for rank in range(size)
        }
        next_server += size
    return mapping


def evaluate(
    datacenter: LeafSpineDatacenter,
    mapping: dict[int, dict[int, int]],
    single_flow_size_bits: int,
    collective: str,
) -> tuple[float, float]:
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        datacenter.paths,
        single_flow_size_bits,
        collective,
    )
    return float(makespan), float(avg_jct)


def normalize_mapping(mapping: dict[int, dict[int, int]]) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {
            str(rank): int(server)
            for rank, server in rank_to_server.items()
        }
        for tenant, rank_to_server in mapping.items()
    }


def run_solver(
    solver_cls: Type[MappingHeuristicSolver],
    datacenter: LeafSpineDatacenter,
    initial_mapping: dict[int, dict[int, int]],
    *,
    collective: str,
    single_flow_size_bits: int,
    time_limit: float | None,
) -> dict[str, object]:
    solver = solver_cls(
        datacenter,
        initial_mapping,
        collective=collective,
        single_flow_size=single_flow_size_bits,
        verbose=False,
    )
    start = time.time()
    solver.solve(time_limit=time_limit)
    runtime = time.time() - start
    final_mapping = solver.get_X_mapping()
    makespan, avg_jct = evaluate(
        datacenter,
        final_mapping,
        single_flow_size_bits,
        collective,
    )
    return {
        "runtime_seconds": float(runtime),
        "mapping": normalize_mapping(final_mapping),
        "avg_jct": float(avg_jct),
        "makespan": float(makespan),
        "surrogate_eval_count": len(solver._surrogate_cache),
        "sim_eval_count_in_solver": len(solver._score_cache),
        "score_is_simulated": bool(solver.final_score_is_simulated),
    }


def summarize_case(
    tenant_count: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    datacenter = LeafSpineDatacenter(
        num_spine=args.num_spine,
        num_leaf=args.num_leaf,
        per_leaf_server=args.per_leaf_server,
    )
    initial_mapping = build_random_server_disjoint_mapping(
        datacenter,
        tenant_count,
        args.seed,
    )
    single_flow_size_bits = int(args.single_flow_mb * BITS_PER_MB)

    base_makespan, base_avg_jct = evaluate(
        datacenter,
        initial_mapping,
        single_flow_size_bits,
        args.collective,
    )
    bnb = run_solver(
        MappingHeuristicSolver,
        datacenter,
        initial_mapping,
        collective=args.collective,
        single_flow_size_bits=single_flow_size_bits,
        time_limit=args.bnb_time_limit,
    )
    local_search = run_solver(
        MappingLocalSearchHeuristicSolver,
        datacenter,
        initial_mapping,
        collective=args.collective,
        single_flow_size_bits=single_flow_size_bits,
        time_limit=args.local_time_limit,
    )

    for result in (bnb, local_search):
        result["avg_jct_gain_pct"] = (
            (base_avg_jct - float(result["avg_jct"])) / base_avg_jct * 100.0
            if base_avg_jct
            else 0.0
        )
        result["makespan_gain_pct"] = (
            (base_makespan - float(result["makespan"])) / base_makespan * 100.0
            if base_makespan
            else 0.0
        )

    return {
        "tenant_count": int(tenant_count),
        "initial_mapping": normalize_mapping(initial_mapping),
        "baseline": {
            "avg_jct": float(base_avg_jct),
            "makespan": float(base_makespan),
        },
        "bnb": bnb,
        "local_search": local_search,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare price-guided BnB and legacy local-search mapping heuristics.",
    )
    parser.add_argument("--num-spine", type=int, default=4)
    parser.add_argument("--num-leaf", type=int, default=8)
    parser.add_argument("--per-leaf-server", type=int, default=8)
    parser.add_argument("--tenant-counts", type=int, nargs="+", default=list(range(2, 9)))
    parser.add_argument("--collective", default="allgather")
    parser.add_argument("--single-flow-mb", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--bnb-time-limit", type=float, default=5.0)
    parser.add_argument(
        "--local-time-limit",
        type=float,
        default=None,
        help="Legacy local-search budget. Omit for natural convergence; pass a positive value to cap runtime.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("compare_bnb_vs_local_search.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = []
    for tenant_count in args.tenant_counts:
        print(f"=== tenant_count={tenant_count} ===")
        case = summarize_case(tenant_count, args)
        cases.append(case)
        bnb = case["bnb"]
        local = case["local_search"]
        print(
            "bnb: "
            f"avg={bnb['avg_jct']:.12f}, gain={bnb['avg_jct_gain_pct']:.2f}%, "
            f"mk_gain={bnb['makespan_gain_pct']:.2f}%, rt={bnb['runtime_seconds']:.2f}s"
        )
        print(
            "local: "
            f"avg={local['avg_jct']:.12f}, gain={local['avg_jct_gain_pct']:.2f}%, "
            f"mk_gain={local['makespan_gain_pct']:.2f}%, rt={local['runtime_seconds']:.2f}s"
        )

    payload = {
        "config": {
            "num_spine": args.num_spine,
            "num_leaf": args.num_leaf,
            "per_leaf_server": args.per_leaf_server,
            "tenant_counts": list(args.tenant_counts),
            "collective": args.collective,
            "single_flow_mb": args.single_flow_mb,
            "seed": args.seed,
            "bnb_time_limit": args.bnb_time_limit,
            "local_time_limit": args.local_time_limit,
            "runtime_definition": "solver_only; final metrics are simulator-evaluated",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
