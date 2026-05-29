from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.config import BITS_PER_MB
from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingHybridHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


TOPOLOGY = {
    "num_spine": 2,
    "num_leaf": 3,
    "per_leaf_server": 4,
}
TENANT_COUNTS = (2, 3, 4)
COLLECTIVE = "allgather"
SINGLE_FLOW_SIZE_BITS = 8 * BITS_PER_MB

_WORKER_DATACENTER: LeafSpineDatacenter | None = None
_WORKER_TENANT_SERVERS: dict[int, tuple[int, ...]] | None = None


def parse_tenant_counts(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


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


def normalize_mapping(mapping: dict[int, dict[int, int]]) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {str(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in mapping.items()
    }


def canonical_server_orders(servers: Iterable[int]) -> list[tuple[int, ...]]:
    ordered = tuple(sorted(int(server) for server in servers))
    if len(ordered) <= 1:
        return [ordered]
    anchor = min(ordered)
    tail = tuple(server for server in ordered if server != anchor)
    return [(anchor, *perm) for perm in itertools.permutations(tail)]


def mapping_from_orders(
    tenant_servers: dict[int, tuple[int, ...]],
    orders_by_tenant: tuple[tuple[int, ...], ...],
) -> dict[int, dict[int, int]]:
    mapping: dict[int, dict[int, int]] = {}
    for tenant_idx, tenant in enumerate(sorted(tenant_servers)):
        order = orders_by_tenant[tenant_idx]
        mapping[int(tenant)] = {rank: int(server) for rank, server in enumerate(order)}
    return mapping


def evaluate_mapping(
    datacenter: LeafSpineDatacenter,
    mapping: dict[int, dict[int, int]],
) -> tuple[float, float]:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        SINGLE_FLOW_SIZE_BITS,
        COLLECTIVE,
    )
    return float(avg_jct), float(makespan)


def solve_with_hybrid(
    datacenter: LeafSpineDatacenter,
    initial_mapping: dict[int, dict[int, int]],
    time_limit: float | None,
) -> tuple[dict[int, dict[int, int]], tuple[float, float], float]:
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=initial_mapping,
        collective=COLLECTIVE,
        single_flow_size=SINGLE_FLOW_SIZE_BITS,
        verbose=False,
        path_table=path_table,
    )
    start = time.time()
    solver.solve(time_limit=time_limit)
    runtime = time.time() - start
    mapping = solver.get_X_mapping()
    score = evaluate_mapping(datacenter, mapping)
    return mapping, score, float(runtime)


def _init_worker(tenant_servers: dict[int, tuple[int, ...]]) -> None:
    global _WORKER_DATACENTER, _WORKER_TENANT_SERVERS
    _WORKER_DATACENTER = LeafSpineDatacenter(**TOPOLOGY)
    _WORKER_TENANT_SERVERS = tenant_servers


def _evaluate_orders(orders_by_tenant: tuple[tuple[int, ...], ...]) -> tuple[float, float, tuple[tuple[int, ...], ...]]:
    if _WORKER_DATACENTER is None or _WORKER_TENANT_SERVERS is None:
        raise RuntimeError("Worker is not initialized")
    mapping = mapping_from_orders(_WORKER_TENANT_SERVERS, orders_by_tenant)
    avg_jct, makespan = evaluate_mapping(_WORKER_DATACENTER, mapping)
    return avg_jct, makespan, orders_by_tenant


def exhaustive_search(
    initial_mapping: dict[int, dict[int, int]],
    *,
    workers: int,
    chunksize: int,
    progress_interval: int,
) -> tuple[dict[int, dict[int, int]], tuple[float, float], int, float]:
    tenant_servers = {
        int(tenant): tuple(sorted(int(server) for server in rank_to_server.values()))
        for tenant, rank_to_server in initial_mapping.items()
    }
    order_spaces = [
        canonical_server_orders(tenant_servers[tenant])
        for tenant in sorted(tenant_servers)
    ]
    total_candidates = 1
    for orders in order_spaces:
        total_candidates *= len(orders)

    start = time.time()
    best_score: tuple[float, float] | None = None
    best_orders: tuple[tuple[int, ...], ...] | None = None
    processed = 0

    order_product = itertools.product(*order_spaces)
    if workers <= 1:
        _init_worker(tenant_servers)
        for result in map(_evaluate_orders, order_product):
            avg_jct, makespan, orders_by_tenant = result
            processed += 1
            score = (float(avg_jct), float(makespan))
            if best_score is None or score < best_score:
                best_score = score
                best_orders = orders_by_tenant
            if progress_interval > 0 and processed % progress_interval == 0:
                elapsed = time.time() - start
                print(
                    f"    enumerated {processed}/{total_candidates} "
                    f"best_avg={best_score[0]:.12f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    else:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(tenant_servers,),
        ) as pool:
            for avg_jct, makespan, orders_by_tenant in pool.imap_unordered(
                _evaluate_orders,
                order_product,
                chunksize=chunksize,
            ):
                processed += 1
                score = (float(avg_jct), float(makespan))
                if best_score is None or score < best_score:
                    best_score = score
                    best_orders = orders_by_tenant
                if progress_interval > 0 and processed % progress_interval == 0:
                    elapsed = time.time() - start
                    print(
                        f"    enumerated {processed}/{total_candidates} "
                        f"best_avg={best_score[0]:.12f} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    if best_score is None or best_orders is None:
        raise RuntimeError("No candidates were evaluated")

    best_mapping = mapping_from_orders(tenant_servers, best_orders)
    return best_mapping, best_score, total_candidates, float(time.time() - start)


def summarize_case(
    tenant_count: int,
    seed: int,
    heuristic_time_limit: float | None,
    workers: int,
    chunksize: int,
    progress_interval: int,
) -> dict[str, object]:
    datacenter = LeafSpineDatacenter(**TOPOLOGY)
    initial_mapping = build_random_server_disjoint_mapping(datacenter, tenant_count, seed)

    print(f"  solving hybrid mapping for tenant_count={tenant_count}", flush=True)
    hybrid_mapping, hybrid_score, hybrid_runtime = solve_with_hybrid(
        datacenter,
        initial_mapping,
        heuristic_time_limit,
    )

    print(f"  enumerating exact dominant optimum for tenant_count={tenant_count}", flush=True)
    optimum_mapping, optimum_score, candidate_count, exhaustive_runtime = exhaustive_search(
        initial_mapping,
        workers=workers,
        chunksize=chunksize,
        progress_interval=progress_interval,
    )

    avg_gap = (
        (hybrid_score[0] - optimum_score[0]) / optimum_score[0]
        if optimum_score[0] > 0
        else 0.0
    )
    makespan_gap = (
        (hybrid_score[1] - optimum_score[1]) / optimum_score[1]
        if optimum_score[1] > 0
        else 0.0
    )

    return {
        "tenant_count": int(tenant_count),
        "candidate_count": int(candidate_count),
        "initial_mapping": normalize_mapping(initial_mapping),
        "hybrid": {
            "mapping": normalize_mapping(hybrid_mapping),
            "avg_jct": hybrid_score[0],
            "makespan": hybrid_score[1],
            "runtime_seconds": hybrid_runtime,
        },
        "optimum": {
            "mapping": normalize_mapping(optimum_mapping),
            "avg_jct": optimum_score[0],
            "makespan": optimum_score[1],
            "runtime_seconds": exhaustive_runtime,
        },
        "gap": {
            "avg_jct_relative": float(avg_gap),
            "makespan_relative": float(makespan_gap),
            "same_mapping": normalize_mapping(hybrid_mapping) == normalize_mapping(optimum_mapping),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively enumerate dominant single-collective rank mappings on the small oracle setup.",
    )
    parser.add_argument("--seed", type=int, default=20250511)
    parser.add_argument("--tenant-counts", type=str, default="2,3,4")
    parser.add_argument("--heuristic-time-limit", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/xiongjiaheng/COCA/MultiTenant/experiment/exhaustive_dominant_mapping.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tenant_counts = parse_tenant_counts(args.tenant_counts)
    results = []
    for tenant_count in tenant_counts:
        print(f"=== exhaustive_dominant tenant_count={tenant_count} ===", flush=True)
        case = summarize_case(
            tenant_count=tenant_count,
            seed=args.seed,
            heuristic_time_limit=args.heuristic_time_limit,
            workers=args.workers,
            chunksize=args.chunksize,
            progress_interval=args.progress_interval,
        )
        results.append(case)
        print(
            "hybrid avg={:.12f}, opt avg={:.12f}, avg_gap={:.6%} | "
            "hybrid mk={:.12f}, opt mk={:.12f}, mk_gap={:.6%}".format(
                case["hybrid"]["avg_jct"],
                case["optimum"]["avg_jct"],
                case["gap"]["avg_jct_relative"],
                case["hybrid"]["makespan"],
                case["optimum"]["makespan"],
                case["gap"]["makespan_relative"],
            ),
            flush=True,
        )

    payload = {
        "metadata": {
            "topology": TOPOLOGY,
            "collective": COLLECTIVE,
            "single_flow_size_bits": SINGLE_FLOW_SIZE_BITS,
            "tenant_counts": list(tenant_counts),
            "seed": args.seed,
            "heuristic_time_limit": args.heuristic_time_limit,
            "workers": args.workers,
            "enumeration": "canonical ring rotations: minimum server fixed at rank 0 for each tenant",
            "objective_order": ["avg_jct", "makespan"],
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
