from __future__ import annotations

import argparse
from collections import defaultdict
import functools
import json
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.solvers import MappingEstimatorBlackBoxOptimizer  # noqa: E402
from multitenant.topology import LeafSpineDatacenter  # noqa: E402


def _parse_mapping(raw):
    return {
        int(tenant): {int(rank): int(server) for rank, server in ranks.items()}
        for tenant, ranks in raw.items()
    }


def _parse_specs(raw):
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def _build_datacenter(metadata):
    topology = metadata["topology"]
    return LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )


def _load_case(path: Path, tenant_count: int, trial_1_based: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for group in payload["results"]:
        if int(group["tenant_count"]) != int(tenant_count):
            continue
        for trial in group["trial_results"]:
            if int(trial["trial_index"]) == int(trial_1_based) - 1:
                return payload["metadata"], trial
    raise ValueError(f"case not found: {path} tenant={tenant_count} trial={trial_1_based}")


def _install_timers(solver, method_names):
    stats = defaultdict(lambda: {"calls": 0, "time": 0.0})

    for name in method_names:
        original = getattr(solver, name, None)
        if original is None:
            continue

        @functools.wraps(original)
        def wrapper(*args, __name=name, __original=original, **kwargs):
            start = time.perf_counter()
            try:
                return __original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                stats[__name]["calls"] += 1
                stats[__name]["time"] += elapsed

        setattr(solver, name, wrapper)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "experiment" / "High_contension_homo.json")
    parser.add_argument("--tenant-count", type=int, default=3)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=20.0)
    args = parser.parse_args()

    metadata, trial = _load_case(args.input, args.tenant_count, args.trial)
    datacenter = _build_datacenter(metadata)
    initial_mapping = _parse_mapping(trial["initial_mapping"])
    specs = _parse_specs(trial["tenant_collective_specs"])
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    solver = MappingEstimatorBlackBoxOptimizer(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    stats = _install_timers(
        solver,
        [
            "_evaluate_mapping",
            "_score_components",
            "_pipeline_score",
            "_analyze_mapping",
            "_time_expanded_epoch_prices",
            "_coarse_local_candidate",
            "_epoch_price_local_candidate",
            "_cached_pair_epoch_prices",
            "_accelerated_coarse_local_descent",
            "_coarse_reassignment_delta",
            "_rank_reassignment_price_delta",
            "_task_pair_prices",
            "_price_guided_remap_pool",
            "_best_neighborhood_candidate",
            "_scored_swap_pairs",
            "_joint_pair_candidate",
            "_tenant_block_recombination_candidates",
            "_tabu_swap_refinement",
        ],
    )

    start = time.time()
    solver.solve(time_limit=float(args.time_limit))
    elapsed = time.time() - start
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"elapsed={elapsed:.3f}s score=({solver.final_avg_jct:.12f},{solver.final_makespan:.12f}) "
        f"maxrss_mb={rss_bytes / (1024.0 * 1024.0):.1f}",
        flush=True,
    )
    for name, item in sorted(stats.items(), key=lambda pair: pair[1]["time"], reverse=True):
        calls = int(item["calls"])
        total = float(item["time"])
        print(f"{name:40s} calls={calls:7d} total={total:10.3f}s avg={total / max(calls, 1):.6f}s")


if __name__ == "__main__":
    main()
