from __future__ import annotations

import argparse
import cProfile
import json
import pstats
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "experiment" / "High_contension_homo.json")
    parser.add_argument("--tenant-count", type=int, default=3)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--sort", default="cumtime")
    parser.add_argument("--limit", type=int, default=40)
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
    profiler = cProfile.Profile()
    start = time.time()
    profiler.enable()
    solver.solve(time_limit=float(args.time_limit))
    profiler.disable()
    elapsed = time.time() - start
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(
        f"profiled time-expanded optimizer: elapsed={elapsed:.3f}s "
        f"score=({solver.final_avg_jct:.12f},{solver.final_makespan:.12f}) "
        f"maxrss_kb={rss_kb}",
        flush=True,
    )
    pstats.Stats(profiler).strip_dirs().sort_stats(args.sort).print_stats(args.limit)


if __name__ == "__main__":
    main()
