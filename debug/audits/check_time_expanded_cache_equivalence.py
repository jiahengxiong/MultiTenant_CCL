from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.simulator import simulate_collective  # noqa: E402
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


def _solve(metadata, trial, *, time_limit: float, disable_caches: bool):
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
    if disable_caches:
        for key in list(getattr(solver, "_cache_limits", {})):
            solver._cache_limits[key] = 0
    start = time.time()
    solver.solve(time_limit=float(time_limit))
    runtime = time.time() - start
    mapping = solver.get_X_mapping()
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        datacenter.build_tenant_ecmp_path_table(mapping),
        tenant_collective_specs=specs,
    )
    return {
        "mapping": mapping,
        "estimator": {
            "avg_jct": float(solver.final_avg_jct),
            "makespan": float(solver.final_makespan),
        },
        "simulator": {
            "avg_jct": float(avg_jct),
            "makespan": float(makespan),
        },
        "runtime": float(runtime),
        "rounds": int(getattr(solver, "search_rounds", -1)),
        "moves": dict(getattr(solver, "move_source_counts", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tenant-count", type=int, required=True)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata, trial = _load_case(args.input, args.tenant_count, args.trial)
    cached = _solve(metadata, trial, time_limit=args.time_limit, disable_caches=False)
    uncached = _solve(metadata, trial, time_limit=args.time_limit, disable_caches=True)
    result = {
        "input": str(args.input),
        "tenant_count": int(args.tenant_count),
        "trial": int(args.trial),
        "time_limit": float(args.time_limit),
        "same_mapping": bool(cached["mapping"] == uncached["mapping"]),
        "cached": cached,
        "uncached": uncached,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "same_mapping": result["same_mapping"],
        "cached_estimator": cached["estimator"],
        "uncached_estimator": uncached["estimator"],
        "cached_simulator": cached["simulator"],
        "uncached_simulator": uncached["simulator"],
        "cached_runtime": cached["runtime"],
        "uncached_runtime": uncached["runtime"],
    }, indent=2))


if __name__ == "__main__":
    main()
