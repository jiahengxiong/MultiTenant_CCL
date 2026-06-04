from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def denormalize_mapping(raw: dict[str, dict[str, int]]) -> dict[int, dict[int, int]]:
    return {
        int(tenant): {int(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in raw.items()
    }


def denormalize_specs(raw: dict[str, dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def normalize_mapping(mapping: dict[int, dict[int, int]]) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {str(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in mapping.items()
    }


def load_trial(path: Path, tenant_count: int, trial_index: int) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for group in payload["results"]:
        if int(group["tenant_count"]) != int(tenant_count):
            continue
        for trial in group["trial_results"]:
            if int(trial["trial_index"]) == int(trial_index):
                return payload["metadata"], trial
    raise ValueError(f"Cannot find tenant_count={tenant_count}, trial_index={trial_index} in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one hybrid mapping case inside one repository version.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tenant-count", type=int, required=True)
    parser.add_argument("--trial-index", type=int, required=True, help="0-based trial index")
    parser.add_argument(
        "--solve-method",
        choices=(
            "solve",
            "neighborhood",
            "portfolio",
            "hierarchical",
            "time-expanded-hybrid",
            "contention-optimizer",
            "estimator-blackbox",
            "time-expanded-blackbox",
        ),
        default="solve",
        help="Debug hook for comparing solver entry points without changing production code.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))

    from multitenant.simulator import simulate_collective
    from multitenant.solvers import MappingHybridHeuristicSolver
    from multitenant.topology import LeafSpineDatacenter

    metadata, trial = load_trial(args.input, args.tenant_count, args.trial_index)
    topology = metadata["topology"]
    datacenter = LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    tenant_collective_specs = denormalize_specs(trial["tenant_collective_specs"])
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    initial_makespan, initial_avg_jct = simulate_collective(
        datacenter.topology,
        initial_mapping,
        path_table,
        tenant_collective_specs=tenant_collective_specs,
    )

    if args.solve_method == "hierarchical":
        from multitenant.solvers import MappingHierarchicalAlignmentSolver

        solver_cls = MappingHierarchicalAlignmentSolver
    elif args.solve_method == "time-expanded-hybrid":
        from multitenant.solvers import MappingTimeExpandedHybridSolver

        solver_cls = MappingTimeExpandedHybridSolver
    elif args.solve_method == "contention-optimizer":
        from multitenant.solvers import MappingContentionOptimizer

        solver_cls = MappingContentionOptimizer
    elif args.solve_method == "estimator-blackbox":
        from multitenant.solvers import MappingEstimatorBlackBoxOptimizer

        solver_cls = MappingEstimatorBlackBoxOptimizer
    elif args.solve_method == "time-expanded-blackbox":
        from multitenant.solvers import MappingTimeExpandedBlackBoxSolver

        solver_cls = MappingTimeExpandedBlackBoxSolver
    else:
        solver_cls = MappingHybridHeuristicSolver
    solver = solver_cls(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=tenant_collective_specs,
        verbose=False,
        path_table=path_table,
    )
    start = time.time()
    if args.solve_method == "neighborhood":
        solver._solve_neighborhood_search()
    elif args.solve_method == "portfolio":
        solver._solve_time_expanded_portfolio()
    else:
        if args.time_limit is None:
            solver.solve()
        else:
            solver.solve(time_limit=args.time_limit)
    runtime = time.time() - start
    mapping = solver.get_X_mapping()
    eval_path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        eval_path_table,
        tenant_collective_specs=tenant_collective_specs,
    )

    result = {
        "repo_root": str(repo_root),
        "input": str(args.input),
        "scene": metadata.get("scene", args.input.stem),
        "tenant_count": int(args.tenant_count),
        "trial_index": int(args.trial_index),
        "avg_jct": float(avg_jct),
        "makespan": float(makespan),
        "runtime_seconds": float(runtime),
        "mapping": normalize_mapping(mapping),
        "same_mapping_as_initial": mapping == initial_mapping,
        "initial_mapping": {
            "avg_jct": float(initial_avg_jct),
            "makespan": float(initial_makespan),
            "mapping": normalize_mapping(initial_mapping),
        },
        "solver_final_avg_jct": getattr(solver, "final_avg_jct", None),
        "solver_final_makespan": getattr(solver, "final_makespan", None),
        "solver_surrogate_mode": getattr(solver, "surrogate_mode", None),
        "solve_method": args.solve_method,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
