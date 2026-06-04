from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.baselines import build_leaf_local_mapping
from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingEstimatorBlackBoxOptimizer, MappingHybridHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


DEFAULT_INPUTS = (
    REPO_ROOT / "experiment" / "Low_contension.json",
    REPO_ROOT / "experiment" / "High_contension.json",
    REPO_ROOT / "experiment" / "Low_contension_homo.json",
    REPO_ROOT / "experiment" / "High_contension_homo.json",
)


Mapping = dict[int, dict[int, int]]
Specs = dict[int, dict[str, object]]
Score = tuple[float, float]  # avg_jct, makespan


def parse_ints(raw: str) -> set[int]:
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def denormalize_mapping(raw: dict[str, dict[str, int]]) -> Mapping:
    return {
        int(tenant): {int(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in raw.items()
    }


def normalize_mapping(mapping: Mapping) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {str(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in mapping.items()
    }


def denormalize_specs(raw: dict[str, dict[str, object]]) -> Specs:
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def build_datacenter(metadata: dict[str, object]) -> LeafSpineDatacenter:
    topology = metadata["topology"]
    return LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )


def evaluate_mapping(
    datacenter: LeafSpineDatacenter,
    mapping: Mapping,
    specs: Specs,
) -> Score:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return float(avg_jct), float(makespan)


def score_dict(score: Score) -> dict[str, float]:
    return {"avg_jct": float(score[0]), "makespan": float(score[1])}


def relative_worse(candidate: Score, baseline: Score) -> dict[str, float]:
    return {
        "avg_jct": (float(candidate[0]) - float(baseline[0])) / max(float(baseline[0]), 1e-12),
        "makespan": (float(candidate[1]) - float(baseline[1])) / max(float(baseline[1]), 1e-12),
    }


def run_solver(
    solver_name: str,
    solver_factory: Callable[..., object],
    datacenter: LeafSpineDatacenter,
    initial_mapping: Mapping,
    specs: Specs,
    *,
    time_limit: float | None,
) -> tuple[Mapping, float]:
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    solver = solver_factory(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    start = time.time()
    solver.solve(time_limit=time_limit)
    runtime = time.time() - start
    mapping = solver.get_X_mapping()
    print(f"    {solver_name} runtime={runtime:.2f}s", flush=True)
    return mapping, float(runtime)


def iter_cases(paths: list[Path], tenant_counts: set[int], trial_indices: set[int]):
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        for group in payload["results"]:
            tenant_count = int(group["tenant_count"])
            if tenant_count not in tenant_counts:
                continue
            for trial in group["trial_results"]:
                trial_index = int(trial["trial_index"])
                if trial_index not in trial_indices:
                    continue
                yield path, metadata, tenant_count, trial


def run_case_task(task: dict[str, object]) -> dict[str, object]:
    return run_case(
        Path(str(task["path"])),
        dict(task["metadata"]),
        int(task["tenant_count"]),
        dict(task["trial"]),
        hybrid_time_limit=task["hybrid_time_limit"],
        time_expanded_time_limit=task["time_expanded_time_limit"],
    )


def run_case(
    path: Path,
    metadata: dict[str, object],
    tenant_count: int,
    trial: dict[str, object],
    *,
    hybrid_time_limit: float | None,
    time_expanded_time_limit: float | None,
) -> dict[str, object]:
    trial_index = int(trial["trial_index"])
    print(
        f"[compare] {path.name} scene={metadata.get('scene', path.stem)} "
        f"tenant_count={tenant_count} trial={trial_index + 1}",
        flush=True,
    )
    datacenter = build_datacenter(metadata)
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    locality_mapping = build_leaf_local_mapping(initial_mapping)
    json_mapping = denormalize_mapping(trial["proposed_mapping"])
    specs = denormalize_specs(trial["tenant_collective_specs"])

    default_score = evaluate_mapping(datacenter, initial_mapping, specs)
    locality_score = evaluate_mapping(datacenter, locality_mapping, specs)
    json_mapping_score = evaluate_mapping(datacenter, json_mapping, specs)

    hybrid_mapping, hybrid_runtime = run_solver(
        "hybrid",
        MappingHybridHeuristicSolver,
        datacenter,
        initial_mapping,
        specs,
        time_limit=hybrid_time_limit,
    )
    hybrid_score = evaluate_mapping(datacenter, hybrid_mapping, specs)

    time_expanded_mapping, time_expanded_runtime = run_solver(
        "time_expanded",
        MappingEstimatorBlackBoxOptimizer,
        datacenter,
        initial_mapping,
        specs,
        time_limit=time_expanded_time_limit,
    )
    time_expanded_score = evaluate_mapping(datacenter, time_expanded_mapping, specs)

    recorded = trial["results"]["mapping"]
    recorded_score = (float(recorded["avg_jct"]), float(recorded["makespan"]))
    methods = {
        "default": {
            "score": score_dict(default_score),
            "mapping": normalize_mapping(initial_mapping),
        },
        "locality": {
            "score": score_dict(locality_score),
            "mapping": normalize_mapping(locality_mapping),
        },
        "json_mapping": {
            "score": score_dict(json_mapping_score),
            "recorded_score": score_dict(recorded_score),
            "recorded_delta": {
                "avg_jct": float(json_mapping_score[0] - recorded_score[0]),
                "makespan": float(json_mapping_score[1] - recorded_score[1]),
            },
            "mapping": normalize_mapping(json_mapping),
        },
        "hybrid": {
            "score": score_dict(hybrid_score),
            "runtime_seconds": hybrid_runtime,
            "same_mapping_as_json": hybrid_mapping == json_mapping,
            "worse_than_json_mapping": relative_worse(hybrid_score, json_mapping_score),
            "mapping": normalize_mapping(hybrid_mapping),
        },
        "time_expanded": {
            "score": score_dict(time_expanded_score),
            "runtime_seconds": time_expanded_runtime,
            "same_mapping_as_json": time_expanded_mapping == json_mapping,
            "same_mapping_as_hybrid": time_expanded_mapping == hybrid_mapping,
            "worse_than_json_mapping": relative_worse(time_expanded_score, json_mapping_score),
            "worse_than_hybrid": relative_worse(time_expanded_score, hybrid_score),
            "mapping": normalize_mapping(time_expanded_mapping),
        },
    }
    print(
        "    scores avg/mk: default={:.6f}/{:.6f}, locality={:.6f}/{:.6f}, "
        "hybrid={:.6f}/{:.6f}, time-expanded={:.6f}/{:.6f}".format(
            default_score[0],
            default_score[1],
            locality_score[0],
            locality_score[1],
            hybrid_score[0],
            hybrid_score[1],
            time_expanded_score[0],
            time_expanded_score[1],
        ),
        flush=True,
    )
    return {
        "input": str(path),
        "scene": metadata.get("scene", path.stem),
        "tenant_count": int(tenant_count),
        "trial_index": trial_index,
        "trial_seed": int(trial.get("trial_seed", -1)),
        "task_size_rule": metadata.get("task_size_rule"),
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run collapsed hybrid and time-expanded mapping on experiment JSON "
            "cases, then evaluate default/locality/hybrid/time-expanded through "
            "one simulator path."
        )
    )
    parser.add_argument("--inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--tenant-counts", default="3")
    parser.add_argument("--trials", default="1", help="1-based trial indices")
    parser.add_argument("--hybrid-time-limit", type=float, default=None)
    parser.add_argument("--time-expanded-time-limit", type=float, default=120.0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of independent cases to run in parallel. Default: 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "debug" / "audits" / "hybrid_vs_time_expanded_runpath.json",
    )
    args = parser.parse_args()

    tenant_counts = parse_ints(args.tenant_counts)
    trial_indices = {idx - 1 for idx in parse_ints(args.trials)}
    tasks = [
        {
            "path": str(path),
            "metadata": metadata,
            "tenant_count": int(tenant_count),
            "trial": trial,
            "hybrid_time_limit": args.hybrid_time_limit,
            "time_expanded_time_limit": args.time_expanded_time_limit,
        }
        for path, metadata, tenant_count, trial in iter_cases(args.inputs, tenant_counts, trial_indices)
    ]
    jobs = max(1, min(int(args.jobs), len(tasks) or 1, os.cpu_count() or 1))
    if jobs == 1:
        rows = [run_case_task(task) for task in tasks]
    else:
        rows = []
        with Pool(processes=jobs, maxtasksperchild=1) as pool:
            for row in pool.imap_unordered(run_case_task, tasks):
                rows.append(row)
        rows.sort(key=lambda row: (str(row["input"]), int(row["tenant_count"]), int(row["trial_index"])))
    output = {
        "metadata": {
            "inputs": [str(path) for path in args.inputs],
            "tenant_counts": sorted(tenant_counts),
            "trials_1_based": sorted(idx + 1 for idx in trial_indices),
            "hybrid_time_limit": args.hybrid_time_limit,
            "time_expanded_time_limit": args.time_expanded_time_limit,
            "jobs": jobs,
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved comparison to {args.output}", flush=True)


if __name__ == "__main__":
    main()
