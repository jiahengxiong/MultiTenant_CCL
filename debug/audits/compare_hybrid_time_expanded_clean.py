from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.simulator import simulate_collective  # noqa: E402
from multitenant.solvers import (  # noqa: E402
    MappingEstimatorBlackBoxOptimizer,
    MappingHybridHeuristicSolver,
)
from multitenant.topology import LeafSpineDatacenter  # noqa: E402


DEFAULT_INPUTS = (
    REPO_ROOT / "experiment" / "Low_contension.json",
    REPO_ROOT / "experiment" / "High_contension.json",
    REPO_ROOT / "experiment" / "Low_contension_homo.json",
    REPO_ROOT / "experiment" / "High_contension_homo.json",
)

Mapping = dict[int, dict[int, int]]
Specs = dict[int, dict[str, Any]]
Score = tuple[float, float]  # avg_jct, makespan


def parse_int_set(raw: str) -> set[int]:
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def denormalize_mapping(raw: dict[str, dict[str, int]]) -> Mapping:
    return {
        int(tenant): {
            int(rank): int(server)
            for rank, server in rank_to_server.items()
        }
        for tenant, rank_to_server in raw.items()
    }


def normalize_mapping(mapping: Mapping) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {
            str(rank): int(server)
            for rank, server in rank_to_server.items()
        }
        for tenant, rank_to_server in mapping.items()
    }


def denormalize_specs(raw: dict[str, dict[str, Any]]) -> Specs:
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def build_datacenter(metadata: dict[str, Any]) -> LeafSpineDatacenter:
    topology = metadata["topology"]
    return LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )


def evaluate_with_simulator(
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


def run_solver(
    solver_cls,
    datacenter: LeafSpineDatacenter,
    initial_mapping: Mapping,
    specs: Specs,
    *,
    time_limit: float | None,
) -> tuple[Mapping, float, dict[str, Any]]:
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    solver = solver_cls(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    start = time.time()
    if time_limit is None:
        solver.solve()
    else:
        solver.solve(time_limit=time_limit)
    runtime = time.time() - start
    mapping = solver.get_X_mapping()
    metadata = {
        "surrogate_mode": getattr(solver, "surrogate_mode", None),
        "solver_final_avg_jct": getattr(solver, "final_avg_jct", None),
        "solver_final_makespan": getattr(solver, "final_makespan", None),
        "move_source_counts": dict(getattr(solver, "move_source_counts", {})),
        "runtime_seconds": float(runtime),
    }
    return mapping, float(runtime), metadata


def score_dict(score: Score) -> dict[str, float]:
    return {"avg_jct": float(score[0]), "makespan": float(score[1])}


def relative_gap(candidate: Score, baseline: Score) -> dict[str, float]:
    return {
        "avg_jct": (float(candidate[0]) - float(baseline[0])) / max(float(baseline[0]), 1e-12),
        "makespan": (float(candidate[1]) - float(baseline[1])) / max(float(baseline[1]), 1e-12),
    }


def mapping_diff_by_tenant(left: Mapping, right: Mapping) -> dict[int, int]:
    return {
        int(tenant): sum(
            1
            for rank, server in left[int(tenant)].items()
            if int(right[int(tenant)][int(rank)]) != int(server)
        )
        for tenant in sorted(left)
    }


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
                yield {
                    "path": str(path),
                    "metadata": metadata,
                    "tenant_count": tenant_count,
                    "trial": trial,
                }


def run_case(task: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(task["path"]))
    metadata = dict(task["metadata"])
    tenant_count = int(task["tenant_count"])
    trial = dict(task["trial"])
    trial_index = int(trial["trial_index"])
    hybrid_time_limit = task.get("hybrid_time_limit")
    time_expanded_time_limit = task.get("time_expanded_time_limit")
    sensitivity = float(task.get("sensitivity", 0.02))

    print(
        f"[clean-compare] {path.name} scene={metadata.get('scene', path.stem)} "
        f"tenant_count={tenant_count} trial={trial_index + 1}",
        flush=True,
    )

    # The JSON is used only as a scenario container.  Do not read proposed
    # mappings or recorded result fields here; both solvers are run live.
    datacenter = build_datacenter(metadata)
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    specs = denormalize_specs(trial["tenant_collective_specs"])
    initial_score = evaluate_with_simulator(datacenter, initial_mapping, specs)

    hybrid_mapping, hybrid_runtime, hybrid_meta = run_solver(
        MappingHybridHeuristicSolver,
        datacenter,
        initial_mapping,
        specs,
        time_limit=hybrid_time_limit,
    )
    hybrid_score = evaluate_with_simulator(datacenter, hybrid_mapping, specs)

    time_expanded_mapping, time_expanded_runtime, time_expanded_meta = run_solver(
        MappingEstimatorBlackBoxOptimizer,
        datacenter,
        initial_mapping,
        specs,
        time_limit=time_expanded_time_limit,
    )
    time_expanded_score = evaluate_with_simulator(datacenter, time_expanded_mapping, specs)

    gap = relative_gap(time_expanded_score, hybrid_score)
    passed = gap["avg_jct"] <= sensitivity

    print(
        "    initial={:.6f}/{:.6f}, hybrid={:.6f}/{:.6f}, "
        "time-expanded={:.6f}/{:.6f}, avg_gap={:+.3%}, pass={}".format(
            initial_score[0],
            initial_score[1],
            hybrid_score[0],
            hybrid_score[1],
            time_expanded_score[0],
            time_expanded_score[1],
            gap["avg_jct"],
            passed,
        ),
        flush=True,
    )

    return {
        "input": str(path),
        "scene": metadata.get("scene", path.stem),
        "tenant_count": tenant_count,
        "trial_index": trial_index,
        "trial_seed": int(trial.get("trial_seed", -1)),
        "scores": {
            "initial": score_dict(initial_score),
            "hybrid": score_dict(hybrid_score),
            "time_expanded": score_dict(time_expanded_score),
        },
        "relative_gap_vs_hybrid": gap,
        "passed_sensitivity": bool(passed),
        "same_mapping": bool(hybrid_mapping == time_expanded_mapping),
        "mapping_diff_by_tenant": mapping_diff_by_tenant(hybrid_mapping, time_expanded_mapping),
        "runtimes": {
            "hybrid": float(hybrid_runtime),
            "time_expanded": float(time_expanded_runtime),
        },
        "solver_metadata": {
            "hybrid": hybrid_meta,
            "time_expanded": time_expanded_meta,
        },
        "mappings": {
            "hybrid": normalize_mapping(hybrid_mapping),
            "time_expanded": normalize_mapping(time_expanded_mapping),
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"case_count": 0}
    avg_gaps = [float(row["relative_gap_vs_hybrid"]["avg_jct"]) for row in rows]
    mk_gaps = [float(row["relative_gap_vs_hybrid"]["makespan"]) for row in rows]
    failed = [row for row in rows if not bool(row["passed_sensitivity"])]
    return {
        "case_count": len(rows),
        "failed_count": len(failed),
        "max_avg_jct_gap": max(avg_gaps),
        "min_avg_jct_gap": min(avg_gaps),
        "max_makespan_gap": max(mk_gaps),
        "min_makespan_gap": min(mk_gaps),
        "failed_cases": [
            {
                "input": row["input"],
                "scene": row["scene"],
                "tenant_count": row["tenant_count"],
                "trial_index": row["trial_index"],
                "avg_jct_gap": row["relative_gap_vs_hybrid"]["avg_jct"],
                "makespan_gap": row["relative_gap_vs_hybrid"]["makespan"],
            }
            for row in failed
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean debug comparison between collapsed hybrid and the "
            "time-expanded-estimator optimizer.  The experiment JSON files are "
            "used only for scenario inputs; stored mappings/results are ignored."
        )
    )
    parser.add_argument("--inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--tenant-counts", default="3")
    parser.add_argument("--trials", default="1", help="1-based trial indices, e.g. 1,2,3")
    parser.add_argument("--hybrid-time-limit", type=float, default=None)
    parser.add_argument("--time-expanded-time-limit", type=float, default=120.0)
    parser.add_argument("--sensitivity", type=float, default=0.02)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "debug" / "audits" / "clean_hybrid_vs_time_expanded.json",
    )
    args = parser.parse_args()

    tenant_counts = parse_int_set(args.tenant_counts)
    trial_indices = {idx - 1 for idx in parse_int_set(args.trials)}
    tasks = list(iter_cases(args.inputs, tenant_counts, trial_indices))
    for task in tasks:
        task["hybrid_time_limit"] = args.hybrid_time_limit
        task["time_expanded_time_limit"] = args.time_expanded_time_limit
        task["sensitivity"] = args.sensitivity

    jobs = max(1, min(int(args.jobs), len(tasks) or 1, os.cpu_count() or 1))
    if jobs == 1:
        rows = [run_case(task) for task in tasks]
    else:
        rows = []
        with Pool(processes=jobs, maxtasksperchild=1) as pool:
            for row in pool.imap_unordered(run_case, tasks):
                rows.append(row)
        rows.sort(key=lambda row: (str(row["input"]), int(row["tenant_count"]), int(row["trial_index"])))

    output = {
        "metadata": {
            "inputs": [str(path) for path in args.inputs],
            "tenant_counts": sorted(tenant_counts),
            "trials_1_based": sorted(idx + 1 for idx in trial_indices),
            "hybrid_time_limit": args.hybrid_time_limit,
            "time_expanded_time_limit": args.time_expanded_time_limit,
            "sensitivity": float(args.sensitivity),
            "jobs": jobs,
            "note": "Experiment JSON files are used only for scenario inputs; stored mappings/results are ignored.",
        },
        "summary": summarize(rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved clean comparison to {args.output}", flush=True)


if __name__ == "__main__":
    main()
