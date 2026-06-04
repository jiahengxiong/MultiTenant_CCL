from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingHybridHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


DEFAULT_INPUTS = (
    REPO_ROOT / "experiment" / "Low_contension.json",
    REPO_ROOT / "experiment" / "High_contension.json",
    REPO_ROOT / "experiment" / "Low_contension_homo.json",
    REPO_ROOT / "experiment" / "High_contension_homo.json",
)


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


def evaluate_mapping(
    datacenter: LeafSpineDatacenter,
    mapping: dict[int, dict[int, int]],
    tenant_collective_specs: dict[int, dict[str, object]],
) -> tuple[float, float]:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=tenant_collective_specs,
    )
    return float(avg_jct), float(makespan)


def score_from_json(raw: dict[str, object]) -> tuple[float, float]:
    return float(raw["avg_jct"]), float(raw["makespan"])


def score_json(score: tuple[float, float]) -> dict[str, float]:
    return {"avg_jct": float(score[0]), "makespan": float(score[1])}


def improvement_against(
    baseline: tuple[float, float],
    current: tuple[float, float],
) -> dict[str, float]:
    avg_delta = float(baseline[0] - current[0])
    mk_delta = float(baseline[1] - current[1])
    return {
        "avg_jct_absolute": avg_delta,
        "avg_jct_relative": avg_delta / baseline[0] if baseline[0] > 0 else 0.0,
        "makespan_absolute": mk_delta,
        "makespan_relative": mk_delta / baseline[1] if baseline[1] > 0 else 0.0,
    }


def run_current_hybrid(
    datacenter: LeafSpineDatacenter,
    initial_mapping: dict[int, dict[int, int]],
    tenant_collective_specs: dict[int, dict[str, object]],
) -> tuple[dict[int, dict[int, int]], tuple[float, float], float]:
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=tenant_collective_specs,
        verbose=False,
        path_table=path_table,
    )
    start = time.time()
    solver.solve()
    runtime = time.time() - start
    mapping = solver.get_X_mapping()
    score = evaluate_mapping(datacenter, mapping, tenant_collective_specs)
    return mapping, score, float(runtime)


def parse_int_filter(raw: str) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def iter_tasks(path: Path, tenant_filter: set[int] | None, trial_filter: set[int] | None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    topology = payload["metadata"]["topology"]
    scene = payload.get("metadata", {}).get("scene", path.stem)
    for tenant_group in payload.get("results", []):
        tenant_count = int(tenant_group["tenant_count"])
        if tenant_filter is not None and tenant_count not in tenant_filter:
            continue
        for trial in tenant_group.get("trial_results", []):
            trial_index = int(trial["trial_index"])
            if trial_filter is not None and trial_index not in trial_filter:
                continue
            yield {
                "input": str(path),
                "scene": scene,
                "topology": topology,
                "tenant_count": tenant_count,
                "trial": trial,
            }


def run_task(task: dict[str, object]) -> dict[str, object]:
    path = Path(str(task["input"]))
    topology = task["topology"]
    tenant_count = int(task["tenant_count"])
    trial = task["trial"]
    trial_index = int(trial["trial_index"])
    datacenter = LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    json_mapping = denormalize_mapping(trial["proposed_mapping"])
    tenant_collective_specs = denormalize_specs(trial["tenant_collective_specs"])

    print(
        f"[audit-current] {path.name} tenant_count={tenant_count} trial={trial_index + 1}",
        flush=True,
    )
    json_recomputed = evaluate_mapping(datacenter, json_mapping, tenant_collective_specs)
    current_mapping, current_score, runtime = run_current_hybrid(
        datacenter,
        initial_mapping,
        tenant_collective_specs,
    )
    recorded_score = score_from_json(trial["results"]["mapping"])
    recompute_delta = {
        "avg_jct": float(json_recomputed[0] - recorded_score[0]),
        "makespan": float(json_recomputed[1] - recorded_score[1]),
    }
    improvement_vs_recorded = improvement_against(recorded_score, current_score)
    improvement_vs_recomputed = improvement_against(json_recomputed, current_score)
    print(
        "  recorded-json avg={:.6f}, mk={:.6f}; "
        "recomputed-json-map avg={:.6f}, mk={:.6f} "
        "(recomputed-recorded avg={:+.6f}, mk={:+.6f})".format(
            recorded_score[0],
            recorded_score[1],
            json_recomputed[0],
            json_recomputed[1],
            recompute_delta["avg_jct"],
            recompute_delta["makespan"],
        ),
        flush=True,
    )
    print(
        "  current avg={:.6f}, mk={:.6f}; "
        "delta vs recorded={:+.3%}/{:+.3%}; "
        "delta vs recomputed={:+.3%}/{:+.3%}".format(
            current_score[0],
            current_score[1],
            improvement_vs_recorded["avg_jct_relative"],
            improvement_vs_recorded["makespan_relative"],
            improvement_vs_recomputed["avg_jct_relative"],
            improvement_vs_recomputed["makespan_relative"],
        ),
        flush=True,
    )
    return {
        "input": str(path),
        "scene": task["scene"],
        "tenant_count": tenant_count,
        "trial_index": trial_index,
        "recorded_json": score_json(recorded_score),
        "json_mapping_recomputed": score_json(json_recomputed),
        "json_recompute_delta_from_recorded": recompute_delta,
        "current_hybrid": {
            "avg_jct": current_score[0],
            "makespan": current_score[1],
            "runtime_seconds": runtime,
            "same_mapping_as_json": current_mapping == json_mapping,
            "mapping": normalize_mapping(current_mapping),
        },
        "improvement_vs_recorded_json": improvement_vs_recorded,
        "improvement_vs_recomputed_json_mapping": improvement_vs_recomputed,
    }


def summarize(rows: list[dict[str, object]], eps: float, primary_baseline: str) -> dict[str, object]:
    def classify(v: float) -> str:
        if v > eps:
            return "improved"
        if v < -eps:
            return "declined"
        return "unchanged"

    def stats(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "mean": float(sum(values) / len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    summary: dict[str, object] = {
        "total_cases": len(rows),
        "eps": eps,
        "baselines": {},
    }
    for baseline_name, row_key in (
        ("recorded_json", "improvement_vs_recorded_json"),
        ("recomputed_json_mapping", "improvement_vs_recomputed_json_mapping"),
    ):
        baseline_summary: dict[str, object] = {"by_metric": {}}
        for metric, key in (("avg_jct", "avg_jct_relative"), ("makespan", "makespan_relative")):
            buckets = {"improved": [], "unchanged": [], "declined": []}
            for row in rows:
                rel = float(row[row_key][key])
                buckets[classify(rel)].append(rel)
            baseline_summary["by_metric"][metric] = {
                bucket: stats(vals) for bucket, vals in buckets.items()
            }
        summary["baselines"][baseline_name] = baseline_summary

    primary_row_key = {
        "recorded": "improvement_vs_recorded_json",
        "recomputed": "improvement_vs_recomputed_json_mapping",
    }[primary_baseline]
    primary_label = {
        "recorded": "recorded_json",
        "recomputed": "recomputed_json_mapping",
    }[primary_baseline]

    by_scene_tenant: dict[str, dict[str, object]] = {}
    for row in rows:
        scene = str(row["scene"])
        tenant_count = int(row["tenant_count"])
        group_key = f"{scene}/T{tenant_count}"
        by_scene_tenant.setdefault(group_key, {"rows": []})["rows"].append(row)
    groups = {}
    for group_key, group in by_scene_tenant.items():
        group_rows = group["rows"]
        avg_rels = [
            float(row[primary_row_key]["avg_jct_relative"])
            for row in group_rows
        ]
        mk_rels = [
            float(row[primary_row_key]["makespan_relative"])
            for row in group_rows
        ]
        groups[group_key] = {
            "cases": len(group_rows),
            "baseline": primary_label,
            "avg_jct_relative_mean": float(sum(avg_rels) / len(avg_rels)),
            "makespan_relative_mean": float(sum(mk_rels) / len(mk_rels)),
            "avg_jct_improved": sum(1 for v in avg_rels if v > eps),
            "avg_jct_unchanged": sum(1 for v in avg_rels if -eps <= v <= eps),
            "avg_jct_declined": sum(1 for v in avg_rels if v < -eps),
        }
    summary["by_scene_tenant"] = groups
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current default hybrid mapping with mappings recorded in main experiment JSON files.",
    )
    parser.add_argument("--inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--tenant-counts", type=str, default="")
    parser.add_argument(
        "--trials",
        type=str,
        default="",
        help="1-based trial ids, comma-separated. Empty means all trials.",
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument(
        "--primary-baseline",
        choices=("recorded", "recomputed"),
        default="recomputed",
        help=(
            "Baseline used for group summaries. 'recomputed' compares mapping quality "
            "under the current simulator; 'recorded' compares against stored JSON metrics."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "current_hybrid_vs_json_audit.json",
    )
    args = parser.parse_args()

    tenant_filter = parse_int_filter(args.tenant_counts)
    trial_filter_1_based = parse_int_filter(args.trials)
    trial_filter = {idx - 1 for idx in trial_filter_1_based} if trial_filter_1_based else None

    tasks = []
    for path in args.inputs:
        tasks.extend(iter_tasks(path, tenant_filter, trial_filter))
    if args.jobs > 1 and len(tasks) > 1:
        with Pool(processes=args.jobs, maxtasksperchild=1) as pool:
            rows = list(pool.imap_unordered(run_task, tasks))
    else:
        rows = [run_task(task) for task in tasks]
    rows.sort(key=lambda row: (str(row["scene"]), int(row["tenant_count"]), int(row["trial_index"])))
    payload = {
        "metadata": {
            "inputs": [str(path) for path in args.inputs],
            "tenant_counts": sorted(tenant_filter) if tenant_filter is not None else "all",
            "trials_1_based": sorted(trial_filter_1_based) if trial_filter_1_based else "all",
            "jobs": args.jobs,
            "eps": args.eps,
            "primary_baseline": args.primary_baseline,
            "note": "Both stored JSON metrics and current-simulator recomputation are reported. Recomputed baseline isolates mapping-quality changes when simulator/path semantics drift from stored results.",
        },
        "summary": summarize(rows, args.eps, args.primary_baseline),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved audit to {args.output}", flush=True)


if __name__ == "__main__":
    main()
