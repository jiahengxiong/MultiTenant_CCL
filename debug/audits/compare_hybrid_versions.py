from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    REPO_ROOT / "experiment" / "Low_contension.json",
    REPO_ROOT / "experiment" / "High_contension.json",
    REPO_ROOT / "experiment" / "Low_contension_homo.json",
    REPO_ROOT / "experiment" / "High_contension_homo.json",
)


def parse_int_filter(raw: str) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def iter_cases(paths: list[Path], tenant_filter: set[int] | None, trial_filter: set[int] | None):
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scene = payload.get("metadata", {}).get("scene", path.stem)
        for group in payload.get("results", []):
            tenant_count = int(group["tenant_count"])
            if tenant_filter is not None and tenant_count not in tenant_filter:
                continue
            for trial in group.get("trial_results", []):
                trial_index = int(trial["trial_index"])
                if trial_filter is not None and trial_index not in trial_filter:
                    continue
                yield {
                    "input": path,
                    "scene": scene,
                    "tenant_count": tenant_count,
                    "trial_index": trial_index,
                }


def run_solver(
    repo_root: Path,
    case: dict[str, object],
    output_path: Path,
    solve_method: str,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "debug" / "audits" / "solve_case_once.py"),
        "--repo-root",
        str(repo_root),
        "--input",
        str(case["input"]),
        "--tenant-count",
        str(case["tenant_count"]),
        "--trial-index",
        str(case["trial_index"]),
        "--solve-method",
        solve_method,
        "--output",
        str(output_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)
    return json.loads(output_path.read_text(encoding="utf-8"))


def improvement_against(baseline: dict[str, object], current: dict[str, object]) -> dict[str, float]:
    base_avg = float(baseline["avg_jct"])
    base_mk = float(baseline["makespan"])
    curr_avg = float(current["avg_jct"])
    curr_mk = float(current["makespan"])
    avg_delta = base_avg - curr_avg
    mk_delta = base_mk - curr_mk
    return {
        "avg_jct_absolute": float(avg_delta),
        "avg_jct_relative": float(avg_delta / base_avg) if base_avg > 0 else 0.0,
        "makespan_absolute": float(mk_delta),
        "makespan_relative": float(mk_delta / base_mk) if base_mk > 0 else 0.0,
    }


def summarize(rows: list[dict[str, object]], eps: float) -> dict[str, object]:
    def classify(value: float) -> str:
        if value > eps:
            return "improved"
        if value < -eps:
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

    summary: dict[str, object] = {"total_cases": len(rows), "eps": eps, "by_metric": {}}
    for metric, key in (("avg_jct", "avg_jct_relative"), ("makespan", "makespan_relative")):
        buckets = {"improved": [], "unchanged": [], "declined": []}
        for row in rows:
            rel = float(row["improvement_vs_baseline_hybrid"][key])
            buckets[classify(rel)].append(rel)
        summary["by_metric"][metric] = {bucket: stats(vals) for bucket, vals in buckets.items()}

    runtimes = {
        "baseline_seconds": [float(row["baseline_hybrid"]["runtime_seconds"]) for row in rows],
        "current_seconds": [float(row["current_time_expanded"]["runtime_seconds"]) for row in rows],
    }
    summary["runtime"] = {name: stats(values) for name, values in runtimes.items()}
    return summary


def run_case_pair(payload: dict[str, object]) -> dict[str, object]:
    idx = int(payload["idx"])
    total = int(payload["total"])
    case = payload["case"]
    baseline_root = Path(str(payload["baseline_root"]))
    current_root = Path(str(payload["current_root"]))
    tmp_root = Path(str(payload["tmp_root"]))
    baseline_solve_method = str(payload["baseline_solve_method"])
    current_solve_method = str(payload["current_solve_method"])

    case_label = f"{case['scene']} T{case['tenant_count']} trial={int(case['trial_index']) + 1}"
    print(f"[compare] {idx}/{total} {case_label}", flush=True)
    baseline = run_solver(
        baseline_root,
        case,
        tmp_root / f"{idx}_baseline.json",
        baseline_solve_method,
    )
    current = run_solver(
        current_root,
        case,
        tmp_root / f"{idx}_current.json",
        current_solve_method,
    )
    improvement = improvement_against(baseline, current)
    print(
        "  default avg={:.6f}, mk={:.6f}; "
        "baseline-hybrid({}/{}) avg={:.6f}, mk={:.6f}, rt={:.2f}s, same_default={}; "
        "time-expanded({}/{}) avg={:.6f}, mk={:.6f}, rt={:.2f}s, same_default={}; "
        "current-vs-baseline={:+.3%}/{:+.3%}".format(
            float(baseline["initial_mapping"]["avg_jct"]),
            float(baseline["initial_mapping"]["makespan"]),
            str(baseline.get("solver_surrogate_mode")),
            str(baseline.get("solve_method")),
            float(baseline["avg_jct"]),
            float(baseline["makespan"]),
            float(baseline["runtime_seconds"]),
            bool(baseline.get("same_mapping_as_initial")),
            str(current.get("solver_surrogate_mode")),
            str(current.get("solve_method")),
            float(current["avg_jct"]),
            float(current["makespan"]),
            float(current["runtime_seconds"]),
            bool(current.get("same_mapping_as_initial")),
            improvement["avg_jct_relative"],
            improvement["makespan_relative"],
        ),
        flush=True,
    )
    return {
        "scene": case["scene"],
        "input": str(case["input"]),
        "tenant_count": int(case["tenant_count"]),
        "trial_index": int(case["trial_index"]),
        "baseline_hybrid": baseline,
        "current_time_expanded": current,
        "same_mapping": baseline["mapping"] == current["mapping"],
        "improvement_vs_baseline_hybrid": improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the latest committed hybrid mapping algorithm with the current time-expanded variant.",
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--tenant-counts", type=str, default="")
    parser.add_argument("--trials", type=str, default="", help="1-based trial ids, comma-separated")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--baseline-solve-method",
        choices=(
            "solve",
            "neighborhood",
            "portfolio",
            "hierarchical",
            "time-expanded-hybrid",
            "contention-optimizer",
            "estimator-blackbox",
        ),
        default="solve",
    )
    parser.add_argument(
        "--current-solve-method",
        choices=(
            "solve",
            "neighborhood",
            "portfolio",
            "hierarchical",
            "time-expanded-hybrid",
            "contention-optimizer",
            "estimator-blackbox",
        ),
        default="solve",
    )
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "hybrid_version_comparison.json",
    )
    args = parser.parse_args()

    tenant_filter = parse_int_filter(args.tenant_counts)
    trial_filter_1_based = parse_int_filter(args.trials)
    trial_filter = {trial - 1 for trial in trial_filter_1_based} if trial_filter_1_based else None
    cases = list(iter_cases(args.inputs, tenant_filter, trial_filter))

    rows = []
    with tempfile.TemporaryDirectory(prefix="hybrid_compare_", dir=str(Path("/private/tmp"))) as tmpdir:
        tmp_root = Path(tmpdir)
        jobs = max(1, int(args.jobs))
        payloads = [
            {
                "idx": idx,
                "total": len(cases),
                "case": case,
                "baseline_root": str(args.baseline_root),
                "current_root": str(args.current_root),
                "baseline_solve_method": args.baseline_solve_method,
                "current_solve_method": args.current_solve_method,
                "tmp_root": str(tmp_root),
            }
            for idx, case in enumerate(cases, start=1)
        ]
        if jobs > 1 and len(payloads) > 1:
            with Pool(processes=jobs, maxtasksperchild=1) as pool:
                rows = list(pool.imap_unordered(run_case_pair, payloads))
        else:
            rows = [run_case_pair(payload) for payload in payloads]

    rows.sort(key=lambda row: (str(row["scene"]), int(row["tenant_count"]), int(row["trial_index"])))
    payload = {
        "metadata": {
            "baseline_root": str(args.baseline_root.resolve()),
            "current_root": str(args.current_root.resolve()),
            "inputs": [str(path) for path in args.inputs],
            "tenant_counts": sorted(tenant_filter) if tenant_filter is not None else "all",
            "trials_1_based": sorted(trial_filter_1_based) if trial_filter_1_based else "all",
            "jobs": args.jobs,
            "baseline_solve_method": args.baseline_solve_method,
            "current_solve_method": args.current_solve_method,
            "eps": args.eps,
        },
        "summary": summarize(rows, args.eps),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved comparison to {args.output}", flush=True)


if __name__ == "__main__":
    main()
