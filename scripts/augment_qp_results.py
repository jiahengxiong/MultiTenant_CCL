"""Add only QP and QP+Harmonics to every saved main-experiment trial.

Run on a staged copy of the four source JSONs. Checkpoints permit resumable,
parallel execution. Existing fields are validated before an augmented JSON is
published; no old mapping method or old experiment is run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_qp_scaling_baseline import build_inputs
from multitenant.baselines.qp_harmonics import QPHarmonicsHeuristic
from multitenant.simulator.adapter import _run_simulation_worker_result
from multitenant.topology import LeafSpineDatacenter

SOURCES = ("Low_contension", "High_contension", "Low_contension_homo", "High_contension_homo")
NEW_METHODS = ("qp", "qp_plus_harmonics")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_trial(job):
    checkpoint = Path(job["checkpoint"])
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get("signature") == job["signature"]:
            return saved
        raise ValueError(f"Checkpoint configuration mismatch: {checkpoint}")
    start = time.time()
    trial = job["trial"]
    print(f"START {job['name']} t={job['tenant_count']} trial={trial['trial_index']} pid={os.getpid()}", flush=True)
    sim_topology, policy, diagnostics = build_inputs(job["metadata"], trial, job["qp_count"], job["hash_seed"])
    qp_result = _run_simulation_worker_result(sim_topology, policy)
    qp_runtime = time.time() - start
    # Persist QP independently so progress is visible before scheduling finishes.
    atomic_json(checkpoint.with_suffix(".qp.json"), {
        "signature": job["signature"], "avg_jct": qp_result["avg_tenant_makespan"],
        "makespan": qp_result["global_makespan"], "runtime_seconds": qp_runtime,
    })
    print(f"QP_DONE {job['name']} t={job['tenant_count']} trial={trial['trial_index']} JCT={qp_result['avg_tenant_makespan']:.9g}", flush=True)
    mapping = {int(t): {int(r): int(s) for r, s in ranks.items()} for t, ranks in trial["initial_mapping"].items()}
    specs = {int(t): dict(spec) for t, spec in trial["tenant_collective_specs"].items()}
    solver = QPHarmonicsHeuristic(
        LeafSpineDatacenter(**job["metadata"]["topology"]), mapping, specs,
        sim_topology, policy, initial_result=qp_result,
        program_time_limit_s=job["metadata"].get("harmonics_time_limit_seconds"),
    )
    del qp_result["tx_complete_time"], qp_result["tx_first_send_time"], qp_result["tx_service_start_time"], qp_result["chunk_ready_time"]
    scheduling_start = time.time()
    solver.solve()
    scheduled = solver.final_result()
    # The inherited search starts from zero offsets and accepts only improvements.
    assert (scheduled["avg_tenant_makespan"], scheduled["global_makespan"]) <= (
        qp_result["avg_tenant_makespan"] + 1e-8, qp_result["global_makespan"] + 1e-8)
    for schedules in (solver.collective_rate_scales, solver.collective_rate_schedule, solver.task_rate_schedule):
        assert all(not value for value in schedules.values()), "Main single-collective path should use offsets only"
    result = {
        "signature": job["signature"], "source_sha256": job["source_sha256"],
        "source_name": job["name"], "tenant_count": job["tenant_count"], "trial_index": trial["trial_index"],
        "trial_seed": trial["trial_seed"], "qp_diagnostics": diagnostics,
        "results": {
            "qp": {"avg_jct": qp_result["avg_tenant_makespan"], "makespan": qp_result["global_makespan"],
                   "runtime_seconds": qp_runtime},
            "qp_plus_harmonics": {
                "avg_jct": scheduled["avg_tenant_makespan"], "makespan": scheduled["global_makespan"],
                "tenant_start_times": {str(t): 0.0 for t in mapping},
                "collective_start_times": {str(t): {str(op): float(v) for op, v in starts.items()}
                                           for t, starts in solver.collective_start_times.items()},
                "collective_rate_scales": solver.collective_rate_scales,
                "collective_rate_schedule": solver.collective_rate_schedule,
                "task_rate_schedule": solver.task_rate_schedule,
                "runtime_seconds": time.time() - scheduling_start,
                "simulation_count": solver.qp_simulation_count, "simulation_cache_hits": solver.qp_cache_hits,
            },
        },
        "total_runtime_seconds": time.time() - start,
    }
    atomic_json(checkpoint, result)
    print(f"DONE {job['name']} t={job['tenant_count']} trial={trial['trial_index']} "
          f"QP={result['results']['qp']['avg_jct']:.9g} QP+H={scheduled['avg_tenant_makespan']:.9g} "
          f"simulations={solver.qp_simulation_count} elapsed={result['total_runtime_seconds']:.1f}s", flush=True)
    return result


def merge_source(original, additions, provenance):
    result = copy.deepcopy(original)
    for group in result["results"]:
        for trial in group["trial_results"]:
            added = additions[(int(group["tenant_count"]), int(trial["trial_index"]))]
            assert trial["trial_seed"] == added["trial_seed"]
            for method in NEW_METHODS:
                trial["results"][method] = added["results"][method]
            trial["qp_diagnostics"] = added["qp_diagnostics"]
        for method in NEW_METHODS:
            group["averages"][method] = {
                metric: sum(float(t["results"][method][metric]) for t in group["trial_results"]) / len(group["trial_results"])
                for metric in ("avg_jct", "makespan")
            }
    result["metadata"]["qp_baseline"] = provenance
    # Prove that removing the additions recovers all old values, not just averages.
    stripped = copy.deepcopy(result)
    del stripped["metadata"]["qp_baseline"]
    for group in stripped["results"]:
        for method in NEW_METHODS:
            del group["averages"][method]
        for trial in group["trial_results"]:
            del trial["qp_diagnostics"]
            for method in NEW_METHODS:
                del trial["results"][method]
    assert stripped == original, "Existing experiment fields must remain exactly unchanged"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "experiment")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 4))
    parser.add_argument("--qp-count", type=int, default=4)
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--tenant-counts", default=None, help="Optional subset for adapter validation")
    parser.add_argument("--trials", type=int, default=None, help="Optional first N trials for adapter validation")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    args = parser.parse_args()
    if args.workers < 1 or args.qp_count < 2:
        parser.error("workers must be positive and qp-count must be >=2")
    import simcore_cpp
    provenance = {
        "description": "Meta-inspired QP message round robin and QP-aware ECMP; original default mapping",
        "qp_count": args.qp_count, "hash_seed": args.hash_seed,
        "harmonics": "existing HarmonicsBaselineHeuristic offset search with QP-aware metadata and simulation callbacks",
        "server": platform.node(), "python": sys.version, "backend": "simcore_cpp",
        "workers": args.workers,
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), ROOT / "scripts/run_qp_scaling_baseline.py", ROOT / "multitenant/baselines/qp_harmonics.py",
            ROOT / "multitenant/baselines/harmonics.py", ROOT / "multitenant/simulator/adapter.py",
            ROOT / "multitenant/simulator/worker.py", ROOT / "CCL_Simulator/simcore_cpp/sim.hpp",
        ]},
        "full_run": args.tenant_counts is None and args.trials is None,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_data = {}
    jobs = []
    for name in args.sources:
        path = args.input_dir / f"{name}.json"
        raw = path.read_bytes()
        original = json.loads(raw)
        source_data[name] = original
        source_hash = hashlib.sha256(raw).hexdigest()
        for group in original["results"]:
            count = int(group["tenant_count"])
            if args.tenant_counts and count not in {int(t) for t in args.tenant_counts.split(",")}:
                continue
            trials = sorted(group["trial_results"], key=lambda t: int(t["trial_index"]))
            if args.trials is not None:
                trials = trials[:args.trials]
            for trial in trials:
                signature = hashlib.sha256(json.dumps([
                    source_hash, count, trial["trial_index"], args.qp_count, args.hash_seed, provenance["code_sha256"],
                ], sort_keys=True).encode()).hexdigest()
                jobs.append({"name": name, "metadata": original["metadata"], "trial": trial,
                             "tenant_count": count, "qp_count": args.qp_count, "hash_seed": args.hash_seed,
                             "source_sha256": source_hash, "signature": signature,
                             "checkpoint": str(output / "checkpoints" / f"{name}_t{count}_trial{trial['trial_index']}.json")})
    all_results = {}
    failures = []
    start = time.time()
    atomic_json(output / "manifest.json", {"provenance": provenance, "total_trials": len(jobs), "completed": 0})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_jobs = {pool.submit(run_trial, job): job for job in jobs}
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            try:
                result = future.result()
                all_results[(job["name"], job["tenant_count"], job["trial"]["trial_index"])] = result
            except Exception as exc:
                failures.append({"name": job["name"], "tenant_count": job["tenant_count"],
                                 "trial_index": job["trial"]["trial_index"], "error": repr(exc)})
                print(f"ERROR {failures[-1]}", flush=True)
            atomic_json(output / "manifest.json", {"provenance": provenance, "total_trials": len(jobs),
                        "completed": len(all_results), "failures": failures, "elapsed_seconds": time.time() - start})
            print(f"PROGRESS {len(all_results)}/{len(jobs)} failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} trials failed; checkpoints retained for resume")
    atomic_json(output / "all_new_results.json", list(all_results.values()))
    if provenance["full_run"]:
        for name, original in source_data.items():
            additions = {(count, index): result for (source, count, index), result in all_results.items() if source == name}
            merged = merge_source(original, additions, provenance)
            atomic_json(output / f"{name}.json", merged)
    print(f"COMPLETE trials={len(all_results)} new_method_results={len(all_results)*2} elapsed={time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
