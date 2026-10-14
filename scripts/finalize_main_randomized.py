"""Audit and plot a completed randomized main run, optionally waiting for it."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.augment_qp_results import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    output = args.run_dir.resolve()
    while True:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest.get("complete"):
            break
        if not args.wait:
            raise SystemExit("Run is not complete; no final figure generated")
        launch_path = ROOT / "launch.json"
        if launch_path.exists():
            pid = json.loads(launch_path.read_text())["pid"]
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                raise SystemExit("Experiment process stopped before completion; resume its checkpoints first")
        # This is the final stage of the server-side experiment pipeline.
        time.sleep(30)
    plan = json.loads((output / "case_plan.json").read_text())
    methods = plan["metadata"]["methods"]
    planned = {c["input_sha256"]: c for c in plan["cases"]}
    seen = set()
    rows, paired_rows = [], []
    for scenario in dict.fromkeys(c["source_name"] for c in plan["cases"]):
        data = json.loads((output / f"{scenario}.json").read_text())
        assert data["complete"]
        for group in data["results"]:
            trials = group["trial_results"]
            assert len(trials) == plan["metadata"]["trials"] == group["trial_count"]
            assert {t["trial_index"] for t in trials} == set(range(plan["metadata"]["trials"]))
            for trial in trials:
                fingerprint = trial["input_sha256"]
                assert fingerprint not in seen
                seen.add(fingerprint)
                original = planned[fingerprint]
                for key in ("initial_mapping", "tenant_collective_specs", "random_seeds", "trial_seed"):
                    assert trial[key] == original[key]
                assert set(trial["results"]) == set(methods)
            for metric in ("avg_jct", "makespan"):
                for method in methods:
                    values = [t["results"][method][metric] for t in trials]
                    assert all(math.isfinite(v) and v > 0 for v in values)
                    average = mean(values)
                    assert math.isclose(average, group["averages"][method][metric], rel_tol=1e-12)
                    rows.append(dict(scenario=scenario, tenant_count=group["tenant_count"], method=method,
                                     metric=metric, trial_count=len(trials), average=average,
                                     standard_error=stdev(values)/math.sqrt(len(values)),
                                     normalized_ratio_of_means=average/group["averages"]["default"][metric]))
                for baseline in ("qp", "qp_plus_harmonics"):
                    for comparator in ("mapping", "mapping_plus_harmonics"):
                        # Paired raw differences capture covariance of shared cases.
                        differences = [t["results"][baseline][metric]-t["results"][comparator][metric] for t in trials]
                        paired_rows.append(dict(scenario=scenario, tenant_count=group["tenant_count"], metric=metric,
                                                baseline=baseline, comparator=comparator,
                                                mean_paired_difference=mean(differences),
                                                standard_error=stdev(differences)/math.sqrt(len(differences)),
                                                baseline_faster_trials=sum(v < 0 for v in differences), trial_count=len(trials)))
    assert seen == set(planned)
    for filename, records in [("summary.csv", rows), ("paired_cape_comparison.csv", paired_rows)]:
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    plotting_path = ROOT / "figures/plot_main_results.py"
    spec = importlib.util.spec_from_file_location("main_figures", plotting_path)
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)
    plotting.EXPERIMENT_DIR = output
    plotting.OUT_DIR = output / "figures"
    plotting.main()
    audit = dict(complete=True, cases=len(seen), methods=len(methods), total_method_results=len(seen)*len(methods),
                 trials_per_group=plan["metadata"]["trials"], all_inputs_match_plan=True,
                 all_means_verified=True, plotting_min_tenant_count=plotting.MIN_TENANT_COUNT,
                 case_plan_sha256=hashlib.sha256((output/"case_plan.json").read_bytes()).hexdigest())
    atomic_json(output / "audit.json", audit)
    archive = output.parent / "main50_results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    print(json.dumps(dict(audit=audit, archive=str(archive), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
