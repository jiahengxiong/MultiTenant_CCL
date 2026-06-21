#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = (
    ("High_resource", "High_resource.py", "High_resource.json"),
    ("Low_resource", "Low_resource.py", "Low_resource.json"),
    ("High_resource_homo", "High_resource_homo.py", "High_resource_homo.json"),
    ("Low_resource_homo", "Low_resource_homo.py", "Low_resource_homo.json"),
)

METHODS = ("baseline", "failover", "local", "coordinate")
ALL_SERIES = ("working", *METHODS)
EXPECTED_MODES = {
    "High_resource": {
        "workload_mode": {"low_contention_dominant"},
        "protection_pool_size_mode": {"high_resource_one_server_per_leaf"},
    },
    "Low_resource": {
        "workload_mode": {"low_contention_dominant"},
        "protection_pool_size_mode": {"low_resource_one_server_per_two_leaves"},
    },
    "High_resource_homo": {
        "workload_mode": {"low_contention_homo_gpt"},
        "protection_pool_size_mode": {"high_resource_one_server_per_leaf"},
    },
    "Low_resource_homo": {
        "workload_mode": {"low_contention_homo_gpt"},
        "protection_pool_size_mode": {"low_resource_one_server_per_two_leaves"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the four failure-aware repair experiment exporters and verify "
            "their paper-facing result JSON files."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("result"),
        help="Directory where the four result JSON files are written.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke experiment scripts.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Pass --rerun to each experiment script instead of exporting saved results.",
    )
    parser.add_argument(
        "--result-workers",
        type=int,
        default=4,
        help="Worker count passed to each script's result exporter.",
    )
    parser.add_argument(
        "--story-candidate-pool-target",
        type=int,
        default=20,
        help=(
            "Initial candidate pool target for the global rerun barrier."
        ),
    )
    parser.add_argument(
        "--story-candidate-batch-size",
        type=int,
        default=20,
        help=(
            "Global rerun barrier batch size. If any scenario fails MILP selection, "
            "all scenarios rerun with one more batch of candidates."
        ),
    )
    parser.add_argument(
        "--story-selection-method",
        choices=("milp", "ranked"),
        default="milp",
        help="When --rerun is used, pass the same final story selector to all scripts.",
    )
    parser.add_argument(
        "--max-story-candidate-pool-target",
        type=int,
        default=200,
        help="Stop the global rerun barrier after this candidate pool target.",
    )
    parser.add_argument(
        "max_story_candidate_pool_target_positional",
        nargs="?",
        type=int,
        help=(
            "Optional shorthand for --max-story-candidate-pool-target, so "
            "`--rerun 2000` is accepted."
        ),
    )
    parser.add_argument(
        "--overwrite-high-resource",
        action="store_true",
        help=(
            "Allow High_resource.py to overwrite an existing High_resource.json. "
            "By default that protected file is reused if already present."
        ),
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help=(
            "Run experiment exporters even when the target result JSON already "
            "exists. By default existing valid JSON files are reused."
        ),
    )
    args = parser.parse_args()
    if args.max_story_candidate_pool_target_positional is not None:
        args.max_story_candidate_pool_target = int(
            args.max_story_candidate_pool_target_positional
        )
    return args


def _env() -> dict[str, str]:
    experiments_dir = Path(__file__).resolve().parent
    project_root = experiments_dir.parent
    repo_root = project_root.parent
    paths = [str(project_root), str(repo_root)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _run_script(
    *,
    python: str,
    script: Path,
    result_json: Path,
    result_workers: int,
    rerun: bool,
    overwrite_high_resource: bool,
    story_candidate_pool_target: int,
    story_selection_method: str,
) -> None:
    cmd = [
        python,
        str(script),
        "--result-json",
        str(result_json),
        "--result-workers",
        str(max(1, int(result_workers))),
    ]
    if rerun:
        cmd.append("--rerun")
        cmd.extend(["--output-json", str(result_json.with_suffix(".summary.json"))])
        cmd.extend(
            [
                "--story-candidate-pool-target",
                str(max(1, int(story_candidate_pool_target))),
                "--story-selection-method",
                str(story_selection_method),
            ]
        )
    if script.name == "High_resource.py" and overwrite_high_resource:
        cmd.append("--overwrite-result-json")

    subprocess.run(
        cmd,
        cwd=script.parent,
        env=_env(),
        check=True,
    )


def _run_script_checked(
    *,
    python: str,
    script: Path,
    result_json: Path,
    result_workers: int,
    story_candidate_pool_target: int,
    story_candidate_batch_size: int,
    story_selection_method: str,
    overwrite_high_resource: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        python,
        str(script),
        "--result-json",
        str(result_json),
        "--result-workers",
        str(max(1, int(result_workers))),
    ]
    if script.name == "High_resource.py" and overwrite_high_resource:
        cmd.append("--overwrite-result-json")
    env = _env()
    env["STORY_CANDIDATE_BATCH_SIZE"] = str(max(1, int(story_candidate_batch_size)))
    env["STORY_CANDIDATE_POOL_TARGET"] = str(
        max(1, int(story_candidate_pool_target))
    )
    return subprocess.run(
        cmd,
        cwd=script.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _require_number(payload: dict[str, object], key: str, label: str) -> None:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise AssertionError(f"{label} missing numeric {key}: {value!r}")


def _validate_top_level_metrics(path: Path, data: dict[str, object]) -> None:
    for series in ALL_SERIES:
        payload = data.get(series)
        if not isinstance(payload, dict):
            raise AssertionError(f"{path.name}: missing top-level {series}")
        _require_number(payload, "avg_jct", f"{path.name}:{series}")
        _require_number(payload, "makespan", f"{path.name}:{series}")
        if series != "working":
            _require_number(payload, "switch_count", f"{path.name}:{series}")


def _validate_tenant_metrics(path: Path, data: dict[str, object]) -> None:
    results_by_tenant = data.get("results_by_tenant")
    if not isinstance(results_by_tenant, dict) or not results_by_tenant:
        raise AssertionError(f"{path.name}: missing results_by_tenant")
    for tenant_count, tenant_payload in results_by_tenant.items():
        if not isinstance(tenant_payload, dict):
            raise AssertionError(f"{path.name}: invalid tenant payload {tenant_count}")
        for series in ALL_SERIES:
            payload = tenant_payload.get(series)
            if not isinstance(payload, dict):
                raise AssertionError(
                    f"{path.name}: tenant={tenant_count} missing {series}"
                )
            _require_number(payload, "avg_jct", f"{path.name}:tenant={tenant_count}:{series}")
            _require_number(payload, "makespan", f"{path.name}:tenant={tenant_count}:{series}")
            if series != "working":
                _require_number(
                    payload,
                    "switch_count",
                    f"{path.name}:tenant={tenant_count}:{series}",
                )


def _validate_design_modes(experiment_name: str, path: Path, data: dict[str, object]) -> None:
    expected = EXPECTED_MODES[experiment_name]
    for key, allowed in expected.items():
        value = data.get(key)
        if value not in allowed:
            raise AssertionError(
                f"{path.name}: {key}={value!r}, expected one of {sorted(allowed)}"
            )


def _validate_result_json(experiment_name: str, path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"expected result JSON was not written: {path}")
    data = json.loads(path.read_text())
    _validate_design_modes(experiment_name, path, data)
    _validate_top_level_metrics(path, data)
    _validate_tenant_metrics(path, data)
    return data


def _trial_story_selection(trial: dict[str, object]) -> dict[str, object] | None:
    metadata = trial.get("failure_selection_metadata")
    if not isinstance(metadata, dict):
        features = trial.get("failure_selection_features")
        if not isinstance(features, dict):
            return None
        selection = features.get("story_selection")
        return selection if isinstance(selection, dict) else None
    selection = metadata.get("story_selection")
    return selection if isinstance(selection, dict) else None


def _iter_result_trials(data: dict[str, object]) -> list[dict[str, object]]:
    trials = data.get("trials")
    if isinstance(trials, list):
        return [trial for trial in trials if isinstance(trial, dict)]
    trials_by_tenant = data.get("trials_by_tenant")
    if isinstance(trials_by_tenant, dict):
        flattened: list[dict[str, object]] = []
        for tenant_trials in trials_by_tenant.values():
            if isinstance(tenant_trials, list):
                flattened.extend(
                    trial for trial in tenant_trials if isinstance(trial, dict)
                )
        return flattened
    return []


def _validate_barrier_story_selection(
    experiment_name: str,
    path: Path,
    data: dict[str, object],
    *,
    expected_pool_target: int,
    expected_method: str,
) -> None:
    trials = _iter_result_trials(data)
    if not trials:
        raise AssertionError(f"{path.name}: missing trials for barrier validation")
    if not bool(data.get("all_ok", False)):
        raise AssertionError(f"{path.name}: result all_ok is false")
    missing = 0
    wrong_pool = 0
    wrong_method = 0
    infeasible = 0
    for trial in trials:
        if not isinstance(trial, dict):
            missing += 1
            continue
        selection = _trial_story_selection(trial)
        if selection is None:
            missing += 1
            continue
        if int(selection.get("candidate_pool_target", -1)) != int(expected_pool_target):
            wrong_pool += 1
        if str(selection.get("method")) != str(expected_method):
            wrong_method += 1
        if not bool(selection.get("feasible", False)):
            infeasible += 1
    if missing or wrong_pool or wrong_method or infeasible:
        raise AssertionError(
            f"{path.name}: invalid barrier MILP metadata for {experiment_name}: "
            f"missing={missing}, wrong_pool={wrong_pool}, wrong_method={wrong_method}, "
            f"infeasible={infeasible}, expected_pool={expected_pool_target}, "
            f"expected_method={expected_method}"
        )


def _write_barrier_metadata(
    path: Path,
    *,
    pool_target: int,
    batch_size: int,
    selection_method: str,
) -> None:
    data = json.loads(path.read_text())
    data["story_candidate_barrier"] = {
        "enabled": True,
        "batch_size": int(batch_size),
        "candidate_pool_target": int(pool_target),
        "selection_method": str(selection_method),
        "all_experiments_share_pool_target": True,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _publish_barrier_results(
    *,
    staging_dir: Path,
    result_dir: Path,
    pool_target: int,
    batch_size: int,
    selection_method: str,
) -> dict[str, str]:
    written: dict[str, str] = {}
    for experiment_name, _script_name, result_name in EXPERIMENTS:
        staged = staging_dir / result_name
        final = result_dir / result_name
        _write_barrier_metadata(
            staged,
            pool_target=int(pool_target),
            batch_size=int(batch_size),
            selection_method=str(selection_method),
        )
        shutil.copy2(staged, final)
        staged_summary = staged.with_suffix(".summary.json")
        if staged_summary.exists():
            shutil.copy2(staged_summary, final.with_suffix(".summary.json"))
        _validate_result_json(experiment_name, final)
        written[experiment_name] = str(final)
    return written


def _run_barrier_rerun(args: argparse.Namespace, experiments_dir: Path, result_dir: Path) -> dict[str, str]:
    if str(args.story_selection_method) != "milp":
        raise ValueError("global story-candidate barrier requires --story-selection-method milp")
    batch_size = max(1, int(args.story_candidate_batch_size))
    initial_target = max(batch_size, int(args.story_candidate_pool_target))
    if initial_target % batch_size:
        initial_target = ((initial_target // batch_size) + 1) * batch_size
    max_target = max(initial_target, int(args.max_story_candidate_pool_target))
    barrier_root = result_dir / ".barrier_rerun"
    barrier_root.mkdir(parents=True, exist_ok=True)

    pool_target = initial_target
    while pool_target <= max_target:
        staging_dir = barrier_root / f"pool_{pool_target}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        print(
            json.dumps(
                {
                    "barrier_round": {
                        "candidate_pool_target": pool_target,
                        "batch_size": batch_size,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for experiment_name, script_name, result_name in EXPERIMENTS:
            script = experiments_dir / script_name
            staged_result = staging_dir / result_name
            completed = _run_script_checked(
                python=str(args.python),
                script=script,
                result_json=staged_result,
                result_workers=int(args.result_workers),
                story_candidate_pool_target=int(pool_target),
                story_candidate_batch_size=int(batch_size),
                story_selection_method=str(args.story_selection_method),
                overwrite_high_resource=True,
            )
            log_path = staging_dir / f"{experiment_name}.log"
            log_path.write_text(completed.stdout or "")
            if completed.returncode != 0:
                failures.append(
                    f"{experiment_name}: script failed with exit={completed.returncode}; "
                    f"log={log_path}"
                )
                continue
            try:
                data = _validate_result_json(experiment_name, staged_result)
                _validate_barrier_story_selection(
                    experiment_name,
                    staged_result,
                    data,
                    expected_pool_target=int(pool_target),
                    expected_method=str(args.story_selection_method),
                )
            except Exception as exc:
                failures.append(f"{experiment_name}: {exc}; log={log_path}")

        if not failures:
            return _publish_barrier_results(
                staging_dir=staging_dir,
                result_dir=result_dir,
                pool_target=int(pool_target),
                batch_size=int(batch_size),
                selection_method=str(args.story_selection_method),
            )

        print(
            json.dumps(
                {
                    "barrier_round_failed": {
                        "candidate_pool_target": pool_target,
                        "failures": failures,
                        "next_candidate_pool_target": pool_target + batch_size,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
        pool_target += batch_size

    raise RuntimeError(
        "global story-candidate barrier did not find MILP-feasible results "
        f"through candidate_pool_target={max_target}; logs are under {barrier_root}"
    )


def main() -> None:
    args = parse_args()
    experiments_dir = Path(__file__).resolve().parent
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.rerun):
        written = _run_barrier_rerun(args, experiments_dir, result_dir)
        print(
            json.dumps(
                {
                    "result_jsons": written,
                    "validated": True,
                    "barrier": True,
                    "story_candidate_batch_size": int(args.story_candidate_batch_size),
                },
                sort_keys=True,
            )
        )
        return

    written: dict[str, str] = {}
    for experiment_name, script_name, result_name in EXPERIMENTS:
        script = experiments_dir / script_name
        result_json = result_dir / result_name
        if result_json.exists() and not bool(args.refresh_existing) and not bool(args.rerun):
            _validate_result_json(experiment_name, result_json)
        else:
            _run_script(
                python=str(args.python),
                script=script,
                result_json=result_json,
                result_workers=int(args.result_workers),
                rerun=bool(args.rerun),
                overwrite_high_resource=bool(args.overwrite_high_resource),
                story_candidate_pool_target=int(args.story_candidate_pool_target),
                story_selection_method=str(args.story_selection_method),
            )
            _validate_result_json(experiment_name, result_json)
        written[experiment_name] = str(result_json)

    print(json.dumps({"result_jsons": written, "validated": True}, sort_keys=True))


if __name__ == "__main__":
    main()
