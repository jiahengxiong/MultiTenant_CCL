#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import select
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = (
    ("High_resource", "High_resource.py", "High_resource.json"),
    ("Low_resource", "Low_resource.py", "Low_resource.json"),
    ("High_resource_homo", "High_resource-homo.py", "High_resource_homo.json"),
    ("Low_resource_homo", "Low_resource_homo.py", "Low_resource_homo.json"),
)
EXPERIMENT_NAMES = tuple(experiment[0] for experiment in EXPERIMENTS)

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

STORY_CACHE_DIRS = {
    "High_resource": Path("/private/tmp/high_resource_story_cache"),
    "Low_resource": Path("/private/tmp/low_resource_story_cache"),
    "High_resource_homo": Path("/private/tmp/high_resource_homo_story_cache"),
    "Low_resource_homo": Path("/private/tmp/low_resource_homo_story_cache"),
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
        "--experiment",
        choices=EXPERIMENT_NAMES,
        action="append",
        default=None,
        help=(
            "Run only the named experiment. May be repeated; defaults to all "
            "experiments."
        ),
    )
    parser.add_argument(
        "--tenant-min",
        type=int,
        default=None,
        help="Optional smoke/debug override passed to all experiment scripts.",
    )
    parser.add_argument(
        "--tenant-max",
        type=int,
        default=None,
        help="Optional smoke/debug override passed to all experiment scripts.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Optional smoke/debug trials-per-tenant override passed to all scripts.",
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
    parser.add_argument(
        "--reproduce-high-resource",
        action="store_true",
        help=(
            "Through this driver, recompute High_resource.json's summary from "
            "stored trials and replay its stored mappings through the simulator."
        ),
    )
    parser.add_argument(
        "--reproduced-high-resource-summary",
        type=Path,
        default=None,
        help="Output path for --reproduce-high-resource summary reproduction.",
    )
    parser.add_argument(
        "--replayed-high-resource-simulator",
        type=Path,
        default=None,
        help="Output path for --reproduce-high-resource simulator replay.",
    )
    parser.add_argument(
        "--candidate-pool-status",
        action="store_true",
        help=(
            "Print current story-cache and evaluated-candidate-pool counts for "
            "the four experiment scripts, then exit."
        ),
    )
    args = parser.parse_args()
    if args.max_story_candidate_pool_target_positional is not None:
        args.max_story_candidate_pool_target = int(
            args.max_story_candidate_pool_target_positional
        )
    return args


def _selected_experiments(args: argparse.Namespace) -> tuple[tuple[str, str, str], ...]:
    requested = getattr(args, "experiment", None)
    if not requested:
        return EXPERIMENTS
    requested_set = set(str(name) for name in requested)
    return tuple(
        experiment for experiment in EXPERIMENTS if experiment[0] in requested_set
    )


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


def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def _evaluated_candidate_pool_by_tenant(
    result_dir: Path,
    experiment_name: str,
    *,
    candidate_pool_dir: Path | None = None,
) -> dict[str, int]:
    pool_dir = (
        candidate_pool_dir
        if candidate_pool_dir is not None
        else result_dir / ".story_candidate_pool"
    )
    counts: dict[str, int] = {}
    if not pool_dir.exists():
        return counts
    prefix = f"{experiment_name}__t"
    for path in sorted(pool_dir.glob(f"{prefix}*.json")):
        tenant = path.stem.removeprefix(prefix)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            counts[str(tenant)] = -1
            continue
        metadata = payload.get("metadata")
        candidate_pool = payload.get("candidate_pool")
        count = None
        if isinstance(metadata, dict):
            raw_count = metadata.get("candidate_pool_size")
            if isinstance(raw_count, int):
                count = int(raw_count)
        if count is None:
            count = len(candidate_pool) if isinstance(candidate_pool, list) else -1
        counts[str(tenant)] = int(count)
    return counts


def _tenant_from_seed_story_path(path: Path) -> str:
    stem = path.stem
    marker = "__t"
    if marker not in stem:
        return "unknown"
    return stem.rsplit(marker, 1)[-1]


def _seed_story_candidate_counts(cache_dir: Path) -> dict[str, object]:
    counts_by_tenant: dict[str, dict[str, int]] = {}
    seed_story_dir = cache_dir / "seed_story"
    if not seed_story_dir.exists():
        return {
            "selected_by_tenant": {},
            "fallback_by_tenant": {},
            "selected_total": 0,
            "fallback_total": 0,
            "unreadable_seed_story_files": 0,
        }
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    for import_root in (project_root, repo_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    unreadable = 0
    for path in sorted(seed_story_dir.glob("*.pkl")):
        tenant = _tenant_from_seed_story_path(path)
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            unreadable += 1
            continue
        seed_result = payload.get("seed_result", payload) if isinstance(payload, dict) else {}
        if not isinstance(seed_result, dict):
            unreadable += 1
            continue
        selected = seed_result.get("selected_failure_cases") or []
        fallback = seed_result.get("fallback_failure_cases") or []
        bucket = counts_by_tenant.setdefault(
            tenant,
            {"selected": 0, "fallback": 0},
        )
        bucket["selected"] += len(selected) if isinstance(selected, list) else 0
        bucket["fallback"] += len(fallback) if isinstance(fallback, list) else 0
    selected_by_tenant = {
        tenant: values["selected"]
        for tenant, values in sorted(counts_by_tenant.items(), key=lambda item: item[0])
    }
    fallback_by_tenant = {
        tenant: values["fallback"]
        for tenant, values in sorted(counts_by_tenant.items(), key=lambda item: item[0])
    }
    return {
        "selected_by_tenant": selected_by_tenant,
        "fallback_by_tenant": fallback_by_tenant,
        "selected_total": sum(selected_by_tenant.values()),
        "fallback_total": sum(fallback_by_tenant.values()),
        "unreadable_seed_story_files": int(unreadable),
    }


def _candidate_pool_status(
    result_dir: Path,
    *,
    story_cache_root: Path | None = None,
    candidate_pool_dir: Path | None = None,
) -> dict[str, object]:
    status: dict[str, object] = {}
    for experiment_name, _script_name, _result_name in EXPERIMENTS:
        cache_dir = (
            story_cache_root / experiment_name
            if story_cache_root is not None
            else STORY_CACHE_DIRS[experiment_name]
        )
        by_tenant = _evaluated_candidate_pool_by_tenant(
            result_dir,
            experiment_name,
            candidate_pool_dir=candidate_pool_dir,
        )
        story_candidate_counts = _seed_story_candidate_counts(cache_dir)
        status[experiment_name] = {
            "story_cache_dir": str(cache_dir),
            "base_experiment_cache_count": _count_files(cache_dir, "*.pkl"),
            "seed_story_cache_count": _count_files(cache_dir / "seed_story", "*.pkl"),
            "selected_story_candidate_count_by_tenant": story_candidate_counts[
                "selected_by_tenant"
            ],
            "selected_story_candidate_total": story_candidate_counts[
                "selected_total"
            ],
            "fallback_story_candidate_count_by_tenant": story_candidate_counts[
                "fallback_by_tenant"
            ],
            "fallback_story_candidate_total": story_candidate_counts[
                "fallback_total"
            ],
            "unreadable_seed_story_files": story_candidate_counts[
                "unreadable_seed_story_files"
            ],
            "simulator_validation_cache_count": _count_files(
                cache_dir / "simulator_validation",
                "*.pkl",
            ),
            "evaluated_candidate_pool_by_tenant": by_tenant,
            "evaluated_candidate_pool_total": sum(
                count for count in by_tenant.values() if count > 0
            ),
        }
    return status


def _print_candidate_pool_status(
    *,
    result_dir: Path,
    event: str,
    candidate_pool_target: int | None = None,
    experiment_name: str | None = None,
    story_cache_root: Path | None = None,
    candidate_pool_dir: Path | None = None,
) -> None:
    payload: dict[str, object] = {
        "candidate_pool_status": _candidate_pool_status(
            result_dir,
            story_cache_root=story_cache_root,
            candidate_pool_dir=candidate_pool_dir,
        ),
        "event": event,
    }
    if candidate_pool_target is not None:
        payload["candidate_pool_target"] = int(candidate_pool_target)
    if experiment_name is not None:
        payload["experiment"] = str(experiment_name)
    print(json.dumps(payload, sort_keys=True), flush=True)


def _run_streaming(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    label: str,
    heartbeat_status=None,
) -> subprocess.CompletedProcess[str]:
    stream_env = dict(env)
    stream_env["PYTHONUNBUFFERED"] = "1"
    print(
        json.dumps(
            {
                "subprocess_start": {
                    "label": label,
                    "cmd": " ".join(cmd),
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=stream_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    lines: list[str] = []
    start_time = time.time()
    last_output_time = start_time
    try:
        assert process.stdout is not None
        while True:
            readable, _writable, _errored = select.select(
                [process.stdout],
                [],
                [],
                60.0,
            )
            if readable:
                line = process.stdout.readline()
                if line:
                    lines.append(line)
                    last_output_time = time.time()
                    print(line, end="", flush=True)
                    continue
                if process.poll() is not None:
                    break
            returncode = process.poll()
            if returncode is not None:
                remainder = process.stdout.read()
                if remainder:
                    lines.append(remainder)
                    print(remainder, end="", flush=True)
                break
            now = time.time()
            heartbeat = {
                "label": label,
                "elapsed_seconds": round(now - start_time, 1),
                "silent_seconds": round(now - last_output_time, 1),
            }
            if heartbeat_status is not None:
                heartbeat["candidate_pool_status"] = heartbeat_status()
            print(
                json.dumps(
                    {"subprocess_heartbeat": heartbeat},
                    sort_keys=True,
                ),
                flush=True,
            )
        returncode = process.wait()
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        raise
    except Exception:
        os.killpg(process.pid, signal.SIGTERM)
        raise
    print(
        json.dumps(
            {
                "subprocess_done": {
                    "label": label,
                    "returncode": returncode,
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout="".join(lines),
    )


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

    completed = _run_streaming(
        cmd,
        cwd=script.parent,
        env=_env(),
        label=f"{script.stem}:export",
    )
    completed.check_returncode()


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
    story_cache_dir: Path,
    candidate_pool_dir: Path | None = None,
    tenant_min: int | None = None,
    tenant_max: int | None = None,
    trials: int | None = None,
    heartbeat_status=None,
) -> subprocess.CompletedProcess[str]:
    generate_env = _env()
    export_env = _env()
    source_cache_overrides = {
        "High_resource-homo.py": story_cache_dir.parent / "High_resource",
        "High_resource_homo.py": story_cache_dir.parent / "High_resource",
        "Low_resource_homo.py": story_cache_dir.parent / "Low_resource",
    }
    if script.name in source_cache_overrides:
        generate_env["WORKING_MAPPING_SOURCE_CACHE_DIR_OVERRIDE"] = str(
            source_cache_overrides[script.name]
        )
        export_env["WORKING_MAPPING_SOURCE_CACHE_DIR_OVERRIDE"] = str(
            source_cache_overrides[script.name]
        )
    export_env["STORY_CACHE_DIR_OVERRIDE"] = str(story_cache_dir)
    if candidate_pool_dir is not None:
        export_env["STORY_CANDIDATE_POOL_DIR_OVERRIDE"] = str(candidate_pool_dir)
    export_env["STORY_CANDIDATE_BATCH_SIZE"] = str(max(1, int(story_candidate_batch_size)))
    export_env["STORY_CANDIDATE_POOL_TARGET"] = str(
        max(1, int(story_candidate_pool_target))
    )
    if tenant_min is not None:
        export_env["STORY_EXPORT_TENANT_MIN"] = str(int(tenant_min))
    if tenant_max is not None:
        export_env["STORY_EXPORT_TENANT_MAX"] = str(int(tenant_max))
    if trials is not None:
        export_env["STORY_EXPORT_TARGET_TRIALS"] = str(int(trials))

    candidates_per_seed = min(10, max(1, int(story_candidate_batch_size)))
    story_search_workers = max(
        1,
        min(
            max(1, int(result_workers)),
            max(
                1,
                (
                    max(1, int(story_candidate_batch_size))
                    + candidates_per_seed
                    - 1
                )
                // candidates_per_seed,
            ),
        ),
    )
    failure_screen_candidates = max(
        1,
        (
            max(1, int(story_candidate_batch_size))
            + story_search_workers
            - 1
        )
        // story_search_workers,
    )
    generate_cmd = [
        python,
        str(script),
        "--result-json",
        str(result_json),
        "--result-workers",
        str(max(1, int(result_workers))),
        "--rerun",
        "--output-json",
        str(result_json.with_suffix(".summary.json")),
        "--story-candidate-pool-target",
        str(max(1, int(story_candidate_pool_target))),
        "--story-selection-method",
        str(story_selection_method),
        "--story-cache-dir",
        str(story_cache_dir),
        "--skip-result-export",
        "--mapping-seed-attempts",
        str(max(30, int(story_candidate_pool_target))),
        "--story-search-workers",
        str(story_search_workers),
        "--failure-screen-mode",
        "structural",
        "--failure-screen-candidates",
        str(failure_screen_candidates),
    ]
    if tenant_min is not None:
        generate_cmd.extend(["--tenant-min", str(int(tenant_min))])
    if tenant_max is not None:
        generate_cmd.extend(["--tenant-max", str(int(tenant_max))])
    if trials is not None:
        generate_cmd.extend(["--trials", str(int(trials))])
    generate = _run_streaming(
        generate_cmd,
        cwd=script.parent,
        env=generate_env,
        label=f"{script.stem}:generate",
        heartbeat_status=heartbeat_status,
    )
    if generate.returncode != 0:
        return generate

    export_cmd = [
        python,
        str(script),
        "--result-json",
        str(result_json),
        "--result-workers",
        str(max(1, int(result_workers))),
        "--export-result-json-only",
    ]
    if tenant_min is not None:
        export_cmd.extend(["--tenant-min", str(int(tenant_min))])
    if tenant_max is not None:
        export_cmd.extend(["--tenant-max", str(int(tenant_max))])
    if trials is not None:
        export_cmd.extend(["--trials", str(int(trials))])
    if script.name == "High_resource.py" and overwrite_high_resource:
        export_cmd.append("--overwrite-result-json")
    export = _run_streaming(
        export_cmd,
        cwd=script.parent,
        env=export_env,
        label=f"{script.stem}:export",
        heartbeat_status=heartbeat_status,
    )
    return subprocess.CompletedProcess(
        args=export.args,
        returncode=export.returncode,
        stdout=(
            "===== generate candidates =====\n"
            + (generate.stdout or "")
            + "\n===== export result =====\n"
            + (export.stdout or "")
        ),
    )


def _reproduce_high_resource(
    *,
    python: str,
    experiments_dir: Path,
    result_dir: Path,
    reproduced_summary_json: Path | None,
    replayed_simulator_json: Path | None,
) -> dict[str, object]:
    script = experiments_dir / "High_resource.py"
    result_json = (result_dir / "High_resource.json").resolve()
    summary_json = reproduced_summary_json or (
        result_dir / "High_resource.reproduced_summary.json"
    )
    replay_json = replayed_simulator_json or (
        result_dir / "High_resource.replayed_simulator.json"
    )
    summary_json = summary_json.resolve()
    replay_json = replay_json.resolve()
    completed_summary = _run_streaming(
        [
            python,
            str(script),
            "--reproduce-result-summary",
            "--result-json",
            str(result_json),
            "--reproduced-summary-json",
            str(summary_json),
        ],
        cwd=script.parent,
        env=_env(),
        label="High_resource:reproduce-summary",
    )
    completed_summary.check_returncode()

    completed_replay = _run_streaming(
        [
            python,
            str(script),
            "--replay-result-simulator",
            "--result-json",
            str(result_json),
            "--replayed-result-json",
            str(replay_json),
            "--overwrite-result-json",
        ],
        cwd=script.parent,
        env=_env(),
        label="High_resource:replay-simulator",
    )
    completed_replay.check_returncode()

    summary_payload = json.loads(summary_json.read_text())
    replay_payload = json.loads(replay_json.read_text())
    return {
        "source_result_json": str(result_json),
        "reproduced_summary_json": str(summary_json),
        "replayed_simulator_json": str(replay_json),
        "summary_all_ok": bool(summary_payload.get("all_ok")),
        "summary_trials": int(summary_payload.get("trials", -1)),
        "replay_trials": int(replay_payload.get("trials", -1)),
        "max_abs_metric_diff": float(replay_payload.get("max_abs_metric_diff", -1.0)),
        "reproduced": (
            bool(summary_payload.get("all_ok"))
            and int(summary_payload.get("trials", -1)) == 60
            and int(replay_payload.get("trials", -1)) == 60
            and float(replay_payload.get("max_abs_metric_diff", -1.0)) == 0.0
        ),
    }


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
    experiments: tuple[tuple[str, str, str], ...] = EXPERIMENTS,
) -> dict[str, str]:
    written: dict[str, str] = {}
    for experiment_name, _script_name, result_name in experiments:
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
    if barrier_root.exists():
        shutil.rmtree(barrier_root)
    barrier_root.mkdir(parents=True, exist_ok=True)
    story_cache_root = barrier_root / "story_cache"
    candidate_pool_dir = barrier_root / "story_candidate_pool"

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
        _print_candidate_pool_status(
            result_dir=staging_dir,
            event="barrier_round_start",
            candidate_pool_target=int(pool_target),
            story_cache_root=story_cache_root,
            candidate_pool_dir=candidate_pool_dir,
        )
        for experiment_name, script_name, result_name in _selected_experiments(args):
            script = experiments_dir / script_name
            staged_result = staging_dir / result_name
            _print_candidate_pool_status(
                result_dir=staging_dir,
                event="experiment_start",
                candidate_pool_target=int(pool_target),
                experiment_name=experiment_name,
                story_cache_root=story_cache_root,
                candidate_pool_dir=candidate_pool_dir,
            )
            completed = _run_script_checked(
                python=str(args.python),
                script=script,
                result_json=staged_result,
                result_workers=int(args.result_workers),
                story_candidate_pool_target=int(pool_target),
                story_candidate_batch_size=int(batch_size),
                story_selection_method=str(args.story_selection_method),
                overwrite_high_resource=True,
                story_cache_dir=story_cache_root / experiment_name,
                candidate_pool_dir=candidate_pool_dir,
                tenant_min=args.tenant_min,
                tenant_max=args.tenant_max,
                trials=args.trials,
                heartbeat_status=lambda staging_dir=staging_dir, story_cache_root=story_cache_root, candidate_pool_dir=candidate_pool_dir: _candidate_pool_status(
                    staging_dir,
                    story_cache_root=story_cache_root,
                    candidate_pool_dir=candidate_pool_dir,
                ),
            )
            _print_candidate_pool_status(
                result_dir=staging_dir,
                event="experiment_done",
                candidate_pool_target=int(pool_target),
                experiment_name=experiment_name,
                story_cache_root=story_cache_root,
                candidate_pool_dir=candidate_pool_dir,
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
                experiments=_selected_experiments(args),
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
        _print_candidate_pool_status(
            result_dir=staging_dir,
            event="barrier_round_failed",
            candidate_pool_target=int(pool_target),
            story_cache_root=story_cache_root,
            candidate_pool_dir=candidate_pool_dir,
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

    if bool(args.candidate_pool_status):
        _print_candidate_pool_status(
            result_dir=result_dir,
            event="candidate_pool_status",
        )
        return

    if bool(args.reproduce_high_resource):
        reproduction = _reproduce_high_resource(
            python=str(args.python),
            experiments_dir=experiments_dir,
            result_dir=result_dir,
            reproduced_summary_json=args.reproduced_high_resource_summary,
            replayed_simulator_json=args.replayed_high_resource_simulator,
        )
        print(
            json.dumps(
                {
                    "High_resource": reproduction,
                    "validated": bool(reproduction["reproduced"]),
                },
                sort_keys=True,
            )
        )
        if not bool(reproduction["reproduced"]):
            raise SystemExit(1)
        return

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
    for experiment_name, script_name, result_name in _selected_experiments(args):
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
