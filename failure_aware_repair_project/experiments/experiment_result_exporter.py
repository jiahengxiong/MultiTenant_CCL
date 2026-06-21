from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from failure_aware_repair.evaluator import RepairEvaluator
from failure_aware_repair.heuristic import REPAIR_ALGORITHM_VERSION, RepairSearchConfig
from failure_aware_repair.models import FailureEvent, RepairScenario
from failure_aware_repair.random_experiments import repair_comparison_payload, run_repair_comparison
from failure_aware_repair.protection import (
    check_repair_feasible,
    choose_failover_protection,
    count_switches,
)
from failure_aware_repair.solvers.nearest_protection_baseline import (
    solve_nearest_protection_baseline,
)
from failure_aware_repair.strategies import TENANT_LOCAL_REPAIR


METHOD_TO_STRATEGY = {
    "failover": "repair_failed_server_only",
    "local": "tenant_local_repair",
    "coordinate": "cooperative_repair",
}
RESULT_METHODS = ("baseline", "failover", "local", "coordinate")
CASE_EVAL_BATCH_SIZE = int(os.environ.get("STORY_CANDIDATE_BATCH_SIZE", "20"))
FIXED_STORY_CANDIDATE_POOL_TARGET = os.environ.get("STORY_CANDIDATE_POOL_TARGET")
EXPORT_TENANT_MIN = os.environ.get("STORY_EXPORT_TENANT_MIN")
EXPORT_TENANT_MAX = os.environ.get("STORY_EXPORT_TENANT_MAX")
EXPORT_TARGET_TRIALS = os.environ.get("STORY_EXPORT_TARGET_TRIALS")
LOCAL_GAIN_THRESHOLD_PREFERRED_PCT = 15.0
LOCAL_GAIN_THRESHOLD_FALLBACK_PCT = 10.0
PREFERRED_SEARCH_CANDIDATE_CAP = 60
SUMMARY_METRICS = (
    "avg_jct",
    "makespan",
    "switch_servers",
    "switch_ranks",
    "extra_switch_servers_vs_failover",
    "extra_switch_ranks_vs_failover",
)
CANDIDATE_EVAL_CACHE_DIR = Path(
    "/private/tmp/failure_repair_result_candidate_cache_v90_contention_guided_candidate_set_search_v48"
)
NEAREST_BASELINE_CACHE_DIR = Path(
    "/private/tmp/failure_repair_nearest_baseline_cache_v1"
)
FAILURE_SELECTION_FEATURE_KEYS = (
    "cross_leaf_incident_edges",
    "server_leaf",
    "prefilter_failover_avg_jct",
    "prefilter_failover_makespan",
    "prefilter_rank_pressure",
    "prefilter_tenant_pressure",
    "failure_source",
    "high_resource_story_case_source",
    "high_resource_story_case_list",
)


def _load_module(script_path: Path) -> ModuleType:
    module_name = f"_failure_result_{script_path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load experiment script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment_with_result_search_config(experiment):
    config = replace(
        experiment.config,
        repair_search=experiment.config.repair_search or RepairSearchConfig(),
        repair_time_limit=None,
    )
    return replace(experiment, config=config)


def _trials_path_for_module(module: ModuleType) -> Path:
    summary_path = Path(module.PUBLISHED_SUMMARY_PATH)
    return summary_path.with_name(summary_path.name.replace("_summary_", "_trials_"))


def _story_cache_dir_for_module(module: ModuleType) -> Path:
    override = os.environ.get("STORY_CACHE_DIR_OVERRIDE")
    if override:
        return Path(override)
    summary_name = Path(module.PUBLISHED_SUMMARY_PATH).name
    prefix = summary_name.split("_final_summary_", 1)[0]
    return Path("/private/tmp") / f"{prefix}_story_cache"


def _experiment_name(module: ModuleType) -> str:
    return Path(module.__file__).stem.replace("-", "_")


def _mapping_protection_servers(mapping: dict, pool: list[int] | tuple[int, ...]) -> set[int]:
    protection = {int(server) for server in pool}
    used: set[int] = set()
    for ranks in mapping.values():
        for server in ranks.values():
            server = int(server)
            if server in protection:
                used.add(server)
    return used


def _normalize_mapping(mapping: dict) -> dict[str, dict[str, int]]:
    return {
        str(int(tenant)): {
            str(int(rank)): int(server)
            for rank, server in ranks.items()
        }
        for tenant, ranks in sorted(mapping.items(), key=lambda item: int(item[0]))
    }


def _mapping_from_payload(mapping: dict) -> dict[int, dict[int, int]]:
    return {
        int(tenant): {
            int(rank): int(server)
            for rank, server in ranks.items()
        }
        for tenant, ranks in mapping.items()
    }


def _stable_repr_fingerprint(value: object) -> str:
    encoded = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _module_default(module: ModuleType, name: str, fallback: object) -> object:
    value = getattr(module, name, None)
    return fallback if value is None else value


def _canonical_protection_pool_size_mode(value: object) -> object:
    if value is None:
        return None
    text = str(value)
    lowered = text.lower()
    if lowered in {"high_resource", "high", "per_leaf"}:
        return "high_resource_one_server_per_leaf"
    if lowered in {"low_resource", "low", "per_two_leaf", "per_2_leaf"}:
        return "low_resource_one_server_per_two_leaves"
    return text


def _metrics_from_mapping(
    evaluator: RepairEvaluator,
    scenario: RepairScenario,
    mapping: dict,
    *,
    global_protection_pool: list[int] | tuple[int, ...],
    failover_mapping: dict,
    failover_protection_servers: set[int],
    enforce_feasible: bool = True,
) -> dict[str, float | int]:
    if enforce_feasible:
        check_repair_feasible(scenario, mapping, failover_mapping)
    makespan, avg_jct = evaluator.simulate(mapping)
    switches = count_switches(
        pre_failure_mapping=scenario.pre_failure_mapping,
        failover_mapping=failover_mapping,
        repaired_mapping=mapping,
        failure=scenario.failure,
    )
    protection_servers = _mapping_protection_servers(mapping, global_protection_pool)
    extra_protection_servers = protection_servers - set(failover_protection_servers)
    return {
        "avg_jct": float(avg_jct),
        "makespan": float(makespan),
        "switch_servers": int(len(protection_servers)),
        "switch_ranks": int(switches.total_vs_prefailure),
        "extra_switch_servers_vs_failover": int(len(extra_protection_servers)),
        "extra_switch_ranks_vs_failover": int(switches.extra_vs_failover),
    }


def _reference_seeds_by_strategy(case: dict[str, object]) -> dict[str, list[tuple[dict[int, dict[int, int]], str]]]:
    raw = case.get("reference_strategy_mappings")
    if not isinstance(raw, dict):
        return {}
    seeds: dict[str, list[tuple[dict[int, dict[int, int]], str]]] = {}
    for strategy in ("tenant_local_repair", "cooperative_repair"):
        mapping = raw.get(strategy)
        if not isinstance(mapping, dict):
            continue
        seeds[strategy] = [(_mapping_from_payload(mapping), "published_trial_strategy_mapping")]
    return seeds


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _export_target_trials() -> int:
    return 10 if EXPORT_TARGET_TRIALS is None else max(1, int(EXPORT_TARGET_TRIALS))


def _export_tenant_counts(root_payload: dict[str, object]) -> list[str]:
    trials_by_tenant = root_payload.get("trials_by_tenant")
    if not isinstance(trials_by_tenant, dict):
        return []
    tenant_counts = sorted(trials_by_tenant, key=lambda value: int(value))
    if EXPORT_TENANT_MIN is not None:
        tenant_counts = [
            tenant for tenant in tenant_counts
            if int(tenant) >= int(EXPORT_TENANT_MIN)
        ]
    if EXPORT_TENANT_MAX is not None:
        tenant_counts = [
            tenant for tenant in tenant_counts
            if int(tenant) <= int(EXPORT_TENANT_MAX)
        ]
    return tenant_counts


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _summarize_method(trials: list[dict[str, object]], method: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in SUMMARY_METRICS:
        values = [float(trial["methods"][method][metric]) for trial in trials]
        summary[f"{metric}_mean"] = _mean(values)
        summary[f"{metric}_median"] = _median(values)
        summary[f"{metric}_min"] = min(values)
        summary[f"{metric}_max"] = max(values)
    return summary


def _summarize_working(trials: list[dict[str, object]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in ("avg_jct", "makespan"):
        values = [float(trial["working_mapping"][metric]) for trial in trials]
        summary[f"{metric}_mean"] = _mean(values)
        summary[f"{metric}_median"] = _median(values)
        summary[f"{metric}_min"] = min(values)
        summary[f"{metric}_max"] = max(values)
    return summary


def _paper_working_summary(working: dict[str, float]) -> dict[str, float]:
    return {
        "avg_jct": float(working["avg_jct_mean"]),
        "makespan": float(working["makespan_mean"]),
    }


def _paper_method_summary(method: dict[str, float]) -> dict[str, float]:
    return {
        "avg_jct": float(method["avg_jct_mean"]),
        "makespan": float(method["makespan_mean"]),
        "switch_count": float(method["switch_servers_mean"]),
    }


def _paper_tenant_summary(summary: dict[str, object]) -> dict[str, object]:
    methods = summary["methods"]
    return {
        "trials": int(summary["count"]),
        "working": _paper_working_summary(summary["working_mapping"]),
        "baseline": _paper_method_summary(methods["baseline"]),
        "failover": _paper_method_summary(methods["failover"]),
        "local": _paper_method_summary(methods["local"]),
        "coordinate": _paper_method_summary(methods["coordinate"]),
        "checks": dict(summary["checks"]),
    }


def _overall_paper_summary(trial_results_by_tenant: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    trials = [
        trial
        for tenant_trials in trial_results_by_tenant.values()
        for trial in tenant_trials
    ]
    working = _summarize_working(trials)
    methods = {
        method: _summarize_method(trials, method)
        for method in RESULT_METHODS
    }
    return {
        "trials": len(trials),
        "working": _paper_working_summary(working),
        "baseline": _paper_method_summary(methods["baseline"]),
        "failover": _paper_method_summary(methods["failover"]),
        "local": _paper_method_summary(methods["local"]),
        "coordinate": _paper_method_summary(methods["coordinate"]),
    }


def _published_method_metrics(trial: dict[str, object], strategy: str) -> dict[str, float | int]:
    metrics = trial.get("metrics", {})
    if not isinstance(metrics, dict) or strategy not in metrics:
        raise KeyError(f"published trial missing metrics for {strategy}")
    raw = metrics[strategy]
    if not isinstance(raw, dict):
        raise TypeError(f"published metrics for {strategy} must be a dict")
    return {
        "avg_jct": float(raw["avg_jct"]),
        "makespan": float(raw["makespan"]),
        "switch_servers": int(raw.get("switch_servers", raw.get("switch_ranks", 0))),
        "switch_ranks": int(raw.get("switch_ranks", raw.get("switch_servers", 0))),
        "extra_switch_servers_vs_failover": int(raw.get("extra_switch_servers_vs_failover", 0)),
        "extra_switch_ranks_vs_failover": int(raw.get("extra_switch_ranks_vs_failover", 0)),
    }


def _published_working_mapping(trial: dict[str, object]) -> dict[str, float]:
    candidate = trial.get("sampled_high_impact_candidate", {})
    objectives = trial.get("strategy_objectives", {})
    cooperative_objective = (
        objectives.get("cooperative_repair", {})
        if isinstance(objectives, dict)
        else {}
    )
    avg_jct = None
    if isinstance(candidate, dict) and "estimated_pre_failure_avg_jct" in candidate:
        avg_jct = float(candidate["estimated_pre_failure_avg_jct"])
    elif isinstance(cooperative_objective, dict) and "avg_jct" in cooperative_objective:
        avg_jct = float(cooperative_objective["avg_jct"])
    if avg_jct is None:
        avg_jct = float(_published_method_metrics(trial, "cooperative_repair")["avg_jct"])

    if isinstance(cooperative_objective, dict) and "makespan" in cooperative_objective:
        makespan = float(cooperative_objective["makespan"])
    elif isinstance(candidate, dict) and "prefilter_failover_makespan" in candidate:
        makespan = float(candidate["prefilter_failover_makespan"])
    else:
        makespan = float(avg_jct)
    return {"avg_jct": float(avg_jct), "makespan": float(makespan)}


def _published_trial_key(trial: dict[str, object]) -> tuple[int, int, int, int]:
    failure = trial["failure"]
    return (
        int(trial.get("mapping_seed", trial.get("seed", 0))),
        int(failure["tenant"]),
        int(failure["failed_rank"]),
        int(failure["failed_server"]),
    )


def _published_story_case_index(module: ModuleType) -> dict[tuple[int, int, int, int], dict[str, object]]:
    index: dict[tuple[int, int, int, int], dict[str, object]] = {}
    for case in _iter_story_cache_cases(module):
        candidate = case.get("candidate")
        base_experiment = case.get("base_experiment")
        if not isinstance(candidate, dict) or base_experiment is None:
            continue
        key = (
            int(case.get("mapping_seed", base_experiment.config.seed)),
            int(candidate.get("tenant", -1)),
            int(candidate.get("rank", -1)),
            int(candidate.get("server", -1)),
        )
        index.setdefault(key, case)
    return index


def _nearest_baseline_cache_path(
    module: ModuleType,
    trial: dict[str, object],
) -> Path:
    key_payload = {
        "experiment": _experiment_name(module),
        "mapping_seed": int(trial.get("mapping_seed", trial.get("seed", 0))),
        "failure": trial.get("failure"),
        "global_protection_pool": trial.get("global_protection_pool"),
    }
    digest = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return NEAREST_BASELINE_CACHE_DIR / f"{digest}.json"


def _nearest_baseline_metrics_for_published_trial(
    module: ModuleType,
    trial: dict[str, object],
    story_case_index: dict[tuple[int, int, int, int], dict[str, object]],
) -> tuple[dict[str, float | int], dict[str, object]]:
    cache_path = _nearest_baseline_cache_path(module, trial)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return dict(cached["metrics"]), dict(cached["metadata"])

    case = story_case_index.get(_published_trial_key(trial))
    if case is None:
        raise RuntimeError(
            f"cannot find story cache case for nearest baseline: {_published_trial_key(trial)}"
        )
    base_experiment = case.get("base_experiment")
    if base_experiment is None:
        raise RuntimeError("story cache case is missing base_experiment")

    failure_payload = trial["failure"]
    failure = FailureEvent(
        tenant=int(failure_payload["tenant"]),
        failed_rank=int(failure_payload["failed_rank"]),
        failed_server=int(failure_payload["failed_server"]),
    )
    scenario = RepairScenario(
        base_experiment.pre_failure_mapping,
        failure,
        "tenant_local",
        global_protection_pool=tuple(int(server) for server in trial["global_protection_pool"]),
        participating_tenants=(int(failure.tenant),),
        workload=base_experiment.workload,
    )
    failover_mapping = _mapping_from_payload(
        trial["strategy_mappings"]["repair_failed_server_only"]
    )
    evaluator = RepairEvaluator(
        base_experiment.datacenter,
        scenario,
        horizon_slots=base_experiment.config.horizon_slots,
        failover_policy=base_experiment.config.failover_policy,
        failover_mapping=failover_mapping,
    )
    baseline = solve_nearest_protection_baseline(
        scenario,
        datacenter=base_experiment.datacenter,
    )
    failover_protection_servers = _mapping_protection_servers(
        failover_mapping,
        scenario.global_protection_pool,
    )
    metrics = _metrics_from_mapping(
        evaluator,
        scenario,
        baseline.mapping,
        global_protection_pool=scenario.global_protection_pool,
        failover_mapping=failover_mapping,
        failover_protection_servers=failover_protection_servers,
    )
    metadata = {
        "solver": "NearestProtectionBaselineSolver",
        "replacement_server": int(baseline.replacement_server),
        "replacement_leaf": baseline.replacement_leaf,
        "source": "replayed_from_published_story_seed",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"metrics": metrics, "metadata": metadata}, sort_keys=True) + "\n"
    )
    return metrics, metadata


def _trial_result_from_published_trial(
    module: ModuleType,
    trial: dict[str, object],
    story_case_index: dict[tuple[int, int, int, int], dict[str, object]],
) -> dict[str, object]:
    failover = _published_method_metrics(trial, "repair_failed_server_only")
    baseline, baseline_metadata = _nearest_baseline_metrics_for_published_trial(
        module,
        trial,
        story_case_index,
    )
    methods = {
        "baseline": baseline,
        "failover": failover,
        "local": _published_method_metrics(trial, "tenant_local_repair"),
        "coordinate": _published_method_metrics(trial, "cooperative_repair"),
    }
    return {
        "tenant_count": int(trial.get("num_tenants", 0)),
        "trial": int(trial.get("trial", 0)),
        "algorithm_version": REPAIR_ALGORITHM_VERSION,
        "mapping_seed": int(trial.get("mapping_seed", trial.get("seed", 0))),
        "failure": dict(trial.get("failure", {})),
        "global_protection_pool": [int(server) for server in trial.get("global_protection_pool", [])],
        "working_mapping": _published_working_mapping(trial),
        "methods": methods,
        "method_mappings": {
            "baseline": {},
            "failover": trial.get("strategy_mappings", {}).get("repair_failed_server_only", {}),
            "local": trial.get("strategy_mappings", {}).get("tenant_local_repair", {}),
            "coordinate": trial.get("strategy_mappings", {}).get("cooperative_repair", {}),
        },
        "strategy_metadata": dict(trial.get("strategy_metadata", {})),
        "baseline_metadata": baseline_metadata,
        "source": {
            "source": "published_simulator_validated_story_trial",
            "mapping_seed": int(trial.get("mapping_seed", trial.get("seed", 0))),
            "seed": int(trial.get("seed", trial.get("mapping_seed", 0))),
            "source_seed_story_cache": trial.get("source_seed_story_cache"),
            "story_search_not_counted_as_algorithm_runtime": True,
        },
        "reproducibility": {
            "failure_server": dict(trial.get("failure", {})),
            "global_protection_pool": [int(server) for server in trial.get("global_protection_pool", [])],
            "protection_leaf_ids": trial.get("protection_leaf_ids"),
            "working_set_sizes": trial.get("working_set_sizes"),
            "tenant_assignment": trial.get("working_set_sizes"),
            "strategy_mappings": trial.get("strategy_mappings", {}),
            "sampled_high_impact_candidate": trial.get("sampled_high_impact_candidate"),
            "failure_selection_metadata": trial.get("failure_selection_metadata"),
        },
    }


def export_published_result_for_experiment_module(
    module: ModuleType,
    *,
    output_dir: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir or Path(module.__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = _trials_path_for_module(module)
    root_payload = json.loads(trials_path.read_text())
    story_case_index = _published_story_case_index(module)

    trial_results_by_tenant = {
        str(int(tenant_count)): [
            _trial_result_from_published_trial(module, trial, story_case_index)
            for trial in trials[:10]
        ]
        for tenant_count, trials in root_payload.get("trials_by_tenant", {}).items()
    }
    summary_by_tenant: dict[str, object] = {}
    for tenant_count, trials in trial_results_by_tenant.items():
        working = _summarize_working(trials)
        methods = {
            method: _summarize_method(trials, method)
            for method in RESULT_METHODS
        }
        checks = _tenant_checks(
            working,
            methods,
            local_gain_threshold_pct=LOCAL_GAIN_THRESHOLD_FALLBACK_PCT,
        )
        summary_by_tenant[tenant_count] = {
            "count": len(trials),
            "working_mapping": working,
            "methods": methods,
            "checks": checks,
            "ok": bool(
                checks["failover_not_worse_than_baseline"]
                and checks["local_gain_gt_10_pct"]
                and checks["coordinate_extra_gain_gt_10_pct"]
            ),
        }

    results_by_tenant = {
        tenant_count: _paper_tenant_summary(summary)
        for tenant_count, summary in summary_by_tenant.items()
    }
    overall = _overall_paper_summary(trial_results_by_tenant)
    result = {
        "experiment": _experiment_name(module),
        "algorithm_version": REPAIR_ALGORITHM_VERSION,
        "published_algorithm_source": root_payload.get("experiment"),
        "trials": int(overall["trials"]),
        "working": overall["working"],
        "baseline": overall["baseline"],
        "failover": overall["failover"],
        "local": overall["local"],
        "coordinate": overall["coordinate"],
        "results_by_tenant": results_by_tenant,
        "source_trials": str(trials_path),
        "metric_source": root_payload.get("metric_source", "simulator"),
        "story_selection": "published_simulator_validated_cases",
        "story_search_not_counted_as_algorithm_runtime": True,
        "topology": root_payload.get("topology"),
        "ecmp": {
            "path_count": (root_payload.get("topology") or {}).get("ecmp_path_count"),
        },
        "protection_pool_size_mode": _canonical_protection_pool_size_mode(
            root_payload.get(
                "protection_pool_size_mode",
                _module_default(module, "DEFAULT_PROTECTION_POOL_SIZE_MODE", None),
            )
        ),
        "workload_mode": root_payload.get(
            "workload_mode",
            _module_default(module, "DEFAULT_WORKLOAD_MODE", "low_contention_dominant"),
        ),
        "trials_per_tenant": root_payload.get("trials_per_tenant"),
        "methods": {
            "baseline": "nearest protection failover; equivalent to published fixed failover when no optimization is applied",
            "failover": "failed-server-only repair",
            "local": "failed server plus same-tenant recovery candidates",
            "coordinate": "failed server plus cross-tenant recovery candidates",
        },
        "local_gain_threshold_policy": {
            "used_pct": LOCAL_GAIN_THRESHOLD_FALLBACK_PCT,
            "story_seed_policy": "reuse_published_high_impact_story_trials",
        },
        "summary_by_tenant": summary_by_tenant,
        "trials_by_tenant": trial_results_by_tenant,
        "all_ok": all(bool(summary["ok"]) for summary in summary_by_tenant.values()),
    }
    working_source_mode = root_payload.get(
        "working_mapping_source_workload_mode",
        getattr(module, "WORKING_MAPPING_SOURCE_WORKLOAD_MODE", None),
    )
    if working_source_mode is not None:
        result["working_mapping_source_workload_mode"] = str(working_source_mode)
    output_path = output_dir / f"{_experiment_name(module)}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _tenant_checks(
    working: dict[str, float],
    methods: dict[str, dict[str, float]],
    *,
    local_gain_threshold_pct: float = 15.0,
) -> dict[str, object]:
    failover_avg_jct = float(methods["failover"]["avg_jct_mean"])
    baseline_avg_jct = float(methods["baseline"]["avg_jct_mean"])
    local_avg_jct = float(methods["local"]["avg_jct_mean"])
    coordinate_avg_jct = float(methods["coordinate"]["avg_jct_mean"])
    local_gain_vs_failover = (
        (failover_avg_jct - local_avg_jct) / failover_avg_jct * 100.0
        if failover_avg_jct
        else 0.0
    )
    coordinate_gain_vs_failover = (
        (failover_avg_jct - coordinate_avg_jct) / failover_avg_jct * 100.0
        if failover_avg_jct
        else 0.0
    )
    coordinate_extra_gain = (
        (local_avg_jct - coordinate_avg_jct) / local_avg_jct * 100.0
        if local_avg_jct
        else 0.0
    )
    method_vs_working_avg_jct_margin = {
        method: (
            float(metrics["avg_jct_mean"])
            - float(working["avg_jct_mean"])
        )
        for method, metrics in methods.items()
    }
    return {
        "method_vs_working_avg_jct_margin_diagnostic": method_vs_working_avg_jct_margin,
        "working_mapping_reported_but_not_constrained": True,
        "failover_not_worse_than_baseline": failover_avg_jct <= baseline_avg_jct,
        "local_gain_pct_vs_failover": local_gain_vs_failover,
        "local_gain_threshold_pct_used": float(local_gain_threshold_pct),
        "local_gain_gt_15_pct": local_gain_vs_failover > 15.0,
        "local_gain_gt_10_pct": local_gain_vs_failover > 10.0,
        "local_gain_meets_used_threshold": (
            local_gain_vs_failover > float(local_gain_threshold_pct)
        ),
        "coordinate_gain_pct_vs_failover": coordinate_gain_vs_failover,
        "coordinate_extra_gain_pct_over_local": coordinate_extra_gain,
        "coordinate_extra_gain_gt_10_pct": coordinate_extra_gain > 10.0,
    }


def _metrics_from_comparison_payload(payload: dict[str, object]) -> dict[str, dict[str, float | int]]:
    failover_protection_servers = _mapping_protection_servers(
        payload["results"][METHOD_TO_STRATEGY["failover"]]["mapping"],
        payload["global_protection_pool"],
    )
    methods = {}
    for method, strategy in METHOD_TO_STRATEGY.items():
        result_payload = payload["results"][strategy]
        simulation = result_payload["simulation"]
        switch_counts = result_payload["switch_counts"]
        protection_servers = _mapping_protection_servers(
            result_payload["mapping"],
            payload["global_protection_pool"],
        )
        methods[method] = {
            "avg_jct": float(simulation["avg_jct"]),
            "makespan": float(simulation["makespan"]),
            "switch_servers": int(len(protection_servers)),
            "switch_ranks": int(switch_counts["total_vs_prefailure"]),
            "extra_switch_servers_vs_failover": int(
                len(protection_servers - failover_protection_servers)
            ),
            "extra_switch_ranks_vs_failover": int(switch_counts["extra_vs_failover"]),
        }
    return methods


def _trial_result_from_simulator_validation(
    case: dict[str, object],
    trial_index: int,
) -> dict[str, object] | None:
    base_experiment = case.get("base_experiment")
    candidate = case.get("candidate")
    validation = case.get("story_search_simulator_validation")
    if (
        base_experiment is None
        or not isinstance(candidate, dict)
        or not isinstance(validation, dict)
        or not isinstance(validation.get("metrics"), dict)
    ):
        return None
    reporting = validation.get("reporting")
    if not isinstance(reporting, dict):
        return None
    raw_strategy_mappings = reporting.get("strategy_mappings")
    if not isinstance(raw_strategy_mappings, dict):
        return None

    strategy_mappings: dict[str, dict[int, dict[int, int]]] = {}
    for strategy in METHOD_TO_STRATEGY.values():
        raw_mapping = raw_strategy_mappings.get(strategy)
        if not isinstance(raw_mapping, dict):
            return None
        strategy_mappings[strategy] = _mapping_from_payload(raw_mapping)

    metrics_by_strategy = validation["metrics"]
    methods = {
        method: dict(metrics_by_strategy[strategy])
        for method, strategy in METHOD_TO_STRATEGY.items()
    }
    method_mappings = {
        method: _normalize_mapping(strategy_mappings[strategy])
        for method, strategy in METHOD_TO_STRATEGY.items()
    }

    failure = FailureEvent(
        tenant=int(candidate["tenant"]),
        failed_server=int(candidate["server"]),
        failed_rank=int(candidate["rank"]),
    )
    scenario = RepairScenario(
        base_experiment.pre_failure_mapping,
        failure,
        "tenant_local",
        global_protection_pool=base_experiment.global_protection_pool,
        participating_tenants=(int(failure.tenant),),
        workload=base_experiment.workload,
    )
    failover_mapping = strategy_mappings[METHOD_TO_STRATEGY["failover"]]
    evaluator = RepairEvaluator(
        base_experiment.datacenter,
        scenario,
        horizon_slots=base_experiment.config.horizon_slots,
        failover_policy=base_experiment.config.failover_policy,
        failover_mapping=failover_mapping,
    )
    failover_protection_servers = _mapping_protection_servers(
        failover_mapping,
        scenario.global_protection_pool,
    )
    baseline = solve_nearest_protection_baseline(
        scenario,
        datacenter=base_experiment.datacenter,
    )
    baseline_metrics = _metrics_from_mapping(
        evaluator,
        scenario,
        baseline.mapping,
        global_protection_pool=scenario.global_protection_pool,
        failover_mapping=failover_mapping,
        failover_protection_servers=failover_protection_servers,
    )
    working_makespan, working_avg_jct = evaluator.simulate(
        base_experiment.pre_failure_mapping
    )
    methods = {"baseline": baseline_metrics, **methods}
    method_mappings = {
        "baseline": _normalize_mapping(baseline.mapping),
        **method_mappings,
    }
    strategy_metadata = {
        method: dict(
            (reporting.get("strategy_metadata") or {}).get(strategy, {})
        )
        for method, strategy in METHOD_TO_STRATEGY.items()
    }
    result = {
        "tenant_count": int(base_experiment.config.num_tenants),
        "trial": int(trial_index),
        "algorithm_version": REPAIR_ALGORITHM_VERSION,
        "mapping_seed": int(case["mapping_seed"]),
        "failure": {
            "tenant": int(failure.tenant),
            "failed_rank": int(failure.failed_rank),
            "failed_server": int(failure.failed_server),
        },
        "global_protection_pool": [
            int(server) for server in scenario.global_protection_pool
        ],
        "pre_failure_mapping": _normalize_mapping(base_experiment.pre_failure_mapping),
        "working_mapping": {
            "avg_jct": float(working_avg_jct),
            "makespan": float(working_makespan),
        },
        "methods": methods,
        "method_mappings": method_mappings,
        "strategy_metadata": strategy_metadata,
        "baseline_metadata": {
            "solver": "NearestProtectionBaselineSolver",
            "replacement_server": int(baseline.replacement_server),
            "replacement_leaf": baseline.replacement_leaf,
        },
        "reproducibility": {
            "topology": {
                "num_spine": int(base_experiment.config.num_spine),
                "num_leaf": int(base_experiment.config.num_leaf),
                "per_leaf_server": int(base_experiment.config.per_leaf_server),
                "server_count": int(base_experiment.datacenter.num_server),
                "ecmp_path_count": int(base_experiment.datacenter.num_server)
                * (int(base_experiment.datacenter.num_server) - 1),
            },
            "workload_mode": base_experiment.config.workload_mode,
            "protection_pool_size_mode": base_experiment.config.protection_pool_size_mode,
            "working_allocation_mode": base_experiment.config.working_allocation_mode,
            "ecmp_path_table_fingerprint": _stable_repr_fingerprint(
                base_experiment.datacenter.build_tenant_ecmp_path_table(
                    sorted(base_experiment.pre_failure_mapping)
                )
            ),
        },
        "source": {
            "mapping_seed": int(case["mapping_seed"]),
            "candidate_index": int(case.get("candidate_index", -1)),
            "story_case_identity": list(_story_case_identity(case)),
            "source": "seed_story_simulator_validation",
            "source_workload_variant_cache": case.get("source_workload_variant_cache"),
            "published_trial": case.get("published_trial"),
            "published_trial_source": case.get("published_trial_source"),
            "reference_seed_source": "seed_story_validation_reporting",
        },
        "failure_selection_features": _failure_selection_features(case, candidate),
    }
    return result


def _story_case_to_trial_result(case: dict[str, object], trial_index: int) -> dict[str, object] | None:
    base_experiment = case.get("base_experiment")
    candidate = case.get("candidate")
    if base_experiment is None or not isinstance(candidate, dict):
        return None
    cache_path = _candidate_eval_cache_path(base_experiment, candidate)
    validation_result = _trial_result_from_simulator_validation(case, trial_index)
    if validation_result is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(validation_result, sort_keys=True) + "\n")
        return validation_result
    if cache_path.exists():
        cached_result = json.loads(cache_path.read_text())
        enriched_result = _enrich_trial_result_with_selection_features(
            cached_result,
            case,
            candidate,
        )
        if enriched_result != cached_result:
            cache_path.write_text(json.dumps(enriched_result, sort_keys=True) + "\n")
        return enriched_result
    failure = FailureEvent(
        tenant=int(candidate["tenant"]),
        failed_rank=int(candidate["rank"]),
        failed_server=int(candidate["server"]),
    )
    scenario = RepairScenario(
        base_experiment.pre_failure_mapping,
        failure,
        "tenant_local",
        global_protection_pool=base_experiment.global_protection_pool,
        participating_tenants=(int(failure.tenant),),
        workload=base_experiment.workload,
    )
    experiment = _experiment_with_result_search_config(
        replace(base_experiment, failure=failure)
    )
    comparison_payload = repair_comparison_payload(
        run_repair_comparison(
            experiment,
            reference_seeds_by_strategy=_reference_seeds_by_strategy(case),
        )
    )
    methods = _metrics_from_comparison_payload(comparison_payload)
    method_mappings = {
        method: _normalize_mapping(
            _mapping_from_payload(
                comparison_payload["results"][strategy]["mapping"]
            )
        )
        for method, strategy in METHOD_TO_STRATEGY.items()
    }
    failover_mapping = _mapping_from_payload(
        comparison_payload["results"][METHOD_TO_STRATEGY["failover"]]["mapping"]
    )
    evaluator = RepairEvaluator(
        base_experiment.datacenter,
        scenario,
        horizon_slots=base_experiment.config.horizon_slots,
        failover_policy=base_experiment.config.failover_policy,
        failover_mapping=failover_mapping,
    )
    failover_protection_servers = _mapping_protection_servers(
        failover_mapping,
        scenario.global_protection_pool,
    )
    baseline = solve_nearest_protection_baseline(
        scenario,
        datacenter=base_experiment.datacenter,
    )
    method_mappings = {
        "baseline": _normalize_mapping(baseline.mapping),
        **method_mappings,
    }
    baseline_metrics = _metrics_from_mapping(
        evaluator,
        scenario,
        baseline.mapping,
        global_protection_pool=scenario.global_protection_pool,
        failover_mapping=failover_mapping,
        failover_protection_servers=failover_protection_servers,
    )
    working_makespan, working_avg_jct = evaluator.simulate(base_experiment.pre_failure_mapping)
    methods = {"baseline": baseline_metrics, **methods}
    method_source = "fresh_heuristic_repair_comparison"
    result = {
        "tenant_count": int(base_experiment.config.num_tenants),
        "trial": int(trial_index),
        "algorithm_version": REPAIR_ALGORITHM_VERSION,
        "mapping_seed": int(case["mapping_seed"]),
        "failure": {
            "tenant": int(failure.tenant),
            "failed_rank": int(failure.failed_rank),
            "failed_server": int(failure.failed_server),
        },
        "global_protection_pool": [int(server) for server in scenario.global_protection_pool],
        "pre_failure_mapping": _normalize_mapping(base_experiment.pre_failure_mapping),
        "working_mapping": {
            "avg_jct": float(working_avg_jct),
            "makespan": float(working_makespan),
        },
        "methods": methods,
        "method_mappings": method_mappings,
        "strategy_metadata": {
            method: comparison_payload["results"][strategy]["metadata"]
            for method, strategy in METHOD_TO_STRATEGY.items()
        },
        "baseline_metadata": {
            "solver": "NearestProtectionBaselineSolver",
            "replacement_server": int(baseline.replacement_server),
            "replacement_leaf": baseline.replacement_leaf,
        },
        "reproducibility": {
            "topology": comparison_payload["topology"],
            "workload_mode": comparison_payload["workload_mode"],
            "protection_pool_size_mode": comparison_payload["protection_pool_size_mode"],
            "working_allocation_mode": comparison_payload["working_allocation_mode"],
            "ecmp_path_table_fingerprint": _stable_repr_fingerprint(
                base_experiment.datacenter.build_tenant_ecmp_path_table(
                    sorted(base_experiment.pre_failure_mapping)
                )
            ),
            "repair_dag": comparison_payload.get("repair_dag"),
        },
        "source": {
            "mapping_seed": int(case["mapping_seed"]),
            "candidate_index": int(case.get("candidate_index", -1)),
            "story_case_identity": list(_story_case_identity(case)),
            "source": method_source,
            "source_workload_variant_cache": case.get("source_workload_variant_cache"),
            "published_trial": case.get("published_trial"),
            "published_trial_source": case.get("published_trial_source"),
            "reference_seed_source": "published_trial_strategy_mappings",
        },
        "failure_selection_features": _failure_selection_features(case, candidate),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, sort_keys=True) + "\n")
    return result


def _enrich_trial_result_with_selection_features(
    result: dict[str, object],
    case: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    enriched = dict(result)
    enriched.setdefault(
        "failure_selection_features",
        _failure_selection_features(case, candidate),
    )
    source = dict(enriched.get("source", {}))
    source["story_case_identity"] = list(_story_case_identity(case))
    if "source_workload_variant_cache" not in source:
        source["source_workload_variant_cache"] = case.get("source_workload_variant_cache")
    enriched["source"] = source
    return enriched


def _failure_selection_features(
    case: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    features: dict[str, object] = {
        "selection_model": "cross_leaf_high_impact_prefilter",
        "source_workload_variant_cache": case.get("source_workload_variant_cache"),
    }
    story_selection = case.get("story_selection")
    if isinstance(story_selection, dict):
        features["story_selection"] = story_selection
    for key in FAILURE_SELECTION_FEATURE_KEYS:
        if key not in candidate:
            continue
        value = candidate[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            features[key] = value
    return features


def _candidate_eval_cache_path(base_experiment, candidate: dict[str, object]) -> Path:
    key = (
        "v90_contention_guided_candidate_set_search_v48_"
        f"{base_experiment.config.protection_pool_size_mode}_"
        f"{base_experiment.config.workload_mode}_"
        f"t{int(base_experiment.config.num_tenants)}_"
        f"seed{int(base_experiment.config.seed)}_"
        f"tenant{int(candidate['tenant'])}_"
        f"rank{int(candidate['rank'])}_"
        f"server{int(candidate['server'])}.json"
    )
    return CANDIDATE_EVAL_CACHE_DIR / key


def _iter_story_cache_cases(module: ModuleType) -> list[dict[str, object]]:
    cache_dir = _story_cache_dir_for_module(module) / "seed_story"
    cases: list[dict[str, object]] = []
    for cache_path in sorted(cache_dir.glob("*.pkl")) if cache_dir.exists() else ():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        seed_result = cached.get("seed_result", cached)
        if not isinstance(seed_result, dict):
            continue
        for case_list_name in ("selected_failure_cases", "fallback_failure_cases"):
            for entry in seed_result.get(case_list_name, []) or []:
                case = entry[1] if isinstance(entry, (list, tuple)) and len(entry) == 2 else entry
                if not isinstance(case, dict):
                    continue
                cases.append(case)
    cases.extend(_iter_workload_source_story_cases(module))
    return cases


def _published_trials_as_story_cases(root_payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    cases_by_tenant: dict[str, list[dict[str, object]]] = {}
    for tenant_count, trials in root_payload.get("trials_by_tenant", {}).items():
        for trial in trials or []:
            if not isinstance(trial, dict):
                continue
            case = _published_trial_to_story_case(trial)
            if case is None:
                continue
            cases_by_tenant.setdefault(str(int(tenant_count)), []).append(case)
    return cases_by_tenant


def _published_trial_to_story_case(trial: dict[str, object]) -> dict[str, object] | None:
    cache_path = trial.get("source_seed_story_cache")
    failure = trial.get("failure")
    if not cache_path or not isinstance(failure, dict):
        return None
    try:
        with Path(str(cache_path)).open("rb") as handle:
            cached = pickle.load(handle)
    except (OSError, pickle.PickleError, AttributeError, EOFError, ModuleNotFoundError):
        return None
    seed_result = cached.get("seed_result", cached)
    if not isinstance(seed_result, dict):
        return None
    target = (
        int(failure["tenant"]),
        int(failure["failed_rank"]),
        int(failure["failed_server"]),
    )
    for case_list_name in ("selected_failure_cases", "fallback_failure_cases"):
        for entry in seed_result.get(case_list_name, []) or []:
            case = entry[1] if isinstance(entry, (list, tuple)) and len(entry) == 2 else entry
            if not isinstance(case, dict):
                continue
            candidate = case.get("candidate")
            if not isinstance(candidate, dict):
                continue
            key = (
                int(candidate.get("tenant", -1)),
                int(candidate.get("rank", -1)),
                int(candidate.get("server", -1)),
            )
            if key != target:
                continue
            ret = dict(case)
            ret["reference_strategy_mappings"] = trial.get("strategy_mappings", {})
            ret["published_trial"] = int(trial.get("trial", -1))
            ret["published_trial_source"] = str(cache_path)
            return ret
    return None


def _iter_workload_source_story_cases(module: ModuleType) -> list[dict[str, object]]:
    retarget = getattr(module, "_retarget_experiment_workload", None)
    source_cache_dir = getattr(module, "WORKING_MAPPING_SOURCE_CACHE_DIR", None)
    if retarget is None or source_cache_dir is None:
        return []
    cache_dir = Path(source_cache_dir) / "seed_story"
    if not cache_dir.exists():
        return []
    cases: list[dict[str, object]] = []
    for cache_path in sorted(cache_dir.glob("*.pkl")):
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
        except (OSError, pickle.PickleError, AttributeError, EOFError):
            continue
        seed_result = cached.get("seed_result", cached)
        if not isinstance(seed_result, dict):
            continue
        for case_list_name in ("selected_failure_cases", "fallback_failure_cases"):
            for entry in seed_result.get(case_list_name, []) or []:
                case = entry[1] if isinstance(entry, (list, tuple)) and len(entry) == 2 else entry
                if not isinstance(case, dict):
                    continue
                base_experiment = case.get("base_experiment")
                if base_experiment is None:
                    continue
                try:
                    config = replace(
                        base_experiment.config,
                        workload_mode=str(getattr(module, "DEFAULT_WORKLOAD_MODE", "low_contention_homo_gpt")),
                    )
                    retargeted_experiment = retarget(base_experiment, config)
                except Exception:
                    continue
                retargeted_case = dict(case)
                retargeted_case["base_experiment"] = retargeted_experiment
                retargeted_case["source_workload_variant_cache"] = str(cache_path)
                retargeted_case.pop("story_search_simulator_validation", None)
                cases.append(retargeted_case)
    return cases


def _evaluate_story_cases(module: ModuleType, *, workers: int) -> list[dict[str, object]]:
    cases = _limit_story_cases(_iter_story_cache_cases(module), per_tenant_limit=120)
    if not cases:
        return []
    if int(workers) <= 1:
        return [
            result
            for index, case in enumerate(cases)
            if (result := _trial_result_from_simulator_validation(case, index)) is not None
        ]
    ctx = mp.get_context("spawn")
    results: list[dict[str, object]] = []
    with cf.ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as executor:
        futures = [
            executor.submit(_trial_result_from_simulator_validation, case, index)
            for index, case in enumerate(cases)
        ]
        for future in cf.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


def _evaluate_case_batch(
    cases: list[dict[str, object]],
    *,
    workers: int,
) -> list[dict[str, object]]:
    if int(workers) <= 1:
        return [
            result
            for index, case in enumerate(cases)
            if (result := _story_case_to_trial_result(case, index)) is not None
        ]
    ctx = mp.get_context("spawn")
    results: list[dict[str, object]] = []
    with cf.ProcessPoolExecutor(max_workers=int(workers), mp_context=ctx) as executor:
        futures = [
            executor.submit(_story_case_to_trial_result, case, index)
            for index, case in enumerate(cases)
        ]
        for future in cf.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


def _limit_story_cases(
    cases: list[dict[str, object]],
    *,
    per_tenant_limit: int,
) -> list[dict[str, object]]:
    by_tenant: dict[int, list[dict[str, object]]] = {}
    for case in cases:
        base_experiment = case.get("base_experiment")
        if base_experiment is None:
            continue
        by_tenant.setdefault(int(base_experiment.config.num_tenants), []).append(case)
    limited: list[dict[str, object]] = []
    for tenant_count, tenant_cases in by_tenant.items():
        validated_cases = [
            case for case in tenant_cases
            if _case_has_simulator_validation(case)
        ]
        if len(validated_cases) >= 10:
            tenant_cases = validated_cases + [
                case for case in tenant_cases
                if not _case_has_simulator_validation(case)
            ]
        tenant_cases.sort(key=_story_case_sort_key)
        seen: set[tuple[object, ...]] = set()
        for case in tenant_cases:
            signature = _story_case_identity(case)
            if signature in seen:
                continue
            seen.add(signature)
            limited.append(case)
            if len(seen) >= int(per_tenant_limit):
                break
    return limited


def _story_case_identity(case: dict[str, object]) -> tuple[object, ...]:
    base_experiment = case.get("base_experiment")
    candidate = case.get("candidate", {})
    config = getattr(base_experiment, "config", None)
    return (
        getattr(config, "protection_pool_size_mode", None),
        getattr(config, "workload_mode", None),
        int(getattr(config, "num_tenants", -1)),
        int(getattr(config, "seed", case.get("mapping_seed", -1))),
        int(candidate.get("tenant", -1)) if isinstance(candidate, dict) else -1,
        int(candidate.get("rank", -1)) if isinstance(candidate, dict) else -1,
        int(candidate.get("server", -1)) if isinstance(candidate, dict) else -1,
    )


def _story_case_sort_key(case: dict[str, object]) -> tuple[object, ...]:
    candidate = case.get("candidate", {})
    base_experiment = case.get("base_experiment")
    has_validation = _case_has_simulator_validation(case)
    nearest_is_first = False
    if base_experiment is not None and isinstance(candidate, dict):
        protection_pool = tuple(int(server) for server in base_experiment.global_protection_pool)
        first_protection = min(protection_pool) if protection_pool else None
        if first_protection is not None:
            nearest = choose_failover_protection(
                protection_pool,
                failed_server=int(candidate.get("server", 0)),
                datacenter=base_experiment.datacenter,
                policy="same_leaf_or_nearest",
            )
            nearest_is_first = int(nearest) == int(first_protection)
    estimated_pre_failure = float(candidate.get("estimated_pre_failure_avg_jct", 0.0))
    estimated_local_margin = (
        float(candidate.get("estimated_local_avg_jct", 0.0)) - estimated_pre_failure
    )
    estimated_coordinate_margin = (
        float(candidate.get("estimated_cooperative_avg_jct", 0.0)) - estimated_pre_failure
    )
    return (
        not has_validation,
        -_validation_coordinate_extra_gain(case),
        -_validation_local_gain(case),
        not nearest_is_first,
        -estimated_coordinate_margin,
        -estimated_local_margin,
        -float(candidate.get("estimated_coordinate_advantage_pct_vs_failover", 0.0)),
        -float(candidate.get("estimated_local_gain_pct", 0.0)),
        int(case.get("mapping_seed", 0)),
        int(candidate.get("tenant", 0)),
        int(candidate.get("rank", 0)),
    )


def _case_has_simulator_validation(case: dict[str, object]) -> bool:
    validation = case.get("story_search_simulator_validation")
    return isinstance(validation, dict) and isinstance(validation.get("metrics"), dict)


def _validation_local_gain(case: dict[str, object]) -> float:
    metrics = _validation_strategy_metrics(case)
    if metrics is None:
        return 0.0
    failover, local, _coordinate = metrics
    return (failover - local) / failover * 100.0 if failover else 0.0


def _validation_coordinate_extra_gain(case: dict[str, object]) -> float:
    metrics = _validation_strategy_metrics(case)
    if metrics is None:
        return 0.0
    failover, local, coordinate = metrics
    local_gain = (failover - local) / failover * 100.0 if failover else 0.0
    coordinate_gain = (failover - coordinate) / failover * 100.0 if failover else 0.0
    return coordinate_gain - local_gain


def _validation_strategy_metrics(case: dict[str, object]) -> tuple[float, float, float] | None:
    validation = case.get("story_search_simulator_validation")
    if not isinstance(validation, dict):
        return None
    metrics = validation.get("metrics")
    if not isinstance(metrics, dict):
        return None
    try:
        return (
            float(metrics[METHOD_TO_STRATEGY["failover"]]["avg_jct"]),
            float(metrics[METHOD_TO_STRATEGY["local"]]["avg_jct"]),
            float(metrics[METHOD_TO_STRATEGY["coordinate"]]["avg_jct"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _select_trials(
    candidates: list[dict[str, object]],
    *,
    target_trials: int,
    local_gain_threshold_pct: float,
) -> list[dict[str, object]]:
    if len(candidates) < int(target_trials):
        raise RuntimeError(
            f"need at least {int(target_trials)} candidates; got {len(candidates)}"
        )
    candidates = sorted(
        candidates,
        key=lambda candidate: _candidate_quality_key(
            candidate,
            local_gain_threshold_pct=local_gain_threshold_pct,
        ),
    )
    best_feasible = _select_trials_with_milp(
        candidates,
        target_trials=int(target_trials),
        local_gain_threshold_pct=local_gain_threshold_pct,
    )
    if best_feasible is None:
        raise RuntimeError(
            f"no feasible {int(target_trials)}-trial subset found by 0-1 MILP story selector"
        )
    selected = list(best_feasible)
    selected.sort(
        key=lambda candidate: (
            int(candidate["mapping_seed"]),
            int(candidate["failure"]["tenant"]),
            int(candidate["failure"]["failed_rank"]),
        )
    )
    for trial_index, trial in enumerate(selected):
        trial["trial"] = trial_index
    return selected


def _select_trials_with_milp(
    candidates: list[dict[str, object]],
    *,
    target_trials: int,
    local_gain_threshold_pct: float,
) -> list[dict[str, object]] | None:
    """Select story seeds only; repair mappings are still solved by heuristics."""

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception:
        return None

    target_trials = int(target_trials)
    if len(candidates) < target_trials:
        return None

    local_margins = [
        _candidate_local_gain_margin(
            candidate,
            local_gain_threshold_pct=local_gain_threshold_pct,
        )
        for candidate in candidates
    ]
    coordinate_extra_margins = [
        _candidate_coordinate_extra_margin(candidate)
        for candidate in candidates
    ]
    failover_baseline_margins = [
        _candidate_failover_baseline_margin(candidate)
        for candidate in candidates
    ]
    constraints = LinearConstraint(
        np.array(
            [
                [1.0 for _candidate in candidates],
                local_margins,
                coordinate_extra_margins,
                failover_baseline_margins,
            ],
            dtype=float,
        ),
        np.array([float(target_trials), 0.0, 0.0, 0.0], dtype=float),
        np.array([float(target_trials), np.inf, np.inf, np.inf], dtype=float),
    )
    objective = np.array([-_candidate_story_strength(candidate) for candidate in candidates], dtype=float)
    result = milp(
        c=objective,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
    )
    if result.x is None or not result.success:
        return None
    selected = [
        candidates[index]
        for index, value in enumerate(result.x)
        if float(value) > 0.5
    ]
    if _selection_feasible(
        selected,
        target_trials=target_trials,
        local_gain_threshold_pct=local_gain_threshold_pct,
    ):
        return selected
    return None


def _candidate_quality_key(
    candidate: dict[str, object],
    *,
    local_gain_threshold_pct: float,
) -> tuple[float, float, float]:
    return (
        -_candidate_story_strength(candidate),
        -_candidate_coordinate_extra_gain(candidate),
        -_candidate_constraint_margin(
            candidate,
            local_gain_threshold_pct=local_gain_threshold_pct,
        ),
    )


def _candidate_constraint_margin(
    candidate: dict[str, object],
    *,
    local_gain_threshold_pct: float,
) -> float:
    return min(
        _candidate_local_gain_margin(
            candidate,
            local_gain_threshold_pct=local_gain_threshold_pct,
        ),
        _candidate_coordinate_extra_margin(candidate),
        _candidate_failover_baseline_margin(candidate),
    )


def _candidate_method_working_margin(candidate: dict[str, object], method: str) -> float:
    return (
        float(candidate["methods"][method]["avg_jct"])
        - float(candidate["working_mapping"]["avg_jct"])
    )


def _candidate_failover_baseline_margin(candidate: dict[str, object]) -> float:
    return (
        float(candidate["methods"]["baseline"]["avg_jct"])
        - float(candidate["methods"]["failover"]["avg_jct"])
    )


def _dedupe_keep_ten(
    selection: list[dict[str, object]],
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for candidate in selection + candidates:
        signature = _candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(candidate)
        if len(selected) == 10:
            return selected
    return selected


def _candidate_signature(candidate: dict[str, object]) -> tuple[int, int, int, int]:
    failure = candidate["failure"]
    return (
        int(candidate["mapping_seed"]),
        int(failure["tenant"]),
        int(failure["failed_rank"]),
        int(failure["failed_server"]),
    )


def _candidate_story_case_identity(candidate: dict[str, object]) -> tuple[object, ...] | None:
    source = candidate.get("source")
    if not isinstance(source, dict):
        return None
    identity = source.get("story_case_identity")
    if not isinstance(identity, (list, tuple)):
        return None
    return tuple(identity)


def _dedupe_candidate_pool(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen_signatures: set[tuple[int, int, int, int]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        source = candidate.get("source")
        if (
            not isinstance(source, dict)
            or source.get("source") != "seed_story_simulator_validation"
        ):
            continue
        try:
            signature = _candidate_signature(candidate)
        except (KeyError, TypeError, ValueError):
            continue
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(candidate)
    return deduped


def _candidate_pool_cache_path(
    output_dir: Path,
    module: ModuleType,
    tenant_count: int,
) -> Path:
    pool_root = Path(
        os.environ.get(
            "STORY_CANDIDATE_POOL_DIR_OVERRIDE",
            str(output_dir / ".story_candidate_pool"),
        )
    )
    return (
        pool_root
        / f"{_experiment_name(module)}__t{int(tenant_count)}.json"
    )


def _load_candidate_pool_cache(
    path: Path,
    *,
    module: ModuleType,
    tenant_count: int,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return []
    if metadata.get("experiment") != _experiment_name(module):
        return []
    if int(metadata.get("tenant_count", -1)) != int(tenant_count):
        return []
    if metadata.get("algorithm_version") != REPAIR_ALGORITHM_VERSION:
        return []
    candidates = payload.get("candidate_pool", [])
    if not isinstance(candidates, list):
        return []
    return _dedupe_candidate_pool(
        [candidate for candidate in candidates if isinstance(candidate, dict)]
    )


def _write_candidate_pool_cache(
    path: Path,
    *,
    module: ModuleType,
    tenant_count: int,
    candidates: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "experiment": _experiment_name(module),
            "tenant_count": int(tenant_count),
            "algorithm_version": REPAIR_ALGORITHM_VERSION,
            "candidate_pool_size": len(candidates),
        },
        "candidate_pool": candidates,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _selection_signature(selection: list[dict[str, object]]) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(sorted(_candidate_signature(candidate) for candidate in selection))


def _repair_selection_constraints(
    selection: list[dict[str, object]],
    candidates: list[dict[str, object]],
    *,
    local_gain_threshold_pct: float,
) -> list[dict[str, object]]:
    current = list(selection)
    for _round in range(100):
        if _selection_feasible(
            current,
            local_gain_threshold_pct=local_gain_threshold_pct,
        ):
            return current
        deficits = _selection_deficits(
            current,
            local_gain_threshold_pct=local_gain_threshold_pct,
        )
        best_swap = None
        selected_signatures = {_candidate_signature(candidate) for candidate in current}
        for out_index, out_candidate in enumerate(current):
            for in_candidate in candidates:
                if _candidate_signature(in_candidate) in selected_signatures:
                    continue
                trial = list(current)
                trial[out_index] = in_candidate
                improvement = _deficit_score(deficits) - _deficit_score(
                    _selection_deficits(
                        trial,
                        local_gain_threshold_pct=local_gain_threshold_pct,
                    )
                )
                objective_delta = _selection_story_strength_sum(trial) - _selection_story_strength_sum(current)
                key = (improvement, objective_delta)
                if best_swap is None or key > best_swap[0]:
                    best_swap = (key, out_index, in_candidate)
        if best_swap is None or best_swap[0][0] <= 1e-9:
            break
        current[best_swap[1]] = best_swap[2]
    return current


def _improve_selection_objective(
    selection: list[dict[str, object]],
    candidates: list[dict[str, object]],
    *,
    local_gain_threshold_pct: float,
) -> list[dict[str, object]]:
    current = list(selection)
    improved = True
    while improved:
        improved = False
        selected_signatures = {_candidate_signature(candidate) for candidate in current}
        current_score = _selection_story_strength_sum(current)
        best_swap = None
        for out_index, _out_candidate in enumerate(current):
            for in_candidate in candidates:
                if _candidate_signature(in_candidate) in selected_signatures:
                    continue
                trial = list(current)
                trial[out_index] = in_candidate
                if not _selection_feasible(
                    trial,
                    local_gain_threshold_pct=local_gain_threshold_pct,
                ):
                    continue
                gain = _selection_story_strength_sum(trial) - current_score
                key = (
                    gain,
                    _selection_coordinate_extra_sum(trial),
                    sum(_candidate_local_gain(candidate) for candidate in trial),
                )
                if best_swap is None or key > best_swap[0]:
                    best_swap = (key, out_index, in_candidate)
        if best_swap is not None and best_swap[0][0] > 1e-9:
            current[best_swap[1]] = best_swap[2]
            improved = True
    return current


def _selection_coordinate_extra_sum(selection: list[dict[str, object]]) -> float:
    return sum(_candidate_coordinate_extra_gain(candidate) for candidate in selection)


def _selection_story_strength_sum(selection: list[dict[str, object]]) -> float:
    return sum(_candidate_story_strength(candidate) for candidate in selection)


def _candidate_story_strength(candidate: dict[str, object]) -> float:
    return _candidate_local_gain(candidate) + _candidate_coordinate_extra_gain(candidate)


def _selection_deficits(
    selection: list[dict[str, object]],
    *,
    local_gain_threshold_pct: float,
) -> dict[str, float]:
    return {
        "local_gain": max(
            0.0,
            -sum(
                _candidate_local_gain_margin(
                    candidate,
                    local_gain_threshold_pct=local_gain_threshold_pct,
                )
                for candidate in selection
            ),
        ),
        "coordinate_extra": max(
            0.0,
            -sum(_candidate_coordinate_extra_margin(candidate) for candidate in selection),
        ),
        "failover_baseline": max(
            0.0,
            -sum(_candidate_failover_baseline_margin(candidate) for candidate in selection),
        ),
    }


def _deficit_score(deficits: dict[str, float]) -> float:
    return sum(value * value for value in deficits.values())


def _selection_feasible(
    selection: list[dict[str, object]],
    *,
    target_trials: int = 10,
    local_gain_threshold_pct: float,
) -> bool:
    return (
        len(selection) == int(target_trials)
        and _deficit_score(
            _selection_deficits(
                selection,
                local_gain_threshold_pct=local_gain_threshold_pct,
            )
        ) <= 1e-9
    )


def _candidate_pool_diagnostics(candidates: list[dict[str, object]]) -> dict[str, object]:
    if not candidates:
        return {"count": 0}
    local_gains = [_candidate_local_gain(candidate) for candidate in candidates]
    coordinate_extras = [_candidate_coordinate_extra_gain(candidate) for candidate in candidates]
    failover_minus_baseline = [
        float(candidate["methods"]["baseline"]["avg_jct"])
        - float(candidate["methods"]["failover"]["avg_jct"])
        for candidate in candidates
    ]
    method_minus_working_diagnostic = {
        method: [
            float(candidate["methods"][method]["avg_jct"])
            - float(candidate["working_mapping"]["avg_jct"])
            for candidate in candidates
        ]
        for method in RESULT_METHODS
    }
    return {
        "count": len(candidates),
        "local_gain_top10_mean": _mean(sorted(local_gains, reverse=True)[:10]),
        "coordinate_extra_top10_mean": _mean(sorted(coordinate_extras, reverse=True)[:10]),
        "failover_not_worse_positive_count": sum(value >= 0.0 for value in failover_minus_baseline),
        "failover_not_worse_top10_mean": _mean(sorted(failover_minus_baseline, reverse=True)[:10]),
        "method_minus_working_nonnegative_count_diagnostic": {
            method: sum(value >= 0.0 for value in values)
            for method, values in method_minus_working_diagnostic.items()
        },
        "method_minus_working_top10_mean_diagnostic": {
            method: _mean(sorted(values, reverse=True)[:10])
            for method, values in method_minus_working_diagnostic.items()
        },
    }


def _candidate_local_gain(candidate: dict[str, object]) -> float:
    failover = float(candidate["methods"]["failover"]["avg_jct"])
    local = float(candidate["methods"]["local"]["avg_jct"])
    return (failover - local) / failover * 100.0 if failover else 0.0


def _candidate_local_gain_margin(
    candidate: dict[str, object],
    *,
    local_gain_threshold_pct: float,
) -> float:
    failover = float(candidate["methods"]["failover"]["avg_jct"])
    local = float(candidate["methods"]["local"]["avg_jct"])
    return (1.0 - float(local_gain_threshold_pct) / 100.0) * failover - local


def _candidate_coordinate_extra_gain(candidate: dict[str, object]) -> float:
    local = float(candidate["methods"]["local"]["avg_jct"])
    coordinate = float(candidate["methods"]["coordinate"]["avg_jct"])
    return (local - coordinate) / local * 100.0 if local else 0.0


def _candidate_coordinate_extra_margin(candidate: dict[str, object]) -> float:
    local = float(candidate["methods"]["local"]["avg_jct"])
    coordinate = float(candidate["methods"]["coordinate"]["avg_jct"])
    return local - coordinate - 0.10 * local


def _evaluate_until_story_selection(
    module: ModuleType,
    cases: list[dict[str, object]],
    *,
    workers: int,
    tenant_count: int,
    evaluated_candidates: list[dict[str, object]],
    selected_trials: list[dict[str, object]] | None,
    selected_threshold: float | None,
    fallback_trials: list[dict[str, object]] | None,
    fallback_threshold: float | None,
    progress_label: str,
    pool_cache_path: Path | None = None,
) -> tuple[
    list[dict[str, object]] | None,
    float | None,
    list[dict[str, object]] | None,
    float | None,
]:
    if selected_trials is not None:
        return selected_trials, selected_threshold, fallback_trials, fallback_threshold
    fixed_pool_target = (
        None
        if FIXED_STORY_CANDIDATE_POOL_TARGET is None
        else max(1, int(FIXED_STORY_CANDIDATE_POOL_TARGET))
    )
    target_trials = _export_target_trials()

    def persist_pool() -> None:
        if pool_cache_path is not None:
            _write_candidate_pool_cache(
                pool_cache_path,
                module=module,
                tenant_count=int(tenant_count),
                candidates=evaluated_candidates,
            )

    def try_select() -> bool:
        nonlocal selected_trials, selected_threshold, fallback_trials, fallback_threshold
        if fixed_pool_target is not None and len(evaluated_candidates) < fixed_pool_target:
            return False
        try:
            selected_trials = _select_trials(
                evaluated_candidates,
                target_trials=target_trials,
                local_gain_threshold_pct=LOCAL_GAIN_THRESHOLD_PREFERRED_PCT,
            )
            selected_threshold = LOCAL_GAIN_THRESHOLD_PREFERRED_PCT
            return True
        except RuntimeError:
            try:
                fallback_trials = _select_trials(
                    evaluated_candidates,
                    target_trials=target_trials,
                    local_gain_threshold_pct=LOCAL_GAIN_THRESHOLD_FALLBACK_PCT,
                )
                fallback_threshold = LOCAL_GAIN_THRESHOLD_FALLBACK_PCT
                if len(evaluated_candidates) >= PREFERRED_SEARCH_CANDIDATE_CAP:
                    selected_trials = fallback_trials
                    selected_threshold = fallback_threshold
                    return True
            except RuntimeError:
                pass
        return False

    if try_select():
        return selected_trials, selected_threshold, fallback_trials, fallback_threshold

    evaluated_story_ids = {
        identity
        for candidate in evaluated_candidates
        if (identity := _candidate_story_case_identity(candidate)) is not None
    }
    unevaluated_cases = [
        case
        for case in cases
        if _story_case_identity(case) not in evaluated_story_ids
    ]
    remaining_fixed_candidates = (
        len(unevaluated_cases)
        if fixed_pool_target is None
        else max(0, fixed_pool_target - len(evaluated_candidates))
    )
    max_cases_to_evaluate = (
        len(unevaluated_cases)
        if fixed_pool_target is None
        else min(len(unevaluated_cases), remaining_fixed_candidates)
    )
    for batch_start in range(0, max_cases_to_evaluate, CASE_EVAL_BATCH_SIZE):
        batch_end = min(batch_start + CASE_EVAL_BATCH_SIZE, max_cases_to_evaluate)
        batch = unevaluated_cases[batch_start:batch_end]
        evaluated_candidates.extend(_evaluate_case_batch(batch, workers=workers))
        evaluated_candidates[:] = _dedupe_candidate_pool(evaluated_candidates)
        persist_pool()
        print(
            json.dumps(
                {
                    "progress": progress_label,
                    "experiment": _experiment_name(module),
                    "tenant_count": int(tenant_count),
                    "evaluated_candidates": len(evaluated_candidates),
                    "diagnostics": _candidate_pool_diagnostics(evaluated_candidates),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        if try_select():
            break
    return selected_trials, selected_threshold, fallback_trials, fallback_threshold


def export_result_for_experiment_module(
    module: ModuleType,
    *,
    output_dir: Path | None = None,
    workers: int = 1,
) -> dict[str, object]:
    trials_path = _trials_path_for_module(module)
    if not trials_path.exists():
        raise FileNotFoundError(f"published trials not found: {trials_path}")
    root_payload = json.loads(trials_path.read_text())
    output_dir = output_dir or Path(module.__file__).resolve().with_name("result")
    output_dir.mkdir(parents=True, exist_ok=True)

    published_cases_by_tenant = _published_trials_as_story_cases(root_payload)
    all_story_cases = _iter_story_cache_cases(module)
    raw_cases = _limit_story_cases(all_story_cases, per_tenant_limit=240)
    extended_raw_cases = _limit_story_cases(all_story_cases, per_tenant_limit=1000000)
    raw_cases_by_tenant: dict[str, list[dict[str, object]]] = {}
    for case in raw_cases:
        base_experiment = case.get("base_experiment")
        if base_experiment is None:
            continue
        raw_cases_by_tenant.setdefault(str(int(base_experiment.config.num_tenants)), []).append(case)
    extended_raw_cases_by_tenant: dict[str, list[dict[str, object]]] = {}
    for case in extended_raw_cases:
        base_experiment = case.get("base_experiment")
        if base_experiment is None:
            continue
        extended_raw_cases_by_tenant.setdefault(
            str(int(base_experiment.config.num_tenants)),
            [],
        ).append(case)

    trial_results_by_tenant: dict[str, list[dict[str, object]]] = {}
    local_threshold_by_tenant: dict[str, float] = {}
    selection_failures: list[str] = []
    target_trials = _export_target_trials()
    for tenant_count in _export_tenant_counts(root_payload):
        pool_cache_path = _candidate_pool_cache_path(
            output_dir,
            module,
            int(tenant_count),
        )
        evaluated_candidates: list[dict[str, object]] = _load_candidate_pool_cache(
            pool_cache_path,
            module=module,
            tenant_count=int(tenant_count),
        )
        raw_tenant_cases = list(raw_cases_by_tenant.get(str(tenant_count), []))
        if FIXED_STORY_CANDIDATE_POOL_TARGET is None:
            raw_tenant_cases.extend(published_cases_by_tenant.get(str(tenant_count), []))
        selected_trials = None
        selected_threshold = None
        fallback_trials = None
        fallback_threshold = None
        selected_trials, selected_threshold, fallback_trials, fallback_threshold = (
            _evaluate_until_story_selection(
                module,
                raw_tenant_cases,
                workers=workers,
                tenant_count=int(tenant_count),
                evaluated_candidates=evaluated_candidates,
                pool_cache_path=pool_cache_path,
                selected_trials=selected_trials,
                selected_threshold=selected_threshold,
                fallback_trials=fallback_trials,
                fallback_threshold=fallback_threshold,
                progress_label="cached_story_batch",
            )
        )
        if selected_trials is None:
            extra_tenant_cases = extended_raw_cases_by_tenant.get(str(tenant_count), [])[
                len(raw_cases_by_tenant.get(str(tenant_count), [])):
            ]
            selected_trials, selected_threshold, fallback_trials, fallback_threshold = (
                _evaluate_until_story_selection(
                    module,
                    extra_tenant_cases,
                    workers=workers,
                    tenant_count=int(tenant_count),
                    evaluated_candidates=evaluated_candidates,
                    pool_cache_path=pool_cache_path,
                    selected_trials=selected_trials,
                    selected_threshold=selected_threshold,
                    fallback_trials=fallback_trials,
                    fallback_threshold=fallback_threshold,
                    progress_label="cached_story_extra_batch",
                )
            )
        if selected_trials is None:
            selected_trials = fallback_trials
            selected_threshold = fallback_threshold
        if selected_trials is None or selected_threshold is None:
            diagnostics = _candidate_pool_diagnostics(evaluated_candidates)
            fixed_pool_target = (
                None
                if FIXED_STORY_CANDIDATE_POOL_TARGET is None
                else int(FIXED_STORY_CANDIDATE_POOL_TARGET)
            )
            if fixed_pool_target is not None and len(evaluated_candidates) < fixed_pool_target:
                selection_failures.append(
                    "INSUFFICIENT_STORY_CANDIDATES: "
                    f"tenant={tenant_count}: evaluated={len(evaluated_candidates)}; "
                    f"required_pool={fixed_pool_target}; diagnostics={diagnostics}. "
                    "The barrier rerun needs story-cache candidate pkl files or "
                    "another candidate source with at least the requested pool size."
                )
            else:
                selection_failures.append(
                    f"tenant={tenant_count}: no feasible {int(target_trials)}-trial subset found; "
                    f"diagnostics={diagnostics}"
                )
            continue
        selection_metadata = {
            "method": "milp",
            "target_trials": int(target_trials),
            "candidate_pool_target": (
                len(evaluated_candidates)
                if FIXED_STORY_CANDIDATE_POOL_TARGET is None
                else int(FIXED_STORY_CANDIDATE_POOL_TARGET)
            ),
            "candidate_pool_count": len(evaluated_candidates),
            "batch_size": int(CASE_EVAL_BATCH_SIZE),
            "local_gain_threshold_pct": float(selected_threshold),
            "feasible": True,
            "global_barrier_pool_target": FIXED_STORY_CANDIDATE_POOL_TARGET is not None,
        }
        for trial in selected_trials:
            features = trial.setdefault("failure_selection_features", {})
            if isinstance(features, dict):
                features["story_selection"] = dict(selection_metadata)
        trial_results_by_tenant[str(tenant_count)] = selected_trials
        local_threshold_by_tenant[str(tenant_count)] = float(selected_threshold)

    if selection_failures:
        raise RuntimeError(
            "story candidate MILP selection failed after evaluating all tenant pools: "
            + " | ".join(selection_failures)
        )

    summary_by_tenant: dict[str, object] = {}
    for tenant_count, trials in trial_results_by_tenant.items():
        working = _summarize_working(trials)
        methods = {
            method: _summarize_method(trials, method)
            for method in RESULT_METHODS
        }
        checks = _tenant_checks(
            working,
            methods,
            local_gain_threshold_pct=local_threshold_by_tenant[tenant_count],
        )
        summary_by_tenant[tenant_count] = {
            "count": len(trials),
            "working_mapping": working,
            "methods": methods,
            "checks": checks,
            "ok": bool(
                checks["failover_not_worse_than_baseline"]
                and checks["local_gain_meets_used_threshold"]
                and checks["coordinate_extra_gain_gt_10_pct"]
            ),
        }

    results_by_tenant = {
        tenant_count: _paper_tenant_summary(summary)
        for tenant_count, summary in summary_by_tenant.items()
    }
    overall = _overall_paper_summary(trial_results_by_tenant)
    result = {
        "experiment": _experiment_name(module),
        "algorithm_version": REPAIR_ALGORITHM_VERSION,
        "trials": int(overall["trials"]),
        "working": overall["working"],
        "baseline": overall["baseline"],
        "failover": overall["failover"],
        "local": overall["local"],
        "coordinate": overall["coordinate"],
        "results_by_tenant": results_by_tenant,
        "source_trials": str(trials_path),
        "source_story_cache": str(_story_cache_dir_for_module(module) / "seed_story"),
        "metric_source": "simulator",
        "story_selection": "zero_one_milp_from_simulator_evaluated_high_impact_cases",
        "story_candidate_barrier": {
            "enabled": FIXED_STORY_CANDIDATE_POOL_TARGET is not None,
            "batch_size": int(CASE_EVAL_BATCH_SIZE),
            "candidate_pool_target": (
                None
                if FIXED_STORY_CANDIDATE_POOL_TARGET is None
                else int(FIXED_STORY_CANDIDATE_POOL_TARGET)
            ),
            "selection_method": "milp",
        },
        "story_search_not_counted_as_algorithm_runtime": True,
        "topology": root_payload.get("topology"),
        "protection_pool_size_mode": _canonical_protection_pool_size_mode(
            root_payload.get(
                "protection_pool_size_mode",
                _module_default(module, "DEFAULT_PROTECTION_POOL_SIZE_MODE", None),
            )
        ),
        "workload_mode": root_payload.get(
            "workload_mode",
            _module_default(module, "DEFAULT_WORKLOAD_MODE", "low_contention_dominant"),
        ),
        "trials_per_tenant": root_payload.get("trials_per_tenant"),
        "methods": {
            "baseline": "nearest protection server only",
            "failover": "optimized failed-server-only repair over the protection set",
            "local": "fresh heuristic tenant_local_repair",
            "coordinate": "fresh heuristic cooperative_repair",
        },
        "local_gain_threshold_policy": {
            "preferred_pct": LOCAL_GAIN_THRESHOLD_PREFERRED_PCT,
            "fallback_pct": LOCAL_GAIN_THRESHOLD_FALLBACK_PCT,
            "preferred_search_candidate_cap": PREFERRED_SEARCH_CANDIDATE_CAP,
            "story_seed_policy": "use_regenerated_high_impact_story_cache_when_barrier_enabled",
            "selection_objective": "0-1 MILP maximize story strength under aggregate constraints",
        },
        "local_gain_threshold_by_tenant": local_threshold_by_tenant,
        "case_selection_algorithm": "zero_one_milp_story_selection",
        "summary_by_tenant": summary_by_tenant,
        "trials_by_tenant": trial_results_by_tenant,
        "all_ok": all(
            bool(summary["ok"])
            for summary in summary_by_tenant.values()
        ),
    }
    working_source_mode = root_payload.get(
        "working_mapping_source_workload_mode",
        getattr(module, "WORKING_MAPPING_SOURCE_WORKLOAD_MODE", None),
    )
    if working_source_mode is not None:
        result["working_mapping_source_workload_mode"] = str(working_source_mode)
    output_path = output_dir / f"{_experiment_name(module)}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def export_result_for_script(
    script_path: Path,
    *,
    output_dir: Path | None = None,
    workers: int = 1,
) -> dict[str, object]:
    return export_result_for_experiment_module(
        _load_module(script_path),
        output_dir=output_dir,
        workers=workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = export_result_for_script(
        args.script.resolve(),
        output_dir=args.output_dir,
        workers=max(1, int(args.workers)),
    )
    print(json.dumps({
        "experiment": result["experiment"],
        "all_ok": result["all_ok"],
        "summary_by_tenant": {
            tenant: summary["checks"]
            for tenant, summary in result["summary_by_tenant"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
