from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.baselines import HarmonicsBaselineHeuristic, build_leaf_local_mapping
from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingEstimatorBlackBoxOptimizer
from multitenant.topology import LeafSpineDatacenter


TOPOLOGY = {
    "num_spine": 4,
    "num_leaf": 8,
    "per_leaf_server": 8,
}
TENANT_COUNTS = (2, 3, 4, 5, 6, 7, 8)
METHOD_NAMES = (
    "default",
    "locality",
    "mapping",
    "default_plus_harmonics",
    "locality_plus_harmonics",
    "mapping_plus_harmonics",
)
WORKLOAD_TRACES = {
    "GPT13B": REPO_ROOT / "workload" / "gpt13B_trace_dp32_ws32.csv",
    "LLaMA65B": REPO_ROOT / "workload" / "llama65B_trace_dp32_ws32.csv",
    "DeepSeek16B": REPO_ROOT / "workload" / "DeepSeek16B_trace_dp32_ws32.csv",
}
TASK_SIZE_MULTIPLIER = 4
MAPPING_TIME_LIMIT_SECONDS = None


def derive_seed(base_seed: int, *components: object) -> int:
    payload = "|".join([str(base_seed), *(str(component) for component in components)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


def log_trial_progress(scene: str, tenant_count: int, trial_index: int, message: str) -> None:
    print(
        f"[worker pid={os.getpid()}] {scene} tenant_count={tenant_count} "
        f"trial={trial_index + 1} {message}",
        flush=True,
    )


def parse_tenant_counts(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_comm_type(comm_type: str) -> str:
    comm_type = comm_type.strip()
    if comm_type.startswith("CommType."):
        comm_type = comm_type.split(".", 1)[1]
    return comm_type.replace("_", "")


def load_dominant_profiles(task_size_multiplier: int) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    for profile_name, path in WORKLOAD_TRACES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing workload trace: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        comm_rows = [
            row
            for row in rows
            if row["comm_type"] not in {"CommType.computation", "CommType.epoch_end"}
            and row["msg_size"] not in {"", "None"}
            and row["comm_group_size"] not in {"", "None"}
        ]
        dominant = max(comm_rows, key=lambda row: int(row["msg_size"]))
        msg_size_bytes = int(dominant["msg_size"])
        group_size = int(dominant["comm_group_size"])
        profiles[profile_name] = {
            "profile_name": profile_name,
            "trace_path": str(path),
            "collective": parse_comm_type(dominant["comm_type"]),
            "msg_size_bytes": msg_size_bytes,
            "group_size": group_size,
            "source_stage": dominant["stage"],
            "source_comm_group": dominant["comm_group"],
            "source_comm_type": dominant["comm_type"],
        }
    return profiles


def _balanced_double_occupancy_assignment(
    servers: list[int],
    tenant_count: int,
    seed: int,
    max_restarts: int = 2000,
) -> dict[int, list[int]]:
    total_slots = len(servers) * 2
    base = total_slots // tenant_count
    rem = total_slots % tenant_count
    quotas = {tenant: base + (1 if tenant < rem else 0) for tenant in range(tenant_count)}

    for attempt in range(max_restarts):
        rng = random.Random(derive_seed(seed, "balanced_double_occupancy", tenant_count, attempt))
        shuffled_servers = list(servers)
        rng.shuffle(shuffled_servers)
        remaining = dict(quotas)
        assignment: dict[int, list[int]] = {}
        success = True

        for index, server in enumerate(shuffled_servers):
            servers_left_after = len(shuffled_servers) - index - 1
            candidates = [tenant for tenant, quota in remaining.items() if quota > 0]
            if len(candidates) < 2:
                success = False
                break

            candidates.sort(key=lambda tenant: (remaining[tenant], rng.random()), reverse=True)
            picked: list[int] = []
            for tenant in candidates:
                if tenant in picked:
                    continue
                picked.append(tenant)
                if len(picked) == 2:
                    break
            if len(picked) < 2:
                success = False
                break

            assignment[int(server)] = picked
            for tenant in picked:
                remaining[tenant] -= 1

            if any(quota < 0 for quota in remaining.values()):
                success = False
                break
            if any(quota > servers_left_after for quota in remaining.values()):
                success = False
                break

        if success and all(quota == 0 for quota in remaining.values()):
            return assignment

    raise RuntimeError(
        f"Failed to construct balanced high-contention mapping for tenant_count={tenant_count}"
    )


def build_high_contention_mapping(
    datacenter: LeafSpineDatacenter,
    tenant_count: int,
    seed: int,
) -> dict[int, dict[int, int]]:
    all_servers = list(datacenter.get_all_servers())
    server_to_tenants = _balanced_double_occupancy_assignment(all_servers, tenant_count, seed)

    per_tenant_servers: dict[int, list[int]] = {tenant: [] for tenant in range(tenant_count)}
    for server in sorted(all_servers):
        for tenant in server_to_tenants[int(server)]:
            per_tenant_servers[int(tenant)].append(int(server))

    mapping: dict[int, dict[int, int]] = {}
    for tenant in range(tenant_count):
        mapping[tenant] = {
            rank: server for rank, server in enumerate(per_tenant_servers[tenant])
        }
    return mapping


def normalize_mapping(mapping: dict[int, dict[int, int]]) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {str(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in mapping.items()
    }


def normalize_specs(specs: dict[int, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(tenant): {
            key: (int(value) if isinstance(value, bool | int) else value)
            for key, value in spec.items()
        }
        for tenant, spec in specs.items()
    }


def sample_tenant_workload_assignment(
    tenant_count: int,
    seed: int,
    profiles: dict[str, dict[str, object]],
) -> dict[int, dict[str, object]]:
    rng = random.Random(seed)
    profile_names = list(profiles)
    tenant_specs: dict[int, dict[str, object]] = {}
    for tenant in range(tenant_count):
        chosen = dict(profiles[rng.choice(profile_names)])
        tenant_specs[tenant] = {
            "collective": str(chosen["collective"]),
            "profile_name": str(chosen["profile_name"]),
            "msg_size_bytes": int(chosen["msg_size_bytes"]),
            "group_size": int(chosen["group_size"]),
            "source_stage": str(chosen["source_stage"]),
            "source_comm_group": str(chosen["source_comm_group"]),
            "source_comm_type": str(chosen["source_comm_type"]),
            "trace_path": str(chosen["trace_path"]),
        }
    return tenant_specs


def build_tenant_collective_specs(
    tenant_workload_assignment: dict[int, dict[str, object]],
    tenant_mapping: dict[int, dict[int, int]],
) -> dict[int, dict[str, object]]:
    tenant_specs: dict[int, dict[str, object]] = {}
    for tenant, chosen in tenant_workload_assignment.items():
        occupied_servers = len(tenant_mapping[tenant])
        task_size_bits = int((int(chosen["msg_size_bytes"]) * TASK_SIZE_MULTIPLIER) // occupied_servers)
        tenant_specs[tenant] = {
            "collective": str(chosen["collective"]),
            "single_flow_size_bits": task_size_bits,
            "profile_name": str(chosen["profile_name"]),
            "msg_size_bytes": int(chosen["msg_size_bytes"]),
            "group_size": int(chosen["group_size"]),
            "occupied_servers": occupied_servers,
            "source_stage": str(chosen["source_stage"]),
            "source_comm_group": str(chosen["source_comm_group"]),
            "source_comm_type": str(chosen["source_comm_type"]),
            "trace_path": str(chosen["trace_path"]),
        }
    return tenant_specs


def average_case_results(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    averages: dict[str, dict[str, float]] = {}
    for method_name in METHOD_NAMES:
        metrics = [trial["results"][method_name] for trial in results]
        averages[method_name] = {
            "avg_jct": sum(float(metric["avg_jct"]) for metric in metrics) / len(metrics),
            "makespan": sum(float(metric["makespan"]) for metric in metrics) / len(metrics),
        }
    return averages


def evaluate_collective(
    datacenter: LeafSpineDatacenter,
    mapping: dict[int, dict[int, int]],
    tenant_collective_specs: dict[int, dict[str, object]],
    *,
    tenant_start_times: dict[int, float] | None = None,
    collective_start_times: dict[int, dict[int, float]] | None = None,
    collective_rate_scales: dict[int, dict[int, float]] | None = None,
    collective_rate_schedule: dict[int, dict[int, list[tuple[float, float]]]] | None = None,
    task_rate_schedule: dict[int, dict[int, dict[int, list[tuple[float, float]]]]] | None = None,
) -> tuple[float, float]:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=tenant_collective_specs,
        tenant_start_times=tenant_start_times,
        collective_start_times=collective_start_times,
        collective_rate_scales=collective_rate_scales,
        collective_rate_schedule=collective_rate_schedule,
        task_rate_schedule=task_rate_schedule,
    )
    return float(makespan), float(avg_jct)


def run_harmonics(
    datacenter: LeafSpineDatacenter,
    mapping: dict[int, dict[int, int]],
    tenant_collective_specs: dict[int, dict[str, object]],
    *,
    time_limit_s: float | None,
) -> tuple[dict[str, object], float]:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    solver_kwargs = {
        "verbose": False,
        "tenant_collective_specs": tenant_collective_specs,
    }
    if time_limit_s is not None:
        solver_kwargs["program_time_limit_s"] = float(time_limit_s)
    solver = HarmonicsBaselineHeuristic(
        datacenter,
        mapping,
        None,
        path_table,
        None,
        None,
        **solver_kwargs,
    )
    start = time.time()
    schedule = solver.solve()
    runtime_s = time.time() - start

    tenant_start_times = {
        int(tenant): float(schedule[int(tenant)][0])
        for tenant in schedule
    }
    collective_start_times = solver.get_collective_start_times()
    collective_rate_scales = solver.get_collective_rate_scales()
    collective_rate_schedule = solver.get_collective_rate_schedule()
    task_rate_schedule = solver.get_task_rate_schedule()

    makespan, avg_jct = evaluate_collective(
        datacenter,
        mapping,
        tenant_collective_specs,
        tenant_start_times=tenant_start_times,
        collective_start_times=collective_start_times,
        collective_rate_scales=collective_rate_scales,
        collective_rate_schedule=collective_rate_schedule,
        task_rate_schedule=task_rate_schedule,
    )
    return {
        "avg_jct": avg_jct,
        "makespan": makespan,
        "tenant_start_times": {str(k): float(v) for k, v in tenant_start_times.items()},
        "collective_start_times": {
            str(tenant): {str(op_idx): float(offset) for op_idx, offset in starts.items()}
            for tenant, starts in collective_start_times.items()
        },
    }, float(runtime_s)


def run_mapping(
    datacenter: LeafSpineDatacenter,
    tenant_mapping: dict[int, dict[int, int]],
    tenant_collective_specs: dict[int, dict[str, object]],
) -> tuple[dict[int, dict[int, int]], float]:
    path_table = datacenter.build_tenant_ecmp_path_table(tenant_mapping)
    solver = MappingEstimatorBlackBoxOptimizer(
        datacenter,
        tenant_mapping=tenant_mapping,
        tenant_collective_specs=tenant_collective_specs,
        verbose=False,
        path_table=path_table,
    )
    start = time.time()
    solver.solve(time_limit=MAPPING_TIME_LIMIT_SECONDS)
    runtime_s = time.time() - start
    return solver.get_X_mapping(), float(runtime_s)


def summarize_single_case(
    tenant_count: int,
    seed: int,
    trial_index: int,
    tenant_workload_assignment: dict[int, dict[str, object]],
    harmonics_time_limit_s: float | None,
) -> dict[str, object]:
    log_trial_progress("high_contention", tenant_count, trial_index, "started")
    datacenter = LeafSpineDatacenter(
        num_leaf=TOPOLOGY["num_leaf"],
        num_spine=TOPOLOGY["num_spine"],
        per_leaf_server=TOPOLOGY["per_leaf_server"],
    )
    default_mapping = build_high_contention_mapping(datacenter, tenant_count, seed)
    tenant_collective_specs = build_tenant_collective_specs(tenant_workload_assignment, default_mapping)
    locality_mapping = build_leaf_local_mapping(default_mapping)
    log_trial_progress("high_contention", tenant_count, trial_index, "solving mapping")
    proposed_mapping, mapping_runtime_s = run_mapping(
        datacenter,
        default_mapping,
        tenant_collective_specs,
    )

    log_trial_progress("high_contention", tenant_count, trial_index, "evaluating default/locality/mapping")
    default_mk, default_avg = evaluate_collective(
        datacenter,
        default_mapping,
        tenant_collective_specs,
    )
    locality_mk, locality_avg = evaluate_collective(
        datacenter,
        locality_mapping,
        tenant_collective_specs,
    )
    mapping_mk, mapping_avg = evaluate_collective(
        datacenter,
        proposed_mapping,
        tenant_collective_specs,
    )

    log_trial_progress("high_contention", tenant_count, trial_index, "running default+harmonics")
    default_harm, default_harm_rt = run_harmonics(
        datacenter,
        default_mapping,
        tenant_collective_specs,
        time_limit_s=harmonics_time_limit_s,
    )
    log_trial_progress("high_contention", tenant_count, trial_index, "running locality+harmonics")
    locality_harm, locality_harm_rt = run_harmonics(
        datacenter,
        locality_mapping,
        tenant_collective_specs,
        time_limit_s=harmonics_time_limit_s,
    )
    log_trial_progress("high_contention", tenant_count, trial_index, "running mapping+harmonics")
    mapping_harm, mapping_harm_rt = run_harmonics(
        datacenter,
        proposed_mapping,
        tenant_collective_specs,
        time_limit_s=harmonics_time_limit_s,
    )
    log_trial_progress("high_contention", tenant_count, trial_index, "finished")

    return {
        "tenant_count": tenant_count,
        "trial_index": trial_index,
        "trial_seed": seed,
        "scene": "high_contention",
        "tenant_collective_specs": normalize_specs(tenant_collective_specs),
        "initial_mapping": normalize_mapping(default_mapping),
        "locality_mapping": normalize_mapping(locality_mapping),
        "proposed_mapping": normalize_mapping(proposed_mapping),
        "results": {
            "default": {
                "avg_jct": default_avg,
                "makespan": default_mk,
            },
            "locality": {
                "avg_jct": locality_avg,
                "makespan": locality_mk,
            },
            "mapping": {
                "avg_jct": mapping_avg,
                "makespan": mapping_mk,
            },
            "default_plus_harmonics": {
                **default_harm,
            },
            "locality_plus_harmonics": {
                **locality_harm,
            },
            "mapping_plus_harmonics": {
                **mapping_harm,
            },
        },
    }


def _run_single_case_worker(payload: dict[str, object]) -> dict[str, object]:
    return summarize_single_case(
        tenant_count=int(payload["tenant_count"]),
        seed=int(payload["seed"]),
        trial_index=int(payload["trial_index"]),
        tenant_workload_assignment=dict(payload["tenant_workload_assignment"]),
        harmonics_time_limit_s=payload["harmonics_time_limit_s"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-contention large-topology comparison using dominant DP collective from GPT13B/LLaMA65B/DeepSeek16B traces.",
    )
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument(
        "--tenant-counts",
        type=str,
        default="2,3,4,5,6,7,8",
        help="Comma-separated tenant counts.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of randomized trials to run per tenant count.",
    )
    parser.add_argument(
        "--harmonics-time-limit",
        type=float,
        default=None,
        help="Optional time limit in seconds for the harmonics heuristic. Default: no time limit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiment" / "High_contension.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tenant_counts = parse_tenant_counts(args.tenant_counts)
    profiles = load_dominant_profiles(TASK_SIZE_MULTIPLIER)
    cpu_count = os.cpu_count() or 1
    pool_size = max(1, min(cpu_count, 10))

    all_work_items: list[dict[str, object]] = []
    results = []
    for tenant_count in tenant_counts:
        print(f"=== high_contention tenant_count={tenant_count} ===", flush=True)
        tenant_workload_assignment = sample_tenant_workload_assignment(
            tenant_count,
            derive_seed(args.seed, "high_contention", "assignment", tenant_count),
            profiles,
        )
        for trial_index in range(args.trials):
            trial_seed = derive_seed(args.seed, "high_contention", "trial", tenant_count, trial_index)
            print(
                f"--- high_contention tenant_count={tenant_count} trial={trial_index + 1}/{args.trials} seed={trial_seed} ---",
                flush=True,
            )
            all_work_items.append(
                {
                    "tenant_count": tenant_count,
                    "seed": trial_seed,
                    "trial_index": trial_index,
                    "tenant_workload_assignment": tenant_workload_assignment,
                    "harmonics_time_limit_s": args.harmonics_time_limit,
                }
            )

    with Pool(processes=pool_size, maxtasksperchild=1) as pool:
        all_trial_results = []
        total_work_items = len(all_work_items)
        for completed_idx, case in enumerate(pool.imap_unordered(_run_single_case_worker, all_work_items), start=1):
            print(
                f"[progress] high_contention tenant_count={int(case['tenant_count'])} "
                f"trial={int(case['trial_index']) + 1}/{args.trials} finished "
                f"({completed_idx}/{total_work_items} total)",
                flush=True,
            )
            all_trial_results.append(case)

    grouped_results: dict[int, list[dict[str, object]]] = {tenant_count: [] for tenant_count in tenant_counts}
    for case in all_trial_results:
        grouped_results[int(case["tenant_count"])].append(case)

    for tenant_count in tenant_counts:
        trial_results = sorted(grouped_results[tenant_count], key=lambda case: int(case["trial_index"]))
        averages = average_case_results(trial_results)
        results.append(
            {
                "tenant_count": tenant_count,
                "scene": "high_contention",
                "trial_count": args.trials,
                "trial_results": trial_results,
                "averages": averages,
            }
        )

        metrics = averages
        print(
            "default avg={:.12f} | locality avg={:.12f} | mapping avg={:.12f}".format(
                metrics["default"]["avg_jct"],
                metrics["locality"]["avg_jct"],
                metrics["mapping"]["avg_jct"],
            ),
            flush=True,
        )
        print(
            "default+harm avg={:.12f} | locality+harm avg={:.12f} | mapping+harm avg={:.12f}".format(
                metrics["default_plus_harmonics"]["avg_jct"],
                metrics["locality_plus_harmonics"]["avg_jct"],
                metrics["mapping_plus_harmonics"]["avg_jct"],
            ),
            flush=True,
        )

    payload = {
        "metadata": {
            "scene": "high_contention",
            "definition": "each server is occupied by exactly two distinct tenants; assignments are as balanced as possible",
            "topology": TOPOLOGY,
            "tenant_counts": list(tenant_counts),
            "seed": args.seed,
            "trials": args.trials,
            "cpu_count": cpu_count,
            "pool_size": pool_size,
            "harmonics_time_limit_seconds": args.harmonics_time_limit,
            "mapping_solver": "MappingEstimatorBlackBoxOptimizer",
            "mapping_solver_time_limit_seconds": MAPPING_TIME_LIMIT_SECONDS,
            "task_size_rule": "single dominant DP collective per tenant; msg_size is in bytes; task_size_bits = (msg_size_bytes * 4) / occupied_servers",
            "dominant_profiles": profiles,
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
