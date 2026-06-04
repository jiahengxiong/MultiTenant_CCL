from __future__ import annotations

import argparse
import json
import hashlib
import os
from multiprocessing import Pool
from pathlib import Path

from Low_contension import (
    TOPOLOGY,
    average_case_results,
    build_tenant_collective_specs,
    load_dominant_profiles,
    parse_tenant_counts,
    summarize_single_case,
)


def derive_seed(base_seed: int, *components: object) -> int:
    payload = "|".join([str(base_seed), *(str(component) for component in components)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


def _run_single_case_worker(payload: dict[str, object]) -> dict[str, object]:
    print(
        f"[worker pid={os.getpid()}] low_contention_homogeneous "
        f"tenant_count={int(payload['tenant_count'])} trial={int(payload['trial_index']) + 1} dispatch",
        flush=True,
    )
    return summarize_single_case(
        tenant_count=int(payload["tenant_count"]),
        seed=int(payload["seed"]),
        trial_index=int(payload["trial_index"]),
        tenant_workload_assignment=dict(payload["tenant_workload_assignment"]),
        harmonics_time_limit_s=payload["harmonics_time_limit_s"],
    )


def build_homogeneous_gpt_specs(
    tenant_count: int,
    profiles: dict[str, dict[str, object]],
) -> dict[int, dict[str, object]]:
    gpt_spec = dict(profiles["GPT13B"])
    return {
        tenant: {
            "collective": str(gpt_spec["collective"]),
            "profile_name": str(gpt_spec["profile_name"]),
            "msg_size_bytes": int(gpt_spec["msg_size_bytes"]),
            "group_size": int(gpt_spec["group_size"]),
            "source_stage": str(gpt_spec["source_stage"]),
            "source_comm_group": str(gpt_spec["source_comm_group"]),
            "source_comm_type": str(gpt_spec["source_comm_type"]),
            "trace_path": str(gpt_spec["trace_path"]),
        }
        for tenant in range(tenant_count)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-contention homogeneous comparison: all tenants use GPT13B dominant DP collective.",
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
        default=Path(__file__).resolve().parent / "Low_contension_homo.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tenant_counts = parse_tenant_counts(args.tenant_counts)
    profiles = load_dominant_profiles(8)
    cpu_count = os.cpu_count() or 1
    pool_size = max(1, cpu_count)

    all_work_items: list[dict[str, object]] = []
    results = []
    for tenant_count in tenant_counts:
        print(f"=== low_contention_homogeneous tenant_count={tenant_count} ===", flush=True)
        tenant_workload_assignment = build_homogeneous_gpt_specs(tenant_count, profiles)
        for trial_index in range(args.trials):
            trial_seed = derive_seed(args.seed, "low_contention_homogeneous", "trial", tenant_count, trial_index)
            print(
                f"--- low_contention_homogeneous tenant_count={tenant_count} trial={trial_index + 1}/{args.trials} seed={trial_seed} ---",
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
                f"[progress] low_contention_homogeneous tenant_count={int(case['tenant_count'])} "
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
                "scene": "low_contention_homogeneous",
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
            "scene": "low_contention_homogeneous",
            "definition": "each server is occupied by exactly one tenant; all tenants use GPT13B dominant DP collective",
            "topology": TOPOLOGY,
            "tenant_counts": list(tenant_counts),
            "seed": args.seed,
            "trials": args.trials,
            "cpu_count": cpu_count,
            "pool_size": pool_size,
            "harmonics_time_limit_seconds": args.harmonics_time_limit,
            "task_size_rule": "GPT13B dominant DP collective only; msg_size is in bytes; task_size_bits = (msg_size_bytes * 8) / occupied_servers",
            "dominant_profile": profiles["GPT13B"],
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
