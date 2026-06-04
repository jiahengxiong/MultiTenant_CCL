from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingHybridHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


TOPOLOGY = {
    "num_spine": 2,
    "num_leaf": 4,
    "per_leaf_server": 6,
}
TENANT_COUNTS = (2, 3, 4)
WORKLOAD_TRACES = {
    "GPT13B": REPO_ROOT / "workload" / "gpt13B_trace_dp32_ws32.csv",
    "LLaMA65B": REPO_ROOT / "workload" / "llama65B_trace_dp32_ws32.csv",
    "DeepSeek16B": REPO_ROOT / "workload" / "DeepSeek16B_trace_dp32_ws32.csv",
}
COLLECTIVE_GAP_S = 0.001
TASK_SIZE_MULTIPLIER = 8
FULL_MAPPING_MIN_MSG_SIZE_BYTES = 64 * 1024 * 1024
SMALL_COLLECTIVE_GAP_S = 0.001


def derive_seed(base_seed: int, *components: object) -> int:
    payload = "|".join([str(base_seed), *(str(component) for component in components)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


def parse_comm_type(comm_type: str) -> str:
    comm_type = comm_type.strip()
    if comm_type.startswith("CommType."):
        comm_type = comm_type.split(".", 1)[1]
    return comm_type.replace("_", "")


def parse_tenant_counts(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def build_balanced_random_mapping(
    datacenter: LeafSpineDatacenter,
    tenant_count: int,
    seed: int,
) -> dict[int, dict[int, int]]:
    all_servers = list(datacenter.get_all_servers())
    total_servers = len(all_servers)
    base = total_servers // tenant_count
    rem = total_servers % tenant_count
    per_tenant_sizes = [base + (1 if idx < rem else 0) for idx in range(tenant_count)]
    rng = random.Random(seed)
    rng.shuffle(all_servers)

    mapping: dict[int, dict[int, int]] = {}
    cursor = 0
    for tenant, size in enumerate(per_tenant_sizes):
        mapping[tenant] = {}
        for rank in range(size):
            mapping[tenant][rank] = int(all_servers[cursor])
            cursor += 1
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


def normalize_programs(programs: dict[int, list[dict[str, object]]]) -> dict[str, list[dict[str, object]]]:
    return {
        str(tenant): [
            {
                key: (int(value) if isinstance(value, int) else float(value) if isinstance(value, float) else value)
                for key, value in op.items()
            }
            for op in program
        ]
        for tenant, program in programs.items()
    }


def normalize_start_times(start_times: dict[int, float]) -> dict[str, float]:
    return {str(tenant): float(start) for tenant, start in start_times.items()}


def load_workload_profiles() -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    for profile_name, path in WORKLOAD_TRACES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing workload trace: {path}")
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

        dominant_row = None
        dominant_size = -1
        program: list[dict[str, object]] = []
        initial_gap_s = 0.0
        skipped_gap_s = 0.0

        for row in rows:
            comm_type = row["comm_type"]
            if comm_type in {"CommType.computation", "CommType.epoch_end"}:
                continue

            group_size_raw = row["comm_group_size"]
            if group_size_raw in {"", "None"}:
                continue
            group_size = int(group_size_raw)

            msg_size_raw = row["msg_size"]
            if msg_size_raw in {"", "None"}:
                continue
            msg_size_bytes = int(msg_size_raw)

            if msg_size_bytes > dominant_size:
                dominant_size = msg_size_bytes
                dominant_row = row

            if group_size != 32:
                skipped_gap_s += COLLECTIVE_GAP_S
                continue

            collective = parse_comm_type(comm_type)

            if program:
                program[-1]["gap_after"] += COLLECTIVE_GAP_S + skipped_gap_s
            else:
                initial_gap_s += skipped_gap_s
            skipped_gap_s = 0.0

            program.append(
                {
                    "collective": collective,
                    "source_msg_size_bytes": msg_size_bytes,
                    "gap_after": 0.0,
                    "source_stage": row["stage"],
                    "source_comm_type": comm_type,
                    "source_comm_group": row["comm_group"],
                    "source_group_size": group_size,
                }
            )

        if not program:
            raise ValueError(f"No executable group_size=32 collectives found in {path}")

        if skipped_gap_s > 0.0:
            program[-1]["gap_after"] += skipped_gap_s

        if dominant_row is None:
            raise ValueError(f"No communication rows found in {path}")

        dominant_group_size = int(dominant_row["comm_group_size"])
        profiles[profile_name] = {
            "profile_name": profile_name,
            "trace_path": str(path),
            "dominant_collective": parse_comm_type(dominant_row["comm_type"]),
            "dominant_msg_size_bytes": int(dominant_row["msg_size"]),
            "dominant_group_size": dominant_group_size,
            "dominant_stage": dominant_row["stage"],
            "initial_gap_s": initial_gap_s,
            "program": program,
        }
    return profiles


def choose_workload_assignment(
    tenant_count: int,
    seed: int,
    profiles: dict[str, dict[str, object]],
) -> dict[int, dict[str, object]]:
    rng = random.Random(seed)
    names = list(profiles)
    rng.shuffle(names)
    chosen: list[str] = []
    while len(chosen) < tenant_count:
        for name in names:
            if len(chosen) < tenant_count:
                chosen.append(name)
        rng.shuffle(names)
    return {tenant: dict(profiles[name]) for tenant, name in enumerate(chosen[:tenant_count])}


def build_dominant_specs(
    assignment: dict[int, dict[str, object]],
    tenant_mapping: dict[int, dict[int, int]],
) -> dict[int, dict[str, object]]:
    specs: dict[int, dict[str, object]] = {}
    for tenant, profile in assignment.items():
        occupied_servers = len(tenant_mapping[tenant])
        specs[tenant] = {
            "collective": str(profile["dominant_collective"]),
            "single_flow_size_bits": int((int(profile["dominant_msg_size_bytes"]) * TASK_SIZE_MULTIPLIER) // occupied_servers),
            "profile_name": str(profile["profile_name"]),
            "msg_size_bytes": int(profile["dominant_msg_size_bytes"]),
            "group_size": int(profile["dominant_group_size"]),
            "occupied_servers": occupied_servers,
            "source_stage": str(profile["dominant_stage"]),
            "trace_path": str(profile["trace_path"]),
        }
    return specs


def build_full_programs(
    assignment: dict[int, dict[str, object]],
    tenant_mapping: dict[int, dict[int, int]],
) -> tuple[dict[int, list[dict[str, object]]], dict[int, float]]:
    programs: dict[int, list[dict[str, object]]] = {}
    start_times: dict[int, float] = {}
    for tenant, profile in assignment.items():
        occupied_servers = len(tenant_mapping[tenant])
        start_times[tenant] = float(profile["initial_gap_s"])
        program: list[dict[str, object]] = []
        for op in profile["program"]:
            program.append(
                {
                    "collective": str(op["collective"]),
                    "single_flow_size_bits": int((int(op["source_msg_size_bytes"]) * TASK_SIZE_MULTIPLIER) // occupied_servers),
                    "gap_after": float(op["gap_after"]),
                }
            )
        programs[tenant] = program
    return programs, start_times


def build_full_mapping_programs(
    assignment: dict[int, dict[str, object]],
    tenant_mapping: dict[int, dict[int, int]],
    *,
    min_msg_size_bytes: int,
) -> tuple[dict[int, list[dict[str, object]]], dict[int, float]]:
    programs: dict[int, list[dict[str, object]]] = {}
    start_times: dict[int, float] = {}
    for tenant, profile in assignment.items():
        occupied_servers = len(tenant_mapping[tenant])
        start_times[tenant] = float(profile["initial_gap_s"])
        program: list[dict[str, object]] = []
        for op in profile["program"]:
            if int(op["source_msg_size_bytes"]) <= int(min_msg_size_bytes):
                folded_gap = SMALL_COLLECTIVE_GAP_S + float(op["gap_after"])
                if program:
                    program[-1]["gap_after"] = float(program[-1]["gap_after"]) + folded_gap
                else:
                    start_times[tenant] += folded_gap
                continue
            program.append(
                {
                    "collective": str(op["collective"]),
                    "single_flow_size_bits": int((int(op["source_msg_size_bytes"]) * TASK_SIZE_MULTIPLIER) // occupied_servers),
                    "gap_after": float(op["gap_after"]),
                }
            )
        if not program:
            dominant_collective = str(profile["dominant_collective"])
            dominant_msg_size_bytes = int(profile["dominant_msg_size_bytes"])
            program.append(
                {
                    "collective": dominant_collective,
                    "single_flow_size_bits": int((dominant_msg_size_bytes * TASK_SIZE_MULTIPLIER) // occupied_servers),
                    "gap_after": 0.0,
                }
            )
        programs[tenant] = program
    return programs, start_times


def evaluate_full_program(
    datacenter: LeafSpineDatacenter,
    mapping: dict[int, dict[int, int]],
    full_programs: dict[int, list[dict[str, object]]],
    initial_start_times: dict[int, float],
) -> tuple[float, float]:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_programs=full_programs,
        tenant_start_times=initial_start_times,
    )
    return float(makespan), float(avg_jct)


def run_mapping_solver(
    datacenter: LeafSpineDatacenter,
    tenant_mapping: dict[int, dict[int, int]],
    *,
    tenant_collective_specs: dict[int, dict[str, object]] | None = None,
    tenant_collective_programs: dict[int, list[dict[str, object]]] | None = None,
    tenant_start_times: dict[int, float] | None = None,
    extra_seed_mappings: list[dict[int, dict[int, int]]] | None = None,
) -> tuple[dict[int, dict[int, int]], float]:
    path_table = datacenter.build_tenant_ecmp_path_table(tenant_mapping)
    solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=tenant_mapping,
        tenant_collective_specs=tenant_collective_specs,
        tenant_collective_programs=tenant_collective_programs,
        tenant_start_times=tenant_start_times,
        verbose=False,
        path_table=path_table,
        extra_seed_mappings=extra_seed_mappings,
    )
    start = time.time()
    solver.solve()
    runtime_s = time.time() - start
    return solver.get_X_mapping(), float(runtime_s)


def summarize_case(
    tenant_count: int,
    seed: int,
    profiles: dict[str, dict[str, object]],
) -> dict[str, object]:
    datacenter = LeafSpineDatacenter(
        num_leaf=TOPOLOGY["num_leaf"],
        num_spine=TOPOLOGY["num_spine"],
        per_leaf_server=TOPOLOGY["per_leaf_server"],
    )
    tenant_mapping = build_balanced_random_mapping(datacenter, tenant_count, seed)
    assignment = choose_workload_assignment(tenant_count, seed, profiles)
    dominant_specs = build_dominant_specs(assignment, tenant_mapping)
    full_programs, initial_start_times = build_full_programs(assignment, tenant_mapping)
    full_mapping_programs, full_mapping_start_times = build_full_mapping_programs(
        assignment,
        tenant_mapping,
        min_msg_size_bytes=FULL_MAPPING_MIN_MSG_SIZE_BYTES,
    )

    print(f"  [tenant_count={tenant_count}] solving dominant mapping...", flush=True)
    dominant_mapping, dominant_runtime_s = run_mapping_solver(
        datacenter,
        tenant_mapping,
        tenant_collective_specs=dominant_specs,
    )
    print(f"  [tenant_count={tenant_count}] solving full-program mapping...", flush=True)
    full_mapping, full_runtime_s = run_mapping_solver(
        datacenter,
        tenant_mapping,
        tenant_collective_programs=full_mapping_programs,
        tenant_start_times=full_mapping_start_times,
        extra_seed_mappings=[dominant_mapping],
    )

    print(f"  [tenant_count={tenant_count}] evaluating full trace on simulator...", flush=True)
    dominant_mk, dominant_avg = evaluate_full_program(
        datacenter,
        dominant_mapping,
        full_programs,
        initial_start_times,
    )
    full_mk, full_avg = evaluate_full_program(
        datacenter,
        full_mapping,
        full_programs,
        initial_start_times,
    )

    return {
        "tenant_count": tenant_count,
        "seed": seed,
        "scene": "single_occupancy_balanced_random",
        "topology": TOPOLOGY,
        "initial_mapping": normalize_mapping(tenant_mapping),
        "workload_assignment": {
            str(tenant): {
                "profile_name": str(profile["profile_name"]),
                "trace_path": str(profile["trace_path"]),
                "dominant_collective": str(profile["dominant_collective"]),
                "dominant_msg_size_bytes": int(profile["dominant_msg_size_bytes"]),
                "dominant_group_size": int(profile["dominant_group_size"]),
                "initial_gap_s": float(profile["initial_gap_s"]),
            }
            for tenant, profile in assignment.items()
        },
        "dominant_specs": normalize_specs(dominant_specs),
        "full_programs": normalize_programs(full_programs),
        "full_mapping_programs": normalize_programs(full_mapping_programs),
        "full_mapping_initial_start_times": normalize_start_times(full_mapping_start_times),
        "initial_start_times": normalize_start_times(initial_start_times),
        "results": {
            "dominant_mapping": {
                "mapping": normalize_mapping(dominant_mapping),
                "avg_jct": dominant_avg,
                "makespan": dominant_mk,
            },
            "full_mapping": {
                "mapping": normalize_mapping(full_mapping),
                "avg_jct": full_avg,
                "makespan": full_mk,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dominant-collective mapping vs full-program mapping, both evaluated on the full workload trace.",
    )
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument(
        "--tenant-counts",
        type=str,
        default="2,3,4",
        help="Comma-separated tenant counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiment" / "dominant vs full.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tenant_counts = parse_tenant_counts(args.tenant_counts)
    profiles = load_workload_profiles()

    results = []
    for tenant_count in tenant_counts:
        print(f"=== dominant_vs_full tenant_count={tenant_count} ===", flush=True)
        case = summarize_case(
            tenant_count=tenant_count,
            seed=derive_seed(args.seed, "dominant_vs_full", tenant_count),
            profiles=profiles,
        )
        results.append(case)
        metrics = case["results"]
        print(
            "dominant avg={:.12f} | full avg={:.12f}".format(
                metrics["dominant_mapping"]["avg_jct"],
                metrics["full_mapping"]["avg_jct"],
            ),
            flush=True,
        )
        print(
            "dominant mk={:.12f} | full mk={:.12f}".format(
                metrics["dominant_mapping"]["makespan"],
                metrics["full_mapping"]["makespan"],
            ),
            flush=True,
        )

    payload = {
        "metadata": {
            "scene": "single_occupancy_balanced_random",
            "definition": "each physical server is occupied by exactly one tenant; all servers are randomly assigned and balanced across tenants as evenly as possible",
            "topology": TOPOLOGY,
            "tenant_counts": list(tenant_counts),
            "seed": args.seed,
            "task_size_rule": "msg_size is in bytes; task_size_bits = (msg_size_bytes * 8) / occupied_servers",
            "collective_gap_seconds": COLLECTIVE_GAP_S,
            "full_mapping_min_msg_size_bytes": FULL_MAPPING_MIN_MSG_SIZE_BYTES,
            "trace_filter_rule": {
                "drop_group_size_none": True,
                "skip_group_size_not_32_as_gap_only": True,
            },
            "profiles": {
                name: {
                    "trace_path": str(profile["trace_path"]),
                    "dominant_collective": str(profile["dominant_collective"]),
                    "dominant_msg_size_bytes": int(profile["dominant_msg_size_bytes"]),
                    "dominant_group_size": int(profile["dominant_group_size"]),
                    "initial_gap_s": float(profile["initial_gap_s"]),
                    "program_length": len(profile["program"]),
                }
                for name, profile in profiles.items()
            },
        },
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
