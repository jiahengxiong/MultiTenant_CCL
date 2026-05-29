from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingHybridHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


DEFAULT_INPUTS = (
    Path("/Users/xiongjiaheng/COCA/MultiTenant/experiment/Low_contension.json"),
    Path("/Users/xiongjiaheng/COCA/MultiTenant/experiment/High_contension.json"),
    Path("/Users/xiongjiaheng/COCA/MultiTenant/experiment/Low_contension_homo.json"),
    Path("/Users/xiongjiaheng/COCA/MultiTenant/experiment/High_contension_homo.json"),
)


def denormalize_mapping(raw: dict[str, dict[str, int]]) -> dict[int, dict[int, int]]:
    return {
        int(tenant): {int(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in raw.items()
    }


def denormalize_specs(raw: dict[str, dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


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


def run_stronger_hybrid(
    datacenter: LeafSpineDatacenter,
    initial_mapping: dict[int, dict[int, int]],
    current_mapping: dict[int, dict[int, int]],
    tenant_collective_specs: dict[int, dict[str, object]],
    *,
    time_limit: float,
    beam_width: int,
    bnb_seconds: float,
    rounds: int,
) -> tuple[dict[int, dict[int, int]], tuple[float, float], float]:
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    tenant_count = len(initial_mapping)
    solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=tenant_collective_specs,
        verbose=False,
        path_table=path_table,
        extra_seed_mappings=[current_mapping],
        beam_width=beam_width,
        bnb_candidate_time_limit=bnb_seconds,
        max_bnb_tenants_per_round=tenant_count,
        max_joint_tenants_per_round=min(max(3, tenant_count), 4),
    )
    solver.max_price_rounds = max(int(solver.max_price_rounds), int(rounds))
    start = time.time()
    solver.solve(time_limit=time_limit)
    runtime = time.time() - start
    mapping = solver.get_X_mapping()
    score = evaluate_mapping(datacenter, mapping, tenant_collective_specs)
    return mapping, score, float(runtime)


def run_default_hybrid(
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


def iter_cases(payload: dict[str, object], tenant_filter: set[int] | None, trial_limit: int):
    for tenant_group in payload.get("results", []):
        tenant_count = int(tenant_group["tenant_count"])
        if tenant_filter is not None and tenant_count not in tenant_filter:
            continue
        trial_results = list(tenant_group.get("trial_results", []))
        if trial_limit >= 0:
            trial_results = trial_results[:trial_limit]
        for trial in trial_results:
            yield tenant_count, trial


def audit_case_task(task: dict[str, object]) -> dict[str, object]:
    path = Path(task["path"])
    topology = task["topology"]
    tenant_count = int(task["tenant_count"])
    trial = task["trial"]
    time_limit = float(task["time_limit"])
    beam_width = int(task["beam_width"])
    bnb_seconds = float(task["bnb_seconds"])
    rounds = int(task["rounds"])
    rerun_default_threshold = float(task["rerun_default_threshold"])

    datacenter = LeafSpineDatacenter(
        num_spine=int(topology["num_spine"]),
        num_leaf=int(topology["num_leaf"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )

    trial_index = int(trial["trial_index"])
    print(f"[audit] {path.name} tenant_count={tenant_count} trial={trial_index + 1}", flush=True)
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    current_mapping = denormalize_mapping(trial["proposed_mapping"])
    tenant_collective_specs = denormalize_specs(trial["tenant_collective_specs"])
    recorded = trial["results"]["mapping"]
    current_score = evaluate_mapping(datacenter, current_mapping, tenant_collective_specs)
    stronger_mapping, stronger_score, stronger_runtime = run_stronger_hybrid(
        datacenter,
        initial_mapping,
        current_mapping,
        tenant_collective_specs,
        time_limit=time_limit,
        beam_width=beam_width,
        bnb_seconds=bnb_seconds,
        rounds=rounds,
    )
    avg_improvement = (
        (current_score[0] - stronger_score[0]) / current_score[0]
        if current_score[0] > 0
        else 0.0
    )
    mk_improvement = (
        (current_score[1] - stronger_score[1]) / current_score[1]
        if current_score[1] > 0
        else 0.0
    )
    case_result = {
        "tenant_count": tenant_count,
        "trial_index": trial_index,
        "current": {
            "recorded_avg_jct": float(recorded["avg_jct"]),
            "recorded_makespan": float(recorded["makespan"]),
            "recomputed_avg_jct": current_score[0],
            "recomputed_makespan": current_score[1],
        },
        "stronger_hybrid": {
            "avg_jct": stronger_score[0],
            "makespan": stronger_score[1],
            "runtime_seconds": stronger_runtime,
            "same_mapping": current_mapping == stronger_mapping,
        },
        "improvement_over_current": {
            "avg_jct_relative": float(avg_improvement),
            "makespan_relative": float(mk_improvement),
        },
    }
    print(
        "  current avg={:.6f}, strong avg={:.6f}, improvement={:.3%}; "
        "current mk={:.6f}, strong mk={:.6f}, improvement={:.3%}".format(
            current_score[0],
            stronger_score[0],
            avg_improvement,
            current_score[1],
            stronger_score[1],
            mk_improvement,
        ),
        flush=True,
    )
    if avg_improvement > rerun_default_threshold:
        default_mapping, default_score, default_runtime = run_default_hybrid(
            datacenter,
            initial_mapping,
            tenant_collective_specs,
        )
        default_vs_strong = (
            (default_score[0] - stronger_score[0]) / stronger_score[0]
            if stronger_score[0] > 0
            else 0.0
        )
        case_result["current_default_hybrid"] = {
            "avg_jct": default_score[0],
            "makespan": default_score[1],
            "runtime_seconds": default_runtime,
            "same_as_recorded_mapping": default_mapping == current_mapping,
            "same_as_stronger_mapping": default_mapping == stronger_mapping,
            "avg_jct_gap_vs_stronger_relative": float(default_vs_strong),
        }
        print(
            "  rerun default hybrid avg={:.6f}, gap_vs_strong={:.3%}".format(
                default_score[0],
                default_vs_strong,
            ),
            flush=True,
        )
    return case_result


def build_audit_tasks(
    path: Path,
    *,
    tenant_filter: set[int] | None,
    trial_limit: int,
    time_limit: float,
    beam_width: int,
    bnb_seconds: float,
    rounds: int,
    rerun_default_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    topology = payload["metadata"]["topology"]
    tasks = []
    for tenant_count, trial in iter_cases(payload, tenant_filter, trial_limit):
        tasks.append(
            {
                "path": str(path),
                "topology": topology,
                "tenant_count": tenant_count,
                "trial": trial,
                "time_limit": time_limit,
                "beam_width": beam_width,
                "bnb_seconds": bnb_seconds,
                "rounds": rounds,
                "rerun_default_threshold": rerun_default_threshold,
            }
        )
    return payload, tasks


def audit_file(
    path: Path,
    *,
    tenant_filter: set[int] | None,
    trial_limit: int,
    time_limit: float,
    beam_width: int,
    bnb_seconds: float,
    rounds: int,
    rerun_default_threshold: float,
    jobs: int = 1,
) -> dict[str, object]:
    payload, tasks = build_audit_tasks(
        path,
        tenant_filter=tenant_filter,
        trial_limit=trial_limit,
        time_limit=time_limit,
        beam_width=beam_width,
        bnb_seconds=bnb_seconds,
        rounds=rounds,
        rerun_default_threshold=rerun_default_threshold,
    )
    if jobs > 1 and len(tasks) > 1:
        with Pool(processes=jobs, maxtasksperchild=1) as pool:
            cases = list(pool.imap_unordered(audit_case_task, tasks))
    else:
        cases = [audit_case_task(task) for task in tasks]
    cases.sort(key=lambda case: (int(case["tenant_count"]), int(case["trial_index"])))

    return {
        "input": str(path),
        "scene": payload.get("metadata", {}).get("scene"),
        "cases": cases,
    }


def parse_tenant_filter(raw: str) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dominant main-result mappings against a stronger hybrid search.",
    )
    parser.add_argument("--inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--tenant-counts", type=str, default="3,4,8")
    parser.add_argument("--trial-limit", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--bnb-seconds", type=float, default=3.0)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--rerun-default-threshold",
        type=float,
        default=0.01,
        help="Rerun the current default hybrid when stronger search improves recorded mapping by more than this relative Avg JCT threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/xiongjiaheng/COCA/MultiTenant/experiment/dominant_mapping_quality_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tenant_filter = parse_tenant_filter(args.tenant_counts)
    results = [
        audit_file(
            path,
            tenant_filter=tenant_filter,
            trial_limit=args.trial_limit,
            time_limit=args.time_limit,
            beam_width=args.beam_width,
            bnb_seconds=args.bnb_seconds,
            rounds=args.rounds,
            rerun_default_threshold=args.rerun_default_threshold,
            jobs=args.jobs,
        )
        for path in args.inputs
    ]
    payload = {
        "metadata": {
            "tenant_counts": sorted(tenant_filter) if tenant_filter is not None else "all",
            "trial_limit_per_tenant_count": args.trial_limit,
            "stronger_hybrid": {
                "time_limit_seconds": args.time_limit,
                "beam_width": args.beam_width,
                "bnb_candidate_time_limit_seconds": args.bnb_seconds,
                "rounds": args.rounds,
            },
            "rerun_default_threshold": args.rerun_default_threshold,
            "jobs": args.jobs,
            "note": "This is an audit only: stronger hybrid uses the current mapping as a seed and does not use simulator feedback during search.",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved audit to {args.output}", flush=True)


if __name__ == "__main__":
    main()
