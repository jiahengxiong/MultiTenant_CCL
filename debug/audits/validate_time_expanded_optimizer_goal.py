from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingEstimatorBlackBoxOptimizer, TimeExpandedContentionEstimator
from multitenant.topology import LeafSpineDatacenter


DEFAULT_INPUTS = (
    REPO_ROOT / "experiment" / "Low_contension.json",
    REPO_ROOT / "experiment" / "High_contension.json",
    REPO_ROOT / "experiment" / "Low_contension_homo.json",
    REPO_ROOT / "experiment" / "High_contension_homo.json",
)


def denormalize_mapping(raw: dict[str, dict[str, int]]) -> dict[int, dict[int, int]]:
    return {
        int(tenant): {int(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in raw.items()
    }


def denormalize_specs(raw: dict[str, dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def evaluate_mapping(datacenter, mapping, specs) -> tuple[float, float]:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return float(avg_jct), float(makespan)


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def release_gate_audit() -> dict[str, object]:
    datacenter = LeafSpineDatacenter(num_spine=2, num_leaf=2, per_leaf_server=4)
    mapping = {0: {0: 0, 1: 1, 2: 4, 3: 5}}
    programs = {
        0: [
            {
                "collective": "allgather",
                "single_flow_size_bits": 1_000_000,
                "gap_after": 0.001,
            },
            {
                "collective": "reducescatter",
                "single_flow_size_bits": 1_000_000,
                "gap_after": 0.0,
            },
        ]
    }
    estimator = TimeExpandedContentionEstimator(
        datacenter,
        tenant_mapping=mapping,
        tenant_collective_programs=programs,
        path_table=datacenter.build_tenant_ecmp_path_table(mapping),
    )
    gates = estimator.audit_release_gates()
    analysis = estimator.analyze(mapping)
    violations = []
    checked_edges = 0
    for tenant, tenant_gates in gates.items():
        for task_id, (previous_tasks, gap_after) in tenant_gates.items():
            for previous_task in previous_tasks:
                checked_edges += 1
                start_time = float(analysis.task_state[(tenant, task_id)]["start_time"])
                previous_finish = float(analysis.task_state[(tenant, previous_task)]["finish_time"])
                if start_time + 1e-9 < previous_finish + float(gap_after):
                    violations.append(
                        {
                            "tenant": int(tenant),
                            "task": int(task_id),
                            "previous_task": int(previous_task),
                            "start_time": start_time,
                            "previous_finish": previous_finish,
                            "gap_after": float(gap_after),
                        }
                    )
    return {
        "checked_release_edges": int(checked_edges),
        "violations": violations,
        "passed": checked_edges > 0 and not violations,
    }


def iter_trials(path: Path, tenant_counts: set[int], trial_indices: set[int]):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for group in payload["results"]:
        tenant_count = int(group["tenant_count"])
        if tenant_count not in tenant_counts:
            continue
        for trial in group["trial_results"]:
            trial_index = int(trial["trial_index"])
            if trial_index not in trial_indices:
                continue
            yield payload["metadata"], tenant_count, trial


def run_case(
    path: Path,
    metadata: dict[str, object],
    tenant_count: int,
    trial: dict[str, object],
    *,
    optimizer_time_limit: float | None,
) -> dict[str, object]:
    topology = metadata["topology"]
    datacenter = LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    json_mapping = denormalize_mapping(trial["proposed_mapping"])
    specs = denormalize_specs(trial["tenant_collective_specs"])
    recorded = trial["results"]["mapping"]
    recorded_score = (float(recorded["avg_jct"]), float(recorded["makespan"]))

    recomputed_score = evaluate_mapping(datacenter, json_mapping, specs)
    solver = MappingEstimatorBlackBoxOptimizer(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=datacenter.build_tenant_ecmp_path_table(initial_mapping),
        verbose=False,
    )
    start = time.time()
    solver.solve(time_limit=optimizer_time_limit)
    runtime = time.time() - start
    candidate_mapping = solver.get_X_mapping()
    candidate_score = evaluate_mapping(datacenter, candidate_mapping, specs)

    def rel_worse(candidate: float, baseline: float) -> float:
        return (float(candidate) - float(baseline)) / max(float(baseline), 1e-12)

    return {
        "input": str(path),
        "scene": metadata.get("scene", path.stem),
        "tenant_count": int(tenant_count),
        "trial_index": int(trial["trial_index"]),
        "recorded": {"avg_jct": recorded_score[0], "makespan": recorded_score[1]},
        "json_mapping_recomputed": {"avg_jct": recomputed_score[0], "makespan": recomputed_score[1]},
        "optimizer": {
            "avg_jct": candidate_score[0],
            "makespan": candidate_score[1],
            "runtime_seconds": float(runtime),
            "surrogate_mode": getattr(solver, "surrogate_mode", None),
            "final_score_is_simulated": bool(getattr(solver, "final_score_is_simulated", False)),
            "same_mapping_as_json": candidate_mapping == json_mapping,
        },
        "recorded_recompute_abs_delta": {
            "avg_jct": float(recomputed_score[0] - recorded_score[0]),
            "makespan": float(recomputed_score[1] - recorded_score[1]),
        },
        "worse_than_recorded_relative": {
            "avg_jct": rel_worse(candidate_score[0], recorded_score[0]),
            "makespan": rel_worse(candidate_score[1], recorded_score[1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the time-expanded contention estimator and estimator-driven "
            "optimizer against representative experiment JSON cases."
        )
    )
    parser.add_argument("--inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--tenant-counts", default="3")
    parser.add_argument("--trials", default="1", help="1-based trial ids")
    parser.add_argument("--sensitivity", type=float, default=0.02)
    parser.add_argument(
        "--optimizer-time-limit",
        type=float,
        default=60.0,
        help="Per-case time limit in seconds for the estimator-driven optimizer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "debug" / "audits" / "time_expanded_optimizer_goal_validation.json",
    )
    args = parser.parse_args()

    tenant_counts = set(parse_ints(args.tenant_counts))
    trial_indices = {idx - 1 for idx in parse_ints(args.trials)}
    rows = []
    for path in args.inputs:
        for metadata, tenant_count, trial in iter_trials(path, tenant_counts, trial_indices):
            print(
                f"[validate] {path.name} tenant_count={tenant_count} trial={int(trial['trial_index']) + 1}",
                flush=True,
            )
            row = run_case(
                path,
                metadata,
                tenant_count,
                trial,
                optimizer_time_limit=args.optimizer_time_limit,
            )
            rows.append(row)
            worse = row["worse_than_recorded_relative"]
            print(
                "  optimizer avg={:.6f}, mk={:.6f}, rt={:.2f}s, worse={:+.3%}/{:+.3%}".format(
                    row["optimizer"]["avg_jct"],
                    row["optimizer"]["makespan"],
                    row["optimizer"]["runtime_seconds"],
                    worse["avg_jct"],
                    worse["makespan"],
                ),
                flush=True,
            )

    release = release_gate_audit()
    failures = []
    if not release["passed"]:
        failures.append({"type": "release_gate", "details": release})
    for row in rows:
        recompute_delta = row["recorded_recompute_abs_delta"]
        if abs(recompute_delta["avg_jct"]) > 1e-9 or abs(recompute_delta["makespan"]) > 1e-9:
            failures.append({"type": "json_recompute", "details": row})
        worse = row["worse_than_recorded_relative"]
        if worse["avg_jct"] > args.sensitivity or worse["makespan"] > args.sensitivity:
            failures.append({"type": "optimizer_sensitivity", "details": row})
        if row["optimizer"]["surrogate_mode"] != "time_expanded":
            failures.append({"type": "wrong_surrogate_mode", "details": row})
        if row["optimizer"]["final_score_is_simulated"]:
            failures.append({"type": "simulator_rerank_used", "details": row})

    payload = {
        "metadata": {
            "inputs": [str(path) for path in args.inputs],
            "tenant_counts": sorted(tenant_counts),
            "trials_1_based": sorted(idx + 1 for idx in trial_indices),
            "sensitivity": float(args.sensitivity),
            "optimizer_time_limit_seconds": args.optimizer_time_limit,
        },
        "release_gate_audit": release,
        "results": rows,
        "failures": failures,
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved validation to {args.output}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
