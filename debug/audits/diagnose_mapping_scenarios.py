from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from solve_case_once import denormalize_mapping, denormalize_specs, load_trial, normalize_mapping


def objective(makespan: float, avg_jct: float) -> tuple[float, float]:
    return (float(avg_jct), float(makespan))


def denormalize_json_mapping(raw: dict[str, dict[str, int]] | None):
    if raw is None:
        return None
    return denormalize_mapping(raw)


def server_occupancy(mapping: dict[int, dict[int, int]]) -> dict[str, Any]:
    server_to_tenants: dict[int, set[int]] = defaultdict(set)
    for tenant, rank_to_server in mapping.items():
        for server in rank_to_server.values():
            server_to_tenants[int(server)].add(int(tenant))
    occupancies = [len(tenants) for tenants in server_to_tenants.values()]
    histogram = Counter(occupancies)
    return {
        "occupied_servers": len(server_to_tenants),
        "max_tenants_per_server": max(occupancies, default=0),
        "tenant_count_histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
    }


def workload_summary(specs: dict[int, dict[str, object]]) -> dict[str, Any]:
    profiles = [str(spec.get("profile_name", "unknown")) for spec in specs.values()]
    collectives = sorted({str(spec.get("collective", "unknown")) for spec in specs.values()})
    sizes = [int(spec.get("single_flow_size_bits", 0)) for spec in specs.values()]
    return {
        "profiles": profiles,
        "collectives": collectives,
        "single_flow_size_bits_min": int(min(sizes, default=0)),
        "single_flow_size_bits_max": int(max(sizes, default=0)),
        "single_flow_size_bits_by_tenant": {
            str(tenant): int(spec.get("single_flow_size_bits", 0))
            for tenant, spec in sorted(specs.items())
        },
    }


def path_overlap_summary(estimator, mapping: dict[int, dict[int, int]]) -> dict[str, Any]:
    task_dag = estimator.task_dag
    unweighted_edge_load: Counter = Counter()
    weighted_edge_load: defaultdict = defaultdict(float)
    tenant_edge_load: dict[int, Counter] = {}
    total_path_uses = 0
    inter_server_tasks = 0
    volume_sum = 0.0

    for tenant, meta in task_dag.items():
        tenant = int(tenant)
        tenant_counter: Counter = Counter()
        for task_id, task_tuple in meta.get("task_info", {}).items():
            _tid, src_rank, dst_rank, volume = task_tuple
            src_server = int(mapping[tenant][int(src_rank)])
            dst_server = int(mapping[tenant][int(dst_rank)])
            if src_server == dst_server:
                continue
            inter_server_tasks += 1
            volume = float(volume)
            volume_sum += volume
            path = estimator.path_edges_for_pair(tenant, src_server, dst_server)
            total_path_uses += len(path)
            for edge in path:
                unweighted_edge_load[edge] += 1
                tenant_counter[edge] += 1
                weighted_edge_load[edge] += volume
        tenant_edge_load[tenant] = tenant_counter

    max_edge_load = max(unweighted_edge_load.values(), default=0)
    shared_edges = {
        edge: load for edge, load in unweighted_edge_load.items() if int(load) > 1
    }
    weighted_values = list(weighted_edge_load.values())
    top_edges = [
        {
            "edge": list(edge),
            "task_count": int(unweighted_edge_load[edge]),
            "volume_bits": float(weighted_edge_load[edge]),
        }
        for edge, _load in sorted(
            unweighted_edge_load.items(),
            key=lambda item: (-int(item[1]), -float(weighted_edge_load[item[0]]), str(item[0])),
        )[:10]
    ]
    return {
        "inter_server_tasks": int(inter_server_tasks),
        "total_task_path_edge_uses": int(total_path_uses),
        "unique_edges": int(len(unweighted_edge_load)),
        "shared_edges": int(len(shared_edges)),
        "max_edge_task_count": int(max_edge_load),
        "sum_edge_task_excess_over_one": int(sum(max(0, int(v) - 1) for v in unweighted_edge_load.values())),
        "max_edge_volume_bits": float(max(weighted_values, default=0.0)),
        "total_inter_server_volume_bits": float(volume_sum),
        "top_edges": top_edges,
    }


def top_hotspots(analysis, limit: int) -> list[dict[str, Any]]:
    rows = []
    for item in analysis.hotspots[:limit]:
        resource_type, resource_id = item["resource"]
        resource = [resource_type, list(resource_id) if isinstance(resource_id, tuple) else resource_id]
        rows.append(
            {
                "slot": int(item["slot"]),
                "resource": resource,
                "load": float(item["load"]),
                "excess": float(item["excess"]),
            }
        )
    return rows


def top_clusters(analysis, limit: int) -> list[dict[str, Any]]:
    rows = []
    for item in analysis.contention_clusters[:limit]:
        resource_type, resource_id = item["resource"]
        resource = [resource_type, list(resource_id) if isinstance(resource_id, tuple) else resource_id]
        rows.append(
            {
                "slot": int(item["slot"]),
                "resource": resource,
                "excess": float(item["excess"]),
                "tenant_count": len(item.get("tenants", ())),
                "rank_count": len(item.get("ranks", ())),
                "task_count": len(item.get("tasks", ())),
                "tenants": [int(tenant) for tenant in item.get("tenants", ())],
                "ranks": [[int(tenant), int(rank)] for tenant, rank in item.get("ranks", ())[:20]],
            }
        )
    return rows


def evaluate_named_mapping(
    *,
    name: str,
    datacenter,
    estimator,
    simulator,
    mapping: dict[int, dict[int, int]],
    specs: dict[int, dict[str, object]],
    hotspot_limit: int,
) -> dict[str, Any]:
    start = time.time()
    estimate = estimator.evaluate(mapping)
    analysis = estimator.analyze(mapping)
    estimator_seconds = time.time() - start
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulator(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return {
        "name": name,
        "estimator": {
            "avg_jct": float(estimate.avg_jct),
            "makespan": float(estimate.makespan),
            "score": list(objective(estimate.makespan, estimate.avg_jct)),
            "seconds": float(estimator_seconds),
        },
        "simulator": {
            "avg_jct": float(avg_jct),
            "makespan": float(makespan),
            "score": list(objective(makespan, avg_jct)),
        },
        "path_overlap": path_overlap_summary(estimator, mapping),
        "tenant_pressure": {
            str(tenant): float(value)
            for tenant, value in sorted(analysis.tenant_pressure.items())
        },
        "tenant_peak_load": {
            str(tenant): float(value)
            for tenant, value in sorted(analysis.tenant_peak_load.items())
        },
        "top_rank_pressure": [
            {"tenant": int(tenant), "rank": int(rank), "pressure": float(value)}
            for (tenant, rank), value in sorted(
                analysis.rank_pressure.items(),
                key=lambda item: (-float(item[1]), int(item[0][0]), int(item[0][1])),
            )[:20]
        ],
        "hotspots": top_hotspots(analysis, hotspot_limit),
        "contention_clusters": top_clusters(analysis, hotspot_limit),
        "mapping": normalize_mapping(mapping),
    }


def run_current_optimizer(datacenter, initial_mapping, specs, solver_time_limit: float | None):
    from multitenant.solvers import MappingContentionOptimizer

    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    solver = MappingContentionOptimizer(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    start = time.time()
    if solver_time_limit is None:
        solver.solve()
    else:
        solver.solve(time_limit=float(solver_time_limit))
    return solver.get_X_mapping(), time.time() - start, {
        "final_avg_jct": getattr(solver, "final_avg_jct", None),
        "final_makespan": getattr(solver, "final_makespan", None),
        "last_move_source": getattr(solver, "last_move_source", None),
        "move_source_counts": getattr(solver, "move_source_counts", None),
        "phase_runtimes": getattr(solver, "phase_runtimes", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose scenario structure and mapping quality without writing into experiment/."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tenant-count", type=int, required=True)
    parser.add_argument("--trial-index", type=int, required=True, help="0-based trial index")
    parser.add_argument("--include-current-optimizer", action="store_true")
    parser.add_argument("--solver-time-limit", type=float, default=None)
    parser.add_argument("--hotspot-limit", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from multitenant.baselines import build_leaf_local_mapping
    from multitenant.simulator import simulate_collective
    from multitenant.solvers import TimeExpandedContentionEstimator
    from multitenant.topology import LeafSpineDatacenter

    metadata, trial = load_trial(args.input, args.tenant_count, args.trial_index)
    topology = metadata["topology"]
    datacenter = LeafSpineDatacenter(
        num_spine=int(topology["num_spine"]),
        num_leaf=int(topology["num_leaf"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    specs = denormalize_specs(trial["tenant_collective_specs"])
    estimator = TimeExpandedContentionEstimator(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=datacenter.build_tenant_ecmp_path_table(initial_mapping),
    )

    mappings: list[tuple[str, dict[int, dict[int, int]], dict[str, Any]]] = [
        ("default", initial_mapping, {}),
        ("locality", build_leaf_local_mapping(initial_mapping), {}),
    ]
    json_mapping = denormalize_json_mapping(trial.get("proposed_mapping"))
    if json_mapping is not None:
        mappings.append(("json_mapping", json_mapping, {}))

    if args.include_current_optimizer:
        current_mapping, runtime, meta = run_current_optimizer(
            datacenter,
            initial_mapping,
            specs,
            args.solver_time_limit,
        )
        meta = dict(meta)
        meta["runtime_seconds"] = float(runtime)
        mappings.append(("current_contention_optimizer", current_mapping, meta))

    evaluated = []
    for name, mapping, mapping_meta in mappings:
        row = evaluate_named_mapping(
            name=name,
            datacenter=datacenter,
            estimator=estimator,
            simulator=simulate_collective,
            mapping=mapping,
            specs=specs,
            hotspot_limit=args.hotspot_limit,
        )
        row["solver_meta"] = mapping_meta
        evaluated.append(row)

    best_by_sim = min(evaluated, key=lambda row: tuple(row["simulator"]["score"]))
    best_by_est = min(evaluated, key=lambda row: tuple(row["estimator"]["score"]))
    payload = {
        "input": str(args.input),
        "scene": metadata.get("scene", args.input.stem),
        "tenant_count": int(args.tenant_count),
        "trial_index": int(args.trial_index),
        "topology": topology,
        "scenario": {
            "server_occupancy": server_occupancy(initial_mapping),
            "workload": workload_summary(specs),
            "release_gate_count": sum(len(gates) for gates in estimator.audit_release_gates().values()),
        },
        "best_by_simulator": best_by_sim["name"],
        "best_by_estimator": best_by_est["name"],
        "mappings": evaluated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    printable = {
        "scene": payload["scene"],
        "tenant_count": payload["tenant_count"],
        "trial_index": payload["trial_index"],
        "scenario": payload["scenario"],
        "best_by_simulator": payload["best_by_simulator"],
        "best_by_estimator": payload["best_by_estimator"],
        "mappings": [
            {
                "name": row["name"],
                "est_avg": row["estimator"]["avg_jct"],
                "est_mk": row["estimator"]["makespan"],
                "sim_avg": row["simulator"]["avg_jct"],
                "sim_mk": row["simulator"]["makespan"],
                "max_edge_task_count": row["path_overlap"]["max_edge_task_count"],
                "shared_edges": row["path_overlap"]["shared_edges"],
                "top_hotspot": row["hotspots"][0] if row["hotspots"] else None,
                "solver_meta": row["solver_meta"],
            }
            for row in evaluated
        ],
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
