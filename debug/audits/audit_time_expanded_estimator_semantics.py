from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.solvers.contention_estimator import TimeExpandedContentionEstimator  # noqa: E402
from multitenant.topology import LeafSpineDatacenter  # noqa: E402


def build_mapping(server_count: int, tenant_count: int, ranks_per_tenant: int) -> dict[int, dict[int, int]]:
    if tenant_count * ranks_per_tenant > server_count:
        raise ValueError("not enough servers for audit mapping")
    server = 0
    mapping: dict[int, dict[int, int]] = {}
    for tenant in range(tenant_count):
        mapping[tenant] = {}
        for rank in range(ranks_per_tenant):
            mapping[tenant][rank] = server
            server += 1
    return mapping


def path_difference_count(path_table: dict[tuple[int, int, int], list[int]], tenants, servers) -> int:
    count = 0
    for src in servers:
        for dst in servers:
            if src == dst:
                continue
            seen = {tuple(path_table[(int(tenant), int(src), int(dst))]) for tenant in tenants}
            if len(seen) > 1:
                count += 1
    return count


def audit() -> dict[str, Any]:
    datacenter = LeafSpineDatacenter(num_leaf=4, num_spine=4, per_leaf_server=4)
    mapping = build_mapping(datacenter.num_server, tenant_count=2, ranks_per_tenant=4)
    programs = {
        tenant: [
            {"collective": "allgather", "single_flow_size_bits": 64 * 1024 * 1024 * 8, "gap_after": 0.001},
            {"collective": "reducescatter", "single_flow_size_bits": 32 * 1024 * 1024 * 8, "gap_after": 0.0},
        ]
        for tenant in mapping
    }
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    estimator = TimeExpandedContentionEstimator(
        datacenter,
        tenant_mapping=mapping,
        tenant_collective_programs=programs,
        path_table=path_table,
        slot_duration=0.00025,
        name="audit_time_expanded_semantics",
    )

    task_dag = estimator.task_dag
    release_gates = estimator.audit_release_gates()
    state = estimator._backbone._compute_time_expanded_surrogate_state(
        estimator.normalize_mapping(mapping),
        collect_signals=True,
        include_hotspots=True,
    )
    task_state = state["task_state"]
    predecessors = state.get("predecessors", {})

    release_gate_checks = []
    for tenant, gates in release_gates.items():
        for task_id, (previous_ids, gap_after) in gates.items():
            task_key = (int(tenant), int(task_id))
            previous_keys = [(int(tenant), int(prev)) for prev in previous_ids]
            previous_finish = max(float(task_state[key]["finish_time"]) for key in previous_keys)
            start_time = float(task_state[task_key]["start_time"])
            release_gate_checks.append(
                {
                    "tenant": int(tenant),
                    "task_id": int(task_id),
                    "previous_task_count": len(previous_ids),
                    "gap_after_s": float(gap_after),
                    "previous_finish_s": previous_finish,
                    "task_start_s": start_time,
                    "satisfies_finish_plus_gap": bool(start_time + 1e-12 >= previous_finish + float(gap_after)),
                    "has_release_predecessor_edges": all(prev in predecessors.get(task_key, []) for prev in previous_keys),
                }
            )

    dag_summary = {}
    for tenant, meta in task_dag.items():
        dag_summary[str(tenant)] = {
            "task_count": len(meta.get("task_info", {})),
            "collective_dependency_edges": sum(len(v) for v in meta.get("collective_preds", {}).values()),
            "execution_dependency_edges": sum(len(v) for v in meta.get("execution_preds", {}).values()),
            "release_gate_count": len(meta.get("release_gates", {})),
            "sender_order_chains": {
                str(sender): len(task_ids)
                for sender, task_ids in meta.get("sender_order", {}).items()
                if len(task_ids) > 1
            },
        }

    servers = datacenter.get_all_servers()
    sample_pair = None
    for src in servers:
        for dst in servers:
            if src == dst:
                continue
            paths = {tenant: path_table[(tenant, src, dst)] for tenant in mapping}
            if len({tuple(path) for path in paths.values()}) > 1:
                sample_pair = {"src": src, "dst": dst, "paths_by_tenant": {str(k): v for k, v in paths.items()}}
                break
        if sample_pair is not None:
            break

    return {
        "estimator_input_contract": {
            "input_is_task_dag": True,
            "input_includes_topology": True,
            "input_includes_tenant_aware_ecmp_path_table": True,
        },
        "tenant_aware_ecmp": {
            "path_table_key_shape": "(tenant, src_server, dst_server)",
            "different_tenant_path_pair_count": path_difference_count(path_table, mapping.keys(), servers),
            "sample_different_path_pair": sample_pair,
        },
        "dag_summary": dag_summary,
        "release_gate_checks": release_gate_checks,
        "all_release_gates_satisfied": all(item["satisfies_finish_plus_gap"] for item in release_gate_checks),
        "all_release_gates_have_predecessor_edges": all(item["has_release_predecessor_edges"] for item in release_gate_checks),
        "score": {
            "makespan": float(state["score"][0]),
            "avg_jct": float(state["score"][1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit time-expanded estimator DAG/release/tenant-aware ECMP semantics.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "debug" / "audits" / "time_expanded_estimator_semantics.json")
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
