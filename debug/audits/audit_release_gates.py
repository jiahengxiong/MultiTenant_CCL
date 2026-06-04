from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from multitenant.solvers import TimeExpandedContentionEstimator
from multitenant.topology import LeafSpineDatacenter


def main() -> None:
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
    samples = []
    for tenant, tenant_gates in gates.items():
        for task_id, (previous_tasks, gap_after) in tenant_gates.items():
            for previous_task in previous_tasks:
                checked_edges += 1
                start_time = analysis.task_state[(tenant, task_id)]["start_time"]
                previous_finish = analysis.task_state[(tenant, previous_task)]["finish_time"]
                sample = {
                    "tenant": int(tenant),
                    "task": int(task_id),
                    "previous_task": int(previous_task),
                    "start_time": float(start_time),
                    "previous_finish": float(previous_finish),
                    "gap_after": float(gap_after),
                }
                if len(samples) < 5:
                    samples.append(sample)
                if start_time + 1e-9 < previous_finish + gap_after:
                    violations.append(sample)

    payload = {
        "gate_counts": {str(tenant): len(gate) for tenant, gate in gates.items()},
        "checked_release_edges": int(checked_edges),
        "violations": violations,
        "sample_edges": samples,
        "avg_jct": analysis.estimate.avg_jct,
        "makespan": analysis.estimate.makespan,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
