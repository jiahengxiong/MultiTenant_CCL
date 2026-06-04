from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingHybridHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


def denormalize_mapping(raw: dict[str, dict[str, int]]) -> dict[int, dict[int, int]]:
    return {
        int(tenant): {int(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in raw.items()
    }


def denormalize_specs(raw: dict[str, dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def normalize_mapping(mapping: dict[int, dict[int, int]]) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {str(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in mapping.items()
    }


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


def run_current_hybrid(
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


def parse_tenant_filter(raw: str) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def parse_trial_filter(raw: str) -> set[int] | None:
    if not raw:
        return None
    # User-facing trial ids are 1-based.
    return {int(part.strip()) - 1 for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun current dominant mapping only for selected result JSON cases.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tenant-counts", type=str, default="")
    parser.add_argument("--trials", type=str, default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    topology = payload["metadata"]["topology"]
    datacenter = LeafSpineDatacenter(
        num_spine=int(topology["num_spine"]),
        num_leaf=int(topology["num_leaf"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )
    tenant_filter = parse_tenant_filter(args.tenant_counts)
    trial_filter = parse_trial_filter(args.trials)

    refreshed = []
    for tenant_group in payload.get("results", []):
        tenant_count = int(tenant_group["tenant_count"])
        if tenant_filter is not None and tenant_count not in tenant_filter:
            continue
        for trial in tenant_group.get("trial_results", []):
            trial_index = int(trial["trial_index"])
            if trial_filter is not None and trial_index not in trial_filter:
                continue
            print(
                f"[refresh] {args.input.name} tenant_count={tenant_count} trial={trial_index + 1}",
                flush=True,
            )
            initial_mapping = denormalize_mapping(trial["initial_mapping"])
            old_mapping = denormalize_mapping(trial["proposed_mapping"])
            tenant_collective_specs = denormalize_specs(trial["tenant_collective_specs"])
            old_score = evaluate_mapping(datacenter, old_mapping, tenant_collective_specs)
            new_mapping, new_score, runtime = run_current_hybrid(
                datacenter,
                initial_mapping,
                tenant_collective_specs,
            )
            avg_improvement = (
                (old_score[0] - new_score[0]) / old_score[0]
                if old_score[0] > 0
                else 0.0
            )
            mk_improvement = (
                (old_score[1] - new_score[1]) / old_score[1]
                if old_score[1] > 0
                else 0.0
            )
            refreshed.append(
                {
                    "tenant_count": tenant_count,
                    "trial_index": trial_index,
                    "old": {
                        "avg_jct": old_score[0],
                        "makespan": old_score[1],
                        "mapping": trial["proposed_mapping"],
                    },
                    "current_hybrid": {
                        "avg_jct": new_score[0],
                        "makespan": new_score[1],
                        "runtime_seconds": runtime,
                        "mapping": normalize_mapping(new_mapping),
                    },
                    "improvement": {
                        "avg_jct_relative": float(avg_improvement),
                        "makespan_relative": float(mk_improvement),
                    },
                }
            )
            print(
                "  old avg={:.6f}, current avg={:.6f}, improvement={:.3%}; "
                "old mk={:.6f}, current mk={:.6f}, improvement={:.3%}".format(
                    old_score[0],
                    new_score[0],
                    avg_improvement,
                    old_score[1],
                    new_score[1],
                    mk_improvement,
                ),
                flush=True,
            )

    output = {
        "metadata": {
            "input": str(args.input),
            "tenant_counts": sorted(tenant_filter) if tenant_filter is not None else "all",
            "trials_1_based": sorted(idx + 1 for idx in trial_filter) if trial_filter is not None else "all",
            "note": "Mapping-only refresh. Harmonics results are not recomputed here.",
        },
        "results": refreshed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved refresh report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
