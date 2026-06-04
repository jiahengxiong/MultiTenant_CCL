from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from solve_case_once import denormalize_mapping, denormalize_specs, load_trial, normalize_mapping


def objective_sort_key(score: tuple[float, float]) -> tuple[float, float]:
    makespan, avg_jct = score
    return (float(avg_jct), float(makespan))


def improvement(base_score: tuple[float, float], candidate_score: tuple[float, float]) -> dict[str, float]:
    base_mk, base_avg = base_score
    cand_mk, cand_avg = candidate_score
    return {
        "avg_jct_abs": float(base_avg - cand_avg),
        "avg_jct_rel": float((base_avg - cand_avg) / base_avg) if base_avg > 0 else 0.0,
        "makespan_abs": float(base_mk - cand_mk),
        "makespan_rel": float((base_mk - cand_mk) / base_mk) if base_mk > 0 else 0.0,
    }


def top_rank_pressure(solver, limit: int) -> list[dict[str, object]]:
    return [
        {"tenant": int(tenant), "rank": int(rank), "pressure": float(value)}
        for (tenant, rank), value in sorted(
            getattr(solver, "rank_pressure", {}).items(),
            key=lambda item: (-float(item[1]), int(item[0][0]), int(item[0][1])),
        )[:limit]
    ]


def tenant_order(solver) -> list[int]:
    return [
        int(tenant)
        for tenant in sorted(
            solver.tenants,
            key=lambda tenant: (
                -float(getattr(solver, "tenant_pressure", {}).get(int(tenant), 0.0)),
                -float(getattr(solver, "tenant_peak_load", {}).get(int(tenant), 0.0)),
                int(tenant),
            ),
        )
    ]


def evaluate_sim(datacenter, mapping, specs) -> tuple[float, float]:
    from multitenant.simulator import simulate_collective

    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return float(makespan), float(avg_jct)


def candidate_summary(solver, datacenter, specs, base_mapping, base_surrogate, base_sim, deadline):
    rows = []
    _loads, _maxima, epoch_prices = solver._compute_epoch_link_prices(base_mapping)
    ordered_tenants = tenant_order(solver)

    for tenant in ordered_tenants:
        if time.time() >= deadline:
            break
        local_mapping, _local_cost = solver._optimize_tenant_block_with_prices(
            base_mapping,
            int(tenant),
            epoch_prices,
            deadline,
        )
        if solver._mapping_signature(local_mapping) != solver._mapping_signature(base_mapping):
            local_surrogate = solver._evaluate_surrogate_mapping(local_mapping)
            local_sim = evaluate_sim(datacenter, local_mapping, specs)
            rows.append(
                {
                    "source": "local",
                    "tenant": int(tenant),
                    "surrogate_score": list(map(float, local_surrogate)),
                    "simulator_score": list(map(float, local_sim)),
                    "surrogate_improvement": improvement(base_surrogate, local_surrogate),
                    "simulator_improvement": improvement(base_sim, local_sim),
                }
            )

        if hasattr(solver, "_build_price_bnb_candidate"):
            bnb_mapping = solver._build_price_bnb_candidate(
                base_mapping,
                int(tenant),
                epoch_prices,
                min(deadline, time.time() + 2.0),
            )
            if solver._mapping_signature(bnb_mapping) != solver._mapping_signature(base_mapping):
                bnb_surrogate = solver._evaluate_surrogate_mapping(bnb_mapping)
                bnb_sim = evaluate_sim(datacenter, bnb_mapping, specs)
                rows.append(
                    {
                        "source": "bnb",
                        "tenant": int(tenant),
                        "surrogate_score": list(map(float, bnb_surrogate)),
                        "simulator_score": list(map(float, bnb_sim)),
                        "surrogate_improvement": improvement(base_surrogate, bnb_surrogate),
                        "simulator_improvement": improvement(base_sim, bnb_sim),
                    }
                )
    rows.sort(key=lambda row: objective_sort_key(tuple(row["surrogate_score"])))
    return rows


def solver_snapshot(mode: str, datacenter, initial_mapping, specs, deadline):
    from multitenant.solvers import MappingHybridHeuristicSolver

    solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=datacenter.build_tenant_ecmp_path_table(initial_mapping),
        surrogate_mode=mode,
        verbose=False,
    )
    base_surrogate = solver._evaluate_surrogate_mapping(initial_mapping)
    base_sim = evaluate_sim(datacenter, initial_mapping, specs)
    _loads, maxima, _prices = solver._compute_epoch_link_prices(initial_mapping)
    candidates = candidate_summary(
        solver,
        datacenter,
        specs,
        initial_mapping,
        base_surrogate,
        base_sim,
        deadline,
    )
    return {
        "mode": mode,
        "base_surrogate_score": list(map(float, base_surrogate)),
        "base_simulator_score": list(map(float, base_sim)),
        "epoch_or_slot_maxima_top": [float(value) for value in sorted(maxima, reverse=True)[:10]],
        "tenant_pressure": {
            str(tenant): float(value)
            for tenant, value in sorted(getattr(solver, "tenant_pressure", {}).items())
        },
        "tenant_peak_load": {
            str(tenant): float(value)
            for tenant, value in sorted(getattr(solver, "tenant_peak_load", {}).items())
        },
        "tenant_order": tenant_order(solver),
        "top_rank_pressure": top_rank_pressure(solver, 20),
        "candidate_rows": candidates[:12],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare collapsed and time-expanded signals exposed to the same hybrid search module."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tenant-count", type=int, required=True)
    parser.add_argument("--trial-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=120.0)
    args = parser.parse_args()

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
    deadline = time.time() + float(args.time_limit)

    payload = {
        "input": str(args.input),
        "scene": metadata.get("scene", args.input.stem),
        "tenant_count": int(args.tenant_count),
        "trial_index": int(args.trial_index),
        "initial_mapping": normalize_mapping(initial_mapping),
        "json_mapping_result": trial.get("results", {}).get("mapping", {}),
        "snapshots": [
            solver_snapshot("collapsed", datacenter, initial_mapping, specs, deadline),
            solver_snapshot("time_expanded", datacenter, initial_mapping, specs, deadline),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "scene": payload["scene"],
        "tenant_count": payload["tenant_count"],
        "trial_index": payload["trial_index"],
        "json_mapping_result": payload["json_mapping_result"],
        "snapshots": [
            {
                "mode": snap["mode"],
                "base_surrogate_score": snap["base_surrogate_score"],
                "base_simulator_score": snap["base_simulator_score"],
                "tenant_order": snap["tenant_order"],
                "tenant_pressure": snap["tenant_pressure"],
                "top_rank_pressure": snap["top_rank_pressure"][:5],
                "top_candidates": snap["candidate_rows"][:5],
            }
            for snap in payload["snapshots"]
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
