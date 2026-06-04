from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.baselines import build_leaf_local_mapping  # noqa: E402
from multitenant.simulator import simulate_collective  # noqa: E402
from multitenant.solvers import (  # noqa: E402
    MappingEstimatorBlackBoxOptimizer,
    MappingHybridHeuristicSolver,
    TimeExpandedContentionEstimator,
)
from multitenant.topology import LeafSpineDatacenter  # noqa: E402


Mapping = dict[int, dict[int, int]]
Specs = dict[int, dict[str, Any]]


def denormalize_mapping(raw: dict[str, dict[str, int]]) -> Mapping:
    return {
        int(tenant): {
            int(rank): int(server)
            for rank, server in rank_to_server.items()
        }
        for tenant, rank_to_server in raw.items()
    }


def denormalize_specs(raw: dict[str, dict[str, Any]]) -> Specs:
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def build_datacenter(metadata: dict[str, Any]) -> LeafSpineDatacenter:
    topology = metadata["topology"]
    return LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )


def load_case(path: Path, tenant_count: int, trial_1_based: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    group = next(group for group in payload["results"] if int(group["tenant_count"]) == tenant_count)
    trial = group["trial_results"][trial_1_based - 1]
    metadata = payload["metadata"]
    datacenter = build_datacenter(metadata)
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    specs = denormalize_specs(trial["tenant_collective_specs"])
    return payload, datacenter, initial_mapping, specs


def simulator_score(datacenter: LeafSpineDatacenter, mapping: Mapping, specs: Specs):
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return {"avg_jct": float(avg_jct), "makespan": float(makespan)}


def score_tuple_to_dict(score: tuple[float, float]):
    makespan, avg_jct = score
    return {"avg_jct": float(avg_jct), "makespan": float(makespan)}


def top_items(mapping: dict[Any, float], limit: int):
    return [
        {"key": repr(key), "value": float(value)}
        for key, value in sorted(mapping.items(), key=lambda item: (-float(item[1]), repr(item[0])))[:limit]
    ]


def tenant_order_from_signals(tenants, tenant_pressure, tenant_peak_load):
    return [
        int(tenant)
        for tenant in sorted(
            tenants,
            key=lambda tenant: (
                -float(tenant_pressure.get(int(tenant), 0.0)),
                -float(tenant_peak_load.get(int(tenant), 0.0)),
                int(tenant),
            ),
        )
    ]


def mapping_diff_count(left: Mapping, right: Mapping) -> int:
    return sum(
        1
        for tenant, ranks in left.items()
        for rank, server in ranks.items()
        if int(right[int(tenant)][int(rank)]) != int(server)
    )


def collapsed_first_round_trace(
    datacenter: LeafSpineDatacenter,
    mapping: Mapping,
    specs: Specs,
    *,
    max_tenants: int,
):
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    base_score = solver._evaluate_surrogate_mapping(mapping)
    _loads, epoch_maxima, epoch_prices = solver._compute_epoch_link_prices(mapping)
    tenant_order = tenant_order_from_signals(
        solver.tenants,
        solver.tenant_pressure,
        solver.tenant_peak_load,
    )
    deadline = time.time() + 20.0
    local_candidates = []
    for tenant in tenant_order[:max_tenants]:
        source, candidate_mapping, candidate_score = solver._best_neighborhood_candidate(
            mapping,
            base_score,
            tenant,
            epoch_prices,
            deadline,
            include_local=True,
            allow_bnb=False,
        )
        local_candidates.append(
            {
                "tenant": int(tenant),
                "source": source,
                "score": score_tuple_to_dict(candidate_score),
                "improves_base": bool(solver._is_better_objective(candidate_score, base_score)),
                "diff_count": mapping_diff_count(mapping, candidate_mapping),
            }
        )
    bnb_candidates = []
    for tenant in tenant_order[:max_tenants]:
        source, candidate_mapping, candidate_score = solver._best_neighborhood_candidate(
            mapping,
            base_score,
            tenant,
            epoch_prices,
            deadline,
            include_local=False,
            allow_bnb=True,
        )
        bnb_candidates.append(
            {
                "tenant": int(tenant),
                "source": source,
                "score": score_tuple_to_dict(candidate_score),
                "improves_base": bool(solver._is_better_objective(candidate_score, base_score)),
                "diff_count": mapping_diff_count(mapping, candidate_mapping),
            }
        )
    return {
        "base_score": score_tuple_to_dict(base_score),
        "tenant_order": tenant_order,
        "tenant_pressure": {str(k): float(v) for k, v in solver.tenant_pressure.items()},
        "tenant_peak_load": {str(k): float(v) for k, v in solver.tenant_peak_load.items()},
        "top_rank_pressure": top_items(solver.rank_pressure, 12),
        "top_tenant_pair_interaction": top_items(solver.tenant_pair_interaction, 12),
        "epoch_count": len(epoch_prices),
        "top_epoch_maxima": top_items({idx: value for idx, value in enumerate(epoch_maxima)}, 12),
        "local_candidates_by_tenant": local_candidates,
        "bnb_candidates_by_tenant": bnb_candidates,
        "search_process_note": (
            "Collapsed hybrid evaluates a beam. In each round it gathers local "
            "candidates for every pressure-ordered tenant, then BnB candidates "
            "for selected high-pressure tenants, then joint candidates, and "
            "chooses the globally best surrogate candidate in the expanded beam."
        ),
    }


def time_expanded_first_round_trace(
    datacenter: LeafSpineDatacenter,
    mapping: Mapping,
    specs: Specs,
    *,
    max_tenants: int,
):
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    estimator = TimeExpandedContentionEstimator(
        datacenter,
        tenant_mapping=mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
    )
    optimizer = MappingEstimatorBlackBoxOptimizer(
        datacenter,
        tenant_mapping=mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    base_estimate = estimator.evaluate(mapping)
    analysis = estimator.analyze(mapping)
    tenant_order = optimizer._tenant_order(analysis)
    local_candidates = []
    deadline = time.time() + 20.0
    for tenant in tenant_order[:max_tenants]:
        source, candidate_mapping, candidate_score = optimizer._best_neighborhood_candidate(
            mapping,
            base_estimate.score,
            tenant,
            analysis,
            deadline,
            include_local=True,
            allow_bnb=False,
        )
        local_candidates.append(
            {
                "tenant": int(tenant),
                "source": source,
                "score": score_tuple_to_dict(candidate_score),
                "improves_base": bool(optimizer._is_better_objective(candidate_score, base_estimate.score)),
                "diff_count": mapping_diff_count(mapping, candidate_mapping),
            }
        )
    bnb_candidates = []
    for tenant in tenant_order[:max_tenants]:
        best = None
        source, candidate_mapping, candidate_score = optimizer._price_guided_remap_pool(
            mapping,
            base_estimate.score,
            tenant,
            analysis,
            time.time() + 20.0,
        )
        if source is not None:
            best = {
                "tenant": int(tenant),
                "source": source,
                "score": score_tuple_to_dict(candidate_score),
                "improves_base": bool(optimizer._is_better_objective(candidate_score, base_estimate.score)),
                "diff_count": mapping_diff_count(mapping, candidate_mapping),
            }
        if best is not None:
            bnb_candidates.append(best)
    return {
        "base_score": score_tuple_to_dict(base_estimate.score),
        "tenant_order": tenant_order,
        "tenant_pressure": {str(k): float(v) for k, v in analysis.tenant_pressure.items()},
        "tenant_peak_load": {str(k): float(v) for k, v in analysis.tenant_peak_load.items()},
        "top_rank_pressure": top_items(analysis.rank_pressure, 12),
        "top_tenant_pair_interaction": top_items(analysis.tenant_pair_interaction, 12),
        "hotspot_count": len(analysis.hotspots),
        "top_hotspots": analysis.hotspots[:8],
        "contention_cluster_count": len(analysis.contention_clusters),
        "local_candidates_by_tenant": local_candidates,
        "bnb_candidates_by_tenant": bnb_candidates,
        "search_process_note": (
            "The standalone time-expanded optimizer uses the same high-level "
            "multi-neighborhood beam skeleton as collapsed hybrid. This trace "
            "compares the first-round signals emitted by the two estimators."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace signal and first-step search differences between collapsed hybrid and time-expanded mapping.",
    )
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "experiment" / "Low_contension.json")
    parser.add_argument("--tenant-count", type=int, default=3)
    parser.add_argument("--trial", type=int, default=1, help="1-based trial index")
    parser.add_argument("--mapping", choices=("initial", "locality"), default="locality")
    parser.add_argument("--max-tenants", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "debug" / "audits" / "trace_hybrid_vs_time_expanded_signals.json",
    )
    args = parser.parse_args()

    _payload, datacenter, initial_mapping, specs = load_case(
        args.input,
        args.tenant_count,
        args.trial,
    )
    mapping = initial_mapping
    if args.mapping == "locality":
        mapping = build_leaf_local_mapping(initial_mapping)

    output = {
        "metadata": {
            "input": str(args.input),
            "tenant_count": int(args.tenant_count),
            "trial_1_based": int(args.trial),
            "mapping_point": args.mapping,
            "note": (
                "This audit compares signals and first-round candidates at the "
                "same mapping point; it does not use stored JSON result fields."
            ),
        },
        "simulator_score_at_mapping": simulator_score(datacenter, mapping, specs),
        "collapsed_hybrid": collapsed_first_round_trace(
            datacenter,
            mapping,
            specs,
            max_tenants=int(args.max_tenants),
        ),
        "time_expanded": time_expanded_first_round_trace(
            datacenter,
            mapping,
            specs,
            max_tenants=int(args.max_tenants),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    collapsed = output["collapsed_hybrid"]
    expanded = output["time_expanded"]
    print(f"Saved trace to {args.output}")
    print("Simulator score:", output["simulator_score_at_mapping"])
    print("Collapsed tenant order:", collapsed["tenant_order"])
    print("Time-expanded tenant order:", expanded["tenant_order"])
    print("Collapsed local candidates:", collapsed["local_candidates_by_tenant"])
    print("Time-expanded local candidates:", expanded["local_candidates_by_tenant"])
    print("Collapsed BnB candidates:", collapsed["bnb_candidates_by_tenant"])
    print("Time-expanded BnB candidates:", expanded["bnb_candidates_by_tenant"])


if __name__ == "__main__":
    main()
