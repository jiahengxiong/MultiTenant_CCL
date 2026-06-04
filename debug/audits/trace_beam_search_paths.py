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
    datacenter = build_datacenter(payload["metadata"])
    return (
        payload,
        datacenter,
        denormalize_mapping(trial["initial_mapping"]),
        denormalize_specs(trial["tenant_collective_specs"]),
    )


def score_dict(score: tuple[float, float]) -> dict[str, float]:
    makespan, avg_jct = score
    return {"avg_jct": float(avg_jct), "makespan": float(makespan)}


def simulator_score(datacenter: LeafSpineDatacenter, mapping: Mapping, specs: Specs) -> tuple[float, float]:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return float(makespan), float(avg_jct)


def mapping_diff_count(left: Mapping, right: Mapping) -> int:
    return sum(
        1
        for tenant, ranks in left.items()
        for rank, server in ranks.items()
        if int(right[int(tenant)][int(rank)]) != int(server)
    )


def normalize_mapping(mapping: Mapping) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {
            str(rank): int(server)
            for rank, server in ranks.items()
        }
        for tenant, ranks in mapping.items()
    }


def annotate_candidates(
    *,
    candidates,
    base_mapping: Mapping,
    scorer_name: str,
    datacenter: LeafSpineDatacenter,
    specs: Specs,
    te_estimator: TimeExpandedContentionEstimator,
    collapsed_solver: MappingHybridHeuristicSolver,
    top_k: int,
):
    annotated = []
    seen = set()
    for score, mapping, source in candidates:
        signature = tuple(
            (tenant, tuple(sorted(ranks.items())))
            for tenant, ranks in sorted(mapping.items())
        )
        if signature in seen:
            continue
        seen.add(signature)
        te_score = te_estimator.evaluate(mapping).score
        collapsed_score = collapsed_solver._evaluate_surrogate_mapping(mapping)
        sim_score = simulator_score(datacenter, mapping, specs)
        annotated.append(
            {
                "source": source,
                f"{scorer_name}_score": score_dict(score),
                "time_expanded_score": score_dict(te_score),
                "collapsed_score": score_dict(collapsed_score),
                "simulator_score": score_dict(sim_score),
                "diff_from_round_base": mapping_diff_count(base_mapping, mapping),
            }
        )
        if len(annotated) >= top_k:
            break
    return annotated


def trace_collapsed_hybrid(
    datacenter: LeafSpineDatacenter,
    initial_mapping: Mapping,
    specs: Specs,
    *,
    max_rounds: int,
    top_k: int,
):
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    te_estimator = TimeExpandedContentionEstimator(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
    )
    deadline = time.time() + 10_000.0
    initial_candidates = []
    best_mapping = None
    best_score = (float("inf"), float("inf"))
    for seed in solver._seed_mappings():
        score = solver._evaluate_surrogate_mapping(seed)
        initial_candidates.append((score, seed, "seed"))
        if solver._is_better_objective(score, best_score):
            best_score = score
            best_mapping = seed
    beam = solver._dedupe_ranked_candidates(initial_candidates) or [(best_score, best_mapping, "seed")]

    rounds = []
    for round_idx in range(1, max_rounds + 1):
        next_candidates = list(beam)
        round_records = []
        for beam_index, (beam_score, beam_mapping, beam_source) in enumerate(beam):
            _loads, _epoch_maxima, epoch_prices = solver._compute_epoch_link_prices(beam_mapping)
            tenant_order = sorted(
                solver.tenants,
                key=lambda tenant: (
                    -solver.tenant_pressure.get(tenant, 0.0),
                    -solver.tenant_peak_load.get(tenant, 0.0),
                    tenant,
                ),
            )
            generated = []
            for tenant in tenant_order:
                source, candidate_mapping, candidate_score = solver._best_neighborhood_candidate(
                    beam_mapping,
                    beam_score,
                    tenant,
                    epoch_prices,
                    deadline,
                    include_local=True,
                    allow_bnb=False,
                )
                if source is not None:
                    entry = (candidate_score, candidate_mapping, f"{source}:t{tenant}")
                    generated.append(entry)
                    next_candidates.append(entry)
            for tenant in tenant_order[: solver._effective_bnb_tenant_count(tenant_order)]:
                source, candidate_mapping, candidate_score = solver._best_neighborhood_candidate(
                    beam_mapping,
                    beam_score,
                    tenant,
                    epoch_prices,
                    deadline,
                    include_local=False,
                    allow_bnb=True,
                )
                if source is not None:
                    entry = (candidate_score, candidate_mapping, f"{source}:t{tenant}")
                    generated.append(entry)
                    next_candidates.append(entry)
            joint_tenants = solver._effective_joint_tenants(tenant_order)
            for idx, tenant_a in enumerate(joint_tenants):
                for tenant_b in joint_tenants[idx + 1:]:
                    for candidate_mapping in solver._joint_pair_candidate(
                        beam_mapping,
                        tenant_a,
                        tenant_b,
                        epoch_prices,
                        deadline,
                    ) or []:
                        candidate_score = solver._evaluate_surrogate_mapping(candidate_mapping)
                        if solver._is_better_objective(candidate_score, beam_score):
                            entry = (candidate_score, candidate_mapping, f"joint:t{tenant_a}-{tenant_b}")
                            generated.append(entry)
                            next_candidates.append(entry)
            generated.sort(key=lambda entry: solver._objective_sort_key(entry[0]))
            round_records.append(
                {
                    "beam_index": int(beam_index),
                    "beam_source": beam_source,
                    "beam_score": score_dict(beam_score),
                    "tenant_order": [int(tenant) for tenant in tenant_order],
                    "top_generated": annotate_candidates(
                        candidates=generated,
                        base_mapping=beam_mapping,
                        scorer_name="collapsed",
                        datacenter=datacenter,
                        specs=specs,
                        te_estimator=te_estimator,
                        collapsed_solver=solver,
                        top_k=top_k,
                    ),
                }
            )
        ranked = solver._dedupe_ranked_candidates(next_candidates)
        top_score, top_mapping, top_source = ranked[0]
        rounds.append(
            {
                "round": int(round_idx),
                "beam": annotate_candidates(
                    candidates=beam,
                    base_mapping=best_mapping,
                    scorer_name="collapsed",
                    datacenter=datacenter,
                    specs=specs,
                    te_estimator=te_estimator,
                    collapsed_solver=solver,
                    top_k=top_k,
                ),
                "expanded": round_records,
                "selected_source": top_source,
                "selected_score": score_dict(top_score),
            }
        )
        if not solver._is_better_objective(top_score, best_score):
            break
        best_score = top_score
        best_mapping = top_mapping
        beam = ranked
    return {
        "best_score": score_dict(best_score),
        "best_mapping": normalize_mapping(best_mapping),
        "rounds": rounds,
    }


def trace_time_expanded(
    datacenter: LeafSpineDatacenter,
    initial_mapping: Mapping,
    specs: Specs,
    *,
    max_rounds: int,
    top_k: int,
):
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    solver = MappingEstimatorBlackBoxOptimizer(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        search_skeleton="hybrid",
        verbose=False,
    )
    collapsed_solver = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    deadline = time.time() + 10_000.0
    initial_candidates = []
    best_mapping = None
    best_score = (float("inf"), float("inf"))
    for seed in solver._seed_mappings():
        score = solver._evaluate_mapping(seed)
        initial_candidates.append((score, seed, "seed"))
        if solver._is_better_objective(score, best_score):
            best_score = score
            best_mapping = seed
    beam = solver._dedupe_ranked_candidates_hybrid_order(initial_candidates) or [(best_score, best_mapping, "seed")]

    rounds = []
    for round_idx in range(1, max_rounds + 1):
        next_candidates = list(beam)
        round_records = []
        for beam_index, (beam_score, beam_mapping, beam_source) in enumerate(beam):
            analysis = solver._analyze_mapping(beam_mapping)
            tenant_order = solver._tenant_order(analysis)
            generated = []
            for tenant in tenant_order:
                source, candidate_mapping, candidate_score = solver._best_neighborhood_candidate(
                    beam_mapping,
                    beam_score,
                    tenant,
                    analysis,
                    deadline,
                    include_local=True,
                    allow_bnb=False,
                )
                if source is not None:
                    entry = (candidate_score, candidate_mapping, f"{source}:t{tenant}")
                    generated.append(entry)
                    next_candidates.append(entry)
            for tenant in tenant_order[: solver._effective_bnb_tenant_count(tenant_order)]:
                source, candidate_mapping, candidate_score = solver._price_guided_remap_pool(
                    beam_mapping,
                    beam_score,
                    tenant,
                    analysis,
                    min(deadline, time.time() + solver.bnb_candidate_time_limit),
                )
                if source is not None:
                    entry = (candidate_score, candidate_mapping, f"{source}:t{tenant}")
                    generated.append(entry)
                    next_candidates.append(entry)
            joint_tenants = solver._effective_joint_tenants(tenant_order)
            for idx, tenant_a in enumerate(joint_tenants):
                for tenant_b in joint_tenants[idx + 1:]:
                    for candidate_mapping in solver._joint_pair_candidate(
                        beam_mapping,
                        tenant_a,
                        tenant_b,
                        analysis,
                        deadline,
                    ):
                        candidate_score = solver._evaluate_mapping(candidate_mapping)
                        if solver._is_better_objective(candidate_score, beam_score):
                            entry = (candidate_score, candidate_mapping, f"joint:t{tenant_a}-{tenant_b}")
                            generated.append(entry)
                            next_candidates.append(entry)
            generated.sort(key=lambda entry: solver._objective_sort_key(entry[0]))
            round_records.append(
                {
                    "beam_index": int(beam_index),
                    "beam_source": beam_source,
                    "beam_score": score_dict(beam_score),
                    "tenant_order": [int(tenant) for tenant in tenant_order],
                    "top_generated": annotate_candidates(
                        candidates=generated,
                        base_mapping=beam_mapping,
                        scorer_name="time_expanded",
                        datacenter=datacenter,
                        specs=specs,
                        te_estimator=solver.estimator,
                        collapsed_solver=collapsed_solver,
                        top_k=top_k,
                    ),
                }
            )
        ranked = solver._dedupe_ranked_candidates_hybrid_order(next_candidates)
        top_score, top_mapping, top_source = ranked[0]
        rounds.append(
            {
                "round": int(round_idx),
                "beam": annotate_candidates(
                    candidates=beam,
                    base_mapping=best_mapping,
                    scorer_name="time_expanded",
                    datacenter=datacenter,
                    specs=specs,
                    te_estimator=solver.estimator,
                    collapsed_solver=collapsed_solver,
                    top_k=top_k,
                ),
                "expanded": round_records,
                "selected_source": top_source,
                "selected_score": score_dict(top_score),
            }
        )
        if not solver._is_better_objective(top_score, best_score):
            break
        best_score = top_score
        best_mapping = top_mapping
        beam = ranked
    return {
        "best_score": score_dict(best_score),
        "best_mapping": normalize_mapping(best_mapping),
        "rounds": rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace collapsed-hybrid and time-expanded beam paths.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "experiment" / "Low_contension.json")
    parser.add_argument("--tenant-count", type=int, default=5)
    parser.add_argument("--trial", type=int, default=1, help="1-based trial index")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "debug" / "audits" / "trace_beam_search_paths.json",
    )
    args = parser.parse_args()

    _payload, datacenter, initial_mapping, specs = load_case(
        args.input,
        args.tenant_count,
        args.trial,
    )
    output = {
        "metadata": {
            "input": str(args.input),
            "tenant_count": int(args.tenant_count),
            "trial_1_based": int(args.trial),
            "note": "Both traces start from the same initial mapping and ignore stored JSON solver results.",
        },
        "collapsed_hybrid": trace_collapsed_hybrid(
            datacenter,
            initial_mapping,
            specs,
            max_rounds=int(args.max_rounds),
            top_k=int(args.top_k),
        ),
        "time_expanded": trace_time_expanded(
            datacenter,
            initial_mapping,
            specs,
            max_rounds=int(args.max_rounds),
            top_k=int(args.top_k),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved trace to {args.output}")
    for label in ("collapsed_hybrid", "time_expanded"):
        trace = output[label]
        print(label, "best", trace["best_score"])
        for round_entry in trace["rounds"]:
            print(
                f"  round {round_entry['round']} selected={round_entry['selected_source']} "
                f"score={round_entry['selected_score']}"
            )
            if round_entry["expanded"]:
                first = round_entry["expanded"][0]
                print("    tenant_order", first["tenant_order"])
                print("    top_generated", first["top_generated"][:2])


if __name__ == "__main__":
    main()
