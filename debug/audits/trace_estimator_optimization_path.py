from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from debug.audits.compare_hybrid_time_expanded_runpath import (  # noqa: E402
    build_datacenter,
    denormalize_mapping,
    denormalize_specs,
    evaluate_mapping,
    normalize_mapping,
)
from multitenant.baselines import build_leaf_local_mapping  # noqa: E402
from multitenant.solvers import MappingEstimatorBlackBoxOptimizer, MappingHybridHeuristicSolver  # noqa: E402


def signature_diff_by_tenant(reference, candidate):
    return {
        int(tenant): sum(
            1
            for rank, server in reference[int(tenant)].items()
            if int(candidate[int(tenant)][int(rank)]) != int(server)
        )
        for tenant in sorted(reference)
    }


def score_pair(score):
    return {"makespan": float(score[0]), "avg_jct": float(score[1])}


def simulator_score(datacenter, specs, mapping):
    avg, mk = evaluate_mapping(datacenter, mapping, specs)
    return {"makespan": float(mk), "avg_jct": float(avg)}


def map_label(mapping, known):
    sig = known["_sig"](mapping)
    return known.get(sig, "unlabeled")


def top_dict_items(mapping, limit=8):
    return [
        [str(key), float(value)]
        for key, value in sorted(mapping.items(), key=lambda item: (-float(item[1]), str(item[0])))[:limit]
    ]


def collect_signals(solver, mapping):
    _, maxima, epoch_prices = solver._compute_epoch_link_prices(mapping)
    tenant_order = sorted(
        solver.tenants,
        key=lambda tenant: (
            -float(solver.tenant_pressure.get(tenant, 0.0)),
            -float(solver.tenant_peak_load.get(tenant, 0.0)),
            int(tenant),
        ),
    )
    return {
        "score": score_pair(solver._evaluate_surrogate_mapping(mapping)),
        "tenant_order": [int(tenant) for tenant in tenant_order],
        "tenant_pressure": {str(k): float(v) for k, v in solver.tenant_pressure.items()},
        "tenant_peak_load": {str(k): float(v) for k, v in solver.tenant_peak_load.items()},
        "top_rank_pressure": top_dict_items(getattr(solver, "rank_pressure", {}), limit=12),
        "top_pair_interaction": top_dict_items(getattr(solver, "tenant_pair_interaction", {}), limit=8),
        "epoch_maxima_head": [float(value) for value in maxima[:12]],
        "epoch_prices": epoch_prices,
    }


def candidate_record(
    *,
    source_solver_name,
    operator,
    tenant_scope,
    base_mapping,
    candidate,
    collapsed,
    time_expanded,
    datacenter,
    specs,
    known_labels,
):
    same = collapsed._mapping_signature(candidate) == collapsed._mapping_signature(base_mapping)
    return {
        "source_solver": source_solver_name,
        "operator": operator,
        "tenant_scope": [int(t) for t in tenant_scope],
        "same_as_base": bool(same),
        "label": map_label(candidate, known_labels),
        "diff_from_base": signature_diff_by_tenant(base_mapping, candidate),
        "collapsed_score": score_pair(collapsed._evaluate_surrogate_mapping(candidate)),
        "time_expanded_score": score_pair(time_expanded._evaluate_surrogate_mapping(candidate)),
        "simulator_score": simulator_score(datacenter, specs, candidate),
        "mapping": normalize_mapping(candidate),
    }


def generate_one_step_candidates(
    solver,
    solver_name,
    base_mapping,
    base_score,
    signals,
    *,
    collapsed,
    time_expanded,
    datacenter,
    specs,
    known_labels,
    deadline,
):
    epoch_prices = signals["epoch_prices"]
    tenant_order = list(signals["tenant_order"])
    records = []

    for tenant in tenant_order:
        local_mapping, _ = solver._optimize_tenant_block_with_prices(
            base_mapping,
            tenant,
            epoch_prices,
            deadline,
        )
        records.append(
            candidate_record(
                source_solver_name=solver_name,
                operator="local",
                tenant_scope=[tenant],
                base_mapping=base_mapping,
                candidate=local_mapping,
                collapsed=collapsed,
                time_expanded=time_expanded,
                datacenter=datacenter,
                specs=specs,
                known_labels=known_labels,
            )
        )

    for tenant in tenant_order[: solver._effective_bnb_tenant_count(tenant_order)]:
        bnb_mapping = solver._build_price_bnb_candidate(
            base_mapping,
            tenant,
            epoch_prices,
            deadline,
        )
        records.append(
            candidate_record(
                source_solver_name=solver_name,
                operator="bnb",
                tenant_scope=[tenant],
                base_mapping=base_mapping,
                candidate=bnb_mapping,
                collapsed=collapsed,
                time_expanded=time_expanded,
                datacenter=datacenter,
                specs=specs,
                known_labels=known_labels,
            )
        )

    joint_tenants = solver._effective_joint_tenants(tenant_order)
    for idx, tenant_a in enumerate(joint_tenants):
        for tenant_b in joint_tenants[idx + 1 :]:
            joint_candidates = solver._joint_pair_candidate(
                base_mapping,
                tenant_a,
                tenant_b,
                epoch_prices,
                deadline,
            ) or []
            for joint_mapping in joint_candidates:
                records.append(
                    candidate_record(
                        source_solver_name=solver_name,
                        operator="joint",
                        tenant_scope=[tenant_a, tenant_b],
                        base_mapping=base_mapping,
                        candidate=joint_mapping,
                        collapsed=collapsed,
                        time_expanded=time_expanded,
                        datacenter=datacenter,
                        specs=specs,
                        known_labels=known_labels,
                    )
                )

    # Time-expanded-only candidates are logged separately so they cannot hide
    # differences in the common local/BnB/joint path.
    if solver.surrogate_mode == "time_expanded":
        for tenant in tenant_order[: min(2, len(tenant_order))]:
            source, cyclic_mapping, _cyclic_score = solver._cyclic_rank_order_candidate(
                base_mapping,
                base_score,
                tenant,
                epoch_prices,
                deadline,
            )
            if source is not None:
                records.append(
                    candidate_record(
                        source_solver_name=solver_name,
                        operator="cyclic",
                        tenant_scope=[tenant],
                        base_mapping=base_mapping,
                        candidate=cyclic_mapping,
                        collapsed=collapsed,
                        time_expanded=time_expanded,
                        datacenter=datacenter,
                        specs=specs,
                        known_labels=known_labels,
                    )
                )
        for tenant in tenant_order:
            source, matching_mapping, _matching_score = solver._time_expanded_rank_order_matching_candidate(
                base_mapping,
                base_score,
                tenant,
                deadline,
            )
            if source is not None:
                records.append(
                    candidate_record(
                        source_solver_name=solver_name,
                        operator="rank_order_matching",
                        tenant_scope=[tenant],
                        base_mapping=base_mapping,
                        candidate=matching_mapping,
                        collapsed=collapsed,
                        time_expanded=time_expanded,
                        datacenter=datacenter,
                        specs=specs,
                        known_labels=known_labels,
                    )
                )

    records.sort(
        key=lambda row: (
            float(row["time_expanded_score"]["avg_jct"]),
            float(row["time_expanded_score"]["makespan"]),
            row["operator"],
            row["tenant_scope"],
        )
    )
    return records


def find_trial(payload, tenant_count, trial_1_based):
    target_index = int(trial_1_based) - 1
    for group in payload["results"]:
        if int(group["tenant_count"]) != int(tenant_count):
            continue
        for trial in group["trial_results"]:
            if int(trial["trial_index"]) == target_index:
                return trial
    raise ValueError(f"tenant_count={tenant_count}, trial={trial_1_based} not found")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trace the first-step optimization path exposed by collapsed and "
            "time-expanded estimators on the same mapping."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tenant-count", type=int, required=True)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument(
        "--base",
        choices=("default", "locality", "json"),
        default="locality",
        help="Mapping from which to trace one-step candidates.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=300.0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    trial = find_trial(payload, args.tenant_count, args.trial)
    datacenter = build_datacenter(metadata)
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    locality_mapping = build_leaf_local_mapping(initial_mapping)
    json_mapping = denormalize_mapping(trial["proposed_mapping"])
    specs = denormalize_specs(trial["tenant_collective_specs"])
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)

    collapsed = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        surrogate_mode="collapsed",
        validate_with_simulator=False,
        verbose=False,
    )
    time_expanded = MappingEstimatorBlackBoxOptimizer(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        validate_with_simulator=False,
        verbose=False,
    )
    mapping_by_name = {
        "default": initial_mapping,
        "locality": locality_mapping,
        "json": json_mapping,
    }
    base_mapping = mapping_by_name[args.base]
    known_labels = {"_sig": collapsed._mapping_signature}
    for label, mapping in mapping_by_name.items():
        known_labels[collapsed._mapping_signature(mapping)] = label

    collapsed_signals = collect_signals(collapsed, base_mapping)
    time_expanded_signals = collect_signals(time_expanded, base_mapping)
    deadline = time.time() + float(args.deadline_seconds)
    collapsed_candidates = generate_one_step_candidates(
        collapsed,
        "collapsed",
        base_mapping,
        tuple(collapsed_signals["score"].values()),
        collapsed_signals,
        collapsed=collapsed,
        time_expanded=time_expanded,
        datacenter=datacenter,
        specs=specs,
        known_labels=known_labels,
        deadline=deadline,
    )
    time_expanded_candidates = generate_one_step_candidates(
        time_expanded,
        "time_expanded",
        base_mapping,
        tuple(time_expanded_signals["score"].values()),
        time_expanded_signals,
        collapsed=collapsed,
        time_expanded=time_expanded,
        datacenter=datacenter,
        specs=specs,
        known_labels=known_labels,
        deadline=deadline,
    )

    output = {
        "case": {
            "input": str(args.input),
            "scene": metadata.get("scene"),
            "tenant_count": int(args.tenant_count),
            "trial": int(args.trial),
            "base": args.base,
        },
        "base_scores": {
            "simulator": simulator_score(datacenter, specs, base_mapping),
            "collapsed": collapsed_signals["score"],
            "time_expanded": time_expanded_signals["score"],
        },
        "signals": {
            "collapsed": {k: v for k, v in collapsed_signals.items() if k != "epoch_prices"},
            "time_expanded": {k: v for k, v in time_expanded_signals.items() if k != "epoch_prices"},
        },
        "candidate_counts": {
            "collapsed": len(collapsed_candidates),
            "time_expanded": len(time_expanded_candidates),
        },
        "top_candidates_by_time_expanded_score": {
            "from_collapsed_path": collapsed_candidates[:20],
            "from_time_expanded_path": time_expanded_candidates[:20],
        },
        "best_by_simulator": {
            "from_collapsed_path": sorted(
                collapsed_candidates,
                key=lambda row: (row["simulator_score"]["avg_jct"], row["simulator_score"]["makespan"]),
            )[:20],
            "from_time_expanded_path": sorted(
                time_expanded_candidates,
                key=lambda row: (row["simulator_score"]["avg_jct"], row["simulator_score"]["makespan"]),
            )[:20],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(
        f"[trace] {args.input.name} T={args.tenant_count} trial={args.trial} base={args.base}"
    )
    print("  base simulator:", output["base_scores"]["simulator"])
    print("  collapsed order:", collapsed_signals["tenant_order"])
    print("  time-expanded order:", time_expanded_signals["tenant_order"])
    print("  candidate counts:", output["candidate_counts"])
    for source, rows in output["top_candidates_by_time_expanded_score"].items():
        print(f"  top TE-score candidates {source}:")
        for row in rows[:5]:
            print(
                "   ",
                row["source_solver"],
                row["operator"],
                row["tenant_scope"],
                "label=" + row["label"],
                "TE=",
                row["time_expanded_score"],
                "sim=",
                row["simulator_score"],
                "diff=",
                row["diff_from_base"],
            )
    print(f"Saved trace to {args.output}")


if __name__ == "__main__":
    main()
