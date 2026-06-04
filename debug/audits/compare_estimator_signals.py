from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from debug.audits.compare_hybrid_time_expanded_runpath import (  # noqa: E402
    build_datacenter,
    denormalize_mapping,
    denormalize_specs,
    evaluate_mapping,
)
from multitenant.baselines import build_leaf_local_mapping  # noqa: E402
from multitenant.solvers import MappingHybridHeuristicSolver  # noqa: E402


def normalize_method_mapping(raw):
    return {
        int(tenant): {int(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in raw.items()
    }


def score_text(score):
    return f"avg={score[0]:.6f}, mk={score[1]:.6f}"


def ordered_tenants(pressure, peak, tenants):
    return sorted(
        tenants,
        key=lambda tenant: (
            -float(pressure.get(int(tenant), 0.0)),
            -float(peak.get(int(tenant), 0.0)),
            int(tenant),
        ),
    )


def top_items(mapping, limit=8):
    return sorted(mapping.items(), key=lambda item: (-float(item[1]), str(item[0])))[:limit]


def summarize_prices(epoch_prices, limit=5):
    rows = []
    for epoch, price_state in enumerate(epoch_prices):
        for resource_type in ("edge", "sender", "receiver"):
            for resource, price in price_state.get(resource_type, {}).items():
                rows.append((float(price), int(epoch), resource_type, resource))
    rows.sort(key=lambda row: (-row[0], row[1], row[2], str(row[3])))
    return rows[:limit]


def mapping_diff_by_tenant(left, right):
    return {
        int(tenant): sum(
            1
            for rank, server in left[int(tenant)].items()
            if int(right[int(tenant)][int(rank)]) != int(server)
        )
        for tenant in sorted(left)
    }


def build_solver(datacenter, initial_mapping, specs, *, path_table, surrogate_mode):
    return MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        surrogate_mode=surrogate_mode,
        validate_with_simulator=False,
        verbose=False,
    )


def collect_collapsed_signals(solver, mapping):
    _, maxima, epoch_prices = solver._compute_epoch_link_prices(mapping)
    return {
        "score": solver._evaluate_surrogate_mapping(mapping),
        "tenant_pressure": dict(solver.tenant_pressure),
        "tenant_peak_load": dict(solver.tenant_peak_load),
        "rank_pressure": dict(solver.rank_pressure),
        "tenant_pair_interaction": dict(solver.tenant_pair_interaction),
        "epoch_maxima": list(maxima),
        "top_prices": summarize_prices(epoch_prices),
    }


def collect_time_expanded_signals(solver, mapping):
    state = solver._get_time_expanded_surrogate_state(mapping)
    maxima, epoch_prices = solver._time_expanded_epoch_prices_from_state(state)
    return {
        "score": state["score"],
        "tenant_pressure": dict(state.get("tenant_pressure", {})),
        "tenant_peak_load": dict(state.get("tenant_peak_load", {})),
        "rank_pressure": dict(state.get("rank_pressure", {})),
        "tenant_pair_interaction": dict(state.get("tenant_pair_interaction", {})),
        "epoch_maxima": list(maxima),
        "top_prices": summarize_prices(epoch_prices),
        "hotspots": list(state.get("hotspots", []))[:5],
        "clusters": list(state.get("contention_clusters", []))[:5],
    }


def print_signal_block(label, signals, tenants):
    score = signals["score"]
    print(f"  {label} estimator score: mk={score[0]:.6f}, avg={score[1]:.6f}")
    order = ordered_tenants(signals["tenant_pressure"], signals["tenant_peak_load"], tenants)
    print("  tenant order:", order)
    print(
        "  tenant pressure:",
        [(tenant, round(float(signals["tenant_pressure"].get(tenant, 0.0)), 6)) for tenant in order],
    )
    print(
        "  tenant peak:",
        [(tenant, round(float(signals["tenant_peak_load"].get(tenant, 0.0)), 6)) for tenant in order],
    )
    print(
        "  top rank pressure:",
        [
            (str(key), round(float(value), 6))
            for key, value in top_items(signals["rank_pressure"], limit=10)
        ],
    )
    print(
        "  tenant pair interaction:",
        [
            (str(key), round(float(value), 6))
            for key, value in top_items(signals["tenant_pair_interaction"], limit=8)
        ],
    )
    print(
        "  top resource prices:",
        [
            (round(price, 9), epoch, resource_type, str(resource))
            for price, epoch, resource_type, resource in signals["top_prices"]
        ],
    )


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
        description="Compare collapsed-hybrid and time-expanded estimator signals for one JSON case."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tenant-count", type=int, required=True)
    parser.add_argument("--trial", type=int, default=1, help="1-based trial index")
    parser.add_argument(
        "--compare-output",
        type=Path,
        default=REPO_ROOT / "debug" / "audits" / "hybrid_vs_time_expanded_t3_t4_trial1.json",
        help="Optional run-path comparison output containing hybrid/time-expanded mappings.",
    )
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

    named_mappings = {
        "default": initial_mapping,
        "locality": locality_mapping,
        "json_mapping": json_mapping,
    }
    if args.compare_output.exists():
        comparison = json.loads(args.compare_output.read_text(encoding="utf-8"))
        for row in comparison.get("results", []):
            if (
                Path(row["input"]).name == args.input.name
                and int(row["tenant_count"]) == int(args.tenant_count)
                and int(row["trial_index"]) == int(args.trial) - 1
            ):
                methods = row["methods"]
                named_mappings["hybrid"] = normalize_method_mapping(methods["hybrid"]["mapping"])
                named_mappings["time_expanded"] = normalize_method_mapping(methods["time_expanded"]["mapping"])
                break

    collapsed = build_solver(
        datacenter,
        initial_mapping,
        specs,
        path_table=path_table,
        surrogate_mode="collapsed",
    )
    time_expanded = build_solver(
        datacenter,
        initial_mapping,
        specs,
        path_table=path_table,
        surrogate_mode="time_expanded",
    )
    tenants = sorted(initial_mapping)

    print(
        f"[case] {args.input.name} tenant_count={args.tenant_count} "
        f"trial={args.trial} scene={metadata.get('scene')}"
    )
    if "hybrid" in named_mappings:
        print("diff locality -> hybrid:", mapping_diff_by_tenant(locality_mapping, named_mappings["hybrid"]))
    if "time_expanded" in named_mappings:
        print(
            "diff locality -> time_expanded:",
            mapping_diff_by_tenant(locality_mapping, named_mappings["time_expanded"]),
        )

    for name, mapping in named_mappings.items():
        print(f"\n[{name}] simulator {score_text(evaluate_mapping(datacenter, mapping, specs))}")
        collapsed_signals = collect_collapsed_signals(collapsed, mapping)
        te_signals = collect_time_expanded_signals(time_expanded, mapping)
        print_signal_block("collapsed", collapsed_signals, tenants)
        print_signal_block("time-expanded", te_signals, tenants)


if __name__ == "__main__":
    main()
