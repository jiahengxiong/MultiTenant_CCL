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
from multitenant.solvers import MappingEstimatorBlackBoxOptimizer, MappingHybridHeuristicSolver  # noqa: E402
from multitenant.topology import LeafSpineDatacenter  # noqa: E402


Mapping = dict[int, dict[int, int]]
Specs = dict[int, dict[str, Any]]


def denormalize_mapping(raw: dict[str, dict[str, int]]) -> Mapping:
    return {
        int(tenant): {int(rank): int(server) for rank, server in ranks.items()}
        for tenant, ranks in raw.items()
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
    group = next(item for item in payload["results"] if int(item["tenant_count"]) == int(tenant_count))
    trial = group["trial_results"][int(trial_1_based) - 1]
    datacenter = build_datacenter(payload["metadata"])
    return (
        payload,
        datacenter,
        denormalize_mapping(trial["initial_mapping"]),
        denormalize_specs(trial["tenant_collective_specs"]),
    )


def score_dict(score: tuple[float, float]) -> dict[str, float]:
    makespan, avg_jct = score
    return {"makespan": float(makespan), "avg_jct": float(avg_jct)}


def simulator_score(datacenter: LeafSpineDatacenter, mapping: Mapping, specs: Specs) -> dict[str, float]:
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return {"makespan": float(makespan), "avg_jct": float(avg_jct)}


def mapping_diff_count(left: Mapping, right: Mapping) -> int:
    return sum(
        1
        for tenant, ranks in left.items()
        for rank, server in ranks.items()
        if int(right[int(tenant)][int(rank)]) != int(server)
    )


def candidate_row(
    *,
    source_family: str,
    source: str | None,
    tenant: int,
    base_mapping: Mapping,
    candidate_mapping: Mapping,
    hybrid: MappingHybridHeuristicSolver,
    expanded: MappingEstimatorBlackBoxOptimizer,
    datacenter: LeafSpineDatacenter,
    specs: Specs,
) -> dict[str, Any]:
    return {
        "source_family": source_family,
        "source": source,
        "tenant": int(tenant),
        "diff_count": int(mapping_diff_count(base_mapping, candidate_mapping)),
        "collapsed_score": score_dict(hybrid._evaluate_surrogate_mapping(candidate_mapping)),
        "time_expanded_score": score_dict(expanded._evaluate_mapping(candidate_mapping)),
        "simulator_score": simulator_score(datacenter, candidate_mapping, specs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate rankings exposed by collapsed-hybrid and "
            "time-expanded local signals at the same mapping point."
        )
    )
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "experiment" / "High_contension.json")
    parser.add_argument("--tenant-count", type=int, default=3)
    parser.add_argument("--trial", type=int, default=1, help="1-based trial index")
    parser.add_argument("--candidate-deadline", type=float, default=180.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "debug" / "audits" / "candidate_signal_rankings.json",
    )
    args = parser.parse_args()

    _payload, datacenter, base_mapping, specs = load_case(args.input, args.tenant_count, args.trial)
    path_table = datacenter.build_tenant_ecmp_path_table(base_mapping)
    hybrid = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=base_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    expanded = MappingEstimatorBlackBoxOptimizer(
        datacenter,
        tenant_mapping=base_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )

    collapsed_base = hybrid._evaluate_surrogate_mapping(base_mapping)
    expanded_base = expanded._evaluate_mapping(base_mapping)
    _loads, _epoch_maxima, epoch_prices = hybrid._compute_epoch_link_prices(base_mapping)
    hybrid_tenant_order = sorted(
        hybrid.tenants,
        key=lambda tenant: (
            -float(hybrid.tenant_pressure.get(tenant, 0.0)),
            -float(hybrid.tenant_peak_load.get(tenant, 0.0)),
            int(tenant),
        ),
    )
    analysis = expanded._analyze_mapping(base_mapping)
    expanded_tenant_order = expanded._tenant_order(analysis)

    rows: list[dict[str, Any]] = []
    deadline = time.time() + float(args.candidate_deadline)
    for tenant in hybrid_tenant_order:
        source, candidate, _score = hybrid._best_neighborhood_candidate(
            base_mapping,
            collapsed_base,
            tenant,
            epoch_prices,
            deadline,
            include_local=True,
            allow_bnb=False,
        )
        if source is not None:
            rows.append(
                candidate_row(
                    source_family="collapsed-local",
                    source=source,
                    tenant=int(tenant),
                    base_mapping=base_mapping,
                    candidate_mapping=candidate,
                    hybrid=hybrid,
                    expanded=expanded,
                    datacenter=datacenter,
                    specs=specs,
                )
            )

    for tenant in expanded_tenant_order:
        source, candidate, _score = expanded._best_neighborhood_candidate(
            base_mapping,
            expanded_base,
            tenant,
            analysis,
            deadline,
            include_local=True,
            allow_bnb=False,
        )
        if source is not None:
            rows.append(
                candidate_row(
                    source_family="time-expanded-local",
                    source=source,
                    tenant=int(tenant),
                    base_mapping=base_mapping,
                    candidate_mapping=candidate,
                    hybrid=hybrid,
                    expanded=expanded,
                    datacenter=datacenter,
                    specs=specs,
                )
            )

    output = {
        "metadata": {
            "input": str(args.input),
            "tenant_count": int(args.tenant_count),
            "trial_1_based": int(args.trial),
            "note": (
                "Stored JSON result mappings are ignored. Rows are generated "
                "live from each solver's first-round local signal at the same "
                "initial mapping and then scored by collapsed surrogate, "
                "time-expanded estimator, and simulator."
            ),
        },
        "base": {
            "collapsed_score": score_dict(collapsed_base),
            "time_expanded_score": score_dict(expanded_base),
            "simulator_score": simulator_score(datacenter, base_mapping, specs),
        },
        "tenant_order": {
            "collapsed": [int(tenant) for tenant in hybrid_tenant_order],
            "time_expanded": [int(tenant) for tenant in expanded_tenant_order],
        },
        "rows": rows,
        "rankings": {
            "by_collapsed": sorted(rows, key=lambda row: (row["collapsed_score"]["avg_jct"], row["collapsed_score"]["makespan"])),
            "by_time_expanded": sorted(rows, key=lambda row: (row["time_expanded_score"]["avg_jct"], row["time_expanded_score"]["makespan"])),
            "by_simulator": sorted(rows, key=lambda row: (row["simulator_score"]["avg_jct"], row["simulator_score"]["makespan"])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved candidate signal audit to {args.output}")
    print("Base:", output["base"])
    print("Tenant order:", output["tenant_order"])
    for key in ("by_collapsed", "by_time_expanded", "by_simulator"):
        print(key)
        for row in output["rankings"][key][:6]:
            print(
                " ",
                row["source_family"],
                "tenant",
                row["tenant"],
                "collapsed",
                row["collapsed_score"],
                "TE",
                row["time_expanded_score"],
                "sim",
                row["simulator_score"],
            )


if __name__ == "__main__":
    main()
