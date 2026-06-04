from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.simulator import simulate_collective  # noqa: E402
from multitenant.solvers import (  # noqa: E402
    MappingEstimatorBlackBoxOptimizer,
    MappingHybridHeuristicSolver,
)
from multitenant.topology import LeafSpineDatacenter  # noqa: E402


def denormalize_mapping(raw):
    return {
        int(tenant): {
            int(rank): int(server)
            for rank, server in rank_to_server.items()
        }
        for tenant, rank_to_server in raw.items()
    }


def denormalize_specs(raw):
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def build_datacenter(metadata):
    topology = metadata["topology"]
    return LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )


def simulator_score(datacenter, mapping, specs):
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return {
        "makespan": float(makespan),
        "avg_jct": float(avg_jct),
    }


def score_dict(score):
    makespan, avg_jct = score
    return {
        "makespan": float(makespan),
        "avg_jct": float(avg_jct),
    }


def best_hybrid_local_step(hybrid, mapping, score, deadline):
    _, _, epoch_prices = hybrid._compute_epoch_link_prices(mapping)
    tenant_order = sorted(
        hybrid.tenants,
        key=lambda tenant: (
            -hybrid.tenant_pressure.get(tenant, 0.0),
            -hybrid.tenant_peak_load.get(tenant, 0.0),
            tenant,
        ),
    )
    best = None
    tried = []
    for tenant in tenant_order:
        source, candidate, candidate_score = hybrid._best_neighborhood_candidate(
            mapping,
            score,
            tenant,
            epoch_prices,
            deadline,
            include_local=True,
            allow_bnb=False,
        )
        tried.append(
            {
                "tenant": int(tenant),
                "source": source,
                "score": score_dict(candidate_score) if source else None,
            }
        )
        if source is None:
            continue
        entry = (candidate_score, candidate, int(tenant), source)
        if best is None or hybrid._is_better_objective(candidate_score, best[0]):
            best = entry
    return best, tried


def inspect_time_expanded_local(optimizer, mapping, score, deadline):
    analysis = optimizer._analyze_mapping(mapping)
    rows = []
    for tenant in optimizer._tenant_order(analysis):
        source, candidate, candidate_score = optimizer._best_neighborhood_candidate(
            mapping,
            score,
            tenant,
            analysis,
            deadline,
            include_local=True,
            allow_bnb=False,
        )
        rows.append(
            {
                "tenant": int(tenant),
                "source": source,
                "score": score_dict(candidate_score) if source else None,
            }
        )
    return rows


def run_case(input_path: Path, tenant_count: int, trial_index: int, max_steps: int):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    group = next(
        item for item in payload["results"]
        if int(item["tenant_count"]) == int(tenant_count)
    )
    trial = group["trial_results"][int(trial_index)]

    datacenter = build_datacenter(metadata)
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    specs = denormalize_specs(trial["tenant_collective_specs"])
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)

    hybrid = MappingHybridHeuristicSolver(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )
    optimizer = MappingEstimatorBlackBoxOptimizer(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
        verbose=False,
    )

    mapping = initial_mapping
    hybrid_score = hybrid._evaluate_surrogate_mapping(mapping)
    trace = [
        {
            "step": 0,
            "source": "initial",
            "tenant": None,
            "hybrid_score": score_dict(hybrid_score),
            "time_expanded_score": score_dict(optimizer._evaluate_mapping(mapping)),
            "simulator_score": simulator_score(datacenter, mapping, specs),
            "time_expanded_local_candidates": inspect_time_expanded_local(
                optimizer,
                mapping,
                optimizer._evaluate_mapping(mapping),
                time.time() + 90,
            ),
        }
    ]

    for step in range(1, max_steps + 1):
        best, tried = best_hybrid_local_step(
            hybrid,
            mapping,
            hybrid_score,
            time.time() + 120,
        )
        if best is None:
            trace.append(
                {
                    "step": step,
                    "source": "hybrid-local-stop",
                    "hybrid_tried": tried,
                }
            )
            break

        hybrid_score, mapping, tenant, source = best
        te_score = optimizer._evaluate_mapping(mapping)
        trace.append(
            {
                "step": step,
                "source": source,
                "tenant": int(tenant),
                "hybrid_score": score_dict(hybrid_score),
                "time_expanded_score": score_dict(te_score),
                "simulator_score": simulator_score(datacenter, mapping, specs),
                "hybrid_tried": tried,
                "time_expanded_local_candidates": inspect_time_expanded_local(
                    optimizer,
                    mapping,
                    te_score,
                    time.time() + 90,
                ),
            }
        )

    return {
        "input": str(input_path),
        "tenant_count": int(tenant_count),
        "trial_index": int(trial_index),
        "trial_index_1_based": int(trial_index) + 1,
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "experiment" / "High_contension.json")
    parser.add_argument("--tenant-count", type=int, default=3)
    parser.add_argument("--trial", type=int, default=1, help="1-based trial index")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "debug" / "audits" / "time_expanded_bridge_trace.json")
    args = parser.parse_args()

    result = run_case(
        args.input,
        args.tenant_count,
        int(args.trial) - 1,
        args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
