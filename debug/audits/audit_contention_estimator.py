from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from solve_case_once import denormalize_mapping, denormalize_specs, load_trial, normalize_mapping


def perturb_mapping(mapping, rng: random.Random, swaps: int):
    candidate = {
        int(tenant): {int(rank): int(server) for rank, server in rank_to_server.items()}
        for tenant, rank_to_server in mapping.items()
    }
    tenants = sorted(candidate)
    for _ in range(max(0, int(swaps))):
        tenant = rng.choice(tenants)
        ranks = sorted(candidate[tenant])
        if len(ranks) < 2:
            continue
        left, right = rng.sample(ranks, 2)
        candidate[tenant][left], candidate[tenant][right] = (
            candidate[tenant][right],
            candidate[tenant][left],
        )
    return candidate


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return float(num / (den_x * den_y))


def rank_order(values):
    return {
        idx: rank
        for rank, (idx, _value) in enumerate(sorted(enumerate(values), key=lambda item: item[1]))
    }


def spearman(xs, ys):
    rx = rank_order(xs)
    ry = rank_order(ys)
    return pearson([rx[idx] for idx in range(len(xs))], [ry[idx] for idx in range(len(ys))])


def main():
    parser = argparse.ArgumentParser(description="Audit time-expanded contention estimator semantics and ranking.")
    parser.add_argument("--input", type=Path, default=Path("experiment/Low_contension.json"))
    parser.add_argument("--tenant-count", type=int, default=3)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--swaps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("debug/audits/contention_estimator_audit.json"))
    args = parser.parse_args()

    from multitenant.simulator import simulate_collective
    from multitenant.solvers import TimeExpandedContentionEstimator
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
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    estimator = TimeExpandedContentionEstimator(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=path_table,
    )

    rng = random.Random(args.seed)
    rows = []
    for idx in range(max(1, int(args.samples))):
        mapping = initial_mapping if idx == 0 else perturb_mapping(initial_mapping, rng, args.swaps)
        estimate = estimator.evaluate(mapping)
        makespan, avg_jct = simulate_collective(
            datacenter.topology,
            mapping,
            datacenter.build_tenant_ecmp_path_table(mapping),
            tenant_collective_specs=specs,
        )
        rows.append(
            {
                "sample": idx,
                "estimator_avg_jct": estimate.avg_jct,
                "estimator_makespan": estimate.makespan,
                "simulator_avg_jct": float(avg_jct),
                "simulator_makespan": float(makespan),
                "mapping": normalize_mapping(mapping),
            }
        )

    est_avg = [row["estimator_avg_jct"] for row in rows]
    sim_avg = [row["simulator_avg_jct"] for row in rows]
    est_mk = [row["estimator_makespan"] for row in rows]
    sim_mk = [row["simulator_makespan"] for row in rows]
    analysis = estimator.analyze(initial_mapping)
    payload = {
        "input": str(args.input),
        "tenant_count": int(args.tenant_count),
        "trial_index": int(args.trial_index),
        "samples": len(rows),
        "release_gates": {
            str(tenant): {
                str(task): {"previous_tasks": list(prev), "gap": gap}
                for task, (prev, gap) in gates.items()
            }
            for tenant, gates in estimator.audit_release_gates().items()
        },
        "initial_hotspot_count": len(analysis.hotspots),
        "initial_contention_cluster_count": len(analysis.contention_clusters),
        "pearson_avg_jct": pearson(est_avg, sim_avg),
        "spearman_avg_jct": spearman(est_avg, sim_avg),
        "pearson_makespan": pearson(est_mk, sim_mk),
        "spearman_makespan": spearman(est_mk, sim_mk),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
