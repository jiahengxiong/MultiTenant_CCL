from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from debug.audits.compare_live_hybrid_vs_time_expanded import (  # noqa: E402
    build_datacenter,
    denormalize_mapping,
    denormalize_specs,
    solver_path_table,
)
from multitenant.simulator import simulate_collective  # noqa: E402
from multitenant.solvers.contention_estimator import TimeExpandedContentionEstimator  # noqa: E402


def simulator_score(datacenter, mapping, specs):
    path_table = datacenter.build_tenant_ecmp_path_table(mapping)
    makespan, avg_jct = simulate_collective(
        datacenter.topology,
        mapping,
        path_table,
        tenant_collective_specs=specs,
    )
    return {"avg_jct": float(avg_jct), "makespan": float(makespan)}


def estimator_score(estimator, mapping):
    estimate = estimator.evaluate(mapping)
    pipeline = estimator.pipeline_score(mapping)
    return {
        "primary": {"makespan": float(estimate.makespan), "avg_jct": float(estimate.avg_jct)},
        "pipeline": {"makespan": float(pipeline[0]), "avg_jct": float(pipeline[1])},
    }


def lex_key(score):
    return (float(score["primary"]["avg_jct"]), float(score["primary"]["makespan"]))


def mapping_diff(left, right):
    return {
        str(tenant): sum(
            1 for rank, server in left[int(tenant)].items()
            if int(right[int(tenant)][int(rank)]) != int(server)
        )
        for tenant in sorted(left)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why time-expanded search missed a hybrid-preferred mapping.")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "debug" / "audits" / "time_expanded_search_miss_diagnosis.json")
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    failures = [row for row in comparison["results"] if not row.get("passed_avg_jct_sensitivity", True)]
    if not failures:
        raise SystemExit("comparison has no failed cases")
    row = failures[int(args.case_index)]

    payload = json.loads(Path(row["input"]).read_text(encoding="utf-8"))
    trial = next(g for g in payload["results"] if int(g["tenant_count"]) == int(row["tenant_count"]))["trial_results"][int(row["trial_index"])]
    datacenter = build_datacenter(payload["metadata"])
    initial_mapping = denormalize_mapping(trial["initial_mapping"])
    specs = denormalize_specs(trial["tenant_collective_specs"])
    estimator = TimeExpandedContentionEstimator(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=solver_path_table(datacenter, initial_mapping),
        name="search_miss_diagnosis_estimator",
    )

    hybrid = denormalize_mapping(row["mappings"]["hybrid_live"])
    te = denormalize_mapping(row["mappings"]["time_expanded_live"])
    base_est = estimator_score(estimator, te)
    hybrid_est = estimator_score(estimator, hybrid)
    te_analysis = estimator.analyze_for_search(te)
    tenant_order = sorted(
        estimator.tenants,
        key=lambda tenant: (
            -float(te_analysis.tenant_pressure.get(int(tenant), 0.0)),
            -float(te_analysis.tenant_peak_load.get(int(tenant), 0.0)),
            int(tenant),
        ),
    )
    tenant_order_pos = {int(tenant): idx for idx, tenant in enumerate(tenant_order)}

    single_replacements: list[dict[str, Any]] = []
    for tenant in sorted(te):
        candidate = {int(t): dict(r2s) for t, r2s in te.items()}
        candidate[int(tenant)] = dict(hybrid[int(tenant)])
        score = estimator_score(estimator, candidate)
        single_replacements.append({
            "tenant": int(tenant),
            "rank_diff_to_hybrid": mapping_diff(te, hybrid)[str(tenant)],
            "tenant_pressure": float(te_analysis.tenant_pressure.get(int(tenant), 0.0)),
            "tenant_peak_load": float(te_analysis.tenant_peak_load.get(int(tenant), 0.0)),
            "tenant_order_position": int(tenant_order_pos[int(tenant)]),
            "estimator_score": score,
            "improves_time_expanded_final": bool(lex_key(score) < lex_key(base_est)),
            "reaches_or_beats_hybrid_estimator": bool(lex_key(score) <= lex_key(hybrid_est)),
        })

    pair_replacements: list[dict[str, Any]] = []
    tenants = sorted(te)
    for idx, tenant_a in enumerate(tenants):
        for tenant_b in tenants[idx + 1:]:
            candidate = {int(t): dict(r2s) for t, r2s in te.items()}
            candidate[int(tenant_a)] = dict(hybrid[int(tenant_a)])
            candidate[int(tenant_b)] = dict(hybrid[int(tenant_b)])
            score = estimator_score(estimator, candidate)
            if lex_key(score) < lex_key(base_est):
                pair_replacements.append({
                    "tenants": [int(tenant_a), int(tenant_b)],
                    "tenant_order_positions": [int(tenant_order_pos[int(tenant_a)]), int(tenant_order_pos[int(tenant_b)])],
                    "estimator_score": score,
                    "reaches_or_beats_hybrid_estimator": bool(lex_key(score) <= lex_key(hybrid_est)),
                })
    pair_replacements.sort(key=lambda item: (item["estimator_score"]["primary"]["avg_jct"], item["estimator_score"]["primary"]["makespan"]))

    result = {
        "case": {
            "input": row["input"],
            "scene": row["scene"],
            "tenant_count": row["tenant_count"],
            "trial_index": row["trial_index"],
        },
        "simulator_scores": row["scores"],
        "estimator_scores": {
            "time_expanded_final": base_est,
            "hybrid_final": hybrid_est,
        },
        "mapping_diff_time_expanded_vs_hybrid": mapping_diff(te, hybrid),
        "tenant_order_under_time_expanded_final": [int(t) for t in tenant_order],
        "single_tenant_hybrid_replacements": sorted(
            single_replacements,
            key=lambda item: (item["estimator_score"]["primary"]["avg_jct"], item["estimator_score"]["primary"]["makespan"]),
        ),
        "improving_pair_hybrid_replacements_top10": pair_replacements[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
