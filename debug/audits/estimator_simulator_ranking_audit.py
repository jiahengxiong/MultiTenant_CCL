from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multitenant.simulator import simulate_collective
from multitenant.solvers import TimeExpandedContentionEstimator
from multitenant.topology import LeafSpineDatacenter


Mapping = dict[int, dict[int, int]]


def _int_mapping(raw: dict[str, dict[str, int]] | dict[int, dict[int, int]]) -> Mapping:
    return {
        int(tenant): {int(rank): int(server) for rank, server in ranks.items()}
        for tenant, ranks in raw.items()
    }


def _int_specs(raw: dict[str, dict[str, object]] | dict[int, dict[str, object]]):
    return {
        int(tenant): dict(spec)
        for tenant, spec in raw.items()
    }


def _copy_mapping(mapping: Mapping) -> Mapping:
    return {
        int(tenant): {int(rank): int(server) for rank, server in ranks.items()}
        for tenant, ranks in mapping.items()
    }


def _signature(mapping: Mapping):
    return tuple(
        (tenant, tuple((rank, mapping[tenant][rank]) for rank in sorted(mapping[tenant])))
        for tenant in sorted(mapping)
    )


def _swap_candidate(mapping: Mapping, tenant: int, left_rank: int, right_rank: int) -> Mapping:
    candidate = _copy_mapping(mapping)
    candidate[tenant][left_rank], candidate[tenant][right_rank] = (
        candidate[tenant][right_rank],
        candidate[tenant][left_rank],
    )
    return candidate


def build_candidates(case: dict[str, object], *, max_swaps: int, seed: int):
    base_candidates: list[tuple[str, Mapping]] = [
        ("default", _int_mapping(case["initial_mapping"])),
        ("locality", _int_mapping(case["locality_mapping"])),
        ("json_mapping", _int_mapping(case["proposed_mapping"])),
    ]
    rng = random.Random(seed)
    seen = {_signature(mapping) for _name, mapping in base_candidates}
    candidates = list(base_candidates)

    anchors = [
        ("default", base_candidates[0][1]),
        ("locality", base_candidates[1][1]),
        ("json_mapping", base_candidates[2][1]),
    ]
    attempts = 0
    while len(candidates) < len(base_candidates) + max_swaps and attempts < max_swaps * 20 + 20:
        attempts += 1
        anchor_name, anchor_mapping = rng.choice(anchors)
        tenant = rng.choice(sorted(anchor_mapping))
        ranks = sorted(anchor_mapping[tenant])
        if len(ranks) < 2:
            continue
        left_rank, right_rank = rng.sample(ranks, 2)
        candidate = _swap_candidate(anchor_mapping, tenant, left_rank, right_rank)
        signature = _signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append((
            f"{anchor_name}_swap_t{tenant}_r{left_rank}_r{right_rank}",
            candidate,
        ))
    return candidates


def score_sign(left: tuple[float, float], right: tuple[float, float], *, tol: float):
    left_avg, left_mk = left
    right_avg, right_mk = right
    if left_avg < right_avg - tol:
        return -1
    if left_avg > right_avg + tol:
        return 1
    if left_mk < right_mk - tol:
        return -1
    if left_mk > right_mk + tol:
        return 1
    return 0


def pairwise_agreement(rows: list[dict[str, object]], *, tol: float):
    comparable = 0
    agreements = 0
    disagreements = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            est_i = rows[i]["estimator"]
            est_j = rows[j]["estimator"]
            sim_i = rows[i]["simulator"]
            sim_j = rows[j]["simulator"]
            est_sign = score_sign(
                (float(est_i["avg_jct"]), float(est_i["makespan"])),
                (float(est_j["avg_jct"]), float(est_j["makespan"])),
                tol=tol,
            )
            sim_sign = score_sign(
                (float(sim_i["avg_jct"]), float(sim_i["makespan"])),
                (float(sim_j["avg_jct"]), float(sim_j["makespan"])),
                tol=tol,
            )
            if est_sign == 0 or sim_sign == 0:
                continue
            comparable += 1
            if est_sign == sim_sign:
                agreements += 1
            else:
                disagreements.append({
                    "left": rows[i]["name"],
                    "right": rows[j]["name"],
                    "estimator_order": est_sign,
                    "simulator_order": sim_sign,
                })
    return {
        "comparable_pairs": comparable,
        "agreements": agreements,
        "agreement_ratio": (agreements / comparable) if comparable else None,
        "disagreements": disagreements,
    }


def audit_case(payload: dict[str, object], case: dict[str, object], *, max_swaps: int, seed: int, tol: float):
    topology = payload["metadata"]["topology"]
    datacenter = LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )
    tenant_collective_specs = _int_specs(case["tenant_collective_specs"])
    initial_mapping = _int_mapping(case["initial_mapping"])
    path_table = datacenter.build_tenant_ecmp_path_table(initial_mapping)
    estimator = TimeExpandedContentionEstimator(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=tenant_collective_specs,
        path_table=path_table,
    )

    rows = []
    for name, mapping in build_candidates(case, max_swaps=max_swaps, seed=seed):
        estimate = estimator.evaluate(mapping)
        sim_makespan, sim_avg = simulate_collective(
            datacenter.topology,
            mapping,
            path_table,
            tenant_collective_specs=tenant_collective_specs,
        )
        rows.append({
            "name": name,
            "estimator": {
                "avg_jct": float(estimate.avg_jct),
                "makespan": float(estimate.makespan),
            },
            "simulator": {
                "avg_jct": float(sim_avg),
                "makespan": float(sim_makespan),
            },
        })

    return {
        "tenant_count": int(case["tenant_count"]),
        "trial_index": int(case["trial_index"]),
        "candidate_count": len(rows),
        "rows": rows,
        "pairwise": pairwise_agreement(rows, tol=tol),
    }


def selected_cases(payload: dict[str, object], tenant_counts: set[int], trials: set[int]):
    for result in payload["results"]:
        tenant_count = int(result["tenant_count"])
        if tenant_counts and tenant_count not in tenant_counts:
            continue
        for case in result["trial_results"]:
            trial_number = int(case["trial_index"]) + 1
            if trials and trial_number not in trials:
                continue
            yield case


def parse_csv_ints(raw: str) -> set[int]:
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare time-expanded estimator and simulator relative ranking over candidate mappings.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tenant-counts", type=str, default="")
    parser.add_argument("--trials", type=str, default="1")
    parser.add_argument("--max-swaps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance in seconds for treating two mappings as tied.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    tenant_counts = parse_csv_ints(args.tenant_counts)
    trials = parse_csv_ints(args.trials)

    reports = []
    for case in selected_cases(payload, tenant_counts, trials):
        print(
            f"[ranking-audit] {args.input.name} tenant_count={case['tenant_count']} "
            f"trial={int(case['trial_index']) + 1}",
            flush=True,
        )
        reports.append(
            audit_case(
                payload,
                case,
                max_swaps=max(0, int(args.max_swaps)),
                seed=int(args.seed) + int(case["tenant_count"]) * 1000 + int(case["trial_index"]),
                tol=float(args.tol),
            )
        )

    output = {
        "input": str(args.input),
        "tenant_counts": sorted(tenant_counts),
        "trials": sorted(trials),
        "max_swaps": int(args.max_swaps),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved ranking audit to {args.output}", flush=True)


if __name__ == "__main__":
    main()
