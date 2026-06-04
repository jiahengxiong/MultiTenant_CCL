from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import multitenant.solvers.mapping_time_expanded_optimizer as optimizer_mod  # noqa: E402
from multitenant.solvers import MappingEstimatorBlackBoxOptimizer  # noqa: E402
from multitenant.topology import LeafSpineDatacenter  # noqa: E402


def _parse_mapping(raw):
    return {
        int(tenant): {int(rank): int(server) for rank, server in ranks.items()}
        for tenant, ranks in raw.items()
    }


def _parse_specs(raw):
    return {int(tenant): dict(spec) for tenant, spec in raw.items()}


def _build_datacenter(metadata):
    topology = metadata["topology"]
    return LeafSpineDatacenter(
        num_leaf=int(topology["num_leaf"]),
        num_spine=int(topology["num_spine"]),
        per_leaf_server=int(topology["per_leaf_server"]),
    )


def _load_case(path: Path, tenant_count: int, trial_1_based: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for group in payload["results"]:
        if int(group["tenant_count"]) != int(tenant_count):
            continue
        for trial in group["trial_results"]:
            if int(trial["trial_index"]) == int(trial_1_based) - 1:
                return payload["metadata"], trial
    raise ValueError(f"case not found: {path} tenant={tenant_count} trial={trial_1_based}")


def _make_solver(metadata, trial):
    datacenter = _build_datacenter(metadata)
    initial_mapping = _parse_mapping(trial["initial_mapping"])
    specs = _parse_specs(trial["tenant_collective_specs"])
    return MappingEstimatorBlackBoxOptimizer(
        datacenter,
        tenant_mapping=initial_mapping,
        tenant_collective_specs=specs,
        path_table=datacenter.build_tenant_ecmp_path_table(initial_mapping),
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tenant-count", type=int, required=True)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--tenant", type=int)
    parser.add_argument("--time-budget", type=float, default=30.0)
    args = parser.parse_args()

    metadata, trial = _load_case(args.input, args.tenant_count, args.trial)
    solver = _make_solver(metadata, trial)
    base_mapping = solver._seed_mappings()[0]
    base_score = solver._evaluate_mapping(base_mapping)
    analysis = solver._analyze_mapping(base_mapping)
    tenant_order = solver._tenant_order(analysis)
    tenant = int(args.tenant if args.tenant is not None else tenant_order[0])

    original_accel = optimizer_mod._te_accel
    deadline = time.time() + float(args.time_budget)
    c_source, c_mapping, c_score = solver._price_guided_remap_pool(
        base_mapping,
        base_score,
        tenant,
        analysis,
        deadline,
    )

    optimizer_mod._te_accel = None
    try:
        deadline = time.time() + float(args.time_budget)
        p_source, p_mapping, p_score = solver._price_guided_remap_pool(
            base_mapping,
            base_score,
            tenant,
            analysis,
            deadline,
        )
    finally:
        optimizer_mod._te_accel = original_accel

    result = {
        "tenant": tenant,
        "same_source": c_source == p_source,
        "same_mapping": c_mapping == p_mapping,
        "same_score": tuple(c_score) == tuple(p_score),
        "cpp": {"source": c_source, "score": tuple(float(v) for v in c_score)},
        "python": {"source": p_source, "score": tuple(float(v) for v in p_score)},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
