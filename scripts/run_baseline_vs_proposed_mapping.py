from __future__ import annotations

import json
from pathlib import Path

from multitenant.config import ExperimentConfig
from multitenant.experiments import evaluate_baseline_vs_proposed_mapping


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_vs_proposed_mapping.json"

    tenant_counts = [2, 3, 4]
    payload = {
        "metadata": {
            "comparison": "baseline_vs_proposed_mapping",
            "tenant_counts": tenant_counts,
        },
        "results_by_tenant_count": {},
    }

    print("=== Baseline vs Proposed Mapping ===")
    for tenant_count in tenant_counts:
        print(f"Running comparison for {tenant_count} tenants...")
        config = ExperimentConfig(num_tenants=tenant_count, num_experiments=20)
        comparison = evaluate_baseline_vs_proposed_mapping(config)
        payload["results_by_tenant_count"][str(tenant_count)] = comparison["results"]

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved baseline comparison to {output_path}")


if __name__ == "__main__":
    main()
