from __future__ import annotations

import json
import sys
from pathlib import Path

# Add the repository root to the python path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from multitenant.config import ExperimentConfig
from multitenant.experiments import evaluate_baseline_vs_proposed_mapping


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_vs_proposed_mapping.json"

    tenant_counts = [2, 4, 8, 16]
    num_exp = 10 # Reduced from 20 to 5 for speed
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
        config = ExperimentConfig(num_tenants=tenant_count, num_experiments=num_exp)
        comparison = evaluate_baseline_vs_proposed_mapping(config)
        payload["results_by_tenant_count"][str(tenant_count)] = comparison["results"]

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved baseline comparison to {output_path}")


if __name__ == "__main__":
    main()
