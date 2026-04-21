from __future__ import annotations

import json
from pathlib import Path

from multitenant.experiments import run_small_scale_proposed_mapping_validation


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "proposed_mapping_small_scale_validation.json"

    print("=== Proposed Mapping Small-Scale Validation ===")
    result = run_small_scale_proposed_mapping_validation()

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved small-scale validation to {output_path}")


if __name__ == "__main__":
    main()
