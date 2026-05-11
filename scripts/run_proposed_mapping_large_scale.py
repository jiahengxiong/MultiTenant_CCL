from __future__ import annotations

import json
from pathlib import Path

from multitenant.experiments import run_large_scale_proposed_mapping


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "proposed_mapping_large_scale.json"

    print("=== Proposed Mapping Large-Scale Study ===")
    result = run_large_scale_proposed_mapping()

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved large-scale mapping study to {output_path}")


if __name__ == "__main__":
    main()
