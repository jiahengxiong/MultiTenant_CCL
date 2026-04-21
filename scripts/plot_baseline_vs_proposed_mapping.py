from __future__ import annotations

import json
from pathlib import Path

from multitenant.plotting import save_baseline_vs_proposed_mapping_figures


def main():
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "results" / "baseline_vs_proposed_mapping.json"
    output_dir = repo_root / "results" / "figures"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run scripts/run_baseline_vs_proposed_mapping.py first."
        )

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    written_files = save_baseline_vs_proposed_mapping_figures(payload, output_dir)
    for written_file in written_files:
        print(f"Saved figure to {written_file}")


if __name__ == "__main__":
    main()
