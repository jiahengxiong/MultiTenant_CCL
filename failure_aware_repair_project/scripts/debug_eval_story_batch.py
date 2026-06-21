from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from failure_aware_repair_project.experiments import experiment_result_exporter as exporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--tenant", type=int, default=3)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = exporter._load_module(args.script)
    root = json.loads(exporter._trials_path_for_module(module).read_text())
    cases = exporter._published_trials_as_story_cases(root)[str(int(args.tenant))][
        : max(1, int(args.limit))
    ]
    results = exporter._evaluate_case_batch(cases, workers=max(1, int(args.workers)))
    rows = []
    for result in results:
        failover = float(result["methods"]["failover"]["avg_jct"])
        local = float(result["methods"]["local"]["avg_jct"])
        coordinate = float(result["methods"]["coordinate"]["avg_jct"])
        rows.append(
            {
                "failure": result["failure"],
                "failover": failover,
                "local": local,
                "coordinate": coordinate,
                "local_gain_pct": (
                    (failover - local) / failover * 100.0 if failover else 0.0
                ),
                "coordinate_extra_pct": (
                    (local - coordinate) / failover * 100.0 if failover else 0.0
                ),
                "local_source": result["strategy_metadata"]["local"].get("best_source"),
                "coordinate_source": result["strategy_metadata"]["coordinate"].get(
                    "best_source"
                ),
            }
        )
    rows.sort(key=lambda row: (-row["local_gain_pct"], -row["coordinate_extra_pct"]))
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
