from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_SCRIPTS = [
    "Low_contension.py",
    "High_contension.py",
    "Low_contension_homo.py",
    "High_contension_homo.py",
    "dominant vs full.py",
    "mapping_vs_ilp_single.py",
    "mapping_vs_ilp_multi.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all experiment scripts in this directory in a fixed order.",
    )
    parser.add_argument(
        "--scripts",
        nargs="*",
        default=DEFAULT_SCRIPTS,
        help="Optional subset of scripts to run. Defaults to all experiment scripts.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later scripts even if one script fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    requested = [Path(name) for name in args.scripts]
    missing = [name for name in requested if not (EXPERIMENT_DIR / name).exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing experiment script(s): {missing_text}")

    failures: list[tuple[str, int]] = []

    for script_name in requested:
        script_path = EXPERIMENT_DIR / script_name
        print(f"=== running {script_path.name} ===", flush=True)
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(EXPERIMENT_DIR),
            check=False,
        )
        elapsed = time.time() - start_time
        print(
            f"=== finished {script_path.name} exit_code={result.returncode} elapsed={elapsed:.2f}s ===",
            flush=True,
        )

        if result.returncode != 0:
            failures.append((script_path.name, int(result.returncode)))
            if not args.continue_on_error:
                break

    if failures:
        summary = ", ".join(f"{name}(exit={code})" for name, code in failures)
        raise SystemExit(f"Experiment runner finished with failures: {summary}")


if __name__ == "__main__":
    main()
