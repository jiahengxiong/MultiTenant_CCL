#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    simcore_cpp = importlib.util.find_spec("simcore_cpp") is not None
    te_accel_files = list((REPO_ROOT / "multitenant" / "solvers").glob("_te_accel*.so"))
    payload = {
        "simcore_cpp_available": simcore_cpp,
        "time_expanded_accelerator_files": [str(path) for path in te_accel_files],
        "large_experiment_compile_commands": [
            "python setup.py build_ext --inplace",
            "python debug/audits/build_te_accel.py",
        ],
        "note": (
            "The failure_aware_repair_project tests run without compiling these "
            "optional accelerators. Compile them before large-scale experiments."
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()

