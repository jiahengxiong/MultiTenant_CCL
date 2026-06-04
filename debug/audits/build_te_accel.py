from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "multitenant" / "solvers" / "_te_accel.cpp"
OUTPUT = (
    REPO_ROOT
    / "multitenant"
    / "solvers"
    / f"_te_accel{sysconfig.get_config_var('EXT_SUFFIX')}"
)


def main() -> None:
    include_flags = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"],
        text=True,
    ).strip().split()
    command = [
        sysconfig.get_config_var("CXX") or "c++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-undefined",
        "dynamic_lookup",
        *include_flags,
        str(SOURCE),
        "-o",
        str(OUTPUT),
    ]
    subprocess.check_call(command, cwd=str(REPO_ROOT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
