from __future__ import annotations

import subprocess
import sys
import sysconfig
import shlex
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
    cxx = sysconfig.get_config_var("CXX") or "c++"
    linker_flags = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
    command = [
        *shlex.split(cxx),
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-fPIC",
        *linker_flags,
        *include_flags,
        str(SOURCE),
        "-o",
        str(OUTPUT),
    ]
    subprocess.check_call(command, cwd=str(REPO_ROOT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
