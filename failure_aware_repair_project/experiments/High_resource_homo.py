#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SOURCE = Path(__file__).resolve().with_name("High_resource-homo.py")
_SPEC = importlib.util.spec_from_file_location("_high_resource_homo_impl", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load experiment script: {_SOURCE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

for _name, _value in vars(_MODULE).items():
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = _value


if __name__ == "__main__":
    main()
