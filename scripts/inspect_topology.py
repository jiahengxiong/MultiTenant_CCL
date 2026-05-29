from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from multitenant.diagnostics import sample_link_capacities
from multitenant.topology import LeafSpineDatacenter


def main():
    datacenter = LeafSpineDatacenter(num_leaf=3, num_spine=2, per_leaf_server=4)

    print("=== Leaf-Spine Topology Inspection ===")
    print(
        "Topology:",
        {
            "num_leaf": datacenter.num_leaf,
            "num_spine": datacenter.num_spine,
            "num_server": datacenter.num_server,
        },
    )

    print("\nSample link capacities:")
    for edge, capacity in sample_link_capacities(datacenter, limit=5):
        print(f"  {edge}: {capacity:.1f} bps")

    example_src, example_dst = 0, min(5, datacenter.num_server - 1)
    path_table = datacenter.build_tenant_ecmp_path_table({0: 0, 1: 1})
    print("\nSample tenant-aware ECMP-style paths:")
    for tenant in (0, 1):
        print(f"  tenant {tenant}, {example_src} -> {example_dst}: {path_table[(tenant, example_src, example_dst)]}")


if __name__ == "__main__":
    main()
