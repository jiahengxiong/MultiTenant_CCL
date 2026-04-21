from __future__ import annotations

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
    print("\nSample ECMP-style path:")
    print(f"  {example_src} -> {example_dst}: {datacenter.paths[(example_src, example_dst)]}")


if __name__ == "__main__":
    main()
