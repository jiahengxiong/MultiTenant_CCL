from __future__ import annotations

from multitenant.config import BITS_PER_MB
from multitenant.diagnostics import resolve_flow_paths, summarize_link_occupancy
from multitenant.solvers import MappingILPSolver
from multitenant.topology import LeafSpineDatacenter
from multitenant.workloads import build_ring_flows


def main():
    datacenter = LeafSpineDatacenter(num_leaf=3, num_spine=2, per_leaf_server=2)
    initial_mapping = {
        0: {0: 0, 1: 2, 2: 4},
        1: {0: 1, 1: 3, 2: 5},
    }
    tenant_flows = build_ring_flows(
        initial_mapping,
        single_flow_size_bits=8 * BITS_PER_MB,
        collective="allgather",
    )

    print("=== Proposed Mapping ILP Debug ===")
    solver = MappingILPSolver(
        datacenter,
        initial_mapping,
        tenant_flows,
        verbose=True,
        collective="allgather",
        single_flow_size=8 * BITS_PER_MB,
    )
    solver.solve()
    solution = solver.get_X_mapping()

    print("\nResolved proposed mapping:")
    print(solution)

    print("\nResolved flow paths:")
    for flow in resolve_flow_paths(datacenter, solution, tenant_flows):
        print(
            f"Tenant {flow['tenant']} flow {flow['flow_index']}: "
            f"{flow['physical_src']} -> {flow['physical_dst']} via {flow['path']} "
            f"(volume={flow['volume_gbit']:.3f} Gbit)"
        )

    print("\nTop link occupancies if the proposed mapping runs alone:")
    for summary in summarize_link_occupancy(datacenter, solution, tenant_flows)[:10]:
        print(
            f"Link {summary['edge']}: "
            f"occupancy={summary['isolated_occupancy_seconds']:.6f}s, "
            f"capacity={summary['capacity_gbps']:.2f}Gbps, "
            f"volume={summary['total_volume_gbit']:.3f}Gbit"
        )


if __name__ == "__main__":
    main()
