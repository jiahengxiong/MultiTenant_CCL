from __future__ import annotations

from multitenant.baselines import HarmonicsBaselineHeuristic, HarmonicsBaselineILP
from multitenant.topology import LeafSpineDatacenter


def main():
    datacenter = LeafSpineDatacenter(num_leaf=3, num_spine=2, per_leaf_server=2)

    tenant_mapping = {
        0: {0: 0, 1: 1},
        1: {0: 2, 1: 4},
        2: {0: 3, 1: 5},
    }
    tenant_flows = {
        0: [(0, 1, 0.1)],
        1: [(0, 1, 0.5)],
        2: [(0, 1, 0.5)],
    }

    print("=== Harmonics Baseline Debug ===")
    heuristic = HarmonicsBaselineHeuristic(
        datacenter,
        tenant_mapping,
        tenant_flows,
        datacenter.paths,
        single_flow_size=8 * 1024 * 1024,
        collective="allreduce",
        verbose=True,
    )
    heuristic_schedule = heuristic.solve()
    print("Harmonics baseline heuristic schedule:", heuristic_schedule)

    ilp = HarmonicsBaselineILP(
        datacenter,
        tenant_mapping,
        tenant_flows,
        datacenter.paths,
        single_flow_size=8 * 1024 * 1024,
        collective="allreduce",
        verbose=True,
        estimation=0.1,
    )
    ilp_schedule = ilp.solve()
    print("Harmonics baseline ILP schedule:", ilp_schedule)


if __name__ == "__main__":
    main()
