from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

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
    path_table = datacenter.build_tenant_ecmp_path_table(tenant_mapping)

    print("=== Harmonics Baseline Debug ===")
    heuristic = HarmonicsBaselineHeuristic(
        datacenter,
        tenant_mapping,
        tenant_flows,
        path_table,
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
        path_table,
        single_flow_size=8 * 1024 * 1024,
        collective="allreduce",
        verbose=True,
        estimation=0.1,
    )
    ilp_schedule = ilp.solve()
    print("Harmonics baseline ILP schedule:", ilp_schedule)


if __name__ == "__main__":
    main()
