from __future__ import annotations

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from multitenant.baselines import HarmonicsBaselineHeuristic, LeafLocalBaseline
from multitenant.simulator.adapter import simulate_collective_details
from multitenant.solvers import MappingHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


BITS_PER_MB = 8 * 1024 * 1024


def _mapping() -> dict[int, dict[int, int]]:
    # Three tenants fully share the same four servers across two leaves.
    return {
        0: {0: 0, 1: 4, 2: 1, 3: 5},
        1: {0: 0, 1: 4, 2: 1, 3: 5},
        2: {0: 0, 1: 4, 2: 1, 3: 5},
    }


def _program() -> dict[int, list[dict[str, object]]]:
    op = {
        "collective": "allgather",
        "single_flow_size_bits": int(8 * BITS_PER_MB),
        "gap_after": 0.0,
    }
    return {0: [dict(op)], 1: [dict(op)], 2: [dict(op)]}


def _pct_improvement(baseline: float, value: float) -> float:
    if baseline == 0.0:
        return 0.0
    return (baseline - value) / baseline * 100.0


def _run_six_way():
    datacenter = LeafSpineDatacenter(num_leaf=3, num_spine=2, per_leaf_server=4)
    tenant_mapping = _mapping()
    program = _program()
    default_path_table = datacenter.build_tenant_ecmp_path_table(tenant_mapping)

    t0 = time.time()
    default = simulate_collective_details(
        datacenter.topology,
        tenant_mapping,
        default_path_table,
        tenant_collective_programs=program,
        policy_qpid_mode="tenant",
    )
    default_wall = time.time() - t0

    t0 = time.time()
    harmonics = HarmonicsBaselineHeuristic(
        datacenter,
        tenant_mapping,
        tenant_flows=None,
        collective="allgather",
        single_flow_size=0,
        tenant_collective_programs=program,
        verbose=False,
    )
    harmonics.solve()
    default_h = simulate_collective_details(
        datacenter.topology,
        tenant_mapping,
        default_path_table,
        tenant_collective_programs=program,
        collective_start_times=harmonics.get_collective_start_times(),
        policy_qpid_mode="tenant",
    )
    default_h_wall = time.time() - t0

    t0 = time.time()
    mapping_solver = MappingHeuristicSolver(
        datacenter,
        tenant_mapping,
        tenant_flows=None,
        collective="allgather",
        single_flow_size=0,
        tenant_collective_programs=program,
        verbose=False,
        validate_with_simulator=False,
        path_table=default_path_table,
    )
    mapping_solver.solve()
    mapped = mapping_solver.get_X_mapping()
    mapped_path_table = datacenter.build_tenant_ecmp_path_table(mapped)
    mapping = simulate_collective_details(
        datacenter.topology,
        mapped,
        mapped_path_table,
        tenant_collective_programs=program,
        policy_qpid_mode="tenant",
    )
    mapping_wall = time.time() - t0

    t0 = time.time()
    mapped_h_solver = HarmonicsBaselineHeuristic(
        datacenter,
        mapped,
        tenant_flows=None,
        collective="allgather",
        single_flow_size=0,
        tenant_collective_programs=program,
        verbose=False,
        path_table=mapped_path_table,
    )
    mapped_h_solver.solve()
    mapping_h = simulate_collective_details(
        datacenter.topology,
        mapped,
        mapped_path_table,
        tenant_collective_programs=program,
        collective_start_times=mapped_h_solver.get_collective_start_times(),
        policy_qpid_mode="tenant",
    )
    mapping_h_wall = time.time() - t0

    t0 = time.time()
    locality_solver = LeafLocalBaseline(tenant_mapping)
    local = locality_solver.solve()
    local_path_table = datacenter.build_tenant_ecmp_path_table(local)
    locality = simulate_collective_details(
        datacenter.topology,
        local,
        local_path_table,
        tenant_collective_programs=program,
        policy_qpid_mode="tenant",
    )
    locality_wall = time.time() - t0

    t0 = time.time()
    local_h_solver = HarmonicsBaselineHeuristic(
        datacenter,
        local,
        tenant_flows=None,
        collective="allgather",
        single_flow_size=0,
        tenant_collective_programs=program,
        verbose=False,
        path_table=local_path_table,
    )
    local_h_solver.solve()
    locality_h = simulate_collective_details(
        datacenter.topology,
        local,
        local_path_table,
        tenant_collective_programs=program,
        collective_start_times=local_h_solver.get_collective_start_times(),
        policy_qpid_mode="tenant",
    )
    locality_h_wall = time.time() - t0

    print("3T-common-bottleneck-4rank-single")
    print(f"default              avg={default['avg_tenant_makespan']:.12f} mk={default['global_makespan']:.12f} wall={default_wall:.4f}s")
    print(
        f"locality             avg={locality['avg_tenant_makespan']:.12f} mk={locality['global_makespan']:.12f} "
        f"wall={locality_wall:.4f}s avg_gain={_pct_improvement(default['avg_tenant_makespan'], locality['avg_tenant_makespan']):.2f}% "
        f"mapping={local}"
    )
    print(
        f"default+harmonics    avg={default_h['avg_tenant_makespan']:.12f} mk={default_h['global_makespan']:.12f} "
        f"wall={default_h_wall:.4f}s avg_gain={_pct_improvement(default['avg_tenant_makespan'], default_h['avg_tenant_makespan']):.2f}% "
        f"starts={harmonics.get_collective_start_times()}"
    )
    print(
        f"locality+harmonics   avg={locality_h['avg_tenant_makespan']:.12f} mk={locality_h['global_makespan']:.12f} "
        f"wall={locality_h_wall:.4f}s gain_over_locality={_pct_improvement(locality['avg_tenant_makespan'], locality_h['avg_tenant_makespan']):.2f}% "
        f"starts={local_h_solver.get_collective_start_times()}"
    )
    print(
        f"mapping              avg={mapping['avg_tenant_makespan']:.12f} mk={mapping['global_makespan']:.12f} "
        f"wall={mapping_wall:.4f}s avg_gain={_pct_improvement(default['avg_tenant_makespan'], mapping['avg_tenant_makespan']):.2f}% "
        f"mapping={mapped}"
    )
    print(
        f"mapping+harmonics    avg={mapping_h['avg_tenant_makespan']:.12f} mk={mapping_h['global_makespan']:.12f} "
        f"wall={mapping_h_wall:.4f}s gain_over_mapping={_pct_improvement(mapping['avg_tenant_makespan'], mapping_h['avg_tenant_makespan']):.2f}% "
        f"starts={mapped_h_solver.get_collective_start_times()}"
    )


if __name__ == "__main__":
    _run_six_way()
