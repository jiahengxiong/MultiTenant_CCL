from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from multitenant.baselines import HarmonicsBaselineHeuristic
from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


BITS_PER_MB = 8 * 1024 * 1024
TOPOLOGY = (3, 2, 4)  # 3 leaves, 2 spines, 4 servers per leaf.


def _extract_start_times(schedule):
    if not schedule:
        return None
    return {tenant: schedule[tenant][0] for tenant in schedule}


def _mapping_from_servers(tenant_servers: dict[int, list[int]]) -> dict[int, dict[int, int]]:
    return {
        int(tenant): {rank: int(server) for rank, server in enumerate(servers)}
        for tenant, servers in tenant_servers.items()
    }


def low_contention_2tenant_6rank() -> dict[int, dict[int, int]]:
    """Each physical server is used by at most one tenant."""
    return _mapping_from_servers(
        {
            0: [0, 1, 2, 3, 4, 5],
            1: [6, 7, 8, 9, 10, 11],
        }
    )


def high_contention_2tenant_6rank() -> dict[int, dict[int, int]]:
    """Both tenants share the same six servers; each shared server has two tenants."""
    return _mapping_from_servers(
        {
            0: [0, 1, 4, 5, 8, 9],
            1: [0, 1, 4, 5, 8, 9],
        }
    )


def low_contention_3tenant_4rank() -> dict[int, dict[int, int]]:
    """Three tenants occupy disjoint server sets."""
    return _mapping_from_servers(
        {
            0: [0, 1, 4, 5],
            1: [2, 3, 8, 9],
            2: [6, 7, 10, 11],
        }
    )


def high_contention_3tenant_4rank() -> dict[int, dict[int, int]]:
    """Pairwise server sharing: each hot server is used by exactly two tenants."""
    return _mapping_from_servers(
        {
            0: [0, 1, 4, 5],
            1: [0, 1, 8, 9],
            2: [4, 5, 8, 9],
        }
    )


def _program(specs: list[tuple[str, int, float]]) -> list[dict[str, object]]:
    return [
        {
            "collective": collective,
            "single_flow_size_bits": int(size_mb * BITS_PER_MB),
            "gap_after": float(gap_after),
        }
        for collective, size_mb, gap_after in specs
    ]


def program_2tenant_asymmetric() -> dict[int, list[dict[str, object]]]:
    return {
        0: _program(
            [
                ("allgather", 64, 1e-3),
                ("allreduce", 32, 1e-3),
                ("allgather", 32, 8e-4),
                ("allreduce", 64, 1e-3),
                ("allgather", 16, 1e-3),
                ("allreduce", 32, 8e-4),
                ("allgather", 16, 1e-3),
                ("allreduce", 16, 0.0),
            ]
        ),
        1: _program(
            [
                ("allgather", 4, 1e-3),
                ("allreduce", 2, 1e-3),
                ("allgather", 4, 8e-4),
                ("allreduce", 2, 1e-3),
                ("allgather", 2, 1e-3),
                ("allreduce", 2, 8e-4),
                ("allgather", 2, 1e-3),
                ("allreduce", 1, 0.0),
            ]
        ),
    }


def program_3tenant_asymmetric() -> dict[int, list[dict[str, object]]]:
    collectives = [
        "allgather",
        "allreduce",
        "allgather",
        "allreduce",
        "allgather",
        "allreduce",
        "allgather",
        "allreduce",
        "allgather",
    ]
    gaps = [1e-3, 1.2e-3, 8e-4, 1e-3, 7e-4, 1e-3, 6e-4, 8e-4, 0.0]
    sizes_by_tenant = {
        0: [64, 32, 32, 64, 16, 32, 16, 16, 8],
        1: [4, 2, 4, 2, 2, 2, 2, 1, 1],
        2: [8, 4, 4, 8, 2, 4, 2, 2, 1],
    }
    return {
        tenant: _program(list(zip(collectives, sizes, gaps)))
        for tenant, sizes in sizes_by_tenant.items()
    }


def _server_sharing_summary(mapping: dict[int, dict[int, int]]) -> str:
    server_use = Counter(server for ranks in mapping.values() for server in ranks.values())
    shared = sorted(server for server, count in server_use.items() if count > 1)
    max_share = max(server_use.values(), default=0)
    return f"shared_servers={shared} max_tenants_per_server={max_share}"


def _pct_improvement(baseline: float, value: float) -> float:
    if baseline == 0.0:
        return 0.0
    return (baseline - value) / baseline * 100.0


def _run_case(
    name: str,
    placement_factory,
    program_factory,
    *,
    mapping_time_limit: float = 2.0,
    harmonics_time_limit: float = 0.5,
):
    datacenter = LeafSpineDatacenter(*TOPOLOGY)
    tenant_mapping = placement_factory()
    program = program_factory()
    default_path_table = datacenter.build_tenant_ecmp_path_table(tenant_mapping)

    default_ms, default_avg = simulate_collective(
        datacenter.topology,
        tenant_mapping,
        default_path_table,
        tenant_collective_programs=program,
    )

    t0 = time.time()
    mapping_solver = MappingHeuristicSolver(
        datacenter,
        tenant_mapping,
        None,
        verbose=False,
        tenant_collective_programs=program,
        validate_with_simulator=False,
        path_table=default_path_table,
    )
    mapping_solver.solve(time_limit=mapping_time_limit)
    mapped = mapping_solver.get_X_mapping()
    mapping_wall = time.time() - t0
    mapped_path_table = datacenter.build_tenant_ecmp_path_table(mapped)

    mapping_ms, mapping_avg = simulate_collective(
        datacenter.topology,
        mapped,
        mapped_path_table,
        tenant_collective_programs=program,
    )

    t0 = time.time()
    harmonics = HarmonicsBaselineHeuristic(
        datacenter,
        tenant_mapping,
        None,
        default_path_table,
        None,
        None,
        verbose=False,
        tenant_collective_programs=program,
        program_time_limit_s=harmonics_time_limit,
    )
    baseline_schedule = harmonics.solve()
    harmonics_wall = time.time() - t0

    default_h_ms, default_h_avg = simulate_collective(
        datacenter.topology,
        tenant_mapping,
        default_path_table,
        tenant_collective_programs=program,
        tenant_start_times=_extract_start_times(baseline_schedule),
        collective_start_times=harmonics.get_collective_start_times(),
        collective_rate_scales=harmonics.get_collective_rate_scales(),
        collective_rate_schedule=harmonics.get_collective_rate_schedule(),
        task_rate_schedule=harmonics.get_task_rate_schedule(),
    )

    t0 = time.time()
    harmonics_on_mapping = HarmonicsBaselineHeuristic(
        datacenter,
        mapped,
        None,
        mapped_path_table,
        None,
        None,
        verbose=False,
        tenant_collective_programs=program,
        program_time_limit_s=harmonics_time_limit,
    )
    mapped_schedule = harmonics_on_mapping.solve()
    mapping_h_wall = time.time() - t0

    mapping_h_ms, mapping_h_avg = simulate_collective(
        datacenter.topology,
        mapped,
        mapped_path_table,
        tenant_collective_programs=program,
        tenant_start_times=_extract_start_times(mapped_schedule),
        collective_start_times=harmonics_on_mapping.get_collective_start_times(),
        collective_rate_scales=harmonics_on_mapping.get_collective_rate_scales(),
        collective_rate_schedule=harmonics_on_mapping.get_collective_rate_schedule(),
        task_rate_schedule=harmonics_on_mapping.get_task_rate_schedule(),
    )

    print(f"\n=== {name} ===")
    print(f"topology={TOPOLOGY}  { _server_sharing_summary(tenant_mapping) }")
    print(f"default_mapping={tenant_mapping}")
    print(f"mapped_mapping={mapped}")
    print(f"{'method':18s} {'makespan':>10s} {'avg_jct':>10s} {'avg_imp':>9s} {'ms_imp':>9s}")
    for method, makespan, avg_jct in [
        ("default", default_ms, default_avg),
        ("default+harmonics", default_h_ms, default_h_avg),
        ("mapping", mapping_ms, mapping_avg),
        ("mapping+harmonics", mapping_h_ms, mapping_h_avg),
    ]:
        print(
            f"{method:18s} "
            f"{makespan:10.6f} "
            f"{avg_jct:10.6f} "
            f"{_pct_improvement(default_avg, avg_jct):8.2f}% "
            f"{_pct_improvement(default_ms, makespan):8.2f}%"
        )
    print(
        "wall "
        f"mapping={mapping_wall:.3f}s "
        f"harmonics={harmonics_wall:.3f}s "
        f"mapping+h={mapping_h_wall:.3f}s"
    )


def main():
    cases = [
        ("2T-low-exclusive-6rank", low_contention_2tenant_6rank, program_2tenant_asymmetric),
        ("2T-high-shared-6rank", high_contention_2tenant_6rank, program_2tenant_asymmetric),
        ("3T-low-exclusive-4rank", low_contention_3tenant_4rank, program_3tenant_asymmetric),
        ("3T-high-shared-4rank", high_contention_3tenant_4rank, program_3tenant_asymmetric),
    ]

    print("=== Explicit Low/High Contention Mapping-Harmonics Cases ===")
    for case in cases:
        _run_case(*case)


if __name__ == "__main__":
    main()
