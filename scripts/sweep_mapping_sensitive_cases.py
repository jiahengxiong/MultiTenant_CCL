from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from multitenant.baselines import HarmonicsBaselineHeuristic
from multitenant.simulator.adapter import simulate_collective_details
from multitenant.solvers import MappingHeuristicSolver
from multitenant.topology import LeafSpineDatacenter


BITS_PER_MB = 8 * 1024 * 1024
TOPOLOGY = (3, 2, 4)


@dataclass(frozen=True)
class Scenario:
    name: str
    tenant_servers: dict[int, list[int]]
    programs: dict[int, list[dict[str, object]]]


def _program(specs: list[tuple[str, int, float]]) -> list[dict[str, object]]:
    return [
        {
            "collective": collective,
            "single_flow_size_bits": int(size_mb * BITS_PER_MB),
            "gap_after": float(gap_after),
        }
        for collective, size_mb, gap_after in specs
    ]


def _mapping_from_servers(tenant_servers: dict[int, list[int]]) -> dict[int, dict[int, int]]:
    return {
        int(tenant): {rank: int(server) for rank, server in enumerate(servers)}
        for tenant, servers in tenant_servers.items()
    }


def _extract_start_times(schedule):
    if not schedule:
        return None
    return {int(tenant): float(schedule[tenant][0]) for tenant in schedule}


def _fixed_gap_by_tenant(programs: dict[int, list[dict[str, object]]]) -> dict[int, float]:
    return {
        int(tenant): sum(float(op.get("gap_after", 0.0)) for op in program)
        for tenant, program in programs.items()
    }


def _communication_metrics(details: dict[str, object], programs: dict[int, list[dict[str, object]]]):
    gaps = _fixed_gap_by_tenant(programs)
    tenant_comm = []
    for tenant_key, finish_time in details.get("tenant_makespans", {}).items():
        tenant = int(tenant_key)
        tenant_comm.append(max(0.0, float(finish_time) - float(gaps.get(tenant, 0.0))))
    if not tenant_comm:
        return 0.0, 0.0
    return max(tenant_comm), sum(tenant_comm) / len(tenant_comm)


def _simulate(datacenter, mapping, programs, harmonics=None, schedule=None):
    kwargs = {}
    if harmonics is not None:
        kwargs.update(
            tenant_start_times=_extract_start_times(schedule),
            collective_start_times=harmonics.get_collective_start_times(),
            collective_rate_scales=harmonics.get_collective_rate_scales(),
            collective_rate_schedule=harmonics.get_collective_rate_schedule(),
            task_rate_schedule=harmonics.get_task_rate_schedule(),
        )
    return simulate_collective_details(
        datacenter.topology,
        mapping,
        datacenter.paths,
        tenant_collective_programs=programs,
        **kwargs,
    )


def _server_sharing(mapping: dict[int, dict[int, int]]) -> tuple[int, int]:
    usage = Counter(server for ranks in mapping.values() for server in ranks.values())
    shared_servers = sum(1 for count in usage.values() if count > 1)
    max_share = max(usage.values(), default=0)
    return shared_servers, max_share


def balanced_ag_ar(num_tenants: int, *, size_mb: int = 16, ops: int = 10, gap_ms: float = 10.0):
    collectives = ["allgather", "allreduce"] * ((ops + 1) // 2)
    collectives = collectives[:ops]
    return {
        tenant: _program(
            [
                (
                    collective,
                    size_mb if collective == "allgather" else max(1, size_mb // 2),
                    0.0 if idx == ops - 1 else (gap_ms + (tenant % 3) * 2.0) / 1000.0,
                )
                for idx, collective in enumerate(collectives)
            ]
        )
        for tenant in range(num_tenants)
    }


def ag_heavy(num_tenants: int, *, size_mb: int = 16, ops: int = 10, gap_ms: float = 10.0):
    collectives = ["allgather", "allgather", "allreduce", "allgather", "allgather"]
    return {
        tenant: _program(
            [
                (
                    collectives[idx % len(collectives)],
                    size_mb if collectives[idx % len(collectives)] == "allgather" else max(1, size_mb // 2),
                    0.0 if idx == ops - 1 else (gap_ms + (idx % 3) * 3.0 + tenant) / 1000.0,
                )
                for idx in range(ops)
            ]
        )
        for tenant in range(num_tenants)
    }


def mild_asym(num_tenants: int, *, ops: int = 10, gap_ms: float = 10.0):
    base_sizes = [32, 24, 16, 12]
    collectives = ["allgather", "allreduce", "allgather", "allreduce", "allgather"]
    return {
        tenant: _program(
            [
                (
                    collectives[idx % len(collectives)],
                    max(1, base_sizes[tenant % len(base_sizes)] // (2 if collectives[idx % len(collectives)] == "allreduce" else 1)),
                    0.0 if idx == ops - 1 else (gap_ms + (tenant * 2 + idx % 2) * 1.0) / 1000.0,
                )
                for idx in range(ops)
            ]
        )
        for tenant in range(num_tenants)
    }


def scenarios() -> list[Scenario]:
    return [
        Scenario(
            "2T-low-interleaved-6rank-balanced",
            {
                0: [0, 4, 8, 1, 5, 9],
                1: [2, 6, 10, 3, 7, 11],
            },
            balanced_ag_ar(2, size_mb=16, ops=10, gap_ms=10.0),
        ),
        Scenario(
            "2T-high-shared-interleaved-6rank-balanced",
            {
                0: [0, 4, 8, 1, 5, 9],
                1: [0, 4, 8, 1, 5, 9],
            },
            balanced_ag_ar(2, size_mb=16, ops=10, gap_ms=10.0),
        ),
        Scenario(
            "2T-high-shared-interleaved-6rank-ag-heavy",
            {
                0: [0, 4, 8, 1, 5, 9],
                1: [0, 4, 8, 1, 5, 9],
            },
            ag_heavy(2, size_mb=16, ops=10, gap_ms=10.0),
        ),
        Scenario(
            "2T-high-shared-interleaved-6rank-mild-asym",
            {
                0: [0, 4, 8, 1, 5, 9],
                1: [0, 4, 8, 1, 5, 9],
            },
            mild_asym(2, ops=10, gap_ms=10.0),
        ),
        Scenario(
            "3T-low-interleaved-4rank-balanced",
            {
                0: [0, 4, 8, 1],
                1: [2, 6, 10, 3],
                2: [5, 9, 7, 11],
            },
            balanced_ag_ar(3, size_mb=16, ops=10, gap_ms=10.0),
        ),
        Scenario(
            "3T-high-pairwise-interleaved-4rank-balanced",
            {
                0: [0, 4, 8, 1],
                1: [0, 4, 10, 3],
                2: [8, 1, 10, 3],
            },
            balanced_ag_ar(3, size_mb=16, ops=10, gap_ms=10.0),
        ),
        Scenario(
            "3T-high-pairwise-interleaved-4rank-ag-heavy",
            {
                0: [0, 4, 8, 1],
                1: [0, 4, 10, 3],
                2: [8, 1, 10, 3],
            },
            ag_heavy(3, size_mb=16, ops=10, gap_ms=10.0),
        ),
        Scenario(
            "3T-high-pairwise-interleaved-4rank-mild-asym",
            {
                0: [0, 4, 8, 1],
                1: [0, 4, 10, 3],
                2: [8, 1, 10, 3],
            },
            mild_asym(3, ops=10, gap_ms=10.0),
        ),
    ]


def run_scenario(scenario: Scenario, *, mapping_time_limit=2.0, harmonics_time_limit=0.5):
    datacenter = LeafSpineDatacenter(*TOPOLOGY)
    default_mapping = _mapping_from_servers(scenario.tenant_servers)
    shared_servers, max_share = _server_sharing(default_mapping)

    t0 = time.time()
    default_details = _simulate(datacenter, default_mapping, scenario.programs)
    default_wall = time.time() - t0

    t0 = time.time()
    harmonics = HarmonicsBaselineHeuristic(
        datacenter,
        default_mapping,
        None,
        datacenter.paths,
        None,
        None,
        verbose=False,
        tenant_collective_programs=scenario.programs,
        program_time_limit_s=harmonics_time_limit,
    )
    harmonics_schedule = harmonics.solve()
    harmonics_wall = time.time() - t0
    harmonics_details = _simulate(datacenter, default_mapping, scenario.programs, harmonics, harmonics_schedule)

    t0 = time.time()
    mapping_solver = MappingHeuristicSolver(
        datacenter,
        default_mapping,
        None,
        verbose=False,
        tenant_collective_programs=scenario.programs,
        validate_with_simulator=False,
    )
    mapping_solver.solve(time_limit=mapping_time_limit)
    mapped = mapping_solver.get_X_mapping()
    mapping_wall = time.time() - t0
    mapping_details = _simulate(datacenter, mapped, scenario.programs)

    t0 = time.time()
    harmonics_on_mapping = HarmonicsBaselineHeuristic(
        datacenter,
        mapped,
        None,
        datacenter.paths,
        None,
        None,
        verbose=False,
        tenant_collective_programs=scenario.programs,
        program_time_limit_s=harmonics_time_limit,
    )
    mapping_h_schedule = harmonics_on_mapping.solve()
    mapping_h_wall = time.time() - t0
    mapping_h_details = _simulate(datacenter, mapped, scenario.programs, harmonics_on_mapping, mapping_h_schedule)

    rows = {}
    for label, details in (
        ("default", default_details),
        ("harmonics", harmonics_details),
        ("mapping", mapping_details),
        ("mapping_harmonics", mapping_h_details),
    ):
        comm_ms, comm_avg = _communication_metrics(details, scenario.programs)
        rows[label] = {
            "comm_makespan": comm_ms,
            "comm_avg": comm_avg,
            "e2e_makespan": float(details["global_makespan"]),
            "e2e_avg": float(details["avg_tenant_makespan"]),
        }

    base = rows["default"]["comm_avg"]
    harmonics_imp = (base - rows["harmonics"]["comm_avg"]) / base * 100.0 if base else 0.0
    mapping_imp = (base - rows["mapping"]["comm_avg"]) / base * 100.0 if base else 0.0
    joint_imp = (base - rows["mapping_harmonics"]["comm_avg"]) / base * 100.0 if base else 0.0
    mapping_over_h = mapping_imp - harmonics_imp
    joint_over_mapping = joint_imp - mapping_imp

    return {
        "name": scenario.name,
        "num_tenants": len(scenario.tenant_servers),
        "num_ops_per_tenant": {tenant: len(program) for tenant, program in scenario.programs.items()},
        "shared_servers": shared_servers,
        "max_share": max_share,
        "rows": rows,
        "improvements": {
            "harmonics": harmonics_imp,
            "mapping": mapping_imp,
            "mapping_harmonics": joint_imp,
            "mapping_over_harmonics": mapping_over_h,
            "joint_over_mapping": joint_over_mapping,
        },
        "wall": {
            "default": default_wall,
            "harmonics": harmonics_wall,
            "mapping": mapping_wall,
            "mapping_harmonics": mapping_h_wall,
        },
        "mapped": mapped,
    }


def _fmt(value):
    return f"{float(value):.6f}"


def main():
    results = []
    print("=== Mapping-Sensitive Scenario Sweep ===")
    print(f"topology={TOPOLOGY}, metric=tenant collective-only time (tenant finish - fixed gaps)")
    for scenario in scenarios():
        result = run_scenario(scenario)
        results.append(result)
        rows = result["rows"]
        imp = result["improvements"]
        print(f"\n=== {result['name']} ===")
        print(
            f"tenants={result['num_tenants']} shared_servers={result['shared_servers']} "
            f"max_share={result['max_share']}"
        )
        print(
            f"default_avg={_fmt(rows['default']['comm_avg'])} "
            f"harmonics_avg={_fmt(rows['harmonics']['comm_avg'])} "
            f"mapping_avg={_fmt(rows['mapping']['comm_avg'])} "
            f"joint_avg={_fmt(rows['mapping_harmonics']['comm_avg'])}"
        )
        print(
            f"imp harmonics={imp['harmonics']:+.2f}% "
            f"mapping={imp['mapping']:+.2f}% "
            f"joint={imp['mapping_harmonics']:+.2f}% "
            f"mapping-over-H={imp['mapping_over_harmonics']:+.2f}pp "
            f"joint-over-M={imp['joint_over_mapping']:+.2f}pp"
        )
        print(
            "wall "
            f"H={result['wall']['harmonics']:.3f}s "
            f"M={result['wall']['mapping']:.3f}s "
            f"M+H={result['wall']['mapping_harmonics']:.3f}s"
        )

    ranked = sorted(
        results,
        key=lambda result: (
            -result["improvements"]["mapping_over_harmonics"],
            -result["improvements"]["mapping"],
            -result["improvements"]["joint_over_mapping"],
        ),
    )
    print("\n=== Top Mapping-Sensitive Cases ===")
    print(f"{'case':52s} {'H':>8s} {'M':>8s} {'M+H':>8s} {'M-H':>8s} {'J-M':>8s}")
    for result in ranked[:8]:
        imp = result["improvements"]
        print(
            f"{result['name'][:52]:52s} "
            f"{imp['harmonics']:8.2f} "
            f"{imp['mapping']:8.2f} "
            f"{imp['mapping_harmonics']:8.2f} "
            f"{imp['mapping_over_harmonics']:8.2f} "
            f"{imp['joint_over_mapping']:8.2f}"
        )


if __name__ == "__main__":
    main()
