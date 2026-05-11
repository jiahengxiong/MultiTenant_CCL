from __future__ import annotations

import sys
from pathlib import Path
import random
from statistics import mean

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from multitenant.baselines import HarmonicsBaselineHeuristic
from multitenant.config import BITS_PER_MB, ExperimentConfig
from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingHeuristicSolver
from multitenant.topology import LeafSpineDatacenter
from multitenant.workloads import build_random_tenant_mapping


def _extract_start_times(schedule):
    if not schedule:
        return None
    return {tenant: schedule[tenant][0] for tenant in schedule}


def _tune_harmonics(harmonics: HarmonicsBaselineHeuristic):
    harmonics.rate_scale_levels = (1.0,)
    harmonics._max_isolated_slots = 8
    harmonics.runtime_cycle_slots = 1
    harmonics.runtime_lookahead_slots = 1
    return harmonics


def _make_program(num_tenants: int, single_flow_size_bits: int):
    ops = (
        {"collective": "allgather", "single_flow_size_bits": single_flow_size_bits, "gap_after": 0.0},
        {"collective": "allreduce", "single_flow_size_bits": single_flow_size_bits, "gap_after": 0.0},
    )
    return {tenant: [dict(op) for op in ops] for tenant in range(num_tenants)}


def _run_one(num_tenants: int, seed: int, *, servers_per_tenant: int):
    rng = random.Random(seed)
    cfg = ExperimentConfig(num_tenants=num_tenants, num_experiments=1)
    dc = LeafSpineDatacenter(cfg.topology.num_leaf, cfg.topology.num_spine, cfg.topology.servers_per_leaf)

    tenant_mapping = build_random_tenant_mapping(
        dc.get_all_servers(),
        cfg.num_tenants,
        rng=rng,
        servers_per_tenant=servers_per_tenant,
    )
    program = _make_program(cfg.num_tenants, single_flow_size_bits=int(0.25 * BITS_PER_MB))

    default_ms, default_avg = simulate_collective(
        dc.topology,
        tenant_mapping,
        dc.paths,
        tenant_collective_programs=program,
    )

    mapping_solver = MappingHeuristicSolver(
        dc,
        tenant_mapping,
        None,
        verbose=False,
        tenant_collective_programs=program,
        validate_with_simulator=False,
    )
    mapping_solver.solve(time_limit=0.2)
    heuristic_mapping = mapping_solver.get_X_mapping()

    mapping_ms, mapping_avg = simulate_collective(
        dc.topology,
        heuristic_mapping,
        dc.paths,
        tenant_collective_programs=program,
    )

    harmonics = HarmonicsBaselineHeuristic(
        dc,
        tenant_mapping,
        None,
        dc.paths,
        None,
        None,
        verbose=False,
        tenant_collective_programs=program,
    )
    _tune_harmonics(harmonics)
    schedule = harmonics.solve()
    default_h_ms, default_h_avg = simulate_collective(
        dc.topology,
        tenant_mapping,
        dc.paths,
        tenant_collective_programs=program,
        tenant_start_times=_extract_start_times(schedule),
        collective_start_times=harmonics.get_collective_start_times(),
        collective_rate_scales=harmonics.get_collective_rate_scales(),
    )

    harmonics_on_mapping = HarmonicsBaselineHeuristic(
        dc,
        heuristic_mapping,
        None,
        dc.paths,
        None,
        None,
        verbose=False,
        tenant_collective_programs=program,
    )
    _tune_harmonics(harmonics_on_mapping)
    schedule_on_mapping = harmonics_on_mapping.solve()
    mapping_h_ms, mapping_h_avg = simulate_collective(
        dc.topology,
        heuristic_mapping,
        dc.paths,
        tenant_collective_programs=program,
        tenant_start_times=_extract_start_times(schedule_on_mapping),
        collective_start_times=harmonics_on_mapping.get_collective_start_times(),
        collective_rate_scales=harmonics_on_mapping.get_collective_rate_scales(),
    )

    return {
        "default": (default_ms, default_avg),
        "default+mapping": (mapping_ms, mapping_avg),
        "default+harmonics": (default_h_ms, default_h_avg),
        "mapping+harmonics": (mapping_h_ms, mapping_h_avg),
    }


def main():
    tenant_counts = (2, 4, 8)
    num_exp = 1
    base_seed = 20240
    servers_per_tenant = 4
    keys = ("default", "default+mapping", "default+harmonics", "mapping+harmonics")

    print("=== Small-scale mapping/harmonics ablation (multi-collective program) ===")
    print(
        "topology=leaf4-spine2-spl8, "
        f"servers_per_tenant={servers_per_tenant}, "
        "program_ops=2, "
        f"seeds={base_seed}..{base_seed + num_exp - 1}"
    )

    for n in tenant_counts:
        acc = {k: [] for k in keys}
        for i in range(num_exp):
            res = _run_one(n, base_seed + i, servers_per_tenant=servers_per_tenant)
            for k in keys:
                acc[k].append(res[k])

        print(f"\nnum_tenants={n}")
        for k in keys:
            ms = mean(v[0] for v in acc[k])
            avg = mean(v[1] for v in acc[k])
            print(f"  {k:17s} ms={ms:.6f}  avg={avg:.6f}")


if __name__ == "__main__":
    main()
