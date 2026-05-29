from __future__ import annotations

import sys
import time
from pathlib import Path
import random

from statistics import mean

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from multitenant.baselines import HarmonicsBaselineHeuristic
from multitenant.config import BITS_PER_MB, ExperimentConfig
from multitenant.simulator import simulate_collective
from multitenant.topology import LeafSpineDatacenter
from multitenant.workloads import build_random_tenant_mapping


def _extract_start_times(schedule):
    if not schedule:
        return None
    return {tenant: schedule[tenant][0] for tenant in schedule}


def _make_program(num_tenants: int, single_flow_size_bits: int):
    ops = (
        {"collective": "allgather", "single_flow_size_bits": single_flow_size_bits, "gap_after": 0.0},
        {"collective": "allreduce", "single_flow_size_bits": single_flow_size_bits, "gap_after": 0.0},
    )
    return {tenant: [dict(op) for op in ops] for tenant in range(num_tenants)}


def _configure_fast_harmonics(harmonics: HarmonicsBaselineHeuristic):
    harmonics._max_isolated_slots = 16
    return harmonics


def run_once(*, num_tenants: int, seed: int, servers_per_tenant: int, op_size_bits: int):
    cfg = ExperimentConfig(num_tenants=num_tenants, num_experiments=1)
    dc = LeafSpineDatacenter(cfg.topology.num_leaf, cfg.topology.num_spine, cfg.topology.servers_per_leaf)

    mapping = build_random_tenant_mapping(
        dc.get_all_servers(),
        cfg.num_tenants,
        rng=random.Random(seed),
        servers_per_tenant=servers_per_tenant,
    )
    path_table = dc.build_tenant_ecmp_path_table(mapping)
    program = _make_program(cfg.num_tenants, single_flow_size_bits=op_size_bits)

    t0 = time.time()
    default_ms, default_avg = simulate_collective(
        dc.topology,
        mapping,
        path_table,
        tenant_collective_programs=program,
    )
    default_wall = time.time() - t0

    harmonics = HarmonicsBaselineHeuristic(
        dc,
        mapping,
        None,
        path_table,
        None,
        None,
        verbose=False,
        tenant_collective_programs=program,
    )
    _configure_fast_harmonics(harmonics)

    t1 = time.time()
    schedule = harmonics.solve()
    harmonics_wall = time.time() - t1

    t2 = time.time()
    dh_ms, dh_avg = simulate_collective(
        dc.topology,
        mapping,
        path_table,
        tenant_collective_programs=program,
        tenant_start_times=_extract_start_times(schedule),
        collective_start_times=harmonics.get_collective_start_times(),
        collective_rate_scales=harmonics.get_collective_rate_scales(),
    )
    dh_wall = time.time() - t2

    return {
        "default": (default_ms, default_avg, default_wall),
        "default+harmonics": (dh_ms, dh_avg, harmonics_wall + dh_wall),
        "harmonics_breakdown": (harmonics_wall, dh_wall),
    }


def main():
    num_tenants = 8
    servers_per_tenant = 4
    seeds = (20240,)
    op_sizes_bits = (
        int(0.25 * BITS_PER_MB),
        int(1.0 * BITS_PER_MB),
        int(8.0 * BITS_PER_MB),
    )

    print("=== default vs default+harmonics (multi-collective program) ===")
    print(f"topology=leaf4-spine2-spl8, num_tenants={num_tenants}, servers_per_tenant={servers_per_tenant}")
    print(f"program_ops=2 (allgather->allreduce), seeds={list(seeds)}")

    for op_size_bits in op_sizes_bits:
        rows = []
        for seed in seeds:
            rows.append(
                run_once(
                    num_tenants=num_tenants,
                    seed=seed,
                    servers_per_tenant=servers_per_tenant,
                    op_size_bits=op_size_bits,
                )
            )

        def_ms = mean(r["default"][0] for r in rows)
        def_avg = mean(r["default"][1] for r in rows)
        dh_ms = mean(r["default+harmonics"][0] for r in rows)
        dh_avg = mean(r["default+harmonics"][1] for r in rows)

        def_wall = mean(r["default"][2] for r in rows)
        dh_wall = mean(r["default+harmonics"][2] for r in rows)
        h_solve = mean(r["harmonics_breakdown"][0] for r in rows)
        h_sim = mean(r["harmonics_breakdown"][1] for r in rows)

        print()
        print(f"op_size={op_size_bits / BITS_PER_MB:.2f} MiB")
        print(f"  default           ms={def_ms:.6f}  avg={def_avg:.6f}  wall={def_wall:.3f}s")
        print(f"  default+harmonics ms={dh_ms:.6f}  avg={dh_avg:.6f}  wall={dh_wall:.3f}s (solve={h_solve:.3f}s, sim={h_sim:.3f}s)")


if __name__ == "__main__":
    main()
