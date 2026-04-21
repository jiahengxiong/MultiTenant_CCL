from __future__ import annotations

import math
import random
import time

from multitenant.baselines import HarmonicsBaselineILP
from multitenant.config import BITS_PER_MB, ExperimentConfig, TopologyConfig
from multitenant.simulator import simulate_collective
from multitenant.solvers import MappingCGSolver, MappingILPSolver
from multitenant.topology import LeafSpineDatacenter
from multitenant.workloads import build_random_tenant_mapping, build_ring_flows


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _extract_start_times(schedule):
    if not schedule:
        return None
    return {tenant: schedule[tenant][0] for tenant in schedule}


def _metadata_from_config(config: ExperimentConfig) -> dict[str, object]:
    return {
        "collective": config.collective,
        "num_experiments": config.num_experiments,
        "num_tenants": config.num_tenants,
        "single_flow_size_bits": config.single_flow_size_bits,
        "topology": {
            "num_leaf": config.topology.num_leaf,
            "num_spine": config.topology.num_spine,
            "servers_per_leaf": config.topology.servers_per_leaf,
        },
    }


def evaluate_baseline_vs_proposed_mapping(
    config: ExperimentConfig,
    *,
    seed: int | None = None,
) -> dict[str, object]:
    """Compare baseline methods against the proposed mapping CG method."""

    rng = random.Random(seed)
    datacenter = LeafSpineDatacenter(
        config.topology.num_leaf,
        config.topology.num_spine,
        config.topology.servers_per_leaf,
    )

    baseline_random_makespan = []
    harmonics_baseline_makespan = []
    proposed_mapping_cg_makespan = []
    proposed_mapping_cg_plus_harmonics_makespan = []

    baseline_random_avg_jct = []
    harmonics_baseline_avg_jct = []
    proposed_mapping_cg_avg_jct = []
    proposed_mapping_cg_plus_harmonics_avg_jct = []

    for _ in range(config.num_experiments):
        tenant_mapping = build_random_tenant_mapping(
            datacenter.get_all_servers(),
            config.num_tenants,
            rng=rng,
        )
        tenant_flows = build_ring_flows(
            tenant_mapping,
            config.single_flow_size_bits,
            config.collective,
        )

        random_makespan, random_avg_jct = simulate_collective(
            datacenter.topology,
            tenant_mapping,
            datacenter.paths,
            config.single_flow_size_bits,
            config.collective,
        )
        baseline_random_makespan.append(random_makespan)
        baseline_random_avg_jct.append(random_avg_jct)

        harmonics_baseline = HarmonicsBaselineILP(
            datacenter,
            tenant_mapping,
            tenant_flows,
            datacenter.paths,
            config.single_flow_size_bits,
            config.collective,
            estimation=random_makespan,
            verbose=False,
        )
        baseline_schedule = harmonics_baseline.solve()
        baseline_start_times = _extract_start_times(baseline_schedule)
        baseline_makespan, baseline_avg_jct = simulate_collective(
            datacenter.topology,
            tenant_mapping,
            datacenter.paths,
            config.single_flow_size_bits,
            config.collective,
            tenant_start_times=baseline_start_times,
        )
        harmonics_baseline_makespan.append(baseline_makespan)
        harmonics_baseline_avg_jct.append(baseline_avg_jct)

        proposed_mapping_cg = MappingCGSolver(datacenter, tenant_mapping, tenant_flows, verbose=False)
        cg_mapping = proposed_mapping_cg.solve()

        if cg_mapping:
            cg_makespan, cg_avg_jct = simulate_collective(
                datacenter.topology,
                cg_mapping,
                datacenter.paths,
                config.single_flow_size_bits,
                config.collective,
            )
            proposed_mapping_cg_makespan.append(cg_makespan)
            proposed_mapping_cg_avg_jct.append(cg_avg_jct)

            harmonics_on_cg = HarmonicsBaselineILP(
                datacenter,
                cg_mapping,
                tenant_flows,
                datacenter.paths,
                config.single_flow_size_bits,
                config.collective,
                estimation=cg_makespan,
                verbose=False,
            )
            cg_schedule = harmonics_on_cg.solve()
            cg_start_times = _extract_start_times(cg_schedule)
            cg_harmonics_makespan, cg_harmonics_avg_jct = simulate_collective(
                datacenter.topology,
                cg_mapping,
                datacenter.paths,
                config.single_flow_size_bits,
                config.collective,
                tenant_start_times=cg_start_times,
            )
            proposed_mapping_cg_plus_harmonics_makespan.append(cg_harmonics_makespan)
            proposed_mapping_cg_plus_harmonics_avg_jct.append(cg_harmonics_avg_jct)
        else:
            proposed_mapping_cg_makespan.append(random_makespan)
            proposed_mapping_cg_avg_jct.append(random_avg_jct)
            proposed_mapping_cg_plus_harmonics_makespan.append(baseline_makespan)
            proposed_mapping_cg_plus_harmonics_avg_jct.append(baseline_avg_jct)

    return {
        "metadata": _metadata_from_config(config),
        "results": {
            "baseline_random_mapping": {
                "makespan": _mean(baseline_random_makespan),
                "avg_jct": _mean(baseline_random_avg_jct),
            },
            "harmonics_baseline": {
                "makespan": _mean(harmonics_baseline_makespan),
                "avg_jct": _mean(harmonics_baseline_avg_jct),
            },
            "proposed_mapping_cg": {
                "makespan": _mean(proposed_mapping_cg_makespan),
                "avg_jct": _mean(proposed_mapping_cg_avg_jct),
            },
            "proposed_mapping_cg_plus_harmonics": {
                "makespan": _mean(proposed_mapping_cg_plus_harmonics_makespan),
                "avg_jct": _mean(proposed_mapping_cg_plus_harmonics_avg_jct),
            },
        },
    }


def run_small_scale_proposed_mapping_validation(
    *,
    num_experiments: int = 10,
    num_tenants: int = 4,
    topology: TopologyConfig | None = None,
    collective: str = "allreduce",
    single_flow_size_bits: int = 8 * BITS_PER_MB,
    seed: int | None = None,
) -> dict[str, object]:
    """Use the exact proposed mapping ILP to validate the scalable CG method."""

    topology = topology or TopologyConfig(num_leaf=3, num_spine=2, servers_per_leaf=4)
    rng = random.Random(seed)

    ilp_objectives = []
    cg_objectives = []
    ilp_runtimes = []
    cg_runtimes = []
    ilp_makespans = []
    cg_makespans = []
    ilp_avg_jcts = []
    cg_avg_jcts = []
    optimality_gaps = []

    for _ in range(num_experiments):
        datacenter = LeafSpineDatacenter(
            topology.num_leaf,
            topology.num_spine,
            topology.servers_per_leaf,
        )
        tenant_mapping = build_random_tenant_mapping(
            datacenter.get_all_servers(),
            num_tenants,
            rng=rng,
        )
        tenant_flows = build_ring_flows(tenant_mapping, single_flow_size_bits, collective)

        ilp_start = time.time()
        proposed_mapping_ilp = MappingILPSolver(datacenter, tenant_mapping, tenant_flows, verbose=False)
        proposed_mapping_ilp.solve()
        ilp_runtimes.append(time.time() - ilp_start)

        if proposed_mapping_ilp.model.Status == 2:
            ilp_objective = proposed_mapping_ilp.model.ObjVal
            ilp_mapping = proposed_mapping_ilp.get_X_mapping()
            ilp_makespan, ilp_avg_jct = simulate_collective(
                datacenter.topology,
                ilp_mapping,
                datacenter.paths,
                single_flow_size_bits,
                collective,
            )
        else:
            ilp_objective = float("inf")
            ilp_makespan = float("inf")
            ilp_avg_jct = float("inf")

        cg_start = time.time()
        proposed_mapping_cg = MappingCGSolver(datacenter, tenant_mapping, tenant_flows, verbose=False)
        cg_mapping = proposed_mapping_cg.solve(max_iter=50)
        cg_runtimes.append(time.time() - cg_start)

        if proposed_mapping_cg.final_obj is not None and cg_mapping:
            cg_objective = proposed_mapping_cg.final_obj
            cg_makespan, cg_avg_jct = simulate_collective(
                datacenter.topology,
                cg_mapping,
                datacenter.paths,
                single_flow_size_bits,
                collective,
            )
        else:
            cg_objective = float("inf")
            cg_makespan = float("inf")
            cg_avg_jct = float("inf")

        ilp_objectives.append(ilp_objective)
        cg_objectives.append(cg_objective)
        ilp_makespans.append(ilp_makespan)
        cg_makespans.append(cg_makespan)
        ilp_avg_jcts.append(ilp_avg_jct)
        cg_avg_jcts.append(cg_avg_jct)

        if ilp_objective > 1e-6 and math.isfinite(ilp_objective) and math.isfinite(cg_objective):
            optimality_gaps.append((cg_objective - ilp_objective) / ilp_objective * 100.0)
        else:
            optimality_gaps.append(0.0)

    return {
        "metadata": {
            "collective": collective,
            "num_experiments": num_experiments,
            "num_tenants": num_tenants,
            "single_flow_size_bits": single_flow_size_bits,
            "topology": {
                "num_leaf": topology.num_leaf,
                "num_spine": topology.num_spine,
                "servers_per_leaf": topology.servers_per_leaf,
            },
        },
        "results": {
            "proposed_mapping_ilp": {
                "objective": _mean(ilp_objectives),
                "runtime_seconds": _mean(ilp_runtimes),
                "makespan": _mean(ilp_makespans),
                "avg_jct": _mean(ilp_avg_jcts),
            },
            "proposed_mapping_cg": {
                "objective": _mean(cg_objectives),
                "runtime_seconds": _mean(cg_runtimes),
                "makespan": _mean(cg_makespans),
                "avg_jct": _mean(cg_avg_jcts),
            },
            "optimality_gap_percent": _mean(optimality_gaps),
        },
    }


def run_large_scale_proposed_mapping_cg(
    *,
    tenant_counts=(4, 8, 16, 32),
    per_tenant_size: int = 4,
    collective: str = "allreduce",
    single_flow_size_bits: int = 8 * BITS_PER_MB,
    seed: int | None = None,
) -> dict[str, object]:
    """Run large-scale experiments using the scalable proposed mapping CG method."""

    rng = random.Random(seed)
    results_by_tenant_count = {}

    for tenant_count in tenant_counts:
        total_servers_needed = tenant_count * per_tenant_size
        per_leaf = 8
        num_leaf = max(2, math.ceil(total_servers_needed / per_leaf))
        num_spine = max(2, num_leaf // 2)
        datacenter = LeafSpineDatacenter(num_leaf, num_spine, per_leaf)

        if len(datacenter.get_all_servers()) < total_servers_needed:
            per_leaf = math.ceil(total_servers_needed / num_leaf)
            datacenter = LeafSpineDatacenter(num_leaf, num_spine, per_leaf)

        selected_servers = datacenter.get_all_servers()[:total_servers_needed]
        rng.shuffle(selected_servers)

        tenant_mapping = {}
        next_server_index = 0
        for tenant in range(tenant_count):
            tenant_mapping[tenant] = {}
            for rank in range(per_tenant_size):
                tenant_mapping[tenant][rank] = selected_servers[next_server_index]
                next_server_index += 1

        tenant_flows = build_ring_flows(tenant_mapping, single_flow_size_bits, collective)

        start_time = time.time()
        proposed_mapping_cg = MappingCGSolver(datacenter, tenant_mapping, tenant_flows, verbose=False)
        cg_mapping = proposed_mapping_cg.solve(max_iter=50)
        runtime_seconds = time.time() - start_time

        cg_makespan = float("inf")
        cg_avg_jct = float("inf")
        if cg_mapping:
            cg_makespan, cg_avg_jct = simulate_collective(
                datacenter.topology,
                cg_mapping,
                datacenter.paths,
                single_flow_size_bits,
                collective,
            )

        results_by_tenant_count[str(tenant_count)] = {
            "proposed_mapping_cg": {
                "runtime_seconds": runtime_seconds,
                "objective": proposed_mapping_cg.final_obj if proposed_mapping_cg.final_obj is not None else float("inf"),
                "makespan": cg_makespan,
                "avg_jct": cg_avg_jct,
            }
        }

    return {
        "metadata": {
            "collective": collective,
            "per_tenant_size": per_tenant_size,
            "single_flow_size_bits": single_flow_size_bits,
            "tenant_counts": list(tenant_counts),
        },
        "results_by_tenant_count": results_by_tenant_count,
    }
