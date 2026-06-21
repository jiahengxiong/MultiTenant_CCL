from __future__ import annotations

import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path

from multitenant.config import BITS_PER_MB
from multitenant.topology import LeafSpineDatacenter

from .heuristic import RepairSearchConfig
from .models import FailureEvent, Mapping, RepairResult, RepairScenario, RepairWorkload
from .objectives import repair_sort_key
from .problem import FailureAwareMappingProblem
from .protection import (
    check_repair_feasible,
    normalize_mapping_for_json,
)
from .solvers.optimized_single_failover import solve_optimized_single_failover
from .strategy_solver import StrategySolveResult, solve_repair_strategies, solve_repair_strategy
from .strategies import (
    COOPERATIVE_REPAIR,
    REPAIR_FAILED_SERVER_ONLY,
    REPAIR_STRATEGIES,
    TENANT_LOCAL_REPAIR,
    RepairStrategySpec,
)


@dataclass(frozen=True)
class RandomRepairExperimentConfig:
    seed: int = 0
    num_leaf: int = 4
    num_spine: int = 2
    per_leaf_server: int = 6
    num_tenants: int = 3
    ranks_per_tenant: int | None = 4
    working_allocation_mode: str = "balanced_remaining"
    protection_pool_size_mode: str = "high_resource"
    contention: str = "low"
    workload_mode: str = "low_contention_dominant"
    collective: str = "allgather"
    single_flow_size_bits: int = 4 * BITS_PER_MB
    working_mapping_time_limit: float | None = None
    repair_time_limit: float | None = None
    horizon_slots: int | None = None
    failover_policy: str = "first"
    failure_selection: str = "random"
    repair_search: RepairSearchConfig | None = None


@dataclass(frozen=True)
class TenantWorkingMappingResult:
    tenant: int
    working_nodes: tuple[int, ...]
    initial_mapping: dict[int, int]
    best_mapping: dict[int, int]
    mapping_avg_jct: float
    mapping_makespan: float
    solver_name: str
    solver_source: str
    runtime_seconds: float


@dataclass(frozen=True)
class RandomRepairExperiment:
    config: RandomRepairExperimentConfig
    datacenter: LeafSpineDatacenter
    working_results: dict[int, TenantWorkingMappingResult]
    workload: RepairWorkload
    global_protection_pool: tuple[int, ...]
    pre_failure_mapping: Mapping
    failure: FailureEvent
    failure_selection_metadata: dict[str, object] | None = None

    @property
    def problem(self) -> FailureAwareMappingProblem:
        return FailureAwareMappingProblem(
            datacenter=self.datacenter,
            pre_failure_mapping=self.pre_failure_mapping,
            failure=self.failure,
            global_protection_pool=self.global_protection_pool,
            workload=self.workload,
            horizon_slots=self.config.horizon_slots,
            failover_policy=self.config.failover_policy,
        )


@dataclass(frozen=True)
class RepairComparison:
    experiment: RandomRepairExperiment
    strategies: tuple[RepairStrategySpec, ...]
    strategy_results: dict[str, StrategySolveResult]
    failover_scenario: RepairScenario
    tenant_local_scenario: RepairScenario
    cooperative_scenario: RepairScenario
    failover_mapping: Mapping
    failover_objective: object
    tenant_local: RepairResult
    cooperative: RepairResult

    def has_strict_repair_gain(self) -> bool:
        return (
            self.tenant_local.objective.avg_jct < self.failover_objective.avg_jct
            and self.cooperative.objective.avg_jct < self.failover_objective.avg_jct
            and self.cooperative.objective.avg_jct < self.tenant_local.objective.avg_jct
        )


def generate_random_repair_experiment(config: RandomRepairExperimentConfig) -> RandomRepairExperiment:
    rng = random.Random(int(config.seed))
    datacenter = LeafSpineDatacenter(
        num_leaf=config.num_leaf,
        num_spine=config.num_spine,
        per_leaf_server=config.per_leaf_server,
    )
    working_sets, global_protection_pool = sample_random_tenant_node_sets(
        datacenter.get_all_servers(),
        num_tenants=config.num_tenants,
        ranks_per_tenant=config.ranks_per_tenant,
        rng=rng,
        contention=config.contention,
        allocation_mode=config.working_allocation_mode,
        protection_pool_size_mode=config.protection_pool_size_mode,
        num_leaf=config.num_leaf,
        per_leaf_server=config.per_leaf_server,
    )

    working_results: dict[int, TenantWorkingMappingResult] = {}
    initial_working_mapping: Mapping = {}
    for tenant in range(config.num_tenants):
        ranked_servers = list(int(server) for server in working_sets[tenant])
        rng.shuffle(ranked_servers)
        initial_working_mapping[tenant] = {
            rank: ranked_servers[rank]
            for rank in range(len(ranked_servers))
        }

    pre_failure_mapping, working_results, workload = solve_best_working_set_mappings(
        datacenter,
        initial_mapping=initial_working_mapping,
        rng=rng,
        seed=config.seed,
        collective=config.collective,
        single_flow_size_bits=config.single_flow_size_bits,
        workload_mode=config.workload_mode,
        time_limit=config.working_mapping_time_limit,
        horizon_slots=config.horizon_slots,
    )
    _validate_full_server_partition(
        datacenter.get_all_servers(),
        {
            int(tenant): tuple(int(server) for server in rank_to_server.values())
            for tenant, rank_to_server in pre_failure_mapping.items()
        },
        global_protection_pool,
    )

    failure, failure_metadata = select_failure_event(
        datacenter,
        pre_failure_mapping=pre_failure_mapping,
        global_protection_pool=global_protection_pool,
        workload=workload,
        rng=rng,
        mode=config.failure_selection,
        horizon_slots=config.horizon_slots,
        failover_policy=config.failover_policy,
    )
    return RandomRepairExperiment(
        config=config,
        datacenter=datacenter,
        working_results=working_results,
        workload=workload,
        global_protection_pool=global_protection_pool,
        pre_failure_mapping=pre_failure_mapping,
        failure=failure,
        failure_selection_metadata=failure_metadata,
    )


def select_failure_event(
    datacenter: LeafSpineDatacenter,
    *,
    pre_failure_mapping: Mapping,
    global_protection_pool: tuple[int, ...],
    workload: RepairWorkload,
    rng: random.Random,
    mode: str = "random",
    horizon_slots: int | None = None,
    failover_policy: str = "same_leaf_or_nearest",
) -> tuple[FailureEvent, dict[str, object]]:
    """Select the server failure used by one repair experiment."""

    mode = str(mode).lower()
    if mode in {"random", "uniform_random"}:
        failure_tenant = rng.choice(sorted(int(tenant) for tenant in pre_failure_mapping))
        failed_rank = rng.choice(sorted(int(rank) for rank in pre_failure_mapping[failure_tenant]))
        failure = FailureEvent(
            tenant=failure_tenant,
            failed_rank=failed_rank,
            failed_server=pre_failure_mapping[failure_tenant][failed_rank],
        )
        return failure, {
            "mode": "random",
            "evaluated_failures": 1,
        }

    if mode not in {"critical_failover", "worst_failover", "max_failover"}:
        raise ValueError(f"unknown failure_selection mode: {mode}")

    best: tuple[float, float, int, int, int] | None = None
    evaluated = 0
    for tenant in sorted(int(tenant) for tenant in pre_failure_mapping):
        for rank in sorted(int(rank) for rank in pre_failure_mapping[tenant]):
            server = int(pre_failure_mapping[tenant][rank])
            failure = FailureEvent(tenant=tenant, failed_rank=rank, failed_server=server)
            problem = FailureAwareMappingProblem(
                datacenter=datacenter,
                pre_failure_mapping=pre_failure_mapping,
                failure=failure,
                global_protection_pool=global_protection_pool,
                workload=workload,
                horizon_slots=horizon_slots,
                failover_policy=failover_policy,
            )
            evaluator = problem.evaluator(REPAIR_FAILED_SERVER_ONLY)
            obj = solve_optimized_single_failover(
                problem.scenario(REPAIR_FAILED_SERVER_ONLY),
                evaluator,
                datacenter=datacenter,
            ).objective
            evaluated += 1
            candidate = (
                float(obj.avg_jct),
                float(obj.makespan),
                int(tenant),
                int(rank),
                int(server),
            )
            if best is None or candidate > best:
                best = candidate

    if best is None:
        raise RuntimeError("cannot select a critical failure from an empty mapping")
    failure = FailureEvent(tenant=best[2], failed_rank=best[3], failed_server=best[4])
    return failure, {
        "mode": "critical_failover",
        "evaluated_failures": evaluated,
        "selected_failover_avg_jct": best[0],
        "selected_failover_makespan": best[1],
    }


def sample_random_tenant_node_sets(
    all_servers,
    *,
    num_tenants: int,
    ranks_per_tenant: int | None,
    rng: random.Random,
    contention: str = "low",
    allocation_mode: str = "balanced_remaining",
    protection_pool_size_mode: str = "high_resource",
    num_leaf: int | None = None,
    per_leaf_server: int | None = None,
) -> tuple[dict[int, tuple[int, ...]], tuple[int, ...]]:
    servers = [int(server) for server in all_servers]
    contention = str(contention).lower()

    if contention == "high":
        raise ValueError(
            "current repair experiments only support low-contention balanced allocation "
            "with topology-based protection modes"
        )

    if contention != "low":
        raise ValueError(f"unknown contention mode: {contention}")
    if num_leaf is None or per_leaf_server is None:
        raise ValueError("num_leaf and per_leaf_server are required for topology-based protection")

    if str(allocation_mode).lower() not in {"balanced_remaining", "balanced"}:
        raise ValueError(
            "low-contention allocation only supports balanced_remaining; "
            f"got {allocation_mode}"
        )
    return sample_balanced_low_contention_node_sets(
        servers,
        num_tenants=int(num_tenants),
        rng=rng,
        protection_pool_size_mode=protection_pool_size_mode,
        num_leaf=int(num_leaf),
        per_leaf_server=int(per_leaf_server),
    )


def sample_balanced_low_contention_node_sets(
    all_servers,
    *,
    num_tenants: int,
    rng: random.Random,
    protection_pool_size_mode: str = "high_resource",
    num_leaf: int | None = None,
    per_leaf_server: int | None = None,
) -> tuple[dict[int, tuple[int, ...]], tuple[int, ...]]:
    """Reserve one global protection pool, then balance all remaining servers."""

    servers = [int(server) for server in all_servers]
    if num_leaf is None or per_leaf_server is None:
        raise ValueError("num_leaf and per_leaf_server are required for topology-based protection")
    global_pool = _sample_topology_protection_pool(
        servers,
        num_leaf=int(num_leaf),
        per_leaf_server=int(per_leaf_server),
        rng=rng,
        mode=protection_pool_size_mode,
    )
    working_servers = [server for server in servers if server not in set(global_pool)]
    if len(working_servers) < int(num_tenants):
        raise ValueError("not enough working servers to give each tenant at least one rank")
    rng.shuffle(working_servers)

    base = len(working_servers) // int(num_tenants)
    rem = len(working_servers) % int(num_tenants)
    working: dict[int, tuple[int, ...]] = {}
    cursor = 0
    for tenant in range(int(num_tenants)):
        size = base + (1 if tenant < rem else 0)
        tenant_nodes = working_servers[cursor: cursor + size]
        cursor += size
        working[int(tenant)] = tuple(sorted(int(server) for server in tenant_nodes))
    _validate_full_server_partition(servers, working, global_pool)
    return working, global_pool


def _sample_topology_protection_pool(
    all_servers: list[int],
    *,
    num_leaf: int,
    per_leaf_server: int,
    rng: random.Random,
    mode: str,
) -> tuple[int, ...]:
    mode = str(mode).lower()
    if mode in {"high_resource", "high", "per_leaf"}:
        group_width = 1
    elif mode in {"low_resource", "low", "per_two_leaf", "per_2_leaf"}:
        group_width = 2
    else:
        raise ValueError(
            "protection_pool_size_mode must be high_resource or low_resource; "
            f"got {mode}"
        )

    expected_server_count = int(num_leaf) * int(per_leaf_server)
    server_set = set(int(server) for server in all_servers)
    if len(all_servers) != expected_server_count or server_set != set(range(expected_server_count)):
        raise ValueError(
            "topology-based protection expects contiguous server ids "
            f"0..{expected_server_count - 1}"
        )

    protection: list[int] = []
    for leaf_start in range(0, int(num_leaf), group_width):
        leaf_end = min(int(num_leaf), leaf_start + group_width)
        group_servers: list[int] = []
        for leaf in range(leaf_start, leaf_end):
            first = leaf * int(per_leaf_server)
            group_servers.extend(range(first, first + int(per_leaf_server)))
        protection.append(int(rng.choice(group_servers)))
    return tuple(sorted(protection))


def _validate_full_server_partition(
    all_servers,
    working_sets: dict[int, tuple[int, ...]],
    global_protection_pool: tuple[int, ...],
) -> None:
    expected = set(int(server) for server in all_servers)
    protection = [int(server) for server in global_protection_pool]
    if len(protection) != len(set(protection)):
        raise ValueError("global protection pool must not contain duplicate servers")

    working_servers: list[int] = []
    for tenant, tenant_servers in working_sets.items():
        tenant_list = [int(server) for server in tenant_servers]
        if len(tenant_list) != len(set(tenant_list)):
            raise ValueError(f"tenant {tenant} working set contains duplicate servers")
        working_servers.extend(tenant_list)

    duplicate_working = {
        server for server in working_servers if working_servers.count(server) > 1
    }
    if duplicate_working:
        raise ValueError(
            "working sets must be tenant-disjoint; duplicate servers: "
            f"{sorted(duplicate_working)}"
        )

    working = set(working_servers)
    protection_set = set(protection)
    overlap = working & protection_set
    if overlap:
        raise ValueError(
            "working sets and global protection pool must be disjoint; "
            f"overlap: {sorted(overlap)}"
        )

    assigned = working | protection_set
    missing = expected - assigned
    extra = assigned - expected
    if missing or extra:
        raise ValueError(
            "working sets plus global protection pool must partition all servers; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def solve_best_working_set_mappings(
    datacenter: LeafSpineDatacenter,
    *,
    initial_mapping: Mapping,
    rng: random.Random,
    seed: int,
    collective: str,
    single_flow_size_bits: int,
    workload_mode: str,
    time_limit: float | None,
    horizon_slots: int | None = None,
) -> tuple[Mapping, dict[int, TenantWorkingMappingResult], RepairWorkload]:
    del rng
    tenant_mapping = {
        int(tenant): {
            int(rank): int(server)
            for rank, server in rank_to_server.items()
        }
        for tenant, rank_to_server in initial_mapping.items()
    }
    low_contention = _load_low_contention_module()
    tenant_collective_specs, workload = _build_low_contention_workload(
        low_contention,
        tenant_mapping=tenant_mapping,
        seed=seed,
        collective=collective,
        single_flow_size_bits=single_flow_size_bits,
        workload_mode=workload_mode,
    )
    del horizon_slots
    original_time_limit = getattr(low_contention, "MAPPING_TIME_LIMIT_SECONDS", None)
    low_contention.MAPPING_TIME_LIMIT_SECONDS = time_limit
    try:
        solved_mapping, runtime_seconds = low_contention.run_mapping(
            datacenter,
            tenant_mapping,
            tenant_collective_specs,
        )
        mapping_makespan, mapping_avg_jct = low_contention.evaluate_collective(
            datacenter,
            solved_mapping,
            tenant_collective_specs,
        )
    finally:
        low_contention.MAPPING_TIME_LIMIT_SECONDS = original_time_limit
    results = {}
    for tenant in sorted(tenant_mapping):
        results[int(tenant)] = TenantWorkingMappingResult(
            tenant=int(tenant),
            working_nodes=tuple(sorted(int(server) for server in tenant_mapping[tenant].values())),
            initial_mapping={int(rank): int(server) for rank, server in tenant_mapping[tenant].items()},
            best_mapping={int(rank): int(server) for rank, server in solved_mapping[int(tenant)].items()},
            mapping_avg_jct=float(mapping_avg_jct),
            mapping_makespan=float(mapping_makespan),
            solver_name="MappingEstimatorBlackBoxOptimizer",
            solver_source="experiment/Low_contension.py::run_mapping",
            runtime_seconds=float(runtime_seconds),
        )
    return (
        {
            int(tenant): {
                int(rank): int(server)
                for rank, server in rank_to_server.items()
            }
            for tenant, rank_to_server in solved_mapping.items()
        },
        results,
        workload,
    )


def _load_low_contention_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "experiment" / "Low_contension.py"
    spec = importlib.util.spec_from_file_location("_failure_repair_low_contention", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Low_contension.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_low_contention_workload(
    low_contention,
    *,
    tenant_mapping: Mapping,
    seed: int,
    collective: str,
    single_flow_size_bits: int,
    workload_mode: str,
) -> tuple[dict[int, dict[str, object]], RepairWorkload]:
    mode = str(workload_mode).lower()
    if mode in {"synthetic", "synthetic_uniform", "uniform"}:
        tenant_collective_specs = {
            int(tenant): {
                "collective": collective,
                "single_flow_size_bits": int(single_flow_size_bits),
            }
            for tenant in tenant_mapping
        }
        return tenant_collective_specs, RepairWorkload(
            tenant_collective_specs=tenant_collective_specs,
            collective=collective,
            single_flow_size_bits=int(single_flow_size_bits),
            source="synthetic uniform collective specs",
        )

    if mode not in {"low_contention_dominant", "dominant", "trace_dominant"}:
        raise ValueError(f"unknown workload_mode: {workload_mode}")

    task_size_multiplier = int(getattr(low_contention, "TASK_SIZE_MULTIPLIER", 8))
    profiles = low_contention.load_dominant_profiles(task_size_multiplier)
    assignment_seed = low_contention.derive_seed(
        int(seed),
        "low_contention",
        "assignment",
        len(tenant_mapping),
    )
    assignment = low_contention.sample_tenant_workload_assignment(
        len(tenant_mapping),
        assignment_seed,
        profiles,
    )
    tenant_collective_specs = low_contention.build_tenant_collective_specs(
        assignment,
        tenant_mapping,
    )
    return tenant_collective_specs, RepairWorkload(
        tenant_collective_specs=tenant_collective_specs,
        source="experiment/Low_contension.py dominant trace-derived tenant_collective_specs",
    )


def run_repair_comparison(
    experiment: RandomRepairExperiment,
    *,
    reference_seeds_by_strategy: dict[str, list[tuple[Mapping, str]]] | None = None,
) -> RepairComparison:
    config = experiment.config
    problem = experiment.problem
    search = config.repair_search or RepairSearchConfig(
        beam_width=5,
        max_rounds=3,
        max_candidates_per_tenant=32,
        max_participating_tenants=config.num_tenants,
    )
    if not reference_seeds_by_strategy:
        strategy_results = solve_repair_strategies(
            problem,
            REPAIR_STRATEGIES,
            search_config=search,
            time_limit=config.repair_time_limit,
        )
    else:
        strategy_results = {}
        failover_reference = None
        for strategy in REPAIR_STRATEGIES:
            result = solve_repair_strategy(
                problem,
                strategy,
                search_config=search,
                time_limit=config.repair_time_limit,
                failover_mapping=failover_reference,
                reference_seeds=reference_seeds_by_strategy.get(strategy.name, []),
            )
            strategy_results[strategy.name] = result
            if strategy.name == REPAIR_FAILED_SERVER_ONLY.name:
                failover_reference = result.mapping
    failover_result = strategy_results[REPAIR_FAILED_SERVER_ONLY.name]
    local_result = strategy_results[TENANT_LOCAL_REPAIR.name]
    cooperative_result = strategy_results[COOPERATIVE_REPAIR.name]
    return RepairComparison(
        experiment=experiment,
        strategies=REPAIR_STRATEGIES,
        strategy_results=strategy_results,
        failover_scenario=failover_result.scenario,
        tenant_local_scenario=local_result.scenario,
        cooperative_scenario=cooperative_result.scenario,
        failover_mapping=failover_result.mapping,
        failover_objective=failover_result.objective,
        tenant_local=_legacy_repair_result(local_result),
        cooperative=_legacy_repair_result(cooperative_result),
    )


def find_random_repair_gain_case(
    base_config: RandomRepairExperimentConfig,
    *,
    max_trials: int = 30,
) -> RepairComparison:
    best: RepairComparison | None = None
    best_key = None
    for offset in range(int(max_trials)):
        config = RandomRepairExperimentConfig(
            **{
                **base_config.__dict__,
                "seed": int(base_config.seed) + offset,
            }
        )
        comparison = run_repair_comparison(generate_random_repair_experiment(config))
        if comparison.has_strict_repair_gain():
            return comparison
        key = (
            repair_sort_key(comparison.cooperative.objective),
            repair_sort_key(comparison.tenant_local.objective),
            comparison.failover_objective.avg_jct,
        )
        if best is None or key < best_key:
            best = comparison
            best_key = key
    if best is None:
        raise RuntimeError("no random repair trial was evaluated")
    raise RuntimeError(
        "no random trial satisfied failover > tenant-local > cooperative within "
        f"{max_trials} seeds; best seed was {best.experiment.config.seed}"
    )


def _legacy_repair_result(strategy_result: StrategySolveResult) -> RepairResult:
    return RepairResult(
        name=strategy_result.strategy.name,
        mapping=strategy_result.mapping,
        objective=strategy_result.objective,
        switch_counts=strategy_result.switch_counts,
        runtime_seconds=strategy_result.runtime_seconds,
        metadata=dict(strategy_result.metadata),
    )


def repair_comparison_payload(comparison: RepairComparison) -> dict[str, object]:
    experiment = comparison.experiment
    config = experiment.config
    return {
        "seed": config.seed,
        "topology": {
            "num_leaf": config.num_leaf,
            "num_spine": config.num_spine,
            "per_leaf_server": config.per_leaf_server,
        },
        "num_tenants": config.num_tenants,
        "ranks_per_tenant": config.ranks_per_tenant,
        "working_allocation_mode": config.working_allocation_mode,
        "protection_pool_size_mode": config.protection_pool_size_mode,
        "contention": config.contention,
        "workload_mode": config.workload_mode,
        "collective": config.collective,
        "single_flow_size_bits": config.single_flow_size_bits,
        "failover_policy": config.failover_policy,
        "failure_selection": config.failure_selection,
        "failure_selection_metadata": dict(experiment.failure_selection_metadata or {}),
        "workload": _workload_payload(experiment.workload),
        "failure_aware_mapping_problem": experiment.problem.payload(),
        "strategies": {
            strategy.name: strategy.payload()
            for strategy in comparison.strategies
        },
        "working_set_best_mappings": {
            str(tenant): {
                "working_nodes": list(result.working_nodes),
                "initial_random_mapping": {str(rank): server for rank, server in result.initial_mapping.items()},
                "best_mapping": {str(rank): server for rank, server in result.best_mapping.items()},
                "mapping_avg_jct": result.mapping_avg_jct,
                "mapping_makespan": result.mapping_makespan,
                "solver": result.solver_name,
                "solver_source": result.solver_source,
                "runtime_seconds": result.runtime_seconds,
            }
            for tenant, result in experiment.working_results.items()
        },
        "working_nodes": {
            str(tenant): sorted(int(server) for server in ranks.values())
            for tenant, ranks in experiment.pre_failure_mapping.items()
        },
        "global_protection_pool": list(experiment.global_protection_pool),
        "repair_dag": _dag_summary_from_workload(experiment),
        "strategy_constraints": _strategy_constraints_payload(comparison),
        "pre_failure_mapping": normalize_mapping_for_json(experiment.pre_failure_mapping),
        "failure": {
            "tenant": experiment.failure.tenant,
            "failed_rank": experiment.failure.failed_rank,
            "failed_server": experiment.failure.failed_server,
        },
        "results": _strategy_results_payload(comparison),
    }


def _strategy_results_payload(comparison: RepairComparison) -> dict[str, object]:
    return {
        strategy.name: _strategy_result_payload(comparison.strategy_results[strategy.name])
        for strategy in comparison.strategies
    }


def _strategy_result_payload(result: StrategySolveResult) -> dict[str, object]:
    sim_makespan = float(result.objective.makespan)
    sim_avg_jct = float(result.objective.avg_jct)
    return {
        "mapping": normalize_mapping_for_json(result.mapping),
        "objective": result.objective.__dict__,
        "simulation": {
            "makespan": float(sim_makespan),
            "avg_jct": float(sim_avg_jct),
        },
        "switch_counts": result.switch_counts.__dict__,
        "runtime_seconds": result.runtime_seconds,
        "metadata": result.metadata,
    }


def _strategy_constraints_payload(comparison: RepairComparison) -> dict[str, object]:
    problem = comparison.experiment.problem
    return {
        strategy.name: problem.constraints_payload(
            strategy,
            failover_mapping=comparison.failover_mapping,
        )
        for strategy in comparison.strategies
    }


def _workload_payload(workload: RepairWorkload) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": workload.source,
        "collective": workload.collective,
        "single_flow_size_bits": workload.single_flow_size_bits,
    }
    if workload.tenant_collective_specs is not None:
        payload["tenant_collective_specs"] = {
            str(tenant): dict(spec)
            for tenant, spec in workload.tenant_collective_specs.items()
        }
    if workload.tenant_collective_programs is not None:
        payload["tenant_collective_programs"] = {
            str(tenant): [dict(op) for op in program]
            for tenant, program in workload.tenant_collective_programs.items()
        }
    if workload.tenant_start_times is not None:
        payload["tenant_start_times"] = {
            str(tenant): float(start_time)
            for tenant, start_time in workload.tenant_start_times.items()
        }
    return payload


def _dag_summary_from_workload(experiment: RandomRepairExperiment) -> dict[str, object]:
    return experiment.problem.evaluator(TENANT_LOCAL_REPAIR).dag_summary()
