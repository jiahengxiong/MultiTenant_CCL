from __future__ import annotations

from collections.abc import Iterable

from .models import FailureEvent, Mapping, RepairScenario, SwitchCounts


def copy_mapping(mapping: Mapping) -> Mapping:
    return {
        int(tenant): {int(rank): int(server) for rank, server in ranks.items()}
        for tenant, ranks in mapping.items()
    }


def infer_failed_rank(mapping: Mapping, failure: FailureEvent) -> int:
    if failure.failed_rank is not None:
        expected = mapping[int(failure.tenant)].get(int(failure.failed_rank))
        if expected != int(failure.failed_server):
            raise ValueError(
                "failure.failed_rank does not map to failure.failed_server "
                f"({failure.failed_rank=} {failure.failed_server=} {expected=})"
            )
        return int(failure.failed_rank)

    for rank, server in mapping[int(failure.tenant)].items():
        if int(server) == int(failure.failed_server):
            return int(rank)
    raise ValueError(
        f"failed_server {failure.failed_server} is not used by tenant {failure.tenant}"
    )


def validate_global_protection_pool(
    mapping: Mapping,
    global_protection_pool: Iterable[int],
) -> None:
    pool = tuple(int(server) for server in global_protection_pool)
    if len(set(pool)) != len(pool):
        raise ValueError("global_protection_pool must be unique")
    if len(pool) < 2:
        raise ValueError("global_protection_pool must contain at least 2 servers")
    mapped_all = {
        int(server)
        for rank_to_server in mapping.values()
        for server in rank_to_server.values()
    }
    if mapped_all & set(pool):
        raise ValueError("global_protection_pool must be disjoint from working servers")


def working_nodes_by_tenant(mapping: Mapping) -> dict[int, tuple[int, ...]]:
    return {
        int(tenant): tuple(sorted(int(server) for server in rank_to_server.values()))
        for tenant, rank_to_server in mapping.items()
    }


def _server_leaf(datacenter, server: int) -> int | None:
    if datacenter is None or not hasattr(datacenter, "get_server_leaf"):
        return None
    try:
        return int(datacenter.get_server_leaf(int(server)))
    except Exception:
        return None


def choose_failover_protection(
    protection_nodes: Iterable[int],
    *,
    failed_server: int,
    datacenter=None,
    policy: str = "same_leaf_or_nearest",
) -> int:
    nodes = sorted(int(node) for node in protection_nodes)
    if not nodes:
        raise ValueError("protection_nodes cannot be empty")
    if policy == "first":
        return nodes[0]
    if policy != "same_leaf_or_nearest":
        raise ValueError(f"unknown failover policy: {policy}")

    failed_leaf = _server_leaf(datacenter, int(failed_server))
    if failed_leaf is None:
        return nodes[0]
    return min(
        nodes,
        key=lambda server: (
            _server_leaf(datacenter, server) != failed_leaf,
            abs(server - int(failed_server)),
            server,
        ),
    )


def build_failover_mapping(
    scenario: RepairScenario,
    *,
    datacenter=None,
    policy: str = "same_leaf_or_nearest",
) -> Mapping:
    validate_global_protection_pool(scenario.pre_failure_mapping, scenario.global_protection_pool)
    mapping = copy_mapping(scenario.pre_failure_mapping)
    failure = scenario.failure
    failed_rank = infer_failed_rank(mapping, failure)
    protection = scenario.global_protection_pool
    replacement = choose_failover_protection(
        protection,
        failed_server=int(failure.failed_server),
        datacenter=datacenter,
        policy=policy,
    )
    mapping[int(failure.tenant)][int(failed_rank)] = int(replacement)
    return mapping


def participating_tenants(scenario: RepairScenario) -> tuple[int, ...]:
    if scenario.mode == "tenant_local":
        return (int(scenario.failure.tenant),)
    if scenario.participating_tenants is not None:
        tenants = tuple(sorted(set(int(t) for t in scenario.participating_tenants)))
        if int(scenario.failure.tenant) not in tenants:
            tenants = tuple(sorted((*tenants, int(scenario.failure.tenant))))
        return tenants
    return tuple(sorted(int(tenant) for tenant in scenario.pre_failure_mapping))


def build_candidate_server_sets(scenario: RepairScenario) -> dict[int, tuple[int, ...]]:
    validate_global_protection_pool(scenario.pre_failure_mapping, scenario.global_protection_pool)
    participants = set(participating_tenants(scenario))
    failure = scenario.failure
    candidate_sets: dict[int, tuple[int, ...]] = {}
    working_by_tenant = working_nodes_by_tenant(scenario.pre_failure_mapping)

    for tenant, ranks in scenario.pre_failure_mapping.items():
        tenant = int(tenant)
        if tenant not in participants:
            candidate_sets[tenant] = tuple(sorted(int(server) for server in ranks.values()))
            continue

        working = set(working_by_tenant[tenant])
        if tenant == int(failure.tenant):
            working.discard(int(failure.failed_server))
        protection = set(int(server) for server in scenario.global_protection_pool)
        candidate_sets[tenant] = tuple(sorted(working | protection))
    return candidate_sets


def repair_strategy_constraints(
    scenario: RepairScenario,
    *,
    strategy_name: str | None = None,
    strategy_spec=None,
    failover_mapping: Mapping | None = None,
) -> dict[str, object]:
    """Summarize the mapping variables and fixed constraints for one strategy."""

    validate_global_protection_pool(scenario.pre_failure_mapping, scenario.global_protection_pool)
    participants = set(participating_tenants(scenario))
    candidate_sets = build_candidate_server_sets(scenario)
    failed_rank = infer_failed_rank(scenario.pre_failure_mapping, scenario.failure)
    tenants = tuple(sorted(int(tenant) for tenant in scenario.pre_failure_mapping))
    fixed_tenants = tuple(int(tenant) for tenant in tenants if int(tenant) not in participants)
    strategy = strategy_name or (
        "tenant_local_repair" if scenario.mode == "tenant_local" else "cooperative_repair"
    )
    objective_order = (
        list(strategy_spec.objective_order)
        if strategy_spec is not None
        else ["avg_jct", "makespan", "extra_switches"]
    )
    return {
        "strategy": strategy,
        "strategy_spec": strategy_spec.payload() if strategy_spec is not None else None,
        "mode": scenario.mode,
        "failed_tenant": int(scenario.failure.tenant),
        "failed_rank": int(failed_rank),
        "failed_server": int(scenario.failure.failed_server),
        "participating_tenants": [int(tenant) for tenant in sorted(participants)],
        "fixed_tenants": [int(tenant) for tenant in fixed_tenants],
        "global_protection_pool": [int(server) for server in scenario.global_protection_pool],
        "candidate_server_sets": {
            str(tenant): [int(server) for server in servers]
            for tenant, servers in sorted(candidate_sets.items())
        },
        "non_participating_tenants_fixed_to_failover": bool(fixed_tenants),
        "failed_server_excluded_for_failed_tenant": (
            int(scenario.failure.failed_server) not in candidate_sets[int(scenario.failure.tenant)]
        ),
        "healthy_rank_switches_minimized": True,
        "objective_order": objective_order,
        "failover_reference_mapping": (
            normalize_mapping_for_json(failover_mapping)
            if failover_mapping is not None
            else None
        ),
    }


def failover_strategy_constraints(
    scenario: RepairScenario,
    failover_mapping: Mapping,
    *,
    strategy_spec=None,
) -> dict[str, object]:
    """Summarize the fixed failed-rank-only replacement baseline."""

    failed_rank = infer_failed_rank(scenario.pre_failure_mapping, scenario.failure)
    tenants = tuple(sorted(int(tenant) for tenant in scenario.pre_failure_mapping))
    return {
        "strategy": "repair_failed_server_only",
        "strategy_spec": strategy_spec.payload() if strategy_spec is not None else None,
        "mode": "failover_only",
        "failed_tenant": int(scenario.failure.tenant),
        "failed_rank": int(failed_rank),
        "failed_server": int(scenario.failure.failed_server),
        "participating_tenants": [int(scenario.failure.tenant)],
        "fixed_tenants": [
            int(tenant)
            for tenant in tenants
            if int(tenant) != int(scenario.failure.tenant)
        ],
        "global_protection_pool": [int(server) for server in scenario.global_protection_pool],
        "candidate_server_sets": {
            str(tenant): [int(server) for server in sorted(ranks.values())]
            for tenant, ranks in sorted(scenario.pre_failure_mapping.items())
        },
        "only_failed_rank_replaced": True,
        "healthy_ranks_fixed_to_prefailure": True,
        "replacement_server": int(failover_mapping[int(scenario.failure.tenant)][int(failed_rank)]),
        "healthy_rank_switches_minimized": False,
        "objective_order": (
            list(strategy_spec.objective_order)
            if strategy_spec is not None
            else ["avg_jct", "makespan"]
        ),
        "failover_reference_mapping": normalize_mapping_for_json(failover_mapping),
    }


def count_switches(
    *,
    pre_failure_mapping: Mapping,
    failover_mapping: Mapping,
    repaired_mapping: Mapping,
    failure: FailureEvent,
) -> SwitchCounts:
    failed_key = (int(failure.tenant), infer_failed_rank(pre_failure_mapping, failure))
    moved_total: list[tuple[int, int]] = []
    moved_extra: list[tuple[int, int]] = []

    for tenant in sorted(pre_failure_mapping):
        for rank in sorted(pre_failure_mapping[tenant]):
            key = (int(tenant), int(rank))
            if int(repaired_mapping[tenant][rank]) != int(pre_failure_mapping[tenant][rank]):
                moved_total.append(key)
            if key == failed_key:
                continue
            if int(repaired_mapping[tenant][rank]) != int(failover_mapping[tenant][rank]):
                moved_extra.append(key)

    return SwitchCounts(
        total_vs_prefailure=len(moved_total),
        extra_vs_failover=len(moved_extra),
        moved_ranks_vs_prefailure=tuple(moved_total),
        extra_moved_ranks_vs_failover=tuple(moved_extra),
    )


def check_repair_feasible(
    scenario: RepairScenario,
    mapping: Mapping,
    failover_mapping: Mapping | None = None,
) -> None:
    candidate_sets = build_candidate_server_sets(scenario)
    participants = set(participating_tenants(scenario))
    failure = scenario.failure
    failed_rank = infer_failed_rank(scenario.pre_failure_mapping, failure)
    protection = set(int(server) for server in scenario.global_protection_pool)
    active_server_owner: dict[int, tuple[int, int]] = {}

    for tenant, ranks in scenario.pre_failure_mapping.items():
        tenant = int(tenant)
        if set(int(rank) for rank in mapping.get(tenant, {})) != set(int(rank) for rank in ranks):
            raise ValueError(f"tenant {tenant} rank set changed")
        values = [int(server) for server in mapping[tenant].values()]
        if len(set(values)) != len(values):
            raise ValueError(f"tenant {tenant} maps multiple ranks to the same active server")
        if tenant == int(failure.tenant) and int(failure.failed_server) in values:
            raise ValueError("failed server is still used")
        if tenant == int(failure.tenant):
            failed_rank_server = int(mapping[tenant][int(failed_rank)])
            if failed_rank_server not in protection:
                raise ValueError(
                    "failed rank must occupy exactly one protection server; "
                    f"tenant {tenant} rank {int(failed_rank)} uses {failed_rank_server}"
                )
        for rank, server in mapping[tenant].items():
            rank = int(rank)
            server = int(server)
            owner = active_server_owner.get(server)
            if owner is not None:
                raise ValueError(
                    "repair maps multiple active ranks to the same physical "
                    f"server {server}: tenant {owner[0]} rank {owner[1]} and "
                    f"tenant {tenant} rank {rank}"
                )
            active_server_owner[server] = (tenant, rank)
        disallowed = set(values) - set(candidate_sets[tenant])
        if disallowed:
            raise ValueError(f"tenant {tenant} uses disallowed servers: {sorted(disallowed)}")
        for rank, server in mapping[tenant].items():
            rank = int(rank)
            server = int(server)
            original_server = int(scenario.pre_failure_mapping[tenant][rank])
            if server != original_server and server not in protection:
                raise ValueError(
                    "repair may only keep a rank on its pre-failure working "
                    f"server or switch it to protection; tenant {tenant} "
                    f"rank {rank} uses {server}"
                )
        if tenant not in participants and mapping[tenant] != scenario.pre_failure_mapping[tenant]:
            raise ValueError(f"non-participating tenant {tenant} changed")


def normalize_mapping_for_json(mapping: Mapping) -> dict[str, dict[str, int]]:
    return {
        str(tenant): {str(rank): int(server) for rank, server in ranks.items()}
        for tenant, ranks in mapping.items()
    }
