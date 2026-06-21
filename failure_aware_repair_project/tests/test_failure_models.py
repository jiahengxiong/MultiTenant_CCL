from __future__ import annotations

import pytest

from multitenant.topology import LeafSpineDatacenter

from failure_aware_repair.models import FailureEvent, RepairScenario
from failure_aware_repair.protection import (
    build_candidate_server_sets,
    build_failover_mapping,
    check_repair_feasible,
    count_switches,
    infer_failed_rank,
)


def test_failover_mapping_changes_only_failed_rank():
    datacenter = LeafSpineDatacenter(num_leaf=4, num_spine=2, per_leaf_server=6)
    mapping = {0: {0: 0, 1: 1}, 1: {0: 2, 1: 3}}
    global_pool = (20, 21)
    scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1),
        mode="tenant_local",
        global_protection_pool=global_pool,
    )

    failover = build_failover_mapping(scenario, datacenter=datacenter)

    assert infer_failed_rank(mapping, scenario.failure) == 1
    assert failover[0][0] == mapping[0][0]
    assert failover[0][1] in global_pool
    assert failover[1] == mapping[1]
    check_repair_feasible(scenario, failover)


def test_candidate_sets_respect_tenant_local_and_cooperative_modes():
    mapping = {0: {0: 0, 1: 1}, 1: {0: 2, 1: 3}}
    global_pool = (20, 21)
    local = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode="tenant_local",
        global_protection_pool=global_pool,
    )
    cooperative = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode="cooperative",
        global_protection_pool=global_pool,
    )

    local_sets = build_candidate_server_sets(local)
    coop_sets = build_candidate_server_sets(cooperative)

    assert 1 not in local_sets[0]
    assert set(local_sets[1]) == {2, 3}
    assert 1 not in coop_sets[0]
    assert set(global_pool).issubset(set(coop_sets[0]))
    assert set(global_pool).issubset(set(coop_sets[1]))


def test_global_protection_pool_rejects_working_overlap():
    scenario = RepairScenario(
        pre_failure_mapping={0: {0: 0, 1: 1}},
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode="tenant_local",
        global_protection_pool=(0, 5),
    )

    with pytest.raises(ValueError, match="disjoint"):
        build_failover_mapping(scenario)


def test_switch_count_separates_unavoidable_failover_from_extra_moves():
    mapping = {0: {0: 0, 1: 1}}
    failover = {0: {0: 0, 1: 5}}
    repaired = {0: {0: 6, 1: 5}}
    counts = count_switches(
        pre_failure_mapping=mapping,
        failover_mapping=failover,
        repaired_mapping=repaired,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
    )
    assert counts.total_vs_prefailure == 2
    assert counts.extra_vs_failover == 1
    assert counts.extra_moved_ranks_vs_failover == ((0, 0),)


def test_feasibility_rejects_failed_server_use():
    mapping = {0: {0: 0, 1: 1}}
    scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode="tenant_local",
        global_protection_pool=(4, 5),
    )
    failover = build_failover_mapping(scenario)
    bad = {0: {0: 0, 1: 1}}
    with pytest.raises(ValueError, match="failed server"):
        check_repair_feasible(scenario, bad, failover)


def test_feasibility_rejects_shared_protection_server():
    mapping = {0: {0: 0, 1: 1}, 1: {0: 2, 1: 3}}
    scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode="cooperative",
        global_protection_pool=(4, 5),
    )
    failover = build_failover_mapping(scenario, policy="first")
    bad = {
        0: {0: 0, 1: 4},
        1: {0: 4, 1: 3},
    }
    with pytest.raises(ValueError, match="same physical server 4"):
        check_repair_feasible(scenario, bad, failover)


def test_feasibility_rejects_same_tenant_shared_protection_server():
    mapping = {0: {0: 0, 1: 1, 2: 2}}
    scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode="tenant_local",
        global_protection_pool=(4, 5),
    )
    failover = build_failover_mapping(scenario, policy="first")
    bad = {
        0: {
            0: 4,
            1: failover[0][1],
            2: 4,
        }
    }
    with pytest.raises(ValueError, match="multiple ranks to the same active server"):
        check_repair_feasible(scenario, bad, failover)


def test_feasibility_allows_local_to_choose_different_protection_than_failover():
    mapping = {0: {0: 0, 1: 1, 2: 2}}
    scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode="tenant_local",
        global_protection_pool=(4, 5),
    )
    failover = build_failover_mapping(scenario, policy="first")
    repaired = {
        0: {
            0: 0,
            1: 5,
            2: 2,
        }
    }

    assert failover[0][1] == 4
    check_repair_feasible(scenario, repaired, failover)


def test_feasibility_rejects_remapping_to_another_working_server():
    mapping = {0: {0: 0, 1: 1, 2: 2}}
    scenario = RepairScenario(
        pre_failure_mapping=mapping,
        failure=FailureEvent(tenant=0, failed_server=1, failed_rank=1),
        mode="tenant_local",
        global_protection_pool=(4, 5),
    )
    failover = build_failover_mapping(scenario, policy="first")
    bad = {
        0: {
            0: 2,
            1: failover[0][1],
            2: 5,
        }
    }
    with pytest.raises(ValueError, match="pre-failure working"):
        check_repair_feasible(scenario, bad, failover)
