from __future__ import annotations

from dataclasses import dataclass

from ..models import FailureEvent, Mapping, RepairScenario
from ..protection import (
    choose_failover_protection,
    copy_mapping,
    infer_failed_rank,
    validate_global_protection_pool,
)


@dataclass(frozen=True)
class NearestProtectionBaselineResult:
    """Mapping produced by the nearest-protection failover baseline."""

    mapping: Mapping
    failed_rank: int
    replacement_server: int
    replacement_leaf: int | None


class NearestProtectionBaselineSolver:
    """Move only the failed rank to the nearest protection server.

    This is intentionally a non-optimizing baseline: it does not search over
    healthy ranks and does not change any tenant other than the failed rank's
    current placement.
    """

    name = "baseline_nearest_protection"

    def __init__(self, scenario: RepairScenario, *, datacenter=None):
        self.scenario = scenario
        self.datacenter = datacenter

    def solve(self) -> NearestProtectionBaselineResult:
        validate_global_protection_pool(
            self.scenario.pre_failure_mapping,
            self.scenario.global_protection_pool,
        )
        failure: FailureEvent = self.scenario.failure
        mapping = copy_mapping(self.scenario.pre_failure_mapping)
        failed_rank = infer_failed_rank(mapping, failure)
        replacement = choose_failover_protection(
            self.scenario.global_protection_pool,
            failed_server=int(failure.failed_server),
            datacenter=self.datacenter,
            policy="same_leaf_or_nearest",
        )
        mapping[int(failure.tenant)][int(failed_rank)] = int(replacement)
        replacement_leaf = None
        if self.datacenter is not None and hasattr(self.datacenter, "get_server_leaf"):
            replacement_leaf = int(self.datacenter.get_server_leaf(int(replacement)))
        return NearestProtectionBaselineResult(
            mapping=mapping,
            failed_rank=int(failed_rank),
            replacement_server=int(replacement),
            replacement_leaf=replacement_leaf,
        )


def solve_nearest_protection_baseline(
    scenario: RepairScenario,
    *,
    datacenter=None,
) -> NearestProtectionBaselineResult:
    return NearestProtectionBaselineSolver(
        scenario,
        datacenter=datacenter,
    ).solve()

