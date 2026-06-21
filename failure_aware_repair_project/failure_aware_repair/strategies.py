from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import FailureEvent, Mapping, RepairMode, RepairScenario, RepairWorkload

RepairStrategyMode = Literal["failover_only", "tenant_local", "cooperative"]
ParticipantScope = Literal["failed_tenant", "all_tenants"]


@dataclass(frozen=True)
class RepairStrategySpec:
    """One protection strategy as a constrained post-failure mapping problem."""

    name: str
    mode: RepairStrategyMode
    participant_scope: ParticipantScope
    optimizes_mapping: bool
    objective_order: tuple[str, ...]
    description: str

    def participating_tenants(self, mapping: Mapping, failure: FailureEvent) -> tuple[int, ...]:
        if self.participant_scope == "failed_tenant":
            return (int(failure.tenant),)
        if self.participant_scope == "all_tenants":
            return tuple(sorted(int(tenant) for tenant in mapping))
        raise ValueError(f"unknown participant scope: {self.participant_scope}")

    def scenario_mode(self) -> RepairMode:
        if self.mode == "cooperative":
            return "cooperative"
        return "tenant_local"

    def build_scenario(
        self,
        *,
        pre_failure_mapping: Mapping,
        failure: FailureEvent,
        global_protection_pool: tuple[int, ...],
        workload: RepairWorkload | None,
    ) -> RepairScenario:
        return RepairScenario(
            pre_failure_mapping,
            failure,
            self.scenario_mode(),
            global_protection_pool=global_protection_pool,
            participating_tenants=self.participating_tenants(pre_failure_mapping, failure),
            workload=workload,
        )

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "participant_scope": self.participant_scope,
            "optimizes_mapping": self.optimizes_mapping,
            "objective_order": list(self.objective_order),
            "description": self.description,
        }


REPAIR_FAILED_SERVER_ONLY = RepairStrategySpec(
    name="repair_failed_server_only",
    mode="failover_only",
    participant_scope="failed_tenant",
    optimizes_mapping=True,
    objective_order=("avg_jct", "makespan"),
    description=(
        "Optimize the failed rank's replacement over the global protection "
        "pool; keep healthy ranks fixed."
    ),
)

TENANT_LOCAL_REPAIR = RepairStrategySpec(
    name="tenant_local_repair",
    mode="tenant_local",
    participant_scope="failed_tenant",
    optimizes_mapping=True,
    objective_order=("avg_jct", "makespan", "extra_switches"),
    description="Allow only the failed tenant to remap onto healthy working nodes plus the global protection pool.",
)

COOPERATIVE_REPAIR = RepairStrategySpec(
    name="cooperative_repair",
    mode="cooperative",
    participant_scope="all_tenants",
    optimizes_mapping=True,
    objective_order=("avg_jct", "makespan", "extra_switches"),
    description="Allow all tenants to cooperatively remap onto healthy working nodes plus the global protection pool.",
)

REPAIR_STRATEGIES = (
    REPAIR_FAILED_SERVER_ONLY,
    TENANT_LOCAL_REPAIR,
    COOPERATIVE_REPAIR,
)

REPAIR_STRATEGY_BY_NAME = {
    strategy.name: strategy
    for strategy in REPAIR_STRATEGIES
}
