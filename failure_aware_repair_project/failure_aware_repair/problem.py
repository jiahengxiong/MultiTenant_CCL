from __future__ import annotations

from dataclasses import dataclass

from .evaluator import RepairEvaluator
from .models import FailureEvent, Mapping, RepairScenario, RepairWorkload
from .protection import (
    build_failover_mapping,
    failover_strategy_constraints,
    repair_strategy_constraints,
)
from .strategies import REPAIR_FAILED_SERVER_ONLY, RepairStrategySpec


@dataclass(frozen=True)
class FailureAwareMappingProblem:
    """Fixed post-failure mapping input shared by all protection strategies."""

    datacenter: object
    pre_failure_mapping: Mapping
    failure: FailureEvent
    global_protection_pool: tuple[int, ...]
    workload: RepairWorkload
    horizon_slots: int | None = None
    failover_policy: str = "same_leaf_or_nearest"

    def scenario(self, strategy: RepairStrategySpec) -> RepairScenario:
        return strategy.build_scenario(
            pre_failure_mapping=self.pre_failure_mapping,
            failure=self.failure,
            global_protection_pool=self.global_protection_pool,
            workload=self.workload,
        )

    def evaluator(
        self,
        strategy: RepairStrategySpec,
        *,
        failover_mapping: Mapping | None = None,
    ) -> RepairEvaluator:
        return RepairEvaluator(
            self.datacenter,
            self.scenario(strategy),
            horizon_slots=self.horizon_slots,
            failover_policy=self.failover_policy,
            failover_mapping=failover_mapping,
        )

    def failover_mapping(self) -> Mapping:
        return build_failover_mapping(
            self.scenario(REPAIR_FAILED_SERVER_ONLY),
            datacenter=self.datacenter,
            policy=self.failover_policy,
        )

    def constraints_payload(
        self,
        strategy: RepairStrategySpec,
        *,
        failover_mapping: Mapping | None = None,
    ) -> dict[str, object]:
        scenario = self.scenario(strategy)
        if strategy.name == REPAIR_FAILED_SERVER_ONLY.name:
            if failover_mapping is None:
                failover_mapping = self.failover_mapping()
            return failover_strategy_constraints(
                scenario,
                failover_mapping,
                strategy_spec=strategy,
            )
        return repair_strategy_constraints(
            scenario,
            strategy_name=strategy.name,
            strategy_spec=strategy,
            failover_mapping=failover_mapping,
        )

    def payload(self) -> dict[str, object]:
        return {
            "failed_tenant": int(self.failure.tenant),
            "failed_rank": (
                int(self.failure.failed_rank)
                if self.failure.failed_rank is not None
                else None
            ),
            "failed_server": int(self.failure.failed_server),
            "global_protection_pool": [int(server) for server in self.global_protection_pool],
            "tenant_count": len(self.pre_failure_mapping),
            "workload_source": self.workload.source,
        }
