from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Mapping = dict[int, dict[int, int]]
RepairMode = Literal["tenant_local", "cooperative"]


@dataclass(frozen=True)
class FailureEvent:
    """A single working-node failure."""

    tenant: int
    failed_server: int
    failed_rank: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant", int(self.tenant))
        object.__setattr__(self, "failed_server", int(self.failed_server))
        if self.failed_rank is not None:
            object.__setattr__(self, "failed_rank", int(self.failed_rank))


@dataclass(frozen=True)
class RepairWorkload:
    """Original mapping workload carried into a post-failure repair scenario."""

    tenant_collective_specs: dict[int, dict[str, object]] | None = None
    tenant_collective_programs: dict[int, list[dict[str, object]]] | None = None
    tenant_start_times: dict[int, float] | None = None
    collective: str | None = None
    single_flow_size_bits: int | None = None
    source: str = "repair_workload"

    def __post_init__(self) -> None:
        if self.tenant_collective_specs is not None:
            specs = {
                int(tenant): dict(spec)
                for tenant, spec in self.tenant_collective_specs.items()
            }
            object.__setattr__(self, "tenant_collective_specs", specs)
        if self.tenant_collective_programs is not None:
            programs = {
                int(tenant): [dict(op) for op in program]
                for tenant, program in self.tenant_collective_programs.items()
            }
            object.__setattr__(self, "tenant_collective_programs", programs)
        if self.tenant_start_times is not None:
            start_times = {
                int(tenant): float(start_time)
                for tenant, start_time in self.tenant_start_times.items()
            }
            object.__setattr__(self, "tenant_start_times", start_times)
        if self.single_flow_size_bits is not None:
            object.__setattr__(self, "single_flow_size_bits", int(self.single_flow_size_bits))
        if self.collective is not None:
            object.__setattr__(self, "collective", str(self.collective))


@dataclass(frozen=True)
class RepairScenario:
    """Complete repair input for one failure event."""

    pre_failure_mapping: Mapping
    failure: FailureEvent
    mode: RepairMode
    global_protection_pool: tuple[int, ...]
    participating_tenants: tuple[int, ...] | None = None
    workload: RepairWorkload | None = None

    def __post_init__(self) -> None:
        mapping = {
            int(tenant): {int(rank): int(server) for rank, server in ranks.items()}
            for tenant, ranks in self.pre_failure_mapping.items()
        }
        object.__setattr__(self, "pre_failure_mapping", mapping)
        pool = tuple(int(server) for server in self.global_protection_pool)
        if len(set(pool)) != len(pool):
            raise ValueError("global_protection_pool must be unique")
        if len(pool) < 2:
            raise ValueError("global_protection_pool must contain at least 2 servers")
        object.__setattr__(self, "global_protection_pool", pool)
        if self.participating_tenants is not None:
            object.__setattr__(
                self,
                "participating_tenants",
                tuple(int(tenant) for tenant in self.participating_tenants),
            )


@dataclass(frozen=True, order=True)
class RepairObjective:
    """Lexicographic repair objective: Avg JCT, makespan, then switch count."""

    avg_jct: float
    makespan: float
    extra_switches: int

    @classmethod
    def infinity(cls) -> "RepairObjective":
        return cls(float("inf"), float("inf"), 10**18)


@dataclass(frozen=True)
class SwitchCounts:
    """Operational movement metrics for a repaired mapping."""

    total_vs_prefailure: int
    extra_vs_failover: int
    moved_ranks_vs_prefailure: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    extra_moved_ranks_vs_failover: tuple[tuple[int, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RepairResult:
    """Result returned by a repair algorithm."""

    name: str
    mapping: Mapping
    objective: RepairObjective
    switch_counts: SwitchCounts
    runtime_seconds: float
    metadata: dict[str, object] = field(default_factory=dict)
