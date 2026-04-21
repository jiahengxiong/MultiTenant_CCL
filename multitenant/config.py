from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CollectiveType = Literal["allgather", "allreduce"]

BITS_PER_MB = 1024 * 1024 * 8


@dataclass(frozen=True)
class TopologyConfig:
    num_leaf: int = 4
    num_spine: int = 2
    servers_per_leaf: int = 8


@dataclass(frozen=True)
class ExperimentConfig:
    num_tenants: int = 3
    num_experiments: int = 10
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    single_flow_size_bits: int = 8 * BITS_PER_MB
    collective: CollectiveType = "allgather"
