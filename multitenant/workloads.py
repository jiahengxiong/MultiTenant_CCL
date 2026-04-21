from __future__ import annotations

import random
from typing import Sequence


def _collective_size_factor(collective: str) -> int:
    return 2 if collective == "allreduce" else 1


def build_random_tenant_mapping(
    all_servers: Sequence[int],
    num_tenants: int,
    *,
    rng: random.Random | None = None,
    servers_per_tenant: int | None = None,
) -> dict[int, dict[int, int]]:
    if num_tenants <= 0:
        raise ValueError("num_tenants must be positive")

    rng = rng or random.Random()
    candidates = list(all_servers)
    if not candidates:
        raise ValueError("all_servers cannot be empty")

    if servers_per_tenant is None:
        if len(candidates) % num_tenants != 0:
            raise ValueError(
                "servers_per_tenant must be provided when the server count is not divisible by num_tenants"
            )
        servers_per_tenant = len(candidates) // num_tenants

    required_servers = num_tenants * servers_per_tenant
    if required_servers > len(candidates):
        raise ValueError("Not enough servers to place all tenants")

    selected = candidates[:required_servers]
    rng.shuffle(selected)

    tenant_mapping: dict[int, dict[int, int]] = {}
    for tenant in range(num_tenants):
        start = tenant * servers_per_tenant
        end = (tenant + 1) * servers_per_tenant
        assigned_servers = selected[start:end]
        tenant_mapping[tenant] = {
            rank: physical_server for rank, physical_server in enumerate(assigned_servers)
        }
    return tenant_mapping


def build_ring_flows(
    tenant_mapping: dict[int, dict[int, int]],
    single_flow_size_bits: int,
    collective: str,
    *,
    scale: float = 1e9,
) -> dict[int, list[tuple[int, int, float]]]:
    size_factor = _collective_size_factor(collective)
    tenant_flows: dict[int, list[tuple[int, int, float]]] = {}

    for tenant, mapping in tenant_mapping.items():
        tenant_flows[tenant] = []
        ranks = sorted(mapping.keys())
        for idx, src in enumerate(ranks):
            dst = ranks[(idx + 1) % len(ranks)]
            volume_bits = (len(ranks) - 1) * single_flow_size_bits * size_factor
            tenant_flows[tenant].append((src, dst, volume_bits / scale))

    return tenant_flows
