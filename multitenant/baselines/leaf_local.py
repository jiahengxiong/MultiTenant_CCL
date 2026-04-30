from __future__ import annotations


def build_leaf_local_mapping(
    tenant_mapping: dict[int, dict[int, int]],
) -> dict[int, dict[int, int]]:
    """Group each tenant's assigned servers by physical order.

    In the current topology construction, server IDs are contiguous within the
    same leaf. Sorting a tenant's assigned servers therefore yields a simple
    leaf-local baseline that prioritizes colocating consecutive ranks on the
    same leaf before spanning additional leaves.
    """

    leaf_local_mapping: dict[int, dict[int, int]] = {}
    for tenant, rank_to_server in tenant_mapping.items():
        ranks = sorted(rank_to_server.keys())
        servers = sorted(rank_to_server.values())
        leaf_local_mapping[tenant] = {
            rank: server for rank, server in zip(ranks, servers)
        }
    return leaf_local_mapping


class LeafLocalBaseline:
    """Deterministic baseline that sorts each tenant's assigned servers."""

    def __init__(self, tenant_mapping: dict[int, dict[int, int]]):
        self.tenant_mapping = {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in tenant_mapping.items()
        }
        self.final_mapping: dict[int, dict[int, int]] | None = None

    def solve(self) -> dict[int, dict[int, int]]:
        self.final_mapping = build_leaf_local_mapping(self.tenant_mapping)
        return {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in self.final_mapping.items()
        }

    def get_mapping(self) -> dict[int, dict[int, int]]:
        if self.final_mapping is None:
            return self.solve()
        return {
            tenant: dict(rank_to_server)
            for tenant, rank_to_server in self.final_mapping.items()
        }
