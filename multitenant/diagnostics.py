from __future__ import annotations

from collections import defaultdict


def sample_link_capacities(datacenter, limit: int = 5) -> list[tuple[tuple[int, int], float]]:
    capacities = []
    for edge in datacenter.topology.edges():
        capacities.append((edge, float(datacenter.topology[edge[0]][edge[1]]["capacity"])))
    return capacities[:limit]


def resolve_flow_paths(
    datacenter,
    tenant_mapping: dict[int, dict[int, int]],
    tenant_flows: dict[int, list[tuple[int, int, float]]],
) -> list[dict[str, object]]:
    resolved = []
    for tenant, flows in tenant_flows.items():
        for flow_index, (logical_src, logical_dst, volume_gbit) in enumerate(flows):
            physical_src = tenant_mapping[tenant][logical_src]
            physical_dst = tenant_mapping[tenant][logical_dst]
            path = datacenter.paths.get((physical_src, physical_dst), [])
            resolved.append(
                {
                    "tenant": tenant,
                    "flow_index": flow_index,
                    "logical_src": logical_src,
                    "logical_dst": logical_dst,
                    "physical_src": physical_src,
                    "physical_dst": physical_dst,
                    "volume_gbit": float(volume_gbit),
                    "path": path,
                }
            )
    return resolved


def summarize_link_occupancy(
    datacenter,
    tenant_mapping: dict[int, dict[int, int]],
    tenant_flows: dict[int, list[tuple[int, int, float]]],
) -> list[dict[str, object]]:
    total_volume_gbit = defaultdict(float)
    contributors = defaultdict(list)

    for flow in resolve_flow_paths(datacenter, tenant_mapping, tenant_flows):
        path = flow["path"]
        if not path:
            continue
        for edge in datacenter.path_to_edges(path):
            total_volume_gbit[edge] += flow["volume_gbit"]
            contributors[edge].append(
                {
                    "tenant": flow["tenant"],
                    "flow_index": flow["flow_index"],
                    "volume_gbit": flow["volume_gbit"],
                }
            )

    summaries = []
    for edge, volume_gbit in total_volume_gbit.items():
        capacity_gbps = float(datacenter.topology[edge[0]][edge[1]]["capacity"]) / 1e9
        summaries.append(
            {
                "edge": edge,
                "total_volume_gbit": volume_gbit,
                "capacity_gbps": capacity_gbps,
                "isolated_occupancy_seconds": volume_gbit / capacity_gbps if capacity_gbps > 0 else float("inf"),
                "contributors": contributors[edge],
            }
        )

    return sorted(summaries, key=lambda item: item["isolated_occupancy_seconds"], reverse=True)
