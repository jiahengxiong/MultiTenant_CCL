from __future__ import annotations

import hashlib

import networkx as nx


class LeafSpineDatacenter:
    """Leaf-spine datacenter with a deterministic ECMP-like path table."""

    def __init__(self, num_leaf: int, num_spine: int, per_leaf_server: int):
        self.topology = nx.DiGraph()
        self.num_leaf = num_leaf
        self.num_spine = num_spine
        self.num_server = per_leaf_server * self.num_leaf
        self.single_num_server = per_leaf_server
        self.link_rate_leaf_spine = 25e9
        self.link_rate_server_leaf = 50e9

        self._build_leaf_spine()
        self._build_server_leaf()

        self.paths = self.build_ecmp_path_table()
        self.ECMP_table = self.build_ecmp_edge_table()
        self.ECMP_edge_set, self.ECMP_hops = self._build_edge_set_and_hops(self.ECMP_table)

    def _build_leaf_spine(self) -> None:
        leaf_index = list(range(self.num_server, self.num_leaf + self.num_server))
        spine_index = list(
            range(
                self.num_leaf + self.num_server,
                self.num_leaf + self.num_server + self.num_spine,
            )
        )
        self.leaf_index = leaf_index
        self.spine_index = spine_index

        for leaf in leaf_index:
            self.topology.add_node(leaf, type="leaf")
        for spine in spine_index:
            self.topology.add_node(spine, type="spine")

        for leaf in leaf_index:
            for spine in spine_index:
                self.topology.add_edge(leaf, spine, capacity=self.link_rate_leaf_spine)
                self.topology.add_edge(spine, leaf, capacity=self.link_rate_leaf_spine)

    def _build_server_leaf(self) -> None:
        server_index = 0
        for leaf in self.leaf_index:
            for _ in range(self.single_num_server):
                self.topology.add_node(server_index, type="server")
                self.topology.add_edge(server_index, leaf, capacity=self.link_rate_server_leaf)
                self.topology.add_edge(leaf, server_index, capacity=self.link_rate_server_leaf)
                server_index += 1

    def get_all_servers(self) -> list[int]:
        return sorted(
            node
            for node, attrs in self.topology.nodes(data=True)
            if attrs.get("type") == "server"
        )

    def get_server_leaf(self, server_id: int) -> int:
        for neighbor in self.topology.successors(server_id):
            if self.topology.nodes[neighbor].get("type") == "leaf":
                return neighbor
        for neighbor in self.topology.predecessors(server_id):
            if self.topology.nodes[neighbor].get("type") == "leaf":
                return neighbor
        raise ValueError(f"Server {server_id} has no attached leaf")

    def _hash_spine(self, src: int, dst: int) -> int:
        key = f"{src}->{dst}".encode("utf-8")
        hashed = hashlib.md5(key).digest()
        value = int.from_bytes(hashed[:-1], "big")
        return self.spine_index[value % self.num_spine]

    def get_ecmp_path(self, src: int, dst: int) -> list[int]:
        if src == dst:
            raise ValueError("src == dst is not allowed")

        leaf_src = self.get_server_leaf(src)
        leaf_dst = self.get_server_leaf(dst)

        if leaf_src == leaf_dst:
            return [src, leaf_src, dst]

        spine = self._hash_spine(src, dst)
        return [src, leaf_src, spine, leaf_dst, dst]

    def build_ecmp_path_table(self) -> dict[tuple[int, int], list[int]]:
        table: dict[tuple[int, int], list[int]] = {}
        servers = self.get_all_servers()
        for src in servers:
            for dst in servers:
                if src == dst:
                    continue
                table[(src, dst)] = self.get_ecmp_path(src, dst)
        return table

    @staticmethod
    def path_to_edges(path: list[int]) -> list[tuple[int, int]]:
        return list(zip(path[:-1], path[1:]))

    def build_ecmp_edge_table(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        return {key: self.path_to_edges(path) for key, path in self.paths.items()}

    @staticmethod
    def _build_edge_set_and_hops(
        edge_table: dict[tuple[int, int], list[tuple[int, int]]]
    ) -> tuple[dict[tuple[int, int], set[tuple[int, int]]], dict[tuple[int, int], int]]:
        edge_set: dict[tuple[int, int], set[tuple[int, int]]] = {}
        hops: dict[tuple[int, int], int] = {}
        for key, edges in edge_table.items():
            edge_set[key] = set(edges)
            hops[key] = len(edges)
        return edge_set, hops
