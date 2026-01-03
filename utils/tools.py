import networkx as nx
from utils.draw import draw_datacenter_topology
import hashlib


class Datacenter:
    def __init__(self, num_leaf: int, num_spine: int, per_leaf_server: int):
        self.topology = nx.DiGraph()
        self.num_leaf = num_leaf
        self.num_spine = num_spine
        self.num_server = per_leaf_server * self.num_leaf
        self.single_num_server = per_leaf_server
        self.link_rate_LS = 25e9
        self.link_rate_SL = 50e9

        self.build_leaf_spine()
        self.build_sever_leaf()

        # Build ECMP path/edge tables
        self.paths = self.build_ecmp_path_table()              # dict[(s,t)] -> [nodes...]
        self.ECMP_table = self.build_ecmp_edge_table()         # dict[(s,t)] -> [(u,v),...]
        self.ECMP_edge_set, self.ECMP_hops = self._build_edge_set_and_hops(self.ECMP_table)

    # ---------- topology ----------
    def build_leaf_spine(self):
        leaf_index = list(range(self.num_server, self.num_leaf + self.num_server))
        spine_index = list(range(self.num_leaf + self.num_server,
                                 self.num_leaf + self.num_server + self.num_spine))
        self.leaf_index = leaf_index
        self.spine_index = spine_index

        for i in leaf_index:
            self.topology.add_node(i, type="leaf")
        for j in spine_index:
            self.topology.add_node(j, type="spine")

        for i in leaf_index:
            for j in spine_index:
                self.topology.add_edge(i, j, capacity=self.link_rate_LS)
                self.topology.add_edge(j, i, capacity=self.link_rate_LS)

    def build_sever_leaf(self):
        server_index = 0
        for leaf in self.leaf_index:
            for _ in range(self.single_num_server):
                self.topology.add_node(server_index, type="server")
                self.topology.add_edge(server_index, leaf, capacity=self.link_rate_SL)
                self.topology.add_edge(leaf, server_index, capacity=self.link_rate_SL)
                server_index += 1

    # ---------- helpers ----------
    def get_all_servers(self):
        return sorted(
            [n for n, attr in self.topology.nodes(data=True)
             if attr.get("type") == "server"]
        )

    def get_server_leaf(self, server_id: int) -> int:
        for v in self.topology.successors(server_id):
            if self.topology.nodes[v].get("type") == "leaf":
                return v
        for v in self.topology.predecessors(server_id):
            if self.topology.nodes[v].get("type") == "leaf":
                return v
        raise ValueError(f"Server {server_id} has no attached leaf")

    # ---------- ECMP-like hashing (src,dst only) ----------
    def _hash_spine(self, src: int, dst: int) -> int:
        key = f"{src}->{dst}".encode("utf-8")
        h = hashlib.md5(key).digest()
        x = int.from_bytes(h[:-1], "big")  # take high bits
        return self.spine_index[x % self.num_spine]

    def get_ecmp_path(self, src: int, dst: int):
        """
        Return exactly ONE path for (src,dst),
        chosen by hash(src,dst), ECMP-style.
        """
        if src == dst:
            raise ValueError("src == dst not allowed")

        leaf_s = self.get_server_leaf(src)
        leaf_d = self.get_server_leaf(dst)

        if leaf_s == leaf_d:
            return [src, leaf_s, dst]

        spine = self._hash_spine(src, dst)
        return [src, leaf_s, spine, leaf_d, dst]

    def build_ecmp_path_table(self):
        """
        dict[(src,dst)] -> unique ECMP-hashed path (node list)
        """
        servers = self.get_all_servers()
        table = {}
        for s in servers:
            for t in servers:
                if s == t:
                    continue
                table[(s, t)] = self.get_ecmp_path(s, t)
        return table

    @staticmethod
    def path_to_edges(path):
        return list(zip(path[:-1], path[1:]))

    def build_ecmp_edge_table(self):
        """
        dict[(src,dst)] -> list of edges on the chosen path
        """
        return {k: self.path_to_edges(v) for k, v in self.paths.items()}

    @staticmethod
    def _build_edge_set_and_hops(edge_table):
        """
        edge_table: dict[(s,t)] -> [(u,v),...]
        returns:
          edge_set[(s,t)] = set((u,v),...)
          hops[(s,t)] = int
        """
        edge_set = {}
        hops = {}
        for k, edges in edge_table.items():
            edge_set[k] = set(edges)
            hops[k] = len(edges)
        return edge_set, hops


if __name__ == "__main__":
    dat = Datacenter(3, 2, 2)
    draw_datacenter_topology(dat)

    print("path 0 -> 5:", dat.paths[(0, 5)])
    print("edges 0 -> 5:", dat.ECMP_table[(0, 5)])
    print("hops  0 -> 5:", dat.ECMP_hops[(0, 5)])
    print("edge_in_path? (0,6) in 0->5:", (0, 6) in dat.ECMP_edge_set[(0, 5)])