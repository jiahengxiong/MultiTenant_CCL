import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def draw_datacenter_topology(dat, figsize=(12, 6), show_capacity=False):
    """
    三层画法：
      y=2: spine
      y=1: leaf
      y=0: server（按其连接的 leaf 分组摆在 leaf 下方）
    dat: 你的 Datacenter 实例（包含 dat.topology / dat.leaf_index / dat.spine_index）
    show_capacity: 是否在边上标注 capacity（会很乱，默认关）
    """
    G = dat.topology
    leaves = list(dat.leaf_index)
    spines = list(dat.spine_index)
    servers = sorted([n for n, attr in G.nodes(data=True) if attr.get("type") == "server"])

    # 1) 建 server -> leaf 的归属（找 server->leaf 或 leaf->server 的邻接都行）
    server_to_leaf = {}
    leaf_to_servers = defaultdict(list)

    leaf_set = set(leaves)
    for s in servers:
        # 找它相邻的 leaf（你的拓扑里 server<->leaf 双向边都存在）
        nbrs = set(G.successors(s)) | set(G.predecessors(s))
        attached = [x for x in nbrs if x in leaf_set]
        if not attached:
            # 没挂到 leaf：就先跳过（理论上你这版不会发生）
            continue
        lf = attached[0]
        server_to_leaf[s] = lf
        leaf_to_servers[lf].append(s)

    # 2) 计算坐标：spine 一排，leaf 一排，server 分组放在 leaf 下
    pos = {}

    # x 坐标给 leaf 等间隔
    # 为了让 server 组不会挤，给每个 leaf 一段 x 区间
    leaf_x = {}
    leaf_spacing = 4.0  # 叶交换机之间的间距（你可以调大一点更松）
    for i, lf in enumerate(sorted(leaves)):
        leaf_x[lf] = i * leaf_spacing
        pos[lf] = (leaf_x[lf], 1.0)

    # spine 放在最上方，均匀铺在 leaf 的上方范围
    if leaves:
        min_x = min(leaf_x.values())
        max_x = max(leaf_x.values())
    else:
        min_x, max_x = 0.0, 1.0

    if len(spines) == 1:
        spine_xs = [(min_x + max_x) / 2.0]
    else:
        spine_xs = [
            min_x + (max_x - min_x) * k / (len(spines) - 1)
            for k in range(len(spines))
        ]

    for j, sp in enumerate(sorted(spines)):
        pos[sp] = (spine_xs[j], 2.0)

    # server：每个 leaf 下方一排
    server_y = 0.0
    server_gap = 0.8  # 同一 leaf 下 server 的横向间隔
    for lf in sorted(leaves):
        s_list = sorted(leaf_to_servers.get(lf, []))
        if not s_list:
            continue
        # 让 server 围绕 leaf 的 x 居中
        start = leaf_x[lf] - server_gap * (len(s_list) - 1) / 2.0
        for k, s in enumerate(s_list):
            pos[s] = (start + k * server_gap, server_y)

    # 3) 画图（分层颜色/形状）
    plt.figure(figsize=figsize)

    # 分层节点
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(spines), node_size=900, node_shape="s")
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(leaves), node_size=900, node_shape="s")
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(servers), node_size=700, node_shape="o")

    # 画边：为了清晰只画无向效果（你的图是双向边，画两次会很黑）
    # 这里用 set 去重，把 (u,v) 和 (v,u) 当成一条线画
    undirected_edges = set()
    for u, v in G.edges():
        a, b = (u, v) if u <= v else (v, u)
        undirected_edges.add((a, b))

    nx.draw_networkx_edges(G, pos, edgelist=list(undirected_edges), width=1.0, alpha=0.7)

    # 标签：直接画节点编号
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    # 可选：边容量标签（会很挤）
    if show_capacity:
        edge_labels = {}
        for (u, v) in undirected_edges:
            cap = None
            # 找任一方向的 capacity
            if G.has_edge(u, v):
                cap = G[u][v].get("capacity", None)
            elif G.has_edge(v, u):
                cap = G[v][u].get("capacity", None)
            if cap is not None:
                edge_labels[(u, v)] = f"{cap/1e9:.0f}G"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.axis("off")
    plt.tight_layout()
    plt.show()