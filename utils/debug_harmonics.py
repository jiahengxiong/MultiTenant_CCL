import gurobipy as gp
from gurobipy import GRB
from Harmonics_tools import HarmonicsILP, HarmonicsHeuristic
from tools import Datacenter

# 模拟数据
# 3个叶交换机, 2个脊交换机, 每个叶下2个服务器
# 拓扑会自动构建
datacenter = Datacenter(num_leaf=3, num_spine=2, per_leaf_server=2)

# 租户映射: 3个租户, 每个占用2个服务器
# server IDs: 0,1 (leaf 3), 2,3 (leaf 4), 4,5 (leaf 5)
tenant_mapping = {
    0: {0: 0, 1: 1},  # Tenant 0 在同一个 rack
    1: {0: 2, 1: 4},  # Tenant 1 跨 rack (leaf 4 -> leaf 5)
    2: {0: 3, 1: 5}   # Tenant 2 跨 rack (leaf 4 -> leaf 5), 应该和 Tenant 1 竞争
}

# 流量定义: (u_log, v_log, volume_normalized)
# volume 假设归一化过 (e.g. 1e9 scale)
# 这里模拟简单的点对点传输
tenant_flows = {
    0: [(0, 1, 0.1)], # 同 rack, 不经过 spine
    1: [(0, 1, 0.5)], # 跨 rack, 经过 spine
    2: [(0, 1, 0.5)]  # 跨 rack, 经过 spine
}

path_table = datacenter.paths
single_flow_size = 8 * 1024 * 1024 # 8MB
collective = "allreduce"

print("=== Running Heuristic Debug ===")
heu = HarmonicsHeuristic(datacenter, tenant_mapping, tenant_flows, path_table, single_flow_size, collective, verbose=True)
sched_heu = heu.solve()
print("Heuristic Schedule:", sched_heu)

print("\n=== Running ILP Debug ===")
# 实例化 ILP
ilp = HarmonicsILP(datacenter, tenant_mapping, tenant_flows, path_table, single_flow_size, collective, verbose=True)

# 求解
sched_ilp = ilp.solve()
print("ILP Schedule:", sched_ilp)

# 简单的结果检查
if sched_ilp:
    t1_start = sched_ilp[1][0]
    t2_start = sched_ilp[2][0]
    print(f"Tenant 1 start: {t1_start}, Tenant 2 start: {t2_start}")
    if abs(t1_start - t2_start) < 1e-4:
        print("Warning: Tenant 1 and 2 started at the same time. Check if they overlap correctly (share bandwidth) or conflict.")
    else:
        print("Tenant 1 and 2 are sequenced (good if capacity is tight).")
else:
    print("ILP failed to solve.")
