from utils.tools import Datacenter
import networkx as nx

MB = 1024 * 1024 * 8 # 1MB

def main():
    print(f"MB constant = {MB}")
    single_flow_size = 8 * MB
    print(f"single_flow_size = {single_flow_size}")
    
    num_tenants = 4
    number_leaf = 3
    num_spine = 2
    per_leaf_servers = 4
    
    datacenter = Datacenter(number_leaf, num_spine, per_leaf_servers)
    
    caps = nx.get_edge_attributes(datacenter.topology, "capacity")
    print("Sample Capacities:")
    count = 0
    for e, cap in caps.items():
        print(f"  Edge {e}: {cap}")
        count += 1
        if count >= 5: break
        
if __name__ == "__main__":
    main()
