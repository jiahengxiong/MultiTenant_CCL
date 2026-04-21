#pragma once
#include "types.hpp"
#include "nodes.hpp"
#include "policy.hpp"
#include <map>
#include <memory>

namespace simcore {

class Sim {
public:
    Environment env;
    std::map<std::string, std::shared_ptr<BaseNode>> nodes;
    std::map<TxId, double> tx_complete_time;
    std::map<std::pair<ChunkId, std::string>, double> chunk_ready_time;
    std::unique_ptr<PolicyEngine> policy;
    std::map<TxId, double> tx_first_send_time;

    Sim(int packet_size_bytes = 1500, int header_size_bytes = 0) {
        PolicySpec spec{packet_size_bytes, header_size_bytes};
        policy = std::make_unique<PolicyEngine>(
            env,
            [this](std::shared_ptr<Packet> pkt) { this->send_from_src(pkt); },
            [this](TxId tx_id) { this->register_tx(tx_id); },
            &nodes,
            spec
        );
    }

    void add_gpu(std::string node_id, int num_qps, int quantum_packets, double tx_proc_delay, double gpu_store_delay) {
        NodeConfig cfg{node_id, "gpu", num_qps, quantum_packets, tx_proc_delay, 0.0, gpu_store_delay};
        nodes[node_id] = std::make_shared<GPUNode>(
            env, cfg,
            [this](TxId tx_id, double t) { this->_on_tx_complete(tx_id, t); },
            [this](std::string node_id, ChunkId chunk_id, double t) { this->_on_chunk_ready(node_id, chunk_id, t); }
        );
    }

    void add_switch(std::string node_id, int num_qps, int quantum_packets, double tx_proc_delay, double sw_proc_delay) {
        NodeConfig cfg{node_id, "switch", num_qps, quantum_packets, tx_proc_delay, sw_proc_delay, 0.0};
        nodes[node_id] = std::make_shared<SwitchNode>(env, cfg);
    }

    void add_link(std::string u, std::string v, double link_rate_bps, double prop_delay) {
        auto src = nodes[u];
        auto deliver_fn = [this, v](std::shared_ptr<Packet> pkt) { this->nodes[v]->receive(pkt); };
        
        int num_qps = (src->cfg.node_type == "switch") ? 1 : src->cfg.num_qps;
        int quantum = (src->cfg.node_type == "switch") ? 1 : src->cfg.quantum_packets;

        src->add_port(v, link_rate_bps, prop_delay, deliver_fn, num_qps, quantum, src->cfg.tx_proc_delay, policy->spec.header_size_bytes);
    }

    void load_policy(const std::vector<PolicyEntry>& entries) {
        policy->install(entries);
    }

    void start() {
        policy->bootstrap();
    }

    void register_tx(TxId tx_id) {
        if (tx_complete_time.find(tx_id) == tx_complete_time.end()) {
            tx_complete_time[tx_id] = NAN;
        }
    }

    void send_from_src(std::shared_ptr<Packet> pkt) {
        std::string src_id = pkt->path[pkt->hop_idx];
        auto node = nodes[src_id];
        if (node->cfg.node_type != "gpu") throw std::runtime_error("Policy src must be a GPU");
        if (pkt->seq == 0 && tx_first_send_time.find(pkt->tx_id) == tx_first_send_time.end()) {
            tx_first_send_time[pkt->tx_id] = env.now;
        }
        node->_send_to_next(pkt);
    }

    void _on_tx_complete(TxId tx_id, double t) {
        if (std::isnan(tx_complete_time[tx_id])) {
            tx_complete_time[tx_id] = t;
        }
    }

    void _on_chunk_ready(std::string node_id, ChunkId chunk_id, double t) {
        auto key = std::make_pair(chunk_id, node_id);
        if (chunk_ready_time.find(key) == chunk_ready_time.end()) {
            chunk_ready_time[key] = t;
        }
        policy->on_chunk_ready(node_id, chunk_id);
    }

    void run(double until = -1.0) {
        env.run(until);
    }
};

} // namespace simcore
