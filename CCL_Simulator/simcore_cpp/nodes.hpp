#pragma once
#include "types.hpp"
#include "port.hpp"
#include <map>

namespace simcore {

struct NodeConfig {
    std::string node_id;
    std::string node_type; // "gpu" or "switch"
    int num_qps = 1;
    int quantum_packets = 1;
    double tx_proc_delay = 0.0;
    double sw_proc_delay = 0.0;
    double gpu_store_delay = 0.0;
};

class BaseNode {
public:
    Environment& env;
    NodeConfig cfg;
    std::string node_id;
    std::map<std::string, std::shared_ptr<Port>> ports;

    BaseNode(Environment& env, NodeConfig cfg) : env(env), cfg(cfg), node_id(cfg.node_id) {}

    void add_port(std::string next_hop_id, double link_rate_bps, double prop_delay,
                  std::function<void(std::shared_ptr<Packet>)> deliver_fn,
                  int num_qps, int quantum_packets, double tx_proc_delay, int header_size_bytes) {
        ports[next_hop_id] = std::make_shared<Port>(
            env, node_id, next_hop_id, LinkSpec{link_rate_bps, prop_delay},
            deliver_fn, num_qps, quantum_packets, tx_proc_delay, header_size_bytes
        );
    }

    void _send_to_next(std::shared_ptr<Packet> pkt) {
        std::string nxt = pkt->next_hop();
        if (nxt.empty()) return;
        if (ports.find(nxt) == ports.end()) throw std::runtime_error(node_id + " has no port to " + nxt);
        pkt->advance();
        ports[nxt]->enqueue(pkt, pkt->qpid);
    }

    virtual void receive(std::shared_ptr<Packet> pkt) = 0;
};

class SwitchNode : public BaseNode {
public:
    SwitchNode(Environment& env, NodeConfig cfg) : BaseNode(env, cfg) {}

    void receive(std::shared_ptr<Packet> pkt) override {
        if (cfg.sw_proc_delay > 0) {
            env.schedule(cfg.sw_proc_delay, [this, pkt]() { this->_send_to_next(pkt); });
        } else {
            _send_to_next(pkt);
        }
    }
};

class GPUNode : public BaseNode {
public:
    std::function<void(TxId, double)> on_tx_complete;
    std::function<void(std::string, ChunkId, double)> on_chunk_ready;

    std::map<ChunkId, bool> have_chunk;
    std::map<TxId, int> _rx_cnt;

    GPUNode(Environment& env, NodeConfig cfg,
            std::function<void(TxId, double)> on_tx_complete,
            std::function<void(std::string, ChunkId, double)> on_chunk_ready)
        : BaseNode(env, cfg), on_tx_complete(on_tx_complete), on_chunk_ready(on_chunk_ready) {}

    void mark_initial_chunk(ChunkId chunk_id) {
        have_chunk[chunk_id] = true;
    }

    void receive(std::shared_ptr<Packet> pkt) override {
        if (pkt->tx_dst == node_id) {
            TxId tx = pkt->tx_id;
            _rx_cnt[tx]++;
            if (_rx_cnt[tx] >= pkt->total_packets) {
                if (cfg.gpu_store_delay > 0) {
                    env.schedule(cfg.gpu_store_delay, [this, tx, pkt]() {
                        this->on_tx_complete(tx, env.now);
                        if (!this->have_chunk[pkt->chunk_id]) {
                            this->have_chunk[pkt->chunk_id] = true;
                            this->on_chunk_ready(this->node_id, pkt->chunk_id, env.now);
                        }
                    });
                } else {
                    this->on_tx_complete(tx, env.now);
                    if (!this->have_chunk[pkt->chunk_id]) {
                        this->have_chunk[pkt->chunk_id] = true;
                        this->on_chunk_ready(this->node_id, pkt->chunk_id, env.now);
                    }
                }
            }
            return;
        }
        _send_to_next(pkt);
    }
};

} // namespace simcore
