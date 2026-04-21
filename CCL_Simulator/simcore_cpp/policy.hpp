#pragma once
#include "types.hpp"
#include "nodes.hpp"
#include <map>
#include <set>
#include <memory>

namespace simcore {

struct PolicySpec {
    int packet_size_bytes = 1500;
    int header_size_bytes = 0;
};

class PolicyEngine {
public:
    Environment& env;
    std::function<void(std::shared_ptr<Packet>)> send_from_src_fn;
    std::function<void(TxId)> register_tx_fn;
    std::map<std::string, std::shared_ptr<BaseNode>>* nodes_ptr;
    PolicySpec spec;

    std::map<std::pair<ChunkId, std::string>, std::vector<PolicyEntry>> rules;
    std::set<std::pair<ChunkId, std::string>> _fired;
    std::set<size_t> _scheduled_entries;
    std::map<std::pair<std::string, ChunkId>, bool> _ready_marked;
    std::map<std::pair<std::string, ChunkId>, std::vector<std::function<void()>>> _ready_events;

    PolicyEngine(Environment& env,
                 std::function<void(std::shared_ptr<Packet>)> send_from_src_fn,
                 std::function<void(TxId)> register_tx_fn,
                 std::map<std::string, std::shared_ptr<BaseNode>>* nodes_ptr,
                 PolicySpec spec)
        : env(env), send_from_src_fn(send_from_src_fn), register_tx_fn(register_tx_fn),
          nodes_ptr(nodes_ptr), spec(spec) {}

    void install(const std::vector<PolicyEntry>& entries) {
        for (const auto& e : entries) {
            rules[{e.chunk_id, e.src}].push_back(e);
        }
    }

    std::map<ChunkId, std::vector<std::string>> infer_initial_sources() {
        std::map<ChunkId, std::set<std::string>> by_chunk_src;
        std::map<ChunkId, std::set<std::string>> by_chunk_dst;

        for (const auto& kv : rules) {
            by_chunk_src[kv.first.first].insert(kv.first.second);
            for (const auto& e : kv.second) {
                by_chunk_dst[e.chunk_id].insert(e.dst);
            }
        }

        std::map<ChunkId, std::vector<std::string>> initial;
        for (const auto& kv : by_chunk_src) {
            ChunkId chunk_id = kv.first;
            std::vector<std::string> init;
            for (const auto& src : kv.second) {
                if (by_chunk_dst[chunk_id].find(src) == by_chunk_dst[chunk_id].end()) {
                    init.push_back(src);
                }
            }
            if (init.empty()) {
                init.assign(kv.second.begin(), kv.second.end());
            }
            std::sort(init.begin(), init.end());
            initial[chunk_id] = init;
        }
        return initial;
    }

    void bootstrap() {
        auto initial = infer_initial_sources();
        for (const auto& kv : initial) {
            for (const auto& s : kv.second) {
                auto node = (*nodes_ptr)[s];
                if (node->cfg.node_type != "gpu") throw std::runtime_error("Initial source must be GPU");
                std::dynamic_pointer_cast<GPUNode>(node)->mark_initial_chunk(kv.first);
                on_chunk_ready(s, kv.first);
            }
        }
    }

    void _mark_ready(std::string node_id, ChunkId chunk_id) {
        auto key = std::make_pair(node_id, chunk_id);
        if (_ready_marked[key]) return;
        _ready_marked[key] = true;
        for (auto& cb : _ready_events[key]) {
            cb();
        }
        _ready_events[key].clear();
    }

    void on_chunk_ready(std::string node_id, ChunkId chunk_id) {
        _mark_ready(node_id, chunk_id);
        auto key = std::make_pair(chunk_id, node_id);
        if (_fired.find(key) != _fired.end()) return;
        _fired.insert(key);

        for (const auto& e : rules[key]) {
            size_t eid = std::hash<std::string>{}(e.chunk_id + e.src + e.dst + std::to_string(e.qpid) + std::to_string(e.time));
            if (_scheduled_entries.find(eid) != _scheduled_entries.end()) continue;
            _scheduled_entries.insert(eid);
            _fire_entry_when_allowed(e);
        }
    }

    void _fire_entry_when_allowed(PolicyEntry e) {
        double wait = std::max(0.0, e.time - env.now);
        auto deps = e.dependency;
        if (deps.empty()) {
            env.schedule(wait, [this, e]() { this->_fire_entry(e); });
        } else {
            auto deps_ready = std::make_shared<int>(0);
            int total_deps = deps.size();
            auto check_deps = [this, e, deps_ready, total_deps]() {
                (*deps_ready)++;
                if (*deps_ready == total_deps) {
                    double wait = std::max(0.0, e.time - env.now);
                    env.schedule(wait, [this, e]() { this->_fire_entry(e); });
                }
            };
            for (const auto& dep : deps) {
                auto key = std::make_pair(e.src, dep);
                if (_ready_marked[key]) {
                    check_deps();
                } else {
                    _ready_events[key].push_back(check_deps);
                }
            }
        }
    }

    void _fire_entry(PolicyEntry e) {
        int ps = spec.packet_size_bytes;
        int total_packets = (e.chunk_size_bytes + ps - 1) / ps;
        total_packets = std::max(1, total_packets);

        TxId tx_id = {e.chunk_id, e.src, e.dst};
        register_tx_fn(tx_id);

        for (int i = 0; i < total_packets; ++i) {
            int remaining = e.chunk_size_bytes - i * ps;
            int sz = (remaining >= ps) ? ps : remaining;
            if (sz <= 0) sz = ps;

            auto pkt = std::make_shared<Packet>();
            pkt->tx_id = tx_id;
            pkt->chunk_id = e.chunk_id;
            pkt->tx_src = e.src;
            pkt->tx_dst = e.dst;
            pkt->seq = i;
            pkt->total_packets = total_packets;
            pkt->size_bytes = sz;
            pkt->path = e.path;
            pkt->hop_idx = 0;
            pkt->qpid = e.qpid;
            pkt->rate_bps = e.rate;
            pkt->use_max_rate = e.use_max_rate;
            pkt->created_time = env.now;

            send_from_src_fn(pkt);
        }
    }
};

} // namespace simcore
