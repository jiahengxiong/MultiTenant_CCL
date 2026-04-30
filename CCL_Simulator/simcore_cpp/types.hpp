#pragma once
#include <string>
#include <vector>
#include <variant>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <deque>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cstdint>

namespace simcore {

using ChunkId = std::string; // Simplify ChunkId to string for C++
using TxId = std::tuple<ChunkId, std::string, std::string>; // chunk_id, src, dst

// Hash for TxId
struct TxIdHash {
    std::size_t operator()(const TxId& tx) const {
        auto h1 = std::hash<std::string>{}(std::get<0>(tx));
        auto h2 = std::hash<std::string>{}(std::get<1>(tx));
        auto h3 = std::hash<std::string>{}(std::get<2>(tx));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct PolicyEntry {
    ChunkId chunk_id;
    std::string src;
    std::string dst;
    int qpid;
    double rate; // 0.0 means "Max"
    bool use_max_rate;
    int chunk_size_bytes;
    std::vector<std::string> path;
    double time = 0.0;
    std::vector<ChunkId> dependency;
};

struct Packet {
    TxId tx_id;
    ChunkId chunk_id;
    std::string tx_src;
    std::string tx_dst;
    int seq;
    int total_packets;
    int size_bytes;
    std::vector<std::string> path;
    int hop_idx;
    int qpid;
    double rate_bps;
    bool use_max_rate;
    double created_time = 0.0;

    std::string next_hop() const {
        if (hop_idx + 1 >= path.size()) return "";
        return path[hop_idx + 1];
    }

    void advance() {
        hop_idx++;
    }
};

// Event System
struct Event {
    double time;
    std::uint64_t sequence;
    std::function<void()> callback;
    bool operator>(const Event& other) const {
        if (time != other.time) return time > other.time;
        return sequence > other.sequence;
    }
};

class Environment {
public:
    double now = 0.0;
    std::uint64_t next_sequence = 0;
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events;

    void schedule(double delay, std::function<void()> callback) {
        events.push({now + delay, next_sequence++, callback});
    }

    void run(double until = -1.0) {
        while (!events.empty()) {
            Event ev = events.top();
            if (until >= 0.0 && ev.time > until) break;
            events.pop();
            now = ev.time;
            ev.callback();
        }
    }
};

} // namespace simcore
