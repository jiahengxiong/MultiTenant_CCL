#pragma once
#include "types.hpp"
#include <deque>
#include <map>
#include <functional>
#include <memory>

namespace simcore {

struct LinkSpec {
    double link_rate_bps;
    double prop_delay;
};

class Port {
public:
    Environment& env;
    std::string owner_id;
    std::string next_hop_id;
    LinkSpec link;
    std::function<void(std::shared_ptr<Packet>)> deliver_fn;
    int num_qps;
    int quantum_packets;
    double tx_proc_delay;
    int header_size_bytes;

    std::map<int, std::deque<std::shared_ptr<Packet>>> qps;
    int _rr = 0;
    bool _draining = false;
    int _nq = 0;

    Port(Environment& env, std::string owner, std::string next_hop, LinkSpec link,
         std::function<void(std::shared_ptr<Packet>)> deliver_fn,
         int num_qps = 1, int quantum_packets = 1, double tx_proc_delay = 0.0, int header_size_bytes = 0)
        : env(env), owner_id(owner), next_hop_id(next_hop), link(link), deliver_fn(deliver_fn),
          num_qps(num_qps), quantum_packets(quantum_packets), tx_proc_delay(tx_proc_delay),
          header_size_bytes(header_size_bytes) {}

    void set_link_rate_bps(double new_rate_bps) {
        link.link_rate_bps = new_rate_bps;
    }

    void enqueue(std::shared_ptr<Packet> pkt, int qpid) {
        int q = qpid % num_qps;
        bool was_empty = (_nq == 0);
        qps[q].push_back(pkt);
        _nq++;

        if (was_empty && !_draining) {
            _draining = true;
            env.schedule(0, [this]() { this->_drain(); });
        }
    }

    int _next_non_empty_qp() {
        for (int i = 0; i < num_qps; ++i) {
            int idx = (_rr + i) % num_qps;
            if (!qps[idx].empty()) return idx;
        }
        return -1;
    }

    double _service_time(const Packet& pkt) {
        double eff = pkt.use_max_rate ? link.link_rate_bps : std::min(pkt.rate_bps, link.link_rate_bps);
        double total_bits = (pkt.size_bytes + header_size_bytes) * 8.0;
        return total_bits / eff;
    }

    void _drain() {
        if (_nq == 0) {
            _draining = false;
            return;
        }

        int qp = _next_non_empty_qp();
        if (qp == -1) {
            _draining = false;
            return;
        }

        int sent = 0;
        double total_delay = 0.0;
        while (sent < quantum_packets && !qps[qp].empty()) {
            auto pkt = qps[qp].front();
            qps[qp].pop_front();
            _nq--;

            if (tx_proc_delay > 0) total_delay += tx_proc_delay;
            double st = _service_time(*pkt);
            if (st > 0) total_delay += st;

            double pd = link.prop_delay;
            env.schedule(total_delay + pd, [this, pkt]() { deliver_fn(pkt); });
            sent++;
        }

        _rr = (qp + 1) % num_qps;
        env.schedule(total_delay, [this]() { this->_drain(); });
    }
};

} // namespace simcore
