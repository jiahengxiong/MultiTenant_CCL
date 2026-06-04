#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

struct PriceKey {
    int epoch;
    int src_server;
    int dst_server;

    bool operator==(const PriceKey &other) const {
        return epoch == other.epoch
            && src_server == other.src_server
            && dst_server == other.dst_server;
    }
};

struct PriceKeyHash {
    std::size_t operator()(const PriceKey &key) const {
        std::size_t value = static_cast<std::size_t>(key.epoch);
        value ^= static_cast<std::size_t>(key.src_server) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        value ^= static_cast<std::size_t>(key.dst_server) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        return value;
    }
};

std::unordered_map<PriceKey, double, PriceKeyHash> build_price_lookup(const py::dict &pair_epoch_price) {
    std::unordered_map<PriceKey, double, PriceKeyHash> lookup;
    lookup.reserve(static_cast<size_t>(pair_epoch_price.size()));
    for (auto item : pair_epoch_price) {
        py::tuple key = py::reinterpret_borrow<py::tuple>(item.first);
        lookup.emplace(
            PriceKey{
                py::cast<int>(key[0]),
                py::cast<int>(key[1]),
                py::cast<int>(key[2]),
            },
            py::cast<double>(item.second)
        );
    }
    return lookup;
}

std::vector<double> max_min_rates(
    const py::list &resources_by_item,
    const py::list &capacities_by_item,
    const py::list &demand_rates
) {
    const py::ssize_t item_count = resources_by_item.size();
    if (capacities_by_item.size() != item_count || demand_rates.size() != item_count) {
        throw std::invalid_argument("max_min_rates inputs must have matching lengths");
    }

    py::dict resource_to_index;
    std::vector<std::vector<int>> item_resources(static_cast<size_t>(item_count));
    std::vector<std::vector<int>> resource_users;
    std::vector<double> residual_capacity;
    std::vector<int> active_count;

    for (py::ssize_t item_idx = 0; item_idx < item_count; ++item_idx) {
        py::sequence resources = py::reinterpret_borrow<py::sequence>(resources_by_item[item_idx]);
        py::sequence capacities = py::reinterpret_borrow<py::sequence>(capacities_by_item[item_idx]);
        if (resources.size() != capacities.size()) {
            throw std::invalid_argument("resource and capacity lists must have matching lengths");
        }

        for (py::ssize_t res_pos = 0; res_pos < resources.size(); ++res_pos) {
            py::object resource = py::reinterpret_borrow<py::object>(resources[res_pos]);
            int resource_idx;
            if (!resource_to_index.contains(resource)) {
                resource_idx = static_cast<int>(residual_capacity.size());
                resource_to_index[resource] = py::int_(resource_idx);
                residual_capacity.push_back(py::cast<double>(capacities[res_pos]));
                active_count.push_back(0);
                resource_users.emplace_back();
            } else {
                resource_idx = py::cast<int>(resource_to_index[resource]);
            }
            item_resources[static_cast<size_t>(item_idx)].push_back(resource_idx);
            resource_users[static_cast<size_t>(resource_idx)].push_back(static_cast<int>(item_idx));
            active_count[static_cast<size_t>(resource_idx)] += 1;
        }
    }

    std::vector<double> rates(static_cast<size_t>(item_count), 0.0);
    std::vector<char> remaining(static_cast<size_t>(item_count), 1);
    int remaining_count = static_cast<int>(item_count);

    while (remaining_count > 0) {
        double limiting_share = std::numeric_limits<double>::infinity();
        std::vector<int> limiting_resources;

        for (int resource_idx = 0; resource_idx < static_cast<int>(active_count.size()); ++resource_idx) {
            const int count = active_count[static_cast<size_t>(resource_idx)];
            if (count <= 0) {
                continue;
            }
            const double share = residual_capacity[static_cast<size_t>(resource_idx)] / static_cast<double>(count);
            if (share < limiting_share - 1e-12) {
                limiting_share = share;
                limiting_resources.clear();
                limiting_resources.push_back(resource_idx);
            } else if (std::abs(share - limiting_share) <= 1e-12) {
                limiting_resources.push_back(resource_idx);
            }
        }

        if (!std::isfinite(limiting_share)) {
            break;
        }

        std::vector<int> frozen;
        std::vector<char> frozen_seen(static_cast<size_t>(item_count), 0);
        for (const int resource_idx : limiting_resources) {
            for (const int item_idx : resource_users[static_cast<size_t>(resource_idx)]) {
                if (remaining[static_cast<size_t>(item_idx)] && !frozen_seen[static_cast<size_t>(item_idx)]) {
                    frozen_seen[static_cast<size_t>(item_idx)] = 1;
                    frozen.push_back(item_idx);
                }
            }
        }
        if (frozen.empty()) {
            for (int item_idx = 0; item_idx < item_count; ++item_idx) {
                if (remaining[static_cast<size_t>(item_idx)]) {
                    frozen.push_back(item_idx);
                }
            }
        }

        for (const int item_idx : frozen) {
            const double demand_rate = py::cast<double>(demand_rates[item_idx]);
            rates[static_cast<size_t>(item_idx)] = std::max(0.0, std::min(limiting_share, demand_rate));
        }

        for (const int item_idx : frozen) {
            const double rate = rates[static_cast<size_t>(item_idx)];
            for (const int resource_idx : item_resources[static_cast<size_t>(item_idx)]) {
                double &residual = residual_capacity[static_cast<size_t>(resource_idx)];
                residual = std::max(0.0, residual - rate);
                active_count[static_cast<size_t>(resource_idx)] -= 1;
            }
            if (remaining[static_cast<size_t>(item_idx)]) {
                remaining[static_cast<size_t>(item_idx)] = 0;
                --remaining_count;
            }
        }
    }

    return rates;
}

struct TaskDef {
    int tenant = 0;
    int task_id = 0;
    int src_rank = 0;
    int dst_rank = 0;
    double volume_bits = 0.0;
    int order_pos = 0;
    std::vector<int> predecessors;
    std::vector<int> release_prev;
    double release_gap = 0.0;
};

struct TaskRuntime {
    int src_server = 0;
    int dst_server = 0;
    std::vector<int> path_edges;
    std::vector<int> resources;
    double remaining_bits = 0.0;
    double start_time = -1.0;
    double finish_time = -1.0;
    long long queue_key = 0;
};

static std::vector<double> max_min_rates_native_dense(
    const std::vector<std::vector<int>> &resources_by_item,
    const std::vector<double> &demand_rates,
    const std::vector<double> &resource_capacity,
    const int resource_count
) {
    const int item_count = static_cast<int>(resources_by_item.size());
    if (item_count == 0) {
        return {};
    }

    std::vector<std::vector<int>> resource_users(static_cast<size_t>(resource_count));
    std::vector<double> residual_capacity(static_cast<size_t>(resource_count), 0.0);
    std::vector<int> active_count(static_cast<size_t>(resource_count), 0);
    std::vector<int> active_resources;
    active_resources.reserve(static_cast<size_t>(resource_count));

    for (int item_idx = 0; item_idx < item_count; ++item_idx) {
        const auto &resources = resources_by_item[static_cast<size_t>(item_idx)];
        for (size_t res_pos = 0; res_pos < resources.size(); ++res_pos) {
            const int resource = resources[res_pos];
            if (resource < 0 || resource >= resource_count) {
                continue;
            }
            if (active_count[static_cast<size_t>(resource)] == 0) {
                active_resources.push_back(resource);
                residual_capacity[static_cast<size_t>(resource)] = resource_capacity[static_cast<size_t>(resource)];
            }
            resource_users[static_cast<size_t>(resource)].push_back(item_idx);
            active_count[static_cast<size_t>(resource)] += 1;
        }
    }

    std::vector<double> rates(static_cast<size_t>(item_count), 0.0);
    std::vector<char> remaining(static_cast<size_t>(item_count), 1);
    int remaining_count = item_count;
    while (remaining_count > 0) {
        double limiting_share = std::numeric_limits<double>::infinity();
        std::vector<int> limiting_resources;
        for (const int resource_idx : active_resources) {
            const int count = active_count[static_cast<size_t>(resource_idx)];
            if (count <= 0) {
                continue;
            }
            const double share = residual_capacity[static_cast<size_t>(resource_idx)] / static_cast<double>(count);
            if (share < limiting_share - 1e-12) {
                limiting_share = share;
                limiting_resources.clear();
                limiting_resources.push_back(resource_idx);
            } else if (std::abs(share - limiting_share) <= 1e-12) {
                limiting_resources.push_back(resource_idx);
            }
        }
        if (!std::isfinite(limiting_share)) {
            break;
        }

        std::vector<int> frozen;
        std::vector<char> seen(static_cast<size_t>(item_count), 0);
        for (const int resource_idx : limiting_resources) {
            for (const int item_idx : resource_users[static_cast<size_t>(resource_idx)]) {
                if (remaining[static_cast<size_t>(item_idx)] && !seen[static_cast<size_t>(item_idx)]) {
                    seen[static_cast<size_t>(item_idx)] = 1;
                    frozen.push_back(item_idx);
                }
            }
        }
        if (frozen.empty()) {
            for (int item_idx = 0; item_idx < item_count; ++item_idx) {
                if (remaining[static_cast<size_t>(item_idx)]) {
                    frozen.push_back(item_idx);
                }
            }
        }

        for (const int item_idx : frozen) {
            rates[static_cast<size_t>(item_idx)] = std::max(
                0.0,
                std::min(limiting_share, demand_rates[static_cast<size_t>(item_idx)])
            );
        }
        for (const int item_idx : frozen) {
            const double rate = rates[static_cast<size_t>(item_idx)];
            for (const int resource_idx : resources_by_item[static_cast<size_t>(item_idx)]) {
                if (resource_idx < 0 || resource_idx >= resource_count) {
                    continue;
                }
                double &residual = residual_capacity[static_cast<size_t>(resource_idx)];
                residual = std::max(0.0, residual - rate);
                active_count[static_cast<size_t>(resource_idx)] -= 1;
            }
            if (remaining[static_cast<size_t>(item_idx)]) {
                remaining[static_cast<size_t>(item_idx)] = 0;
                --remaining_count;
            }
        }
    }
    return rates;
}

class TimeExpandedScoreEngine {
public:
    TimeExpandedScoreEngine(
        std::vector<int> tenant_ids,
        std::vector<double> tenant_start_times,
        std::vector<double> server_send_capacity,
        std::vector<double> server_recv_capacity,
        std::vector<double> edge_capacity,
        std::vector<std::vector<std::vector<std::vector<int>>>> path_edges,
        py::list task_entries,
        py::object slot_duration_override,
        py::object horizon_slots_override
    )
        : tenant_ids_(std::move(tenant_ids)),
          tenant_start_times_(std::move(tenant_start_times)),
          server_send_capacity_(std::move(server_send_capacity)),
          server_recv_capacity_(std::move(server_recv_capacity)),
          edge_capacity_(std::move(edge_capacity)),
          path_edges_(std::move(path_edges)),
          slot_duration_override_(slot_duration_override.is_none() ? -1.0 : py::cast<double>(slot_duration_override)),
          horizon_slots_override_(horizon_slots_override.is_none() ? -1 : py::cast<int>(horizon_slots_override)) {
        server_count_ = static_cast<int>(server_send_capacity_.size());
        edge_count_ = static_cast<int>(edge_capacity_.size());
        const int tenant_count = static_cast<int>(tenant_ids_.size());
        tasks_by_tenant_.assign(static_cast<size_t>(tenant_count), {});

        for (py::handle entry_handle : task_entries) {
            py::sequence entry = py::reinterpret_borrow<py::sequence>(entry_handle);
            if (entry.size() != 9) {
                throw std::invalid_argument("task entry must have 9 fields");
            }
            TaskDef task;
            task.tenant = py::cast<int>(entry[0]);
            task.task_id = py::cast<int>(entry[1]);
            task.src_rank = py::cast<int>(entry[2]);
            task.dst_rank = py::cast<int>(entry[3]);
            task.volume_bits = py::cast<double>(entry[4]);
            task.order_pos = py::cast<int>(entry[5]);
            task.predecessors = py::cast<std::vector<int>>(entry[6]);
            task.release_prev = py::cast<std::vector<int>>(entry[7]);
            task.release_gap = py::cast<double>(entry[8]);
            const int task_idx = static_cast<int>(tasks_.size());
            tasks_.push_back(std::move(task));
            tasks_by_tenant_[static_cast<size_t>(tasks_.back().tenant)].push_back(task_idx);
        }

        successors_.assign(tasks_.size(), {});
        for (int task_idx = 0; task_idx < static_cast<int>(tasks_.size()); ++task_idx) {
            for (const int pred_idx : tasks_[static_cast<size_t>(task_idx)].predecessors) {
                if (pred_idx >= 0 && pred_idx < static_cast<int>(tasks_.size())) {
                    successors_[static_cast<size_t>(pred_idx)].push_back(task_idx);
                }
            }
        }
    }

    py::tuple evaluate(const std::vector<std::vector<int>> &server_by_tenant_rank) const {
        const double slot_duration = estimate_slot_duration(server_by_tenant_rank);
        const int max_slots = max_slot_count();
        const int task_count = static_cast<int>(tasks_.size());

        std::vector<TaskRuntime> runtime(static_cast<size_t>(task_count));
        std::vector<int> remaining_preds(static_cast<size_t>(task_count), 0);
        std::vector<char> pending(static_cast<size_t>(task_count), 1);
        std::vector<char> active(static_cast<size_t>(task_count), 0);
        int pending_count = task_count;
        int active_count = 0;
        const int resource_count = 2 * server_count_ + edge_count_;
        std::vector<double> resource_capacity(static_cast<size_t>(resource_count), 0.0);
        for (int server = 0; server < server_count_; ++server) {
            resource_capacity[static_cast<size_t>(server)] = server_send_capacity_[static_cast<size_t>(server)];
            resource_capacity[static_cast<size_t>(server_count_ + server)] = server_recv_capacity_[static_cast<size_t>(server)];
        }
        for (int edge_idx = 0; edge_idx < edge_count_; ++edge_idx) {
            resource_capacity[static_cast<size_t>(2 * server_count_ + edge_idx)] = edge_capacity_[static_cast<size_t>(edge_idx)];
        }

        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
            const TaskDef &task = tasks_[static_cast<size_t>(task_idx)];
            TaskRuntime &state = runtime[static_cast<size_t>(task_idx)];
            state.src_server = server_by_tenant_rank[static_cast<size_t>(task.tenant)][static_cast<size_t>(task.src_rank)];
            state.dst_server = server_by_tenant_rank[static_cast<size_t>(task.tenant)][static_cast<size_t>(task.dst_rank)];
            state.path_edges = path_edges_[static_cast<size_t>(task.tenant)]
                [static_cast<size_t>(state.src_server)]
                [static_cast<size_t>(state.dst_server)];
            state.remaining_bits = task.volume_bits;
            state.resources.clear();
            state.resources.push_back(state.src_server);
            state.resources.push_back(server_count_ + state.dst_server);
            for (const int edge_idx : state.path_edges) {
                state.resources.push_back(2 * server_count_ + edge_idx);
            }
            const int port_id = state.path_edges.empty() ? (edge_count_ + state.src_server) : state.path_edges.front();
            state.queue_key = static_cast<long long>(task.tenant) * static_cast<long long>(edge_count_ + server_count_ + 1) + port_id;
            remaining_preds[static_cast<size_t>(task_idx)] = static_cast<int>(task.predecessors.size());
        }

        double current_time = 0.0;
        int slot_idx = 0;
        double contention_potential = 0.0;

        while ((pending_count > 0 || active_count > 0) && slot_idx < max_slots) {
            std::vector<int> newly_ready;
            for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                if (!pending[static_cast<size_t>(task_idx)] || remaining_preds[static_cast<size_t>(task_idx)] != 0) {
                    continue;
                }
                if (release_time(task_idx, runtime, slot_duration) <= current_time + 1e-12) {
                    newly_ready.push_back(task_idx);
                }
            }
            for (const int task_idx : newly_ready) {
                pending[static_cast<size_t>(task_idx)] = 0;
                --pending_count;
                active[static_cast<size_t>(task_idx)] = 1;
                ++active_count;
                if (runtime[static_cast<size_t>(task_idx)].start_time < 0.0) {
                    runtime[static_cast<size_t>(task_idx)].start_time = current_time;
                }
            }

            if (active_count == 0) {
                double next_release = current_time + slot_duration;
                bool found_release = false;
                for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                    if (!pending[static_cast<size_t>(task_idx)] || remaining_preds[static_cast<size_t>(task_idx)] != 0) {
                        continue;
                    }
                    const double candidate = release_time(task_idx, runtime, slot_duration);
                    if (std::isfinite(candidate) && (!found_release || candidate < next_release)) {
                        next_release = candidate;
                        found_release = true;
                    }
                }
                current_time = std::max(current_time + slot_duration, next_release);
                ++slot_idx;
                continue;
            }

            std::vector<int> service_active = service_frontier(active, runtime);
            std::vector<double> normalized_load(static_cast<size_t>(resource_count), 0.0);
            std::vector<int> loaded_resources;
            loaded_resources.reserve(static_cast<size_t>(resource_count));
            auto add_normalized_load = [&](int resource, double value) {
                if (resource < 0 || resource >= resource_count) {
                    return;
                }
                double &slot_load = normalized_load[static_cast<size_t>(resource)];
                if (slot_load == 0.0) {
                    loaded_resources.push_back(resource);
                }
                slot_load += value;
            };
            for (const int task_idx : service_active) {
                const TaskRuntime &state = runtime[static_cast<size_t>(task_idx)];
                const double remaining = state.remaining_bits;
                const double send_cap = server_send_capacity_[static_cast<size_t>(state.src_server)];
                const double recv_cap = server_recv_capacity_[static_cast<size_t>(state.dst_server)];
                add_normalized_load(
                    state.src_server,
                    std::min(remaining, send_cap * slot_duration) / std::max(send_cap * slot_duration, 1e-12)
                );
                add_normalized_load(
                    server_count_ + state.dst_server,
                    std::min(remaining, recv_cap * slot_duration) / std::max(recv_cap * slot_duration, 1e-12)
                );
                for (const int edge_idx : state.path_edges) {
                    const double cap = edge_capacity_[static_cast<size_t>(edge_idx)];
                    add_normalized_load(
                        2 * server_count_ + edge_idx,
                        std::min(remaining, cap * slot_duration) / std::max(cap * slot_duration, 1e-12)
                    );
                }
            }
            for (const int resource : loaded_resources) {
                const double excess = std::max(0.0, normalized_load[static_cast<size_t>(resource)] - 1.0);
                contention_potential += excess * excess;
            }

            std::vector<std::vector<int>> resources_by_item;
            std::vector<double> demand_rates;
            resources_by_item.reserve(service_active.size());
            demand_rates.reserve(service_active.size());
            for (const int task_idx : service_active) {
                const TaskRuntime &state = runtime[static_cast<size_t>(task_idx)];
                resources_by_item.push_back(state.resources);
                demand_rates.push_back(state.remaining_bits / std::max(slot_duration, 1e-12));
            }
            std::vector<double> rates = max_min_rates_native_dense(
                resources_by_item,
                demand_rates,
                resource_capacity,
                resource_count
            );

            std::vector<int> completed;
            for (size_t local_idx = 0; local_idx < service_active.size(); ++local_idx) {
                const int task_idx = service_active[local_idx];
                const double rate = rates[local_idx];
                if (rate <= 0.0) {
                    continue;
                }
                TaskRuntime &state = runtime[static_cast<size_t>(task_idx)];
                const double service_bits = std::min(state.remaining_bits, rate * slot_duration);
                state.remaining_bits -= service_bits;
                if (state.remaining_bits <= 1e-9) {
                    const double finish_time = current_time + service_bits / std::max(rate, 1e-12);
                    state.finish_time = finish_time;
                    completed.push_back(task_idx);
                }
            }

            for (const int task_idx : completed) {
                if (!active[static_cast<size_t>(task_idx)]) {
                    continue;
                }
                active[static_cast<size_t>(task_idx)] = 0;
                --active_count;
                for (const int succ_idx : successors_[static_cast<size_t>(task_idx)]) {
                    remaining_preds[static_cast<size_t>(succ_idx)] -= 1;
                }
            }

            current_time += slot_duration;
            ++slot_idx;
        }

        if (pending_count > 0 || active_count > 0) {
            double penalty_start = current_time;
            for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                TaskRuntime &state = runtime[static_cast<size_t>(task_idx)];
                if (state.finish_time >= 0.0) {
                    continue;
                }
                double bottleneck = std::min(
                    server_send_capacity_[static_cast<size_t>(state.src_server)],
                    server_recv_capacity_[static_cast<size_t>(state.dst_server)]
                );
                for (const int edge_idx : state.path_edges) {
                    bottleneck = std::min(bottleneck, edge_capacity_[static_cast<size_t>(edge_idx)]);
                }
                penalty_start += state.remaining_bits / std::max(bottleneck, 1e-12);
                state.finish_time = penalty_start;
            }
        }

        std::vector<double> tenant_finish(tenant_ids_.size(), 0.0);
        for (int tenant_pos = 0; tenant_pos < static_cast<int>(tasks_by_tenant_.size()); ++tenant_pos) {
            double finish = 0.0;
            for (const int task_idx : tasks_by_tenant_[static_cast<size_t>(tenant_pos)]) {
                finish = std::max(finish, runtime[static_cast<size_t>(task_idx)].finish_time);
            }
            tenant_finish[static_cast<size_t>(tenant_pos)] = finish;
        }
        double makespan = 0.0;
        double sum_finish = 0.0;
        for (const double finish : tenant_finish) {
            makespan = std::max(makespan, finish);
            sum_finish += finish;
        }
        const double avg_jct = sum_finish / std::max<size_t>(tenant_finish.size(), 1);
        const double tiebreak = 1e-9 * contention_potential;
        return py::make_tuple(makespan + tiebreak, avg_jct + tiebreak, makespan, avg_jct, contention_potential, slot_duration);
    }

    py::tuple evaluate_pipeline(const std::vector<std::vector<int>> &server_by_tenant_rank) const {
        const double slot_duration = estimate_slot_duration(server_by_tenant_rank);
        const int max_slots = horizon_slots_override_ > 0
            ? std::max(1, horizon_slots_override_)
            : std::max(512, std::min(16384, 16 * std::max(static_cast<int>(tasks_.size()), 1)));
        const int task_count = static_cast<int>(tasks_.size());

        std::vector<TaskRuntime> runtime(static_cast<size_t>(task_count));
        std::vector<int> remaining_preds(static_cast<size_t>(task_count), 0);
        std::vector<char> pending(static_cast<size_t>(task_count), 1);
        std::vector<char> active(static_cast<size_t>(task_count), 0);
        std::vector<std::vector<double>> buffers(static_cast<size_t>(task_count));
        std::vector<double> delivered(static_cast<size_t>(task_count), 0.0);
        int pending_count = task_count;
        int active_count = 0;
        const int resource_count = 2 * server_count_ + edge_count_;
        std::vector<double> resource_capacity(static_cast<size_t>(resource_count), 0.0);
        for (int server = 0; server < server_count_; ++server) {
            resource_capacity[static_cast<size_t>(server)] = server_send_capacity_[static_cast<size_t>(server)];
            resource_capacity[static_cast<size_t>(server_count_ + server)] = server_recv_capacity_[static_cast<size_t>(server)];
        }
        for (int edge_idx = 0; edge_idx < edge_count_; ++edge_idx) {
            resource_capacity[static_cast<size_t>(2 * server_count_ + edge_idx)] = edge_capacity_[static_cast<size_t>(edge_idx)];
        }

        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
            const TaskDef &task = tasks_[static_cast<size_t>(task_idx)];
            TaskRuntime &state = runtime[static_cast<size_t>(task_idx)];
            state.src_server = server_by_tenant_rank[static_cast<size_t>(task.tenant)][static_cast<size_t>(task.src_rank)];
            state.dst_server = server_by_tenant_rank[static_cast<size_t>(task.tenant)][static_cast<size_t>(task.dst_rank)];
            state.path_edges = path_edges_[static_cast<size_t>(task.tenant)]
                [static_cast<size_t>(state.src_server)]
                [static_cast<size_t>(state.dst_server)];
            state.remaining_bits = task.volume_bits;
            const int port_id = state.path_edges.empty() ? (edge_count_ + state.src_server) : state.path_edges.front();
            state.queue_key = static_cast<long long>(task.tenant) * static_cast<long long>(edge_count_ + server_count_ + 1) + port_id;
            remaining_preds[static_cast<size_t>(task_idx)] = static_cast<int>(task.predecessors.size());
            buffers[static_cast<size_t>(task_idx)].assign(state.path_edges.size(), 0.0);
        }

        struct Item {
            int task_idx;
            int hop_idx;
        };

        auto task_remaining = [&](int task_idx) {
            return std::max(0.0, tasks_[static_cast<size_t>(task_idx)].volume_bits - delivered[static_cast<size_t>(task_idx)]);
        };

        auto item_resources = [&](const Item &item, std::vector<int> &resources) {
            const TaskRuntime &state = runtime[static_cast<size_t>(item.task_idx)];
            const int edge_idx = state.path_edges[static_cast<size_t>(item.hop_idx)];
            resources.clear();
            resources.push_back(2 * server_count_ + edge_idx);
            if (item.hop_idx == 0) {
                resources.push_back(state.src_server);
            }
            if (item.hop_idx == static_cast<int>(state.path_edges.size()) - 1) {
                resources.push_back(server_count_ + state.dst_server);
            }
        };

        auto service_frontier_items = [&](const std::vector<Item> &items) {
            const int queue_count = static_cast<int>(tenant_ids_.size()) * std::max(edge_count_ + 1, 1);
            std::vector<int> best_item(static_cast<size_t>(queue_count), -1);
            std::vector<int> touched_queues;
            touched_queues.reserve(static_cast<size_t>(queue_count));
            for (int item_idx = 0; item_idx < static_cast<int>(items.size()); ++item_idx) {
                const Item &item = items[static_cast<size_t>(item_idx)];
                const TaskDef &task = tasks_[static_cast<size_t>(item.task_idx)];
                const TaskRuntime &state = runtime[static_cast<size_t>(item.task_idx)];
                const int edge_idx = state.path_edges[static_cast<size_t>(item.hop_idx)];
                const int queue_idx = task.tenant * std::max(edge_count_ + 1, 1) + edge_idx;
                if (queue_idx < 0 || queue_idx >= queue_count) {
                    continue;
                }
                const int incumbent_item_idx = best_item[static_cast<size_t>(queue_idx)];
                if (incumbent_item_idx < 0) {
                    best_item[static_cast<size_t>(queue_idx)] = item_idx;
                    touched_queues.push_back(queue_idx);
                    continue;
                }
                const Item &incumbent_item = items[static_cast<size_t>(incumbent_item_idx)];
                const TaskDef &incumbent_task = tasks_[static_cast<size_t>(incumbent_item.task_idx)];
                const TaskRuntime &incumbent_state = runtime[static_cast<size_t>(incumbent_item.task_idx)];
                const auto candidate_key = std::make_tuple(
                    state.start_time < 0.0 ? 0.0 : state.start_time,
                    task.order_pos,
                    task.task_id,
                    item.hop_idx
                );
                const auto incumbent_key = std::make_tuple(
                    incumbent_state.start_time < 0.0 ? 0.0 : incumbent_state.start_time,
                    incumbent_task.order_pos,
                    incumbent_task.task_id,
                    incumbent_item.hop_idx
                );
                if (candidate_key < incumbent_key) {
                    best_item[static_cast<size_t>(queue_idx)] = item_idx;
                }
            }
            std::vector<int> result;
            result.reserve(touched_queues.size());
            for (const int queue_idx : touched_queues) {
                const int item_idx = best_item[static_cast<size_t>(queue_idx)];
                if (item_idx >= 0) {
                    result.push_back(item_idx);
                }
            }
            return result;
        };

        double current_time = 0.0;
        int slot_idx = 0;
        double contention_potential = 0.0;

        while ((pending_count > 0 || active_count > 0) && slot_idx < max_slots) {
            std::vector<int> newly_ready;
            for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                if (!pending[static_cast<size_t>(task_idx)] || remaining_preds[static_cast<size_t>(task_idx)] != 0) {
                    continue;
                }
                if (release_time(task_idx, runtime, slot_duration) <= current_time + 1e-12) {
                    newly_ready.push_back(task_idx);
                }
            }
            for (const int task_idx : newly_ready) {
                pending[static_cast<size_t>(task_idx)] = 0;
                --pending_count;
                TaskRuntime &state = runtime[static_cast<size_t>(task_idx)];
                if (state.start_time < 0.0) {
                    state.start_time = current_time;
                }
                if (state.path_edges.empty()) {
                    state.finish_time = current_time;
                    for (const int succ_idx : successors_[static_cast<size_t>(task_idx)]) {
                        remaining_preds[static_cast<size_t>(succ_idx)] -= 1;
                    }
                    continue;
                }
                buffers[static_cast<size_t>(task_idx)][0] += tasks_[static_cast<size_t>(task_idx)].volume_bits;
                active[static_cast<size_t>(task_idx)] = 1;
                ++active_count;
            }

            if (active_count == 0) {
                double next_release = current_time + slot_duration;
                bool found_release = false;
                for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                    if (!pending[static_cast<size_t>(task_idx)] || remaining_preds[static_cast<size_t>(task_idx)] != 0) {
                        continue;
                    }
                    const double candidate = release_time(task_idx, runtime, slot_duration);
                    if (std::isfinite(candidate) && (!found_release || candidate < next_release)) {
                        next_release = candidate;
                        found_release = true;
                    }
                }
                current_time = std::max(current_time + slot_duration, next_release);
                ++slot_idx;
                continue;
            }

            std::vector<Item> candidate_items;
            std::vector<double> item_available;
            for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                if (!active[static_cast<size_t>(task_idx)]) {
                    continue;
                }
                for (int hop_idx = 0; hop_idx < static_cast<int>(buffers[static_cast<size_t>(task_idx)].size()); ++hop_idx) {
                    const double amount = buffers[static_cast<size_t>(task_idx)][static_cast<size_t>(hop_idx)];
                    if (amount <= 1e-9) {
                        continue;
                    }
                    candidate_items.push_back(Item{task_idx, hop_idx});
                    item_available.push_back(amount);
                }
            }
            std::vector<int> service_item_indices = service_frontier_items(candidate_items);
            std::vector<std::vector<int>> resources_by_item;
            std::vector<double> demand_rates;
            std::vector<double> normalized_load(static_cast<size_t>(resource_count), 0.0);
            std::vector<int> loaded_resources;
            loaded_resources.reserve(static_cast<size_t>(resource_count));
            auto add_normalized_load = [&](int resource, double value) {
                if (resource < 0 || resource >= resource_count) {
                    return;
                }
                double &slot_load = normalized_load[static_cast<size_t>(resource)];
                if (slot_load == 0.0) {
                    loaded_resources.push_back(resource);
                }
                slot_load += value;
            };

            for (const int item_idx : service_item_indices) {
                const Item &item = candidate_items[static_cast<size_t>(item_idx)];
                std::vector<int> resources;
                item_resources(item, resources);
                for (size_t res_idx = 0; res_idx < resources.size(); ++res_idx) {
                    const int resource = resources[res_idx];
                    const double capacity = resource_capacity[static_cast<size_t>(resource)];
                    add_normalized_load(
                        resource,
                        std::min(
                            item_available[static_cast<size_t>(item_idx)],
                            capacity * slot_duration
                        ) / std::max(capacity * slot_duration, 1e-12)
                    );
                }
                resources_by_item.push_back(std::move(resources));
                demand_rates.push_back(item_available[static_cast<size_t>(item_idx)] / std::max(slot_duration, 1e-12));
            }

            for (const int resource : loaded_resources) {
                const double excess = std::max(0.0, normalized_load[static_cast<size_t>(resource)] - 1.0);
                contention_potential += excess * excess;
            }

            std::vector<double> rates = max_min_rates_native_dense(
                resources_by_item,
                demand_rates,
                resource_capacity,
                resource_count
            );
            std::vector<double> task_rate_by_slot(static_cast<size_t>(task_count), 0.0);
            std::vector<std::pair<Item, double>> next_hop_add;
            std::vector<double> delivered_this_slot(static_cast<size_t>(task_count), 0.0);
            for (size_t local_idx = 0; local_idx < service_item_indices.size(); ++local_idx) {
                const int item_idx = service_item_indices[local_idx];
                const Item item = candidate_items[static_cast<size_t>(item_idx)];
                const double rate = rates[local_idx];
                if (rate <= 0.0) {
                    continue;
                }
                double &buffer = buffers[static_cast<size_t>(item.task_idx)][static_cast<size_t>(item.hop_idx)];
                const double service_bits = std::min(buffer, rate * slot_duration);
                if (service_bits <= 0.0) {
                    continue;
                }
                buffer -= service_bits;
                task_rate_by_slot[static_cast<size_t>(item.task_idx)] += service_bits / std::max(slot_duration, 1e-12);
                if (item.hop_idx + 1 < static_cast<int>(buffers[static_cast<size_t>(item.task_idx)].size())) {
                    next_hop_add.push_back({Item{item.task_idx, item.hop_idx + 1}, service_bits});
                } else {
                    delivered_this_slot[static_cast<size_t>(item.task_idx)] += service_bits;
                }
            }

            for (const auto &entry : next_hop_add) {
                buffers[static_cast<size_t>(entry.first.task_idx)][static_cast<size_t>(entry.first.hop_idx)] += entry.second;
            }

            std::vector<int> completed;
            for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                const double amount = delivered_this_slot[static_cast<size_t>(task_idx)];
                if (amount <= 0.0) {
                    continue;
                }
                delivered[static_cast<size_t>(task_idx)] += amount;
                runtime[static_cast<size_t>(task_idx)].remaining_bits = task_remaining(task_idx);
                if (task_remaining(task_idx) <= 1e-6) {
                    const double last_rate = std::max(task_rate_by_slot[static_cast<size_t>(task_idx)], 1e-12);
                    runtime[static_cast<size_t>(task_idx)].finish_time = current_time + std::min(slot_duration, amount / last_rate);
                    completed.push_back(task_idx);
                }
            }

            for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                if (active[static_cast<size_t>(task_idx)]) {
                    runtime[static_cast<size_t>(task_idx)].remaining_bits = task_remaining(task_idx);
                }
            }
            for (const int task_idx : completed) {
                if (!active[static_cast<size_t>(task_idx)]) {
                    continue;
                }
                active[static_cast<size_t>(task_idx)] = 0;
                --active_count;
                for (const int succ_idx : successors_[static_cast<size_t>(task_idx)]) {
                    remaining_preds[static_cast<size_t>(succ_idx)] -= 1;
                }
            }
            current_time += slot_duration;
            ++slot_idx;
        }

        if (pending_count > 0 || active_count > 0) {
            double penalty_start = current_time;
            for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                TaskRuntime &state = runtime[static_cast<size_t>(task_idx)];
                if (state.finish_time >= 0.0) {
                    continue;
                }
                double bottleneck = std::min(
                    server_send_capacity_[static_cast<size_t>(state.src_server)],
                    server_recv_capacity_[static_cast<size_t>(state.dst_server)]
                );
                for (const int edge_idx : state.path_edges) {
                    bottleneck = std::min(bottleneck, edge_capacity_[static_cast<size_t>(edge_idx)]);
                }
                penalty_start += state.remaining_bits / std::max(bottleneck, 1e-12);
                state.finish_time = penalty_start;
            }
        }

        double makespan = 0.0;
        double sum_finish = 0.0;
        for (const auto &tenant_tasks : tasks_by_tenant_) {
            double finish = 0.0;
            for (const int task_idx : tenant_tasks) {
                finish = std::max(finish, runtime[static_cast<size_t>(task_idx)].finish_time);
            }
            makespan = std::max(makespan, finish);
            sum_finish += finish;
        }
        const double avg_jct = sum_finish / std::max<size_t>(tasks_by_tenant_.size(), 1);
        const double tiebreak = 1e-9 * contention_potential;
        return py::make_tuple(makespan + tiebreak, avg_jct + tiebreak, makespan, avg_jct, contention_potential, slot_duration);
    }

private:
    static double slot_aligned_time(double value, double slot_duration) {
        const double safe_slot = std::max(slot_duration, 1e-12);
        return std::ceil(std::max(value, 0.0) / safe_slot - 1e-9) * safe_slot;
    }

    int max_slot_count() const {
        if (horizon_slots_override_ > 0) {
            return std::max(1, horizon_slots_override_);
        }
        const int task_count = static_cast<int>(tasks_.size());
        return std::max(256, std::min(8192, 8 * std::max(task_count, 1)));
    }

    double estimate_slot_duration(const std::vector<std::vector<int>> &server_by_tenant_rank) const {
        if (slot_duration_override_ > 0.0) {
            return std::max(slot_duration_override_, 1e-12);
        }
        std::vector<double> samples;
        for (const TaskDef &task : tasks_) {
            const int src_server = server_by_tenant_rank[static_cast<size_t>(task.tenant)][static_cast<size_t>(task.src_rank)];
            const int dst_server = server_by_tenant_rank[static_cast<size_t>(task.tenant)][static_cast<size_t>(task.dst_rank)];
            double bottleneck = std::min(
                server_send_capacity_[static_cast<size_t>(src_server)],
                server_recv_capacity_[static_cast<size_t>(dst_server)]
            );
            const auto &path = path_edges_[static_cast<size_t>(task.tenant)][static_cast<size_t>(src_server)][static_cast<size_t>(dst_server)];
            for (const int edge_idx : path) {
                bottleneck = std::min(bottleneck, edge_capacity_[static_cast<size_t>(edge_idx)]);
            }
            if (bottleneck > 0.0) {
                samples.push_back(task.volume_bits / bottleneck);
            }
        }
        if (samples.empty()) {
            return 1e-6;
        }
        std::sort(samples.begin(), samples.end());
        const size_t index = static_cast<size_t>(std::max(0, std::min(
            static_cast<int>(samples.size()) - 1,
            static_cast<int>(0.25 * static_cast<double>(samples.size() - 1))
        )));
        const double p25 = samples[index];
        const double smallest = samples.front();
        return std::max(std::min(p25 / 2.0, smallest), 1e-9);
    }

    double release_time(int task_idx, const std::vector<TaskRuntime> &runtime, double slot_duration) const {
        const TaskDef &task = tasks_[static_cast<size_t>(task_idx)];
        double release = slot_aligned_time(tenant_start_times_[static_cast<size_t>(task.tenant)], slot_duration);
        if (!task.release_prev.empty()) {
            double previous_release = 0.0;
            bool first = true;
            for (const int prev_idx : task.release_prev) {
                const double finish = runtime[static_cast<size_t>(prev_idx)].finish_time;
                if (finish < 0.0) {
                    return std::numeric_limits<double>::infinity();
                }
                const double aligned = slot_aligned_time(finish, slot_duration);
                previous_release = first ? aligned : std::max(previous_release, aligned);
                first = false;
            }
            const int gap_slots = static_cast<int>(std::ceil(task.release_gap / std::max(slot_duration, 1e-12) - 1e-9));
            release = std::max(release, previous_release + std::max(gap_slots, 0) * slot_duration);
        } else if (task.release_gap > 0.0) {
            release = std::max(release, slot_aligned_time(task.release_gap, slot_duration));
        }
        return release;
    }

    std::vector<int> service_frontier(
        const std::vector<char> &active,
        const std::vector<TaskRuntime> &runtime
    ) const {
        const int queue_count = static_cast<int>(tenant_ids_.size()) * std::max(edge_count_ + server_count_ + 1, 1);
        std::vector<int> best_task(static_cast<size_t>(queue_count), -1);
        std::vector<int> touched_queues;
        touched_queues.reserve(static_cast<size_t>(queue_count));
        for (int task_idx = 0; task_idx < static_cast<int>(active.size()); ++task_idx) {
            if (!active[static_cast<size_t>(task_idx)]) {
                continue;
            }
            const TaskDef &task = tasks_[static_cast<size_t>(task_idx)];
            const TaskRuntime &state = runtime[static_cast<size_t>(task_idx)];
            const int queue_idx = static_cast<int>(state.queue_key);
            if (queue_idx < 0 || queue_idx >= queue_count) {
                continue;
            }
            const int incumbent_task_idx = best_task[static_cast<size_t>(queue_idx)];
            if (incumbent_task_idx < 0) {
                best_task[static_cast<size_t>(queue_idx)] = task_idx;
                touched_queues.push_back(queue_idx);
                continue;
            }
            const TaskDef &incumbent_task = tasks_[static_cast<size_t>(incumbent_task_idx)];
            const TaskRuntime &incumbent_state = runtime[static_cast<size_t>(incumbent_task_idx)];
            const auto candidate_key = std::make_tuple(
                state.start_time < 0.0 ? 0.0 : state.start_time,
                task.order_pos,
                task.task_id
            );
            const auto incumbent_key = std::make_tuple(
                incumbent_state.start_time < 0.0 ? 0.0 : incumbent_state.start_time,
                incumbent_task.order_pos,
                incumbent_task.task_id
            );
            if (candidate_key < incumbent_key) {
                best_task[static_cast<size_t>(queue_idx)] = task_idx;
            }
        }
        std::vector<int> result;
        result.reserve(touched_queues.size());
        for (const int queue_idx : touched_queues) {
            const int task_idx = best_task[static_cast<size_t>(queue_idx)];
            if (task_idx >= 0) {
                result.push_back(task_idx);
            }
        }
        return result;
    }

    std::vector<int> tenant_ids_;
    std::vector<double> tenant_start_times_;
    std::vector<double> server_send_capacity_;
    std::vector<double> server_recv_capacity_;
    std::vector<double> edge_capacity_;
    std::vector<std::vector<std::vector<std::vector<int>>>> path_edges_;
    std::vector<TaskDef> tasks_;
    std::vector<std::vector<int>> successors_;
    std::vector<std::vector<int>> tasks_by_tenant_;
    double slot_duration_override_;
    int horizon_slots_override_;
    int server_count_ = 0;
    int edge_count_ = 0;
};

py::object best_rank_swap_delta(
    const py::list &flows,
    const py::dict &rank_to_server,
    const py::list &branch_order,
    int anchor_limit,
    int partner_limit,
    const py::dict &pair_epoch_price
) {
    struct Flow {
        int epoch;
        int src_rank;
        int dst_rank;
        double volume;
    };

    std::vector<int> ranks;
    ranks.reserve(static_cast<size_t>(branch_order.size()));
    for (py::handle rank_handle : branch_order) {
        ranks.push_back(py::cast<int>(rank_handle));
    }

    std::unordered_map<int, int> server_by_rank;
    for (auto item : rank_to_server) {
        server_by_rank.emplace(py::cast<int>(item.first), py::cast<int>(item.second));
    }

    std::vector<Flow> flow_vec;
    flow_vec.reserve(static_cast<size_t>(flows.size()));
    std::unordered_map<int, std::vector<int>> incidence;
    for (py::ssize_t idx = 0; idx < flows.size(); ++idx) {
        py::tuple entry = py::reinterpret_borrow<py::tuple>(flows[idx]);
        Flow flow{
            py::cast<int>(entry[0]),
            py::cast<int>(entry[1]),
            py::cast<int>(entry[2]),
            py::cast<double>(entry[3]),
        };
        const int flow_idx = static_cast<int>(flow_vec.size());
        incidence[flow.src_rank].push_back(flow_idx);
        incidence[flow.dst_rank].push_back(flow_idx);
        flow_vec.push_back(flow);
    }

    const int effective_anchor_limit = std::max(0, std::min(anchor_limit, static_cast<int>(ranks.size())));
    const int effective_partner_limit = std::max(0, std::min(partner_limit, static_cast<int>(ranks.size())));
    double best_delta = 0.0;
    int best_left = -1;
    int best_right = -1;
    std::unordered_map<long long, char> tried;
    const auto price_lookup = build_price_lookup(pair_epoch_price);

    auto price = [&](int epoch, int src_server, int dst_server) -> double {
        return price_lookup.at(PriceKey{epoch, src_server, dst_server});
    };

    for (int left_idx = 0; left_idx < effective_anchor_limit; ++left_idx) {
        const int left_rank = ranks[static_cast<size_t>(left_idx)];
        for (int right_idx = 0; right_idx < effective_partner_limit; ++right_idx) {
            const int right_rank = ranks[static_cast<size_t>(right_idx)];
            if (left_rank == right_rank) {
                continue;
            }
            const int a = std::min(left_rank, right_rank);
            const int b = std::max(left_rank, right_rank);
            const long long pair_key = (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
            if (tried.find(pair_key) != tried.end()) {
                continue;
            }
            tried[pair_key] = 1;

            std::vector<int> affected;
            std::unordered_map<int, char> affected_seen;
            for (const int rank : {a, b}) {
                auto it = incidence.find(rank);
                if (it == incidence.end()) {
                    continue;
                }
                for (const int flow_idx : it->second) {
                    if (affected_seen.find(flow_idx) == affected_seen.end()) {
                        affected_seen[flow_idx] = 1;
                        affected.push_back(flow_idx);
                    }
                }
            }

            const int server_a = server_by_rank[a];
            const int server_b = server_by_rank[b];
            double delta = 0.0;
            for (const int flow_idx : affected) {
                const Flow &flow = flow_vec[static_cast<size_t>(flow_idx)];
                const int old_src = server_by_rank[flow.src_rank];
                const int old_dst = server_by_rank[flow.dst_rank];
                int new_src = old_src;
                int new_dst = old_dst;
                if (flow.src_rank == a) {
                    new_src = server_b;
                } else if (flow.src_rank == b) {
                    new_src = server_a;
                }
                if (flow.dst_rank == a) {
                    new_dst = server_b;
                } else if (flow.dst_rank == b) {
                    new_dst = server_a;
                }
                delta += flow.volume * (
                    price(flow.epoch, new_src, new_dst) - price(flow.epoch, old_src, old_dst)
                );
            }
            if (delta < best_delta - 1e-12) {
                best_delta = delta;
                best_left = a;
                best_right = b;
            }
        }
    }

    if (best_left < 0) {
        return py::none();
    }
    return py::make_tuple(best_delta, best_left, best_right);
}

py::object best_coarse_move_delta(
    const py::list &flows,
    const py::dict &rank_to_server,
    const py::list &branch_order,
    int anchor_limit,
    int partner_limit,
    const py::list &block_groups,
    const py::dict &pair_epoch_price
) {
    struct Flow {
        int epoch;
        int src_rank;
        int dst_rank;
        double volume;
    };

    std::vector<int> ranks;
    ranks.reserve(static_cast<size_t>(branch_order.size()));
    for (py::handle rank_handle : branch_order) {
        ranks.push_back(py::cast<int>(rank_handle));
    }

    std::unordered_map<int, int> server_by_rank;
    for (auto item : rank_to_server) {
        server_by_rank.emplace(py::cast<int>(item.first), py::cast<int>(item.second));
    }

    std::vector<Flow> flow_vec;
    flow_vec.reserve(static_cast<size_t>(flows.size()));
    std::unordered_map<int, std::vector<int>> incidence;
    for (py::ssize_t idx = 0; idx < flows.size(); ++idx) {
        py::tuple entry = py::reinterpret_borrow<py::tuple>(flows[idx]);
        Flow flow{
            py::cast<int>(entry[0]),
            py::cast<int>(entry[1]),
            py::cast<int>(entry[2]),
            py::cast<double>(entry[3]),
        };
        const int flow_idx = static_cast<int>(flow_vec.size());
        incidence[flow.src_rank].push_back(flow_idx);
        incidence[flow.dst_rank].push_back(flow_idx);
        flow_vec.push_back(flow);
    }

    const auto price_lookup = build_price_lookup(pair_epoch_price);

    auto price = [&](int epoch, int src_server, int dst_server) -> double {
        return price_lookup.at(PriceKey{epoch, src_server, dst_server});
    };

    auto delta_for_reassignment = [&](const std::unordered_map<int, int> &new_server_by_rank) -> double {
        std::vector<int> affected;
        std::unordered_map<int, char> affected_seen;
        for (const auto &entry : new_server_by_rank) {
            auto it = incidence.find(entry.first);
            if (it == incidence.end()) {
                continue;
            }
            for (const int flow_idx : it->second) {
                if (affected_seen.find(flow_idx) == affected_seen.end()) {
                    affected_seen[flow_idx] = 1;
                    affected.push_back(flow_idx);
                }
            }
        }

        double delta = 0.0;
        for (const int flow_idx : affected) {
            const Flow &flow = flow_vec[static_cast<size_t>(flow_idx)];
            const int old_src = server_by_rank[flow.src_rank];
            const int old_dst = server_by_rank[flow.dst_rank];
            int new_src = old_src;
            int new_dst = old_dst;
            auto src_it = new_server_by_rank.find(flow.src_rank);
            if (src_it != new_server_by_rank.end()) {
                new_src = src_it->second;
            }
            auto dst_it = new_server_by_rank.find(flow.dst_rank);
            if (dst_it != new_server_by_rank.end()) {
                new_dst = dst_it->second;
            }
            delta += flow.volume * (
                price(flow.epoch, new_src, new_dst) - price(flow.epoch, old_src, old_dst)
            );
        }
        return delta;
    };

    double best_delta = 0.0;
    std::unordered_map<int, int> best_move;
    const int effective_anchor_limit = std::max(0, std::min(anchor_limit, static_cast<int>(ranks.size())));
    const int effective_partner_limit = std::max(0, std::min(partner_limit, static_cast<int>(ranks.size())));
    std::unordered_map<long long, char> tried;

    for (int left_idx = 0; left_idx < effective_anchor_limit; ++left_idx) {
        const int left_rank = ranks[static_cast<size_t>(left_idx)];
        for (int right_idx = 0; right_idx < effective_partner_limit; ++right_idx) {
            const int right_rank = ranks[static_cast<size_t>(right_idx)];
            if (left_rank == right_rank) {
                continue;
            }
            const int a = std::min(left_rank, right_rank);
            const int b = std::max(left_rank, right_rank);
            const long long pair_key = (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
            if (tried.find(pair_key) != tried.end()) {
                continue;
            }
            tried[pair_key] = 1;
            std::unordered_map<int, int> reassignment{
                {a, server_by_rank[b]},
                {b, server_by_rank[a]},
            };
            const double delta = delta_for_reassignment(reassignment);
            if (delta < best_delta - 1e-12) {
                best_delta = delta;
                best_move = std::move(reassignment);
            }
        }
    }

    for (py::handle group_handle : block_groups) {
        py::sequence group_seq = py::reinterpret_borrow<py::sequence>(group_handle);
        const int group_size = static_cast<int>(group_seq.size());
        if (group_size < 3) {
            continue;
        }
        std::vector<int> group;
        group.reserve(static_cast<size_t>(group_size));
        for (py::ssize_t idx = 0; idx < group_seq.size(); ++idx) {
            group.push_back(py::cast<int>(group_seq[idx]));
        }
        for (int shift = 1; shift < group_size; ++shift) {
            std::unordered_map<int, int> reassignment;
            for (int idx = 0; idx < group_size; ++idx) {
                const int rank = group[static_cast<size_t>(idx)];
                const int donor_rank = group[static_cast<size_t>((idx + shift) % group_size)];
                reassignment[rank] = server_by_rank[donor_rank];
            }
            const double delta = delta_for_reassignment(reassignment);
            if (delta < best_delta - 1e-12) {
                best_delta = delta;
                best_move = std::move(reassignment);
            }
        }
    }

    if (best_move.empty()) {
        return py::none();
    }
    py::dict result;
    for (const auto &entry : best_move) {
        result[py::int_(entry.first)] = py::int_(entry.second);
    }
    return py::make_tuple(best_delta, result);
}

py::object coarse_local_descent(
    const py::list &flows,
    const py::dict &rank_to_server,
    const py::list &branch_order,
    int anchor_limit,
    int partner_limit,
    const py::list &block_groups,
    const py::dict &pair_epoch_price,
    int passes
) {
    struct Flow {
        int epoch;
        int src_rank;
        int dst_rank;
        double volume;
    };

    std::vector<int> ranks;
    ranks.reserve(static_cast<size_t>(branch_order.size()));
    for (py::handle rank_handle : branch_order) {
        ranks.push_back(py::cast<int>(rank_handle));
    }

    std::unordered_map<int, int> server_by_rank;
    for (auto item : rank_to_server) {
        server_by_rank.emplace(py::cast<int>(item.first), py::cast<int>(item.second));
    }

    std::vector<Flow> flow_vec;
    flow_vec.reserve(static_cast<size_t>(flows.size()));
    std::unordered_map<int, std::vector<int>> incidence;
    for (py::ssize_t idx = 0; idx < flows.size(); ++idx) {
        py::tuple entry = py::reinterpret_borrow<py::tuple>(flows[idx]);
        Flow flow{
            py::cast<int>(entry[0]),
            py::cast<int>(entry[1]),
            py::cast<int>(entry[2]),
            py::cast<double>(entry[3]),
        };
        const int flow_idx = static_cast<int>(flow_vec.size());
        incidence[flow.src_rank].push_back(flow_idx);
        incidence[flow.dst_rank].push_back(flow_idx);
        flow_vec.push_back(flow);
    }

    std::vector<std::vector<int>> groups;
    groups.reserve(static_cast<size_t>(block_groups.size()));
    for (py::handle group_handle : block_groups) {
        py::sequence group_seq = py::reinterpret_borrow<py::sequence>(group_handle);
        std::vector<int> group;
        group.reserve(static_cast<size_t>(group_seq.size()));
        for (py::ssize_t idx = 0; idx < group_seq.size(); ++idx) {
            group.push_back(py::cast<int>(group_seq[idx]));
        }
        groups.push_back(std::move(group));
    }

    const auto price_lookup = build_price_lookup(pair_epoch_price);
    auto price = [&](int epoch, int src_server, int dst_server) -> double {
        return price_lookup.at(PriceKey{epoch, src_server, dst_server});
    };

    auto delta_for_reassignment = [&](const std::unordered_map<int, int> &new_server_by_rank) -> double {
        std::vector<int> affected;
        std::unordered_map<int, char> affected_seen;
        for (const auto &entry : new_server_by_rank) {
            auto it = incidence.find(entry.first);
            if (it == incidence.end()) {
                continue;
            }
            for (const int flow_idx : it->second) {
                if (affected_seen.find(flow_idx) == affected_seen.end()) {
                    affected_seen[flow_idx] = 1;
                    affected.push_back(flow_idx);
                }
            }
        }

        double delta = 0.0;
        for (const int flow_idx : affected) {
            const Flow &flow = flow_vec[static_cast<size_t>(flow_idx)];
            const int old_src = server_by_rank[flow.src_rank];
            const int old_dst = server_by_rank[flow.dst_rank];
            int new_src = old_src;
            int new_dst = old_dst;
            auto src_it = new_server_by_rank.find(flow.src_rank);
            if (src_it != new_server_by_rank.end()) {
                new_src = src_it->second;
            }
            auto dst_it = new_server_by_rank.find(flow.dst_rank);
            if (dst_it != new_server_by_rank.end()) {
                new_dst = dst_it->second;
            }
            delta += flow.volume * (
                price(flow.epoch, new_src, new_dst) - price(flow.epoch, old_src, old_dst)
            );
        }
        return delta;
    };

    const int effective_anchor_limit = std::max(0, std::min(anchor_limit, static_cast<int>(ranks.size())));
    const int effective_partner_limit = std::max(0, std::min(partner_limit, static_cast<int>(ranks.size())));
    bool changed = false;

    for (int pass_idx = 0; pass_idx < std::max(0, passes); ++pass_idx) {
        double best_delta = 0.0;
        std::unordered_map<int, int> best_move;
        std::unordered_map<long long, char> tried;

        for (int left_idx = 0; left_idx < effective_anchor_limit; ++left_idx) {
            const int left_rank = ranks[static_cast<size_t>(left_idx)];
            for (int right_idx = 0; right_idx < effective_partner_limit; ++right_idx) {
                const int right_rank = ranks[static_cast<size_t>(right_idx)];
                if (left_rank == right_rank) {
                    continue;
                }
                const int a = std::min(left_rank, right_rank);
                const int b = std::max(left_rank, right_rank);
                const long long pair_key = (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
                if (tried.find(pair_key) != tried.end()) {
                    continue;
                }
                tried[pair_key] = 1;
                std::unordered_map<int, int> reassignment{
                    {a, server_by_rank[b]},
                    {b, server_by_rank[a]},
                };
                const double delta = delta_for_reassignment(reassignment);
                if (delta < best_delta - 1e-12) {
                    best_delta = delta;
                    best_move = std::move(reassignment);
                }
            }
        }

        for (const auto &group : groups) {
            const int group_size = static_cast<int>(group.size());
            if (group_size < 3) {
                continue;
            }
            std::vector<int> current_servers;
            current_servers.reserve(static_cast<size_t>(group_size));
            for (const int rank : group) {
                current_servers.push_back(server_by_rank[rank]);
            }
            for (int shift = 1; shift < group_size; ++shift) {
                std::unordered_map<int, int> reassignment;
                for (int idx = 0; idx < group_size; ++idx) {
                    reassignment[group[static_cast<size_t>(idx)]] =
                        current_servers[static_cast<size_t>((idx + shift) % group_size)];
                }
                const double delta = delta_for_reassignment(reassignment);
                if (delta < best_delta - 1e-12) {
                    best_delta = delta;
                    best_move = std::move(reassignment);
                }
            }
        }

        if (best_move.empty()) {
            break;
        }
        for (const auto &entry : best_move) {
            server_by_rank[entry.first] = entry.second;
        }
        changed = true;
    }

    if (!changed) {
        return py::none();
    }
    py::dict result;
    for (const auto &entry : server_by_rank) {
        result[py::int_(entry.first)] = py::int_(entry.second);
    }
    return result;
}

py::dict task_pair_price_lookup_dense(
    const std::vector<int> &task_ids,
    const py::list &exposure_slots_by_task,
    const std::vector<int> &candidate_servers,
    const std::vector<std::vector<double>> &sender_prices,
    const std::vector<std::vector<double>> &receiver_prices,
    const std::vector<std::vector<double>> &edge_prices,
    const std::vector<std::vector<std::vector<int>>> &path_edges_by_pair
) {
    py::dict result;
    const int slot_count = static_cast<int>(sender_prices.size());
    for (size_t task_pos = 0; task_pos < task_ids.size(); ++task_pos) {
        std::vector<int> exposure_slots = py::cast<std::vector<int>>(exposure_slots_by_task[task_pos]);
        if (exposure_slots.empty()) {
            exposure_slots.push_back(0);
        }
        const int task_id = task_ids[task_pos];
        for (const int src_server : candidate_servers) {
            for (const int dst_server : candidate_servers) {
                if (src_server == dst_server) {
                    continue;
                }
                double price_sum = 0.0;
                for (const int raw_slot : exposure_slots) {
                    const int slot = std::max(0, std::min(raw_slot, slot_count - 1));
                    double price = 0.0;
                    if (slot_count > 0) {
                        price = std::max(price, sender_prices[static_cast<size_t>(slot)][static_cast<size_t>(src_server)]);
                        price = std::max(price, receiver_prices[static_cast<size_t>(slot)][static_cast<size_t>(dst_server)]);
                        const auto &path = path_edges_by_pair[static_cast<size_t>(src_server)][static_cast<size_t>(dst_server)];
                        for (const int edge_idx : path) {
                            price = std::max(price, edge_prices[static_cast<size_t>(slot)][static_cast<size_t>(edge_idx)]);
                        }
                    }
                    price_sum += price;
                }
                result[py::make_tuple(task_id, src_server, dst_server)] = price_sum / std::max<size_t>(exposure_slots.size(), 1);
            }
        }
    }
    return result;
}

struct SwapTaskInfo {
    int task_id = 0;
    int src_rank = 0;
    int dst_rank = 0;
    double volume = 0.0;
};

struct SwapScore {
    double delta = 0.0;
    int left_rank = 0;
    int right_rank = 0;
};

struct SwapTaskPairPriceKey {
    int task_id;
    int src_server;
    int dst_server;

    bool operator==(const SwapTaskPairPriceKey &other) const {
        return task_id == other.task_id
            && src_server == other.src_server
            && dst_server == other.dst_server;
    }
};

struct SwapTaskPairPriceKeyHash {
    std::size_t operator()(const SwapTaskPairPriceKey &key) const {
        std::size_t value = static_cast<std::size_t>(key.task_id);
        value ^= static_cast<std::size_t>(key.src_server) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        value ^= static_cast<std::size_t>(key.dst_server) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        return value;
    }
};

py::list scored_swap_pairs_from_state(
    const std::vector<int> &hot_ranks,
    const std::vector<int> &partner_ranks,
    const py::list &task_infos_py,
    const py::dict &tenant_mapping,
    const py::dict &task_active_slots,
    const py::dict &task_ready_slots,
    int tenant,
    const py::list &slot_prices,
    const std::vector<double> &base_sender,
    const std::vector<double> &base_receiver,
    const std::vector<double> &base_edge,
    const py::dict &edge_to_idx,
    const std::vector<std::vector<std::vector<int>>> &path_edges_by_pair
) {
    const int slot_count = std::max<int>(static_cast<int>(slot_prices.size()), 1);
    const int server_count = static_cast<int>(base_sender.size());
    if (base_receiver.size() != base_sender.size()) {
        throw std::invalid_argument("base sender/receiver price vectors must have matching lengths");
    }

    std::vector<std::vector<double>> sender_prices(static_cast<size_t>(slot_count), base_sender);
    std::vector<std::vector<double>> receiver_prices(static_cast<size_t>(slot_count), base_receiver);
    std::vector<std::vector<double>> edge_prices(static_cast<size_t>(slot_count), base_edge);

    for (int slot_idx = 0; slot_idx < static_cast<int>(slot_prices.size()); ++slot_idx) {
        py::dict price_state = py::reinterpret_borrow<py::dict>(slot_prices[slot_idx]);
        if (price_state.contains("sender")) {
            py::dict sender = py::reinterpret_borrow<py::dict>(price_state["sender"]);
            for (auto item : sender) {
                const int server = py::cast<int>(item.first);
                if (0 <= server && server < server_count) {
                    sender_prices[static_cast<size_t>(slot_idx)][static_cast<size_t>(server)] =
                        py::cast<double>(item.second);
                }
            }
        }
        if (price_state.contains("receiver")) {
            py::dict receiver = py::reinterpret_borrow<py::dict>(price_state["receiver"]);
            for (auto item : receiver) {
                const int server = py::cast<int>(item.first);
                if (0 <= server && server < server_count) {
                    receiver_prices[static_cast<size_t>(slot_idx)][static_cast<size_t>(server)] =
                        py::cast<double>(item.second);
                }
            }
        }
        if (price_state.contains("edge")) {
            py::dict edge = py::reinterpret_borrow<py::dict>(price_state["edge"]);
            for (auto item : edge) {
                py::object edge_key = py::reinterpret_borrow<py::object>(item.first);
                if (!edge_to_idx.contains(edge_key)) {
                    continue;
                }
                const int edge_idx = py::cast<int>(edge_to_idx[edge_key]);
                if (0 <= edge_idx && edge_idx < static_cast<int>(base_edge.size())) {
                    edge_prices[static_cast<size_t>(slot_idx)][static_cast<size_t>(edge_idx)] =
                        py::cast<double>(item.second);
                }
            }
        }
    }

    std::unordered_map<int, int> server_by_rank;
    server_by_rank.reserve(static_cast<size_t>(tenant_mapping.size()));
    for (auto item : tenant_mapping) {
        server_by_rank.emplace(py::cast<int>(item.first), py::cast<int>(item.second));
    }

    std::vector<SwapTaskInfo> tasks;
    tasks.reserve(static_cast<size_t>(task_infos_py.size()));
    std::unordered_map<int, std::vector<int>> incidence;
    for (py::handle item_handle : task_infos_py) {
        py::tuple item = py::reinterpret_borrow<py::tuple>(item_handle);
        if (item.size() < 4) {
            throw std::invalid_argument("task_infos entries must be (task_id, src_rank, dst_rank, volume)");
        }
        SwapTaskInfo info;
        info.task_id = py::cast<int>(item[0]);
        info.src_rank = py::cast<int>(item[1]);
        info.dst_rank = py::cast<int>(item[2]);
        info.volume = py::cast<double>(item[3]);
        const int task_index = static_cast<int>(tasks.size());
        tasks.push_back(info);
        incidence[info.src_rank].push_back(task_index);
        incidence[info.dst_rank].push_back(task_index);
    }

    std::vector<std::vector<int>> exposure_slots_by_task(tasks.size());
    for (size_t task_index = 0; task_index < tasks.size(); ++task_index) {
        const int task_id = tasks[task_index].task_id;
        std::unordered_set<int> seen_slots;
        auto append_slots = [&](const py::dict &slot_map) {
            py::tuple key = py::make_tuple(tenant, task_id);
            if (!slot_map.contains(key)) {
                return;
            }
            py::object slot_object = py::reinterpret_borrow<py::object>(slot_map[key]);
            for (py::handle slot_handle : slot_object) {
                const int slot = py::cast<int>(slot_handle);
                if (seen_slots.insert(slot).second) {
                    exposure_slots_by_task[task_index].push_back(slot);
                }
            }
        };
        append_slots(task_active_slots);
        append_slots(task_ready_slots);
        if (exposure_slots_by_task[task_index].empty()) {
            exposure_slots_by_task[task_index].push_back(0);
        }
    }

    std::unordered_map<SwapTaskPairPriceKey, double, SwapTaskPairPriceKeyHash> price_cache;
    auto price_for = [&](int task_index, int src_server, int dst_server) -> double {
        if (src_server == dst_server) {
            return 0.0;
        }
        const int task_id = tasks[static_cast<size_t>(task_index)].task_id;
        SwapTaskPairPriceKey key{task_id, src_server, dst_server};
        const auto cached = price_cache.find(key);
        if (cached != price_cache.end()) {
            return cached->second;
        }
        double price_sum = 0.0;
        for (const int raw_slot : exposure_slots_by_task[static_cast<size_t>(task_index)]) {
            const int slot = std::max(0, std::min(raw_slot, slot_count - 1));
            double price = 0.0;
            price = std::max(
                price,
                sender_prices[static_cast<size_t>(slot)][static_cast<size_t>(src_server)]
            );
            price = std::max(
                price,
                receiver_prices[static_cast<size_t>(slot)][static_cast<size_t>(dst_server)]
            );
            const auto &path = path_edges_by_pair[static_cast<size_t>(src_server)][static_cast<size_t>(dst_server)];
            for (const int edge_idx : path) {
                price = std::max(
                    price,
                    edge_prices[static_cast<size_t>(slot)][static_cast<size_t>(edge_idx)]
                );
            }
            price_sum += price;
        }
        const double value = price_sum / std::max<size_t>(
            exposure_slots_by_task[static_cast<size_t>(task_index)].size(),
            1
        );
        price_cache.emplace(key, value);
        return value;
    };

    std::vector<SwapScore> scored;
    std::unordered_set<std::uint64_t> seen_pairs;
    auto pair_key = [](int left, int right) -> std::uint64_t {
        const std::uint32_t a = static_cast<std::uint32_t>(std::min(left, right));
        const std::uint32_t b = static_cast<std::uint32_t>(std::max(left, right));
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    };

    for (const int raw_left_rank : hot_ranks) {
        for (const int raw_right_rank : partner_ranks) {
            if (raw_left_rank == raw_right_rank) {
                continue;
            }
            const int left_rank = std::min(raw_left_rank, raw_right_rank);
            const int right_rank = std::max(raw_left_rank, raw_right_rank);
            const std::uint64_t seen_key = pair_key(left_rank, right_rank);
            if (!seen_pairs.insert(seen_key).second) {
                continue;
            }
            const int left_server = server_by_rank.at(left_rank);
            const int right_server = server_by_rank.at(right_rank);

            std::unordered_set<int> affected;
            if (incidence.count(left_rank) != 0U) {
                for (const int task_index : incidence[left_rank]) {
                    affected.insert(task_index);
                }
            }
            if (incidence.count(right_rank) != 0U) {
                for (const int task_index : incidence[right_rank]) {
                    affected.insert(task_index);
                }
            }

            auto swapped_server = [&](int rank) -> int {
                if (rank == left_rank) {
                    return right_server;
                }
                if (rank == right_rank) {
                    return left_server;
                }
                return server_by_rank.at(rank);
            };

            double delta = 0.0;
            for (const int task_index : affected) {
                const SwapTaskInfo &task = tasks[static_cast<size_t>(task_index)];
                const int old_src = server_by_rank.at(task.src_rank);
                const int old_dst = server_by_rank.at(task.dst_rank);
                const int new_src = swapped_server(task.src_rank);
                const int new_dst = swapped_server(task.dst_rank);
                const double old_cost = price_for(task_index, old_src, old_dst);
                const double new_cost = price_for(task_index, new_src, new_dst);
                delta += task.volume * (new_cost - old_cost);
            }
            scored.push_back(SwapScore{delta, left_rank, right_rank});
        }
    }

    std::sort(scored.begin(), scored.end(), [](const SwapScore &lhs, const SwapScore &rhs) {
        if (lhs.delta != rhs.delta) {
            return lhs.delta < rhs.delta;
        }
        if (lhs.left_rank != rhs.left_rank) {
            return lhs.left_rank < rhs.left_rank;
        }
        return lhs.right_rank < rhs.right_rank;
    });

    py::list result;
    for (const SwapScore &item : scored) {
        result.append(py::make_tuple(item.delta, py::make_tuple(item.left_rank, item.right_rank)));
    }
    return result;
}

struct RemapTaskInfo {
    int task_id = 0;
    int src_rank = 0;
    int dst_rank = 0;
    int src_pos = 0;
    int dst_pos = 0;
    double volume = 0.0;
};

struct TaskPairPriceKey {
    int task_id;
    int src_server;
    int dst_server;

    bool operator==(const TaskPairPriceKey &other) const {
        return task_id == other.task_id
            && src_server == other.src_server
            && dst_server == other.dst_server;
    }
};

struct TaskPairPriceKeyHash {
    std::size_t operator()(const TaskPairPriceKey &key) const {
        std::size_t value = static_cast<std::size_t>(key.task_id);
        value ^= static_cast<std::size_t>(key.src_server) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        value ^= static_cast<std::size_t>(key.dst_server) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        return value;
    }
};

struct RemapMinKey {
    int mode;
    int task_id;
    int fixed_server;
    std::uint64_t mask;

    bool operator==(const RemapMinKey &other) const {
        return mode == other.mode
            && task_id == other.task_id
            && fixed_server == other.fixed_server
            && mask == other.mask;
    }
};

struct RemapMinKeyHash {
    std::size_t operator()(const RemapMinKey &key) const {
        std::size_t value = static_cast<std::size_t>(key.task_id);
        value ^= static_cast<std::size_t>(key.mode) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        value ^= static_cast<std::size_t>(key.fixed_server) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        value ^= static_cast<std::size_t>(key.mask) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        return value;
    }
};

struct RemapCandidate {
    double cost = 0.0;
    std::vector<int> assignment;
};

struct IntVectorHash {
    std::size_t operator()(const std::vector<int> &values) const {
        std::size_t hash = 0;
        for (const int value : values) {
            hash ^= static_cast<std::size_t>(value) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct SlotTaskInfo {
    int src_server = 0;
    int dst_server = 0;
    std::vector<py::object> path_edges;
};

struct ResourceKey {
    int type = 0;  // 0=edge, 1=sender, 2=receiver.
    std::string id;

    bool operator==(const ResourceKey &other) const {
        return type == other.type && id == other.id;
    }
};

struct ResourceKeyHash {
    std::size_t operator()(const ResourceKey &key) const {
        std::size_t value = static_cast<std::size_t>(key.type);
        value ^= std::hash<std::string>{}(key.id) + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
        return value;
    }
};

struct EdgeServiceAccum {
    py::object key = py::none();
    double service = 0.0;
};

long long pack_task_tuple(const py::handle &task_key_handle) {
    py::tuple task_key = py::reinterpret_borrow<py::tuple>(task_key_handle);
    const int tenant = py::cast<int>(task_key[0]);
    const int task_id = py::cast<int>(task_key[1]);
    return (static_cast<long long>(tenant) << 32) ^ static_cast<unsigned int>(task_id);
}

ResourceKey parse_resource_key(const py::handle &resource_handle) {
    py::tuple resource = py::reinterpret_borrow<py::tuple>(resource_handle);
    const std::string type = py::cast<std::string>(resource[0]);
    const std::string id = py::cast<std::string>(py::str(resource[1]));
    if (type == "edge") {
        return ResourceKey{0, id};
    }
    if (type == "sender") {
        return ResourceKey{1, id};
    }
    return ResourceKey{2, id};
}

double resource_price_value(double capacity, double normalized_load, double beta, double gamma) {
    const double base_cost = 1.0 / std::max(capacity, 1e-12);
    return base_cost * (1.0 + beta * std::pow(static_cast<double>(normalized_load), gamma));
}

py::list time_expanded_slot_prices_batch(
    const py::list &slot_resource_pressure,
    const py::list &slot_active_tasks,
    const py::list &slot_service_rates,
    const py::dict &task_state,
    const py::list &critical_resources_by_slot,
    const py::dict &edge_capacity,
    const std::vector<double> &server_send_capacity,
    const std::vector<double> &server_recv_capacity,
    double link_price_beta,
    double link_price_gamma,
    double critical_path_price_beta
) {
    std::unordered_map<long long, SlotTaskInfo> task_info;
    task_info.reserve(static_cast<size_t>(task_state.size()));
    for (auto item : task_state) {
        const long long key = pack_task_tuple(item.first);
        py::dict state = py::reinterpret_borrow<py::dict>(item.second);
        SlotTaskInfo info;
        info.src_server = py::cast<int>(state["src_server"]);
        info.dst_server = py::cast<int>(state["dst_server"]);
        py::sequence path = py::reinterpret_borrow<py::sequence>(state["path"]);
        info.path_edges.reserve(static_cast<size_t>(path.size()));
        for (py::handle edge_handle : path) {
            info.path_edges.push_back(py::reinterpret_borrow<py::object>(edge_handle));
        }
        task_info.emplace(key, std::move(info));
    }

    const py::ssize_t slot_count = slot_resource_pressure.size();
    py::list result;
    for (py::ssize_t slot_idx = 0; slot_idx < slot_count; ++slot_idx) {
        std::unordered_map<std::string, EdgeServiceAccum> edge_service;
        std::unordered_map<int, double> sender_service;
        std::unordered_map<int, double> receiver_service;
        py::dict service_rates = py::reinterpret_borrow<py::dict>(slot_service_rates[slot_idx]);
        py::iterable active_tasks = py::reinterpret_borrow<py::iterable>(slot_active_tasks[slot_idx]);

        for (py::handle task_key_handle : active_tasks) {
            const long long task_key = pack_task_tuple(task_key_handle);
            auto info_iter = task_info.find(task_key);
            if (info_iter == task_info.end()) {
                continue;
            }
            const double rate = service_rates.contains(task_key_handle)
                ? py::cast<double>(service_rates[task_key_handle])
                : 0.0;
            if (rate <= 0.0) {
                continue;
            }
            const SlotTaskInfo &info = info_iter->second;
            sender_service[info.src_server] += rate;
            receiver_service[info.dst_server] += rate;
            for (const py::object &edge : info.path_edges) {
                const std::string edge_id = py::cast<std::string>(py::str(edge));
                auto &entry = edge_service[edge_id];
                if (entry.key.is_none()) {
                    entry.key = edge;
                }
                entry.service += rate;
            }
        }

        std::unordered_set<ResourceKey, ResourceKeyHash> critical_resources;
        if (slot_idx < critical_resources_by_slot.size()) {
            py::iterable resources = py::reinterpret_borrow<py::iterable>(critical_resources_by_slot[slot_idx]);
            for (py::handle resource_handle : resources) {
                critical_resources.insert(parse_resource_key(resource_handle));
            }
        }

        auto critical_multiplier = [&](const ResourceKey &resource) -> double {
            return critical_resources.find(resource) == critical_resources.end()
                ? 1.0
                : 1.0 + critical_path_price_beta;
        };

        py::dict pressure_state = py::reinterpret_borrow<py::dict>(slot_resource_pressure[slot_idx]);
        py::dict edge_pressure = py::reinterpret_borrow<py::dict>(pressure_state["edge"]);
        py::dict sender_pressure = py::reinterpret_borrow<py::dict>(pressure_state["sender"]);
        py::dict receiver_pressure = py::reinterpret_borrow<py::dict>(pressure_state["receiver"]);

        py::dict edge_prices;
        for (const auto &entry : edge_service) {
            const py::object &edge_key = entry.second.key;
            const double capacity = edge_capacity.contains(edge_key)
                ? py::cast<double>(edge_capacity[edge_key])
                : 0.0;
            const double utilization = std::max(0.0, std::min(1.0, entry.second.service / std::max(capacity, 1e-12)));
            const double pressure = edge_pressure.contains(edge_key)
                ? py::cast<double>(edge_pressure[edge_key])
                : 0.0;
            const ResourceKey resource{0, entry.first};
            edge_prices[edge_key] = py::float_(
                critical_multiplier(resource)
                * utilization
                * resource_price_value(capacity, pressure, link_price_beta, link_price_gamma)
            );
        }

        py::dict sender_prices;
        for (const auto &entry : sender_service) {
            const int server = entry.first;
            const double capacity = server >= 0 && server < static_cast<int>(server_send_capacity.size())
                ? server_send_capacity[static_cast<size_t>(server)]
                : 0.0;
            const double utilization = std::max(0.0, std::min(1.0, entry.second / std::max(capacity, 1e-12)));
            py::int_ sender_key(server);
            const double pressure = sender_pressure.contains(sender_key)
                ? py::cast<double>(sender_pressure[sender_key])
                : 0.0;
            const ResourceKey resource{1, std::to_string(server)};
            sender_prices[py::int_(server)] = py::float_(
                critical_multiplier(resource)
                * utilization
                * resource_price_value(capacity, pressure, link_price_beta, link_price_gamma)
            );
        }

        py::dict receiver_prices;
        for (const auto &entry : receiver_service) {
            const int server = entry.first;
            const double capacity = server >= 0 && server < static_cast<int>(server_recv_capacity.size())
                ? server_recv_capacity[static_cast<size_t>(server)]
                : 0.0;
            const double utilization = std::max(0.0, std::min(1.0, entry.second / std::max(capacity, 1e-12)));
            py::int_ receiver_key(server);
            const double pressure = receiver_pressure.contains(receiver_key)
                ? py::cast<double>(receiver_pressure[receiver_key])
                : 0.0;
            const ResourceKey resource{2, std::to_string(server)};
            receiver_prices[py::int_(server)] = py::float_(
                critical_multiplier(resource)
                * utilization
                * resource_price_value(capacity, pressure, link_price_beta, link_price_gamma)
            );
        }

        py::dict slot_prices;
        slot_prices["edge"] = edge_prices;
        slot_prices["sender"] = sender_prices;
        slot_prices["receiver"] = receiver_prices;
        result.append(slot_prices);
    }
    return result;
}

py::object coarse_local_descent_dense(
    const py::list &flows,
    const py::dict &rank_to_server,
    const py::list &branch_order,
    int anchor_limit,
    int partner_limit,
    const py::list &block_groups,
    const std::vector<double> &pair_epoch_price_dense,
    int epoch_count,
    int server_count,
    int passes
) {
    struct Flow {
        int epoch;
        int src_rank;
        int dst_rank;
        double volume;
    };

    if (epoch_count <= 0 || server_count <= 0) {
        return py::none();
    }
    const std::size_t expected_size = static_cast<std::size_t>(epoch_count)
        * static_cast<std::size_t>(server_count)
        * static_cast<std::size_t>(server_count);
    if (pair_epoch_price_dense.size() < expected_size) {
        throw std::invalid_argument("dense pair-epoch price table has invalid size");
    }

    std::vector<int> ranks;
    ranks.reserve(static_cast<std::size_t>(branch_order.size()));
    for (py::handle rank_handle : branch_order) {
        ranks.push_back(py::cast<int>(rank_handle));
    }

    std::unordered_map<int, int> server_by_rank;
    for (auto item : rank_to_server) {
        server_by_rank.emplace(py::cast<int>(item.first), py::cast<int>(item.second));
    }

    std::vector<Flow> flow_vec;
    flow_vec.reserve(static_cast<std::size_t>(flows.size()));
    std::unordered_map<int, std::vector<int>> incidence;
    for (py::ssize_t idx = 0; idx < flows.size(); ++idx) {
        py::tuple entry = py::reinterpret_borrow<py::tuple>(flows[idx]);
        Flow flow{
            py::cast<int>(entry[0]),
            py::cast<int>(entry[1]),
            py::cast<int>(entry[2]),
            py::cast<double>(entry[3]),
        };
        const int flow_idx = static_cast<int>(flow_vec.size());
        incidence[flow.src_rank].push_back(flow_idx);
        incidence[flow.dst_rank].push_back(flow_idx);
        flow_vec.push_back(flow);
    }

    std::vector<std::vector<int>> groups;
    groups.reserve(static_cast<std::size_t>(block_groups.size()));
    for (py::handle group_handle : block_groups) {
        py::sequence group_seq = py::reinterpret_borrow<py::sequence>(group_handle);
        std::vector<int> group;
        group.reserve(static_cast<std::size_t>(group_seq.size()));
        for (py::ssize_t idx = 0; idx < group_seq.size(); ++idx) {
            group.push_back(py::cast<int>(group_seq[idx]));
        }
        groups.push_back(std::move(group));
    }

    auto price = [&](int epoch, int src_server, int dst_server) -> double {
        if (src_server == dst_server) {
            return 0.0;
        }
        if (epoch < 0 || epoch >= epoch_count || src_server < 0 || dst_server < 0
            || src_server >= server_count || dst_server >= server_count) {
            throw std::out_of_range("dense pair-epoch price index out of range");
        }
        const std::size_t offset = (
            (static_cast<std::size_t>(epoch) * static_cast<std::size_t>(server_count)
                + static_cast<std::size_t>(src_server))
            * static_cast<std::size_t>(server_count)
            + static_cast<std::size_t>(dst_server)
        );
        const double value = pair_epoch_price_dense[offset];
        if (!std::isfinite(value)) {
            throw std::out_of_range("missing dense pair-epoch price entry");
        }
        return value;
    };

    auto delta_for_reassignment = [&](const std::unordered_map<int, int> &new_server_by_rank) -> double {
        std::vector<int> affected;
        std::unordered_map<int, char> affected_seen;
        for (const auto &entry : new_server_by_rank) {
            auto it = incidence.find(entry.first);
            if (it == incidence.end()) {
                continue;
            }
            for (const int flow_idx : it->second) {
                if (affected_seen.find(flow_idx) == affected_seen.end()) {
                    affected_seen[flow_idx] = 1;
                    affected.push_back(flow_idx);
                }
            }
        }

        double delta = 0.0;
        for (const int flow_idx : affected) {
            const Flow &flow = flow_vec[static_cast<std::size_t>(flow_idx)];
            const int old_src = server_by_rank[flow.src_rank];
            const int old_dst = server_by_rank[flow.dst_rank];
            int new_src = old_src;
            int new_dst = old_dst;
            auto src_it = new_server_by_rank.find(flow.src_rank);
            if (src_it != new_server_by_rank.end()) {
                new_src = src_it->second;
            }
            auto dst_it = new_server_by_rank.find(flow.dst_rank);
            if (dst_it != new_server_by_rank.end()) {
                new_dst = dst_it->second;
            }
            delta += flow.volume * (
                price(flow.epoch, new_src, new_dst) - price(flow.epoch, old_src, old_dst)
            );
        }
        return delta;
    };

    const int effective_anchor_limit = std::max(0, std::min(anchor_limit, static_cast<int>(ranks.size())));
    const int effective_partner_limit = std::max(0, std::min(partner_limit, static_cast<int>(ranks.size())));
    bool changed = false;

    for (int pass_idx = 0; pass_idx < std::max(0, passes); ++pass_idx) {
        double best_delta = 0.0;
        std::unordered_map<int, int> best_move;
        std::unordered_map<long long, char> tried;

        for (int left_idx = 0; left_idx < effective_anchor_limit; ++left_idx) {
            const int left_rank = ranks[static_cast<std::size_t>(left_idx)];
            for (int right_idx = 0; right_idx < effective_partner_limit; ++right_idx) {
                const int right_rank = ranks[static_cast<std::size_t>(right_idx)];
                if (left_rank == right_rank) {
                    continue;
                }
                const int a = std::min(left_rank, right_rank);
                const int b = std::max(left_rank, right_rank);
                const long long pair_key = (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
                if (tried.find(pair_key) != tried.end()) {
                    continue;
                }
                tried[pair_key] = 1;
                std::unordered_map<int, int> reassignment{
                    {a, server_by_rank[b]},
                    {b, server_by_rank[a]},
                };
                const double delta = delta_for_reassignment(reassignment);
                if (delta < best_delta - 1e-12) {
                    best_delta = delta;
                    best_move = std::move(reassignment);
                }
            }
        }

        for (const auto &group : groups) {
            const int group_size = static_cast<int>(group.size());
            if (group_size < 3) {
                continue;
            }
            std::vector<int> current_servers;
            current_servers.reserve(static_cast<std::size_t>(group_size));
            for (const int rank : group) {
                current_servers.push_back(server_by_rank[rank]);
            }
            for (int shift = 1; shift < group_size; ++shift) {
                std::unordered_map<int, int> reassignment;
                for (int idx = 0; idx < group_size; ++idx) {
                    reassignment[group[static_cast<std::size_t>(idx)]] =
                        current_servers[static_cast<std::size_t>((idx + shift) % group_size)];
                }
                const double delta = delta_for_reassignment(reassignment);
                if (delta < best_delta - 1e-12) {
                    best_delta = delta;
                    best_move = std::move(reassignment);
                }
            }
        }

        if (best_move.empty()) {
            break;
        }
        for (const auto &entry : best_move) {
            server_by_rank[entry.first] = entry.second;
        }
        changed = true;
    }

    if (!changed) {
        return py::none();
    }
    py::dict result;
    for (const auto &entry : server_by_rank) {
        result[py::int_(entry.first)] = py::int_(entry.second);
    }
    return result;
}

py::list price_guided_remap_candidates(
    const std::vector<int> &ranks,
    const std::vector<int> &current_servers,
    const std::vector<int> &branch_ranks,
    const py::list &task_infos_py,
    const py::dict &task_pair_price_py,
    int max_price_candidates,
    double time_budget_seconds
) {
    const int rank_count = static_cast<int>(ranks.size());
    if (rank_count <= 0 || current_servers.size() != ranks.size()) {
        return py::list();
    }
    if (rank_count > 64) {
        throw std::invalid_argument("price_guided_remap_candidates supports at most 64 ranks");
    }
    max_price_candidates = std::max(1, max_price_candidates);
    const auto start_time = std::chrono::steady_clock::now();
    const bool has_time_budget = std::isfinite(time_budget_seconds) && time_budget_seconds >= 0.0;

    auto timed_out = [&]() -> bool {
        if (!has_time_budget) {
            return false;
        }
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        return elapsed >= time_budget_seconds;
    };

    std::unordered_map<int, int> rank_to_pos;
    rank_to_pos.reserve(static_cast<size_t>(rank_count));
    for (int pos = 0; pos < rank_count; ++pos) {
        rank_to_pos[ranks[static_cast<size_t>(pos)]] = pos;
    }

    std::unordered_map<int, int> server_to_index;
    server_to_index.reserve(current_servers.size());
    for (int idx = 0; idx < rank_count; ++idx) {
        server_to_index[current_servers[static_cast<size_t>(idx)]] = idx;
    }
    const std::uint64_t full_server_mask = rank_count == 64
        ? std::numeric_limits<std::uint64_t>::max()
        : ((std::uint64_t{1} << rank_count) - 1U);

    std::vector<RemapTaskInfo> tasks;
    tasks.reserve(static_cast<size_t>(task_infos_py.size()));
    std::vector<std::vector<int>> rank_incidence(static_cast<size_t>(rank_count));
    std::unordered_map<int, int> task_id_to_index;
    for (py::handle item_handle : task_infos_py) {
        py::tuple item = py::reinterpret_borrow<py::tuple>(item_handle);
        const int task_id = py::cast<int>(item[0]);
        const int src_rank = py::cast<int>(item[1]);
        const int dst_rank = py::cast<int>(item[2]);
        const double volume = py::cast<double>(item[3]);
        auto src_iter = rank_to_pos.find(src_rank);
        auto dst_iter = rank_to_pos.find(dst_rank);
        if (src_iter == rank_to_pos.end() || dst_iter == rank_to_pos.end()) {
            continue;
        }
        const int task_index = static_cast<int>(tasks.size());
        tasks.push_back(RemapTaskInfo{
            task_id,
            src_rank,
            dst_rank,
            src_iter->second,
            dst_iter->second,
            volume,
        });
        task_id_to_index[task_id] = task_index;
        rank_incidence[static_cast<size_t>(src_iter->second)].push_back(task_index);
        rank_incidence[static_cast<size_t>(dst_iter->second)].push_back(task_index);
    }

    const int task_count = static_cast<int>(tasks.size());
    const double missing_price = std::numeric_limits<double>::infinity();
    std::vector<double> task_pair_price(
        static_cast<size_t>(std::max(task_count, 0) * rank_count * rank_count),
        missing_price
    );
    auto price_offset = [&](int task_index, int src_idx, int dst_idx) -> size_t {
        return (static_cast<size_t>(task_index) * static_cast<size_t>(rank_count)
            + static_cast<size_t>(src_idx)) * static_cast<size_t>(rank_count)
            + static_cast<size_t>(dst_idx);
    };
    for (auto item : task_pair_price_py) {
        py::tuple key = py::reinterpret_borrow<py::tuple>(item.first);
        const int task_id = py::cast<int>(key[0]);
        const int src_server = py::cast<int>(key[1]);
        const int dst_server = py::cast<int>(key[2]);
        auto task_iter = task_id_to_index.find(task_id);
        auto src_iter = server_to_index.find(src_server);
        auto dst_iter = server_to_index.find(dst_server);
        if (task_iter == task_id_to_index.end() || src_iter == server_to_index.end() || dst_iter == server_to_index.end()) {
            continue;
        }
        task_pair_price[price_offset(task_iter->second, src_iter->second, dst_iter->second)] = py::cast<double>(item.second);
    }

    auto lookup_price = [&](int task_index, int src_server, int dst_server) -> double {
        const int src_idx = server_to_index.at(src_server);
        const int dst_idx = server_to_index.at(dst_server);
        const double price = task_pair_price[price_offset(task_index, src_idx, dst_idx)];
        if (!std::isfinite(price)) {
            throw std::out_of_range("missing task-pair price entry");
        }
        return price;
    };

    auto lookup_price_by_index = [&](int task_index, int src_idx, int dst_idx) -> double {
        const double price = task_pair_price[price_offset(task_index, src_idx, dst_idx)];
        if (!std::isfinite(price)) {
            throw std::out_of_range("missing task-pair price entry");
        }
        return price;
    };

    std::vector<int> branch_positions;
    branch_positions.reserve(branch_ranks.size());
    for (const int rank : branch_ranks) {
        auto iter = rank_to_pos.find(rank);
        if (iter != rank_to_pos.end()) {
            branch_positions.push_back(iter->second);
        }
    }
    if (static_cast<int>(branch_positions.size()) != rank_count) {
        branch_positions.clear();
        for (int pos = 0; pos < rank_count; ++pos) {
            branch_positions.push_back(pos);
        }
    }

    auto assignment_cost = [&](const std::vector<int> &assignment) -> double {
        double total = 0.0;
        for (int task_index = 0; task_index < task_count; ++task_index) {
            const RemapTaskInfo &task = tasks[static_cast<size_t>(task_index)];
            const int src_server = assignment[static_cast<size_t>(task.src_pos)];
            const int dst_server = assignment[static_cast<size_t>(task.dst_pos)];
            total += task.volume * lookup_price(task_index, src_server, dst_server);
        }
        return total;
    };

    std::vector<RemapCandidate> candidates;
    std::unordered_set<std::vector<int>, IntVectorHash> candidate_signatures;
    candidates.reserve(static_cast<size_t>(max_price_candidates));

    auto add_candidate = [&](double cost, const std::vector<int> &assignment) {
        for (const int server : assignment) {
            if (server < 0) {
                return;
            }
        }
        if (candidate_signatures.find(assignment) != candidate_signatures.end()) {
            return;
        }
        candidates.push_back(RemapCandidate{cost, assignment});
        candidate_signatures.insert(assignment);
        std::sort(candidates.begin(), candidates.end(), [](const RemapCandidate &lhs, const RemapCandidate &rhs) {
            return lhs.cost < rhs.cost;
        });
        while (static_cast<int>(candidates.size()) > max_price_candidates) {
            candidate_signatures.erase(candidates.back().assignment);
            candidates.pop_back();
        }
    };

    auto current_prune_cost = [&]() -> double {
        if (static_cast<int>(candidates.size()) < max_price_candidates) {
            return std::numeric_limits<double>::infinity();
        }
        return candidates.back().cost;
    };

    std::vector<int> initial_assignment = current_servers;
    add_candidate(assignment_cost(initial_assignment), initial_assignment);

    std::unordered_map<RemapMinKey, double, RemapMinKeyHash> min_price_cache;
    min_price_cache.reserve(tasks.size() * 16U);
    constexpr std::size_t kMaxMinPriceCacheEntries = 200000;
    auto remember_min_price = [&](const RemapMinKey &key, double value) {
        if (min_price_cache.size() >= kMaxMinPriceCacheEntries) {
            min_price_cache.clear();
        }
        min_price_cache.emplace(key, value);
    };

    auto min_task_price_any = [&](int task_index, std::uint64_t remaining_mask) -> double {
        const RemapMinKey key{0, task_index, -1, remaining_mask};
        auto cached = min_price_cache.find(key);
        if (cached != min_price_cache.end()) {
            return cached->second;
        }
        double best = std::numeric_limits<double>::infinity();
        for (int src_idx = 0; src_idx < rank_count; ++src_idx) {
            if ((remaining_mask & (std::uint64_t{1} << src_idx)) == 0U) {
                continue;
            }
            for (int dst_idx = 0; dst_idx < rank_count; ++dst_idx) {
                if (src_idx == dst_idx || (remaining_mask & (std::uint64_t{1} << dst_idx)) == 0U) {
                    continue;
                }
                best = std::min(best, lookup_price_by_index(task_index, src_idx, dst_idx));
            }
        }
        remember_min_price(key, best);
        return best;
    };

    auto min_task_price_dst = [&](int task_index, int src_server, std::uint64_t remaining_mask) -> double {
        const int src_idx = server_to_index.at(src_server);
        const RemapMinKey key{1, task_index, src_idx, remaining_mask};
        auto cached = min_price_cache.find(key);
        if (cached != min_price_cache.end()) {
            return cached->second;
        }
        double best = std::numeric_limits<double>::infinity();
        for (int dst_idx = 0; dst_idx < rank_count; ++dst_idx) {
            if ((remaining_mask & (std::uint64_t{1} << dst_idx)) == 0U) {
                continue;
            }
            const int dst_server = current_servers[static_cast<size_t>(dst_idx)];
            if (dst_server == src_server) {
                continue;
            }
            best = std::min(best, lookup_price_by_index(task_index, src_idx, dst_idx));
        }
        remember_min_price(key, best);
        return best;
    };

    auto min_task_price_src = [&](int task_index, int dst_server, std::uint64_t remaining_mask) -> double {
        const int dst_idx = server_to_index.at(dst_server);
        const RemapMinKey key{2, task_index, dst_idx, remaining_mask};
        auto cached = min_price_cache.find(key);
        if (cached != min_price_cache.end()) {
            return cached->second;
        }
        double best = std::numeric_limits<double>::infinity();
        for (int src_idx = 0; src_idx < rank_count; ++src_idx) {
            if ((remaining_mask & (std::uint64_t{1} << src_idx)) == 0U) {
                continue;
            }
            const int src_server = current_servers[static_cast<size_t>(src_idx)];
            if (src_server == dst_server) {
                continue;
            }
            best = std::min(best, lookup_price_by_index(task_index, src_idx, dst_idx));
        }
        remember_min_price(key, best);
        return best;
    };

    std::vector<int> assignment(static_cast<size_t>(rank_count), -1);
    std::uint64_t used_server_mask = 0U;

    auto lower_bound = [&](double current_partial_cost) -> double {
        const std::uint64_t remaining_mask = full_server_mask ^ used_server_mask;
        double bound = current_partial_cost;
        for (int task_index = 0; task_index < task_count; ++task_index) {
            if (timed_out()) {
                return std::numeric_limits<double>::infinity();
            }
            const RemapTaskInfo &task = tasks[static_cast<size_t>(task_index)];
            const bool src_assigned = assignment[static_cast<size_t>(task.src_pos)] >= 0;
            const bool dst_assigned = assignment[static_cast<size_t>(task.dst_pos)] >= 0;
            if (src_assigned && dst_assigned) {
                continue;
            }
            double best_price = std::numeric_limits<double>::infinity();
            if (src_assigned) {
                best_price = min_task_price_dst(task_index, assignment[static_cast<size_t>(task.src_pos)], remaining_mask);
            } else if (dst_assigned) {
                best_price = min_task_price_src(task_index, assignment[static_cast<size_t>(task.dst_pos)], remaining_mask);
            } else {
                best_price = min_task_price_any(task_index, remaining_mask);
            }
            if (std::isfinite(best_price)) {
                bound += task.volume * best_price;
            }
        }
        return bound;
    };

    auto incremental_rank_server_price = [&](int rank_pos, int server, const std::vector<int> &partial_assignment, std::uint64_t used_mask) {
        double exact_delta = 0.0;
        double optimistic_delta = 0.0;
        const std::uint64_t remaining_after = (full_server_mask ^ used_mask) & ~(std::uint64_t{1} << server_to_index.at(server));
        for (const int task_index : rank_incidence[static_cast<size_t>(rank_pos)]) {
            const RemapTaskInfo &task = tasks[static_cast<size_t>(task_index)];
            const bool rank_is_src = task.src_pos == rank_pos;
            const int other_pos = rank_is_src ? task.dst_pos : task.src_pos;
            if (partial_assignment[static_cast<size_t>(other_pos)] >= 0) {
                const int src_server = rank_is_src ? server : partial_assignment[static_cast<size_t>(other_pos)];
                const int dst_server = rank_is_src ? partial_assignment[static_cast<size_t>(other_pos)] : server;
                exact_delta += task.volume * lookup_price(task_index, src_server, dst_server);
                continue;
            }
            double best = std::numeric_limits<double>::infinity();
            const int server_idx = server_to_index.at(server);
            for (int idx = 0; idx < rank_count; ++idx) {
                if ((remaining_after & (std::uint64_t{1} << idx)) == 0U) {
                    continue;
                }
                if (rank_is_src) {
                    best = std::min(best, lookup_price_by_index(task_index, server_idx, idx));
                } else {
                    best = std::min(best, lookup_price_by_index(task_index, idx, server_idx));
                }
            }
            if (std::isfinite(best)) {
                optimistic_delta += task.volume * best;
            }
        }
        return std::make_pair(exact_delta, optimistic_delta);
    };

    auto candidate_servers_for_rank = [&](int rank_pos) {
        std::vector<std::tuple<double, int, int>> scored;
        const int preferred = current_servers[static_cast<size_t>(rank_pos)];
        const std::uint64_t available_mask = full_server_mask ^ used_server_mask;
        for (int idx = 0; idx < rank_count; ++idx) {
            if (timed_out()) {
                return std::vector<int>();
            }
            if ((available_mask & (std::uint64_t{1} << idx)) == 0U) {
                continue;
            }
            const int server = current_servers[static_cast<size_t>(idx)];
            const auto deltas = incremental_rank_server_price(rank_pos, server, assignment, used_server_mask);
            scored.emplace_back(deltas.first + deltas.second, server == preferred ? 0 : 1, server);
        }
        std::sort(scored.begin(), scored.end(), [](const auto &lhs, const auto &rhs) {
            if (std::get<0>(lhs) != std::get<0>(rhs)) {
                return std::get<0>(lhs) < std::get<0>(rhs);
            }
            if (std::get<1>(lhs) != std::get<1>(rhs)) {
                return std::get<1>(lhs) < std::get<1>(rhs);
            }
            return std::get<2>(lhs) < std::get<2>(rhs);
        });
        std::vector<int> ordered;
        ordered.reserve(scored.size());
        for (const auto &entry : scored) {
            ordered.push_back(std::get<2>(entry));
        }
        return ordered;
    };

    auto apply_rank = [&](int rank_pos, int server, double current_partial_cost) -> double {
        assignment[static_cast<size_t>(rank_pos)] = server;
        used_server_mask |= (std::uint64_t{1} << server_to_index.at(server));
        double delta = 0.0;
        for (const int task_index : rank_incidence[static_cast<size_t>(rank_pos)]) {
            const RemapTaskInfo &task = tasks[static_cast<size_t>(task_index)];
            const bool rank_is_src = task.src_pos == rank_pos;
            const int other_pos = rank_is_src ? task.dst_pos : task.src_pos;
            if (assignment[static_cast<size_t>(other_pos)] < 0) {
                continue;
            }
            const int src_server = rank_is_src ? server : assignment[static_cast<size_t>(other_pos)];
            const int dst_server = rank_is_src ? assignment[static_cast<size_t>(other_pos)] : server;
            delta += task.volume * lookup_price(task_index, src_server, dst_server);
        }
        return current_partial_cost + delta;
    };

    auto rollback_rank = [&](int rank_pos, int server) {
        used_server_mask &= ~(std::uint64_t{1} << server_to_index.at(server));
        assignment[static_cast<size_t>(rank_pos)] = -1;
    };

    auto greedy_price_assignment = [&]() {
        std::vector<int> greedy_assignment(static_cast<size_t>(rank_count), -1);
        std::uint64_t greedy_used = 0U;
        double greedy_cost = 0.0;
        for (const int rank_pos : branch_positions) {
            if (timed_out()) {
                return;
            }
            std::vector<std::tuple<double, double, int>> scored;
            const std::uint64_t available_mask = full_server_mask ^ greedy_used;
            for (int idx = 0; idx < rank_count; ++idx) {
                if (timed_out()) {
                    return;
                }
                if ((available_mask & (std::uint64_t{1} << idx)) == 0U) {
                    continue;
                }
                const int server = current_servers[static_cast<size_t>(idx)];
                const auto deltas = incremental_rank_server_price(rank_pos, server, greedy_assignment, greedy_used);
                scored.emplace_back(deltas.first + deltas.second, deltas.first, server);
            }
            if (scored.empty()) {
                continue;
            }
            std::sort(scored.begin(), scored.end(), [](const auto &lhs, const auto &rhs) {
                if (std::get<0>(lhs) != std::get<0>(rhs)) {
                    return std::get<0>(lhs) < std::get<0>(rhs);
                }
                return std::get<2>(lhs) < std::get<2>(rhs);
            });
            const int server = std::get<2>(scored.front());
            greedy_assignment[static_cast<size_t>(rank_pos)] = server;
            greedy_used |= (std::uint64_t{1} << server_to_index.at(server));
            greedy_cost += std::get<1>(scored.front());
        }
        bool complete = true;
        for (const int server : greedy_assignment) {
            if (server < 0) {
                complete = false;
                break;
            }
        }
        if (complete) {
            add_candidate(greedy_cost, greedy_assignment);
        }
    };

    greedy_price_assignment();

    std::function<void(int, double)> dfs = [&](int depth, double current_partial_cost) {
        if (timed_out()) {
            return;
        }
        if (lower_bound(current_partial_cost) >= current_prune_cost() - 1e-12) {
            return;
        }
        if (depth >= static_cast<int>(branch_positions.size())) {
            add_candidate(current_partial_cost, assignment);
            return;
        }
        const int rank_pos = branch_positions[static_cast<size_t>(depth)];
        for (const int server : candidate_servers_for_rank(rank_pos)) {
            if (timed_out()) {
                break;
            }
            const double next_cost = apply_rank(rank_pos, server, current_partial_cost);
            dfs(depth + 1, next_cost);
            rollback_rank(rank_pos, server);
        }
    };
    dfs(0, 0.0);

    py::list result;
    for (const RemapCandidate &candidate : candidates) {
        py::dict assignment_dict;
        for (int pos = 0; pos < rank_count; ++pos) {
            assignment_dict[py::int_(ranks[static_cast<size_t>(pos)])] = py::int_(candidate.assignment[static_cast<size_t>(pos)]);
        }
        result.append(py::make_tuple(candidate.cost, assignment_dict));
    }
    return result;
}

py::list price_guided_remap_candidates_dense(
    const std::vector<int> &ranks,
    const std::vector<int> &current_servers,
    const std::vector<int> &branch_ranks,
    const py::list &task_infos_py,
    const py::list &exposure_slots_by_task,
    const std::vector<std::vector<double>> &sender_prices,
    const std::vector<std::vector<double>> &receiver_prices,
    const std::vector<std::vector<double>> &edge_prices,
    const std::vector<std::vector<std::vector<int>>> &path_edges_by_pair,
    int max_price_candidates,
    double time_budget_seconds
) {
    const int rank_count = static_cast<int>(ranks.size());
    if (rank_count <= 0 || current_servers.size() != ranks.size()) {
        return py::list();
    }
    if (rank_count > 64) {
        throw std::invalid_argument("price_guided_remap_candidates_dense supports at most 64 ranks");
    }
    max_price_candidates = std::max(1, max_price_candidates);
    const auto start_time = std::chrono::steady_clock::now();
    const bool has_time_budget = std::isfinite(time_budget_seconds) && time_budget_seconds >= 0.0;

    auto timed_out = [&]() -> bool {
        if (!has_time_budget) {
            return false;
        }
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        return elapsed >= time_budget_seconds;
    };

    std::unordered_map<int, int> rank_to_pos;
    rank_to_pos.reserve(static_cast<size_t>(rank_count));
    for (int pos = 0; pos < rank_count; ++pos) {
        rank_to_pos[ranks[static_cast<size_t>(pos)]] = pos;
    }

    std::unordered_map<int, int> server_to_index;
    server_to_index.reserve(current_servers.size());
    for (int idx = 0; idx < rank_count; ++idx) {
        server_to_index[current_servers[static_cast<size_t>(idx)]] = idx;
    }
    const std::uint64_t full_server_mask = rank_count == 64
        ? std::numeric_limits<std::uint64_t>::max()
        : ((std::uint64_t{1} << rank_count) - 1U);

    std::vector<RemapTaskInfo> tasks;
    tasks.reserve(static_cast<size_t>(task_infos_py.size()));
    std::vector<std::vector<int>> rank_incidence(static_cast<size_t>(rank_count));
    for (py::handle item_handle : task_infos_py) {
        py::tuple item = py::reinterpret_borrow<py::tuple>(item_handle);
        const int task_id = py::cast<int>(item[0]);
        const int src_rank = py::cast<int>(item[1]);
        const int dst_rank = py::cast<int>(item[2]);
        const double volume = py::cast<double>(item[3]);
        auto src_iter = rank_to_pos.find(src_rank);
        auto dst_iter = rank_to_pos.find(dst_rank);
        if (src_iter == rank_to_pos.end() || dst_iter == rank_to_pos.end()) {
            continue;
        }
        const int task_index = static_cast<int>(tasks.size());
        tasks.push_back(RemapTaskInfo{
            task_id,
            src_rank,
            dst_rank,
            src_iter->second,
            dst_iter->second,
            volume,
        });
        rank_incidence[static_cast<size_t>(src_iter->second)].push_back(task_index);
        rank_incidence[static_cast<size_t>(dst_iter->second)].push_back(task_index);
    }

    const int task_count = static_cast<int>(tasks.size());
    if (static_cast<int>(exposure_slots_by_task.size()) < task_count) {
        throw std::invalid_argument("exposure_slots_by_task is shorter than task_infos");
    }
    const int slot_count = static_cast<int>(sender_prices.size());
    std::vector<double> task_pair_price(
        static_cast<size_t>(std::max(task_count, 0) * rank_count * rank_count),
        std::numeric_limits<double>::infinity()
    );
    auto price_offset = [&](int task_index, int src_idx, int dst_idx) -> size_t {
        return (static_cast<size_t>(task_index) * static_cast<size_t>(rank_count)
            + static_cast<size_t>(src_idx)) * static_cast<size_t>(rank_count)
            + static_cast<size_t>(dst_idx);
    };
    for (int task_index = 0; task_index < task_count; ++task_index) {
        std::vector<int> exposure_slots = py::cast<std::vector<int>>(exposure_slots_by_task[task_index]);
        if (exposure_slots.empty()) {
            exposure_slots.push_back(0);
        }
        for (int src_idx = 0; src_idx < rank_count; ++src_idx) {
            const int src_server = current_servers[static_cast<size_t>(src_idx)];
            for (int dst_idx = 0; dst_idx < rank_count; ++dst_idx) {
                if (src_idx == dst_idx) {
                    continue;
                }
                const int dst_server = current_servers[static_cast<size_t>(dst_idx)];
                double price_sum = 0.0;
                for (const int raw_slot : exposure_slots) {
                    const int slot = slot_count > 0
                        ? std::max(0, std::min(raw_slot, slot_count - 1))
                        : 0;
                    double price = 0.0;
                    if (slot_count > 0) {
                        price = std::max(price, sender_prices[static_cast<size_t>(slot)][static_cast<size_t>(src_server)]);
                        price = std::max(price, receiver_prices[static_cast<size_t>(slot)][static_cast<size_t>(dst_server)]);
                        const auto &path = path_edges_by_pair[static_cast<size_t>(src_server)][static_cast<size_t>(dst_server)];
                        for (const int edge_idx : path) {
                            price = std::max(price, edge_prices[static_cast<size_t>(slot)][static_cast<size_t>(edge_idx)]);
                        }
                    }
                    price_sum += price;
                }
                task_pair_price[price_offset(task_index, src_idx, dst_idx)] =
                    price_sum / std::max<size_t>(exposure_slots.size(), 1);
            }
        }
    }

    auto lookup_price = [&](int task_index, int src_server, int dst_server) -> double {
        const int src_idx = server_to_index.at(src_server);
        const int dst_idx = server_to_index.at(dst_server);
        const double price = task_pair_price[price_offset(task_index, src_idx, dst_idx)];
        if (!std::isfinite(price)) {
            throw std::out_of_range("missing dense task-pair price entry");
        }
        return price;
    };

    auto lookup_price_by_index = [&](int task_index, int src_idx, int dst_idx) -> double {
        const double price = task_pair_price[price_offset(task_index, src_idx, dst_idx)];
        if (!std::isfinite(price)) {
            throw std::out_of_range("missing dense task-pair price entry");
        }
        return price;
    };

    std::vector<int> branch_positions;
    branch_positions.reserve(branch_ranks.size());
    for (const int rank : branch_ranks) {
        auto iter = rank_to_pos.find(rank);
        if (iter != rank_to_pos.end()) {
            branch_positions.push_back(iter->second);
        }
    }
    if (static_cast<int>(branch_positions.size()) != rank_count) {
        branch_positions.clear();
        for (int pos = 0; pos < rank_count; ++pos) {
            branch_positions.push_back(pos);
        }
    }

    auto assignment_cost = [&](const std::vector<int> &assignment) -> double {
        double total = 0.0;
        for (int task_index = 0; task_index < task_count; ++task_index) {
            const RemapTaskInfo &task = tasks[static_cast<size_t>(task_index)];
            const int src_server = assignment[static_cast<size_t>(task.src_pos)];
            const int dst_server = assignment[static_cast<size_t>(task.dst_pos)];
            total += task.volume * lookup_price(task_index, src_server, dst_server);
        }
        return total;
    };

    std::vector<RemapCandidate> candidates;
    std::unordered_set<std::vector<int>, IntVectorHash> candidate_signatures;
    candidates.reserve(static_cast<size_t>(max_price_candidates));

    auto add_candidate = [&](double cost, const std::vector<int> &assignment) {
        for (const int server : assignment) {
            if (server < 0) {
                return;
            }
        }
        if (candidate_signatures.find(assignment) != candidate_signatures.end()) {
            return;
        }
        candidates.push_back(RemapCandidate{cost, assignment});
        candidate_signatures.insert(assignment);
        std::sort(candidates.begin(), candidates.end(), [](const RemapCandidate &lhs, const RemapCandidate &rhs) {
            return lhs.cost < rhs.cost;
        });
        while (static_cast<int>(candidates.size()) > max_price_candidates) {
            candidate_signatures.erase(candidates.back().assignment);
            candidates.pop_back();
        }
    };

    auto current_prune_cost = [&]() -> double {
        if (static_cast<int>(candidates.size()) < max_price_candidates) {
            return std::numeric_limits<double>::infinity();
        }
        return candidates.back().cost;
    };

    std::vector<int> initial_assignment = current_servers;
    add_candidate(assignment_cost(initial_assignment), initial_assignment);

    std::unordered_map<RemapMinKey, double, RemapMinKeyHash> min_price_cache;
    min_price_cache.reserve(tasks.size() * 16U);
    constexpr std::size_t kMaxDenseMinPriceCacheEntries = 200000;
    auto remember_min_price = [&](const RemapMinKey &key, double value) {
        if (min_price_cache.size() >= kMaxDenseMinPriceCacheEntries) {
            min_price_cache.clear();
        }
        min_price_cache.emplace(key, value);
    };

    auto min_task_price_any = [&](int task_index, std::uint64_t remaining_mask) -> double {
        const RemapMinKey key{0, task_index, -1, remaining_mask};
        auto cached = min_price_cache.find(key);
        if (cached != min_price_cache.end()) {
            return cached->second;
        }
        double best = std::numeric_limits<double>::infinity();
        for (int src_idx = 0; src_idx < rank_count; ++src_idx) {
            if ((remaining_mask & (std::uint64_t{1} << src_idx)) == 0U) {
                continue;
            }
            for (int dst_idx = 0; dst_idx < rank_count; ++dst_idx) {
                if (src_idx == dst_idx || (remaining_mask & (std::uint64_t{1} << dst_idx)) == 0U) {
                    continue;
                }
                best = std::min(best, lookup_price_by_index(task_index, src_idx, dst_idx));
            }
        }
        remember_min_price(key, best);
        return best;
    };

    auto min_task_price_dst = [&](int task_index, int src_server, std::uint64_t remaining_mask) -> double {
        const int src_idx = server_to_index.at(src_server);
        const RemapMinKey key{1, task_index, src_idx, remaining_mask};
        auto cached = min_price_cache.find(key);
        if (cached != min_price_cache.end()) {
            return cached->second;
        }
        double best = std::numeric_limits<double>::infinity();
        for (int dst_idx = 0; dst_idx < rank_count; ++dst_idx) {
            if ((remaining_mask & (std::uint64_t{1} << dst_idx)) == 0U) {
                continue;
            }
            const int dst_server = current_servers[static_cast<size_t>(dst_idx)];
            if (dst_server == src_server) {
                continue;
            }
            best = std::min(best, lookup_price_by_index(task_index, src_idx, dst_idx));
        }
        remember_min_price(key, best);
        return best;
    };

    auto min_task_price_src = [&](int task_index, int dst_server, std::uint64_t remaining_mask) -> double {
        const int dst_idx = server_to_index.at(dst_server);
        const RemapMinKey key{2, task_index, dst_idx, remaining_mask};
        auto cached = min_price_cache.find(key);
        if (cached != min_price_cache.end()) {
            return cached->second;
        }
        double best = std::numeric_limits<double>::infinity();
        for (int src_idx = 0; src_idx < rank_count; ++src_idx) {
            if ((remaining_mask & (std::uint64_t{1} << src_idx)) == 0U) {
                continue;
            }
            const int src_server = current_servers[static_cast<size_t>(src_idx)];
            if (src_server == dst_server) {
                continue;
            }
            best = std::min(best, lookup_price_by_index(task_index, src_idx, dst_idx));
        }
        remember_min_price(key, best);
        return best;
    };

    std::vector<int> assignment(static_cast<size_t>(rank_count), -1);
    std::uint64_t used_server_mask = 0U;

    auto lower_bound = [&](double current_partial_cost) -> double {
        const std::uint64_t remaining_mask = full_server_mask ^ used_server_mask;
        double bound = current_partial_cost;
        for (int task_index = 0; task_index < task_count; ++task_index) {
            if (timed_out()) {
                return std::numeric_limits<double>::infinity();
            }
            const RemapTaskInfo &task = tasks[static_cast<size_t>(task_index)];
            const bool src_assigned = assignment[static_cast<size_t>(task.src_pos)] >= 0;
            const bool dst_assigned = assignment[static_cast<size_t>(task.dst_pos)] >= 0;
            if (src_assigned && dst_assigned) {
                continue;
            }
            double best_price = std::numeric_limits<double>::infinity();
            if (src_assigned) {
                best_price = min_task_price_dst(task_index, assignment[static_cast<size_t>(task.src_pos)], remaining_mask);
            } else if (dst_assigned) {
                best_price = min_task_price_src(task_index, assignment[static_cast<size_t>(task.dst_pos)], remaining_mask);
            } else {
                best_price = min_task_price_any(task_index, remaining_mask);
            }
            if (std::isfinite(best_price)) {
                bound += task.volume * best_price;
            }
        }
        return bound;
    };

    auto incremental_rank_server_price = [&](int rank_pos, int server, const std::vector<int> &partial_assignment, std::uint64_t used_mask) {
        double exact_delta = 0.0;
        double optimistic_delta = 0.0;
        const std::uint64_t remaining_after = (full_server_mask ^ used_mask) & ~(std::uint64_t{1} << server_to_index.at(server));
        for (const int task_index : rank_incidence[static_cast<size_t>(rank_pos)]) {
            const RemapTaskInfo &task = tasks[static_cast<size_t>(task_index)];
            const bool rank_is_src = task.src_pos == rank_pos;
            const int other_pos = rank_is_src ? task.dst_pos : task.src_pos;
            if (partial_assignment[static_cast<size_t>(other_pos)] >= 0) {
                const int src_server = rank_is_src ? server : partial_assignment[static_cast<size_t>(other_pos)];
                const int dst_server = rank_is_src ? partial_assignment[static_cast<size_t>(other_pos)] : server;
                exact_delta += task.volume * lookup_price(task_index, src_server, dst_server);
                continue;
            }
            double best = std::numeric_limits<double>::infinity();
            const int server_idx = server_to_index.at(server);
            for (int idx = 0; idx < rank_count; ++idx) {
                if ((remaining_after & (std::uint64_t{1} << idx)) == 0U) {
                    continue;
                }
                if (rank_is_src) {
                    best = std::min(best, lookup_price_by_index(task_index, server_idx, idx));
                } else {
                    best = std::min(best, lookup_price_by_index(task_index, idx, server_idx));
                }
            }
            if (std::isfinite(best)) {
                optimistic_delta += task.volume * best;
            }
        }
        return std::make_pair(exact_delta, optimistic_delta);
    };

    auto candidate_servers_for_rank = [&](int rank_pos) {
        std::vector<std::tuple<double, int, int>> scored;
        const int preferred = current_servers[static_cast<size_t>(rank_pos)];
        const std::uint64_t available_mask = full_server_mask ^ used_server_mask;
        for (int idx = 0; idx < rank_count; ++idx) {
            if (timed_out()) {
                return std::vector<int>();
            }
            if ((available_mask & (std::uint64_t{1} << idx)) == 0U) {
                continue;
            }
            const int server = current_servers[static_cast<size_t>(idx)];
            const auto deltas = incremental_rank_server_price(rank_pos, server, assignment, used_server_mask);
            scored.emplace_back(deltas.first + deltas.second, server == preferred ? 0 : 1, server);
        }
        std::sort(scored.begin(), scored.end(), [](const auto &lhs, const auto &rhs) {
            if (std::get<0>(lhs) != std::get<0>(rhs)) {
                return std::get<0>(lhs) < std::get<0>(rhs);
            }
            if (std::get<1>(lhs) != std::get<1>(rhs)) {
                return std::get<1>(lhs) < std::get<1>(rhs);
            }
            return std::get<2>(lhs) < std::get<2>(rhs);
        });
        std::vector<int> ordered;
        ordered.reserve(scored.size());
        for (const auto &entry : scored) {
            ordered.push_back(std::get<2>(entry));
        }
        return ordered;
    };

    auto apply_rank = [&](int rank_pos, int server, double current_partial_cost) -> double {
        assignment[static_cast<size_t>(rank_pos)] = server;
        used_server_mask |= (std::uint64_t{1} << server_to_index.at(server));
        double delta = 0.0;
        for (const int task_index : rank_incidence[static_cast<size_t>(rank_pos)]) {
            const RemapTaskInfo &task = tasks[static_cast<size_t>(task_index)];
            const bool rank_is_src = task.src_pos == rank_pos;
            const int other_pos = rank_is_src ? task.dst_pos : task.src_pos;
            if (assignment[static_cast<size_t>(other_pos)] < 0) {
                continue;
            }
            const int src_server = rank_is_src ? server : assignment[static_cast<size_t>(other_pos)];
            const int dst_server = rank_is_src ? assignment[static_cast<size_t>(other_pos)] : server;
            delta += task.volume * lookup_price(task_index, src_server, dst_server);
        }
        return current_partial_cost + delta;
    };

    auto rollback_rank = [&](int rank_pos, int server) {
        used_server_mask &= ~(std::uint64_t{1} << server_to_index.at(server));
        assignment[static_cast<size_t>(rank_pos)] = -1;
    };

    auto greedy_price_assignment = [&]() {
        std::vector<int> greedy_assignment(static_cast<size_t>(rank_count), -1);
        std::uint64_t greedy_used = 0U;
        double greedy_cost = 0.0;
        for (const int rank_pos : branch_positions) {
            if (timed_out()) {
                return;
            }
            std::vector<std::tuple<double, double, int>> scored;
            const std::uint64_t available_mask = full_server_mask ^ greedy_used;
            for (int idx = 0; idx < rank_count; ++idx) {
                if (timed_out()) {
                    return;
                }
                if ((available_mask & (std::uint64_t{1} << idx)) == 0U) {
                    continue;
                }
                const int server = current_servers[static_cast<size_t>(idx)];
                const auto deltas = incremental_rank_server_price(rank_pos, server, greedy_assignment, greedy_used);
                scored.emplace_back(deltas.first + deltas.second, deltas.first, server);
            }
            if (scored.empty()) {
                continue;
            }
            std::sort(scored.begin(), scored.end(), [](const auto &lhs, const auto &rhs) {
                if (std::get<0>(lhs) != std::get<0>(rhs)) {
                    return std::get<0>(lhs) < std::get<0>(rhs);
                }
                return std::get<2>(lhs) < std::get<2>(rhs);
            });
            const int server = std::get<2>(scored.front());
            greedy_assignment[static_cast<size_t>(rank_pos)] = server;
            greedy_used |= (std::uint64_t{1} << server_to_index.at(server));
            greedy_cost += std::get<1>(scored.front());
        }
        bool complete = true;
        for (const int server : greedy_assignment) {
            if (server < 0) {
                complete = false;
                break;
            }
        }
        if (complete) {
            add_candidate(greedy_cost, greedy_assignment);
        }
    };

    greedy_price_assignment();

    std::function<void(int, double)> dfs = [&](int depth, double current_partial_cost) {
        if (timed_out()) {
            return;
        }
        if (lower_bound(current_partial_cost) >= current_prune_cost() - 1e-12) {
            return;
        }
        if (depth >= static_cast<int>(branch_positions.size())) {
            add_candidate(current_partial_cost, assignment);
            return;
        }
        const int rank_pos = branch_positions[static_cast<size_t>(depth)];
        for (const int server : candidate_servers_for_rank(rank_pos)) {
            if (timed_out()) {
                break;
            }
            const double next_cost = apply_rank(rank_pos, server, current_partial_cost);
            dfs(depth + 1, next_cost);
            rollback_rank(rank_pos, server);
        }
    };
    dfs(0, 0.0);

    py::list result;
    for (const RemapCandidate &candidate : candidates) {
        py::dict assignment_dict;
        for (int pos = 0; pos < rank_count; ++pos) {
            assignment_dict[py::int_(ranks[static_cast<size_t>(pos)])] = py::int_(candidate.assignment[static_cast<size_t>(pos)]);
        }
        result.append(py::make_tuple(candidate.cost, assignment_dict));
    }
    return result;
}

py::tuple aggregate_epoch_prices(
    const py::list &slot_prices,
    const py::list &epoch_active_slots,
    const int global_max_epoch
) {
    py::list epoch_maxima;
    py::list epoch_prices;
    const py::ssize_t slot_count = slot_prices.size();

    for (int epoch = 0; epoch <= global_max_epoch; ++epoch) {
        py::dict edge_prices;
        std::unordered_map<int, double> sender_prices;
        std::unordered_map<int, double> receiver_prices;

        if (epoch >= 0 && epoch < static_cast<int>(epoch_active_slots.size())) {
            py::iterable slot_iterable = py::reinterpret_borrow<py::iterable>(epoch_active_slots[epoch]);
            for (py::handle slot_handle : slot_iterable) {
                const int slot_idx = py::cast<int>(slot_handle);
                if (slot_idx < 0 || slot_idx >= slot_count) {
                    continue;
                }

                py::dict price_state = py::reinterpret_borrow<py::dict>(slot_prices[slot_idx]);
                if (price_state.contains("edge")) {
                    py::dict edge_state = py::reinterpret_borrow<py::dict>(price_state["edge"]);
                    for (auto item : edge_state) {
                        py::object edge_key = py::reinterpret_borrow<py::object>(item.first);
                        const double price = py::cast<double>(item.second);
                        double current = 0.0;
                        if (edge_prices.contains(edge_key)) {
                            current = py::cast<double>(edge_prices[edge_key]);
                        }
                        if (price > current) {
                            edge_prices[edge_key] = py::float_(price);
                        }
                    }
                }
                if (price_state.contains("sender")) {
                    py::dict sender_state = py::reinterpret_borrow<py::dict>(price_state["sender"]);
                    for (auto item : sender_state) {
                        const int server = py::cast<int>(item.first);
                        const double price = py::cast<double>(item.second);
                        auto existing = sender_prices.find(server);
                        if (existing == sender_prices.end() || price > existing->second) {
                            sender_prices[server] = price;
                        }
                    }
                }
                if (price_state.contains("receiver")) {
                    py::dict receiver_state = py::reinterpret_borrow<py::dict>(price_state["receiver"]);
                    for (auto item : receiver_state) {
                        const int server = py::cast<int>(item.first);
                        const double price = py::cast<double>(item.second);
                        auto existing = receiver_prices.find(server);
                        if (existing == receiver_prices.end() || price > existing->second) {
                            receiver_prices[server] = price;
                        }
                    }
                }
            }
        }

        py::dict sender_dict;
        double epoch_max = 0.0;
        for (const auto &[server, price] : sender_prices) {
            sender_dict[py::int_(server)] = py::float_(price);
            if (price > epoch_max) {
                epoch_max = price;
            }
        }

        py::dict receiver_dict;
        for (const auto &[server, price] : receiver_prices) {
            receiver_dict[py::int_(server)] = py::float_(price);
            if (price > epoch_max) {
                epoch_max = price;
            }
        }

        for (auto item : edge_prices) {
            const double price = py::cast<double>(item.second);
            if (price > epoch_max) {
                epoch_max = price;
            }
        }

        py::dict state;
        state["edge"] = edge_prices;
        state["sender"] = sender_dict;
        state["receiver"] = receiver_dict;
        epoch_prices.append(state);
        epoch_maxima.append(py::float_(epoch_max));
    }

    return py::make_tuple(epoch_maxima, epoch_prices);
}

static double dict_get_double(const py::dict &dict, const py::object &key, const double default_value) {
    if (dict.contains(key)) {
        return py::cast<double>(dict[key]);
    }
    return default_value;
}

static double dict_get_double_int(const py::dict &dict, const int key, const double default_value) {
    py::int_ py_key(key);
    if (dict.contains(py_key)) {
        return py::cast<double>(dict[py_key]);
    }
    return default_value;
}

py::dict coarse_pair_epoch_prices(
    const int tenant,
    const std::vector<int> &servers,
    const py::list &epoch_prices,
    const py::dict &pair_edges,
    const py::dict &default_edge_price,
    const py::dict &default_sender_price,
    const py::dict &default_receiver_price
) {
    py::dict lookup;
    const py::ssize_t epoch_count = epoch_prices.size();

    for (py::ssize_t epoch = 0; epoch < epoch_count; ++epoch) {
        py::dict state = py::reinterpret_borrow<py::dict>(epoch_prices[epoch]);
        py::dict edge_prices = py::reinterpret_borrow<py::dict>(state["edge"]);
        py::dict sender_prices = py::reinterpret_borrow<py::dict>(state["sender"]);
        py::dict receiver_prices = py::reinterpret_borrow<py::dict>(state["receiver"]);

        for (const int src_server : servers) {
            const double default_sender = dict_get_double_int(default_sender_price, src_server, 0.0);
            const double sender_price = dict_get_double_int(sender_prices, src_server, default_sender);

            for (const int dst_server : servers) {
                if (src_server == dst_server) {
                    continue;
                }
                const double default_receiver = dict_get_double_int(default_receiver_price, dst_server, 0.0);
                double price = sender_price + dict_get_double_int(receiver_prices, dst_server, default_receiver);

                py::tuple pair_key = py::make_tuple(src_server, dst_server);
                if (pair_edges.contains(pair_key)) {
                    py::iterable edges = py::reinterpret_borrow<py::iterable>(pair_edges[pair_key]);
                    for (py::handle edge_handle : edges) {
                        py::object edge_key = py::reinterpret_borrow<py::object>(edge_handle);
                        const double default_edge = dict_get_double(default_edge_price, edge_key, 0.0);
                        price += dict_get_double(edge_prices, edge_key, default_edge);
                    }
                }

                lookup[py::make_tuple(static_cast<int>(epoch), src_server, dst_server)] = py::float_(price);
            }
        }
    }

    (void)tenant;
    return lookup;
}

PYBIND11_MODULE(_te_accel, m) {
    m.doc() = "C++ kernels for the time-expanded contention estimator";
    m.def("max_min_rates", &max_min_rates, py::arg("resources_by_item"), py::arg("capacities_by_item"), py::arg("demand_rates"));
    m.def("best_rank_swap_delta", &best_rank_swap_delta,
          py::arg("flows"),
          py::arg("rank_to_server"),
          py::arg("branch_order"),
          py::arg("anchor_limit"),
          py::arg("partner_limit"),
          py::arg("pair_epoch_price"));
    m.def("best_coarse_move_delta", &best_coarse_move_delta,
          py::arg("flows"),
          py::arg("rank_to_server"),
          py::arg("branch_order"),
          py::arg("anchor_limit"),
          py::arg("partner_limit"),
          py::arg("block_groups"),
          py::arg("pair_epoch_price"));
    m.def("coarse_local_descent", &coarse_local_descent,
          py::arg("flows"),
          py::arg("rank_to_server"),
          py::arg("branch_order"),
          py::arg("anchor_limit"),
          py::arg("partner_limit"),
          py::arg("block_groups"),
          py::arg("pair_epoch_price"),
          py::arg("passes"));
    m.def("coarse_local_descent_dense", &coarse_local_descent_dense,
          py::arg("flows"),
          py::arg("rank_to_server"),
          py::arg("branch_order"),
          py::arg("anchor_limit"),
          py::arg("partner_limit"),
          py::arg("block_groups"),
          py::arg("pair_epoch_price_dense"),
          py::arg("epoch_count"),
          py::arg("server_count"),
          py::arg("passes"));
    m.def("task_pair_price_lookup_dense", &task_pair_price_lookup_dense,
          py::arg("task_ids"),
          py::arg("exposure_slots_by_task"),
          py::arg("candidate_servers"),
          py::arg("sender_prices"),
          py::arg("receiver_prices"),
          py::arg("edge_prices"),
          py::arg("path_edges_by_pair"));
    m.def("scored_swap_pairs_from_state", &scored_swap_pairs_from_state,
          py::arg("hot_ranks"),
          py::arg("partner_ranks"),
          py::arg("task_infos"),
          py::arg("tenant_mapping"),
          py::arg("task_active_slots"),
          py::arg("task_ready_slots"),
          py::arg("tenant"),
          py::arg("slot_prices"),
          py::arg("base_sender"),
          py::arg("base_receiver"),
          py::arg("base_edge"),
          py::arg("edge_to_idx"),
          py::arg("path_edges_by_pair"));
    m.def("time_expanded_slot_prices_batch", &time_expanded_slot_prices_batch,
          py::arg("slot_resource_pressure"),
          py::arg("slot_active_tasks"),
          py::arg("slot_service_rates"),
          py::arg("task_state"),
          py::arg("critical_resources_by_slot"),
          py::arg("edge_capacity"),
          py::arg("server_send_capacity"),
          py::arg("server_recv_capacity"),
          py::arg("link_price_beta"),
          py::arg("link_price_gamma"),
          py::arg("critical_path_price_beta"));
    m.def("price_guided_remap_candidates", &price_guided_remap_candidates,
          py::arg("ranks"),
          py::arg("current_servers"),
          py::arg("branch_ranks"),
          py::arg("task_infos"),
          py::arg("task_pair_price"),
          py::arg("max_price_candidates"),
          py::arg("time_budget_seconds"));
    m.def("price_guided_remap_candidates_dense", &price_guided_remap_candidates_dense,
          py::arg("ranks"),
          py::arg("current_servers"),
          py::arg("branch_ranks"),
          py::arg("task_infos"),
          py::arg("exposure_slots_by_task"),
          py::arg("sender_prices"),
          py::arg("receiver_prices"),
          py::arg("edge_prices"),
          py::arg("path_edges_by_pair"),
          py::arg("max_price_candidates"),
          py::arg("time_budget_seconds"));
    m.def("aggregate_epoch_prices", &aggregate_epoch_prices,
          py::arg("slot_prices"),
          py::arg("epoch_active_slots"),
          py::arg("global_max_epoch"));
    m.def("coarse_pair_epoch_prices", &coarse_pair_epoch_prices,
          py::arg("tenant"),
          py::arg("servers"),
          py::arg("epoch_prices"),
          py::arg("pair_edges"),
          py::arg("default_edge_price"),
          py::arg("default_sender_price"),
          py::arg("default_receiver_price"));
    py::class_<TimeExpandedScoreEngine>(m, "TimeExpandedScoreEngine")
        .def(py::init<
             std::vector<int>,
             std::vector<double>,
             std::vector<double>,
             std::vector<double>,
             std::vector<double>,
             std::vector<std::vector<std::vector<std::vector<int>>>>,
             py::list,
             py::object,
             py::object
        >())
        .def("evaluate", &TimeExpandedScoreEngine::evaluate)
        .def("evaluate_pipeline", &TimeExpandedScoreEngine::evaluate_pipeline);
}
