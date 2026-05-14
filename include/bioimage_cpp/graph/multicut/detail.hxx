#pragma once

#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::multicut::detail {

struct DynamicGraph {
    explicit DynamicGraph(const std::size_t size) : adjacency(size), constraints(size), alive(size, true), alive_count(size) {
    }

    [[nodiscard]] bool edge_exists(const std::size_t u, const std::size_t v) const {
        return alive[u] && alive[v] && adjacency[u].find(v) != adjacency[u].end();
    }

    [[nodiscard]] bool has_constraint(const std::size_t u, const std::size_t v) const {
        return constraints[u].find(v) != constraints[u].end();
    }

    void add_constraint(const std::size_t u, const std::size_t v) {
        constraints[u].insert(v);
        constraints[v].insert(u);
    }

    void set_edge(const std::size_t u, const std::size_t v, const double weight) {
        adjacency[u][v] = weight;
        adjacency[v][u] = weight;
    }

    void remove_node(const std::size_t node) {
        for (const auto &entry : adjacency[node]) {
            adjacency[entry.first].erase(node);
            constraints[entry.first].erase(node);
        }
        adjacency[node].clear();
        constraints[node].clear();
        alive[node] = false;
        --alive_count;
    }

    std::vector<std::unordered_map<std::size_t, double>> adjacency;
    std::vector<std::unordered_set<std::size_t>> constraints;
    std::vector<bool> alive;
    std::size_t alive_count;
};

struct QueueEdge {
    std::size_t u = 0;
    std::size_t v = 0;
    double priority = 0.0;
    std::uint64_t edition = 0;

    bool operator<(const QueueEdge &other) const {
        return priority < other.priority;
    }
};

inline void initialize_dynamic_graph(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    DynamicGraph &dynamic_graph,
    std::vector<std::unordered_map<std::size_t, std::uint64_t>> &editions,
    std::priority_queue<QueueEdge> &queue,
    const bool absolute_priority,
    const bool add_noise = false,
    const int seed = 42,
    const double sigma = 1.0
) {
    std::mt19937 generator(seed);
    std::normal_distribution<double> noise(0.0, sigma);
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        const auto uv = graph.uv(edge);
        const auto u = static_cast<std::size_t>(uv.first);
        const auto v = static_cast<std::size_t>(uv.second);
        double weight = costs[static_cast<std::size_t>(edge)];
        if (add_noise) {
            weight += noise(generator);
        }
        dynamic_graph.set_edge(u, v, weight);
        editions[u][v] = 0;
        editions[v][u] = 0;
        queue.push(QueueEdge{u, v, absolute_priority ? std::abs(weight) : weight, 0});
    }
}

inline std::vector<std::uint64_t> labels_from_sets(UnionFind &sets, const UndirectedGraph &graph) {
    return dense_labels_from_union_find(sets, graph.number_of_nodes());
}

inline std::size_t stop_node_count(const UndirectedGraph &graph, const double node_num_stop) {
    return node_num_stop >= 1.0
        ? static_cast<std::size_t>(node_num_stop)
        : static_cast<std::size_t>(double(graph.number_of_nodes()) * node_num_stop + 0.5);
}

inline std::size_t merge_dynamic_nodes(
    DynamicGraph &dynamic_graph,
    UnionFind &sets,
    std::vector<std::unordered_map<std::size_t, std::uint64_t>> &editions,
    std::priority_queue<QueueEdge> &queue,
    std::size_t u,
    std::size_t v,
    const bool absolute_priority
) {
    u = static_cast<std::size_t>(sets.find(u));
    v = static_cast<std::size_t>(sets.find(v));
    if (u == v) {
        return u;
    }
    sets.merge(u, v);
    const auto stable = static_cast<std::size_t>(sets.find(u));
    const auto removed = (stable == u) ? v : u;
    std::vector<std::pair<std::size_t, double>> neighbors(
        dynamic_graph.adjacency[removed].begin(),
        dynamic_graph.adjacency[removed].end()
    );
    dynamic_graph.adjacency[stable].erase(removed);
    dynamic_graph.constraints[stable].erase(removed);
    for (const auto &[neighbor, removed_weight] : neighbors) {
        if (neighbor == stable) {
            continue;
        }
        const auto current = dynamic_graph.edge_exists(stable, neighbor)
            ? dynamic_graph.adjacency[stable][neighbor]
            : 0.0;
        const auto merged_weight = current + removed_weight;
        if (dynamic_graph.has_constraint(removed, neighbor)) {
            dynamic_graph.add_constraint(stable, neighbor);
        }
        dynamic_graph.set_edge(stable, neighbor, merged_weight);
        const auto edition = ++editions[std::min(stable, neighbor)][std::max(stable, neighbor)];
        queue.push(QueueEdge{
            std::min(stable, neighbor),
            std::max(stable, neighbor),
            absolute_priority ? std::abs(merged_weight) : merged_weight,
            edition
        });
    }
    dynamic_graph.remove_node(removed);
    return stable;
}

} // namespace bioimage_cpp::graph::multicut::detail
