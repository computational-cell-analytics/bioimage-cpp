#pragma once

#include "bioimage_cpp/detail/union_find.hxx"
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

struct DynamicEdge {
    double weight = 0.0;
    std::uint64_t edition = 0;
};

struct DynamicGraph {
    explicit DynamicGraph(const std::size_t size) : adjacency(size), constraints(size), alive(size, true), alive_count(size) {
    }

    explicit DynamicGraph(const UndirectedGraph &graph)
        : adjacency(static_cast<std::size_t>(graph.number_of_nodes())),
          constraints(static_cast<std::size_t>(graph.number_of_nodes())),
          alive(static_cast<std::size_t>(graph.number_of_nodes()), true),
          alive_count(static_cast<std::size_t>(graph.number_of_nodes())) {
        for (std::uint64_t node = 0; node < graph.number_of_nodes(); ++node) {
            const auto degree = graph.node_adjacency(node).size();
            adjacency[static_cast<std::size_t>(node)].reserve(degree);
        }
    }

    [[nodiscard]] bool edge_exists(const std::size_t u, const std::size_t v) const {
        return alive[u] && alive[v] && adjacency[u].find(v) != adjacency[u].end();
    }

    [[nodiscard]] const DynamicEdge *edge(const std::size_t u, const std::size_t v) const {
        if (!alive[u] || !alive[v]) {
            return nullptr;
        }
        const auto found = adjacency[u].find(v);
        return found == adjacency[u].end() ? nullptr : &found->second;
    }

    [[nodiscard]] bool has_constraint(const std::size_t u, const std::size_t v) const {
        return constraints[u].find(v) != constraints[u].end();
    }

    void add_constraint(const std::size_t u, const std::size_t v) {
        constraints[u].insert(v);
        constraints[v].insert(u);
    }

    void set_initial_edge(const std::size_t u, const std::size_t v, const double weight) {
        adjacency[u][v] = DynamicEdge{weight, 0};
        adjacency[v][u] = DynamicEdge{weight, 0};
    }

    std::uint64_t update_edge(const std::size_t u, const std::size_t v, const double weight) {
        const auto current = adjacency[u].find(v);
        const auto edition = current == adjacency[u].end() ? 0 : current->second.edition + 1;
        adjacency[u][v] = DynamicEdge{weight, edition};
        adjacency[v][u] = DynamicEdge{weight, edition};
        return edition;
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

    std::vector<std::unordered_map<std::size_t, DynamicEdge>> adjacency;
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
        dynamic_graph.set_initial_edge(u, v, weight);
        queue.push(QueueEdge{u, v, absolute_priority ? std::abs(weight) : weight, 0});
    }
}

inline std::vector<std::uint64_t> labels_from_sets(
    bioimage_cpp::detail::UnionFind &sets,
    const UndirectedGraph &graph
) {
    return dense_labels_from_union_find(sets, graph.number_of_nodes());
}

inline std::size_t stop_node_count(const UndirectedGraph &graph, const double node_num_stop) {
    return node_num_stop >= 1.0
        ? static_cast<std::size_t>(node_num_stop)
        : static_cast<std::size_t>(double(graph.number_of_nodes()) * node_num_stop + 0.5);
}

inline std::size_t merge_dynamic_nodes(
    DynamicGraph &dynamic_graph,
    bioimage_cpp::detail::UnionFind &sets,
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
    auto stable = u;
    auto removed = v;
    if (dynamic_graph.adjacency[stable].size() < dynamic_graph.adjacency[removed].size()) {
        std::swap(stable, removed);
    }
    sets.merge_to(stable, removed);
    std::vector<std::pair<std::size_t, DynamicEdge>> neighbors(
        dynamic_graph.adjacency[removed].begin(),
        dynamic_graph.adjacency[removed].end()
    );
    dynamic_graph.adjacency[stable].erase(removed);
    dynamic_graph.constraints[stable].erase(removed);
    for (const auto &[neighbor, removed_edge] : neighbors) {
        if (neighbor == stable) {
            continue;
        }
        const auto current = dynamic_graph.edge(stable, neighbor);
        const auto merged_weight = (current == nullptr ? 0.0 : current->weight) + removed_edge.weight;
        if (dynamic_graph.has_constraint(removed, neighbor)) {
            dynamic_graph.add_constraint(stable, neighbor);
        }
        const auto edition = dynamic_graph.update_edge(stable, neighbor, merged_weight);
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
