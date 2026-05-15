#pragma once

#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

// One (node, distance) entry from a breadth-first search.
struct BfsEntry {
    std::uint64_t node;
    std::uint64_t distance;
};

// Reusable scratch state for `breadth_first_search`. Reset via `reset(graph)`
// before each call so the visited-flag and distance buffers grow once and stay
// allocated across many BFS invocations on the same graph.
class BfsWorkspace {
public:
    BfsWorkspace() = default;

    void reset(const UndirectedGraph &graph) {
        const auto n_nodes = static_cast<std::size_t>(graph.number_of_nodes());
        visited_.assign(n_nodes, 0);
        distance_.assign(n_nodes, 0);
    }

    [[nodiscard]] std::vector<unsigned char> &visited() { return visited_; }
    [[nodiscard]] std::vector<std::uint64_t> &distance() { return distance_; }

private:
    std::vector<unsigned char> visited_;
    std::vector<std::uint64_t> distance_;
};

// Distance value reported for the source node itself (distance == 0).
inline constexpr std::uint64_t bfs_source_distance = 0;

// Sentinel for "no maximum distance" — the BFS expands until the entire
// connected component of `source` has been reported.
inline constexpr std::uint64_t bfs_no_max_distance =
    std::numeric_limits<std::uint64_t>::max();

// Run a breadth-first search on `graph` starting from `source`, reporting every
// node reached within `max_distance` hops (inclusive) in BFS order.
//
// The `source` node itself is reported with distance 0. When
// `include_source` is false the source is excluded from the output (useful for
// "nodes within k hops, excluding self" queries, e.g. lifted-edge insertion).
// When `max_distance == bfs_no_max_distance` the search expands until the
// entire connected component is visited.
//
// `workspace` lets the caller reuse internal buffers across calls on the same
// graph; pass a fresh workspace for a one-off call.
inline std::vector<BfsEntry> breadth_first_search(
    const UndirectedGraph &graph,
    const std::uint64_t source,
    const std::uint64_t max_distance,
    const bool include_source,
    BfsWorkspace &workspace
) {
    if (source >= graph.number_of_nodes()) {
        throw std::invalid_argument(
            "source must be < number_of_nodes"
        );
    }
    workspace.reset(graph);
    auto &visited = workspace.visited();
    auto &distance = workspace.distance();

    std::vector<BfsEntry> result;
    std::queue<std::uint64_t> queue;
    queue.push(source);
    visited[static_cast<std::size_t>(source)] = 1;
    distance[static_cast<std::size_t>(source)] = 0;
    if (include_source) {
        result.push_back({source, 0});
    }

    while (!queue.empty()) {
        const auto node = queue.front();
        queue.pop();
        const auto node_distance = distance[static_cast<std::size_t>(node)];
        if (node_distance >= max_distance) {
            continue;
        }
        const auto next_distance = node_distance + 1;
        for (const auto adjacency : graph.node_adjacency(node)) {
            const auto neighbor = adjacency.node;
            if (visited[static_cast<std::size_t>(neighbor)] != 0) {
                continue;
            }
            visited[static_cast<std::size_t>(neighbor)] = 1;
            distance[static_cast<std::size_t>(neighbor)] = next_distance;
            result.push_back({neighbor, next_distance});
            queue.push(neighbor);
        }
    }
    return result;
}

inline std::vector<BfsEntry> breadth_first_search(
    const UndirectedGraph &graph,
    const std::uint64_t source,
    const std::uint64_t max_distance = bfs_no_max_distance,
    const bool include_source = true
) {
    BfsWorkspace workspace;
    return breadth_first_search(graph, source, max_distance, include_source, workspace);
}

} // namespace bioimage_cpp::graph
