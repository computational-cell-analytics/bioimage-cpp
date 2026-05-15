#pragma once

#include "bioimage_cpp/detail/indexed_heap.hxx"
#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::lifted_multicut::detail {

inline constexpr std::size_t no_edge = std::numeric_limits<std::size_t>::max();

// Per-super-node adjacency entry.
struct NeighborEntry {
    std::size_t neighbor;
    std::size_t edge_id;
};

// Per-edge data stored in a flat vector indexed by stable edge_id. Same shape
// as multicut::detail::DynamicEdge but with `is_lifted` replacing the multicut
// constraint flag, since lifted edges have different heap semantics:
//   * lifted edges are excluded from the heap until they become non-lifted;
//   * lifted propagates iff *both* inputs are lifted (vs. constraint, which
//     propagates if *either* input is constrained).
struct DynamicEdge {
    std::size_t u = 0;
    std::size_t v = 0;
    double weight = 0.0;
    unsigned char is_lifted = 0;
};

using EdgeHeap = bioimage_cpp::detail::DenseIndexedHeap<double>;

struct DynamicGraph {
    DynamicGraph() = default;

    explicit DynamicGraph(const UndirectedGraph &lifted_graph) {
        reset(lifted_graph);
    }

    void reset(const UndirectedGraph &lifted_graph) {
        const auto n_nodes = static_cast<std::size_t>(lifted_graph.number_of_nodes());
        const auto n_edges = static_cast<std::size_t>(lifted_graph.number_of_edges());

        for (auto &adj : adjacency) {
            adj.clear();
        }
        adjacency.resize(n_nodes);
        for (std::uint64_t node = 0; node < lifted_graph.number_of_nodes(); ++node) {
            const auto degree = lifted_graph.node_adjacency(node).size();
            adjacency[static_cast<std::size_t>(node)].reserve(degree);
        }

        alive.assign(n_nodes, true);
        alive_count = n_nodes;
        scratch_edge_id.assign(n_nodes, no_edge);
        edges.resize(n_edges);
    }

    std::vector<std::vector<NeighborEntry>> adjacency;
    std::vector<DynamicEdge> edges;
    std::vector<bool> alive;
    std::size_t alive_count;
    std::vector<std::size_t> scratch_edge_id;
};

// Initialize the dynamic graph from a lifted graph + per-edge weights, where
// the first `n_base_edges` entries of the lifted graph are the base edges.
// Optional Gaussian noise is added to the weights, mirroring the multicut
// greedy additive flow.
//
// Lifted edges enter the dynamic graph but are *not* pushed onto the heap (the
// driver guards them via `is_lifted`). Non-lifted (base) edges are heap-pushed
// at their (possibly noisy) weight.
inline void initialize_dynamic_graph(
    const UndirectedGraph &lifted_graph,
    const std::vector<double> &weights,
    const std::uint64_t n_base_edges,
    DynamicGraph &dynamic_graph,
    EdgeHeap &heap,
    const bool add_noise = false,
    const int seed = 42,
    const double sigma = 1.0
) {
    const auto n_edges = static_cast<std::size_t>(lifted_graph.number_of_edges());
    heap.reset_capacity(n_edges);

    std::vector<EdgeHeap::Entry> heap_entries;
    heap_entries.reserve(static_cast<std::size_t>(n_base_edges));

    std::mt19937 generator(seed);
    std::normal_distribution<double> noise(0.0, sigma);
    for (std::uint64_t edge = 0; edge < lifted_graph.number_of_edges(); ++edge) {
        const auto uv = lifted_graph.uv(edge);
        const auto u = static_cast<std::size_t>(uv.first);
        const auto v = static_cast<std::size_t>(uv.second);
        double weight = weights[static_cast<std::size_t>(edge)];
        if (add_noise) {
            weight += noise(generator);
        }
        const auto edge_id = static_cast<std::size_t>(edge);
        auto &e = dynamic_graph.edges[edge_id];
        e.u = u;
        e.v = v;
        e.weight = weight;
        e.is_lifted = (edge < n_base_edges) ? 0 : 1;
        dynamic_graph.adjacency[u].push_back({v, edge_id});
        dynamic_graph.adjacency[v].push_back({u, edge_id});
        if (e.is_lifted == 0) {
            heap_entries.push_back({edge_id, weight});
        }
    }

    heap.build_heap(std::move(heap_entries));
}

namespace internal {

inline bool erase_by_neighbor(std::vector<NeighborEntry> &list, const std::size_t target) {
    for (std::size_t i = 0; i < list.size(); ++i) {
        if (list[i].neighbor == target) {
            list[i] = list.back();
            list.pop_back();
            return true;
        }
    }
    return false;
}

inline void rename_neighbor(
    std::vector<NeighborEntry> &list,
    const std::size_t from_node,
    const std::size_t to_node
) {
    for (auto &entry : list) {
        if (entry.neighbor == from_node) {
            entry.neighbor = to_node;
            return;
        }
    }
}

} // namespace internal

// Contract the edge between u and v in the dynamic graph. Folds the
// smaller-degree super-node into the larger one. Heap is kept in sync with the
// following lifted-aware rules for parallel-edge merges:
//   * both inputs lifted  → result lifted (stays out of heap).
//   * both inputs base    → result base, priority = summed weight.
//   * one of each kind    → result base, edge re-enters or is updated in the
//                           heap with priority = summed weight.
//
// Precondition: the edge being contracted is non-lifted (the driver only
// pops base edges off the heap).
inline std::size_t merge_dynamic_nodes(
    DynamicGraph &dynamic_graph,
    bioimage_cpp::detail::UnionFind &sets,
    EdgeHeap &heap,
    std::size_t u,
    std::size_t v
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

    for (const auto &entry : dynamic_graph.adjacency[stable]) {
        dynamic_graph.scratch_edge_id[entry.neighbor] = entry.edge_id;
    }

    const auto contracted_edge_id = dynamic_graph.scratch_edge_id[removed];
    heap.erase(contracted_edge_id);
    dynamic_graph.scratch_edge_id[removed] = no_edge;
    internal::erase_by_neighbor(dynamic_graph.adjacency[stable], removed);

    const auto removed_neighbors = dynamic_graph.adjacency[removed];

    for (const auto &entry : removed_neighbors) {
        const auto neighbor = entry.neighbor;
        const auto removed_edge_id = entry.edge_id;
        if (neighbor == stable) {
            continue;
        }

        const auto existing_id = dynamic_graph.scratch_edge_id[neighbor];
        if (existing_id == no_edge) {
            // No stable-side edge to `neighbor`: re-key the removed-side edge
            // by replacing `removed` with `stable`. Lifted flag and weight are
            // preserved, so the heap state stays consistent (lifted edges stay
            // off the heap; base edges keep their existing heap entry).
            dynamic_graph.adjacency[stable].push_back({neighbor, removed_edge_id});
            dynamic_graph.scratch_edge_id[neighbor] = removed_edge_id;
            internal::rename_neighbor(dynamic_graph.adjacency[neighbor], removed, stable);
            auto &e = dynamic_graph.edges[removed_edge_id];
            if (e.u == removed) {
                e.u = stable;
            } else {
                e.v = stable;
            }
        } else {
            // Stable already has an edge to `neighbor`. Sum the weights and
            // resolve the lifted flag.
            auto &keep = dynamic_graph.edges[existing_id];
            const auto &fold = dynamic_graph.edges[removed_edge_id];
            const bool keep_was_lifted = keep.is_lifted != 0;
            const bool fold_was_lifted = fold.is_lifted != 0;
            keep.weight += fold.weight;
            const bool result_lifted = keep_was_lifted && fold_was_lifted;
            keep.is_lifted = result_lifted ? 1 : 0;

            internal::erase_by_neighbor(dynamic_graph.adjacency[neighbor], removed);

            if (result_lifted) {
                // Both inputs were lifted: the result is still lifted, so the
                // heap should not contain either side. (Both were already off.)
            } else if (keep_was_lifted && !fold_was_lifted) {
                // The fold side was the base edge and lives on the heap; we
                // are dropping that id. The keep side (formerly lifted) needs
                // to be inserted into the heap with the summed weight.
                heap.erase(removed_edge_id);
                heap.push(existing_id, keep.weight);
            } else if (!keep_was_lifted && fold_was_lifted) {
                // The keep side is on the heap; the fold side was lifted and
                // is not. Update the keep entry's priority to the summed weight.
                heap.change(existing_id, keep.weight);
            } else {
                // Both base edges: drop the removed-side heap entry and
                // refresh the kept entry's priority.
                heap.erase(removed_edge_id);
                heap.change(existing_id, keep.weight);
            }
        }
    }

    for (const auto &entry : dynamic_graph.adjacency[stable]) {
        dynamic_graph.scratch_edge_id[entry.neighbor] = no_edge;
    }

    dynamic_graph.adjacency[removed].clear();
    dynamic_graph.alive[removed] = false;
    --dynamic_graph.alive_count;
    return stable;
}

inline std::vector<std::uint64_t> labels_from_sets(
    bioimage_cpp::detail::UnionFind &sets,
    const UndirectedGraph &graph
) {
    return dense_labels_from_union_find(sets, graph.number_of_nodes());
}

inline std::size_t stop_node_count(
    const UndirectedGraph &graph,
    const double node_num_stop
) {
    return node_num_stop >= 1.0
        ? static_cast<std::size_t>(node_num_stop)
        : static_cast<std::size_t>(double(graph.number_of_nodes()) * node_num_stop + 0.5);
}

} // namespace bioimage_cpp::graph::lifted_multicut::detail
