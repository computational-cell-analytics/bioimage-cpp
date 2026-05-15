#pragma once

#include "bioimage_cpp/detail/indexed_heap.hxx"
#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/connected_components.hxx"
#include "bioimage_cpp/graph/multicut/objective.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::multicut::detail {

inline constexpr std::size_t no_edge = std::numeric_limits<std::size_t>::max();

// Per-super-node adjacency entry.
struct NeighborEntry {
    std::size_t neighbor;
    std::size_t edge_id;
};

// Per-edge data stored in a flat vector indexed by stable edge_id. Edge ids
// never grow past the original number of input edges — when two edges fold
// into one during a merge, one id survives and the other becomes orphaned
// (no adjacency entry or heap entry references it again).
struct DynamicEdge {
    std::size_t u = 0;
    std::size_t v = 0;
    double weight = 0.0;
    unsigned char is_constraint = 0;
};

// Heap keyed by dense stable edge_id. The vector-backed DenseLocator updates
// in O(1) per sift step, which was the missing ingredient for greedy_additive
// to beat the std::priority_queue baseline.
using EdgeHeap = bioimage_cpp::detail::DenseIndexedHeap<double>;

struct DynamicGraph {
    DynamicGraph() = default;

    explicit DynamicGraph(const UndirectedGraph &graph) {
        reset(graph);
    }

    // Reuse the buffers of an existing DynamicGraph for a new input graph.
    // Inner adjacency vectors are `clear()`-ed (keeping capacity) and the
    // outer container is resized; degree-based reserves prevent any per-edge
    // adjacency growth during initialise.
    void reset(const UndirectedGraph &graph) {
        const auto n_nodes = static_cast<std::size_t>(graph.number_of_nodes());
        const auto n_edges = static_cast<std::size_t>(graph.number_of_edges());

        for (auto &adj : adjacency) {
            adj.clear();
        }
        adjacency.resize(n_nodes);
        for (std::uint64_t node = 0; node < graph.number_of_nodes(); ++node) {
            const auto degree = graph.node_adjacency(node).size();
            adjacency[static_cast<std::size_t>(node)].reserve(degree);
        }

        alive.assign(n_nodes, true);
        alive_count = n_nodes;
        scratch_edge_id.assign(n_nodes, no_edge);
        edges.resize(n_edges);
    }

    // O(degree(u)). Returns no_edge when (u, v) is not an edge.
    [[nodiscard]] std::size_t find_edge(const std::size_t u, const std::size_t v) const {
        if (u >= adjacency.size() || v >= adjacency.size()) {
            return no_edge;
        }
        for (const auto &entry : adjacency[u]) {
            if (entry.neighbor == v) {
                return entry.edge_id;
            }
        }
        return no_edge;
    }

    [[nodiscard]] bool edge_exists(const std::size_t u, const std::size_t v) const {
        return find_edge(u, v) != no_edge;
    }

    [[nodiscard]] bool has_constraint(const std::size_t u, const std::size_t v) const {
        const auto id = find_edge(u, v);
        return id != no_edge && edges[id].is_constraint != 0;
    }

    std::vector<std::vector<NeighborEntry>> adjacency;
    std::vector<DynamicEdge> edges;
    std::vector<bool> alive;
    std::size_t alive_count;
    // Sized to #super-nodes, all entries == no_edge between merges. Used by
    // merge_dynamic_nodes to find the existing stable-side edge for each of
    // removed's neighbors in O(1) without hashing.
    std::vector<std::size_t> scratch_edge_id;
};

inline double priority_for(const double weight, const bool absolute_priority) {
    return absolute_priority ? std::abs(weight) : weight;
}

inline void initialize_dynamic_graph(
    const UndirectedGraph &graph,
    const std::vector<double> &costs,
    DynamicGraph &dynamic_graph,
    EdgeHeap &heap,
    const bool absolute_priority,
    const bool add_noise = false,
    const int seed = 42,
    const double sigma = 1.0
) {
    const auto n_edges = static_cast<std::size_t>(graph.number_of_edges());
    heap.reset_capacity(n_edges);

    std::vector<EdgeHeap::Entry> heap_entries;
    heap_entries.reserve(n_edges);

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
        const auto edge_id = static_cast<std::size_t>(edge);
        auto &e = dynamic_graph.edges[edge_id];
        e.u = u;
        e.v = v;
        e.weight = weight;
        e.is_constraint = 0;
        dynamic_graph.adjacency[u].push_back({v, edge_id});
        dynamic_graph.adjacency[v].push_back({u, edge_id});
        heap_entries.push_back({edge_id, priority_for(weight, absolute_priority)});
    }

    // Floyd's heapify is O(n_edges) — meaningfully faster than n_edges
    // successive `push` calls when n_edges is in the hundreds of thousands
    // (the common case for graphs we run the multicut on).
    heap.build_heap(std::move(heap_entries));
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

namespace internal {

// Erase the first entry with neighbor == target from `list` (swap-with-back).
// Returns whether anything was erased.
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

// Rename the first entry whose neighbor == from_node to point to to_node.
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

// Contract the edge between u and v in the dynamic graph. The smaller-degree
// super-node is folded into the larger-degree one to keep the per-super-node
// adjacency growth amortized. The heap is kept in sync without staleness: each
// edge id appears at most once.
inline std::size_t merge_dynamic_nodes(
    DynamicGraph &dynamic_graph,
    bioimage_cpp::detail::UnionFind &sets,
    EdgeHeap &heap,
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

    // Stamp stable's neighbors so each removed-neighbor lookup is O(1).
    for (const auto &entry : dynamic_graph.adjacency[stable]) {
        dynamic_graph.scratch_edge_id[entry.neighbor] = entry.edge_id;
    }

    // The contracted edge (stable, removed) must be in stable's adjacency.
    const auto contracted_edge_id = dynamic_graph.scratch_edge_id[removed];
    heap.erase(contracted_edge_id);
    dynamic_graph.scratch_edge_id[removed] = no_edge;
    internal::erase_by_neighbor(dynamic_graph.adjacency[stable], removed);

    // Snapshot removed's neighbors before mutating its adjacency.
    const auto removed_neighbors = dynamic_graph.adjacency[removed];

    for (const auto &entry : removed_neighbors) {
        const auto neighbor = entry.neighbor;
        const auto removed_edge_id = entry.edge_id;
        if (neighbor == stable) {
            continue;
        }

        const auto existing_id = dynamic_graph.scratch_edge_id[neighbor];
        if (existing_id == no_edge) {
            // No existing stable-side edge to `neighbor`. Re-key the
            // removed-side edge by replacing `removed` with `stable` on both
            // sides. The edge_id, weight and constraint flag are preserved,
            // so the heap entry remains valid without any priority change.
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
            // Stable already has an edge to `neighbor`; fold removed's weight
            // (and constraint flag) into it. Then drop the removed-side edge.
            auto &keep = dynamic_graph.edges[existing_id];
            const auto &fold = dynamic_graph.edges[removed_edge_id];
            keep.weight += fold.weight;
            const bool propagated_constraint =
                keep.is_constraint != 0 || fold.is_constraint != 0;
            keep.is_constraint = propagated_constraint ? 1 : 0;

            heap.erase(removed_edge_id);
            internal::erase_by_neighbor(dynamic_graph.adjacency[neighbor], removed);

            if (propagated_constraint) {
                heap.erase(existing_id);
            } else {
                heap.change(existing_id, priority_for(keep.weight, absolute_priority));
            }
        }
    }

    // Clear scratch via the updated stable adjacency (includes appended entries).
    for (const auto &entry : dynamic_graph.adjacency[stable]) {
        dynamic_graph.scratch_edge_id[entry.neighbor] = no_edge;
    }

    dynamic_graph.adjacency[removed].clear();
    dynamic_graph.alive[removed] = false;
    --dynamic_graph.alive_count;
    return stable;
}

} // namespace bioimage_cpp::graph::multicut::detail
