#pragma once

#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph::detail {

struct AgreementContraction {
    // Contracted graph: one node per agreement component, dense ids in
    // [0, number_of_components).
    UndirectedGraph contracted_graph;
    // For every original edge id: the contracted edge id it maps onto, or
    // -1 if both endpoints collapsed into the same component.
    std::vector<std::int64_t> contracted_edge_of_original;
    // For every original node id: its dense agreement-component id.
    std::vector<std::uint64_t> root_of_node;
};

// Build the agreement-projection contracted graph: merge `(u, v)` iff every
// proposal labels `u` and `v` identically, then dense-relabel and assemble the
// contracted graph with one edge per distinct surviving (root_u, root_v) pair.
//
// `proposals` is a row-major buffer of shape (n_proposals, number_of_nodes).
// Passing 0 proposals collapses every edge (all proposals trivially agree)
// and yields a single-node contracted graph; we reject that as a usage error.
//
// Implementation notes:
//   - Dense relabeling uses a flat sentinel array indexed by root id rather
//     than a hash map; roots live in [0, number_of_nodes) so this is O(N).
//   - Surviving (min_root, max_root) pairs are collected and sorted by a
//     packed 64-bit key, then deduped sequentially. The contracted graph is
//     built in one pass via `UndirectedGraph::from_sorted_unique_edges`,
//     bypassing per-edge hash insertion in `insert_edge`.
//   - The contracted graph's `edge_lookup_` is left empty because the
//     fusion-move sub-solver only iterates edges and adjacency, never calls
//     `find_edge`.
inline AgreementContraction contract_by_agreement(
    const UndirectedGraph &graph,
    const std::uint64_t *proposals,
    const std::size_t n_proposals,
    const std::size_t n_nodes_per_proposal
) {
    if (n_proposals == 0) {
        throw std::invalid_argument("at least one proposal is required");
    }
    const auto number_of_nodes = graph.number_of_nodes();
    const auto number_of_edges = graph.number_of_edges();
    if (n_nodes_per_proposal != static_cast<std::size_t>(number_of_nodes)) {
        throw std::invalid_argument(
            "proposal width must equal number_of_nodes, got " +
            std::to_string(n_nodes_per_proposal) + " for number_of_nodes=" +
            std::to_string(number_of_nodes)
        );
    }

    bioimage_cpp::detail::UnionFind sets(static_cast<std::size_t>(number_of_nodes));
    for (std::uint64_t edge = 0; edge < number_of_edges; ++edge) {
        const auto uv = graph.uv(edge);
        const auto u = static_cast<std::size_t>(uv.first);
        const auto v = static_cast<std::size_t>(uv.second);
        bool agree = true;
        for (std::size_t p = 0; p < n_proposals; ++p) {
            const auto *row = proposals + p * n_nodes_per_proposal;
            if (row[u] != row[v]) {
                agree = false;
                break;
            }
        }
        if (agree) {
            sets.merge(uv.first, uv.second);
        }
    }

    // Dense-relabel UFD roots in one O(N) pass with a sentinel array.
    constexpr std::uint64_t unset = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> dense_of_raw(
        static_cast<std::size_t>(number_of_nodes), unset
    );
    std::vector<std::uint64_t> root_of_node(static_cast<std::size_t>(number_of_nodes));
    std::uint64_t number_of_components = 0;
    for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
        const auto raw = sets.find(node);
        auto dense = dense_of_raw[static_cast<std::size_t>(raw)];
        if (dense == unset) {
            dense = number_of_components++;
            dense_of_raw[static_cast<std::size_t>(raw)] = dense;
        }
        root_of_node[static_cast<std::size_t>(node)] = dense;
    }

    // Sort key for surviving edges. The lower 32 bits hold `max_root` and
    // the upper 32 bits hold `min_root` so a single uint64 comparison
    // suffices. This requires number_of_components to fit in 32 bits, which
    // is always true for graphs we can fit in memory.
    if (number_of_components > (std::uint64_t{1} << 32)) {
        throw std::runtime_error(
            "number_of_components exceeds 2^32 — contraction packing assumption violated"
        );
    }

    struct Survivor {
        std::uint64_t key;             // (min_root << 32) | max_root
        std::uint64_t original_edge;
    };
    std::vector<Survivor> survivors;
    survivors.reserve(static_cast<std::size_t>(number_of_edges));

    std::vector<std::int64_t> contracted_edge_of_original(
        static_cast<std::size_t>(number_of_edges), -1
    );

    for (std::uint64_t edge = 0; edge < number_of_edges; ++edge) {
        const auto uv = graph.uv(edge);
        auto ru = root_of_node[static_cast<std::size_t>(uv.first)];
        auto rv = root_of_node[static_cast<std::size_t>(uv.second)];
        if (ru == rv) {
            continue;
        }
        if (ru > rv) {
            std::swap(ru, rv);
        }
        survivors.push_back(Survivor{(ru << 32) | rv, edge});
    }

    std::sort(
        survivors.begin(),
        survivors.end(),
        [](const Survivor &a, const Survivor &b) {
            return a.key < b.key;
        }
    );

    std::vector<UndirectedGraph::Edge> contracted_edges;
    contracted_edges.reserve(survivors.size());

    constexpr std::uint64_t no_key = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t last_key = no_key;
    std::int64_t current_contracted = -1;
    for (const auto &survivor : survivors) {
        if (survivor.key != last_key) {
            const auto ru = survivor.key >> 32;
            const auto rv = survivor.key & std::uint64_t{0xFFFFFFFF};
            current_contracted = static_cast<std::int64_t>(contracted_edges.size());
            contracted_edges.push_back(UndirectedGraph::Edge{ru, rv});
            last_key = survivor.key;
        }
        contracted_edge_of_original[static_cast<std::size_t>(survivor.original_edge)] =
            current_contracted;
    }

    auto contracted_graph = UndirectedGraph::from_sorted_unique_edges(
        number_of_components, std::move(contracted_edges), /*populate_lookup=*/false
    );

    return AgreementContraction{
        std::move(contracted_graph),
        std::move(contracted_edge_of_original),
        std::move(root_of_node),
    };
}

} // namespace bioimage_cpp::graph::detail
