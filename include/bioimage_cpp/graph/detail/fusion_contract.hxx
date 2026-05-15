#pragma once

#include "bioimage_cpp/detail/relabel.hxx"
#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
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

    std::vector<std::uint64_t> raw_root(static_cast<std::size_t>(number_of_nodes));
    for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
        raw_root[static_cast<std::size_t>(node)] = sets.find(node);
    }
    auto root_of_node = bioimage_cpp::detail::dense_relabel(raw_root);

    std::uint64_t number_of_components = 0;
    for (const auto root : root_of_node) {
        if (root + 1 > number_of_components) {
            number_of_components = root + 1;
        }
    }

    UndirectedGraph contracted_graph(number_of_components);
    std::vector<std::int64_t> contracted_edge_of_original(
        static_cast<std::size_t>(number_of_edges), -1
    );

    for (std::uint64_t edge = 0; edge < number_of_edges; ++edge) {
        const auto uv = graph.uv(edge);
        const auto ru = root_of_node[static_cast<std::size_t>(uv.first)];
        const auto rv = root_of_node[static_cast<std::size_t>(uv.second)];
        if (ru == rv) {
            continue;
        }
        const auto inserted = contracted_graph.insert_edge(ru, rv);
        contracted_edge_of_original[static_cast<std::size_t>(edge)] =
            static_cast<std::int64_t>(inserted);
    }

    return AgreementContraction{
        std::move(contracted_graph),
        std::move(contracted_edge_of_original),
        std::move(root_of_node),
    };
}

} // namespace bioimage_cpp::graph::detail
