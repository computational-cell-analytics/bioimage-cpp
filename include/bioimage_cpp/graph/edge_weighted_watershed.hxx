#pragma once

#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace bioimage_cpp::graph {

// Kruskal-style edge-weighted seeded watershed.
//
// Edges are visited in ascending weight order. Two distinct components are
// merged iff at least one of them is unlabeled (seed label 0); the non-zero
// seed label then propagates. Two distinct already-labeled components are
// never merged, so seed boundaries are preserved. Nodes that no seed can
// reach retain label 0.
//
// `WeightT` is any type orderable with `<` (typically float32 or float64).
// `SeedT` is any integer label type. Signed types must not contain negative
// values; this is checked at the boundary.
template <class WeightT, class SeedT>
inline std::vector<SeedT> edge_weighted_watershed(
    const UndirectedGraph &graph,
    const std::vector<WeightT> &edge_weights,
    const std::vector<SeedT> &seeds
) {
    const auto number_of_nodes = graph.number_of_nodes();
    const auto number_of_edges = graph.number_of_edges();

    if (edge_weights.size() != static_cast<std::size_t>(number_of_edges)) {
        throw std::invalid_argument(
            "edge_weights length must equal number_of_edges, got " +
            std::to_string(edge_weights.size()) + " for number_of_edges=" +
            std::to_string(number_of_edges)
        );
    }
    if (seeds.size() != static_cast<std::size_t>(number_of_nodes)) {
        throw std::invalid_argument(
            "seeds length must equal number_of_nodes, got " +
            std::to_string(seeds.size()) + " for number_of_nodes=" +
            std::to_string(number_of_nodes)
        );
    }
    if constexpr (std::is_signed_v<SeedT>) {
        for (const auto value : seeds) {
            if (value < 0) {
                throw std::invalid_argument("seeds must not contain negative values");
            }
        }
    }

    std::vector<SeedT> labels(seeds);

    if (number_of_edges == 0) {
        return labels;
    }

    std::vector<std::uint64_t> order(static_cast<std::size_t>(number_of_edges));
    for (std::uint64_t edge = 0; edge < number_of_edges; ++edge) {
        order[static_cast<std::size_t>(edge)] = edge;
    }
    std::stable_sort(
        order.begin(),
        order.end(),
        [&edge_weights](const std::uint64_t a, const std::uint64_t b) {
            return edge_weights[static_cast<std::size_t>(a)] <
                   edge_weights[static_cast<std::size_t>(b)];
        }
    );

    detail::UnionFind sets(static_cast<std::size_t>(number_of_nodes));

    constexpr SeedT zero{0};
    for (const auto edge : order) {
        const auto uv = graph.uv(edge);
        const auto ru = sets.find(uv.first);
        const auto rv = sets.find(uv.second);
        if (ru == rv) {
            continue;
        }
        const auto lu = labels[static_cast<std::size_t>(ru)];
        const auto lv = labels[static_cast<std::size_t>(rv)];
        if (lu != zero && lv != zero) {
            continue;
        }
        const auto new_label = std::max(lu, lv);
        const auto new_root = sets.unite_roots(ru, rv);
        labels[static_cast<std::size_t>(new_root)] = new_label;
    }

    for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
        const auto root = sets.find(node);
        labels[static_cast<std::size_t>(node)] = labels[static_cast<std::size_t>(root)];
    }

    return labels;
}

} // namespace bioimage_cpp::graph
