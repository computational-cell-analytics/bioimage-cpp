#pragma once

#include "bioimage_cpp/detail/union_find.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

namespace detail {

template <class WeightT>
struct EdgeWeightedWatershedScratch {
    std::vector<std::pair<WeightT, std::uint64_t>> sort_buffer;
};

template <class WeightT, class SeedT>
inline void edge_weighted_watershed_kernel(
    const UndirectedGraph &graph,
    const WeightT *edge_weights,
    const std::uint64_t number_of_edges,
    SeedT *labels,
    const std::uint64_t number_of_nodes,
    EdgeWeightedWatershedScratch<WeightT> &scratch
) {
    if (number_of_edges == 0) {
        return;
    }

    scratch.sort_buffer.resize(static_cast<std::size_t>(number_of_edges));
    for (std::uint64_t edge = 0; edge < number_of_edges; ++edge) {
        scratch.sort_buffer[static_cast<std::size_t>(edge)] = {
            edge_weights[static_cast<std::size_t>(edge)], edge
        };
    }
    // The pair's first element is the weight, so the default less-than gives
    // ascending sort order. `stable_sort` keeps the existing tie-breaking
    // semantics (matches the docstring's "ascending by weight" guarantee).
    std::stable_sort(
        scratch.sort_buffer.begin(),
        scratch.sort_buffer.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; }
    );

    bioimage_cpp::detail::UnionFind sets(static_cast<std::size_t>(number_of_nodes));

    constexpr SeedT zero{0};
    for (const auto &item : scratch.sort_buffer) {
        const auto edge = item.second;
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
}

} // namespace detail

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
    detail::EdgeWeightedWatershedScratch<WeightT> scratch;
    detail::edge_weighted_watershed_kernel<WeightT, SeedT>(
        graph,
        edge_weights.data(),
        number_of_edges,
        labels.data(),
        number_of_nodes,
        scratch
    );
    return labels;
}

// Buffer-reusing variant: caller provides scratch + output buffers. Validates
// nothing; intended for tight inner loops (e.g. proposal generators that call
// the watershed once per iteration on the same graph). The caller must:
//   - size `labels` to `graph.number_of_nodes()` and copy seeds into it before
//     the call (the algorithm propagates labels in place on this buffer);
//   - ensure `edge_weights` has length `graph.number_of_edges()`.
template <class WeightT, class SeedT>
inline void edge_weighted_watershed_into(
    const UndirectedGraph &graph,
    const std::vector<WeightT> &edge_weights,
    std::vector<SeedT> &labels,
    detail::EdgeWeightedWatershedScratch<WeightT> &scratch
) {
    detail::edge_weighted_watershed_kernel<WeightT, SeedT>(
        graph,
        edge_weights.data(),
        graph.number_of_edges(),
        labels.data(),
        graph.number_of_nodes(),
        scratch
    );
}

} // namespace bioimage_cpp::graph
