#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// Whole-volume merge primitives for the distributed region-adjacency-graph and
// edge-feature pipeline. These take the per-block artifacts produced by
// `block_extraction.hxx` (which the Python orchestration layer serializes and
// reads back) and combine them. Orchestration — deciding which blocks to merge,
// hierarchical scheduling, and I/O — lives in Python and is not implemented
// here.
namespace bioimage_cpp::graph::distributed {

// Merge many blocks' edge arrays into the whole-volume edge set. `concatenated`
// is an `(N, 2)` array of `(u, v)` pairs (typically all blocks' edges stacked).
// Edges are canonicalized to `u < v`, self-edges (`u == v`) are dropped, and the
// result is deduplicated and sorted ascending by `(u, v)` — exactly the
// precondition of `UndirectedGraph::from_sorted_unique_edges`, so the merged
// output can be turned into the global graph directly.
inline std::vector<bioimage_cpp::detail::Edge> merge_edges(
    const ConstArrayView<std::uint64_t> &concatenated
) {
    if (concatenated.ndim() != 2 || concatenated.shape[1] != 2) {
        throw std::invalid_argument("edges must have shape (n_edges, 2)");
    }

    const auto number_of_input_edges = static_cast<std::size_t>(concatenated.shape[0]);
    std::unordered_set<bioimage_cpp::detail::Edge, bioimage_cpp::detail::EdgeHash> merged;
    merged.reserve(number_of_input_edges);
    for (std::size_t index = 0; index < number_of_input_edges; ++index) {
        const auto u = concatenated.data[2 * index];
        const auto v = concatenated.data[2 * index + 1];
        if (u == v) {
            continue;
        }
        merged.insert(bioimage_cpp::detail::edge_key(u, v));
    }

    std::vector<bioimage_cpp::detail::Edge> sorted_edges(merged.begin(), merged.end());
    std::sort(sorted_edges.begin(), sorted_edges.end());
    return sorted_edges;
}

// Fold one block's partial edge statistics into a running whole-volume
// accumulator, returning the updated accumulator. `current_stats` and the return
// value are `(number_of_edges, 5)` rows aligned to `global_graph`'s edge ids;
// `block_edges`/`block_stats` are the `(n, 2)` / `(n, 5)` outputs of a block
// extraction. Each block edge is mapped to its global edge id via
// `global_graph.find_edge`; edges absent from the global graph (id `-1`) are
// skipped. `count/sum/sum_of_squares` add; `min/max` reduce, seeded from the
// first contribution to each edge (so `current_stats` may be zero-initialized).
//
// Block edge endpoints must be valid node ids of `global_graph`
// (`< number_of_nodes`); `find_edge` throws `std::out_of_range` otherwise.
inline std::vector<double> merge_block_edge_stats(
    const UndirectedGraph &global_graph,
    const ConstArrayView<double> &current_stats,
    const ConstArrayView<std::uint64_t> &block_edges,
    const ConstArrayView<double> &block_stats
) {
    if (current_stats.ndim() != 2 || current_stats.shape[1] != 5) {
        throw std::invalid_argument("current_stats must have shape (number_of_edges, 5)");
    }
    if (block_edges.ndim() != 2 || block_edges.shape[1] != 2) {
        throw std::invalid_argument("block_edges must have shape (n_edges, 2)");
    }
    if (block_stats.ndim() != 2 || block_stats.shape[1] != 5) {
        throw std::invalid_argument("block_stats must have shape (n_edges, 5)");
    }
    if (block_stats.shape[0] != block_edges.shape[0]) {
        throw std::invalid_argument(
            "block_edges and block_stats must have the same number of rows"
        );
    }
    if (static_cast<std::uint64_t>(current_stats.shape[0]) != global_graph.number_of_edges()) {
        throw std::invalid_argument(
            "current_stats rows must match global_graph number_of_edges"
        );
    }

    const auto number_of_edges = static_cast<std::size_t>(current_stats.shape[0]);
    std::vector<double> out(
        current_stats.data, current_stats.data + number_of_edges * 5
    );

    const auto number_of_block_edges = static_cast<std::size_t>(block_edges.shape[0]);
    for (std::size_t index = 0; index < number_of_block_edges; ++index) {
        const auto u = block_edges.data[2 * index];
        const auto v = block_edges.data[2 * index + 1];
        const auto edge = global_graph.find_edge(u, v);
        if (edge < 0) {
            continue;
        }

        double *const o = out.data() + static_cast<std::size_t>(edge) * 5;
        const double *const b = block_stats.data + index * 5;
        if (o[0] == 0.0) {
            o[3] = b[3];
            o[4] = b[4];
        } else {
            o[3] = std::min(o[3], b[3]);
            o[4] = std::max(o[4], b[4]);
        }
        o[0] += b[0];
        o[1] += b[1];
        o[2] += b[2];
    }
    return out;
}

// Turn accumulated partial statistics `(number_of_edges, 5)` into edge features.
// `compute_complex_features` selects the output width:
//   false -> `(number_of_edges, 2)`: [mean, size]
//   true  -> `(number_of_edges, 5)`: [mean, std, min, max, size]
// Edges with zero count produce all-zero rows. These columns equal the in-core
// `accumulate_*_features` results; the complex output is the moment subset of
// the 12-column in-core complex features (median and percentiles are not
// recoverable from block partials).
inline void finalize_edge_features(
    const ConstArrayView<double> &stats,
    const bool compute_complex_features,
    const ArrayView<double> &out
) {
    if (stats.ndim() != 2 || stats.shape[1] != 5) {
        throw std::invalid_argument("stats must have shape (number_of_edges, 5)");
    }
    const auto number_of_edges = static_cast<std::size_t>(stats.shape[0]);
    const auto number_of_features = compute_complex_features ? 5 : 2;
    if (out.ndim() != 2 ||
        out.shape[0] != stats.shape[0] ||
        out.shape[1] != number_of_features) {
        throw std::invalid_argument("out shape must be (number_of_edges, number_of_features)");
    }

    for (std::size_t edge = 0; edge < number_of_edges; ++edge) {
        const double *const s = stats.data + edge * 5;
        double *const o = out.data + edge * static_cast<std::size_t>(number_of_features);
        const auto count = s[0];
        if (count == 0.0) {
            for (int feature = 0; feature < number_of_features; ++feature) {
                o[feature] = 0.0;
            }
            continue;
        }

        const auto mean = s[1] / count;
        if (compute_complex_features) {
            const auto variance = std::max(0.0, s[2] / count - mean * mean);
            o[0] = mean;
            o[1] = std::sqrt(variance);
            o[2] = s[3];
            o[3] = s[4];
            o[4] = count;
        } else {
            o[0] = mean;
            o[1] = count;
        }
    }
}

} // namespace bioimage_cpp::graph::distributed
