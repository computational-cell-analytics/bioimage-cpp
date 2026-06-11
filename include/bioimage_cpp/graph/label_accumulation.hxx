#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/graph/node_label_projection.hxx"
#include "bioimage_cpp/graph/region_adjacency_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::graph {

namespace detail_label_accumulation {

// Combined (node, other) histogram key. A single map per thread keyed by this
// pair avoids the n_threads * n_nodes map-of-maps (one std::unordered_map per
// node per thread), which dominates allocation for RAGs with many nodes.
template <class OtherT>
struct NodeOther {
    std::uint64_t node;
    OtherT other;
    bool operator==(const NodeOther &rhs) const noexcept {
        return node == rhs.node && other == rhs.other;
    }
};

template <class OtherT>
struct NodeOtherHash {
    std::size_t operator()(const NodeOther<OtherT> &key) const noexcept {
        std::uint64_t h = key.node * 0x9E3779B97F4A7C15ULL;
        h ^= static_cast<std::uint64_t>(key.other) + 0x9E3779B97F4A7C15ULL +
             (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

template <class OtherT>
using NodeOtherHistogram =
    std::unordered_map<NodeOther<OtherT>, std::uint64_t, NodeOtherHash<OtherT>>;

template <class LabelT, class OtherT>
void scan_chunk(
    const LabelT *labels,
    const OtherT *other_labels,
    const std::size_t pixel_begin,
    const std::size_t pixel_end,
    const bool has_ignore_value,
    const OtherT ignore_value,
    NodeOtherHistogram<OtherT> &histogram
) {
    for (std::size_t index = pixel_begin; index < pixel_end; ++index) {
        const auto other = other_labels[index];
        if (has_ignore_value && other == ignore_value) {
            continue;
        }
        const auto node = detail::checked_label_to_node(labels[index]);
        ++histogram[NodeOther<OtherT>{static_cast<std::uint64_t>(node), other}];
    }
}

} // namespace detail_label_accumulation

template <class LabelT, class OtherT>
void accumulate_labels(
    const RegionAdjacencyGraph &rag,
    const ConstArrayView<LabelT> &labels,
    const ConstArrayView<OtherT> &other_labels,
    const bool has_ignore_value,
    const OtherT ignore_value,
    const std::size_t number_of_threads,
    const ArrayView<OtherT> &out
) {
    if (labels.ndim() != 2 && labels.ndim() != 3) {
        throw std::invalid_argument(
            "labels must be a 2D or 3D array, got ndim=" +
            std::to_string(labels.ndim())
        );
    }
    detail_projection::require_rag_shape_matches_labels(rag, labels.shape);
    if (other_labels.shape != labels.shape) {
        throw std::invalid_argument("other_labels shape must match labels shape");
    }
    const auto number_of_nodes = static_cast<std::size_t>(rag.number_of_nodes());
    if (out.shape != std::vector<std::ptrdiff_t>{static_cast<std::ptrdiff_t>(number_of_nodes)}) {
        throw std::invalid_argument(
            "out shape must be (number_of_nodes,)"
        );
    }
    if (detail::max_label(labels) >= static_cast<std::uint64_t>(number_of_nodes)) {
        throw std::invalid_argument("labels contain a node id outside the rag");
    }

    const auto n_pixels = bioimage_cpp::detail::number_of_elements(labels.shape);
    const auto n_threads = detail::normalize_thread_count(number_of_threads, n_pixels);

    std::vector<detail_label_accumulation::NodeOtherHistogram<OtherT>> per_thread(n_threads);

    bioimage_cpp::detail::parallel_for_chunks(
        n_threads,
        n_pixels,
        [&](const std::size_t thread_id, const std::size_t begin, const std::size_t end) {
            detail_label_accumulation::scan_chunk(
                labels.data,
                other_labels.data,
                begin,
                end,
                has_ignore_value,
                ignore_value,
                per_thread[thread_id]
            );
        }
    );

    // Merge the per-thread histograms into one, then pick the majority `other`
    // label per node (smaller label wins ties; nodes with no pixels stay 0).
    // The argmax pass is single-threaded over distinct (node, other) pairs,
    // which is cheap relative to the parallel pixel scan above.
    auto &merged = per_thread[0];
    for (std::size_t thread_id = 1; thread_id < per_thread.size(); ++thread_id) {
        for (const auto &entry : per_thread[thread_id]) {
            merged[entry.first] += entry.second;
        }
        detail_label_accumulation::NodeOtherHistogram<OtherT>().swap(per_thread[thread_id]);
    }

    for (std::size_t node = 0; node < number_of_nodes; ++node) {
        out.data[node] = OtherT{0};
    }
    std::vector<std::uint64_t> best_count(number_of_nodes, 0);
    std::vector<unsigned char> has_best(number_of_nodes, 0);
    for (const auto &entry : merged) {
        const auto node = static_cast<std::size_t>(entry.first.node);
        const auto other = entry.first.other;
        const auto count = entry.second;
        if (has_best[node] == 0 ||
            count > best_count[node] ||
            (count == best_count[node] && other < out.data[node])) {
            out.data[node] = other;
            best_count[node] = count;
            has_best[node] = 1;
        }
    }
}

} // namespace bioimage_cpp::graph
