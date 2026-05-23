#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/graph/node_label_projection.hxx"
#include "bioimage_cpp/graph/region_adjacency_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace bioimage_cpp::graph {

namespace detail_label_accumulation {

inline std::size_t number_of_pixels(const std::vector<std::ptrdiff_t> &shape) {
    return static_cast<std::size_t>(std::accumulate(
        shape.begin(),
        shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    ));
}

template <class LabelT, class OtherT>
void scan_chunk(
    const LabelT *labels,
    const OtherT *other_labels,
    const std::size_t pixel_begin,
    const std::size_t pixel_end,
    const bool has_ignore_value,
    const OtherT ignore_value,
    std::vector<std::unordered_map<OtherT, std::uint64_t>> &histograms
) {
    for (std::size_t index = pixel_begin; index < pixel_end; ++index) {
        const auto other = other_labels[index];
        if (has_ignore_value && other == ignore_value) {
            continue;
        }
        const auto node = detail::checked_label_to_node(labels[index]);
        ++histograms[static_cast<std::size_t>(node)][other];
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

    const auto n_pixels = detail_label_accumulation::number_of_pixels(labels.shape);
    const auto n_threads = detail::normalize_thread_count(number_of_threads, n_pixels);

    std::vector<std::vector<std::unordered_map<OtherT, std::uint64_t>>> per_thread(
        n_threads,
        std::vector<std::unordered_map<OtherT, std::uint64_t>>(number_of_nodes)
    );

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

    // Merge per-thread histograms and pick majority per node in one pass over
    // nodes. This is embarrassingly parallel across nodes.
    const auto node_threads = detail::normalize_thread_count(number_of_threads, number_of_nodes);
    bioimage_cpp::detail::parallel_for_chunks(
        node_threads,
        number_of_nodes,
        [&](const std::size_t, const std::size_t node_begin, const std::size_t node_end) {
            for (std::size_t node = node_begin; node < node_end; ++node) {
                std::unordered_map<OtherT, std::uint64_t> merged;
                for (auto &thread_histograms : per_thread) {
                    auto &node_hist = thread_histograms[node];
                    for (const auto &entry : node_hist) {
                        merged[entry.first] += entry.second;
                    }
                    node_hist.clear();
                }
                OtherT best_label = OtherT{0};
                std::uint64_t best_count = 0;
                bool has_best = false;
                for (const auto &entry : merged) {
                    if (!has_best ||
                        entry.second > best_count ||
                        (entry.second == best_count && entry.first < best_label)) {
                        best_label = entry.first;
                        best_count = entry.second;
                        has_best = true;
                    }
                }
                out.data[node] = has_best ? best_label : OtherT{0};
            }
        }
    );
}

} // namespace bioimage_cpp::graph
