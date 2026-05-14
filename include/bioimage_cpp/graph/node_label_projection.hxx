#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/graph/region_adjacency_graph.hxx"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::graph {

namespace detail_projection {

inline std::size_t number_of_pixels(const std::vector<std::ptrdiff_t> &shape) {
    return static_cast<std::size_t>(std::accumulate(
        shape.begin(),
        shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    ));
}

inline void require_rag_shape_matches_labels(
    const RegionAdjacencyGraph &rag,
    const std::vector<std::ptrdiff_t> &labels_shape
) {
    const auto &rag_shape = rag.shape();
    if (rag_shape.size() != labels_shape.size()) {
        throw std::invalid_argument("rag shape must match labels shape");
    }
    for (std::size_t axis = 0; axis < rag_shape.size(); ++axis) {
        if (rag_shape[axis] != static_cast<std::uint64_t>(labels_shape[axis])) {
            throw std::invalid_argument("rag shape must match labels shape");
        }
    }
}

} // namespace detail_projection

template <class LabelT>
void project_node_labels_to_pixels(
    const RegionAdjacencyGraph &rag,
    const ConstArrayView<LabelT> &labels,
    const ConstArrayView<std::uint64_t> &node_labels,
    const std::size_t number_of_threads,
    const ArrayView<std::uint64_t> &out
) {
    if (labels.ndim() != 2 && labels.ndim() != 3) {
        throw std::invalid_argument(
            "labels must be a 2D or 3D array, got ndim=" +
            std::to_string(labels.ndim())
        );
    }
    detail_projection::require_rag_shape_matches_labels(rag, labels.shape);
    if (node_labels.ndim() != 1) {
        throw std::invalid_argument("node_labels must be a 1D uint64 array");
    }
    if (node_labels.shape[0] != static_cast<std::ptrdiff_t>(rag.number_of_nodes())) {
        throw std::invalid_argument("node_labels length must match rag number_of_nodes");
    }
    if (out.shape != labels.shape) {
        throw std::invalid_argument("out shape must match labels shape");
    }
    if (detail::max_label(labels) >= static_cast<std::uint64_t>(node_labels.shape[0])) {
        throw std::invalid_argument("labels contain a node id outside node_labels");
    }

    const auto n_pixels = detail_projection::number_of_pixels(labels.shape);
    const auto n_threads = detail::normalize_thread_count(number_of_threads, n_pixels);
    bioimage_cpp::detail::parallel_for_chunks(
        n_threads,
        n_pixels,
        [&](const std::size_t, const std::size_t begin, const std::size_t end) {
            for (std::size_t index = begin; index < end; ++index) {
                const auto node = static_cast<std::uint64_t>(labels.data[index]);
                out.data[index] = node_labels.data[static_cast<std::size_t>(node)];
            }
        }
    );
}

} // namespace bioimage_cpp::graph
