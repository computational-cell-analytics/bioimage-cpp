#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/mutex_storage.hxx"
#include "bioimage_cpp/detail/semantic_labels.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp {

// Semantic mutex watershed on a 2D or 3D image-derived grid graph.
//
// `affinities` carries three groups of channels stacked along axis 0:
//   - [0, number_of_attractive_channels) : attractive grid edges
//   - [number_of_attractive_channels, number_of_offsets) : mutex grid edges
//   - [number_of_offsets, channels) : semantic-class affinities (one per class)
// The grid offsets in `offsets` apply to the first `number_of_offsets`
// channels; semantic channels are not spatial edges and so have no offset.
//
// `valid_edges` shares the affinity shape and gates every edge (including
// semantic ones) on/off; the Python wrapper computes this mask. Output
// `node_labels_out` is 1-based dense node labels with shape `affinities.shape[1:]`
// (matching `mutex_watershed_grid`). Output `semantic_labels_out` has the same
// spatial shape and an ``int64`` value per node, ``-1`` for clusters that
// received no semantic assignment.
//
// Ported from `compute_semantic_mws_segmentation` in the affogato library.
template <class T>
void semantic_mutex_watershed_grid(
    const ConstArrayView<T> &affinities,
    const ConstArrayView<std::uint8_t> &valid_edges,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const std::size_t number_of_attractive_channels,
    const std::size_t number_of_offsets,
    const ArrayView<std::uint64_t> &node_labels_out,
    const ArrayView<std::int64_t> &semantic_labels_out
) {
    if (affinities.ndim() != 3 && affinities.ndim() != 4) {
        throw std::invalid_argument(
            "affinities must have shape (channels, y, x) or (channels, z, y, x), got ndim=" +
            std::to_string(affinities.ndim())
        );
    }
    if (offsets.empty()) {
        throw std::invalid_argument("offsets must not be empty");
    }
    if (valid_edges.shape != affinities.shape) {
        throw std::invalid_argument("valid_edges shape must match affinities shape");
    }

    const auto number_of_channels = static_cast<std::size_t>(affinities.shape[0]);
    const auto spatial_ndim = static_cast<std::size_t>(affinities.ndim() - 1);
    if (offsets.size() != number_of_offsets) {
        throw std::invalid_argument(
            "offsets length must equal number_of_offsets, got offsets length=" +
            std::to_string(offsets.size()) +
            ", number_of_offsets=" + std::to_string(number_of_offsets)
        );
    }
    if (number_of_attractive_channels > number_of_offsets) {
        throw std::invalid_argument(
            "number_of_attractive_channels must be <= number_of_offsets"
        );
    }
    if (number_of_offsets > number_of_channels) {
        throw std::invalid_argument(
            "number_of_offsets must be <= number of affinity channels"
        );
    }
    for (const auto &offset : offsets) {
        if (offset.size() != spatial_ndim) {
            throw std::invalid_argument(
                "each offset must have length matching the spatial ndim, got spatial ndim=" +
                std::to_string(spatial_ndim)
            );
        }
    }

    std::vector<std::ptrdiff_t> spatial_shape(
        affinities.shape.begin() + 1,
        affinities.shape.end()
    );
    if (node_labels_out.shape != spatial_shape) {
        throw std::invalid_argument("node_labels_out shape must match affinities spatial shape");
    }
    if (semantic_labels_out.shape != spatial_shape) {
        throw std::invalid_argument(
            "semantic_labels_out shape must match affinities spatial shape"
        );
    }

    const auto number_of_nodes = static_cast<std::uint64_t>(std::accumulate(
        spatial_shape.begin(),
        spatial_shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    ));
    const auto spatial_strides = detail::c_order_strides(spatial_shape);

    std::vector<std::ptrdiff_t> offset_strides(number_of_offsets, 0);
    for (std::size_t channel = 0; channel < number_of_offsets; ++channel) {
        for (std::size_t axis = 0; axis < spatial_ndim; ++axis) {
            offset_strides[channel] += offsets[channel][axis] * spatial_strides[axis];
        }
    }

    struct WeightedGridEdge {
        T weight;
        std::uint64_t id;
    };

    const auto number_of_edges = number_of_nodes * number_of_channels;
    std::vector<WeightedGridEdge> edge_order;
    edge_order.reserve(static_cast<std::size_t>(number_of_edges));
    for (std::uint64_t edge_id = 0; edge_id < number_of_edges; ++edge_id) {
        if (valid_edges.data[edge_id] != 0) {
            edge_order.push_back(WeightedGridEdge{affinities.data[edge_id], edge_id});
        }
    }
    std::sort(edge_order.begin(), edge_order.end(), [](const auto &first, const auto &second) {
        if (first.weight == second.weight) {
            return first.id < second.id;
        }
        return first.weight > second.weight;
    });

    bioimage_cpp::util::UnionFind sets(static_cast<std::size_t>(number_of_nodes));
    MutexStorage mutexes(static_cast<std::size_t>(number_of_nodes));
    SemanticLabeling semantic_labels(static_cast<std::size_t>(number_of_nodes), -1);

    const auto number_of_attractive_edges = number_of_nodes * static_cast<std::uint64_t>(
        number_of_attractive_channels
    );
    const auto number_of_spatial_edges = number_of_nodes * static_cast<std::uint64_t>(
        number_of_offsets
    );

    for (const auto &edge : edge_order) {
        const auto edge_id = edge.id;
        const auto channel = static_cast<std::size_t>(edge_id / number_of_nodes);
        const auto u = edge_id % number_of_nodes;

        if (edge_id >= number_of_spatial_edges) {
            const auto class_id = static_cast<std::int64_t>(channel - number_of_offsets);
            const auto root_u = sets.find(u);
            assign_semantic_label(root_u, class_id, semantic_labels);
            continue;
        }

        const auto v_signed = static_cast<std::int64_t>(u) +
            static_cast<std::int64_t>(offset_strides[channel]);
        const auto v = static_cast<std::uint64_t>(v_signed);
        const auto root_u = sets.find(u);
        const auto root_v = sets.find(v);
        if (root_u == root_v) {
            continue;
        }
        if (check_semantic_constraint(root_u, root_v, semantic_labels)) {
            continue;
        }
        if (check_mutex(root_u, root_v, mutexes)) {
            continue;
        }

        const bool is_mutex_edge = edge_id >= number_of_attractive_edges;
        if (is_mutex_edge) {
            insert_mutex(root_u, root_v, mutexes);
        } else {
            const auto new_root = sets.unite_roots(root_u, root_v);
            const auto old_root = (new_root == root_u) ? root_v : root_u;
            merge_mutexes(old_root, new_root, mutexes);
            merge_semantic_labels(new_root, old_root, semantic_labels);
        }
    }

    std::vector<std::uint64_t> root_labels(static_cast<std::size_t>(number_of_nodes), 0);
    std::uint64_t next_label = 1;
    for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
        const auto root = sets.find(node);
        auto &label = root_labels[static_cast<std::size_t>(root)];
        if (label == 0) {
            label = next_label;
            ++next_label;
        }
        node_labels_out.data[node] = label;
        semantic_labels_out.data[node] = semantic_labels[root];
    }
}

} // namespace bioimage_cpp
