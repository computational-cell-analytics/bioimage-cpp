#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/graph/grid_graph.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

namespace detail_grid_features {

template <std::size_t D>
std::vector<std::ptrdiff_t> graph_shape_vector(const GridGraph<D> &graph) {
    std::vector<std::ptrdiff_t> shape(D);
    const auto &grid_shape = graph.shape();
    for (std::size_t axis = 0; axis < D; ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(grid_shape[axis]);
    }
    return shape;
}

inline std::ptrdiff_t l1_norm(const std::vector<std::ptrdiff_t> &offset) {
    std::ptrdiff_t result = 0;
    for (const auto value : offset) {
        result += value < 0 ? -value : value;
    }
    return result;
}

template <std::size_t D>
void require_graph_shape(
    const GridGraph<D> &graph,
    const std::vector<std::ptrdiff_t> &shape,
    const char *argument_name
) {
    if (shape.size() != D) {
        throw std::invalid_argument(
            std::string(argument_name) + " ndim must match graph ndim"
        );
    }
    const auto &graph_shape = graph.shape();
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (shape[axis] != static_cast<std::ptrdiff_t>(graph_shape[axis])) {
            throw std::invalid_argument(
                std::string(argument_name) + " shape must match graph shape"
            );
        }
    }
}

template <std::size_t D>
void require_affinity_shape(
    const GridGraph<D> &graph,
    const ConstArrayView<double> &affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets
) {
    if (affinities.ndim() != static_cast<std::ptrdiff_t>(D + 1)) {
        throw std::invalid_argument("affinities must have shape (channels, *graph.shape)");
    }
    if (static_cast<std::size_t>(affinities.shape[0]) != offsets.size()) {
        throw std::invalid_argument("offsets length must match affinities channel count");
    }
    const auto &graph_shape = graph.shape();
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (affinities.shape[axis + 1] != static_cast<std::ptrdiff_t>(graph_shape[axis])) {
            throw std::invalid_argument("affinities spatial shape must match graph shape");
        }
    }
    for (const auto &offset : offsets) {
        if (offset.size() != D) {
            throw std::invalid_argument("each offset must have length matching graph ndim");
        }
        if (l1_norm(offset) == 0) {
            throw std::invalid_argument("offsets must not contain the zero offset");
        }
    }
}

template <std::size_t D>
std::uint64_t local_edge_id(
    const GridGraph<D> &graph,
    const std::uint64_t node,
    const std::vector<std::ptrdiff_t> &offset
) {
    std::size_t axis = D;
    std::ptrdiff_t step = 0;
    for (std::size_t candidate = 0; candidate < D; ++candidate) {
        if (offset[candidate] != 0) {
            axis = candidate;
            step = offset[candidate];
            break;
        }
    }
    if (axis == D || (step != 1 && step != -1) || l1_norm(offset) != 1) {
        throw std::invalid_argument("local offsets must have L1 norm 1");
    }

    auto pivot = graph.coordinates(node);
    if (step < 0) {
        --pivot[axis];
    }
    return graph.edge_id(axis, pivot);
}

} // namespace detail_grid_features

template <std::size_t D>
void grid_boundary_features(
    const GridGraph<D> &graph,
    const ConstArrayView<double> &boundary_map,
    const ArrayView<double> &out
) {
    detail_grid_features::require_graph_shape(graph, boundary_map.shape, "boundary_map");
    if (out.shape != std::vector<std::ptrdiff_t>{
            static_cast<std::ptrdiff_t>(graph.number_of_edges())}) {
        throw std::invalid_argument("out shape must be (number_of_edges,)");
    }

    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        const auto uv = graph.uv(edge);
        out.data[edge] = 0.5 * (boundary_map.data[uv.first] + boundary_map.data[uv.second]);
    }
}

template <std::size_t D>
void grid_local_affinity_features(
    const GridGraph<D> &graph,
    const ConstArrayView<double> &affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets,
    const ArrayView<double> &weights,
    const ArrayView<std::uint8_t> &valid_edges
) {
    detail_grid_features::require_affinity_shape(graph, affinities, offsets);
    if (weights.shape != std::vector<std::ptrdiff_t>{
            static_cast<std::ptrdiff_t>(graph.number_of_edges())}) {
        throw std::invalid_argument("weights shape must be (number_of_edges,)");
    }
    if (valid_edges.shape != weights.shape) {
        throw std::invalid_argument("valid_edges shape must match weights shape");
    }
    for (const auto &offset : offsets) {
        if (detail_grid_features::l1_norm(offset) != 1) {
            throw std::invalid_argument("grid_affinity_features accepts only local offsets");
        }
    }

    const auto spatial_shape = detail_grid_features::graph_shape_vector(graph);
    const auto spatial_strides = bioimage_cpp::detail::c_order_strides(spatial_shape);
    const auto number_of_nodes = graph.number_of_nodes();
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        weights.data[edge] = 0.0;
        valid_edges.data[edge] = 0;
    }

    for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
        const auto channel_offset = static_cast<std::uint64_t>(channel) * number_of_nodes;
        for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
            std::uint64_t target = 0;
            if (!bioimage_cpp::detail::valid_offset_target(
                    node, offsets[channel], spatial_shape, spatial_strides, target)) {
                continue;
            }
            const auto edge = detail_grid_features::local_edge_id(graph, node, offsets[channel]);
            if (valid_edges.data[edge] != 0) {
                throw std::invalid_argument("offsets produce duplicate local grid edges");
            }
            weights.data[edge] = affinities.data[channel_offset + node];
            valid_edges.data[edge] = 1;
        }
    }
}

struct GridLiftedAffinityFeatures {
    std::vector<double> local_weights;
    std::vector<std::uint8_t> valid_local_edges;
    std::vector<bioimage_cpp::detail::Edge> lifted_uvs;
    std::vector<double> lifted_weights;
    std::vector<std::uint64_t> lifted_offset_ids;
};

template <std::size_t D>
GridLiftedAffinityFeatures grid_affinity_features_with_lifted(
    const GridGraph<D> &graph,
    const ConstArrayView<double> &affinities,
    const std::vector<std::vector<std::ptrdiff_t>> &offsets
) {
    detail_grid_features::require_affinity_shape(graph, affinities, offsets);

    const auto number_of_nodes = graph.number_of_nodes();
    GridLiftedAffinityFeatures result;
    result.local_weights.assign(static_cast<std::size_t>(graph.number_of_edges()), 0.0);
    result.valid_local_edges.assign(static_cast<std::size_t>(graph.number_of_edges()), 0);

    const auto spatial_shape = detail_grid_features::graph_shape_vector(graph);
    const auto spatial_strides = bioimage_cpp::detail::c_order_strides(spatial_shape);
    std::unordered_set<bioimage_cpp::detail::Edge, bioimage_cpp::detail::EdgeHash> lifted_seen;
    std::uint64_t lifted_capacity = 0;
    for (const auto &offset : offsets) {
        if (detail_grid_features::l1_norm(offset) > 1) {
            lifted_capacity += number_of_nodes;
        }
    }
    result.lifted_uvs.reserve(static_cast<std::size_t>(lifted_capacity));
    result.lifted_weights.reserve(static_cast<std::size_t>(lifted_capacity));
    result.lifted_offset_ids.reserve(static_cast<std::size_t>(lifted_capacity));
    lifted_seen.reserve(static_cast<std::size_t>(lifted_capacity));

    for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
        const auto l1 = detail_grid_features::l1_norm(offsets[channel]);
        const auto channel_offset = static_cast<std::uint64_t>(channel) * number_of_nodes;
        for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
            std::uint64_t target = 0;
            if (!bioimage_cpp::detail::valid_offset_target(
                    node, offsets[channel], spatial_shape, spatial_strides, target)) {
                continue;
            }
            if (l1 == 1) {
                const auto edge = detail_grid_features::local_edge_id(graph, node, offsets[channel]);
                if (result.valid_local_edges[static_cast<std::size_t>(edge)] != 0) {
                    throw std::invalid_argument("offsets produce duplicate local grid edges");
                }
                result.local_weights[static_cast<std::size_t>(edge)] =
                    affinities.data[channel_offset + node];
                result.valid_local_edges[static_cast<std::size_t>(edge)] = 1;
            } else {
                const auto uv = bioimage_cpp::detail::edge_key(node, target);
                if (!lifted_seen.insert(uv).second) {
                    throw std::invalid_argument("offsets produce duplicate long-range grid edges");
                }
                result.lifted_uvs.push_back(uv);
                result.lifted_weights.push_back(affinities.data[channel_offset + node]);
                result.lifted_offset_ids.push_back(static_cast<std::uint64_t>(channel));
            }
        }
    }

    return result;
}

} // namespace bioimage_cpp::graph
