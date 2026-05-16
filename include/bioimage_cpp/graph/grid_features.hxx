#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/edge_hash.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/graph/grid_graph.hxx"

#include <algorithm>
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

template <std::size_t D>
std::array<std::uint64_t, D> uint64_strides_from_shape(
    const std::array<std::uint64_t, D> &shape
) {
    std::array<std::uint64_t, D> strides{};
    strides[D - 1] = 1;
    for (std::size_t axis = D - 1; axis > 0; --axis) {
        strides[axis - 1] = strides[axis] * shape[axis];
    }
    return strides;
}

template <std::size_t D>
std::size_t local_offset_axis(const std::vector<std::ptrdiff_t> &offset) {
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (offset[axis] != 0) {
            return axis;
        }
    }
    return D;
}

template <std::size_t D, class Callback>
void for_each_valid_offset_link(
    const GridGraph<D> &graph,
    const std::vector<std::ptrdiff_t> &offset,
    Callback &&callback
) {
    const auto &shape = graph.shape();
    const auto &strides = graph.strides();
    std::array<std::uint64_t, D> begin{};
    std::array<std::uint64_t, D> end{};
    std::int64_t delta = 0;
    for (std::size_t axis = 0; axis < D; ++axis) {
        const auto axis_offset = offset[axis];
        if (axis_offset < 0) {
            begin[axis] = static_cast<std::uint64_t>(-axis_offset);
            end[axis] = shape[axis];
        } else {
            begin[axis] = 0;
            end[axis] = shape[axis] - static_cast<std::uint64_t>(axis_offset);
        }
        delta += static_cast<std::int64_t>(axis_offset) *
                 static_cast<std::int64_t>(strides[axis]);
    }

    std::array<std::uint64_t, D> coordinate{};
    auto recurse = [&](auto &&self, const std::size_t axis, const std::uint64_t node) -> void {
        if (axis == D) {
            const auto target =
                static_cast<std::uint64_t>(static_cast<std::int64_t>(node) + delta);
            callback(node, target, coordinate);
            return;
        }
        auto axis_node = node + begin[axis] * strides[axis];
        for (std::uint64_t coord = begin[axis]; coord < end[axis]; ++coord) {
            coordinate[axis] = coord;
            self(self, axis + 1, axis_node);
            axis_node += strides[axis];
        }
    };
    recurse(recurse, 0, 0);
}

template <std::size_t D>
std::uint64_t local_edge_id_from_source_coordinate(
    const GridGraph<D> &graph,
    const std::vector<std::ptrdiff_t> &offset,
    const std::array<std::uint64_t, D> &source_coordinate,
    const std::size_t axis,
    const std::array<std::uint64_t, D> &edge_strides
) {
    std::uint64_t local_edge = 0;
    for (std::size_t coord_axis = 0; coord_axis < D; ++coord_axis) {
        auto pivot = source_coordinate[coord_axis];
        if (coord_axis == axis && offset[axis] < 0) {
            --pivot;
        }
        local_edge += pivot * edge_strides[coord_axis];
    }
    return graph.edge_offset(axis) + local_edge;
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

    const auto number_of_nodes = graph.number_of_nodes();
    for (std::uint64_t edge = 0; edge < graph.number_of_edges(); ++edge) {
        weights.data[edge] = 0.0;
        valid_edges.data[edge] = 0;
    }

    for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
        const auto axis = detail_grid_features::local_offset_axis<D>(offsets[channel]);
        auto edge_shape = graph.shape();
        --edge_shape[axis];
        const auto edge_strides = detail_grid_features::uint64_strides_from_shape(edge_shape);
        const auto channel_offset = static_cast<std::uint64_t>(channel) * number_of_nodes;
        detail_grid_features::for_each_valid_offset_link<D>(
            graph,
            offsets[channel],
            [&](const std::uint64_t node, const std::uint64_t, const auto &coordinate) {
            const auto edge = detail_grid_features::local_edge_id_from_source_coordinate<D>(
                graph, offsets[channel], coordinate, axis, edge_strides
            );
            if (valid_edges.data[edge] != 0) {
                throw std::invalid_argument("offsets produce duplicate local grid edges");
            }
            weights.data[edge] = affinities.data[channel_offset + node];
            valid_edges.data[edge] = 1;
        });
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

    std::uint64_t lifted_capacity = 0;
    for (const auto &offset : offsets) {
        if (detail_grid_features::l1_norm(offset) > 1) {
            lifted_capacity += number_of_nodes;
        }
    }
    result.lifted_uvs.reserve(static_cast<std::size_t>(lifted_capacity));
    result.lifted_weights.reserve(static_cast<std::size_t>(lifted_capacity));
    result.lifted_offset_ids.reserve(static_cast<std::size_t>(lifted_capacity));

    for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
        const auto l1 = detail_grid_features::l1_norm(offsets[channel]);
        const auto channel_offset = static_cast<std::uint64_t>(channel) * number_of_nodes;
        if (l1 == 1) {
            const auto axis = detail_grid_features::local_offset_axis<D>(offsets[channel]);
            auto edge_shape = graph.shape();
            --edge_shape[axis];
            const auto edge_strides = detail_grid_features::uint64_strides_from_shape(edge_shape);
            detail_grid_features::for_each_valid_offset_link<D>(
                graph,
                offsets[channel],
                [&](const std::uint64_t node, const std::uint64_t, const auto &coordinate) {
                const auto edge = detail_grid_features::local_edge_id_from_source_coordinate<D>(
                    graph, offsets[channel], coordinate, axis, edge_strides
                );
                if (result.valid_local_edges[static_cast<std::size_t>(edge)] != 0) {
                    throw std::invalid_argument("offsets produce duplicate local grid edges");
                }
                result.local_weights[static_cast<std::size_t>(edge)] =
                    affinities.data[channel_offset + node];
                result.valid_local_edges[static_cast<std::size_t>(edge)] = 1;
            });
        } else {
            detail_grid_features::for_each_valid_offset_link<D>(
                graph,
                offsets[channel],
                [&](const std::uint64_t node, const std::uint64_t target, const auto &) {
                const auto uv = bioimage_cpp::detail::edge_key(node, target);
                result.lifted_uvs.push_back(uv);
                result.lifted_weights.push_back(affinities.data[channel_offset + node]);
                result.lifted_offset_ids.push_back(static_cast<std::uint64_t>(channel));
            });
        }
    }

    auto sorted_lifted_uvs = result.lifted_uvs;
    std::sort(sorted_lifted_uvs.begin(), sorted_lifted_uvs.end());
    if (std::adjacent_find(sorted_lifted_uvs.begin(), sorted_lifted_uvs.end()) !=
        sorted_lifted_uvs.end()) {
        throw std::invalid_argument("offsets produce duplicate long-range grid edges");
    }

    return result;
}

} // namespace bioimage_cpp::graph
