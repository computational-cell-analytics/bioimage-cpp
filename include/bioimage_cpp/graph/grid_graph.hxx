#pragma once

#include "bioimage_cpp/graph/undirected_graph.hxx"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::graph {

template <std::size_t D>
class GridGraph : public UndirectedGraph {
public:
    using UndirectedGraph::number_of_edges;

    using Coordinate = std::array<std::uint64_t, D>;

    explicit GridGraph(const Coordinate &shape)
        : UndirectedGraph(checked_node_count(shape), checked_edge_count(shape)),
          shape_(shape),
          strides_(compute_strides(shape)),
          edge_offsets_(compute_edge_offsets(shape)) {
        build_edges();
    }

    explicit GridGraph(const std::vector<std::uint64_t> &shape)
        : GridGraph(vector_to_coordinate(shape)) {
    }

    [[nodiscard]] const Coordinate &shape() const {
        return shape_;
    }

    [[nodiscard]] const Coordinate &strides() const {
        return strides_;
    }

    [[nodiscard]] std::size_t ndim() const {
        return D;
    }

    [[nodiscard]] std::uint64_t node_id(const Coordinate &coordinate) const {
        std::uint64_t node = 0;
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (coordinate[axis] >= shape_[axis]) {
                throw std::out_of_range(
                    "coordinate[" + std::to_string(axis) + "] must be < shape[" +
                    std::to_string(axis) + "]"
                );
            }
            node += coordinate[axis] * strides_[axis];
        }
        return node;
    }

    [[nodiscard]] Coordinate coordinates(std::uint64_t node) const {
        validate_node(node);
        Coordinate coordinate{};
        for (std::size_t axis = 0; axis < D; ++axis) {
            coordinate[axis] = node / strides_[axis];
            node -= coordinate[axis] * strides_[axis];
        }
        return coordinate;
    }

    [[nodiscard]] std::size_t edge_axis(const std::uint64_t edge) const {
        validate_edge(edge);
        const auto found = std::upper_bound(
            edge_offsets_.begin(),
            edge_offsets_.end(),
            edge
        );
        return static_cast<std::size_t>(found - edge_offsets_.begin() - 1);
    }

    [[nodiscard]] std::uint64_t edge_offset(const std::size_t axis) const {
        if (axis >= D) {
            throw std::out_of_range("axis must be < ndim");
        }
        return edge_offsets_[axis];
    }

    [[nodiscard]] std::uint64_t number_of_edges(const std::size_t axis) const {
        if (axis >= D) {
            throw std::out_of_range("axis must be < ndim");
        }
        return edge_offsets_[axis + 1] - edge_offsets_[axis];
    }

    [[nodiscard]] std::uint64_t edge_id(
        const std::size_t axis,
        const Coordinate &pivot_coordinate
    ) const {
        if (axis >= D) {
            throw std::out_of_range("axis must be < ndim");
        }
        const auto shape = edge_shape(axis);
        const auto strides = compute_strides(shape);
        std::uint64_t local_edge = 0;
        for (std::size_t coord_axis = 0; coord_axis < D; ++coord_axis) {
            if (pivot_coordinate[coord_axis] >= shape[coord_axis]) {
                throw std::out_of_range(
                    "pivot_coordinate[" + std::to_string(coord_axis) +
                    "] is outside the edge grid"
                );
            }
            local_edge += pivot_coordinate[coord_axis] * strides[coord_axis];
        }
        return edge_offsets_[axis] + local_edge;
    }

    [[nodiscard]] std::pair<Coordinate, std::size_t>
    edge_coordinates(const std::uint64_t edge) const {
        const auto axis = edge_axis(edge);
        const auto local_edge = edge - edge_offsets_[axis];
        return {edge_pivot_coordinates(axis, local_edge), axis};
    }

    [[nodiscard]] bool valid_offset_target(
        const std::uint64_t node,
        const std::array<std::int64_t, D> &offset,
        std::uint64_t &target_out
    ) const {
        const auto coordinate = coordinates(node);
        std::int64_t signed_delta = 0;
        for (std::size_t axis = 0; axis < D; ++axis) {
            const auto coord = static_cast<std::int64_t>(coordinate[axis]);
            const auto neighbor = coord + offset[axis];
            if (neighbor < 0 || neighbor >= static_cast<std::int64_t>(shape_[axis])) {
                return false;
            }
            signed_delta += offset[axis] * static_cast<std::int64_t>(strides_[axis]);
        }
        target_out = static_cast<std::uint64_t>(static_cast<std::int64_t>(node) + signed_delta);
        return true;
    }

private:
    static Coordinate vector_to_coordinate(const std::vector<std::uint64_t> &shape) {
        if (shape.size() != D) {
            throw std::invalid_argument(
                "shape must have length " + std::to_string(D) +
                ", got length=" + std::to_string(shape.size())
            );
        }
        Coordinate coordinate{};
        std::copy(shape.begin(), shape.end(), coordinate.begin());
        return coordinate;
    }

    static std::uint64_t checked_multiply(
        const std::uint64_t a,
        const std::uint64_t b,
        const char *name
    ) {
        if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
            throw std::overflow_error(std::string(name) + " exceeds uint64 range");
        }
        return a * b;
    }

    static std::uint64_t checked_node_count(const Coordinate &shape) {
        std::uint64_t count = 1;
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (shape[axis] == 0) {
                throw std::invalid_argument("shape dimensions must be greater than zero");
            }
            count = checked_multiply(count, shape[axis], "number_of_nodes");
        }
        return count;
    }

    static Coordinate compute_strides(const Coordinate &shape) {
        Coordinate strides{};
        strides[D - 1] = 1;
        for (std::size_t axis = D - 1; axis > 0; --axis) {
            strides[axis - 1] = checked_multiply(strides[axis], shape[axis], "stride");
        }
        return strides;
    }

    static std::uint64_t axis_edge_count(const Coordinate &shape, const std::size_t axis) {
        if (shape[axis] <= 1) {
            return 0;
        }
        std::uint64_t count = shape[axis] - 1;
        for (std::size_t other_axis = 0; other_axis < D; ++other_axis) {
            if (other_axis != axis) {
                count = checked_multiply(count, shape[other_axis], "number_of_edges");
            }
        }
        return count;
    }

    static std::uint64_t checked_edge_count(const Coordinate &shape) {
        std::uint64_t count = 0;
        for (std::size_t axis = 0; axis < D; ++axis) {
            const auto axis_count = axis_edge_count(shape, axis);
            if (axis_count > std::numeric_limits<std::uint64_t>::max() - count) {
                throw std::overflow_error("number_of_edges exceeds uint64 range");
            }
            count += axis_count;
        }
        return count;
    }

    static std::array<std::uint64_t, D + 1> compute_edge_offsets(const Coordinate &shape) {
        std::array<std::uint64_t, D + 1> offsets{};
        for (std::size_t axis = 0; axis < D; ++axis) {
            offsets[axis + 1] = offsets[axis] + axis_edge_count(shape, axis);
        }
        return offsets;
    }

    [[nodiscard]] Coordinate edge_shape(const std::size_t axis) const {
        auto result = shape_;
        --result[axis];
        return result;
    }

    [[nodiscard]] Coordinate edge_pivot_coordinates(
        const std::size_t axis,
        std::uint64_t local_edge
    ) const {
        const auto shape = edge_shape(axis);
        const auto strides = compute_strides(shape);
        Coordinate coordinate{};
        for (std::size_t coord_axis = 0; coord_axis < D; ++coord_axis) {
            coordinate[coord_axis] = local_edge / strides[coord_axis];
            local_edge -= coordinate[coord_axis] * strides[coord_axis];
        }
        return coordinate;
    }

    void build_edges() {
        for (std::size_t axis = 0; axis < D; ++axis) {
            const auto axis_edges = edge_offsets_[axis + 1] - edge_offsets_[axis];
            for (std::uint64_t local_edge = 0; local_edge < axis_edges; ++local_edge) {
                const auto coordinate = edge_pivot_coordinates(axis, local_edge);
                const auto u = node_id(coordinate);
                insert_new_edge(u, u + strides_[axis]);
            }
        }
    }

    Coordinate shape_{};
    Coordinate strides_{};
    std::array<std::uint64_t, D + 1> edge_offsets_{};
};

} // namespace bioimage_cpp::graph
