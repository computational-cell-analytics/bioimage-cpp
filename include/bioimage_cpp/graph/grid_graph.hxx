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
        : UndirectedGraph(checked_node_count(shape), checked_edge_count(shape), 0),
          shape_(shape),
          strides_(compute_strides(shape)),
          edge_offsets_(compute_edge_offsets(shape)),
          edge_strides_(compute_edge_strides(shape)) {
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
        std::uint64_t local_edge = 0;
        for (std::size_t coord_axis = 0; coord_axis < D; ++coord_axis) {
            if (pivot_coordinate[coord_axis] >= shape[coord_axis]) {
                throw std::out_of_range(
                    "pivot_coordinate[" + std::to_string(coord_axis) +
                    "] is outside the edge grid"
                );
            }
            local_edge += pivot_coordinate[coord_axis] * edge_strides_[axis][coord_axis];
        }
        return edge_offsets_[axis] + local_edge;
    }

    EdgeId insert_edge(const NodeId u, const NodeId v) override {
        if (u == v) {
            throw std::invalid_argument("self edges are not supported");
        }
        const auto existing = find_edge(u, v);
        if (existing >= 0) {
            return static_cast<EdgeId>(existing);
        }
        // `find_edge` already validated the node ids and missed both the grid
        // analytical check and the parent hash map, so we can insert directly
        // without paying for another hash lookup inside `UndirectedGraph::insert_edge`.
        const auto key = detail::edge_key(u, v);
        return UndirectedGraph::insert_new_edge(key.first, key.second);
    }

    [[nodiscard]] std::int64_t find_edge(const NodeId u, const NodeId v) const override {
        validate_node(u);
        validate_node(v);
        if (u == v) {
            return -1;
        }
        const auto lower = std::min(u, v);
        const auto upper = std::max(u, v);
        const auto diff = upper - lower;
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (diff != strides_[axis]) {
                continue;
            }
            const auto axis_coordinate = (lower / strides_[axis]) % shape_[axis];
            if (axis_coordinate + 1 >= shape_[axis]) {
                return -1;
            }
            std::uint64_t local_edge = 0;
            for (std::size_t coord_axis = 0; coord_axis < D; ++coord_axis) {
                const auto coordinate = (lower / strides_[coord_axis]) % shape_[coord_axis];
                local_edge += coordinate * edge_strides_[axis][coord_axis];
            }
            return static_cast<std::int64_t>(edge_offsets_[axis] + local_edge);
        }
        return UndirectedGraph::find_edge(u, v);
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

    static std::array<Coordinate, D> compute_edge_strides(const Coordinate &shape) {
        std::array<Coordinate, D> strides{};
        for (std::size_t axis = 0; axis < D; ++axis) {
            auto edge_shape = shape;
            --edge_shape[axis];
            strides[axis] = compute_strides(edge_shape);
        }
        return strides;
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
        Coordinate coordinate{};
        for (std::size_t coord_axis = 0; coord_axis < D; ++coord_axis) {
            coordinate[coord_axis] = local_edge / edge_strides_[axis][coord_axis];
            local_edge -= coordinate[coord_axis] * edge_strides_[axis][coord_axis];
        }
        return coordinate;
    }

    // Walk a sub-shape of the node grid in C-order and invoke `callback(flat)`
    // for every node id in it. Flat node ids are computed incrementally by
    // addition only — no per-step divisions, no per-step bounds checks. The
    // template recursion lets the compiler unroll the entire nest for D = 2/3.
    template <std::size_t Axis, class Callback>
    static void enumerate_subshape_in_c_order(
        const Coordinate &subshape,
        const Coordinate &node_strides,
        std::uint64_t base_flat,
        Callback &&callback
    ) {
        if constexpr (Axis == D) {
            callback(base_flat);
        } else {
            std::uint64_t flat = base_flat;
            for (std::uint64_t i = 0; i < subshape[Axis]; ++i) {
                enumerate_subshape_in_c_order<Axis + 1>(
                    subshape, node_strides, flat, callback
                );
                flat += node_strides[Axis];
            }
        }
    }

    void build_edges() {
        // Emit every edge into `edges_` in canonical order (axis-major,
        // C-order within each axis). Sequential `emplace_back` into a
        // single pre-reserved vector is cache-friendly. `edges_` was
        // already reserved by the `UndirectedGraph` ctor.
        //
        // We deliberately do NOT call `rebuild_adjacency_from_edges()`
        // here. The dominant grid-graph workflow (build graph → compute
        // edge features via `uv_ids()`) never touches adjacency, so the
        // ~400 ms rebuild on a 12 M-edge 3D grid is pure waste in that
        // case. Callers that need adjacency (BFS, connected components,
        // `extract_subgraph_from_nodes`) trigger a lazy rebuild via the
        // base-class `node_adjacency` path on first use, paying the cost
        // exactly once. Multi-threaded readers should call `freeze()` on
        // the construction thread before fan-out — the same convention
        // any insert-built `UndirectedGraph` already follows.
        auto &edges = access_edges();
        for (std::size_t axis = 0; axis < D; ++axis) {
            auto pivot_shape = shape_;
            --pivot_shape[axis];
            const auto axis_step = strides_[axis];
            enumerate_subshape_in_c_order<0>(
                pivot_shape, strides_, 0,
                [&edges, axis_step](const std::uint64_t u) {
                    edges.emplace_back(u, u + axis_step);
                }
            );
        }
    }

    Coordinate shape_{};
    Coordinate strides_{};
    std::array<std::uint64_t, D + 1> edge_offsets_{};
    std::array<Coordinate, D> edge_strides_{};
};

} // namespace bioimage_cpp::graph
