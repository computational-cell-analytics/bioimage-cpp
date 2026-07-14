#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/indexed_heap.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::distance {

// Cost assigned to a directed grid edge u -> v.
enum class DijkstraCostMode {
    Physical,             // Euclidean length of the neighbour offset.
    Node,                 // costs[v].
    NodeTimesPhysical,    // costs[v] * physical edge length.
};

struct DijkstraOptions {
    int connectivity = 0;  // 0 means full connectivity (ndim).
    // Empty means unit spacing. Must be empty for Node mode, where physical
    // step lengths do not contribute to the edge cost.
    std::vector<double> spacing;
    DijkstraCostMode cost_mode = DijkstraCostMode::Physical;
};

struct DijkstraResult {
    std::vector<double> distances;
    // Flat C-order predecessor indices. Empty when predecessors were not
    // requested. Sources point to themselves; unreachable/background is -1.
    std::vector<std::int64_t> predecessors;
};

namespace detail_grid_dijkstra {

inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct Neighbor {
    std::vector<std::ptrdiff_t> offset;
    double physical_length = 0.0;
};

inline void build_neighbors_recursive(
    const std::size_t axis,
    const int connectivity,
    const std::vector<double> &spacing,
    std::vector<std::ptrdiff_t> &offset,
    int nonzero_axes,
    std::vector<Neighbor> &neighbors
) {
    if (axis == spacing.size()) {
        if (nonzero_axes == 0 || nonzero_axes > connectivity) {
            return;
        }
        double squared_length = 0.0;
        for (std::size_t d = 0; d < spacing.size(); ++d) {
            const double step = static_cast<double>(offset[d]) * spacing[d];
            squared_length += step * step;
        }
        neighbors.push_back({offset, std::sqrt(squared_length)});
        return;
    }

    for (std::ptrdiff_t step = -1; step <= 1; ++step) {
        offset[axis] = step;
        build_neighbors_recursive(
            axis + 1,
            connectivity,
            spacing,
            offset,
            nonzero_axes + (step != 0 ? 1 : 0),
            neighbors
        );
    }
}

inline std::vector<Neighbor> build_neighbors(
    const std::size_t ndim,
    const int connectivity,
    const std::vector<double> &spacing
) {
    std::vector<Neighbor> neighbors;
    std::vector<std::ptrdiff_t> offset(ndim, 0);
    build_neighbors_recursive(0, connectivity, spacing, offset, 0, neighbors);
    return neighbors;
}

using HeapPriority = std::pair<double, std::size_t>;
using MinHeap = bioimage_cpp::detail::DenseIndexedHeap<
    HeapPriority,
    std::greater<HeapPriority>
>;

struct SolveResult {
    DijkstraResult result;
    std::size_t reached_target = std::numeric_limits<std::size_t>::max();
};

inline void validate_inputs(
    const ConstArrayView<std::uint8_t> &mask,
    DijkstraOptions &options,
    const ConstArrayView<double> *costs
) {
    const auto ndim = mask.shape.size();
    if (ndim != 2 && ndim != 3) {
        throw std::invalid_argument(
            "mask must have ndim 2 or 3, got ndim=" + std::to_string(ndim)
        );
    }
    for (std::size_t axis = 0; axis < ndim; ++axis) {
        if (mask.shape[axis] < 0) {
            throw std::invalid_argument("mask shape entries must be non-negative");
        }
    }

    if (options.connectivity == 0) {
        options.connectivity = static_cast<int>(ndim);
    }
    if (options.connectivity < 1 || options.connectivity > static_cast<int>(ndim)) {
        throw std::invalid_argument(
            "connectivity must be in [1, ndim], got connectivity=" +
            std::to_string(options.connectivity) + " for ndim=" + std::to_string(ndim)
        );
    }

    if (options.cost_mode == DijkstraCostMode::Node && !options.spacing.empty()) {
        throw std::invalid_argument("spacing must be omitted for node cost mode");
    }
    if (options.spacing.empty()) {
        options.spacing.assign(ndim, 1.0);
    }
    if (options.spacing.size() != ndim) {
        throw std::invalid_argument(
            "spacing must have length matching mask ndim, got ndim=" +
            std::to_string(ndim) + ", spacing length=" +
            std::to_string(options.spacing.size())
        );
    }
    for (std::size_t axis = 0; axis < ndim; ++axis) {
        if (!(std::isfinite(options.spacing[axis]) && options.spacing[axis] > 0.0)) {
            throw std::invalid_argument(
                "spacing values must be positive and finite, got spacing[" +
                std::to_string(axis) + "]=" + std::to_string(options.spacing[axis])
            );
        }
    }

    const bool needs_costs = options.cost_mode != DijkstraCostMode::Physical;
    if (needs_costs != (costs != nullptr)) {
        throw std::invalid_argument(
            needs_costs ? "costs are required for the selected cost mode"
                        : "costs must be omitted for physical cost mode"
        );
    }
    if (costs == nullptr) {
        return;
    }
    if (costs->shape != mask.shape) {
        throw std::invalid_argument("costs must have the same shape as mask");
    }
    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);
    for (std::size_t index = 0; index < n; ++index) {
        const double value = costs->data[index];
        if (!(std::isfinite(value) && value >= 0.0)) {
            throw std::invalid_argument(
                "costs must contain finite non-negative values, got costs[" +
                std::to_string(index) + "]=" + std::to_string(value)
            );
        }
    }
}

inline SolveResult solve(
    const ConstArrayView<std::uint8_t> &mask,
    const std::vector<std::size_t> &sources,
    DijkstraOptions options,
    const ConstArrayView<double> *costs,
    const bool return_predecessors,
    const std::vector<std::uint8_t> *target_mask = nullptr
) {
    validate_inputs(mask, options, costs);
    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);
    if (sources.empty()) {
        throw std::invalid_argument("sources must contain at least one coordinate");
    }
    for (const auto source : sources) {
        if (source >= n) {
            throw std::invalid_argument("source index is out of bounds");
        }
        if (mask.data[source] == 0) {
            throw std::invalid_argument("sources must lie inside the foreground mask");
        }
    }
    if (target_mask != nullptr && target_mask->size() != n) {
        throw std::invalid_argument("target mask must have the same number of elements as mask");
    }

    SolveResult solved;
    solved.result.distances.assign(n, kInfinity);
    if (return_predecessors) {
        solved.result.predecessors.assign(n, -1);
    }

    std::vector<std::uint8_t> settled(n, 0);
    MinHeap heap(n);
    std::vector<std::size_t> ordered_sources = sources;
    std::sort(ordered_sources.begin(), ordered_sources.end());
    ordered_sources.erase(
        std::unique(ordered_sources.begin(), ordered_sources.end()),
        ordered_sources.end()
    );
    for (const auto source : ordered_sources) {
        solved.result.distances[source] = 0.0;
        if (return_predecessors) {
            solved.result.predecessors[source] = static_cast<std::int64_t>(source);
        }
        heap.push(source, {0.0, source});
    }

    const auto strides = bioimage_cpp::detail::c_order_strides(mask.shape);
    const auto neighbors = build_neighbors(
        mask.shape.size(), options.connectivity, options.spacing
    );

    while (!heap.empty()) {
        const auto entry = heap.pop();
        const std::size_t node = entry.key;
        if (settled[node] != 0) {
            continue;
        }
        settled[node] = 1;

        if (target_mask != nullptr && (*target_mask)[node] != 0) {
            solved.reached_target = node;
            break;
        }

        const double node_distance = solved.result.distances[node];
        for (const auto &neighbor : neighbors) {
            std::uint64_t target_u64 = 0;
            if (!bioimage_cpp::detail::valid_offset_target(
                    static_cast<std::uint64_t>(node),
                    neighbor.offset,
                    mask.shape,
                    strides,
                    target_u64
                )) {
                continue;
            }
            const auto target = static_cast<std::size_t>(target_u64);
            if (mask.data[target] == 0 || settled[target] != 0) {
                continue;
            }

            double edge_cost = neighbor.physical_length;
            if (options.cost_mode == DijkstraCostMode::Node) {
                edge_cost = costs->data[target];
            } else if (options.cost_mode == DijkstraCostMode::NodeTimesPhysical) {
                edge_cost = costs->data[target] * neighbor.physical_length;
            }
            const double candidate = node_distance + edge_cost;
            if (candidate < solved.result.distances[target]) {
                solved.result.distances[target] = candidate;
                if (return_predecessors) {
                    solved.result.predecessors[target] = static_cast<std::int64_t>(node);
                }
                heap.push_or_change(target, {candidate, target});
            }
        }
    }

    return solved;
}

} // namespace detail_grid_dijkstra

// Full multi-source Dijkstra distance field on a 2D/3D mask. `sources` are
// flat C-order indices. See DijkstraCostMode for the edge-cost contract.
inline DijkstraResult dijkstra_distance_field(
    const ConstArrayView<std::uint8_t> &mask,
    const std::vector<std::size_t> &sources,
    DijkstraOptions options = {},
    const ConstArrayView<double> *costs = nullptr,
    const bool return_predecessors = false
) {
    return detail_grid_dijkstra::solve(
        mask, sources, std::move(options), costs, return_predecessors
    ).result;
}

// Early-stopping one-source/multi-target Dijkstra. Returns flat C-order voxel
// indices from source to the cheapest reached target. Equal target distances
// are resolved by flat index because the heap priority includes the key.
inline std::vector<std::size_t> dijkstra_path(
    const ConstArrayView<std::uint8_t> &mask,
    const std::size_t source,
    const std::vector<std::size_t> &targets,
    DijkstraOptions options = {},
    const ConstArrayView<double> *costs = nullptr
) {
    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);
    if (targets.empty()) {
        throw std::invalid_argument("targets must contain at least one coordinate");
    }
    std::vector<std::uint8_t> target_mask(n, 0);
    for (const auto target : targets) {
        if (target >= n) {
            throw std::invalid_argument("target index is out of bounds");
        }
        if (mask.data[target] == 0) {
            throw std::invalid_argument("targets must lie inside the foreground mask");
        }
        target_mask[target] = 1;
    }

    auto solved = detail_grid_dijkstra::solve(
        mask, {source}, std::move(options), costs, true, &target_mask
    );
    if (solved.reached_target == std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("no target is reachable from source");
    }

    std::vector<std::size_t> reverse_path;
    std::size_t node = solved.reached_target;
    while (true) {
        reverse_path.push_back(node);
        const auto parent = solved.result.predecessors[node];
        if (parent < 0) {
            throw std::runtime_error("invalid predecessor chain while reconstructing path");
        }
        if (static_cast<std::size_t>(parent) == node) {
            break;
        }
        node = static_cast<std::size_t>(parent);
        if (reverse_path.size() > n) {
            throw std::runtime_error("cycle in predecessor chain while reconstructing path");
        }
    }
    std::reverse(reverse_path.begin(), reverse_path.end());
    return reverse_path;
}

} // namespace bioimage_cpp::distance
