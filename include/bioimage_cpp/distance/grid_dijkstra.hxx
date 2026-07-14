#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/indexed_heap.hxx"
#include "bioimage_cpp/distance/detail/delta_stepping.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp::distance {

enum class DijkstraCostMode {
    Physical,
    Node,
    NodeTimesPhysical,
};

struct DijkstraOptions {
    int connectivity = 0;
    std::vector<double> spacing;
    DijkstraCostMode cost_mode = DijkstraCostMode::Physical;
    std::size_t number_of_threads = 1;
};

struct DijkstraResult {
    std::vector<double> distances;
    std::vector<std::int64_t> predecessors;
};

namespace detail_grid_dijkstra {

inline constexpr double kInfinity = std::numeric_limits<double>::infinity();
inline constexpr std::size_t kNoTarget = std::numeric_limits<std::size_t>::max();
inline constexpr std::uint8_t kDiscovered = 1;
inline constexpr std::uint8_t kSettled = 2;
inline constexpr std::uint8_t kTarget = 4;

struct Neighbor {
    std::array<std::int8_t, 3> step{0, 0, 0};
    std::ptrdiff_t delta = 0;
    double physical_length = 0.0;
};

struct HeapEntry {
    double distance = 0.0;
    std::size_t node = 0;
};

struct HeapGreater {
    bool operator()(const HeapEntry &a, const HeapEntry &b) const noexcept {
        return a.distance > b.distance ||
            (a.distance == b.distance && a.node > b.node);
    }
};

using IndexedPriority = std::pair<double, std::size_t>;
using IndexedMinHeap = bioimage_cpp::detail::DenseIndexedHeap<
    IndexedPriority,
    std::greater<IndexedPriority>
>;

inline void lazy_push(std::vector<HeapEntry> &heap, const HeapEntry entry) {
    heap.push_back(entry);
    std::push_heap(heap.begin(), heap.end(), HeapGreater{});
}

inline HeapEntry lazy_pop(std::vector<HeapEntry> &heap) {
    std::pop_heap(heap.begin(), heap.end(), HeapGreater{});
    const auto entry = heap.back();
    heap.pop_back();
    return entry;
}

} // namespace detail_grid_dijkstra

// Reusable scratch storage for dense grid Dijkstra calls. Capacities are kept
// between invocations. A workspace is single-threaded and must not be used by
// concurrent calls.
struct DijkstraWorkspace {
    std::vector<std::uint8_t> state;
    std::vector<double> scratch_distances;
    std::vector<std::int64_t> scratch_predecessors;
    std::vector<std::size_t> touched;
    std::vector<detail_grid_dijkstra::HeapEntry> lazy_heap;
    detail_grid_dijkstra::IndexedMinHeap indexed_heap;
    std::size_t indexed_capacity = 0;

    std::array<detail_grid_dijkstra::Neighbor, 26> neighbors{};
    std::size_t neighbor_count = 0;
    std::vector<std::ptrdiff_t> prepared_shape;
    std::vector<std::ptrdiff_t> strides;
    std::vector<double> prepared_spacing;
    int prepared_connectivity = -1;
    bool state_is_clean = true;
    detail_delta_stepping::DeltaSteppingWorkspace<std::size_t> parallel;
};

namespace detail_grid_dijkstra {

inline void validate_inputs(
    const ConstArrayView<std::uint8_t> &mask,
    DijkstraOptions &options,
    const ConstArrayView<double> *costs,
    const bool validate_cost_values = true
) {
    const auto ndim = mask.shape.size();
    if (ndim != 2 && ndim != 3) {
        throw std::invalid_argument(
            "mask must have ndim 2 or 3, got ndim=" + std::to_string(ndim)
        );
    }
    for (const auto extent : mask.shape) {
        if (extent < 0) {
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
    if (!validate_cost_values) {
        return;
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

inline void validate_sources(
    const ConstArrayView<std::uint8_t> &mask,
    const std::span<const std::size_t> sources
) {
    if (sources.empty()) {
        throw std::invalid_argument("sources must contain at least one coordinate");
    }
    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);
    for (const auto source : sources) {
        if (source >= n) {
            throw std::invalid_argument("source index is out of bounds");
        }
        if (mask.data[source] == 0) {
            throw std::invalid_argument("sources must lie inside the foreground mask");
        }
    }
}

inline void validate_targets(
    const ConstArrayView<std::uint8_t> &mask,
    const std::span<const std::size_t> targets
) {
    if (targets.empty()) {
        throw std::invalid_argument("targets must contain at least one coordinate");
    }
    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);
    for (const auto target : targets) {
        if (target >= n) {
            throw std::invalid_argument("target index is out of bounds");
        }
        if (mask.data[target] == 0) {
            throw std::invalid_argument("targets must lie inside the foreground mask");
        }
    }
}

inline void build_neighbors(
    DijkstraWorkspace &workspace,
    const std::vector<std::ptrdiff_t> &shape,
    const int connectivity,
    const std::vector<double> &spacing
) {
    workspace.neighbor_count = 0;
    workspace.strides = bioimage_cpp::detail::c_order_strides(shape);
    if (shape.size() == 2) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int nonzero = (dy != 0) + (dx != 0);
                if (nonzero == 0 || nonzero > connectivity) {
                    continue;
                }
                const double py = static_cast<double>(dy) * spacing[0];
                const double px = static_cast<double>(dx) * spacing[1];
                workspace.neighbors[workspace.neighbor_count++] = {
                    {static_cast<std::int8_t>(dy), static_cast<std::int8_t>(dx), 0},
                    static_cast<std::ptrdiff_t>(dy) * workspace.strides[0] + dx,
                    std::sqrt(py * py + px * px),
                };
            }
        }
    } else {
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nonzero = (dz != 0) + (dy != 0) + (dx != 0);
                    if (nonzero == 0 || nonzero > connectivity) {
                        continue;
                    }
                    const double pz = static_cast<double>(dz) * spacing[0];
                    const double py = static_cast<double>(dy) * spacing[1];
                    const double px = static_cast<double>(dx) * spacing[2];
                    workspace.neighbors[workspace.neighbor_count++] = {
                        {
                            static_cast<std::int8_t>(dz),
                            static_cast<std::int8_t>(dy),
                            static_cast<std::int8_t>(dx),
                        },
                        static_cast<std::ptrdiff_t>(dz) * workspace.strides[0] +
                            static_cast<std::ptrdiff_t>(dy) * workspace.strides[1] + dx,
                        std::sqrt(pz * pz + py * py + px * px),
                    };
                }
            }
        }
    }
    workspace.prepared_shape = shape;
    workspace.prepared_connectivity = connectivity;
    workspace.prepared_spacing = spacing;
}

inline void prepare_geometry(
    DijkstraWorkspace &workspace,
    const std::vector<std::ptrdiff_t> &shape,
    const DijkstraOptions &options
) {
    if (workspace.prepared_shape != shape ||
        workspace.prepared_connectivity != options.connectivity ||
        workspace.prepared_spacing != options.spacing) {
        build_neighbors(workspace, shape, options.connectivity, options.spacing);
    }
}

inline void ensure_state(DijkstraWorkspace &workspace, const std::size_t n) {
    if (workspace.state.size() != n) {
        workspace.state.assign(n, 0);
        workspace.state_is_clean = true;
    }
}

inline void begin_full(DijkstraWorkspace &workspace, const std::size_t n) {
    ensure_state(workspace, n);
    std::fill(workspace.state.begin(), workspace.state.end(), std::uint8_t{0});
    workspace.state_is_clean = false;
    workspace.touched.clear();
    workspace.lazy_heap.clear();
}

inline void begin_path(DijkstraWorkspace &workspace, const std::size_t n) {
    ensure_state(workspace, n);
    if (!workspace.state_is_clean) {
        std::fill(workspace.state.begin(), workspace.state.end(), std::uint8_t{0});
        workspace.state_is_clean = true;
    }
    workspace.touched.clear();
    workspace.lazy_heap.clear();
}

inline void prepare_indexed_heap(DijkstraWorkspace &workspace, const std::size_t n) {
    if (workspace.indexed_capacity != n) {
        workspace.indexed_heap.reset_capacity(n);
        workspace.indexed_capacity = n;
    } else {
        workspace.indexed_heap.clear();
    }
}

inline void cleanup_path_state(
    DijkstraWorkspace &workspace,
    const std::span<const std::size_t> targets
) {
    for (const auto node : workspace.touched) {
        workspace.state[node] = 0;
    }
    for (const auto target : targets) {
        workspace.state[target] = 0;
    }
    workspace.touched.clear();
    workspace.lazy_heap.clear();
    workspace.indexed_heap.clear();
    workspace.state_is_clean = true;
}

template <int NDim, bool ZeroHalo, class Body>
inline void for_each_neighbor(
    const std::size_t node,
    const ConstArrayView<std::uint8_t> &mask,
    const DijkstraWorkspace &workspace,
    const Body &body
) {
    if constexpr (ZeroHalo) {
        for (std::size_t i = 0; i < workspace.neighbor_count; ++i) {
            const auto &neighbor = workspace.neighbors[i];
            const auto target = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(node) + neighbor.delta
            );
            body(target, neighbor.physical_length);
        }
        return;
    }

    if constexpr (NDim == 2) {
        const auto width = static_cast<std::size_t>(mask.shape[1]);
        const auto y = node / width;
        const auto x = node - y * width;
        const bool interior = y > 0 && y + 1 < static_cast<std::size_t>(mask.shape[0]) &&
            x > 0 && x + 1 < width;
        for (std::size_t i = 0; i < workspace.neighbor_count; ++i) {
            const auto &neighbor = workspace.neighbors[i];
            if (!interior && (
                static_cast<std::size_t>(static_cast<std::ptrdiff_t>(y) + neighbor.step[0]) >=
                    static_cast<std::size_t>(mask.shape[0]) ||
                static_cast<std::size_t>(static_cast<std::ptrdiff_t>(x) + neighbor.step[1]) >=
                    width)) {
                continue;
            }
            const auto target = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(node) + neighbor.delta
            );
            body(target, neighbor.physical_length);
        }
    } else {
        const auto height = static_cast<std::size_t>(mask.shape[1]);
        const auto width = static_cast<std::size_t>(mask.shape[2]);
        const auto slice = height * width;
        const auto z = node / slice;
        const auto remainder = node - z * slice;
        const auto y = remainder / width;
        const auto x = remainder - y * width;
        const bool interior = z > 0 && z + 1 < static_cast<std::size_t>(mask.shape[0]) &&
            y > 0 && y + 1 < height && x > 0 && x + 1 < width;
        for (std::size_t i = 0; i < workspace.neighbor_count; ++i) {
            const auto &neighbor = workspace.neighbors[i];
            if (!interior && (
                static_cast<std::size_t>(static_cast<std::ptrdiff_t>(z) + neighbor.step[0]) >=
                    static_cast<std::size_t>(mask.shape[0]) ||
                static_cast<std::size_t>(static_cast<std::ptrdiff_t>(y) + neighbor.step[1]) >=
                    height ||
                static_cast<std::size_t>(static_cast<std::ptrdiff_t>(x) + neighbor.step[2]) >=
                    width)) {
                continue;
            }
            const auto target = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(node) + neighbor.delta
            );
            body(target, neighbor.physical_length);
        }
    }
}

inline std::size_t foreground_count(const ConstArrayView<std::uint8_t> &mask) {
    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);
    return static_cast<std::size_t>(std::count_if(
        mask.data, mask.data + n,
        [](const std::uint8_t value) { return value != 0; }
    ));
}

inline double sampled_positive_cost_median(
    const ConstArrayView<std::uint8_t> &mask,
    const ConstArrayView<double> &costs
) {
    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);
    std::size_t positive_count = 0;
    for (std::size_t index = 0; index < n; ++index) {
        positive_count += mask.data[index] != 0 && costs.data[index] > 0.0 ? 1 : 0;
    }
    if (positive_count == 0) {
        return 1.0;
    }
    constexpr std::size_t maximum_samples = 4096;
    const auto stride = std::max<std::size_t>(
        1, (positive_count + maximum_samples - 1) / maximum_samples
    );
    std::vector<double> samples;
    samples.reserve(std::min(positive_count, maximum_samples));
    std::size_t positive_index = 0;
    for (std::size_t index = 0; index < n && samples.size() < maximum_samples; ++index) {
        if (mask.data[index] == 0 || costs.data[index] == 0.0) {
            continue;
        }
        if (positive_index % stride == stride / 2) {
            samples.push_back(costs.data[index]);
        }
        ++positive_index;
    }
    if (samples.empty()) {
        return 1.0;
    }
    const auto middle = samples.begin() + static_cast<std::ptrdiff_t>(samples.size() / 2);
    std::nth_element(samples.begin(), middle, samples.end());
    return *middle;
}

inline double parallel_delta(
    const ConstArrayView<std::uint8_t> &mask,
    const DijkstraOptions &options,
    const ConstArrayView<double> *costs,
    const DijkstraWorkspace &workspace
) {
    double minimum_length = std::numeric_limits<double>::infinity();
    for (std::size_t neighbor = 0; neighbor < workspace.neighbor_count; ++neighbor) {
        minimum_length = std::min(
            minimum_length, workspace.neighbors[neighbor].physical_length
        );
    }
    if (options.cost_mode == DijkstraCostMode::Physical) {
        return minimum_length;
    }
    const double node_scale = sampled_positive_cost_median(mask, *costs);
    return options.cost_mode == DijkstraCostMode::Node
        ? node_scale
        : node_scale * minimum_length;
}

template <DijkstraCostMode Mode, int NDim, bool ZeroHalo>
inline detail_delta_stepping::DeltaSteppingResult<std::size_t> run_parallel(
    const ConstArrayView<std::uint8_t> &mask,
    const std::span<const std::size_t> sources,
    const std::span<const std::size_t> targets,
    const ConstArrayView<double> *costs,
    DijkstraWorkspace &workspace,
    const double delta,
    const std::size_t number_of_threads
) {
    const auto neighbors = [&](const std::size_t node, const auto &body) {
        for_each_neighbor<NDim, ZeroHalo>(
            node, mask, workspace,
            [&](const std::size_t target, const double physical_length) {
                if (mask.data[target] == 0) {
                    return;
                }
                double weight = physical_length;
                if constexpr (Mode == DijkstraCostMode::Node) {
                    weight = costs->data[target];
                } else if constexpr (Mode == DijkstraCostMode::NodeTimesPhysical) {
                    weight = costs->data[target] * physical_length;
                }
                body(target, weight);
            }
        );
    };
    return detail_delta_stepping::run<std::size_t>(
        bioimage_cpp::detail::number_of_elements(mask.shape),
        sources,
        targets,
        delta,
        number_of_threads,
        workspace.neighbor_count,
        neighbors,
        workspace.parallel
    );
}

template <DijkstraCostMode Mode>
inline detail_delta_stepping::DeltaSteppingResult<std::size_t>
dispatch_parallel_dimension(
    const ConstArrayView<std::uint8_t> &mask,
    const std::span<const std::size_t> sources,
    const std::span<const std::size_t> targets,
    const ConstArrayView<double> *costs,
    DijkstraWorkspace &workspace,
    const double delta,
    const std::size_t number_of_threads,
    const bool zero_halo
) {
    if (mask.shape.size() == 2) {
        return zero_halo
            ? run_parallel<Mode, 2, true>(
                  mask, sources, targets, costs, workspace, delta, number_of_threads
              )
            : run_parallel<Mode, 2, false>(
                  mask, sources, targets, costs, workspace, delta, number_of_threads
              );
    }
    return zero_halo
        ? run_parallel<Mode, 3, true>(
              mask, sources, targets, costs, workspace, delta, number_of_threads
          )
        : run_parallel<Mode, 3, false>(
              mask, sources, targets, costs, workspace, delta, number_of_threads
          );
}

inline detail_delta_stepping::DeltaSteppingResult<std::size_t>
dispatch_parallel_mode(
    const ConstArrayView<std::uint8_t> &mask,
    const std::span<const std::size_t> sources,
    const std::span<const std::size_t> targets,
    const DijkstraOptions &options,
    const ConstArrayView<double> *costs,
    DijkstraWorkspace &workspace,
    const bool zero_halo
) {
    const double delta = parallel_delta(mask, options, costs, workspace);
    switch (options.cost_mode) {
        case DijkstraCostMode::Physical:
            return dispatch_parallel_dimension<DijkstraCostMode::Physical>(
                mask, sources, targets, costs, workspace, delta,
                options.number_of_threads, zero_halo
            );
        case DijkstraCostMode::Node:
            return dispatch_parallel_dimension<DijkstraCostMode::Node>(
                mask, sources, targets, costs, workspace, delta,
                options.number_of_threads, zero_halo
            );
        case DijkstraCostMode::NodeTimesPhysical:
            return dispatch_parallel_dimension<DijkstraCostMode::NodeTimesPhysical>(
                mask, sources, targets, costs, workspace, delta,
                options.number_of_threads, zero_halo
            );
    }
    return {};
}

inline bool use_parallel_backend(
    const ConstArrayView<std::uint8_t> &mask,
    const DijkstraOptions &options,
    const std::size_t number_of_sources,
    const bool stop_at_target
) {
    // Directed node costs admit the especially efficient one-insertion heap
    // kernel. The staged parallel backend loses decisively to it in the
    // benchmark matrix, so keep that specialization for every thread request.
    if (options.number_of_threads == 1 || stop_at_target ||
        options.cost_mode == DijkstraCostMode::Node) {
        return false;
    }
    const auto foreground = foreground_count(mask);
    if (foreground < detail_delta_stepping::kSequentialProblemThreshold) {
        return false;
    }
    // Delta stepping pays off when a field has a broad initial wavefront. The
    // optimized heap wins for one/few-source fields and every path solve on the
    // current benchmark matrix.
    const auto minimum_sources = std::max<std::size_t>(2, foreground / 256);
    if (number_of_sources < minimum_sources) {
        return false;
    }
    return bioimage_cpp::detail::normalize_thread_count(
        options.number_of_threads, foreground
    ) > 1;
}

template <DijkstraCostMode Mode, int NDim, bool ZeroHalo>
inline std::size_t run_lazy(
    const ConstArrayView<std::uint8_t> &mask,
    const std::span<const std::size_t> sources,
    const ConstArrayView<double> *costs,
    double *distances,
    std::int64_t *predecessors,
    DijkstraWorkspace &workspace,
    const bool stop_at_target,
    const bool track_touched
) {
    for (const auto source : sources) {
        if ((workspace.state[source] & kDiscovered) != 0) {
            continue;
        }
        workspace.state[source] |= kDiscovered;
        if (track_touched) {
            workspace.touched.push_back(source);
        }
        if (distances != nullptr) {
            distances[source] = 0.0;
        }
        if (predecessors != nullptr) {
            predecessors[source] = static_cast<std::int64_t>(source);
        }
        lazy_push(workspace.lazy_heap, {0.0, source});
    }

    while (!workspace.lazy_heap.empty()) {
        const auto entry = lazy_pop(workspace.lazy_heap);
        const auto node = entry.node;
        if ((workspace.state[node] & kSettled) != 0) {
            continue;
        }
        workspace.state[node] |= kSettled;
        if (stop_at_target && (workspace.state[node] & kTarget) != 0) {
            return node;
        }

        for_each_neighbor<NDim, ZeroHalo>(
            node, mask, workspace,
            [&](const std::size_t target, const double physical_length) {
                if (mask.data[target] == 0 ||
                    (workspace.state[target] & kSettled) != 0) {
                    return;
                }
                if constexpr (Mode == DijkstraCostMode::Node) {
                    // Every incoming edge to target has the same cost. The
                    // first settled neighbor to discover it is therefore
                    // optimal, so no decrease-key is possible.
                    if ((workspace.state[target] & kDiscovered) != 0) {
                        return;
                    }
                    const double candidate = entry.distance + costs->data[target];
                    workspace.state[target] |= kDiscovered;
                    if (track_touched) {
                        workspace.touched.push_back(target);
                    }
                    if (distances != nullptr) {
                        distances[target] = candidate;
                    }
                    if (predecessors != nullptr) {
                        predecessors[target] = static_cast<std::int64_t>(node);
                    }
                    lazy_push(workspace.lazy_heap, {candidate, target});
                } else {
                    const double candidate = entry.distance + physical_length;
                    const bool discovered =
                        (workspace.state[target] & kDiscovered) != 0;
                    if (discovered && !(candidate < distances[target])) {
                        return;
                    }
                    if (!discovered) {
                        workspace.state[target] |= kDiscovered;
                        if (track_touched) {
                            workspace.touched.push_back(target);
                        }
                    }
                    distances[target] = candidate;
                    if (predecessors != nullptr) {
                        predecessors[target] = static_cast<std::int64_t>(node);
                    }
                    lazy_push(workspace.lazy_heap, {candidate, target});
                }
            }
        );
    }
    return kNoTarget;
}

template <int NDim, bool ZeroHalo>
inline std::size_t run_indexed(
    const ConstArrayView<std::uint8_t> &mask,
    const std::span<const std::size_t> sources,
    const ConstArrayView<double> *costs,
    double *distances,
    std::int64_t *predecessors,
    DijkstraWorkspace &workspace,
    const bool stop_at_target,
    const bool track_touched
) {
    for (const auto source : sources) {
        if ((workspace.state[source] & kDiscovered) != 0) {
            continue;
        }
        workspace.state[source] |= kDiscovered;
        if (track_touched) {
            workspace.touched.push_back(source);
        }
        distances[source] = 0.0;
        if (predecessors != nullptr) {
            predecessors[source] = static_cast<std::int64_t>(source);
        }
        workspace.indexed_heap.push(source, {0.0, source});
    }

    while (!workspace.indexed_heap.empty()) {
        const auto entry = workspace.indexed_heap.pop();
        const auto node = entry.key;
        workspace.state[node] |= kSettled;
        if (stop_at_target && (workspace.state[node] & kTarget) != 0) {
            return node;
        }
        const double node_distance = entry.priority.first;
        for_each_neighbor<NDim, ZeroHalo>(
            node, mask, workspace,
            [&](const std::size_t target, const double physical_length) {
                if (mask.data[target] == 0 ||
                    (workspace.state[target] & kSettled) != 0) {
                    return;
                }
                const double candidate =
                    node_distance + costs->data[target] * physical_length;
                const bool discovered =
                    (workspace.state[target] & kDiscovered) != 0;
                if (discovered && !(candidate < distances[target])) {
                    return;
                }
                if (!discovered) {
                    workspace.state[target] |= kDiscovered;
                    if (track_touched) {
                        workspace.touched.push_back(target);
                    }
                }
                distances[target] = candidate;
                if (predecessors != nullptr) {
                    predecessors[target] = static_cast<std::int64_t>(node);
                }
                workspace.indexed_heap.push_or_change(target, {candidate, target});
            }
        );
    }
    return kNoTarget;
}

template <DijkstraCostMode Mode>
inline std::size_t dispatch_run(
    const ConstArrayView<std::uint8_t> &mask,
    const std::span<const std::size_t> sources,
    const ConstArrayView<double> *costs,
    double *distances,
    std::int64_t *predecessors,
    DijkstraWorkspace &workspace,
    const bool stop_at_target,
    const bool track_touched,
    const bool zero_halo
) {
    const bool is_2d = mask.shape.size() == 2;
    if constexpr (Mode == DijkstraCostMode::NodeTimesPhysical) {
        if (is_2d) {
            return zero_halo
                ? run_indexed<2, true>(mask, sources, costs, distances, predecessors,
                    workspace, stop_at_target, track_touched)
                : run_indexed<2, false>(mask, sources, costs, distances, predecessors,
                    workspace, stop_at_target, track_touched);
        }
        return zero_halo
            ? run_indexed<3, true>(mask, sources, costs, distances, predecessors,
                workspace, stop_at_target, track_touched)
            : run_indexed<3, false>(mask, sources, costs, distances, predecessors,
                workspace, stop_at_target, track_touched);
    } else {
        if (is_2d) {
            return zero_halo
                ? run_lazy<Mode, 2, true>(mask, sources, costs, distances, predecessors,
                    workspace, stop_at_target, track_touched)
                : run_lazy<Mode, 2, false>(mask, sources, costs, distances, predecessors,
                    workspace, stop_at_target, track_touched);
        }
        return zero_halo
            ? run_lazy<Mode, 3, true>(mask, sources, costs, distances, predecessors,
                workspace, stop_at_target, track_touched)
            : run_lazy<Mode, 3, false>(mask, sources, costs, distances, predecessors,
                workspace, stop_at_target, track_touched);
    }
}

inline std::size_t dispatch_mode(
    const ConstArrayView<std::uint8_t> &mask,
    const std::span<const std::size_t> sources,
    const DijkstraOptions &options,
    const ConstArrayView<double> *costs,
    double *distances,
    std::int64_t *predecessors,
    DijkstraWorkspace &workspace,
    const bool stop_at_target,
    const bool track_touched,
    const bool zero_halo
) {
    switch (options.cost_mode) {
        case DijkstraCostMode::Physical:
            return dispatch_run<DijkstraCostMode::Physical>(
                mask, sources, costs, distances, predecessors, workspace,
                stop_at_target, track_touched, zero_halo
            );
        case DijkstraCostMode::Node:
            return dispatch_run<DijkstraCostMode::Node>(
                mask, sources, costs, distances, predecessors, workspace,
                stop_at_target, track_touched, zero_halo
            );
        case DijkstraCostMode::NodeTimesPhysical:
            return dispatch_run<DijkstraCostMode::NodeTimesPhysical>(
                mask, sources, costs, distances, predecessors, workspace,
                stop_at_target, track_touched, zero_halo
            );
    }
    throw std::invalid_argument("invalid Dijkstra cost mode");
}

inline void distance_field_impl(
    const ConstArrayView<std::uint8_t> &mask,
    const std::vector<std::size_t> &sources,
    DijkstraOptions options,
    const ConstArrayView<double> *costs,
    const bool return_predecessors,
    DijkstraWorkspace &workspace,
    DijkstraResult &result,
    const bool validate_cost_values,
    const bool zero_halo
) {
    validate_inputs(mask, options, costs, validate_cost_values);
    validate_sources(mask, sources);
    prepare_geometry(workspace, mask.shape, options);
    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);
    std::vector<std::size_t> ordered_sources = sources;
    std::sort(ordered_sources.begin(), ordered_sources.end());
    ordered_sources.erase(
        std::unique(ordered_sources.begin(), ordered_sources.end()),
        ordered_sources.end()
    );

    if (use_parallel_backend(mask, options, ordered_sources.size(), false)) {
        const auto parallel_result = dispatch_parallel_mode(
            mask, ordered_sources, {}, options, costs, workspace, zero_halo
        );
        if (parallel_result.completed) {
            result.distances.assign(n, kInfinity);
            if (return_predecessors) {
                result.predecessors.assign(n, -1);
            } else {
                result.predecessors.clear();
            }
            for (const auto node : workspace.parallel.touched) {
                result.distances[node] = workspace.parallel.distances[node];
                if (return_predecessors) {
                    result.predecessors[node] = static_cast<std::int64_t>(
                        workspace.parallel.predecessors[node]
                    );
                }
            }
            return;
        }
    }

    begin_full(workspace, n);
    if (options.cost_mode == DijkstraCostMode::NodeTimesPhysical) {
        prepare_indexed_heap(workspace, n);
    }
    result.distances.assign(n, kInfinity);
    if (return_predecessors) {
        result.predecessors.assign(n, -1);
    } else {
        result.predecessors.clear();
    }
    dispatch_mode(
        mask, ordered_sources, options, costs, result.distances.data(),
        return_predecessors ? result.predecessors.data() : nullptr,
        workspace, false, false, zero_halo
    );
}

inline void path_impl(
    const ConstArrayView<std::uint8_t> &mask,
    const std::size_t source,
    const std::vector<std::size_t> &targets,
    DijkstraOptions options,
    const ConstArrayView<double> *costs,
    DijkstraWorkspace &workspace,
    std::vector<std::size_t> &path,
    const bool validate_cost_values,
    const bool zero_halo
) {
    validate_inputs(mask, options, costs, validate_cost_values);
    const std::array<std::size_t, 1> source_array{source};
    validate_sources(mask, source_array);
    validate_targets(mask, targets);
    prepare_geometry(workspace, mask.shape, options);
    const auto n = bioimage_cpp::detail::number_of_elements(mask.shape);

    if (use_parallel_backend(mask, options, 1, true)) {
        const auto parallel_result = dispatch_parallel_mode(
            mask, source_array, targets, options, costs, workspace, zero_halo
        );
        if (parallel_result.completed) {
            const auto reached = parallel_result.reached_target;
            if (reached == kNoTarget) {
                throw std::runtime_error("no target is reachable from source");
            }
            path.clear();
            auto node = reached;
            while (true) {
                path.push_back(node);
                const auto parent = workspace.parallel.predecessors[node];
                if (parent == node) {
                    break;
                }
                node = parent;
                if (path.size() > n) {
                    throw std::runtime_error(
                        "cycle in predecessor chain while reconstructing path"
                    );
                }
            }
            std::reverse(path.begin(), path.end());
            return;
        }
    }

    begin_path(workspace, n);
    if (options.cost_mode == DijkstraCostMode::NodeTimesPhysical) {
        prepare_indexed_heap(workspace, n);
    }
    workspace.scratch_predecessors.resize(n);
    double *distances = nullptr;
    if (options.cost_mode != DijkstraCostMode::Node) {
        workspace.scratch_distances.resize(n);
        distances = workspace.scratch_distances.data();
    }
    for (const auto target : targets) {
        workspace.state[target] |= kTarget;
    }

    const auto reached = dispatch_mode(
        mask, source_array, options, costs, distances,
        workspace.scratch_predecessors.data(), workspace,
        true, true, zero_halo
    );
    if (reached == kNoTarget) {
        cleanup_path_state(workspace, targets);
        throw std::runtime_error("no target is reachable from source");
    }

    path.clear();
    auto node = reached;
    while (true) {
        path.push_back(node);
        const auto parent = workspace.scratch_predecessors[node];
        if (parent < 0) {
            cleanup_path_state(workspace, targets);
            throw std::runtime_error("invalid predecessor chain while reconstructing path");
        }
        if (static_cast<std::size_t>(parent) == node) {
            break;
        }
        node = static_cast<std::size_t>(parent);
        if (path.size() > n) {
            cleanup_path_state(workspace, targets);
            throw std::runtime_error("cycle in predecessor chain while reconstructing path");
        }
    }
    std::reverse(path.begin(), path.end());
    cleanup_path_state(workspace, targets);
}

} // namespace detail_grid_dijkstra

inline void dijkstra_distance_field(
    const ConstArrayView<std::uint8_t> &mask,
    const std::vector<std::size_t> &sources,
    DijkstraWorkspace &workspace,
    DijkstraResult &result,
    DijkstraOptions options = {},
    const ConstArrayView<double> *costs = nullptr,
    const bool return_predecessors = false
) {
    detail_grid_dijkstra::distance_field_impl(
        mask, sources, std::move(options), costs, return_predecessors,
        workspace, result, true, false
    );
}

inline DijkstraResult dijkstra_distance_field(
    const ConstArrayView<std::uint8_t> &mask,
    const std::vector<std::size_t> &sources,
    DijkstraOptions options = {},
    const ConstArrayView<double> *costs = nullptr,
    const bool return_predecessors = false
) {
    DijkstraWorkspace workspace;
    DijkstraResult result;
    dijkstra_distance_field(
        mask, sources, workspace, result, std::move(options), costs,
        return_predecessors
    );
    return result;
}

inline void dijkstra_path(
    const ConstArrayView<std::uint8_t> &mask,
    const std::size_t source,
    const std::vector<std::size_t> &targets,
    DijkstraWorkspace &workspace,
    std::vector<std::size_t> &path,
    DijkstraOptions options = {},
    const ConstArrayView<double> *costs = nullptr
) {
    detail_grid_dijkstra::path_impl(
        mask, source, targets, std::move(options), costs, workspace, path,
        true, false
    );
}

inline std::vector<std::size_t> dijkstra_path(
    const ConstArrayView<std::uint8_t> &mask,
    const std::size_t source,
    const std::vector<std::size_t> &targets,
    DijkstraOptions options = {},
    const ConstArrayView<double> *costs = nullptr
) {
    DijkstraWorkspace workspace;
    std::vector<std::size_t> path;
    dijkstra_path(
        mask, source, targets, workspace, path, std::move(options), costs
    );
    return path;
}

} // namespace bioimage_cpp::distance
