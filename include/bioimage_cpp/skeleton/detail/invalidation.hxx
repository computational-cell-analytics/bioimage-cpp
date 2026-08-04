#pragma once

#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/skeleton/detail/compact_grid_dijkstra.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <vector>

namespace bioimage_cpp::skeleton::detail {

struct BallInvalidationEntry {
    double distance = 0.0;
    std::size_t source = 0;
    std::size_t voxel = 0;
};

struct BallInvalidationGreater {
    bool operator()(
        const BallInvalidationEntry &first,
        const BallInvalidationEntry &second
    ) const noexcept {
        if (first.distance != second.distance) {
            return first.distance > second.distance;
        }
        if (first.source != second.source) {
            return first.source > second.source;
        }
        return first.voxel > second.voxel;
    }
};

struct CompactBallInvalidationEntry {
    double distance = 0.0;
    std::uint32_t source = 0;
    std::uint32_t node = 0;
};

struct CompactBallInvalidationGreater {
    bool operator()(
        const CompactBallInvalidationEntry &first,
        const CompactBallInvalidationEntry &second
    ) const noexcept {
        if (first.distance != second.distance) {
            return first.distance > second.distance;
        }
        if (first.source != second.source) {
            return first.source > second.source;
        }
        return first.node > second.node;
    }
};

struct CompactBallInvalidationStats {
    std::size_t offers = 0;
    std::size_t accepted = 0;
    std::size_t pops = 0;
    std::size_t stale_pops = 0;
    std::size_t invalidated = 0;
    std::size_t peak_heap = 0;

    void reset() noexcept {
        *this = {};
    }

    void merge(const CompactBallInvalidationStats &other) noexcept {
        offers += other.offers;
        accepted += other.accepted;
        pops += other.pops;
        stale_pops += other.stale_pops;
        invalidated += other.invalidated;
        peak_heap = std::max(peak_heap, other.peak_heap);
    }
};

struct CompactBallInvalidationWorkspace {
    std::vector<double> pending_distance;
    std::vector<std::uint32_t> pending_source;
    std::vector<std::uint32_t> pending_generation;
    std::vector<CompactBallInvalidationEntry> heap;
    std::vector<std::array<std::ptrdiff_t, 3>> source_coordinates;
    std::uint32_t generation = 0;

    void begin(const std::size_t number_of_nodes, const std::size_t number_of_sources) {
        if (pending_generation.size() != number_of_nodes) {
            pending_distance.resize(number_of_nodes);
            pending_source.resize(number_of_nodes);
            pending_generation.assign(number_of_nodes, 0);
            generation = 0;
        }
        if (generation == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(pending_generation.begin(), pending_generation.end(), 0);
            generation = 1;
        } else {
            ++generation;
        }
        heap.clear();
        source_coordinates.resize(number_of_sources);
    }
};

inline void compact_ball_heap_push(
    CompactBallInvalidationWorkspace &workspace,
    const CompactBallInvalidationEntry entry,
    CompactBallInvalidationStats *stats
) {
    workspace.heap.push_back(entry);
    std::push_heap(
        workspace.heap.begin(), workspace.heap.end(),
        CompactBallInvalidationGreater{}
    );
    if (stats != nullptr) {
        ++stats->accepted;
        stats->peak_heap = std::max(stats->peak_heap, workspace.heap.size());
    }
}

inline CompactBallInvalidationEntry compact_ball_heap_pop(
    CompactBallInvalidationWorkspace &workspace,
    CompactBallInvalidationStats *stats
) {
    std::pop_heap(
        workspace.heap.begin(), workspace.heap.end(),
        CompactBallInvalidationGreater{}
    );
    const auto entry = workspace.heap.back();
    workspace.heap.pop_back();
    if (stats != nullptr) {
        ++stats->pops;
    }
    return entry;
}

// Invalidate strict physical-radius balls in compact foreground space. Keep
// only the best discovered entry per active node. A displaced heap entry is
// stale and cannot affect traversal because the better pending entry sorts
// before it.
template <CompactAdjacency Adjacency>
inline std::size_t invalidate_compact_path_balls(
    std::vector<std::uint8_t> &active,
    const std::span<const std::uint32_t> path,
    const std::span<const double> radii,
    const CompactGridDomain &domain,
    const std::array<double, 3> &spacing,
    CompactBallInvalidationWorkspace &workspace,
    CompactBallInvalidationStats *stats = nullptr
) {
    if (domain.shape.size() != 3) {
        throw std::invalid_argument("ball invalidation requires a 3D domain");
    }
    if (path.size() != radii.size()) {
        throw std::invalid_argument("ball invalidation path and radii must match");
    }
    if (active.size() != domain.size()) {
        throw std::invalid_argument("ball invalidation active domain mismatch");
    }
    if constexpr (Adjacency == CompactAdjacency::Csr) {
        if (!domain.has_csr()) {
            throw std::invalid_argument("compact CSR adjacency is not available");
        }
    } else if (!domain.has_full_lookup()) {
        throw std::invalid_argument("compact full-index lookup is not available");
    }
    if (path.size() > static_cast<std::size_t>(kNoCompactNode)) {
        throw std::invalid_argument("ball invalidation path exceeds uint32 range");
    }

    workspace.begin(domain.size(), path.size());
    for (std::size_t source = 0; source < path.size(); ++source) {
        if (path[source] >= domain.size()) {
            throw std::invalid_argument("ball invalidation path node is out of bounds");
        }
        bioimage_cpp::detail::coords_from_index(
            domain.compact_to_full[path[source]], domain.strides, 3,
            workspace.source_coordinates[source].data()
        );
    }

    const auto offer = [&](const std::uint32_t node,
                           const double distance,
                           const std::uint32_t source) {
        if (stats != nullptr) {
            ++stats->offers;
        }
        if (active[node] == 0) {
            return;
        }
        const bool pending =
            workspace.pending_generation[node] == workspace.generation;
        if (
            pending &&
            !(distance < workspace.pending_distance[node] ||
              (distance == workspace.pending_distance[node] &&
               source < workspace.pending_source[node]))
        ) {
            return;
        }
        workspace.pending_distance[node] = distance;
        workspace.pending_source[node] = source;
        workspace.pending_generation[node] = workspace.generation;
        compact_ball_heap_push(workspace, {distance, source, node}, stats);
    };

    for (std::uint32_t source = 0; source < path.size(); ++source) {
        offer(path[source], 0.0, source);
    }

    std::size_t invalidated = 0;
    std::array<std::ptrdiff_t, 3> coordinate{};
    while (!workspace.heap.empty()) {
        const auto entry = compact_ball_heap_pop(workspace, stats);
        const bool current =
            workspace.pending_generation[entry.node] == workspace.generation &&
            workspace.pending_distance[entry.node] == entry.distance &&
            workspace.pending_source[entry.node] == entry.source;
        if (!current || active[entry.node] == 0) {
            if (stats != nullptr) {
                ++stats->stale_pops;
            }
            continue;
        }

        active[entry.node] = 0;
        ++invalidated;
        if (stats != nullptr) {
            ++stats->invalidated;
        }
        bioimage_cpp::detail::coords_from_index(
            domain.compact_to_full[entry.node], domain.strides, 3,
            coordinate.data()
        );
        for_each_compact_neighbor_with_metadata<Adjacency>(
            domain, entry.node,
            [&](const std::uint32_t target, const CompactNeighbor &neighbor) {
                if (active[target] == 0) {
                    return;
                }
                double distance_squared = 0.0;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    const auto neighbor_coordinate =
                        coordinate[axis] + neighbor.coordinate_delta[axis];
                    const auto delta = static_cast<double>(
                        neighbor_coordinate -
                        workspace.source_coordinates[entry.source][axis]
                    ) * spacing[axis];
                    distance_squared += delta * delta;
                }
                const double distance = std::sqrt(distance_squared);
                if (distance < radii[entry.source]) {
                    offer(target, distance, entry.source);
                }
            }
        );
    }
    return invalidated;
}

// Invalidate the active, 26-connected part of each strict physical-radius ball.
// Inactive voxels are barriers. Active source voxels are always invalidated.
inline std::size_t invalidate_path_balls(
    std::vector<std::uint8_t> &active,
    const std::span<const std::size_t> path,
    const std::span<const double> radii,
    const std::vector<std::ptrdiff_t> &shape,
    const std::array<double, 3> &spacing
) {
    if (shape.size() != 3) {
        throw std::invalid_argument("ball invalidation requires a 3D shape");
    }
    if (path.size() != radii.size()) {
        throw std::invalid_argument("ball invalidation path and radii must match");
    }
    if (active.size() != bioimage_cpp::detail::number_of_elements(shape)) {
        throw std::invalid_argument("ball invalidation active mask shape mismatch");
    }

    const auto strides = bioimage_cpp::detail::c_order_strides(shape);
    std::vector<std::array<std::ptrdiff_t, 3>> source_coordinates(path.size());
    std::priority_queue<
        BallInvalidationEntry,
        std::vector<BallInvalidationEntry>,
        BallInvalidationGreater
    > queue;
    for (std::size_t source = 0; source < path.size(); ++source) {
        if (path[source] >= active.size()) {
            throw std::invalid_argument("ball invalidation path voxel is out of bounds");
        }
        bioimage_cpp::detail::coords_from_index(
            static_cast<std::uint64_t>(path[source]),
            strides,
            3,
            source_coordinates[source].data()
        );
        queue.push({0.0, source, path[source]});
    }

    std::size_t invalidated = 0;
    std::array<std::ptrdiff_t, 3> coordinate{};
    while (!queue.empty()) {
        const auto entry = queue.top();
        queue.pop();
        if (active[entry.voxel] == 0) {
            continue;
        }

        active[entry.voxel] = 0;
        ++invalidated;
        bioimage_cpp::detail::coords_from_index(
            static_cast<std::uint64_t>(entry.voxel),
            strides,
            3,
            coordinate.data()
        );
        for (std::ptrdiff_t dz = -1; dz <= 1; ++dz) {
            for (std::ptrdiff_t dy = -1; dy <= 1; ++dy) {
                for (std::ptrdiff_t dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) {
                        continue;
                    }
                    const std::array<std::ptrdiff_t, 3> neighbor_coordinate{
                        coordinate[0] + dz,
                        coordinate[1] + dy,
                        coordinate[2] + dx,
                    };
                    if (
                        neighbor_coordinate[0] < 0 ||
                        neighbor_coordinate[1] < 0 ||
                        neighbor_coordinate[2] < 0 ||
                        neighbor_coordinate[0] >= shape[0] ||
                        neighbor_coordinate[1] >= shape[1] ||
                        neighbor_coordinate[2] >= shape[2]
                    ) {
                        continue;
                    }
                    const auto neighbor = static_cast<std::size_t>(
                        neighbor_coordinate[0] * strides[0] +
                        neighbor_coordinate[1] * strides[1] +
                        neighbor_coordinate[2]
                    );
                    if (active[neighbor] == 0) {
                        continue;
                    }

                    double distance_squared = 0.0;
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        const auto delta = static_cast<double>(
                            neighbor_coordinate[axis] -
                            source_coordinates[entry.source][axis]
                        ) * spacing[axis];
                        distance_squared += delta * delta;
                    }
                    const double distance = std::sqrt(distance_squared);
                    if (distance < radii[entry.source]) {
                        queue.push({distance, entry.source, neighbor});
                    }
                }
            }
        }
    }
    return invalidated;
}

} // namespace bioimage_cpp::skeleton::detail
