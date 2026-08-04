#pragma once

#include "bioimage_cpp/detail/grid.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
