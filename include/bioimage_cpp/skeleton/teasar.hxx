#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/distance/distance_transform.hxx"
#include "bioimage_cpp/distance/grid_dijkstra.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::skeleton {

struct TeasarOptions {
    std::array<double, 3> spacing{1.0, 1.0, 1.0};
    double scale = 1.5;
    double constant = 0.0;
    double pdrf_scale = 100000.0;
    double pdrf_exponent = 4.0;
};

struct SkeletonGraph {
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<std::uint64_t, 2>> edges;
    std::vector<float> radii;
};

namespace detail_teasar {

inline void validate_options(
    const ConstArrayView<std::uint8_t> &mask,
    const TeasarOptions &options
) {
    if (mask.shape.size() != 3) {
        throw std::invalid_argument(
            "mask must have ndim 3, got ndim=" + std::to_string(mask.shape.size())
        );
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (mask.shape[axis] < 0) {
            throw std::invalid_argument("mask shape entries must be non-negative");
        }
        if (!(std::isfinite(options.spacing[axis]) && options.spacing[axis] > 0.0)) {
            throw std::invalid_argument(
                "spacing values must be positive and finite, got spacing[" +
                std::to_string(axis) + "]=" + std::to_string(options.spacing[axis])
            );
        }
    }
    if (!(std::isfinite(options.scale) && options.scale >= 0.0)) {
        throw std::invalid_argument("scale must be finite and non-negative");
    }
    if (!(std::isfinite(options.constant) && options.constant >= 0.0)) {
        throw std::invalid_argument("constant must be finite and non-negative");
    }
    if (!(std::isfinite(options.pdrf_scale) && options.pdrf_scale >= 0.0)) {
        throw std::invalid_argument("pdrf_scale must be finite and non-negative");
    }
    if (!(std::isfinite(options.pdrf_exponent) && options.pdrf_exponent > 0.0)) {
        throw std::invalid_argument("pdrf_exponent must be positive and finite");
    }
}

inline std::size_t farthest_foreground(
    const std::vector<std::uint8_t> &mask,
    const std::vector<double> &distances
) {
    std::size_t farthest = std::numeric_limits<std::size_t>::max();
    double farthest_distance = -1.0;
    for (std::size_t index = 0; index < mask.size(); ++index) {
        if (mask[index] == 0 || !std::isfinite(distances[index])) {
            continue;
        }
        if (distances[index] > farthest_distance) {
            farthest = index;
            farthest_distance = distances[index];
        }
    }
    return farthest;
}

} // namespace detail_teasar

// Skeletonize a binary 3D mask with the core TEASAR procedure. A non-empty
// input must contain exactly one 26-connected foreground component.
inline SkeletonGraph teasar(
    const ConstArrayView<std::uint8_t> &mask,
    const TeasarOptions &options = {}
) {
    detail_teasar::validate_options(mask, options);
    BIOIMAGE_PROFILE_INIT(profile)

    SkeletonGraph graph;
    const auto input_n = bioimage_cpp::detail::number_of_elements(mask.shape);
    std::size_t foreground_count = 0;
    for (std::size_t index = 0; index < input_n; ++index) {
        foreground_count += mask.data[index] != 0 ? 1 : 0;
    }
    if (foreground_count == 0) {
        return graph;
    }

    // A zero halo makes the exterior an explicit background feature for EDT,
    // including when the input object touches a volume boundary.
    const std::vector<std::ptrdiff_t> shape{
        mask.shape[0] + 2,
        mask.shape[1] + 2,
        mask.shape[2] + 2,
    };
    const auto n = bioimage_cpp::detail::number_of_elements(shape);
    const auto strides = bioimage_cpp::detail::c_order_strides(shape);
    std::vector<std::uint8_t> padded_mask(n, 0);
    std::size_t first_foreground = std::numeric_limits<std::size_t>::max();
    for (std::ptrdiff_t z = 0; z < mask.shape[0]; ++z) {
        for (std::ptrdiff_t y = 0; y < mask.shape[1]; ++y) {
            for (std::ptrdiff_t x = 0; x < mask.shape[2]; ++x) {
                const auto input_index = static_cast<std::size_t>(
                    (z * mask.shape[1] + y) * mask.shape[2] + x
                );
                if (mask.data[input_index] == 0) {
                    continue;
                }
                const auto padded_index = static_cast<std::size_t>(
                    (z + 1) * strides[0] + (y + 1) * strides[1] + (x + 1)
                );
                padded_mask[padded_index] = 1;
                if (first_foreground == std::numeric_limits<std::size_t>::max()) {
                    first_foreground = padded_index;
                }
            }
        }
    }

    std::vector<float> dbf(n, 0.0f);
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "distance_transform")
        ConstArrayView<std::uint8_t> padded_view{padded_mask.data(), shape, {}};
        ArrayView<float> distances_view{dbf.data(), shape, {}};
        distance::distance_transform(
            padded_view,
            {options.spacing[0], options.spacing[1], options.spacing[2]},
            {distances_view, {}, {}},
            1
        );
    }

    const distance::DijkstraOptions physical_options{
        3,
        {options.spacing[0], options.spacing[1], options.spacing[2]},
        distance::DijkstraCostMode::Physical,
    };
    distance::DijkstraResult first_field;
    distance::DijkstraResult root_field;
    std::size_t root = first_foreground;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "root_dijkstra")
        ConstArrayView<std::uint8_t> padded_view{padded_mask.data(), shape, {}};
        first_field = distance::dijkstra_distance_field(
            padded_view, {first_foreground}, physical_options
        );
        for (std::size_t index = 0; index < n; ++index) {
            if (padded_mask[index] != 0 && !std::isfinite(first_field.distances[index])) {
                throw std::invalid_argument(
                    "mask foreground must contain exactly one 26-connected component"
                );
            }
        }
        root = detail_teasar::farthest_foreground(padded_mask, first_field.distances);
        root_field = distance::dijkstra_distance_field(
            padded_view, {root}, physical_options
        );
    }

    double dbf_max = 0.0;
    double daf_max = 0.0;
    for (std::size_t index = 0; index < n; ++index) {
        if (padded_mask[index] == 0) {
            continue;
        }
        dbf_max = std::max(dbf_max, static_cast<double>(dbf[index]));
        daf_max = std::max(daf_max, root_field.distances[index]);
    }

    std::vector<double> pdrf(n, 0.0);
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "pdrf")
        for (std::size_t index = 0; index < n; ++index) {
            if (padded_mask[index] == 0) {
                continue;
            }
            const double normalized_dbf = dbf_max > 0.0
                ? std::clamp(1.0 - static_cast<double>(dbf[index]) / dbf_max, 0.0, 1.0)
                : 0.0;
            const double normalized_daf = daf_max > 0.0
                ? root_field.distances[index] / daf_max
                : 0.0;
            pdrf[index] = options.pdrf_scale *
                std::pow(normalized_dbf, options.pdrf_exponent) + normalized_daf;
        }
    }

    std::vector<std::uint8_t> active = padded_mask;
    std::size_t active_count = foreground_count;
    std::vector<std::int64_t> vertex_of_voxel(n, -1);
    std::vector<std::size_t> skeleton_voxels;
    std::vector<std::ptrdiff_t> coords(3, 0);

    const auto add_vertex = [&](const std::size_t voxel) -> std::uint64_t {
        bioimage_cpp::detail::coords_from_index(
            static_cast<std::uint64_t>(voxel), strides, 3, coords.data()
        );
        const auto vertex_id = static_cast<std::uint64_t>(graph.vertices.size());
        graph.vertices.push_back({
            static_cast<double>(coords[0] - 1) * options.spacing[0],
            static_cast<double>(coords[1] - 1) * options.spacing[1],
            static_cast<double>(coords[2] - 1) * options.spacing[2],
        });
        graph.radii.push_back(dbf[voxel]);
        vertex_of_voxel[voxel] = static_cast<std::int64_t>(vertex_id);
        skeleton_voxels.push_back(voxel);
        pdrf[voxel] = 0.0;
        return vertex_id;
    };

    add_vertex(root);
    ConstArrayView<std::uint8_t> padded_view{padded_mask.data(), shape, {}};
    ConstArrayView<double> pdrf_view{pdrf.data(), shape, {}};
    const distance::DijkstraOptions node_options{
        3, {}, distance::DijkstraCostMode::Node
    };

    while (active_count > 0) {
        std::size_t target = std::numeric_limits<std::size_t>::max();
        double target_distance = -1.0;
        for (std::size_t index = 0; index < n; ++index) {
            if (active[index] != 0 && root_field.distances[index] > target_distance) {
                target = index;
                target_distance = root_field.distances[index];
            }
        }
        if (target == std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("TEASAR active-voxel accounting became inconsistent");
        }

        std::vector<std::size_t> path;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "path_dijkstra")
            path = distance::dijkstra_path(
                padded_view, target, skeleton_voxels, node_options, &pdrf_view
            );
        }

        std::uint64_t previous = static_cast<std::uint64_t>(
            vertex_of_voxel[path.back()]
        );
        for (auto it = path.rbegin() + 1; it != path.rend(); ++it) {
            const std::size_t voxel = *it;
            std::uint64_t current = 0;
            if (vertex_of_voxel[voxel] >= 0) {
                current = static_cast<std::uint64_t>(vertex_of_voxel[voxel]);
            } else {
                current = add_vertex(voxel);
            }
            graph.edges.push_back({previous, current});
            previous = current;
        }

        {
            BIOIMAGE_PROFILE_SCOPE(profile, "invalidation")
            for (const auto voxel : path) {
                pdrf[voxel] = 0.0;
                bioimage_cpp::detail::coords_from_index(
                    static_cast<std::uint64_t>(voxel), strides, 3, coords.data()
                );
                const double radius =
                    options.scale * static_cast<double>(dbf[voxel]) + options.constant;
                if (!std::isfinite(radius)) {
                    throw std::runtime_error("TEASAR invalidation radius overflowed");
                }
                std::array<std::ptrdiff_t, 3> lo{};
                std::array<std::ptrdiff_t, 3> hi{};
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    const double half_width = radius / options.spacing[axis];
                    const double lo_value = std::max(
                        0.0, static_cast<double>(coords[axis]) - half_width
                    );
                    const double hi_value = std::min(
                        static_cast<double>(shape[axis] - 1),
                        static_cast<double>(coords[axis]) + half_width
                    );
                    lo[axis] = static_cast<std::ptrdiff_t>(std::ceil(lo_value));
                    hi[axis] = static_cast<std::ptrdiff_t>(std::floor(hi_value));
                }
                for (std::ptrdiff_t z = lo[0]; z <= hi[0]; ++z) {
                    for (std::ptrdiff_t y = lo[1]; y <= hi[1]; ++y) {
                        for (std::ptrdiff_t x = lo[2]; x <= hi[2]; ++x) {
                            const auto index = static_cast<std::size_t>(
                                z * strides[0] + y * strides[1] + x
                            );
                            if (active[index] != 0) {
                                active[index] = 0;
                                --active_count;
                            }
                        }
                    }
                }
            }
        }
    }

    BIOIMAGE_PROFILE_REPORT(profile)
    return graph;
}

} // namespace bioimage_cpp::skeleton
