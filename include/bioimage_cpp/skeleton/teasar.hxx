#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/distance/distance_transform.hxx"
#include "bioimage_cpp/distance/grid_dijkstra.hxx"
#include "bioimage_cpp/skeleton/detail/compact_grid_dijkstra.hxx"
#include "bioimage_cpp/skeleton/detail/components.hxx"
#include "bioimage_cpp/skeleton/detail/invalidation.hxx"
#include "bioimage_cpp/skeleton/detail/row_interval_union.hxx"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace bioimage_cpp::skeleton {

enum class TeasarInvalidation {
    Cube,
    Ball,
};

struct TeasarOptions {
    std::array<double, 3> spacing{1.0, 1.0, 1.0};
    double scale = 1.5;
    double constant = 0.0;
    double pdrf_scale = 100000.0;
    double pdrf_exponent = 4.0;
    std::size_t number_of_threads = 1;
    TeasarInvalidation invalidation = TeasarInvalidation::Cube;
    bool fix_branching = true;
};

using VoxelCoordinate = std::array<std::int64_t, 3>;

struct SkeletonGraph {
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<std::uint64_t, 2>> edges;
    std::vector<float> radii;
};

struct LatticeSkeletonGraph {
    std::vector<VoxelCoordinate> vertices;
    std::vector<std::array<std::uint64_t, 2>> edges;
    std::vector<float> radii;
};

template <class LabelT>
struct LabeledVoxelTarget {
    LabelT label{};
    VoxelCoordinate coordinate{};
};

// Kept public at the C++ level so development benchmarks can compare the
// sequential implementations. The Python API always uses TeasarBackend::Auto.
enum class TeasarBackend {
    Auto,
    DenseFloat64,
    CompactOnTheFlyFloat64,
    CompactCsrFloat64,
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
    if (
        options.invalidation != TeasarInvalidation::Cube &&
        options.invalidation != TeasarInvalidation::Ball
    ) {
        throw std::invalid_argument("invalid TEASAR invalidation mode");
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

inline void invalidation_bounds(
    const std::vector<std::ptrdiff_t> &coords,
    const double radius,
    const std::array<double, 3> &spacing,
    const std::vector<std::ptrdiff_t> &shape,
    std::array<std::ptrdiff_t, 3> &lo,
    std::array<std::ptrdiff_t, 3> &hi
) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const double half_width = radius / spacing[axis];
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
}

} // namespace detail_teasar

// Skeletonize a binary 3D mask with the core TEASAR procedure. A non-empty
// input must contain exactly one 26-connected foreground component.
inline LatticeSkeletonGraph teasar_dense_impl(
    const ConstArrayView<std::uint8_t> &mask,
    detail::PreparedTeasarComponent *prepared,
    const TeasarOptions &options,
    const bool report_profile
) {
    if (prepared == nullptr) {
        detail_teasar::validate_options(mask, options);
    }
    BIOIMAGE_PROFILE_INIT(profile)

    LatticeSkeletonGraph graph;
    std::size_t foreground_count = 0;
    std::array<std::ptrdiff_t, 3> input_origin{0, 0, 0};
    std::vector<std::ptrdiff_t> shape;
    std::vector<std::uint8_t> padded_mask;
    std::vector<std::uint8_t> distance_mask;
    std::vector<std::size_t> required_targets;
    std::size_t required_root = std::numeric_limits<std::size_t>::max();
    std::size_t first_foreground = std::numeric_limits<std::size_t>::max();
    if (prepared != nullptr) {
        foreground_count = prepared->foreground_count;
        input_origin = prepared->input_origin;
        shape = std::move(prepared->padded_shape);
        padded_mask = std::move(prepared->padded_mask);
        distance_mask = std::move(prepared->distance_mask);
        required_targets = std::move(prepared->required_target_voxels);
        required_root = prepared->required_root_voxel;
        for (std::size_t index = 0; index < padded_mask.size(); ++index) {
            if (padded_mask[index] != 0) {
                first_foreground = index;
                break;
            }
        }
    } else {
        const auto input_n = bioimage_cpp::detail::number_of_elements(mask.shape);
        for (std::size_t index = 0; index < input_n; ++index) {
            foreground_count += mask.data[index] != 0 ? 1 : 0;
        }
        shape = {
            mask.shape[0] + 2,
            mask.shape[1] + 2,
            mask.shape[2] + 2,
        };
        const auto local_strides = bioimage_cpp::detail::c_order_strides(shape);
        padded_mask.assign(
            bioimage_cpp::detail::number_of_elements(shape), 0
        );
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
                        (z + 1) * local_strides[0] +
                        (y + 1) * local_strides[1] + (x + 1)
                    );
                    padded_mask[padded_index] = 1;
                    if (first_foreground == std::numeric_limits<std::size_t>::max()) {
                        first_foreground = padded_index;
                    }
                }
            }
        }
    }
    if (foreground_count == 0) {
        return graph;
    }
    if (first_foreground == std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("prepared TEASAR foreground count is inconsistent");
    }
    const auto effective_threads = bioimage_cpp::detail::normalize_thread_count(
        options.number_of_threads, foreground_count
    );
    const auto n = padded_mask.size();
    const auto strides = bioimage_cpp::detail::c_order_strides(shape);

    std::vector<float> dbf(n, 0.0f);
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "distance_transform")
        const auto &distance_input = distance_mask.empty()
            ? padded_mask : distance_mask;
        ConstArrayView<std::uint8_t> padded_view{
            distance_input.data(), shape, {}
        };
        ArrayView<float> distances_view{dbf.data(), shape, {}};
        distance::distance_transform(
            padded_view,
            {options.spacing[0], options.spacing[1], options.spacing[2]},
            {distances_view, {}, {}},
            effective_threads
        );
    }

    const distance::DijkstraOptions physical_options{
        3,
        {options.spacing[0], options.spacing[1], options.spacing[2]},
        distance::DijkstraCostMode::Physical,
        effective_threads,
    };
    distance::DijkstraResult root_field;
    std::size_t root = first_foreground;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "root_dijkstra")
        ConstArrayView<std::uint8_t> padded_view{padded_mask.data(), shape, {}};
        if (required_root != std::numeric_limits<std::size_t>::max()) {
            if (required_root >= n || padded_mask[required_root] == 0) {
                throw std::runtime_error(
                    "prepared required root is not foreground"
                );
            }
            root = required_root;
        } else {
            auto first_field = distance::dijkstra_distance_field(
                padded_view, {first_foreground}, physical_options
            );
            for (std::size_t index = 0; index < n; ++index) {
                if (
                    padded_mask[index] != 0 &&
                    !std::isfinite(first_field.distances[index])
                ) {
                    throw std::invalid_argument(
                        "mask foreground must contain exactly one 26-connected component"
                    );
                }
            }
            root = detail_teasar::farthest_foreground(
                padded_mask, first_field.distances
            );
        }
        root_field = distance::dijkstra_distance_field(
            padded_view, {root}, physical_options
        );
        for (std::size_t index = 0; index < n; ++index) {
            if (
                padded_mask[index] != 0 &&
                !std::isfinite(root_field.distances[index])
            ) {
                throw std::invalid_argument(
                    "mask foreground must contain exactly one 26-connected component"
                );
            }
        }
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
            static_cast<std::int64_t>(coords[0] - 1 + input_origin[0]),
            static_cast<std::int64_t>(coords[1] - 1 + input_origin[1]),
            static_cast<std::int64_t>(coords[2] - 1 + input_origin[2]),
        });
        graph.radii.push_back(dbf[voxel]);
        vertex_of_voxel[voxel] = static_cast<std::int64_t>(vertex_id);
        skeleton_voxels.push_back(voxel);
        if (options.fix_branching) {
            pdrf[voxel] = 0.0;
        }
        return vertex_id;
    };

    ConstArrayView<std::uint8_t> padded_view{padded_mask.data(), shape, {}};
    ConstArrayView<double> pdrf_view{pdrf.data(), shape, {}};
    const distance::DijkstraOptions node_options{
        3, {}, distance::DijkstraCostMode::Node, effective_threads
    };
    std::vector<std::int64_t> fixed_predecessors;
    if (!options.fix_branching) {
        BIOIMAGE_PROFILE_SCOPE(profile, "parental_field")
        auto field = distance::dijkstra_distance_field(
            padded_view, {root}, node_options, &pdrf_view, true
        );
        fixed_predecessors = std::move(field.predecessors);
    }

    add_vertex(root);
    std::vector<std::size_t> path;
    const auto trace_target = [&](const std::size_t target) {
        if (options.fix_branching) {
            BIOIMAGE_PROFILE_SCOPE(profile, "path_dijkstra")
            path = distance::dijkstra_path(
                padded_view, target, skeleton_voxels, node_options, &pdrf_view
            );
        } else {
            BIOIMAGE_PROFILE_SCOPE(profile, "path_from_parents")
            path.clear();
            auto voxel = target;
            while (vertex_of_voxel[voxel] < 0) {
                path.push_back(voxel);
                const auto predecessor = fixed_predecessors[voxel];
                if (predecessor < 0) {
                    throw std::runtime_error(
                        "invalid fixed-parent predecessor chain"
                    );
                }
                voxel = static_cast<std::size_t>(predecessor);
                if (path.size() > n) {
                    throw std::runtime_error("cycle in fixed-parent predecessor chain");
                }
            }
            path.push_back(voxel);
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
            if (options.invalidation == TeasarInvalidation::Ball) {
                std::vector<double> radii;
                radii.reserve(path.size());
                for (const auto voxel : path) {
                    const double radius =
                        options.scale * static_cast<double>(dbf[voxel]) +
                        options.constant;
                    if (!std::isfinite(radius)) {
                        throw std::runtime_error("TEASAR invalidation radius overflowed");
                    }
                    radii.push_back(radius);
                }
                const auto invalidated = detail::invalidate_path_balls(
                    active, path, radii, shape, options.spacing
                );
                if (invalidated > active_count) {
                    throw std::runtime_error(
                        "TEASAR ball invalidation count is inconsistent"
                    );
                }
                active_count -= invalidated;
            } else {
                for (const auto voxel : path) {
                    bioimage_cpp::detail::coords_from_index(
                        static_cast<std::uint64_t>(voxel), strides, 3, coords.data()
                    );
                    const double radius =
                        options.scale * static_cast<double>(dbf[voxel]) +
                        options.constant;
                    if (!std::isfinite(radius)) {
                        throw std::runtime_error("TEASAR invalidation radius overflowed");
                    }
                    std::array<std::ptrdiff_t, 3> lo{};
                    std::array<std::ptrdiff_t, 3> hi{};
                    detail_teasar::invalidation_bounds(
                        coords, radius, options.spacing, shape, lo, hi
                    );
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
            if (options.fix_branching) {
                for (const auto voxel : path) {
                    pdrf[voxel] = 0.0;
                }
            }
        }
    };

    for (const auto target : required_targets) {
        if (target >= n || padded_mask[target] == 0) {
            throw std::runtime_error("prepared required target is not foreground");
        }
    }
    std::sort(
        required_targets.begin(), required_targets.end(),
        [&](const std::size_t first, const std::size_t second) {
            if (root_field.distances[first] != root_field.distances[second]) {
                return root_field.distances[first] > root_field.distances[second];
            }
            return first < second;
        }
    );
    for (const auto target : required_targets) {
        if (vertex_of_voxel[target] < 0) {
            trace_target(target);
        }
    }

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
        trace_target(target);
    }

    if (report_profile) {
        BIOIMAGE_PROFILE_REPORT(profile)
    }
    return graph;
}

inline LatticeSkeletonGraph teasar_dense(
    const ConstArrayView<std::uint8_t> &mask,
    const TeasarOptions &options = {}
) {
    return teasar_dense_impl(mask, nullptr, options, true);
}

inline LatticeSkeletonGraph teasar_dense_prepared(
    detail::PreparedTeasarComponent prepared,
    const TeasarOptions &options
) {
    const ConstArrayView<std::uint8_t> unused{};
    return teasar_dense_impl(
        unused, &prepared, options, false
    );
}

template <detail::CompactAdjacency Adjacency, class Distance>
inline LatticeSkeletonGraph teasar_compact_impl(
    const ConstArrayView<std::uint8_t> &mask,
    detail::PreparedTeasarComponent *prepared,
    const TeasarOptions &options,
    const bool report_profile,
    detail::CompactBallInvalidationStats *ball_invalidation_stats = nullptr,
    bioimage_cpp::detail::ActiveProfiler *component_profile = nullptr
) {
    if (prepared == nullptr) {
        detail_teasar::validate_options(mask, options);
    }
    BIOIMAGE_PROFILE_INIT(profile)

    LatticeSkeletonGraph graph;
    std::array<std::ptrdiff_t, 3> crop_begin{0, 0, 0};
    std::array<std::ptrdiff_t, 3> crop_end{0, 0, 0};
    std::size_t foreground_count = 0;
    std::vector<std::ptrdiff_t> shape;
    std::vector<std::ptrdiff_t> strides;
    std::size_t n = 0;
    std::vector<std::uint8_t> padded_mask;
    std::vector<std::uint8_t> distance_mask;
    std::vector<float> compact_dbf;
    std::vector<std::size_t> required_targets;
    std::size_t required_root = std::numeric_limits<std::size_t>::max();
    if (prepared != nullptr) {
        crop_begin = prepared->input_origin;
        foreground_count = prepared->foreground_count;
        shape = std::move(prepared->padded_shape);
        padded_mask = std::move(prepared->padded_mask);
        distance_mask = std::move(prepared->distance_mask);
        compact_dbf = std::move(prepared->compact_dbf);
        required_targets = std::move(prepared->required_target_voxels);
        required_root = prepared->required_root_voxel;
        n = padded_mask.size();
        strides = bioimage_cpp::detail::c_order_strides(shape);
        if (foreground_count == 0) {
            return graph;
        }
    } else {
        BIOIMAGE_PROFILE_SCOPE(profile, "input_crop")
        crop_begin = {mask.shape[0], mask.shape[1], mask.shape[2]};
        for (std::ptrdiff_t z = 0; z < mask.shape[0]; ++z) {
            for (std::ptrdiff_t y = 0; y < mask.shape[1]; ++y) {
                for (std::ptrdiff_t x = 0; x < mask.shape[2]; ++x) {
                    const auto input_index = static_cast<std::size_t>(
                        (z * mask.shape[1] + y) * mask.shape[2] + x
                    );
                    if (mask.data[input_index] == 0) {
                        continue;
                    }
                    ++foreground_count;
                    crop_begin[0] = std::min(crop_begin[0], z);
                    crop_begin[1] = std::min(crop_begin[1], y);
                    crop_begin[2] = std::min(crop_begin[2], x);
                    crop_end[0] = std::max(crop_end[0], z + 1);
                    crop_end[1] = std::max(crop_end[1], y + 1);
                    crop_end[2] = std::max(crop_end[2], x + 1);
                }
            }
        }
        if (foreground_count == 0) {
            return graph;
        }

        shape = {
            crop_end[0] - crop_begin[0] + 2,
            crop_end[1] - crop_begin[1] + 2,
            crop_end[2] - crop_begin[2] + 2,
        };
        n = bioimage_cpp::detail::number_of_elements(shape);
        strides = bioimage_cpp::detail::c_order_strides(shape);
        padded_mask.assign(n, 0);
        for (std::ptrdiff_t z = crop_begin[0]; z < crop_end[0]; ++z) {
            for (std::ptrdiff_t y = crop_begin[1]; y < crop_end[1]; ++y) {
                for (std::ptrdiff_t x = crop_begin[2]; x < crop_end[2]; ++x) {
                    const auto input_index = static_cast<std::size_t>(
                        (z * mask.shape[1] + y) * mask.shape[2] + x
                    );
                    if (mask.data[input_index] == 0) {
                        continue;
                    }
                    const auto padded_index = static_cast<std::size_t>(
                        (z - crop_begin[0] + 1) * strides[0] +
                        (y - crop_begin[1] + 1) * strides[1] +
                        (x - crop_begin[2] + 1)
                    );
                    padded_mask[padded_index] = 1;
                }
            }
        }
    }
    const auto effective_threads = bioimage_cpp::detail::normalize_thread_count(
        options.number_of_threads, foreground_count
    );

    std::unique_ptr<float[]> dbf;
    if (compact_dbf.empty()) {
        dbf = std::make_unique_for_overwrite<float[]>(n);
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "distance_transform")
            const auto &distance_input = distance_mask.empty()
                ? padded_mask : distance_mask;
            ConstArrayView<std::uint8_t> padded_view{
                distance_input.data(), shape, {}
            };
            ArrayView<float> distances_view{dbf.get(), shape, {}};
            distance::distance_transform(
                padded_view,
                {options.spacing[0], options.spacing[1], options.spacing[2]},
                {distances_view, {}, {}},
                effective_threads
            );
        }
    }

    detail::CompactGridDomain domain;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "compact_domain")
        ConstArrayView<std::uint8_t> padded_view{padded_mask.data(), shape, {}};
        if (!detail::build_compact_grid_domain(
                padded_view, options.spacing, Adjacency, domain)) {
            if (prepared == nullptr) {
                return teasar_dense(mask, options);
            }
            detail::PreparedTeasarComponent dense_prepared;
            dense_prepared.padded_shape = std::move(shape);
            dense_prepared.padded_mask = std::move(padded_mask);
            dense_prepared.distance_mask = std::move(distance_mask);
            dense_prepared.input_origin = crop_begin;
            dense_prepared.foreground_count = foreground_count;
            dense_prepared.required_target_voxels = std::move(required_targets);
            dense_prepared.required_root_voxel = required_root;
            return teasar_dense_prepared(
                std::move(dense_prepared), options
            );
        }
    }
    if (domain.size() != foreground_count) {
        throw std::runtime_error("TEASAR compact foreground count is inconsistent");
    }

    double dbf_max = 0.0;
    if (compact_dbf.empty()) {
        compact_dbf.reserve(domain.size());
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "dbf_compaction")
            for (std::uint32_t node = 0; node < domain.size(); ++node) {
                const auto value = dbf[domain.compact_to_full[node]];
                compact_dbf.push_back(value);
                dbf_max = std::max(dbf_max, static_cast<double>(value));
            }
            dbf.reset();
        }
    } else {
        if (compact_dbf.size() != domain.size()) {
            throw std::runtime_error(
                "TEASAR precomputed distance count is inconsistent"
            );
        }
        for (const auto value : compact_dbf) {
            dbf_max = std::max(dbf_max, static_cast<double>(value));
        }
    }

    detail::CompactDijkstraWorkspace<Distance> dijkstra_workspace;
    std::vector<Distance> root_field;
    std::uint32_t root = 0;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "root_dijkstra")
        if (required_root != std::numeric_limits<std::size_t>::max()) {
            root = domain.compact_node_from_full(required_root);
            if (root == detail::kNoCompactNode) {
                throw std::runtime_error(
                    "prepared required root is not foreground"
                );
            }
        } else {
            std::vector<Distance> first_field;
            detail::compact_physical_distance_field<Adjacency>(
                domain, 0, dijkstra_workspace, first_field
            );
            Distance farthest_distance = Distance{-1};
            for (std::uint32_t node = 0; node < domain.size(); ++node) {
                if (!std::isfinite(first_field[node])) {
                    throw std::invalid_argument(
                        "mask foreground must contain exactly one 26-connected component"
                    );
                }
                if (first_field[node] > farthest_distance) {
                    root = node;
                    farthest_distance = first_field[node];
                }
            }
        }
        detail::compact_physical_distance_field<Adjacency>(
            domain, root, dijkstra_workspace, root_field
        );
        for (const auto distance : root_field) {
            if (!std::isfinite(distance)) {
                throw std::invalid_argument(
                    "mask foreground must contain exactly one 26-connected component"
                );
            }
        }
    }

    Distance daf_max = Distance{0};
    for (std::uint32_t node = 0; node < domain.size(); ++node) {
        daf_max = std::max(daf_max, root_field[node]);
    }

    std::vector<Distance> pdrf(domain.size(), Distance{0});
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "pdrf")
        for (std::uint32_t node = 0; node < domain.size(); ++node) {
            const double normalized_dbf = dbf_max > 0.0
                ? std::clamp(
                    1.0 - static_cast<double>(compact_dbf[node]) / dbf_max,
                    0.0,
                    1.0
                )
                : 0.0;
            const double normalized_daf = daf_max > Distance{0}
                ? static_cast<double>(root_field[node] / daf_max)
                : 0.0;
            pdrf[node] = static_cast<Distance>(
                options.pdrf_scale *
                    std::pow(normalized_dbf, options.pdrf_exponent) +
                normalized_daf
            );
        }
    }

    std::vector<std::uint8_t> active;
    if (options.invalidation == TeasarInvalidation::Ball) {
        active.assign(domain.size(), std::uint8_t{1});
        std::vector<std::uint8_t>().swap(padded_mask);
    } else {
        active = std::move(padded_mask);
    }
    std::size_t active_count = foreground_count;
    std::vector<std::int64_t> vertex_of_node(domain.size(), -1);
    std::vector<std::uint32_t> skeleton_nodes;
    std::vector<std::ptrdiff_t> coords(3, 0);

    const auto add_vertex = [&](const std::uint32_t node) -> std::uint64_t {
        const auto voxel = static_cast<std::size_t>(domain.compact_to_full[node]);
        bioimage_cpp::detail::coords_from_index(
            static_cast<std::uint64_t>(voxel), strides, 3, coords.data()
        );
        const auto vertex_id = static_cast<std::uint64_t>(graph.vertices.size());
        graph.vertices.push_back({
            static_cast<std::int64_t>(coords[0] - 1 + crop_begin[0]),
            static_cast<std::int64_t>(coords[1] - 1 + crop_begin[1]),
            static_cast<std::int64_t>(coords[2] - 1 + crop_begin[2]),
        });
        graph.radii.push_back(compact_dbf[node]);
        vertex_of_node[node] = static_cast<std::int64_t>(vertex_id);
        skeleton_nodes.push_back(node);
        if (options.fix_branching) {
            pdrf[node] = Distance{0};
        }
        return vertex_id;
    };

    std::vector<std::uint32_t> fixed_predecessors;
    if (!options.fix_branching) {
        BIOIMAGE_PROFILE_SCOPE(profile, "parental_field")
        detail::compact_node_cost_parental_field<Adjacency>(
            domain, root, pdrf, dijkstra_workspace, fixed_predecessors
        );
    }

    add_vertex(root);
    std::vector<std::uint32_t> path;
    std::vector<double> ball_radii;
    detail::CompactBallInvalidationWorkspace ball_invalidation_workspace;
    detail::RowIntervalUnion invalidated_rows(
        n / static_cast<std::size_t>(shape[2]), shape[2]
    );
    const auto trace_target = [&](const std::uint32_t target) {
        if (options.fix_branching) {
            BIOIMAGE_PROFILE_SCOPE(profile, "path_dijkstra")
            detail::compact_node_cost_path<Adjacency>(
                domain, target, skeleton_nodes, pdrf, dijkstra_workspace, path
            );
        } else {
            BIOIMAGE_PROFILE_SCOPE(profile, "path_from_parents")
            path.clear();
            auto node = target;
            while (vertex_of_node[node] < 0) {
                path.push_back(node);
                node = fixed_predecessors[node];
                if (path.size() > domain.size()) {
                    throw std::runtime_error("cycle in fixed-parent predecessor chain");
                }
            }
            path.push_back(node);
        }

        std::uint64_t previous = static_cast<std::uint64_t>(
            vertex_of_node[path.back()]
        );
        for (auto it = path.rbegin() + 1; it != path.rend(); ++it) {
            const auto node = *it;
            std::uint64_t current = 0;
            if (vertex_of_node[node] >= 0) {
                current = static_cast<std::uint64_t>(vertex_of_node[node]);
            } else {
                current = add_vertex(node);
            }
            graph.edges.push_back({previous, current});
            previous = current;
        }

        {
            BIOIMAGE_PROFILE_SCOPE(profile, "invalidation")
            if (options.invalidation == TeasarInvalidation::Ball) {
                ball_radii.clear();
                ball_radii.reserve(path.size());
                for (const auto node : path) {
                    const double radius =
                        options.scale * static_cast<double>(compact_dbf[node]) +
                        options.constant;
                    if (!std::isfinite(radius)) {
                        throw std::runtime_error("TEASAR invalidation radius overflowed");
                    }
                    ball_radii.push_back(radius);
                }
                const auto invalidated = detail::invalidate_compact_path_balls<Adjacency>(
                    active, path, ball_radii, domain, options.spacing,
                    ball_invalidation_workspace, ball_invalidation_stats
                );
                if (invalidated > active_count) {
                    throw std::runtime_error(
                        "TEASAR ball invalidation count is inconsistent"
                    );
                }
                active_count -= invalidated;
            } else {
                for (const auto node : path) {
                    const auto voxel = static_cast<std::size_t>(
                        domain.compact_to_full[node]
                    );
                    bioimage_cpp::detail::coords_from_index(
                        static_cast<std::uint64_t>(voxel), strides, 3, coords.data()
                    );
                    const double radius =
                        options.scale * static_cast<double>(compact_dbf[node]) +
                        options.constant;
                    if (!std::isfinite(radius)) {
                        throw std::runtime_error("TEASAR invalidation radius overflowed");
                    }
                    std::array<std::ptrdiff_t, 3> lo{};
                    std::array<std::ptrdiff_t, 3> hi{};
                    detail_teasar::invalidation_bounds(
                        coords, radius, options.spacing, shape, lo, hi
                    );
                    for (std::ptrdiff_t z = lo[0]; z <= hi[0]; ++z) {
                        for (std::ptrdiff_t y = lo[1]; y <= hi[1]; ++y) {
                            const auto row = static_cast<std::size_t>(
                                z * shape[1] + y
                            );
                            const auto row_begin = static_cast<std::size_t>(
                                z * strides[0] + y * strides[1]
                            );
                            invalidated_rows.insert(
                                row,
                                lo[2],
                                hi[2],
                                [&](const std::ptrdiff_t begin, const std::ptrdiff_t end) {
                                    for (auto x = begin; x <= end; ++x) {
                                        const auto index = row_begin +
                                            static_cast<std::size_t>(x);
                                        if (active[index] != 0) {
                                            active[index] = 0;
                                            --active_count;
                                        }
                                    }
                                }
                            );
                        }
                    }
                }
            }
            if (options.fix_branching) {
                for (const auto node : path) {
                    pdrf[node] = Distance{0};
                }
            }
        }
    };

    std::vector<std::uint32_t> compact_required_targets;
    compact_required_targets.reserve(required_targets.size());
    for (const auto full_target : required_targets) {
        if (full_target >= n) {
            throw std::runtime_error("prepared required target is out of bounds");
        }
        const auto target = domain.compact_node_from_full(full_target);
        if (target == detail::kNoCompactNode) {
            throw std::runtime_error("prepared required target is not foreground");
        }
        compact_required_targets.push_back(target);
    }
    std::sort(
        compact_required_targets.begin(), compact_required_targets.end(),
        [&](const std::uint32_t first, const std::uint32_t second) {
            if (root_field[first] != root_field[second]) {
                return root_field[first] > root_field[second];
            }
            return first < second;
        }
    );
    for (const auto target : compact_required_targets) {
        if (vertex_of_node[target] < 0) {
            trace_target(target);
        }
    }

    std::vector<std::uint32_t> ordered_targets;
    std::size_t ordered_target_cursor = 0;
    std::size_t linear_target_selections = 0;
    const auto linear_target_limit = std::max<std::size_t>(
        16, std::bit_width(domain.size())
    );
    constexpr std::size_t ordered_target_minimum_nodes = 1U << 16;
    const bool allow_ordered_targets =
        domain.size() >= ordered_target_minimum_nodes;
    bool targets_ordered = false;
    while (active_count > 0) {
        auto target = detail::kNoCompactNode;
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "target_selection")
            if (
                !targets_ordered &&
                (!allow_ordered_targets ||
                 linear_target_selections < linear_target_limit)
            ) {
                Distance target_distance = Distance{-1};
                for (std::uint32_t node = 0; node < domain.size(); ++node) {
                    const auto active_index = options.invalidation ==
                            TeasarInvalidation::Ball
                        ? static_cast<std::size_t>(node)
                        : static_cast<std::size_t>(domain.compact_to_full[node]);
                    if (active[active_index] != 0 && root_field[node] > target_distance) {
                        target = node;
                        target_distance = root_field[node];
                    }
                }
                ++linear_target_selections;
            } else {
                if (!targets_ordered) {
                    ordered_targets.reserve(active_count);
                    for (std::uint32_t node = 0; node < domain.size(); ++node) {
                        const auto active_index = options.invalidation ==
                                TeasarInvalidation::Ball
                            ? static_cast<std::size_t>(node)
                            : static_cast<std::size_t>(domain.compact_to_full[node]);
                        if (active[active_index] != 0) {
                            ordered_targets.push_back(node);
                        }
                    }
                    std::sort(
                        ordered_targets.begin(), ordered_targets.end(),
                        [&](const std::uint32_t first, const std::uint32_t second) {
                            if (root_field[first] != root_field[second]) {
                                return root_field[first] > root_field[second];
                            }
                            return first < second;
                        }
                    );
                    targets_ordered = true;
                }
                while (
                    ordered_target_cursor < ordered_targets.size() &&
                    active[
                        options.invalidation == TeasarInvalidation::Ball
                            ? static_cast<std::size_t>(
                                ordered_targets[ordered_target_cursor]
                            )
                            : static_cast<std::size_t>(domain.compact_to_full[
                                ordered_targets[ordered_target_cursor]
                            ])
                    ] == 0
                ) {
                    ++ordered_target_cursor;
                }
                if (ordered_target_cursor < ordered_targets.size()) {
                    target = ordered_targets[ordered_target_cursor++];
                }
            }
        }
        if (target == detail::kNoCompactNode) {
            throw std::runtime_error("TEASAR active-voxel accounting became inconsistent");
        }
        trace_target(target);
    }

    if (report_profile) {
        BIOIMAGE_PROFILE_REPORT(profile)
    }
    if (component_profile != nullptr) {
        component_profile->merge(profile);
    }
    return graph;
}

template <detail::CompactAdjacency Adjacency, class Distance>
inline LatticeSkeletonGraph teasar_compact(
    const ConstArrayView<std::uint8_t> &mask,
    const TeasarOptions &options
) {
    return teasar_compact_impl<Adjacency, Distance>(
        mask, nullptr, options, true
    );
}

template <detail::CompactAdjacency Adjacency, class Distance>
inline LatticeSkeletonGraph teasar_compact_prepared(
    detail::PreparedTeasarComponent prepared,
    const TeasarOptions &options,
    detail::CompactBallInvalidationStats *ball_invalidation_stats = nullptr,
    bioimage_cpp::detail::ActiveProfiler *component_profile = nullptr
) {
    const ConstArrayView<std::uint8_t> unused{};
    return teasar_compact_impl<Adjacency, Distance>(
        unused, &prepared, options, false, ball_invalidation_stats,
        component_profile
    );
}

namespace detail_teasar {

inline void append_skeleton_graph(
    LatticeSkeletonGraph &destination,
    LatticeSkeletonGraph &&source
) {
    if (destination.vertices.size() > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("skeleton vertex offset exceeds uint64 range");
    }
    const auto vertex_offset = static_cast<std::uint64_t>(
        destination.vertices.size()
    );
    destination.vertices.insert(
        destination.vertices.end(),
        std::make_move_iterator(source.vertices.begin()),
        std::make_move_iterator(source.vertices.end())
    );
    destination.radii.insert(
        destination.radii.end(), source.radii.begin(), source.radii.end()
    );
    for (const auto &edge : source.edges) {
        if (
            edge[0] > std::numeric_limits<std::uint64_t>::max() - vertex_offset ||
            edge[1] > std::numeric_limits<std::uint64_t>::max() - vertex_offset
        ) {
            throw std::overflow_error("skeleton edge offset exceeds uint64 range");
        }
        destination.edges.push_back({
            edge[0] + vertex_offset, edge[1] + vertex_offset
        });
    }
}

inline LatticeSkeletonGraph assemble_skeleton_graphs(
    std::vector<LatticeSkeletonGraph> &graphs,
    const std::vector<std::size_t> &component_ids
) {
    std::size_t vertices = 0;
    std::size_t edges = 0;
    for (const auto component_id : component_ids) {
        vertices = detail::checked_add_size(
            vertices, graphs[component_id].vertices.size(),
            "assembled skeleton vertex count overflows size_t"
        );
        edges = detail::checked_add_size(
            edges, graphs[component_id].edges.size(),
            "assembled skeleton edge count overflows size_t"
        );
    }
    LatticeSkeletonGraph result;
    result.vertices.reserve(vertices);
    result.radii.reserve(vertices);
    result.edges.reserve(edges);
    for (const auto component_id : component_ids) {
        append_skeleton_graph(result, std::move(graphs[component_id]));
    }
    return result;
}

inline SkeletonGraph lattice_to_physical(
    LatticeSkeletonGraph graph,
    const std::array<double, 3> &spacing
) {
    SkeletonGraph result;
    result.vertices.reserve(graph.vertices.size());
    for (const auto &coordinate : graph.vertices) {
        result.vertices.push_back({
            static_cast<double>(coordinate[0]) * spacing[0],
            static_cast<double>(coordinate[1]) * spacing[1],
            static_cast<double>(coordinate[2]) * spacing[2],
        });
    }
    result.edges = std::move(graph.edges);
    result.radii = std::move(graph.radii);
    return result;
}

template <class LabelT>
std::string component_context(
    const detail::ComponentDescriptor<LabelT> &component,
    const bool include_label
) {
    std::string context = "TEASAR component";
    if (include_label) {
        if constexpr (std::is_signed_v<LabelT>) {
            context += " label=" + std::to_string(
                static_cast<long long>(component.label)
            );
        } else {
            context += " label=" + std::to_string(
                static_cast<unsigned long long>(component.label)
            );
        }
    }
    context += " first_coordinate=(" +
        std::to_string(component.first_coordinate[0]) + "," +
        std::to_string(component.first_coordinate[1]) + "," +
        std::to_string(component.first_coordinate[2]) + ")";
    return context;
}

template <class LabelT>
std::vector<std::size_t> component_thread_budgets(
    const detail::ComponentSet<LabelT> &components,
    const std::size_t total_budget
) {
    const auto count = components.components.size();
    std::vector<std::size_t> budgets(count, 1);
    if (count == 0 || total_budget <= count) {
        return budgets;
    }
    std::vector<std::size_t> capacities(count, 0);
    std::vector<std::size_t> weights(count, 0);
    for (std::size_t component = 0; component < count; ++component) {
        const auto voxels = static_cast<std::size_t>(
            components.components[component].voxel_count
        );
        capacities[component] = voxels - 1;
        weights[component] = detail::padded_component_volume(
            components.components[component]
        );
    }

    std::size_t remaining = total_budget - count;
    while (remaining > 0) {
        long double total_weight = 0.0L;
        for (std::size_t component = 0; component < count; ++component) {
            if (capacities[component] > 0) {
                total_weight += static_cast<long double>(weights[component]);
            }
        }
        if (total_weight == 0.0L) {
            throw std::runtime_error("TEASAR thread-budget capacity is inconsistent");
        }

        struct Remainder {
            long double fraction;
            std::size_t component;
        };
        std::vector<Remainder> remainders;
        remainders.reserve(count);
        std::size_t allocated = 0;
        const auto round_budget = remaining;
        for (std::size_t component = 0; component < count; ++component) {
            if (capacities[component] == 0) {
                continue;
            }
            const auto exact = static_cast<long double>(round_budget) *
                static_cast<long double>(weights[component]) / total_weight;
            const auto floor_share = static_cast<std::size_t>(exact);
            const auto share = std::min(floor_share, capacities[component]);
            budgets[component] += share;
            capacities[component] -= share;
            allocated += share;
            remainders.push_back({
                exact - static_cast<long double>(floor_share), component
            });
        }
        if (allocated > remaining) {
            throw std::runtime_error("TEASAR thread-budget allocation overflowed");
        }
        remaining -= allocated;
        if (remaining == 0) {
            break;
        }
        std::sort(
            remainders.begin(), remainders.end(),
            [](const Remainder &first, const Remainder &second) {
                if (first.fraction != second.fraction) {
                    return first.fraction > second.fraction;
                }
                return first.component < second.component;
            }
        );
        bool gave_remainder = false;
        for (const auto &entry : remainders) {
            if (remaining == 0) {
                break;
            }
            if (capacities[entry.component] == 0) {
                continue;
            }
            ++budgets[entry.component];
            --capacities[entry.component];
            --remaining;
            gave_remainder = true;
        }
        if (!gave_remainder && allocated == 0) {
            throw std::runtime_error("TEASAR thread-budget allocation stalled");
        }
    }
    if (
        std::accumulate(budgets.begin(), budgets.end(), std::size_t{0}) !=
        total_budget
    ) {
        throw std::runtime_error("TEASAR thread budgets do not sum to call budget");
    }
    return budgets;
}

enum class ComponentEdtStrategy {
    Auto,
    Local,
    Shared,
};

enum class SharedEdtDecision {
    Selected,
    ForcedLocal,
    FewerThanTwoComponents,
    CompactRange,
    VolumeRatio,
    ScratchLimit,
};

inline constexpr std::size_t kSharedEdtScratchLimitBytes =
    std::size_t{256} * 1024 * 1024;

inline const char *shared_edt_decision_name(const SharedEdtDecision decision) {
    switch (decision) {
        case SharedEdtDecision::Selected:
            return "shared";
        case SharedEdtDecision::ForcedLocal:
            return "forced-local";
        case SharedEdtDecision::FewerThanTwoComponents:
            return "fewer-than-two-components";
        case SharedEdtDecision::CompactRange:
            return "compact-range";
        case SharedEdtDecision::VolumeRatio:
            return "volume-ratio";
        case SharedEdtDecision::ScratchLimit:
            return "scratch-limit";
    }
    return "unknown";
}

struct SharedEdtPreparation {
    std::vector<std::vector<float>> component_dbf;
    SharedEdtDecision decision = SharedEdtDecision::ForcedLocal;
    std::size_t estimated_scratch_bytes = 0;

    [[nodiscard]] bool selected() const noexcept {
        return decision == SharedEdtDecision::Selected;
    }
};

struct SharedEdtGatherContext {
    const detail::ComponentSet<std::uint8_t> *components = nullptr;
    std::array<std::ptrdiff_t, 3> global_begin{};
    const std::vector<std::ptrdiff_t> *shared_shape = nullptr;
    std::vector<std::vector<float>> *component_dbf = nullptr;
};

inline void gather_shared_squared_distances(
    const double *squared_distances,
    const std::size_t number_of_values,
    void *raw_context
) {
    auto &context = *static_cast<SharedEdtGatherContext *>(raw_context);
    if (
        context.components == nullptr || context.shared_shape == nullptr ||
        context.component_dbf == nullptr
    ) {
        throw std::invalid_argument("shared EDT gather context is incomplete");
    }
    const auto expected_values = detail::checked_shape_size(
        *context.shared_shape, "shared EDT gather shape overflows size_t"
    );
    if (number_of_values != expected_values) {
        throw std::runtime_error("shared EDT gather volume is inconsistent");
    }
    const auto strides = bioimage_cpp::detail::c_order_strides(
        *context.shared_shape
    );
    const auto &components = *context.components;
    auto &component_dbf = *context.component_dbf;
    for (std::size_t component_id = 0;
         component_id < components.components.size(); ++component_id) {
        const auto &component = components.components[component_id];
        auto &values = component_dbf[component_id];
        values.reserve(static_cast<std::size_t>(component.voxel_count));
        for (std::size_t offset = 0; offset < component.number_of_runs; ++offset) {
            const auto run_id = components.component_run_ids[
                component.run_offset + offset
            ];
            const auto &run = components.runs[run_id];
            const auto z = run.z - context.global_begin[0] + 1;
            const auto y = run.y - context.global_begin[1] + 1;
            const auto row_begin = static_cast<std::size_t>(
                z * strides[0] + y * strides[1]
            );
            for (auto x = run.x_begin; x <= run.x_end; ++x) {
                const auto local_x = static_cast<std::size_t>(
                    x - context.global_begin[2] + 1
                );
                values.push_back(static_cast<float>(
                    std::sqrt(squared_distances[row_begin + local_x])
                ));
            }
        }
        if (values.size() != static_cast<std::size_t>(component.voxel_count)) {
            throw std::runtime_error(
                "shared EDT component value count is inconsistent"
            );
        }
    }
}

inline std::size_t shared_edt_scratch_estimate(
    const std::size_t shared_volume,
    const std::size_t foreground_count,
    const std::size_t maximum_extent,
    const std::size_t number_of_threads
) {
    const auto volume_edt_bytes = detail::checked_multiply_size(
        shared_volume, sizeof(double),
        "shared EDT volume scratch estimate overflows size_t"
    );
    const auto compact_output_bytes = detail::checked_multiply_size(
        foreground_count, sizeof(float),
        "shared EDT compact output estimate overflows size_t"
    );
    const auto workspace_line_bytes = detail::checked_add_size(
        detail::checked_multiply_size(
            maximum_extent, std::size_t{32},
            "shared EDT line scratch estimate overflows size_t"
        ),
        std::size_t{8 + 6 * sizeof(std::vector<double>)},
        "shared EDT line scratch estimate overflows size_t"
    );
    const auto workspace_bytes = detail::checked_multiply_size(
        workspace_line_bytes, number_of_threads,
        "shared EDT thread scratch estimate overflows size_t"
    );
    const auto edt_peak = detail::checked_add_size(
        detail::checked_add_size(
            volume_edt_bytes, compact_output_bytes,
            "shared EDT peak estimate overflows size_t"
        ),
        workspace_bytes,
        "shared EDT peak estimate overflows size_t"
    );
    const auto compact_dbf_bytes = detail::checked_multiply_size(
        foreground_count, sizeof(float),
        "shared EDT compact DBF estimate overflows size_t"
    );
    const auto gather_peak = detail::checked_multiply_size(
        compact_dbf_bytes, std::size_t{1},
        "shared EDT gather peak estimate overflows size_t"
    );
    const auto raw_peak = std::max(edt_peak, gather_peak);
    const auto headroom = detail::checked_add_size(
        raw_peak, std::size_t{9},
        "shared EDT headroom estimate overflows size_t"
    ) / 10;
    return detail::checked_add_size(
        raw_peak, headroom,
        "shared EDT scratch estimate overflows size_t"
    );
}

template <class Profiler>
SharedEdtPreparation prepare_shared_binary_edt(
    const detail::ComponentSet<std::uint8_t> &components,
    const TeasarOptions &options,
    const ComponentEdtStrategy strategy,
    const std::size_t scratch_limit_bytes,
    Profiler &profile
) {
    SharedEdtPreparation result;
    const auto count = components.components.size();
    if (strategy == ComponentEdtStrategy::Local) {
        result.decision = SharedEdtDecision::ForcedLocal;
        return result;
    }
    if (count == 0) {
        result.decision = SharedEdtDecision::FewerThanTwoComponents;
        return result;
    }
    if (strategy == ComponentEdtStrategy::Auto && count < 2) {
        result.decision = SharedEdtDecision::FewerThanTwoComponents;
        return result;
    }

    auto global_begin = components.components.front().begin;
    auto global_end = components.components.front().end;
    std::size_t local_volume_sum = 0;
    for (const auto &component : components.components) {
        const auto padded_volume = detail::padded_component_volume(component);
        if (
            padded_volume > static_cast<std::size_t>(detail::kNoCompactNode) ||
            component.voxel_count > detail::kNoCompactNode
        ) {
            if (strategy == ComponentEdtStrategy::Shared) {
                throw std::invalid_argument(
                    "forced shared EDT requires uint32-compatible components"
                );
            }
            result.decision = SharedEdtDecision::CompactRange;
            return result;
        }
        local_volume_sum = detail::checked_add_size(
            local_volume_sum, padded_volume,
            "summed component EDT volume overflows size_t"
        );
        for (std::size_t axis = 0; axis < 3; ++axis) {
            global_begin[axis] = std::min(global_begin[axis], component.begin[axis]);
            global_end[axis] = std::max(global_end[axis], component.end[axis]);
        }
    }

    std::vector<std::ptrdiff_t> shared_shape(3);
    std::size_t maximum_extent = 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto extent = global_end[axis] - global_begin[axis];
        if (extent > std::numeric_limits<std::ptrdiff_t>::max() - 2) {
            throw std::overflow_error("shared EDT shape overflows ptrdiff_t");
        }
        shared_shape[axis] = extent + 2;
        maximum_extent = std::max(
            maximum_extent, static_cast<std::size_t>(shared_shape[axis])
        );
    }
    const auto shared_volume = detail::checked_shape_size(
        shared_shape, "shared EDT volume overflows size_t"
    );
    if (shared_volume > static_cast<std::size_t>(detail::kNoCompactNode)) {
        if (strategy == ComponentEdtStrategy::Shared) {
            throw std::invalid_argument(
                "forced shared EDT volume exceeds uint32 index range"
            );
        }
        result.decision = SharedEdtDecision::CompactRange;
        return result;
    }
    if (
        strategy == ComponentEdtStrategy::Auto &&
        static_cast<long double>(shared_volume) >
            0.75L * static_cast<long double>(local_volume_sum)
    ) {
        result.decision = SharedEdtDecision::VolumeRatio;
        return result;
    }

    const auto effective_threads = bioimage_cpp::detail::normalize_thread_count(
        options.number_of_threads, components.foreground_count
    );
    result.estimated_scratch_bytes = shared_edt_scratch_estimate(
        shared_volume, components.foreground_count, maximum_extent,
        effective_threads
    );
    if (
        strategy == ComponentEdtStrategy::Auto &&
        result.estimated_scratch_bytes > scratch_limit_bytes
    ) {
        result.decision = SharedEdtDecision::ScratchLimit;
        return result;
    }

    std::unique_ptr<double[]> initialized_squared;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "shared_edt_setup")
        initialized_squared = std::make_unique_for_overwrite<double[]>(
            shared_volume
        );
        std::fill(
            initialized_squared.get(),
            initialized_squared.get() + shared_volume,
            0.0
        );
        const auto strides = bioimage_cpp::detail::c_order_strides(shared_shape);
        for (const auto &run : components.runs) {
            const auto z = run.z - global_begin[0] + 1;
            const auto y = run.y - global_begin[1] + 1;
            const auto x_begin = run.x_begin - global_begin[2] + 1;
            const auto x_end = run.x_end - global_begin[2] + 1;
            const auto row_begin = static_cast<std::size_t>(
                z * strides[0] + y * strides[1]
            );
            for (auto x = x_begin; x <= x_end; ++x) {
                const auto index = row_begin + static_cast<std::size_t>(x);
                initialized_squared[index] = distance::detail::kInfinity;
            }
        }
    }

    result.component_dbf.resize(count);
    SharedEdtGatherContext gather_context{
        &components, global_begin, &shared_shape, &result.component_dbf
    };
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "shared_distance_transform")
        distance::detail::distance_transform_from_squared(
            shared_shape,
            {options.spacing[0], options.spacing[1], options.spacing[2]},
            std::move(initialized_squared),
            &gather_shared_squared_distances, &gather_context,
            effective_threads
        );
    }
    result.decision = SharedEdtDecision::Selected;
    return result;
}

template <class LabelT>
std::vector<LatticeSkeletonGraph> skeletonize_components(
    const detail::ComponentSet<LabelT> &components,
    const TeasarOptions &options,
    const bool include_label_in_errors,
    const std::vector<std::vector<std::array<std::ptrdiff_t, 3>>> *required_targets = nullptr,
    const detail::OpenBlockFaces *open_faces = nullptr,
    std::vector<std::vector<float>> *precomputed_component_dbf = nullptr,
    std::vector<detail::CompactBallInvalidationStats> *ball_invalidation_stats = nullptr,
    bioimage_cpp::detail::ActiveProfiler *component_profile = nullptr
) {
    const auto count = components.components.size();
    std::vector<LatticeSkeletonGraph> results(count);
    if (count == 0) {
        return results;
    }
    if (ball_invalidation_stats != nullptr) {
        ball_invalidation_stats->assign(count, {});
    }
    std::vector<bioimage_cpp::detail::ActiveProfiler> component_profiles;
    if (component_profile != nullptr) {
        component_profiles.resize(count);
    }
    const auto merge_component_profiles = [&] {
        if (component_profile == nullptr) {
            return;
        }
        for (const auto &local_profile : component_profiles) {
            component_profile->merge(local_profile);
        }
    };
    const auto total_budget = bioimage_cpp::detail::normalize_thread_count(
        options.number_of_threads, components.foreground_count
    );
    std::vector<std::size_t> task_order(count);
    std::iota(task_order.begin(), task_order.end(), std::size_t{0});
    std::sort(
        task_order.begin(), task_order.end(),
        [&](const std::size_t first, const std::size_t second) {
            const auto first_work = detail::padded_component_volume(
                components.components[first]
            );
            const auto second_work = detail::padded_component_volume(
                components.components[second]
            );
            if (first_work != second_work) {
                return first_work > second_work;
            }
            return components.components[first].first_flat_index <
                components.components[second].first_flat_index;
        }
    );

    const auto run_component = [&] (
        const std::size_t component_id,
        const std::size_t local_budget
    ) {
        try {
            auto prepared = required_targets == nullptr
                ? detail::prepare_component(components, component_id)
                : detail::prepare_component(
                    components, component_id, required_targets->at(component_id),
                    open_faces
                );
            if (precomputed_component_dbf != nullptr) {
                prepared.compact_dbf = std::move(
                    precomputed_component_dbf->at(component_id)
                );
            }
            auto local_options = options;
            local_options.number_of_threads = local_budget;
            results[component_id] = teasar_compact_prepared<
                detail::CompactAdjacency::OnTheFly, double
            >(
                std::move(prepared), local_options,
                ball_invalidation_stats == nullptr
                    ? nullptr : &ball_invalidation_stats->at(component_id),
                component_profile == nullptr
                    ? nullptr : &component_profiles[component_id]
            );
        } catch (const std::exception &error) {
            throw std::runtime_error(
                component_context(
                    components.components[component_id], include_label_in_errors
                ) + ": " + error.what()
            );
        }
    };

    if (count == 1) {
        run_component(0, total_budget);
        merge_component_profiles();
        return results;
    }
    if (count >= total_budget) {
        std::atomic<std::size_t> cursor{0};
        bioimage_cpp::detail::parallel_for_chunks(
            total_budget, total_budget,
            [&](const std::size_t, const std::size_t, const std::size_t) {
                while (true) {
                    const auto task = cursor.fetch_add(1, std::memory_order_relaxed);
                    if (task >= count) {
                        break;
                    }
                    run_component(task_order[task], 1);
                }
            }
        );
        merge_component_profiles();
        return results;
    }

    const auto budgets = component_thread_budgets(components, total_budget);
    bioimage_cpp::detail::parallel_for_chunks(
        count, count,
        [&](const std::size_t, const std::size_t begin, const std::size_t end) {
            for (auto task = begin; task < end; ++task) {
                const auto component_id = task_order[task];
                run_component(component_id, budgets[component_id]);
            }
        }
    );
    merge_component_profiles();
    return results;
}

} // namespace detail_teasar

inline SkeletonGraph teasar_with_backend(
    const ConstArrayView<std::uint8_t> &mask,
    const TeasarOptions &options,
    const TeasarBackend backend
) {
    LatticeSkeletonGraph lattice;
    switch (backend) {
        case TeasarBackend::Auto:
        case TeasarBackend::CompactOnTheFlyFloat64:
            lattice = teasar_compact<detail::CompactAdjacency::OnTheFly, double>(
                mask, options
            );
            break;
        case TeasarBackend::DenseFloat64:
            lattice = teasar_dense(mask, options);
            break;
        case TeasarBackend::CompactCsrFloat64:
            lattice = teasar_compact<detail::CompactAdjacency::Csr, double>(
                mask, options
            );
            break;
        default:
            throw std::invalid_argument("invalid TEASAR backend");
    }
    return detail_teasar::lattice_to_physical(
        std::move(lattice), options.spacing
    );
}

inline SkeletonGraph teasar_with_component_edt(
    const ConstArrayView<std::uint8_t> &mask,
    const TeasarOptions &options,
    const detail_teasar::ComponentEdtStrategy edt_strategy,
    const std::size_t shared_edt_scratch_limit,
    detail_teasar::SharedEdtDecision *used_decision = nullptr,
    std::size_t *estimated_scratch_bytes = nullptr
) {
    detail_teasar::validate_options(mask, options);
    BIOIMAGE_PROFILE_INIT(profile)
    auto components = detail::extract_binary_components(mask, profile);
    auto shared_edt = detail_teasar::prepare_shared_binary_edt(
        components, options, edt_strategy, shared_edt_scratch_limit, profile
    );
    if (used_decision != nullptr) {
        *used_decision = shared_edt.decision;
    }
    if (estimated_scratch_bytes != nullptr) {
        *estimated_scratch_bytes = shared_edt.estimated_scratch_bytes;
    }
    std::vector<LatticeSkeletonGraph> results;
    std::vector<detail::CompactBallInvalidationStats> ball_stats;
    bioimage_cpp::detail::ActiveProfiler component_profile;
#ifdef BIOIMAGE_PROFILE
    auto *ball_stats_output = &ball_stats;
    auto *component_profile_output = &component_profile;
#else
    auto *ball_stats_output =
        static_cast<std::vector<detail::CompactBallInvalidationStats> *>(nullptr);
    auto *component_profile_output =
        static_cast<bioimage_cpp::detail::ActiveProfiler *>(nullptr);
#endif
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "component_teasar")
        results = detail_teasar::skeletonize_components(
            components, options, false, nullptr, nullptr,
            shared_edt.selected() ? &shared_edt.component_dbf : nullptr,
            ball_stats_output, component_profile_output
        );
    }
    std::vector<std::size_t> component_ids(results.size());
    std::iota(component_ids.begin(), component_ids.end(), std::size_t{0});
    LatticeSkeletonGraph output;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "forest_assembly")
        output = detail_teasar::assemble_skeleton_graphs(results, component_ids);
    }
    BIOIMAGE_PROFILE_REPORT(profile)
    BIOIMAGE_PROFILE_REPORT_NAMED(
        component_profile, "[bioimage TEASAR component profile]"
    )
#ifdef BIOIMAGE_PROFILE
    detail::CompactBallInvalidationStats combined_ball_stats;
    for (const auto &stats : ball_stats) {
        combined_ball_stats.merge(stats);
    }
    std::fprintf(
        stderr,
        "[bioimage TEASAR diagnostics]\n"
        "  shared_edt            %s\n"
        "  shared_scratch        %zu bytes\n"
        "  ball_offers           %zu\n"
        "  ball_accepted         %zu\n"
        "  ball_pops             %zu\n"
        "  ball_stale_pops       %zu\n"
        "  ball_invalidated      %zu\n"
        "  ball_peak_heap        %zu\n",
        detail_teasar::shared_edt_decision_name(shared_edt.decision),
        shared_edt.estimated_scratch_bytes,
        combined_ball_stats.offers,
        combined_ball_stats.accepted,
        combined_ball_stats.pops,
        combined_ball_stats.stale_pops,
        combined_ball_stats.invalidated,
        combined_ball_stats.peak_heap
    );
#endif
    return detail_teasar::lattice_to_physical(
        std::move(output), options.spacing
    );
}

inline SkeletonGraph teasar(
    const ConstArrayView<std::uint8_t> &mask,
    const TeasarOptions &options = {}
) {
    return teasar_with_component_edt(
        mask, options, detail_teasar::ComponentEdtStrategy::Auto,
        detail_teasar::kSharedEdtScratchLimitBytes
    );
}

inline LatticeSkeletonGraph teasar_block(
    const ConstArrayView<std::uint8_t> &mask,
    std::vector<VoxelCoordinate> required_targets,
    const detail::OpenBlockFaces &open_faces,
    const TeasarOptions &options = {}
) {
    detail_teasar::validate_options(mask, options);
    std::vector<detail::ComponentTarget<std::uint8_t>> local_targets;
    local_targets.reserve(required_targets.size());
    for (std::size_t row = 0; row < required_targets.size(); ++row) {
        const auto &target = required_targets[row];
        std::array<std::ptrdiff_t, 3> coordinate{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (
                target[axis] < 0 ||
                static_cast<std::uint64_t>(target[axis]) >=
                    static_cast<std::uint64_t>(mask.shape[axis])
            ) {
                throw std::invalid_argument(
                    "required_targets row " + std::to_string(row) +
                    " is out of bounds at axis " + std::to_string(axis)
                );
            }
            coordinate[axis] = static_cast<std::ptrdiff_t>(target[axis]);
        }
        const auto flat = static_cast<std::size_t>(
            (coordinate[0] * mask.shape[1] + coordinate[1]) * mask.shape[2] +
            coordinate[2]
        );
        if (mask.data[flat] == 0) {
            throw std::invalid_argument(
                "required_targets row " + std::to_string(row) +
                " must lie on foreground"
            );
        }
        local_targets.push_back({std::uint8_t{1}, coordinate});
    }
    std::sort(
        local_targets.begin(), local_targets.end(),
        [](const auto &first, const auto &second) {
            return first.coordinate < second.coordinate;
        }
    );
    local_targets.erase(
        std::unique(
            local_targets.begin(), local_targets.end(),
            [](const auto &first, const auto &second) {
                return first.coordinate == second.coordinate;
            }
        ),
        local_targets.end()
    );

    BIOIMAGE_PROFILE_INIT(profile)
    auto components = detail::extract_binary_components(mask, profile);
    auto component_targets = detail::assign_targets_to_components(
        components, local_targets
    );
    std::vector<LatticeSkeletonGraph> results;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "component_teasar")
        results = detail_teasar::skeletonize_components(
            components, options, false, &component_targets, &open_faces
        );
    }
    std::vector<std::size_t> component_ids(results.size());
    std::iota(component_ids.begin(), component_ids.end(), std::size_t{0});
    LatticeSkeletonGraph output;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "forest_assembly")
        output = detail_teasar::assemble_skeleton_graphs(results, component_ids);
    }
    BIOIMAGE_PROFILE_REPORT(profile)
    return output;
}

} // namespace bioimage_cpp::skeleton
