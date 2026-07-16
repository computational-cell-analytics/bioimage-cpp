#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/distance/distance_transform.hxx"
#include "bioimage_cpp/distance/grid_dijkstra.hxx"
#include "bioimage_cpp/skeleton/detail/compact_grid_dijkstra.hxx"
#include "bioimage_cpp/skeleton/detail/components.hxx"
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

struct TeasarOptions {
    std::array<double, 3> spacing{1.0, 1.0, 1.0};
    double scale = 1.5;
    double constant = 0.0;
    double pdrf_scale = 100000.0;
    double pdrf_exponent = 4.0;
    std::size_t number_of_threads = 1;
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
        pdrf[voxel] = 0.0;
        return vertex_id;
    };

    add_vertex(root);
    ConstArrayView<std::uint8_t> padded_view{padded_mask.data(), shape, {}};
    ConstArrayView<double> pdrf_view{pdrf.data(), shape, {}};
    const distance::DijkstraOptions node_options{
        3, {}, distance::DijkstraCostMode::Node, effective_threads
    };
    std::vector<std::size_t> path;
    const auto trace_target = [&](const std::size_t target) {
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
    const bool report_profile
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
    std::vector<std::size_t> required_targets;
    std::size_t required_root = std::numeric_limits<std::size_t>::max();
    if (prepared != nullptr) {
        crop_begin = prepared->input_origin;
        foreground_count = prepared->foreground_count;
        shape = std::move(prepared->padded_shape);
        padded_mask = std::move(prepared->padded_mask);
        distance_mask = std::move(prepared->distance_mask);
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

    auto dbf = std::make_unique_for_overwrite<float[]>(n);
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
    std::vector<float> compact_dbf;
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

    detail::CompactDijkstraWorkspace<Distance> dijkstra_workspace;
    std::vector<Distance> root_field;
    std::uint32_t root = 0;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "root_dijkstra")
        if (required_root != std::numeric_limits<std::size_t>::max()) {
            if (required_root > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "prepared required root exceeds compact index range"
                );
            }
            const auto it = std::lower_bound(
                domain.compact_to_full.begin(), domain.compact_to_full.end(),
                static_cast<std::uint32_t>(required_root)
            );
            if (
                it == domain.compact_to_full.end() ||
                *it != static_cast<std::uint32_t>(required_root)
            ) {
                throw std::runtime_error(
                    "prepared required root is not foreground"
                );
            }
            root = static_cast<std::uint32_t>(
                std::distance(domain.compact_to_full.begin(), it)
            );
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

    std::vector<std::uint8_t> active = std::move(padded_mask);
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
        pdrf[node] = Distance{0};
        return vertex_id;
    };

    add_vertex(root);
    std::vector<std::uint32_t> path;
    detail::RowIntervalUnion invalidated_rows(
        n / static_cast<std::size_t>(shape[2]), shape[2]
    );
    const auto trace_target = [&](const std::uint32_t target) {
        {
            BIOIMAGE_PROFILE_SCOPE(profile, "path_dijkstra")
            detail::compact_node_cost_path<Adjacency>(
                domain, target, skeleton_nodes, pdrf, dijkstra_workspace, path
            );
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
            for (const auto node : path) {
                pdrf[node] = Distance{0};
                const auto voxel = static_cast<std::size_t>(domain.compact_to_full[node]);
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
    };

    std::vector<std::uint32_t> compact_required_targets;
    compact_required_targets.reserve(required_targets.size());
    for (const auto full_target : required_targets) {
        if (full_target >= n) {
            throw std::runtime_error("prepared required target is out of bounds");
        }
        const auto it = std::lower_bound(
            domain.compact_to_full.begin(), domain.compact_to_full.end(),
            static_cast<std::uint32_t>(full_target)
        );
        if (
            it == domain.compact_to_full.end() ||
            *it != static_cast<std::uint32_t>(full_target)
        ) {
            throw std::runtime_error("prepared required target is not foreground");
        }
        compact_required_targets.push_back(static_cast<std::uint32_t>(
            std::distance(domain.compact_to_full.begin(), it)
        ));
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
                    const auto full = domain.compact_to_full[node];
                    if (active[full] != 0 && root_field[node] > target_distance) {
                        target = node;
                        target_distance = root_field[node];
                    }
                }
                ++linear_target_selections;
            } else {
                if (!targets_ordered) {
                    ordered_targets.reserve(active_count);
                    for (std::uint32_t node = 0; node < domain.size(); ++node) {
                        if (active[domain.compact_to_full[node]] != 0) {
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
                        domain.compact_to_full[ordered_targets[ordered_target_cursor]]
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
    const TeasarOptions &options
) {
    const ConstArrayView<std::uint8_t> unused{};
    return teasar_compact_impl<Adjacency, Distance>(
        unused, &prepared, options, false
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

template <class LabelT>
std::vector<LatticeSkeletonGraph> skeletonize_components(
    const detail::ComponentSet<LabelT> &components,
    const TeasarOptions &options,
    const bool include_label_in_errors,
    const std::vector<std::vector<std::array<std::ptrdiff_t, 3>>> *required_targets = nullptr,
    const detail::OpenBlockFaces *open_faces = nullptr
) {
    const auto count = components.components.size();
    std::vector<LatticeSkeletonGraph> results(count);
    if (count == 0) {
        return results;
    }
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
            auto local_options = options;
            local_options.number_of_threads = local_budget;
            results[component_id] = teasar_compact_prepared<
                detail::CompactAdjacency::OnTheFly, double
            >(std::move(prepared), local_options);
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

inline SkeletonGraph teasar(
    const ConstArrayView<std::uint8_t> &mask,
    const TeasarOptions &options = {}
) {
    detail_teasar::validate_options(mask, options);
    BIOIMAGE_PROFILE_INIT(profile)
    auto components = detail::extract_binary_components(mask, profile);
    std::vector<LatticeSkeletonGraph> results;
    {
        BIOIMAGE_PROFILE_SCOPE(profile, "component_teasar")
        results = detail_teasar::skeletonize_components(
            components, options, false
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
    return detail_teasar::lattice_to_physical(
        std::move(output), options.spacing
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
