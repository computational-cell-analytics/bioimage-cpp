#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/distance/detail/fast_marching.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Geodesic distances within a mask (regular Cartesian grid).
//
// A "mask" defines the geometry: distances are shortest-path lengths that stay
// inside the nonzero (foreground) region and never cross background voxels.
// Solved with the first-order Godunov fast marching method (see
// detail/fast_marching.hxx), matching the scikit-fmm scheme. The reference
// oracle lives in development/distance/.

namespace bioimage_cpp::distance {

namespace detail {

// Convert an (m, ndim) array of int64 voxel coordinates (NumPy axis order) to
// flat row-major indices, validating that every coordinate is in bounds.
inline std::vector<std::size_t> linear_indices_from_coords(
    const ConstArrayView<std::int64_t> &coords,
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<std::ptrdiff_t> &strides,
    const char *name
) {
    const auto m = static_cast<std::size_t>(coords.shape[0]);
    const auto ndim = shape.size();
    std::vector<std::size_t> indices(m);
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t linear = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            const std::int64_t c = coords.data[i * ndim + d];
            if (c < 0 || c >= shape[d]) {
                throw std::invalid_argument(
                    std::string(name) + " coordinate out of bounds: " + name + "[" +
                    std::to_string(i) + ", " + std::to_string(d) + "]=" +
                    std::to_string(c) + " not in [0, " + std::to_string(shape[d]) + ")"
                );
            }
            linear += static_cast<std::size_t>(c) * static_cast<std::size_t>(strides[d]);
        }
        indices[i] = linear;
    }
    return indices;
}

} // namespace detail

// Geodesic distance field within a mask from a set of source coordinates.
//
// mask nonzero = inside domain; distances propagate only through nonzero voxels.
// sources: (n_sources, ndim) int64 voxel coords (NumPy axis order). A single
// source is a 1-row array. sampling: per-axis spacing (length ndim). speed:
// optional, same shape as mask (nullptr => unit speed). distances (out):
// float64, mask.shape; +inf where unreachable / outside the domain. gradient:
// optional (nullptr to skip); when given, the first-order upwind gradient of
// the field is written into it as float32 with shape (*mask.shape, ndim) — see
// GridFastMarching::write_gradient.
inline void geodesic_distance_field(
    const ConstArrayView<std::uint8_t> &mask,
    const ConstArrayView<std::int64_t> &sources,
    const std::vector<double> &sampling,
    const ConstArrayView<double> *speed,
    ArrayView<double> &distances,
    std::size_t /*n_threads*/ = 1,
    ArrayView<float> *gradient = nullptr
) {
    const auto strides = bioimage_cpp::detail::c_order_strides(mask.shape);
    const auto source_indices =
        detail::linear_indices_from_coords(sources, mask.shape, strides, "sources");

    detail::GridFastMarching solver(mask, sampling, speed);
    solver.solve(source_indices);

    const auto &dist = solver.distances();
    std::copy(dist.begin(), dist.end(), distances.data);

    if (gradient != nullptr) {
        solver.write_gradient(*gradient);
    }
}

// Full pairwise geodesic distance matrix between points within a mask.
//
// points: (n_points, ndim) int64. distances (out): (n_points, n_points) float64,
// symmetric with a zero diagonal; +inf where two points are not connected inside
// the domain.
inline void geodesic_distances(
    const ConstArrayView<std::uint8_t> &mask,
    const ConstArrayView<std::int64_t> &points,
    const std::vector<double> &sampling,
    const ConstArrayView<double> *speed,
    ArrayView<double> &distances,
    std::size_t n_threads = 1
) {
    const auto strides = bioimage_cpp::detail::c_order_strides(mask.shape);
    const auto point_indices =
        detail::linear_indices_from_coords(points, mask.shape, strides, "points");
    const std::size_t n = point_indices.size();
    double *out = distances.data;

    const std::size_t threads =
        bioimage_cpp::detail::normalize_thread_count(n_threads, n);

    // One independent single-source solve per point (row of the matrix). The
    // mask/speed inputs are read-only, so the per-thread solves are safe.
    bioimage_cpp::detail::parallel_for_chunks(
        threads, n,
        [&](std::size_t /*thread_id*/, std::size_t begin, std::size_t end) {
            detail::GridFastMarching solver(mask, sampling, speed);
            for (std::size_t i = begin; i < end; ++i) {
                solver.solve({point_indices[i]});
                for (std::size_t j = 0; j < n; ++j) {
                    out[i * n + j] = solver.distance(point_indices[j]);
                }
            }
        }
    );

    // Symmetrize (per-source solves may differ by a tiny numerical amount) and
    // pin the diagonal to exactly zero.
    for (std::size_t i = 0; i < n; ++i) {
        out[i * n + i] = 0.0;
        for (std::size_t j = i + 1; j < n; ++j) {
            const double avg = 0.5 * (out[i * n + j] + out[j * n + i]);
            out[i * n + j] = avg;
            out[j * n + i] = avg;
        }
    }
}

} // namespace bioimage_cpp::distance
