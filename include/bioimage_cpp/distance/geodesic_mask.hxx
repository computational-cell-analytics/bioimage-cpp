#pragma once

#include "bioimage_cpp/array_view.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Geodesic distances within a mask (regular Cartesian grid).
//
// A "mask" defines the geometry: distances are shortest-path lengths that
// stay inside the nonzero (foreground) region and never cross background
// voxels. This is the fast-marching / Eikonal formulation used by scikit-fmm
// (see development/distance/ for the reference oracle).
//
// STATUS: interface only. The functions below are stubs that throw until the
// fast-marching solver is implemented. The intended implementation reuses the
// shared infrastructure rather than rolling its own:
//   - detail/indexed_heap.hxx  -> DenseIndexedHeap keyed on the linear voxel
//                                 id for the fast-marching narrow band.
//   - detail/grid.hxx          -> c_order_strides / valid_offset_target for
//                                 neighbour offsets and stride math.
//   - detail/threading.hxx     -> parallel_for_chunks to run the per-source
//                                 solves of the pairwise matrix in parallel.
// If a threaded implementation reads any lazily-built structure, freeze it on
// the calling thread before the fan-out (see the CLAUDE.md thread-safety
// contract).

namespace bioimage_cpp::distance {

// Geodesic distance field within a mask from a set of source coordinates.
//
// mask:      nonzero = inside the domain; distances propagate only through
//            nonzero voxels. Row-major (C-contiguous), any ndim.
// sources:   (n_sources, ndim) int64 voxel coordinates in NumPy axis order.
//            A single source is just a 1-row array.
// sampling:  per-axis voxel spacing, length == mask.ndim().
// speed:     optional per-voxel speed, same shape as mask. nullptr => unit
//            speed (plain geodesic distance). When given, the result is the
//            weighted travel time (skfmm.travel_time semantics).
// distances: output, float64, same shape as mask. Unreachable voxels and
//            voxels outside the domain (mask == 0) are set to +inf.
inline void geodesic_distance_field(
    const ConstArrayView<std::uint8_t> & /*mask*/,
    const ConstArrayView<std::int64_t> & /*sources*/,
    const std::vector<double> & /*sampling*/,
    const ConstArrayView<double> * /*speed*/,
    ArrayView<double> & /*distances*/,
    std::size_t /*n_threads*/ = 1
) {
    throw std::runtime_error("geodesic_distance_field: not yet implemented");
}

// Full pairwise geodesic distance matrix between points within a mask.
//
// points:    (n_points, ndim) int64 voxel coordinates in NumPy axis order.
// distances: output, (n_points, n_points) float64. Symmetric; entry (i, j)
//            is the geodesic distance from points[i] to points[j] within the
//            mask, or +inf when the two points are not connected inside the
//            domain. The diagonal is 0.
// Remaining arguments match geodesic_distance_field.
inline void geodesic_distances(
    const ConstArrayView<std::uint8_t> & /*mask*/,
    const ConstArrayView<std::int64_t> & /*points*/,
    const std::vector<double> & /*sampling*/,
    const ConstArrayView<double> * /*speed*/,
    ArrayView<double> & /*distances*/,
    std::size_t /*n_threads*/ = 1
) {
    throw std::runtime_error("geodesic_distances: not yet implemented");
}

} // namespace bioimage_cpp::distance
