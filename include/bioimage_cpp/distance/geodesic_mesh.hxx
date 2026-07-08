#pragma once

#include "bioimage_cpp/array_view.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Geodesic distances on a triangle-mesh surface.
//
// A triangle mesh defines the geometry: distances are shortest-path lengths
// measured across the 2-manifold surface (not through the embedding space).
// The reference oracle uses the exact MMP algorithm (pygeodesic); see
// development/distance/.
//
// STATUS: interface only. The functions below are stubs that throw until the
// mesh geodesic solver is implemented. The intended implementation reuses the
// shared infrastructure:
//   - detail/indexed_heap.hxx  -> SparseIndexedHeap / DenseIndexedHeap keyed
//                                 on the vertex id for the marching front.
//   - detail/threading.hxx     -> parallel_for_chunks for the per-source
//                                 solves of the pairwise matrix.
// Vertices carry real coordinates, so there is no sampling argument.

namespace bioimage_cpp::distance {

// Geodesic distance field on a triangle-mesh surface from a set of sources.
//
// vertices:  (n_vertices, 3) float64 vertex positions.
// faces:     (n_faces, 3) int64 triangle vertex indices.
// sources:   (n_sources,) int64 source vertex indices. A single source is a
//            1-element array.
// speed:     optional (n_vertices,) per-vertex speed. nullptr => unit speed.
// distances: output, (n_vertices,) float64. Vertices unreachable from any
//            source (a disconnected component) are set to +inf.
inline void geodesic_distance_field_mesh(
    const ConstArrayView<double> & /*vertices*/,
    const ConstArrayView<std::int64_t> & /*faces*/,
    const ConstArrayView<std::int64_t> & /*sources*/,
    const ConstArrayView<double> * /*speed*/,
    ArrayView<double> & /*distances*/,
    std::size_t /*n_threads*/ = 1
) {
    throw std::runtime_error("geodesic_distance_field_mesh: not yet implemented");
}

// Full pairwise geodesic distance matrix between mesh vertices.
//
// points:    (n_points,) int64 vertex indices.
// distances: output, (n_points, n_points) float64. Symmetric; entry (i, j) is
//            the surface geodesic distance from points[i] to points[j], or
//            +inf when they lie in different connected components. The
//            diagonal is 0.
// Remaining arguments match geodesic_distance_field_mesh.
inline void geodesic_distances_mesh(
    const ConstArrayView<double> & /*vertices*/,
    const ConstArrayView<std::int64_t> & /*faces*/,
    const ConstArrayView<std::int64_t> & /*points*/,
    const ConstArrayView<double> * /*speed*/,
    ArrayView<double> & /*distances*/,
    std::size_t /*n_threads*/ = 1
) {
    throw std::runtime_error("geodesic_distances_mesh: not yet implemented");
}

} // namespace bioimage_cpp::distance
