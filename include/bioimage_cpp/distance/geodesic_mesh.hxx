#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/threading.hxx"
#include "bioimage_cpp/distance/detail/mesh_fast_marching.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Geodesic distances on a triangle-mesh surface.
//
// A triangle mesh defines the geometry: distances are shortest-path lengths
// measured across the 2-manifold surface (not through the embedding space).
// Solved with the first-order Kimmel-Sethian fast marching method (see
// detail/mesh_fast_marching.hxx). The exact reference oracle (pygeodesic) lives
// in development/distance/. Vertices carry real coordinates, so there is no
// sampling argument.

namespace bioimage_cpp::distance {

namespace detail {

// Validate 1-D int64 vertex indices against the vertex count and return them as
// linear indices.
inline std::vector<std::size_t> vertex_indices(
    const ConstArrayView<std::int64_t> &indices,
    std::size_t n_vertices,
    const char *name
) {
    const auto m = static_cast<std::size_t>(indices.shape[0]);
    std::vector<std::size_t> out(m);
    for (std::size_t i = 0; i < m; ++i) {
        const std::int64_t v = indices.data[i];
        if (v < 0 || static_cast<std::size_t>(v) >= n_vertices) {
            throw std::invalid_argument(
                std::string(name) + " contains vertex index " + std::to_string(v) +
                " out of range [0, " + std::to_string(n_vertices) + ")"
            );
        }
        out[i] = static_cast<std::size_t>(v);
    }
    return out;
}

} // namespace detail

// Geodesic distance field on a triangle-mesh surface from a set of sources.
//
// vertices: (n_vertices, 3) float64. faces: (n_faces, 3) int64 triangle
// indices. sources: (n_sources,) int64 vertex indices. speed: optional
// (n_vertices,) per-vertex speed (nullptr => unit speed). distances (out):
// (n_vertices,) float64; +inf for vertices unreachable from any source.
inline void geodesic_distance_field_mesh(
    const ConstArrayView<double> &vertices,
    const ConstArrayView<std::int64_t> &faces,
    const ConstArrayView<std::int64_t> &sources,
    const ConstArrayView<double> *speed,
    ArrayView<double> &distances,
    std::size_t /*n_threads*/ = 1
) {
    detail::MeshFastMarching solver(vertices, faces, speed);
    const auto source_indices =
        detail::vertex_indices(sources, solver.size(), "sources");
    solver.solve(source_indices);

    const auto &dist = solver.distances();
    std::copy(dist.begin(), dist.end(), distances.data);
}

// Full pairwise geodesic distance matrix between mesh vertices.
//
// points: (n_points,) int64 vertex indices. distances (out): (n_points,
// n_points) float64, symmetric with a zero diagonal; +inf when two points lie
// in different connected components.
inline void geodesic_distances_mesh(
    const ConstArrayView<double> &vertices,
    const ConstArrayView<std::int64_t> &faces,
    const ConstArrayView<std::int64_t> &points,
    const ConstArrayView<double> *speed,
    ArrayView<double> &distances,
    std::size_t n_threads = 1
) {
    // Build one solver up front to validate the mesh and the point indices.
    detail::MeshFastMarching probe(vertices, faces, speed);
    const auto point_indices = detail::vertex_indices(points, probe.size(), "points");
    const std::size_t n = point_indices.size();
    double *out = distances.data;

    const std::size_t threads =
        bioimage_cpp::detail::normalize_thread_count(n_threads, n);

    // One independent single-source solve per point (row of the matrix); the
    // mesh/speed inputs are read-only, so the per-thread solves are safe.
    bioimage_cpp::detail::parallel_for_chunks(
        threads, n,
        [&](std::size_t /*thread_id*/, std::size_t begin, std::size_t end) {
            detail::MeshFastMarching solver(vertices, faces, speed);
            for (std::size_t i = begin; i < end; ++i) {
                solver.solve({point_indices[i]});
                for (std::size_t j = 0; j < n; ++j) {
                    out[i * n + j] = solver.distance(point_indices[j]);
                }
            }
        }
    );

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
