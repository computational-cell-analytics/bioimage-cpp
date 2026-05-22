#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp {

namespace detail::mesh_smoothing {

template <class I>
struct Adjacency {
    std::vector<std::size_t> offsets;
    std::vector<I> neighbours;
};

template <class I>
Adjacency<I> build_adjacency(const ConstArrayView<I> &faces, std::ptrdiff_t n_verts) {
    const std::ptrdiff_t n_faces = faces.shape[0];

    std::vector<std::pair<I, I>> edges;
    edges.reserve(static_cast<std::size_t>(n_faces) * 3);

    const auto signed_n_verts = static_cast<std::int64_t>(n_verts);
    auto check_index = [&](const I value) {
        const auto signed_value = static_cast<std::int64_t>(value);
        if (signed_value < 0 || signed_value >= signed_n_verts) {
            throw std::invalid_argument(
                "faces contains index " + std::to_string(signed_value)
                + " out of range [0, " + std::to_string(n_verts) + ")"
            );
        }
    };

    for (std::ptrdiff_t f = 0; f < n_faces; ++f) {
        const I a = faces.data[f * 3 + 0];
        const I b = faces.data[f * 3 + 1];
        const I c = faces.data[f * 3 + 2];
        check_index(a);
        check_index(b);
        check_index(c);
        edges.emplace_back(std::min(a, b), std::max(a, b));
        edges.emplace_back(std::min(b, c), std::max(b, c));
        edges.emplace_back(std::min(a, c), std::max(a, c));
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    std::vector<std::size_t> offsets(static_cast<std::size_t>(n_verts) + 1, 0);
    for (const auto &edge : edges) {
        ++offsets[static_cast<std::size_t>(edge.first) + 1];
        ++offsets[static_cast<std::size_t>(edge.second) + 1];
    }
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        offsets[i] += offsets[i - 1];
    }

    std::vector<I> neighbours(offsets.back());
    std::vector<std::size_t> insert_pos(offsets.begin(), offsets.end() - 1);
    for (const auto &edge : edges) {
        neighbours[insert_pos[static_cast<std::size_t>(edge.first)]++] = edge.second;
        neighbours[insert_pos[static_cast<std::size_t>(edge.second)]++] = edge.first;
    }

    return Adjacency<I>{std::move(offsets), std::move(neighbours)};
}

} // namespace detail::mesh_smoothing

// Laplacian smoothing of a triangular mesh: each vertex (and corresponding
// normal) is replaced by the mean of itself and its 1-ring neighbours,
// repeated for `iterations` passes. `verts` and `normals` are (n_verts, dim);
// `faces` is (n_faces, 3) with values in [0, n_verts). Adjacency is built once
// and reused across iterations. For `iterations == 0`, inputs are copied to
// outputs unchanged.
template <class V, class I>
void smooth_mesh(
    const ConstArrayView<V> &verts,
    const ConstArrayView<V> &normals,
    const ConstArrayView<I> &faces,
    std::size_t iterations,
    int n_threads,
    const ArrayView<V> &out_verts,
    const ArrayView<V> &out_normals
) {
    if (verts.shape.size() != 2) {
        throw std::invalid_argument(
            "verts must have ndim=2, got ndim=" + std::to_string(verts.shape.size())
        );
    }
    if (normals.shape != verts.shape) {
        throw std::invalid_argument("normals shape must match verts shape");
    }
    if (faces.shape.size() != 2 || faces.shape[1] != 3) {
        throw std::invalid_argument("faces must have shape (n_faces, 3)");
    }
    if (out_verts.shape != verts.shape) {
        throw std::invalid_argument("out_verts shape must match verts shape");
    }
    if (out_normals.shape != verts.shape) {
        throw std::invalid_argument("out_normals shape must match verts shape");
    }

    const std::ptrdiff_t n_verts = verts.shape[0];
    const std::ptrdiff_t dim = verts.shape[1];
    const std::size_t n_total = static_cast<std::size_t>(n_verts) * static_cast<std::size_t>(dim);

    if (iterations == 0) {
        std::copy(verts.data, verts.data + n_total, out_verts.data);
        std::copy(normals.data, normals.data + n_total, out_normals.data);
        return;
    }

    const auto adjacency = detail::mesh_smoothing::build_adjacency<I>(faces, n_verts);

    std::vector<V> scratch_verts(n_total);
    std::vector<V> scratch_normals(n_total);

    // Choose initial write target so the final write lands in out_*.
    // iter 0 reads from `verts`/`normals` (input); subsequent iters alternate
    // between out_* and scratch. The last write must be to out_*, so when
    // iterations is even, iter 0 writes scratch (and iter 1 writes out).
    V *buf_a_verts = out_verts.data;
    V *buf_a_normals = out_normals.data;
    V *buf_b_verts = scratch_verts.data();
    V *buf_b_normals = scratch_normals.data();
    if (iterations % 2 == 0) {
        std::swap(buf_a_verts, buf_b_verts);
        std::swap(buf_a_normals, buf_b_normals);
    }

    const auto &offsets = adjacency.offsets;
    const auto &neighbours = adjacency.neighbours;
    const std::size_t threads = detail::normalize_thread_count(
        n_threads < 0 ? 0 : static_cast<std::size_t>(n_threads),
        static_cast<std::size_t>(n_verts)
    );

    auto smooth_pass = [&](const V *src_verts, const V *src_normals, V *dst_verts, V *dst_normals) {
        detail::parallel_for_chunks(
            threads,
            static_cast<std::size_t>(n_verts),
            [&](std::size_t, std::size_t begin, std::size_t end) {
                for (std::size_t v = begin; v < end; ++v) {
                    const std::size_t beg = offsets[v];
                    const std::size_t fin = offsets[v + 1];
                    const auto count = static_cast<V>(fin - beg + 1);
                    const std::size_t row = v * static_cast<std::size_t>(dim);
                    for (std::ptrdiff_t d = 0; d < dim; ++d) {
                        V sum_v = src_verts[row + static_cast<std::size_t>(d)];
                        V sum_n = src_normals[row + static_cast<std::size_t>(d)];
                        for (std::size_t k = beg; k < fin; ++k) {
                            const std::size_t nbr_row =
                                static_cast<std::size_t>(neighbours[k]) * static_cast<std::size_t>(dim);
                            sum_v += src_verts[nbr_row + static_cast<std::size_t>(d)];
                            sum_n += src_normals[nbr_row + static_cast<std::size_t>(d)];
                        }
                        dst_verts[row + static_cast<std::size_t>(d)] = sum_v / count;
                        dst_normals[row + static_cast<std::size_t>(d)] = sum_n / count;
                    }
                }
            }
        );
    };

    // First iteration: read from inputs, write to buf_a.
    smooth_pass(verts.data, normals.data, buf_a_verts, buf_a_normals);

    for (std::size_t it = 1; it < iterations; ++it) {
        const V *src_verts = (it % 2 == 1) ? buf_a_verts : buf_b_verts;
        const V *src_normals = (it % 2 == 1) ? buf_a_normals : buf_b_normals;
        V *dst_verts = (it % 2 == 1) ? buf_b_verts : buf_a_verts;
        V *dst_normals = (it % 2 == 1) ? buf_b_normals : buf_a_normals;
        smooth_pass(src_verts, src_normals, dst_verts, dst_normals);
    }
}

} // namespace bioimage_cpp
