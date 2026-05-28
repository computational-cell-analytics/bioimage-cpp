#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/graph/grid_graph.hxx"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bioimage_cpp::graph {

namespace detail_grid_edge_projection {

// Walk a sub-shape in C-order, computing a flat output offset incrementally
// using `strides` (the strides of the *full* containing array, not of the
// sub-shape itself). Calls `callback(flat)` at every leaf. With `Axis` known
// at compile time the compiler unrolls the entire nest for D = 2 / D = 3.
template <std::size_t Axis, std::size_t D, class Callback>
void enumerate_in_c_order(
    const std::array<std::uint64_t, D> &subshape,
    const std::array<std::ptrdiff_t, D> &strides,
    std::ptrdiff_t base,
    Callback &&callback
) {
    if constexpr (Axis == D) {
        callback(base);
    } else {
        for (std::uint64_t i = 0; i < subshape[Axis]; ++i) {
            enumerate_in_c_order<Axis + 1, D>(subshape, strides, base, callback);
            base += strides[Axis];
        }
    }
}

// Walk every coordinate in C-order over `shape`, tracking whether
// `coord + offset` is in bounds, and call `callback(in_bounds, coord)` at the
// leaf. `coord` is filled in-place and re-used across calls so the sampler
// can read it without copying.
template <std::size_t Axis, std::size_t D, class Callback>
void enumerate_with_offset(
    const std::array<std::uint64_t, D> &shape,
    const std::array<std::ptrdiff_t, D> &offset,
    std::array<std::ptrdiff_t, D> &coord,
    bool in_bounds,
    Callback &&callback
) {
    if constexpr (Axis == D) {
        callback(in_bounds, coord);
    } else {
        for (std::uint64_t c = 0; c < shape[Axis]; ++c) {
            coord[Axis] = static_cast<std::ptrdiff_t>(c);
            const std::ptrdiff_t target = coord[Axis] + offset[Axis];
            const bool axis_ok =
                target >= 0 &&
                target < static_cast<std::ptrdiff_t>(shape[Axis]);
            enumerate_with_offset<Axis + 1, D>(
                shape, offset, coord, in_bounds && axis_ok, callback
            );
        }
    }
}

template <std::size_t D>
std::array<std::ptrdiff_t, D> full_c_strides(
    const std::array<std::uint64_t, D> &shape
) {
    std::array<std::ptrdiff_t, D> strides{};
    std::ptrdiff_t s = 1;
    for (std::size_t i = D; i-- > 0; ) {
        strides[i] = s;
        s *= static_cast<std::ptrdiff_t>(shape[i]);
    }
    return strides;
}

template <std::size_t D>
std::ptrdiff_t shape_product(const std::array<std::uint64_t, D> &shape) {
    std::ptrdiff_t p = 1;
    for (std::size_t i = 0; i < D; ++i) {
        p *= static_cast<std::ptrdiff_t>(shape[i]);
    }
    return p;
}

template <std::size_t D>
void require_projection_output_shape(
    const std::array<std::uint64_t, D> &graph_shape,
    const std::vector<std::ptrdiff_t> &out_shape,
    const std::ptrdiff_t expected_leading,
    const char *argument_name
) {
    if (out_shape.size() != D + 1) {
        throw std::invalid_argument(
            std::string(argument_name) + " must have ndim == graph.ndim + 1"
        );
    }
    if (out_shape[0] != expected_leading) {
        throw std::invalid_argument(
            std::string(argument_name) + " leading dimension does not match expected length"
        );
    }
    for (std::size_t d = 0; d < D; ++d) {
        if (out_shape[d + 1] != static_cast<std::ptrdiff_t>(graph_shape[d])) {
            throw std::invalid_argument(
                std::string(argument_name) + " spatial shape must match graph shape"
            );
        }
    }
}

} // namespace detail_grid_edge_projection

// Writes graph edge ids into `output` of shape (D, *graph.shape) and dtype
// int64. The caller pre-fills `output` with -1; this function then sets
// output[axis, c_0, ..., c_{D-1}] = edge_id for every edge whose spanning
// axis is `axis` and whose pivot (smaller-endpoint) coordinate is
// (c_0, ..., c_{D-1}). Slots where c_axis == shape[axis] - 1 are not visited
// and keep their pre-filled -1.
//
// Exploits the fact that the GridGraph stores edges axis-major + C-order
// within each axis: the k-th edge of axis `d` corresponds exactly to the
// k-th pivot coordinate of the sub-shape (s_0, ..., s_d - 1, ..., s_{D-1})
// walked in C-order. So we walk that sub-shape with the *full* output
// strides and emit consecutive edge ids — no per-edge coordinate division.
template <std::size_t D>
void project_edge_ids_to_pixels(
    const GridGraph<D> &graph,
    const ArrayView<std::int64_t> &output
) {
    namespace dgep = detail_grid_edge_projection;
    const auto &shape = graph.shape();
    dgep::require_projection_output_shape<D>(
        shape, output.shape, static_cast<std::ptrdiff_t>(D), "output"
    );

    const auto strides = dgep::full_c_strides<D>(shape);
    const std::ptrdiff_t plane = dgep::shape_product<D>(shape);

    auto *data = output.data;
    std::int64_t edge_id = 0;
    for (std::size_t axis = 0; axis < D; ++axis) {
        auto pivot_shape = shape;
        if (pivot_shape[axis] == 0) {
            continue;
        }
        --pivot_shape[axis];
        const std::ptrdiff_t base =
            static_cast<std::ptrdiff_t>(axis) * plane;
        dgep::enumerate_in_c_order<0, D>(
            pivot_shape, strides, base,
            [data, &edge_id](std::ptrdiff_t flat) {
                data[flat] = edge_id++;
            }
        );
    }
}

// Enumerates "lifted" edges defined by a set of per-channel grid offsets.
// Walks (offset_idx, *coord) in C-order over (offsets.size(), *graph.shape).
// For each coord whose target `coord + offsets[offset_idx]` stays in bounds
// AND the sampler accepts, writes a sequential counter starting at 0;
// rejected slots get -1. Returns the total number of valid entries written.
//
// The counter is NOT a graph edge id — it indexes into the implicit array
// of lifted edges derived from an affinity volume.
template <std::size_t D, class Sampler>
std::uint64_t project_edge_ids_to_pixels_with_offsets_impl(
    const GridGraph<D> &graph,
    const std::vector<std::array<std::ptrdiff_t, D>> &offsets,
    Sampler &&sampler,
    const ArrayView<std::int64_t> &output
) {
    namespace dgep = detail_grid_edge_projection;
    const auto &shape = graph.shape();
    dgep::require_projection_output_shape<D>(
        shape, output.shape, static_cast<std::ptrdiff_t>(offsets.size()), "output"
    );

    auto *out = output.data;
    std::int64_t edge_id = 0;
    std::size_t out_idx = 0;
    std::array<std::ptrdiff_t, D> coord{};
    for (std::size_t off_idx = 0; off_idx < offsets.size(); ++off_idx) {
        const auto &off = offsets[off_idx];
        dgep::enumerate_with_offset<0, D>(
            shape, off, coord, /*in_bounds=*/true,
            [&](bool in_bounds, const std::array<std::ptrdiff_t, D> &c) {
                if (in_bounds && sampler(off_idx, c)) {
                    out[out_idx++] = edge_id++;
                } else {
                    out[out_idx++] = -1;
                }
            }
        );
    }
    return static_cast<std::uint64_t>(edge_id);
}

// Convenience overload: no filter.
template <std::size_t D>
std::uint64_t project_edge_ids_to_pixels_with_offsets(
    const GridGraph<D> &graph,
    const std::vector<std::array<std::ptrdiff_t, D>> &offsets,
    const ArrayView<std::int64_t> &output
) {
    return project_edge_ids_to_pixels_with_offsets_impl<D>(
        graph, offsets,
        [](std::size_t, const std::array<std::ptrdiff_t, D> &) { return true; },
        output
    );
}

// Convenience overload: only coords aligned with `strides` along every axis.
template <std::size_t D>
std::uint64_t project_edge_ids_to_pixels_with_offsets(
    const GridGraph<D> &graph,
    const std::vector<std::array<std::ptrdiff_t, D>> &offsets,
    const std::array<std::ptrdiff_t, D> &strides,
    const ArrayView<std::int64_t> &output
) {
    for (std::size_t d = 0; d < D; ++d) {
        if (strides[d] <= 0) {
            throw std::invalid_argument("strides must be positive");
        }
    }
    return project_edge_ids_to_pixels_with_offsets_impl<D>(
        graph, offsets,
        [&strides](std::size_t, const std::array<std::ptrdiff_t, D> &c) {
            for (std::size_t d = 0; d < D; ++d) {
                if (c[d] % strides[d] != 0) {
                    return false;
                }
            }
            return true;
        },
        output
    );
}

// Convenience overload: only coords where the (n_offsets, *graph.shape)
// `mask` is non-zero.
template <std::size_t D>
std::uint64_t project_edge_ids_to_pixels_with_offsets(
    const GridGraph<D> &graph,
    const std::vector<std::array<std::ptrdiff_t, D>> &offsets,
    const ConstArrayView<std::uint8_t> &mask,
    const ArrayView<std::int64_t> &output
) {
    namespace dgep = detail_grid_edge_projection;
    const auto &shape = graph.shape();
    dgep::require_projection_output_shape<D>(
        shape, mask.shape, static_cast<std::ptrdiff_t>(offsets.size()), "mask"
    );

    const auto mask_spatial_strides = dgep::full_c_strides<D>(shape);
    const std::ptrdiff_t mask_plane = dgep::shape_product<D>(shape);
    const auto *mdata = mask.data;
    return project_edge_ids_to_pixels_with_offsets_impl<D>(
        graph, offsets,
        [mdata, mask_spatial_strides, mask_plane](
            std::size_t off_idx, const std::array<std::ptrdiff_t, D> &c
        ) {
            std::ptrdiff_t flat =
                static_cast<std::ptrdiff_t>(off_idx) * mask_plane;
            for (std::size_t d = 0; d < D; ++d) {
                flat += c[d] * mask_spatial_strides[d];
            }
            return mdata[flat] != 0;
        },
        output
    );
}

} // namespace bioimage_cpp::graph
