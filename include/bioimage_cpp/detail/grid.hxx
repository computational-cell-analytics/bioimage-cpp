#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bioimage_cpp::detail {

// Total number of elements in a row-major array of the given shape (the product
// of the per-axis extents). Shape entries are assumed non-negative.
inline std::size_t number_of_elements(const std::vector<std::ptrdiff_t> &shape) {
    std::size_t total = 1;
    for (const auto extent : shape) {
        total *= static_cast<std::size_t>(extent);
    }
    return total;
}

// C-order strides for a row-major array of the given shape, in units of array
// elements (not bytes). The innermost (last) axis has stride 1.
inline std::vector<std::ptrdiff_t> c_order_strides(const std::vector<std::ptrdiff_t> &shape) {
    std::vector<std::ptrdiff_t> strides(shape.size(), 1);
    for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(shape.size()) - 2; axis >= 0; --axis) {
        strides[static_cast<std::size_t>(axis)] =
            strides[static_cast<std::size_t>(axis + 1)] *
            shape[static_cast<std::size_t>(axis + 1)];
    }
    return strides;
}

// Decode a flat (row-major) node index into its per-axis coordinates, writing
// `ndim` entries into `coords_out`. `strides` must be `c_order_strides(shape)`.
// Uses one division + one subtraction per axis (no modulo); the innermost axis
// (stride 1) reduces to a copy of the remaining index. This is the cheap
// decode to prefer in hot loops that need a node's coordinates once, rather
// than calling `valid_offset_target` (div + mod per axis) repeatedly.
inline void coords_from_index(
    std::uint64_t node,
    const std::vector<std::ptrdiff_t> &strides,
    std::size_t ndim,
    std::ptrdiff_t *coords_out
) {
    for (std::size_t axis = 0; axis < ndim; ++axis) {
        const auto stride = static_cast<std::uint64_t>(strides[axis]);
        const auto coord = node / stride;
        coords_out[axis] = static_cast<std::ptrdiff_t>(coord);
        node -= coord * stride;
    }
}

// Translate a flat node index by a per-axis offset on a row-major grid.
//
// Returns true when the offset keeps the result inside the grid, in which case
// `target_out` is set to the neighbor's flat index. Returns false otherwise and
// leaves `target_out` unchanged. `strides` must match `shape` and is typically
// `c_order_strides(shape)`.
inline bool valid_offset_target(
    const std::uint64_t node,
    const std::vector<std::ptrdiff_t> &offset,
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<std::ptrdiff_t> &strides,
    std::uint64_t &target_out
) {
    std::int64_t target_signed = static_cast<std::int64_t>(node);
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        const auto coord =
            static_cast<std::ptrdiff_t>(node / static_cast<std::uint64_t>(strides[axis])) %
            shape[axis];
        const auto neighbor = coord + offset[axis];
        if (neighbor < 0 || neighbor >= shape[axis]) {
            return false;
        }
        target_signed += static_cast<std::int64_t>(offset[axis] * strides[axis]);
    }
    target_out = static_cast<std::uint64_t>(target_signed);
    return true;
}

inline bool is_valid_grid_edge(
    const std::uint64_t node,
    const std::vector<std::ptrdiff_t> &offset,
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<std::ptrdiff_t> &strides
) {
    std::uint64_t unused = 0;
    return valid_offset_target(node, offset, shape, strides, unused);
}

// Given a signed offset `delta` along one axis and the axis `length`, return the
// half-open range of reference coordinates `[lo, hi)` for which `coord + delta`
// stays in `[0, length)`. Returns `lo >= hi` when the offset is larger than the
// axis (no valid reference coordinate). This is the per-axis primitive behind
// the offset-box sweeps in feature accumulation and the distributed block
// extraction — it depends only on grid geometry, not on any array data.
inline void valid_axis_range(
    const std::ptrdiff_t delta,
    const std::size_t length,
    std::size_t &lo,
    std::size_t &hi
) {
    if (delta >= 0) {
        lo = 0;
        const auto d = static_cast<std::size_t>(delta);
        hi = (d >= length) ? 0 : (length - d);
    } else {
        const auto d = static_cast<std::size_t>(-delta);
        lo = (d >= length) ? length : d;
        hi = length;
    }
}

// Sweep every (node, target) pair on a 2D row-major grid for which
// `node + offset` stays inside the grid and the reference node lies in the
// half-open clip box `[clip_y_lo, clip_y_hi) x [clip_x_lo, clip_x_hi)`. The
// body receives flat C-order indices for both endpoints and is expected to
// inline at -O2 since this is a header-only template with a fully-known
// callable type at instantiation. Callers fold any additional restriction —
// a per-thread axis-0 slab, an owned block box — into the clip box; passing
// the full grid extent sweeps every valid reference node. Shared by the
// in-core feature accumulation / lifted-edge sweeps and the distributed
// block extraction.
template <class Body>
void sweep_clipped_box_2d(
    const std::ptrdiff_t dy,
    const std::ptrdiff_t dx,
    const std::size_t height,
    const std::size_t width,
    const std::size_t clip_y_lo,
    const std::size_t clip_y_hi,
    const std::size_t clip_x_lo,
    const std::size_t clip_x_hi,
    const Body &body
) {
    std::size_t y_lo_v, y_hi_v, x_lo_v, x_hi_v;
    valid_axis_range(dy, height, y_lo_v, y_hi_v);
    valid_axis_range(dx, width, x_lo_v, x_hi_v);
    const auto y_lo = std::max(y_lo_v, clip_y_lo);
    const auto y_hi = std::min(y_hi_v, clip_y_hi);
    const auto x_lo = std::max(x_lo_v, clip_x_lo);
    const auto x_hi = std::min(x_hi_v, clip_x_hi);
    if (y_lo >= y_hi || x_lo >= x_hi) {
        return;
    }
    const auto offset_stride = dy * static_cast<std::ptrdiff_t>(width) + dx;
    for (std::size_t y = y_lo; y < y_hi; ++y) {
        const auto row_offset = y * width;
        for (std::size_t x = x_lo; x < x_hi; ++x) {
            const auto node = row_offset + x;
            const auto target = static_cast<std::uint64_t>(
                static_cast<std::ptrdiff_t>(node) + offset_stride
            );
            body(static_cast<std::uint64_t>(node), target);
        }
    }
}

// 3D variant of `sweep_clipped_box_2d`.
template <class Body>
void sweep_clipped_box_3d(
    const std::ptrdiff_t dz,
    const std::ptrdiff_t dy,
    const std::ptrdiff_t dx,
    const std::size_t depth,
    const std::size_t height,
    const std::size_t width,
    const std::size_t clip_z_lo,
    const std::size_t clip_z_hi,
    const std::size_t clip_y_lo,
    const std::size_t clip_y_hi,
    const std::size_t clip_x_lo,
    const std::size_t clip_x_hi,
    const Body &body
) {
    std::size_t z_lo_v, z_hi_v, y_lo_v, y_hi_v, x_lo_v, x_hi_v;
    valid_axis_range(dz, depth, z_lo_v, z_hi_v);
    valid_axis_range(dy, height, y_lo_v, y_hi_v);
    valid_axis_range(dx, width, x_lo_v, x_hi_v);
    const auto z_lo = std::max(z_lo_v, clip_z_lo);
    const auto z_hi = std::min(z_hi_v, clip_z_hi);
    const auto y_lo = std::max(y_lo_v, clip_y_lo);
    const auto y_hi = std::min(y_hi_v, clip_y_hi);
    const auto x_lo = std::max(x_lo_v, clip_x_lo);
    const auto x_hi = std::min(x_hi_v, clip_x_hi);
    if (z_lo >= z_hi || y_lo >= y_hi || x_lo >= x_hi) {
        return;
    }
    const auto slice_size = height * width;
    const auto offset_stride =
        dz * static_cast<std::ptrdiff_t>(slice_size) +
        dy * static_cast<std::ptrdiff_t>(width) + dx;
    for (std::size_t z = z_lo; z < z_hi; ++z) {
        const auto slice_offset = z * slice_size;
        for (std::size_t y = y_lo; y < y_hi; ++y) {
            const auto row_offset = slice_offset + y * width;
            for (std::size_t x = x_lo; x < x_hi; ++x) {
                const auto node = row_offset + x;
                const auto target = static_cast<std::uint64_t>(
                    static_cast<std::ptrdiff_t>(node) + offset_stride
                );
                body(static_cast<std::uint64_t>(node), target);
            }
        }
    }
}

} // namespace bioimage_cpp::detail
