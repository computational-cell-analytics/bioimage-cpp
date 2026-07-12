#pragma once

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
        // Avoid negating PTRDIFF_MIN. Converting first and subtracting from
        // zero computes the magnitude in the unsigned domain.
        const auto d = std::size_t{0} - static_cast<std::size_t>(delta);
        lo = (d >= length) ? length : d;
        hi = length;
    }
}

// Number of leading positions to skip along an axis of length `length` for a
// signed offset `delta` (i.e. `max(0, -delta)` clamped to `length`). Returns a
// plain `std::ptrdiff_t` so the affinity kernels keep their inline
// `ptrdiff_t` loop bounds and inner-loop codegen, while avoiding the undefined
// behaviour of negating `delta == PTRDIFF_MIN`: magnitudes at least as large as
// the axis (which make the whole offset channel empty) are clamped to `length`,
// so the negation only runs when `-length < delta < 0` and is always safe.
inline std::ptrdiff_t axis_begin_offset(
    const std::ptrdiff_t delta,
    const std::ptrdiff_t length
) {
    if (delta >= 0) {
        return 0;
    }
    if (delta <= -length) {
        return length;
    }
    return -delta;
}

} // namespace bioimage_cpp::detail
