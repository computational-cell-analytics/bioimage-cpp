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

} // namespace bioimage_cpp::detail
