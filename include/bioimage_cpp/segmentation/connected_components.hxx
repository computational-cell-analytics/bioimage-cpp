#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/util/union_find.hxx"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::segmentation {

namespace detail_cc {

// Enumerate the *backward* half of the k-connectivity neighbourhood for an
// `ndim`-dimensional raster-scan: every offset in {-1, 0, 1}^ndim with at
// least one and at most `connectivity` non-zero components, restricted to
// those whose first non-zero component is -1 (so they point to pixels that
// have already been visited under C-order traversal).
inline std::vector<std::vector<std::ptrdiff_t>> make_backward_offsets(
    const int ndim, const int connectivity
) {
    std::vector<std::vector<std::ptrdiff_t>> offsets;
    std::size_t total = 1;
    for (int axis = 0; axis < ndim; ++axis) {
        total *= 3;
    }
    for (std::size_t code = 0; code < total; ++code) {
        std::vector<std::ptrdiff_t> offset(static_cast<std::size_t>(ndim), 0);
        std::size_t value = code;
        for (int axis = ndim - 1; axis >= 0; --axis) {
            offset[static_cast<std::size_t>(axis)] =
                static_cast<std::ptrdiff_t>(value % 3) - 1;
            value /= 3;
        }
        int nonzero = 0;
        for (const auto component : offset) {
            if (component != 0) {
                ++nonzero;
            }
        }
        if (nonzero == 0 || nonzero > connectivity) {
            continue;
        }
        bool backward = false;
        for (const auto component : offset) {
            if (component < 0) {
                backward = true;
                break;
            }
            if (component > 0) {
                break;
            }
        }
        if (backward) {
            offsets.push_back(std::move(offset));
        }
    }
    return offsets;
}

} // namespace detail_cc

// Two-pass connected-components labeling.
//
// Pixels with `image[i] == background` are background and written as 0.
// Two non-background pixels share an output label iff there is a path between
// them through `connectivity`-neighbour steps along which the input value is
// constant. Output labels are dense, start at 1, and are assigned in
// first-occurrence (C-order) order.
//
// Supports 2D and 3D arrays. `out` must have the same shape as `image`.
// `connectivity` is in [1, ndim] (1 = orthogonal neighbours only, ndim = full
// diagonal connectivity).
template <class InT>
void label(
    const ConstArrayView<InT> &image,
    const InT background,
    const int connectivity,
    const ArrayView<std::uint64_t> &out
) {
    const auto &shape = image.shape;
    const int ndim = static_cast<int>(shape.size());
    if (ndim != 2 && ndim != 3) {
        throw std::invalid_argument(
            "image must have ndim 2 or 3, got ndim=" + std::to_string(ndim)
        );
    }
    if (connectivity < 1 || connectivity > ndim) {
        throw std::invalid_argument(
            "connectivity must be in [1, ndim], got connectivity=" +
            std::to_string(connectivity) + " for ndim=" + std::to_string(ndim)
        );
    }

    std::uint64_t number_of_pixels = 1;
    for (const auto extent : shape) {
        number_of_pixels *= static_cast<std::uint64_t>(extent);
    }

    for (std::uint64_t i = 0; i < number_of_pixels; ++i) {
        out.data[i] = 0;
    }
    if (number_of_pixels == 0) {
        return;
    }

    const auto strides = bioimage_cpp::detail::c_order_strides(shape);
    const auto offsets = detail_cc::make_backward_offsets(ndim, connectivity);

    bioimage_cpp::util::UnionFind sets(static_cast<std::size_t>(number_of_pixels));

    for (std::uint64_t i = 0; i < number_of_pixels; ++i) {
        const InT value = image.data[i];
        if (value == background) {
            continue;
        }
        for (const auto &offset : offsets) {
            std::uint64_t neighbor = 0;
            if (!bioimage_cpp::detail::valid_offset_target(
                    i, offset, shape, strides, neighbor
                )) {
                continue;
            }
            if (image.data[neighbor] == value) {
                sets.merge(i, neighbor);
            }
        }
    }

    std::vector<std::uint64_t> root_to_dense(
        static_cast<std::size_t>(number_of_pixels), 0
    );
    std::uint64_t next_label = 1;
    for (std::uint64_t i = 0; i < number_of_pixels; ++i) {
        if (image.data[i] == background) {
            continue;
        }
        const std::uint64_t root = sets.find(i);
        std::uint64_t dense = root_to_dense[static_cast<std::size_t>(root)];
        if (dense == 0) {
            dense = next_label++;
            root_to_dense[static_cast<std::size_t>(root)] = dense;
        }
        out.data[i] = dense;
    }
}

} // namespace bioimage_cpp::segmentation
