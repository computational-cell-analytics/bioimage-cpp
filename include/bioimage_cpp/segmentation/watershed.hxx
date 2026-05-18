#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bioimage_cpp {

// Marker-controlled flooding watershed on a 2D or 3D image.
//
// `image` is the heightmap. `markers` carries non-zero seed labels that get
// propagated to neighbouring pixels in order of increasing height. If `mask`
// is non-empty, only pixels with a non-zero mask value participate; the
// remaining pixels stay 0 in the output. Connectivity is 1 (axis-aligned
// 4-neighbours in 2D, 6-neighbours in 3D).
template <class HeightT, class LabelT>
void watershed(
    const ConstArrayView<HeightT> &image,
    const ConstArrayView<LabelT> &markers,
    const ConstArrayView<std::uint8_t> &mask,
    const ArrayView<LabelT> &out
) {
    if (image.ndim() != 2 && image.ndim() != 3) {
        throw std::invalid_argument(
            "image must have ndim 2 or 3, got ndim=" + std::to_string(image.ndim())
        );
    }
    if (markers.shape != image.shape) {
        throw std::invalid_argument("markers shape must match image shape");
    }
    if (out.shape != image.shape) {
        throw std::invalid_argument("out shape must match image shape");
    }
    const bool has_mask = !mask.shape.empty();
    if (has_mask && mask.shape != image.shape) {
        throw std::invalid_argument("mask shape must match image shape");
    }

    const auto spatial_ndim = static_cast<std::size_t>(image.ndim());
    const auto number_of_nodes = static_cast<std::uint64_t>(std::accumulate(
        image.shape.begin(),
        image.shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    ));

    const auto strides = detail::c_order_strides(image.shape);

    // Connectivity-1 offsets: ±1 along each spatial axis.
    std::vector<std::vector<std::ptrdiff_t>> offsets;
    offsets.reserve(2 * spatial_ndim);
    for (std::size_t axis = 0; axis < spatial_ndim; ++axis) {
        std::vector<std::ptrdiff_t> neg(spatial_ndim, 0);
        std::vector<std::ptrdiff_t> pos(spatial_ndim, 0);
        neg[axis] = -1;
        pos[axis] = 1;
        offsets.push_back(std::move(neg));
        offsets.push_back(std::move(pos));
    }

    // Zero out the output array first so unreachable pixels stay at 0.
    for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
        out.data[node] = LabelT{0};
    }

    using QueueEntry = std::pair<HeightT, std::uint64_t>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> heap;

    for (std::uint64_t node = 0; node < number_of_nodes; ++node) {
        if (has_mask && mask.data[node] == 0) {
            continue;
        }
        const auto marker = markers.data[node];
        if (marker != LabelT{0}) {
            out.data[node] = marker;
            heap.emplace(image.data[node], node);
        }
    }

    while (!heap.empty()) {
        const auto [height, node] = heap.top();
        heap.pop();
        (void)height;
        const auto label = out.data[node];

        for (const auto &offset : offsets) {
            std::uint64_t neighbor = 0;
            if (!detail::valid_offset_target(node, offset, image.shape, strides, neighbor)) {
                continue;
            }
            if (has_mask && mask.data[neighbor] == 0) {
                continue;
            }
            if (out.data[neighbor] != LabelT{0}) {
                continue;
            }
            out.data[neighbor] = label;
            heap.emplace(image.data[neighbor], neighbor);
        }
    }
}

} // namespace bioimage_cpp
