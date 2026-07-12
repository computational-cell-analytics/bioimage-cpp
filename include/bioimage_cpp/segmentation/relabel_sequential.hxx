#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/checked_arithmetic.hxx"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace bioimage_cpp::segmentation {

template <class T>
struct RelabelSequentialMaps {
    std::vector<T> forward_map;
    std::vector<T> inverse_map;
};

// Relabel an integer array so all non-zero labels become consecutive starting
// at offset, in sorted order. Label 0 is treated as background and always
// maps to 0. Matches skimage.segmentation.relabel_sequential.
//
// The maps are dense NumPy lookup arrays (forward_map indexed by old label,
// inverse_map indexed by new label), so their sizes scale with `max(label)`
// and `offset`. Inputs whose required maps would overflow `size_t`, the label
// dtype, or (via the pre-sized inverse map) available memory are rejected with
// a clear exception rather than wrapping a size to zero (a heap overflow) or
// silently truncating a new label.
template <class T>
RelabelSequentialMaps<T> relabel_sequential(
    const ConstArrayView<T> &input,
    const T offset,
    const ArrayView<T> &out
) {
    if (input.shape != out.shape) {
        throw std::invalid_argument("out shape must match input shape");
    }

    const auto n = std::accumulate(
        input.shape.begin(),
        input.shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );

    RelabelSequentialMaps<T> maps;

    if (n == 0) {
        // No labels: inverse_map is just the `offset`-length background prefix,
        // forward_map is empty.
        maps.inverse_map.assign(
            detail::checked_size_cast(static_cast<std::uint64_t>(offset), "relabel offset"),
            T{0}
        );
        return maps;
    }

    // Pass 1: find the maximum label value. Tight, vectorizable loop.
    T max_value = T{0};
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const T value = input.data[i];
        if (value > max_value) {
            max_value = value;
        }
    }

    // The forward LUT doubles as a presence bitmap in pass 2 (any non-zero
    // value means "present"), then gets rewritten with the final new labels by
    // the sorted scan below. Its size is `max_value + 1`; the checked add turns
    // a maximal label (e.g. UINT64_MAX, whose `+ 1` would wrap the size to
    // zero and make pass 2 write out of bounds) into a clean overflow error.
    const auto forward_size = detail::checked_size_add(
        detail::checked_size_cast(static_cast<std::uint64_t>(max_value), "relabel max label"),
        1,
        "relabel forward map size"
    );
    maps.forward_map.assign(forward_size, T{0});

    // Pass 2: mark each input value as present in the LUT.
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        maps.forward_map[static_cast<std::size_t>(input.data[i])] = T{1};
    }
    // 0 is reserved background and maps to 0 even if it appeared in input.
    maps.forward_map[0] = T{0};

    // Sorted-order assignment: walk 1..max_value, replace the presence sentinel
    // with the next sequential label, and collect the old labels. The new
    // labels occupy [offset, max(T)]; `capacity` is how many fit. Guarding the
    // count here — before the (potentially large) inverse-map allocation —
    // rejects an out-of-range `offset` cleanly instead of wrapping `new_label`
    // past the dtype maximum or pre-sizing a multi-gigabyte inverse map.
    const std::uint64_t capacity =
        static_cast<std::uint64_t>(std::numeric_limits<T>::max()) -
        static_cast<std::uint64_t>(offset) + 1;
    std::vector<T> old_labels;
    T new_label = offset;
    for (std::size_t v = 1; v <= static_cast<std::size_t>(max_value); ++v) {
        if (maps.forward_map[v] != T{0}) {
            if (static_cast<std::uint64_t>(old_labels.size()) >= capacity) {
                throw std::overflow_error(
                    "relabel_sequential: offset + number of distinct labels "
                    "exceeds the label dtype range"
                );
            }
            maps.forward_map[v] = new_label;
            old_labels.push_back(static_cast<T>(v));
            ++new_label;
        }
    }

    // inverse_map = `offset` background zeros followed by the old labels, in
    // ascending new-label order.
    const auto prefix = detail::checked_size_cast(
        static_cast<std::uint64_t>(offset), "relabel offset"
    );
    const auto inverse_size = detail::checked_size_add(
        prefix, old_labels.size(), "relabel inverse map size"
    );
    maps.inverse_map.assign(inverse_size, T{0});
    std::copy(
        old_labels.begin(),
        old_labels.end(),
        maps.inverse_map.begin() + static_cast<std::ptrdiff_t>(prefix)
    );

    // Pass 3: apply the forward LUT to write the relabeled output.
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        out.data[i] = maps.forward_map[static_cast<std::size_t>(input.data[i])];
    }

    return maps;
}

} // namespace bioimage_cpp::segmentation
