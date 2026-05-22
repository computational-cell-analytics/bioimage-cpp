#pragma once

#include "bioimage_cpp/array_view.hxx"

#include <cstddef>
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
    maps.inverse_map.assign(static_cast<std::size_t>(offset), T{0});

    if (n == 0) {
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
    // value means "present"), then gets rewritten with the final new labels
    // by the sorted scan below.
    maps.forward_map.assign(static_cast<std::size_t>(max_value) + 1u, T{0});

    // Pass 2: mark each input value as present in the LUT.
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        maps.forward_map[static_cast<std::size_t>(input.data[i])] = T{1};
    }
    // 0 is reserved background and maps to 0 even if it appeared in input.
    maps.forward_map[0] = T{0};

    // Sorted-order assignment: walk 1..max_value, replace presence sentinel
    // with the next sequential label, and push the old label onto the
    // inverse map (already pre-sized to `offset` zeros).
    T new_label = offset;
    for (std::size_t v = 1; v <= static_cast<std::size_t>(max_value); ++v) {
        if (maps.forward_map[v] != T{0}) {
            maps.forward_map[v] = new_label;
            maps.inverse_map.push_back(static_cast<T>(v));
            ++new_label;
        }
    }

    // Pass 3: apply the forward LUT to write the relabeled output.
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        out.data[i] = maps.forward_map[static_cast<std::size_t>(input.data[i])];
    }

    return maps;
}

} // namespace bioimage_cpp::segmentation
