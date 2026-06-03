#pragma once

#include "bioimage_cpp/array_view.hxx"

#include <cstdint>
#include <numeric>
#include <vector>

namespace bioimage_cpp {

// Compute the COCO-style binary run-length encoding of `mask`.
//
// The array is treated as a flat C-order buffer and interpreted as binary
// (zero vs. nonzero). The returned run lengths always start with a run of
// zeros and then alternate (zeros, ones, zeros, ...). If the first element is
// nonzero a leading 0 is emitted. An empty input yields an empty result.
template <class T>
std::vector<std::int64_t> compute_rle(const ConstArrayView<T> &mask) {
    const auto n = std::accumulate(
        mask.shape.begin(),
        mask.shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );

    std::vector<std::int64_t> counts;
    if (n <= 0) {
        return counts;
    }

    bool current_value = false;  // runs start with zeros
    std::int64_t run_length = 0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const bool value = mask.data[i] != T{0};
        if (value == current_value) {
            ++run_length;
        } else {
            counts.push_back(run_length);
            current_value = value;
            run_length = 1;
        }
    }
    counts.push_back(run_length);

    return counts;
}

} // namespace bioimage_cpp
