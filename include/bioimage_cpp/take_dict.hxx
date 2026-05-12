#pragma once

#include "bioimage_cpp/array_view.hxx"

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace bioimage_cpp {

template <class T>
void take_dict(
    const std::unordered_map<T, T> &relabeling,
    const ConstArrayView<T> &to_relabel,
    const ArrayView<T> &out
) {
    if (to_relabel.shape != out.shape) {
        throw std::invalid_argument(
            "out shape must match to_relabel shape"
        );
    }

    const auto n = std::accumulate(
        to_relabel.shape.begin(),
        to_relabel.shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const T value = to_relabel.data[i];
        const auto found = relabeling.find(value);
        if (found == relabeling.end()) {
            throw std::out_of_range("relabeling is missing key " + std::to_string(value));
        }
        out.data[i] = found->second;
    }
}

} // namespace bioimage_cpp
