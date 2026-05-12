#pragma once

#include <cstddef>
#include <vector>

namespace bioimage_cpp {

template <class T>
struct ArrayView {
    T *data = nullptr;
    std::vector<std::ptrdiff_t> shape;
    std::vector<std::ptrdiff_t> strides;

    [[nodiscard]] std::ptrdiff_t ndim() const {
        return static_cast<std::ptrdiff_t>(shape.size());
    }
};

template <class T>
struct ConstArrayView {
    const T *data = nullptr;
    std::vector<std::ptrdiff_t> shape;
    std::vector<std::ptrdiff_t> strides;

    [[nodiscard]] std::ptrdiff_t ndim() const {
        return static_cast<std::ptrdiff_t>(shape.size());
    }
};

} // namespace bioimage_cpp
