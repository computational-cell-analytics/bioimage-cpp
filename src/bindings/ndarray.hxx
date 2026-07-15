#pragma once

#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings::detail {

template <class T>
using NumpyArray = nb::ndarray<nb::numpy, T, nb::c_contig>;

inline std::size_t checked_array_size(const std::span<const std::size_t> shape) {
    std::size_t size = 1;
    for (const auto extent : shape) {
        if (extent != 0 && size > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::overflow_error("NumPy output shape overflows size_t");
        }
        size *= extent;
    }
    return size;
}

template <class T>
NumpyArray<T> make_array(const std::span<const std::size_t> shape) {
    auto *data = new T[checked_array_size(shape)]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return NumpyArray<T>(data, shape.size(), shape.data(), owner);
}

template <class T>
NumpyArray<T> make_array(const std::initializer_list<std::size_t> shape) {
    return make_array<T>(std::span<const std::size_t>(shape.begin(), shape.size()));
}

template <class T>
NumpyArray<T> make_array_for_overwrite(const std::span<const std::size_t> shape) {
    auto *data = new T[checked_array_size(shape)];
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return NumpyArray<T>(data, shape.size(), shape.data(), owner);
}

template <class T>
NumpyArray<T> make_array_for_overwrite(
    const std::initializer_list<std::size_t> shape
) {
    return make_array_for_overwrite<T>(
        std::span<const std::size_t>(shape.begin(), shape.size())
    );
}

template <class T>
NumpyArray<T> copy_vector_to_array(
    const std::vector<T> &values,
    const std::span<const std::size_t> shape
) {
    if (checked_array_size(shape) != values.size()) {
        throw std::invalid_argument("NumPy output shape does not match vector size");
    }
    auto output = make_array_for_overwrite<T>(shape);
    std::copy(values.begin(), values.end(), output.data());
    return output;
}

template <class T>
NumpyArray<T> copy_vector_to_array(
    const std::vector<T> &values,
    const std::initializer_list<std::size_t> shape
) {
    return copy_vector_to_array<T>(
        values, std::span<const std::size_t>(shape.begin(), shape.size())
    );
}

template <class T>
NumpyArray<T> copy_vector_to_array(const std::vector<T> &values) {
    return copy_vector_to_array<T>(values, {values.size()});
}

} // namespace bioimage_cpp::bindings::detail
