#include "distance.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/distance_transform.hxx"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

using UInt8Input = nb::ndarray<nb::numpy, const std::uint8_t, nb::c_contig>;
using FloatArray = nb::ndarray<nb::numpy, float, nb::c_contig>;
using Int32Array = nb::ndarray<nb::numpy, std::int32_t, nb::c_contig>;

template <class Array>
std::vector<std::ptrdiff_t> ndarray_shape(const Array &array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

std::vector<std::size_t> size_t_shape(const std::vector<std::ptrdiff_t> &shape) {
    std::vector<std::size_t> out(shape.size());
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        out[axis] = static_cast<std::size_t>(shape[axis]);
    }
    return out;
}

template <class T, class Array>
Array make_array(const std::vector<std::size_t> &shape) {
    std::size_t size = 1;
    for (const auto axis_size : shape) {
        size *= axis_size;
    }
    auto *data = new T[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return Array(data, shape.size(), shape.data(), owner);
}

nb::tuple distance_transform_uint8(
    UInt8Input input,
    const std::vector<double> &sampling,
    const bool return_distances,
    const bool return_indices
) {
    if (!return_distances && !return_indices) {
        throw std::invalid_argument("at least one of return_distances/return_indices must be True");
    }

    const auto shape = ndarray_shape(input);
    const auto out_shape = size_t_shape(shape);
    std::vector<std::size_t> indices_shape;
    indices_shape.reserve(shape.size() + 1);
    indices_shape.push_back(shape.size());
    for (const auto axis_size : shape) {
        indices_shape.push_back(static_cast<std::size_t>(axis_size));
    }

    FloatArray distances = return_distances
        ? make_array<float, FloatArray>(out_shape)
        : FloatArray(nullptr, 0, nullptr);
    Int32Array indices = return_indices
        ? make_array<std::int32_t, Int32Array>(indices_shape)
        : Int32Array(nullptr, 0, nullptr);

    ConstArrayView<std::uint8_t> input_view{
        input.data(),
        shape,
        {},
    };
    ArrayView<float> distances_view{
        return_distances ? distances.data() : nullptr,
        shape,
        {},
    };
    ArrayView<std::int32_t> indices_view{
        return_indices ? indices.data() : nullptr,
        std::vector<std::ptrdiff_t>(indices_shape.begin(), indices_shape.end()),
        {},
    };
    ArrayView<float> vectors_view{
        nullptr,
        {},
        {},
    };

    {
        nb::gil_scoped_release release;
        distance::distance_transform(
            input_view,
            sampling,
            {distances_view, indices_view, vectors_view}
        );
    }

    return nb::make_tuple(
        return_distances ? nb::cast(distances) : nb::none(),
        return_indices ? nb::cast(indices) : nb::none()
    );
}

FloatArray vector_difference_transform_uint8(
    UInt8Input input,
    const std::vector<double> &sampling
) {
    const auto shape = ndarray_shape(input);
    std::vector<std::size_t> vector_shape = size_t_shape(shape);
    vector_shape.push_back(shape.size());
    FloatArray vectors = make_array<float, FloatArray>(vector_shape);

    ConstArrayView<std::uint8_t> input_view{
        input.data(),
        shape,
        {},
    };
    ArrayView<float> vectors_view{
        vectors.data(),
        std::vector<std::ptrdiff_t>(vector_shape.begin(), vector_shape.end()),
        {},
    };

    {
        nb::gil_scoped_release release;
        distance::distance_transform(
            input_view,
            sampling,
            {
                {nullptr, {}, {}},
                {nullptr, {}, {}},
                vectors_view,
            }
        );
    }

    return vectors;
}

} // namespace

void bind_distance(nb::module_ &m) {
    m.def(
        "_distance_transform_uint8",
        &distance_transform_uint8,
        nb::arg("input"),
        nb::arg("sampling"),
        nb::arg("return_distances"),
        nb::arg("return_indices"),
        "Distance transform for a C-contiguous uint8 binary array."
    );
    m.def(
        "_vector_difference_transform_uint8",
        &vector_difference_transform_uint8,
        nb::arg("input"),
        nb::arg("sampling"),
        "Vector difference transform for a C-contiguous uint8 binary array."
    );
}

} // namespace bioimage_cpp::bindings
