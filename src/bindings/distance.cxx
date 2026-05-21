#include "distance.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/distance_transform.hxx"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <optional>
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

void check_buffer_shape(
    const char *name,
    const std::vector<std::ptrdiff_t> &actual,
    const std::vector<std::size_t> &expected
) {
    if (actual.size() != expected.size()) {
        throw std::invalid_argument(
            std::string(name) + ": buffer ndim mismatch, expected " +
            std::to_string(expected.size()) + ", got " + std::to_string(actual.size())
        );
    }
    for (std::size_t axis = 0; axis < expected.size(); ++axis) {
        if (static_cast<std::size_t>(actual[axis]) != expected[axis]) {
            throw std::invalid_argument(
                std::string(name) + ": buffer shape mismatch at axis " +
                std::to_string(axis) + ", expected " + std::to_string(expected[axis]) +
                ", got " + std::to_string(actual[axis])
            );
        }
    }
}

nb::tuple distance_transform_uint8(
    UInt8Input input,
    const std::vector<double> &sampling,
    const bool return_distances,
    const bool return_indices,
    const bool return_vectors,
    std::optional<FloatArray> distances_buf,
    std::optional<Int32Array> indices_buf,
    std::optional<FloatArray> vectors_buf,
    const std::size_t n_threads
) {
    if (input.ndim() == 0) {
        throw std::invalid_argument("input must have ndim >= 1, got ndim=0");
    }
    if (sampling.size() != input.ndim()) {
        throw std::invalid_argument(
            "sampling must have length matching input ndim, got ndim=" +
            std::to_string(input.ndim()) + ", sampling length=" +
            std::to_string(sampling.size())
        );
    }

    const auto shape = ndarray_shape(input);
    const auto distances_shape = size_t_shape(shape);
    std::vector<std::size_t> indices_shape;
    indices_shape.reserve(shape.size() + 1);
    indices_shape.push_back(shape.size());
    for (const auto axis_size : shape) {
        indices_shape.push_back(static_cast<std::size_t>(axis_size));
    }
    std::vector<std::size_t> vectors_shape = distances_shape;
    vectors_shape.push_back(shape.size());

    // Pre-allocated buffers: validate shape; otherwise allocate new arrays.
    bool distances_user_provided = false;
    bool indices_user_provided = false;
    bool vectors_user_provided = false;

    FloatArray distances_array;
    if (return_distances) {
        if (distances_buf.has_value()) {
            check_buffer_shape("distances", ndarray_shape(*distances_buf), distances_shape);
            distances_array = *distances_buf;
            distances_user_provided = true;
        } else {
            distances_array = make_array<float, FloatArray>(distances_shape);
        }
    }

    Int32Array indices_array;
    if (return_indices) {
        if (indices_buf.has_value()) {
            check_buffer_shape("indices", ndarray_shape(*indices_buf), indices_shape);
            indices_array = *indices_buf;
            indices_user_provided = true;
        } else {
            indices_array = make_array<std::int32_t, Int32Array>(indices_shape);
        }
    }

    FloatArray vectors_array;
    if (return_vectors) {
        if (vectors_buf.has_value()) {
            check_buffer_shape("vectors", ndarray_shape(*vectors_buf), vectors_shape);
            vectors_array = *vectors_buf;
            vectors_user_provided = true;
        } else {
            vectors_array = make_array<float, FloatArray>(vectors_shape);
        }
    }

    ConstArrayView<std::uint8_t> input_view{input.data(), shape, {}};
    ArrayView<float> distances_view{
        return_distances ? distances_array.data() : nullptr,
        shape,
        {},
    };
    std::vector<std::ptrdiff_t> indices_view_shape(indices_shape.begin(), indices_shape.end());
    ArrayView<std::int32_t> indices_view{
        return_indices ? indices_array.data() : nullptr,
        indices_view_shape,
        {},
    };
    std::vector<std::ptrdiff_t> vectors_view_shape(vectors_shape.begin(), vectors_shape.end());
    ArrayView<float> vectors_view{
        return_vectors ? vectors_array.data() : nullptr,
        vectors_view_shape,
        {},
    };

    {
        nb::gil_scoped_release release;
        distance::distance_transform(
            input_view,
            sampling,
            {distances_view, indices_view, vectors_view},
            n_threads
        );
    }

    auto distances_result = (!return_distances || distances_user_provided)
        ? nb::object(nb::none())
        : nb::cast(distances_array);
    auto indices_result = (!return_indices || indices_user_provided)
        ? nb::object(nb::none())
        : nb::cast(indices_array);
    auto vectors_result = (!return_vectors || vectors_user_provided)
        ? nb::object(nb::none())
        : nb::cast(vectors_array);

    return nb::make_tuple(distances_result, indices_result, vectors_result);
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
        nb::arg("return_vectors"),
        nb::arg("distances_buf").none(),
        nb::arg("indices_buf").none(),
        nb::arg("vectors_buf").none(),
        nb::arg("n_threads"),
        "Distance transform for a C-contiguous uint8 binary array. Computes any\n"
        "combination of (distances, indices, vectors) in a single separable F&H\n"
        "sweep. Pre-allocated output buffers are written into directly."
    );
}

} // namespace bioimage_cpp::bindings
