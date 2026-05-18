#include "transformation.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/transformation/affine.hxx"

#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace bioimage_cpp::bindings {
namespace {

template <class T>
using ConstArray = nb::ndarray<nb::numpy, const T, nb::c_contig>;

template <class T>
using OutputArray = nb::ndarray<nb::numpy, T, nb::c_contig>;

using MatrixArray = nb::ndarray<nb::numpy, const double, nb::c_contig>;
using StartsArray = nb::ndarray<nb::numpy, const std::ptrdiff_t, nb::c_contig>;

template <class Array>
std::vector<std::ptrdiff_t> shape_of(const Array &array) {
    std::vector<std::ptrdiff_t> shape(array.ndim());
    for (std::size_t axis = 0; axis < array.ndim(); ++axis) {
        shape[axis] = static_cast<std::ptrdiff_t>(array.shape(axis));
    }
    return shape;
}

std::vector<std::ptrdiff_t> c_order_strides(const std::vector<std::ptrdiff_t> &shape) {
    std::vector<std::ptrdiff_t> strides(shape.size(), 1);
    for (std::size_t axis = shape.size(); axis-- > 1;) {
        strides[axis - 1] = strides[axis] * shape[axis];
    }
    return strides;
}

std::size_t number_of_elements(const std::vector<std::ptrdiff_t> &shape) {
    return static_cast<std::size_t>(std::accumulate(
        shape.begin(),
        shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    ));
}

std::vector<std::size_t> ndarray_shape(const std::vector<std::ptrdiff_t> &shape) {
    std::vector<std::size_t> result(shape.size());
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] < 0) {
            throw std::invalid_argument("output_shape values must be non-negative");
        }
        result[axis] = static_cast<std::size_t>(shape[axis]);
    }
    return result;
}

template <class T>
OutputArray<T> make_output(const std::vector<std::ptrdiff_t> &shape) {
    const auto out_shape = ndarray_shape(shape);
    const std::size_t size = number_of_elements(shape);
    auto *data = new T[size]();
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<T *>(p); });
    return OutputArray<T>(data, out_shape.size(), out_shape.data(), owner);
}

template <std::size_t D, class T>
OutputArray<T> affine_transform_t(
    ConstArray<T> input,
    MatrixArray matrix,
    StartsArray starts,
    StartsArray output_shape,
    const int order,
    const T fill_value
) {
    if (input.ndim() != D) {
        throw std::invalid_argument(
            "input must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(input.ndim())
        );
    }

    const auto input_shape = shape_of(input);
    const auto input_strides = c_order_strides(input_shape);
    const auto matrix_shape = shape_of(matrix);
    const auto matrix_strides = c_order_strides(matrix_shape);
    const auto starts_shape = shape_of(starts);
    const auto starts_strides = c_order_strides(starts_shape);

    if (output_shape.ndim() != 1 || output_shape.shape(0) != D) {
        throw std::invalid_argument(
            "output_shape must have shape (" + std::to_string(D) + ",)"
        );
    }

    std::vector<std::ptrdiff_t> out_shape(D);
    for (std::size_t axis = 0; axis < D; ++axis) {
        out_shape[axis] = output_shape.data()[axis];
    }
    auto output = make_output<T>(out_shape);
    const auto output_strides = c_order_strides(out_shape);

    ConstArrayView<T> input_view{
        input.data(),
        input_shape,
        input_strides,
    };
    ArrayView<T> output_view{
        output.data(),
        out_shape,
        output_strides,
    };
    ConstArrayView<double> matrix_view{
        matrix.data(),
        matrix_shape,
        matrix_strides,
    };
    ConstArrayView<std::ptrdiff_t> starts_view{
        starts.data(),
        starts_shape,
        starts_strides,
    };

    {
        nb::gil_scoped_release release;
        transformation::affine_transform<D, T>(
            input_view, output_view, matrix_view, starts_view, order, fill_value
        );
    }

    return output;
}

template <class T>
void bind_affine_for_dtype(nb::module_ &m, const char *name_2d, const char *name_3d) {
    m.def(
        name_2d,
        &affine_transform_t<2, T>,
        nb::arg("input"),
        nb::arg("matrix"),
        nb::arg("starts"),
        nb::arg("output_shape"),
        nb::arg("order"),
        nb::arg("fill_value"),
        "Apply a 2D affine transformation to a contiguous NumPy array."
    );
    m.def(
        name_3d,
        &affine_transform_t<3, T>,
        nb::arg("input"),
        nb::arg("matrix"),
        nb::arg("starts"),
        nb::arg("output_shape"),
        nb::arg("order"),
        nb::arg("fill_value"),
        "Apply a 3D affine transformation to a contiguous NumPy array."
    );
}

} // namespace

void bind_transformation(nb::module_ &m) {
    bind_affine_for_dtype<std::uint8_t>(
        m, "_affine_transform_2d_uint8", "_affine_transform_3d_uint8"
    );
    bind_affine_for_dtype<std::uint16_t>(
        m, "_affine_transform_2d_uint16", "_affine_transform_3d_uint16"
    );
    bind_affine_for_dtype<std::uint32_t>(
        m, "_affine_transform_2d_uint32", "_affine_transform_3d_uint32"
    );
    bind_affine_for_dtype<std::uint64_t>(
        m, "_affine_transform_2d_uint64", "_affine_transform_3d_uint64"
    );
    bind_affine_for_dtype<std::int8_t>(
        m, "_affine_transform_2d_int8", "_affine_transform_3d_int8"
    );
    bind_affine_for_dtype<std::int16_t>(
        m, "_affine_transform_2d_int16", "_affine_transform_3d_int16"
    );
    bind_affine_for_dtype<std::int32_t>(
        m, "_affine_transform_2d_int32", "_affine_transform_3d_int32"
    );
    bind_affine_for_dtype<std::int64_t>(
        m, "_affine_transform_2d_int64", "_affine_transform_3d_int64"
    );
    bind_affine_for_dtype<float>(
        m, "_affine_transform_2d_float32", "_affine_transform_3d_float32"
    );
    bind_affine_for_dtype<double>(
        m, "_affine_transform_2d_float64", "_affine_transform_3d_float64"
    );
}

} // namespace bioimage_cpp::bindings
