#include "transformation.hxx"

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"
#include "bioimage_cpp/transformation/affine.hxx"

#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
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

template <std::size_t D, class T>
OutputArray<T> affine_transform_t(
    ConstArray<T> input,
    MatrixArray matrix,
    StartsArray starts,
    OutputArray<T> output,
    const int order,
    const T fill_value
) {
    if (input.ndim() != D) {
        throw std::invalid_argument(
            "input must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(input.ndim())
        );
    }
    if (output.ndim() != D) {
        throw std::invalid_argument(
            "output must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(output.ndim())
        );
    }

    const auto input_shape = shape_of(input);
    const auto input_strides = detail::c_order_strides(input_shape);
    const auto matrix_shape = shape_of(matrix);
    const auto matrix_strides = detail::c_order_strides(matrix_shape);
    const auto starts_shape = shape_of(starts);
    const auto starts_strides = detail::c_order_strides(starts_shape);
    const auto output_shape = shape_of(output);
    const auto output_strides = detail::c_order_strides(output_shape);

    ConstArrayView<T> input_view{input.data(), input_shape, input_strides};
    ArrayView<T> output_view{output.data(), output_shape, output_strides};
    ConstArrayView<double> matrix_view{matrix.data(), matrix_shape, matrix_strides};
    ConstArrayView<std::ptrdiff_t> starts_view{starts.data(), starts_shape, starts_strides};

    {
        nb::gil_scoped_release release;
        if constexpr (D == 2) {
            transformation::affine_transform_2d<T>(
                input_view, output_view, matrix_view, starts_view, order, fill_value
            );
        } else {
            transformation::affine_transform_3d<T>(
                input_view, output_view, matrix_view, starts_view, order, fill_value
            );
        }
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
        nb::arg("output"),
        nb::arg("order"),
        nb::arg("fill_value"),
        "Apply a 2D affine transformation into a pre-allocated NumPy array."
    );
    m.def(
        name_3d,
        &affine_transform_t<3, T>,
        nb::arg("input"),
        nb::arg("matrix"),
        nb::arg("starts"),
        nb::arg("output"),
        nb::arg("order"),
        nb::arg("fill_value"),
        "Apply a 3D affine transformation into a pre-allocated NumPy array."
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
