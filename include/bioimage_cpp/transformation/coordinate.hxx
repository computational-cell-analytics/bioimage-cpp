#pragma once

// Coordinate-based resampling (the analogue of ``scipy.ndimage.map_coordinates``). For every output
// voxel the source coordinate to sample is read from an explicit ``coordinates`` array, instead of
// being computed from an affine matrix. The per-voxel interpolation reuses the affine samplers
// (``detail::sample_2d`` / ``detail::sample_3d``) so the interpolation backend lives in one place.
//
// This kernel is pure and in-memory (NumPy in, NumPy out); reading the source data and producing the
// coordinate (deformation) field are the caller's responsibility.

#include "bioimage_cpp/transformation/affine.hxx"

namespace bioimage_cpp::transformation {

namespace detail {

// Validate that ``coordinates`` is a ``(D, *output_shape)`` field, i.e. it carries one source
// coordinate per output voxel along its leading axis.
template <std::size_t D, class T>
void require_coordinates(const ConstArrayView<double> &coordinates, const ArrayView<T> &output) {
    if (coordinates.ndim() != static_cast<std::ptrdiff_t>(D + 1)) {
        throw std::invalid_argument(
            "coordinates must have ndim=" + std::to_string(D + 1) +
            ", got ndim=" + std::to_string(coordinates.ndim())
        );
    }
    if (coordinates.shape[0] != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            "coordinates.shape[0] must equal the data dimension " + std::to_string(D) +
            ", got " + std::to_string(coordinates.shape[0])
        );
    }
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (coordinates.shape[axis + 1] != output.shape[axis]) {
            throw std::invalid_argument(
                "coordinates spatial shape (coordinates.shape[1:]) must match the output shape"
            );
        }
    }
}

} // namespace detail

// ----- 2D entry point -------------------------------------------------------

template <class T>
void map_coordinates_2d(
    const ConstArrayView<T> &input,
    ArrayView<T> &output,
    const ConstArrayView<double> &coordinates,
    const int order,
    const T fill_value
) {
    detail::require_views<2, T>(input, output);
    detail::require_coordinates<2, T>(coordinates, output);
    if (order < 0 || order > 5) {
        throw std::invalid_argument(
            "order must be in 0..5, got " + std::to_string(order)
        );
    }

    const auto out_h = output.shape[0];
    const auto out_w = output.shape[1];
    if (out_h == 0 || out_w == 0) return;

    const auto in_h = input.shape[0];
    const auto in_w = input.shape[1];

    // coordinates is C-contiguous with shape (2, out_h, out_w), so axis-d coordinates form a
    // contiguous block of n_out doubles starting at d * n_out.
    const std::ptrdiff_t n_out = out_h * out_w;
    const double *cy = coordinates.data;
    const double *cx = coordinates.data + n_out;

    const double fill = static_cast<double>(fill_value);
    const T *in_data = input.data;
    T *out_ptr = output.data;

    for (std::ptrdiff_t p = 0; p < n_out; ++p) {
        const double value = detail::sample_2d(in_data, in_h, in_w, cy[p], cx[p], order, fill);
        out_ptr[p] = detail::to_output<T>(value);
    }
}

// ----- 3D entry point -------------------------------------------------------

template <class T>
void map_coordinates_3d(
    const ConstArrayView<T> &input,
    ArrayView<T> &output,
    const ConstArrayView<double> &coordinates,
    const int order,
    const T fill_value
) {
    detail::require_views<3, T>(input, output);
    detail::require_coordinates<3, T>(coordinates, output);
    if (order < 0 || order > 5) {
        throw std::invalid_argument(
            "order must be in 0..5, got " + std::to_string(order)
        );
    }

    const auto out_d = output.shape[0];
    const auto out_h = output.shape[1];
    const auto out_w = output.shape[2];
    if (out_d == 0 || out_h == 0 || out_w == 0) return;

    const auto in_d = input.shape[0];
    const auto in_h = input.shape[1];
    const auto in_w = input.shape[2];

    // coordinates is C-contiguous with shape (3, out_d, out_h, out_w).
    const std::ptrdiff_t n_out = out_d * out_h * out_w;
    const double *cz = coordinates.data;
    const double *cy = coordinates.data + n_out;
    const double *cx = coordinates.data + 2 * n_out;

    const double fill = static_cast<double>(fill_value);
    const T *in_data = input.data;
    T *out_ptr = output.data;

    for (std::ptrdiff_t p = 0; p < n_out; ++p) {
        const double value = detail::sample_3d(in_data, in_d, in_h, in_w,
                                               cz[p], cy[p], cx[p], order, fill);
        out_ptr[p] = detail::to_output<T>(value);
    }
}

} // namespace bioimage_cpp::transformation
