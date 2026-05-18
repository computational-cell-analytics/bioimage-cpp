#pragma once

#include "bioimage_cpp/array_view.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace bioimage_cpp::transformation {

namespace detail {

template <std::size_t D>
using Coordinate = std::array<std::ptrdiff_t, D>;

template <std::size_t D>
using FloatingCoordinate = std::array<double, D>;

template <std::size_t D>
void require_shape(const ConstArrayView<double> &matrix, const char *name) {
    if (matrix.ndim() != 2 || matrix.shape[0] != static_cast<std::ptrdiff_t>(D) ||
        matrix.shape[1] != static_cast<std::ptrdiff_t>(D + 1)) {
        throw std::invalid_argument(
            std::string(name) + " must have shape (" + std::to_string(D) + ", " +
            std::to_string(D + 1) + ")"
        );
    }
}

template <std::size_t D>
void require_shape(const ConstArrayView<std::ptrdiff_t> &starts, const char *name) {
    if (starts.ndim() != 1 || starts.shape[0] != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            std::string(name) + " must have shape (" + std::to_string(D) + ",)"
        );
    }
}

template <std::size_t D, class T>
void require_view_shape(const ConstArrayView<T> &input, const ArrayView<T> &output) {
    if (input.ndim() != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            "input must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(input.ndim())
        );
    }
    if (output.ndim() != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            "output must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(output.ndim())
        );
    }
}

template <std::size_t D, class T>
std::ptrdiff_t offset_of(const ArrayView<T> &view, const Coordinate<D> &coord) {
    std::ptrdiff_t offset = 0;
    for (std::size_t axis = 0; axis < D; ++axis) {
        offset += coord[axis] * view.strides[axis];
    }
    return offset;
}

template <std::size_t D, class T>
std::ptrdiff_t offset_of(const ConstArrayView<T> &view, const Coordinate<D> &coord) {
    std::ptrdiff_t offset = 0;
    for (std::size_t axis = 0; axis < D; ++axis) {
        offset += coord[axis] * view.strides[axis];
    }
    return offset;
}

template <std::size_t D, class T>
bool is_inside_index(const ConstArrayView<T> &input, const Coordinate<D> &coord) {
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (coord[axis] < 0 || coord[axis] >= input.shape[axis]) return false;
    }
    return true;
}

template <std::size_t D, class T>
bool is_inside_coordinate(const ConstArrayView<T> &input, const FloatingCoordinate<D> &coord) {
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (coord[axis] < 0.0 || coord[axis] > static_cast<double>(input.shape[axis] - 1)) {
            return false;
        }
    }
    return true;
}

template <std::size_t D>
FloatingCoordinate<D> transform_coordinate(
    const Coordinate<D> &output_coord,
    const ConstArrayView<double> &matrix
) {
    FloatingCoordinate<D> input_coord{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        double value = matrix.data[axis * matrix.strides[0] + static_cast<std::ptrdiff_t>(D) * matrix.strides[1]];
        for (std::size_t inner = 0; inner < D; ++inner) {
            value += matrix.data[axis * matrix.strides[0] + inner * matrix.strides[1]] *
                     static_cast<double>(output_coord[inner]);
        }
        input_coord[axis] = value;
    }
    return input_coord;
}

inline double cubic_weight(const double x) {
    // Catmull-Rom / Keys cubic convolution with a = -0.5.
    const double ax = std::abs(x);
    if (ax < 1.0) {
        return (1.5 * ax - 2.5) * ax * ax + 1.0;
    }
    if (ax < 2.0) {
        return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0;
    }
    return 0.0;
}

template <std::size_t D, class T>
double sample_or_fill(
    const ConstArrayView<T> &input,
    const Coordinate<D> &coord,
    const T fill_value
) {
    if (!is_inside_index(input, coord)) return static_cast<double>(fill_value);
    return static_cast<double>(input.data[offset_of(input, coord)]);
}

template <std::size_t D, class T>
double interpolate_nearest(
    const ConstArrayView<T> &input,
    const FloatingCoordinate<D> &coord,
    const T fill_value
) {
    Coordinate<D> nearest{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        nearest[axis] = static_cast<std::ptrdiff_t>(std::floor(coord[axis] + 0.5));
    }
    return sample_or_fill(input, nearest, fill_value);
}

template <std::size_t D, class T>
double interpolate_linear_impl(
    const ConstArrayView<T> &input,
    const T fill_value,
    Coordinate<D> &sample_coord,
    const Coordinate<D> &lower,
    const std::array<double, D> &fraction,
    const std::size_t axis,
    const double weight
) {
    if (axis == D) {
        return weight * sample_or_fill(input, sample_coord, fill_value);
    }

    sample_coord[axis] = lower[axis];
    double value = interpolate_linear_impl(
        input, fill_value, sample_coord, lower, fraction, axis + 1,
        weight * (1.0 - fraction[axis])
    );

    sample_coord[axis] = lower[axis] + 1;
    value += interpolate_linear_impl(
        input, fill_value, sample_coord, lower, fraction, axis + 1,
        weight * fraction[axis]
    );
    return value;
}

template <std::size_t D, class T>
double interpolate_linear(
    const ConstArrayView<T> &input,
    const FloatingCoordinate<D> &coord,
    const T fill_value
) {
    if (!is_inside_coordinate(input, coord)) return static_cast<double>(fill_value);

    Coordinate<D> lower{};
    Coordinate<D> sample_coord{};
    std::array<double, D> fraction{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        const double floored = std::floor(coord[axis]);
        lower[axis] = static_cast<std::ptrdiff_t>(floored);
        fraction[axis] = coord[axis] - floored;
    }
    return interpolate_linear_impl(
        input, fill_value, sample_coord, lower, fraction, 0, 1.0
    );
}

template <std::size_t D, class T>
double interpolate_cubic_impl(
    const ConstArrayView<T> &input,
    Coordinate<D> &sample_coord,
    const std::array<std::array<std::ptrdiff_t, 4>, D> &indices,
    const std::array<std::array<double, 4>, D> &weights,
    const T fill_value,
    const std::size_t axis,
    const double weight
) {
    if (axis == D) {
        return weight * sample_or_fill(input, sample_coord, fill_value);
    }

    double value = 0.0;
    for (std::size_t k = 0; k < 4; ++k) {
        sample_coord[axis] = indices[axis][k];
        value += interpolate_cubic_impl(
            input, sample_coord, indices, weights, fill_value, axis + 1,
            weight * weights[axis][k]
        );
    }
    return value;
}

template <std::size_t D, class T>
double interpolate_cubic(
    const ConstArrayView<T> &input,
    const FloatingCoordinate<D> &coord,
    const T fill_value
) {
    if (!is_inside_coordinate(input, coord)) return static_cast<double>(fill_value);

    std::array<std::array<std::ptrdiff_t, 4>, D> indices{};
    std::array<std::array<double, 4>, D> weights{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        const auto base = static_cast<std::ptrdiff_t>(std::floor(coord[axis]));
        for (std::size_t k = 0; k < 4; ++k) {
            const auto index = base + static_cast<std::ptrdiff_t>(k) - 1;
            indices[axis][k] = index;
            weights[axis][k] = cubic_weight(coord[axis] - static_cast<double>(index));
        }
    }

    Coordinate<D> sample_coord{};
    return interpolate_cubic_impl(input, sample_coord, indices, weights, fill_value, 0, 1.0);
}

template <class T>
T cast_output(const double value) {
    return static_cast<T>(value);
}

template <std::size_t D, class T>
void advance_coordinate(Coordinate<D> &coord, const ArrayView<T> &output) {
    for (std::size_t axis = D; axis-- > 0;) {
        ++coord[axis];
        if (coord[axis] < output.shape[axis]) return;
        coord[axis] = 0;
    }
}

} // namespace detail

template <std::size_t D, class T>
void affine_transform(
    const ConstArrayView<T> &input,
    ArrayView<T> &output,
    const ConstArrayView<double> &matrix,
    const ConstArrayView<std::ptrdiff_t> &starts,
    const int order,
    const T fill_value
) {
    detail::require_view_shape<D>(input, output);
    detail::require_shape<D>(matrix, "matrix");
    detail::require_shape<D>(starts, "starts");
    if (order != 0 && order != 1 && order != 3) {
        throw std::invalid_argument("order must be 0, 1 or 3, got " + std::to_string(order));
    }

    std::ptrdiff_t total = 1;
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (output.shape[axis] < 0) {
            throw std::invalid_argument("output shape must be non-negative");
        }
        total *= output.shape[axis];
    }
    if (total == 0) return;

    detail::Coordinate<D> local_coord{};
    for (std::ptrdiff_t linear = 0; linear < total; ++linear) {
        detail::Coordinate<D> output_coord{};
        for (std::size_t axis = 0; axis < D; ++axis) {
            output_coord[axis] = starts.data[axis * starts.strides[0]] + local_coord[axis];
        }

        const auto input_coord = detail::transform_coordinate(output_coord, matrix);
        double value = 0.0;
        if (order == 0) {
            value = detail::interpolate_nearest(input, input_coord, fill_value);
        } else if (order == 1) {
            value = detail::interpolate_linear(input, input_coord, fill_value);
        } else {
            value = detail::interpolate_cubic(input, input_coord, fill_value);
        }

        output.data[detail::offset_of(output, local_coord)] = detail::cast_output<T>(value);
        detail::advance_coordinate(local_coord, output);
    }
}

} // namespace bioimage_cpp::transformation
