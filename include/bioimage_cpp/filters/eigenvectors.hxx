#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace bioimage_cpp::filters {

namespace detail {

template <class T, std::size_t D>
inline void canonicalize_eigenvector(std::array<T, D> &vector) {
    std::size_t pivot = 0;
    T largest = std::abs(vector[0]);
    for (std::size_t axis = 1; axis < D; ++axis) {
        const T magnitude = std::abs(vector[axis]);
        if (magnitude > largest) {
            largest = magnitude;
            pivot = axis;
        }
    }
    if (vector[pivot] < T{0}) {
        for (auto &value : vector) {
            value = -value;
        }
    }
}

template <class T, std::size_t D>
inline void jacobi_rotation(
    std::array<std::array<T, D>, D> &matrix,
    std::array<std::array<T, D>, D> &vectors,
    const std::size_t p,
    const std::size_t q
) {
    const T off_diagonal = matrix[p][q];
    if (off_diagonal == T{0}) {
        return;
    }

    const T tau = (matrix[q][q] - matrix[p][p]) / (T{2} * off_diagonal);
    const T sign = tau < T{0} ? T{-1} : T{1};
    const T tangent = sign / (std::abs(tau) + std::hypot(T{1}, tau));
    const T cosine = T{1} / std::sqrt(T{1} + tangent * tangent);
    const T sine = tangent * cosine;

    const T diagonal_p = matrix[p][p];
    const T diagonal_q = matrix[q][q];
    matrix[p][p] = diagonal_p - tangent * off_diagonal;
    matrix[q][q] = diagonal_q + tangent * off_diagonal;
    matrix[p][q] = T{0};
    matrix[q][p] = T{0};

    for (std::size_t axis = 0; axis < D; ++axis) {
        if (axis == p || axis == q) {
            continue;
        }
        const T value_p = matrix[axis][p];
        const T value_q = matrix[axis][q];
        const T rotated_p = cosine * value_p - sine * value_q;
        const T rotated_q = sine * value_p + cosine * value_q;
        matrix[axis][p] = rotated_p;
        matrix[p][axis] = rotated_p;
        matrix[axis][q] = rotated_q;
        matrix[q][axis] = rotated_q;
    }

    for (std::size_t axis = 0; axis < D; ++axis) {
        const T value_p = vectors[axis][p];
        const T value_q = vectors[axis][q];
        vectors[axis][p] = cosine * value_p - sine * value_q;
        vectors[axis][q] = sine * value_p + cosine * value_q;
    }
}

template <class T, std::size_t D>
inline std::array<T, D> symmetric_selected_eigenvector_descending(
    std::array<std::array<T, D>, D> matrix,
    const std::size_t index
) {
    T scale = T{0};
    for (const auto &row : matrix) {
        for (const T value : row) {
            scale = std::max(scale, std::abs(value));
        }
    }

    if (scale == T{0}) {
        std::array<T, D> result{};
        result[index] = T{1};
        return result;
    }

    if (scale < std::numeric_limits<T>::min()) {
        for (auto &row : matrix) {
            for (auto &value : row) {
                value /= scale;
            }
        }
    } else {
        const T inverse_scale = T{1} / scale;
        for (auto &row : matrix) {
            for (auto &value : row) {
                value *= inverse_scale;
            }
        }
    }

    std::array<std::array<T, D>, D> vectors{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        vectors[axis][axis] = T{1};
    }

    constexpr std::size_t kMaxSweeps = 12;
    const T tolerance = T{8} * std::numeric_limits<T>::epsilon();
    for (std::size_t sweep = 0; sweep < kMaxSweeps; ++sweep) {
        T off_diagonal_squared = T{0};
        for (std::size_t row = 0; row < D; ++row) {
            for (std::size_t column = row + 1; column < D; ++column) {
                off_diagonal_squared +=
                    T{2} * matrix[row][column] * matrix[row][column];
            }
        }
        T diagonal_squared = T{0};
        for (std::size_t axis = 0; axis < D; ++axis) {
            diagonal_squared += matrix[axis][axis] * matrix[axis][axis];
        }
        if (std::sqrt(off_diagonal_squared) <=
            tolerance * std::max(T{1}, std::sqrt(diagonal_squared))) {
            break;
        }

        for (std::size_t p = 0; p < D; ++p) {
            for (std::size_t q = p + 1; q < D; ++q) {
                jacobi_rotation(matrix, vectors, p, q);
            }
        }
    }

    std::array<std::size_t, D> order{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        order[axis] = axis;
    }
    std::stable_sort(order.begin(), order.end(), [&](const auto first, const auto second) {
        return matrix[first][first] > matrix[second][second];
    });

    std::array<T, D> result{};
    T squared_norm = T{0};
    const std::size_t column = order[index];
    for (std::size_t axis = 0; axis < D; ++axis) {
        result[axis] = vectors[axis][column];
        squared_norm += result[axis] * result[axis];
    }
    const T inverse_norm = T{1} / std::sqrt(squared_norm);
    for (auto &value : result) {
        value *= inverse_norm;
    }
    canonicalize_eigenvector(result);
    return result;
}

} // namespace detail

// Select one eigenvector from 2x2 symmetric matrices in descending eigenvalue
// order. Inputs use upper-triangle structure-of-arrays layout.
template <class T>
inline void ev2_symmetric_eigenvector_descending(
    const T *__restrict a00,
    const T *__restrict a01,
    const T *__restrict a11,
    const std::size_t index,
    const std::uint8_t *__restrict mask,
    T *__restrict out,
    const std::ptrdiff_t n
) {
    if (index >= 2) {
        throw std::invalid_argument("eigenvector index must be in [0, 2)");
    }
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        if (mask != nullptr && mask[i] == 0) {
            out[2 * i] = T{0};
            out[2 * i + 1] = T{0};
            continue;
        }
        const auto vector = detail::symmetric_selected_eigenvector_descending<T, 2>(
            {{{a00[i], a01[i]}, {a01[i], a11[i]}}}, index
        );
        out[2 * i] = vector[0];
        out[2 * i + 1] = vector[1];
    }
}

// Select one eigenvector from 3x3 symmetric matrices in descending eigenvalue
// order. Inputs use upper-triangle structure-of-arrays layout.
template <class T>
inline void ev3_symmetric_eigenvector_descending(
    const T *__restrict a00,
    const T *__restrict a01,
    const T *__restrict a02,
    const T *__restrict a11,
    const T *__restrict a12,
    const T *__restrict a22,
    const std::size_t index,
    const std::uint8_t *__restrict mask,
    T *__restrict out,
    const std::ptrdiff_t n
) {
    if (index >= 3) {
        throw std::invalid_argument("eigenvector index must be in [0, 3)");
    }
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        if (mask != nullptr && mask[i] == 0) {
            out[3 * i] = T{0};
            out[3 * i + 1] = T{0};
            out[3 * i + 2] = T{0};
            continue;
        }
        const auto vector = detail::symmetric_selected_eigenvector_descending<T, 3>(
            {{{a00[i], a01[i], a02[i]},
              {a01[i], a11[i], a12[i]},
              {a02[i], a12[i], a22[i]}}},
            index
        );
        out[3 * i] = vector[0];
        out[3 * i + 1] = vector[1];
        out[3 * i + 2] = vector[2];
    }
}

} // namespace bioimage_cpp::filters
