#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/grid.hxx"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::distance {

namespace detail {

inline std::ptrdiff_t number_of_elements(const std::vector<std::ptrdiff_t> &shape) {
    std::ptrdiff_t n = 1;
    for (const auto axis_size : shape) {
        n *= axis_size;
    }
    return n;
}

inline void flat_index_to_coords(
    const std::ptrdiff_t index,
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<std::ptrdiff_t> &strides,
    std::vector<std::ptrdiff_t> &coords
) {
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        coords[axis] = (index / strides[axis]) % shape[axis];
    }
}

inline double sampled_squared_distance(
    const std::vector<std::ptrdiff_t> &coords,
    const std::vector<std::ptrdiff_t> &target,
    const std::vector<double> &sampling
) {
    double squared = 0.0;
    for (std::size_t axis = 0; axis < coords.size(); ++axis) {
        const double diff = static_cast<double>(target[axis] - coords[axis]) * sampling[axis];
        squared += diff * diff;
    }
    return squared;
}

} // namespace detail

struct DistanceTransformResult {
    ArrayView<float> distances;
    ArrayView<std::int32_t> indices;
    ArrayView<float> vectors;
};

inline void distance_transform(
    const ConstArrayView<std::uint8_t> &input,
    const std::vector<double> &sampling,
    const DistanceTransformResult &result
) {
    const auto ndim = input.ndim();
    if (ndim < 1) {
        throw std::invalid_argument("input must have ndim >= 1");
    }
    if (sampling.size() != static_cast<std::size_t>(ndim)) {
        throw std::invalid_argument(
            "sampling must have length matching input ndim, got ndim=" +
            std::to_string(ndim) + ", sampling length=" + std::to_string(sampling.size())
        );
    }
    for (std::size_t axis = 0; axis < sampling.size(); ++axis) {
        if (!(std::isfinite(sampling[axis]) && sampling[axis] > 0.0)) {
            throw std::invalid_argument(
                "sampling values must be positive and finite, got sampling[" +
                std::to_string(axis) + "]=" + std::to_string(sampling[axis])
            );
        }
    }

    const auto n = detail::number_of_elements(input.shape);
    if (n == 0) {
        return;
    }

    const auto strides = bioimage_cpp::detail::c_order_strides(input.shape);
    std::vector<std::vector<std::ptrdiff_t>> targets;
    std::vector<std::ptrdiff_t> coords(static_cast<std::size_t>(ndim), 0);

    for (std::ptrdiff_t index = 0; index < n; ++index) {
        if (input.data[index] == 0) {
            detail::flat_index_to_coords(index, input.shape, strides, coords);
            targets.push_back(coords);
        }
    }

    const bool use_virtual_target = targets.empty();
    const bool has_distances = result.distances.data != nullptr;
    const bool has_indices = result.indices.data != nullptr;
    const bool has_vectors = result.vectors.data != nullptr;

    std::vector<std::ptrdiff_t> best(static_cast<std::size_t>(ndim), 0);
    if (use_virtual_target) {
        best[0] = -1;
    }

    for (std::ptrdiff_t index = 0; index < n; ++index) {
        detail::flat_index_to_coords(index, input.shape, strides, coords);

        if (input.data[index] == 0 && !use_virtual_target) {
            best = coords;
            if (has_distances) {
                result.distances.data[index] = 0.0f;
            }
        } else if (use_virtual_target) {
            best.assign(static_cast<std::size_t>(ndim), 0);
            best[0] = -1;
            if (has_distances) {
                const auto dist = std::sqrt(detail::sampled_squared_distance(coords, best, sampling));
                result.distances.data[index] = static_cast<float>(dist);
            }
        } else {
            double best_squared = std::numeric_limits<double>::infinity();
            for (const auto &target : targets) {
                const auto squared = detail::sampled_squared_distance(coords, target, sampling);
                if (squared < best_squared) {
                    best_squared = squared;
                    best = target;
                }
            }
            if (has_distances) {
                result.distances.data[index] = static_cast<float>(std::sqrt(best_squared));
            }
        }

        if (has_indices) {
            for (std::ptrdiff_t axis = 0; axis < ndim; ++axis) {
                result.indices.data[axis * n + index] = static_cast<std::int32_t>(best[axis]);
            }
        }
        if (has_vectors) {
            for (std::ptrdiff_t axis = 0; axis < ndim; ++axis) {
                const auto diff =
                    static_cast<double>(best[axis] - coords[axis]) *
                    sampling[static_cast<std::size_t>(axis)];
                result.vectors.data[index * ndim + axis] = static_cast<float>(diff);
            }
        }
    }
}

} // namespace bioimage_cpp::distance
