#pragma once

#include "bioimage_cpp/array_view.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace bioimage_cpp::flow {
namespace detail {

template <std::size_t D>
void require_flow_views(
    const ConstArrayView<float> &flow,
    const ConstArrayView<std::uint8_t> &mask,
    const ArrayView<float> &density
) {
    if (flow.ndim() != static_cast<std::ptrdiff_t>(D + 1)) {
        throw std::invalid_argument(
            "flow must have ndim=" + std::to_string(D + 1) +
            ", got ndim=" + std::to_string(flow.ndim())
        );
    }
    if (mask.ndim() != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            "fg_mask must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(mask.ndim())
        );
    }
    if (density.ndim() != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            "density must have ndim=" + std::to_string(D) +
            ", got ndim=" + std::to_string(density.ndim())
        );
    }
    if (flow.shape[0] != static_cast<std::ptrdiff_t>(D)) {
        throw std::invalid_argument(
            "flow first axis must match spatial ndim=" + std::to_string(D) +
            ", got " + std::to_string(flow.shape[0])
        );
    }
    for (std::size_t axis = 0; axis < D; ++axis) {
        const auto expected = mask.shape[axis];
        if (flow.shape[axis + 1] != expected) {
            throw std::invalid_argument("flow spatial shape must match fg_mask shape");
        }
        if (density.shape[axis] != expected) {
            throw std::invalid_argument("density shape must match fg_mask shape");
        }
    }
}

template <std::size_t D>
std::ptrdiff_t flat_index(
    const std::array<std::ptrdiff_t, D> &coord,
    const std::vector<std::ptrdiff_t> &strides
) {
    std::ptrdiff_t index = 0;
    for (std::size_t axis = 0; axis < D; ++axis) {
        index += coord[axis] * strides[axis];
    }
    return index;
}

template <std::size_t D>
float sample_linear_nearest(
    const float *channel,
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<std::ptrdiff_t> &strides,
    const std::array<float, D> &position
) {
    std::array<std::ptrdiff_t, D> lower{};
    std::array<float, D> frac{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        const float pos = position[axis];
        const auto lo = static_cast<std::ptrdiff_t>(std::floor(pos));
        lower[axis] = lo;
        frac[axis] = pos - static_cast<float>(lo);
    }

    float value = 0.0f;
    constexpr std::size_t n_corners = std::size_t{1} << D;
    for (std::size_t corner = 0; corner < n_corners; ++corner) {
        std::array<std::ptrdiff_t, D> sampled{};
        float weight = 1.0f;
        for (std::size_t axis = 0; axis < D; ++axis) {
            const bool upper = ((corner >> axis) & std::size_t{1}) != 0;
            if (upper) {
                weight *= frac[axis];
                sampled[axis] = lower[axis] + 1;
            } else {
                weight *= 1.0f - frac[axis];
                sampled[axis] = lower[axis];
            }
            if (sampled[axis] < 0) {
                sampled[axis] = 0;
            } else if (sampled[axis] >= shape[axis]) {
                sampled[axis] = shape[axis] - 1;
            }
        }
        value += weight * channel[flat_index<D>(sampled, strides)];
    }
    return value;
}

template <std::size_t D>
std::array<std::ptrdiff_t, D> unravel_index(
    std::ptrdiff_t index,
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<std::ptrdiff_t> &strides
) {
    std::array<std::ptrdiff_t, D> coord{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        coord[axis] = (index / strides[axis]) % shape[axis];
    }
    return coord;
}

} // namespace detail

template <std::size_t D>
void compute_flow_density(
    const ConstArrayView<float> &flow,
    const ConstArrayView<std::uint8_t> &fg_mask,
    ArrayView<float> &density,
    const std::size_t n_iter,
    const float dt
) {
    detail::require_flow_views<D>(flow, fg_mask, density);

    const std::ptrdiff_t n_pixels = std::accumulate(
        fg_mask.shape.begin(),
        fg_mask.shape.end(),
        std::ptrdiff_t{1},
        [](const std::ptrdiff_t a, const std::ptrdiff_t b) { return a * b; }
    );
    for (std::ptrdiff_t i = 0; i < n_pixels; ++i) {
        density.data[i] = 0.0f;
    }

    std::vector<std::array<float, D>> positions;
    positions.reserve(static_cast<std::size_t>(n_pixels));
    for (std::ptrdiff_t index = 0; index < n_pixels; ++index) {
        if (fg_mask.data[index] == 0) {
            continue;
        }
        const auto coord = detail::unravel_index<D>(index, fg_mask.shape, fg_mask.strides);
        std::array<float, D> position{};
        for (std::size_t axis = 0; axis < D; ++axis) {
            position[axis] = static_cast<float>(coord[axis]);
        }
        positions.push_back(position);
    }
    if (positions.empty()) {
        return;
    }

    const std::ptrdiff_t channel_size = n_pixels;
    for (std::size_t iter = 0; iter < n_iter; ++iter) {
        for (auto &position : positions) {
            for (std::size_t axis = 0; axis < D; ++axis) {
                const float upper = static_cast<float>(fg_mask.shape[axis] - 1);
                if (position[axis] < 0.0f) {
                    position[axis] = 0.0f;
                } else if (position[axis] > upper) {
                    position[axis] = upper;
                }
            }

            std::array<float, D> step{};
            for (std::size_t axis = 0; axis < D; ++axis) {
                const float *channel = flow.data + static_cast<std::ptrdiff_t>(axis) * channel_size;
                step[axis] = detail::sample_linear_nearest<D>(
                    channel, fg_mask.shape, fg_mask.strides, position
                );
            }
            for (std::size_t axis = 0; axis < D; ++axis) {
                position[axis] += dt * step[axis];
            }
        }
    }

    for (const auto &position : positions) {
        std::array<std::ptrdiff_t, D> coord{};
        for (std::size_t axis = 0; axis < D; ++axis) {
            const float upper = static_cast<float>(fg_mask.shape[axis] - 1);
            float clipped = position[axis];
            if (clipped < 0.0f) {
                clipped = 0.0f;
            } else if (clipped > upper) {
                clipped = upper;
            }
            coord[axis] = static_cast<std::ptrdiff_t>(std::nearbyint(clipped));
        }
        density.data[detail::flat_index<D>(coord, fg_mask.strides)] += 1.0f;
    }

    for (std::ptrdiff_t index = 0; index < n_pixels; ++index) {
        if (fg_mask.data[index] == 0) {
            density.data[index] = 0.0f;
        }
    }
}

inline void compute_flow_density_2d(
    const ConstArrayView<float> &flow,
    const ConstArrayView<std::uint8_t> &fg_mask,
    ArrayView<float> &density,
    const std::size_t n_iter,
    const float dt
) {
    compute_flow_density<2>(flow, fg_mask, density, n_iter, dt);
}

inline void compute_flow_density_3d(
    const ConstArrayView<float> &flow,
    const ConstArrayView<std::uint8_t> &fg_mask,
    ArrayView<float> &density,
    const std::size_t n_iter,
    const float dt
) {
    compute_flow_density<3>(flow, fg_mask, density, n_iter, dt);
}

} // namespace bioimage_cpp::flow
