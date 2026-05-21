#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bioimage_cpp::flow {
namespace detail {

template <std::size_t D>
struct GridLayout {
    std::array<std::ptrdiff_t, D> shape{};
    std::array<std::ptrdiff_t, D> strides{};
    std::array<float, D> upper{};
};

template <std::size_t D>
GridLayout<D> make_grid_layout(
    const std::vector<std::ptrdiff_t> &shape,
    const std::vector<std::ptrdiff_t> &strides
) {
    GridLayout<D> layout;
    for (std::size_t axis = 0; axis < D; ++axis) {
        layout.shape[axis] = shape[axis];
        layout.strides[axis] = strides[axis];
        layout.upper[axis] = static_cast<float>(shape[axis] - 1);
    }
    return layout;
}

// Per-position cache of the 2^D corners used for linear interpolation. Offsets
// are in elements relative to the channel base pointer; weights sum to 1.
template <std::size_t D>
struct SamplingCorners {
    std::array<std::ptrdiff_t, (std::size_t{1} << D)> offsets{};
    std::array<float, (std::size_t{1} << D)> weights{};
};

template <std::size_t D>
SamplingCorners<D> compute_corners(
    const std::array<float, D> &position,
    const GridLayout<D> &grid
) {
    std::array<std::ptrdiff_t, D> lower{};
    std::array<float, D> frac{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        const auto lo = static_cast<std::ptrdiff_t>(std::floor(position[axis]));
        lower[axis] = lo;
        frac[axis] = position[axis] - static_cast<float>(lo);
    }

    SamplingCorners<D> corners{};
    constexpr std::size_t n_corners = std::size_t{1} << D;
    for (std::size_t corner = 0; corner < n_corners; ++corner) {
        std::ptrdiff_t offset = 0;
        float weight = 1.0f;
        for (std::size_t axis = 0; axis < D; ++axis) {
            const bool upper_side = ((corner >> axis) & std::size_t{1}) != 0;
            std::ptrdiff_t coord;
            if (upper_side) {
                weight *= frac[axis];
                coord = lower[axis] + 1;
            } else {
                weight *= 1.0f - frac[axis];
                coord = lower[axis];
            }
            if (coord < 0) {
                coord = 0;
            } else if (coord >= grid.shape[axis]) {
                coord = grid.shape[axis] - 1;
            }
            offset += coord * grid.strides[axis];
        }
        corners.offsets[corner] = offset;
        corners.weights[corner] = weight;
    }
    return corners;
}

template <std::size_t D>
inline float sample_channel(
    const float *channel,
    const SamplingCorners<D> &corners
) {
    constexpr std::size_t n_corners = std::size_t{1} << D;
    float value = 0.0f;
    for (std::size_t corner = 0; corner < n_corners; ++corner) {
        value += corners.weights[corner] * channel[corners.offsets[corner]];
    }
    return value;
}

template <std::size_t D>
inline std::ptrdiff_t round_to_flat_index(
    const std::array<float, D> &position,
    const GridLayout<D> &grid
) {
    std::ptrdiff_t flat = 0;
    for (std::size_t axis = 0; axis < D; ++axis) {
        float clipped = position[axis];
        if (clipped < 0.0f) {
            clipped = 0.0f;
        } else if (clipped > grid.upper[axis]) {
            clipped = grid.upper[axis];
        }
        const auto coord = static_cast<std::ptrdiff_t>(std::nearbyint(clipped));
        flat += coord * grid.strides[axis];
    }
    return flat;
}

} // namespace detail

// Preconditions (validated in the binding layer):
//   * flow.ndim() == D + 1, flow.shape[0] == D, flow.shape[1..] == fg_mask.shape
//   * fg_mask.ndim() == D and density.shape == fg_mask.shape
//   * flow / fg_mask / density are C-contiguous
//   * n_iter >= 0, dt finite and >= 0
template <std::size_t D>
void compute_flow_density(
    const ConstArrayView<float> &flow,
    const ConstArrayView<std::uint8_t> &fg_mask,
    ArrayView<float> &density,
    const std::size_t n_iter,
    const float dt,
    const std::size_t number_of_threads = 1
) {
    const auto grid = detail::make_grid_layout<D>(fg_mask.shape, fg_mask.strides);

    std::ptrdiff_t n_pixels = 1;
    for (std::size_t axis = 0; axis < D; ++axis) {
        n_pixels *= grid.shape[axis];
    }
    for (std::ptrdiff_t i = 0; i < n_pixels; ++i) {
        density.data[i] = 0.0f;
    }

    std::vector<std::array<float, D>> positions;
    for (std::ptrdiff_t index = 0; index < n_pixels; ++index) {
        if (fg_mask.data[index] == 0) {
            continue;
        }
        std::array<float, D> position{};
        std::ptrdiff_t remainder = index;
        for (std::size_t axis = 0; axis < D; ++axis) {
            position[axis] = static_cast<float>(remainder / grid.strides[axis]);
            remainder = remainder % grid.strides[axis];
        }
        positions.push_back(position);
    }
    if (positions.empty()) {
        return;
    }

    const std::ptrdiff_t channel_stride = flow.strides[0];
    std::array<const float *, D> channels{};
    for (std::size_t axis = 0; axis < D; ++axis) {
        channels[axis] = flow.data + static_cast<std::ptrdiff_t>(axis) * channel_stride;
    }

    const auto n_threads = ::bioimage_cpp::detail::normalize_thread_count(
        number_of_threads, positions.size()
    );

    for (std::size_t iter = 0; iter < n_iter; ++iter) {
        ::bioimage_cpp::detail::parallel_for_chunks(
            n_threads,
            positions.size(),
            [&](const std::size_t, const std::size_t begin, const std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    auto &position = positions[i];
                    for (std::size_t axis = 0; axis < D; ++axis) {
                        if (position[axis] < 0.0f) {
                            position[axis] = 0.0f;
                        } else if (position[axis] > grid.upper[axis]) {
                            position[axis] = grid.upper[axis];
                        }
                    }
                    const auto corners = detail::compute_corners<D>(position, grid);
                    for (std::size_t axis = 0; axis < D; ++axis) {
                        const float step = detail::sample_channel<D>(channels[axis], corners);
                        position[axis] += dt * step;
                    }
                }
            }
        );
    }

    for (const auto &position : positions) {
        density.data[detail::round_to_flat_index<D>(position, grid)] += 1.0f;
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
    const float dt,
    const std::size_t number_of_threads = 1
) {
    compute_flow_density<2>(flow, fg_mask, density, n_iter, dt, number_of_threads);
}

inline void compute_flow_density_3d(
    const ConstArrayView<float> &flow,
    const ConstArrayView<std::uint8_t> &fg_mask,
    ArrayView<float> &density,
    const std::size_t n_iter,
    const float dt,
    const std::size_t number_of_threads = 1
) {
    compute_flow_density<3>(flow, fg_mask, density, n_iter, dt, number_of_threads);
}

} // namespace bioimage_cpp::flow
