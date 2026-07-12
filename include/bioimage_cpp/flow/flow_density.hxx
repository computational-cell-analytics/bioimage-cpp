#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"
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
        // Round half up, matching the nearest-neighbor convention in
        // transformation/affine.hxx and segmentation/watershed.hxx.
        // std::nearbyint would honor the FP rounding mode (round-half-to-even).
        const auto coord = static_cast<std::ptrdiff_t>(std::floor(clipped + 0.5f));
        flat += coord * grid.strides[axis];
    }
    return flat;
}

template <std::size_t D>
inline bool position_is_in_mask(
    const std::array<float, D> &position,
    const GridLayout<D> &grid,
    const std::uint8_t *mask
) {
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (position[axis] < 0.0f || position[axis] > grid.upper[axis]) {
            return false;
        }
    }
    return mask[round_to_flat_index(position, grid)] != 0;
}

} // namespace detail

enum class IntegrationMethod {
    Euler,
    RK2,
};

// Preconditions (validated in the binding layer):
//   * flow.ndim() == D + 1, flow.shape[0] == D, flow.shape[1..] == fg_mask.shape
//   * fg_mask.ndim() == D and density.shape == fg_mask.shape
//   * flow / fg_mask / density are C-contiguous
//   * n_iter >= 0, dt finite and >= 0, tol >= 0
template <std::size_t D>
void compute_flow_density(
    const ConstArrayView<float> &flow,
    const ConstArrayView<std::uint8_t> &fg_mask,
    ArrayView<float> &density,
    const std::size_t n_iter,
    const float dt,
    const float tol = 0.0f,
    const IntegrationMethod method = IntegrationMethod::Euler,
    const bool restrict_to_mask = false,
    const std::size_t number_of_threads = 1
) {
    BIOIMAGE_PROFILE_INIT(profiler);

    const auto grid = detail::make_grid_layout<D>(fg_mask.shape, fg_mask.strides);

    std::ptrdiff_t n_pixels = 1;
    for (std::size_t axis = 0; axis < D; ++axis) {
        n_pixels *= grid.shape[axis];
    }

    std::vector<std::array<float, D>> positions;
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "init");
        for (std::ptrdiff_t i = 0; i < n_pixels; ++i) {
            density.data[i] = 0.0f;
        }
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
    }
    if (positions.empty()) {
        BIOIMAGE_PROFILE_REPORT(profiler);
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

    std::vector<std::uint8_t> alive(positions.size(), 1);
    const bool use_rk2 = (method == IntegrationMethod::RK2);
    const bool check_convergence = (tol > 0.0f);

    auto clip_position = [&grid](std::array<float, D> &p) {
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (p[axis] < 0.0f) {
                p[axis] = 0.0f;
            } else if (p[axis] > grid.upper[axis]) {
                p[axis] = grid.upper[axis];
            }
        }
    };

    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "iter_loop");
        for (std::size_t iter = 0; iter < n_iter; ++iter) {
            ::bioimage_cpp::detail::parallel_for_chunks(
                n_threads,
                positions.size(),
                [&](const std::size_t, const std::size_t begin, const std::size_t end) {
                    for (std::size_t i = begin; i < end; ++i) {
                        if (alive[i] == 0) {
                            continue;
                        }
                        auto &position = positions[i];
                        clip_position(position);

                        const auto corners = detail::compute_corners<D>(position, grid);
                        std::array<float, D> step{};
                        for (std::size_t axis = 0; axis < D; ++axis) {
                            step[axis] = detail::sample_channel<D>(channels[axis], corners);
                        }

                        if (use_rk2) {
                            std::array<float, D> mid{};
                            for (std::size_t axis = 0; axis < D; ++axis) {
                                mid[axis] = position[axis] + 0.5f * dt * step[axis];
                            }
                            clip_position(mid);
                            const auto mid_corners = detail::compute_corners<D>(mid, grid);
                            for (std::size_t axis = 0; axis < D; ++axis) {
                                step[axis] = detail::sample_channel<D>(channels[axis], mid_corners);
                            }
                        }

                        float max_step = 0.0f;
                        for (std::size_t axis = 0; axis < D; ++axis) {
                            const float abs_step = std::fabs(dt * step[axis]);
                            if (abs_step > max_step) {
                                max_step = abs_step;
                            }
                        }
                        if (check_convergence && max_step < tol) {
                            alive[i] = 0;
                            continue;
                        }
                        auto proposed = position;
                        for (std::size_t axis = 0; axis < D; ++axis) {
                            proposed[axis] += dt * step[axis];
                        }
                        // A particle whose proposed endpoint leaves the
                        // foreground is frozen at its last in-mask position
                        // (only the endpoint is mask-tested, not the RK2
                        // midpoint). Every alive particle is seeded in the mask
                        // and only commits in-mask endpoints, so its current
                        // position is always a valid last position and no
                        // separate start-of-step mask test is needed.
                        if (restrict_to_mask && !detail::position_is_in_mask<D>(
                            proposed, grid, fg_mask.data
                        )) {
                            alive[i] = 0;
                            continue;
                        }
                        position = proposed;
                    }
                }
            );

            if (check_convergence || restrict_to_mask) {
                std::size_t still_alive = 0;
                for (const auto a : alive) {
                    still_alive += a;
                }
                if (still_alive == 0) {
                    break;
                }
            }
        }
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "scatter");
        for (const auto &position : positions) {
            density.data[detail::round_to_flat_index<D>(position, grid)] += 1.0f;
        }
    }

    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "mask_zero");
        for (std::ptrdiff_t index = 0; index < n_pixels; ++index) {
            if (fg_mask.data[index] == 0) {
                density.data[index] = 0.0f;
            }
        }
    }

    BIOIMAGE_PROFILE_REPORT(profiler);
}

inline void compute_flow_density_2d(
    const ConstArrayView<float> &flow,
    const ConstArrayView<std::uint8_t> &fg_mask,
    ArrayView<float> &density,
    const std::size_t n_iter,
    const float dt,
    const float tol,
    const IntegrationMethod method,
    const bool restrict_to_mask,
    const std::size_t number_of_threads = 1
) {
    compute_flow_density<2>(
        flow, fg_mask, density, n_iter, dt, tol, method, restrict_to_mask, number_of_threads
    );
}

inline void compute_flow_density_3d(
    const ConstArrayView<float> &flow,
    const ConstArrayView<std::uint8_t> &fg_mask,
    ArrayView<float> &density,
    const std::size_t n_iter,
    const float dt,
    const float tol,
    const IntegrationMethod method,
    const bool restrict_to_mask,
    const std::size_t number_of_threads = 1
) {
    compute_flow_density<3>(
        flow, fg_mask, density, n_iter, dt, tol, method, restrict_to_mask, number_of_threads
    );
}

} // namespace bioimage_cpp::flow
