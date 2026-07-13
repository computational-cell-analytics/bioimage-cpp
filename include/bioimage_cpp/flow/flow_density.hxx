#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(BIOIMAGE_FLOW_FMA_DISPATCH) && defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#endif

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

// Linearly interpolate all D flow channels at `position` and write the result
// to `out`. This is explicit bilinear (D==2) / trilinear (D==3) sampling rather
// than a generic 2^D corner table: the lower/upper index and fractional weight
// are computed once per axis and shared across channels, and the nested lerps
// avoid materializing 2^D product weights and offsets.
//
// Precondition: 0 <= position[axis] <= shape[axis]-1 for every axis (callers
// clip before sampling). Because the coordinate is nonnegative, truncation
// equals std::floor, so the integer cast is exact; this matters because on the
// portable (no -march, SSE2) wheel build std::floor is an inlined multi-branch
// software routine while the cast is a single instruction. At an upper boundary
// the lower and upper index coincide, matching nearest-boundary behavior.
template <std::size_t D>
inline void sample_flow(
    const std::array<const float *, D> &channels,
    const std::array<float, D> &position,
    const GridLayout<D> &grid,
    std::array<float, D> &out
) {
    if constexpr (D == 2) {
        const std::ptrdiff_t sy = grid.strides[0];
        const std::ptrdiff_t sx = grid.strides[1];
        const std::ptrdiff_t y0 = static_cast<std::ptrdiff_t>(position[0]);
        const std::ptrdiff_t x0 = static_cast<std::ptrdiff_t>(position[1]);
        const float fy = position[0] - static_cast<float>(y0);
        const float fx = position[1] - static_cast<float>(x0);
        const std::ptrdiff_t y1 = (y0 + 1 < grid.shape[0]) ? y0 + 1 : y0;
        const std::ptrdiff_t x1 = (x0 + 1 < grid.shape[1]) ? x0 + 1 : x0;
        const std::ptrdiff_t o00 = y0 * sy + x0 * sx;
        const std::ptrdiff_t o01 = y0 * sy + x1 * sx;
        const std::ptrdiff_t o10 = y1 * sy + x0 * sx;
        const std::ptrdiff_t o11 = y1 * sy + x1 * sx;
        for (std::size_t axis = 0; axis < D; ++axis) {
            const float *c = channels[axis];
            const float top = c[o00] + fx * (c[o01] - c[o00]);
            const float bot = c[o10] + fx * (c[o11] - c[o10]);
            out[axis] = top + fy * (bot - top);
        }
    } else {
        const std::ptrdiff_t sz = grid.strides[0];
        const std::ptrdiff_t sy = grid.strides[1];
        const std::ptrdiff_t sx = grid.strides[2];
        const std::ptrdiff_t z0 = static_cast<std::ptrdiff_t>(position[0]);
        const std::ptrdiff_t y0 = static_cast<std::ptrdiff_t>(position[1]);
        const std::ptrdiff_t x0 = static_cast<std::ptrdiff_t>(position[2]);
        const float fz = position[0] - static_cast<float>(z0);
        const float fy = position[1] - static_cast<float>(y0);
        const float fx = position[2] - static_cast<float>(x0);
        const std::ptrdiff_t z1 = (z0 + 1 < grid.shape[0]) ? z0 + 1 : z0;
        const std::ptrdiff_t y1 = (y0 + 1 < grid.shape[1]) ? y0 + 1 : y0;
        const std::ptrdiff_t x1 = (x0 + 1 < grid.shape[2]) ? x0 + 1 : x0;
        const std::ptrdiff_t z0s = z0 * sz, z1s = z1 * sz;
        const std::ptrdiff_t y0s = y0 * sy, y1s = y1 * sy;
        const std::ptrdiff_t x0s = x0 * sx, x1s = x1 * sx;
        const std::ptrdiff_t o000 = z0s + y0s + x0s;
        const std::ptrdiff_t o001 = z0s + y0s + x1s;
        const std::ptrdiff_t o010 = z0s + y1s + x0s;
        const std::ptrdiff_t o011 = z0s + y1s + x1s;
        const std::ptrdiff_t o100 = z1s + y0s + x0s;
        const std::ptrdiff_t o101 = z1s + y0s + x1s;
        const std::ptrdiff_t o110 = z1s + y1s + x0s;
        const std::ptrdiff_t o111 = z1s + y1s + x1s;
        for (std::size_t axis = 0; axis < D; ++axis) {
            const float *c = channels[axis];
            const float c00 = c[o000] + fx * (c[o001] - c[o000]);
            const float c01 = c[o010] + fx * (c[o011] - c[o010]);
            const float c10 = c[o100] + fx * (c[o101] - c[o100]);
            const float c11 = c[o110] + fx * (c[o111] - c[o110]);
            const float c0 = c00 + fy * (c01 - c00);
            const float c1 = c10 + fy * (c11 - c10);
            out[axis] = c0 + fz * (c1 - c0);
        }
    }
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
        // transformation/affine.hxx and segmentation/watershed.hxx. clipped is
        // nonnegative, so (clipped + 0.5f) >= 0 and the truncating cast equals
        // std::floor(clipped + 0.5f) exactly, while avoiding the comparatively
        // expensive inlined floor on the portable (non-SSE4.1) wheel build.
        const auto coord = static_cast<std::ptrdiff_t>(clipped + 0.5f);
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

namespace detail {

// Trace a single particle through its whole trajectory. The integration flags
// are compile-time parameters so the per-step branches on them fold away and
// the sampler/RK2/convergence/mask code inlines into one specialized loop.
//
// CodegenVariant must match the instantiating trace_all (see there): the
// compiler is free to emit this function out-of-line (observed with GCC 14
// under mild size pressure), and without the tag the FMA translation unit
// would emit AVX code under the same weak symbol name as the portable
// instantiation, letting COMDAT selection ship AVX code to the portable
// fallback path (SIGILL on pre-AVX CPUs) or silently discard the FMA kernel.
template <
    std::size_t D, bool UseRK2, bool CheckConvergence, bool RestrictToMask,
    bool CodegenVariant = false>
inline void trace_particle(
    std::array<float, D> &position,
    const std::array<const float *, D> &channels,
    const GridLayout<D> &grid,
    const std::uint8_t *mask,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    const auto clip = [&grid](std::array<float, D> &p) {
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (p[axis] < 0.0f) {
                p[axis] = 0.0f;
            } else if (p[axis] > grid.upper[axis]) {
                p[axis] = grid.upper[axis];
            }
        }
    };

    // When restricting to the mask, only in-mask (hence in-bounds) endpoints are
    // ever committed and the seed is an in-bounds integer voxel, so `position`
    // is already inside the domain at the start of every step and the
    // start-of-step clip is redundant. Without mask restriction an endpoint may
    // leave the domain and must be clipped back before the next sample.
    if constexpr (!RestrictToMask) {
        clip(position);
    }

    for (std::size_t iter = 0; iter < n_iter; ++iter) {
        std::array<float, D> step{};
        sample_flow<D>(channels, position, grid, step);

        if constexpr (UseRK2) {
            std::array<float, D> mid{};
            for (std::size_t axis = 0; axis < D; ++axis) {
                mid[axis] = position[axis] + 0.5f * dt * step[axis];
            }
            clip(mid);
            sample_flow<D>(channels, mid, grid, step);
        }

        if constexpr (CheckConvergence) {
            float max_step = 0.0f;
            for (std::size_t axis = 0; axis < D; ++axis) {
                const float abs_step = std::fabs(dt * step[axis]);
                if (abs_step > max_step) {
                    max_step = abs_step;
                }
            }
            if (max_step < tol) {
                break;
            }
        }

        std::array<float, D> proposed = position;
        for (std::size_t axis = 0; axis < D; ++axis) {
            proposed[axis] += dt * step[axis];
        }

        if constexpr (RestrictToMask) {
            // A particle whose proposed endpoint leaves the foreground is frozen
            // at its last in-mask position (only the endpoint is mask-tested,
            // not the RK2 midpoint).
            if (!position_is_in_mask<D>(proposed, grid, mask)) {
                break;
            }
        } else {
            clip(proposed);
        }
        position = proposed;
    }
}

// Trace K consecutive particles in lockstep. The trajectories are independent,
// so the out-of-order core can overlap one lane's serial sample->update chain
// with the other lanes' chains; a converged or mask-frozen lane costs one
// predictable branch per remaining group iteration, and the group ends when
// every lane is done. The per-step body is identical to trace_particle (with
// `break` expressed as clearing the lane's alive flag), so the traced
// positions are bitwise equal to K independent trace_particle calls.
template <
    std::size_t D, std::size_t K, bool UseRK2, bool CheckConvergence,
    bool RestrictToMask, bool CodegenVariant = false>
inline void trace_particle_block(
    std::array<float, D> *positions,
    const std::array<const float *, D> &channels,
    const GridLayout<D> &grid,
    const std::uint8_t *mask,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    static_assert(K >= 2, "use trace_particle for single trajectories");

    const auto clip = [&grid](std::array<float, D> &p) {
        for (std::size_t axis = 0; axis < D; ++axis) {
            if (p[axis] < 0.0f) {
                p[axis] = 0.0f;
            } else if (p[axis] > grid.upper[axis]) {
                p[axis] = grid.upper[axis];
            }
        }
    };

    // Local copies keep the lane state in registers across the group loop.
    std::array<std::array<float, D>, K> pos;
    std::array<bool, K> alive;
    for (std::size_t k = 0; k < K; ++k) {
        pos[k] = positions[k];
        alive[k] = true;
    }
    if constexpr (!RestrictToMask) {
        for (std::size_t k = 0; k < K; ++k) {
            clip(pos[k]);
        }
    }

    for (std::size_t iter = 0; iter < n_iter; ++iter) {
        for (std::size_t k = 0; k < K; ++k) {
            if (!alive[k]) {
                continue;
            }
            std::array<float, D> step{};
            sample_flow<D>(channels, pos[k], grid, step);

            if constexpr (UseRK2) {
                std::array<float, D> mid{};
                for (std::size_t axis = 0; axis < D; ++axis) {
                    mid[axis] = pos[k][axis] + 0.5f * dt * step[axis];
                }
                clip(mid);
                sample_flow<D>(channels, mid, grid, step);
            }

            if constexpr (CheckConvergence) {
                float max_step = 0.0f;
                for (std::size_t axis = 0; axis < D; ++axis) {
                    const float abs_step = std::fabs(dt * step[axis]);
                    if (abs_step > max_step) {
                        max_step = abs_step;
                    }
                }
                if (max_step < tol) {
                    alive[k] = false;
                    continue;
                }
            }

            std::array<float, D> proposed = pos[k];
            for (std::size_t axis = 0; axis < D; ++axis) {
                proposed[axis] += dt * step[axis];
            }

            if constexpr (RestrictToMask) {
                if (!position_is_in_mask<D>(proposed, grid, mask)) {
                    alive[k] = false;
                    continue;
                }
            } else {
                clip(proposed);
            }
            pos[k] = proposed;
        }

        bool any_alive = false;
        for (std::size_t k = 0; k < K; ++k) {
            any_alive = any_alive || alive[k];
        }
        if (!any_alive) {
            break;
        }
    }

    for (std::size_t k = 0; k < K; ++k) {
        positions[k] = pos[k];
    }
}

// CodegenVariant gives separately compiled ISA variants a distinct linker
// identity. Without it, COMDAT selection may replace an FMA instantiation with
// the portable instantiation that has the otherwise-identical template name.
template <
    std::size_t D, bool UseRK2, bool CheckConvergence, bool RestrictToMask,
    bool CodegenVariant = false>
void trace_all(
    std::vector<std::array<float, D>> &positions,
    const std::array<const float *, D> &channels,
    const GridLayout<D> &grid,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    // Particle-major: each worker traces its whole contiguous range of particles
    // across all integration steps in one fan-out. Trajectories are independent
    // until the sequential scatter, so no global alive state, per-step barrier,
    // or per-step alive scan is needed.
    ::bioimage_cpp::detail::parallel_for_chunks(
        n_threads,
        positions.size(),
        [&](const std::size_t, const std::size_t begin, const std::size_t end) {
            // Lockstep interleaving only pays for RK2: its two dependent
            // samples per step leave latency bubbles that other lanes fill.
            // The shorter Euler chain measured ~10% slower when interleaved
            // (extra lane state without enough latency to hide). K=3 beat
            // K=2/4 and the per-particle loop on paired benchmarks (see
            // development/flow/PERFORMANCE_NOTES.md, trajectory interleave).
            constexpr std::size_t K = UseRK2 ? 3 : 1;
            std::size_t i = begin;
            if constexpr (K > 1) {
                for (; i + K <= end; i += K) {
                    trace_particle_block<
                        D, K, UseRK2, CheckConvergence, RestrictToMask,
                        CodegenVariant>(
                        &positions[i], channels, grid, mask, n_iter, dt, tol
                    );
                }
            }
            for (; i < end; ++i) {
                trace_particle<
                    D, UseRK2, CheckConvergence, RestrictToMask, CodegenVariant>(
                    positions[i], channels, grid, mask, n_iter, dt, tol
                );
            }
        }
    );
}

#if defined(BIOIMAGE_FLOW_FMA_DISPATCH)

inline bool runtime_fma_supported() noexcept {
    static const bool supported = []() noexcept {
#if defined(_MSC_VER)
        int registers[4]{};
        __cpuid(registers, 1);
        constexpr int fma_bit = 1 << 12;
        constexpr int osxsave_bit = 1 << 27;
        constexpr int avx_bit = 1 << 28;
        if ((registers[2] & (fma_bit | osxsave_bit | avx_bit)) !=
            (fma_bit | osxsave_bit | avx_bit)) {
            return false;
        }
        if ((_xgetbv(0) & 0x6) != 0x6) {
            return false;
        }
#if defined(BIOIMAGE_FLOW_FMA_REQUIRES_AVX2)
        __cpuidex(registers, 7, 0);
        constexpr int avx2_bit = 1 << 5;
        if ((registers[1] & avx2_bit) == 0) {
            return false;
        }
#endif
        return true;
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma");
#else
        return false;
#endif
    }();
    return supported;
}

void trace_all_fma_2d(
    std::vector<std::array<float, 2>> &positions,
    const std::array<const float *, 2> &channels,
    const GridLayout<2> &grid,
    const std::uint8_t *mask,
    std::size_t n_threads,
    std::size_t n_iter,
    float dt,
    float tol
);

void trace_all_fma_3d(
    std::vector<std::array<float, 3>> &positions,
    const std::array<const float *, 3> &channels,
    const GridLayout<3> &grid,
    const std::uint8_t *mask,
    std::size_t n_threads,
    std::size_t n_iter,
    float dt,
    float tol
);

template <std::size_t D>
bool try_trace_all_fma(
    std::vector<std::array<float, D>> &positions,
    const std::array<const float *, D> &channels,
    const GridLayout<D> &grid,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    if (!runtime_fma_supported()) {
        return false;
    }
    if constexpr (D == 2) {
        trace_all_fma_2d(
            positions, channels, grid, mask, n_threads, n_iter, dt, tol
        );
    } else {
        trace_all_fma_3d(
            positions, channels, grid, mask, n_threads, n_iter, dt, tol
        );
    }
    return true;
}

#endif

} // namespace detail

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

    const bool use_rk2 = (method == IntegrationMethod::RK2);
    const bool check_convergence = (tol > 0.0f);

    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "iter_loop");
        // Dispatch the three loop-invariant flags to a compile-time specialized
        // tracer (see trace_particle). The flags are constant for the whole
        // call, so this hoists their branches out of the innermost step loop.
        const int selector =
            (use_rk2 ? 4 : 0) | (check_convergence ? 2 : 0) | (restrict_to_mask ? 1 : 0);
#define BIOIMAGE_FLOW_TRACE(SELECTOR, RK2, CONV, RESTRICT)                \
    case SELECTOR:                                                        \
        detail::trace_all<D, RK2, CONV, RESTRICT>(                        \
            positions, channels, grid, fg_mask.data, n_threads, n_iter, dt, tol); \
        break;
        switch (selector) {
            BIOIMAGE_FLOW_TRACE(0, false, false, false)
            BIOIMAGE_FLOW_TRACE(1, false, false, true)
            BIOIMAGE_FLOW_TRACE(2, false, true,  false)
            BIOIMAGE_FLOW_TRACE(3, false, true,  true)
            BIOIMAGE_FLOW_TRACE(4, true,  false, false)
            BIOIMAGE_FLOW_TRACE(5, true,  false, true)
            BIOIMAGE_FLOW_TRACE(6, true,  true,  false)
            case 7:
#if defined(BIOIMAGE_FLOW_FMA_DISPATCH)
                if (detail::try_trace_all_fma<D>(
                        positions,
                        channels,
                        grid,
                        fg_mask.data,
                        n_threads,
                        n_iter,
                        dt,
                        tol
                    )) {
                    break;
                }
#endif
                detail::trace_all<D, true, true, true>(
                    positions, channels, grid, fg_mask.data, n_threads, n_iter, dt, tol
                );
                break;
        }
#undef BIOIMAGE_FLOW_TRACE
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
