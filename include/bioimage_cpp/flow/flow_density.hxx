#pragma once

#include "bioimage_cpp/array_view.hxx"
#include "bioimage_cpp/detail/force_inline.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef BIOIMAGE_PROFILE
#include <chrono>
#include <cstdio>
#endif

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

// Executed-step accounting for profile builds. `record(steps, exit)` is called
// once per traced particle with the number of integration steps whose samples
// were computed and why tracing stopped. Outside BIOIMAGE_PROFILE builds the
// Null variant is used and every call site is guarded by `if constexpr`, so
// the production kernels carry no counters.
struct NullTraceStats {
    static constexpr bool enabled = false;
    enum Exit : int { Converged = 0, LeftMask = 1, MaxIter = 2 };
    explicit NullTraceStats(std::size_t = 0) noexcept {}
    void record(std::size_t, int) noexcept {}
    void merge(const NullTraceStats &) noexcept {}
    static void report(const std::vector<NullTraceStats> &, std::size_t, bool) noexcept {}
};

#ifdef BIOIMAGE_PROFILE
struct TraceStats {
    static constexpr bool enabled = true;
    enum Exit : int { Converged = 0, LeftMask = 1, MaxIter = 2 };

    std::uint64_t steps = 0;
    std::uint64_t particles = 0;
    std::array<std::uint64_t, 3> exits{};
    std::vector<std::uint64_t> histogram;  // histogram[s] = particles that executed s steps
    double seconds = 0.0;

    explicit TraceStats(std::size_t n_iter = 0) : histogram(n_iter + 1, 0) {}

    void record(std::size_t n_steps, int reason) noexcept {
        steps += n_steps;
        ++particles;
        ++exits[static_cast<std::size_t>(reason)];
        if (n_steps < histogram.size()) {
            ++histogram[n_steps];
        }
    }

    void merge(const TraceStats &other) {
        steps += other.steps;
        particles += other.particles;
        for (std::size_t i = 0; i < exits.size(); ++i) {
            exits[i] += other.exits[i];
        }
        if (histogram.size() < other.histogram.size()) {
            histogram.resize(other.histogram.size(), 0);
        }
        for (std::size_t i = 0; i < other.histogram.size(); ++i) {
            histogram[i] += other.histogram[i];
        }
        seconds += other.seconds;
    }

    // Print the merged totals plus one line per worker (load balance).
    static void report(
        const std::vector<TraceStats> &per_thread, const std::size_t n_iter, const bool rk2
    ) {
        TraceStats total(n_iter);
        for (const auto &t : per_thread) {
            total.merge(t);
        }
        const double particles = static_cast<double>(std::max<std::uint64_t>(total.particles, 1));
        std::fprintf(stderr, "[bioimage flow trace]\n");
        std::fprintf(
            stderr, "  particles %llu  n_iter %zu  %s\n",
            static_cast<unsigned long long>(total.particles), n_iter, rk2 ? "rk2" : "euler"
        );
        std::fprintf(
            stderr, "  steps %llu  mean_steps_per_particle %.3f  samples_per_step %d\n",
            static_cast<unsigned long long>(total.steps), total.steps / particles, rk2 ? 2 : 1
        );
        std::fprintf(
            stderr, "  exits converged %llu (%.1f%%)  left_mask %llu (%.1f%%)  hit_n_iter %llu (%.1f%%)\n",
            static_cast<unsigned long long>(total.exits[0]), 100.0 * total.exits[0] / particles,
            static_cast<unsigned long long>(total.exits[1]), 100.0 * total.exits[1] / particles,
            static_cast<unsigned long long>(total.exits[2]), 100.0 * total.exits[2] / particles
        );
        std::fprintf(stderr, "  histogram(steps:particles)");
        std::size_t lo = 0;
        for (std::size_t width = 1; lo < total.histogram.size(); width *= 2) {
            const std::size_t hi = std::min(total.histogram.size(), lo + width);
            std::uint64_t count = 0;
            for (std::size_t s = lo; s < hi; ++s) {
                count += total.histogram[s];
            }
            if (hi - lo == 1) {
                std::fprintf(stderr, " %zu:%llu", lo, static_cast<unsigned long long>(count));
            } else {
                std::fprintf(stderr, " %zu-%zu:%llu", lo, hi - 1, static_cast<unsigned long long>(count));
            }
            lo = hi;
        }
        std::fprintf(stderr, "\n");
        double max_seconds = 0.0, sum_seconds = 0.0;
        std::uint64_t max_steps = 0;
        for (std::size_t t = 0; t < per_thread.size(); ++t) {
            std::fprintf(
                stderr, "  thread %zu  %.4f s  steps %llu  particles %llu\n", t, per_thread[t].seconds,
                static_cast<unsigned long long>(per_thread[t].steps),
                static_cast<unsigned long long>(per_thread[t].particles)
            );
            max_seconds = std::max(max_seconds, per_thread[t].seconds);
            sum_seconds += per_thread[t].seconds;
            max_steps = std::max(max_steps, per_thread[t].steps);
        }
        if (per_thread.size() > 1) {
            const double mean_seconds = sum_seconds / per_thread.size();
            const double mean_steps = static_cast<double>(total.steps) / per_thread.size();
            std::fprintf(
                stderr, "  imbalance time max/mean-1 %.1f%%  steps max/mean-1 %.1f%%\n",
                100.0 * (max_seconds / std::max(mean_seconds, 1e-12) - 1.0),
                100.0 * (max_steps / std::max(mean_steps, 1e-12) - 1.0)
            );
        }
    }
};
using ActiveTraceStats = TraceStats;
#else
using ActiveTraceStats = NullTraceStats;
#endif

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
BIOIMAGE_FORCE_INLINE void sample_flow(
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
BIOIMAGE_FORCE_INLINE std::ptrdiff_t round_to_flat_index(
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
BIOIMAGE_FORCE_INLINE bool position_is_in_mask(
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
// CodegenVariant must match the instantiating trace_all (see there): without
// the tag the FMA translation unit would emit AVX code under the same weak
// symbol name as the portable instantiation, letting COMDAT selection ship
// AVX code to the portable fallback path (SIGILL on pre-AVX CPUs) or silently
// discard the FMA kernel. The per-step helpers are additionally force-inlined
// because GCC 14 was observed to stop inlining them once the FMA translation
// unit grew (which both slowed the loop ~1.5x and recreated the hazard).
template <
    std::size_t D, bool UseRK2, bool CheckConvergence, bool RestrictToMask,
    bool CodegenVariant = false>
BIOIMAGE_FORCE_INLINE void trace_particle(
    std::array<float, D> &position,
    const std::array<const float *, D> &channels,
    const GridLayout<D> &grid,
    const std::uint8_t *mask,
    const std::size_t n_iter,
    const float dt,
    const float tol,
    ActiveTraceStats &stats
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
    [[maybe_unused]] std::size_t executed_steps = 0;
    [[maybe_unused]] int exit_reason = ActiveTraceStats::MaxIter;

    // When restricting to the mask, only in-mask (hence in-bounds) endpoints are
    // ever committed and the seed is an in-bounds integer voxel, so `position`
    // is already inside the domain at the start of every step and the
    // start-of-step clip is redundant. Without mask restriction an endpoint may
    // leave the domain and must be clipped back before the next sample.
    if constexpr (!RestrictToMask) {
        clip(position);
    }

    for (std::size_t iter = 0; iter < n_iter; ++iter) {
        if constexpr (ActiveTraceStats::enabled) {
            ++executed_steps;
        }
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
                if constexpr (ActiveTraceStats::enabled) {
                    exit_reason = ActiveTraceStats::Converged;
                }
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
                if constexpr (ActiveTraceStats::enabled) {
                    exit_reason = ActiveTraceStats::LeftMask;
                }
                break;
            }
        } else {
            clip(proposed);
        }
        position = proposed;
    }
    if constexpr (ActiveTraceStats::enabled) {
        stats.record(executed_steps, exit_reason);
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
BIOIMAGE_FORCE_INLINE void trace_particle_block(
    std::array<float, D> *positions,
    const std::array<const float *, D> &channels,
    const GridLayout<D> &grid,
    const std::uint8_t *mask,
    const std::size_t n_iter,
    const float dt,
    const float tol,
    ActiveTraceStats &stats
) {
    static_assert(K >= 2, "use trace_particle for single trajectories");
    [[maybe_unused]] std::array<std::size_t, K> lane_steps{};
    [[maybe_unused]] std::array<int, K> lane_exit{};
    if constexpr (ActiveTraceStats::enabled) {
        lane_exit.fill(ActiveTraceStats::MaxIter);
    }

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
            if constexpr (ActiveTraceStats::enabled) {
                ++lane_steps[k];
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
                    if constexpr (ActiveTraceStats::enabled) {
                        lane_exit[k] = ActiveTraceStats::Converged;
                    }
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
                    if constexpr (ActiveTraceStats::enabled) {
                        lane_exit[k] = ActiveTraceStats::LeftMask;
                    }
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
    if constexpr (ActiveTraceStats::enabled) {
        for (std::size_t k = 0; k < K; ++k) {
            stats.record(lane_steps[k], lane_exit[k]);
        }
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
    ActiveTraceStats null_stats{};
    std::vector<ActiveTraceStats> per_thread_stats;
    if constexpr (ActiveTraceStats::enabled) {
        per_thread_stats.assign(n_threads, ActiveTraceStats(n_iter));
    }
    ::bioimage_cpp::detail::parallel_for_chunks(
        n_threads,
        positions.size(),
        [&](const std::size_t thread_id, const std::size_t begin, const std::size_t end) {
            ActiveTraceStats *stats = &null_stats;
            (void)thread_id;
#ifdef BIOIMAGE_PROFILE
            stats = &per_thread_stats[thread_id];
            const auto chunk_start = std::chrono::steady_clock::now();
#endif
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
                        &positions[i], channels, grid, mask, n_iter, dt, tol, *stats
                    );
                }
            }
            for (; i < end; ++i) {
                trace_particle<
                    D, UseRK2, CheckConvergence, RestrictToMask, CodegenVariant>(
                    positions[i], channels, grid, mask, n_iter, dt, tol, *stats
                );
            }
#ifdef BIOIMAGE_PROFILE
            stats->seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - chunk_start
            ).count();
#endif
        }
    );
    if constexpr (ActiveTraceStats::enabled) {
        ActiveTraceStats::report(per_thread_stats, n_iter, UseRK2);
    }
}

// Runtime opt-out of the FMA-specialized tracer (mirrors
// BIOIMAGE_CPP_FILTERS_FORCE_SCALAR). Read on every call, not cached, so tests
// can toggle it in-process.
inline bool force_scalar_requested() noexcept {
    const char *value = std::getenv("BIOIMAGE_CPP_FLOW_FORCE_SCALAR");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
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
    if (force_scalar_requested() || !runtime_fma_supported()) {
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

// Name of the tracer the default RK2/convergence/mask path will use on this
// machine: "fma" (runtime-dispatched FMA translation unit) or "scalar"
// (portable kernel). Exposed to Python as `_core._flow_trace_backend`.
inline const char *trace_backend() noexcept {
#if defined(BIOIMAGE_FLOW_FMA_DISPATCH)
    if (!detail::force_scalar_requested() && detail::runtime_fma_supported()) {
        return "fma";
    }
#endif
    return "scalar";
}

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
