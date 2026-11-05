#include "bioimage_cpp/flow/flow_density.hxx"

#include "bioimage_cpp/detail/force_inline.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#if !defined(BIOIMAGE_FLOW_NEON)
#error "flow_density_neon.cxx must only be built with BIOIMAGE_FLOW_NEON"
#endif
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error "flow_density_neon.cxx requires an arm64 target"
#endif

#include <arm_neon.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

// NEON port of the packed-channel tracer in flow_density_fma.cxx.
// See development/flow/PERFORMANCE_NOTES.md for the design.

namespace bioimage_cpp::flow::detail {
namespace {

constexpr std::ptrdiff_t kPackedTailSlack = 4;  // floats past the last voxel

template <std::size_t D>
struct PackedGrid {
    const float *data = nullptr;
    float32x4_t upper_f;  // shape-1 per axis (float), 0 in unused lanes
    uint32x4_t lane_mask;  // all-ones in the D used lanes
    std::array<int, D> pstride{};  // float-element strides of the padded packed grid
    std::array<int, D> mstride{};  // element strides of the mask grid
};

BIOIMAGE_FORCE_INLINE float32x4_t make_f32x4(const float a, const float b, const float c, const float d) {
    const float values[4] = {a, b, c, d};
    return vld1q_f32(values);
}

// Repack the flow channel-last, padded by one replicated voxel per axis.
// Returns false if the offsets do not fit int32 or the allocation fails.
template <std::size_t D>
bool make_packed_grid(
    const std::array<const float *, D> &channels,
    const GridLayout<D> &grid,
    const std::size_t n_threads,
    std::unique_ptr<float[]> &storage,
    PackedGrid<D> &out
) {
    static_assert(D == 2 || D == 3, "packed tracer supports 2D and 3D");
    constexpr std::ptrdiff_t int_max = std::numeric_limits<std::int32_t>::max();
    constexpr std::ptrdiff_t channels_per_voxel = static_cast<std::ptrdiff_t>(D);

    std::array<std::ptrdiff_t, D> padded{};
    std::ptrdiff_t n_padded = 1;
    for (std::size_t axis = 0; axis < D; ++axis) {
        padded[axis] = grid.shape[axis] + 1;
        n_padded *= padded[axis];
    }
    const std::ptrdiff_t total = channels_per_voxel * n_padded + kPackedTailSlack;
    std::array<std::ptrdiff_t, D> pstride{};
    pstride[D - 1] = channels_per_voxel;
    for (std::size_t axis = D - 1; axis > 0; --axis) {
        pstride[axis - 1] = pstride[axis] * padded[axis];
    }
    if (total > int_max) {
        return false;
    }
    for (std::size_t axis = 0; axis < D; ++axis) {
        if (grid.strides[axis] > int_max) {
            return false;
        }
    }
    try {
        storage.reset(new float[static_cast<std::size_t>(total)]);
    } catch (const std::bad_alloc &) {
        return false;
    }
    float *packed = storage.get();
    for (std::ptrdiff_t i = total - kPackedTailSlack; i < total; ++i) {
        packed[i] = 0.0f;
    }

    // Copy one source row and duplicate its last voxel.
    const std::ptrdiff_t width = grid.shape[D - 1];
    const auto fill_row = [&](float *o, const std::ptrdiff_t src_offset) {
        if constexpr (D == 3) {
            const float *c0 = channels[0] + src_offset;
            const float *c1 = channels[1] + src_offset;
            const float *c2 = channels[2] + src_offset;
            for (std::ptrdiff_t x = 0; x < width; ++x) {
                o[3 * x] = c0[x];
                o[3 * x + 1] = c1[x];
                o[3 * x + 2] = c2[x];
            }
            o[3 * width] = c0[width - 1];
            o[3 * width + 1] = c1[width - 1];
            o[3 * width + 2] = c2[width - 1];
        } else {
            const float *c0 = channels[0] + src_offset;
            const float *c1 = channels[1] + src_offset;
            for (std::ptrdiff_t x = 0; x < width; ++x) {
                o[2 * x] = c0[x];
                o[2 * x + 1] = c1[x];
            }
            o[2 * width] = c0[width - 1];
            o[2 * width + 1] = c1[width - 1];
        }
    };

    if constexpr (D == 3) {
        const std::ptrdiff_t n_planes = padded[0];
        const auto pack_threads = ::bioimage_cpp::detail::normalize_thread_count(
            n_threads, static_cast<std::size_t>(n_planes)
        );
        ::bioimage_cpp::detail::parallel_for_chunks(
            pack_threads, static_cast<std::size_t>(n_planes),
            [&](const std::size_t, const std::size_t begin, const std::size_t end) {
                for (std::size_t zp = begin; zp < end; ++zp) {
                    const auto z = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(zp), grid.shape[0] - 1);
                    for (std::ptrdiff_t yp = 0; yp < padded[1]; ++yp) {
                        const std::ptrdiff_t y = std::min<std::ptrdiff_t>(yp, grid.shape[1] - 1);
                        fill_row(
                            packed + static_cast<std::ptrdiff_t>(zp) * pstride[0] + yp * pstride[1],
                            z * grid.strides[0] + y * grid.strides[1]
                        );
                    }
                }
            }
        );
    } else {
        const std::ptrdiff_t n_rows = padded[0];
        const auto pack_threads = ::bioimage_cpp::detail::normalize_thread_count(
            n_threads, static_cast<std::size_t>(n_rows)
        );
        ::bioimage_cpp::detail::parallel_for_chunks(
            pack_threads, static_cast<std::size_t>(n_rows),
            [&](const std::size_t, const std::size_t begin, const std::size_t end) {
                for (std::size_t yp = begin; yp < end; ++yp) {
                    const auto y = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(yp), grid.shape[0] - 1);
                    fill_row(packed + static_cast<std::ptrdiff_t>(yp) * pstride[0], y * grid.strides[0]);
                }
            }
        );
    }

    out.data = packed;
    for (std::size_t axis = 0; axis < D; ++axis) {
        out.pstride[axis] = static_cast<int>(pstride[axis]);
        out.mstride[axis] = static_cast<int>(grid.strides[axis]);
    }
    if constexpr (D == 3) {
        out.upper_f = make_f32x4(grid.upper[0], grid.upper[1], grid.upper[2], 0.0f);
        const std::uint32_t lanes[4] = {~0u, ~0u, ~0u, 0u};
        out.lane_mask = vld1q_u32(lanes);
    } else {
        out.upper_f = make_f32x4(grid.upper[0], grid.upper[1], 0.0f, 0.0f);
        const std::uint32_t lanes[4] = {~0u, ~0u, 0u, 0u};
        out.lane_mask = vld1q_u32(lanes);
    }
    return true;
}

// lo + f * (hi - lo) with one rounding, like the contracted scalar lerp.
BIOIMAGE_FORCE_INLINE float32x4_t lerp(const float32x4_t lo, const float32x4_t hi, const float32x4_t f) {
    return vfmaq_f32(lo, f, vsubq_f32(hi, lo));
}

// Lower corner, fraction and flat offset. Truncation equals floor because p >= 0.
// vcvtq_s32_f32 differs from cvttps only for NaN and overflow, and p is clipped.
struct Stencil {
    float32x4_t frac;
    int base;
};

template <std::size_t D>
BIOIMAGE_FORCE_INLINE Stencil make_stencil(const PackedGrid<D> &g, const float32x4_t p) {
    const int32x4_t i0 = vcvtq_s32_f32(p);
    const float32x4_t frac = vsubq_f32(p, vcvtq_f32_s32(i0));
    int base = vgetq_lane_s32(i0, 0) * g.pstride[0] + vgetq_lane_s32(i0, 1) * g.pstride[1];
    if constexpr (D == 3) {
        base += vgetq_lane_s32(i0, 2) * g.pstride[2];
    }
    return Stencil{frac, base};
}

// 3D: nested lerps in the order of the scalar kernel. Lane 3 holds finite junk.
BIOIMAGE_FORCE_INLINE float32x4_t sample_packed(const PackedGrid<3> &g, const float32x4_t p) {
    const Stencil s = make_stencil<3>(g, p);
    constexpr int dx = 3;
    const int dy = g.pstride[1];
    const int dz = g.pstride[0];
    const float *p0 = g.data + s.base;
    const float *p1 = p0 + dz;
    const float32x4_t c000 = vld1q_f32(p0);
    const float32x4_t c001 = vld1q_f32(p0 + dx);
    const float32x4_t c010 = vld1q_f32(p0 + dy);
    const float32x4_t c011 = vld1q_f32(p0 + dy + dx);
    const float32x4_t c100 = vld1q_f32(p1);
    const float32x4_t c101 = vld1q_f32(p1 + dx);
    const float32x4_t c110 = vld1q_f32(p1 + dy);
    const float32x4_t c111 = vld1q_f32(p1 + dy + dx);
    const float32x4_t fz = vdupq_laneq_f32(s.frac, 0);
    const float32x4_t fy = vdupq_laneq_f32(s.frac, 1);
    const float32x4_t fx = vdupq_laneq_f32(s.frac, 2);
    const float32x4_t c00 = lerp(c000, c001, fx);
    const float32x4_t c01 = lerp(c010, c011, fx);
    const float32x4_t c10 = lerp(c100, c101, fx);
    const float32x4_t c11 = lerp(c110, c111, fx);
    const float32x4_t c0 = lerp(c00, c01, fy);
    const float32x4_t c1 = lerp(c10, c11, fy);
    return lerp(c0, c1, fz);
}

// 2D: weighted sum of the four corners, in the same order as the SSE kernel.
BIOIMAGE_FORCE_INLINE float32x4_t sample_packed(const PackedGrid<2> &g, const float32x4_t p) {
    const Stencil s = make_stencil<2>(g, p);
    const float *r0p = g.data + s.base;
    // Row loads: [c0(x0), c1(x0), c0(x1), c1(x1)].
    const float32x4_t r0 = vld1q_f32(r0p);
    const float32x4_t r1 = vld1q_f32(r0p + g.pstride[0]);
    const float32x4_t f = s.frac;  // [fy, fx, 0, 0]
    const float32x4_t omf = vsubq_f32(vdupq_n_f32(1.0f), f);  // [gy, gx, 1, 1]
    const float32x4_t vx = vcombine_f32(vdup_laneq_f32(omf, 1), vdup_laneq_f32(f, 1));  // [gx, gx, fx, fx]
    const float32x4_t w0 = vmulq_laneq_f32(vx, omf, 0);  // row y0
    const float32x4_t w1 = vmulq_laneq_f32(vx, f, 0);  // row y1
    const float32x4_t t = vfmaq_f32(vmulq_f32(r0, w0), r1, w1);  // [x0 terms | x1 terms]
    return vaddq_f32(t, vextq_f32(t, t, 2));  // lanes 0,1 = the sum
}

struct StepConstants {
    float32x4_t zero;
    float32x4_t half;
    float32x4_t half_dt;
    float32x4_t dt;
    float32x4_t tol;
};

inline StepConstants make_step_constants(const float dt, const float tol) {
    return StepConstants{
        vdupq_n_f32(0.0f),
        vdupq_n_f32(0.5f),
        vdupq_n_f32(0.5f * dt),
        vdupq_n_f32(dt),
        vdupq_n_f32(tol),
    };
}

// True if any lane of the comparison mask is set.
BIOIMAGE_FORCE_INLINE bool any_lane(const uint32x4_t m) {
    return vget_lane_u64(vreinterpret_u64_u16(vshrn_n_u32(m, 16)), 0) != 0;
}

constexpr int kStillAlive = -1;

// One RK2 step of a single particle with convergence and mask termination.
// Returns kStillAlive, or the ActiveTraceStats exit reason if tracing stops.
template <std::size_t D>
BIOIMAGE_FORCE_INLINE int step_packed(
    float32x4_t &pos, const PackedGrid<D> &g, const StepConstants &c, const std::uint8_t *mask
) {
    // The clip zeroes the junk lane of the first sample. The mask zeroes it in the second.
    float32x4_t step = sample_packed(g, pos);
    float32x4_t mid = vfmaq_f32(pos, c.half_dt, step);
    // vmaxnmq/vminnmq match maxps/minps: a NaN lane clips to 0.
    mid = vminnmq_f32(vmaxnmq_f32(mid, c.zero), g.upper_f);
    step = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(sample_packed(g, mid)), g.lane_mask));

    // The particle converged if no lane has |dt * step| >= tol.
    const float32x4_t disp = vmulq_f32(c.dt, step);
    if (!any_lane(vcgeq_f32(vabsq_f32(disp), c.tol))) {
        return ActiveTraceStats::Converged;
    }

    // Fused, because clang fuses `proposed += dt * step` in the scalar kernel.
    const float32x4_t proposed = vfmaq_f32(pos, c.dt, step);
    const uint32x4_t outside = vorrq_u32(vcltq_f32(proposed, c.zero), vcgtq_f32(proposed, g.upper_f));
    if (any_lane(outside)) {
        return ActiveTraceStats::LeftMask;
    }
    // Round half up to the mask voxel. Truncation equals floor because proposed >= 0.
    const int32x4_t rounded = vcvtq_s32_f32(vaddq_f32(proposed, c.half));
    int flat = vgetq_lane_s32(rounded, 0) * g.mstride[0] + vgetq_lane_s32(rounded, 1) * g.mstride[1];
    if constexpr (D == 3) {
        flat += vgetq_lane_s32(rounded, 2) * g.mstride[2];
    }
    if (mask[flat] == 0) {
        return ActiveTraceStats::LeftMask;
    }
    pos = proposed;
    return kStillAlive;
}

template <std::size_t D>
BIOIMAGE_FORCE_INLINE float32x4_t load_position(const std::array<float, D> &p) {
    if constexpr (D == 3) {
        return make_f32x4(p[0], p[1], p[2], 0.0f);
    } else {
        return make_f32x4(p[0], p[1], 0.0f, 0.0f);
    }
}

template <std::size_t D>
BIOIMAGE_FORCE_INLINE void store_position(std::array<float, D> &p, const float32x4_t v) {
    float tmp[4];
    vst1q_f32(tmp, v);
    for (std::size_t axis = 0; axis < D; ++axis) {
        p[axis] = tmp[axis];
    }
}

template <class F, std::size_t... Ks>
BIOIMAGE_FORCE_INLINE void for_each_lane(F &&f, std::index_sequence<Ks...>) {
    (f(std::integral_constant<std::size_t, Ks>{}), ...);
}

// Trace K particles in lockstep. Compile-time lane indices keep the state in registers.
template <std::size_t D, std::size_t K>
void trace_block_packed(
    std::array<float, D> *positions,
    const PackedGrid<D> &g,
    const StepConstants &c,
    const std::uint8_t *mask,
    const std::size_t n_iter,
    ActiveTraceStats &stats
) {
    float32x4_t pos[K];
    bool alive[K];
    [[maybe_unused]] std::size_t lane_steps[K] = {};
    [[maybe_unused]] int lane_exit[K] = {};
    for_each_lane([&](auto k) {
        pos[k] = load_position<D>(positions[k]);
        alive[k] = true;
        if constexpr (ActiveTraceStats::enabled) {
            lane_exit[k] = ActiveTraceStats::MaxIter;
        }
    }, std::make_index_sequence<K>{});

    for (std::size_t iter = 0; iter < n_iter; ++iter) {
        bool any_alive = false;
        for_each_lane([&](auto k) {
            if (!alive[k]) {
                return;
            }
            if constexpr (ActiveTraceStats::enabled) {
                ++lane_steps[k];
            }
            const int exit_reason = step_packed<D>(pos[k], g, c, mask);
            if (exit_reason != kStillAlive) {
                alive[k] = false;
                if constexpr (ActiveTraceStats::enabled) {
                    lane_exit[k] = exit_reason;
                }
                return;
            }
            any_alive = true;
        }, std::make_index_sequence<K>{});
        if (!any_alive) {
            break;
        }
    }

    for_each_lane([&](auto k) {
        store_position<D>(positions[k], pos[k]);
        if constexpr (ActiveTraceStats::enabled) {
            stats.record(lane_steps[k], lane_exit[k]);
        }
    }, std::make_index_sequence<K>{});
}

template <std::size_t D, std::size_t K>
void trace_all_packed(
    std::vector<std::array<float, D>> &positions,
    const PackedGrid<D> &g,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    const StepConstants c = make_step_constants(dt, tol);
    ActiveTraceStats null_stats{};
    std::vector<ActiveTraceStats> per_thread_stats;
    if constexpr (ActiveTraceStats::enabled) {
        per_thread_stats.assign(n_threads, ActiveTraceStats(n_iter));
    }
    ::bioimage_cpp::detail::parallel_for_chunks(
        n_threads, positions.size(),
        [&](const std::size_t thread_id, const std::size_t begin, const std::size_t end) {
            ActiveTraceStats *stats = &null_stats;
            (void)thread_id;
#ifdef BIOIMAGE_PROFILE
            stats = &per_thread_stats[thread_id];
            const auto chunk_start = std::chrono::steady_clock::now();
#endif
            std::size_t i = begin;
            if constexpr (K > 1) {
                for (; i + K <= end; i += K) {
                    trace_block_packed<D, K>(&positions[i], g, c, mask, n_iter, *stats);
                }
            }
            for (; i < end; ++i) {
                trace_block_packed<D, 1>(&positions[i], g, c, mask, n_iter, *stats);
            }
#ifdef BIOIMAGE_PROFILE
            stats->seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - chunk_start
            ).count();
#endif
        }
    );
    if constexpr (ActiveTraceStats::enabled) {
        ActiveTraceStats::report(per_thread_stats, n_iter, true);
    }
}

// K = 16 measured best on Apple M2. See PERFORMANCE_NOTES.md for the sweep.
#if defined(BIOIMAGE_FLOW_NEON_LANES)
constexpr std::size_t kPackedLanes = BIOIMAGE_FLOW_NEON_LANES;
#else
constexpr std::size_t kPackedLanes = 16;
#endif

template <std::size_t D>
void trace_all_neon(
    std::vector<std::array<float, D>> &positions,
    const std::array<const float *, D> &channels,
    const GridLayout<D> &grid,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    BIOIMAGE_PROFILE_INIT(profiler);
    {
        std::unique_ptr<float[]> storage;
        PackedGrid<D> packed{};
        bool ready = false;
        {
            BIOIMAGE_PROFILE_SCOPE(profiler, "neon_pack");
            ready = make_packed_grid<D>(channels, grid, n_threads, storage, packed);
        }
        if (ready) {
            BIOIMAGE_PROFILE_SCOPE(profiler, "neon_trace_packed");
            trace_all_packed<D, kPackedLanes>(positions, packed, mask, n_threads, n_iter, dt, tol);
            BIOIMAGE_PROFILE_REPORT_NAMED(profiler, "[bioimage profile: flow neon]");
            return;
        }
    }
    // Fallback if the packed grid cannot be built.
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "neon_trace_scalar");
        trace_all<D, true, true, true>(positions, channels, grid, mask, n_threads, n_iter, dt, tol);
    }
    BIOIMAGE_PROFILE_REPORT_NAMED(profiler, "[bioimage profile: flow neon]");
}

} // namespace

void trace_all_neon_2d(
    std::vector<std::array<float, 2>> &positions,
    const std::array<const float *, 2> &channels,
    const GridLayout<2> &grid,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    trace_all_neon<2>(positions, channels, grid, mask, n_threads, n_iter, dt, tol);
}

void trace_all_neon_3d(
    std::vector<std::array<float, 3>> &positions,
    const std::array<const float *, 3> &channels,
    const GridLayout<3> &grid,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    trace_all_neon<3>(positions, channels, grid, mask, n_threads, n_iter, dt, tol);
}

} // namespace bioimage_cpp::flow::detail
