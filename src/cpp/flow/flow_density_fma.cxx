#include "bioimage_cpp/flow/flow_density.hxx"

#include "bioimage_cpp/detail/force_inline.hxx"
#include "bioimage_cpp/detail/profile.hxx"
#include "bioimage_cpp/detail/threading.hxx"

#if !defined(BIOIMAGE_FLOW_FMA_DISPATCH)
#error "flow_density_fma.cxx must only be built with BIOIMAGE_FLOW_FMA_DISPATCH"
#endif
#if !defined(__AVX__) || !(defined(__FMA__) || defined(_MSC_VER))
#error "flow_density_fma.cxx must be compiled with AVX and FMA enabled (-mavx -mfma or /arch:AVX2)"
#endif

#include <immintrin.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

// Runtime-dispatched tracer for the default path (RK2, convergence check, mask
// restriction), selected by try_trace_all_fma when the CPU supports AVX+FMA.
//
// Packed-channel kernel. The flow field is repacked once into channel-last
// storage (D floats per voxel, no padding between channels) with the grid
// padded by one duplicated voxel along every axis. Every interpolation corner
// is then one 16-byte load holding all channels, the per-channel scalar lerps
// become packed operations, and the duplicated border makes the upper corner
// index always valid (the scalar kernel's nearest-boundary behaviour without a
// clamp). The particle position lives in one 4-lane register, so the RK2
// midpoint, clipping, the convergence test and the mask lookup are packed too,
// while integer address arithmetic runs in general-purpose registers to keep it
// out of the FP register file. The per-lane arithmetic is the same IEEE
// single-precision operation sequence as the scalar kernel (fused lerp, fused
// midpoint, unfused position update).
//
// The scalar-FMA instantiation of the header kernel is kept as fallback when
// the packed buffer cannot be allocated or its offsets exceed int32.

namespace bioimage_cpp::flow::detail {
namespace {

constexpr std::ptrdiff_t kPackedTailSlack = 4;  // floats past the last voxel

template <std::size_t D>
struct PackedGrid {
    const float *data = nullptr;
    __m128 upper_f;                  // shape-1 per axis (float), 0 in unused lanes
    __m128 lane_mask;                // all-ones in the D used lanes
    std::array<int, D> pstride{};    // float-element strides of the padded packed grid
    std::array<int, D> mstride{};    // element strides of the mask grid
};

// Repack channel-first flow into channel-last storage on a grid padded by one
// voxel per axis (padding replicates the last voxel/row/plane). Returns false,
// and the caller falls back to the scalar kernel, if offsets would not fit
// int32 lanes or the buffer cannot be allocated.
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

    // Copy one source row (clamped y/z) into a padded row and duplicate its
    // last voxel. Rows are the unit of work; planes/rows beyond the source
    // extent replicate the last source plane/row.
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
                    const std::ptrdiff_t z = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(zp), grid.shape[0] - 1);
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
                    const std::ptrdiff_t y = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(yp), grid.shape[0] - 1);
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
        out.upper_f = _mm_setr_ps(grid.upper[0], grid.upper[1], grid.upper[2], 0.0f);
        out.lane_mask = _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0));
    } else {
        out.upper_f = _mm_setr_ps(grid.upper[0], grid.upper[1], 0.0f, 0.0f);
        out.lane_mask = _mm_castsi128_ps(_mm_setr_epi32(-1, -1, 0, 0));
    }
    return true;
}

// lo + f * (hi - lo) with one rounding, the same contraction the scalar
// kernel's lerp compiles to under -mfma.
BIOIMAGE_FORCE_INLINE __m128 lerp(const __m128 lo, const __m128 hi, const __m128 f) {
    return _mm_fmadd_ps(f, _mm_sub_ps(hi, lo), lo);
}

// Lower corner (by truncation: position lanes are clipped and nonnegative, so
// this equals floor, as in the scalar kernel), fractional part, and the flat
// float offset of the lower corner in the padded packed grid. The integer
// work runs in general-purpose registers.
struct Stencil {
    __m128 frac;
    int base;
};

template <std::size_t D>
BIOIMAGE_FORCE_INLINE Stencil make_stencil(const PackedGrid<D> &g, const __m128 p) {
    const __m128i i0 = _mm_cvttps_epi32(p);
    const __m128 frac = _mm_sub_ps(p, _mm_cvtepi32_ps(i0));
    int base = _mm_cvtsi128_si32(i0) * g.pstride[0] + _mm_extract_epi32(i0, 1) * g.pstride[1];
    if constexpr (D == 3) {
        base += _mm_extract_epi32(i0, 2) * g.pstride[2];
    }
    return Stencil{frac, base};
}

// All channels of the flow at packed position p = [z, y, x, 0] (3D) or
// [y, x, 0, 0] (2D). Lanes beyond D hold finite junk (the next voxel's first
// channel or tail slack); the caller cleans them where it matters.
// 3D: nested trilinear lerps, the same association order as the scalar kernel.
// (A weighted-sum tree with a shorter dependent chain was measured within noise
// here because the 3D loop is bound by the reorder window, not by FP register
// pressure; see the 2D sampler.)
BIOIMAGE_FORCE_INLINE __m128 sample_packed(const PackedGrid<3> &g, const __m128 p) {
    const Stencil s = make_stencil<3>(g, p);
    constexpr int dx = 3;
    const int dy = g.pstride[1];
    const int dz = g.pstride[0];
    const float *p0 = g.data + s.base;
    const float *p1 = p0 + dz;
    // Each load is [c0, c1, c2, junk]: junk is the next voxel's first channel
    // or the tail slack, always finite.
    const __m128 c000 = _mm_loadu_ps(p0);
    const __m128 c001 = _mm_loadu_ps(p0 + dx);
    const __m128 c010 = _mm_loadu_ps(p0 + dy);
    const __m128 c011 = _mm_loadu_ps(p0 + dy + dx);
    const __m128 c100 = _mm_loadu_ps(p1);
    const __m128 c101 = _mm_loadu_ps(p1 + dx);
    const __m128 c110 = _mm_loadu_ps(p1 + dy);
    const __m128 c111 = _mm_loadu_ps(p1 + dy + dx);
    const __m128 fz = _mm_shuffle_ps(s.frac, s.frac, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128 fy = _mm_shuffle_ps(s.frac, s.frac, _MM_SHUFFLE(1, 1, 1, 1));
    const __m128 fx = _mm_shuffle_ps(s.frac, s.frac, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128 c00 = lerp(c000, c001, fx);
    const __m128 c01 = lerp(c010, c011, fx);
    const __m128 c10 = lerp(c100, c101, fx);
    const __m128 c11 = lerp(c110, c111, fx);
    const __m128 c0 = lerp(c00, c01, fy);
    const __m128 c1 = lerp(c10, c11, fy);
    return lerp(c0, c1, fz);
}

// 2D: bilinear interpolation as a weighted sum of the four corners. The four
// weights are formed from the fractional coordinates off the load path, so the
// dependent chain after the loads is mul -> fma -> add (11 cycles) instead of
// two nested lerp levels (16 cycles). The 2D loop is bound by FP register-file
// occupancy, where this measured 12 % faster. The association order differs
// from the scalar kernel's nested lerps; the value is the same bilinear
// interpolation.
BIOIMAGE_FORCE_INLINE __m128 sample_packed(const PackedGrid<2> &g, const __m128 p) {
    const Stencil s = make_stencil<2>(g, p);
    const float *r0p = g.data + s.base;
    // Row loads: [c0(x0), c1(x0), c0(x1), c1(x1)]; x1 = x0 + 1 is valid thanks
    // to the padding column.
    const __m128 r0 = _mm_loadu_ps(r0p);
    const __m128 r1 = _mm_loadu_ps(r0p + g.pstride[0]);
    const __m128 f = s.frac;                                                    // [fy, fx, 0, 0]
    const __m128 omf = _mm_sub_ps(_mm_set1_ps(1.0f), f);                        // [gy, gx, 1, 1]
    const __m128 vx = _mm_shuffle_ps(omf, f, _MM_SHUFFLE(1, 1, 1, 1));          // [gx, gx, fx, fx]
    const __m128 w0 = _mm_mul_ps(vx, _mm_shuffle_ps(omf, omf, _MM_SHUFFLE(0, 0, 0, 0)));  // row y0
    const __m128 w1 = _mm_mul_ps(vx, _mm_shuffle_ps(f, f, _MM_SHUFFLE(0, 0, 0, 0)));      // row y1
    const __m128 t = _mm_fmadd_ps(r1, w1, _mm_mul_ps(r0, w0));  // [x0 terms | x1 terms]
    return _mm_add_ps(t, _mm_movehl_ps(t, t));                   // lanes 0,1 = the sum
}

struct StepConstants {
    __m128 zero;
    __m128 half;
    __m128 half_dt;
    __m128 dt;
    __m128 tol;
    __m128 sign_bit;
};

inline StepConstants make_step_constants(const float dt, const float tol) {
    return StepConstants{
        _mm_setzero_ps(),
        _mm_set1_ps(0.5f),
        _mm_set1_ps(0.5f * dt),
        _mm_set1_ps(dt),
        _mm_set1_ps(tol),
        _mm_set1_ps(-0.0f),
    };
}

constexpr int kStillAlive = -1;

// One RK2 step of a single particle with convergence and mask termination.
// Returns kStillAlive, or the ActiveTraceStats exit reason if tracing stops.
template <std::size_t D>
BIOIMAGE_FORCE_INLINE int step_packed(
    __m128 &pos, const PackedGrid<D> &g, const StepConstants &c, const std::uint8_t *mask
) {
    // The junk lane of the first sample is neutralised by the midpoint clip
    // (upper bound 0 in unused lanes); the second sample is masked explicitly
    // because it feeds the convergence test, the bounds test and the position.
    __m128 step = sample_packed(g, pos);
    __m128 mid = _mm_fmadd_ps(c.half_dt, step, pos);
    mid = _mm_min_ps(_mm_max_ps(mid, c.zero), g.upper_f);
    step = _mm_and_ps(sample_packed(g, mid), g.lane_mask);

    // Scalar kernel: converged iff max_axis |dt*step| < tol, i.e. no lane has
    // |dt*step| >= tol.
    const __m128 disp = _mm_mul_ps(c.dt, step);
    const __m128 abs_disp = _mm_andnot_ps(c.sign_bit, disp);
    if (_mm_movemask_ps(_mm_cmpge_ps(abs_disp, c.tol)) == 0) {
        return ActiveTraceStats::Converged;
    }

    const __m128 proposed = _mm_add_ps(pos, disp);
    const __m128 outside =
        _mm_or_ps(_mm_cmplt_ps(proposed, c.zero), _mm_cmpgt_ps(proposed, g.upper_f));
    if (_mm_movemask_ps(outside) != 0) {
        return ActiveTraceStats::LeftMask;
    }
    // Round half up to the mask voxel (proposed is in-bounds and nonnegative
    // here, so the truncating conversion equals floor(x + 0.5)).
    const __m128i rounded = _mm_cvttps_epi32(_mm_add_ps(proposed, c.half));
    int flat = _mm_cvtsi128_si32(rounded) * g.mstride[0] + _mm_extract_epi32(rounded, 1) * g.mstride[1];
    if constexpr (D == 3) {
        flat += _mm_extract_epi32(rounded, 2) * g.mstride[2];
    }
    if (mask[flat] == 0) {
        return ActiveTraceStats::LeftMask;
    }
    pos = proposed;
    return kStillAlive;
}

template <std::size_t D>
BIOIMAGE_FORCE_INLINE __m128 load_position(const std::array<float, D> &p) {
    if constexpr (D == 3) {
        return _mm_setr_ps(p[0], p[1], p[2], 0.0f);
    } else {
        return _mm_setr_ps(p[0], p[1], 0.0f, 0.0f);
    }
}

template <std::size_t D>
BIOIMAGE_FORCE_INLINE void store_position(std::array<float, D> &p, const __m128 v) {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, v);
    for (std::size_t axis = 0; axis < D; ++axis) {
        p[axis] = tmp[axis];
    }
}

template <class F, std::size_t... Ks>
BIOIMAGE_FORCE_INLINE void for_each_lane(F &&f, std::index_sequence<Ks...>) {
    (f(std::integral_constant<std::size_t, Ks>{}), ...);
}

// Trace K consecutive particles in lockstep (see trace_particle_block in the
// header). Lanes are addressed with compile-time indices so their state stays
// in registers.
template <std::size_t D, std::size_t K>
void trace_block_packed(
    std::array<float, D> *positions,
    const PackedGrid<D> &g,
    const StepConstants &c,
    const std::uint8_t *mask,
    const std::size_t n_iter,
    ActiveTraceStats &stats
) {
    __m128 pos[K];
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

// Lanes traced in lockstep per group. K = 4 measured 1-6 % faster than K = 3
// on the registered fixtures and every adversarial case (2026-09-02, Zen 3);
// K = 6 was parity, K = 2 and K = 1 lose.
constexpr std::size_t kPackedLanes = 4;

template <std::size_t D>
void trace_all_fma(
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
            BIOIMAGE_PROFILE_SCOPE(profiler, "fma_pack");
            ready = make_packed_grid<D>(channels, grid, n_threads, storage, packed);
        }
        if (ready) {
            BIOIMAGE_PROFILE_SCOPE(profiler, "fma_trace_packed");
            trace_all_packed<D, kPackedLanes>(positions, packed, mask, n_threads, n_iter, dt, tol);
            BIOIMAGE_PROFILE_REPORT_NAMED(profiler, "[bioimage profile: flow fma]");
            return;
        }
    }
    // Fallback: packed offsets would not fit int32 or the buffer could not be
    // allocated. Same header kernel as the portable path, compiled with FMA.
    {
        BIOIMAGE_PROFILE_SCOPE(profiler, "fma_trace_scalar");
        trace_all<D, true, true, true, true>(
            positions, channels, grid, mask, n_threads, n_iter, dt, tol
        );
    }
    BIOIMAGE_PROFILE_REPORT_NAMED(profiler, "[bioimage profile: flow fma]");
}

} // namespace

void trace_all_fma_2d(
    std::vector<std::array<float, 2>> &positions,
    const std::array<const float *, 2> &channels,
    const GridLayout<2> &grid,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    trace_all_fma<2>(positions, channels, grid, mask, n_threads, n_iter, dt, tol);
}

void trace_all_fma_3d(
    std::vector<std::array<float, 3>> &positions,
    const std::array<const float *, 3> &channels,
    const GridLayout<3> &grid,
    const std::uint8_t *mask,
    const std::size_t n_threads,
    const std::size_t n_iter,
    const float dt,
    const float tol
) {
    trace_all_fma<3>(positions, channels, grid, mask, n_threads, n_iter, dt, tol);
}

} // namespace bioimage_cpp::flow::detail
