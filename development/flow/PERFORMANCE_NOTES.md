# `compute_flow_density` — performance notes

A running record of the optimization work on `bioimage_cpp.flow.compute_flow_density`.
Captures the baseline, the changes that landed, the experiments that were rolled
back, and a roofline-style diagnosis of where remaining time goes.

Test data, here and below, refers to the registered `flow_data_2d.h5` /
`flow_data_3d.h5` fixtures (`bioimage_cpp._data.load_flow_data`), invoked via
`development/flow/check_flow_density.py`.

Hardware on which these numbers were taken: **11th Gen Intel Core i7-1185G7
@ 3.00 GHz** (Tiger Lake, 4 physical cores / 8 SMT threads, 12 MB L3, AVX2 +
AVX-512 capable).

## Headline numbers

```
                       baseline       current      gain
3D, 1 thread        :  5.55 s   →    5.40 s       1.03×
3D, 8 threads       :  5.55 s   →    1.51 s       3.67×
2D, 1 thread        :  0.43 s   →    0.45 s       0.96×
2D, 8 threads       :  0.43 s   →    0.18 s       2.4×

Accuracy (3D vs reference):
  rel_diff = mean(|ours - ref|) / mean(ref) = 0.021   (gate ≤ 0.15)
  pearson                                    = 0.998
```

The big win is threading. Algorithmic changes only marginally help the
single-threaded path; they pay off because they cap the work and give the
threaded path something to scale.

## Changes that landed

In rough order of impact:

1. **Threading via `detail::parallel_for_chunks`** — particle tracing is
   embarrassingly parallel; the final density scatter stays single-threaded
   (`<1 %` of time per the profile). Determinism preserved.
2. **Convergence-based early exit** (`tol` parameter) — per-particle alive
   flag; freeze a particle once `max_axis(|dt·step|) < tol`. Outer iteration
   loop breaks once all particles are frozen.
3. **Mask-restricted tracing** (`restrict_to_mask` parameter) — a particle
   that rounds to a background voxel freezes in place. Combined with (2) it
   sheds slow trajectories that would otherwise run the full `n_iter`.
4. **RK2 (midpoint) integrator** (`method="rk2"`) — two flow samples per
   step but allows larger `dt`; combined with early-exit it converges in
   fewer iterations.
5. **Corner-table sharing inside `sample_channel`** — `compute_corners`
   builds the `2^D` (offset, weight) table once per particle per iteration;
   all `D` channel samples reuse it. Cuts the redundant lower/frac work that
   the original `sample_linear_nearest` did per axis.
6. **Hoisted invariants** out of the per-particle loop (`upper[axis]`,
   channel base pointers, channel stride).
7. **Profiling instrumentation** (`BIOIMAGE_PROFILE_SCOPE`) around `init`,
   `iter_loop`, `scatter`, `mask_zero`. Free when the CMake flag is off; live
   when `-C cmake.define.BIOIMAGE_PROFILE=ON`.

New chosen defaults (see `src/bioimage_cpp/flow/_flow.py`):

```python
n_iter=50, dt=0.2, tol=0.005, method="rk2", restrict_to_mask=True,
number_of_threads=1
```

A caller who wants the threaded speedup passes `number_of_threads=8` (or
whatever value fits their box).

## Profile breakdown (1-thread, 3D, profile build)

```
init        ~0.4 %    density-zero + positions collection
iter_loop  ~99   %    the particle tracing loop
scatter     ~0.1 %    final density write
mask_zero   ~0.1 %    final mask zeroing
```

The iteration loop dominates so completely that none of the other phases
were worth touching. A "single-pass init" cleanup and per-thread scatter
buffers were planned and skipped after the profile.

## Thread scaling (3D, current defaults)

```
threads   runtime   speedup vs 1T
   1      5.40 s    1.00×
   2      3.76 s    1.44×
   4      2.07 s    2.61×
   6      1.71 s    3.17×
   8      1.52 s    3.57×
```

Sub-linear from `1→4` (2.61× on 4 physical cores) and a marginal
improvement from `4→8` (SMT pair on each core). Likely contributors:

- The per-iter `alive` scan is sequential.
- Single-thread TurboBoost is higher than all-core sustained clock.
- Memory subsystem contention even before bandwidth saturation
  (L3 sharing, prefetcher load).

It is **not** memory-bandwidth saturation — see the next section.

## Bandwidth diagnosis (no `perf` available, used a streaming probe)

Sustained streaming read bandwidth, measured via `numpy.sum` on float32
arrays larger than L3:

```
array size   best wall    GB/s
   16 MB      1.44 ms    11.67
   64 MB      5.83 ms    11.52
  144 MB     13.08 ms    11.54   (matches the 3D flow array layout)
  256 MB     23.63 ms    11.36
```

Sustained read bandwidth on this machine: **≈ 11.5 GB/s**.

Upper-bound bytes read from the flow array per `compute_flow_density` call,
3D RK2 path:

```
729 236 particles × 50 iter × 2 samples (RK2) × 3 channels × 8 corners × 4 B
  ≈ 7.0 GB   (upper bound — convergence early-exit reduces this in practice)
```

Effective kernel bandwidth, against that upper bound:

```
1 thread @ 5.40 s   →  1.3 GB/s  =  11 % of sustained streaming
8 threads @ 1.52 s  →  4.6 GB/s  =  40 % of sustained streaming
```

So even at 8 threads the kernel only utilises ~40 % of the memory subsystem.
The conclusion is that **the hot loop is compute-bound**, not
bandwidth-bound. Almost certainly bound by gather latency / per-particle
serial instruction count, not RAM throughput.

## SIMD experiment (rolled back)

GCC/Clang `__attribute__((target_clones("default,arch=haswell,arch=skylake-avx512")))`
was wired up end-to-end so the IFUNC resolver would pick the AVX2 /
AVX-512F clone at first call on capable CPUs. Wheel-portability was
preserved: the attribute is only enabled on Linux x86_64 GCC/Clang; macOS
(Mach-O lacks IFUNC) and MSVC fall back to a single default clone.

The dispatcher was confirmed to work — the IFUNC resolver was emitted, and
the three clones diverged in their object code once the templated kernel was
forced inline via `[[gnu::always_inline]]` + `[[gnu::flatten]]`. Result on
the AVX2 clone:

- ✓ `vfmadd231ss` + ymm registers in scalar FMA encoding.
- ✗ No packed-vector ops (`*ps` instructions, `vfmadd*ps`).
- ✗ No `vpgatherdd` or `vgatherqps`.

GCC's autovectorizer evaluated the `sample_channel` corner-sum (8 gather
loads + multiply-add) and decided the gather pattern wasn't profitable.
Tried in sequence:

1. `target_clones` alone — all clones folded to the same code (the template
   was called via a normal function boundary, target attributes did not
   propagate). Net: no change.
2. `[[gnu::always_inline]] inline` on the template + `[[gnu::flatten]]` on
   the dispatch wrappers — clones diverged but only emitted scalar FMA.
   8-thread runtime regressed from 1.7 s to 2.3 s (icache pressure from
   three full inlined copies in the same TU).
3. `#pragma omp simd reduction(+:value)` on the corner sum + `-fopenmp-simd`
   in CMake — packed SIMD still did not materialise.
4. Narrowing corner offsets from `ptrdiff_t` to `int32_t` to enable 8-wide
   `vpgatherdd` — no codegen change observed.

Final decision: **roll back**. The complexity was non-zero (a new TU, IFUNC
plumbing, force-inline annotations on the templated kernel, an `omp-simd`
flag) and the 8-thread regression was real.

What would actually work, but was out of scope for the session: hand-written
AVX2 intrinsics for `sample_channel<3>` (and `<2>`), specifically:

```cpp
__attribute__((target("avx2,fma")))
inline float sample_channel_avx2_3d(const float *channel,
                                    const SamplingCorners<3> &c) {
    const __m256  w = _mm256_loadu_ps(c.weights.data());
    const __m256i o = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(c.offsets.data()));
    const __m256  v = _mm256_i32gather_ps(channel, o, sizeof(float));
    const __m256  p = _mm256_mul_ps(w, v);
    // horizontal reduce
    __m128 lo = _mm256_castps256_ps128(p);
    __m128 hi = _mm256_extractf128_ps(p, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}
```

Wired up behind a runtime dispatch (`__builtin_cpu_supports("avx2")`) so the
wheel still loads on SSE2-only CPUs. Expected speedup at 1T: ~20–30 %, less
at 8T due to closer-to-bandwidth saturation. Estimated effort: half a day
plus a careful microbenchmark of `vpgatherdd` cost on the target CPUs.

## What was tried and rejected

- **Per-thread density scatter buffers** — scatter is <1 % of runtime,
  not worth it.
- **Single-pass init** (fuse density-zero with positions collection) — init
  is 0.4 % of runtime, not worth it.
- **Active-list compaction** (rebuild a `vector<size_t> active_idx` to skip
  dead-particle scans) — projected impact <1 %; the current alive-byte
  check is one load + one branch.
- **`target_clones` SIMD** — see above, no net win, rolled back.
- **Half-precision (fp16) flow storage** — would halve memory traffic, but
  the kernel isn't bandwidth-bound (see diagnosis), and it's a breaking API
  change.

## What remains worth trying

In rough order of expected return on effort:

1. **Hand-written AVX2 (and SSE2-fallback) intrinsics for the corner sum**
   inside a runtime-dispatched specialization. Plausible 20–30 % at 1T. The
   plumbing scaffolding is documented above in the SIMD section.
2. **SoA position layout** (`vector<float> pos_x, pos_y, pos_z` instead of
   `vector<array<float,3>>`). Lets the autovectorizer batch the clip + step
   + convergence check across N particles. Pairs naturally with (1).
3. **Reduce gather miss latency** via batch prefetching: prefetch the
   next-K particles' corner cache lines before computing the current one.
   Likely modest gain; needs profiling.
4. **A real `perf stat`** run on a host where `linux-perf-tools` is
   installed, to confirm where cycles actually go (front-end, back-end
   memory, back-end core, retired-FMA-ratio). The streaming-probe upper
   bound is necessary but not sufficient — counter data would localise the
   bottleneck precisely.

## Reproducing these measurements

```bash
# Build with profiling instrumentation
pip install -e . --no-build-isolation -C cmake.define.BIOIMAGE_PROFILE=ON

# Per-phase breakdown
python development/flow/check_flow_density.py --dim both --repeats 1

# Production build (no profile)
pip install -e . --no-build-isolation

# Thread scaling
for nt in 1 2 4 6 8; do
    python development/flow/check_flow_density.py --dim 3 --repeats 3 --threads $nt
done

# Accuracy gate at the chosen defaults
python development/flow/check_flow_density.py --dim both --repeats 3
```

`check_flow_density.py` accepts `--method`, `--dt`, `--tol`, `--n-iter`,
`--restrict-to-mask` / `--no-restrict-to-mask`, and `--threads` to override
defaults; useful for sweep work. The PASS gate is
`rel_diff = mean(|ours-ref|)/mean(ref) ≤ 0.15` by default
(`--rel-tol` to override).
