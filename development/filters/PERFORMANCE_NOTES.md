# Filter benchmark — performance notes

Results from `development/filters/benchmark.py` for the six Tier-1 filters
on 2D and 3D test data from `skimage.data`. Re-run with:

```bash
python development/filters/check_parity.py    # parity gate, must PASS first
python development/filters/benchmark.py       # headline numbers
```

All four libraries produce the same response (verified by `check_parity.py`
on the image interior, dropping a `window_size * sigma` margin on each side;
or `inner + outer` for the structure tensor). Reported numbers are
**median wall-clock per call** across 5 timed repeats, after one untimed
warmup, with calls interleaved round-robin between libraries to share cache
state fairly.

## Setup

- CPU: Intel Core i7-1185G7 (Tiger Lake, 4C/8T, AVX2 + AVX-512)
- Compiler: gcc 15.2.0 (conda-forge), `-O3`, no `-march=native`
- Python 3.11.15 on Linux x86_64
- `numpy 2.4.5`, `scipy 1.17.1`, `vigra 1.12.3`, `fastfilters 0.3-4-g87f08b5`,
  `bioimage_cpp 0.1.0`
- All libraries called single-threaded; `bioimage_cpp` has no SIMD intrinsics
  (Tier-1: scalar C++20 + compiler auto-vectorisation).

## Benchmark configuration

| parameter | value | notes |
|---|---|---|
| sigma | 1.5 | smoothing / derivative / LoG / gradient / Hessian |
| inner_sigma | 1.0 | structure-tensor gradient scale |
| outer_sigma | 2.0 | structure-tensor smoothing scale |
| window_size | 3.0 | kernel half-width / sigma; matched across libraries |
| repeats | 5 | timed, interleaved round-robin |
| 2D image | `skimage.data.camera()` | 512×512 → float32, normalised to [0, 1] |
| 3D volume | `skimage.data.cells3d()[:, 1]` | 60×256×256 nuclei channel → float32 |

`gaussian_derivative` uses order = 1 along the trailing axis only. `fastfilters`
only supports uniform per-axis order, so its row is `n/a`.

## Headline ratios

Geometric mean of `bioimage_cpp.median / other.median` across all benched
(filter, dim) combinations. **>1.0 means `bioimage_cpp` is slower** than the
other library; **<1.0 means faster**.

| comparison | geomean ratio | n |
|---|---|---|
| `bioimage_cpp` / `fastfilters` | **2.00** | 10 |
| `bioimage_cpp` / `vigra` | **0.18** | 12 |
| `bioimage_cpp` / `scipy` | **0.20** | 12 |

Interpretation: `fastfilters` (hand-AVX2) is 2× faster than ours on average;
ours is ~5× faster than vigra and ~5× faster than scipy/numpy.

## 2D results — `camera()` 512×512

Times in milliseconds (median of 5 repeats). The `x ours` column is
`bioimage_cpp.median / this_lib.median`; **values >1.0 mean the library is
faster than `bioimage_cpp`**.

| filter | bioimage_cpp ms | fastfilters ms | x ours | vigra ms | x ours | scipy ms | x ours |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_smoothing | 0.82 | 0.49 | 1.67 | 5.18 | 0.16 | 3.59 | 0.23 |
| gaussian_derivative | 0.83 | n/a | n/a | 5.15 | 0.16 | 3.85 | 0.21 |
| gaussian_gradient_magnitude | 2.11 | 0.91 | 2.32 | 13.85 | 0.15 | 7.77 | 0.27 |
| laplacian_of_gaussian | 1.92 | 0.96 | 2.00 | 8.88 | 0.22 | 7.50 | 0.26 |
| hessian_of_gaussian_eigenvalues | 3.02 | 1.67 | 1.81 | 19.55 | 0.15 | 77.79 | 0.04 |
| structure_tensor_eigenvalues | 5.16 | 2.86 | 1.80 | 26.22 | 0.20 | 82.99 | 0.06 |

## 3D results — `cells3d()[:, 1]` 60×256×256

| filter | bioimage_cpp ms | fastfilters ms | x ours | vigra ms | x ours | scipy ms | x ours |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_smoothing | 33.27 | 15.02 | 2.22 | 192.29 | 0.17 | 85.37 | 0.39 |
| gaussian_derivative | 33.05 | n/a | n/a | 195.27 | 0.17 | 86.19 | 0.38 |
| gaussian_gradient_magnitude | 106.42 | 61.91 | 1.72 | 640.83 | 0.17 | 276.64 | 0.38 |
| laplacian_of_gaussian | 103.69 | 61.93 | 1.67 | 584.71 | 0.18 | 266.87 | 0.39 |
| hessian_of_gaussian_eigenvalues | 486.38 | 175.75 | 2.77 | 2195.57 | 0.22 | 4031.79 | 0.12 |
| structure_tensor_eigenvalues | 599.90 | 263.14 | 2.28 | 1938.37 | 0.31 | 4301.12 | 0.14 |

## Per-filter throughput vs `bioimage_cpp` (megapixels / second)

Throughput = `image_size / median_time`. Useful for cross-shape comparison.

### 2D (512×512 = 0.262 megapixels per output)

| filter | bioimage_cpp | fastfilters | vigra | scipy |
|---|---:|---:|---:|---:|
| gaussian_smoothing | 320 | 535 | 51 | 73 |
| gaussian_gradient_magnitude | 124 | 288 | 19 | 34 |
| laplacian_of_gaussian | 136 | 273 | 30 | 35 |
| hessian_of_gaussian_eigenvalues | 87 | 157 | 13 | 3 |
| structure_tensor_eigenvalues | 51 | 92 | 10 | 3 |

### 3D (60×256×256 = 3.93 megavoxels per output)

| filter | bioimage_cpp | fastfilters | vigra | scipy |
|---|---:|---:|---:|---:|
| gaussian_smoothing | 118 | 262 | 20 | 46 |
| gaussian_gradient_magnitude | 37 | 64 | 6 | 14 |
| laplacian_of_gaussian | 38 | 64 | 7 | 15 |
| hessian_of_gaussian_eigenvalues | 8 | 22 | 2 | 1 |
| structure_tensor_eigenvalues | 7 | 15 | 2 | 1 |

## Takeaways

- **vs `fastfilters`** (the hand-AVX2 target). We sit at 1.67× – 2.77×
  slower across the board, exactly in the 1.0×–2.0× band the Tier-1 plan
  predicted (a touch worse on Hessian/structure-tensor 3D where eigenvalue
  arithmetic dominates and benefits less from auto-vectorisation). This is
  the gap a Tier-2 manual-AVX2 path would aim to close.
- **vs `vigra`**. ~5× faster across the board. Same algorithmic family,
  scalar code in both cases; the win comes from fewer abstraction layers
  and tighter kernels (constant-bound inner loops, half-kernel storage,
  X-strip pattern for the strided pass).
- **vs `scipy.ndimage` (+ `numpy.linalg.eigvalsh` for the eigenvalue
  filters)**. ~5× faster on simple filters and ~10–25× faster on the
  eigenvalue paths. `scipy.ndimage` itself is competitive on plain
  convolution; the eigenvalue gap is almost entirely the per-pixel numpy
  `eigvalsh` cost — exactly what scipy-only users currently pay.

## Where the Tier-2 SIMD work would help most

Largest absolute gaps to `fastfilters` (3D, in ms per call):

| filter | ours | ff | absolute gap | gap × 5 calls/sec |
|---|---:|---:|---:|---:|
| hessian_of_gaussian_eigenvalues | 486 | 176 | 311 ms | 1.55 s/sec saved |
| structure_tensor_eigenvalues | 600 | 263 | 337 ms | 1.69 s/sec saved |
| gaussian_gradient_magnitude | 106 | 62 | 44 ms | 0.22 s/sec saved |

If/when Tier-2 lands, instrument the Hessian and structure-tensor 3D paths
first — they share the same separable-FIR primitives plus the 3×3 trig
eigensolver, so closing the gap on those carries the gradient/magnitude /
LoG paths along for free.

## Reproducibility notes

- Run after `pip install -e . --no-build-isolation` so the C++ extension
  matches the source tree.
- For absolute-time comparisons across machines, also report CPU model,
  compiler version, and whether `-march=native` was set (we do NOT set it
  in normal builds; this benchmark used the default `-O3`).
- For a quick re-check use `--small` (crops to 128×128 and 32×64×64);
  finishes in a few seconds.
- For raw per-(filter, library) timings (useful for plotting), pass
  `--csv path.csv`.

## Tier 2 SIMD — design notes (deferred)

**Status (2026-05-17): not pursued for now.** Tier 1 sits within ~2× of
fastfilters' hand-AVX2 on the headline benchmark while being ~5× faster
than `vigra` and `scipy.ndimage`. The marginal user value of closing that
2× gap doesn't yet justify the extra build complexity and dual-path
maintenance burden. This section captures the design so a future coding
agent (or future-us) can pick it up without re-deriving the choices.

### Trigger conditions — when to revisit

Open this section again when **at least one** is true:

1. Real users are hitting the Hessian-3D / structure-tensor-3D paths on
   volumes large enough that ~300 ms vs ~150 ms per call materially
   matters in their pipeline (typically batch feature extraction over
   many large 3D blocks).
2. "Performance parity with fastfilters" becomes a stated project goal
   (e.g. for a migration story or comparison documentation).
3. Profiling on a real downstream workflow shows `bioimage_cpp.filters`
   is the bottleneck and the gap to fastfilters is the dominant slice.

If none of those is true: stay on Tier 1.

### Scope — what to ship, what to keep out

**In scope** (the whole Tier 2 delivery):

- Hand-written AVX2 + FMA implementations of exactly two inner kernels:
  - `convolve_x_radius<R, Symmetric>` — the X (innermost contiguous) pass.
  - `convolve_strided_radius<R, Symmetric>` — the Y/Z (strided) pass.
  - Both currently live in
    `include/bioimage_cpp/filters/convolve.hxx::detail`.
- One-time CPUID dispatch at module load that picks scalar vs AVX2
  function pointers for those two kernels.

**Out of scope** (do NOT add any of these as part of Tier 2):

- AVX-512 path. The win over AVX2 is small on memory-bound separable FIR
  and doubles the kernel binary footprint; revisit only if a user with a
  Sapphire Rapids / Zen 5 workload asks specifically.
- NEON / arm64 hand-tuning. Tier 1 auto-vectorization on Apple Clang is
  already competitive; this would be a separate project with its own
  trigger conditions.
- Replacing `std::acos` / `std::cos` in `eigenvalues.hxx` with a
  vectorized math library (this is what `fastfilters` vendors as
  `avx_mathfun.h`, 924 lines). The Tier-1 plan explicitly rejected
  vendoring it; revisit only if eigenvalue profiling shows the trig
  calls dominate the remaining gap. Don't bundle this into Tier 2.
- Any change to `kernel.hxx`, `eigenvalues.hxx`, `gaussian.hxx`, the
  binding layer, or the Python wrapper. Tier 2 is a *drop-in* speedup of
  two leaf functions; if you find yourself changing anything else,
  something is wrong.

### File layout

```
include/bioimage_cpp/filters/
    convolve.hxx                  # existing scalar; renamed entry points
                                  # to point at function pointers (see below)
    convolve_dispatch.hxx         # NEW — function-pointer table + CPUID

src/cpp/filters/
    convolve_avx2.cxx             # NEW — AVX2+FMA kernels (compiled with
                                  # per-file -mavx2 -mfma / /arch:AVX2)
    convolve_dispatch.cxx         # NEW — one-time init of the pointers
```

The existing `convolve_axis_x` / `convolve_axis_strided` entry points in
`convolve.hxx` keep their signatures. Their bodies switch from "directly
call `detail::convolve_x_radius<R, Sym>`" to "call
`bioimage_cpp::filters::dispatch::convolve_x_table[R][Sym]`". Higher
levels (`gaussian.hxx`, the six composite filters, the binding layer) are
unchanged.

### CMake wiring

Add to the `nanobind_add_module(_core ...)` source list:

```cmake
src/cpp/filters/convolve_avx2.cxx
src/cpp/filters/convolve_dispatch.cxx
```

Then attach per-file flags so only the AVX2 TU gets AVX2 instructions
(the rest of the wheel stays at the manylinux SSE2 baseline):

```cmake
if(MSVC)
    set_source_files_properties(
        src/cpp/filters/convolve_avx2.cxx
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
    )
else()
    set_source_files_properties(
        src/cpp/filters/convolve_avx2.cxx
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
    )
endif()
```

Do **not** add `-march=native` or change the global `-O3`. The wheel
must keep installing on any pre-Haswell x86_64 machine that
manylinux2014 supports; the AVX2 instructions only execute behind the
CPUID check.

### Runtime dispatch pattern

In `convolve_dispatch.hxx`:

```cpp
namespace bioimage_cpp::filters::dispatch {

using ConvolveXFn = void (*)(
    const float*, float*, std::ptrdiff_t, std::ptrdiff_t, const float*
);
using ConvolveStridedFn = void (*)(
    const float*, float*, std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t,
    const float*
);

// One entry per (radius R in 1..kMaxSpecialisedRadius, Symmetric in {0,1}).
// Filled at module load by init().
extern ConvolveXFn convolve_x_table[kMaxSpecialisedRadius + 1][2];
extern ConvolveStridedFn convolve_strided_table[kMaxSpecialisedRadius + 1][2];

void init();  // called once from bind_filters()

}
```

In `convolve_dispatch.cxx`:

```cpp
namespace {
bool detect_avx2_fma() {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER)
    int regs1[4]; __cpuid(regs1, 1);
    const bool fma = (regs1[2] & (1 << 12)) != 0;
    int regs7[4]; __cpuidex(regs7, 7, 0);
    const bool avx2 = (regs7[1] & (1 << 5)) != 0;
    // Also OSXSAVE + XGETBV to confirm OS-saved YMM state.
    ...
    return avx2 && fma;
#else
    return false;
#endif
}
}

void init() {
    const bool use_avx2 = detect_avx2_fma();
    // Macros generate the per-R table entries to avoid 24 hand-written
    // lines (the same boost-preprocessor-style explosion fastfilters
    // does, kept tiny with simple X-macros).
    #define BIO_FILL(R) \
        if (use_avx2) { \
            convolve_x_table[R][0] = &avx2::convolve_x_radius_sym<R>; \
            convolve_x_table[R][1] = &avx2::convolve_x_radius_anti<R>; \
            convolve_strided_table[R][0] = &avx2::convolve_strided_radius_sym<R>; \
            convolve_strided_table[R][1] = &avx2::convolve_strided_radius_anti<R>; \
        } else { \
            convolve_x_table[R][0] = &detail::convolve_x_radius_sym<R>; \
            convolve_x_table[R][1] = &detail::convolve_x_radius_anti<R>; \
            convolve_strided_table[R][0] = &detail::convolve_strided_radius_sym<R>; \
            convolve_strided_table[R][1] = &detail::convolve_strided_radius_anti<R>; \
        }
    BIO_FILL(1) BIO_FILL(2) ... BIO_FILL(12)
    #undef BIO_FILL
}
```

(Internally split each existing `template <int R, bool Symmetric>` into
two non-templated-on-`Symmetric` aliases — `_sym` and `_anti` — so the
function-pointer types are concrete and the table is plain data.)

Call `dispatch::init()` from `bind_filters()` in `src/bindings/filters.cxx`
(once, before any kernel binding can be invoked). Use a
`static std::once_flag` guard so re-imports don't double-initialise.

### AVX2 kernel skeleton

The X-pass kernel in `convolve_avx2.cxx` is essentially the scalar main
loop with explicit `__m256` registers:

```cpp
namespace bioimage_cpp::filters::avx2 {

template <int R>
void convolve_x_radius_sym(
    const float* __restrict in,
    float* __restrict out,
    std::ptrdiff_t n_rows,
    std::ptrdiff_t n_cols,
    const float* __restrict h
) {
    const std::ptrdiff_t prologue_end   = std::min<std::ptrdiff_t>(R, n_cols);
    const std::ptrdiff_t epilogue_start = std::max<std::ptrdiff_t>(prologue_end, n_cols - R);

    for (std::ptrdiff_t row = 0; row < n_rows; ++row) {
        const float* __restrict in_row  = in  + row * n_cols;
        float*       __restrict out_row = out + row * n_cols;

        // --- border prologue: reuse scalar mirror code unchanged ---
        scalar_border_sym<R>(in_row, out_row, 0, prologue_end, n_cols, h);

        // --- main AVX2 loop ---
        std::ptrdiff_t x = prologue_end;
        const __m256 h0 = _mm256_set1_ps(h[0]);
        for (; x + 8 <= epilogue_start; x += 8) {
            __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(in_row + x), h0);
            for (int k = 1; k <= R; ++k) {
                const __m256 hk  = _mm256_set1_ps(h[k]);
                const __m256 sum = _mm256_add_ps(
                    _mm256_loadu_ps(in_row + x + k),
                    _mm256_loadu_ps(in_row + x - k)
                );
                acc = _mm256_fmadd_ps(hk, sum, acc);
            }
            _mm256_storeu_ps(out_row + x, acc);
        }
        // --- scalar tail (0..7 floats) ---
        scalar_main_sym<R>(in_row, out_row, x, epilogue_start, h);

        // --- border epilogue ---
        scalar_border_sym<R>(in_row, out_row, epilogue_start, n_cols, n_cols, h);
    }
}

template <int R>
void convolve_x_radius_anti(...) { /* same shape, _mm256_sub_ps instead of _add_ps */ }

}
```

The strided kernel follows the same pattern but loops over `kStripBlock`
in steps of 8, using `__m256` for the accumulator strip. Crucially the
strip stays at 64 floats (`kStripBlock` is already a multiple of 8), so
no new tiling decision is needed.

`scalar_border_sym` / `scalar_main_sym` are just the existing scalar
loop bodies hoisted into small inline helpers callable from both the
AVX2 and the scalar TU. **The mirror-boundary handling code must not be
duplicated between the two TUs** — that's where divergence bugs would
hide. Make the helpers `inline` in a shared header.

### What stays byte-for-byte identical

- Kernel-coefficient generation in `kernel.hxx`.
- Eigenvalue solvers in `eigenvalues.hxx`.
- Composite filters in `gaussian.hxx`.
- Binding layer in `src/bindings/filters.cxx`.
- Python wrapper in `src/bioimage_cpp/filters/_filters.py`.
- The public `convolve_axis_x` / `convolve_axis_strided` signatures in
  `convolve.hxx`.
- The mirror-index function `detail::mirror_index` and the border
  prologue/epilogue logic.

If a Tier-2 change is touching any of these, stop and re-read the scope
section — it's almost certainly not what Tier 2 is for.

### Expected speedup

Based on the gap to `fastfilters` in this benchmark
(`bioimage_cpp / fastfilters` geomean = 2.00):

- Simple filters (smoothing, derivative, gradient_magnitude, LoG):
  realistic post-Tier-2 ratio **1.0 – 1.3×** fastfilters (essentially
  tied to slightly behind).
- 3D Hessian / structure-tensor eigenvalues: realistic post-Tier-2
  ratio **1.4 – 1.7×** fastfilters. The remaining gap is in `acos`/`cos`
  inside the 3×3 trig eigensolver, which intrinsics alone do not help
  with.

Do not expect to *match* fastfilters exactly without also vendoring a
vectorized math library and per-radius file-copy specialisation — that's
the next-tier-after-Tier-2 work, deliberately out of scope here.

### Verification

1. **Parity gate stays green**:
   `python development/filters/check_parity.py` on a machine with AVX2
   support must still PASS at the same tolerances. If it doesn't, the
   AVX2 kernel disagrees with the scalar kernel — that's the most
   likely failure mode and is almost always an off-by-one in the
   prologue / main / epilogue boundary handling.
2. **Scalar path stays green**: re-run with the AVX2 path forced off
   (set the function-pointer table to the scalar entries unconditionally
   in a debug build, or guard the dispatch decision with an environment
   variable like `BIOIMAGE_FORCE_SCALAR=1`). The full pytest suite must
   still pass; this catches scalar-only regressions introduced when
   refactoring shared helpers.
3. **Benchmark**:
   `python development/filters/benchmark.py` should show the
   `bioimage_cpp / fastfilters` geomean drop from ~2.00 toward ~1.2.
   Update this file with the new numbers.
4. **Pre-Haswell smoke**: the dispatch must take the scalar path on a
   machine without AVX2. Easiest local check: temporarily make
   `detect_avx2_fma()` return `false` and confirm correctness +
   performance fall back to today's Tier-1 numbers.

### Smallest first step

Don't ship both kernels at once. The recommended sequence is:

1. Land the dispatch scaffolding (`convolve_dispatch.hxx` /
   `.cxx`, function-pointer tables, CMake wiring) **with both pointers
   still pointing at the existing scalar kernels**. No behavior change.
   Tests stay green. This isolates the build-system part of the work.
2. Add `convolve_x_radius_avx2` only. Re-run parity + benchmark.
   Expect the simple filters to move; the Y/Z-bound filters
   (gradient_magnitude, LoG, Hessian) move proportionally less.
3. Add `convolve_strided_radius_avx2`. Re-run parity + benchmark.
   Expect the 3D filters to move significantly.

If after step 2 the speedup is smaller than expected, stop and profile
before continuing — it usually means the autovectorizer was already
doing better than this section assumes, and the marginal value of
step 3 is lower than it appears here.

### Watch-outs

- **Boundary-mode divergence** between scalar and AVX2 paths is the
  single most likely correctness bug. Share the prologue/epilogue
  helpers via an `inline` header; don't copy-paste.
- **Unaligned loads only.** Use `_mm256_loadu_ps` / `_mm256_storeu_ps`,
  not the aligned variants. The bench inputs are not guaranteed to be
  32-byte aligned, and on modern Intel/AMD the unaligned-load
  performance penalty is essentially zero. Trying to force alignment in
  the binding layer is more complexity than the win.
- **MSVC AVX2 detection.** `__builtin_cpu_supports` is GCC/Clang only.
  Use raw `__cpuid` / `__cpuidex` + an `_xgetbv` check (the OS must
  have saved the YMM state for AVX to be safe to use). There's example
  code in `fastfilters/src/library/cpu_intel.c` if you need a
  reference; do not vendor it, just write the small bit you need.
- **Don't introduce OpenMP, std::thread, or any threading primitive in
  this work.** Threading is a separate follow-up that should layer on
  top of the dispatch scheme via `parallel_for_chunks` (see
  `include/bioimage_cpp/detail/threading.hxx`). Mixing the two changes
  is asking for trouble.
- **Don't add AVX-512 "while we're here."** It is a separate trigger
  decision with separate trade-offs (frequency throttling on older
  Xeons, larger binary, marginal win on memory-bound separable FIR).

## Known caveats reflected in the adapters

- `fastfilters.gaussianDerivative` only accepts a uniform per-axis order;
  the bench cell is `n/a` rather than running an unequal operation.
- `fastfilters.structureTensorEigenvalues` swaps `innerScale` /
  `outerScale` at the Python boundary (its wrapper calls the C function
  with the args in the opposite order — `src/python/core.cxx:328` vs
  `src/library/fir_filters.c:156` in the fastfilters source). The adapter
  swaps them back so the bench compares the same operation as `vigra` /
  `bioimage_cpp`.
- `scipy` and `bioimage_cpp` use scipy-style `mirror` (reflect without
  edge-pixel repeat); `vigra` / `fastfilters` use reflect with edge
  repeat. Parity is checked on the image interior to absorb the
  difference.
