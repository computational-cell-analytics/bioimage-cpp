# `compute_flow_density` — performance notes

A running record of the optimization work on `bioimage_cpp.flow.compute_flow_density`.
Captures the baseline, the changes that landed, the experiments that were rolled
back, and a roofline-style diagnosis of where remaining time goes.

Test data, here and below, refers to the registered `flow_data_2d.h5` /
`flow_data_3d.h5` fixtures (`bioimage_cpp._data.load_flow_data`), invoked via
`development/flow/check_flow_density.py`.

Hardware on which these numbers were taken: **11th Gen Intel Core i7-1185G7
@ 3.00 GHz** (Tiger Lake, 4 physical cores / 8 SMT threads, 12 MB L3, AVX2 +
AVX-512 capable). The 2026-07-12 pass below was cross-checked against
GPT 5.6-Sol's independent investigation, which measured the same changes on an
AMD EPYC 7513 (Zen 3); see `FLOW_OPTIM.md` in the repo root. Absolute timings
differ between the hosts; the relative gains agree closely across both.

## Headline numbers

Current state after the 2026-07-12 scalar-kernel rewrite (particle-major +
direct interpolation + truncation + compile-time specialization). Timings are
`check_flow_density.py` medians of 5, measured in one session on the Tiger Lake
host. The "pre-rewrite" column is the previous kernel (generic `2^D` corner
table, iteration-major with a per-iteration alive scan):

```
                    pre-rewrite     current      gain
3D, 1 thread     :  5.66 s     →    2.45 s       2.31×   (−57 %)
3D, 8 threads    :  1.51 s     →    0.64 s       2.36×   (−58 %)
2D, 1 thread     :  0.63 s     →    0.29 s       2.18×   (−54 %)
2D, 8 threads    :  0.18 s     →    0.057 s      3.07×   (−67 %)

Accuracy vs reference (unchanged — the rewrite is bitwise identical):
  3D  rel_diff = 0.0470   pearson = 0.9690   (gate ≤ 0.15)
  2D  rel_diff = 0.0201   pearson = 0.9975
```

The rewrite produces a **byte-for-byte identical** density on both fixtures:
`max_abs_diff`, `mean_abs_diff`, `rel_diff`, and `pearson` are unchanged from the
pre-rewrite build. It was also verified bitwise identical across 1056 randomized
small/edge-case inputs (see "Correctness" below).

Two eras of work are recorded here:

1. **Threading + early-exit era** (earlier). Threading was the big lever; the
   algorithmic changes capped work so the threaded path had something to scale.
2. **Scalar-kernel rewrite** (2026-07-12). Roughly halved single-threaded *and*
   multi-threaded runtime by cutting per-step instruction count and removing the
   per-iteration synchronization, on the existing channel-first layout with no
   SIMD, no relayout, and no precision change.

## Changes that landed

In rough order of when they landed:

1. **Threading via `detail::parallel_for_chunks`** — particle tracing is
   embarrassingly parallel; the final density scatter stays single-threaded
   (`<1 %` of time per the profile). Determinism preserved.
2. **Convergence-based early exit** (`tol` parameter) — freeze a particle once
   `max_axis(|dt·step|) < tol`.
3. **Mask-restricted tracing** (`restrict_to_mask` parameter) — a particle whose
   proposed endpoint rounds to a background voxel freezes in place. Combined with
   (2) it sheds slow trajectories that would otherwise run the full `n_iter`.
4. **RK2 (midpoint) integrator** (`method="rk2"`) — two flow samples per step but
   allows larger `dt`; combined with early-exit it converges in fewer iterations.
5. **Profiling instrumentation** (`BIOIMAGE_PROFILE_SCOPE`) around `init`,
   `iter_loop`, `scatter`, `mask_zero`. Free when the CMake flag is off.

The 2026-07-12 scalar-kernel rewrite then replaced the generic `2^D` corner
table and the iteration-major loop (with its per-iteration alive scan) with:

6. **Direct bilinear / trilinear interpolation** (`detail::sample_flow`) —
   replaces the generic `2^D` corner-offset/weight table with explicit nested
   lerps. The lower/upper index and fractional weight are computed once per axis
   and shared across channels; no `2^D` product weights or offsets are
   materialized. `if constexpr` selects the 2D vs 3D path.
7. **Truncation instead of `std::floor`** for the (clipped, nonnegative) sampling
   and rounding coordinates. See "Why truncation is worth ~10-35 %" below.
8. **Particle-major traversal** (`detail::trace_particle` / `trace_all`) — the
   integration loop moved *inside* one `parallel_for_chunks` fan-out. Each worker
   traces its contiguous range of particles across all `n_iter` steps. Removes
   the global `alive` vector, the up-to-`n_iter` thread create/join cycles, and
   the per-iteration sequential alive scan. Trajectories are independent until the
   sequential scatter, so results stay deterministic.
9. **Compile-time specialization** of the trace loop on the three loop-invariant
   flags `(use_rk2, check_convergence, restrict_to_mask)` via an 8-way switch
   dispatching to `trace_particle<D, RK2, Conv, Restrict>`. Hoists the per-step
   branches on these flags out of the innermost loop. When `restrict_to_mask` is
   set, the specialization also drops the start-of-step clip, which is provably
   redundant there (committed endpoints already passed the in-bounds mask test,
   and the seed is an in-bounds integer voxel).

10. **Runtime-dispatched FMA specialization** for the common default mode
    `(RK2, convergence, restrict_to_mask)`. On supported x86 CPUs the extension
    dispatches once, before thread fan-out, to a separately compiled tracer that
    uses scalar FMA contraction in the nested lerps. All other modes and CPUs use
    the portable kernel. Only `src/cpp/flow/flow_density_fma.cxx` receives the
    ISA flags (`-mavx -mfma` for GCC/Clang, `/arch:AVX2` for MSVC); the extension
    and binding layer remain baseline-safe. The build can disable this path with
    `-C cmake.define.BIOIMAGE_FLOW_FMA_DISPATCH=OFF`.

Chosen defaults (see `src/bioimage_cpp/flow/_flow.py`) are unchanged:

```python
n_iter=50, dt=0.2, tol=0.005, method="rk2", restrict_to_mask=True,
number_of_threads=1
```

## Attribution of the 2026-07-12 rewrite

Standalone C++ kernel harness, `-O3 -DNDEBUG` gcc-14, min-of-N, Tiger Lake.
Every row was verified bitwise identical to the pre-rewrite kernel on both
fixtures. `PM` = particle-major, `DIRECT` = direct interpolation, `TRUNC` =
truncation (the three toggles were measured in every combination):

```
PM DIRECT TRUNC | 3D 1T   3D 8T  | 2D 1T   2D 8T
 0    0     0   | 5.45    1.34   | 0.598   0.118    (pre-rewrite baseline)
 0    0     1   | 4.89    1.23   | 0.391   0.103
 0    1     0   | 4.35    1.01   | 0.501   0.125
 0    1     1   | 3.79    0.90   | 0.374   0.088
 1    0     0   | 4.48    1.05   | 0.469   0.085
 1    0     1   | 3.95    0.87   | 0.319   0.059
 1    1     0   | 3.18    0.68   | 0.424   0.072
 1    1     1   | 2.51    0.55   | 0.303   0.054    (all three)
```

Single-thread marginal contributions from the pre-rewrite baseline:

```
                       3D        2D
direct interpolation   ~20 %     ~16 %
particle-major         ~18 %     ~22 %
truncation             ~10 %     ~35 %
all three combined     ~54 %     ~49 %
```

- All three compose well and are independent wins.
- **In 2D, truncation is the single biggest lever** (~35 %): fewer corners (4 vs
  8) and channels (2 vs 3) make the floor operations a larger share of the work.
- Particle-major helps 8T more than 1T (barrier + alive-scan removal), so it
  improves both raw throughput and scaling.

On top of the three changes, **compile-time specialization (item 9) adds a
further ~6-9 % on 3D** (3D 1T 2.51 → ~2.29 s; total ~59 %). Isolating it: ~5.6 %
is from hoisting the three per-step flag branches, ~1 % from dropping the
redundant start-of-step clip. This is the same start-clip removal Sol tried as a
*runtime* `if (!restrict_to_mask)` and found to regress (~3.87 → 4.97 s): the
regression was the added invariant branch, not the removed clip. Done at compile
time there is no added branch, so it is a small positive instead.

The rounding used by the per-step mask test (`round_to_flat_index`) also had a
`std::floor(x + 0.5f)`; it is now a truncating cast (0.5-2.6 %, bitwise safe by
the same nonnegativity argument).

## Why truncation is worth ~10-35 %

The gain is real *because of*, not despite, the portable wheel build. On the
baseline x86-64 / SSE2 target the wheels compile for (the project forbids
`-march=native`), gcc-14 lowers `std::floor` to an **inlined ~15-instruction
software routine with a branch** — `roundss` requires SSE4.1, which the baseline
target does not assume. A truncating `static_cast<std::ptrdiff_t>` is a single
`cvttss2si`. Every sampling coordinate is clipped to `[0, shape-1]` before use,
so it is nonnegative and truncation equals `std::floor` exactly.

Consequences:

- The win **holds on the shipped wheels** (same SSE2 target). It would only
  shrink if the build enabled SSE4.1.
- Keep truncation local to helpers whose contract requires clipped, nonnegative
  coordinates (`sample_flow`, `round_to_flat_index`). Do not push it into a
  general sampler that might later be called with negative positions.

## Particle-major load balance (adversarial check)

Particle-major assigns each thread a static contiguous range for the whole trace.
The concern is that spatially-clustered slow trajectories could leave threads
idle. Measured on a worst-case synthetic fixture — `(48,512,512)`, all
foreground, zero flow for `z<24` (converges at iteration 0) and a small constant
in-mask drift for `z>=24` (runs all 50 steps). In C-order every slow particle is
in the upper half of the index range, so with 8 static chunks the lower 4 threads
are idle:

```
                       1T        8T       8T scaling
baseline (iter-major)  44.4 s    12.2 s   3.65×
opt (particle-major)   22.2 s    6.0 s    3.70×
```

Particle-major scales **as well** as iteration-major even in this worst case, and
is ~2× faster overall. This matches the theory: iteration-major pays a
create/join barrier on every one of the 50 steps, so it is gated by
`Σ_iter max_thread(work)`, whereas particle-major has one barrier and is gated by
`max_thread(Σ_iter work)`, and `max-of-sums ≤ sum-of-maxes`. The ~3.7× ceiling is
imposed by the shared static partition (4 idle threads), identical for both, not
by particle-major. **A dynamic scheduler is not warranted** on this evidence.

Realistic-fixture scaling (optimized kernel, 3D, `check_flow_density.py` median):

```
threads   runtime   speedup vs 1T
   1      2.46 s    1.00×
   2      1.33 s    1.85×
   4      0.73 s    3.37×
   6      0.66 s    3.73×
   8      0.54 s    4.56×
```

Better than the pre-rewrite kernel's 3.57× at 8T; barrier and alive-scan removal
let it scale further before hitting the 4-physical-core / SMT ceiling.

## Runtime FMA dispatch (2026-07-12)

The direct bilinear/trilinear sampler is a chain of scalar lerps. On baseline
x86-64 those lerps compile as separate multiply/add operations; an FMA-enabled
build contracts them and provides a further portable-at-runtime speedup without
changing the flow layout or API.

Paired `check_flow_density.py` medians on the AMD EPYC 7513, from the same source
revision and build environment:

| Fixture | Threads | Dispatch OFF | Dispatch ON | Improvement |
|---|---:|---:|---:|---:|
| 3D | 1 | 1.4900 s | 1.3084 s | 12.2% |
| 3D | 8 | 0.3035 s | 0.2553 s | 15.9% |
| 2D | 1 | 0.2021 s | 0.1801 s | 10.9% |

The full registered 3D density remained byte-for-byte identical, both stored
reference checks passed with unchanged metrics, and the complete test suite
reported 1057 passed / 8 skipped. A further 400 randomized 2D/3D default-mode
cases, including axis-of-length-one shapes and 1/4 threads, produced identical
aggregate output digests with dispatch enabled and disabled.

Implementation details that matter:

- Runtime feature detection happens in the baseline-compiled caller before the
  specialized function is entered.
- GCC/Clang require AVX+FMA; MSVC uses `/arch:AVX2` and therefore also checks AVX2
  before dispatch.
- Only the common selector 7 specialization is duplicated. Less common Euler,
  no-convergence, and unrestricted modes keep using the portable implementation.
- The FMA translation unit uses a distinct `CodegenVariant` template argument.
  Without a distinct mangled name, linker COMDAT selection can silently retain
  the portable `trace_all` instantiation and discard the FMA implementation.
- Automatic `target_clones` remains rejected: putting a target boundary around
  the driver or chunk inhibited the hot-loop inlining and regressed runtime.

## Correctness

- The current suite reports 1057 passed / 8 skipped. The 17
  `tests/test_flow.py` cases cover
  2D/3D, Euler/RK2, mask on/off, convergence on/off, single-vs-multithread
  equality, non-contiguous inputs, degenerate `n_iter`, and invalid inputs.
- A differential harness ran 1056 randomized cases — axis-of-length-1 shapes,
  tiny grids, zero/integer-aligned flows, Euler and RK2, `tol=0` and `>0`,
  `restrict_to_mask` on/off, 1 vs 4 threads — all bitwise identical to the
  pre-rewrite kernel.

Bitwise identity is robust here because the density scatter rounds each endpoint
to an integer voxel, which absorbs the sub-half-pixel differences from the
changed floating-point association order of the nested lerps. It is not a theorem
for every possible flow field, which is why the differential edge-case coverage
above matters.

## Profile breakdown (1-thread, 3D, profile build)

```
init        ~0.4 %    density-zero + positions collection
iter_loop  ~99   %    the particle tracing loop
scatter     ~0.1 %    final density write
mask_zero   ~0.1 %    final mask zeroing
```

The iteration loop still dominates after the rewrite, so init/scatter/mask-zero
remain not worth touching (per-thread scatter buffers and single-pass init were
scoped and skipped for this reason).

## Bandwidth diagnosis (streaming probe, earlier era)

Sustained streaming read bandwidth on this host is **≈ 11.5 GB/s** (measured via
`numpy.sum` on float32 arrays larger than L3). Against the upper-bound flow bytes
read per 3D RK2 call (~7.0 GB), the pre-rewrite kernel used ~11 % of sustained
bandwidth at 1T and ~40 % at 8T — i.e. the hot loop was **compute-bound, not
bandwidth-bound**, dominated by per-particle serial instruction count and gather
latency rather than RAM throughput.

The 2026-07-12 rewrite is consistent with and sharpens that diagnosis: it won by
**removing instructions and synchronization**, not by touching the memory layout.
The earlier conclusion was too narrow only if read as "only prefetching can
help" — cutting the coordinate/branch/weight/alive-state work around the loads
was the larger lever.

## Rejected / did-not-help experiments

Kept so these avenues are not blindly re-attempted.

- **`target_clones` autovectorization SIMD** (earlier, rolled back) — the AVX2
  clone emitted only scalar FMA (gcc judged the 8-corner gather unprofitable) and
  regressed 8T from icache pressure of three inlined clones.
- **Current-sample channel prefetching** after interpolation offsets were known
  regressed the EPYC 3D fixture by about 5% at 1T and 2% at 8T. The warm-kernel
  counter run reported only about 1.45% L1-data load misses, so the extra
  prefetch instructions cost more than the avoided demand-load latency.
- **Interleaved (channel-last) layout + hand-written AVX2 FMA** (earlier, rolled
  back, 2026-06-12) — codegen was confirmed optimal via `objdump` (8× packed
  `vfmadd132ps`, zero gathers) yet was no faster than scalar, and `VS=4` padding
  cost +33 % flow memory and worse line density than channel-first. Confirms the
  loop is load-latency / instruction-count bound, not ALU bound. The scalar
  rewrite targets exactly that (fewer instructions), which is why it succeeded
  where SIMD did not.
- **Half-precision (fp16) flow storage** — would halve traffic, but the kernel is
  not bandwidth-bound, and it is a breaking API change.
- **Per-thread density scatter buffers** / **single-pass init** — scatter and
  init are each `<0.5 %` of runtime.
- **Active-list compaction** (superseded) — was projected `<1 %` against the old
  alive-byte scan; particle-major removes the alive state entirely, so this is
  moot.
- **Runtime `if (!restrict_to_mask)` start-clip removal** (Sol, reverted) —
  regressed ~3.87 → 4.97 s from the added invariant branch. Now achieved for free
  via compile-time specialization (item 9).
- **Explicit `dt·step` displacement reuse** (Sol) — neutral (~0.02 s); the
  compiler already handles it.
- **16-particle iteration-major sub-blocking inside a persistent worker** (Sol) —
  regressed ~34 %. Keeping one trajectory in registers beats interleaving 16 to
  expose memory-level parallelism. This is also why a per-particle prefetch of the
  *next* particle is unlikely to pay: the dependent chain is within one trajectory.

## What remains worth trying

Lower priority than the landed work; all need a fixed-clock host with `perf`
(the Tiger Lake laptop's ~±8 % thermal-throttle noise swamps sub-10 % effects).

1. **Software prefetching of the next sample's cache lines.** Targets the residual
   load latency on the existing channel-first layout. Expected modest; the failed
   16-block experiment shows added machinery to expose independent loads can lose
   more than it gains, so measure carefully.
2. **A real `perf stat`** around a warm-kernel harness (fixture loaded once,
   several kernel calls) to confirm where the remaining cycles go (front-end vs
   back-end memory vs back-end core) and to provide a low-noise environment for
   evaluating (1).
3. **Cross-architecture confirmation** on macOS x86-64 and Windows x86-64 for
   the FMA dispatch, plus the truncation gain magnitude on arm64 (macOS
   AppleClang and Linux aarch64). The particle-major and direct-interpolation
   gains are portable C++20; the x86 dispatch is compiler-specific and the
   truncation magnitude depends on how `floor` is lowered without SSE4.1 / on
   the arm equivalent.

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
    python development/flow/check_flow_density.py --dim 3 --repeats 5 --threads $nt
done

# Accuracy gate at the chosen defaults
python development/flow/check_flow_density.py --dim both --repeats 3
```

`check_flow_density.py` accepts `--method`, `--dt`, `--tol`, `--n-iter`,
`--restrict-to-mask` / `--no-restrict-to-mask`, and `--threads` to override
defaults. The PASS gate is `rel_diff = mean(|ours-ref|)/mean(ref) ≤ 0.15`
(`--rel-tol` to override).
