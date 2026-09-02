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
AMD EPYC 7513 (Zen 3); see `development/flow/FLOW_OPTIM.md`. Absolute timings
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

**2026-09-02 update (Zen 3, see "Packed-channel FMA tracer" below):** the
default path on AVX+FMA x86-64 CPUs now uses a packed-channel vectorized
tracer. On the same machine and flags, the registered fixtures run
**1.052 s -> 0.637 s (3D, 1T)** and **0.110 s -> 0.069 s (2D, 1T)**,
-39 % and -37 %, with identical densities; 2T and 4T gain 30-42 %, adversarial
inputs 17-44 %.

Three eras of work are recorded here:

1. **Threading + early-exit era** (earlier). Threading was the big lever; the
   algorithmic changes capped work so the threaded path had something to scale.
2. **Scalar-kernel rewrite** (2026-07-12). Roughly halved single-threaded *and*
   multi-threaded runtime by cutting per-step instruction count and removing the
   per-iteration synchronization, on the existing channel-first layout with no
   SIMD, no relayout, and no precision change.
3. **Packed-channel FMA tracer** (2026-09-02). Another -37..-44 % by repacking
   the flow field channel-last once and tracing with 4-lane packed arithmetic
   inside the runtime-dispatched FMA translation unit, after a per-step counter
   budget showed the scalar kernel retired ~330 instructions per step and spent
   a third of its cycles with the FP register file full.

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

11. **3-way lockstep trajectory interleave for RK2**
    (`detail::trace_particle_block`, 2026-07-12). Each worker traces three
    particles per group in lockstep instead of one trajectory to completion.
    The trajectories are independent, so the out-of-order core fills one
    lane's serial sample→lerp→update latency chain with the other lanes'
    work; a converged/frozen lane costs one predictable branch per remaining
    group iteration. Euler keeps the plain per-particle loop — its shorter
    single-sample chain measured ~10 % *slower* interleaved. Results are
    bitwise identical (per-trajectory arithmetic is untouched). See
    "Trajectory interleave (2026-07-12)" below for measurements and K choice.

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

## Trajectory interleave (2026-07-12)

The rejected midpoint-reuse experiment showed the per-step loads are
latency-hidden *within* one trajectory — so the remaining 1T lever is ILP
*across* trajectories. `trace_particle_block<D, K, ...>` traces K consecutive
particles in lockstep (local position/alive lane state, per-step body identical
to `trace_particle`); the RK2 selectors use K=3, with a plain `trace_particle`
remainder loop.

Measured with tightly paired `.so`-swap ABA runs (K1 = per-particle loop,
a local predecessor of today's `development/flow/paired_bench.py`, min-of-3 per
invocation, Tiger Lake). Two flag regimes,
because the conda toolchain exports default `CXXFLAGS`
(`-march=nocona -mtune=haswell -ftree-vectorize ...`) while wheel builds get
plain flags — see the pitfall below:

| Workload (RK2 defaults) | conda flags | plain flags |
|---|---:|---:|
| registered 3D, 1T | **−8.6 %** (1.93 → 1.76 s) | **−13 %** |
| registered 2D, 1T | **−27 %** (0.33 → 0.24 s) | **−28 %** |
| registered 3D, 8T | −3.5 % | ~parity |
| euler 3D, 1T (gated to K=1) | parity | — |
| stripes 1/4 long-lived, 1T (lane-divergence worst case) | +1.2 % | parity |
| stripes 3/4 long-lived, 1T | −1.6 % | — |
| random flow s=1 / s=10, 1T | −7 % / −10 % | — |

- **K choice**: K=3 and K=4 both beat K=1 by 6–16 % on the 3D fixture in the
  triage; K=2 was inside noise. Head-to-head, K=3 beat K=4 by ~4 % and carries
  less lane state, so K=3 landed. GCC 14 does *not* unroll the K-lane loop —
  the ILP comes from the out-of-order window overlapping loop iterations, and
  that was measurably sufficient; manual unrolling was not needed.
- **2D vs 3D**: the 2D win (~27 %) dwarfs 3D — 2D has less per-lane state
  (position + 4-corner sampling), so the interleave adds ILP without register
  pressure.
- **Lane divergence** (the lockstep concern: a group lives as long as its
  slowest lane) is benign: the stripe worst case — one never-converging
  orbiter per group of 4, neighbours converging on step 1 — costs ~1 %,
  because dead lanes skip their step body and only pay an alive-bit branch.
- **Adversarial cases** live in `development/flow/benchmark_interleave.py`.
- Verified bitwise identical: 320-case randomized digest harness, registered
  2D/3D fixtures at 1/4/8 threads, full suite (1070 tests).

### Build-flag pitfall (affects all local benchmarking)

The micromamba GCC 14 environment exports `CXXFLAGS`. CMake appends it to
every TU, so *normal* local builds are `-march=nocona -mtune=haswell
-ftree-vectorize ...` on top of the project's `-O3`. Setting `CXXFLAGS=<x>`
for an experiment build **replaces** those defaults rather than adding to
them, silently changing codegen of every TU — an early triage here compared
mixed flag regimes before this was caught. Wheel builds (cibuildwheel, no
conda) get the plain flags, so plain-flag numbers are the shipping-relevant
ones. When A/B-benchmarking local builds: `echo $CXXFLAGS` first, and keep
the regime identical on both sides.

## Packed-channel FMA tracer (2026-09-02)

Third optimization round, on an **AMD EPYC 7513 (Zen 3)** node under a SLURM
allocation of 2 physical cores / 4 logical CPUs (CPUs 36,37 + SMT siblings
100,101; NUMA node 1), governor `performance`, THP `always`, `nmi_watchdog=1`
(five programmable counters per `perf` group). GCC 14.3 (conda-forge), Python
3.14.5, NumPy 2.4.6, **plain flags** (`CXXFLAGS= CFLAGS=`; the conda toolchain's
`-march=nocona ... -O2` defaults were cleared), commit `492ff9b` as baseline.
All A/B numbers come from `development/flow/paired_bench.py` (fresh pinned
subprocess per measurement, A B B A per repeat, prebuilt `.so` files swapped
through `sys.modules`, never rebuilt between sides) and from the kernel-only
counter harness `development/flow/perf_kernel.py`. This machine cannot measure
beyond two physical cores; 8T-class numbers remain to be taken elsewhere.

### What changed

The default path (RK2, `tol > 0`, `restrict_to_mask`) on CPUs with AVX+FMA now
runs a **packed-channel kernel** in `src/cpp/flow/flow_density_fma.cxx`:

- The flow field is repacked once (parallel over planes/rows) into channel-last
  storage, D floats per voxel, on a grid padded by one replicated voxel per
  axis (+2.5 % voxels on the 3D fixture). Every trilinear corner is then a
  single 16-byte load carrying all channels, and the padding makes the upper
  corner index always valid, so the scalar kernel's nearest-boundary rule
  needs no clamp. Cost: a transient buffer the size of the flow input
  (151 MB on the 3D fixture) and a pack pass of ~16 ms at 1T / ~10 ms at 2T.
- The particle position is one 4-lane register `[z, y, x, 0]`. Truncation,
  fraction, midpoint, clip, convergence test (one compare + `movemask`),
  bounds test and the rounding for the mask lookup are packed; the flat
  offsets are formed in general-purpose registers from three extracted lanes.
- 3D interpolation keeps the nested-lerp association order of the scalar
  kernel (fused `lo + f*(hi-lo)`, fused midpoint, unfused position update),
  so the traced positions match the scalar-FMA kernel bit for bit on the
  fixtures. 2D uses a four-weight sum (`mul -> fma -> add`), which shortens
  the post-load chain from 16 to 11 cycles; its association order differs,
  and densities were still bitwise identical on both fixtures and all 400
  randomized cases.
- Lockstep interleave of **K = 4** trajectories per group (K = 3 for the
  scalar kernel), lanes addressed with compile-time indices so the state
  stays in registers.
- The scalar-FMA instantiation of the header kernel remains as fallback when
  the packed buffer cannot be allocated or its offsets exceed int32, and the
  portable SSE2 kernel is untouched for selectors 0-6 and non-AVX CPUs.
- New runtime opt-out `BIOIMAGE_CPP_FLOW_FORCE_SCALAR=1` (mirrors the filters
  module) and `_core._flow_trace_backend()` returning `"fma"` or `"scalar"`,
  with a parity test in `tests/test_flow.py`.
- The binding's finiteness scan over the flow buffer (previously a serial
  `std::isfinite` loop holding the GIL, ~30 ms of the 3D fixture) now runs
  inside the GIL release through `detail/finite.hxx::all_finite` (branch-free
  per-block exponent test, parallel with the caller's thread count).
- Hot helpers (header `sample_flow`, `round_to_flat_index`,
  `position_is_in_mask`, `trace_particle`, `trace_particle_block`; the TU's
  samplers and step) carry `BIOIMAGE_FORCE_INLINE`
  (`include/bioimage_cpp/detail/force_inline.hxx`). See the codegen trap below.

### Why: per-step diagnosis of the previous kernel

The 2026-07-13 gate had closed prefetching on the grounds that misses were rare;
it had not converted the counters into a per-step budget. A profile-build step
counter (`TraceStats`, printed as `[bioimage flow trace]`) now gives:

```
3D fixture: 33,959,237 executed steps for 729,236 particles (46.6 per particle)
            exits: converged 4.4 %, left mask 5.6 %, hit n_iter 90.0 %
2D fixture:  7,230,607 executed steps for 151,019 particles (47.9 per particle)
            exits: converged 6.1 %, left mask 0.3 %, hit n_iter 93.6 %
```

Baseline kernel, 3D fixture, 1T, per executed step (perf_kernel, 8 warm calls):

```
cycles 114   instructions 326   loads 95 (49 essential)   IPC 2.87
L1d misses 1.4/step (1.5 %)   L2 misses 0.03/step   DRAM fills 0.007/step
dTLB misses ~0.0007/step (THP effective)
FP-register-file dispatch stalls: 40 cycles/step (35 % of cycles)
```

So the kernel was neither memory- nor branch-bound; it retired ~330
instructions per step, half of them address/weight bookkeeping in the FP
register domain, and spent a third of its cycles with the FP register file full.
Static disassembly of the FMA TU confirmed ~330-360 instructions and 42-47
stack operands per lane-step. That is what the packed kernel attacks: fewer
instructions, fewer FP-domain micro-ops, shorter chain.

### Results

Paired A/B (`paired_bench.py`, ABBA, best-of-subprocess minima, `.so` swap),
previous kernel (`492ff9b`) vs packed kernel, plain flags, Zen 3:

```
case        threads  cpus            previous    packed    change   noise  parity
fixture3d   1        36              1.0515 s    0.6372 s  -39.4 %  0.6 %  identical
fixture2d   1        36              0.1099 s    0.0689 s  -37.3 %  1.8 %  identical
fixture3d   2 phys   36,37           0.5520 s    0.3388 s  -38.6 %  0.4 %  identical
fixture2d   2 phys   36,37           0.0561 s    0.0360 s  -35.9 %  1.3 %  identical
fixture3d   2 SMT    36,100          0.8489 s    0.4914 s  -42.1 %  0.5 %  identical
fixture3d   4        36,37,100,101   0.4576 s    0.2675 s  -41.6 %  1.9 %  identical
fixture2d   4        36,37,100,101   0.0441 s    0.0309 s  -29.9 %  1.0 %  identical
stripes 1/4 1        36              0.5817 s    0.4821 s  -17.1 %  1.3 %  identical
stripes 3/4 1        36              1.2842 s    0.7707 s  -40.0 %  2.8 %  identical
random s=1  1        36              1.5414 s    0.8931 s  -42.1 %  3.0 %  identical
random s=10 1        36              1.6568 s    0.9259 s  -44.1 %  0.9 %  identical
random s=5  1        36              1.1499 s    0.6816 s  -40.7 %  0.3 %  identical
random s=20 1        36              1.5028 s    0.8957 s  -40.4 %  2.7 %  identical
random s=40 1        36              0.8152 s    0.5425 s  -33.5 %  3.2 %  identical
2D random 10 1       36              0.1727 s    0.1103 s  -36.1 %  0.4 %  differs (2D tree)
```

The base-vs-base calibration of the same harness reads `noise` on both fixtures
(|dMin| <= 0.12 %). The stripes 1/4 case (three of four lanes converge on the
first step) gains least, as expected for a lockstep group; K = 4 still beat
K = 3 there by 3.6 %.

Accuracy gate (`check_flow_density.py --dim both --repeats 3`): unchanged,
3D `rel_diff = 0.0470`, `pearson = 0.9690`; 2D `rel_diff = 0.0201`,
`pearson = 0.9975`; particle sums 729,236 / 151,019; densities bitwise equal to
the previous kernel on both fixtures. `differential_check.py` (400 randomized
cases: length-1 axes, tiny grids, Euler/RK2, `tol`/`dt`/`n_iter` variants, mask
on/off, 1/4 threads) is 400/400 bitwise identical previous-vs-packed, and
399/400 packed-vs-forced-scalar (one scale-50 random flow: 0.63 % of voxels
differ by <= 2 counts, sums equal; contraction noise on a chaotic field). Full
suite: 1457 passed, 7 skipped.

Per executed step, 3D fixture, 1T (2D in parentheses):

```
                          previous            packed
cycles/step               114    (56)         67     (35)
instructions/step         326    (145)        133    (81)
loads/step                95     (39)         38     (16)
IPC                       2.87   (2.60)       1.98   (2.33)
L1d misses/step           1.4                 0.34
FP-RF dispatch stall      40 cyc (35 %)       36 cyc (54 %)   (2D: 20 cyc, 57 %)
flops/step (FMA = 2)      n/a                 196    (84)
```

Static K = 4 lane body (final build): 3D 122 instructions per lane-step
(110 GPR, 107 FP ALU, 74 loads incl. 8 constant operands, 59 FMA, 24 shuffles,
24 vector-int, 20 cvt per group of four; 2 stack operands per lane-step);
2D 78 per lane-step. The remaining cost is the dependent chain per step
(~90 cycles in 3D: two samples of cvt -> extract -> imul/add -> load -> three
lerp levels, plus midpoint and update) against the reorder window and the FP
register file; four lanes recover ~1.35 lane-steps of overlap.

Thread balance (profile build, `[bioimage flow trace]` per-thread lines):
2T physical on the 3D fixture 0.9 % time / 1.1 % steps imbalance; stripes 1/4
0.1 %. Static contiguous chunks remain adequate. Phase split at 1T on the 3D
fixture: pack 16.6 ms (2.6 %), tracing 96.5 %, init 1.7 %, scatter 0.8 %,
mask zero 1.0 %.

Non-kernel: the parallel, GIL-free finiteness check cuts the `n_iter=0` call
from ~40 ms to a few ms on the 3D fixture (was 3.8 % at 1T, 7 % at 2T of the
previous kernel's runtime, growing with thread count).

### Codegen trap: translation-unit growth disables inlining

While the FMA TU held many template instantiations (K and layout variants for
the sweep), GCC 14 stopped inlining `step_packed`, `sample_packed` and even the
header's `sample_flow<3>`: the scalar-FMA fallback slowed from 1.05 s to 1.59 s
and the packed kernel from 0.65 s to 1.24 s, with out-of-line calls in the hot
loop. An out-of-line `sample_flow<3>` compiled with AVX is also exactly the
COMDAT/ODR hazard the dispatch design guards against. `BIOIMAGE_FORCE_INLINE`
on the per-step helpers removes the dependence on the unit-growth budget in both
translation units.

### Rejected in this round

- **Vector-integer address math** (`pmulld` + horizontal sums for the flat
  offset, `pminsd`-clamped upper indices): 0.720 s vs 0.645 s for the
  integer-register version at K=3 in 3D, and a 2D regression (0.113 s vs the
  scalar 0.110 s). Vector-integer ops occupy the FP register file, which was
  the bottleneck.
- **16-byte voxels (zero fourth lane, aligned loads, no lane clean-up)**:
  −2 % in 3D for +33 % transient memory; split-line loads (3.3 per step, ~20 %
  of corner loads) are not a measurable cost. Not worth the memory.
- **Weight-tree interpolation in 3D**: within noise (−1 %); the 3D loop is
  bound by the reorder window (~150 micro-ops per lane-step against a
  ~90-cycle chain), so the extra weight micro-ops offset the shorter chain.
  Kept for 2D only, where it measured −12 %.
- **K = 6 lanes**: parity with K = 4 in both dimensions; K = 2 and K = 1 lose
  (3D: 0.652 s / 1.19 s vs 0.645 s).
- **Padding away the boundary clamps**: only ~1 % (integer clamps were not on
  the critical path), kept because it simplifies the sampler.

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
- **Next-sample software prefetching** (closed after the `perf` gate,
  2026-07-13; not implemented). A production, plain-flag build of `e426920` was
  measured on an AMD EPYC 7513 (Zen 3) with the registered 3D fixture, pinned to
  one core. Each `perf stat -r 3` process loaded the fixture once, made one warm
  kernel call, then made eight more calls; process setup and loading were
  included but amortized by the repeated kernel work. The performance governor
  was active (boost remained enabled), so cycles and counter repeatability were
  used as the gate rather than wall time alone.
  - Measurement noise was low enough for the decision: cycles varied by
    0.03--0.12% and wall time by 0.3--0.7% across the three runs.
  - The warm process sustained 2.77 instructions/cycle. L1 data-load misses
    were about 1.54% (roughly 500 million misses from 32.3 billion loads).
  - Of the observed demand data reads reaching L2, about 453.5 million hit and
    19.5 million missed (approximately a 4.1% L2 miss rate). An L2 fill was
    pending for roughly 7% of counted cycles; this is an upper bound on cycles
    prefetching might overlap, not a direct stall measurement.
  These counters do not show enough residual cache-miss cost to justify the
  extra instructions and code complexity. The existing 3-way RK2 interleave
  already overlaps independent load chains, current-sample prefetching regressed
  on this CPU, and a next-iteration address is only known late in the dependent
  trajectory update. The prefetch target is therefore closed unless a future
  workload or architecture supplies contrary profile evidence.
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
- **RK2 midpoint same-cell corner reuse** (2026-07-12, reverted). After the first
  sample of a step, keep the 2^D·D corner values and skip the midpoint's offset
  computation and reloads when `trunc(mid) == trunc(position)` (bitwise-exact by
  construction: equal lower coords imply equal clamped offsets and corner
  values). The mechanism worked as designed — measured with temporary counters,
  the reuse hit rate is **96.7 %** on the registered 3D fixture (default
  `dt=0.2`; the midpoint moves `0.1·|flow|` voxels) — and a 320-case randomized
  digest harness plus the full suite confirmed bitwise-identical output. It was
  still rejected on tightly A/B-paired benchmarks (alternating the two built
  `.so` files per run; stash-and-rebuild rounds proved useless against this
  laptop's 15–20 % thermal drift):
  - Registered fixtures: 3D 1T ~2–3 % faster, 2D 1T ~5 %, 3D 8T ~2–5 % — the
    midpoint loads are L1 hits whose latency the out-of-order core already hides
    behind the trajectory's dependent chain, so removing them buys little.
  - Adversarial random flows (`development/flow/benchmark_midpoint_reuse.py`,
    scale sweep): at scale 10 (~30 % hit rate, maximally unpredictable branch)
    the kernel regressed **13–15 %** at 1T, consistently across pairs; ~1–4 %
    regression even at scale 1 (92 % hit).
  - Two codegen traps worth remembering: (a) expressing `sample_flow` as
    load-cell + interpolate-cell made GCC 14 materialize the corner array on the
    stack in the single-sample Euler tracer (~20 % Euler regression; fixed by
    keeping `sample_flow` monolithic); (b) the fatter RK2 `trace_particle` made
    GCC stop inlining it, emitting it out-of-line as an *untagged* weak symbol
    with AVX code in the FMA TU — the COMDAT/ODR hazard the dispatch design had
    only avoided through full inlining. The `CodegenVariant` tag is therefore
    now propagated through `trace_particle` (landed; instruction streams verified
    identical to the pre-change build in both TUs).

## Optimization status and future validation

The 2026-07-13 Zen 3 counter run had closed the round at `e426920`. The
2026-09-02 round reopened it on new evidence (a per-step budget and FP
register-file stall counters, see "Packed-channel FMA tracer") and landed the
packed tracer. What remains is bounded by the per-step dependent chain against
the out-of-order window: further gains would need fewer FP-domain micro-ops per
step (e.g. cheaper lane extraction for the integer address path) or a shorter
chain, both of which are now measurable with `perf_kernel.py --steps`.
Reopen on a demonstrated regression, a new representative workload, or
counter evidence of a bottleneck not covered above.

Cross-architecture validation remains useful but is not an active optimization
target: confirm the FMA dispatch on macOS x86-64 and Windows x86-64, and confirm
the truncation gain magnitude on arm64 (macOS AppleClang and a Linux arm64 wheel
environment). The particle-major and direct-interpolation changes are portable
C++20; the x86 dispatch is compiler-specific and the truncation magnitude can
vary with how `floor` is lowered on each target.

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

Kernel-level tooling added in the 2026-09-02 round (all in `development/flow/`,
none required by the test suite):

```bash
# Kernel-only timing + hardware counters (perf stat attached around the timed
# calls only; five-event groups). --steps converts totals into per-step budgets
# using the executed-step count printed by a BIOIMAGE_PROFILE=ON build.
taskset -c 36 python development/flow/perf_kernel.py --case fixture3d --cpu 36 \
    --perf-group core,stall,fp --steps 33959237

# Paired A/B of two prebuilt _core .so files (fresh pinned subprocess per run,
# ABBA, noise estimate, density parity). Calibrate with --a X --b X first.
python development/flow/paired_bench.py --a base.so --b cand.so --cases all \
    --threads 1 --cpu 36 --repeats 4 --inner 3

# Randomized differential check (400 cases) between two builds or between the
# FMA path and BIOIMAGE_CPP_FLOW_FORCE_SCALAR=1.
python development/flow/differential_check.py --a base.so --b cand.so
python development/flow/differential_check.py --a cand.so --b cand.so \
    --env-b BIOIMAGE_CPP_FLOW_FORCE_SCALAR=1
```

A `BIOIMAGE_PROFILE=ON` build prints, per call, a `[bioimage flow trace]` block
(executed steps, exit reasons, steps-per-particle histogram, per-thread time and
imbalance) in addition to the phase profile.
