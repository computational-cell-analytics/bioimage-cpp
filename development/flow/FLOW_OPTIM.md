# Flow-density optimization investigation

Date: 2026-07-12

## Executive summary

This document records a new performance investigation of
`bioimage_cpp.flow.compute_flow_density`, building on the experiments already
documented in `development/flow/PERFORMANCE_NOTES.md`.

The investigation found three complementary changes that are substantially
more promising than the SIMD, channel-layout, scatter, and active-list ideas
previously explored:

1. Trace a particle through all of its integration steps inside one parallel
   worker invocation, instead of launching and joining workers once per global
   integration step.
2. Replace the generic `2^D` corner-table construction with explicit bilinear
   and trilinear interpolation.
3. Convert already-clipped, nonnegative sampling coordinates to integers by
   truncation instead of calling `std::floor`.

In a temporary implementation, the combined change reduced the registered 3D
fixture runtime from 7.96 s to 3.87 s with one thread and from 2.05 s to 0.90 s
with eight threads on an AMD EPYC 7513. This is a reduction of approximately
51% and 56%, respectively. The registered 2D fixture improved from 1.09 s to
0.54 s with one thread, a reduction of approximately 51%.

The best temporary implementation passed all 17 tests in `tests/test_flow.py`,
retained the same stored-reference accuracy metrics, and produced a density
array that was bitwise identical to the current implementation on the full 3D
fixture. No algorithm changes from this investigation have been applied to the
repository; the experiments were made in a copy under `/tmp`.

## Scope and constraints

The goals of this investigation were:

- Take all existing flow benchmarks and optimization notes into account.
- Identify remaining performance opportunities in the current implementation.
- Validate ideas by measurement rather than relying on instruction-count or
  memory-bandwidth intuition alone.
- Keep experiments outside the repository until the evidence justified a
  concrete recommendation.
- Preserve the existing dependency-free, portable C++20 implementation and
  deterministic Python API.

The investigation did not change public parameters, defaults, numerical data
types, the channel-first flow layout, the integration methods, or the density
scatter semantics.

## Current implementation

The current implementation is in
`include/bioimage_cpp/flow/flow_density.hxx`. Its main phases are:

1. Zero the output and collect the coordinates of foreground voxels into an
   array of particle positions.
2. For every global integration iteration:
   - call `detail::parallel_for_chunks` over all particle positions;
   - skip particles whose `alive` byte is zero;
   - clip the current position;
   - build a shared table of corner offsets and weights;
   - sample all flow channels from that table;
   - optionally take a second sample for RK2;
   - test convergence and the foreground mask;
   - update the position or mark the particle dead;
   - join all worker threads;
   - scan the complete `alive` array to determine whether tracing can stop.
3. Scatter the final particle positions into the density array.
4. Zero density outside the foreground mask.

The existing profiler shows that approximately 99% of the runtime is in the
integration loop. Initialization, scatter, and mask-zeroing are each well below
1%, so this investigation continued to focus exclusively on particle tracing.

### Generic interpolation work

For every flow sample, `compute_corners<D>` currently:

- calls `std::floor` once per spatial axis;
- computes a fractional coordinate once per axis;
- visits all `2^D` corners;
- for every corner, revisits all `D` axes;
- selects the lower or upper coordinate;
- multiplies the corresponding weight;
- clamps the selected coordinate;
- multiplies it by the grid stride and accumulates the flat offset.

The resulting offsets and weights are shared across flow channels, which was a
useful improvement over recomputing them independently for every channel.
Nevertheless, the generic construction performs much more coordinate,
branching, and weight-product work than fixed 2D and 3D interpolation require.

### Parallel-loop structure

`detail::parallel_for_chunks` creates `n_threads - 1` `std::thread` objects and
joins them before returning. The current integration loop calls it once per
global integration step. With the default `n_iter=50`, this can create and join
the worker set up to 50 times per function call.

The iteration-major order also requires a global byte per particle to retain
the alive state, scans dead particles in later iterations, and performs a
sequential complete-array alive reduction after every step when convergence or
mask restriction is enabled.

Every trajectory is independent until the final sequential density scatter.
There is therefore no algorithmic requirement for particles to advance in
lockstep.

## Prior optimization work considered

The existing `development/flow/PERFORMANCE_NOTES.md` was treated as the starting
point, not as a list of experiments to repeat. The following successful changes
were retained in all experiments:

- Multithreaded particle tracing through `detail::parallel_for_chunks`.
- Per-particle convergence detection via `tol`.
- Optional termination when a proposed endpoint leaves the foreground mask.
- RK2 integration, which enables useful accuracy/runtime defaults.
- Sharing a corner table across the flow channels.
- Hoisting channel pointers, strides, and grid bounds.
- Existing phase profiling through `BIOIMAGE_PROFILE`.

The following previously rejected approaches were not reintroduced:

- Parallel or per-thread density scatter buffers, because scatter is below 1%
  of runtime.
- Fusing the initialization passes, because initialization is approximately
  0.4% of runtime.
- A globally compacted active-particle list.
- Compiler-generated SIMD through target clones.
- Handwritten AVX2 for the existing corner sum.
- Channel-last/interleaved flow storage, including padded SIMD-friendly
  storage.
- Half-precision flow storage.
- A standalone structure-of-arrays position layout whose main purpose would be
  SIMD.

The prior AVX2 and interleaved-layout experiment is particularly important. It
produced the intended packed FMA instructions but did not improve runtime. This
showed that reducing only interpolation arithmetic, while retaining the same
corner-address and load behavior, was insufficient on the earlier Intel test
machine.

## Experimental environment

All new source experiments were performed in a copied repository at:

```text
/tmp/bic-flowopt.5WuTmL/repo
```

An isolated virtual environment and persistent build directory were also kept
under that temporary directory. The production repository remained unchanged
during the experiments.

Hardware and runtime environment:

- CPU: AMD EPYC 7513 32-Core Processor, Zen 3 generation.
- Host topology: two sockets, 32 physical cores per socket, two hardware
  threads per core.
- CPUs available to the process: `32-35,96-99`, corresponding to eight logical
  CPUs.
- Compiler optimization: the project's normal non-Debug `-O3` build.
- Python: 3.14.
- Performance counters: Linux `perf` was available.
- Registered fixture shape: `(48, 512, 512)` for 3D, with 729,236 foreground
  particles.
- Registered fixture shape: `(520, 704)` for 2D, with 151,019 foreground
  particles.
- Flow defaults: `n_iter=50`, `dt=0.2`, `tol=0.005`, `method="rk2"`,
  `restrict_to_mask=True`, and no density smoothing.

This is a different machine from the Intel Tiger Lake laptop used for the
existing performance notes. Absolute timings should therefore not be compared
between the two documents. Relative timings within each experiment are the
relevant measurements.

The baseline was built from an untouched temporary copy and reproduced the
timings of the already installed current implementation, which reduced the
risk that build-directory or compiler differences biased the comparison.

## Baseline measurements

Registered fixture timings for the current implementation were:

| Fixture | Threads | Median | Minimum | Repeats |
|---|---:|---:|---:|---:|
| 3D | 1 | 7.9621 s | 7.9464 s | 3 |
| 3D | 8 | 2.0529 s | 2.0472 s | 3 |
| 2D | 1 | 1.0876 s | 1.0845 s | 5 |

The initial hardware-counter run on the current implementation reported high
instruction throughput and a low branch-miss rate:

| Counter | Current implementation |
|---|---:|
| Cycles | 16.394 billion |
| Instructions | 51.949 billion |
| Instructions per cycle | 3.17 |
| Branches | 8.945 billion |
| Branch misses | 44.45 million, 0.50% |
| Cache references | 405.3 million |
| Cache misses | 63.35 million |

These were whole-process `perf stat -r 3` measurements of the validation script,
not counters around only the C++ kernel. They include data loading and accuracy
calculation, and the events were multiplexed at approximately 83%. They are
still useful for comparing large changes made under the same conditions.

The low branch-miss rate argues against branch prediction hints as a useful
primary optimization. The high total instruction count suggested that removing
work could be valuable even though the existing roofline analysis correctly
showed that the kernel does not saturate streaming memory bandwidth.

## Successful experiment 1: particle-major traversal

### Idea

Move the integration loop inside the per-particle loop:

```text
parallel over contiguous particle chunks once:
    for each particle in the chunk:
        for up to n_iter steps:
            sample, test, and update this particle
            break when it converges or leaves the mask
```

This is valid because particles do not interact during tracing. Only the final
density scatter combines their results, and that scatter remains sequential and
deterministic.

### Work removed

- Up to 49 of the 50 worker-thread creation/join cycles.
- The global `alive` vector.
- Alive-byte loads and dead-particle branches in later global iterations.
- The complete sequential alive reduction after each global iteration.
- Repeated loads and stores of a particle position between global iterations;
  the compiler has a better opportunity to keep one trajectory in registers.

### Measured result

| Fixture | Threads | Current | Particle-major | Improvement |
|---|---:|---:|---:|---:|
| 3D | 1 | 7.9621 s | 6.8670 s | 13.8% |
| 3D | 8 | 2.0529 s | 1.8783 s | 8.5% |

The one-thread improvement proves that repeated worker creation is not the only
mechanism. Removing alive-state traffic and keeping a trajectory locally are
also important.

### Risks

The current iteration-major traversal gives each thread the same static range
of particles on every iteration. Particle-major traversal also assigns static
contiguous ranges, but each thread now owns the entire remaining lifetime of
its range. If slow-converging trajectories are spatially clustered, threads may
finish at different times and reduce parallel utilization.

The registered 3D fixture still improved at eight threads, but an implementation
should add a deliberately imbalanced benchmark or test before assuming this is
universal. A dynamic scheduler should not be introduced casually: it would add
new infrastructure, scheduling overhead, and portability complexity. First
measure whether real workloads exhibit enough imbalance to need it.

## Successful experiment 2: direct bilinear/trilinear interpolation

### Idea

Replace the generic corner table with a small sampling description containing
the lower coordinate, upper coordinate, and fractional coordinate for each
axis. Sample each channel through nested interpolation.

Conceptually, 3D sampling becomes:

1. Interpolate along x for each of the four `(z, y)` pairs.
2. Interpolate the resulting values along y for each z plane.
3. Interpolate the two plane values along z.

The 2D case performs two x interpolations followed by one y interpolation.

At an upper grid boundary, the lower and upper coordinate are identical, which
preserves the current nearest-boundary behavior.

### Work removed

- The generic loop over every corner and every axis.
- Repeated lower/upper coordinate selection for every corner.
- Repeated per-corner bounds checks.
- Construction and storage of all product weights.
- Construction and storage of a full corner table when only fixed 2D or 3D
  interpolation is supported.

The flow loads themselves remain channel-first and use the existing data. This
is fundamentally different from the rejected SIMD experiment: the new approach
reduces address-generation, coordinate, branch, and weight-construction work,
not just the arithmetic instructions used to sum already-constructed corners.

### Measured result

| Fixture | Threads | Current | Direct interpolation | Improvement |
|---|---:|---:|---:|---:|
| 3D | 1 | 7.9621 s | 6.7392 s | 15.4% |
| 3D | 8 | 2.0529 s | 1.5889 s | 22.6% |
| 2D | 1 | 1.0876 s | 0.8853 s | 18.6% |

Direct interpolation and particle-major traversal composed well:

| Fixture | Threads | Current | Combined | Improvement |
|---|---:|---:|---:|---:|
| 3D | 1 | 7.9621 s | 5.0713 s | 36.3% |
| 3D | 8 | 2.0529 s | 1.2550 s | 38.9% |
| 2D | 1 | 1.0876 s | 0.7447 s | 31.5% |

### Numerical considerations

Nested interpolation changes the association order of floating-point
operations relative to multiplying eight precomputed weights and summing them.
The mathematical interpolation is the same, but intermediate float32 rounding
is not guaranteed to be bitwise identical for every possible flow field.

On the registered fixtures, all reported accuracy metrics were unchanged. The
final best 3D density was also bitwise identical to the current density.
Nevertheless, an implementation should include targeted differential tests of
the sampler itself and end-to-end randomized tests, rather than relying on the
integer-valued density scatter to hide small trajectory differences.

## Successful experiment 3: truncate clipped coordinates

### Idea

Every position passed to interpolation is clipped to the closed grid domain:

- the current particle position is clipped before its first sample;
- the RK2 midpoint is clipped before its second sample.

Sampling coordinates are therefore finite and nonnegative. For a nonnegative
float, conversion to an integer by truncation has the same result as
`floor(position)`. Replacing:

```cpp
static_cast<std::ptrdiff_t>(std::floor(position[axis]))
```

with:

```cpp
static_cast<std::ptrdiff_t>(position[axis])
```

is exact under this precondition.

This optimization should remain local to a helper whose contract requires
clipped coordinates. It should not silently change a general-purpose sampling
helper that might later be called with negative positions.

### Measured result

Adding truncation to the interpolation-only implementation reduced the 3D
one-thread runtime from 6.7392 s to 5.5406 s, an additional 17.8% improvement.

Adding it to the combined particle-major and direct-interpolation implementation
produced the best result:

| Fixture | Threads | Current | Best temporary variant | Improvement |
|---|---:|---:|---:|---:|
| 3D | 1 | 7.9621 s | 3.8657 s | 51.5% |
| 3D | 8 | 2.0529 s | 0.9046 s | 55.9% |
| 2D | 1 | 1.0876 s | 0.5351 s | 50.8% |

The size of this effect was unexpected. It may depend strongly on compiler and
microarchitecture, so the exact gain must be checked on Intel x86-64, arm64,
and the wheel build toolchains. The semantic equivalence does not depend on the
CPU as long as the clipped-coordinate precondition holds.

## Counter evidence for the combined change

The best temporary implementation was measured with the same whole-process
`perf stat -r 3` command as the current implementation:

| Counter | Current | Best temporary variant | Change |
|---|---:|---:|---:|
| Cycles | 16.394 B | 10.269 B | -37.4% |
| Instructions | 51.949 B | 25.391 B | -51.1% |
| Instructions per cycle | 3.17 | 2.48 | lower, but much less total work |
| Branches | 8.945 B | 2.248 B | -74.9% |
| Branch misses | 44.45 M | 43.79 M | approximately unchanged absolute count |
| Cache references | 405.3 M | 275.5 M | -32.0% |
| Cache misses | 63.35 M | 27.64 M | -56.4% |

Because these counters include Python data loading and result checking, they
understate the proportional reduction inside the C++ kernel. They nevertheless
confirm the main mechanism: the improvement comes from executing much less
work. It is not an artifact of a different flow layout, reduced precision,
non-deterministic scatter, or architecture-specific packed SIMD.

The reduced IPC does not indicate a regression. The current implementation
has abundant independent generic corner arithmetic that can retire at high
throughput. The optimized version removes much of that arithmetic, leaving a
larger fraction of the remaining cycles exposed to the unavoidable dependent
flow sampling. Total cycles and wall time still fall substantially.

## Correctness validation

Every major successful variant was checked with:

```bash
python -m pytest tests/test_flow.py -q
```

All 17 tests passed. These tests cover:

- 2D and 3D tracing.
- Euler and RK2 paths.
- Mask restriction enabled and disabled.
- Convergence enabled and disabled.
- Single-threaded and multithreaded equality.
- Non-contiguous Python inputs.
- Degenerate iteration counts and zero flows.
- Invalid inputs.
- Density smoothing integration.

The registered 2D and 3D validation scripts continued to pass their stored
reference gates. For the final 3D variant:

- relative difference against the stored reference remained 0.0470;
- Pearson correlation remained 0.9690;
- the density sum remained exactly 729,236 particles;
- the full density array was bitwise equal to the current implementation;
- zero of 12,582,912 voxels differed.

Bitwise equality on this fixture is strong evidence, but it is not a proof that
nested interpolation will produce identical trajectories for every input.

## New experiments that did not help

### Conditional removal of start-of-step clipping

With `restrict_to_mask=True`, every committed endpoint is in bounds, so clipping
the current position at the start of the next iteration is logically redundant.
The experiment wrapped clipping in a runtime `if (!restrict_to_mask)` condition.

This regressed the best 3D one-thread result from approximately 3.87 s to
4.97 s. The likely cause is worse compiler optimization or hot-loop layout from
the additional invariant runtime branch. The experiment was reverted.

If this is revisited, it should only be through compile-time specialization of
the mask-restricted and unrestricted kernels, followed by careful code-size and
cross-platform measurements. The current evidence does not justify that added
complexity.

### Explicit displacement reuse

The current code calculates `dt * step[axis]` once for the convergence maximum
and again when forming the proposed endpoint. A temporary implementation stored
the displacement in an array and reused it.

The result was neutral: 3.8871 s versus 3.8657 s for the best implementation,
within normal run variation. The compiler already appears capable of handling
this arithmetic efficiently, or the extra temporary storage cancels the saved
multiplications.

### Small-block iteration-major traversal

Particle-major traversal introduces a dependent chain: the next flow address
for one particle is unknown until the current sample updates its position. A
possible latency-hiding strategy is to process a small block of neighboring
particles in iteration-major order inside each persistent worker.

A block size of 16 was tested. It retained one parallel invocation but used a
small local alive table and interleaved 16 trajectories. It regressed the best
implementation:

| Threads | Particle-major best | Block size 16 | Regression |
|---:|---:|---:|---:|
| 1 | 3.8657 s | 5.1927 s | 34.3% |
| 8 | 0.9046 s | 1.2043 s | 33.1% |

The benefit of keeping one trajectory in registers outweighed any additional
memory-level parallelism or local spatial reuse. Other block sizes were not
tested because the result was not marginal and the simpler particle-major
design already performed well.

## Revised performance diagnosis

The existing notes concluded that the hot loop was not limited by streaming
memory bandwidth and that flow-load latency was likely important. That remains
compatible with the new evidence, but the earlier conclusion was too narrow if
interpreted as meaning that only prefetching could help.

The current generic interpolation loop executes a very large amount of
coordinate, bounds, offset, weight, alive-state, and loop-control work around
the actual loads. It can sustain high IPC, but it also retires approximately
twice as many instructions as the best temporary implementation in the
whole-process comparison.

The rejected SIMD experiment reduced arithmetic for an already-constructed
corner set without removing most address-generation and control work. The new
direct interpolation changes the algorithmic formulation of that work. The
particle-major traversal independently removes synchronization and global
state traffic. The truncation change removes an expensive coordinate operation
under a valid local precondition.

After these changes, load latency may become a larger fraction of the remaining
runtime. Software prefetching is therefore still a possible later experiment,
but it is lower priority than implementing and validating the demonstrated
instruction-count reductions. The failed 16-particle interleaving experiment
also shows that adding machinery solely to expose more independent loads can
easily lose more than it gains.

## Recommended implementation plan

### 1. Introduce a narrow direct sampling helper

Add a small internal sampling description or explicit 2D/3D helpers in
`bioimage_cpp::flow::detail`. Keep the API focused on clipped coordinates and
C-contiguous grid strides. Avoid introducing a generic interpolation framework
or new dependency.

Recommended properties:

- Separate, readable 2D and 3D code paths selected with `if constexpr` or two
  small overloads.
- Compute lower, upper, and fractional coordinates once per sample position.
- Use truncation only after documenting or asserting the nonnegative clipped
  precondition.
- Load each corner value into a named local before interpolation, both for
  readability and to make single-load intent obvious.
- Preserve the current duplicated-coordinate behavior at upper boundaries.
- Reuse the same sampling description across all flow channels.

The temporary proof of concept prioritized experimental speed and should be
cleaned up before landing, particularly by shortening long interpolation
expressions and documenting boundary behavior.

### 2. Change to particle-major tracing

Move the per-particle iteration loop inside one call to
`detail::parallel_for_chunks`. Preserve contiguous static chunks and the
existing final sequential scatter. This remains compatible with the project's
single threading primitive and deterministic behavior.

The implementation can remove:

- the global `alive` vector;
- the outer global iteration loop;
- the sequential complete alive scan.

For each particle, convergence or mask exit becomes a simple `break` from its
local integration loop.

### 3. Keep existing profiling scopes and add temporary subphase evidence

The existing `iter_loop` profile scope should remain. During implementation,
temporary profiling or benchmark variants can distinguish:

- sampling-coordinate preparation;
- first flow sample;
- RK2 midpoint and second sample;
- convergence/mask/update logic.

Use the existing profiling macros rather than ad hoc timers. Fine-grained
scopes inside every particle step may themselves be intrusive, so use them only
in a dedicated profiling build or compare separately compiled variants.

### 4. Add differential tests

In addition to the existing suite, add tests for:

- direct sampling at integer coordinates;
- fractional coordinates in 2D and 3D;
- the last coordinate on every axis;
- shapes containing an axis of length one;
- random positions near voxel-rounding boundaries;
- random finite flow fields comparing old and new implementations during
  development;
- both Euler and RK2;
- `restrict_to_mask` enabled and disabled;
- `tol=0` and positive convergence tolerance;
- masks designed so different contiguous particle ranges require very
  different iteration counts;
- equality across thread counts.

The old sampler can be kept only in a development comparison or temporary test
helper while validating the change; it should not remain as duplicate
production infrastructure after confidence is established.

### 5. Benchmark multiple architectures

At minimum, repeat the registered benchmark matrix on:

- an Intel x86-64 machine representative of the earlier Tiger Lake results;
- the AMD Zen 3 host used here;
- macOS arm64 with AppleClang;
- one Linux arm64 wheel environment when available;
- Windows x86-64 with MSVC.

The particle-major and direct-interpolation changes are portable C++20 and are
expected to generalize. The magnitude of the truncation gain is the most likely
to vary with compiler and CPU.

Recommended measurements:

```bash
python development/flow/check_flow_density.py --dim both --repeats 5 --threads 1
python development/flow/check_flow_density.py --dim 3 --repeats 5 --threads 2
python development/flow/check_flow_density.py --dim 3 --repeats 5 --threads 4
python development/flow/check_flow_density.py --dim 3 --repeats 5 --threads 8
python -m pytest tests/test_flow.py -q
```

Where available, collect `perf stat` counters around a script that loads the
fixture once and executes several warm kernel calls. This would isolate the C++
kernel better than the whole-process counters used in this investigation.

## Priority ranking

1. **Implement direct 2D/3D interpolation plus clipped-coordinate truncation.**
   This removes the largest clearly identified body of unnecessary work and
   has no threading/load-balance risk.
2. **Implement particle-major traversal.** It provides a separate substantial
   gain and composes exceptionally well with direct interpolation. Validate
   load balance on heterogeneous trajectories.
3. **Run cross-architecture benchmarks and hardware counters.** The combined
   gain is large enough that it should survive ordinary noise, but compiler and
   architecture coverage are essential for a wheel-oriented project.
4. **Only then reconsider prefetching or compile-time mode specialization.**
   These are more complex and currently lack positive experimental evidence.

## Conclusion

The strongest remaining opportunity is not more SIMD or a new memory layout.
It is simplifying the scalar algorithm and matching the loop structure to the
independence of particle trajectories.

The combined temporary implementation approximately halved 2D and 3D runtime,
improved both single-threaded and eight-threaded execution, retained all
existing test behavior, and produced a bitwise-identical full 3D density result
for the registered fixture. The evidence is strong enough to justify an
implementation pass, provided it includes targeted numerical tests,
load-balance checks, and validation on the project's major wheel platforms.
