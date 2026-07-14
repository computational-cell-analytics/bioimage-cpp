# Grid Dijkstra performance notes

Baseline measurements and initial optimization recommendations for the
correctness-first masked-grid Dijkstra implementation in
`include/bioimage_cpp/distance/grid_dijkstra.hxx`. These notes cover the public
`dijkstra_distance_field` and `dijkstra_path` primitives independently of
TEASAR. No Dijkstra-specific optimizations had been applied when these numbers
were collected.

## Measurement setup

Measured on 2026-07-14:

- Intel Core i7-1185G7, 4 cores / 8 hardware threads, up to 4.8 GHz.
- Optimized editable build (`-O3` through the normal CMake configuration).
- Single-threaded. The initial Dijkstra implementation has no threaded path;
  the FMM comparison used `number_of_threads=1`.
- One warmup followed by three measured calls; the median is the headline
  value and all raw measurements are reported below.
- Full connectivity: 8 neighbors in 2D and 26 neighbors in 3D.
- The mask is one connected, nearly dense domain with two walls and alternating
  openings. The source and target lie near opposite corners, forcing a long
  detour. This is intentionally a difficult field/path workload.
- `physical-field` uses anisotropic spacing. `node-field` uses a smooth,
  positive spatial cost field. `early-path` targets the opposite corner.
- The existing first-order fast-marching geodesic field is included as a
  performance reference only. It solves a different continuous approximation
  and is not a Dijkstra correctness oracle.

Reproduce from the repository root:

```bash
python development/distance/benchmark_dijkstra.py \
    --large --repeats 3 --warmup 1 --include-geodesic \
    --json /tmp/dijkstra_large.json
```

The benchmark tiers are:

| tier | 2D | 3D |
| --- | --- | --- |
| small | `256²` | `32×96×96` |
| default | `1024²` | `64×160×160` |
| large | `2048²` | `128×256×256` |

## Wall-clock results

Median of three measured calls:

| operation | `2048²` (4.19M foreground) | ns/fg voxel | `128×256×256` (8.32M foreground) | ns/fg voxel |
| --- | ---: | ---: | ---: | ---: |
| physical field | 1.001 s | 238.9 | 10.114 s | 1215.1 |
| field + predecessors | 1.028 s | 245.4 | 10.002 s | 1201.6 |
| node-cost field | 1.020 s | 243.4 | 8.943 s | 1074.5 |
| early-stopping physical path | 0.999 s | 238.3 | 9.968 s | 1197.6 |
| geodesic FMM field | 1.295 s | 309.1 | 5.709 s | 685.9 |

Raw timings in seconds:

| operation | `2048²` | `128×256×256` |
| --- | --- | --- |
| physical field | 1.001, 1.001, 0.996 | 10.114, 10.132, 9.898 |
| field + predecessors | 1.032, 1.023, 1.028 | 10.002, 10.576, 9.962 |
| node-cost field | 1.020, 1.033, 1.004 | 8.883, 8.943, 9.041 |
| early path | 1.018, 0.999, 0.979 | 9.968, 9.746, 10.040 |
| geodesic FMM field | 1.231, 1.295, 1.373 | 5.619, 5.730, 5.709 |

## Findings

1. **The 3D neighbor loop is the main visible scaling problem.** The physical
   field takes 239 ns/foreground voxel in 2D and 1215 ns/foreground voxel in
   3D, a 5.1× increase per voxel. The neighbor count grows from 8 to 26, and
   the 3D case also puts substantially more pressure on the heap and memory
   hierarchy.

2. **Dijkstra beats FMM in this 2D case but loses clearly in 3D.** Physical
   Dijkstra is 1.29× faster than FMM at `2048²`; FMM is 1.77× faster at
   `128×256×256`. This does not make the algorithms interchangeable: only
   Dijkstra provides the exact discrete predecessor chain required by TEASAR.

3. **Writing predecessors is not the bottleneck.** The predecessor result adds
   about 2.7% in 2D and is within run-to-run noise in 3D. Avoiding predecessor
   writes alone will not materially improve the rail-path use case.

4. **Early stopping does not help when the target is far away.** The opposite
   target and wall layout require nearly the whole reachable domain to settle,
   so `dijkstra_path` costs essentially the same as a full field. Early
   stopping should still help for nearby targets and should not be removed.

5. **The node-cost 3D field is about 11.6% faster than the physical field.**
   This may reflect a different priority distribution and relaxation pattern;
   it is not yet phase-profiled and should not be treated as a general property
   of node costs.

6. **The current solver constructs and initializes dense state for every
   call.** For `N` voxels, a solve allocates/fills at least the `float64`
   distance field, one-byte settled field, and `size_t` dense heap locator.
   Paths additionally allocate an `int64` predecessor field and a one-byte
   target bitmap. On the measured 64-bit build, the indexed heap stores up to
   one 24-byte entry per active key. At 8.4M voxels this makes allocation,
   zero/fill bandwidth, and heap locality important even before neighbor
   arithmetic is considered.

7. **Weighted calls validate the full cost volume on every invocation.** This
   is correct for the public API but redundant for TEASAR, which owns a valid
   PDRF field and invokes repeated path solves on it.

8. **Neighbor metadata and boundary work are dynamic.** Every solve rebuilds
   the offset vectors and physical lengths. Every settled voxel calls
   `valid_offset_target` for each candidate neighbor. The optimized FMM solver
   previously gained substantially by decoding a voxel once and replacing
   repeated generic offset validation with direct coordinate/bounds tests; the
   Dijkstra loop has the same structural opportunity.

## Initial optimization recommendations

Recommendations are ordered by expected value and implementation risk. They
are hypotheses until confirmed by the repository profiler and repeated
benchmarks.

### 1. Profile initialization, heap, and relaxation separately

Add `BIOIMAGE_PROFILE_SCOPE` regions around:

- input/cost validation;
- dense state initialization;
- heap pop;
- coordinate decoding and neighbor relaxation;
- path reconstruction.

Run the large 3D physical and node cases once with `BIOIMAGE_PROFILE=ON`. This
will distinguish memory initialization, heap maintenance, and neighbor
arithmetic before changing data structures.

### 2. Specialize 2D and 3D neighbor traversal

Decode the popped flat index once, then use fixed-size 2D/3D offset tables and
direct bounds tests. Precompute signed linear deltas and physical edge lengths
once in a reusable solver/workspace. Avoid `vector<vector<ptrdiff_t>>` and
generic `valid_offset_target` calls in the inner loop.

This is the safest first kernel optimization and should especially benefit the
26-neighbor 3D case. Preserve lexicographic neighbor ordering and `(distance,
flat_index)` tie-breaking so paths remain deterministic.

### 3. Add a reusable Dijkstra workspace

Follow the existing workspace pattern used by the graph solvers. Reuse:

- distances and predecessors;
- settled/state storage;
- dense heap locator and heap capacity;
- target marks;
- strides and neighbor metadata.

For repeated solves, use touched-index lists or generation counters so reset
cost is proportional to visited voxels rather than the full bounding volume.
Keep the current allocation-owning public functions as convenient wrappers.

Provide a trusted internal entry point for callers such as TEASAR that have
already validated an unchanged mask and cost field. Public Python calls must
continue to perform full validation.

### 4. Benchmark heap alternatives rather than assuming one winner

The addressable dense heap avoids stale entries but requires an `N`-entry
locator and random locator updates on every swap. Compare it against:

- `std::priority_queue` with lazy stale-entry rejection;
- a Dijkstra-specific indexed heap with a smaller entry/priority layout;
- integer/radix or bucket queues only if realistic costs admit them.

The physical and floating node-cost APIs are general, so bucket/radix schemes
cannot replace the binary heap universally. Measure runtime, peak heap size,
and memory before choosing.

### 5. Reduce dense memory traffic

After profiling, consider:

- merging settled/heap state into a compact byte state where practical;
- avoiding a full target bitmap for small target sets via generation marks or
  an indexed target lookup;
- using touched lists to initialize only reached foreground;
- compact foreground indexing for very sparse masks.

Compact foreground indexing adds mapping overhead and is unlikely to help the
nearly dense standalone benchmark, but it may be decisive for sparse TEASAR
objects.

### 6. Defer parallel shortest-path algorithms

The current global priority order is inherently serial. Delta stepping or
other parallel variants would be a substantial algorithmic change with harder
determinism and weighted-cost semantics. First exhaust single-threaded
neighbor, workspace, heap, and memory improvements.

## Correctness and acceptance gates

Any optimization should preserve:

- exact fields against the independent Python heap oracle;
- deterministic paths and target tie-breaking;
- predecessor-chain costs;
- all connectivity, anisotropy, zero-cost, and weighted-mode tests;
- public validation and error behavior.

Current full-suite baseline: `1126 passed`. Re-run
`tests/distance/test_grid_dijkstra.py` after every kernel/data-structure change,
then the full suite before accepting benchmark gains.
