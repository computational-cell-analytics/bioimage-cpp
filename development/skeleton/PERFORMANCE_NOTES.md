# Binary TEASAR performance notes

Baseline measurements and initial optimization recommendations for the first
correctness-oriented binary, single-component 3D TEASAR implementation in
`include/bioimage_cpp/skeleton/teasar.hxx`. The comparison backend is kimimaro
5.8.1. No TEASAR-specific performance optimization had been applied when these
numbers were collected.

## Measurement setup

Measured on 2026-07-14:

- Intel Core i7-1185G7, 4 cores / 8 hardware threads, up to 4.8 GHz.
- Optimized editable bioimage-cpp build (`-O3`).
- Kimimaro 5.8.1. Importing this installation required adding
  `crackle-codec 0.42.0`; kimimaro imports `crackle` although that package is
  declared only by an optional installation extra.
- Single-threaded comparison: bioimage-cpp's initial implementation is
  single-threaded and kimimaro used `parallel=1`.
- One warmup followed by three measured calls. Backend order was shuffled
  deterministically for each repetition; the median is the headline value.
- Deterministic branching tubes span each volume. Radius increases with volume
  size; foreground occupies about 0.8–1.0% of the full volume.
- Spacing `(1.5, 1.0, 1.0)`, `scale=1.5`, `constant=1.0`,
  `pdrf_scale=100000`, and `pdrf_exponent=4` for both implementations.
- Kimimaro soma handling was disabled with infinite thresholds. Hole filling,
  border fixing, and progress reporting were disabled; `fix_branching=True`.

Reproduce from the repository root:

```bash
python development/skeleton/benchmark_teasar.py \
    --large --kimimaro --repeats 3 --warmup 1 \
    --json /tmp/teasar_large.json
```

The benchmark tiers are:

| tier | volumes |
| --- | --- |
| small | `64³` |
| default | `96³`, `128³`, `192³` |
| large | `128³`, `192³`, `256³` |

## Wall-clock results

Median of three measured calls:

| volume | full voxels | foreground | bioimage-cpp | kimimaro | bio / kimimaro | vertices (bio / kimi) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `128³` | 2.10M | 20,988 | 213.43 ms | 94.71 ms | 2.25× | 213 / 213 |
| `192³` | 7.08M | 59,657 | 1.092 s | 381.13 ms | 2.87× | 340 / 323 |
| `256³` | 16.78M | 132,619 | 1.917 s | 998.21 ms | 1.92× | 415 / 443 |

Raw timings in milliseconds:

| volume | bioimage-cpp | kimimaro |
| --- | --- | --- |
| `128³` | 213.43, 214.00, 205.25 | 109.05, 93.94, 94.71 |
| `192³` | 1092.50, 1112.77, 1074.82 | 381.13, 387.05, 364.05 |
| `256³` | 1917.01, 1912.05, 1948.18 | 934.06, 998.97, 998.21 |

Normalized medians:

| volume | bio µs/foreground | kimi µs/foreground | bio ns/full-volume voxel | kimi ns/full-volume voxel |
| --- | ---: | ---: | ---: | ---: |
| `128³` | 10.17 | 4.51 | 101.8 | 45.2 |
| `192³` | 18.31 | 6.39 | 154.4 | 53.9 |
| `256³` | 14.46 | 7.53 | 114.3 | 59.5 |

## Comparison limitations

This is an end-to-end performance comparison, not a strict parity test:

- Bioimage-cpp uses a padded EDT, deterministic two-sweep root, repeated
  exact grid Dijkstra rail paths, and an axis-aligned physical invalidation
  cube.
- Kimimaro uses its public multi-label/component pipeline and a different
  inside-component invalidation implementation plus additional tracing
  machinery.
- Equal vertex counts do not prove identical geometry or topology. Different
  counts do not by themselves imply that either result is more accurate.

Both backends returned trees (`E = V - 1`) in all measured cases. Graph sizes
match at `128³`. Bioimage-cpp returns 5.3% more vertices at `192³` and 6.3%
fewer at `256³`, consistent with the known invalidation/heuristic differences.

## Findings

1. **The initial implementation is within roughly 2–3× of kimimaro.** The gap
   is 2.25× at `128³`, 2.87× at `192³`, and 1.92× at `256³`. This is a useful
   baseline for a correctness-first implementation, but there is enough gap to
   justify optimizing before adding multi-component dispatch.

2. **Runtime is strongly tied to the full bounding volume, not only foreground
   count.** Foreground occupies less than 1% of these volumes, while most core
   arrays and several scans cover every voxel. From `128³` to `256³`, full
   volume grows 8×, foreground 6.32×, bioimage-cpp time 8.98×, and kimimaro time
   10.54×.

3. **The 3D Dijkstra primitive is already a known hot candidate.** Its separate
   dense `128×256×256` benchmark takes about 10.1 s per physical field and 8.9 s
   per node-cost field. TEASAR's foreground is sparse and its paths stop early,
   so those standalone values do not predict TEASAR time directly, but repeated
   fresh Dijkstra state remains an obvious target.

4. **Large dense buffers remain alive together.** At padded `256³`, the current
   implementation retains the padded mask, float DBF, first-root field,
   distance-from-root field, double PDRF, active mask, and int64
   voxel-to-vertex map. These persistent arrays alone are roughly 620 MiB.
   A path solve adds dense distances, predecessors, settled state, heap locator,
   and target marks, bringing the fixed live state to roughly 1 GiB before EDT
   scratch and dynamic heap entries.

5. **The first root field is retained after its last use.** It is needed only
   to verify connectivity and select the final root, but remains allocated for
   the rest of skeletonization. At the largest case this unnecessarily retains
   about 131 MiB.

6. **Every rail path currently pays public-solver setup costs.** Repeated path
   calls allocate/fill dense Dijkstra state, rebuild neighbor metadata, create
   a full target bitmap, and rescan the entire PDRF to validate finite
   non-negative costs. TEASAR owns and maintains these arrays, so most of this
   work is redundant internally.

7. **Target selection scans the entire padded volume once per path.** The loop
   searches all voxels for the farthest active DAF value even though DAF is
   immutable and only active/inactive status changes.

8. **Invalidation may revisit overlapping cubes.** Each path vertex walks its
   full physical cube and checks `active` voxel-by-voxel. Correctness is simple,
   but long thick paths can cause substantial overlap and repeated visits.

9. **The existing phase profiler is already wired into TEASAR.** Current labels
   are `distance_transform`, `root_dijkstra`, `pdrf`, `path_dijkstra`, and
   `invalidation`. The next optimization pass should collect these phase totals
   on the `256³` case before changing code.

## Initial optimization recommendations

Recommendations are prioritized by expected impact, memory reduction, and
risk. Speedups remain hypotheses until phase profiling and repeated benchmark
confirmation.

### 1. Profile the `256³` case by phase

Build with profiling enabled and use one measured call to avoid accumulated
reports becoming noisy:

```bash
pip install -e . --no-build-isolation \
    -C cmake.define.BIOIMAGE_PROFILE=ON
python development/skeleton/benchmark_teasar.py \
    --large --repeats 1 --warmup 0
```

Record both phase totals and peak RSS. Restore a normal build before publishing
headline timings. Add finer Dijkstra scopes for validation, initialization,
heap pop, and relaxation as described in
`development/distance/DIJKSTRA_PERFORMANCE_NOTES.md`.

### 2. Crop to the foreground bounding box before dense processing

Compute a tight bounding box for the single component, add the required zero
halo, and carry the origin offset into returned physical coordinates. The
synthetic tubes span most axes but still leave sizeable margins; real binary
objects embedded in larger volumes may benefit much more.

Cropping reduces EDT work, every dense field, every target scan, and Dijkstra
workspace size simultaneously. Preserve lexicographic coordinate order so
root/path tie-breaking remains deterministic.

### 3. Reuse a trusted Dijkstra workspace across all TEASAR solves

Create one workspace for root and path solves and reuse distances,
predecessors, state, heap capacity, target marks, strides, and neighbor tables.
Use generation counters or touched-index lists so early-stopping path solves
reset only visited voxels.

Add an internal already-validated cost-field path so TEASAR does not rescan the
entire PDRF before every rail. The public Dijkstra functions must retain their
current validation contract.

### 4. Release or reuse full-volume fields at their last use

- Destroy/reuse `first_field.distances` immediately after the final root is
  selected.
- Reuse compatible buffers between first-root Dijkstra, final-root DAF, and
  path workspaces where lifetimes do not overlap.
- Evaluate `float32` for PDRF and selected internal fields only with explicit
  path-parity tests; kimimaro uses lower-precision fields in parts of its
  pipeline, but bioimage-cpp must not silently lose deterministic correctness.

Releasing the first field is a low-risk memory win even if it has little
wall-clock effect.

### 5. Replace repeated full-volume target scans

DAF never changes. Build candidates once in descending `(DAF, flat_index)`
order, then advance lazily past inactive voxels. A heap is also possible, but a
one-time sorted foreground list may have better locality and deterministic tie
behavior. Compare construction cost and memory on both sparse and dense masks.

### 6. Optimize invalidation after measuring its phase

Potential approaches include:

- merging overlapping axis-aligned spans per `z/y` row before clearing;
- tracking active foreground coordinates so background is never revisited;
- a component-aware distance-limited invalidation traversal;
- changing to spherical/inside-component invalidation for closer kimimaro
  behavior, treated as an algorithmic change with new correctness fixtures.

Do not combine a semantic invalidation change with a low-level optimization;
otherwise performance and graph differences cannot be attributed cleanly.

### 7. Add controlled threading only after workspace optimization

The EDT already accepts a thread count and can parallelize safely. Exposing a
TEASAR thread setting for EDT and embarrassingly parallel setup phases is lower
risk than parallel Dijkstra. A single-component rail loop is serial by design;
future multi-component dispatch can parallelize components independently.

## Correctness and acceptance gates

For low-level optimizations, require unchanged deterministic vertices, edges,
and radii on the existing straight, Y-branch, thick-tube, anisotropic, diagonal,
and non-contiguous fixtures. Cropping and workspace reuse should be bitwise
identical.

Algorithmic changes such as spherical/component-aware invalidation may
intentionally change the graph. Land those separately with explicit expected
topology/radius tests and rerun the kimimaro comparison without asserting
vertex-for-vertex parity.

Current full-suite baseline: `1126 passed`. Use the `256³` three-repeat median
and peak RSS as the performance acceptance benchmark for the next optimization
step.
