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

Pre-optimization full-suite baseline: `1126 passed`. Use the `256³` three-repeat median
and peak RSS as the performance acceptance benchmark for the next optimization
step.

## Implemented compact-indexing results

The sequential design matrix in `OPTIM_DIJKSTRA.md` was implemented and
measured on 2026-07-14. Reproduce the size sweep and extended regimes with:

```bash
python development/skeleton/benchmark_teasar.py \
    --large --sequential-backends --repeats 5
python development/skeleton/benchmark_teasar_sequential.py --repeats 5
```

Compact IDs follow ascending full C-order indices. Compact FP64 therefore
preserves dense FP64 heap tie-breaking and produced bitwise-identical vertices,
edges, and radii for every size, spacing, density, and PDRF regime tested.

Five-repeat end-to-end medians for the main anisotropic branching tubes:

| backend | `128^3` | `192^3` | `256^3` |
| --- | ---: | ---: | ---: |
| optimized dense FP64 | 134.66 ms | 618.82 ms | 1.169 s |
| compact on-the-fly FP64 | **74.04 ms** | **267.21 ms** | **686.58 ms** |
| compact CSR FP64 | 75.19 ms | 284.57 ms | 730.72 ms |
| compact CSR FP32 | 74.66 ms | 284.70 ms | 730.21 ms |

CSR was 1.6%, 6.5%, and 6.4% slower than on-the-fly FP64 on these tiers. It
was only 2.3% faster on the separate extremely sparse `192^3` case and was
40% slower on the relatively dense `96^3` ball. It therefore failed the rule
requiring a 5% win on both large tiers. The selected production backend is
compact on-the-fly FP64: a temporary/full `uint32` lookup plus
`compact_to_full`, with root, PDRF, heap state, predecessors, skeleton targets,
and voxel-to-vertex data indexed only over foreground.

On the `256^3` case, process peak RSS fell from 1,143,572 KiB in the original
implementation to 266,536 KiB, a 76.7% reduction. The optimized dense backend
measured 839,596 KiB in the same single-call process. Compact on-the-fly and
CSR had effectively identical peaks because both must first construct the
65.5 MiB full-to-compact lookup; CSR releases it only after its adjacency is
built.

Profile-build phase totals on `256^3` show where the improvement lands:

| phase | optimized dense FP64 | selected compact FP64 |
| --- | ---: | ---: |
| compact-domain build | -- | 41.6 ms |
| root Dijkstra (two fields) | 178.7 ms | 58.8 ms |
| PDRF construction | 53.4 ms | 3.7 ms |
| rail-path Dijkstra | 274.3 ms | 29.0 ms |
| total TEASAR | 995.0 ms | 639.9 ms |

The exact distance-to-boundary transform is now 77% of selected-backend time;
combined root/path Dijkstra is 5.2x faster than the optimized dense backend and
12.5x faster than the original recorded Dijkstra phases.

### FP32 decision

Internal FP32 was rejected. On `256^3`, CSR FP32 combined root/path Dijkstra
was 113.9 ms versus 108.9 ms for CSR FP64, end-to-end time was slightly slower,
and peak RSS was unchanged at about 266.5 MiB. It also failed the quality gate
on `192^3`: the contracted degree signature changed, bidirectional 95th
percentile distance was 2.0 voxels, Hausdorff distance was 7.07 voxels, and
total physical length differed by 2.26%. Automatic TEASAR execution therefore
remains FP64. Parallel shortest paths remain a separate follow-up.

Final verification after selecting compact on-the-fly FP64: `1129 passed`.

## Initial parallel follow-up (compact dispatch later removed)

The shared thread budget for the distance transform and compact Dijkstra solves
was implemented and measured on 2026-07-15. The reference matrix now passes
the same worker counts to kimimaro through `parallel=N`. Reproduce it with:

```bash
python development/skeleton/benchmark_teasar.py \
    --large --kimimaro --threads 1 2 4 8 \
    --repeats 3 --warmup 1 \
    --json /tmp/teasar_large_parallel.json
```

The environment was the same Intel Core i7-1185G7 host (4 cores / 8 hardware
threads), with kimimaro 5.8.1 and EDT 3.1.1. Each backend received one warmup;
backend order was shuffled deterministically for each of the three measured
calls. The table reports the median with the minimum in parentheses. Scaling
is relative to the corresponding single-worker median, and `bio / kimi` is the
matched median-time ratio (lower favors bioimage-cpp).

| volume | workers | bioimage-cpp median (min) | bio scaling | kimimaro median (min) | kimi scaling | bio / kimi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `128^3` | 1 | 77.60 ms (73.83 ms) | 1.00x | 110.43 ms (103.32 ms) | 1.00x | 0.70x |
| `128^3` | 2 | 60.42 ms (56.16 ms) | 1.28x | 143.28 ms (136.60 ms) | 0.77x | 0.42x |
| `128^3` | 4 | 50.01 ms (43.15 ms) | 1.55x | 152.51 ms (149.16 ms) | 0.72x | 0.33x |
| `128^3` | 8 | 55.71 ms (45.21 ms) | 1.39x | 172.32 ms (171.98 ms) | 0.64x | 0.32x |
| `192^3` | 1 | 303.83 ms (283.12 ms) | 1.00x | 381.02 ms (364.20 ms) | 1.00x | 0.80x |
| `192^3` | 2 | 235.93 ms (216.57 ms) | 1.29x | 393.61 ms (389.73 ms) | 0.97x | 0.60x |
| `192^3` | 4 | 209.54 ms (205.03 ms) | 1.45x | 413.88 ms (409.37 ms) | 0.92x | 0.51x |
| `192^3` | 8 | 195.08 ms (188.79 ms) | 1.56x | 431.37 ms (417.77 ms) | 0.88x | 0.45x |
| `256^3` | 1 | 694.69 ms (681.63 ms) | 1.00x | 931.81 ms (883.13 ms) | 1.00x | 0.75x |
| `256^3` | 2 | 510.91 ms (509.73 ms) | 1.36x | 950.05 ms (933.76 ms) | 0.98x | 0.54x |
| `256^3` | 4 | 444.05 ms (424.72 ms) | 1.56x | 902.87 ms (902.36 ms) | 1.03x | 0.49x |
| `256^3` | 8 | 434.67 ms (424.57 ms) | 1.60x | 938.22 ms (911.45 ms) | 0.99x | 0.46x |

Bioimage-cpp improved through four workers on every tier and was fastest at
eight workers on `192^3` and `256^3`. Its best measured scaling was 1.60x on
`256^3`; at four workers it was approximately 3.0x, 2.0x, and 2.0x faster than
kimimaro on the three volumes.

Kimimaro's `parallel` path is relevant but not fully exercised by this fixture.
It threads the full-volume EDT and then uses a process pool to dispatch
connected components. Each branching tube contains only one component, so the
pool has one useful tracing task and additional workers mostly add process and
shared-memory overhead. The only measured kimimaro improvement was about 3%
at four workers on `256^3`.

This section records the initial implementation. The threshold-focused review
follow-up below supersedes its compact-Dijkstra dispatch decision.

Worker count did not change either implementation's graph size:

| volume | bioimage-cpp vertices / edges | kimimaro vertices / edges |
| --- | ---: | ---: |
| `128^3` | 213 / 212 | 213 / 212 |
| `192^3` | 340 / 339 | 323 / 322 |
| `256^3` | 415 / 414 | 443 / 442 |

Verification after the benchmark update: `24 passed` in
`tests/skeleton/test_teasar.py`; the matched-matrix and sequential-backend
smoke runs also completed successfully.

## Implemented non-Dijkstra optimization follow-up

Profiling after the compact and parallel-Dijkstra work showed that further
Dijkstra changes were no longer the best target. On the `256^3` branching tube
at four workers, the two root fields plus all rail paths took about 85 ms, while
the full-volume EDT alone took about 219 ms. The following exact-semantics
changes were implemented on 2026-07-15:

- the compact TEASAR backends crop to the tight foreground bounding box plus a
  one-voxel zero halo and restore the integer crop origin before converting
  vertices to physical coordinates;
- DBF values are gathered into compact-node order and the dense DBF is released
  before the root/path solves; the padded mask is moved into active state rather
  than copied;
- the shared EDT initializes its uninitialized squared-distance buffer during
  the first-axis gather and parallelizes distance materialization;
- invalidation maintains a persistent union of x intervals for each `(z, y)`
  row, so overlapping cubes scan only newly covered spans.

The dense backend remains unchanged as a correctness oracle. Compact FP64
vertices, edges, and radii remained array-exact with dense FP64 across embedded,
boundary-touching, sparse, dense, isotropic, anisotropic, and low/high-PDRF
cases.

### Profile and memory result

One profile-build call on the `256^3` branching tube at four workers:

| phase | before | after |
| --- | ---: | ---: |
| input crop / padding | unprofiled full volume | 30.6 ms |
| distance transform | 218.5 ms | 48.6 ms |
| compact domain | 42.0 ms | 14.1 ms |
| DBF compaction | -- | 3.3 ms |
| root Dijkstra | 56.6 ms | 57.1 ms |
| PDRF | 3.8 ms | 3.3 ms |
| target selection | 0.8 ms | 0.8 ms |
| rail-path Dijkstra | 28.7 ms | 26.6 ms |
| invalidation | 11.0 ms | 3.7 ms |
| measured Python call | 446.3 ms | 202.3 ms |
| peak RSS | 266,804 KiB | 120,288 KiB |

The unchanged Dijkstra phases confirm that the gain came from the intended
non-Dijkstra work. Peak RSS fell by 54.9%. On the relatively dense `96^3` ball,
the row-interval union reduced invalidation from 27.2 ms to 2.7 ms and the full
call from 159.8 ms to 118.1 ms, comfortably clearing the invalidation retention
gate.

### Normal-build wall-clock result

Five-repeat medians with one warmup, using the same deterministic backend-order
shuffle as the earlier matrix:

| volume | workers | optimized median | previous median | speedup |
| --- | ---: | ---: | ---: | ---: |
| `128^3` | 1 | 36.71 ms | 77.60 ms | 2.11x |
| `128^3` | 2 | 29.06 ms | 60.42 ms | 2.08x |
| `128^3` | 4 | 24.17 ms | 50.01 ms | 2.07x |
| `128^3` | 8 | 24.09 ms | 55.71 ms | 2.31x |
| `192^3` | 1 | 124.50 ms | 303.83 ms | 2.44x |
| `192^3` | 2 | 94.47 ms | 235.93 ms | 2.50x |
| `192^3` | 4 | 85.25 ms | 209.54 ms | 2.46x |
| `192^3` | 8 | 78.89 ms | 195.08 ms | 2.47x |
| `256^3` | 1 | 338.64 ms | 694.69 ms | 2.05x |
| `256^3` | 2 | 241.10 ms | 510.91 ms | 2.12x |
| `256^3` | 4 | 207.33 ms | 444.05 ms | 2.14x |
| `256^3` | 8 | 203.90 ms | 434.67 ms | 2.13x |

The extended sequential matrix also retained exact dense parity. Its production
compact medians were 65.67 ms for the sparse embedded `192^3` tube, 118.03 ms
for the relatively dense `96^3` ball, and 35.61--38.84 ms across the `128^3`
spacing/PDRF cases.

A fresh three-repeat matched reference run put bioimage-cpp at 0.19x, 0.21x,
and 0.20x kimimaro's four-worker time on `128^3`, `192^3`, and `256^3`
respectively (28.75 vs 152.76 ms, 81.64 vs 386.13 ms, and 203.82 vs
1002.83 ms). Kimimaro still has only one useful component-tracing task on these
single-component fixtures.

The shared EDT change was checked separately on a deterministic 50%-foreground
`64 x 512 x 512` volume: five-call medians were 1.028 s at one worker and
344.33 ms at four workers, with exact threaded distances, indices, vectors,
and preallocated outputs.

Final verification: `1141 passed` with third-party pytest plugin autoload
disabled. The normal editable build was restored after profiling.

## Review follow-up: compact dispatch and deferred optimizations

The post-optimization review was resolved on 2026-07-15. The compact
delta-stepping adapter was removed rather than retained behind its
`1 << 20`-foreground threshold. It was unreachable in the established sparse
tube benchmarks and, when exercised by new threshold-focused cases, it caused
a large regression. Compact root and rail solves now always use their optimized
heap; `number_of_threads` still controls the exact distance transform.

Reproduce the dispatch matrix with:

```bash
python development/skeleton/benchmark_teasar_dispatch.py \
    --threads 1 2 4 --repeats 5 --warmup 1
```

The benchmark checks array-exact vertices, edges, and radii across worker
counts. Five-repeat medians before and after removing the adapter were:

| case | workers | before | after | change |
| --- | ---: | ---: | ---: | ---: |
| solid cube, 1,061,208 foreground | 1 | 1037.89 ms | 1059.10 ms | +2.0% |
| solid cube, 1,061,208 foreground | 2 | 1970.56 ms | 1044.70 ms | -47.0% |
| solid cube, 1,061,208 foreground | 4 | 1872.36 ms | 1028.86 ms | -45.1% |
| solid cuboid, 1,048,576 foreground | 1 | 1066.11 ms | 1115.94 ms | +4.7% |
| solid cuboid, 1,048,576 foreground | 2 | 1985.48 ms | 1130.47 ms | -43.1% |
| solid cuboid, 1,048,576 foreground | 4 | 1943.07 ms | 1093.50 ms | -43.7% |
| branching tube, 132,619 foreground | 1 | 305.56 ms | 313.56 ms | +2.6% |
| branching tube, 132,619 foreground | 2 | 228.84 ms | 239.68 ms | +4.7% |
| branching tube, 132,619 foreground | 4 | 198.84 ms | 199.86 ms | +0.5% |

The small single-thread differences are within the predeclared 5% noise gate;
the exact-threshold cases show that the removed dispatch nearly doubled
runtime. The branching tube still scales through the threaded EDT without
parallelizing its narrow shortest-path wavefronts.

Other review-driven cleanup retained exact behavior: dense TEASAR releases its
first root field after selecting the final root; dense and compact TEASAR share
one invalidation-bounds calculation; invalid compact targets are checked before
workspace mutation; and the failed experimental FP32 backend is no longer
compiled or exposed by the development selector. Dense FP64 remains an
independent correctness oracle rather than being folded into a template-heavy
common rail driver.

The following proposed optimizations were deliberately deferred:

- A pre-sorted target list would replace an O(P x V) scan, but the measured
  target-selection phase is only 0.8 ms on the `256^3` fixture. Its O(V log V)
  setup, extra memory, and invalidation bookkeeping cannot pay back materially
  at present.
- Touched-node or generation-based compact state reset could avoid a V-byte
  clear per rail, but all rail-path Dijkstra work is only 26.6 ms in the same
  profile. Adding a branch/write to every discovery without first isolating
  reset time risks moving cost into the hotter relaxation loop. Revisit only
  after profiling reset as its own phase on a many-rail case.
- A full dense/compact rail-loop abstraction was rejected for now. The indexing,
  storage, crop handling, and invalidation strategies are intentionally
  different, and keeping the dense oracle structurally independent helps it
  catch compact-specific regressions. Only the identical, policy-free bounds
  calculation was shared.
- Candidate-target and state-reset changes are therefore benchmark follow-ups,
  not pending correctness fixes.
