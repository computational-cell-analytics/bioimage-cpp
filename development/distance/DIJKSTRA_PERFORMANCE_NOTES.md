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

Pre-optimization full-suite baseline: `1126 passed`. Re-run
`tests/distance/test_grid_dijkstra.py` after every kernel/data-structure change,
then the full suite before accepting benchmark gains.

## Implemented sequential optimization results

Implemented and measured on 2026-07-14. The public API and FP64 results remain
unchanged. The kernel now uses fixed 2D/3D neighbor tables, direct boundary
tests, cost-mode dispatch outside the neighbor loop, a lazy binary heap for
physical costs, first-discovery insertion for directed node costs, and the
dense indexed heap only for node-times-physical costs. `DijkstraWorkspace`
retains state, heap, predecessor, and geometry capacities for C++ callers.

Default-tier medians (one warmup, three measured calls) changed as follows:

| operation | `1024^2` before | after | speedup | `64x160x160` before | after | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| physical field | 224.94 ms | 129.80 ms | 1.73x | 1.102 s | 384.40 ms | 2.87x |
| field + predecessors | 239.55 ms | 138.53 ms | 1.73x | 1.120 s | 403.02 ms | 2.78x |
| node-cost field | 215.45 ms | 128.04 ms | 1.68x | 1.113 s | 306.53 ms | 3.63x |
| far-target physical path | 224.11 ms | 139.83 ms | 1.60x | 1.101 s | 373.07 ms | 2.95x |

All modes continue to pass the independent Python heap oracle, deterministic
path, predecessor-chain, zero-cost, connectivity, anisotropy, and validation
tests. The public implementation stays dense and FP64; compact indexing and
reduced precision were evaluated only for the internal TEASAR workload.

The final large-tier medians (same one-warmup/three-repeat protocol as the
original table) were 568.17 ms / 626.03 ms / 569.91 ms / 620.64 ms in 2D and
3.565 s / 3.813 s / 2.179 s / 3.629 s in 3D for physical, predecessors, node,
and far-target path respectively. Relative to the committed baseline, the
large 3D speedups are 2.84x, 2.62x, 4.10x, and 2.75x.

Final verification after the optimization: `1129 passed`.

## Implementation details and alternatives evaluated

The final implementation is not one universal Dijkstra loop with a different
edge-cost callback. The measurements showed that the three cost conventions
have different queue behavior, so dimension and cost mode are dispatched once
before entering the hot loop.

### Fixed grid traversal

The old kernel recursively constructed dynamic neighbor offsets for every
solve and called generic offset-validation logic for every candidate edge. The
new workspace caches at most 8 neighbors in 2D or 26 in 3D. Each entry stores:

- signed per-axis steps for border checks;
- the signed C-order flat-index delta;
- the physical step length under the requested spacing.

A popped node is decoded once. Interior nodes use only flat deltas; border
nodes additionally check the signed steps against the shape. Neighbor order is
still lexicographic, and heap entries are ordered by `(distance, flat_index)`,
so deterministic tie behavior is retained.

A zero-halo specialization was also prototyped for TEASAR, where every
foreground node is known to be surrounded by an explicit background border.
Removing all bounds checks from that path was only about 5% faster after the
fixed-neighbor rewrite. Compact foreground indexing offered much larger gains,
so a separate public zero-halo API was not added.

### Queue choice by cost mode

Three queue strategies were tested:

1. the original generic dense addressable heap;
2. a smaller Dijkstra-specific addressable heap with fixed grid traversal;
3. a vector-backed binary heap with lazy duplicate entries and stale-pop
   rejection.

Representative direct-kernel prototype timings were:

| workload and mode | generic indexed | specialized indexed | specialized lazy |
| --- | ---: | ---: | ---: |
| default 3D physical | 1.063 s | 430 ms | **386 ms** |
| default 3D node | 1.077 s | 420 ms | **349 ms** |
| large 3D physical | 8.257 s | 5.027 s | **4.310 s** |
| large 3D node | 7.576 s | 4.249 s | **3.460 s** |

These were development-kernel measurements rather than the public Python
benchmark, so their absolute numbers should not be mixed with the final tables.
They were used to choose the queue design.

For directed node costs, every edge entering node `v` has the same weight,
`costs[v]`. The first settled neighbor that discovers `v` therefore supplies
an optimal distance; a later decrease is impossible. Random-cost stress tests
confirmed zero decreases. The final node kernel consequently inserts each node
only once and needs neither a locator nor stale duplicates. On a random 3D
field this specialized lazy/discovery kernel took 763 ms versus 1.468 s for the
generic indexed implementation.

That property does not hold for `node_times_physical`: different incoming
directions have different physical lengths. A random weighted benchmark
measured 896 ms for the specialized indexed heap and 1.089 s for lazy
duplicates. The lazy heap reached 199,205 entries versus 83,243 live entries
for the indexed heap. The final implementation therefore keeps the dense
indexed heap for this mode, while physical distances use lazy duplicates and
node costs use one-time discovery.

Bucket and radix queues were not pursued for the sequential public solver.
Physical step lengths and user node costs are arbitrary floating-point values,
including zero, so an integer queue would either narrow the API or introduce a
quantization policy. Bucketing is reconsidered below as a parallel scheduling
mechanism, where it need not change the exact FP64 distance calculation.

### Workspace, state, and validation

`DijkstraWorkspace` retains state, heap storage, predecessor scratch space,
distance scratch space, strides, and neighbor metadata. Allocation-owning C++
and Python functions remain available as before. Path calls track touched
indices and clear only reached state after early termination; full fields still
initialize the requested full output because background and unreachable voxels
must be `+inf`/`-1`.

Discovered, settled, and target flags share one byte per dense node. Weighted
cost validation now happens once in the C++ boundary/kernel path; the duplicate
full-array NumPy scan was removed. The Python wrapper still converts costs to a
C-contiguous FP64 array and checks shape before releasing the GIL.

### Compact foreground indexing for TEASAR

The standalone public functions stay dense because callers expect a dense
field. TEASAR only needs distances and predecessors on its foreground, so it
now builds deterministic compact IDs in ascending padded flat-index order. The
selected on-the-fly representation stores:

- a `uint32` full-to-compact lookup over the padded volume;
- a `uint32` compact-to-full array;
- compact FP64 root/PDRF fields;
- compact byte state, `uint32` predecessors and heap nodes;
- compact skeleton-target and voxel-to-vertex arrays.

The padded mask, exact distance-to-boundary field, and invalidation bitmap stay
dense. The implementation falls back to optimized dense FP64 if padded flat
indices or foreground IDs cannot be represented below the `uint32` sentinel.

For the padded `256^3` branching tube, the domain has 17,173,512 total voxels,
132,619 foreground voxels (129.5x compression), and 3,180,936 directed
foreground adjacencies. The full-to-compact lookup is 65.5 MiB and
compact-to-full is 0.51 MiB. The alternative CSR representation adds about
1.01 MiB of `size_t` offsets, 12.13 MiB of targets, and 3.03 MiB of neighbor
codes, then releases the full lookup.

Both compact FP64 variants produced bitwise-identical TEASAR vertices, edges,
and radii versus dense FP64 across all tested sizes, density regimes, spacings,
and PDRF ranges. End-to-end medians were:

| backend | `128^3` | `192^3` | `256^3` |
| --- | ---: | ---: | ---: |
| optimized dense FP64 | 134.66 ms | 618.82 ms | 1.169 s |
| compact on-the-fly FP64 | **74.04 ms** | **267.21 ms** | **686.58 ms** |
| compact CSR FP64 | 75.19 ms | 284.57 ms | 730.72 ms |

CSR was 6--7% slower on the two large tiers. It was 2.3% faster on a very
sparse `192^3` object but 40% slower on a relatively dense `96^3` ball. Peak
RSS was effectively the same because CSR must construct the full lookup before
releasing it. On-the-fly traversal was selected, but the compact ID scheme does
not lock a future parallel backend into that choice; parallel CSR locality must
be measured again.

### Reduced precision result

Compact CSR was also instantiated with FP32 root distances, PDRF values, and
heap priorities. It did not pass either side of the adoption gate. On `256^3`,
combined root/path Dijkstra was 113.9 ms in FP32 versus 108.9 ms in FP64,
end-to-end time was slightly slower, and peak RSS stayed near 266.5 MiB because
dense EDT/mask storage and compact adjacency dominated. On `192^3`, FP32 also
changed the contracted topology, had a 2.0-voxel bidirectional 95th-percentile
distance, a 7.07-voxel Hausdorff distance, and a 2.26% skeleton-length error.
Production TEASAR therefore remains FP64.

### Remaining sequential bottleneck and speedup ceiling

The selected `256^3` profile is:

| phase | time | fraction |
| --- | ---: | ---: |
| exact distance transform | 495.0 ms | 77.4% |
| compact-domain construction | 41.6 ms | 6.5% |
| two root Dijkstra fields | 58.8 ms | 9.2% |
| all rail-path Dijkstra calls | 29.0 ms | 4.5% |
| PDRF, selection and invalidation | 15.4 ms | 2.4% |

Root and path shortest paths together are now only 13.7% of end-to-end time.
Even an infinitely fast parallel shortest-path backend would improve this case
by at most 1.16x. A fourfold Dijkstra speedup alone would improve end-to-end
time by about 1.11x. The parallel follow-up should therefore expose one thread
budget to the existing threaded EDT as well as to shortest paths; otherwise a
successful parallel Dijkstra implementation will look modest end-to-end.

## Parallel implementation strategy

The first parallel target should be the compact internal TEASAR backend, not
the dense public field. It has much less mutable state, immutable on-the-fly
neighbor lookup, and realistic downstream workloads. The current sequential
heap remains the `number_of_threads=1` implementation and fallback.

### What can and cannot run concurrently

- The two root sweeps are dependent: the second source is selected from the
  first field, so they cannot run concurrently.
- TEASAR rails are also dependent. Each accepted rail changes the skeleton
  target set, PDRF zeros, and active invalidation state used by the next rail.
  Running complete rails concurrently would change the algorithm and is not
  the initial plan.
- Neighbor relaxations within one root field or one rail search are the useful
  parallel unit.
- EDT and compact-domain setup are independent of the rail loop and should use
  the same user thread budget. Compact ID construction can later use block
  counts plus a prefix sum if profiling shows it remains significant.

### Proposed shortest-path algorithm: deterministic delta stepping

Implement a separate compact FP64 delta-stepping backend. Delta stepping groups
tentative distances into buckets of width `Delta`, permitting many nodes to be
relaxed together while retaining exact non-negative FP64 distances. It is a
scheduling change, not distance quantization.

For each non-empty bucket:

1. Take its current frontier in compact-ID order.
2. Use `detail::parallel_for_chunks` to scan frontier nodes. Workers only read
   the immutable compact domain and the current distance snapshot and append
   relaxation proposals to per-thread buffers.
3. Concatenate and deterministically reduce proposals by
   `(target, candidate_distance, predecessor_key)`. Apply the winning updates
   on the calling thread, eliminating data races on distances and
   predecessors.
4. Repeat light-edge relaxations (`weight <= Delta`) until the current bucket
   is closed.
5. Relax heavy edges from all nodes removed from the bucket and insert their
   updates into later buckets.

This bulk-synchronous proposal/reduction design avoids non-portable atomic
`{distance, predecessor}` pairs and makes results repeatable. The predecessor
key should include predecessor distance, compact ID, and neighbor code so equal
candidate distances have an explicit order. Exact fields should match the
sequential solver; an equal-cost predecessor path may differ only if the new
documented tie rule differs from serial discovery order.

Physical edge classification uses the precomputed neighbor length. Directed
node-cost classification uses `pdrf[target]`; zero-cost edges are light and
must participate in the current-bucket closure. `node_times_physical` can use
`cost[target] * length` if the backend is later generalized to the public
solver.

For early-stop rail searches, do not stop when a worker first sees a skeleton
target. Finish closing the current bucket and applying all proposals that can
reach it from the same or an earlier bucket. Then select the target with the
minimum final `(distance, compact_id)` and reconstruct its predecessor chain.

### Bucket width and small-frontier fallback

`Delta` controls available parallelism versus redundant work and must be
benchmarked rather than fixed from theory alone:

- physical fields: start with `min(spacing)` and test 0.5x, 1x, 2x and 4x;
- node-cost paths: sample positive foreground PDRF values and test lower
  quartile, median and scaled-median widths;
- treat an all-zero sampled cost field as one light-edge closure.

The existing threading helper creates and joins workers per invocation. Thin
buckets or short rail searches should therefore stay on the sequential heap.
Start with a frontier threshold around several thousand nodes, record worker
launch and deterministic-merge time, and tune the threshold on `64^3` through
`256^3`. This uses the repository's existing threading primitive and avoids a
second worker-pool implementation.

Both compact adjacency layouts should be retested. On-the-fly neighbors won
sequentially, but CSR's contiguous edge ranges may scale better when several
threads scan different frontier chunks. Select on end-to-end time and peak RSS,
not relaxation throughput alone.

### API and workspace shape

- Add `number_of_threads` to TEASAR, normalized with
  `detail::normalize_thread_count`; retain `1` as the conservative behavior
  while the backend is experimental.
- Pass the normalized count to the existing distance transform and compact
  shortest-path backend.
- Add a parallel workspace containing FP64 distances, `uint32` predecessors,
  bucket membership/generation marks, and one reusable proposal buffer per
  thread. Bound and reuse proposal capacity so a difficult bucket cannot cause
  repeated allocation spikes.
- Keep the sequential `CompactDijkstraWorkspace` intact. Do not put atomics in
  its fast single-threaded arrays or regress the established fallback.
- Keep the GIL released for the full C++ call and touch no Python objects from
  workers.

The existing choices do not need to be unwound: compact IDs and on-the-fly
lookup are immutable and safe for concurrent reads, FP64 is suitable for the
parallel backend, and the heap implementation is isolated behind the
sequential path. The byte state array is not safe for concurrent mutation, so
the parallel workspace must use staged proposal application or explicit atomic
state rather than sharing it directly.

### Alternatives not selected initially

- **Concurrent complete rails:** conflicts with TEASAR's evolving target and
  invalidation state and would be a separate approximate algorithm.
- **Relaxed multi-queue Dijkstra:** offers more asynchronous parallelism but
  makes early stopping, repeatability, and predecessor ties harder. It is a
  fallback experiment only if deterministic delta stepping cannot scale.
- **Parallel Bellman-Ford/frontier relaxation:** simple, but likely performs
  too many grid-edge scans on long weighted paths.
- **Reintroducing FP32:** does not address synchronization or bucket work and
  already failed the sequential performance and quality gates.
- **Replacing `detail::parallel_for_chunks`:** would violate the repository's
  single-threading-primitive policy; first make bucket/frontier size large
  enough to amortize its launches.

### Parallel benchmarks and acceptance gates

Extend the sequential matrix with `number_of_threads = 1, 2, 4, 8` and record:

- EDT, compact-build, root-field and rail-path phase totals;
- bucket count, light-closure rounds, relaxations, accepted updates,
  duplicates, proposal/merge time, and peak per-thread buffer capacity;
- early-stop settled-node counts and rail count;
- end-to-end time and peak RSS for on-the-fly and CSR adjacency.

Use deterministic backend-order shuffling, one warmup and five measured calls.
The first acceptance target is at least 1.75x combined root/path speedup at four
threads on both `192^3` and `256^3`, at least 5% end-to-end improvement when
shortest-path threading is considered alone, and no regression over 5% on
small or relatively dense cases. Peak RSS should stay within 30% of compact
sequential FP64.

Correctness tests must cover exact fields, zero weights, anisotropic physical
weights, disconnected masks, multiple/equal targets, and targets reached in a
light-edge closure. Calls at each thread count must be repeatable. If exact
predecessor parity is intentionally relaxed, retain the previously used TEASAR
gate: matching contracted branch/leaf topology, bidirectional 95th-percentile
distance at most one normalized voxel, Hausdorff distance at most two voxels,
and total physical length within 2% of sequential compact FP64.

## Initial parallel follow-up (dispatch later narrowed)

Implemented after reviewing the draft above. The implementation scope was
expanded to the dense public Dijkstra fields as well as the compact TEASAR
backend. Both use one deterministic FP64 delta-stepping primitive in
`distance/detail/delta_stepping.hxx`; the optimized heaps remain unchanged.

This section records the initial implementation. The review follow-up below
supersedes its weighted-field and compact-TEASAR dispatch decisions.

The concrete design closes several gaps in the draft:

- frontier nodes and proposals are processed in fixed global batches independent
  of thread count, with at most 1,048,576 worst-case proposals per batch;
- workers write only per-thread buffers, then the calling thread sorts, reduces,
  and applies strict improvements in a deterministic order;
- equal-distance updates do not rewrite existing predecessors, preventing cycles
  in zero-weight components;
- the heavy phase uses the unique set removed during the complete light closure;
- early stopping happens only after both phases and after checking that FP64
  rounding did not reinsert work into the current bucket;
- generation-marked compact state avoids clearing every rail-search array;
- invalid bucket widths or bucket-index overflow restart on the sequential heap.

The bucket width is the minimum physical neighbor length for physical costs, the
deterministically sampled positive-cost median for node costs, and their product
for node-times-physical costs. All-zero costs use a positive sentinel width and
remain entirely in the light closure.

Initial measurements on the four-core development machine also rejected the
assumption that every large one-source solve benefits from delta stepping.
Representative one-shot default-tier dense physical fields took about 0.38 s on
the heap versus 0.64 s with four-thread staged relaxation, while a full source
plane on the same `64 x 160 x 160` domain improved from about 0.72 s to 0.30 s.
The directed-node heap remained decisively faster. TEASAR's compact wavefronts
were likewise too narrow through the `256^3` tier, although threading its EDT
still improved end-to-end time.

The production dispatch therefore follows the benchmark rather than forcing the
new backend: one-thread calls, paths, directed-node costs, small fields, and
narrow multi-source fields retain the heap. Broad multi-source physical and
node-times-physical fields use delta stepping. Compact TEASAR keeps heap Dijkstra
through the measured tiers while sharing the requested thread budget with EDT;
the delta backend remains available for much larger compact domains. The
development benchmarks now accept thread matrices so these thresholds can be
revisited with evidence on other machines.

## Review follow-up: retained delta-stepping scope

The review follow-up on 2026-07-15 narrowed delta stepping to the case for which
the benchmark shows a consistent benefit: broad multi-source **physical**
fields. Paths, one/few-source fields, both weighted modes, small problems, and
all compact TEASAR solves use their specialized heaps. The compact adapter and
its duplicated sampled-cost-median heuristic were removed completely.

`development/distance/benchmark_dijkstra.py --broad-multisource` adds a full
source plane and can reproduce the dispatch. Three-repeat final medians were:

| shape / mode | 1 thread | 2 threads | 4 threads | 1-to-4 scaling |
| --- | ---: | ---: | ---: | ---: |
| `256 x 256`, physical | 13.57 ms | 5.89 ms | 5.85 ms | 2.32x |
| `256 x 256`, node-times-physical | 9.80 ms | 9.62 ms | 9.49 ms | 1.03x |
| `32 x 96 x 96`, physical | 120.55 ms | 60.46 ms | 55.22 ms | 2.18x |
| `32 x 96 x 96`, node-times-physical | 92.31 ms | 89.31 ms | 92.27 ms | 1.00x |

The weighted rows now show heap noise rather than parallel scaling. Before the
dispatch was narrowed, their 2/4-thread medians were 149.35/148.64 ms in 3D,
versus the final 89.31/92.27 ms. Retaining delta stepping for that mode would
have knowingly shipped a roughly 1.6x regression. The physical 3D row retains
the useful broad-wavefront speedup (the pre-change medians were 122.79, 60.81,
and 51.06 ms; the four-thread final minimum was 50.79 ms).

The bucket-index quotient now uses `double`, matching the actual distance and
delta types on every wheel target. Conversion uses an exclusive, exactly
representable `2^64` upper bound to avoid undefined out-of-range conversion.
A decimal-spacing case above the parallel threshold (`0.1, 0.5, 1.25`) asserts
array-exact distances against the heap and deterministic predecessors between
two and four threads.

Two proposed changes remain intentionally unimplemented:

- Equal-candidate predecessor handling still uses the existing strict-update
  behavior. Distances are exact, two/four-thread predecessor fields are
  deterministic, and refusing equal-distance rewrites prevents cycles in
  zero-weight components. Harmonizing the small-frontier and merged-proposal
  tie rules would change an observable predecessor/path choice, so it needs a
  separately chosen public tie contract rather than being bundled into a
  portability fix.
- `std::map` buckets were not replaced speculatively. A flat vector gives O(1)
  lookup for dense bucket ranges but can allocate excessively when FP64 weights
  create large sparse index gaps; a ring or segmented design adds rebasing and
  overflow complexity. Profile bucket lookup/allocation separately on the
  retained broad physical workload before changing the data structure.

The dispatch threshold (`foreground >= 32768` and at least approximately one
source per 256 foreground voxels) remains benchmark-derived and intentionally
conservative. The standalone benchmark records both heap and dispatched cases
so it can be retuned on materially different wheel hardware.

Final validation used a normal editable build and reported `1145 passed`. The
optional array-view lifetime fix was also suite-tested in a separate build with
`-fno-elide-constructors`, specifically removing any dependence on NRVO.
