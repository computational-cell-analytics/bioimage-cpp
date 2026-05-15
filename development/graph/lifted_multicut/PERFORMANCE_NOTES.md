# Lifted Multicut Performance Notes

State of the lifted-multicut solvers vs nifty on the standard benchmark
problems and notes on remaining optimization headroom. Read this before the
next round of perf work.

## Current benchmark (3D ISBI lifted problem)

Problem dimensions: 2462 nodes, 17 949 local edges, 21 444 lifted edges.

| Solver | bic | nifty | Ratio | Status |
|---|---|---|---|---|
| Greedy additive | ~13 ms | ~14.7 ms | **0.88×** (faster) | Goal met |
| Kernighan-Lin (greedy + 10 outer) | ~200 ms | ~102 ms | **1.95×** (slower) | Outside 30%-of-nifty target |

Energies match nifty to within numerical noise on both solvers (greedy diff
~0.3, KL diff ~0.08).

## What's done

### Greedy additive

Three Python-side wins (~3× speedup, 39 → 13 ms):

1. **Bulk `_add_lifted_edges` fast path** in `src/bioimage_cpp/graph/__init__.py`.
   Replaced a per-row Python loop over `lifted_uvs` with one
   `insert_edges` call plus `np.bincount` for the residual
   collision case.
2. **`UndirectedGraph.from_unique_edges` binding** in `src/bindings/graph.cxx`.
   Bypasses the per-edge hash dedup that `insert_edge` performs; used
   from `_copy_graph` and from the lifted-graph construction path.
3. **Dropped defensive base-graph copy** in `LiftedMulticutObjective`.
   The C++ `Objective` already only holds a `const UndirectedGraph &`; the
   Python wrapper now matches.

The C++ greedy kernel itself runs in ~4 ms — already at the algorithmic
floor for ~21 k heap operations.

### Kernighan-Lin

One C++ optimization (~22% speedup, 245 → 200 ms):

1. **Pre-built per-node filtered adjacency** in
   `include/bioimage_cpp/graph/lifted_multicut/kernighan_lin.hxx`.
   `ChainScratch` gained `filtered_offset`, `filtered_count`, and
   `filtered_entries` (each entry caches `{node, weight, is_base}`).
   `chain_gain_init` populates these alongside the gain accumulation; the
   chain loop iterates only in-pair neighbors and skips the
   `bufs.in_pair[u_key]` filter check. Also caches `was_in_heap` once per
   iteration so the three subsequent heap operations don't each re-query
   the locator.

## What remains — KL is the open item

C++ KL takes ~200 ms; nifty ~102 ms. Profile breakdown
(`BIOIMAGE_PROFILE=ON`, 1 repeat) — note the inner scopes inflate total by
~80% so trust the relative shares:

```
  cc_repartition           0.0033 s  (  0.7%)
  energy_eval              0.0013 s  (  0.4%)
  compute_pairs            0.0073 s  (  2.0%)
  chain_init               0.0037 s  (  1.0%)
  chain_gain_init          0.0812 s  ( 22.1%)   <-- ~36% of pair_chains
  chain_loop               0.0880 s  ( 23.9%)   <-- ~63% of pair_chains
  chain_cleanup            0.0017 s  (  0.5%)
  pair_chains              0.1744 s  ( 47.4%)
  cluster_splits           0.0083 s  (  2.3%)
```

Workload statistics on the 3D problem after the greedy warm-start:

- 643 clusters. Sizes: min 1, max 139, mean 3.8, median 2.
- 4443 base cluster pairs per outer iter. Pair sizes: mean 20, median 7,
  max 198 (long tail).
- Average lifted node degree: 32.

## Ranked future optimizations

### 1. Pre-bucketed gain init (estimated landing point: ~150 ms total, −25%)

**Idea.** For each outer iteration, bucket every lifted edge by its
endpoint cluster pair (sorted `(min_label, max_label)`). For pair-chain
`(A, B)` the gain init iterates only the three relevant buckets —
`(A, A)`, `(B, B)`, `(A, B)` — instead of walking the full lifted
adjacency of every node in the pair.

**Why it would help.** Each edge currently contributes to `chain_gain_init`
exactly once per pair-chain that touches one of its endpoint clusters.
With buckets, each edge contributes O(1) per outer iter. The 80 ms
`chain_gain_init` collapses to roughly O(E) = ~5 ms per outer iter, saving
~70 ms.

**The wrinkle.** Bucket membership goes stale as nodes move between
clusters during the sequential pair-chains. The right fix is **incremental
bucket maintenance** — every node move performs O(degree) bucket-membership
updates (remove from old bucket, push into new). Bucket entries store
their position via an `edge_id → bucket_pos` index so removal is
`swap-with-back` in O(1). Total maintenance cost per outer iter:
`O(num_moves × avg_degree)` ≈ 30 µs on our problem.

**Complexity.** Substantial: ~200 lines, new `LiftedEdgeBuckets` struct in
`detail_kl`, hooks in `chain_loop` to call `buckets.relabel(v, old, new)`
on every committed move, careful invariant management.

**Sanity check before implementing.** Try a quick prototype where buckets
are rebuilt fresh at the start of each outer iter (O(E) per outer iter)
and gain init uses buckets, accepting that within-outer-iter moves create
stale entries. If the resulting energy is comparable to the exact version,
the incremental maintenance is worthwhile.

### 2. CSR adjacency layout for lifted graph (estimated: 10–15% off)

**Idea.** `UndirectedGraph` stores adjacency as `vector<vector<Adjacency>>`
— one heap allocation per node. Walking adjacency for many distinct nodes
in a pair pays per-node pointer chasing. A flat CSR (`offsets[n+1]` +
`entries[2E]`) built once at the start of `kernighan_lin` would be more
cache-friendly.

**Caveat.** The actual access pattern in `chain_gain_init` walks
adjacency for nodes in arbitrary order (whichever pair we're processing),
so spatial locality across nodes is poor regardless of layout. The win is
limited to per-node cache-line savings (one miss per node vs one per
adjacency vector header). Estimate ~10% based on rough cycle counting.

**Complexity.** Localized: ~50 lines, build CSR in `kernighan_lin`,
replace `lifted_graph.node_adjacency(v)` calls in `chain_gain_init` with
CSR iteration.

**Worth combining** with optimization 1, since CSR walking is what bucket
maintenance would need anyway.

### 3. Inspect nifty's internals (estimated: unknown, possibly clarifying)

The gap between our `chain_gain_init` and nifty's equivalent
`computeDifferences` is suspiciously ~2× per adjacency entry given that
both algorithms walk the same data. nifty's source is at:

```
/home/pape/Work/software/src/nifty/include/nifty/graph/opt/lifted_multicut/detail/lifted_twocut_kernighan_lin.hxx
```

Worth checking:
- How nifty stores `liftedGraph_.adjacency(v)` — is it CSR-like?
- Whether nifty's `referencedBy` array is `uint32` or larger (we use
  `uint32`).
- Whether nifty's `differences` (= our `stash_gain`) cache line layout
  differs.

### 4. Linear-scan border instead of a heap

**Verdict: not worth pursuing for this problem.**

I worked through it: heap is faster than linear scan for our pair-size
distribution. The heap pop is O(log N) and heap.change is also O(log N);
for pair size 7 (the median) that's ~2× faster than O(N) linear scan,
and the gap widens for larger pairs.

nifty uses linear scan, but that's not where its speed advantage comes
from — likely it's the adjacency-walking constants (optimization 3).

### 5. Skip pair-chains where heap stays empty (estimated: <5 ms)

After `chain_gain_init`, if `heap.empty()` we already skip
`chain_loop`. But we still pay for `chain_init` and the full
`chain_gain_init` walk. For pair-chains where the cluster pair has only
one alive node per side (post-staleness filtering of `cluster_to_nodes`),
we could skip earlier. Need to maintain live cluster sizes.

Low priority — only a few ms.

## Where to start next time

1. Re-run `cd development/graph/lifted_multicut && python check_kernighan_lin.py --size 3d --repeats 5` to confirm the baseline hasn't drifted.
2. Build with `pip install -e . --no-build-isolation -C cmake.define.BIOIMAGE_PROFILE=ON` to get the profile breakdown back.
3. Prototype the rebuild-per-outer-iter bucket approach (optimization 1
   simplified) to validate the energy quality before committing to the
   incremental maintenance version.
4. Targets:
   - 30%-of-nifty: ≤132 ms.
   - Match nifty: ≤102 ms.

## Files that matter

- `include/bioimage_cpp/graph/lifted_multicut/kernighan_lin.hxx` —
  KL kernel; existing profile scopes wrap each phase.
- `include/bioimage_cpp/graph/lifted_multicut/greedy_additive.hxx` —
  greedy kernel; profile scopes already present, currently at ~4 ms
  so not the focus.
- `include/bioimage_cpp/graph/lifted_multicut/objective.hxx` — objective
  state, including `n_base_edges` and the lifted graph.
- `src/bioimage_cpp/graph/__init__.py::LiftedMulticutObjective` —
  Python construction path (already optimized).
- `development/graph/lifted_multicut/_compatibility.py` —
  bic-vs-nifty harness; uses `run_comparison(...)`.
- `tests/graph/lifted_multicut/test_external_problem.py` — regression
  test on the 2D problem; energy bound is ENERGY_BOUND = -1574.5.
