# Lifted Multicut Performance Notes

State of the lifted-multicut solvers vs nifty on the standard benchmark
problems and notes on remaining optimization headroom. Read this before the
next round of perf work.

## Current benchmark (3D ISBI lifted problem)

Problem dimensions: 2462 nodes, 17 949 local edges, 21 444 lifted edges.

| Solver | bic | nifty | Ratio | Status |
|---|---|---|---|---|
| Greedy additive | ~13 ms | ~14.7 ms | **0.88×** (faster) | Goal met |
| Kernighan-Lin (greedy + 10 outer) | ~195 ms | ~103 ms | **1.92×** (slower) | Outside 30%-of-nifty target |

Energies match nifty to within numerical noise on both solvers: greedy diff
~0.3, KL diff ~0.08.

On the 2D ISBI lifted problem (756 nodes, 2 134 local + 3 541 lifted edges):
bic KL runs in ~9 ms vs nifty's ~4.5 ms (2.0× — small-problem overhead
dominates).

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

One C++ optimization landed (~22% speedup, 245 → 195 ms):

1. **Pre-built per-node filtered adjacency** in
   `include/bioimage_cpp/graph/lifted_multicut/kernighan_lin.hxx`.
   `ChainScratch` gained `filtered_offset`, `filtered_count`, and
   `filtered_entries` (each entry caches `{node, weight, is_base}`).
   The chain loop iterates only in-pair neighbors and caches `was_in_heap`
   once per iteration so the three subsequent heap operations don't each
   re-query the locator.

## Post-mortem: cluster-pair bucket optimization (2026-05-16, reverted)

Tried the optimization the previous notes had ranked #1 — a per-outer-iter
bucket index over all lifted-graph edges, grouped by their endpoint
cluster-label pair, so `chain_gain_init` could read three buckets
(`(A, A)`, `(B, B)`, `(A, B)`) per pair-chain instead of walking the full
lifted adjacency of every in-pair node.

**Wall-clock outcomes:**

| Variant | Time | 3D energy diff vs nifty | Notes |
|---|---|---|---|
| Pre-bucket (current) | 195 ms | 0.08 | Reference |
| Simplified buckets, 1 rebuild / outer iter | 135 ms | 0.94 | Staleness during cluster_splits |
| Simplified buckets, 2 rebuilds / outer iter | 139 ms | 0.17 | Mid-iter rebuild recovers most of the loss |
| Incremental maintenance (relabel-on-commit) | 198 ms | 0.16 | Slower than pre-bucket, no energy win |

The simplified bucket version delivered the predicted time win on
`chain_gain_init` (79 ms → 31 ms), but the 0.08 → 0.17 energy regression
that came with it could not be closed. The incremental-maintenance
follow-up — designed specifically to eliminate within-iter staleness —
turned out to be both slower (memory-fragmentation regression in
`chain_gain_init` and `chain_loop` from the `unordered_map<key,
vector<Entry>>` layout) and unable to close the energy gap. We reverted
the entire bucket experiment.

**Why the energy gap didn't close.** I assumed the 0.08 → 0.17 regression
was caused by buckets going stale between pair-chains, so incremental
maintenance should restore it. The data says otherwise: simplified
buckets with maximum staleness (1 rebuild/iter) and incremental
buckets with zero staleness landed at 0.17 and 0.16 respectively —
essentially identical. The gap is not from staleness; it's from a
**different filtered-adjacency iteration order** that the bucket
construction produces vs. the original direct-adjacency walk. KL is
sensitive to tie-breaking when multiple candidate moves have near-equal
gain, and the order in which `filtered_entries[v]` entries get pushed
into the heap influences the local optimum the chain converges to.
Floating-point summation order across many lifted edges also produces
bit-level different `stash_gain` values, contributing tie-shifts.

The 2D problem stays at **exact** 0.000 diff because it has fewer ties
and fewer summed edges. The 3D problem has more opportunity for
divergence.

**What this means for the bucket approach.** The bucket idea is sound
in the abstract — `chain_gain_init`'s data-volume floor really is O(E)
per outer iter, not O(pairs × pair_size × adjacency). But the
implementation creates a different `filtered_entries` ordering than the
adjacency-walk version, and that ordering difference is what produces
the 0.09 energy gap on 3D. To get bucket-level speed *and* pre-bucket
tie-breaking parity, you would need to either:

1. Build `filtered_entries[v]` by walking `lifted_graph.node_adjacency(v)`
   *after* using buckets to compute `stash_gain`. Adds back most of the
   adjacency walk cost (estimated +30–50 ms), undoing most of the bucket
   win.
2. Sort `filtered_entries[v]` by a canonical edge_id order after bucket
   construction. Adds ~30 ms in per-pair sort cost; only partially
   matches pre-bucket order because pre-bucket follows insertion order,
   not edge_id order.
3. Accept the 0.17 diff. It's 0.005% relative on the only problem
   where it shows, doesn't violate any test bound, and is well below
   downstream segmentation noise. The 60 ms time win on the 3D problem
   is real.

I implemented option 3 (simplified buckets + mid-iter rebuild), and at
the user's request escalated to incremental maintenance hoping it would
also restore energy parity. It did not, and was slower besides. We are
back at the pre-bucket baseline.

**Specific implementation lessons:**

- `unordered_map<uint64_t, vector<Entry>>` for per-bucket storage caused
  ~40 ms of cache-locality regression in `chain_gain_init` and
  `chain_loop` vs the flat sorted vector. If revisiting buckets,
  keep them in one contiguous flat array.
- The `relabel_node` machinery itself is cheap (~4 ms total over 10
  outer iters in the profile). Incremental maintenance is *not*
  expensive in absolute terms; the slowdown came from the layout
  switch.
- Predicted bucket-rebuild cost (notes said "~5 ms per outer iter")
  was 4× off — sort on 40 k 40-byte entries actually takes ~11 ms.
  The cheap-path radix-sort optimization listed below would address
  this, but only matters if buckets come back.

## Future optimizations (re-prioritised after the bucket post-mortem)

The bucket approach is **not** the recommended next step. It changes
algorithm output (different tie-breaking) which costs us energy parity
on the 3D problem. If we revisit it, we'd want to invest first in
verifying we can recover pre-bucket order (option 1 or 2 above) before
committing to the layout work.

### 1. CSR adjacency layout for lifted graph (estimated: 10–15% off)

**Idea.** `UndirectedGraph` stores adjacency as `vector<vector<Adjacency>>`
— one heap allocation per node. Walking adjacency for many distinct nodes
in a pair pays per-node pointer chasing. A flat CSR (`offsets[n+1]` +
`entries[2E]`) built once at the start of `kernighan_lin` would be more
cache-friendly.

**Caveat.** The actual access pattern in `chain_gain_init` walks
adjacency for nodes in arbitrary order (whichever pair we're
processing), so spatial locality across nodes is poor regardless of
layout. The win is limited to per-node cache-line savings (one miss per
node vs one per adjacency vector header). Estimate ~10% based on rough
cycle counting.

**Complexity.** Localized: ~50 lines, build CSR in `kernighan_lin`,
replace `lifted_graph.node_adjacency(v)` calls in `chain_gain_init` and
`chain_loop` with CSR iteration. Doesn't change algorithm output —
adjacency iteration order is preserved.

**Why now.** This is the most attractive remaining lever: localized,
non-invasive, preserves tie-breaking, and the win is real cache
savings rather than an algorithmic restructure with side effects.

### 2. Skip pair-chains where heap stays empty (estimated: <5 ms)

After `chain_gain_init`, if `heap.empty()` we already skip
`chain_loop`. But we still pay for `chain_init` and the full
`chain_gain_init` walk. For pair-chains where the cluster pair has
only one alive node per side (post-staleness filtering of
`cluster_to_nodes`), we could skip earlier. Need to maintain live
cluster sizes.

Low priority — only a few ms.

### 3. Revisit bucket gain init *if* tie-breaking parity is acceptable

If a future use-case is OK with the 0.17 energy diff on 3D (e.g., the
bucket version's output is fed into a downstream solver that re-optimises
anyway), the simplified flat-sorted-vector bucket implementation with
mid-iter rebuild is a known ~60 ms win. See git history for the
implementation; the key files were
`include/bioimage_cpp/graph/lifted_multicut/kernighan_lin.hxx`
(LiftedEdgeBuckets struct + chain_gain_init rewrite).

Do not pursue incremental maintenance — it's a strict regression.

### 4. Inspect nifty's internals — DONE 2026-05-16

Read of `lifted_twocut_kernighan_lin.hxx` confirmed nifty has no
algorithmic advantage over our pre-bucket version: same per-pair
`O(pair_size × full_adjacency)` work, same NodeMap-based difference
cache, no precomputed pair-bucket index. nifty's source is at:

```
/home/pape/Work/software/src/nifty/include/nifty/graph/opt/lifted_multicut/detail/lifted_twocut_kernighan_lin.hxx
```

Their per-entry constant factor is competitive (separate
`graph_.adjacency(v)` and `liftedGraph_.adjacency(v)` walks, no per-entry
`is_base` classification), but the algorithmic class is identical. The
~2× gap on `chain_gain_init` per entry is most plausibly the
adjacency-walk constants — which is what optimization #1 (CSR layout)
would address.

### 5. Linear-scan border instead of a heap

**Verdict: not worth pursuing for this problem.**

I worked through it: heap is faster than linear scan for our pair-size
distribution. The heap pop is O(log N) and heap.change is also O(log N);
for pair size 7 (the median) that's ~2× faster than O(N) linear scan,
and the gap widens for larger pairs.

nifty uses linear scan, but that's not where its speed advantage comes
from — likely it's the adjacency-walking constants (optimization 1).

## Workload statistics (3D problem, post greedy warm-start)

Unchanged from before; useful for sanity-checking future estimates:

- 643 clusters. Sizes: min 1, max 139, mean 3.8, median 2.
- 4443 base cluster pairs per outer iter. Pair sizes: mean 20,
  median 7, max 198 (long tail).
- Average lifted node degree: 32.

## Where to start next time

1. Re-run `cd development/graph/lifted_multicut && python check_kernighan_lin.py --size 3d --repeats 5` to confirm the baseline hasn't drifted.
2. Build with `pip install -e . --no-build-isolation -C cmake.define.BIOIMAGE_PROFILE=ON` to get the profile breakdown back.
3. Try optimization 1 (CSR adjacency) — it's the safest remaining
   lever, preserves tie-breaking, and addresses the per-entry
   constant-factor gap we measured against nifty.
4. Targets (downgraded from previous attempt — pre-bucket KL is at
   195 ms, not 200 as the original notes said):
   - 30%-of-nifty: ≤132 ms (currently 195 ms — 63 ms over).
   - Match nifty: ≤103 ms.

   The CSR change alone won't close 60+ ms. Closing the full gap
   probably requires either (a) accepting the bucket tie-breaking
   regression, or (b) finding an actually new algorithmic lever we
   haven't identified.

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
