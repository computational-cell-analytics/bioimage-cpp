# Lifted Multicut Performance Notes

State of the lifted-multicut solvers vs nifty on the standard benchmark
problems and notes on remaining optimization headroom. Read this before the
next round of perf work.

## Current benchmark matrix

Produced by `python evaluate_solvers.py` (2026-05-17). All runs
single-threaded; fusion-move uses `n_seeds_fraction=0.1`,
`number_of_iterations=10`, `stop_if_no_improvement=4`. nifty side uses
matched settings via the chained-solver factory (greedy warm-start +
KL/fusion with the same iteration counts, `SEED_FROM_LOCAL`).
`runtime ratio` is `nifty_runtime / bic_runtime` — values > 1 mean bic
is faster.

| Problem | Solver | bic energy | nifty energy | Δenergy | bic runtime | nifty runtime | runtime ratio |
|---|---|---|---|---|---|---|---|
| 2D    | greedy           |    -1575.04 |    -1575.04 |  0.00    | 0.70 ms | 1.50 ms | 2.15× faster |
| 2D    | KL (10 outer)    |    -1575.21 |    -1575.21 |  0.00    | 4.14 ms | 4.77 ms | 1.15× faster |
| 2D    | fusion-move      |    -1575.43 |    -1575.43 |  0.00    | 8.31 ms | 12.3 ms | 1.48× faster |
| 3D    | greedy           |   -15891.4  |   -15891.0  | −0.35    | 6.40 ms | 15.9 ms | 2.48× faster |
| 3D    | KL (10 outer)    |   -15921.0  |   -15921.1  | +0.07    | 78.1 ms |  103 ms | 1.32× faster |
| 3D    | fusion-move      |   -15915.1  |   -15915.1  |  0.00    |  128 ms |  196 ms | 1.53× faster |
| grid  | greedy           |  -690 014   |  -690 050   | +35.9    | 16.4 s  | 20.6 s  | 1.25× faster |
| grid  | fusion-move      |  -690 271   |  -690 356   | +84.7    | 51.8 s  | 60.2 s  | 1.16× faster |

Δenergy = bic − nifty; negative means bic is better, positive means
nifty is better. Energies are exact matches on 2D/3D fusion-move and
within 0.05 % on the rest; bic is faster than nifty on every row.

KL on the 262 k-node grid is omitted from the matrix — it is correct
but takes several minutes (heavy chain-init work scales with cluster
count). See the fusion-move post-script below for the grid-specific
behavior.

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

Two C++ optimizations landed (cumulative 2.85× speedup, 245 → 86 ms):

1. **Pre-built per-node filtered adjacency** (~22% speedup, 245 → 195 ms) in
   `include/bioimage_cpp/graph/lifted_multicut/kernighan_lin.hxx`.
   `ChainScratch` gained `filtered_offset`, `filtered_count`, and
   `filtered_entries` (each entry caches `{node, weight, is_base}`).
   The chain loop iterates only in-pair neighbors and caches `was_in_heap`
   once per iteration so the three subsequent heap operations don't each
   re-query the locator.

2. **`changed_[]` cross-iteration pair-skip** (~2.27× speedup, 195 → 86 ms)
   in the outer `kernighan_lin()` driver. Mirrors nifty's
   `checkIfPartitonChanged()` / `changed_[piU] || changed_[piV]` gate. A
   pair `(A, B)` whose endpoints' node-sets are identical to the previous
   outer iter must produce the same result it did last time (zero gain —
   otherwise A or B would have changed). After iter 1, this typically
   skips >50% of pairs. New helper `detail_kl::compute_cluster_changed`
   runs in <0.2 ms per outer iter. Tie-breaking is preserved (iter 0
   processes every pair); only pairs whose runs would commit no moves are
   skipped. Same gate also applied to `cluster_splits`.

### Fusion-move

No solver-level optimizations were needed — the driver, fuse step and
sub-solver are already competitive with nifty's. The only change was a
**proposal-generator parameter-semantics fix** in
`include/bioimage_cpp/graph/proposal_generators/watershed.hxx`. Pre-fix
`n_seeds_fraction=0.1` produced 2× nifty's seed density: the loop
iterated `0.1 * N` times placing 2 seeds per iter (0.2 N total seeds),
whereas nifty iterates `nSeeds / 2` times placing 2 each (`0.1 * N`
total seeds). Effect on grid: -690099 → -690270 (closes 67 % of the
256-unit energy gap), 2D becomes an exact energy match, 3D drops 1.5
energy units (still matches nifty). Loop now runs `n_seeds / 2` times
to match nifty exactly. Documented in the header and Python docstring.

**Diagnosis of the grid energy gap (2026-05-17).** On grid, bic-fusion
finishes 85 units worse than nifty-fusion, despite bic-greedy starting
36 units *better* than nifty-greedy. The 121-unit reversal across the
fusion-move step is the thing to explain.

1. **Cross-feed test localizes 100 % of the final gap to the greedy
   warm-start, not to the fusion-move code.** Feeding nifty-greedy
   labels into bic-fusion produced -690 355.67, matching nifty-fusion's
   -690 355.63 (within float noise). bic's fusion-move algorithm
   reproduces nifty's result from the same starting state.

2. **bic-greedy and nifty-greedy land in structurally different local
   optima**, not in two equivalent tie-breaks of the same optimum. The
   reversal (bic ahead by 36 after greedy, behind by 85 after fusion)
   means bic-greedy converges to a *deeper* local minimum that the
   watershed proposals cannot escape — agreement-contraction between
   bic-greedy's labels and the proposals leaves the sub-solver no room
   to commit improvements. nifty-greedy's higher-energy optimum has
   partition boundaries that the same proposals *can* perturb, so its
   fusion-move makes much more progress per iteration.

3. **Tie density is the structural cause.** 47 % of grid base edges
   (339 946 of 718 848) carry the same weight ~+0.1. With so many
   identical priorities, greedy-additive's merge order can pick wildly
   different partition topologies. 2D and 3D have far fewer ties, so
   both sides land in similar optima and the fusion-move ratios stay
   exact.

4. **bic's existing merge direction is the right one on this problem.**
   Swapping bic's union-by-adjacency-size for nifty's union-by-rank
   regressed bic-greedy by 100 units. Reverted.

To close the gap we have to escape bic-greedy's deeper-but-rigid
optimum before invoking fusion. KL refinement does exactly this:

| Variant on grid | Energy | Runtime | Δ vs nifty energy |
|---|---|---|---|
| current (greedy + fusion 10/4) | -690 270.93 | 52 s | +85   (worse) |
| fusion 20/8 (more iters, no KL) | -690 341.37 | 87 s | +14 |
| fusion 50/15                    | -690 357.68 | 193 s | -2 |
| **greedy + KL(1) + fusion**     | **-690 392.90** | **82 s** | **-37 (better)** |
| greedy + KL(2) + fusion         | -690 539.43 | 173 s | -184 |

At equal runtime, one outer iter of KL between greedy and fusion buys
~50 more energy units than the equivalent extra fusion iterations.
KL(1) on 2D shifts the result by 0.02 units (still matches nifty
within noise), 3D improves by ~5 units, runtime cost is 10–15 % per
problem.

**Not adopted as a default.** The current `FusionMoveLiftedMulticut`
matches nifty exactly on 2D/3D and runs 1.16× faster than nifty on
grid at 0.012 % worse energy; this is acceptable for downstream
segmentation use. Users who need the better grid energy can chain
greedy → KL → fusion explicitly via `LiftedChainedSolvers`. If that
turns out to be the common workflow, a `kl_warm_refinement_iters`
parameter on `FusionMoveLiftedMulticut` would make the opt-in
self-contained.

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

## Post-script: how the `changed_[]` flag closed the gap (2026-05-16)

The bucket post-mortem (above) concluded with two recommended levers
(CSR adjacency, accept bucket tie-breaking) and the note that "nifty has
no algorithmic advantage" — both rooted in a careful read of
`lifted_twocut_kernighan_lin.hxx` (the per-pair two-cut routine).

That read missed the outer driver. `lifted_multicut_kernighan_lin.hxx`
maintains a `changed_[]` flag per partition and gates the inner two-cut
on it:

```cpp
if (!pV.empty() && (changed_[piU] || changed_[piV]))
    twoCut_.optimizeTwoCut(pU, pV, twoCutBuffers_);
```

The flag is refreshed each outer iter by `checkIfPartitonChanged()`, a
linear-time CC-style walk over base adjacency that marks a new partition
as "changed" iff its node-set differs from the previous iter's
partition that contained it (split or merge). The same gate is applied
to `introduceNewPartitions` (== our `cluster_splits`).

We added the equivalent: `detail_kl::compute_cluster_changed` (~60 lines)
plus gates in the `kernighan_lin()` driver. Result on 3D:

| Phase | Pre-flag | Post-flag |
|---|---|---|
| `pair_chains` | 167 ms | 61 ms |
| `chain_gain_init` | 78 ms | 28 ms |
| `chain_loop` | 85 ms | 31 ms |
| `cluster_splits` | 6.9 ms | 1.8 ms |
| `compute_changed` | — | 0.2 ms |
| total (profiled) | 354 ms | 135 ms |
| total (wall) | 195 ms | 86 ms |

Energy stayed at 0.07 diff (vs 0.08 pre-flag — within noise); 2D stayed
at exact 0.000 diff. Tie-breaking is preserved by construction: iter 0
has all partitions "changed" so it processes every pair (and every
split) — identical to today's algorithm. From iter 1, the only pairs
skipped are those where neither input changed since the previous iter,
where the chain is mathematically guaranteed to commit zero moves.

Why this works so well on the benchmark: the workload is 10 outer iters
on a problem that essentially converges after ~3 iters. Late iterations
were 80%+ wasted re-walking adjacency for partitions that hadn't moved.

## Future optimizations

The KL solver now beats nifty by ~17% on 3D and matches it on 2D.
Further optimization is not currently a priority. Sketched levers, in
case the workload changes:

### CSR adjacency layout (estimated: 10–15% off `pair_chains`)

`UndirectedGraph` stores adjacency as `vector<vector<Adjacency>>` — one
heap allocation per node. A flat CSR built once at the start of
`kernighan_lin` would reduce per-node cache-line misses in
`chain_gain_init` and `chain_loop`. Doesn't change algorithm output.

### Skip pair-chains where heap stays empty (estimated: <5 ms)

Already implicitly handled by `heap.empty()` check, but `chain_init`
and `chain_gain_init` still run. Track live cluster sizes incrementally
to skip earlier when one side is empty/singleton.

### Bucket gain init

Documented in the post-mortem (above). Would give ~60 ms on the
pre-flag baseline but only ~30 ms on the post-flag baseline (most of
the `chain_gain_init` time is now in iter 0 work, which buckets would
still cover). Same tie-breaking concern stands; not recommended without
a use-case that tolerates the 0.17 energy diff.

### Inspect nifty's internals — DONE 2026-05-16

Read of `lifted_twocut_kernighan_lin.hxx` confirmed nifty's per-pair
two-cut has no algorithmic advantage over ours. The outer-driver
`changed_[]` flag (in `lifted_multicut_kernighan_lin.hxx`) was the
missing piece — now implemented. Source at:

```
/home/pape/Work/software/src/nifty/include/nifty/graph/opt/lifted_multicut/lifted_multicut_kernighan_lin.hxx
/home/pape/Work/software/src/nifty/include/nifty/graph/opt/lifted_multicut/detail/lifted_twocut_kernighan_lin.hxx
```

### Linear-scan border instead of a heap

**Verdict: not worth pursuing.** Heap is faster than linear scan for
our pair-size distribution; both `pop` and `change` are O(log N) on the
addressable indexed heap, beating O(N) linear scan for the median pair
size of 7.

## Workload statistics (3D problem, post greedy warm-start)

Unchanged from before; useful for sanity-checking future estimates:

- 643 clusters. Sizes: min 1, max 139, mean 3.8, median 2.
- 4443 base cluster pairs per outer iter. Pair sizes: mean 20,
  median 7, max 198 (long tail).
- Average lifted node degree: 32.

## Where to start next time

1. Re-run `cd development/graph/lifted_multicut && python check_kernighan_lin.py --size 3d --repeats 5` to confirm the baseline hasn't drifted.
2. Build with `pip install -e . --no-build-isolation -C cmake.define.BIOIMAGE_PROFILE=ON` for the profile breakdown.
3. The next-most-attractive lever is CSR adjacency layout (10–15%),
   but only worth doing if a heavier workload reveals the need.

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
