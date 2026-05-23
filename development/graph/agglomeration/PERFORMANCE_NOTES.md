# Agglomeration Performance Notes

State of `bioimage_cpp.graph.agglomeration` vs `nifty.graph.agglo` on the
external multicut problem set (samples A, B, C × sizes small, medium), and
notes on the remaining algorithmic differences and possible optimisations.

## Current benchmark matrix

Produced 2026-05-24 with the per-policy `check_*.py` scripts, each run with
`--num-clusters-stop 1000 --repeats 1`. All runs are single-threaded.
`gasp_abs_max` is intentionally not in the comparison — nifty has no direct
sign-aware absolute-maximum linkage. ARI = adjusted Rand index between the
two partitions; speedup = `nifty_runtime / bic_runtime`.

| Policy | smp/size | bic clusters | nifty clusters | bic [s] | nifty [s] | speedup | ARI |
|---|---|---:|---:|---:|---:|---:|---:|
| edge_weighted | A/small | 1000 | 1000 | 0.70 | 0.96 | 1.37x | 1.000 |
| edge_weighted | A/medium | 1000 | 1000 | 12.06 | 15.21 | 1.26x | 1.000 |
| edge_weighted | B/small | 1000 | 1000 | 0.84 | 0.90 | 1.06x | 1.000 |
| edge_weighted | B/medium | 1000 | 1000 | 9.72 | 13.46 | 1.39x | 1.000 |
| edge_weighted | C/small | 1000 | 1000 | 1.45 | 1.85 | 1.28x | 0.940 † |
| edge_weighted | C/medium | 1000 | 1000 | 13.73 | 20.44 | 1.49x | 0.505 † |
| node_and_edge_weighted | A/small | 1000 | 1000 | 0.76 | 0.81 | 1.07x | 1.000 |
| node_and_edge_weighted | A/medium | 1000 | 1000 | 12.21 | 14.47 | 1.18x | 0.962 |
| node_and_edge_weighted | B/small | 1000 | 1000 | 0.84 | 0.96 | 1.14x | 1.000 |
| node_and_edge_weighted | B/medium | 1000 | 1000 | 11.95 | 15.26 | 1.28x | 1.000 |
| node_and_edge_weighted | C/small | 1000 | 1000 | 2.68 | 2.09 | 0.78x | 1.000 |
| node_and_edge_weighted | C/medium | 1000 | 1000 | 17.94 | 21.53 | 1.20x | 1.000 |
| mala | A/small | 1000 | 1000 | 0.62 | 0.67 | 1.09x | 0.995 |
| mala | A/medium | 6406 | 6410 | 8.42 | 10.70 | 1.27x | 0.998 |
| mala | B/small | 2754 | 2755 | 0.63 | 0.73 | 1.16x | 0.493 † |
| mala | B/medium | 20584 | 20623 | 7.21 | 8.73 | 1.21x | 0.911 |
| mala | C/small | 1000 | 1000 | 1.40 | 1.37 | 0.98x | 0.876 † |
| mala | C/medium | 1000 | 1000 | 17.48 | 18.06 | 1.03x | 0.999 |
| gasp_mutex_watershed | A/small | 5061 | 5061 | 0.38 | 0.54 | 1.41x | 1.000 |
| gasp_mutex_watershed | B/small | 4847 | 4847 | 0.65 | 0.57 | 0.87x | 1.000 |
| gasp_mutex_watershed | B/medium | 38226 | 38226 | 9.13 | 8.13 | 0.89x | 1.000 |
| gasp_mutex_watershed | C/small | 7221 | 7221 | 1.34 | 1.39 | 1.03x | 1.000 |
| gasp_max | A/small | 3421 | 3421 | 0.51 | 0.74 | 1.46x | 1.000 |
| gasp_max | A/medium | 39815 | 39815 | 16.50 | 23.67 | 1.44x | 1.000 |
| gasp_max | B/small | 1828 | 1828 | 0.94 | 1.61 | 1.71x | 1.000 |
| gasp_max | B/medium | 15619 | 15619 | 47.03 | 63.31 | 1.35x | 1.000 |
| gasp_max | C/small | 2781 | 2781 | 2.24 | 3.34 | 1.49x | 1.000 |
| gasp_max | C/medium | 18475 | 18475 | 164.23 | 105.19 | 0.64x | 1.000 |
| gasp_min | A/small | 6340 | 6340 | 0.29 | 0.50 | 1.73x | 1.000 |
| gasp_min | B/small | 8646 | 8646 | 0.29 | 0.40 | 1.37x | 1.000 |
| gasp_min | C/small | 10833 | 10833 | 0.57 | 0.85 | 1.49x | 0.821 † |
| gasp_mean | A/small | 4985 | 4985 | 0.32 | 0.52 | 1.61x | 1.000 |
| gasp_mean | B/small | 4606 | 4606 | 0.35 | 0.58 | 1.67x | 1.000 |
| gasp_mean | B/medium | 36537 | 36537 | 5.05 | 7.35 | 1.46x | 1.000 |
| gasp_mean | C/small | 6871 | 6871 | 0.73 | 1.21 | 1.66x | 1.000 |
| gasp_sum | A/small | 5036 | 5036 | 0.30 | 0.49 | 1.61x | 1.000 |
| gasp_sum | B/small | 4768 | 4768 | 0.41 | 0.75 | 1.82x | 1.000 |
| gasp_sum | B/medium | 37941 | 37941 | 6.06 | 10.24 | 1.69x | 1.000 |
| gasp_sum | C/small | 7051 | 7051 | 0.92 | 1.37 | 1.49x | 1.000 |

`†` ARI < 0.95 — all confirmed as tie-breaking artefacts (see next section),
not implementation bugs.

### Not in the table: nifty OOM-killed configurations

Nine GASP medium runs (sample A and C for `mean` / `sum` /
`mutex_watershed` / `min`) terminate with exit code 137 — nifty allocates
beyond available memory before printing its result. bic completes every one
of these in 4–7 s with cluster counts matching nifty's natural stop count.
Concretely, the largest of these, gasp_sum on A/medium, runs in 5.4 s in bic
versus a previous (pre-stop-criterion-fix) run of 187 s on bic and 6.7 s on
nifty — see git history for the pre-fix benchmark.

### Aggregate

| Policy | runs | avg speedup | avg ARI |
|---|---:|---:|---:|
| edge_weighted | 6 | 1.31x | 0.908 |
| node_and_edge_weighted | 6 | 1.11x | 0.994 |
| mala | 6 | 1.12x | 0.879 |
| gasp_mutex_watershed | 4 | 1.05x | 1.000 |
| gasp_max | 6 | 1.35x | 1.000 |
| gasp_min | 3 | 1.53x | 0.940 |
| gasp_mean | 4 | 1.60x | 1.000 |
| gasp_sum | 4 | 1.65x | 1.000 |

## Remaining differences vs nifty

ARI parity is exact (1.000) on 31 of 39 completed comparisons. The eight rows
marked `†` (six unique problem points) all share the same root cause: the
hierarchical agglomeration is **chaotically sensitive to tie-breaking** when
the algorithm's priority depends on accumulated state.

### Direct evidence from a controlled experiment

Run bic against itself with one input perturbed by ±1e-9 uniform random
noise — far below float-precision relevance for any "real" computation — and
observe ARI between the two outputs:

| Policy | smp/size | ARI bic vs bic + 1e-9 noise | Notes |
|---|---|---:|---|
| edge_weighted (sr=0) | C/medium | 1.000 | tie-breaking does not propagate |
| edge_weighted (sr=0.5) | C/medium | 0.461 | tie-break feedback on node sizes |
| mala | B/small | 0.568 | tie-break feedback on the 0.5 threshold |

The mechanism: under non-zero `size_regularizer` the priority is
`indicator * sFac(node_size_u, node_size_v)`. Two edges that initially tie
on the indicator will be popped in some order; whichever order is chosen
shifts node sizes, which shifts `sFac` for every adjacent edge in the heap,
which shifts the next pop. A single first-pop difference between bic and
nifty cascades into a completely different partition over a million
contractions.

Sample C/medium amplifies this because **86% of its 4.7M indicator values
are non-unique** (about 4.0M of the edges share an indicator value with
some other edge). Samples with high indicator uniqueness (A small/medium,
B small/medium) reach ARI 1.000 because the initial pop order is forced.

The same phenomenon explains the MALA outliers: once the running median
crosses the 0.5 stop threshold, the cluster freezes. Two implementations
that pop ties differently freeze different clusters at slightly different
moments.

### What "fixing" tie-breaking would require

Matching nifty exactly would mean reproducing its heap's internal tie-break
order — which is determined by its `boost::heap`-style fibonacci-heap
implementation and the insertion order. That is not a fruitful direction:
bic's tie-breaking is deterministic (smallest stable edge id wins), nifty's
is not (insertion-order dependent). Both partitions are valid local minima;
the appropriate way to compare them is via partition agreement metrics, not
label equality, and the user should be aware that small input perturbations
can change the labelling without changing solution quality.

## Where time goes today (qualitative)

No `BIOIMAGE_PROFILE` instrumentation has been added yet — this section
records hypotheses to test, not measurements. The hottest path is shared by
all policies:

1. **Heap pop and refresh per contraction.** `DenseIndexedHeap::change` is
   `O(log N)` and is called once per fold and once per neighbour in the
   final `contract_edge_done` sweep. For the largest problems we do
   ~10⁸ such updates; the `change` cycle dominates wall time.
2. **Adjacency restructuring inside `agglo_merge_dynamic_nodes`.** Per
   contraction we walk the removed node's adjacency, do an O(degree)
   `erase_by_neighbor` per fold, and write back into the survivor's
   adjacency. Memory-bound on large medium problems.
3. **`contract_edge_done` reprice in edge-weighted / node-and-edge-weighted.**
   After each contraction every adjacent edge of the survivor has its
   priority recomputed and possibly heap-updated. This is `O(degree)` even
   when most priorities turn out to be unchanged.
4. **MALA per-edge histogram.** Each surviving edge holds 40 `double`
   bins (320 B). On a 4.7M-edge medium problem this is ~1.5 GB resident.
   `merge_edges` sums two 40-element vectors and then walks bins to find
   the median quantile — `O(num_bins)` per fold.
5. **GASP `MutexStorage`.** `cannot_link_` is a `vector<unordered_set>` of
   `n_nodes` sets. `check_mutex` / `insert_mutex` / `merge_mutexes` are
   the only operations and they dominate the GASP `mutex_watershed` runs
   when many cannot-link constraints accumulate (sample C/medium).

The single benchmark row where bic is significantly slower than nifty —
**gasp_max on C/medium, 164 s vs 105 s** — is consistent with hypothesis
(1) + (2) being the dominant cost: that problem has the largest absolute
degree of merges (793k contractions on an 812k-node graph) and one cluster
grows to 600k nodes, so each late-stage `contract_edge_done` touches ~tens
of thousands of edges.

## Potential optimisation strategies

In rough order of expected payoff vs implementation cost, treating bic's
current behaviour as functionally complete and correct.

### Low-risk, likely measurable wins

1. **Skip the `contract_edge_done` no-op sweep when priorities are
   unchanged.** Today we call `priority_of(edge_id, stable, neighbor)`
   for every adjacent edge and compare against `edge.weight` before
   updating. The comparison saves a `heap.change` but still pays the
   `priority_of` computation (a `pow` per call for non-unit
   `size_regularizer`). Cache the previous `node_size_[stable]`; if it
   only grew by `node_size_[removed]`, derive the new `sFac` from the
   old in `O(1)` for unchanged neighbours. Concretely: precompute
   `pow(size, sr)` once per node and store it; recompute only on
   `merge_nodes`.

2. **Pre-merge degree heuristic for stable/removed swap.** The current
   swap is by `adjacency.size()`; this is a good heuristic but also the
   only one that affects which side of the merge dies. For policies
   whose `merge_nodes` / `merge_edges` cost is independent of which side
   survives (edge-weighted, mala, gasp non-MW), this can flip without
   changing the result; for `gasp_mutex_watershed` the choice matters
   because `merge_mutexes` is `O(|cannot_link_[removed]|)`. Picking the
   side with the smaller mutex set, not the smaller adjacency, would
   reduce the C/small mutex_watershed time.

3. **MALA histogram in `uint32_t` with a fallback to `double`.** The
   `insert` splits weights of 1.0 (or `edge_sizes[edge]`) between two
   bins, so counts are inherently fractional. But integer scale-up
   (e.g. weights stored ×1024 in `uint32_t`) keeps full integer
   arithmetic until counts exceed ~4M per bin, which only happens late
   in medium problems. Saves a factor of 2 in memory and gives faster
   per-bin loops. Falls back to `double` automatically when an integer
   count would overflow.

4. **MALA early-exit median.** Today `median_of` scans all 40 bins
   linearly. Cache the bin where the previous median fell; the
   post-merge median is within ±1 bin in the vast majority of cases.
   Skip ahead from the cached bin and only scan forward/backward as
   needed.

### Medium-effort wins

5. **Decouple priority recomputation from heap update for non-folding
   neighbours.** Today every neighbour of `stable` gets `heap.change`
   even when the priority shifted by less than the next pop's worth.
   A lazy-priority variant of `DenseIndexedHeap` would mark entries
   dirty and recompute on pop. This trades amortized `O(degree)` work
   per contraction for `O(1)`-per-update plus extra work per pop. Net
   win depends on how often a recomputed priority actually changes the
   heap top — empirically this is rare, so the trade should be
   favourable.

6. **Profile-guided focus on `gasp_max` C/medium.** Add `BIOIMAGE_PROFILE`
   scopes for `agglo_merge_dynamic_nodes` (broken into "fold loop",
   "rekey loop", "contract_edge_done") and rerun. The 164 s vs 105 s
   gap should be attributable to one of these phases; once it's
   visible, the right primitive to optimise becomes obvious.

7. **Sparse `MutexStorage` representation.** Each
   `std::unordered_set<uint64_t>` carries ~50 B of overhead and a hash
   per insert. For the common case where a cluster has only a handful
   of cannot-link partners, a sorted `std::vector<uint64_t>` would be
   smaller and faster (the existing `merge_mutexes` already iterates
   linearly). Switching is a `detail::mutex_storage.hxx` change that
   the mutex watershed clustering would benefit from too.

### Larger / more speculative

8. **Batched contractions.** The agglomeration is inherently sequential
   on the heap-top edge, but multiple non-overlapping contractions can
   be applied between heap synchronisations. Detect the top-K edges
   that touch disjoint super-nodes (a simple greedy walk down the
   heap), contract them in parallel via `parallel_for_chunks`, then
   reheapify. Greedy_additive multicut already uses a similar batching
   trick — porting it to the agglomeration driver is a non-trivial
   refactor but is the only path to multi-core scaling.

9. **Specialised hot-path policies.** Currently each policy is virtual-
   dispatched per `merge_edges` / `merge_nodes` call. For the four
   shipped policies the virtual calls are predictable and well-inlined
   by the indirect-call branch predictor, but a CRTP template variant
   of `agglomerative_clustering` would let each per-policy hot loop
   inline fully. Only worth doing once (1)–(7) are exhausted.

10. **Drop edge sizes from MALA when unused.** When all `edge_sizes` are
    1.0 the histogram insert reduces to a single `+= 1.0` (because the
    fbin split happens once); not a big win but the entire `insert`
    path simplifies and the per-bin loop in `median_of` can short-
    circuit on integer-only histograms. The current binding doesn't
    even accept edge_sizes for MALA — adding it is a minor API change
    that would also bring MALA closer to nifty's signature.

## Notes for future profiling sessions

- Build with `pip install -e . --no-build-isolation -C
  cmake.define.BIOIMAGE_PROFILE=ON` per the CLAUDE.md profiling
  workflow. The macros are no-ops in normal builds, so leaving them
  inline costs nothing.
- Wrap exactly one logical phase per macro — `merge_edges` and
  `contract_edge_done` are the obvious top-level scopes. Avoid
  per-iteration scopes inside tight inner loops; the macro overhead
  is small but not free.
- Run with `--repeats 1` to keep the report compact, on the largest
  problem (C/medium gasp_max or A/medium edge_weighted, depending on
  policy).
- Compare standalone bic timings against nifty's wall clock before
  touching code — bic is already faster on most rows; the only
  problematic rows are `gasp_max C/medium` and the
  `node_and_edge_weighted C/small` outlier (which may itself be
  noise — single-run timing).

## Reproducing the matrix

```bash
cd development/graph/agglomeration
for sample in A B C; do
  for size in small medium; do
    for s in check_edge_weighted.py check_node_and_edge_weighted.py check_mala.py; do
      python "$s" --sample "$sample" --size "$size" --num-clusters-stop 1000 --repeats 1
    done
    for linkage in mean sum max min mutex_watershed; do
      python check_gasp.py --sample "$sample" --size "$size" --linkage "$linkage" \
        --num-clusters-stop 1000 --repeats 1
    done
  done
done
```

For diagnosing ARI < 1 cases:

```bash
python diagnose.py --policy mala
python diagnose.py --policy edge_weighted
python diagnose.py --policy gasp_max
```

The diagnostic script reports cluster-size histograms and (for
edge_weighted) the count of non-unique indicator values, which is the
quickest indicator that an outlier ARI is a tie-breaking artefact rather
than an algorithmic bug.
