# Multicut Performance Notes

State of the multicut solvers vs nifty on the standard benchmark problems and
notes on remaining optimization headroom. Read this before the next round of
perf work.

## Shared contraction topology

Greedy additive and greedy fixation now use
`graph::detail::BasicContractionTopology`. An edge payload keeps each weight
and constraint flag next to its endpoints. Mutation observers update the heap
and union-find without an intermediate event buffer. The solver workspace
retains all scratch allocations across calls.

Profile builds report reset, initialization, contraction, and label
materialization separately.

The refactor keeps the previous merge direction: the larger-degree endpoint
survives, and the edge's first endpoint survives a degree tie.

## Kernighan-Lin optimization

The P1 optimization ran on 2026-07-27. The acceptance run used a production
build, one thread, five repeats, and alternating backend order:

```bash
python development/graph/multicut/evaluate_solvers.py \
    --solvers kernighan_lin \
    --problems A_small B_small C_small A_medium B_medium C_medium \
    --n-repeats 5 --backend both --require-reference nifty \
    --results-jsonl \
    development/graph/multicut/benchmark_results/kl_p1_final_accepted.jsonl \
    --build-command 'pip install -e . --no-build-isolation'
```

The acceptance criteria were:

- Each bioimage-cpp median must be at most 1.03 times its baseline median.
- `B_medium` must be no slower than nifty.
- Every bioimage-cpp repeat must reproduce its baseline energy and label
  SHA-256 digest.

Run the executable gate with:

```bash
python development/graph/multicut/check_benchmark_acceptance.py \
    development/graph/multicut/benchmark_results/kl_p1_baseline.jsonl \
    development/graph/multicut/benchmark_results/kl_p1_final_accepted.jsonl \
    --reference-parity B_medium
```

All six problems passed. This includes `A_medium` and `C_medium`.

| Problem | Baseline bic | Final bic | Final / baseline | Final nifty | Nifty / bic | Exact output |
|---|---:|---:|---:|---:|---:|---|
| A_small | 1.974 s | 1.691 s | 0.857 | 2.533 s | 1.498 | yes |
| B_small | 4.343 s | 3.759 s | 0.865 | 5.308 s | 1.412 | yes |
| C_small | 6.496 s | 6.126 s | 0.943 | 7.856 s | 1.282 | yes |
| A_medium | 125.723 s | 60.070 s | 0.478 | 134.766 s | 2.243 | yes |
| B_medium | 275.021 s | 156.324 s | 0.568 | 165.150 s | 1.056 | yes |
| C_medium | 220.694 s | 130.206 s | 0.590 | 234.075 s | 1.798 | yes |

The implementation skips pair chains and split checks for clusters that did
not change in the previous iteration. The final configured iteration always
runs all checks. This final pass preserves the previous fixed output.

Each chain stores only adjacency entries whose other endpoint is in the active
cluster pair. The move loop reuses this filtered adjacency. It also derives
heap membership from the maintained cross-border count and resets scratch
state only for nodes that the chain touched.

The ordinary and lifted implementations now share the cluster-change helper in
`detail/relabel.hxx`. Phase instrumentation uses `detail/profile.hxx` and has no
production-build cost.

The `B_medium` profile changed as follows:

| Scope | Before | After |
|---|---:|---:|
| End-to-end bic runtime | 267.927 s | 153.626 s |
| `pair_chains` | 257.111 s | 143.886 s |
| `chain_gain_init` | 110.342 s | 84.403 s |
| `chain_loop` | 134.652 s | 54.715 s |
| `chain_cleanup` | 7.520 s | 1.918 s |

Profile scopes are nested and must not be summed. The retained evidence is in:

- `benchmark_results/kl_p1_baseline.jsonl`
- `benchmark_results/kl_p1_final_accepted.jsonl`
- `benchmark_results/kl_p1_final_accepted.txt`
- `benchmark_results/kl_p1_profile_before.jsonl`
- `benchmark_results/kl_p1_profile_before.txt`
- `benchmark_results/kl_p1_profile_accepted.jsonl`
- `benchmark_results/kl_p1_profile_accepted.txt`

The benchmark harness now records raw runtimes, medians, minima, energies,
label digests, commands, build commands, source state, package paths, compiler
details, CPU details, and thread-related environment variables. It loads nifty
only when requested and can require the reference with
`--require-reference nifty`.

## Previous benchmark matrix

Produced by `python evaluate_solvers.py` (2026-05-17). Small problems were run
with both implementations in one pass:

```bash
python evaluate_solvers.py --problems A_small B_small C_small \
    --results-jsonl benchmark_results/small_both.jsonl
```

Medium problems were run in two passes so completed rows were preserved even if
a long nifty row had to be stopped:

```bash
python evaluate_solvers.py --problems A_medium B_medium C_medium \
    --backend bic --results-jsonl benchmark_results/medium_bic.jsonl
python evaluate_solvers.py --problems A_medium B_medium C_medium \
    --backend nifty --results-jsonl benchmark_results/medium_nifty.jsonl
```

All runs are single-threaded with `n_repeats=1`. `KernighanLinMulticut` and the
KL stage inside `ChainedMulticutSolvers` use 5 outer iterations. Fusion-move
uses watershed proposals, `numberOfIterations=10`, `stopIfNoImprovement=4`, and
a greedy-additive fusion sub-solver on the nifty side. `runtime ratio` is
`nifty_runtime / bic_runtime` - values reported as "faster" mean bic is faster.

| Problem | Solver | bic energy | nifty energy | Δenergy | bic runtime | nifty runtime | runtime ratio |
|---|---|---|---|---|---|---|---|
| A_small | greedy_additive | -76 914.52 | -76 914.52 | 0.00 | 0.28 s | 0.38 s | 1.32x faster |
| A_small | kernighan_lin | -76 916.06 | -76 916.06 | 0.00 | 2.04 s | 2.54 s | 1.24x faster |
| A_small | greedy_fixation | -76 914.13 | -76 914.13 | 0.00 | 0.28 s | 1.96 s | 7.09x faster |
| A_small | chained | -76 916.06 | -76 916.06 | 0.00 | 2.02 s | 2.56 s | 1.27x faster |
| A_small | decomposer | -76 914.52 | -76 914.52 | 0.00 | 0.26 s | 0.34 s | 1.30x faster |
| A_small | fusion_move | -76 915.29 | -76 915.29 | 0.00 | 1.60 s | 2.35 s | 1.46x faster |
| B_small | greedy_additive | -437 001.4 | -437 001.4 | 0.00 | 0.32 s | 0.44 s | 1.36x faster |
| B_small | kernighan_lin | -437 023.6 | -437 023.6 | 0.00 | 4.40 s | 5.32 s | 1.21x faster |
| B_small | greedy_fixation | -436 943.8 | -436 943.8 | 0.00 | 0.34 s | 1.86 s | 5.43x faster |
| B_small | chained | -437 023.6 | -437 023.6 | 0.00 | 4.36 s | 5.27 s | 1.21x faster |
| B_small | decomposer | -437 001.4 | -437 001.4 | 0.00 | 0.32 s | 0.49 s | 1.50x faster |
| B_small | fusion_move | -437 034.1 | -437 034.1 | 0.00 | 1.60 s | 2.62 s | 1.64x faster |
| C_small | greedy_additive | -24 189.93 | -24 189.93 | 0.00 | 0.74 s | 0.87 s | 1.17x faster |
| C_small | kernighan_lin | -24 191.91 | -24 191.95 | +0.03698 | 6.52 s | 7.41 s | 1.14x faster |
| C_small | greedy_fixation | -24 162.43 | -24 162.43 | 0.00 | 0.76 s | 2.79 s | 3.66x faster |
| C_small | chained | -24 191.91 | -24 191.95 | +0.03698 | 6.52 s | 7.33 s | 1.12x faster |
| C_small | decomposer | -24 189.93 | -24 189.93 | 0.00 | 0.74 s | 0.91 s | 1.23x faster |
| C_small | fusion_move | -24 191.45 | -24 191.45 | 0.00 | 4.90 s | 7.42 s | 1.52x faster |
| A_medium | greedy_additive | -535 228.2 | -535 228.2 | 0.00 | 6.21 s | 6.72 s | 1.08x faster |
| A_medium | kernighan_lin | -535 251.2 | -535 251.3 | +0.0821 | 124.75 s | 137.72 s | 1.10x faster |
| A_medium | greedy_fixation | -535 214.6 | -535 214.6 | 0.00 | 6.16 s | 25.26 s | 4.10x faster |
| A_medium | chained | -535 251.2 | -535 251.3 | +0.0821 | 122.41 s | 134.68 s | 1.10x faster |
| A_medium | decomposer | -535 228.2 | -535 228.2 | 0.00 | 6.15 s | 5.81 s | 1.06x slower |
| A_medium | fusion_move | -535 235.3 | -535 235.3 | 0.00 | 51.34 s | 65.61 s | 1.28x faster |
| B_medium | greedy_additive | -2 141 349 | -2 141 349 | 0.00 | 5.71 s | 6.78 s | 1.19x faster |
| B_medium | kernighan_lin | -2 141 705 | -2 141 700 | -5.246 | 263.46 s | 157.91 s | 1.67x slower |
| B_medium | greedy_fixation | -2 141 097 | -2 141 097 | 0.00 | 5.63 s | 24.22 s | 4.30x faster |
| B_medium | chained | -2 141 705 | -2 141 700 | -5.246 | 259.35 s | 156.60 s | 1.66x slower |
| B_medium | decomposer | -2 141 349 | -2 141 349 | 0.00 | 5.62 s | 7.00 s | 1.25x faster |
| B_medium | fusion_move | -2 141 572 | -2 141 572 | 0.00 | 43.82 s | 67.12 s | 1.53x faster |
| C_medium | greedy_additive | -90 732.51 | -90 732.51 | 0.00 | 11.05 s | 11.53 s | 1.04x faster |
| C_medium | kernighan_lin | -90 768.01 | -90 768.56 | +0.5491 | 219.36 s | 240.26 s | 1.10x faster |
| C_medium | greedy_fixation | -90 631.67 | -90 631.67 | 0.00 | 11.11 s | 30.28 s | 2.73x faster |
| C_medium | chained | -90 768.01 | -90 768.56 | +0.5491 | 218.77 s | 234.48 s | 1.07x faster |
| C_medium | decomposer | -90 732.51 | -90 732.51 | 0.00 | 10.84 s | 12.12 s | 1.12x faster |
| C_medium | fusion_move | -90 746.68 | -90 746.68 | 0.00 | 58.34 s | 86.42 s | 1.48x faster |

Δenergy = bic - nifty; negative means bic found the lower-energy labeling,
positive means nifty did. The only nonzero differences are KL/chained rows, and
they are tiny relative to the objective scale. Fusion-move, greedy-additive,
greedy-fixation, and decomposer match nifty energies on every benchmark row.

The decomposer rows above are legacy measurements. The old bioimage-cpp
configuration used `GreedyAdditiveMulticut` without a fallthrough solver, so
it selected the documented greedy-additive fast path and did not run
decomposition. The benchmark now supplies matching greedy-additive sub- and
fallthrough solvers to both implementations, passes `number_of_threads`, and
requires multiple non-singleton positive-cost components. Do not use the
legacy decomposer runtimes for performance comparisons.

### Corrected decomposer smoke benchmark

The corrected benchmark ran on `A_small` on 2026-07-26. Each row used one
repeat. Both implementations used greedy-additive sub- and fallthrough
solvers. The benchmark verified that the graph had at least two non-singleton
positive-cost components.

| Threads | bic energy | nifty energy | bic runtime | nifty runtime |
|---|---|---|---|---|
| 1 | -76 914.5 | -76 914.5 | 0.469 s | 0.335 s |
| 4 | -76 914.5 | -76 914.5 | 0.429 s | 0.378 s |

These single-repeat rows confirm matched objective values and exercise the
real decomposer path. They are smoke measurements, not stable performance
estimates.

### Native decomposer benchmark

The decomposer moved from Python orchestration to C++ on 2026-07-26. Matched
`A_small` runs used five repeats with greedy-additive sub- and fallthrough
solvers.

| Threads | bic energy | nifty energy | bic runtime | nifty runtime |
|---|---|---|---|---|
| 1 | -76 914.5 | -76 914.5 | 0.240 s | 0.354 s |
| 4 | -76 914.5 | -76 914.5 | 0.221 s | 0.359 s |

The native implementation matched the reference energy. It was 1.48 times
faster than nifty with one thread and 1.62 times faster with four threads on
this problem.

## Previous read

`KernighanLinMulticut` is the dominant runtime target. It is faster than nifty
on most rows, but `B_medium` is the clear exception: bic KL/chained takes about
260 s vs nifty's 157 s. Because `ChainedMulticutSolvers` is greedy-additive +
KL, it inherits the same behavior.

Fusion-move is not the immediate bottleneck. It is faster than nifty on every
row in the matrix, including all medium problems, with exact energy matches.

Greedy fixation is substantially faster than nifty, and greedy additive plus
the decomposer are already close enough that further work there is unlikely to
move end-to-end runtimes unless a downstream workflow uses only those solvers.

Raw intermediate rows are in `benchmark_results/*.jsonl`; the `.md` files next
to them contain the direct stdout tables from each run.
