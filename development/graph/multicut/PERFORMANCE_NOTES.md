# Multicut Performance Notes

State of the multicut solvers vs nifty on the standard benchmark problems and
notes on remaining optimization headroom. Read this before the next round of
perf work.

## Current benchmark matrix

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

## Current read

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
