# Marching cubes performance

## Codex-Sol / Claude consolidation (2026-07-11)

The consolidation pass kept the Codex-Sol float32 MC33 kernel, exact
scikit-image normal/value semantics, and zero-copy NumPy handoff. It moved
axis conversion and winding into triangle emission, added robust transitive
degenerate-vertex merging with the shared `UnionFind`, and broadened the
correctness and benchmark suites. Scalar spacing is now accepted as an
isotropic convenience.

The benchmark now has a backward-compatible single-size mode and a full
scaling suite matching the independent implementation comparison:

```bash
python development/mesh/benchmark_marching_cubes.py --size medium
python development/mesh/benchmark_marching_cubes.py \
    --suite scaling --repeats 5 --batches 1 --warmup 1 --memory \
    --json /tmp/marching-cubes.json
```

The scaling suite alternates paired bioimage-cpp/scikit-image calls and covers
Lewiner spheres/scalar fields through 512³, dense 10%-foreground masks through
256³, and all three Lorensen workloads at 128³. Fresh subprocesses measure
peak RSS before the timing process allocates any large volumes.

Same-harness before/after results on the machine described below:

| case | before ms | after ms | change | scikit-image / after |
|---|---:|---:|---:|---:|
| sphere, Lewiner, 512³ | 1,898.68 | 1,786.24 | -5.9% | 1.06× |
| dense mask, Lewiner, 256³ | 2,119.26 | 2,063.38 | -2.6% | 1.98× |
| scalar field, Lewiner, 512³ | 2,268.58 | 2,295.50 | +1.2% | 1.30× |
| sphere, Lorensen, 128³ | 30.31 | 28.39 | -6.3% | 1.31× |
| dense mask, Lorensen, 128³ | 192.19 | 212.75 | +10.7% | 2.16× |
| scalar field, Lorensen, 128³ | 48.72 | 49.80 | +2.2% | 1.84× |

Absolute timings moved with CPU state: across all 15 cases the geometric-mean
wall time improved 2.4%, while normalization by each paired scikit-image time
showed a 1.1% improvement. Targeted same-session A/B repeats against the exact
pre-change commit resolved the apparent small-scalar and dense-Lorensen
regressions: scalar 64³ retained a ~2.04–2.06× paired speedup, while dense
Lorensen improved from ~2.01–2.02× to ~2.08×. No reproducible regression above
3% remained.

Peak RSS was unchanged within measurement noise:

| case | before MiB | after MiB | scikit-image MiB |
|---|---:|---:|---:|
| sphere 512³ | 725.8 | 725.7 | 734.6 |
| dense mask 256³ | 643.5 | 643.5 | 1,108.6 |
| scalar field 512³ | 703.3 | 703.4 | 843.3 |

Correctness validation finished with 29 mesh tests, 1,024 full-suite tests,
all 254 nontrivial binary cube configurations for both methods, 1,024 random
scalar cubes for both methods, and 1,000 randomized multi-cube mask/stride/
degeneracy cases. Twelve `allow_degenerate=False` cases intentionally differed
from scikit-image's negative-index remapping quirk; all returned faces were
valid and contained no collapsed-coordinate triangles.

## Two-slice face-cache optimization

`bic.mesh.marching_cubes` remains deterministic and single-threaded. The
optimization pass replaced the volume-growing edge hash map with Lewiner's
two-slice face cache. Several smaller candidates were measured first and
rejected when their gains did not survive repeated benchmarks.

## Measurement setup

- CPU: Intel Core i7-1185G7, 4 cores / 8 threads; the kernel uses one thread.
- OS: Linux 5.15 x86_64.
- Compiler/build: conda-forge GCC 14.3, editable release build (`-O3`).
- Python 3.13.13, NumPy 2.4.6, scikit-image 0.26.0.
- Workloads: a binary sphere, a reproducible 10%-foreground random binary
  mask, and a deterministic smooth scalar field.
- Medium timings are medians of three seven-call batch medians. Small timings
  use three batches of three calls; large timings use one batch of three calls.
- Every timed case passes the geometry/topology reference comparison first.

Reproduce:

```bash
python development/mesh/benchmark_marching_cubes.py --size small --repeats 3 --batches 3
python development/mesh/benchmark_marching_cubes.py --size medium --repeats 7 --batches 3
python development/mesh/benchmark_marching_cubes.py --size large --repeats 3 --batches 1
```

## Candidate evaluation

Each candidate was rebuilt, checked against the mesh tests and scikit-image
oracle, then timed twice at 96³. A candidate needed a repeatable improvement
of at least 3% without a regression above 3%.

| candidate | observed result | decision |
|---|---|---|
| Slice-sized output reserves and scalar normal appends | First run helped spheres ~3%, repeat regressed spheres 3–4% and dense Lewiner 11% | Reverted |
| Hoisted flat indexing plus one-pass hash insertion | Dense masks improved ~2.5%, but spheres repeatedly regressed 3–4% | Reverted |
| Cached corner strengths and per-cell edge indices | Mostly 1–3% changes; dense Lorensen regressed ~2% | Reverted |
| Two-slice face cache | Repeatable 17–22% sphere, 55% scalar, and 80–81% dense-mask reductions at 96³ | Retained |
| Flat cube loads after the face cache | Helped sparse Lewiner, but repeat regressed scalar Lewiner 4% and left dense masks unchanged | Reverted |

The retained cache uses two `int32` arrays with four slots per `(y, x)`
position. The upper edge layer becomes the next z-slice's lower layer; the new
upper layer is cleared. MC33 center vertices remain cell-local. Deduplication
memory is therefore `O(nx * ny)` instead of growing with the total surface.

## Profiling

Profiling used the repository's `BIOIMAGE_PROFILE` build on a 160³ Lewiner
call. The baseline's global map destruction happened after the original core
report, so it appears as the gap between core traversal/finalization and the
binding's `core_call`.

| dense-mask phase | hash-map baseline | two-slice cache |
|---|---:|---:|
| cell traversal | 2.208 s | 0.449 s |
| normal finalization | 0.009 s | 0.009 s |
| cache/map cleanup | ~0.537 s | <0.001 s |
| output orientation | 0.009 s | 0.016 s |
| measured public core/orientation total | 2.762 s | 0.475 s |

After the cache change, traversal is still 98% of the measured core work;
normalization, output orientation, cleanup, and NumPy handoff are individually
too small to justify further single-threaded complexity in this pass.

Peak RSS for a single 160³ dense-mask Lewiner call fell from 323,048 KiB to
235,760 KiB, a 27% reduction.

## Final results

`baseline / final` reports the speedup from this optimization pass.
`skimage / final` above one means bioimage-cpp is faster.

| shape | workload | method | baseline ms | final ms | baseline / final | skimage / final |
|---|---|---|---:|---:|---:|---:|
| 48³ | sphere | lewiner | 2.56 | 1.85 | 1.38× | 1.34× |
| 48³ | sphere | lorensen | 2.53 | 2.00 | 1.27× | 1.24× |
| 48³ | dense mask | lewiner | 26.76 | 10.45 | 2.56× | 2.17× |
| 48³ | dense mask | lorensen | 23.34 | 9.06 | 2.58× | 2.08× |
| 48³ | scalar field | lewiner | 10.51 | 4.45 | 2.36× | 2.13× |
| 48³ | scalar field | lorensen | 9.91 | 4.34 | 2.28× | 2.20× |
| 96³ | sphere | lewiner | 15.15 | 12.15 | 1.25× | 1.35× |
| 96³ | sphere | lorensen | 15.49 | 12.36 | 1.25× | 1.35× |
| 96³ | dense mask | lewiner | 427.71 | 85.60 | 5.00× | 2.16× |
| 96³ | dense mask | lorensen | 389.43 | 74.49 | 5.23× | 2.12× |
| 96³ | scalar field | lewiner | 52.73 | 23.71 | 2.22× | 1.85× |
| 96³ | scalar field | lorensen | 52.51 | 23.41 | 2.24× | 1.90× |
| 160³ | sphere | lewiner | 63.65 | 56.15 | 1.13× | 1.14× |
| 160³ | sphere | lorensen | 66.41 | 55.20 | 1.20× | 1.21× |
| 160³ | dense mask | lewiner | 2,824.26 | 456.11 | 6.19× | 2.10× |
| 160³ | dense mask | lorensen | 2,507.33 | 378.46 | 6.63× | 2.17× |
| 160³ | scalar field | lewiner | 247.95 | 87.80 | 2.82× | 1.64× |
| 160³ | scalar field | lorensen | 227.09 | 86.28 | 2.63× | 1.63× |

The size-dependent hash-map regression is gone. The implementation is faster
than scikit-image in every measured case without threading, SIMD, or API
changes.
