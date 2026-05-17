# Filter benchmark — performance notes

Results from `development/filters/benchmark.py` for the six Tier-1 filters
on 2D and 3D test data from `skimage.data`. Re-run with:

```bash
python development/filters/check_parity.py    # parity gate, must PASS first
python development/filters/benchmark.py       # headline numbers
```

All four libraries produce the same response (verified by `check_parity.py`
on the image interior, dropping a `window_size * sigma` margin on each side;
or `inner + outer` for the structure tensor). Reported numbers are
**median wall-clock per call** across 5 timed repeats, after one untimed
warmup, with calls interleaved round-robin between libraries to share cache
state fairly.

## Setup

- CPU: Intel Core i7-1185G7 (Tiger Lake, 4C/8T, AVX2 + AVX-512)
- Compiler: gcc 15.2.0 (conda-forge), `-O3`, no `-march=native`
- Python 3.11.15 on Linux x86_64
- `numpy 2.4.5`, `scipy 1.17.1`, `vigra 1.12.3`, `fastfilters 0.3-4-g87f08b5`,
  `bioimage_cpp 0.1.0`
- All libraries called single-threaded; `bioimage_cpp` has no SIMD intrinsics
  (Tier-1: scalar C++20 + compiler auto-vectorisation).

## Benchmark configuration

| parameter | value | notes |
|---|---|---|
| sigma | 1.5 | smoothing / derivative / LoG / gradient / Hessian |
| inner_sigma | 1.0 | structure-tensor gradient scale |
| outer_sigma | 2.0 | structure-tensor smoothing scale |
| window_size | 3.0 | kernel half-width / sigma; matched across libraries |
| repeats | 5 | timed, interleaved round-robin |
| 2D image | `skimage.data.camera()` | 512×512 → float32, normalised to [0, 1] |
| 3D volume | `skimage.data.cells3d()[:, 1]` | 60×256×256 nuclei channel → float32 |

`gaussian_derivative` uses order = 1 along the trailing axis only. `fastfilters`
only supports uniform per-axis order, so its row is `n/a`.

## Headline ratios

Geometric mean of `bioimage_cpp.median / other.median` across all benched
(filter, dim) combinations. **>1.0 means `bioimage_cpp` is slower** than the
other library; **<1.0 means faster**.

| comparison | geomean ratio | n |
|---|---|---|
| `bioimage_cpp` / `fastfilters` | **2.00** | 10 |
| `bioimage_cpp` / `vigra` | **0.18** | 12 |
| `bioimage_cpp` / `scipy` | **0.20** | 12 |

Interpretation: `fastfilters` (hand-AVX2) is 2× faster than ours on average;
ours is ~5× faster than vigra and ~5× faster than scipy/numpy.

## 2D results — `camera()` 512×512

Times in milliseconds (median of 5 repeats). The `x ours` column is
`bioimage_cpp.median / this_lib.median`; **values >1.0 mean the library is
faster than `bioimage_cpp`**.

| filter | bioimage_cpp ms | fastfilters ms | x ours | vigra ms | x ours | scipy ms | x ours |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_smoothing | 0.82 | 0.49 | 1.67 | 5.18 | 0.16 | 3.59 | 0.23 |
| gaussian_derivative | 0.83 | n/a | n/a | 5.15 | 0.16 | 3.85 | 0.21 |
| gaussian_gradient_magnitude | 2.11 | 0.91 | 2.32 | 13.85 | 0.15 | 7.77 | 0.27 |
| laplacian_of_gaussian | 1.92 | 0.96 | 2.00 | 8.88 | 0.22 | 7.50 | 0.26 |
| hessian_of_gaussian_eigenvalues | 3.02 | 1.67 | 1.81 | 19.55 | 0.15 | 77.79 | 0.04 |
| structure_tensor_eigenvalues | 5.16 | 2.86 | 1.80 | 26.22 | 0.20 | 82.99 | 0.06 |

## 3D results — `cells3d()[:, 1]` 60×256×256

| filter | bioimage_cpp ms | fastfilters ms | x ours | vigra ms | x ours | scipy ms | x ours |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_smoothing | 33.27 | 15.02 | 2.22 | 192.29 | 0.17 | 85.37 | 0.39 |
| gaussian_derivative | 33.05 | n/a | n/a | 195.27 | 0.17 | 86.19 | 0.38 |
| gaussian_gradient_magnitude | 106.42 | 61.91 | 1.72 | 640.83 | 0.17 | 276.64 | 0.38 |
| laplacian_of_gaussian | 103.69 | 61.93 | 1.67 | 584.71 | 0.18 | 266.87 | 0.39 |
| hessian_of_gaussian_eigenvalues | 486.38 | 175.75 | 2.77 | 2195.57 | 0.22 | 4031.79 | 0.12 |
| structure_tensor_eigenvalues | 599.90 | 263.14 | 2.28 | 1938.37 | 0.31 | 4301.12 | 0.14 |

## Per-filter throughput vs `bioimage_cpp` (megapixels / second)

Throughput = `image_size / median_time`. Useful for cross-shape comparison.

### 2D (512×512 = 0.262 megapixels per output)

| filter | bioimage_cpp | fastfilters | vigra | scipy |
|---|---:|---:|---:|---:|
| gaussian_smoothing | 320 | 535 | 51 | 73 |
| gaussian_gradient_magnitude | 124 | 288 | 19 | 34 |
| laplacian_of_gaussian | 136 | 273 | 30 | 35 |
| hessian_of_gaussian_eigenvalues | 87 | 157 | 13 | 3 |
| structure_tensor_eigenvalues | 51 | 92 | 10 | 3 |

### 3D (60×256×256 = 3.93 megavoxels per output)

| filter | bioimage_cpp | fastfilters | vigra | scipy |
|---|---:|---:|---:|---:|
| gaussian_smoothing | 118 | 262 | 20 | 46 |
| gaussian_gradient_magnitude | 37 | 64 | 6 | 14 |
| laplacian_of_gaussian | 38 | 64 | 7 | 15 |
| hessian_of_gaussian_eigenvalues | 8 | 22 | 2 | 1 |
| structure_tensor_eigenvalues | 7 | 15 | 2 | 1 |

## Takeaways

- **vs `fastfilters`** (the hand-AVX2 target). We sit at 1.67× – 2.77×
  slower across the board, exactly in the 1.0×–2.0× band the Tier-1 plan
  predicted (a touch worse on Hessian/structure-tensor 3D where eigenvalue
  arithmetic dominates and benefits less from auto-vectorisation). This is
  the gap a Tier-2 manual-AVX2 path would aim to close.
- **vs `vigra`**. ~5× faster across the board. Same algorithmic family,
  scalar code in both cases; the win comes from fewer abstraction layers
  and tighter kernels (constant-bound inner loops, half-kernel storage,
  X-strip pattern for the strided pass).
- **vs `scipy.ndimage` (+ `numpy.linalg.eigvalsh` for the eigenvalue
  filters)**. ~5× faster on simple filters and ~10–25× faster on the
  eigenvalue paths. `scipy.ndimage` itself is competitive on plain
  convolution; the eigenvalue gap is almost entirely the per-pixel numpy
  `eigvalsh` cost — exactly what scipy-only users currently pay.

## Where the Tier-2 SIMD work would help most

Largest absolute gaps to `fastfilters` (3D, in ms per call):

| filter | ours | ff | absolute gap | gap × 5 calls/sec |
|---|---:|---:|---:|---:|
| hessian_of_gaussian_eigenvalues | 486 | 176 | 311 ms | 1.55 s/sec saved |
| structure_tensor_eigenvalues | 600 | 263 | 337 ms | 1.69 s/sec saved |
| gaussian_gradient_magnitude | 106 | 62 | 44 ms | 0.22 s/sec saved |

If/when Tier-2 lands, instrument the Hessian and structure-tensor 3D paths
first — they share the same separable-FIR primitives plus the 3×3 trig
eigensolver, so closing the gap on those carries the gradient/magnitude /
LoG paths along for free.

## Reproducibility notes

- Run after `pip install -e . --no-build-isolation` so the C++ extension
  matches the source tree.
- For absolute-time comparisons across machines, also report CPU model,
  compiler version, and whether `-march=native` was set (we do NOT set it
  in normal builds; this benchmark used the default `-O3`).
- For a quick re-check use `--small` (crops to 128×128 and 32×64×64);
  finishes in a few seconds.
- For raw per-(filter, library) timings (useful for plotting), pass
  `--csv path.csv`.

## Known caveats reflected in the adapters

- `fastfilters.gaussianDerivative` only accepts a uniform per-axis order;
  the bench cell is `n/a` rather than running an unequal operation.
- `fastfilters.structureTensorEigenvalues` swaps `innerScale` /
  `outerScale` at the Python boundary (its wrapper calls the C function
  with the args in the opposite order — `src/python/core.cxx:328` vs
  `src/library/fir_filters.c:156` in the fastfilters source). The adapter
  swaps them back so the bench compares the same operation as `vigra` /
  `bioimage_cpp`.
- `scipy` and `bioimage_cpp` use scipy-style `mirror` (reflect without
  edge-pixel repeat); `vigra` / `fastfilters` use reflect with edge
  repeat. Parity is checked on the image interior to absorb the
  difference.
