# Filter benchmark — performance notes

This document records the P2 filter optimization from `REVIEW.md` and the
follow-up vector-math work.

Run the parity gate before the benchmark:

```bash
python development/filters/check_parity.py
python development/filters/check_parity.py --force-scalar
python development/filters/validate_eigenvalue_approximation.py
python development/filters/benchmark_eigenvalues.py
python development/filters/benchmark.py --repeats 5
python development/filters/benchmark_structure_tensor.py --repeats 5
```

The benchmark reports the median wall time across five interleaved calls after
one warmup call.

## Environment

- CPU: Intel Core i7-1185G7, 4 cores and 8 threads, AVX2 and FMA
- Compiler: conda-forge GCC 14.3.0, `-O3`, no `-march=native`
- Python 3.13.13 on Linux x86-64
- `bioimage_cpp 0.8.0`
- `fastfilters 0.3-5-ge484a99`
- `numpy 2.4.6`
- `scipy 1.17.1`
- `scikit-image 0.26.0`

All filter calls are single-threaded.

## Configuration

| Parameter | Value |
| --- | ---: |
| sigma | 1.5 |
| inner sigma | 1.0 |
| outer sigma | 2.0 |
| window size | 3.0 |
| 2D input | `camera()`, 512 × 512, float32 |
| 3D input | `cells3d()[:, 1]`, 60 × 256 × 256, float32 |

## Result

The matched `bioimage_cpp / fastfilters` geometric-mean ratio improved from
`2.194` to `1.498`. Values above one mean that bioimage-cpp is slower.

| Filter | Dim. | Before, ms | After, ms | fastfilters, ms | Final ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian smoothing | 2D | 0.81 | 0.88 | 0.60 | 1.47 |
| Gradient magnitude | 2D | 2.06 | 1.67 | 1.03 | 1.63 |
| Laplacian of Gaussian | 2D | 1.78 | 1.23 | 0.88 | 1.40 |
| Hessian eigenvalues | 2D | 3.01 | 2.77 | 1.70 | 1.63 |
| Structure-tensor eigenvalues | 2D | 5.13 | 4.11 | 3.00 | 1.37 |
| Gaussian smoothing | 3D | 36.86 | 32.24 | 14.29 | 2.26 |
| Gradient magnitude | 3D | 108.44 | 72.44 | 53.07 | 1.37 |
| Laplacian of Gaussian | 3D | 104.82 | 68.93 | 47.29 | 1.46 |
| Hessian eigenvalues | 3D | 492.18 | 207.25 | 148.64 | 1.39 |
| Structure-tensor eigenvalues | 3D | 621.41 | 293.79 | 242.84 | 1.21 |

The final geometric-mean ratios against the other references are:

| Comparison | Ratio | Cases |
| --- | ---: | ---: |
| bioimage-cpp / fastfilters | 1.498 | 10 |
| bioimage-cpp / vigra | 0.136 | 12 |
| bioimage-cpp / scipy | 0.124 | 12 |

The result does not reach the `1.2` fastfilters stretch target. The current
implementation remains about seven times faster than vigra and eight times
faster than SciPy on this matrix.

## Implemented changes

### Scratch storage

Each filter call now owns one uninitialized scratch allocation. It divides the
allocation into full-volume slots.

- Hessian uses four slots in 2D and seven slots in 3D.
- Structure tensor uses six slots in 2D and ten slots in 3D.
- The previous structure-tensor implementation used seven and eleven slots.

The implementation does not retain scratch memory between calls. Concurrent
calls do not share writable state.

### Shared 3D passes

Composite filters reuse common Z derivatives.

- Gradient magnitude and Laplacian of Gaussian use two Z passes instead of
  three.
- Hessian uses one Z pass for each derivative order, for three passes instead
  of six.
- Structure-tensor gradient calculation uses two Z passes instead of three.

This scalar stage reduced the 3D gradient-magnitude time from `108.44 ms` to
about `77.49 ms`. It reduced the 3D Laplacian time from `104.82 ms` to about
`76.34 ms`.

### Fused structure-tensor products

A fused pass forms all unique gradient products and applies the first outer
Gaussian axis in one traversal. It removes the temporary product volume.

The AVX2 version reduced the profiled 3D outer-product phase from `120.5 ms` to
`85.5 ms`. Scalar and AVX2 results differ by less than `1.3e-8` in the direct
backend comparison.

### AVX2 and FMA dispatch

The build option `BIOIMAGE_FILTERS_AVX2_DISPATCH` is enabled by default on
supported x86 builds. Only `src/cpp/filters/convolve_avx2.cxx` and
`src/cpp/filters/eigenvalues_avx2.cxx` receive AVX2 and FMA compiler flags.

Runtime dispatch checks CPU and operating-system AVX support. It falls back to
the scalar kernels in these cases:

- the CPU does not support AVX2 and FMA;
- the build target is not supported x86;
- a convolution kernel radius is above 12;
- `BIOIMAGE_CPP_FILTERS_FORCE_SCALAR` is set to a nonzero value.

The specialized kernels use unaligned loads. They share scalar mirror-border
helpers with the portable implementation.

### 3 × 3 eigenvalues

Profiling before the final eigensolver change showed:

| Filter | Convolution and products | Eigenvalues |
| --- | ---: | ---: |
| 3D Hessian | 34.7% | 65.3% |
| 3D structure tensor | 47.8% | 52.2% |

The AVX2 eigensolver processes eight symmetric matrices per batch. It reads six
structure-of-arrays component streams and writes interleaved triples. The
scalar implementation handles the tail and unsupported targets.

The vector path uses fixed float32 polynomials:

- degree-7 Chebyshev for `acos(abs(r)) / sqrt(1 - abs(r))`;
- degree-4 power polynomials for `cos(phi)` and `sin(phi) / phi`.

The `acos` implementation reflects negative inputs. The direct sine polynomial
avoids cancellation near repeated roots. The implementation clamps the
invariant to `[-1, 1]` and handles zero, isotropic, and subnormal matrices.

The approximation validator reports:

| Measurement | Maximum error |
| --- | ---: |
| `acos` absolute error | `4.223e-7` |
| `cos` absolute error | `8.511e-8` |
| `sin` absolute error | `8.277e-8` |
| random eigenvalue scaled error | `1.363e-5` |
| repeated-root scaled error | `1.982e-4` |
| AVX2 versus scalar scaled error | `2.038e-5` |

The direct benchmark uses `3,932,160` matrices. The AVX2 median is `49.871 ms`.
The scalar median is `246.645 ms`. The AVX2 path is `4.946` times faster.

A paired end-to-end benchmark captured immediately before this change used the
same environment and input:

| Filter | Before, ms | After, ms | Reduction |
| --- | ---: | ---: | ---: |
| 3D Hessian eigenvalues | 404.57 | 207.25 | 48.8% |
| 3D structure-tensor eigenvalues | 480.03 | 293.79 | 38.8% |

Both reductions exceed the 15% acceptance threshold. The unaffected 3D
smoothing, gradient-magnitude, and Laplacian times changed by at most 3.5%
relative to the preceding P2 benchmark.

The change adds no external dependency. The portable build continues to use
the scalar standard-library implementation.

## Correctness and portability

- The full parity matrix passes on scalar and AVX2 paths with the original
  tolerances.
- The automatic filter test run passes 91 tests.
- The full repository test suite passes 1378 tests.
- A build with `BIOIMAGE_FILTERS_AVX2_DISPATCH=OFF` passes 77 filter tests. It
  skips 14 AVX2-only checks.
- Tests cover vector boundaries, random scales, repeated roots, subnormal
  inputs, short image axes, radii above 12, concurrent calls, and
  scalar-to-AVX2 agreement.

## Known reference differences

- fastfilters does not support the benchmark's per-axis derivative order. The
  derivative row is excluded from the fastfilters geometric mean.
- The fastfilters Python structure-tensor wrapper swaps its inner and outer
  scales. The adapter swaps them back before comparison.
- SciPy and bioimage-cpp use mirror reflection without an edge-pixel repeat.
  Vigra and fastfilters use reflection with an edge-pixel repeat. Parity checks
  compare only the image interior.
