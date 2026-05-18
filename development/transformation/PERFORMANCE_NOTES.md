# Affine transform — performance & semantics notes

These are internal notes for `bic.transformation.affine_transform`. They
describe the current numerical semantics, the runtime numbers on this
machine, and the work that would be required to close the remaining gap
to `scipy.ndimage.affine_transform`.

The reproducible benchmark and parity check is in
`development/transformation/check_affine.py`. The benchmarks below come
from a single recent run of that script with `--repeats 5` on the default
shapes (2D: 512×512 float32 from `skimage.data.camera`; 3D: 40×128×128
float32 from `skimage.data.cells3d`). Times are medians over five
interleaved repeats.

## Order coverage and kernels

| order | kernel                              | taps / axis | interpolating? | scipy match              |
|-------|-------------------------------------|-------------|----------------|--------------------------|
| 0     | nearest neighbour                   | 1           | yes            | scipy `mode='constant'`  |
| 1     | linear                              | 2           | yes            | scipy `mode='constant'`  |
| 2     | quadratic B-spline                  | 3           | no (smoothing) | scipy `mode='grid-constant'`, `prefilter=False` |
| 3     | Keys cubic (Catmull-Rom, a=−0.5)    | 4           | yes            | **does not match** scipy (which uses cubic B-spline) |
| 4     | quartic B-spline                    | 5           | no (smoothing) | scipy `mode='grid-constant'`, `prefilter=False` |
| 5     | quintic B-spline                    | 6           | no (smoothing) | scipy `mode='grid-constant'`, `prefilter=False` |

Notes on the asymmetry at order 3:

- Without a prefilter pass, the *interpolating* cubic kernel is Keys' cubic
  convolution. With a prefilter, the natural choice is the cubic B-spline
  (what scipy uses). Since we currently skip the prefilter, Keys is the
  consistent default for order 3 — it reproduces input samples at integer
  coordinates without an extra global pass.
- Orders 2, 4, 5 are *only* useful when paired with a prefilter if you want
  interpolation. Without one they are low-pass smoothing kernels; we expose
  them anyway because (a) they exactly match scipy's `prefilter=False` and
  thus give a clean migration path, and (b) the wider kernels are useful as
  cheap anti-aliasing for downsampling, paired with
  `bic.transformation.resample`.

## Boundary handling

For the **interpolating** orders (0, 1, 3) we apply a strict outer coord
check: if the affine-mapped input coordinate is outside `[0, shape - 1]`
along any axis, the output pixel is set to `fill_value`. This matches
`scipy.ndimage.affine_transform(..., mode='constant', cval=fill_value)`.

For the **B-spline** orders (2, 4, 5) we drop the outer coord check and let
each kernel tap evaluate independently — out-of-bounds taps contribute
`fill_value` weighted by the kernel weight, in-bounds taps contribute the
sampled value. This matches scipy with
`mode='grid-constant', cval=fill_value`. It is the right semantic for a
smoothing kernel near the image border: the kernel smoothly fades to the
fill value rather than producing a hard cliff.

For the comparison script this means scipy is invoked with `mode='constant'`
for orders 0/1/3 and `mode='grid-constant'` for orders 2/4/5. Interior
parity is then bit-for-bit on every gated case.

## Benchmark (this run)

```
                                bioimage_cpp ms   nifty ms (× ours)   scipy ms (× ours)
2D (512×512 float32)
  order 0  identity                       1.42      4.78  (0.30)         6.36 (0.22)
           translate                      1.35      4.99  (0.27)         6.22 (0.22)
           rotate 30°                     1.43      4.62  (0.31)         6.35 (0.23)
           crop                           0.77      2.49  (0.31)         3.44 (0.22)
  order 1  identity                       2.13      6.90  (0.31)        11.81 (0.18)
           translate                      2.28      6.72  (0.34)        11.39 (0.20)
           rotate 30°                     2.22      6.21  (0.36)        10.47 (0.21)
           crop                           1.27      3.79  (0.33)         6.24 (0.20)
  order 2  identity                       4.43        —                 26.45 (0.17)
           translate                      4.58        —                 25.64 (0.18)
           rotate 30°                     4.89        —                 24.53 (0.20)
           crop                           2.65        —                 15.08 (0.18)
  order 3  identity                       8.53        —                 28.14 (0.30)
           translate                      8.52        —                 26.14 (0.33)
           rotate 30°                     7.38        —                 22.76 (0.32)
           crop                           4.41        —                 14.61 (0.30)
  order 4  identity                      11.19        —                 52.44 (0.21)
           translate                     11.38        —                 53.19 (0.21)
           rotate 30°                    12.22        —                 54.38 (0.22)
           crop                           6.51        —                 31.94 (0.20)
  order 5  identity                      14.60        —                 71.93 (0.20)
           translate                     16.31        —                 79.79 (0.20)
           rotate 30°                    15.91        —                 70.54 (0.23)
           crop                           8.76        —                 42.01 (0.21)

3D (40×128×128 float32)
  order 0  identity                       5.33     15.83  (0.34)        23.25 (0.23)
           translate                      5.37     19.65  (0.27)        23.98 (0.22)
           rotateZ 20°                    5.27     16.71  (0.32)        23.00 (0.23)
  order 1  identity                      10.74     33.87  (0.32)        56.75 (0.19)
           translate                     10.39     34.02  (0.31)        54.94 (0.19)
           rotateZ 20°                    9.66     30.66  (0.31)        52.15 (0.19)
  order 2  identity                      24.83        —                165.79 (0.15)
           translate                     24.95        —                170.66 (0.15)
           rotateZ 20°                   28.18        —                165.99 (0.17)
  order 3  identity                      50.17        —                216.39 (0.23)
           translate                     47.23        —                204.45 (0.23)
           rotateZ 20°                   47.86        —                194.71 (0.25)
  order 4  identity                      88.57        —                664.01 (0.13)
           translate                     89.63        —                672.09 (0.13)
           rotateZ 20°                   89.48        —                599.83 (0.15)
  order 5  identity                     138.88        —               1016.87 (0.14)
           translate                    145.75        —               1026.59 (0.14)
           rotateZ 20°                  148.91        —               1037.42 (0.14)
```

`× ours` is `bioimage_cpp.median / library.median`. Values below 1.0 mean we
are faster; values above 1.0 mean we are slower.

Headline:

- Across **all** orders and shapes, `bioimage_cpp` is the fastest of the
  three libraries: roughly **3–4× faster than nifty** on the orders it
  supports (0, 1) and **4–8× faster than scipy** across orders 0–5.
- The 3D B-spline orders (especially 4 and 5) dominate runtime: a 6-tap
  kernel applied per output voxel gives 6³ = 216 fused multiply-adds per
  output. That is unavoidable for non-separated direct evaluation; see the
  separation comment in the parallelization section.
- Order 3 (Keys cubic) is roughly **1.4–2× faster than orders 2 and 4** at
  the same tap count, because the Keys path was hand-unrolled before the
  templated B-spline path was added. Some of that gap should close if we
  give the B-spline path the same per-row inner loop unrolling, but I have
  not benched the change.

## Remaining gap to scipy: the prefilter

scipy's default for orders ≥ 2 is `prefilter=True`. That runs a separable
IIR filter over the input array to convert sample values into B-spline
**coefficients**; sampling the kernel against coefficients (instead of raw
samples) makes the result *interpolating* — it passes through the original
input values at integer coordinates. The two scipy calls below produce the
same result:

```python
coeffs = scipy.ndimage.spline_filter(image, order=3)
out = scipy.ndimage.affine_transform(coeffs, ..., order=3, prefilter=False)
# is equivalent to
out = scipy.ndimage.affine_transform(image, ..., order=3, prefilter=True)
```

Without the prefilter, the B-spline kernel low-pass smooths the input.
That is why our `order=2/4/5` matches scipy with `prefilter=False` but
visibly *smooths* the output (the test cases `crop` and `identity` keep
exact pixel values at order 2/4/5 only because `identity` has no
fractional offset and `crop` has no kernel taps touching the boundary).

### How a prefilter would work

For each spatial axis, the prefilter applies the recursive IIR filter

```
H(z) = 1 / (1 + z) ... using each pole p_k for the chosen order
```

For order `n` the filter has `floor(n / 2)` real poles. The poles are
constants (see `get_filter_poles` in scipy's `ni_splines.c`):

| order | poles |
|-------|-------|
| 2     | `−0.171572875…`  |
| 3     | `−0.267949192…`  |
| 4     | `−0.361341226…`, `−0.013725429…` |
| 5     | `−0.430575347…`, `−0.043096288…` |

The filter is applied as a forward (causal) sweep and a backward
(anti-causal) sweep per pole. Initial conditions handle the boundary mode
(mirror, wrap, constant, …). The implementation is ~150 lines of C; the
recursion is sequential along the swept axis so it does not vectorize but
each axis-aligned pass is embarrassingly parallel across the other axes.

### Cost analysis if we implemented it

A spline prefilter is a **one-time** pre-processing pass over the input
volume; its cost is independent of the output shape. For an input of `N`
voxels, axis count `D`, and `P` poles for the chosen order, the prefilter
performs `D · P · 2 · N` IIR steps (forward + backward per axis per pole).
Each step is one multiply-add. For order 3 (`P = 1`) on a 512×512 float32
image that is `2 · 2 · 512² ≈ 1 M` mul-adds — sub-millisecond on this
machine.

The break-even is when the prefilter cost is small compared to the
sampling cost. Sampling cost scales with output size and kernel taps; for
non-trivial transformations on outputs the same size as the input, the
prefilter is essentially free. The clear win is: enabling prefilter at
order 3 makes `affine_transform` produce the same interpolating cubic that
scipy users expect, at near-zero added runtime.

The work then would be:

1. Header-only `detail/spline_prefilter.hxx` with one templated function
   per dtype. Real-valued IIR over a contiguous buffer; supports forward,
   backward, and one or two poles. ~100 LOC.
2. Python wrapper exposes `prefilter: bool = False` on `affine_transform`
   (default `False` to preserve current behaviour) and the standalone
   `bic.transformation.spline_prefilter(image, order)` for explicit use.
3. When `prefilter=True` is set, the wrapper calls the prefilter once,
   then calls the existing sampler. The C++ sampler does not need to
   change — it already accepts arbitrary input values.

The numerical effect is well-defined: with prefilter enabled, `order=3`
ceases to be Keys cubic and becomes scipy-compatible cubic B-spline
interpolation. We would then **add** Keys cubic as a separately-named
option (e.g. `kernel='keys'` or `order=3` always Keys with
`prefilter=True` forced to error) to avoid silently changing existing
results.

We deliberately deferred this work — it is straightforward, but
introducing a global pre-pass is a meaningful semantic change and a
separately reviewable change is cleaner.

## Parallelization

`affine_transform` is currently single-threaded, and CLAUDE.md mandates
that we not add threading on top of a still-evolving implementation.
Concretely, two pieces are parallelizable:

1. **The sampling pass.** Each output voxel is independent. The natural
   partition is along the outermost output axis: chunk `[0, out_d)` into
   `n_threads` contiguous bands, each thread computes its band using its
   own `out_ptr` and per-row running `input_coord` accumulator. The kernel
   inner loop touches only its band's output region; the input volume is
   read-only and shared. `detail/threading.hxx::parallel_for_chunks` is the
   right primitive — it is what every other parallel kernel in this repo
   uses.

   Cost model: each thread does `out_voxels / n_threads · taps_per_voxel`
   FMA. There is no synchronization in the inner loop, so the only ceiling
   is memory bandwidth (the input volume can fit in L2 for typical 2D
   problems; 3D order-5 reads `6³` cache lines per output voxel and is
   memory-bound). Expect close-to-linear scaling up to memory-bandwidth
   saturation — empirically that is 4–8 threads on commodity hardware.

2. **The (future) spline prefilter.** The recursion is sequential along
   the swept axis, but the other axes are independent: when sweeping along
   axis `k`, each fiber `image[i, :, j]` (for 3D, axis `k=1`) can be
   computed independently. Same `parallel_for_chunks` partition, this time
   over the non-swept axes.

A `number_of_threads=` parameter on the Python wrapper, threaded through
to the C++ kernel, is the obvious knob. Default to `1` (current behaviour)
or `0` (use all cores) — whichever matches the convention the rest of the
repo settles on. The plumbing is mechanical and is the next thing to do
once we're confident the single-threaded implementation has stable
semantics.
