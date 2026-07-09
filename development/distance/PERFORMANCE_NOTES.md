# Geodesic FMM performance notes

Optimization of the two geodesic Fast Marching Method solvers, with the
before/after measurements that justify the changes. Both changes are pure
reorganizations of identical arithmetic — the solved fields are **bitwise
identical** to the pre-optimization solvers on every code path tested.

## What changed

- **Lead A — mask/grid solver** (`distance/detail/fast_marching.hxx`): the FMM
  march no longer calls `detail::valid_offset_target` (one integer division +
  modulo *per axis, per call*, ~42 calls / popped voxel in 3D). Each popped voxel
  is decoded **once** with the new `detail::coords_from_index` (divide-and-subtract,
  no modulo); axis neighbours are reached with a single `p ± strides_[d]` add and an
  integer bounds compare, and the neighbour's coordinates are handed to
  `solve_eikonal` so it never re-decodes. The `offsets_` `vector<vector>` is gone.
- **Lead B — mesh solver** (`distance/detail/mesh_fast_marching.hxx`): when a
  vertex `v` freezes, each incident face relaxes its other corners from *that face
  alone* instead of calling `update_vertex`, which rescanned every incident face of
  every touched vertex. `dist_[w]` is a running minimum that already reflects the
  other faces' contributions (applied when their corners froze), so the single-face
  update reaches the same value. `update_vertex` was removed. Operands are passed to
  `triangle_update` in face-storage order so the acute update is bit-identical, not
  just equal to within rounding.

## Measurement setup

8-core machine, `powersave` governor (frequency-scaling noise present; `perf`
unavailable, `perf_event_paranoid=4`). Mitigations: 1 warmup call + 20 timed
repeats, **min** used as the headline estimator (least-interference), spread
reported. Single-source **field** solve, `--threads 1` — the single-source solve is
the whole kernel (pairwise = N × field). Baseline and optimized were built
separately; the **`mesh/field` case is a null control for Lead A** (its code is
untouched between the baseline and A-only builds) and moved <1%, bounding machine
drift below the measured mask gains. Lead A's mask numbers also reproduced within
~1% across two independent builds (A-only and A+B).

Reproduce (from `development/distance/`):

```
python benchmark_geodesic.py --large  --only field --threads 1 --repeats 20 --no-ref --json base.json
python benchmark_geodesic.py --xlarge --only field --threads 1 --repeats 20 --no-ref --json opt.json
python /path/to/compare.py base.json opt.json      # or diff the JSON directly
```

## Wall-clock: baseline → optimized (single-thread field, min of 20)

| case | baseline min | optimized min | speedup |
|---|---|---|---|
| mask2d/field (1024²)      |  294.0 ms |  208.9 ms | **1.41×** |
| mask2d/field (1536²)      |  685.6 ms |  492.7 ms | **1.39×** |
| mask3d/field (64·128²)    |  745.1 ms |  370.2 ms | **2.01×** |
| mask3d/field (128³)       | 1811.6 ms |  912.4 ms | **1.99×** |
| mesh/field (V=20000)      |   23.2 ms |    7.9 ms | **2.94×** |
| mesh/field (V=40000)      |   49.1 ms |   17.5 ms | **2.81×** |

Lead A: ~2.0× in 3D (matches the div/mod arithmetic dominating 3 axes), ~1.4× in
2D. Lead B: ~2.8–2.9× on the mesh. Both exceed the pre-work predictions (A: 1.5–2×
3D; B: ~2×).

## Mechanism confirmation (BIOIMAGE_PROFILE=ON, one field solve)

Coarse phases in `solve()`: `pop` (heap pop, untouched by either change) vs `relax`
(the neighbour/face loop where the removed work lived).

| solve | phase | baseline | optimized | change |
|---|---|---|---|---|
| mask 128³ | pop   | 0.319 s | 0.316 s | unchanged |
| mask 128³ | relax | 1.700 s | 0.762 s | **2.23× smaller** |
| mesh 40k  | pop   | 0.0035 s | 0.0034 s | unchanged |
| mesh 40k  | relax | 0.0500 s | 0.0163 s | **3.07× smaller** |

The heap phase is invariant; the targeted `relax` phase shrank by 2.2× (mask) and
3.1× (mesh). `pop`'s *share* rose (15.8%→29.3% mask; 6.5%→17.1% mesh) only because
`relax` shrank around it — the gain came from exactly the phase each change targeted.

## Versus reference (single-thread field, min of 10)

| case | bioimage-cpp | reference | speedup |
|---|---|---|---|
| mask2d/field (1024²)   | 210.5 ms | scikit-fmm 331.8 ms | **1.58×** |
| mask3d/field (64·128²) | 368.9 ms | scikit-fmm 761.4 ms | **2.06×** |
| mesh/field (V=20000)   |   8.4 ms | pygeodesic 1745 ms  | 208× (apples-to-oranges: pygeodesic is *exact* MMP, ours is first-order) |

The mask solver went from parity with scikit-fmm's compiled C (pre-work ~1.0–1.1×)
to a clear lead (1.58× in 2D, 2.06× in 3D).

## Correctness (all green on the shipping build)

- `pytest tests/distance/ -q` — 64 passed. The tight external cross-checks ran
  (not skipped): `test_mask_matches_scikit_fmm` (residual < 1e-6),
  `test_mesh_matches_pygeodesic` (mean rel < 0.06, p95 < 0.12).
- `python check_geodesic_distance.py` — all 6 cases OK, exit 0 (mask residuals
  ~1e-12 vs scikit-fmm; mesh rel mean 0.027).
- **Bitwise equivalence**: 13 solver outputs (2D/3D field, anisotropic `sampling`,
  `speed`, gradient, mask + mesh pairwise, mesh `speed`) are exactly equal
  (`np.array_equal`) between the baseline and optimized builds. Determinism verified
  (repeat solves identical). This is the dev-time check behind the "bitwise
  identical" claim; the pytest suite guards correctness against the external
  references on every run.

The profiler instrumentation (`BIOIMAGE_PROFILE_SCOPE` "pop"/"relax") is left in both
solvers; it is a no-op unless built with `-C cmake.define.BIOIMAGE_PROFILE=ON`.
