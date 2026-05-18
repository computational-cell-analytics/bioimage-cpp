# Watershed — performance notes

Optimization log for `bioimage_cpp.segmentation.watershed`. Re-run with:

```bash
python development/segmentation/check_watershed_2d.py --repeats 5
python development/segmentation/check_watershed_3d.py --repeats 3
```

Both scripts build the same heightmap from the cached ISBI affinity volume
(1 − mean of the nearest-neighbour affinity channels), generate seeds by
labelling local minima of a Gaussian-smoothed copy, then time
`bioimage_cpp.segmentation.watershed` against
`skimage.segmentation.watershed(connectivity=1)` on identical inputs with one
untimed warmup and interleaved repeats. Partition agreement is measured with
Rand Index / VI / adapted-rand-error from `elf.evaluation`. Exact label
equality is not expected — tie-breaking on equal heights is documented as
unspecified for `bioimage_cpp.watershed`.

## Setup

- CPU: 11th Gen Intel Core i7-1185G7 (Tiger Lake, 4C/8T)
- Compiler: gcc 14.3.0 (conda-forge), `-O3`, no `-march=native`
- Python 3.12.12 on Linux x86_64
- `scikit-image 0.25.2`, `numpy 1.26.4`, `bioimage_cpp 0.1.0`

## Headline numbers

Median wall-clock per call on the ISBI test volume.

| problem | shape | seeds | baseline (v1) | optimized | speedup over baseline | skimage | speedup vs skimage | Rand Index |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2D | (512, 512) | 1162 | 47.1 ms | **8.1 ms** | **5.8×** | 66.0 ms | **8.1×** | 0.982 |
| 3D | (6, 512, 512) | 2407 | 720 ms | **77.3 ms** | **9.3×** | 937 ms | **12.1×** | 0.997 |

`baseline (v1)` is the heap-based version from the initial watershed
implementation, before this optimization round. `optimized` is the
65536-bucket Meyer-flooding variant landed below. Rand Index is measured
against `skimage.segmentation.watershed` — values above 0.97 indicate
near-identical partitions with boundary jitter of a few percent.

## Optimization phases

### Baseline profile

With `BIOIMAGE_PROFILE=ON` on a single 3D run:

```
init_output            0.0003 s  (  0.0%)
seed_pass              0.0024 s  (  0.4%)
main_loop              0.6183 s  ( 99.6%)
total                  0.6210 s
```

Everything is in the heap loop. The inner loop calls
`detail::valid_offset_target` once per neighbour, which does `ndim`
divisions and modulos per call — 18 div+mod per heap pop in 3D.

### Phase 2+3 — Precomputed offsets + 2D/3D specialization

`watershed<HeightT, LabelT>` split into `detail_ws::watershed_2d` and
`detail_ws::watershed_3d` with manually-unrolled 4- / 6-neighbour blocks.
Replaced `valid_offset_target` (which does per-axis div+mod every call) with:

- one (2D) or two (3D) div+mods per heap pop to decompose `node` into
  `(y, x)` or `(z, y, x)`;
- branchless boundary checks of the form `coord[axis] > 0` /
  `coord[axis] + 1 < shape[axis]`;
- flat-index neighbour arithmetic via `node ± strides[axis]`.

Heap stays as `std::priority_queue<std::pair<HeightT, uint64_t>>`.

| problem | before | after | delta |
|---|---:|---:|---:|
| 2D | 38.8 ms | 36.1 ms | 1.07× |
| 3D | 602 ms | 484 ms | 1.24× |

3D got the bigger win because it had more div+mod to remove per pop (18 → 2)
than 2D (8 → 1). Rand Index unchanged (algorithm semantics preserved).

### Phase 4 — Smaller heap entries

Skipped. The profile showed the entire cost is in `main_loop` and the next
phase replaces the heap entirely, so the 10–20% potential from a tighter
heap entry isn't worth the binding-layer churn.

### Phase 5 — Bucket-queue Meyer flooding

Replaced `std::priority_queue` with a 65536-bucket queue. Algorithm:

1. Scan the heightmap (skipping masked pixels) to find `[h_min, h_max]`.
2. Quantize each pixel to `uint16_t` with
   `level = floor((image[i] - h_min) / (h_max - h_min) * 65535)`.
3. For each level, maintain a `std::vector<uint64_t>` of pending pixel
   indices. A `current_level` cursor advances monotonically.
4. Pop pixels from `buckets[current_level]` via `pop_back` (LIFO inside a
   level). For each unlabeled neighbour, set its label and push it into
   `buckets[max(level[neighbour], current_level)]` — the Meyer monotone
   semantic: a neighbour pushed from above never lands below the cursor.
5. Advance `current_level` when the current bucket drains.

`O(N + L)` total work where `L = 65536` levels. For `N >> L` this is
effectively `O(N)` instead of the heap's `O(N log N)`.

| problem | before (heap+specialization) | after (bucket) | delta |
|---|---:|---:|---:|
| 2D | 36.1 ms | 8.1 ms | 4.5× |
| 3D | 484 ms | 77.3 ms | 6.3× |

Post-Phase-5 3D profile:

```
init_output            0.0002 s  (  0.3%)
range                  0.0023 s  (  2.9%)
quantize               0.0020 s  (  2.5%)
seed_pass              0.0013 s  (  1.7%)
main_loop              0.0721 s  ( 92.6%)
total                  0.0779 s
```

`main_loop` is still ~93% of the time; the per-pixel work in the bucket
queue is dramatically lower than in the heap, but it remains the dominant
phase. `range`+`quantize`+`seed_pass` together are ~7% and aren't worth
chasing further at this stage — saving all of them would be ~5 ms.

## Tradeoff: partition agreement

Rand Index drops from `0.998 / 0.999` (heap) to `0.982 / 0.997` (bucket).
2D agreement dropped more because the ISBI 2D crop has small regions
(avg ~225 pixels per seed) where boundary tie-breaking dominates the
metric.

Two semantic differences cause this:

1. **Quantization.** 65536 levels can't distinguish ULP-close floats. Two
   pixels at heights `1e-6` apart end up in the same level instead of being
   strictly ordered as the heap would. Tested at 65536 levels; bumping to
   1M didn't materially improve agreement (tried during this round) but
   costs more memory, so kept at 65536.
2. **Meyer monotone flooding.** A "valley pixel" reached from a higher
   cursor is processed at the cursor's level, not its own. The heap-based
   watershed would pop it immediately and let it propagate from there. The
   bucket-queue version delays its processing until the cursor naturally
   reaches it, which can change which seed claims it.

Both are documented as unspecified tie-breaking in the Python wrapper.
The 21-test correctness suite (deterministic ridge tests, mask handling,
dtype matrix, error cases) still passes; the agreement metric vs
`skimage.segmentation.watershed` quantifies the boundary jitter on a real
problem.

The benchmark script's `min_rand_index` gate is set to 0.97 to reflect
this — below that we'd treat it as a real regression to investigate.

## What was tried and not landed

- **More quantization levels (1M).** Marginal improvement in agreement
  (0.982 → 0.985 on 2D), 4× memory cost for `levels[]`, and the bucket
  vector grows to 24 MB of empty `std::vector` shells. Not worth it.
- **`std::deque` for FIFO inside a level.** Doesn't change Meyer semantics
  (the issue isn't intra-level order). Adds per-op overhead. Discarded.
- **`uint32_t` indices in the bucket queue.** Cuts bucket storage in half
  on the inner index path; preserves the existing `uint64_t` flat-index
  semantics in the rest of the codebase. Skipped to keep the change
  minimal; revisit if a single watershed > 2³² voxels ever shows up
  (currently ruled out by memory anyway).

## How to reproduce

```bash
# Clean build (production)
pip install -e . --no-build-isolation
python -m pytest tests/segmentation/test_watershed.py -q     # 21 pass
python development/segmentation/check_watershed_2d.py --repeats 5
python development/segmentation/check_watershed_3d.py --repeats 3

# Profile build (per-phase breakdown to stderr)
pip install -e . --no-build-isolation -C cmake.define.BIOIMAGE_PROFILE=ON
python development/segmentation/check_watershed_3d.py --repeats 1
```

The profile macros (`BIOIMAGE_PROFILE_INIT/SCOPE/REPORT`) are gated by the
`BIOIMAGE_PROFILE` cmake option and are no-ops in production builds, so
they're left in `watershed.hxx` for the next round of work.
