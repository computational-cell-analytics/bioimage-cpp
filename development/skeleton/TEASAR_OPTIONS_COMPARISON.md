# TEASAR option comparison on the example MRC mask

This report compares the new bioimage-cpp TEASAR settings with Kimimaro 5.8.1.
It uses the example mask and centered crop from the initial skeleton diagnostic.

## Summary

- Ball invalidation moves the bioimage-cpp topology closer to Kimimaro.
- `fix_branching=False` reduces degree-3 nodes and short terminal spurs in both
  implementations.
- Bioimage-cpp ball modes are 1.9--2.0x faster than Kimimaro with one worker.
  They are approximately 3.5x faster with eight workers on this crop.
- Coalesced invalidation and a guarded shared distance transform reduced the
  one-worker ball time by 52--53% without changing the measured skeletons.
- The default bioimage-cpp behavior remains cube invalidation with
  `fix_branching=True`.

The outputs do not match exactly. Bioimage-cpp and Kimimaro still differ in
component handling, root selection, distance fields, and path tie-breaking.

## Measurement setup

Measured on 2026-08-04.

The input is `examples/skeleton/00004_gt_mask.mrc`. The source shape is
`(324, 1251, 1251)`. The benchmark uses the centered crop with origin
`(62, 525, 525)`, shape `(200, 200, 200)`, and 585,875 foreground voxels.

The parameters are:

| parameter | cube modes | ball modes and Kimimaro |
| --- | ---: | ---: |
| spacing | 10 Å | 10 Å |
| `scale` | 0 | 0 |
| `constant` | 70 Å | 140 Å |
| `pdrf_scale` | 100,000 | 100,000 |
| `pdrf_exponent` | 4 | 4 |

These constants reproduce the working radii from the initial investigation.
The cube and ball timing comparison therefore includes both a geometry change
and a radius change. The bioimage-cpp ball and Kimimaro rows use identical
TEASAR parameters.

Kimimaro runs with soma handling disabled. Border fixing, hole filling, dust
filtering, and progress reporting are also disabled. `fix` means
`fix_branching=True`. `parental` means `fix_branching=False`.

The host has an Intel Core i7-1185G7 with four cores and eight hardware threads.
The environment uses Python 3.13.13, NumPy 2.4.6, bioimage-cpp 0.8.0,
Kimimaro 5.8.1, and EDT 3.1.1.

Reproduce the measurement from the repository root:

```bash
python development/skeleton/benchmark_teasar_mrc.py \
    --crop-size 200 \
    --pixel-size 10 \
    --scale 0 \
    --constant 70 \
    --ball-constant 140 \
    --all-settings \
    --kimimaro \
    --threads 1 8 \
    --repeats 7 \
    --warmup 2 \
    --json /tmp/teasar_options_mrc.json
```

## Skeleton comparison

The table reports raw skeletons before tick removal. A spur is a degree-3 node
with a terminal arm of at most 200 Å. A real junction has three arms longer
than 200 Å. Percentages use the number of degree-3 nodes as the denominator.

| implementation and setting | constant | vertices | components | length | degree 1 | degree 3 | degree 4 | spurs | real junctions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bioimage-cpp cube/fix | 70 Å | 11,850 | 72 | 16.723 µm | 340 | 198 | 0 | 110 (55.6%) | 22 (11.1%) |
| bioimage-cpp cube/parental | 70 Å | 11,875 | 72 | 16.727 µm | 293 | 147 | 2 | 58 (39.5%) | 22 (15.0%) |
| bioimage-cpp ball/fix | 140 Å | 11,256 | 72 | 15.857 µm | 270 | 128 | 0 | 50 (39.1%) | 26 (20.3%) |
| bioimage-cpp ball/parental | 140 Å | 11,399 | 72 | 16.039 µm | 258 | 110 | 3 | 31 (28.2%) | 25 (22.7%) |
| Kimimaro ball/fix | 140 Å | 11,328 | 71 | 16.308 µm | 283 | 141 | 0 | 67 (47.5%) | 21 (14.9%) |
| Kimimaro ball/parental | 140 Å | 11,507 | 71 | 16.505 µm | 268 | 126 | 0 | 44 (34.9%) | 24 (19.0%) |

Ball invalidation has the largest topology effect in bioimage-cpp. Relative to
the default cube/fix mode, ball/fix reduces degree-3 nodes from 198 to 128 and
spurs from 110 to 50. The number of real junctions increases from 22 to 26.

The fixed parental field reduces branching further. It changes the ball result
from 128 to 110 degree-3 nodes and from 50 to 31 spurs. The real-junction count
remains similar at 25. The same direction appears in Kimimaro, where the
parental mode changes 141 to 126 degree-3 nodes and 67 to 44 spurs.

The bioimage-cpp ball modes produce fewer degree-3 nodes and spurs than their
matched Kimimaro modes. They also introduce degree-4 nodes when the parental
field is active. These metrics describe topology only; they do not establish
which skeleton is more accurate.

## Performance comparison

Each backend receives two warmups. The benchmark then measures seven calls in
a deterministically shuffled order. The table reports the median. MRC loading,
crop extraction, and graph-statistic calculation are outside the timed region.

| implementation and setting | one worker | eight workers | speedup |
| --- | ---: | ---: | ---: |
| bioimage-cpp cube/fix | 0.713 s | 0.219 s | 3.26x |
| bioimage-cpp cube/parental | 0.692 s | 0.210 s | 3.30x |
| bioimage-cpp ball/fix | 0.944 s | 0.272 s | 3.47x |
| bioimage-cpp ball/parental | 0.911 s | 0.276 s | 3.30x |
| Kimimaro ball/fix | 1.833 s | 0.946 s | 1.94x |
| Kimimaro ball/parental | 1.817 s | 0.958 s | 1.90x |

Bioimage-cpp is 1.94x faster than Kimimaro for ball/fix with one worker and
3.48x faster with eight workers. The corresponding ball/parental factors are
1.99x and 3.47x.

The parental field reduces the one-worker bioimage-cpp time by approximately
3%. It does not provide a consistent gain with eight workers on this crop.
The saved path work is small relative to the shared distance transform and
component dispatch.

Ball invalidation remains more expensive than cube invalidation. It is 32%
slower for fix mode with one worker and 24% slower with eight workers. This is
not a primitive-level comparison. The ball uses twice the constant radius and
produces a different set of paths.

### Optimization stages

The baseline is the implementation measured before this optimization pass.
The queue stage keeps only the best pending invalidation distance for each
compact foreground voxel. The final stage also shares one guarded distance
transform across ordinary binary components when its memory and volume tests
pass.

| ball setting | workers | baseline | coalesced queue | final | final change |
| --- | ---: | ---: | ---: | ---: | ---: |
| fix | 1 | 1.985 s | 1.121 s | 0.944 s | -52.4% |
| fix | 8 | 0.518 s | 0.306 s | 0.272 s | -47.5% |
| parental | 1 | 1.945 s | 1.096 s | 0.911 s | -53.1% |
| parental | 8 | 0.516 s | 0.311 s | 0.276 s | -46.6% |

The queue change accounts for a 39.8--43.7% reduction from the baseline. The
shared distance transform gives a further 11.2--16.8% reduction relative to
the queue stage. The measured vertices, edges, and radii did not change across
these stages.

### Profile and memory

A one-worker profile of ball/fix selected the shared distance transform and
estimated 53.9 MB of scratch memory. The main phase totals were:

| phase | time |
| --- | ---: |
| shared distance-transform setup | 21.8 ms |
| shared distance transform | 151.4 ms |
| compact-domain construction | 27.0 ms |
| root Dijkstra | 283.6 ms |
| rail-path Dijkstra | 167.4 ms |
| invalidation | 273.8 ms |
| measured TEASAR total | 980.9 ms |

The invalidation queue received 5,976,824 offers and accepted 693,630 of them.
It rejected 88.4% of offers before heap insertion. Of the accepted entries,
107,755 became stale before removal. The peak heap size was 18,059 entries.
The invalidation phase fell from approximately 1.074 s to 273.8 ms in the
profile build. The shared distance transform reduced the corresponding local
component-transform total from approximately 423 ms to 151 ms.

Fresh worker processes measured incremental process-tree peak RSS. Each value
is the median of three processes after imports and input allocation:

| implementation and setting | one worker | eight workers |
| --- | ---: | ---: |
| bioimage-cpp ball/fix | 52.7 MiB | 77.1 MiB |
| bioimage-cpp ball/parental | 52.7 MiB | 77.1 MiB |
| Kimimaro ball/fix | 89.8 MiB | 1,575 MiB |
| Kimimaro ball/parental | 89.8 MiB | 1,555 MiB |

The shared transform raises the one-worker ball/fix peak from the 42.5 MiB
baseline to 52.7 MiB. It reduces the eight-worker peak from 129.8 MiB to
77.1 MiB because workers no longer retain separate component transforms.
The automatic strategy uses local transforms when the shared volume or the
estimated scratch memory exceeds its guard.

The results cover one crop on one host. They do not predict full-volume runtime
or performance on masks with a different component-size distribution.

## Later queue evaluation

A follow-up tested 4-ary and radix queues for compact Dijkstra and an indexed
queue for ball invalidation. None met the retention gates, so the production
algorithm and the results in this report remain unchanged.

The Dijkstra candidates changed a physical field by at most 1.9% and regressed
the parental field. The indexed invalidation queue removed 107,755 stale
entries on ball/fix, but one-worker MRC time changed from 974.9 ms to 982.0 ms.
Ball/parental changed from 956.6 ms to 975.8 ms. See
`PERFORMANCE_NOTES.md` for the complete queue, synthetic, and memory results.
