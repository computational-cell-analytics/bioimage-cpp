# Migration Guide

This guide explains how to migrate code that used selected `nifty` or
`affogato` functionality to `bioimage-cpp`.

`bioimage-cpp` is not a drop-in compatibility layer. The package keeps a
smaller, NumPy-first API with Python-style method names, explicit dtype support,
and no I/O dependencies in the C++ core.

## Imports

Use:

```python
import bioimage_cpp as bic
```

Graph functionality is under `bic.graph`, segmentation functionality is under
`bic.segmentation`, ground-truth comparison functionality is under
`bic.ground_truth`, and small utility functions are under `bic.utils`.

## Blocking

Nifty:

```python
import nifty.tools as nt

blocking = nt.blocking([0, 0], [100, 80], [32, 32])
block = blocking.getBlock(0)
block_with_halo = blocking.getBlockWithHalo(0, [8, 8])
```

bioimage-cpp:

```python
import bioimage_cpp as bic

blocking = bic.Blocking([0, 0], [100, 80], [32, 32])
block = blocking.get_block(0)
block_with_halo = blocking.get_block_with_halo(0, [8, 8])
```

Name changes:

| nifty-style name | bioimage-cpp name |
| --- | --- |
| `roiBegin` | `roi_begin` |
| `roiEnd` | `roi_end` |
| `blockShape` | `block_shape` |
| `blockShift` | `block_shift` |
| `blocksPerAxis` | `blocks_per_axis` |
| `numberOfBlocks` | `number_of_blocks` |
| `blockGridPosition` | `block_grid_position` |
| `getNeighborId` | `get_neighbor_id` |
| `getBlock` | `get_block` |
| `getBlockWithHalo` | `get_block_with_halo` |
| `addHalo` | `add_halo` |
| `coordinatesToBlockId` | `coordinates_to_block_id` |
| `getBlockIdsInBoundingBox` | `get_block_ids_in_bounding_box` |
| `getBlockIdsOverlappingBoundingBox` | `get_block_ids_overlapping_bounding_box` |
| `getLocalOverlaps` | `get_local_overlaps` |
| `getBlockIdsInSlice` | `get_block_ids_in_slice` |

Intentional improvements over nifty:

- `coordinates_to_block_id` accounts for both `roi_begin` and `block_shift`.
- `get_block_ids_overlapping_bounding_box` works for any dimensionality, not
  only 3D.
- Bounding boxes use NumPy-style half-open intervals: `[begin, end)`.
- `get_local_overlaps` returns `None` if blocks do not overlap; otherwise it
  returns `(begin_a, end_a, begin_b, end_b)` in local coordinates.

## Undirected Graphs

Nifty:

```python
import nifty.graph as ng

graph = ng.undirectedGraph(4)
edge_id = graph.insertEdge(0, 1)
uvs = graph.uvIds()
```

bioimage-cpp:

```python
import bioimage_cpp as bic

graph = bic.graph.UndirectedGraph(4)
edge_id = graph.insert_edge(0, 1)
uvs = graph.uv_ids()
```

The convenience constructor is:

```python
graph = bic.graph.undirected_graph(4)
graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [1, 2]])
```

Important differences:

- Nodes are fixed at construction and have ids `0 .. number_of_nodes - 1`.
- Re-inserting an existing undirected edge returns the existing edge id.
- Bulk methods accept array-like inputs and return NumPy arrays.
- Python-style names are preferred. A few nifty-style aliases are still present
  on `UndirectedGraph` for convenience, but new code should use snake_case.

Common method/property mapping:

| nifty-style name | bioimage-cpp name |
| --- | --- |
| `numberOfNodes` | `number_of_nodes` |
| `numberOfEdges` | `number_of_edges` |
| `nodeIdUpperBound` | `node_id_upper_bound` |
| `edgeIdUpperBound` | `edge_id_upper_bound` |
| `insertEdge` | `insert_edge` |
| `insertEdges` | `insert_edges` |
| `findEdge` | `find_edge` |
| `findEdges` | `find_edges` |
| `uvIds` | `uv_ids` |
| `nodeAdjacency` | `node_adjacency` |
| `serializationSize` | `serialization_size` |
| `extractSubgraphFromNodes` | `extract_subgraph_from_nodes` |
| `edgesFromNodeList` | `edges_from_node_list` |

## Region Adjacency Graphs

Nifty:

```python
import nifty.graph.rag as nrag

rag = nrag.gridRag(labels)
uvs = rag.uvIds()
```

bioimage-cpp:

```python
import bioimage_cpp as bic

rag = bic.graph.region_adjacency_graph(labels)
uvs = rag.uv_ids()
```

Notes:

- Supported label dtypes are `uint32`, `uint64`, `int32`, and `int64`.
- Labels must be 2D or 3D.
- Negative signed labels are rejected.
- Nodes correspond to label ids from `0` to `labels.max()`.
- Edge ids are deterministic; RAG edges are sorted lexicographically by
  endpoint ids.
- Non-contiguous labels are copied to contiguous memory before entering C++.

## RAG Boundary and Affinity Features

Nifty has RAG feature helpers such as `accumulateEdgeMeanAndLength`,
`accumulateEdgeStandartFeatures`, and affinity feature accumulation helpers.
In `bioimage-cpp`, these are exposed as explicit NumPy-returning functions.

Simple edge-map features:

```python
rag = bic.graph.region_adjacency_graph(labels)
features = bic.graph.edge_map_features(rag, labels, edge_map)
```

The columns are:

```python
bic.graph.SIMPLE_EDGE_FEATURE_NAMES
# ("mean", "size")
```

Complex edge-map features:

```python
features = bic.graph.edge_map_features_complex(rag, labels, edge_map)
```

The columns are:

```python
bic.graph.COMPLEX_EDGE_FEATURE_NAMES
# ("mean", "median", "std", "min", "max", "p5", "p10",
#  "p25", "p75", "p90", "p95", "size")
```

Affinity features:

```python
features = bic.graph.affinity_features(
    rag,
    labels,
    affinities,
    offsets=[[0, 1], [1, 0]],
)
```

Complex affinity features:

```python
features = bic.graph.affinity_features_complex(
    rag,
    labels,
    affinities,
    offsets=[[0, 1], [1, 0]],
)
```

Notes:

- `edge_map` must have the same shape as `labels`.
- `affinities` must have shape `(channels, *labels.shape)`.
- `offsets` must have one offset per channel in NumPy axis order.
- Feature arrays use `float64` output.
- `number_of_threads=0` uses the library default; pass a positive integer for a
  fixed thread count.

## Projecting RAG Node Labels to Pixels

Nifty projects scalar node data back to pixels with
`projectScalarNodeDataToPixels`.

Nifty:

```python
import nifty.graph.rag as nrag

rag = nrag.gridRag(labels)
pixel_labels = nrag.projectScalarNodeDataToPixels(rag, node_labels)
```

bioimage-cpp:

```python
rag = bic.graph.region_adjacency_graph(labels)
pixel_labels = bic.graph.project_node_labels_to_pixels(
    rag,
    labels,
    node_labels,
)
```

Notes:

- `labels` must be the over-segmentation used to construct `rag`.
- `node_labels` must be a 1D array with length `rag.number_of_nodes`.
- The output has the same shape as `labels` and dtype `uint64`.
- `number_of_threads=0` uses the library default; pass a positive integer for a
  fixed thread count.

## Multicut

Nifty exposes multicut through an objective + factory-style solver hierarchy.
`bioimage-cpp` uses an explicit `MulticutObjective` and a `MulticutSolver` class
hierarchy with a single `optimize(objective)` entry point.

Nifty:

```python
import nifty.graph.opt.multicut as nmc

objective = nmc.multicutObjective(graph, edge_costs)
solver = objective.greedyAdditiveFactory().create(objective)
labels = solver.optimize()
energy = objective.evalNodeLabels(labels)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

objective = bic.graph.MulticutObjective(graph, edge_costs)
labels = bic.graph.GreedyAdditiveMulticut().optimize(objective)
energy = objective.energy(labels)
```

`MulticutObjective` accepts an `UndirectedGraph` or a `RegionAdjacencyGraph` and
a 1D `edge_costs` array of length `graph.number_of_edges`. The objective owns
the current best `labels`; `optimize` updates them in place and also returns
the new array.

Available solvers:

| nifty factory | bioimage-cpp solver |
| --- | --- |
| `greedyAdditiveFactory()` | `GreedyAdditiveMulticut()` |
| `greedyFixationFactory()` | `GreedyFixationMulticut()` |
| `kernighanLinFactory(...)` | `KernighanLinMulticut(...)` |
| `chainedSolversFactory([...])` | `ChainedMulticutSolvers([...])` |
| `multicutDecomposer(submodelFactory=...)` | `MulticutDecomposer(sub_solver=...)` |

Constructor argument mapping:

| nifty argument | bioimage-cpp argument |
| --- | --- |
| `weightStop` | `weight_stop` |
| `nodeNumStop` | `node_num_stop` |
| `addNoise` | `add_noise` |
| `numberOfOuterIterations` | `number_of_outer_iterations` |
| `numberOfInnerIterations` | `number_of_inner_iterations` |
| `epsilon` | `epsilon` |
| `submodelFactory` | `sub_solver` |
| `fallthroughFactory` | `fallthrough_solver` |
| `numberOfThreads` | `number_of_threads` |

Kernighan-Lin example:

```python
solver = bic.graph.KernighanLinMulticut(number_of_outer_iterations=5)
labels = solver.optimize(objective)
```

If the objective's labels are left at the default (one cluster per node),
`KernighanLinMulticut` warm-starts from a greedy-additive solution
internally, matching `kernighanLinFactory(warmStartGreedy=True)`. To skip the
warm-start, set `objective.set_labels(...)` to a non-trivial labeling first.

Chaining solvers:

```python
solver = bic.graph.ChainedMulticutSolvers([
    bic.graph.GreedyAdditiveMulticut(),
    bic.graph.KernighanLinMulticut(number_of_outer_iterations=5),
])
labels = solver.optimize(objective)
```

Decomposing a problem into positive-cost connected components and solving each
sub-problem with a cheaper solver:

```python
solver = bic.graph.MulticutDecomposer(
    sub_solver=bic.graph.KernighanLinMulticut(number_of_outer_iterations=5),
    fallthrough_solver=bic.graph.GreedyAdditiveMulticut(),
    number_of_threads=0,
)
labels = solver.optimize(objective)
```

Notes:

- `edge_costs` must be `float64` and 1D with length `graph.number_of_edges`.
- Output labels are dense `uint64` ids in `0 .. number_of_clusters - 1`.
- `MulticutObjective.energy(labels)` is the multicut energy used internally; it
  matches `nmc.multicutObjective(...).evalNodeLabels(labels)`.
- `objective.reset_labels()` restores the per-node initial labeling, useful when
  re-running solvers from a clean state.

Intentional differences vs. nifty:

- Solvers are plain Python classes — no `factory().create(objective)` step.
- Solver arguments use snake_case and are keyword-only where appropriate.
- `KernighanLinMulticut` runs a border-restricted move chain plus an explicit
  cluster-split phase, matching nifty's local optima on the standard multicut
  benchmark while being noticeably faster.
- `MulticutDecomposer` short-circuits the trivial case where the sub-solver is
  `GreedyAdditiveMulticut` and no fallthrough is given — the greedy solver
  already operates on each connected component internally.

## Segmentation Overlaps

Nifty:

```python
import nifty.ground_truth as ngt

overlap = ngt.overlap(segmentation, ground_truth)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

overlap = bic.ground_truth.segmentation_overlap(segmentation, ground_truth)
```

The first input is called `labels_a` and the second input is called `labels_b`
in the `bioimage-cpp` API. Use named structured tables instead of positional
arrays:

```python
table = overlap.overlap_table()
# fields: "label_a", "label_b", "count"

table = overlap.overlap_table(normalize_by="a")
# fields: "label_a", "label_b", "count", "fraction"

overlaps = overlap.overlaps_for_label_a(12, normalize=True)
# fields: "label", "count", "fraction"

best = overlap.best_overlap_for_label_a(12, ignore_zero=True)
# BestOverlap(label=..., count=..., fraction=..., found=...)
```

Other common queries:

```python
overlap.labels_a
overlap.labels_b
overlap.count_a(12)
overlap.count_b(4)
overlap.overlap_count(12, 4)
overlap.counts_a_table()
overlap.counts_b_table()
overlap.best_overlap_for_label_b(4)
overlap.is_label_a_overlapping_with_zero(12)
overlap.different_overlap(12, 13)
```

Intentional improvements over nifty:

- Labels are stored sparsely, so large sparse label ids do not require a dense
  vector up to `max_label + 1`.
- The Python API returns structured arrays with named fields and a
  `BestOverlap` dataclass instead of ambiguous positional arrays.
- Both overlap directions are supported explicitly:
  `overlaps_for_label_a(...)` and `overlaps_for_label_b(...)`.
- Normalization is explicit via `normalize_by="a"`, `"b"`, or `"total"`.
- Missing labels return count `0`; best-overlap queries expose `found=False`.

Notes:

- Inputs must be integer arrays with identical shape.
- Signed integer inputs must not contain negative labels.
- Inputs are converted to contiguous `uint64` arrays before entering C++.

## Mutex Watershed

Affogato:

```python
from affogato.segmentation import compute_mws_segmentation

seg = compute_mws_segmentation(
    weights,
    offsets,
    number_of_attractive_channels=3,
    strides=[1, 1, 1],
)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

seg = bic.segmentation.mutex_watershed(
    weights,
    offsets,
    number_of_attractive_channels=3,
    strides=[1, 1, 1],
)
```

Important migration notes:

- `bioimage-cpp` expects the first `number_of_attractive_channels` channels to
  be attractive merge-edge weights and the remaining channels to be mutex-edge
  weights.
- Supported affinity dtypes are `float32` and `float64`.
- Inputs must represent 2D or 3D grids with shapes `(channels, y, x)` or
  `(channels, z, y, x)`.
- Non-contiguous affinity arrays are copied to contiguous memory.
- `strides` sub-sample mutex edges only; attractive edges are always kept.
- `randomized_strides=True` uses NumPy's global random state, so existing
  `np.random.seed(...)` workflows remain deterministic.
- A boolean `mask` may be passed. Edges touching `False` pixels are ignored and
  masked pixels are set to label `0`.
- Output labels are `uint64`, consecutive, and 1-based for foreground pixels.

## Dictionary-Based Relabeling

If you used a small helper to apply a dictionary to an integer label array, use
`take_dict`:

```python
labels = np.array([1, 3, 2, 1], dtype=np.uint64)
relabeling = {1: 10, 2: 20, 3: 30}

out = bic.utils.take_dict(relabeling, labels)
```

Notes:

- Supported input dtypes are `uint32`, `uint64`, `int32`, and `int64`.
- Output has the same shape and dtype as the input array.
- Every value in the input must be present in the mapping.
- Non-contiguous inputs are copied before entering C++.

## I/O and Build Dependencies

`bioimage-cpp` intentionally does not replace nifty or affogato I/O helpers.
Load TIFF, HDF5, zarr, N5, OME-NGFF, and related formats with existing Python
libraries, then pass NumPy arrays to `bioimage-cpp`.

The package is designed for small PyPI wheels and does not depend on nifty,
vigra, HDF5, z5, xtensor, pybind11, or other large C++ libraries.
