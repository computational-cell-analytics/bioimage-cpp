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
`bic.segmentation`, and utility functionality (blocking, relabeling, overlap
measurement, union-find, etc.) is under `bic.utils`.

`bic.graph` keeps the core graph types and graph-level algorithms
(`UndirectedGraph`, `GridGraph2D`, `GridGraph3D`, `RegionAdjacencyGraph`,
`connected_components`, `breadth_first_search`, `edge_weighted_watershed`,
`region_adjacency_graph`, `project_node_labels_to_pixels`) at the top level.
Algorithmic domains live in dedicated submodules:

- `bic.graph.multicut` — multicut objective and solvers, fusion-move
  proposal generators, multicut problem loaders.
- `bic.graph.lifted_multicut` — lifted multicut objective and solvers,
  lifted multicut problem loaders. Proposal generators are re-exported
  from `bic.graph.multicut` here for convenience.
- `bic.graph.mutex_watershed` — graph-based mutex watershed clustering
  (with and without semantic constraints).
- `bic.graph.features` — edge-feature accumulation for RAGs and grid
  graphs (boundary maps, affinity channels, lifted edge features).

## Affogato

### Affinities

Affogato:

```python
from affogato.affinities import compute_affinities

affinities, mask = compute_affinities(labels, offsets, ignore_label=0)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

affinities, mask = bic.affinities.compute_affinities(
    labels,
    offsets,
    ignore_label=0,
)
```

Notes:

- Supported label dtypes are `uint32`, `uint64`, `int32`, and `int64`.
- Labels must be 2D or 3D and are copied to C-contiguous memory when needed.
- Offsets are in NumPy axis order and must have one entry per spatial axis.
- The affinity output is `float32` with shape `(n_offsets, *labels.shape)`.
- Pass `return_mask=False` to skip the validity-mask allocation when only the
  affinity array is needed.
- `number_of_threads` must be a positive integer; the default is `1`.

### Embedding Distances

Affogato:

```python
from affogato.affinities import compute_embedding_distances

distances = compute_embedding_distances(values, offsets, norm="l2")
```

bioimage-cpp:

```python
import bioimage_cpp as bic

distances = bic.affinities.compute_embedding_distances(
    values,
    offsets,
    norm="l2",
)
```

Notes:

- `values` has shape `(C, *spatial)` and dtype `float32`. `spatial` must be 2D
  or 3D; non-contiguous arrays are copied to C-contiguous memory when needed.
- Offsets are in NumPy axis order and must have one entry per spatial axis.
- The output is `float32` with shape `(n_offsets, *spatial)`. Out-of-bounds
  positions are left at `0.0`.
- Supported norms are `"l1"` (new in bioimage-cpp), `"l2"`, and `"cosine"`.
  Cosine on a zero-norm channel vector yields `NaN`/`Inf` (matching affogato).
- No mask is returned (matches affogato).
- `number_of_threads` parallelizes over offset channels.

### Mutex Watershed

`bioimage-cpp` ships two mutex-watershed entry points, mirroring the two
affogato APIs: one that consumes a dense affinity grid, and one that
consumes an arbitrary graph with a separate list of mutex (long-range
repulsive) edges.

#### Grid-based mutex watershed (affinity volumes)

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

#### Mutex watershed on a generic graph

For mutex watershed on an arbitrary undirected graph (region adjacency graph
or otherwise) with a separate list of long-range repulsive edges,
`bioimage-cpp` provides `bic.graph.mutex_watershed.mutex_watershed_clustering`. This is a
port of affogato's `compute_mws_clustering` using the same input format as
`LiftedMulticutObjective`: a base graph carries the attractive edges, and
long-range (called *mutex* here) edges are supplied alongside as a `(M, 2)`
node-pair array. The same `(graph, edge_costs, lifted_uvs, lifted_costs)`
tuple used to build a lifted multicut problem can be passed to the mutex
watershed clustering without any reshaping.

Affogato:

```python
from affogato.segmentation import compute_mws_clustering

labels = compute_mws_clustering(
    number_of_nodes,
    uvs.astype(np.uint64),
    mutex_uvs.astype(np.uint64),
    weights.astype(np.float32),
    mutex_weights.astype(np.float32),
)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

graph = bic.graph.UndirectedGraph.from_edges(number_of_nodes, uvs)
labels = bic.graph.mutex_watershed.mutex_watershed_clustering(
    graph,
    weights,
    mutex_uvs,
    mutex_weights,
)
```

Notes:

- The attractive edges are the edges of the base graph; the count and the
  ordering of `weights` must match `graph.number_of_edges`. Mutex edges
  are supplied separately as `(M, 2)` `uint64` pairs with matching
  `mutex_weights`.
- Both `weights` and `mutex_weights` accept `float32` and `float64`. The
  wrapper dispatches to a templated C++ instantiation per dtype; other
  floating dtypes are cast to `float32`. If the two arrays' dtypes do not
  match, both are promoted to `float64` rather than silently downcast.
- Higher weights are processed first (in descending order) — the same
  convention affogato uses.
- The implementation reuses the union-find and per-root mutex-set helpers
  shared with the grid-based mutex watershed (`detail/mutex_storage.hxx`),
  so behavior is consistent between the two entry points.
- Output labels are dense `uint64` ids in `0 .. number_of_clusters - 1`,
  assigned in first-occurrence order (matches the convention of the graph
  multicut solvers, *not* the 1-based foreground labels produced by the
  grid-based variant).
- The function accepts both `UndirectedGraph` and `RegionAdjacencyGraph`.
- Tie-breaking is deterministic: when weights are equal, attractive edges
  are processed before mutex edges, then by index. Affogato's reference
  uses a non-stable `std::sort`, so on inputs with many ties the two
  implementations may produce slightly different (but very similar)
  partitions. See `development/graph/check_mutex_clustering.py` for a
  comparison harness.

### Semantic Mutex Watershed

`bioimage-cpp` mirrors affogato's two semantic-mutex-watershed entry points
— `compute_semantic_mws_segmentation` for affinity volumes and
`compute_semantic_mws_clustering` for an arbitrary graph. Both extend the
regular mutex watershed with a third edge type: per-pixel (or per-node)
"semantic edges" that tag each cluster with a class id. Two clusters that
have been tagged with different class ids cannot subsequently merge.

#### Grid-based semantic mutex watershed (affinity volumes)

Affogato:

```python
from affogato.segmentation import compute_semantic_mws_segmentation

labels, semantic_labels = compute_semantic_mws_segmentation(
    weights,
    offsets,
    number_of_attractive_channels=3,
    strides=[1, 1, 1],
)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

labels, semantic_labels = bic.segmentation.semantic_mutex_watershed(
    weights,
    offsets,
    number_of_attractive_channels=3,
    strides=[1, 1, 1],
)
```

Channel layout (identical to affogato):

- Channels `[0, number_of_attractive_channels)` are attractive grid edges.
- Channels `[number_of_attractive_channels, len(offsets))` are mutex grid
  edges.
- Channels `[len(offsets), affinities.shape[0])` are per-semantic-class
  affinities; channel `len(offsets) + c` scores how strongly each pixel
  belongs to class `c`.

Important migration notes:

- Inputs must represent 2D or 3D grids with shapes `(channels, y, x)` or
  `(channels, z, y, x)` and `channels > len(offsets)` (use
  `bic.segmentation.mutex_watershed` if there are no semantic channels).
- Supported affinity dtypes are `float32` and `float64`.
- Returned `labels` are `uint64`, consecutive, and 1-based for foreground
  pixels (matching the regular grid-based mutex watershed). `semantic_labels`
  is `int64` with `-1` reserved for clusters that received no class
  assignment.
- A boolean `mask` may be passed. Edges touching `False` pixels are
  ignored. Masked pixels are set to label `0` in `labels` and to the
  `mask_label` parameter (default `0`) in `semantic_labels`.
- `strides` and `randomized_strides` follow the same convention as the
  regular grid mutex watershed (mutex channels only; attractive channels
  are always kept).

#### Graph-based semantic mutex watershed

Affogato:

```python
from affogato.segmentation import compute_semantic_mws_clustering

labels, semantic_labels = compute_semantic_mws_clustering(
    number_of_nodes,
    uvs.astype(np.uint64),
    mutex_uvs.astype(np.uint64),
    semantic_node_classes.astype(np.uint64),
    weights.astype(np.float32),
    mutex_weights.astype(np.float32),
    semantic_weights.astype(np.float32),
)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

graph = bic.graph.UndirectedGraph.from_edges(number_of_nodes, uvs)
labels, semantic_labels = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
    graph,
    weights,
    mutex_uvs,
    mutex_weights,
    semantic_node_classes,
    semantic_weights,
)
```

Input format mirrors `mutex_watershed_clustering` plus two extra arrays:

- `semantic_node_classes` is an `(n_semantic, 2)` `uint64` table. Column 0
  is a node id and column 1 is the semantic class id (a non-negative
  integer interpreted as `int64` internally).
- `semantic_weights` is a 1D `float32`/`float64` array of length
  `n_semantic` giving one weight per `(node, class)` candidate.

Notes:

- All three weight arrays (`weights`, `mutex_weights`, `semantic_weights`)
  must have the same floating dtype, or all three are promoted to
  `float64`.
- Output `labels` are dense `uint64` ids in `0 .. number_of_clusters - 1`
  (first-occurrence order, matching the regular graph mutex watershed —
  *not* the 1-based foreground labels produced by the grid variant).
  `semantic_labels` is `int64` with `-1` for unassigned clusters.
- Accepts both `UndirectedGraph` and `RegionAdjacencyGraph`.

#### Divergence from affogato

The bioimage-cpp port fixes a missing `merge_semantic_labels` call on
attractive merges in affogato's graph kernel (`compute_semantic_mws_clustering`):
without that call, a node that has been tagged with a class can have its
tag dropped when it later becomes the non-root of a merge. Affogato's
array kernel additionally invokes `boost::disjoint_sets::link(u, v)` on
raw node ids rather than their roots, which corrupts the union-find tree
on multi-class inputs and over-fragments the result. The unit-test
problems shipped with affogato do not exercise these paths heavily, so
this only shows up on realistic multi-class data.

For most inputs the two implementations agree. On dense multi-class
inputs they may not; the development scripts under
`development/segmentation/check_semantic_mutex_watershed_{2d,3d}.py` and
`development/graph/check_semantic_mutex_clustering.py` print VI/ARI
partition metrics and a semantic-label match fraction so the deviation is
measurable. The bioimage-cpp partitions match an independent Python
reference implementation of the algorithm.

### Watershed From Affinities

`bioimage-cpp` also provides an affinity-driven marker-controlled watershed
for nearest-neighbour affinity maps. It is useful when edge priorities are
already available and no heightmap derivation is needed.

```python
import bioimage_cpp as bic

# Affinity-driven: edge priorities, no heightmap derivation needed
labels = bic.segmentation.watershed_from_affinities(
    affinities,                       # (C, *spatial), C == spatial_ndim
    offsets=[(-1, 0), (0, -1)],       # one NN offset per channel, same sign
    markers=markers,
    mask=optional_mask,
)
```

Notes for `watershed_from_affinities`:

- Each channel must encode a single nearest-neighbour edge (exactly one
  ±1 entry, the rest zero). All offsets must have the same sign — mixing
  positive and negative directions is rejected. The function dispatches
  to a positive-direction or negative-direction specialisation at the C++
  layer so the inner loop has no per-channel sign branches.
- Offsets may be passed in any axis order; the channel ↔ axis mapping is
  rebuilt internally.
- Higher affinity is processed first (high affinity = strong bond).
- Compared to `affogato.segmentation.compute_mws_segmentation`, there are
  no mutex (repulsive) channels and no long-range offsets — use
  `bic.segmentation.mutex_watershed` for that.

## Nifty

### Undirected Graphs

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
- `graph.clone()` returns an independent deep copy. The C++ class is
  move-only (it owns a CSR adjacency buffer), so prefer this over
  reassignment-by-value.
- The internal adjacency is built *lazily* on the first `node_adjacency`
  read, and that lazy build is **not thread-safe**. The built-in
  multi-threaded algorithms freeze the graph internally before fanning out, so
  passing a graph straight into them is safe. But if you build a graph and then
  share it across **your own** threads (concurrent `node_adjacency` reads, a
  BFS, etc.), call `graph.freeze()` once on the construction thread first —
  racing the first read across threads corrupts the adjacency (nondeterministic
  results, possible crashes). `freeze()` eagerly builds the adjacency and is a
  no-op once built; it also removes the first-call rebuild cost from later
  `node_adjacency` reads. This applies to all graph types (`GridGraph2D`,
  `GridGraph3D`, `RegionAdjacencyGraph`).

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

### Region Adjacency Graphs

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

### Distributed Region Adjacency Graphs and Features

For volumes too large to hold in memory, `nifty.distributed` builds a RAG and
its edge features block by block and merges the results. `bioimage-cpp` provides
the equivalent **low-level primitives** under `bic.graph.distributed`; the
orchestration nifty bundles (iterating blocks, sizing halos, and serializing the
per-block subgraphs/features to zarr/N5/HDF5) is intentionally left to the
caller, since I/O and block scheduling belong in Python.

The primitives assume **globally consistent labels** (a segment has the same id
in every block — as after a stitched distributed watershed). Labels must also
be reasonably **dense**: the global graph allocates memory proportional to the
largest node id (`from_unique_edges(number_of_nodes, ...)` builds a dense CSR
over ids `0 .. number_of_nodes - 1`), so sparse or very large globally unique
id ranges need a relabeling pass before building the graph. A block owns the
pixel-pairs whose reference pixel lies in its inner (non-halo) box, so the
caller reads each block with a halo (≥1 on the forward faces for the region
graph / an edge map; ≥ `max |offset|` per side for affinities) and passes the
owned box as `own_begin` / `own_shape` (e.g. from `bic.utils.Blocking`'s
`get_block_with_halo(...).inner_block_local`).

```python
d = bic.graph.distributed

# per block (labels read with a halo):
edges = d.block_region_adjacency_edges(labels_block, own_begin, own_shape)
block_edges, block_stats = d.block_edge_map_stats(labels_block, edge_map_block, own_begin, own_shape)
block_edges, block_stats = d.block_affinity_stats(labels_block, aff_block, offsets, own_begin, own_shape)

# merge the graph, then build the global graph:
global_edges = d.merge_edges([edges_block_0, edges_block_1, ...])
graph = bic.graph.UndirectedGraph.from_unique_edges(number_of_nodes, global_edges)

# fold per-block features onto the global edges (in place), then finalize:
acc = d.empty_edge_stats(graph.number_of_edges)
for be, bs in per_block_stats:
    d.merge_block_edge_stats(graph, acc, be, bs)
features = d.finalize_edge_features(acc, compute_complex_features=True)
```

Notes:

- Blocked results reproduce the whole-volume `region_adjacency_graph` /
  `features.*_features` exactly for `size`, `min` and `max`, and to
  floating-point tolerance for `mean` / `std` (the running moments depend on
  thread count and merge order). The `(n, 5)` partial statistics use the
  numerically stable Welford/Chan representation `[count, mean, M2, min, max]`
  (`M2` = sum of squared deviations from the mean), so `std` stays accurate
  for values with a large baseline and small spread.
- **Median and percentiles are not distributable.** The distributed complex
  output is the moment subset `[mean, std, min, max, size]` — the corresponding
  columns of the in-core 12-column complex features.
- Making per-block-local ids globally consistent (label stitching) is a separate
  step and is not part of these primitives.

### Breadth-First Search

Nifty has an internal `BreadthFirstSearch` template used during lifted-edge
insertion. `bioimage-cpp` exposes a Python-friendly free function:

```python
nodes, distances = bic.graph.breadth_first_search(
    graph,
    source,
    max_distance=3,           # optional, default: full component
    include_source=True,      # set to False for k-hop neighborhoods excluding self
)
```

Both output arrays are 1D `uint64`, listing reached nodes in BFS order with
their hop distance from the source. Useful for building lifted-edge sets
manually, sampling local neighborhoods, or computing graph distances.

### Affine Transformations

`bioimage-cpp` exposes NumPy-only affine transformations under
`bic.transformation`. HDF5, zarr, N5, and OME-NGFF loading stays in Python;
load the desired chunk or subvolume first, then pass the NumPy array here.

Nifty:

```python
import nifty.transformation as nt

out = nt.affineTransformation(
    data,
    matrix,
    order=1,
    bounding_box=(slice(0, 64), slice(0, 64)),
    fill_value=0,
)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

out = bic.transformation.affine_transform(
    data,
    matrix,
    bounding_box=(slice(0, 64), slice(0, 64)),
    order=1,
    fill_value=0,
)
```

Important differences from nifty:

- Only NumPy arrays are accepted. `affineTransformationH5`,
  `affineTransformationZ5`, and coordinate-file transformations are not
  reproduced.
- The API is snake_case only: `affine_transform`.
- `matrix` maps output coordinates to input coordinates in NumPy axis order.
  Matrix shapes `(ndim, ndim + 1)` and homogeneous `(ndim + 1, ndim + 1)` are
  accepted.
- `bounding_box=None` transforms `slice(0, data.shape[d])` for every axis.
  Custom bounding boxes are one slice per axis and cannot use a step.
- Supported interpolation orders are `0` (nearest), `1` (linear),
  `2`/`4`/`5` (quadratic / quartic / quintic B-spline), and `3` (Keys cubic
  convolution, `a = -0.5`). The order set matches `scipy.ndimage`.
- Order `3` is *interpolating* (reproduces input samples at integer
  coordinates). Orders `2`, `4`, `5` are *smoothing* B-spline kernels: they
  exactly match `scipy.ndimage.affine_transform(..., prefilter=False,
  mode='grid-constant')`, which is **not** scipy's default. We do not run
  the cubic-spline IIR prefilter that scipy applies when `prefilter=True`,
  so `bic.transformation.affine_transform(..., order=3)` is **not**
  numerically equivalent to scipy's default `order=3`. See
  `development/transformation/PERFORMANCE_NOTES.md` for the prefilter cost
  analysis and the sketch of how we would add it. Practical guidance:
    - For nifty parity, use `order=0` or `order=1`.
    - For OpenCV-style "smooth cubic that hits the samples", use `order=3`.
    - For scipy `prefilter=False` parity, use `order=2/4/5`.
    - For scipy `prefilter=True` parity, you currently have to prefilter
      the input yourself with `scipy.ndimage.spline_filter` before calling
      our `affine_transform`.
- Border handling for orders 0, 1, and 3 follows
  `scipy.ndimage.affine_transform(..., mode='constant')`: any output
  coordinate that maps to an input coordinate inside `[0, shape - 1]`
  along every axis is interpolated; coordinates fully outside are replaced
  with `fill_value`. In particular the last row/column/slice is sampled
  (nifty's older NumPy affine path treats the last index as out-of-bounds).
  Orders `2/4/5` use `mode='grid-constant'` semantics: each kernel tap
  independently picks up `fill_value` when it is out of bounds, with no
  outer cliff at the input border.
- Output dtype is preserved for all supported input dtypes, including
  integer inputs with linear, cubic, or spline interpolation. Integer
  outputs round to the nearest integer and clamp to the dtype range, so
  cubic / spline overshoots are well defined for `uint8`/`int8`/etc.
- An optional `out=` keyword writes the result into a pre-allocated
  C-contiguous NumPy array of matching shape and dtype.

#### Re-creating nifty's HDF5/zarr affine in Python

`bioimage-cpp` deliberately stops at NumPy; format-specific entry points
(`affineTransformationH5`, `affineTransformationZ5`) are out of scope for
the C++ core. For a downstream library that wants to recreate them, the
NumPy primitives compose naturally — chunk the **output** frame, read
just the input bounding box needed for each output chunk, transform with
`bic.transformation.affine_transform`, write the result back:

```python
import numpy as np
import bioimage_cpp as bic

def affine_transform_chunked(in_dataset, out_dataset, matrix, *,
                             output_shape, order=1, fill_value=0,
                             out_block_shape=(64, 256, 256), halo=None):
    """Apply an affine to a large array, one output block at a time.

    `in_dataset` and `out_dataset` are array-like (numpy / h5py.Dataset /
    zarr.Array / tensorstore / ...). `matrix` maps output coordinates to
    input coordinates in NumPy axis order (the same convention as
    `bic.transformation.affine_transform`).
    """
    ndim = len(output_shape)
    linear = np.asarray(matrix, dtype=np.float64)[:ndim, :ndim]
    translation = np.asarray(matrix, dtype=np.float64)[:ndim, ndim]
    # Default halo: kernel half-width per axis (order/2 rounded up) plus a
    # safety margin for floating-point coordinate drift.
    if halo is None:
        halo = tuple([order + 2] * ndim)

    in_shape = np.asarray(in_dataset.shape)
    out_block = np.asarray(out_block_shape)

    # Walk the output frame block by block.
    block_starts = [
        range(0, output_shape[k], out_block[k]) for k in range(ndim)
    ]
    for corner in np.ndindex(*(len(b) for b in block_starts)):
        out_start = np.array([block_starts[k][corner[k]] for k in range(ndim)])
        out_stop = np.minimum(out_start + out_block, output_shape)

        # 1. Find the input bounding box that all output voxels in this
        #    block could possibly sample. The 8 (2D: 4) corners of the
        #    output block are mapped through `matrix`; the axis-aligned
        #    bounding box of those mapped points (plus a halo) is what we
        #    need from the input array.
        corners = np.stack(np.meshgrid(*[
            [out_start[k], out_stop[k] - 1] for k in range(ndim)
        ], indexing="ij"), axis=-1).reshape(-1, ndim).astype(np.float64)
        in_corners = corners @ linear.T + translation
        in_lo = np.floor(in_corners.min(axis=0)).astype(np.int64) - np.asarray(halo)
        in_hi = np.ceil(in_corners.max(axis=0)).astype(np.int64) + np.asarray(halo)

        # 2. Clip to the input array. Anything outside becomes fill_value
        #    via the affine_transform's border handling.
        in_lo_clipped = np.maximum(in_lo, 0)
        in_hi_clipped = np.minimum(in_hi, in_shape)
        if np.any(in_hi_clipped <= in_lo_clipped):
            # Output block lies entirely outside the input frame.
            out_block_data = np.full(
                tuple((out_stop - out_start).tolist()),
                fill_value, dtype=out_dataset.dtype,
            )
        else:
            slicer = tuple(slice(int(lo), int(hi))
                           for lo, hi in zip(in_lo_clipped, in_hi_clipped))
            in_block = np.ascontiguousarray(in_dataset[slicer])

            # 3. Translate `matrix` into the input-block-local frame.
            #    Our convention: input = linear @ output + translation.
            #    For the local block, input_local = input - in_lo_clipped.
            local_matrix = np.hstack([linear, (translation - in_lo_clipped)[:, None]])

            # 4. Run the affine on the in-memory block. We pass the local
            #    bounding box in **output** coordinates: this block of the
            #    output spans (out_start, out_stop).
            out_block_data = bic.transformation.affine_transform(
                in_block,
                local_matrix,
                bounding_box=tuple(slice(int(a), int(b))
                                   for a, b in zip(out_start, out_stop)),
                order=order,
                fill_value=fill_value,
            )

        out_dataset[tuple(slice(int(a), int(b))
                          for a, b in zip(out_start, out_stop))] = out_block_data
```

Notes:

- The halo accounts for the kernel's tap reach; the safety margin handles
  floating-point drift in the corner mapping. `order + 2` is conservative
  for orders ≤ 5.
- This pattern works for `h5py.Dataset`, `zarr.Array`, `tensorstore`, or
  any other lazy-array library that supports NumPy-style indexing — there
  is nothing format-specific in the body.
- For best throughput, choose `out_block_shape` to match the on-disk
  chunking of `out_dataset` (one block = one chunk write) and large enough
  in each axis that the input-side read is also full chunks.
- Anti-aliasing for downsampling pipelines: replace
  `bic.transformation.affine_transform(...)` in step 4 with
  `bic.transformation.resample(...)`. The Gaussian sigma is derived from
  `matrix` and is identical for every block, so the per-block smoothing
  cost is constant.
- For random-access transformations (large rotations, perspective warps)
  the per-block input bounding box can be much larger than the output
  block. A real implementation should either cap the read size and skip
  obviously-empty output blocks, or partition the output into a finer grid
  for those cases.

### Blocking

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

blocking = bic.utils.Blocking([0, 0], [100, 80], [32, 32])
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

### Dictionary-Based Relabeling

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

### Run-Length Encoding

`nifty.tools.computeRLE` computes the COCO-style binary run-length encoding of an
array. Use `bic.utils.compute_rle`:

```python
import nifty.tools as nt

mask = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.uint8)

rle = nt.computeRLE(mask)            # nifty: Python list [2, 3, 2, 2]
rle = bic.utils.compute_rle(mask)    # bioimage-cpp: np.int64 array([2, 3, 2, 2])
```

Notes:

- The array is flattened in **C-order** and interpreted as binary
  (zero vs. nonzero). Counts always start with a run of zeros and then
  alternate; a leading `0` is emitted when the first element is nonzero.
- Supported input dtypes are `bool`, `uint8`, `uint16`, `uint32`, `uint64`,
  `int32`, and `int64`.
- **Behavioral difference:** nifty returns a Python `list`; `bioimage-cpp`
  returns a 1-D NumPy `int64` array.
- Non-contiguous inputs are copied before entering C++.

### Edge-Weighted Watershed

Nifty:

```python
import nifty.graph as ng

labels = ng.edgeWeightedWatershedsSegmentation(graph, edge_weights, seeds)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

labels = bic.graph.edge_weighted_watershed(graph, edge_weights, seeds)
```

Notes:

- Only the Kruskal variant of nifty's algorithm is provided. Edges are visited
  in ascending weight order; two distinct components merge iff at least one is
  unlabeled (seed label `0`), so seed boundaries are preserved.
- `graph` may be an `UndirectedGraph` or a `RegionAdjacencyGraph`.
- `edge_weights` must be 1D with length `graph.number_of_edges`. Supported
  dtypes are `float32` and `float64`; other floating dtypes are promoted to
  `float32` (matching nifty, whose Python binding is `float32`-only). Non-float
  dtypes raise `TypeError`.
- `seeds` must be 1D with length `graph.number_of_nodes`. Supported dtypes are
  `uint32`, `uint64`, `int32`, `int64`. The value `0` marks unlabeled nodes;
  non-zero ids are propagated along low-weight paths. Signed seed arrays must
  not contain negative values.
- The output is 1D with length `graph.number_of_nodes` and the same dtype as
  `seeds`. Seed label values are preserved (no dense relabeling). Nodes that
  no seed can reach remain `0`.

Intentional differences vs. nifty:

- No priority-queue variant — only the simpler sort + union-find Kruskal flow.
  For the same input it matches nifty's default behavior (which also dispatches
  to the Kruskal implementation).
- No carving / background-bias variant. Build a carving prior into the edge
  weights before calling the function if needed.

### External Problem Instances

bioimage-cpp ships pooch-backed downloaders for the multicut and lifted
multicut benchmark problems used by the development comparison scripts and
the regression tests. Files are cached under `~/.cache/bioimage-cpp/`,
overridable via the `BIOIMAGE_CPP_CACHE` environment variable.

`pooch` is an optional runtime dependency — install via the `test` or `data`
extras, e.g. `pip install bioimage-cpp[data]`.

Multicut problems (3 samples × 2 sizes, originally from
`elf.segmentation.utils.load_multicut_problem`):

```python
# Returns (UndirectedGraph, edge_costs)
graph, costs = bic.graph.multicut.load_multicut_problem(sample="A", size="small")
# Or just the underlying arrays
uv_ids, costs = bic.graph.multicut.load_multicut_problem_data(sample="B", size="medium")
# Or the cached file path
path = bic.graph.multicut.multicut_problem_path(sample="C", size="medium")
```

Valid samples are `"A"`, `"B"`, `"C"`; valid sizes are `"small"` and
`"medium"`. The legacy `load_external_multicut_problem` /
`load_external_multicut_problem_data` / `external_multicut_problem_path`
shims default to sample A, size small and continue to honor the
`BIOIMAGE_CPP_EXTERNAL_MULTICUT_PATH` and
`BIOIMAGE_CPP_EXTERNAL_MULTICUT_CACHE` environment variables.

Lifted multicut problems (2D ISBI slice, RAG-based 3D volume, and grid-graph
volume):

```python
problem = bic.graph.lifted_multicut.load_lifted_multicut_problem(size="2d")
# Fields: n_nodes (int), local_uvs, local_costs, lifted_uvs, lifted_costs.
graph = bic.graph.UndirectedGraph.from_edges(problem.n_nodes, problem.local_uvs)
objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
    graph,
    problem.local_costs,
    lifted_uvs=problem.lifted_uvs,
    lifted_costs=problem.lifted_costs,
)
```

Valid sizes are `"2d"`, `"3d"`, and `"grid"`.

Notes:

- Every download is integrity-checked against a SHA256 in the registry; a
  corrupted cache file is detected on the next `load_*` call.
- Downloads are lazy: nothing happens until you call a loader. Re-runs are
  free (the cached file is reused).
- For air-gapped use, fetch the file once on a machine with network access
  and copy `~/.cache/bioimage-cpp/<filename>` to the same path on the target
  machine.

### Grid Graphs

Nifty-style regular grid graphs map to explicit 2D or 3D grid graph classes:

```python
graph = bic.graph.GridGraph2D((height, width))
graph = bic.graph.GridGraph3D((depth, height, width))
graph = bic.graph.grid_graph((height, width))
```

Grid graph nodes use NumPy C-order ids. For a 2D shape `(y, x)`, node
`(row, col)` has id `row * x + col`; for 3D `(z, y, x)`, ids follow the same
row-major convention. `GridGraph2D` and `GridGraph3D` inherit the regular
`UndirectedGraph` API, so solvers, connected components, breadth-first search,
and `uv_ids()` work unchanged.

Important differences:

- Only nearest-neighbor 2D and 3D grids are exposed for now.
- Edge ids are deterministic axis blocks: axis 0 edges first, then axis 1, and
  axis 2 for 3D.
- Scalar boundary maps can be converted to edge weights with
  `grid_boundary_features(graph, boundary_map)`.
- Local affinity channels can be converted to edge-aligned weights with
  `grid_affinity_features(graph, affinities, offsets)`.
- Mixed local and long-range affinity offsets can be converted with
  `grid_affinity_features_with_lifted(...)`, which returns local graph weights
  plus explicit long-range `uv_ids` and weights for lifted multicut or mutex
  watershed style workflows.
- Nifty's `projectEdgeIdsToPixels` and `projectEdgeIdsToPixelsWithOffsets` map
  to `graph.project_edge_ids_to_pixels()` and
  `graph.project_edge_ids_to_pixels_with_offsets(offsets, *, strides=None, mask=None)`
  on `GridGraph2D` / `GridGraph3D`. The basic form returns an `int64` array of
  shape `(ndim, *graph.shape)` with each grid edge id written at its pivot
  pixel and `-1` elsewhere. The offsets form returns
  `(array, n_valid)`: an `int64` array of shape `(len(offsets), *graph.shape)`
  whose non-`-1` entries are a sequential counter over the in-bounds (and
  filter-accepted) targets, plus the total count. `strides` keeps only coords
  aligned along every axis; `mask` keeps only coords where a boolean array
  of shape `(len(offsets), *graph.shape)` is true. Like in nifty, `strides`
  and `mask` are mutually exclusive — passing both raises `ValueError`.
- The three grid feature functions preserve `float32` and `float64` input
  dtype end-to-end (no internal copy to `float64`); other dtypes are
  promoted to `float64`. Output weight arrays match the input dtype.
- Grid graph construction does not materialize the per-node adjacency
  list. If you only need `uv_ids()` and edge features (the common case)
  you pay nothing for adjacency. The first call to `node_adjacency`,
  `connected_components`, `breadth_first_search`, or
  `extract_subgraph_from_nodes` on a grid graph triggers a one-shot
  rebuild; call `graph.freeze()` on the construction thread before
  fan-out if you intend to use those from multiple threads.
- Affogato-style masks and seed edges are not part of the public grid feature
  API yet; the implementation is structured so these filters/extra edges can
  be added later.

### Label Multiset

The label-multiset data structure stores, for each spatial block of a label
volume, a histogram of `(label, count)` pairs over the underlying voxels,
with identical histograms across blocks deduplicated into shared storage.
It is used by Paintera to build multi-resolution label pyramids.

`nifty.tools` exposes three functions / classes — `readSubset`,
`downsampleMultiset`, and `MultisetMerger` — operating on five flat arrays
(`offsets`, `entry_sizes`, `entry_offsets`, `ids`, `counts`). `bioimage-cpp`
keeps the same algorithm and storage layout but wraps it in a
`LabelMultiset` dataclass plus a level-0 bootstrap helper.

Nifty:

```python
import nifty.tools as nt

# nifty does not provide a "from labels" helper; level-0 multisets are
# typically constructed by the caller (e.g. by writing histograms manually
# to N5 chunks).
blocking = nt.blocking([0, 0, 0], list(labels.shape), [2, 2, 2])
argmax, new_offsets, new_ids, new_counts = nt.downsampleMultiset(
    blocking, offsets, entry_sizes, entry_offsets, ids, counts,
    restrict_set=-1,
)

ids, counts = nt.readSubset(offsets, sizes, ids, counts, True)

merger = nt.MultisetMerger(unique_offsets, entry_sizes, ids, counts)
merger.update(unique_offsets, entry_sizes, ids, counts, offsets)
```

bioimage-cpp:

```python
import bioimage_cpp as bic
from bioimage_cpp.label_multiset import (
    LabelMultiset,
    MultisetMerger,
    downsample_multiset,
    multiset_from_labels,
    read_subset,
)

# Build the level-0 multiset directly from a label volume.
ms0 = multiset_from_labels(labels, block_shape=(1, 1, 1))

# Downsample one level. `blocking.roi_end` must match the input's spatial
# extent (i.e. the shape used to build ms0).
blocking = bic.utils.Blocking([0, 0, 0], list(labels.shape), [2, 2, 2])
ms1 = downsample_multiset(ms0, blocking, restrict_set=-1)

# Merge entries from a list of (offset, size) ranges into one histogram.
ids, counts = read_subset(offsets, sizes, ms1.ids, ms1.counts)

# Deduplicating merger — constructor takes one offset per unique entry.
merger = MultisetMerger.from_multiset(ms1)
merger.update(unique_offsets, entry_sizes, ids, counts, offsets)
```

Name and API changes:

| nifty-style name | bioimage-cpp name |
| --- | --- |
| `readSubset` | `read_subset` |
| `downsampleMultiset` | `downsample_multiset` |
| `MultisetMerger` | `MultisetMerger` |
| `MultisetMerger.get_ids()` | `MultisetMerger.ids` (property) |
| `MultisetMerger.get_counts()` | `MultisetMerger.counts` |
| `restrict_set` (keyword) | `restrict_set` (keyword, same default `-1`) |

Notes:

- A `LabelMultiset` carries all five arrays (`offsets`, `entry_offsets`,
  `entry_sizes`, `ids`, `counts`) plus `argmax`. Nifty's
  `downsampleMultiset` returns only four of them and leaves the caller to
  reconstruct `entry_offsets` / `entry_sizes`; `bioimage-cpp` returns them
  directly so multi-level downsample chains compose without bookkeeping.
- `multiset_from_labels(labels, block_shape)` builds the level-0 multiset
  from a `uint32` or `uint64` label volume in one call. There is no nifty
  equivalent.
- `MultisetMerger.__init__` takes one offset per unique entry (length
  `n_unique`), matching nifty's contract. Use
  `MultisetMerger.from_multiset(ms)` to construct one directly from a
  `LabelMultiset`.
- Count dtype is `uint32` (nifty uses `int32`). Convert at the boundary
  if you are reading nifty-written data.
- 2D and 3D blockings are both supported. The bindings instantiate `uint64`
  ids, `uint32` counts, and `uint64` offsets; wider dtype matrices can be
  added on demand.
- The `LabelMultisetWrapper` z5/N5 reader from
  `nifty/tools/label_multiset_wrapper.hxx` is intentionally **not** ported
  — I/O stays out of the C++ core. Read/write Paintera-format chunks with
  `zarr`/`numpy` in Python if needed.

### Lifted Multicut

Nifty exposes lifted multicut through a separate objective + solver hierarchy.
`bioimage-cpp` mirrors the structure with `LiftedMulticutObjective` and a
`LiftedMulticutSolver` class hierarchy.

Nifty:

```python
import nifty.graph.opt.lifted_multicut as nlmc

objective = nlmc.liftedMulticutObjective(graph)
objective.insertLiftedEdgesBfs(max_distance=3)
for u, v, w in lifted_weights:
    objective.setCost(u, v, w)
solver = objective.liftedMulticutGreedyAdditiveFactory().create(objective)
labels = solver.optimize()
```

bioimage-cpp:

```python
import bioimage_cpp as bic

objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
    graph,
    edge_costs,
    lifted_uvs=lifted_uvs,
    lifted_costs=lifted_costs,
    bfs_distance=3,  # optional: also insert zero-weight lifted edges within k hops
)
labels = bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut().optimize(objective)
energy = objective.energy(labels)
```

`LiftedMulticutObjective` accepts:

- `graph` — an `UndirectedGraph` or `RegionAdjacencyGraph`. The constructor
  copies the topology, so further mutations on the input graph do not affect
  the objective.
- `edge_costs` — 1D `float64` array of length `graph.number_of_edges`.
- `lifted_uvs` / `lifted_costs` — optional `(n_lifted, 2)` uint64 array and 1D
  float64 array of equal length, listing the additional lifted edges and
  their weights.
- `bfs_distance` — optional positive integer. Adds a zero-weight lifted edge
  for every pair of nodes within this many base-graph hops of each other
  (excluding nodes already connected by a base edge). Pairs with both
  `lifted_uvs` and `bfs_distance` to seed the topology and then update
  specific weights.
- `overwrite_existing` — when `True`, lifted entries that coincide with an
  existing edge replace its weight; the default accumulates.

Available solvers (no ILP solvers yet):

| nifty factory | bioimage-cpp solver |
| --- | --- |
| `liftedMulticutGreedyAdditiveFactory()` | `LiftedGreedyAdditiveMulticut()` |
| `liftedMulticutKernighanLinFactory(...)` | `LiftedKernighanLinMulticut(...)` |
| `fusionMoveBasedFactory(...)` | `FusionMoveLiftedMulticut(...)` |
| `chainedSolversFactory([...])` | `LiftedChainedSolvers([...])` |

`FusionMoveLiftedMulticut` mirrors `FusionMoveMulticut` (same proposal-generator
plumbing, same threading + multi-proposal joint-fuse semantics, same best-of
safety net). The differences are:

- Proposal generators operate on the *base* graph and base edge costs (only
  base-graph edges are candidate cut edges; lifted edges contribute to energy
  but cannot be contracted directly). The driver extracts the base costs from
  `objective.weights[:objective.number_of_base_edges]` automatically.
- Each fuse contracts the base graph by agreement, aggregates *both* base and
  lifted weights onto the contracted lifted-multicut subproblem (lifted edges
  whose endpoints land on already-existing contracted base edges fold into
  them; the rest become new contracted lifted edges), and solves the
  subproblem with a `LiftedMulticutSolver`.
- The default sub-solver and warm-start are `LiftedGreedyAdditiveMulticut`.
  Both `LiftedGreedyAdditiveMulticut` and `LiftedKernighanLinMulticut` are
  pluggable via `sub_solver=`.

```python
solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
    proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(
        sigma=1.0, n_seeds_fraction=0.1, seed=0,
    ),
    sub_solver=bic.graph.lifted_multicut.LiftedKernighanLinMulticut(number_of_outer_iterations=3),
    number_of_iterations=10,
    stop_if_no_improvement=4,
    number_of_threads=4,
)
labels = solver.optimize(objective)
```

A typical warm-started solve combines greedy and KL:

```python
solver = bic.graph.lifted_multicut.LiftedChainedSolvers([
    bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut(),
    bic.graph.lifted_multicut.LiftedKernighanLinMulticut(number_of_outer_iterations=10),
])
labels = solver.optimize(objective)
```

Notes:

- Output labels are dense `uint64` ids in `0 .. number_of_clusters - 1`.
- Every output cluster is *base-graph connected* — both solvers enforce this
  invariant. A strongly attractive lifted edge between two nodes that have no
  base-graph path between them will not merge their clusters.
- `LiftedKernighanLinMulticut` warm-starts from the lifted greedy-additive
  solution when the objective's current labels are the trivial singleton
  labeling.
- `objective.set_cost(u, v, weight, overwrite=False)` updates or inserts a
  single lifted edge.
- The lifted graph is exposed via `objective.lifted_graph`; the first
  `objective.number_of_base_edges` edges are exactly the base edges in the
  same order as in `graph`.

#### Building a lifted multicut problem from affinities

For the common case of lifted multicut on a watershed over-segmentation,
nifty offers `nifty.graph.rag.computeLiftedEdgesFromRagAndOffsets` (lifted
edge discovery) and per-channel affinity accumulators. bioimage-cpp exposes
two focused helpers that cover the same workflow:

```python
# Discover lifted edges implied by long-range affinity offsets. 1-hop offsets
# are skipped automatically, so the full offset list can be passed in.
lifted_uvs = bic.graph.features.lifted_edges_from_affinities(
    rag, oversegmentation, offsets, number_of_threads=0,
)

# Accumulate (mean, size) statistics per lifted edge. Pixel pairs whose
# (u, v) does not appear in `lifted_uvs` are skipped, so local edges are
# never contaminated with long-range affinities.
lifted_features = bic.graph.features.lifted_affinity_features(
    oversegmentation, affinities, offsets, lifted_uvs,
    number_of_threads=0,
)
# For the 12-column feature set (mean, median, std, min, max, percentiles, size):
lifted_features = bic.graph.features.lifted_affinity_features_complex(...)
```

The output column conventions match the local-edge variants
(`SIMPLE_EDGE_FEATURE_NAMES`, `COMPLEX_EDGE_FEATURE_NAMES`).

#### Building lifted edges from per-node labels

When the lifted edges come from semantic / class labels per RAG node rather
than from long-range affinities, nifty offers
`nifty.distributed.liftedNeighborhoodFromNodeLabels`. The bioimage-cpp
equivalent lives under `bic.graph.lifted_multicut`:

```python
# nifty
lifted_uvs = nifty.distributed.liftedNeighborhoodFromNodeLabels(
    graph, node_labels, graphDepth=2, numberOfThreads=4,
    mode='all', ignoreLabel=0,
)

# bioimage-cpp
lifted_uvs = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
    graph, node_labels, graph_depth=2,
    mode='all', ignore_label=0, number_of_threads=4,
)
```

Both functions return an `(n_lifted, 2)` `uint64` array of `(u, v)` pairs
with `u < v`, sorted lexicographically. The BFS hop distance is restricted
to `[2, graph_depth]`, so base-graph edges are excluded. `mode='same'` /
`'different'` filter by whether `node_labels[u] == node_labels[v]`;
`ignore_label` drops every pair where either endpoint label matches.

Intentional differences vs. nifty:

- snake_case parameter names (`graph_depth`, `ignore_label`,
  `number_of_threads`);
- `ignore_label` defaults to `None` (no filtering) instead of `0`;
- node `0` is iterated as a source (nifty's distributed variant has an
  off-by-one that silently skips it).

End-to-end pipeline (also in `examples/segmentation/lifted_multicut_from_affinities.py`):

```python
rag = bic.graph.region_adjacency_graph(oversegmentation)
local_costs = local_threshold - bic.graph.features.affinity_features(
    rag, oversegmentation, direct_affinities, direct_offsets,
)[:, 0]
lifted_uvs = bic.graph.features.lifted_edges_from_affinities(
    rag, oversegmentation, long_range_offsets,
)
lifted_costs = lifted_threshold - bic.graph.features.lifted_affinity_features(
    oversegmentation, long_range_affinities, long_range_offsets, lifted_uvs,
)[:, 0]
objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
    rag, local_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs,
)
```

### Multicut

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

objective = bic.graph.multicut.MulticutObjective(graph, edge_costs)
labels = bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective)
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
solver = bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5)
labels = solver.optimize(objective)
```

If the objective's labels are left at the default (one cluster per node),
`KernighanLinMulticut` warm-starts from a greedy-additive solution
internally, matching `kernighanLinFactory(warmStartGreedy=True)`. To skip the
warm-start, set `objective.set_labels(...)` to a non-trivial labeling first.

Chaining solvers:

```python
solver = bic.graph.multicut.ChainedMulticutSolvers([
    bic.graph.multicut.GreedyAdditiveMulticut(),
    bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
])
labels = solver.optimize(objective)
```

Decomposing a problem into positive-cost connected components and solving each
sub-problem with a cheaper solver:

```python
solver = bic.graph.multicut.MulticutDecomposer(
    sub_solver=bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
    fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
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

#### Fusion Moves

Nifty exposes the fusion-move multicut solver via the factory hierarchy with a
chosen proposal generator and sub-solver factory.

Nifty:

```python
import nifty.graph.opt.multicut as nmc

objective = nmc.multicutObjective(graph, edge_costs)
pgen = nmc.watershedProposals(sigma=1.0, numberOfSeeds=0.1)
factory = nmc.fusionMoveBasedFactory(
    proposalGenerator=pgen,
    fusionMove=nmc.fusionMoveSettings(
        mcFactory=nmc.greedyAdditiveFactory(),
    ),
    numberOfIterations=10,
    stopIfNoImprovement=4,
)
labels = factory.create(objective).optimize()
```

bioimage-cpp:

```python
import bioimage_cpp as bic

objective = bic.graph.multicut.MulticutObjective(graph, edge_costs)
solver = bic.graph.multicut.FusionMoveMulticut(
    proposal_generator=bic.graph.multicut.WatershedProposalGenerator(
        sigma=1.0, n_seeds_fraction=0.1, seed=0,
    ),
    sub_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
    number_of_iterations=10,
    stop_if_no_improvement=4,
)
labels = solver.optimize(objective)
```

Proposal generators:

| nifty proposal generator | bioimage-cpp proposal generator |
| --- | --- |
| `watershedProposals(sigma=..., numberOfSeeds=...)` | `WatershedProposalGenerator(sigma=..., n_seeds_fraction=..., seed=...)` |
| `greedyAdditiveProposals(sigma=..., weightStopCond=..., nodeNumStopCond=...)` | `GreedyAdditiveProposalGenerator(sigma=..., weight_stop=..., node_num_stop=..., seed=...)` |

Sub-solvers: any built-in multicut solver (`GreedyAdditiveMulticut`,
`GreedyFixationMulticut`, `KernighanLinMulticut`). If `sub_solver` is omitted,
the default is `GreedyAdditiveMulticut` constructed with no-noise defaults.

Intentional differences vs. nifty:

- Single object construction: no separate factory / solver step.
- Proposal generators are Python classes carrying their settings; the C++
  proposal-generator object is built lazily when `optimize` is called.
- The driver warm-starts from the trivial singleton labeling by running the
  default greedy-additive sub-solver once before the proposal loop.
- A best-of safety net keeps the running energy monotonically non-increasing
  across iterations (compared against current, proposals, fused, and the
  stage-2 joint fuse).
- Parallel proposal generation and a multi-proposal joint fuse are supported:
  `number_of_threads=T` runs `number_of_parallel_proposals=P` proposal
  generators in parallel within each iteration. By default `P=2` when `T=1`
  and `P=T` when `T>1`; pass an explicit `number_of_parallel_proposals` to
  override. Each parallel slot uses an independent proposal generator with
  seed `proposal_generator.seed + slot_index` so the result is deterministic
  for a given `(seed, T, P)`. When at least two parallel pairwise fuses fail
  to improve on the current best, a joint multi-proposal fuse runs over the
  surviving fused candidates (matches nifty's `ccFusionMoveBased` stage-2
  behaviour).

Notes:

- Custom Python proposal generators are not yet supported; subclass
  `ProposalGenerator` and provide your own `_build` returning a C++
  proposal-generator object if you need to extend the set.

### Agglomerative Cluster Policies

`bioimage_cpp.graph.agglomeration` provides hierarchical agglomerative
clustering driven by a small set of policy classes, matching the policies
in `nifty.graph.agglo`. Each policy is a max-heap-style driver (smaller
edge indicator = stronger merge candidate, matching nifty's convention)
with policy-specific priority computation, merge rule, and stopping
criterion. All policies accept any `UndirectedGraph` subclass —
`RegionAdjacencyGraph`, `GridGraph2D`/`GridGraph3D` included.

Nifty:

```python
import nifty.graph.agglo as nagglo

# Hierarchical, edge-weighted clustering.
policy = nagglo.edgeWeightedClusterPolicy(
    graph=graph,
    edgeIndicators=edge_indicators,
    edgeSizes=edge_sizes,
    nodeSizes=node_sizes,
    numberOfNodesStop=number_of_clusters_stop,
    sizeRegularizer=0.5,
)
labels = nagglo.agglomerativeClustering(policy).run().result()
```

bioimage-cpp:

```python
labels = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
    num_clusters_stop=number_of_clusters_stop,
    size_regularizer=0.5,
).optimize(graph, edge_indicators, edge_sizes=edge_sizes, node_sizes=node_sizes)
```

Mapping:

| Nifty | bioimage-cpp |
| --- | --- |
| `edgeWeightedClusterPolicy(...)` | `EdgeWeightedClusterPolicy(num_clusters_stop=, size_regularizer=).optimize(graph, edge_indicators, edge_sizes=, node_sizes=)` |
| `nodeAndEdgeWeightedClusterPolicy(...)` | `NodeAndEdgeWeightedClusterPolicy(num_clusters_stop=, size_regularizer=, beta=).optimize(graph, edge_indicators, node_features, edge_sizes=, node_sizes=)` |
| `malaClusterPolicy(...)` | `MalaClusterPolicy(num_bins=, bin_min=, bin_max=, num_clusters_stop=, num_edges_stop=, threshold=).optimize(graph, edge_indicators)` |
| `gaspClusterPolicy(...)` (signed weights + linkage) | `GaspClusterPolicy(num_clusters_stop=, linkage=).optimize(graph, edge_weights, edge_sizes=, is_mergeable=)` |

`GaspClusterPolicy` linkage strings map to the rules in Bailoni et al.'s
GASP framework: `"sum"`, `"mean"`, `"max"`, `"min"`, `"abs_max"`,
`"mutex_watershed"`. The `mutex_watershed` linkage treats a negative
heap-top weight as a cannot-link constraint; the others apply the chosen
linkage update without imposing hard constraints from signs. The
optional `is_mergeable` mask marks edges that should be used only to
install cluster-level cannot-link constraints.

Differences from nifty:

- `optimize` returns dense `uint64` node labels directly. Nifty exposes a
  separate driver (`agglomerativeClustering(policy).run().result()`); the
  underlying loop is the same.
- Both `float32` and `float64` inputs are accepted; computation runs in
  `float64` internally.
- Tie-breaks follow the deterministic order of edge ids returned by
  `UndirectedGraph`, which may differ from nifty's. On inputs where many
  edges share the same indicator value, this combines with the
  hierarchical agglomeration's positive feedback loop (each tied merge
  changes node sizes, which changes the harmonic size factor `sFac`,
  which changes future priorities) to give cascading divergence. On the
  external multicut problem sample C/medium, where 86% of indicator
  values are non-unique, perturbing the indicators of a single bic run by
  1e-9 random noise can change the final partition's adjusted Rand index
  vs. its own unperturbed output by ~0.5 (the algorithm is chaotically
  sensitive to tie-breaking under non-zero `size_regularizer`). Both
  partitions are valid clusterings; partition agreement (VI, ARI) is the
  appropriate comparison metric, not label equality.

### Projecting RAG Node Labels to Pixels

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

### Mapping RAG Edges to Pixel Coordinates

Nifty's `ragCoordinates` scans the label volume once and caches, per edge, the
pixel coordinates of the boundary between the two adjacent regions, so that
per-edge values (e.g. edge probabilities) can be painted back onto a volume.

Nifty:

```python
import nifty.graph.rag as nrag

rag = nrag.gridRag(labels)
rag_coords = nrag.ragCoordinates(rag)
storage = rag_coords.storageLengths()
volume = rag_coords.edgesToVolume(edge_values, edgeDirection=0)
```

bioimage-cpp:

```python
rag = bic.graph.region_adjacency_graph(labels)
rag_coords = bic.graph.rag_coordinates(rag, labels)

storage = rag_coords.storage_lengths()
coords = rag_coords.edge_coordinates(edge_id)          # (n_points, ndim)
volume = rag_coords.edges_to_volume(edge_values, edge_direction=0)
```

Notes:

- `labels` must be the over-segmentation used to construct `rag`, and is passed
  explicitly (nifty's RAG holds an internal reference to it; ours does not).
  Supported label dtypes: `uint32`, `uint64`, `int32`, `int64`.
- A boundary "contact" is a pair of directly adjacent pixels with different
  labels. Each contact contributes two coordinates to its edge: the
  lower-coordinate pixel followed by its `+axis` neighbor. `storage_lengths`
  therefore reports `2 * n_contacts` per edge, and coordinates are stored in
  scan order (NumPy axis order, C-contiguous).
- `edge_direction` selects which side(s) to report / paint: `0` = both (default),
  `1` = lower-coordinate pixel only, `2` = higher-coordinate pixel only.
- `edges_to_volume` returns a volume of the label shape and dtype matching
  `edge_values` (supported: `float32`, `float64`, `uint32`, `uint64`).
  Non-boundary pixels are set to `ignore_value`. Painting is sequential in
  ascending edge id, so where several edges' boundaries coincide on a pixel the
  highest edge id wins — a deterministic, race-free rule (nifty's parallel
  `edgesToVolume` does not guarantee a tie-break order).
- The cached object can be reused across many `edges_to_volume` calls without
  re-scanning the labels. `number_of_threads=0` (on `rag_coordinates`) uses the
  library default.

### Accumulating Labels on a RAG

Nifty's `gridRagAccumulateLabels` projects a second label volume onto a RAG
by taking a per-node majority vote (commonly used to project a ground-truth
segmentation onto an over-segmentation).

Nifty:

```python
import nifty.graph.rag as nrag

rag = nrag.gridRag(labels)
node_labels = nrag.gridRagAccumulateLabels(rag, gt)
# ignore label 0 in the ground truth (nifty's "ignoreBackground"):
node_labels = nrag.gridRagAccumulateLabels(rag, gt, ignoreBackground=True)
```

bioimage-cpp:

```python
rag = bic.graph.region_adjacency_graph(labels)
node_labels = bic.graph.features.accumulate_labels(rag, labels, gt)
# arbitrary ignore value (covers nifty's ignoreBackground=True by passing 0):
node_labels = bic.graph.features.accumulate_labels(
    rag, labels, gt, ignore_value=0
)
```

Notes:

- `labels` must be the over-segmentation used to construct `rag`.
- `other_labels` must have the same shape as `labels`. Supported dtypes for
  both arrays: `uint32`, `uint64`, `int32`, `int64`; they may differ.
- The output has length `rag.number_of_nodes` and the same dtype as
  `other_labels`. Nodes whose pixels are all ignored receive `0`.
- Ties in the majority vote are broken by smaller label id (deterministic).
  Nifty's tie-breaking depends on `std::unordered_map` iteration order and
  is therefore platform-dependent; `bic` resolves ties deterministically.
- `ignore_value` is more general than nifty's boolean `ignoreBackground`:
  pass `0` to reproduce `ignoreBackground=True`, or any other value to skip
  arbitrary sentinels (e.g. `255` or `-1`).

### RAG Boundary and Affinity Features

Nifty has RAG feature helpers such as `accumulateEdgeMeanAndLength`,
`accumulateEdgeStandartFeatures`, and affinity feature accumulation helpers.
In `bioimage-cpp`, these are exposed as explicit NumPy-returning functions.

Simple edge-map features:

```python
rag = bic.graph.region_adjacency_graph(labels)
features = bic.graph.features.edge_map_features(rag, labels, edge_map)
```

The columns are:

```python
bic.graph.features.SIMPLE_EDGE_FEATURE_NAMES
# ("mean", "size")
```

Complex edge-map features:

```python
features = bic.graph.features.edge_map_features_complex(rag, labels, edge_map)
```

The columns are:

```python
bic.graph.features.COMPLEX_EDGE_FEATURE_NAMES
# ("mean", "median", "std", "min", "max", "p5", "p10",
#  "p25", "p75", "p90", "p95", "size")
```

Affinity features:

```python
features = bic.graph.features.affinity_features(
    rag,
    labels,
    affinities,
    offsets=[[0, 1], [1, 0]],
)
```

Complex affinity features:

```python
features = bic.graph.features.affinity_features_complex(
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

### Segmentation Overlaps

Nifty:

```python
import nifty.ground_truth as ngt

overlap = ngt.overlap(segmentation, ground_truth)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

overlap = bic.utils.segmentation_overlap(segmentation, ground_truth)
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

### Union-Find

Nifty exposes a disjoint-set / union-find structure as `nifty.ufd.ufd`.
`bioimage-cpp` provides the same primitive under `bic.utils`.

Nifty:

```python
import nifty.ufd as nufd
import numpy as np

uf = nufd.ufd(5)
uf.merge(0, 1)
uf.merge(np.array([[2, 3], [3, 4]], dtype="uint64"))
roots = uf.find(np.array([0, 1, 2, 3, 4], dtype="uint64"))
labels = uf.elementLabeling()
```

bioimage-cpp:

```python
import bioimage_cpp as bic
import numpy as np

uf = bic.utils.UnionFind(5)
uf.merge(0, 1)
uf.merge(np.array([[2, 3], [3, 4]], dtype=np.uint64))
roots = uf.find(np.array([0, 1, 2, 3, 4], dtype=np.uint64))
labels = uf.element_labeling()
```

Method mapping:

| nifty-style name | bioimage-cpp name |
| --- | --- |
| `find(node)` / `find(array)` | `find(node)` / `find(array)` |
| `merge(u, v)` / `merge(array)` | `merge(u, v)` / `merge(array)` |
| `elementLabeling` | `element_labeling` |
| `numberOfElements` | `size` (property) |

Notes:

- The constructor takes a single `size` argument; all elements start as
  singletons.
- Scalar `find`/`merge` accept Python integers and return Python integers.
- Bulk `find(nodes)` accepts a 1D `uint64` array and returns a 1D `uint64`
  array of roots of the same length.
- Bulk `merge(edges)` accepts an `(N, 2)` `uint64` array of node-pair edges
  and applies the merges in row order.
- `element_labeling()` returns a `uint64` array of length `size`, each entry
  the (path-compressed) root of that element. Use this when you want the
  final labeling as one array rather than via repeated `find` calls.
- `merge_to(stable, removed)` is also available: it forces `stable`'s root
  to survive the union regardless of rank.
- `reset(n)` reinitialises the structure to `n` singletons, reusing
  capacity where possible.
- The GIL is released around bulk operations, so multiple threads can run
  bulk merges on independent `UnionFind` instances in parallel.

## Skimage

### Marching Cubes

`bioimage-cpp` provides dependency-free isosurface extraction under
`bic.mesh`, including both the topology-resolving Lewiner/MC33 method used by
default in scikit-image and the classic Lorensen lookup-table variant.

```python
import bioimage_cpp as bic

# Extract one object from a label image. `pad=True` closes objects that touch
# the volume boundary by adding a temporary zero-valued halo.
vertices, faces, normals, values = bic.mesh.marching_cubes(
    labels == label_id,
    level=0.5,
    spacing=(z_spacing, y_spacing, x_spacing),
    method="lewiner",
    pad=True,
)
```

The signature follows `skimage.measure.marching_cubes`: `level`, `spacing`,
`gradient_direction`, `step_size`, `allow_degenerate`, `method`, and an
optional boolean `mask` have the same purpose. Coordinates use NumPy
`(z, y, x)` order. Vertices are `float32` at unit spacing and `float64` after
non-unit spacing; faces are consistently `int32`, and normals/values are
`float32`.

Important details:

- Inputs are converted to contiguous `float32` before extraction. Any real
  numeric or boolean input dtype is accepted; complex inputs are rejected.
- `method="lewiner"` resolves ambiguous cases and is the default;
  `method="lorensen"` selects the original 256-case algorithm.
- `spacing` accepts either one positive finite scalar for isotropic data or a
  length-three `(z, y, x)` sequence.
- Normals and local-range values follow scikit-image semantics. As in
  scikit-image, `gradient_direction` reverses face winding without changing
  normals, and anisotropic spacing scales vertices without transforming
  normals.
- `pad=False` matches scikit-image's open-boundary behavior. The additional
  `pad=True` option uses a zero-valued halo and is intended for
  foreground-positive segmentation masks. The iso-level is determined from
  the original unpadded volume.
- Spacing entries must be positive and finite, and faces remain `int32` when
  degenerate faces are removed. Duplicate vertices in a collapsed face are
  merged transitively with the first vertex as representative, and faces that
  still collapse after remapping are discarded. This guarantees in-range face
  indices and intentionally avoids a rare scikit-image negative-index
  remapping quirk. These validation and cleanup choices are intentional
  differences from scikit-image edge cases.

See `development/mesh/check_marching_cubes.py` for reference comparisons and
`development/mesh/benchmark_marching_cubes.py` for reproducible timings.

### Anti-Aliased Resampling

`affine_transform` itself never pre-smooths the input; downsampling without
prior low-pass filtering aliases. `bic.transformation.resample` is a thin
Python wrapper that computes a per-input-axis Gaussian sigma from the
matrix's linear part and pre-smooths via `bic.filters.gaussian_smoothing`
before sampling:

```python
import bioimage_cpp as bic

# Downsample by 2x on each axis, anti-aliased:
matrix = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
small = bic.transformation.resample(
    image, matrix,
    bounding_box=(slice(0, h // 2), slice(0, w // 2)),
    order=1,                # any supported order
    anti_aliasing=True,     # default; uses the heuristic sigma
)

# Explicit sigma (skips the heuristic):
small = bic.transformation.resample(image, matrix, anti_aliasing_sigma=[1.0, 1.0])

# Inspect what the heuristic would pick without resampling:
sigma = bic.transformation.compute_anti_aliasing_sigma(matrix, image.ndim)
```

The heuristic mirrors `skimage.transform.resize`: per input axis,
`sigma = max(0, (||row_of_linear_part|| - 1) / 2)`. Pure rotations
produce all-zero sigma (no smoothing); a uniform 2× downsample produces
`sigma = 0.5` per axis.

### Marker-Controlled Watershed

`bioimage-cpp` ships a marker-controlled watershed entry point that consumes a
node-valued heightmap, analogous to `skimage.segmentation.watershed`. Markers
are mandatory, connectivity is 1 (4-neighbour in 2D, 6-neighbour in 3D), an
optional foreground mask is supported, and tie-breaking on equal heights is
unspecified.

```python
# Heightmap-driven (analogous to skimage.segmentation.watershed)
labels = bic.segmentation.watershed(image, markers, mask=optional_mask)
```

### Sequential Label Relabeling

`bic.segmentation.relabel_sequential` mirrors
`skimage.segmentation.relabel_sequential`: it remaps an integer label array so
all non-zero labels become consecutive starting at `offset`, in sorted order
of the original label values. Label `0` is preserved as background. The
return is a `(relabeled, forward_map, inverse_map)` tuple with the same
indexing semantics as skimage (`forward_map[old] == new`,
`inverse_map[new] == old`).

```python
labels = np.array([0, 5, 10, 5, 0, 200], dtype=np.uint32)
relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(labels)
# relabeled  -> [0, 1, 2, 1, 0, 3]
# forward_map[5] == 1, forward_map[10] == 2, forward_map[200] == 3
# inverse_map -> [0, 5, 10, 200]

# Custom offset works the same way; only label 0 is treated as background.
relabeled, _, _ = bic.segmentation.relabel_sequential(labels, offset=10)
# relabeled  -> [0, 10, 11, 10, 0, 12]
```

Notes:

- Supported input dtypes are `uint32`, `uint64`, `int32`, and `int64`. The
  `relabeled`, `forward_map`, and `inverse_map` arrays all share the input
  dtype (skimage picks the smallest dtype that fits the output range; this
  implementation does not).
- `offset` must be a positive integer (`>= 1`).
- Negative values in signed-dtype inputs are rejected.
- Non-contiguous inputs are copied before entering C++.
- Single-threaded but typically ~7–11× faster than skimage and ~12–28×
  faster than `vigra.analysis.relabelConsecutive` on dense label fields
  (1024² and 128³ arrays with hundreds to hundreds-of-thousands of distinct
  labels). The internal kernel allocates a forward-map LUT of size
  `max(label_field) + 1`, so adversarial inputs with very large `max` and few
  distinct labels will use more memory than a hashmap-based implementation.

### Connected-Components Labeling

`bioimage-cpp` provides pixel-grid connected-components labeling for 2D and
3D arrays, mirroring `skimage.measure.label`. Two non-background pixels share
a component iff there is a path of `connectivity`-neighbour steps between
them along which the input value is constant.

Skimage:

```python
from skimage.measure import label

labels = label(image, background=0, connectivity=None)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

labels = bic.segmentation.label(image, background=0, connectivity=None)
```

Vigra has a closely related entry point on binary / labeled inputs:

```python
import vigra.analysis as va

labels = va.labelMultiArrayWithBackground(
    image, neighborhood="direct", background_value=0,
)
```

`bic.segmentation.label` covers both cases — it labels equal-value runs (as
`skimage.measure.label` does), and for binary masks it agrees with vigra's
`labelMultiArrayWithBackground` partition. `neighborhood="direct"` maps to
`connectivity=1`, `neighborhood="indirect"` maps to `connectivity=image.ndim`.

Important migration notes:

- Supported input dtypes are `bool`, `uint8`, `uint16`, `uint32`, `uint64`,
  `int32`, `int64`. Floating-point inputs are rejected. Non-contiguous
  arrays are copied to contiguous memory.
- `connectivity` is an integer in `[1, image.ndim]`. `1` is orthogonal
  neighbours only (4-connectivity in 2D, 6-connectivity in 3D);
  `image.ndim` enables full diagonal connectivity (8-connectivity in 2D,
  26-connectivity in 3D); `2` in 3D is 18-connectivity. `connectivity=None`
  defaults to `image.ndim`, matching `skimage.measure.label`.
- `background` is the pixel value treated as background. Background pixels
  stay `0` in the output; other equal-valued pixels start at label `1`.
- The output dtype is always `uint64`. `skimage.measure.label` returns
  `intp`; cast if you need bit-for-bit dtype parity.
- Output labels are dense, start at `1`, and are assigned in row-major
  first-occurrence order — same convention as skimage.
- Passing a `bool` array enables an internal fast path that skips
  per-pixel value-equality compares. Convert `uint8` masks to `bool` first
  if your data is binary.
- Only 2D and 3D inputs are supported in v1. `skimage.measure.label`
  accepts arbitrary ndim; loop over slices externally if you need 4D+.
- `return_num=True` from `skimage.measure.label` is not provided. Use
  `int(labels.max())` to get the component count.

Performance characteristics (single-threaded, against `skimage 0.25` and
`vigra 1.11`):

- On integer inputs (`uint8`/`uint16`/…), bioimage-cpp clearly beats both
  skimage and vigra across the tested grid (2D 512²–2048², 3D 64³–128³, all
  connectivities, binary and multi-value). Typical margin is **1.5×–3×**
  faster than skimage and **2×–8×** faster than vigra.
- On `bool` inputs, skimage ships a separately tuned 2D kernel that is very
  fast at large sizes. bioimage-cpp matches it at small/medium sizes and on
  all 3D cases; at 2D 2048² the skimage-bool path is currently ahead by
  roughly 1.7×. Convert to `uint8` to fall back onto the general path if
  you need to win at every 2D size.

## Vigra / fastfilters

### Image Filters

`bioimage-cpp` ships a small Gaussian-derivative filter set under
`bic.filters`. The scope is the "ilastik filter set" exposed by
`fastfilters`, which is also the most-used subset of `vigra.filters`.

Vigra / fastfilters:

```python
import vigra.filters as vf
out = vf.gaussianSmoothing(img, sigma=1.5)
ev = vf.hessianOfGaussianEigenvalues(img, scale=1.5)

import fastfilters as ff
out = ff.gaussianSmoothing(img, sigma=1.5)
ev = ff.hessianOfGaussianEigenvalues(img, scale=1.5)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

out = bic.filters.gaussian_smoothing(img, sigma=1.5)
ev = bic.filters.hessian_of_gaussian_eigenvalues(img, sigma=1.5)
```

Name mapping:

| vigra / fastfilters name | bioimage-cpp name |
| --- | --- |
| `gaussianSmoothing` | `gaussian_smoothing` |
| `gaussianDerivative` | `gaussian_derivative` |
| `gaussianGradientMagnitude` | `gaussian_gradient_magnitude` |
| `laplacianOfGaussian` | `laplacian_of_gaussian` |
| `hessianOfGaussianEigenvalues` | `hessian_of_gaussian_eigenvalues` |
| `structureTensorEigenvalues` | `structure_tensor_eigenvalues` |

Common parameters:

- `sigma` is a positive scalar or a per-axis sequence of length
  `image.ndim`. Anisotropic sigma is supported on every filter.
- `gaussian_derivative` takes an `order` argument that is a scalar or a
  per-axis sequence of ints in `{0, 1, 2}`.
- `structure_tensor_eigenvalues` takes positional `inner_sigma` and
  `outer_sigma` (vigra calls them `innerScale` / `outerScale`).
- `window_size` controls the kernel radius:
  `radius = ceil(window_size * sigma)`. `0.0` (the default) selects the
  vigra-style default `3 + 0.5 * order`. Matches the same-named parameter
  in vigra/fastfilters.

Important differences from vigra and fastfilters:

- Only 2D and 3D scalar (single-channel) inputs are supported in v1.
  Channels and leading batch axes should be looped externally — matches
  fastfilters' convention. Vigra's `taggedView`/`AxisInfo` machinery is
  not reproduced.
- C++ kernels operate on `float32`. `float64` inputs are accepted and the
  output is cast back to `float64`. `uint8` and `uint16` are accepted with
  a `float32` output (the typical ML-feature use case).
- Boundary handling is `mirror` (matches scipy `mode="mirror"` —
  reflection without edge-pixel repeat). Other boundary modes are not
  exposed yet; the C++ layer carries an enum for future tiled processing.
- Eigenvalue outputs have a trailing axis of size `image.ndim`, sorted
  largest → smallest. This matches `fastfilters`. To get vigra's
  ascending order, reverse with `result[..., ::-1]`.
- No IIR / recursive Gaussian, no `convolve` / `recursiveFilter2D`, no
  morphology, no nonlinear diffusion, and no
  non-local means in v1. Use `scipy.ndimage`, `skimage`, or the original
  vigra/fastfilters bindings if you need those.

Implementation notes:

- All six filters are written as portable C++20 scalar code that the
  compiler auto-vectorizes. No SIMD intrinsics, no per-file ISA flags, no
  runtime CPU dispatch, no vendored SIMD library. This keeps the build
  light enough to ship as portable PyPI wheels across Linux/macOS/Windows
  and x86_64/arm64.
- Single-threaded for now. Threading can be added later via
  `detail/threading.hxx::parallel_for_chunks` without changing the
  public API.

### Distance Transforms

`bioimage-cpp` exposes exact binary Euclidean distance transforms under
`bic.distance`. The implementation uses the separable
Felzenszwalb–Huttenlocher algorithm, complexity O(N · ndim), with optional
multithreading across the orthogonal lines of each axis sweep.

SciPy / vigra:

```python
from scipy import ndimage
dist = ndimage.distance_transform_edt(mask, sampling=(2.0, 1.0))

import vigra.filters as vf
vec = vf.vectorDistanceTransform(mask)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

# One call can return any combination of distances, feature indices, and
# difference vectors — the C++ kernel computes them in a single sweep.
dist = bic.distance.distance_transform(mask, sampling=(2.0, 1.0))
dist, idx, vec = bic.distance.distance_transform(
    mask,
    sampling=(2.0, 1.0),
    return_distances=True,
    return_indices=True,
    return_vectors=True,
)

# Short alias kept for parity with vigra; equivalent to the call above with
# return_distances=False, return_indices=False, return_vectors=True.
vec = bic.distance.vector_difference_transform(mask, sampling=(2.0, 1.0))
```

Name mapping:

| scipy / vigra name | bioimage-cpp name |
| --- | --- |
| `scipy.ndimage.distance_transform_edt` | `distance_transform` |
| `vigra.filters.vectorDistanceTransform` | `vector_difference_transform` |

Important differences:

- Distance-valued outputs are `float32`, not SciPy's `float64`. Indices are
  `int32` with shape `(ndim, *mask.shape)` (matches SciPy's layout). Vectors
  are `float32` with shape `(*mask.shape, ndim)`; components are sampled
  displacements `(feature_coord - pixel_coord) * sampling[ax]` per axis.
- `distance_transform` follows SciPy's binary convention: nonzero values are
  foreground and distances are measured to the nearest zero-valued element.
  `bool` and `uint8` C-contiguous inputs are fast-pathed without a copy;
  other dtypes are converted via `array != 0`.
- A single `distance_transform` call can return any non-empty subset of
  `distances`, `indices`, and `vectors` via the corresponding
  `return_distances` / `return_indices` / `return_vectors` flags. The result
  is the array itself when only one output is requested, otherwise a tuple
  in `(distances, indices, vectors)` order with omitted entries skipped.
- Pre-allocated output buffers are supported via the `distances=`,
  `indices=`, and `vectors=` keyword arguments. They must be C-contiguous,
  writable, of the documented shape and dtype, and are written into in
  place. Pre-allocated outputs are excluded from the return value (matching
  SciPy's convention); the call returns `None` if every requested output
  was preallocated.
- `number_of_threads` selects the thread count for the per-axis sweep.
  `1` (the default) is single-threaded; `0` uses
  `std::thread::hardware_concurrency()`; positive values pin an explicit
  count. Output is deterministic and bitwise identical across thread counts.
- For an all-foreground input (no zero-valued elements), the result matches
  SciPy: distances and indices report a virtual background point at
  axis-0 coordinate `-1` and `0` on all other axes. The first row of
  `indices` will then contain `-1` everywhere.

### Non-Maximum Distance Suppression

`nifty.filters.nonMaximumDistanceSuppression` filters a set of candidate
points using a distance map: each point's suppression radius is the distance
value at its own location, and from every group of points that fall within
one another's radius only the one with the largest distance value is kept.
`bioimage-cpp` exposes the same algorithm as
`bic.distance.non_maximum_distance_suppression`.

nifty:

```python
from nifty.filters import nonMaximumDistanceSuppression

# distanceMap: float32 array; points: uint64 array of shape (N, ndim)
kept = nonMaximumDistanceSuppression(distanceMap, points)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

kept = bic.distance.non_maximum_distance_suppression(distance_map, points)
```

Name mapping:

| nifty name | bioimage-cpp name |
| --- | --- |
| `nifty.filters.nonMaximumDistanceSuppression` | `non_maximum_distance_suppression` |

Important differences:

- Snake_case naming, consistent with the rest of `bic.distance`.
- `points` may be `int64`, `uint64`, `int32`, or `uint32`; the returned array
  has shape `(K, ndim)` and preserves the input `points` dtype (nifty always
  returned `uint64`). Output rows are the retained points in ascending
  input-index order.
- `distance_map` is coerced to C-contiguous `float32` if needed. The
  per-point radius is dynamic (the distance value at each point), matching
  nifty; there is no fixed-radius mode.
- The algorithm is otherwise identical to nifty, including its float
  arithmetic, so results match element-for-element. It is O(N²) in time but
  uses only O(`number_of_threads` · N) auxiliary memory (no dense N×N matrix),
  and `number_of_threads` parallelizes the pairwise evaluation. Still threshold
  the distance map first to keep the candidate count modest.

## scikit-fmm

`scikit-fmm` computes geodesic distances on regular grids with the fast
marching method. `bioimage-cpp` groups geodesic distances under `bic.distance`
with two operations — a **distance field** to a set of sources, and the full
**pairwise distance matrix** between a set of points — for two geometry types:
regular-grid **masks** (the scikit-fmm equivalent) and triangle **meshes**
(which scikit-fmm does not support; the reference for those is the exact MMP
algorithm in `pygeodesic`).

Both are solved with a first-order fast marching method (Godunov/Eikonal on the
grid, Kimmel–Sethian on triangles). Masks match scikit-fmm's `order=1` scheme;
mesh distances are first-order approximations of the exact surface geodesics.

scikit-fmm:

```python
import numpy as np
import skfmm

# Distance from a set of seed voxels, constrained to a mask (domain).
phi = np.ones(mask.shape)
phi[tuple(sources.T)] = -1            # zero contour marks the seeds
phi = np.ma.MaskedArray(phi, ~mask)   # obstacles / outside-domain
field = np.abs(skfmm.distance(phi, dx=sampling))
# Weighted (travel-time) variant:
tt = skfmm.travel_time(phi, speed, dx=sampling)
```

bioimage-cpp:

```python
import bioimage_cpp as bic

# masks (regular grid); sources/points are (n, ndim) int64 voxel coordinates
field = bic.distance.geodesic_distance_field(mask, sources, sampling=None,
                                             speed=None)      # -> mask.shape, float64
matrix = bic.distance.geodesic_distances(mask, points)       # -> (N, N) float64

# optional per-axis gradient of the field (like vector_difference_transform)
field, grad = bic.distance.geodesic_distance_field(mask, sources,
                                                   return_gradient=True)
# grad: (*mask.shape, ndim) float32, d(field)/d(axis), points away from source
gradient = bic.distance.geodesic_gradient_field(mask, sources)  # just the gradient

# surfaces (triangle mesh); sources/points are 1-D int64 vertex indices
field = bic.distance.geodesic_distance_field_mesh(vertices, faces, sources,
                                                  speed=None) # -> (n_vertices,) float64
matrix = bic.distance.geodesic_distances_mesh(vertices, faces, points)  # -> (N, N) float64
```

Name mapping:

| scikit-fmm / pygeodesic | bioimage-cpp name |
| --- | --- |
| `skfmm.distance` / `skfmm.travel_time` (from seed voxels, masked) | `geodesic_distance_field` |
| gradient of the distance field (per-axis) | `geodesic_distance_field(..., return_gradient=True)` / `geodesic_gradient_field` |
| pairwise via repeated `skfmm.distance` | `geodesic_distances` |
| `pygeodesic … geodesicDistances(sources, None)` | `geodesic_distance_field_mesh` |
| `pygeodesic … geodesicDistances` (pairwise) | `geodesic_distances_mesh` |

Important differences:

- Explicit, geometry-specific functions rather than a level-set encoding: the
  domain is passed directly as a `mask` (nonzero = inside) or as a
  `(vertices, faces)` mesh, and the sources are given as coordinates / vertex
  indices instead of being baked into a signed `phi`.
- Outputs are `float64`. Voxels outside the mask and points/vertices
  unreachable from any source are `+inf`; pairwise matrices are symmetric with
  a zero diagonal.
- `sampling` (per-axis voxel spacing, scalar or per-axis) maps to scikit-fmm's
  `dx` and applies to masks only — meshes carry real vertex coordinates.
- `speed` is optional (`None` = unit-speed geodesic distance). When supplied it
  generalizes to a weighted travel time, matching `skfmm.travel_time`; for
  masks it has the mask's shape, for meshes it is per-vertex.
- `number_of_threads` follows the `bic.distance` convention (`1` default,
  `0` = hardware concurrency); the pairwise solves parallelize over sources.
- `geodesic_distance_field(..., return_gradient=True)` also returns the per-axis
  gradient `∇T` of the field (or use `geodesic_gradient_field` for the gradient
  alone), analogous to `vector_difference_transform`. It is `float32` with shape
  `(*mask.shape, ndim)` (channel-last, NumPy axis order); component `i` is
  `d(field)/d(axis_i)`, pointing **away** from the nearest source with
  `‖∇T‖ ≈ 1/speed`. Negate it to trace back toward the source — e.g. feed
  `-grad` (transposed to channel-first) to `bic.flow.compute_flow_density`. It
  is zero at sources, background, and unreachable voxels. Masks only.
- Mesh geodesics are surface (2-manifold) distances from the Kimmel–Sethian
  triangle fast-marching method — a first-order approximation (a few % mean
  error vs the exact `pygeodesic` MMP algorithm, larger on very obtuse
  triangles, since obtuse-angle unfolding is not done yet). Like the mask
  solver it slightly overestimates. See `development/distance/` for the
  reference oracles and `benchmark_geodesic.py` for timings.

## I/O and Build Dependencies

`bioimage-cpp` intentionally does not replace nifty or affogato I/O helpers.
Load TIFF, HDF5, zarr, N5, OME-NGFF, and related formats with existing Python
libraries, then pass NumPy arrays to `bioimage-cpp`.

The package is designed for small PyPI wheels and does not depend on nifty,
vigra, HDF5, z5, xtensor, pybind11, or other large C++ libraries.
