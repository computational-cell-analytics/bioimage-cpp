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

## Edge-Weighted Watershed

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

## Fusion Moves (Multicut)

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

objective = bic.graph.MulticutObjective(graph, edge_costs)
solver = bic.graph.FusionMoveMulticut(
    proposal_generator=bic.graph.WatershedProposalGenerator(
        sigma=1.0, n_seeds_fraction=0.1, seed=0,
    ),
    sub_solver=bic.graph.GreedyAdditiveMulticut(),
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

## Lifted Multicut

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

objective = bic.graph.LiftedMulticutObjective(
    graph,
    edge_costs,
    lifted_uvs=lifted_uvs,
    lifted_costs=lifted_costs,
    bfs_distance=3,  # optional: also insert zero-weight lifted edges within k hops
)
labels = bic.graph.LiftedGreedyAdditiveMulticut().optimize(objective)
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
solver = bic.graph.FusionMoveLiftedMulticut(
    proposal_generator=bic.graph.WatershedProposalGenerator(
        sigma=1.0, n_seeds_fraction=0.1, seed=0,
    ),
    sub_solver=bic.graph.LiftedKernighanLinMulticut(number_of_outer_iterations=3),
    number_of_iterations=10,
    stop_if_no_improvement=4,
    number_of_threads=4,
)
labels = solver.optimize(objective)
```

A typical warm-started solve combines greedy and KL:

```python
solver = bic.graph.LiftedChainedSolvers([
    bic.graph.LiftedGreedyAdditiveMulticut(),
    bic.graph.LiftedKernighanLinMulticut(number_of_outer_iterations=10),
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

### Building a lifted multicut problem from affinities

For the common case of lifted multicut on a watershed over-segmentation,
nifty offers `nifty.graph.rag.computeLiftedEdgesFromRagAndOffsets` (lifted
edge discovery) and per-channel affinity accumulators. bioimage-cpp exposes
two focused helpers that cover the same workflow:

```python
# Discover lifted edges implied by long-range affinity offsets. 1-hop offsets
# are skipped automatically, so the full offset list can be passed in.
lifted_uvs = bic.graph.lifted_edges_from_affinities(
    rag, oversegmentation, offsets, number_of_threads=0,
)

# Accumulate (mean, size) statistics per lifted edge. Pixel pairs whose
# (u, v) does not appear in `lifted_uvs` are skipped, so local edges are
# never contaminated with long-range affinities.
lifted_features = bic.graph.lifted_affinity_features(
    oversegmentation, affinities, offsets, lifted_uvs,
    number_of_threads=0,
)
# For the 12-column feature set (mean, median, std, min, max, percentiles, size):
lifted_features = bic.graph.lifted_affinity_features_complex(...)
```

The output column conventions match the local-edge variants
(`SIMPLE_EDGE_FEATURE_NAMES`, `COMPLEX_EDGE_FEATURE_NAMES`).

End-to-end pipeline (also in `examples/segmentation/lifted_multicut_from_affinities.py`):

```python
rag = bic.graph.region_adjacency_graph(oversegmentation)
local_costs = local_threshold - bic.graph.affinity_features(
    rag, oversegmentation, direct_affinities, direct_offsets,
)[:, 0]
lifted_uvs = bic.graph.lifted_edges_from_affinities(
    rag, oversegmentation, long_range_offsets,
)
lifted_costs = lifted_threshold - bic.graph.lifted_affinity_features(
    oversegmentation, long_range_affinities, long_range_offsets, lifted_uvs,
)[:, 0]
objective = bic.graph.LiftedMulticutObjective(
    rag, local_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs,
)
```

## External Problem Instances

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
graph, costs = bic.graph.load_multicut_problem(sample="A", size="small")
# Or just the underlying arrays
uv_ids, costs = bic.graph.load_multicut_problem_data(sample="B", size="medium")
# Or the cached file path
path = bic.graph.multicut_problem_path(sample="C", size="medium")
```

Valid samples are `"A"`, `"B"`, `"C"`; valid sizes are `"small"` and
`"medium"`. The legacy `load_external_multicut_problem` /
`load_external_multicut_problem_data` / `external_multicut_problem_path`
shims default to sample A, size small and continue to honor the
`BIOIMAGE_CPP_EXTERNAL_MULTICUT_PATH` and
`BIOIMAGE_CPP_EXTERNAL_MULTICUT_CACHE` environment variables.

Lifted multicut problems (2D ISBI slice and full 3D volume, built by
`examples/segmentation/serialize_lifted_problem.py`):

```python
problem = bic.graph.load_lifted_multicut_problem(size="2d")
# Fields: n_nodes (int), local_uvs, local_costs, lifted_uvs, lifted_costs.
graph = bic.graph.UndirectedGraph.from_edges(problem.n_nodes, problem.local_uvs)
objective = bic.graph.LiftedMulticutObjective(
    graph,
    problem.local_costs,
    lifted_uvs=problem.lifted_uvs,
    lifted_costs=problem.lifted_costs,
)
```

Notes:

- Every download is integrity-checked against a SHA256 in the registry; a
  corrupted cache file is detected on the next `load_*` call.
- Downloads are lazy: nothing happens until you call a loader. Re-runs are
  free (the cached file is reused).
- For air-gapped use, fetch the file once on a machine with network access
  and copy `~/.cache/bioimage-cpp/<filename>` to the same path on the target
  machine.

## Breadth-First Search

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
