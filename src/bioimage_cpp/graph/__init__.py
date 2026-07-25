"""Graph data structures and graph-level algorithms.

Top-level surface:

- Graph structures: :class:`UndirectedGraph`, :class:`ContractionGraph`,
  :class:`GridGraph2D`, :class:`GridGraph3D`,
  :class:`RegionAdjacencyGraph`.
- Constructors: :func:`undirected_graph`, :func:`grid_graph`,
  :func:`region_adjacency_graph`.
- Algorithms: :func:`connected_components`, :func:`breadth_first_search`,
  :func:`edge_weighted_watershed`, :func:`project_node_labels_to_pixels`.

Algorithm domains live in dedicated submodules:

- :mod:`bioimage_cpp.graph.multicut` — multicut objective and solvers,
  proposal generators, multicut problem loaders.
- :mod:`bioimage_cpp.graph.lifted_multicut` — lifted multicut objective and
  solvers, lifted problem loaders.
- :mod:`bioimage_cpp.graph.mutex_watershed` — mutex watershed clustering
  (with and without semantic constraints).
- :mod:`bioimage_cpp.graph.features` — edge-feature accumulation on RAGs and
  grid graphs.
- :mod:`bioimage_cpp.graph.distributed` — low-level per-block extraction and
  whole-volume merge primitives for building RAGs and edge features on volumes
  too large to hold in memory (block iteration and I/O are left to the caller).

Thread safety
-------------
All graph types (:class:`UndirectedGraph`, :class:`GridGraph2D`,
:class:`GridGraph3D`, :class:`RegionAdjacencyGraph`) build their internal
adjacency representation *lazily*, on the first call that reads it. The
built-in multi-threaded algorithms freeze the graph internally before fanning
out, so passing a graph straight into them is safe and needs no extra step.

If you build a graph yourself and then share it across **your own** threads
(reading adjacency, running a BFS, etc. concurrently), call ``graph.freeze()``
once on the construction thread first: the lazy build is not thread-safe, and
racing the first read across threads corrupts the adjacency. ``freeze()`` is a
no-op on an already-built graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import _core
from .._validation import strict_integer_array, strict_offsets
from ._shared import (
    _as_1d_array,
    _as_coordinate_array,
    _as_node_array,
    _as_offset_array,
    _as_serialization_array,
    _as_shape,
    _as_uv_array,
    _normalize_labels,
    _normalize_number_of_threads,
    _resolve_weight_dtype,
)


class UndirectedGraph(_core.UndirectedGraph):
    """Undirected graph with consecutive node and edge ids.

    Nodes are fixed at construction and addressed by ids
    ``0 .. number_of_nodes - 1``. Edges are inserted lazily and receive
    consecutive ids in insertion order. Re-inserting an existing undirected edge
    returns the existing edge id.

    The adjacency representation is built lazily on first use. Before sharing a
    freshly built graph across threads of your own, call :meth:`freeze` once on
    the construction thread — see the module-level "Thread safety" note. The
    built-in multi-threaded algorithms already freeze internally.
    """

    def insert_edges(self, uvs):
        return super().insert_edges(_as_uv_array(uvs, "uvs"))

    def find_edges(self, uvs):
        return super().find_edges(_as_uv_array(uvs, "uvs"))

    def insertEdges(self, uvs):
        return self.insert_edges(uvs)

    def findEdges(self, uvs):
        return self.find_edges(uvs)

    def extract_subgraph_from_nodes(self, nodes):
        return super().extract_subgraph_from_nodes(_as_node_array(nodes, "nodes"))

    def edges_from_node_list(self, nodes):
        return super().edges_from_node_list(_as_node_array(nodes, "nodes"))

    def extractSubgraphFromNodes(self, nodes):
        return self.extract_subgraph_from_nodes(nodes)

    def edgesFromNodeList(self, nodes):
        return self.edges_from_node_list(nodes)

    @classmethod
    def from_edges(cls, number_of_nodes: int, uvs):
        return cls(int(number_of_nodes), _as_uv_array(uvs, "uvs"), False)

    @classmethod
    def from_unique_edges(cls, number_of_nodes: int, uvs):
        """Bulk-construct from a pre-deduplicated ``(n, 2)`` edge array.

        The caller asserts that no undirected ``(u, v)`` pair appears twice and
        that ``u != v`` in every row; edges receive ids matching their position
        in ``uvs``. This skips the per-edge dedup of :meth:`from_edges` and is
        the fast path for e.g. the merged edge set of
        :func:`bioimage_cpp.graph.distributed.merge_edges`.
        """
        return cls(int(number_of_nodes), _as_uv_array(uvs, "uvs"), True)

    @classmethod
    def deserialize(cls, serialization):
        serialization = _as_serialization_array(serialization)
        number_of_nodes = int(serialization[0])
        number_of_edges = int(serialization[1])
        uvs = serialization[2:].reshape(number_of_edges, 2)
        return cls.from_edges(number_of_nodes, uvs)


@dataclass(frozen=True)
class ContractionResult:
    """Materialized graph and values from a :class:`ContractionGraph`.

    Rows in each node or edge value array match the corresponding ids in
    ``graph``. ``node_mapping`` and ``edge_mapping`` map ids from the input
    graph to these dense ids. Multiple input ids can map to one output id. An
    entry is ``-1`` when no materialized item represents the input item.
    """

    graph: UndirectedGraph
    node_values: dict[str, np.ndarray]
    edge_values: dict[str, np.ndarray]
    node_mapping: np.ndarray
    edge_mapping: np.ndarray


_CONTRACTION_REDUCTIONS = {
    "sum": 0,
    "mean": 1,
    "min": 2,
    "max": 3,
}

_CONTRACTION_PARALLEL_EDGES = {
    "merge": 0,
    "keep": 1,
}


def _contraction_values(values, name: str, expected_items: int) -> np.ndarray:
    array = _resolve_weight_dtype(values, name)
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if array.shape[0] != expected_items:
        raise ValueError(
            f"{name}.shape[0] must be {expected_items}, got {array.shape[0]}"
        )
    return np.ascontiguousarray(array)


def _contraction_reduction(reduction: str) -> int:
    try:
        return _CONTRACTION_REDUCTIONS[reduction]
    except KeyError as error:
        supported = ", ".join(_CONTRACTION_REDUCTIONS)
        raise ValueError(
            f"reduction must be one of ({supported}), got {reduction!r}"
        ) from error


def _contraction_parallel_edges(parallel_edges: str) -> int:
    try:
        return _CONTRACTION_PARALLEL_EDGES[parallel_edges]
    except KeyError as error:
        supported = ", ".join(_CONTRACTION_PARALLEL_EDGES)
        raise ValueError(
            f"parallel_edges must be one of ({supported}), got {parallel_edges!r}"
        ) from error


class ContractionGraph:
    """Mutable graph for edge contractions and value reduction.

    The graph copies the input topology. Node and edge ids stay stable while
    they are active. Mutations can leave gaps in the id range. The input graph
    must be simple.

    Register value maps before the first mutation. A map must have shape
    ``(number_of_items, ...)`` and a floating dtype. ``float32`` and
    ``float64`` are preserved. The reduction is component-wise:

    - ``"sum"`` adds represented values.
    - ``"mean"`` computes the mean over represented input items.
    - ``"min"`` and ``"max"`` select the extrema.

    Node maps reduce when an edge contraction merges its endpoints. Edge maps
    reduce when two surviving edges are folded into one. The contracted edge
    has no surviving target, so its value is discarded.

    ``parallel_edges="merge"`` folds parallel edges during each mutation.
    ``parallel_edges="keep"`` keeps them separate in the mutable graph.
    :meth:`materialize` always creates a simple graph and folds remaining
    parallel edges only in the returned snapshot.

    This class is not thread-safe.
    """

    def __init__(
        self,
        graph: UndirectedGraph,
        *,
        parallel_edges: str = "merge",
    ):
        parallel_edges_code = _contraction_parallel_edges(parallel_edges)
        self._core = _core._ContractionGraph(graph, parallel_edges_code)
        self._original_number_of_nodes = int(graph.number_of_nodes)
        self._original_number_of_edges = int(graph.number_of_edges)
        self._parallel_edges = parallel_edges
        self._mutation_started = False

    @property
    def parallel_edges(self) -> str:
        """Policy for parallel edges created by mutations."""
        return self._parallel_edges

    @property
    def number_of_nodes(self) -> int:
        """Number of active nodes."""
        return int(self._core.number_of_nodes)

    @property
    def number_of_edges(self) -> int:
        """Number of active edges."""
        return int(self._core.number_of_edges)

    def nodes(self) -> np.ndarray:
        """Return active stable node ids in ascending order."""
        return np.asarray(self._core.nodes(), dtype=np.uint64)

    def edges(self) -> np.ndarray:
        """Return active stable edge ids in ascending order."""
        return np.asarray(self._core.edges(), dtype=np.uint64)

    def uv(self, edge: int) -> tuple[int, int]:
        u, v = self._core.uv(int(edge))
        return int(u), int(v)

    def find_edge(self, u: int, v: int) -> int:
        """Return the smallest matching active edge id, or ``-1``."""
        return int(self._core.find_edge(int(u), int(v)))

    def find_edges(self, u: int, v: int) -> np.ndarray:
        """Return all matching active edge ids in ascending order."""
        return np.asarray(self._core.find_edges(int(u), int(v)), dtype=np.uint64)

    def node_adjacency(self, node: int) -> np.ndarray:
        """Return ``(neighbor, edge)`` rows for an active node."""
        return self._core.node_adjacency(int(node))

    def degree(self, node: int) -> int:
        return int(self._core.degree(int(node)))

    def is_node_active(self, node: int) -> bool:
        return bool(self._core.is_node_active(int(node)))

    def is_edge_active(self, edge: int) -> bool:
        return bool(self._core.is_edge_active(int(edge)))

    def representative(self, original_node: int) -> int:
        """Return the current representative of an input node."""
        return int(self._core.representative(int(original_node)))

    def add_node_values(self, name: str, values, *, reduction: str) -> None:
        """Register a node value map before the first mutation."""
        if self._mutation_started:
            raise ValueError(
                "value maps must be registered before the first graph mutation"
            )
        array = _contraction_values(
            values,
            name,
            self._original_number_of_nodes,
        )
        code = _contraction_reduction(reduction)
        getattr(self._core, f"_add_node_values_{array.dtype.name}")(
            str(name), array, code
        )

    def add_edge_values(self, name: str, values, *, reduction: str) -> None:
        """Register an edge value map before the first mutation."""
        if self._mutation_started:
            raise ValueError(
                "value maps must be registered before the first graph mutation"
            )
        array = _contraction_values(
            values,
            name,
            self._original_number_of_edges,
        )
        code = _contraction_reduction(reduction)
        getattr(self._core, f"_add_edge_values_{array.dtype.name}")(
            str(name), array, code
        )

    def contract_edge(self, edge: int, *, keep_node: int | None = None) -> int:
        """Contract an active edge and return the retained node id."""
        keep = None if keep_node is None else int(keep_node)
        result = int(self._core.contract_edge(int(edge), keep))
        self._mutation_started = True
        return result

    def erase_edge(self, edge: int) -> None:
        """Erase an active edge without merging its endpoints."""
        self._core.erase_edge(int(edge))
        self._mutation_started = True

    def suppress_node(self, node: int) -> int:
        """Suppress a degree-2 node and return the replacement edge id.

        The operation removes the node and its two incident edges. It connects
        the two neighbors and reduces both edge value rows into the replacement
        edge. The suppressed node value has no target and is discarded.
        """
        result = int(self._core.suppress_node(int(node)))
        self._mutation_started = True
        return result

    def node_value(self, name: str, node: int) -> np.ndarray:
        """Return a copy of one active node value."""
        return self._core.node_value(str(name), int(node))

    def edge_value(self, name: str, edge: int) -> np.ndarray:
        """Return a copy of one active edge value."""
        return self._core.edge_value(str(name), int(edge))

    def active_node_values(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return active node ids and their value rows."""
        return self._core.active_node_values(str(name))

    def active_edge_values(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return active edge ids and their value rows."""
        return self._core.active_edge_values(str(name))

    def materialize(self) -> ContractionResult:
        """Create a dense graph and aligned copies of all registered values.

        The output graph is simple. The snapshot folds active parallel edges
        with each edge map's registered reduction.

        Materialization does not change this contraction graph. The result
        remains valid after later mutations.
        """
        (
            number_of_nodes,
            uvs,
            node_values,
            edge_values,
            node_mapping,
            edge_mapping,
        ) = self._core.materialize()
        graph = UndirectedGraph.from_unique_edges(int(number_of_nodes), uvs)
        return ContractionResult(
            graph=graph,
            node_values=dict(node_values),
            edge_values=dict(edge_values),
            node_mapping=node_mapping,
            edge_mapping=edge_mapping,
        )


def _normalize_projection_offsets(offsets, ndim: int) -> list[list[int]]:
    return [list(offset) for offset in strict_offsets(offsets, ndim)]


def _normalize_projection_strides(strides, ndim: int) -> list[int]:
    values_array = strict_integer_array(strides, "strides", dtype=np.uint64, ndim=1)
    values = [int(v) for v in values_array]
    if len(values) != ndim:
        raise ValueError(
            f"strides must have length {ndim}, got length={len(values)}"
        )
    if any(v <= 0 for v in values):
        raise ValueError("strides must be positive")
    return values


def _normalize_projection_mask(mask, n_offsets: int, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(mask)
    expected = (n_offsets, *shape)
    if array.shape != expected:
        raise ValueError(
            f"mask shape must be {expected}, got shape={array.shape}"
        )
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


class GridGraph2D(_core.GridGraph2D):
    """Regular 2D nearest-neighbor grid graph.

    Nodes use C-order ids for ``shape=(y, x)``. Edge ids are deterministic:
    all y-axis edges first, then all x-axis edges.
    """

    def __init__(self, shape):
        super().__init__(_as_shape(shape, 2))

    def node_id(self, coordinate):
        return super().node_id(_as_coordinate_array(coordinate, 2, "coordinate"))

    def nodeId(self, coordinate):
        return self.node_id(coordinate)

    def offset_target(self, node: int, offset):
        return super().offset_target(int(node), _as_offset_array(offset, 2, "offset"))

    def offsetTarget(self, node: int, offset):
        return self.offset_target(node, offset)

    def project_edge_ids_to_pixels(self) -> np.ndarray:
        """Project edge ids onto a per-pixel array of shape ``(2, *shape)``.

        Returns an ``int64`` array initialised to ``-1``. For every grid edge
        with spanning axis ``d`` and pivot (smaller-endpoint) coordinate
        ``c``, sets ``out[d, *c] = edge_id``. Slots where the pivot lies at
        the last index of axis ``d`` remain ``-1``.
        """
        return super().project_edge_ids_to_pixels()

    def project_edge_ids_to_pixels_with_offsets(
        self,
        offsets,
        *,
        strides=None,
        mask=None,
    ) -> tuple[np.ndarray, int]:
        """Enumerate lifted edges defined by per-channel ``offsets``.

        Walks ``(offset_idx, *coord)`` in C-order over
        ``(len(offsets), *shape)``. For every coord whose target
        ``coord + offsets[offset_idx]`` is in bounds (and survives the
        optional ``strides`` or ``mask`` filter), writes a sequential
        counter starting at 0; rejected slots get ``-1``.

        ``strides`` and ``mask`` are mutually exclusive. ``strides`` keeps
        only coords where every ``coord[d] % strides[d] == 0``; ``mask``
        keeps only coords where ``mask[offset_idx, *coord]`` is true.

        Returns ``(array, n_valid)`` — the ``int64`` array and the total
        number of valid entries written. The counter is **not** a graph
        edge id; it indexes into the implicit array of lifted edges.
        """
        if strides is not None and mask is not None:
            raise ValueError("strides and mask cannot be given together")
        offsets = _normalize_projection_offsets(offsets, 2)
        normalized_strides = (
            None if strides is None else _normalize_projection_strides(strides, 2)
        )
        normalized_mask = (
            None
            if mask is None
            else _normalize_projection_mask(mask, len(offsets), tuple(self.shape))
        )
        return super().project_edge_ids_to_pixels_with_offsets(
            offsets, normalized_strides, normalized_mask
        )


class GridGraph3D(_core.GridGraph3D):
    """Regular 3D nearest-neighbor grid graph.

    Nodes use C-order ids for ``shape=(z, y, x)``. Edge ids are deterministic:
    all z-axis edges first, then y-axis edges, then x-axis edges.
    """

    def __init__(self, shape):
        super().__init__(_as_shape(shape, 3))

    def node_id(self, coordinate):
        return super().node_id(_as_coordinate_array(coordinate, 3, "coordinate"))

    def nodeId(self, coordinate):
        return self.node_id(coordinate)

    def offset_target(self, node: int, offset):
        return super().offset_target(int(node), _as_offset_array(offset, 3, "offset"))

    def offsetTarget(self, node: int, offset):
        return self.offset_target(node, offset)

    def project_edge_ids_to_pixels(self) -> np.ndarray:
        """Project edge ids onto a per-pixel array of shape ``(3, *shape)``.

        See :meth:`GridGraph2D.project_edge_ids_to_pixels` for the
        semantics; the only difference is the leading dimension matches the
        graph rank.
        """
        return super().project_edge_ids_to_pixels()

    def project_edge_ids_to_pixels_with_offsets(
        self,
        offsets,
        *,
        strides=None,
        mask=None,
    ) -> tuple[np.ndarray, int]:
        """Enumerate lifted edges defined by per-channel ``offsets``.

        See :meth:`GridGraph2D.project_edge_ids_to_pixels_with_offsets` for
        the semantics; the only difference is the input shape.
        """
        if strides is not None and mask is not None:
            raise ValueError("strides and mask cannot be given together")
        offsets = _normalize_projection_offsets(offsets, 3)
        normalized_strides = (
            None if strides is None else _normalize_projection_strides(strides, 3)
        )
        normalized_mask = (
            None
            if mask is None
            else _normalize_projection_mask(mask, len(offsets), tuple(self.shape))
        )
        return super().project_edge_ids_to_pixels_with_offsets(
            offsets, normalized_strides, normalized_mask
        )


RegionAdjacencyGraph = _core.RegionAdjacencyGraph


class RagCoordinates:
    """Mapping from RAG edges to the pixel coordinates of region boundaries.

    Created via :func:`rag_coordinates`. The label volume is scanned once at
    construction and the per-edge boundary coordinates are cached, so the same
    object can be reused across many :meth:`edges_to_volume` calls.

    Each boundary "contact" — a pair of directly adjacent pixels with different
    labels — contributes two coordinates to its edge: the lower-coordinate pixel
    and its ``+axis`` neighbor. The ``edge_direction`` argument selects which
    side(s) are reported: ``0`` = both (default), ``1`` = lower-side only,
    ``2`` = higher-side only.
    """

    def __init__(self, core):
        self._core = core

    @property
    def ndim(self) -> int:
        return self._core.ndim

    @property
    def shape(self):
        return self._core.shape

    @property
    def number_of_edges(self) -> int:
        return self._core.number_of_edges

    def storage_lengths(self) -> np.ndarray:
        """Number of stored boundary points per edge (``2 * n_contacts``)."""
        return self._core.storage_lengths()

    def edge_coordinates(self, edge: int, *, edge_direction: int = 0) -> np.ndarray:
        """Boundary coordinates of one edge as an ``(n_points, ndim)`` array."""
        if edge_direction not in (0, 1, 2):
            raise ValueError("edge_direction must be 0, 1, or 2")
        return self._core.edge_coordinates(int(edge), int(edge_direction))

    def edges_to_volume(
        self,
        edge_values,
        *,
        edge_direction: int = 0,
        ignore_value=0,
    ) -> np.ndarray:
        """Paint per-edge values onto a volume of the label shape.

        Every pixel is set to ``ignore_value`` and then each selected boundary
        point receives its edge's value. ``edge_values`` is a 1D array of length
        ``number_of_edges``; supported dtypes are ``float32``, ``float64``,
        ``uint32`` and ``uint64``. The returned volume has the same dtype.

        Painting is in ascending edge id, so where several edges' boundaries
        share a pixel the highest edge id wins (deterministic).
        """
        if edge_direction not in (0, 1, 2):
            raise ValueError("edge_direction must be 0, 1, or 2")
        values = np.asarray(edge_values)
        try:
            method_name = _EDGES_TO_VOLUME_BY_DTYPE[values.dtype]
        except KeyError as error:
            supported = ", ".join(str(dtype) for dtype in _EDGES_TO_VOLUME_BY_DTYPE)
            raise TypeError(
                f"edge_values must have one of dtypes ({supported}), "
                f"got dtype={values.dtype}"
            ) from error
        if values.ndim != 1:
            raise ValueError("edge_values must be a 1D array")
        if values.shape[0] != self.number_of_edges:
            raise ValueError("edge_values length must match number_of_edges")
        method = getattr(self._core, method_name)
        return method(
            np.ascontiguousarray(values),
            int(edge_direction),
            values.dtype.type(ignore_value),
        )

    def storageLengths(self) -> np.ndarray:
        return self.storage_lengths()

    def edgeCoordinates(self, edge: int, *, edge_direction: int = 0) -> np.ndarray:
        return self.edge_coordinates(edge, edge_direction=edge_direction)

    def edgesToVolume(self, edge_values, **kwargs) -> np.ndarray:
        return self.edges_to_volume(edge_values, **kwargs)


def undirected_graph(number_of_nodes: int) -> UndirectedGraph:
    """Create an :class:`UndirectedGraph`."""
    return UndirectedGraph(number_of_nodes)


def grid_graph(shape):
    """Create a regular 2D or 3D nearest-neighbor grid graph."""
    ndim = np.asarray(shape).ndim
    if ndim != 1:
        raise ValueError("shape must be a 1D sequence")
    n_axes = len(shape)
    if n_axes == 2:
        return GridGraph2D(shape)
    if n_axes == 3:
        return GridGraph3D(shape)
    raise ValueError(f"shape must have length 2 or 3, got length={n_axes}")


_EDGE_WEIGHTED_WATERSHED_BY_DTYPE = {
    (np.dtype("float32"), np.dtype("uint32")): _core._edge_weighted_watershed_float32_uint32,
    (np.dtype("float32"), np.dtype("uint64")): _core._edge_weighted_watershed_float32_uint64,
    (np.dtype("float32"), np.dtype("int32")): _core._edge_weighted_watershed_float32_int32,
    (np.dtype("float32"), np.dtype("int64")): _core._edge_weighted_watershed_float32_int64,
    (np.dtype("float64"), np.dtype("uint32")): _core._edge_weighted_watershed_float64_uint32,
    (np.dtype("float64"), np.dtype("uint64")): _core._edge_weighted_watershed_float64_uint64,
    (np.dtype("float64"), np.dtype("int32")): _core._edge_weighted_watershed_float64_int32,
    (np.dtype("float64"), np.dtype("int64")): _core._edge_weighted_watershed_float64_int64,
}


_REGION_ADJACENCY_GRAPH_BY_DTYPE = {
    np.dtype("uint32"): _core._region_adjacency_graph_uint32,
    np.dtype("uint64"): _core._region_adjacency_graph_uint64,
    np.dtype("int32"): _core._region_adjacency_graph_int32,
    np.dtype("int64"): _core._region_adjacency_graph_int64,
}


_RAG_COORDINATES_BY_DTYPE = {
    np.dtype("uint32"): _core._rag_coordinates_uint32,
    np.dtype("uint64"): _core._rag_coordinates_uint64,
    np.dtype("int32"): _core._rag_coordinates_int32,
    np.dtype("int64"): _core._rag_coordinates_int64,
}


_EDGES_TO_VOLUME_BY_DTYPE = {
    np.dtype("float32"): "_edges_to_volume_float32",
    np.dtype("float64"): "_edges_to_volume_float64",
    np.dtype("uint32"): "_edges_to_volume_uint32",
    np.dtype("uint64"): "_edges_to_volume_uint64",
}


_PROJECT_NODE_LABELS_TO_PIXELS_BY_DTYPE = {
    np.dtype("uint32"): _core._project_node_labels_to_pixels_uint32,
    np.dtype("uint64"): _core._project_node_labels_to_pixels_uint64,
    np.dtype("int32"): _core._project_node_labels_to_pixels_int32,
    np.dtype("int64"): _core._project_node_labels_to_pixels_int64,
}


def connected_components(
    graph,
    edge_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute dense connected-component labels for graph nodes.

    If ``edge_mask`` is given, only edges with a true mask value contribute to
    the connected components.
    """
    if edge_mask is None:
        return _core._connected_components(graph)

    mask = np.asarray(edge_mask)
    if mask.dtype != np.dtype("bool"):
        raise TypeError(f"edge_mask must have dtype bool, got dtype={mask.dtype}")
    if mask.ndim != 1:
        raise ValueError("edge_mask must be a 1D array")
    if mask.shape[0] != graph.number_of_edges:
        raise ValueError("edge_mask length must match graph number_of_edges")
    return _core._connected_components_masked(
        graph, np.ascontiguousarray(mask.astype(np.uint8, copy=False))
    )


def breadth_first_search(
    graph,
    source: int,
    *,
    max_distance: int | None = None,
    include_source: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Breadth-first search from ``source`` on ``graph``.

    Returns ``(nodes, distances)`` — two 1D ``uint64`` arrays of equal length,
    listing every reachable node within ``max_distance`` hops (inclusive) in
    BFS order along with its hop distance from the source.

    Parameters
    ----------
    graph:
        :class:`UndirectedGraph` or :class:`RegionAdjacencyGraph`.
    source:
        Source node id.
    max_distance:
        Maximum hop distance from ``source`` to report. ``None`` (default)
        means no limit — the search expands until the entire connected
        component of ``source`` is visited.
    include_source:
        If ``True`` (default), the source itself is reported with distance 0.
        Set to ``False`` for "nodes within k hops, excluding self" queries.
    """
    if int(source) < 0 or int(source) >= int(graph.number_of_nodes):
        raise ValueError(
            f"source must be in [0, number_of_nodes), got source={source}, "
            f"number_of_nodes={int(graph.number_of_nodes)}"
        )
    if max_distance is None:
        limit = (1 << 64) - 1
    else:
        if int(max_distance) < 0:
            raise ValueError("max_distance must be non-negative")
        limit = int(max_distance)
    return _core._breadth_first_search(
        graph, int(source), limit, bool(include_source)
    )


def edge_weighted_watershed(
    graph,
    edge_weights,
    seeds,
) -> np.ndarray:
    """Kruskal-style edge-weighted seeded watershed on an undirected graph.

    Edges are visited in ascending weight order. Two distinct components are
    merged iff at least one of them is unlabeled (seed label ``0``); the
    non-zero seed label then propagates. Two distinct already-labeled
    components are never merged, so seed boundaries are preserved.

    Parameters
    ----------
    graph:
        :class:`UndirectedGraph` or :class:`RegionAdjacencyGraph`.
    edge_weights:
        1D array of length ``graph.number_of_edges``. Supported dtypes are
        ``float32`` and ``float64``. Other floating dtypes are cast to
        ``float32`` (matches nifty); other dtypes raise ``TypeError``.
    seeds:
        1D array of length ``graph.number_of_nodes``. Supported dtypes are
        ``uint32``, ``uint64``, ``int32``, ``int64``. ``0`` marks unlabeled
        nodes; positive ids are seed labels and propagate along low-weight
        paths. Signed seed arrays must not contain negative values.

    Returns
    -------
    np.ndarray
        1D array of length ``graph.number_of_nodes`` with the same dtype as
        ``seeds``. Nodes reachable from a seed receive that seed's label;
        unreachable nodes remain ``0``. Seed label values are preserved (no
        dense relabeling).
    """
    weight_array = np.asarray(edge_weights)
    if weight_array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        if np.issubdtype(weight_array.dtype, np.floating):
            weight_array = weight_array.astype(np.float32, copy=False)
        else:
            raise TypeError(
                "edge_weights must have dtype float32 or float64, got "
                f"dtype={weight_array.dtype}"
            )

    seed_array = np.asarray(seeds)
    if seed_array.dtype not in (
        np.dtype("uint32"),
        np.dtype("uint64"),
        np.dtype("int32"),
        np.dtype("int64"),
    ):
        raise TypeError(
            "seeds must have dtype uint32, uint64, int32, or int64, got "
            f"dtype={seed_array.dtype}"
        )

    weight_array = _as_1d_array(
        weight_array, weight_array.dtype, "edge_weights", int(graph.number_of_edges)
    )
    seed_array = _as_1d_array(
        seed_array, seed_array.dtype, "seeds", int(graph.number_of_nodes)
    )

    run = _EDGE_WEIGHTED_WATERSHED_BY_DTYPE[(weight_array.dtype, seed_array.dtype)]
    return run(graph, weight_array, seed_array)


def region_adjacency_graph(
    labels: np.ndarray,
    *,
    number_of_threads: int = 0,
) -> RegionAdjacencyGraph:
    """Build a region adjacency graph from a 2D or 3D label image.

    Nodes correspond to label ids from ``0`` to ``labels.max()``. Undirected
    edges connect different labels that touch along the pixel or voxel grid's
    direct neighborhood. The edge ids are deterministic and sorted
    lexicographically by their endpoint ids.
    """
    array = np.asarray(labels)
    if array.ndim not in (2, 3):
        raise ValueError(f"labels must be a 2D or 3D array, got ndim={array.ndim}")
    if any(size == 0 for size in array.shape):
        raise ValueError("labels must not have empty dimensions")

    dtype = array.dtype
    try:
        run = _REGION_ADJACENCY_GRAPH_BY_DTYPE[dtype]
    except KeyError as error:
        supported = ", ".join(
            str(dtype) for dtype in _REGION_ADJACENCY_GRAPH_BY_DTYPE
        )
        raise TypeError(
            f"labels must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    number_of_threads = _normalize_number_of_threads(number_of_threads)
    return run(np.ascontiguousarray(array), number_of_threads)


def rag_coordinates(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    *,
    number_of_threads: int = 0,
) -> RagCoordinates:
    """Map RAG edges to the pixel coordinates of the region boundaries.

    Scans ``labels`` (the over-segmentation used to construct ``rag``) once and
    caches, per edge, the boundary coordinates between the two adjacent regions.
    The returned :class:`RagCoordinates` exposes :meth:`~RagCoordinates.storage_lengths`,
    :meth:`~RagCoordinates.edge_coordinates`, and
    :meth:`~RagCoordinates.edges_to_volume`.
    """
    array = np.asarray(labels)
    if array.ndim not in (2, 3):
        raise ValueError(f"labels must be a 2D or 3D array, got ndim={array.ndim}")
    if tuple(int(size) for size in rag.shape) != array.shape:
        raise ValueError(
            "rag shape must match labels shape, got "
            f"rag shape={tuple(rag.shape)}, labels shape={array.shape}"
        )

    dtype = array.dtype
    try:
        run = _RAG_COORDINATES_BY_DTYPE[dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in _RAG_COORDINATES_BY_DTYPE)
        raise TypeError(
            f"labels must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    number_of_threads = _normalize_number_of_threads(number_of_threads)
    core = run(rag, np.ascontiguousarray(array), number_of_threads)
    return RagCoordinates(core)


def project_node_labels_to_pixels(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    node_labels,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Map RAG node labels back to a pixel-wise segmentation.

    ``labels`` is the over-segmentation used to construct ``rag``. Each pixel
    value is interpreted as a RAG node id and replaced by the corresponding
    entry in the 1D ``node_labels`` array. The returned segmentation has the
    same shape as ``labels`` and dtype ``uint64``.
    """
    label_array = _normalize_labels(labels)
    if tuple(int(size) for size in rag.shape) != label_array.shape:
        raise ValueError(
            "rag shape must match labels shape, got "
            f"rag shape={tuple(rag.shape)}, labels shape={label_array.shape}"
        )

    node_label_array = np.asarray(node_labels, dtype=np.uint64)
    if node_label_array.ndim != 1:
        raise ValueError("node_labels must be a 1D array")
    if node_label_array.shape[0] != rag.number_of_nodes:
        raise ValueError("node_labels length must match rag number_of_nodes")

    run = _PROJECT_NODE_LABELS_TO_PIXELS_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        np.ascontiguousarray(node_label_array),
        _normalize_number_of_threads(number_of_threads),
    )


from . import agglomeration  # noqa: E402  (must follow class/function definitions)
from . import distributed  # noqa: E402
from . import features  # noqa: E402
from . import lifted_multicut  # noqa: E402
from . import multicut  # noqa: E402
from . import mutex_watershed  # noqa: E402


__all__ = [
    "ContractionGraph",
    "ContractionResult",
    "GridGraph2D",
    "GridGraph3D",
    "RagCoordinates",
    "RegionAdjacencyGraph",
    "UndirectedGraph",
    "agglomeration",
    "breadth_first_search",
    "connected_components",
    "distributed",
    "edge_weighted_watershed",
    "features",
    "grid_graph",
    "lifted_multicut",
    "multicut",
    "mutex_watershed",
    "project_node_labels_to_pixels",
    "rag_coordinates",
    "region_adjacency_graph",
    "undirected_graph",
]
