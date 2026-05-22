"""Graph data structures and graph-level algorithms.

Top-level surface:

- Graph structures: :class:`UndirectedGraph`, :class:`GridGraph2D`,
  :class:`GridGraph3D`, :class:`RegionAdjacencyGraph`.
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
"""

from __future__ import annotations

import numpy as np

from .. import _core
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
)


class UndirectedGraph(_core.UndirectedGraph):
    """Undirected graph with consecutive node and edge ids.

    Nodes are fixed at construction and addressed by ids
    ``0 .. number_of_nodes - 1``. Edges are inserted lazily and receive
    consecutive ids in insertion order. Re-inserting an existing undirected edge
    returns the existing edge id.
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
        graph = cls(number_of_nodes)
        graph.insert_edges(uvs)
        return graph

    @classmethod
    def deserialize(cls, serialization):
        serialization = _as_serialization_array(serialization)
        number_of_nodes = int(serialization[0])
        number_of_edges = int(serialization[1])
        uvs = serialization[2:].reshape(number_of_edges, 2)
        return cls.from_edges(number_of_nodes, uvs)


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


RegionAdjacencyGraph = _core.RegionAdjacencyGraph


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


from . import features  # noqa: E402  (must follow class/function definitions)
from . import lifted_multicut  # noqa: E402
from . import multicut  # noqa: E402
from . import mutex_watershed  # noqa: E402


__all__ = [
    "GridGraph2D",
    "GridGraph3D",
    "RegionAdjacencyGraph",
    "UndirectedGraph",
    "breadth_first_search",
    "connected_components",
    "edge_weighted_watershed",
    "features",
    "grid_graph",
    "lifted_multicut",
    "multicut",
    "mutex_watershed",
    "project_node_labels_to_pixels",
    "region_adjacency_graph",
    "undirected_graph",
]
