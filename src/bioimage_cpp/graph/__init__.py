"""Graph data structures."""

from __future__ import annotations

import numpy as np

from .. import _core

_REGION_ADJACENCY_GRAPH_BY_DTYPE = {
    np.dtype("uint32"): _core._region_adjacency_graph_uint32,
    np.dtype("uint64"): _core._region_adjacency_graph_uint64,
    np.dtype("int32"): _core._region_adjacency_graph_int32,
    np.dtype("int64"): _core._region_adjacency_graph_int64,
}

_EDGE_MAP_FEATURES_BY_DTYPE = {
    np.dtype("uint32"): _core._accumulate_edge_map_features_uint32,
    np.dtype("uint64"): _core._accumulate_edge_map_features_uint64,
    np.dtype("int32"): _core._accumulate_edge_map_features_int32,
    np.dtype("int64"): _core._accumulate_edge_map_features_int64,
}

_AFFINITY_FEATURES_BY_DTYPE = {
    np.dtype("uint32"): _core._accumulate_affinity_features_uint32,
    np.dtype("uint64"): _core._accumulate_affinity_features_uint64,
    np.dtype("int32"): _core._accumulate_affinity_features_int32,
    np.dtype("int64"): _core._accumulate_affinity_features_int64,
}

SIMPLE_EDGE_FEATURE_NAMES = ("mean", "size")
COMPLEX_EDGE_FEATURE_NAMES = (
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p5",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
    "size",
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


def _as_uv_array(uvs, name: str) -> np.ndarray:
    array = np.asarray(uvs, dtype=np.uint64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_edges, 2)")
    return np.ascontiguousarray(array)


def _as_node_array(nodes, name: str) -> np.ndarray:
    array = np.asarray(nodes, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return np.ascontiguousarray(array)


def _as_serialization_array(serialization) -> np.ndarray:
    array = np.asarray(serialization, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError("serialization must be a 1D array")
    if array.size < 2:
        raise ValueError("serialization must have at least two entries")
    number_of_edges = int(array[1])
    if array.size != 2 + 2 * number_of_edges:
        raise ValueError("serialization size must be 2 + 2 * number_of_edges")
    return np.ascontiguousarray(array)


def undirected_graph(number_of_nodes: int) -> UndirectedGraph:
    """Create an :class:`UndirectedGraph`."""
    return UndirectedGraph(number_of_nodes)


RegionAdjacencyGraph = _core.RegionAdjacencyGraph


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

    number_of_threads = int(number_of_threads)
    if number_of_threads < 0:
        raise ValueError("number_of_threads must be non-negative")
    return run(np.ascontiguousarray(array), number_of_threads)


def edge_map_features(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    edge_map: np.ndarray,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute mean and size features for edge-map values on RAG boundaries."""
    return _accumulate_edge_map_features(
        rag,
        labels,
        edge_map,
        compute_complex_features=False,
        number_of_threads=number_of_threads,
    )


def edge_map_features_complex(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    edge_map: np.ndarray,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute complex edge-map features on RAG boundaries.

    The output columns are given by ``COMPLEX_EDGE_FEATURE_NAMES``.
    """
    return _accumulate_edge_map_features(
        rag,
        labels,
        edge_map,
        compute_complex_features=True,
        number_of_threads=number_of_threads,
    )


def affinity_features(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute mean and size features for affinity links crossing RAG edges."""
    return _accumulate_affinity_features(
        rag,
        labels,
        affinities,
        offsets,
        compute_complex_features=False,
        number_of_threads=number_of_threads,
    )


def affinity_features_complex(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute complex affinity features for links crossing RAG edges.

    The output columns are given by ``COMPLEX_EDGE_FEATURE_NAMES``.
    """
    return _accumulate_affinity_features(
        rag,
        labels,
        affinities,
        offsets,
        compute_complex_features=True,
        number_of_threads=number_of_threads,
    )


def _accumulate_edge_map_features(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    edge_map: np.ndarray,
    *,
    compute_complex_features: bool,
    number_of_threads: int,
) -> np.ndarray:
    label_array = _normalize_labels(labels)
    edge_map_array = np.asarray(edge_map, dtype=np.float64)
    if edge_map_array.shape != label_array.shape:
        raise ValueError(
            "edge_map shape must match labels shape, got "
            f"edge_map shape={edge_map_array.shape}, labels shape={label_array.shape}"
        )
    run = _EDGE_MAP_FEATURES_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        np.ascontiguousarray(edge_map_array),
        bool(compute_complex_features),
        _normalize_number_of_threads(number_of_threads),
    )


def _accumulate_affinity_features(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    *,
    compute_complex_features: bool,
    number_of_threads: int,
) -> np.ndarray:
    label_array = _normalize_labels(labels)
    affinity_array = np.asarray(affinities, dtype=np.float64)
    if affinity_array.ndim != label_array.ndim + 1:
        raise ValueError("affinities must have shape (channels, *labels.shape)")
    if affinity_array.shape[1:] != label_array.shape:
        raise ValueError(
            "affinities spatial shape must match labels shape, got "
            f"affinities shape={affinity_array.shape}, labels shape={label_array.shape}"
        )

    normalized_offsets = [tuple(int(value) for value in offset) for offset in offsets]
    if len(normalized_offsets) != affinity_array.shape[0]:
        raise ValueError(
            "offsets length must match affinities channel count, got "
            f"offsets length={len(normalized_offsets)}, channels={affinity_array.shape[0]}"
        )
    if any(len(offset) != label_array.ndim for offset in normalized_offsets):
        raise ValueError("each offset must have length matching labels ndim")

    run = _AFFINITY_FEATURES_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        np.ascontiguousarray(affinity_array),
        normalized_offsets,
        bool(compute_complex_features),
        _normalize_number_of_threads(number_of_threads),
    )


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim not in (2, 3):
        raise ValueError(f"labels must be a 2D or 3D array, got ndim={array.ndim}")
    try:
        _REGION_ADJACENCY_GRAPH_BY_DTYPE[array.dtype]
    except KeyError as error:
        supported = ", ".join(
            str(dtype) for dtype in _REGION_ADJACENCY_GRAPH_BY_DTYPE
        )
        raise TypeError(
            f"labels must have one of dtypes ({supported}), got dtype={array.dtype}"
        ) from error
    return np.ascontiguousarray(array)


def _normalize_number_of_threads(number_of_threads: int) -> int:
    number_of_threads = int(number_of_threads)
    if number_of_threads < 0:
        raise ValueError("number_of_threads must be non-negative")
    return number_of_threads


__all__ = [
    "COMPLEX_EDGE_FEATURE_NAMES",
    "RegionAdjacencyGraph",
    "SIMPLE_EDGE_FEATURE_NAMES",
    "UndirectedGraph",
    "affinity_features",
    "affinity_features_complex",
    "edge_map_features",
    "edge_map_features_complex",
    "region_adjacency_graph",
    "undirected_graph",
]
