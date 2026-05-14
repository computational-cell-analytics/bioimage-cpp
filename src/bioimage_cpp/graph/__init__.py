"""Graph data structures."""

from __future__ import annotations

import numpy as np

from .. import _core

_GRID_REGION_ADJACENCY_GRAPH_BY_DTYPE = {
    np.dtype("uint32"): _core._grid_region_adjacency_graph_uint32,
    np.dtype("uint64"): _core._grid_region_adjacency_graph_uint64,
    np.dtype("int32"): _core._grid_region_adjacency_graph_int32,
    np.dtype("int64"): _core._grid_region_adjacency_graph_int64,
}


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


def grid_region_adjacency_graph(
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
        run = _GRID_REGION_ADJACENCY_GRAPH_BY_DTYPE[dtype]
    except KeyError as error:
        supported = ", ".join(
            str(dtype) for dtype in _GRID_REGION_ADJACENCY_GRAPH_BY_DTYPE
        )
        raise TypeError(
            f"labels must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    number_of_threads = int(number_of_threads)
    if number_of_threads < 0:
        raise ValueError("number_of_threads must be non-negative")
    return run(np.ascontiguousarray(array), number_of_threads)


def grid_rag(labels: np.ndarray, *, number_of_threads: int = 0) -> RegionAdjacencyGraph:
    """Alias for :func:`grid_region_adjacency_graph`."""
    return grid_region_adjacency_graph(
        labels,
        number_of_threads=number_of_threads,
    )


__all__ = [
    "RegionAdjacencyGraph",
    "UndirectedGraph",
    "grid_rag",
    "grid_region_adjacency_graph",
    "undirected_graph",
]
