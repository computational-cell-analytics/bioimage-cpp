"""Mutex watershed clustering on undirected graphs.

The functions in this submodule operate on an attractive base graph plus a
set of explicit repulsive (mutex) constraints. They are conceptually close to
the lifted multicut input format — the same ``(graph, edge_costs, mutex_uvs,
mutex_costs)`` arrays can be reused — but the algorithms differ: mutex
constraints are *hard* (a single mutex edge separates two components forever),
whereas lifted costs are soft.
"""

from __future__ import annotations

import numpy as np

from .. import _core
from ._shared import (
    _as_1d_array,
    _as_uv_array,
    _resolve_weight_dtype,
)


_MUTEX_WATERSHED_CLUSTERING_BY_DTYPE = {
    np.dtype("float32"): _core._mutex_watershed_clustering_float32,
    np.dtype("float64"): _core._mutex_watershed_clustering_float64,
}

_SEMANTIC_MUTEX_WATERSHED_CLUSTERING_BY_DTYPE = {
    np.dtype("float32"): _core._semantic_mutex_watershed_clustering_float32,
    np.dtype("float64"): _core._semantic_mutex_watershed_clustering_float64,
}


def mutex_watershed_clustering(
    graph,
    edge_costs,
    mutex_uvs,
    mutex_costs,
) -> np.ndarray:
    """Mutex watershed clustering on an undirected graph.

    Introduced in "The Mutex Watershed and its Objective: Efficient,
    Parameter-Free Image Partitioning":
    https://arxiv.org/pdf/1904.12654.pdf

    Attractive edges come from ``graph`` (one cost per edge in
    ``edge_costs``); repulsive long-range edges are supplied separately as
    ``mutex_uvs`` with weights ``mutex_costs``. All edges are jointly sorted
    by descending weight and processed in a single pass: an attractive edge
    merges its two components unless a mutex constraint already separates
    them; a mutex edge installs a constraint between the two current
    components.

    The input format matches
    :class:`bioimage_cpp.graph.lifted_multicut.LiftedMulticutObjective` — the
    same ``(graph, edge_costs, lifted_uvs, lifted_costs)`` arrays can be
    passed here as ``(graph, edge_costs, mutex_uvs, mutex_costs)`` — though
    the algorithms differ (mutex constraints are hard; lifted costs are
    soft).

    Parameters
    ----------
    graph:
        :class:`bioimage_cpp.graph.UndirectedGraph` or
        :class:`bioimage_cpp.graph.RegionAdjacencyGraph` defining the
        attractive edges.
    edge_costs:
        1D array of length ``graph.number_of_edges``. Supported dtypes are
        ``float32`` and ``float64``; other floating dtypes are cast to
        ``float32``. Higher values are more attractive.
    mutex_uvs:
        ``(n_mutex, 2)`` uint64 array of (u, v) pairs for the mutex edges.
    mutex_costs:
        1D array of length ``n_mutex``. Same dtype rules as ``edge_costs``;
        if the two dtypes differ both are promoted to ``float64``. Higher
        values are stronger repulsions.

    Returns
    -------
    np.ndarray
        ``(graph.number_of_nodes,)`` uint64 array. Dense node labels in
        ``[0, k)`` in first-occurrence order.
    """
    edge_cost_array = _resolve_weight_dtype(edge_costs, "edge_costs")
    mutex_cost_array = _resolve_weight_dtype(mutex_costs, "mutex_costs")
    # Use a single instantiation for both arrays. If the user supplied
    # mismatched dtypes (one float32, one float64) we promote both to
    # float64 — the wider type — rather than silently downcasting.
    if edge_cost_array.dtype != mutex_cost_array.dtype:
        edge_cost_array = edge_cost_array.astype(np.float64, copy=False)
        mutex_cost_array = mutex_cost_array.astype(np.float64, copy=False)

    edge_cost_array = _as_1d_array(
        edge_cost_array,
        edge_cost_array.dtype,
        "edge_costs",
        int(graph.number_of_edges),
    )
    mutex_uv_array = _as_uv_array(mutex_uvs, "mutex_uvs")
    mutex_cost_array = _as_1d_array(
        mutex_cost_array,
        mutex_cost_array.dtype,
        "mutex_costs",
        int(mutex_uv_array.shape[0]),
    )
    run = _MUTEX_WATERSHED_CLUSTERING_BY_DTYPE[edge_cost_array.dtype]
    return run(graph, edge_cost_array, mutex_uv_array, mutex_cost_array)


def semantic_mutex_watershed_clustering(
    graph,
    edge_costs,
    mutex_uvs,
    mutex_costs,
    semantic_node_classes,
    semantic_costs,
) -> tuple[np.ndarray, np.ndarray]:
    """Semantic mutex watershed clustering on an undirected graph.

    Introduced in "The Semantic Mutex Watershed for Efficient Bottom-Up
    Semantic Instance Segmentation":
    https://arxiv.org/pdf/1912.12717.pdf

    Extends :func:`mutex_watershed_clustering` with a third group of edges
    that attach semantic class labels to clusters. Two clusters carrying
    different semantic class labels are forbidden from merging; otherwise
    the algorithm proceeds as in the non-semantic case (attractive edges
    merge; mutex edges block).

    Parameters
    ----------
    graph:
        :class:`bioimage_cpp.graph.UndirectedGraph` or
        :class:`bioimage_cpp.graph.RegionAdjacencyGraph` defining the
        attractive edges.
    edge_costs:
        1D array of length ``graph.number_of_edges``. Same dtype rules as
        :func:`mutex_watershed_clustering`.
    mutex_uvs:
        ``(n_mutex, 2)`` uint64 array of (u, v) pairs for the mutex edges.
    mutex_costs:
        1D array of length ``n_mutex``.
    semantic_node_classes:
        ``(n_semantic, 2)`` uint64 array. Column 0 is a node id; column 1
        is the semantic class id (non-negative integer). The semantic class
        id is interpreted as a signed integer when reported back, so values
        above ``np.iinfo(np.int64).max`` are out of range.
    semantic_costs:
        1D array of length ``n_semantic``. Same dtype rules as
        ``edge_costs``; if the floating dtypes of the three weight arrays
        do not all agree, all three are promoted to ``float64``.

    Returns
    -------
    node_labels:
        ``(graph.number_of_nodes,)`` uint64 array. Dense node labels in
        ``[0, k)`` in first-occurrence order.
    semantic_labels:
        ``(graph.number_of_nodes,)`` int64 array. Per-node semantic class
        id, or ``-1`` for clusters that received no semantic assignment.
    """
    edge_cost_array = _resolve_weight_dtype(edge_costs, "edge_costs")
    mutex_cost_array = _resolve_weight_dtype(mutex_costs, "mutex_costs")
    semantic_cost_array = _resolve_weight_dtype(semantic_costs, "semantic_costs")

    dtypes = {edge_cost_array.dtype, mutex_cost_array.dtype, semantic_cost_array.dtype}
    if len(dtypes) > 1:
        edge_cost_array = edge_cost_array.astype(np.float64, copy=False)
        mutex_cost_array = mutex_cost_array.astype(np.float64, copy=False)
        semantic_cost_array = semantic_cost_array.astype(np.float64, copy=False)

    edge_cost_array = _as_1d_array(
        edge_cost_array,
        edge_cost_array.dtype,
        "edge_costs",
        int(graph.number_of_edges),
    )
    mutex_uv_array = _as_uv_array(mutex_uvs, "mutex_uvs")
    mutex_cost_array = _as_1d_array(
        mutex_cost_array,
        mutex_cost_array.dtype,
        "mutex_costs",
        int(mutex_uv_array.shape[0]),
    )
    semantic_uv_array = _as_uv_array(semantic_node_classes, "semantic_node_classes")
    semantic_cost_array = _as_1d_array(
        semantic_cost_array,
        semantic_cost_array.dtype,
        "semantic_costs",
        int(semantic_uv_array.shape[0]),
    )

    run = _SEMANTIC_MUTEX_WATERSHED_CLUSTERING_BY_DTYPE[edge_cost_array.dtype]
    return run(
        graph,
        edge_cost_array,
        mutex_uv_array,
        mutex_cost_array,
        semantic_uv_array,
        semantic_cost_array,
    )


__all__ = [
    "mutex_watershed_clustering",
    "semantic_mutex_watershed_clustering",
]
