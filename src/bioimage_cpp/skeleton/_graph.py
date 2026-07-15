"""Utilities for working with skeleton topology."""

from __future__ import annotations

import numpy as np

from ..graph import UndirectedGraph


def skeleton_to_graph(
    vertices: np.ndarray,
    edges: np.ndarray,
) -> UndirectedGraph:
    """Convert skeleton vertex and edge arrays to an undirected graph.

    This accepts the ``vertices`` and ``edges`` arrays returned by
    :func:`bioimage_cpp.skeleton.teasar`, or by an individual result from
    :func:`bioimage_cpp.skeleton.teasar_labels`. Graph node ids correspond
    directly to rows in ``vertices`` (and in the accompanying TEASAR
    ``radii`` array). The returned graph stores topology only; coordinates and
    radii remain in their original arrays.

    Parameters
    ----------
    vertices:
        Two-dimensional array with one row per skeleton vertex. Only the
        number of rows is used to construct the graph.
    edges:
        Integer array with shape ``(E, 2)`` containing vertex ids. Edge
        endpoints must be smaller than ``len(vertices)`` and self-edges are
        not supported.

    Returns
    -------
    bioimage_cpp.graph.UndirectedGraph
        An undirected graph with one node per vertex and the topology given by
        ``edges``. Empty skeletons and isolated vertices are preserved.

    Raises
    ------
    ValueError
        If ``vertices`` is not two-dimensional or the edge shape/topology is
        invalid.
    TypeError
        If ``edges`` is not an integer array.
    IndexError
        If an edge endpoint is outside the vertex range.
    """
    vertex_array = np.asarray(vertices)
    if vertex_array.ndim != 2:
        raise ValueError(
            f"vertices must be a 2D array, got ndim={vertex_array.ndim}"
        )
    return UndirectedGraph.from_edges(vertex_array.shape[0], edges)


__all__ = ["skeleton_to_graph"]
