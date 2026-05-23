"""Small graph constructions and helpers shared by the agglomeration tests."""

from __future__ import annotations

import numpy as np

import bioimage_cpp as bic


def chain_graph(n: int):
    """A path graph 0-1-2-...-(n-1) with ``n - 1`` edges."""
    uvs = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.uint64)
    return bic.graph.UndirectedGraph.from_edges(n, uvs)


def two_clusters_graph():
    """Two triangles connected by one weak bridge edge.

    Edge order:
        0: 0-1, 1: 1-2, 2: 0-2,        (cluster A)
        3: 3-4, 4: 4-5, 5: 3-5,        (cluster B)
        6: 2-3                          (bridge)
    """
    uvs = np.array(
        [
            [0, 1], [1, 2], [0, 2],
            [3, 4], [4, 5], [3, 5],
            [2, 3],
        ],
        dtype=np.uint64,
    )
    return bic.graph.UndirectedGraph.from_edges(6, uvs)


def canonical_labels(labels):
    """Map labels to first-occurrence dense ids for partition comparison."""
    array = np.asarray(labels, dtype=np.uint64)
    out = np.empty_like(array)
    seen: dict[int, int] = {}
    for index, value in enumerate(array):
        key = int(value)
        if key not in seen:
            seen[key] = len(seen)
        out[index] = seen[key]
    return out


def assert_same_partition(actual, expected):
    np.testing.assert_array_equal(canonical_labels(actual), canonical_labels(expected))
