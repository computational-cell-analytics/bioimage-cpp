import numpy as np
import pytest

import bioimage_cpp as bic


def test_bfs_chain_distances():
    graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [1, 2], [2, 3]])
    nodes, distances = bic.graph.breadth_first_search(graph, 0)
    np.testing.assert_array_equal(nodes, np.array([0, 1, 2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(distances, np.array([0, 1, 2, 3], dtype=np.uint64))


def test_bfs_max_distance_limits_expansion():
    graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [1, 2], [2, 3]])
    nodes, distances = bic.graph.breadth_first_search(graph, 0, max_distance=2)
    np.testing.assert_array_equal(nodes, np.array([0, 1, 2], dtype=np.uint64))
    np.testing.assert_array_equal(distances, np.array([0, 1, 2], dtype=np.uint64))


def test_bfs_exclude_source():
    graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [1, 2], [2, 3]])
    nodes, distances = bic.graph.breadth_first_search(
        graph, 1, max_distance=1, include_source=False
    )
    np.testing.assert_array_equal(sorted(nodes.tolist()), [0, 2])
    np.testing.assert_array_equal(distances, np.array([1, 1], dtype=np.uint64))


def test_bfs_branching_distances():
    # Star: 0 -- 1, 0 -- 2, 1 -- 3
    graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [0, 2], [1, 3]])
    nodes, distances = bic.graph.breadth_first_search(graph, 0)
    order = np.argsort(nodes)
    np.testing.assert_array_equal(nodes[order], np.array([0, 1, 2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(distances[order], np.array([0, 1, 1, 2], dtype=np.uint64))


def test_bfs_disconnected_component_not_reached():
    graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [2, 3]])
    nodes, distances = bic.graph.breadth_first_search(graph, 0)
    np.testing.assert_array_equal(sorted(nodes.tolist()), [0, 1])
    np.testing.assert_array_equal(distances, np.array([0, 1], dtype=np.uint64))


def test_bfs_invalid_source_raises():
    graph = bic.graph.UndirectedGraph.from_edges(2, [[0, 1]])
    with pytest.raises(ValueError, match="source"):
        bic.graph.breadth_first_search(graph, 2)


def test_bfs_negative_max_distance_raises():
    graph = bic.graph.UndirectedGraph.from_edges(2, [[0, 1]])
    with pytest.raises(ValueError, match="max_distance"):
        bic.graph.breadth_first_search(graph, 0, max_distance=-1)


def test_bfs_zero_max_distance_returns_only_source():
    graph = bic.graph.UndirectedGraph.from_edges(2, [[0, 1]])
    nodes, distances = bic.graph.breadth_first_search(graph, 0, max_distance=0)
    np.testing.assert_array_equal(nodes, np.array([0], dtype=np.uint64))
    np.testing.assert_array_equal(distances, np.array([0], dtype=np.uint64))


def test_bfs_zero_max_distance_exclude_source_returns_empty():
    graph = bic.graph.UndirectedGraph.from_edges(2, [[0, 1]])
    nodes, distances = bic.graph.breadth_first_search(
        graph, 0, max_distance=0, include_source=False
    )
    assert nodes.shape == (0,)
    assert distances.shape == (0,)
