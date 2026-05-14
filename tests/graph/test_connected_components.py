import numpy as np
import pytest

import bioimage_cpp as bic


def test_connected_components_without_edges():
    graph = bic.graph.UndirectedGraph(3)

    np.testing.assert_array_equal(
        bic.graph.connected_components(graph),
        np.array([0, 1, 2], dtype=np.uint64),
    )


def test_connected_components_for_undirected_graph():
    graph = bic.graph.UndirectedGraph.from_edges(5, [[0, 1], [2, 3], [1, 4]])

    np.testing.assert_array_equal(
        bic.graph.connected_components(graph),
        np.array([0, 0, 1, 1, 0], dtype=np.uint64),
    )


def test_connected_components_with_edge_mask():
    graph = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])

    np.testing.assert_array_equal(
        bic.graph.connected_components(graph, edge_mask=np.array([True, False])),
        np.array([0, 0, 1], dtype=np.uint64),
    )


def test_connected_components_accepts_region_adjacency_graph():
    labels = np.array([[1, 1, 2], [3, 3, 2]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)

    components = bic.graph.connected_components(rag)

    assert components.dtype == np.uint64
    assert components.shape == (4,)
    assert components[0] != components[1]
    np.testing.assert_array_equal(components[[1, 2, 3]], np.repeat(components[1], 3))


def test_connected_components_rejects_invalid_mask():
    graph = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])

    with pytest.raises(TypeError, match="edge_mask must have dtype bool"):
        bic.graph.connected_components(graph, edge_mask=np.array([1, 0], dtype=np.uint8))
    with pytest.raises(ValueError, match="1D"):
        bic.graph.connected_components(graph, edge_mask=np.ones((1, 2), dtype=bool))
    with pytest.raises(ValueError, match="number_of_edges"):
        bic.graph.connected_components(graph, edge_mask=np.array([True]))
