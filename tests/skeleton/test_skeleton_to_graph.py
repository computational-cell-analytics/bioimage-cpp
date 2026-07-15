import numpy as np
import pytest

import bioimage_cpp as bic


def _node_degrees(graph):
    return np.fromiter(
        (
            len(graph.node_adjacency(node))
            for node in range(graph.number_of_nodes)
        ),
        dtype=np.uint64,
        count=graph.number_of_nodes,
    )


def test_skeleton_to_graph_preserves_teasar_topology_and_vertex_ids():
    mask = np.zeros((9, 11, 10), dtype=np.uint8)
    mask[4, 5, 1:5] = 1
    for step in range(5):
        mask[4, 5 - step, 4 + step] = 1
        mask[4, 5 + step, 4 + step] = 1

    vertices, edges, _ = bic.skeleton.teasar(mask)
    graph = bic.skeleton.skeleton_to_graph(vertices, edges)

    assert isinstance(graph, bic.graph.UndirectedGraph)
    assert graph.number_of_nodes == len(vertices)
    assert graph.number_of_edges == len(edges)
    np.testing.assert_array_equal(graph.uv_ids(), edges)

    degrees = _node_degrees(graph)
    endpoints = {
        tuple(vertex.astype(int)) for vertex in vertices[degrees <= 1]
    }
    assert endpoints == {(4, 5, 1), (4, 1, 8), (4, 9, 8)}
    assert np.count_nonzero(degrees > 2) == 1


def test_skeleton_to_graph_preserves_empty_and_isolated_vertices():
    empty = bic.skeleton.skeleton_to_graph(
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 2), dtype=np.uint64),
    )
    assert empty.number_of_nodes == 0
    assert empty.number_of_edges == 0

    isolated = bic.skeleton.skeleton_to_graph(
        np.array([[2.0, 3.0, 4.0]]),
        np.empty((0, 2), dtype=np.uint64),
    )
    assert isolated.number_of_nodes == 1
    assert isolated.number_of_edges == 0
    np.testing.assert_array_equal(_node_degrees(isolated), [0])


def test_skeleton_to_graph_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="vertices must be a 2D array"):
        bic.skeleton.skeleton_to_graph(
            np.array([0.0, 1.0, 2.0]),
            np.empty((0, 2), dtype=np.uint64),
        )

    vertices = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="uvs must have shape"):
        bic.skeleton.skeleton_to_graph(vertices, [0, 1])
    with pytest.raises(IndexError, match="node id must be < number_of_nodes"):
        bic.skeleton.skeleton_to_graph(vertices, [[0, 2]])
