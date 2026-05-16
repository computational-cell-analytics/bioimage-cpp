import numpy as np
import pytest

import bioimage_cpp as bic


def test_grid_graph_2d_topology_and_coordinates():
    graph = bic.graph.GridGraph2D((2, 3))

    assert graph.number_of_nodes == 6
    assert graph.number_of_edges == 7
    assert graph.ndim == 2
    np.testing.assert_array_equal(graph.shape, np.array([2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(graph.strides, np.array([3, 1], dtype=np.uint64))
    np.testing.assert_array_equal(
        graph.uv_ids(),
        np.array(
            [[0, 3], [1, 4], [2, 5], [0, 1], [1, 2], [3, 4], [4, 5]],
            dtype=np.uint64,
        ),
    )

    assert graph.node_id((1, 2)) == 5
    np.testing.assert_array_equal(graph.coordinates(4), np.array([1, 1], dtype=np.uint64))
    assert graph.find_edge(0, 3) == 0
    assert graph.find_edge(2, 5) == 2
    assert graph.find_edge(0, 1) == 3
    assert graph.find_edge(0, 2) == -1
    assert graph.edge_axis(0) == 0
    assert graph.edge_axis(3) == 1
    np.testing.assert_array_equal(
        graph.edge_coordinates(6), np.array([1, 1], dtype=np.uint64)
    )


def test_grid_graph_3d_topology_and_offset_targets():
    graph = bic.graph.GridGraph3D((2, 2, 2))

    assert graph.number_of_nodes == 8
    assert graph.number_of_edges == 12
    np.testing.assert_array_equal(graph.shape, np.array([2, 2, 2], dtype=np.uint64))
    np.testing.assert_array_equal(graph.strides, np.array([4, 2, 1], dtype=np.uint64))
    np.testing.assert_array_equal(
        graph.uv_ids(),
        np.array(
            [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [0, 2],
                [1, 3],
                [4, 6],
                [5, 7],
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
            ],
            dtype=np.uint64,
        ),
    )

    assert graph.node_id((1, 1, 1)) == 7
    np.testing.assert_array_equal(
        graph.coordinates(6), np.array([1, 1, 0], dtype=np.uint64)
    )
    assert graph.offset_target(0, (1, 0, 0)) == 4
    assert graph.offset_target(3, (0, 0, -1)) == 2
    assert graph.offset_target(3, (0, 0, 1)) == -1


def test_grid_graph_factory_and_algorithm_interop():
    graph = bic.graph.grid_graph((3, 3))

    assert isinstance(graph, bic.graph.GridGraph2D)
    components = bic.graph.connected_components(graph)
    np.testing.assert_array_equal(components, np.zeros(9, dtype=np.uint64))

    nodes, distances = bic.graph.breadth_first_search(
        graph, 0, max_distance=1, include_source=True
    )
    np.testing.assert_array_equal(nodes, np.array([0, 3, 1], dtype=np.uint64))
    np.testing.assert_array_equal(distances, np.array([0, 1, 1], dtype=np.uint64))


def test_grid_graph_rejects_invalid_shapes_and_coordinates():
    with pytest.raises(ValueError, match="length 2"):
        bic.graph.GridGraph2D((2, 3, 4))
    with pytest.raises(ValueError, match="greater than zero"):
        bic.graph.GridGraph3D((2, 0, 4))
    with pytest.raises(ValueError, match="length 2 or 3"):
        bic.graph.grid_graph((5,))

    graph = bic.graph.GridGraph2D((2, 3))
    with pytest.raises(ValueError, match="coordinate must be"):
        graph.node_id((1, 2, 3))
    with pytest.raises(IndexError, match="coordinate\\[0\\]"):
        graph.node_id((2, 0))
    with pytest.raises(ValueError, match="offset must be"):
        graph.offset_target(0, (1, 0, 0))
