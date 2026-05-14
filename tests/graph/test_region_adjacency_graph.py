import numpy as np
import pytest

import bioimage_cpp as bic


def test_grid_region_adjacency_graph_2d():
    labels = np.array(
        [
            [1, 1, 2],
            [1, 3, 2],
            [4, 3, 3],
        ],
        dtype=np.uint64,
    )

    rag = bic.graph.grid_region_adjacency_graph(labels)

    assert isinstance(rag, bic.graph.RegionAdjacencyGraph)
    assert rag.number_of_nodes == 5
    assert rag.number_of_edges == 5
    assert rag.shape == [3, 3]
    np.testing.assert_array_equal(
        rag.uv_ids(),
        np.array([[1, 2], [1, 3], [1, 4], [2, 3], [3, 4]], dtype=np.uint64),
    )
    assert rag.find_edge(3, 1) == 1


def test_grid_region_adjacency_graph_3d():
    labels = np.array(
        [
            [[1, 1], [2, 2]],
            [[1, 3], [2, 3]],
        ],
        dtype=np.uint32,
    )

    rag = bic.graph.grid_rag(labels, number_of_threads=1)

    assert rag.number_of_nodes == 4
    assert rag.shape == [2, 2, 2]
    np.testing.assert_array_equal(
        rag.uv_ids(), np.array([[1, 2], [1, 3], [2, 3]], dtype=np.uint64)
    )


def test_grid_region_adjacency_graph_parallel_matches_single_thread():
    labels = np.array(
        [
            [0, 1, 1, 2],
            [0, 0, 3, 2],
            [4, 4, 3, 2],
            [4, 5, 5, 5],
        ],
        dtype=np.int64,
    )

    single_threaded = bic.graph.grid_region_adjacency_graph(
        labels, number_of_threads=1
    )
    parallel = bic.graph.grid_region_adjacency_graph(labels, number_of_threads=3)

    np.testing.assert_array_equal(parallel.uv_ids(), single_threaded.uv_ids())


def test_grid_region_adjacency_graph_accepts_non_contiguous_input():
    labels = np.array(
        [
            [1, 9, 2],
            [1, 9, 2],
            [3, 9, 2],
        ],
        dtype=np.uint64,
    )[:, ::2]

    rag = bic.graph.grid_region_adjacency_graph(labels)

    np.testing.assert_array_equal(
        rag.uv_ids(), np.array([[1, 2], [1, 3], [2, 3]], dtype=np.uint64)
    )


def test_grid_region_adjacency_graph_rejects_invalid_input():
    with pytest.raises(ValueError, match="2D or 3D"):
        bic.graph.grid_region_adjacency_graph(np.ones((1,), dtype=np.uint64))

    with pytest.raises(TypeError, match="labels must have one of dtypes"):
        bic.graph.grid_region_adjacency_graph(np.ones((2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="negative"):
        bic.graph.grid_region_adjacency_graph(
            np.array([[0, -1]], dtype=np.int64)
        )

    with pytest.raises(ValueError, match="number_of_threads"):
        bic.graph.grid_region_adjacency_graph(
            np.ones((2, 2), dtype=np.uint64), number_of_threads=-1
        )
