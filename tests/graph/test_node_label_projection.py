import numpy as np
import pytest

import bioimage_cpp as bic


def test_project_node_labels_to_pixels_matches_nifty():
    nrag = pytest.importorskip("nifty.graph.rag")

    labels = np.array([[1, 1, 2], [3, 2, 2]], dtype=np.uint64)
    node_labels = np.array([10, 11, 12, 13], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)

    actual = bic.graph.project_node_labels_to_pixels(rag, labels, node_labels)
    expected = nrag.projectScalarNodeDataToPixels(nrag.gridRag(labels), node_labels)

    np.testing.assert_array_equal(actual, expected)


def test_project_node_labels_to_pixels_parallel_matches_single_thread():
    labels = np.arange(6 * 7, dtype=np.uint32).reshape(6, 7)
    rag = bic.graph.region_adjacency_graph(labels, number_of_threads=2)
    node_labels = (np.arange(rag.number_of_nodes, dtype=np.uint64) // 3) + 17

    single_threaded = bic.graph.project_node_labels_to_pixels(
        rag, labels, node_labels, number_of_threads=1
    )
    parallel = bic.graph.project_node_labels_to_pixels(
        rag, labels, node_labels, number_of_threads=4
    )

    np.testing.assert_array_equal(parallel, single_threaded)


def test_project_node_labels_to_pixels_validation():
    labels = np.array([[0, 1]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)

    with pytest.raises(ValueError, match="node_labels must be a 1D array"):
        bic.graph.project_node_labels_to_pixels(rag, labels, [[0, 1]])

    with pytest.raises(ValueError, match="node_labels length"):
        bic.graph.project_node_labels_to_pixels(rag, labels, [0])

    other_labels = np.array([[0, 1], [0, 1]], dtype=np.uint64)
    with pytest.raises(ValueError, match="rag shape"):
        bic.graph.project_node_labels_to_pixels(rag, other_labels, [0, 1])
