import numpy as np
import pytest

import bioimage_cpp as bic


def test_edge_map_features_simple():
    labels = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.uint64)
    edge_map = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
    rag = bic.graph.region_adjacency_graph(labels)

    features = bic.graph.edge_map_features(rag, labels, edge_map)

    assert tuple(bic.graph.SIMPLE_EDGE_FEATURE_NAMES) == ("mean", "size")
    np.testing.assert_allclose(
        features,
        np.array(
            [
                [6.0, 1.0],
                [3.25, 2.0],
                [12.0, 1.0],
            ]
        ),
    )


def test_edge_map_features_complex():
    labels = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.uint32)
    edge_map = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
    rag = bic.graph.region_adjacency_graph(labels)

    features = bic.graph.edge_map_features_complex(rag, labels, edge_map)

    assert tuple(bic.graph.COMPLEX_EDGE_FEATURE_NAMES) == (
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p5",
        "p10",
        "p25",
        "p75",
        "p90",
        "p95",
        "size",
    )
    np.testing.assert_allclose(
        features,
        np.array(
            [
                [
                    6.0, 6.0, 0.0, 6.0, 6.0, 6.0,
                    6.0, 6.0, 6.0, 6.0, 6.0, 1.0,
                ],
                [
                    3.25, 3.25, 0.25, 3.0, 3.5, 3.025,
                    3.05, 3.125, 3.375, 3.45, 3.475, 2.0,
                ],
                [
                    12.0, 12.0, 0.0, 12.0, 12.0, 12.0,
                    12.0, 12.0, 12.0, 12.0, 12.0, 1.0,
                ],
            ]
        ),
    )


def test_affinity_features_simple():
    labels = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.int64)
    rag = bic.graph.region_adjacency_graph(labels)
    affinities = np.zeros((2, 2, 3), dtype=np.float32)
    affinities[0] = np.array([[0.0, 6.0, 0.0], [7.0, 8.0, 0.0]])
    affinities[1] = np.array([[0.0, 9.0, 0.0], [0.0, 0.0, 0.0]])

    features = bic.graph.affinity_features(
        rag, labels, affinities, offsets=[[0, 1], [1, 0]]
    )

    np.testing.assert_allclose(
        features,
        np.array(
            [
                [6.0, 1.0],
                [8.0, 2.0],
                [8.0, 1.0],
            ]
        ),
    )


def test_affinity_features_complex_parallel_matches_single_thread():
    labels = np.array([[0, 1, 1], [2, 2, 3], [2, 4, 3]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)
    affinities = np.arange(2 * labels.size, dtype=np.float64).reshape((2,) + labels.shape)
    offsets = [[0, 1], [1, 0]]

    single_threaded = bic.graph.affinity_features_complex(
        rag, labels, affinities, offsets, number_of_threads=1
    )
    parallel = bic.graph.affinity_features_complex(
        rag, labels, affinities, offsets, number_of_threads=3
    )

    np.testing.assert_allclose(parallel, single_threaded)


def test_rag_feature_validation():
    labels = np.array([[1, 2]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)

    with pytest.raises(ValueError, match="edge_map shape"):
        bic.graph.edge_map_features(rag, labels, np.ones((2, 2)))

    with pytest.raises(ValueError, match="affinities must have shape"):
        bic.graph.affinity_features(rag, labels, np.ones((2, 2)), offsets=[[0, 1]])

    with pytest.raises(ValueError, match="offsets length"):
        bic.graph.affinity_features(
            rag, labels, np.ones((2, 1, 2)), offsets=[[0, 1]]
        )
