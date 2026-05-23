from __future__ import annotations

import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import (
    assert_same_partition,
    chain_graph,
    two_clusters_graph,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_chain_merges_to_single_cluster(dtype):
    graph = chain_graph(4)
    # Low indicators / low feature distance everywhere — all edges merge.
    indicators = np.array([0.1, 0.2, 0.3], dtype=dtype)
    features = np.array([[0.0], [0.1], [0.2], [0.3]], dtype=dtype)

    labels = bic.graph.agglomeration.NodeAndEdgeWeightedClusterPolicy(
        num_clusters_stop=1, beta=0.5, size_regularizer=0.0
    ).optimize(graph, indicators, features)

    assert_same_partition(labels, [0, 0, 0, 0])


def test_beta_zero_reproduces_edge_weighted():
    graph = two_clusters_graph()
    # High boundary on the bridge edge, low boundaries inside each triangle.
    indicators = np.array(
        [0.1, 0.15, 0.12, 0.08, 0.09, 0.11, 0.9], dtype=np.float64
    )
    features = np.random.RandomState(0).rand(6, 3).astype(np.float64)

    labels_node = bic.graph.agglomeration.NodeAndEdgeWeightedClusterPolicy(
        num_clusters_stop=2, beta=0.0
    ).optimize(graph, indicators, features)
    labels_edge = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=2
    ).optimize(graph, indicators)

    assert_same_partition(labels_node, labels_edge)


def test_beta_one_uses_feature_distance():
    # All edge indicators are large, so only the feature distance distinguishes
    # the candidate priorities. Features place the first three and the last
    # three nodes close together, with the bridge far apart. The min-heap
    # pops the smallest-distance edges first, so intra-cluster edges merge
    # before the bridge.
    graph = two_clusters_graph()
    indicators = np.ones(graph.number_of_edges, dtype=np.float64)
    features = np.array(
        [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]], dtype=np.float64
    )

    labels = bic.graph.agglomeration.NodeAndEdgeWeightedClusterPolicy(
        num_clusters_stop=2, beta=1.0, size_regularizer=0.0
    ).optimize(graph, indicators, features)

    assert_same_partition(labels, [0, 0, 0, 1, 1, 1])


def test_node_features_shape_mismatch_raises():
    graph = chain_graph(4)
    with pytest.raises(ValueError):
        bic.graph.agglomeration.NodeAndEdgeWeightedClusterPolicy(
            num_clusters_stop=1
        ).optimize(
            graph,
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
            np.array([[0.0], [0.1], [0.2]], dtype=np.float64),
        )


def test_float32_and_float64_match():
    graph = two_clusters_graph()
    indicators_f32 = np.array(
        [0.1, 0.15, 0.12, 0.08, 0.09, 0.11, 0.9], dtype=np.float32
    )
    features_f32 = np.random.RandomState(1).rand(6, 2).astype(np.float32)

    labels_f32 = bic.graph.agglomeration.NodeAndEdgeWeightedClusterPolicy(
        num_clusters_stop=2
    ).optimize(graph, indicators_f32, features_f32)
    labels_f64 = bic.graph.agglomeration.NodeAndEdgeWeightedClusterPolicy(
        num_clusters_stop=2
    ).optimize(
        graph,
        indicators_f32.astype(np.float64),
        features_f32.astype(np.float64),
    )

    assert_same_partition(labels_f32, labels_f64)
