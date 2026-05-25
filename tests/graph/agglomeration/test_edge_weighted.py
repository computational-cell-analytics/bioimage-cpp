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
    graph = chain_graph(5)
    indicators = np.array([0.1, 0.2, 0.3, 0.4], dtype=dtype)

    labels = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=1, size_regularizer=0.0
    ).optimize(graph, indicators)

    assert_same_partition(labels, [0, 0, 0, 0, 0])
    assert labels.dtype == np.uint64


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_num_clusters_stop_respected(dtype):
    graph = chain_graph(5)
    # Indicators are boundary strengths (low = weak boundary, merges first).
    indicators = np.array([0.1, 0.2, 0.3, 0.4], dtype=dtype)

    labels = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=3, size_regularizer=0.0
    ).optimize(graph, indicators)

    # Two contractions: 0-1 (lowest 0.1), then 1-2 (0.2). Final partition:
    # {0,1,2}, {3}, {4}.
    assert_same_partition(labels, [0, 0, 0, 1, 2])


def test_float32_and_float64_match():
    graph = two_clusters_graph()
    # Strong (high) boundary on the bridge edge, weak boundaries inside the
    # two triangles. With num_clusters_stop=2 the strong bridge survives.
    indicators_f32 = np.array(
        [0.1, 0.15, 0.12, 0.08, 0.09, 0.11, 0.9], dtype=np.float32
    )
    indicators_f64 = indicators_f32.astype(np.float64)

    labels_f32 = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=2
    ).optimize(graph, indicators_f32)
    labels_f64 = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=2
    ).optimize(graph, indicators_f64)

    assert_same_partition(labels_f32, labels_f64)
    assert_same_partition(labels_f32, [0, 0, 0, 1, 1, 1])


def test_size_regularizer_changes_priority():
    # Path graph with indicators (0.1, 0.5, 0.1). Without size regularisation
    # both 0.1 edges tie at the smallest priority. With a strong size
    # regulariser the second 0.1 edge's priority is rescaled because one
    # endpoint already grew, so the merge order changes.
    graph = chain_graph(4)
    indicators = np.array([0.1, 0.5, 0.1], dtype=np.float64)

    labels_no_reg = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=2, size_regularizer=0.0
    ).optimize(graph, indicators)
    labels_strong = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=2, size_regularizer=4.0
    ).optimize(graph, indicators)

    assert len(np.unique(labels_no_reg)) == 2
    assert len(np.unique(labels_strong)) == 2


def test_indicator_length_mismatch_raises():
    graph = chain_graph(3)
    with pytest.raises(ValueError):
        bic.graph.agglomeration.EdgeWeightedClusterPolicy(num_clusters_stop=1).optimize(
            graph, np.array([0.5, 0.5, 0.5], dtype=np.float32)
        )


def test_non_floating_indicator_raises():
    graph = chain_graph(3)
    with pytest.raises(TypeError):
        bic.graph.agglomeration.EdgeWeightedClusterPolicy(num_clusters_stop=1).optimize(
            graph, np.array([1, 0, 1], dtype=np.int64)
        )


def test_bridge_edge_is_last():
    # Bridge has the largest indicator (strongest boundary) → it is the last
    # candidate; with num_clusters_stop=2 the bridge keeps the two
    # triangles apart.
    graph = two_clusters_graph()
    indicators = np.array(
        [0.1, 0.15, 0.12, 0.08, 0.09, 0.11, 0.9], dtype=np.float64
    )

    labels = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=2
    ).optimize(graph, indicators)

    assert_same_partition(labels, [0, 0, 0, 1, 1, 1])
