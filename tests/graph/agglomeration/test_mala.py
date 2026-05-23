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
def test_low_threshold_collapses_all(dtype):
    graph = chain_graph(5)
    indicators = np.array([0.1, 0.2, 0.15, 0.05], dtype=dtype)

    labels = bic.graph.agglomeration.MalaClusterPolicy(
        threshold=1.0, num_clusters_stop=1
    ).optimize(graph, indicators)

    assert_same_partition(labels, [0, 0, 0, 0, 0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_threshold_stops_early(dtype):
    graph = chain_graph(5)
    indicators = np.array([0.1, 0.9, 0.15, 0.05], dtype=dtype)

    # Threshold 0.5: the 0.9 edge is above threshold and never gets popped
    # for merging. The other three edges are below threshold and merge in
    # ascending priority order until they hit the 0.9 boundary.
    labels = bic.graph.agglomeration.MalaClusterPolicy(
        threshold=0.5, num_clusters_stop=1
    ).optimize(graph, indicators)

    # Nodes around the 0.9 edge should stay split.
    assert len(np.unique(labels)) >= 2


def test_num_clusters_stop_respected():
    graph = chain_graph(5)
    indicators = np.array([0.1, 0.2, 0.15, 0.05], dtype=np.float64)

    labels = bic.graph.agglomeration.MalaClusterPolicy(
        threshold=1.0, num_clusters_stop=3
    ).optimize(graph, indicators)

    assert len(np.unique(labels)) == 3


def test_float32_and_float64_match():
    graph = two_clusters_graph()
    indicators_f32 = np.array(
        [0.1, 0.15, 0.12, 0.08, 0.09, 0.11, 0.8], dtype=np.float32
    )
    labels_f32 = bic.graph.agglomeration.MalaClusterPolicy(
        threshold=0.5, num_clusters_stop=1
    ).optimize(graph, indicators_f32)
    labels_f64 = bic.graph.agglomeration.MalaClusterPolicy(
        threshold=0.5, num_clusters_stop=1
    ).optimize(graph, indicators_f32.astype(np.float64))

    assert_same_partition(labels_f32, labels_f64)


def test_bad_bin_range_raises():
    graph = chain_graph(3)
    with pytest.raises(Exception):
        bic.graph.agglomeration.MalaClusterPolicy(
            bin_min=1.0, bin_max=0.0
        ).optimize(graph, np.array([0.1, 0.1], dtype=np.float64))


def test_zero_bins_raises():
    graph = chain_graph(3)
    with pytest.raises(Exception):
        bic.graph.agglomeration.MalaClusterPolicy(num_bins=0).optimize(
            graph, np.array([0.1, 0.1], dtype=np.float64)
        )


def test_indicator_length_mismatch_raises():
    graph = chain_graph(3)
    with pytest.raises(ValueError):
        bic.graph.agglomeration.MalaClusterPolicy().optimize(
            graph, np.array([0.1, 0.2, 0.3], dtype=np.float64)
        )
