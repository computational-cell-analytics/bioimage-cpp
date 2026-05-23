from __future__ import annotations

import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import (
    assert_same_partition,
    chain_graph,
    two_clusters_graph,
)


LINKAGES = ["sum", "mean", "max", "min", "abs_max", "mutex_watershed"]


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_positive_collapses_to_one_cluster(linkage, dtype):
    graph = chain_graph(4)
    weights = np.array([0.9, 0.5, 0.7], dtype=dtype)

    labels = bic.graph.agglomeration.GaspClusterPolicy(
        num_clusters_stop=1, linkage=linkage
    ).optimize(graph, weights)

    assert_same_partition(labels, [0, 0, 0, 0])


@pytest.mark.parametrize("linkage", LINKAGES)
def test_negative_bridge_keeps_two_clusters(linkage):
    # All linkages observe the "stop when no positive edges remain" rule,
    # matching ``nifty.graph.agglo``. The ``mutex_watershed`` linkage gets
    # there via cannot-link constraints on negative heap pops; the others
    # via the global signed-priority stop check in ``next_action``.
    graph = two_clusters_graph()
    weights = np.array(
        [0.9, 0.8, 0.85, 0.95, 0.92, 0.94, -1.0], dtype=np.float64
    )

    labels = bic.graph.agglomeration.GaspClusterPolicy(
        num_clusters_stop=1, linkage=linkage
    ).optimize(graph, weights)

    assert_same_partition(labels, [0, 0, 0, 1, 1, 1])


def test_mutex_watershed_linkage_matches_mutex_watershed():
    # Run GASP-mutex_watershed and the reference mutex_watershed_clustering
    # on the same data; the partitions should agree.
    graph = two_clusters_graph()
    weights = np.array(
        [0.9, 0.8, 0.85, 0.95, 0.92, 0.94, -1.0], dtype=np.float64
    )

    gasp_labels = bic.graph.agglomeration.GaspClusterPolicy(
        num_clusters_stop=1, linkage="mutex_watershed"
    ).optimize(graph, weights)

    # Mutex-watershed reference: split positive (attractive) and negative
    # (repulsive) edges into the two arrays it expects.
    positive_mask = weights >= 0
    uvs = np.array(
        [
            [0, 1], [1, 2], [0, 2],
            [3, 4], [4, 5], [3, 5],
            [2, 3],
        ],
        dtype=np.uint64,
    )
    pos_uvs = uvs[positive_mask]
    pos_costs = weights[positive_mask]
    neg_uvs = uvs[~positive_mask]
    neg_costs = -weights[~positive_mask]

    # The attractive base graph must only contain positive edges; rebuild it.
    base_graph = bic.graph.UndirectedGraph.from_edges(6, pos_uvs)
    pos_edge_costs = np.ascontiguousarray(pos_costs.astype(np.float64))
    mw_labels = bic.graph.mutex_watershed.mutex_watershed_clustering(
        base_graph, pos_edge_costs, neg_uvs, np.ascontiguousarray(neg_costs)
    )

    assert_same_partition(gasp_labels, mw_labels)


def test_is_mergeable_mask_creates_extra_cluster():
    graph = two_clusters_graph()
    weights = np.array(
        [0.9, 0.8, 0.85, 0.95, 0.92, 0.94, 0.99], dtype=np.float64
    )
    is_mergeable = np.array([1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)

    labels = bic.graph.agglomeration.GaspClusterPolicy(
        num_clusters_stop=1, linkage="mean"
    ).optimize(graph, weights, is_mergeable=is_mergeable)

    assert_same_partition(labels, [0, 0, 0, 1, 1, 1])


def test_invalid_linkage_raises():
    with pytest.raises(ValueError):
        bic.graph.agglomeration.GaspClusterPolicy(linkage="bogus")


def test_weight_length_mismatch_raises():
    graph = chain_graph(3)
    with pytest.raises(ValueError):
        bic.graph.agglomeration.GaspClusterPolicy(num_clusters_stop=1).optimize(
            graph, np.array([0.1, 0.2, 0.3], dtype=np.float64)
        )


@pytest.mark.parametrize(
    "linkage,expect_one_cluster",
    [
        ("sum", True),       # combined = 0.4 > 0 → next pop still merges
        ("mean", True),      # combined = 0.2 > 0 → next pop still merges
        ("max", True),       # combined = max(0.5, -0.1) = 0.5 > 0
        ("abs_max", True),   # combined = 0.5 (largest |w|) > 0
        ("min", False),      # combined = min(0.5, -0.1) = -0.1 → stop
    ],
)
def test_linkage_combines_parallel_edges(linkage, expect_one_cluster):
    # Triangle: 0-1 (0.9), 0-2 (0.5), 1-2 (-0.1). The first contraction
    # merges 0 and 1 (top heap), folding edges 0-2 and 1-2 into one. The
    # combined weight depends on the linkage rule; with the signed-priority
    # stop criterion, ``min`` is the only rule that drops the combined
    # weight below zero and therefore halts before the second merge.
    uvs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.uint64)
    graph = bic.graph.UndirectedGraph.from_edges(3, uvs)
    weights = np.array([0.9, 0.5, -0.1], dtype=np.float64)

    labels = bic.graph.agglomeration.GaspClusterPolicy(
        num_clusters_stop=1, linkage=linkage
    ).optimize(graph, weights)

    expected = 1 if expect_one_cluster else 2
    assert len(np.unique(labels)) == expected


def test_signed_priority_stop_matches_nifty_semantics():
    # With non-mutex_watershed linkages and ``num_clusters_stop=1``, the
    # agglomeration must still leave clusters separated by negative-weight
    # bridges. Mirrors `nifty.graph.agglo`'s "no attractive edges remain"
    # termination.
    graph = two_clusters_graph()
    weights = np.array(
        [0.9, 0.8, 0.85, 0.95, 0.92, 0.94, -10.0], dtype=np.float64
    )
    for linkage in ("sum", "mean", "max", "abs_max"):
        labels = bic.graph.agglomeration.GaspClusterPolicy(
            num_clusters_stop=1, linkage=linkage
        ).optimize(graph, weights)
        assert_same_partition(labels, [0, 0, 0, 1, 1, 1])
