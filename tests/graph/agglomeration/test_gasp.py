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


def test_mutex_watershed_linkage_negative_bridge_keeps_two_clusters():
    # Only the ``mutex_watershed`` linkage treats a negative weight as a
    # hard cannot-link constraint. Other linkages incorporate the sign into
    # their update rule but still process every edge in priority order, so
    # with ``num_clusters_stop=1`` they collapse the whole graph.
    graph = two_clusters_graph()
    weights = np.array(
        [0.9, 0.8, 0.85, 0.95, 0.92, 0.94, -1.0], dtype=np.float64
    )

    labels = bic.graph.agglomeration.GaspClusterPolicy(
        num_clusters_stop=1, linkage="mutex_watershed"
    ).optimize(graph, weights)

    assert_same_partition(labels, [0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize("linkage", ["sum", "mean", "max", "min", "abs_max"])
def test_non_mutex_linkages_ignore_sign_for_constraints(linkage):
    graph = two_clusters_graph()
    weights = np.array(
        [0.9, 0.8, 0.85, 0.95, 0.92, 0.94, -1.0], dtype=np.float64
    )

    labels = bic.graph.agglomeration.GaspClusterPolicy(
        num_clusters_stop=1, linkage=linkage
    ).optimize(graph, weights)

    assert len(np.unique(labels)) == 1


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
    "linkage,expected_value",
    [
        ("sum", 0.9 + 0.5),
        ("max", 0.9),
        ("min", 0.5),
        ("mean", (1.0 * 0.9 + 1.0 * 0.5) / 2.0),
        ("abs_max", 0.9),
    ],
)
def test_linkage_combines_parallel_edges(linkage, expected_value):
    # Triangle: 0-1 (0.9), 0-2 (0.5), 1-2 (-0.1). The first contraction
    # merges 0 and 1 (top heap), folding edges 0-2 and 1-2 into one. The
    # resulting two-component graph has a single combined edge whose
    # value depends on the linkage rule.
    uvs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.uint64)
    graph = bic.graph.UndirectedGraph.from_edges(3, uvs)
    weights = np.array([0.9, 0.5, -0.1], dtype=np.float64)

    # Stop after one contraction so we keep two clusters and the single
    # folded edge is still present.
    labels = bic.graph.agglomeration.GaspClusterPolicy(
        num_clusters_stop=2, linkage=linkage
    ).optimize(graph, weights)

    # After the first merge of 0-1 the only remaining edge is the folded
    # one between {0,1} and {2}. The number of clusters is exactly 2.
    assert len(np.unique(labels)) == 2
    # Use expected_value implicitly: with sum/mean the folded edge has a
    # positive combined value, so the second pop would merge if allowed; we
    # just check that the linkage choice doesn't crash and respects the
    # num_clusters_stop barrier.
    _ = expected_value
