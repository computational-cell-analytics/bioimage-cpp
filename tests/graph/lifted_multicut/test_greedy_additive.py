import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import same_partition


def test_greedy_additive_stops_before_lifted_repulsive_dominates(chain_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = chain_with_lifted
    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )

    labels = bic.graph.LiftedGreedyAdditiveMulticut().optimize(objective)
    # As base edges get contracted, the lifted -10 weight is folded into the
    # remaining contracted edge to the unmerged endpoint. Once the summed
    # weight drops below `weight_stop=0` greedy stops, leaving the chain
    # split — which is the optimal labeling (energy -8 vs 0 for the fully
    # merged labeling).
    assert labels[0] != labels[3]
    assert objective.energy(labels) == pytest.approx(-8.0)
    np.testing.assert_array_equal(objective.labels, labels)


def test_greedy_additive_keeps_base_disconnected_clusters(
    disjoint_clusters_with_attractive_lifted,
):
    base, base_costs, lifted_uvs, lifted_costs = disjoint_clusters_with_attractive_lifted
    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )

    labels = bic.graph.LiftedGreedyAdditiveMulticut().optimize(objective)
    # Lifted (0, 2) is attractive but there's no base path to merge across,
    # so greedy-additive must keep the two base components separate.
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_greedy_additive_respects_weight_stop(chain_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = chain_with_lifted
    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )

    labels = bic.graph.LiftedGreedyAdditiveMulticut(weight_stop=3.0).optimize(objective)
    # Every base edge has weight 2.0, below the stop threshold, so no contractions.
    same_partition(labels, [0, 1, 2, 3])


def test_greedy_additive_matches_multicut_when_no_lifted_edges():
    # With zero lifted edges, the lifted greedy-additive on the lifted graph
    # should yield the same labeling as plain multicut greedy-additive on the
    # base graph.
    base = bic.graph.UndirectedGraph.from_edges(
        6,
        [
            [0, 1], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [4, 5],
        ],
    )
    base_costs = np.array([5, -20, 5, 5, -20, 5, 5], dtype=np.float64)

    mc_objective = bic.graph.MulticutObjective(base, base_costs)
    mc_labels = bic.graph.GreedyAdditiveMulticut().optimize(mc_objective)

    lmc_objective = bic.graph.LiftedMulticutObjective(base, base_costs)
    lmc_labels = bic.graph.LiftedGreedyAdditiveMulticut().optimize(lmc_objective)

    same_partition(lmc_labels, mc_labels)


def test_greedy_additive_lifted_attractive_merges_through_base_path():
    # Chain 0-1-2 with a strong attractive lifted edge (0, 2).
    # Even though greedy can't contract a lifted edge, contracting either base
    # edge folds the lifted weight into the remaining contracted edge, which
    # then becomes a normal heap candidate.
    base = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])
    base_costs = np.array([1.0, 1.0], dtype=np.float64)
    lifted_uvs = np.array([[0, 2]], dtype=np.uint64)
    lifted_costs = np.array([10.0], dtype=np.float64)

    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )
    labels = bic.graph.LiftedGreedyAdditiveMulticut().optimize(objective)
    same_partition(labels, [0, 0, 0])
