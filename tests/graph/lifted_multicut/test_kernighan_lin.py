import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import same_partition


def test_kl_splits_chain_along_repulsive_lifted_edge(chain_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = chain_with_lifted
    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )

    # Warm-starting from a single cluster (everything merged) and running KL
    # should split the chain so that the repulsive (0, 3) lifted edge is cut.
    objective.labels = np.zeros(4, dtype=np.uint64)
    labels = bic.graph.LiftedKernighanLinMulticut(
        number_of_outer_iterations=10
    ).optimize(objective)

    # Energy should improve over the single-cluster labeling.
    assert objective.energy(labels) < objective.energy(np.zeros(4, dtype=np.uint64))
    # The lifted edge endpoints must end up in different clusters.
    assert labels[0] != labels[3]


def test_kl_keeps_base_disconnected_clusters_separate(
    disjoint_clusters_with_attractive_lifted,
):
    base, base_costs, lifted_uvs, lifted_costs = disjoint_clusters_with_attractive_lifted
    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )

    labels = bic.graph.LiftedKernighanLinMulticut(
        number_of_outer_iterations=10
    ).optimize(objective)
    # Lifted (0, 2) is attractive but the base graph offers no path between
    # the two components; lifted KL must keep every cluster base-graph
    # connected.
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_kl_matches_multicut_without_lifted_edges():
    base = bic.graph.UndirectedGraph.from_edges(
        6,
        [
            [0, 1], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [4, 5],
        ],
    )
    base_costs = np.array([5, -20, 5, 5, -20, 5, 5], dtype=np.float64)

    mc_objective = bic.graph.MulticutObjective(base, base_costs)
    mc_labels = bic.graph.KernighanLinMulticut(
        number_of_outer_iterations=10
    ).optimize(mc_objective)

    lmc_objective = bic.graph.LiftedMulticutObjective(base, base_costs)
    lmc_labels = bic.graph.LiftedKernighanLinMulticut(
        number_of_outer_iterations=10
    ).optimize(lmc_objective)

    same_partition(lmc_labels, mc_labels)


def test_kl_lifted_attractive_overrides_repulsive_base():
    # Chain 0-1-2 with one weakly attractive base edge and one strongly
    # repulsive base edge; an attractive lifted (0, 2) tips the balance back
    # toward keeping everything in one cluster. KL warm-starts from the
    # greedy result {0,1}|{2} (which only sees the base graph) and must
    # discover the merge once the lifted edge is considered.
    base = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])
    base_costs = np.array([1.0, -5.0], dtype=np.float64)
    lifted_uvs = np.array([[0, 2]], dtype=np.uint64)
    lifted_costs = np.array([10.0], dtype=np.float64)

    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )
    labels = bic.graph.LiftedKernighanLinMulticut(
        number_of_outer_iterations=10
    ).optimize(objective)

    same_partition(labels, [0, 0, 0])
    assert objective.energy(labels) == pytest.approx(0.0)


def test_kl_warm_starts_from_singleton():
    base = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])
    base_costs = np.array([2.0, 2.0], dtype=np.float64)
    objective = bic.graph.LiftedMulticutObjective(base, base_costs)
    # Singleton labels trigger an internal greedy-additive warm start.
    labels = bic.graph.LiftedKernighanLinMulticut(
        number_of_outer_iterations=5
    ).optimize(objective)
    same_partition(labels, [0, 0, 0])
