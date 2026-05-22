import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import same_partition


def test_greedy_additive_merges_positive_components(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective)

    same_partition(labels, [0, 0, 0, 1])
    np.testing.assert_array_equal(objective.labels, labels)
    assert objective.energy() == pytest.approx(-5.0)


def test_greedy_additive_respects_weight_stop(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.GreedyAdditiveMulticut(weight_stop=1.5).optimize(objective)

    same_partition(labels, [0, 1, 1, 2])


def test_greedy_additive_on_external_toy_problem(external_toy_problem):
    graph, costs, _ = external_toy_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective)

    assert objective.energy(labels) <= -35.0


def test_greedy_additive_energy_bound_on_grid_problem(grid_problem):
    graph, costs = grid_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective)

    assert objective.energy(labels) <= -20.0
