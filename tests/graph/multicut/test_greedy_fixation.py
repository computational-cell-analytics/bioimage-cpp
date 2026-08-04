import numpy as np
import pytest

import bioimage_cpp as bic


def test_greedy_fixation_respects_negative_constraints(frustrated_triangle):
    graph, costs = frustrated_triangle
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.GreedyFixationMulticut().optimize(objective)

    assert labels[0] != labels[2]
    assert objective.energy(labels) <= -3.0


def test_greedy_fixation_propagates_two_folded_constraints():
    graph = bic.graph.UndirectedGraph.from_edges(
        3,
        [[0, 1], [0, 2], [1, 2]],
    )
    costs = np.array([3.0, -5.0, -4.0], dtype=np.float64)
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.GreedyFixationMulticut().optimize(objective)

    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
    assert objective.energy(labels) == pytest.approx(-9.0)


def test_greedy_fixation_node_num_stop(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.GreedyFixationMulticut(node_num_stop=3).optimize(objective)

    assert len(set(labels.tolist())) == 3


def test_greedy_fixation_on_external_toy_problem(external_toy_problem):
    graph, costs, _ = external_toy_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.GreedyFixationMulticut().optimize(objective)

    assert objective.energy(labels) <= -35.0


def test_greedy_fixation_energy_bound_on_grid_problem(grid_problem):
    graph, costs = grid_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.GreedyFixationMulticut().optimize(objective)

    assert objective.energy(labels) <= -20.0
