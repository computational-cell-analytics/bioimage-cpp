import numpy as np
import pytest

import bioimage_cpp as bic
from bioimage_cpp import _core


def test_watershed_proposal_generator_is_deterministic_given_seed():
    # Two generators with the same seed must produce the same proposal series.
    graph = bic.graph.UndirectedGraph.from_edges(
        6,
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 3]],
    )
    costs = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 0.5], dtype=np.float64)

    objective_a = bic.graph.multicut.MulticutObjective(graph, costs)
    objective_b = bic.graph.multicut.MulticutObjective(graph, costs)

    solver_a = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=42),
        number_of_iterations=3,
    )
    solver_b = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=42),
        number_of_iterations=3,
    )

    labels_a = solver_a.optimize(objective_a)
    labels_b = solver_b.optimize(objective_b)
    np.testing.assert_array_equal(labels_a, labels_b)


def test_greedy_additive_proposal_generator_is_deterministic_given_seed(grid_problem):
    graph, costs = grid_problem
    objective_a = bic.graph.multicut.MulticutObjective(graph, costs)
    objective_b = bic.graph.multicut.MulticutObjective(graph, costs)

    solver_a = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.GreedyAdditiveProposalGenerator(seed=5),
        number_of_iterations=3,
    )
    solver_b = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.GreedyAdditiveProposalGenerator(seed=5),
        number_of_iterations=3,
    )
    np.testing.assert_array_equal(
        solver_a.optimize(objective_a),
        solver_b.optimize(objective_b),
    )


def test_cpp_proposal_generator_validates_costs_length():
    graph = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])
    bad_costs = np.array([0.5], dtype=np.float64)
    with pytest.raises((ValueError, RuntimeError)):
        _core._WatershedProposalGenerator(graph, bad_costs)
    with pytest.raises((ValueError, RuntimeError)):
        _core._GreedyAdditiveMulticutProposalGenerator(graph, bad_costs)


def test_proposal_generator_isinstance_check():
    assert isinstance(
        bic.graph.multicut.WatershedProposalGenerator(), bic.graph.multicut.ProposalGenerator
    )
    assert isinstance(
        bic.graph.multicut.GreedyAdditiveProposalGenerator(), bic.graph.multicut.ProposalGenerator
    )
