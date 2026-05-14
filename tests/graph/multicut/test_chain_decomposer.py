import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import same_partition


def test_chained_multicut_solvers(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.MulticutObjective(graph, costs)
    solver = bic.graph.ChainedMulticutSolvers(
        [
            bic.graph.GreedyAdditiveMulticut(),
            bic.graph.KernighanLinMulticut(number_of_outer_iterations=5),
        ]
    )

    labels = solver.optimize(objective)

    same_partition(labels, [0, 0, 0, 1])
    assert objective.energy(labels) == pytest.approx(-5.0)


def test_chained_solver_rejects_empty_chain():
    with pytest.raises(ValueError, match="at least one"):
        bic.graph.ChainedMulticutSolvers([])


def test_multicut_decomposer_solves_positive_components():
    graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [1, 2], [2, 3]])
    objective = bic.graph.MulticutObjective(graph, [1.0, -5.0, 1.0])
    solver = bic.graph.MulticutDecomposer(bic.graph.GreedyAdditiveMulticut())

    labels = solver.optimize(objective)

    same_partition(labels, [0, 0, 1, 1])
    assert objective.energy() == pytest.approx(-5.0)


def test_multicut_decomposer_uses_fallthrough_solver_for_single_component():
    class SingletonSolver(bic.graph.MulticutSolver):
        def optimize(self, objective):
            objective.labels = np.arange(objective.graph.number_of_nodes, dtype=np.uint64)
            return objective.labels

    graph = bic.graph.UndirectedGraph.from_edges(2, [[0, 1]])
    objective = bic.graph.MulticutObjective(graph, [1.0])
    solver = bic.graph.MulticutDecomposer(
        bic.graph.GreedyAdditiveMulticut(),
        fallthrough_solver=SingletonSolver(),
    )

    labels = solver.optimize(objective)

    same_partition(labels, [0, 1])


def test_decomposer_on_external_toy_problem(external_toy_problem):
    graph, costs, _ = external_toy_problem
    objective = bic.graph.MulticutObjective(graph, costs)
    solver = bic.graph.MulticutDecomposer(
        bic.graph.ChainedMulticutSolvers(
            [
                bic.graph.GreedyAdditiveMulticut(),
                bic.graph.KernighanLinMulticut(number_of_outer_iterations=10),
            ]
        )
    )

    labels = solver.optimize(objective)

    assert objective.energy(labels) <= -35.0


def test_decomposer_energy_bound_on_grid_problem(grid_problem):
    graph, costs = grid_problem
    objective = bic.graph.MulticutObjective(graph, costs)
    solver = bic.graph.MulticutDecomposer(bic.graph.GreedyAdditiveMulticut())

    labels = solver.optimize(objective)

    assert objective.energy(labels) <= -20.0
