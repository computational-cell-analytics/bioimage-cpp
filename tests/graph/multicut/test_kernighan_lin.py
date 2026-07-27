import numpy as np
import pytest

import bioimage_cpp as bic


def test_kernighan_lin_improves_or_preserves_energy(frustrated_triangle):
    graph, costs = frustrated_triangle
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    before = objective.energy()

    labels = bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=10).optimize(objective)

    assert objective.energy(labels) <= before
    assert labels.dtype == np.uint64


def test_kernighan_lin_finds_frustrated_triangle_optimum(frustrated_triangle):
    graph, costs = frustrated_triangle
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=10).optimize(objective)

    assert objective.energy(labels) == pytest.approx(-3.0)


def test_kernighan_lin_accepts_initial_labels(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs, initial_labels=[0, 0, 0, 1])
    before = objective.energy()

    labels = bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5).optimize(objective)

    assert objective.energy(labels) <= before


def test_kernighan_lin_on_external_toy_problem(external_toy_problem):
    graph, costs, _ = external_toy_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=20).optimize(objective)

    # The move-chain implementation reaches the optimum on this instance.
    assert objective.energy(labels) == pytest.approx(-35.0)


def test_kernighan_lin_energy_bound_on_grid_problem(grid_problem):
    graph, costs = grid_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    labels = bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=10).optimize(objective)

    # Regression guard pinned to the move-chain implementation's converged energy.
    assert objective.energy(labels) <= -34.0


def test_kernighan_lin_multi_iteration_regression():
    edges = np.array(
        [
            [0, 2],
            [0, 3],
            [0, 6],
            [0, 7],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 6],
            [3, 4],
            [3, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [4, 7],
            [5, 6],
            [5, 7],
            [6, 7],
        ],
        dtype=np.uint64,
    )
    costs = np.array(
        [
            4, 3, -1, 3, 1, -5, 0, 0, -4, -5,
            -3, 5, -4, 0, 3, 5, 5, -5, 5, -4,
        ],
        dtype=np.float64,
    )
    graph = bic.graph.UndirectedGraph.from_edges(8, edges)

    expected = {
        1: (np.array([0, 0, 0, 1, 1, 1, 2, 1], dtype=np.uint64), -19.0),
        2: (np.array([0, 1, 1, 0, 0, 0, 2, 0], dtype=np.uint64), -21.0),
        10: (np.array([0, 1, 1, 0, 0, 0, 2, 0], dtype=np.uint64), -21.0),
    }
    for number_of_outer_iterations, (
        expected_labels,
        expected_energy,
    ) in expected.items():
        outputs = []
        for _ in range(3):
            objective = bic.graph.multicut.MulticutObjective(graph, costs)
            labels = bic.graph.multicut.KernighanLinMulticut(
                number_of_outer_iterations=number_of_outer_iterations
            ).optimize(objective)
            outputs.append(labels)
            assert objective.energy(labels) == pytest.approx(expected_energy)
        assert all(np.array_equal(labels, expected_labels) for labels in outputs)
