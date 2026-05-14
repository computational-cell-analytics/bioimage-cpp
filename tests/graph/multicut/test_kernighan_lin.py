import numpy as np
import bioimage_cpp as bic


def test_kernighan_lin_improves_or_preserves_energy(frustrated_triangle):
    graph, costs = frustrated_triangle
    objective = bic.graph.MulticutObjective(graph, costs)
    before = objective.energy()

    labels = bic.graph.KernighanLinMulticut(number_of_outer_iterations=10).optimize(objective)

    assert objective.energy(labels) <= before
    assert labels.dtype == np.uint64


def test_kernighan_lin_accepts_initial_labels(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.MulticutObjective(graph, costs, initial_labels=[0, 0, 0, 1])
    before = objective.energy()

    labels = bic.graph.KernighanLinMulticut(number_of_outer_iterations=5).optimize(objective)

    assert objective.energy(labels) <= before


def test_kernighan_lin_on_external_toy_problem(external_toy_problem):
    graph, costs, _ = external_toy_problem
    objective = bic.graph.MulticutObjective(graph, costs)

    labels = bic.graph.KernighanLinMulticut(number_of_outer_iterations=20).optimize(objective)

    assert objective.energy(labels) <= -35.0


def test_kernighan_lin_energy_bound_on_grid_problem(grid_problem):
    graph, costs = grid_problem
    objective = bic.graph.MulticutObjective(graph, costs)

    labels = bic.graph.KernighanLinMulticut(number_of_outer_iterations=10).optimize(objective)

    assert objective.energy(labels) <= -20.0
