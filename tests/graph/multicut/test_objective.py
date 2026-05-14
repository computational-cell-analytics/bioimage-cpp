import numpy as np
import pytest

import bioimage_cpp as bic


def test_multicut_objective_energy_and_validation(frustrated_triangle):
    graph, costs = frustrated_triangle
    objective = bic.graph.MulticutObjective(graph, costs)

    assert objective.energy([0, 0, 1]) == pytest.approx(-3.0)
    objective.labels = [0, 0, 1]
    assert objective.energy() == pytest.approx(-3.0)

    with pytest.raises(ValueError, match="edge_costs"):
        bic.graph.MulticutObjective(graph, [1.0, 2.0])
    with pytest.raises(ValueError, match="labels"):
        objective.labels = [0, 1]


def test_multicut_objective_copies_graph_topology(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.MulticutObjective(graph, costs)

    graph.insert_edge(0, 3)

    assert objective.graph.number_of_edges == 3
    np.testing.assert_array_equal(
        objective.graph.uv_ids(),
        np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint64),
    )


def test_multicut_objective_reset_labels(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.MulticutObjective(graph, costs, initial_labels=[0, 0, 1, 1])

    objective.reset_labels()

    np.testing.assert_array_equal(objective.labels, np.arange(4, dtype=np.uint64))
