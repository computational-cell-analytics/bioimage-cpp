import pytest

import bioimage_cpp as bic


def test_solvers_on_full_external_problem(tmp_path):
    try:
        graph, costs = bic.graph.multicut.load_external_multicut_problem(timeout=30)
    except (FileNotFoundError, RuntimeError) as error:
        pytest.skip(str(error))

    solvers = [
        bic.graph.multicut.GreedyAdditiveMulticut(),
        bic.graph.multicut.GreedyFixationMulticut(),
        bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
        bic.graph.multicut.ChainedMulticutSolvers(
            [
                bic.graph.multicut.GreedyAdditiveMulticut(),
                bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
            ]
        ),
        bic.graph.multicut.MulticutDecomposer(bic.graph.multicut.GreedyAdditiveMulticut()),
    ]

    for solver in solvers:
        objective = bic.graph.multicut.MulticutObjective(graph, costs)
        labels = solver.optimize(objective)
        assert labels.shape == (graph.number_of_nodes,)
        assert objective.energy(labels) <= -76900.0
