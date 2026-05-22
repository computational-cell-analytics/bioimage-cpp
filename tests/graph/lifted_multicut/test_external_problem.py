import pytest

import bioimage_cpp as bic


# Energy bound for the 2D ISBI lifted multicut problem. Every shipped solver
# (including the chained greedy + KL combination) should reach an energy at
# most this high. Baseline measurements at the time of writing:
#   singleton labelling        : -1263.9
#   LiftedGreedyAdditive       : -1575.0
#   LiftedKernighanLin (10)    : -1575.2
#   chained (greedy + KL)      : -1575.2
# Margin of ~0.5 catches meaningful regressions while tolerating numerical drift.
ENERGY_BOUND = -1574.5


def _load_problem():
    try:
        return bic.graph.lifted_multicut.load_lifted_multicut_problem("2d", timeout=30)
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError) as error:
        pytest.skip(str(error))


def test_solvers_on_2d_lifted_problem():
    problem = _load_problem()
    graph = bic.graph.UndirectedGraph.from_edges(problem.n_nodes, problem.local_uvs)

    solvers = [
        bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut(),
        bic.graph.lifted_multicut.LiftedKernighanLinMulticut(number_of_outer_iterations=10),
        bic.graph.lifted_multicut.LiftedChainedSolvers(
            [
                bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut(),
                bic.graph.lifted_multicut.LiftedKernighanLinMulticut(number_of_outer_iterations=10),
            ]
        ),
    ]

    for solver in solvers:
        objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
            graph,
            problem.local_costs,
            lifted_uvs=problem.lifted_uvs,
            lifted_costs=problem.lifted_costs,
        )
        labels = solver.optimize(objective)
        assert labels.shape == (graph.number_of_nodes,)
        assert objective.energy(labels) <= ENERGY_BOUND, (
            f"{type(solver).__name__} energy {objective.energy(labels):.3f} "
            f"exceeds regression bound {ENERGY_BOUND}"
        )
