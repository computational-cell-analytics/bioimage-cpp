import threading

import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import same_partition


def test_chained_multicut_solvers(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    solver = bic.graph.multicut.ChainedMulticutSolvers(
        [
            bic.graph.multicut.GreedyAdditiveMulticut(),
            bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
        ]
    )

    labels = solver.optimize(objective)

    same_partition(labels, [0, 0, 0, 1])
    assert objective.energy(labels) == pytest.approx(-5.0)


def test_chained_solver_rejects_empty_chain():
    with pytest.raises(ValueError, match="at least one"):
        bic.graph.multicut.ChainedMulticutSolvers([])


def test_multicut_decomposer_solves_positive_components():
    graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [1, 2], [2, 3]])
    objective = bic.graph.multicut.MulticutObjective(graph, [1.0, -5.0, 1.0])
    solver = bic.graph.multicut.MulticutDecomposer(bic.graph.multicut.GreedyAdditiveMulticut())

    labels = solver.optimize(objective)

    same_partition(labels, [0, 0, 1, 1])
    assert objective.energy() == pytest.approx(-5.0)


def test_multicut_decomposer_uses_fallthrough_solver_for_single_component():
    class SingletonSolver(bic.graph.multicut.MulticutSolver):
        def optimize(self, objective):
            objective.labels = np.arange(objective.graph.number_of_nodes, dtype=np.uint64)
            return objective.labels

    graph = bic.graph.UndirectedGraph.from_edges(2, [[0, 1]])
    objective = bic.graph.multicut.MulticutObjective(graph, [1.0])
    solver = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.GreedyAdditiveMulticut(),
        fallthrough_solver=SingletonSolver(),
    )

    labels = solver.optimize(objective)

    same_partition(labels, [0, 1])


def test_decomposer_on_external_toy_problem(external_toy_problem):
    graph, costs, _ = external_toy_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    solver = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.ChainedMulticutSolvers(
            [
                bic.graph.multicut.GreedyAdditiveMulticut(),
                bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=10),
            ]
        )
    )

    labels = solver.optimize(objective)

    assert objective.energy(labels) <= -35.0


def test_decomposer_energy_bound_on_grid_problem(grid_problem):
    graph, costs = grid_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    solver = bic.graph.multicut.MulticutDecomposer(bic.graph.multicut.GreedyAdditiveMulticut())

    labels = solver.optimize(objective)

    assert objective.energy(labels) <= -20.0


class _BarrierSolver(bic.graph.multicut.MulticutSolver):
    def __init__(self, barrier, calls, call_lock):
        self.barrier = barrier
        self.calls = calls
        self.call_lock = call_lock

    def clone(self):
        return _BarrierSolver(self.barrier, self.calls, self.call_lock)

    def optimize(self, objective):
        with self.call_lock:
            self.calls.append((id(self), threading.get_ident()))
        self.barrier.wait(timeout=5.0)
        objective.labels = np.zeros(objective.graph.number_of_nodes, dtype=np.uint64)
        return objective.labels


def _two_component_objective():
    graph = bic.graph.UndirectedGraph.from_edges(
        4,
        [[0, 1], [1, 2], [2, 3]],
    )
    return bic.graph.multicut.MulticutObjective(graph, [1.0, -5.0, 1.0])


def test_decomposer_runs_distinct_solver_clones_in_parallel():
    calls = []
    solver = _BarrierSolver(
        threading.Barrier(2),
        calls,
        threading.Lock(),
    )
    objective = _two_component_objective()
    labels = bic.graph.multicut.MulticutDecomposer(
        solver,
        fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
        number_of_threads=2,
    ).optimize(objective)

    same_partition(labels, [0, 0, 1, 1])
    assert len(calls) == 2
    assert len({solver_id for solver_id, _ in calls}) == 2
    assert len({thread_id for _, thread_id in calls}) == 2


@pytest.mark.parametrize("number_of_threads", [0, 1, 2, 8])
def test_decomposer_is_deterministic_across_thread_counts(number_of_threads):
    objective = _two_component_objective()
    labels = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.GreedyFixationMulticut(),
        fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
        number_of_threads=number_of_threads,
    ).optimize(objective)

    same_partition(labels, [0, 0, 1, 1])
    assert objective.energy() == pytest.approx(-5.0)


def test_decomposer_handles_singleton_components_without_solver_call():
    class FailSolver(bic.graph.multicut.MulticutSolver):
        def optimize(self, objective):
            raise AssertionError("singleton component reached the solver")

    graph = bic.graph.UndirectedGraph(3)
    objective = bic.graph.multicut.MulticutObjective(graph, [])
    labels = bic.graph.multicut.MulticutDecomposer(
        FailSolver(),
        fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
        number_of_threads=2,
    ).optimize(objective)
    same_partition(labels, [0, 1, 2])


def test_decomposer_rejects_clone_that_returns_original_solver():
    class BadCloneSolver(bic.graph.multicut.MulticutSolver):
        def clone(self):
            return self

        def optimize(self, objective):
            return objective.labels

    objective = _two_component_objective()
    original_labels = objective.labels.copy()
    with pytest.raises(TypeError, match="distinct solver"):
        bic.graph.multicut.MulticutDecomposer(
            BadCloneSolver(),
            fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
            number_of_threads=2,
        ).optimize(objective)
    np.testing.assert_array_equal(objective.labels, original_labels)


def test_decomposer_propagates_worker_error_without_partial_labels():
    class ErrorSolver(bic.graph.multicut.MulticutSolver):
        def clone(self):
            return ErrorSolver()

        def optimize(self, objective):
            raise RuntimeError("worker failed")

    objective = _two_component_objective()
    original_labels = objective.labels.copy()
    with pytest.raises(RuntimeError, match="worker failed"):
        bic.graph.multicut.MulticutDecomposer(
            ErrorSolver(),
            fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
            number_of_threads=2,
        ).optimize(objective)
    np.testing.assert_array_equal(objective.labels, original_labels)


def test_builtin_and_chained_solver_clone_configuration():
    solver = bic.graph.multicut.ChainedMulticutSolvers(
        [
            bic.graph.multicut.GreedyAdditiveMulticut(seed=7),
            bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=3),
        ]
    )
    cloned = solver.clone()
    assert cloned is not solver
    assert all(
        cloned_solver is not original_solver
        for cloned_solver, original_solver in zip(cloned.solvers, solver.solvers)
    )
    assert cloned.solvers[0].seed == 7
    assert cloned.solvers[1].number_of_outer_iterations == 3
