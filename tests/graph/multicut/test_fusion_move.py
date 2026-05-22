import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import edge_cut_labels


def test_fuses_a_two_cluster_problem(external_toy_problem):
    graph, costs, _ = external_toy_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    baseline = bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective)
    baseline_energy = objective.energy(baseline)

    objective.reset_labels()
    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=1),
        number_of_iterations=10,
    )
    labels = solver.optimize(objective)

    assert labels.dtype == np.uint64
    assert labels.shape == (graph.number_of_nodes,)
    assert objective.energy(labels) <= baseline_energy + 1e-9


def test_safety_net_never_regresses(grid_problem):
    graph, costs = grid_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    baseline_energy = objective.energy(
        bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective)
    )

    objective.reset_labels()
    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=7),
        number_of_iterations=8,
    )
    labels = solver.optimize(objective)

    assert objective.energy(labels) <= baseline_energy + 1e-9


def test_warm_starts_from_singleton(chain_problem):
    graph, costs = chain_problem
    # The default objective labels are the singleton (arange) labeling; the
    # driver should warm-start with greedy-additive and reach the optimum.
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=0),
        number_of_iterations=3,
    )
    labels = solver.optimize(objective)

    # Optimal: cut only the negative edge between nodes 2 and 3.
    expected_cut = np.array([False, False, True])
    np.testing.assert_array_equal(edge_cut_labels(graph, labels), expected_cut)


def test_greedy_additive_proposal_generator_runs(grid_problem):
    graph, costs = grid_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    baseline = objective.energy(bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective))

    objective.reset_labels()
    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.GreedyAdditiveProposalGenerator(seed=3, sigma=1.0),
        number_of_iterations=5,
    )
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline + 1e-9


@pytest.mark.parametrize(
    "sub_solver",
    [
        bic.graph.multicut.GreedyAdditiveMulticut(),
        bic.graph.multicut.GreedyFixationMulticut(),
        bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=3),
    ],
)
def test_sub_solver_pluggability(grid_problem, sub_solver):
    graph, costs = grid_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)

    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=2),
        sub_solver=sub_solver,
        number_of_iterations=4,
    )
    labels = solver.optimize(objective)

    assert labels.shape == (graph.number_of_nodes,)
    assert labels.dtype == np.uint64


def test_stops_after_no_improvement(frustrated_triangle):
    # Tiny problem with many iterations and an aggressive non-improvement
    # threshold: the loop must terminate quickly without scanning all 1000
    # iterations (the test would time out otherwise).
    graph, costs = frustrated_triangle
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    baseline = objective.energy(
        bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective)
    )

    objective.reset_labels()
    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=0),
        number_of_iterations=1000,
        stop_if_no_improvement=1,
    )
    labels = solver.optimize(objective)
    # Best-of safety net guarantees energy never regresses past baseline.
    assert objective.energy(labels) <= baseline + 1e-9


def test_chains_with_kernighan_lin(external_toy_problem):
    graph, costs, _ = external_toy_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    baseline = objective.energy(
        bic.graph.multicut.GreedyAdditiveMulticut().optimize(objective)
    )

    objective.reset_labels()
    solver = bic.graph.multicut.ChainedMulticutSolvers([
        bic.graph.multicut.FusionMoveMulticut(
            proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=11),
            number_of_iterations=5,
        ),
        bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=3),
    ])
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline + 1e-9


def test_rejects_non_proposal_generator():
    with pytest.raises(TypeError, match="proposal_generator"):
        bic.graph.multicut.FusionMoveMulticut(proposal_generator=object())


def test_rejects_zero_thread_count():
    with pytest.raises(ValueError, match="number_of_threads"):
        bic.graph.multicut.FusionMoveMulticut(
            proposal_generator=bic.graph.multicut.WatershedProposalGenerator(),
            number_of_threads=0,
        )


def test_rejects_zero_parallel_proposals():
    with pytest.raises(ValueError, match="number_of_parallel_proposals"):
        bic.graph.multicut.FusionMoveMulticut(
            proposal_generator=bic.graph.multicut.WatershedProposalGenerator(),
            number_of_parallel_proposals=0,
        )


def test_rejects_invalid_iteration_settings():
    with pytest.raises(ValueError, match="number_of_iterations"):
        bic.graph.multicut.FusionMoveMulticut(
            proposal_generator=bic.graph.multicut.WatershedProposalGenerator(),
            number_of_iterations=-1,
        )
    with pytest.raises(ValueError, match="stop_if_no_improvement"):
        bic.graph.multicut.FusionMoveMulticut(
            proposal_generator=bic.graph.multicut.WatershedProposalGenerator(),
            stop_if_no_improvement=0,
        )


def test_runs_on_empty_graph():
    graph = bic.graph.UndirectedGraph(0)
    objective = bic.graph.multicut.MulticutObjective(graph, np.zeros(0))
    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(),
    )
    labels = solver.optimize(objective)
    assert labels.shape == (0,)


def test_parallel_threads_match_single_threaded_safety_net(grid_problem):
    # With T>1 the loop should never regress past the greedy-additive baseline
    # (best-of safety net is enforced in C++).
    graph, costs = grid_problem
    baseline = bic.graph.multicut.MulticutObjective(graph, costs).energy(
        bic.graph.multicut.GreedyAdditiveMulticut().optimize(bic.graph.multicut.MulticutObjective(graph, costs))
    )

    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=11),
        number_of_threads=4,
        number_of_iterations=5,
    )
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline + 1e-9


def test_multi_proposal_runs(grid_problem):
    # number_of_parallel_proposals > 2 exercises the stage-2 joint fuse path
    # (no determinism guarantee against T=1 here — the algorithm shape changes).
    graph, costs = grid_problem
    baseline = bic.graph.multicut.MulticutObjective(graph, costs).energy(
        bic.graph.multicut.GreedyAdditiveMulticut().optimize(bic.graph.multicut.MulticutObjective(graph, costs))
    )

    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=3),
        number_of_threads=2,
        number_of_parallel_proposals=4,
        number_of_iterations=5,
    )
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline + 1e-9


def test_parallel_is_deterministic_given_settings(grid_problem):
    # Same seed + same T + same P + same iterations → same result.
    graph, costs = grid_problem

    def run():
        objective = bic.graph.multicut.MulticutObjective(graph, costs)
        solver = bic.graph.multicut.FusionMoveMulticut(
            proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=7),
            number_of_threads=4,
            number_of_iterations=5,
        )
        return solver.optimize(objective)

    np.testing.assert_array_equal(run(), run())


def test_default_parallel_proposals_tracks_threads():
    pgen = bic.graph.multicut.WatershedProposalGenerator()
    one_thread = bic.graph.multicut.FusionMoveMulticut(proposal_generator=pgen)
    four_threads = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=pgen, number_of_threads=4
    )
    assert one_thread.number_of_parallel_proposals == 2
    assert four_threads.number_of_parallel_proposals == 4


def test_explicit_parallel_proposals_overrides_default():
    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(),
        number_of_threads=4,
        number_of_parallel_proposals=8,
    )
    assert solver.number_of_parallel_proposals == 8


def test_runs_on_graph_without_negative_edges(chain_problem):
    # WatershedProposalGenerator yields all-zero proposals if no negative
    # edges exist; the driver must still terminate cleanly and produce the
    # warm-started greedy-additive result.
    graph, _ = chain_problem
    costs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    solver = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(seed=0),
        number_of_iterations=3,
    )
    labels = solver.optimize(objective)
    # All-positive costs → no cut.
    assert np.all(edge_cut_labels(graph, labels) == False)  # noqa: E712
