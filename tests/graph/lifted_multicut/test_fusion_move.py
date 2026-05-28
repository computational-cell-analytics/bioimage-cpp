import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import same_partition


def _grid_lifted_problem(shape=(4, 4), bfs_distance=2, seed=0):
    """A small grid base graph plus zero-weight BFS-induced lifted edges,
    with some lifted edges then re-weighted negatively to make the problem
    non-trivial."""
    rng = np.random.default_rng(seed)
    edges = []
    base_costs = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            node = y * shape[1] + x
            if x + 1 < shape[1]:
                edges.append([node, node + 1])
                base_costs.append(1.0 if x != shape[1] // 2 else -3.0)
            if y + 1 < shape[0]:
                edges.append([node, node + shape[1]])
                base_costs.append(1.0 if y != shape[0] // 2 else -3.0)
    base = bic.graph.UndirectedGraph.from_edges(shape[0] * shape[1], edges)
    base_costs = np.array(base_costs, dtype=np.float64)

    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        base, base_costs, bfs_distance=bfs_distance
    )
    # Reweight a handful of lifted edges to non-zero, mixed sign, so they
    # actually influence the energy.
    n_lifted = objective.number_of_lifted_edges
    n_base = objective.number_of_base_edges
    weights = objective.weights.copy()
    for i in range(min(n_lifted, 6)):
        weights[n_base + i] = float(rng.normal(0.0, 2.0))
    objective._weights = weights
    return objective


def test_fusion_move_splits_chain_along_repulsive_lifted_edge(chain_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = chain_with_lifted
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )

    # Singleton labels trigger the lifted greedy-additive warm start, which
    # sees lifted weights and discovers the cut along the chain. The
    # subsequent proposal/fuse loop must not regress past that.
    merged_energy = objective.energy(np.zeros(4, dtype=np.uint64))

    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=1),
        number_of_iterations=10,
    )
    labels = solver.optimize(objective)

    assert labels.dtype == np.uint64
    assert labels.shape == (base.number_of_nodes,)
    assert objective.energy(labels) <= merged_energy + 1e-9
    # Repulsive lifted edge must end up cut.
    assert labels[0] != labels[3]


def test_fusion_move_safety_net_never_regresses(chain_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = chain_with_lifted
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )
    baseline = bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut().optimize(objective)
    baseline_energy = objective.energy(baseline)

    objective.reset_labels()
    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=7),
        number_of_iterations=8,
    )
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline_energy + 1e-9


def test_fusion_move_matches_multicut_without_lifted_edges():
    base = bic.graph.UndirectedGraph.from_edges(
        6,
        [
            [0, 1], [0, 3], [1, 2], [1, 4], [2, 5], [3, 4], [4, 5],
        ],
    )
    base_costs = np.array([5, -20, 5, 5, -20, 5, 5], dtype=np.float64)

    mc_objective = bic.graph.multicut.MulticutObjective(base, base_costs)
    mc_labels = bic.graph.multicut.FusionMoveMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=3),
        number_of_iterations=5,
    ).optimize(mc_objective)

    lmc_objective = bic.graph.lifted_multicut.LiftedMulticutObjective(base, base_costs)
    lmc_labels = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=3),
        number_of_iterations=5,
    ).optimize(lmc_objective)

    # Energies should match (the lifted problem has no lifted edges, so it's
    # equivalent to the base multicut problem).
    assert mc_objective.energy(mc_labels) == pytest.approx(
        lmc_objective.energy(lmc_labels), abs=1e-9
    )


def test_fusion_move_warm_starts_from_singleton():
    base = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])
    base_costs = np.array([2.0, 2.0], dtype=np.float64)
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(base, base_costs)
    # Singleton labels trigger the internal lifted greedy-additive warm start.
    labels = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=0),
        number_of_iterations=3,
    ).optimize(objective)
    same_partition(labels, [0, 0, 0])


def test_fusion_move_keeps_base_disconnected_clusters_separate(
    disjoint_clusters_with_attractive_lifted,
):
    base, base_costs, lifted_uvs, lifted_costs = (
        disjoint_clusters_with_attractive_lifted
    )
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )
    labels = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=0),
        number_of_iterations=5,
    ).optimize(objective)
    # The (0, 2) lifted edge is attractive, but the base graph has no path
    # between the two components — base-graph contraction never merges them.
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]


def test_fusion_move_greedy_additive_proposal_generator_runs():
    objective = _grid_lifted_problem()
    baseline = objective.energy(
        bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut().optimize(objective)
    )

    objective.reset_labels()
    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.GreedyAdditiveProposalGenerator(
            seed=3, sigma=1.0
        ),
        number_of_iterations=5,
    )
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline + 1e-9


@pytest.mark.parametrize(
    "sub_solver",
    [
        bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut(),
        bic.graph.lifted_multicut.LiftedKernighanLinMulticut(number_of_outer_iterations=3),
    ],
)
def test_fusion_move_sub_solver_pluggability(sub_solver):
    objective = _grid_lifted_problem()
    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=2),
        sub_solver=sub_solver,
        number_of_iterations=4,
    )
    labels = solver.optimize(objective)
    assert labels.shape == (objective.graph.number_of_nodes,)
    assert labels.dtype == np.uint64


def test_fusion_move_stops_after_no_improvement():
    # Tiny problem with many iterations and an aggressive non-improvement
    # threshold: the loop must terminate quickly.
    base = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2], [0, 2]])
    base_costs = np.array([2.0, 2.0, -5.0], dtype=np.float64)
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(base, base_costs)
    baseline = objective.energy(
        bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut().optimize(objective)
    )

    objective.reset_labels()
    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=0),
        number_of_iterations=1000,
        stop_if_no_improvement=1,
    )
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline + 1e-9


def test_fusion_move_chains_with_kernighan_lin(chain_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = chain_with_lifted
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )
    baseline = objective.energy(
        bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut().optimize(objective)
    )

    objective.reset_labels()
    solver = bic.graph.lifted_multicut.LiftedChainedSolvers([
        bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
            proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=11),
            number_of_iterations=5,
        ),
        bic.graph.lifted_multicut.LiftedKernighanLinMulticut(number_of_outer_iterations=3),
    ])
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline + 1e-9


def test_fusion_move_rejects_non_proposal_generator():
    with pytest.raises(TypeError, match="proposal_generator"):
        bic.graph.lifted_multicut.FusionMoveLiftedMulticut(proposal_generator=object())


def test_fusion_move_rejects_non_lifted_sub_solver():
    with pytest.raises(TypeError, match="sub_solver"):
        bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
            proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(),
            sub_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
        )


def test_fusion_move_rejects_zero_thread_count():
    with pytest.raises(ValueError, match="number_of_threads"):
        bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
            proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(),
            number_of_threads=0,
        )


def test_fusion_move_rejects_zero_parallel_proposals():
    with pytest.raises(ValueError, match="number_of_parallel_proposals"):
        bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
            proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(),
            number_of_parallel_proposals=0,
        )


def test_fusion_move_runs_on_empty_graph():
    base = bic.graph.UndirectedGraph(0)
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(base, np.zeros(0))
    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(),
    )
    labels = solver.optimize(objective)
    assert labels.shape == (0,)


def test_fusion_move_parallel_threads_match_single_threaded_safety_net():
    objective = _grid_lifted_problem(seed=11)
    baseline = objective.energy(
        bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut().optimize(objective)
    )

    objective.reset_labels()
    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=11),
        number_of_threads=4,
        number_of_iterations=5,
    )
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline + 1e-9


def test_fusion_move_multi_proposal_runs():
    objective = _grid_lifted_problem(seed=3)
    baseline = objective.energy(
        bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut().optimize(objective)
    )

    objective.reset_labels()
    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=3),
        number_of_threads=2,
        number_of_parallel_proposals=4,
        number_of_iterations=5,
    )
    labels = solver.optimize(objective)
    assert objective.energy(labels) <= baseline + 1e-9


def test_fusion_move_parallel_is_deterministic_given_settings():
    def run():
        objective = _grid_lifted_problem(seed=7)
        solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
            proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(seed=7),
            number_of_threads=4,
            number_of_iterations=5,
        )
        return solver.optimize(objective)

    np.testing.assert_array_equal(run(), run())


def test_fusion_move_default_parallel_proposals_tracks_threads():
    pgen = bic.graph.lifted_multicut.WatershedProposalGenerator()
    one_thread = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(proposal_generator=pgen)
    four_threads = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=pgen, number_of_threads=4
    )
    assert one_thread.number_of_parallel_proposals == 2
    assert four_threads.number_of_parallel_proposals == 4


def test_greedy_proposals_parallel_is_deterministic_on_dirty_base_graph():
    # Regression guard for the lazy-CSR-adjacency data race on the *base* graph.
    # The greedy-additive proposal generator reads base_graph.node_adjacency()
    # (via DynamicGraph::reset); with T>1 the parallel proposal slots used to
    # race on the first rebuild of a not-yet-frozen base graph. Unlike the
    # multicut driver, here the singleton warm-start only freezes the *lifted*
    # graph, so the race is reachable from the default start. The solver now
    # freezes the base graph before fan-out; the multi-threaded result must equal
    # the single-threaded reference on every run.
    #
    # Note: a regression here can surface as a process crash (it is a data race),
    # not just a value mismatch.
    n = 2000
    base_edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.uint64)
    base_costs = np.array(
        [1.0 if i % 3 else -2.0 for i in range(n - 1)], dtype=np.float64
    )
    # A handful of lifted edges keeps the lifted graph small (fast warm-start)
    # while the large base graph drives the parallel proposal generation.
    lifted_uvs = np.array(
        [[i, i + 5] for i in range(0, n - 5, 250)], dtype=np.uint64
    )
    lifted_costs = np.array([-3.0] * len(lifted_uvs), dtype=np.float64)
    parallel_proposals = 4

    def run(threads):
        # Fresh base graph per run so each multi-threaded run starts dirty.
        base = bic.graph.UndirectedGraph.from_edges(n, base_edges)
        objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
            base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
        )
        solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
            proposal_generator=bic.graph.lifted_multicut.GreedyAdditiveProposalGenerator(seed=0),
            number_of_threads=threads,
            number_of_parallel_proposals=parallel_proposals,
            number_of_iterations=3,
        )
        return solver.optimize(objective)

    reference = run(1)
    for _ in range(15):
        np.testing.assert_array_equal(run(4), reference)
