import numpy as np
import pytest

import bioimage_cpp as bic

from ._helpers import edge_cut_labels, same_partition


def _copy_supported_solver(solver):
    multicut = bic.graph.multicut
    if isinstance(solver, multicut.GreedyAdditiveMulticut):
        return multicut.GreedyAdditiveMulticut(
            weight_stop=solver.weight_stop,
            node_num_stop=solver.node_num_stop,
            add_noise=solver.add_noise,
            seed=solver.seed,
            sigma=solver.sigma,
        )
    if isinstance(solver, multicut.GreedyFixationMulticut):
        return multicut.GreedyFixationMulticut(
            weight_stop=solver.weight_stop,
            node_num_stop=solver.node_num_stop,
        )
    if isinstance(solver, multicut.KernighanLinMulticut):
        return multicut.KernighanLinMulticut(
            number_of_outer_iterations=solver.number_of_outer_iterations,
            number_of_inner_iterations=solver.number_of_inner_iterations,
            epsilon=solver.epsilon,
        )
    if isinstance(solver, multicut.ChainedMulticutSolvers):
        return multicut.ChainedMulticutSolvers(
            [_copy_supported_solver(child) for child in solver.solvers]
        )
    raise TypeError(f"unsupported test solver: {type(solver).__name__}")


def _python_reference_decomposition(
    graph,
    edge_costs,
    sub_solver,
    fallthrough_solver,
):
    objective = bic.graph.multicut.MulticutObjective(graph, edge_costs)
    components = bic.graph.connected_components(
        objective.graph,
        edge_mask=objective.edge_costs > 0.0,
    )
    number_of_components = int(components.max()) + 1 if components.size else 0
    if number_of_components <= 1:
        solver = fallthrough_solver or sub_solver
        return _copy_supported_solver(solver).optimize(objective)

    component_index = components.astype(np.intp, copy=False)
    node_counts = np.bincount(
        component_index, minlength=number_of_components
    )
    node_offsets = np.concatenate(
        [np.array([0], dtype=np.intp), np.cumsum(node_counts, dtype=np.intp)]
    )
    grouped_nodes = np.argsort(component_index, kind="stable").astype(
        np.uint64, copy=False
    )
    global_to_local = np.empty(graph.number_of_nodes, dtype=np.uint64)
    for component in range(number_of_components):
        nodes = grouped_nodes[
            node_offsets[component] : node_offsets[component + 1]
        ]
        global_to_local[nodes] = np.arange(nodes.size, dtype=np.uint64)

    uvs = graph.uv_ids()
    u_components = component_index[uvs[:, 0]]
    v_components = component_index[uvs[:, 1]]
    internal_edges = np.flatnonzero(u_components == v_components)
    edge_order = np.argsort(u_components[internal_edges], kind="stable")
    grouped_edges = internal_edges[edge_order]
    edge_counts = np.bincount(
        u_components[internal_edges], minlength=number_of_components
    )
    edge_offsets = np.concatenate(
        [np.array([0], dtype=np.intp), np.cumsum(edge_counts, dtype=np.intp)]
    )

    labels = np.empty(graph.number_of_nodes, dtype=np.uint64)
    label_offset = 0
    for component in range(number_of_components):
        nodes = grouped_nodes[
            node_offsets[component] : node_offsets[component + 1]
        ]
        if nodes.size == 1:
            labels[int(nodes[0])] = label_offset
            label_offset += 1
            continue

        edge_ids = grouped_edges[
            edge_offsets[component] : edge_offsets[component + 1]
        ]
        local_uvs = global_to_local[uvs[edge_ids]]
        subgraph = bic.graph.UndirectedGraph(
            int(nodes.size), int(edge_ids.size)
        )
        subgraph.insert_edges(local_uvs)
        sub_objective = bic.graph.multicut.MulticutObjective(
            subgraph, objective.edge_costs[edge_ids]
        )
        sub_labels = _copy_supported_solver(sub_solver).optimize(
            sub_objective
        )
        _, sub_labels = np.unique(sub_labels, return_inverse=True)
        sub_labels = sub_labels.astype(np.uint64, copy=False)
        labels[nodes] = sub_labels + label_offset
        label_offset += int(sub_labels.max()) + 1

    return labels


def test_chained_multicut_solvers(chain_problem):
    graph, costs = chain_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    solver = bic.graph.multicut.ChainedMulticutSolvers(
        [
            bic.graph.multicut.GreedyAdditiveMulticut(),
            bic.graph.multicut.KernighanLinMulticut(
                number_of_outer_iterations=5
            ),
        ]
    )

    labels = solver.optimize(objective)

    same_partition(labels, [0, 0, 0, 1])
    assert objective.energy(labels) == pytest.approx(-5.0)


def test_chained_solver_rejects_empty_chain():
    with pytest.raises(ValueError, match="at least one"):
        bic.graph.multicut.ChainedMulticutSolvers([])


def test_multicut_decomposer_solves_positive_components():
    graph = bic.graph.UndirectedGraph.from_edges(
        4, [[0, 1], [1, 2], [2, 3]]
    )
    objective = bic.graph.multicut.MulticutObjective(
        graph, [1.0, -5.0, 1.0]
    )
    solver = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.GreedyAdditiveMulticut()
    )

    labels = solver.optimize(objective)

    same_partition(labels, [0, 0, 1, 1])
    assert objective.energy() == pytest.approx(-5.0)


def test_multicut_decomposer_uses_fallthrough_for_single_component():
    graph = bic.graph.UndirectedGraph.from_edges(2, [[0, 1]])
    objective = bic.graph.multicut.MulticutObjective(graph, [1.0])
    solver = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.GreedyAdditiveMulticut(),
        fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(
            weight_stop=2.0
        ),
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
                bic.graph.multicut.KernighanLinMulticut(
                    number_of_outer_iterations=10
                ),
            ]
        )
    )

    labels = solver.optimize(objective)

    assert objective.energy(labels) <= -35.0


def test_decomposer_energy_bound_on_grid_problem(grid_problem):
    graph, costs = grid_problem
    objective = bic.graph.multicut.MulticutObjective(graph, costs)
    solver = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.GreedyAdditiveMulticut()
    )

    labels = solver.optimize(objective)

    assert objective.energy(labels) <= -20.0


def _two_component_objective():
    graph = bic.graph.UndirectedGraph.from_edges(
        4,
        [[0, 1], [1, 2], [2, 3]],
    )
    return bic.graph.multicut.MulticutObjective(
        graph, [1.0, -5.0, 1.0]
    )


@pytest.mark.parametrize("number_of_threads", [0, 1, 2, 8])
def test_decomposer_is_deterministic_across_thread_counts(number_of_threads):
    objective = _two_component_objective()
    labels = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.GreedyFixationMulticut(),
        fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
        number_of_threads=number_of_threads,
    ).optimize(objective)

    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    assert objective.energy() == pytest.approx(-5.0)


def test_decomposer_handles_singleton_components():
    graph = bic.graph.UndirectedGraph(3)
    objective = bic.graph.multicut.MulticutObjective(graph, [])
    labels = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.KernighanLinMulticut(
            number_of_outer_iterations=0
        ),
        fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
        number_of_threads=2,
    ).optimize(objective)

    np.testing.assert_array_equal(labels, [0, 1, 2])


def test_decomposer_handles_empty_graph():
    graph = bic.graph.UndirectedGraph()
    objective = bic.graph.multicut.MulticutObjective(graph, [])
    labels = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.KernighanLinMulticut(),
        fallthrough_solver=bic.graph.multicut.GreedyFixationMulticut(),
        number_of_threads=0,
    ).optimize(objective)

    np.testing.assert_array_equal(labels, np.array([], dtype=np.uint64))


def test_decomposer_kernighan_lin_keeps_greedy_warm_start():
    objective = _two_component_objective()
    labels = bic.graph.multicut.MulticutDecomposer(
        bic.graph.multicut.KernighanLinMulticut(
            number_of_outer_iterations=0
        ),
        fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
        number_of_threads=2,
    ).optimize(objective)

    np.testing.assert_array_equal(labels, [0, 0, 1, 1])


class _CustomSolver(bic.graph.multicut.MulticutSolver):
    def optimize(self, objective):
        return objective.labels


@pytest.mark.parametrize(
    "unsupported_solver",
    [
        pytest.param(_CustomSolver(), id="custom-python"),
        pytest.param(
            bic.graph.multicut.FusionMoveMulticut(
                proposal_generator=(
                    bic.graph.multicut.WatershedProposalGenerator()
                )
            ),
            id="fusion-move",
        ),
        pytest.param(
            bic.graph.multicut.MulticutDecomposer(
                bic.graph.multicut.GreedyAdditiveMulticut()
            ),
            id="nested-decomposer",
        ),
        pytest.param(
            bic.graph.multicut.ChainedMulticutSolvers(
                [
                    bic.graph.multicut.GreedyAdditiveMulticut(),
                    _CustomSolver(),
                ]
            ),
            id="chain-with-custom-python",
        ),
    ],
)
def test_decomposer_rejects_unsupported_sub_solver(unsupported_solver):
    with pytest.raises(TypeError, match="sub_solver"):
        bic.graph.multicut.MulticutDecomposer(unsupported_solver)


def test_decomposer_rejects_unsupported_fallthrough_solver():
    with pytest.raises(TypeError, match="fallthrough_solver"):
        bic.graph.multicut.MulticutDecomposer(
            bic.graph.multicut.GreedyAdditiveMulticut(),
            fallthrough_solver=_CustomSolver(),
        )


def test_decomposer_rejects_negative_thread_count():
    with pytest.raises(ValueError, match="number_of_threads"):
        bic.graph.multicut.MulticutDecomposer(
            bic.graph.multicut.GreedyAdditiveMulticut(),
            number_of_threads=-1,
        )


def test_multicut_solvers_do_not_expose_clone_api():
    assert not hasattr(bic.graph.multicut.MulticutSolver, "clone")


@pytest.mark.parametrize(
    "sub_solver",
    [
        pytest.param(
            bic.graph.multicut.GreedyAdditiveMulticut(
                weight_stop=0.05,
                add_noise=True,
                seed=7,
                sigma=0.2,
            ),
            id="greedy-additive",
        ),
        pytest.param(
            bic.graph.multicut.GreedyFixationMulticut(weight_stop=0.05),
            id="greedy-fixation",
        ),
        pytest.param(
            bic.graph.multicut.KernighanLinMulticut(
                number_of_outer_iterations=5,
                epsilon=1.0e-5,
            ),
            id="kernighan-lin",
        ),
        pytest.param(
            bic.graph.multicut.ChainedMulticutSolvers(
                [
                    bic.graph.multicut.GreedyAdditiveMulticut(
                        weight_stop=0.02,
                        add_noise=True,
                        seed=11,
                        sigma=0.1,
                    ),
                    bic.graph.multicut.KernighanLinMulticut(
                        number_of_outer_iterations=3,
                        epsilon=1.0e-5,
                    ),
                ]
            ),
            id="chain",
        ),
    ],
)
def test_native_decomposer_matches_python_reference(sub_solver):
    rng = np.random.default_rng(27)
    all_edges = np.array(
        [
            (u, v)
            for u in range(12)
            for v in range(u + 1, 12)
        ],
        dtype=np.uint64,
    )
    fallthrough_solver = bic.graph.multicut.GreedyFixationMulticut(
        weight_stop=2.0
    )

    for _ in range(8):
        edge_indices = rng.choice(
            all_edges.shape[0], size=24, replace=False
        )
        rng.shuffle(edge_indices)
        edges = all_edges[edge_indices]
        costs = rng.normal(size=edges.shape[0])
        costs[rng.random(edges.shape[0]) < 0.4] -= 2.0
        graph = bic.graph.UndirectedGraph.from_edges(12, edges)

        expected = _python_reference_decomposition(
            graph,
            costs,
            sub_solver,
            fallthrough_solver,
        )
        objective = bic.graph.multicut.MulticutObjective(graph, costs)
        actual = bic.graph.multicut.MulticutDecomposer(
            sub_solver,
            fallthrough_solver=fallthrough_solver,
            number_of_threads=4,
        ).optimize(objective)

        np.testing.assert_array_equal(
            edge_cut_labels(graph, actual),
            edge_cut_labels(graph, expected),
        )
        assert objective.energy(actual) == pytest.approx(
            objective.energy(expected)
        )


def test_decomposer_preserves_labels_after_binding_validation_error():
    objective = _two_component_objective()
    original_labels = objective.labels.copy()
    objective._edge_costs[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bic.graph.multicut.MulticutDecomposer(
            bic.graph.multicut.GreedyAdditiveMulticut(),
            fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
            number_of_threads=2,
        ).optimize(objective)

    np.testing.assert_array_equal(objective.labels, original_labels)
