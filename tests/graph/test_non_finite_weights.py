from __future__ import annotations

import numpy as np
import pytest

import bioimage_cpp as bic


NON_FINITE = [np.nan, np.inf, -np.inf]


def _chain_graph():
    return bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])


@pytest.mark.parametrize("value", NON_FINITE)
def test_multicut_rejects_non_finite_costs(value):
    graph = _chain_graph()
    with pytest.raises(ValueError, match="edge_costs.*finite"):
        bic.graph.multicut.MulticutObjective(graph, [1.0, value])


@pytest.mark.parametrize(
    "solver",
    [
        bic.graph.multicut.GreedyAdditiveMulticut(),
        bic.graph.multicut.GreedyFixationMulticut(),
        bic.graph.multicut.KernighanLinMulticut(),
        bic.graph.multicut.FusionMoveMulticut(
            proposal_generator=bic.graph.multicut.WatershedProposalGenerator(),
            number_of_iterations=1,
        ),
    ],
)
def test_multicut_solvers_revalidate_mutated_costs(solver):
    objective = bic.graph.multicut.MulticutObjective(_chain_graph(), [1.0, -1.0])
    objective.edge_costs[1] = np.nan
    with pytest.raises(ValueError, match="edge_costs.*finite"):
        solver.optimize(objective)
    with pytest.raises(ValueError, match="edge_costs.*finite"):
        objective.energy()


@pytest.mark.parametrize("value", NON_FINITE)
def test_lifted_multicut_rejects_non_finite_costs(value):
    graph = _chain_graph()
    with pytest.raises(ValueError, match="edge_costs.*finite"):
        bic.graph.lifted_multicut.LiftedMulticutObjective(graph, [1.0, value])
    with pytest.raises(ValueError, match="lifted_costs.*finite"):
        bic.graph.lifted_multicut.LiftedMulticutObjective(
            graph,
            [1.0, 1.0],
            lifted_uvs=[[0, 2]],
            lifted_costs=[value],
        )


@pytest.mark.parametrize("value", NON_FINITE)
def test_lifted_set_cost_rejects_non_finite_without_mutation(value):
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        _chain_graph(), [1.0, 1.0]
    )
    with pytest.raises(ValueError, match="weight.*finite"):
        objective.set_cost(0, 2, value)
    assert objective.number_of_lifted_edges == 0


def test_lifted_set_cost_rejects_non_finite_accumulation_without_mutation():
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        _chain_graph(),
        [1.0, 1.0],
        lifted_uvs=[[0, 2]],
        lifted_costs=[1.0e308],
    )
    with pytest.raises(ValueError, match="accumulated weight.*finite"):
        objective.set_cost(0, 2, 1.0e308)
    assert objective.weights[-1] == 1.0e308


def test_lifted_solver_revalidates_mutated_weights():
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        _chain_graph(),
        [1.0, 1.0],
        lifted_uvs=[[0, 2]],
        lifted_costs=[-1.0],
    )
    objective.weights[-1] = np.nan
    solver = bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
        proposal_generator=bic.graph.multicut.WatershedProposalGenerator(),
        number_of_iterations=1,
    )
    with pytest.raises(ValueError, match="weights.*finite"):
        solver.optimize(objective)
    with pytest.raises(ValueError, match="weights.*finite"):
        objective.energy()


@pytest.mark.parametrize("value", NON_FINITE)
def test_edge_weighted_watershed_rejects_non_finite_weights(value):
    with pytest.raises(ValueError, match="edge_weights.*finite"):
        bic.graph.edge_weighted_watershed(
            _chain_graph(),
            np.array([0.1, value], dtype=np.float64),
            np.array([1, 0, 2], dtype=np.uint64),
        )


@pytest.mark.parametrize("argument", ["edge_costs", "mutex_costs"])
def test_mutex_watershed_rejects_non_finite_weights(argument):
    kwargs = {
        "graph": _chain_graph(),
        "edge_costs": np.array([1.0, 1.0], dtype=np.float64),
        "mutex_uvs": np.array([[0, 2]], dtype=np.uint64),
        "mutex_costs": np.array([1.0], dtype=np.float64),
    }
    kwargs[argument][-1] = np.nan
    with pytest.raises(ValueError, match=rf"{argument}.*finite"):
        bic.graph.mutex_watershed.mutex_watershed_clustering(**kwargs)


@pytest.mark.parametrize(
    "argument",
    ["edge_costs", "mutex_costs", "semantic_costs"],
)
def test_semantic_mutex_watershed_rejects_non_finite_weights(argument):
    kwargs = {
        "graph": _chain_graph(),
        "edge_costs": np.array([1.0, 1.0], dtype=np.float64),
        "mutex_uvs": np.array([[0, 2]], dtype=np.uint64),
        "mutex_costs": np.array([1.0], dtype=np.float64),
        "semantic_node_classes": np.array([[0, 1]], dtype=np.uint64),
        "semantic_costs": np.array([1.0], dtype=np.float64),
    }
    kwargs[argument][-1] = np.nan
    with pytest.raises(ValueError, match=rf"{argument}.*finite"):
        bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(**kwargs)


@pytest.mark.parametrize(
    ("make_policy", "arguments", "name"),
    [
        (
            bic.graph.agglomeration.EdgeWeightedClusterPolicy,
            {"edge_indicators": [0.1, np.nan]},
            "edge_indicators",
        ),
        (
            bic.graph.agglomeration.GaspClusterPolicy,
            {"edge_weights": [1.0, np.nan]},
            "edge_weights",
        ),
        (
            bic.graph.agglomeration.MalaClusterPolicy,
            {"edge_indicators": [0.1, np.nan]},
            "edge_indicators",
        ),
    ],
)
def test_agglomeration_rejects_non_finite_primary_weights(
    make_policy, arguments, name
):
    with pytest.raises(ValueError, match=rf"{name}.*finite"):
        make_policy().optimize(_chain_graph(), **arguments)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("edge_sizes", [1.0, np.nan]),
        ("node_sizes", [1.0, 1.0, np.nan]),
    ],
)
def test_agglomeration_rejects_non_finite_auxiliary_weights(argument, value):
    kwargs = {
        "edge_indicators": [0.1, 0.2],
        argument: value,
    }
    with pytest.raises(ValueError, match=rf"{argument}.*finite"):
        bic.graph.agglomeration.EdgeWeightedClusterPolicy().optimize(
            _chain_graph(), **kwargs
        )


def test_node_and_edge_agglomeration_rejects_non_finite_features():
    features = np.array([[0.0], [np.nan], [1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="node_features.*finite"):
        bic.graph.agglomeration.NodeAndEdgeWeightedClusterPolicy().optimize(
            _chain_graph(), [0.1, 0.2], features
        )


@pytest.mark.parametrize(
    ("policy", "name"),
    [
        (
            lambda: bic.graph.agglomeration.EdgeWeightedClusterPolicy(
                size_regularizer=np.nan
            ),
            "size_regularizer",
        ),
        (
            lambda: bic.graph.agglomeration.NodeAndEdgeWeightedClusterPolicy(
                beta=np.inf
            ),
            "beta",
        ),
        (
            lambda: bic.graph.agglomeration.MalaClusterPolicy(threshold=-np.inf),
            "threshold",
        ),
    ],
)
def test_agglomeration_rejects_non_finite_policy_parameters(policy, name):
    with pytest.raises(ValueError, match=rf"{name}.*finite"):
        policy()
