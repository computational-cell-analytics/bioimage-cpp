import numpy as np
import pytest

import bioimage_cpp as bic


def _graph_from_edges(n: int, uvs):
    uvs = np.asarray(uvs, dtype=np.uint64)
    return bic.graph.UndirectedGraph.from_edges(n, uvs)


def _canonicalize(labels):
    # Map labels to first-occurrence dense ids so two segmentations agree iff
    # they produce the same partition (regardless of integer values).
    array = np.asarray(labels, dtype=np.uint64)
    _, inverse = np.unique(array, return_inverse=True)
    # `np.unique`'s `return_inverse` is sorted by value, not by first
    # occurrence — remap one more time to get first-occurrence order.
    seen = {}
    out = np.empty_like(inverse)
    for index, value in enumerate(array):
        key = int(value)
        if key not in seen:
            seen[key] = len(seen)
        out[index] = seen[key]
    return out


def test_all_attractive_merges_to_one_component():
    graph = _graph_from_edges(5, [[0, 1], [1, 2], [2, 3], [3, 4]])
    edge_costs = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)

    labels = bic.graph.mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs
    )

    assert labels.dtype == np.uint64
    assert labels.shape == (5,)
    np.testing.assert_array_equal(labels, np.zeros(5, dtype=np.uint64))


def test_mutex_only_keeps_singletons():
    graph = bic.graph.UndirectedGraph(4)
    edge_costs = np.zeros(0, dtype=np.float64)
    mutex_uvs = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.uint64
    )
    mutex_costs = np.ones(6, dtype=np.float64)

    labels = bic.graph.mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs
    )

    np.testing.assert_array_equal(_canonicalize(labels), np.array([0, 1, 2, 3]))


def test_mutex_beats_attractive_when_higher_weight():
    # Triangle a-b, b-c attractive (weights 1.0, 1.0) + mutex a-c (weight 2.0).
    # Mutex arrives first (highest weight), then a-b merges, then b-c is
    # blocked because root(a)=root(b) carries a mutex with c.
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.array([[0, 2]], dtype=np.uint64)
    mutex_costs = np.array([2.0], dtype=np.float64)

    labels = bic.graph.mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs
    )

    # {0,1} together, {2} alone.
    np.testing.assert_array_equal(_canonicalize(labels), np.array([0, 0, 1]))


def test_mutex_constraint_propagates_through_merge():
    # Nodes 0,1,2 chained attractively (weights 10, 9). Mutex 0-3 with
    # weight 5. Attractive 2-3 with weight 4. After 0,1,2 merge into one
    # root, the mutex propagated via merge_mutexes must block 2-3.
    graph = _graph_from_edges(4, [[0, 1], [1, 2], [2, 3]])
    edge_costs = np.array([10.0, 9.0, 4.0], dtype=np.float64)
    mutex_uvs = np.array([[0, 3]], dtype=np.uint64)
    mutex_costs = np.array([5.0], dtype=np.float64)

    labels = bic.graph.mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs
    )

    np.testing.assert_array_equal(_canonicalize(labels), np.array([0, 0, 0, 1]))


def test_attractive_higher_than_mutex_still_merges():
    # Mirror of test_mutex_beats_attractive but with the order swapped:
    # attractive triangle wins so all three nodes end up together.
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([5.0, 5.0], dtype=np.float64)
    mutex_uvs = np.array([[0, 2]], dtype=np.uint64)
    mutex_costs = np.array([0.1], dtype=np.float64)

    labels = bic.graph.mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs
    )

    np.testing.assert_array_equal(_canonicalize(labels), np.array([0, 0, 0]))


def test_lifted_multicut_inputs_accepted_unchanged():
    # The (graph, edge_costs, lifted_uvs, lifted_costs) shape used to build
    # a LiftedMulticutObjective must also be a valid call to
    # mutex_watershed_clustering — same arrays, no reshape. We do not assert
    # that the algorithms agree on the labels (they don't, in general),
    # only that the input format is compatible.
    graph = _graph_from_edges(4, [[0, 1], [1, 2], [2, 3]])
    edge_costs = np.array([1.0, -0.5, 1.0], dtype=np.float64)
    lifted_uvs = np.array([[0, 2], [0, 3], [1, 3]], dtype=np.uint64)
    lifted_costs = np.array([-1.0, -1.0, 0.5], dtype=np.float64)

    objective = bic.graph.LiftedMulticutObjective(
        graph,
        edge_costs,
        lifted_uvs=lifted_uvs,
        lifted_costs=lifted_costs,
    )
    assert objective is not None

    labels = bic.graph.mutex_watershed_clustering(
        graph, edge_costs, lifted_uvs, lifted_costs
    )
    assert labels.shape == (4,)
    assert labels.dtype == np.uint64


def test_deterministic_across_runs():
    rng = np.random.default_rng(7)
    n = 30
    uvs = []
    for u in range(n):
        for v in range(u + 1, min(u + 4, n)):
            uvs.append([u, v])
    uvs = np.array(uvs, dtype=np.uint64)
    graph = _graph_from_edges(n, uvs)
    edge_costs = rng.uniform(-1.0, 1.0, size=int(graph.number_of_edges)).astype(np.float64)

    mutex_uvs = rng.integers(0, n, size=(40, 2), dtype=np.uint64)
    mutex_uvs = mutex_uvs[mutex_uvs[:, 0] != mutex_uvs[:, 1]]
    mutex_costs = rng.uniform(0.0, 1.0, size=mutex_uvs.shape[0]).astype(np.float64)

    first = bic.graph.mutex_watershed_clustering(graph, edge_costs, mutex_uvs, mutex_costs)
    second = bic.graph.mutex_watershed_clustering(graph, edge_costs, mutex_uvs, mutex_costs)
    np.testing.assert_array_equal(first, second)


def test_dense_label_range():
    graph = _graph_from_edges(6, [[0, 1], [2, 3], [4, 5]])
    edge_costs = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)

    labels = bic.graph.mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs
    )

    # Three components → labels must be {0, 1, 2}, dense.
    assert set(int(value) for value in labels) == {0, 1, 2}
    assert labels.max() == 2


def test_invalid_edge_costs_length_raises():
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    bad_costs = np.array([1.0], dtype=np.float64)  # expected size 2
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)

    with pytest.raises(ValueError):
        bic.graph.mutex_watershed_clustering(graph, bad_costs, mutex_uvs, mutex_costs)


def test_invalid_mutex_uvs_shape_raises():
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    bad_mutex_uvs = np.array([0, 2], dtype=np.uint64)  # 1D, not (n, 2)
    mutex_costs = np.array([1.0], dtype=np.float64)

    with pytest.raises(ValueError):
        bic.graph.mutex_watershed_clustering(graph, edge_costs, bad_mutex_uvs, mutex_costs)


def test_mismatched_mutex_costs_raises():
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.array([[0, 2]], dtype=np.uint64)
    bad_mutex_costs = np.array([1.0, 2.0], dtype=np.float64)  # expected size 1

    with pytest.raises(ValueError):
        bic.graph.mutex_watershed_clustering(graph, edge_costs, mutex_uvs, bad_mutex_costs)


def test_out_of_range_mutex_endpoint_raises():
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    bad_mutex_uvs = np.array([[0, 99]], dtype=np.uint64)
    mutex_costs = np.array([1.0], dtype=np.float64)

    with pytest.raises((ValueError, RuntimeError)):
        bic.graph.mutex_watershed_clustering(graph, edge_costs, bad_mutex_uvs, mutex_costs)
