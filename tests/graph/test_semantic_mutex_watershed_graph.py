import numpy as np
import pytest

import bioimage_cpp as bic


def _graph_from_edges(n: int, uvs):
    uvs = np.asarray(uvs, dtype=np.uint64)
    return bic.graph.UndirectedGraph.from_edges(n, uvs)


def _canonicalize(labels):
    array = np.asarray(labels, dtype=np.uint64)
    seen = {}
    out = np.empty(array.shape, dtype=np.uint64)
    for index, value in enumerate(array):
        key = int(value)
        if key not in seen:
            seen[key] = len(seen)
        out[index] = seen[key]
    return out


def _empty_semantic(dtype=np.float64):
    return (
        np.zeros((0, 2), dtype=np.uint64),
        np.zeros(0, dtype=dtype),
    )


def test_without_semantic_edges_matches_regular_mutex_watershed():
    graph = _graph_from_edges(4, [[0, 1], [1, 2], [2, 3]])
    edge_costs = np.array([10.0, 9.0, 4.0], dtype=np.float64)
    mutex_uvs = np.array([[0, 3]], dtype=np.uint64)
    mutex_costs = np.array([5.0], dtype=np.float64)
    semantic_uvs, semantic_costs = _empty_semantic()

    labels, semantic = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs, semantic_uvs, semantic_costs
    )
    regular_labels = bic.graph.mutex_watershed.mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs
    )

    np.testing.assert_array_equal(_canonicalize(labels), _canonicalize(regular_labels))
    np.testing.assert_array_equal(semantic, -np.ones(4, dtype=np.int64))
    assert semantic.dtype == np.int64
    assert labels.dtype == np.uint64


def test_semantic_constraint_blocks_merge():
    # Two attractive edges 0-1 and 1-2. Node 0 has semantic class 0 (high weight),
    # node 2 has semantic class 1 (high weight). The chain would normally merge
    # all three nodes, but the semantic constraint should block the 1-2 merge
    # after 0 and 1 unite and inherit class 0.
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)
    semantic_node_classes = np.array([[0, 0], [2, 1]], dtype=np.uint64)
    semantic_costs = np.array([10.0, 10.0], dtype=np.float64)

    labels, semantic = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs,
        semantic_node_classes, semantic_costs,
    )

    # Semantic edges have the highest weight so they're processed first.
    # After that, the 0-1 attractive merge brings class 0 onto the joint
    # root. The 1-2 attractive merge is then blocked because 2 has class 1.
    np.testing.assert_array_equal(_canonicalize(labels), np.array([0, 0, 1]))
    assert int(semantic[0]) == 0
    assert int(semantic[1]) == 0
    assert int(semantic[2]) == 1


def test_semantic_label_propagates_across_merge():
    # 0 is assigned class 3 via a high-weight semantic edge. Then the chain
    # 0-1-2-3 merges via attractive edges. All four nodes must end up with
    # the same cluster label AND with semantic_label == 3.
    graph = _graph_from_edges(4, [[0, 1], [1, 2], [2, 3]])
    edge_costs = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)
    semantic_node_classes = np.array([[0, 3]], dtype=np.uint64)
    semantic_costs = np.array([10.0], dtype=np.float64)

    labels, semantic = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs,
        semantic_node_classes, semantic_costs,
    )

    np.testing.assert_array_equal(_canonicalize(labels), np.zeros(4, dtype=np.uint64))
    np.testing.assert_array_equal(semantic, np.full(4, 3, dtype=np.int64))


def test_same_semantic_label_does_not_block_merge():
    # Two roots with the same class label can still merge — the semantic
    # constraint only fires when labels differ.
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)
    semantic_node_classes = np.array([[0, 7], [2, 7]], dtype=np.uint64)
    semantic_costs = np.array([10.0, 10.0], dtype=np.float64)

    labels, semantic = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs,
        semantic_node_classes, semantic_costs,
    )

    np.testing.assert_array_equal(_canonicalize(labels), np.zeros(3, dtype=np.uint64))
    np.testing.assert_array_equal(semantic, np.full(3, 7, dtype=np.int64))


def test_mutex_still_separates_under_semantic():
    # Mutex edges should still block merges regardless of semantic tags.
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.array([[0, 2]], dtype=np.uint64)
    mutex_costs = np.array([5.0], dtype=np.float64)
    semantic_node_classes = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.uint64)
    semantic_costs = np.array([0.5, 0.5, 0.5], dtype=np.float64)

    labels, semantic = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs,
        semantic_node_classes, semantic_costs,
    )

    # Mutex 0-2 (weight 5) is processed first. Then attractive 0-1 (weight
    # 1) merges {0,1}; attractive 1-2 is blocked by the propagated mutex.
    # Semantic class 0 still propagates to all reachable clusters.
    np.testing.assert_array_equal(_canonicalize(labels), np.array([0, 0, 1]))
    np.testing.assert_array_equal(semantic, np.zeros(3, dtype=np.int64))


def test_unassigned_clusters_keep_minus_one():
    graph = _graph_from_edges(3, [[0, 1]])
    edge_costs = np.array([1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)
    semantic_node_classes = np.array([[0, 5]], dtype=np.uint64)
    semantic_costs = np.array([10.0], dtype=np.float64)

    labels, semantic = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs,
        semantic_node_classes, semantic_costs,
    )

    # {0, 1} merges and inherits class 5. {2} is alone and unassigned.
    assert int(semantic[0]) == 5
    assert int(semantic[1]) == 5
    assert int(semantic[2]) == -1


def test_dense_label_range():
    graph = _graph_from_edges(6, [[0, 1], [2, 3], [4, 5]])
    edge_costs = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)
    semantic_uvs, semantic_costs = _empty_semantic()

    labels, _ = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs, semantic_uvs, semantic_costs
    )

    assert set(int(value) for value in labels) == {0, 1, 2}
    assert int(labels.max()) == 2


def test_deterministic_across_runs():
    rng = np.random.default_rng(11)
    n = 20
    uvs = []
    for u in range(n):
        for v in range(u + 1, min(u + 4, n)):
            uvs.append([u, v])
    uvs = np.array(uvs, dtype=np.uint64)
    graph = _graph_from_edges(n, uvs)
    edge_costs = rng.uniform(-1.0, 1.0, size=int(graph.number_of_edges)).astype(np.float64)
    mutex_uvs = rng.integers(0, n, size=(20, 2), dtype=np.uint64)
    mutex_uvs = mutex_uvs[mutex_uvs[:, 0] != mutex_uvs[:, 1]]
    mutex_costs = rng.uniform(0.0, 1.0, size=mutex_uvs.shape[0]).astype(np.float64)
    semantic_nodes = rng.integers(0, n, size=10, dtype=np.uint64)
    semantic_classes = rng.integers(0, 3, size=10, dtype=np.uint64)
    semantic_node_classes = np.stack([semantic_nodes, semantic_classes], axis=1)
    semantic_costs = rng.uniform(0.0, 1.0, size=10).astype(np.float64)

    first = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs, semantic_node_classes, semantic_costs
    )
    second = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs, semantic_node_classes, semantic_costs
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_float32_inputs_supported():
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float32)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float32)
    semantic_node_classes = np.array([[0, 0], [2, 1]], dtype=np.uint64)
    semantic_costs = np.array([10.0, 10.0], dtype=np.float32)

    labels, semantic = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs,
        semantic_node_classes, semantic_costs,
    )

    assert labels.dtype == np.uint64
    assert semantic.dtype == np.int64
    np.testing.assert_array_equal(_canonicalize(labels), np.array([0, 0, 1]))


def test_mismatched_dtypes_are_promoted():
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float32)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)
    semantic_node_classes = np.array([[0, 0], [2, 1]], dtype=np.uint64)
    semantic_costs = np.array([10.0, 10.0], dtype=np.float64)

    labels, semantic = bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
        graph, edge_costs, mutex_uvs, mutex_costs,
        semantic_node_classes, semantic_costs,
    )

    np.testing.assert_array_equal(_canonicalize(labels), np.array([0, 0, 1]))
    assert int(semantic[0]) == 0
    assert int(semantic[2]) == 1


def test_invalid_semantic_shape_raises():
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)
    bad_semantic = np.array([0, 1], dtype=np.uint64)  # 1D, not (n, 2)
    semantic_costs = np.array([1.0], dtype=np.float64)

    with pytest.raises(ValueError):
        bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
            graph, edge_costs, mutex_uvs, mutex_costs, bad_semantic, semantic_costs
        )


def test_mismatched_semantic_costs_raises():
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)
    semantic_node_classes = np.array([[0, 0]], dtype=np.uint64)
    bad_costs = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError):
        bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
            graph, edge_costs, mutex_uvs, mutex_costs,
            semantic_node_classes, bad_costs,
        )


def test_out_of_range_semantic_node_raises():
    graph = _graph_from_edges(3, [[0, 1], [1, 2]])
    edge_costs = np.array([1.0, 1.0], dtype=np.float64)
    mutex_uvs = np.zeros((0, 2), dtype=np.uint64)
    mutex_costs = np.zeros(0, dtype=np.float64)
    semantic_node_classes = np.array([[99, 0]], dtype=np.uint64)
    semantic_costs = np.array([1.0], dtype=np.float64)

    with pytest.raises((ValueError, RuntimeError)):
        bic.graph.mutex_watershed.semantic_mutex_watershed_clustering(
            graph, edge_costs, mutex_uvs, mutex_costs,
            semantic_node_classes, semantic_costs,
        )
