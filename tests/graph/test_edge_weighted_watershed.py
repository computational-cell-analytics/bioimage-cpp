import numpy as np
import pytest

import bioimage_cpp as bic


def _path_graph(n: int) -> bic.graph.UndirectedGraph:
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.uint64)
    return bic.graph.UndirectedGraph.from_edges(n, edges)


def test_propagates_along_lowest_weight_path():
    graph = _path_graph(5)
    # Weights: 0-1=0.1, 1-2=0.5, 2-3=0.2, 3-4=0.1
    # Seeds at node 0 (label 1) and node 4 (label 2).
    # Ascending sort: edges (0-1, 3-4) tie, then (2-3), then (1-2).
    # After 0-1 and 3-4: nodes 0,1 -> 1; nodes 3,4 -> 2.
    # 2-3: node 2 unlabeled, joins label 2 component.
    # 1-2: both labeled, no merge — boundary stays.
    weights = np.array([0.1, 0.5, 0.2, 0.1], dtype=np.float64)
    seeds = np.array([1, 0, 0, 0, 2], dtype=np.uint64)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    assert labels.dtype == np.uint64
    np.testing.assert_array_equal(labels, np.array([1, 1, 2, 2, 2], dtype=np.uint64))


def test_distinct_labeled_components_are_not_merged():
    graph = _path_graph(3)
    weights = np.array([0.1, 0.2], dtype=np.float64)
    seeds = np.array([1, 0, 2], dtype=np.uint64)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    np.testing.assert_array_equal(labels, np.array([1, 1, 2], dtype=np.uint64))


def test_components_with_same_label_stay_separate():
    # Matches nifty's Kruskal variant: two pre-labeled components with the
    # same label are not merged because the condition requires at least one
    # to be unlabeled.
    graph = _path_graph(3)
    weights = np.array([0.1, 0.2], dtype=np.float64)
    seeds = np.array([1, 0, 1], dtype=np.uint64)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    # Edge 0-1 merges node 1 into label 1. Edge 1-2 then has both endpoints
    # labeled (1 and 1), so no merge happens — but the result still gives
    # node 2 label 1 from its own seed.
    np.testing.assert_array_equal(labels, np.array([1, 1, 1], dtype=np.uint64))


def test_unreachable_nodes_remain_zero():
    graph = bic.graph.UndirectedGraph(4)  # no edges
    weights = np.zeros(0, dtype=np.float64)
    seeds = np.array([1, 0, 2, 0], dtype=np.uint64)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    np.testing.assert_array_equal(labels, seeds)


def test_no_seeds_returns_zero_labels():
    graph = _path_graph(4)
    weights = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    seeds = np.zeros(4, dtype=np.uint64)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    np.testing.assert_array_equal(labels, np.zeros(4, dtype=np.uint64))


def test_high_weight_edge_is_processed_last():
    # Y-shape: nodes 0 and 2 are seeds, node 1 is the junction.
    # The 1-3 edge has a high weight, so node 3 is grabbed by node 1's
    # component last; the seed reached via the lower-weight path wins.
    graph = bic.graph.UndirectedGraph.from_edges(
        4, np.array([[0, 1], [1, 2], [1, 3]], dtype=np.uint64)
    )
    weights = np.array([0.1, 0.2, 0.9], dtype=np.float64)
    seeds = np.array([1, 0, 2, 0], dtype=np.uint64)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    # Edge 0 (0-1, w=0.1): node 1 gets label 1.
    # Edge 1 (1-2, w=0.2): both labeled (1 vs 2), no merge.
    # Edge 2 (1-3, w=0.9): node 3 joins component of node 1 (label 1).
    np.testing.assert_array_equal(labels, np.array([1, 1, 2, 1], dtype=np.uint64))


def test_accepts_region_adjacency_graph():
    labels_image = np.array([[1, 1, 2], [3, 3, 2]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels_image)

    weights = np.ones(rag.number_of_edges, dtype=np.float64)
    seeds = np.array([0, 7, 0, 0], dtype=np.uint64)

    labels = bic.graph.edge_weighted_watershed(rag, weights, seeds)

    assert labels.dtype == np.uint64
    assert labels.shape == (rag.number_of_nodes,)
    # node 1 is the only seed, so all reachable nodes get label 7.
    np.testing.assert_array_equal(labels[1:], np.full(3, 7, dtype=np.uint64))


def test_preserves_seed_label_values():
    graph = _path_graph(4)
    weights = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    seeds = np.array([42, 0, 0, 99], dtype=np.uint64)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    assert set(np.unique(labels).tolist()).issubset({42, 99})


def test_accepts_list_inputs():
    graph = _path_graph(3)

    labels = bic.graph.edge_weighted_watershed(graph, [0.1, 0.2], [1, 0, 2])

    # numpy default int dtype is int64 on this platform; the seed dtype is
    # preserved on the output.
    np.testing.assert_array_equal(labels, np.array([1, 1, 2]))


@pytest.mark.parametrize("weight_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed_dtype", [np.uint32, np.uint64, np.int32, np.int64])
def test_dtype_combinations_preserve_seed_dtype(weight_dtype, seed_dtype):
    graph = _path_graph(5)
    weights = np.array([0.1, 0.5, 0.2, 0.1], dtype=weight_dtype)
    seeds = np.array([1, 0, 0, 0, 2], dtype=seed_dtype)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    assert labels.dtype == np.dtype(seed_dtype)
    np.testing.assert_array_equal(
        labels, np.array([1, 1, 2, 2, 2], dtype=seed_dtype)
    )


def test_float32_weights_are_not_copied_to_float64():
    # The float32 path should accept float32 inputs without an upcast.
    graph = _path_graph(3)
    weights = np.array([0.1, 0.2], dtype=np.float32)
    seeds = np.array([1, 0, 2], dtype=np.uint32)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    assert labels.dtype == np.dtype("uint32")
    np.testing.assert_array_equal(labels, np.array([1, 1, 2], dtype=np.uint32))


def test_float16_weights_are_promoted_to_float32():
    graph = _path_graph(3)
    weights = np.array([0.1, 0.2], dtype=np.float16)
    seeds = np.array([1, 0, 2], dtype=np.uint64)

    labels = bic.graph.edge_weighted_watershed(graph, weights, seeds)

    np.testing.assert_array_equal(labels, np.array([1, 1, 2], dtype=np.uint64))


def test_rejects_unsupported_weight_dtype():
    graph = _path_graph(3)
    with pytest.raises(TypeError, match="edge_weights must have dtype"):
        bic.graph.edge_weighted_watershed(
            graph,
            np.array([1, 2], dtype=np.int32),
            np.array([1, 0, 2], dtype=np.uint64),
        )


def test_rejects_unsupported_seed_dtype():
    graph = _path_graph(3)
    with pytest.raises(TypeError, match="seeds must have dtype"):
        bic.graph.edge_weighted_watershed(
            graph,
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([1, 0, 2], dtype=np.uint8),
        )


def test_rejects_negative_signed_seeds():
    graph = _path_graph(3)
    with pytest.raises(ValueError, match="seeds must not contain negative values"):
        bic.graph.edge_weighted_watershed(
            graph,
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([1, -1, 2], dtype=np.int32),
        )


def test_rejects_wrong_weight_length():
    graph = _path_graph(3)
    with pytest.raises(ValueError, match="edge_weights length"):
        bic.graph.edge_weighted_watershed(
            graph,
            np.zeros(3, dtype=np.float64),
            np.array([1, 0, 2], dtype=np.uint64),
        )


def test_rejects_wrong_seeds_length():
    graph = _path_graph(3)
    with pytest.raises(ValueError, match="seeds length"):
        bic.graph.edge_weighted_watershed(
            graph,
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([1, 0], dtype=np.uint64),
        )


def test_rejects_non_1d_inputs():
    graph = _path_graph(3)
    with pytest.raises(ValueError, match="edge_weights must be a 1D array"):
        bic.graph.edge_weighted_watershed(
            graph,
            np.zeros((2, 1), dtype=np.float64),
            np.array([1, 0, 2], dtype=np.uint64),
        )
    with pytest.raises(ValueError, match="seeds must be a 1D array"):
        bic.graph.edge_weighted_watershed(
            graph,
            np.array([0.1, 0.2], dtype=np.float64),
            np.zeros((3, 1), dtype=np.uint64),
        )


def _reference_kruskal_watershed(graph, weights, seeds):
    """Pure-Python reference implementation of the Kruskal seeded watershed.

    Mirrors nifty's edgeWeightedWatershedsSegmentationKruskalImpl: walks edges
    in ascending weight order, merges components iff at least one is unlabeled,
    propagates the non-zero label.
    """
    n = int(graph.number_of_nodes)
    weights = np.asarray(weights, dtype=np.float64)
    labels = np.asarray(seeds, dtype=np.uint64).copy()
    uvs = graph.uv_ids()

    parent = np.arange(n, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    order = np.argsort(weights, kind="stable")
    for edge_index in order:
        u, v = int(uvs[edge_index, 0]), int(uvs[edge_index, 1])
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        lu, lv = labels[ru], labels[rv]
        if lu != 0 and lv != 0:
            continue
        new_label = max(int(lu), int(lv))
        parent[rv] = ru
        labels[ru] = new_label
        labels[rv] = new_label

    for node in range(n):
        labels[node] = labels[find(node)]
    return labels


def test_matches_python_reference_on_random_graph():
    rng = np.random.default_rng(42)
    n_nodes = 50
    n_edges = 150

    edges = set()
    while len(edges) < n_edges:
        u, v = rng.integers(0, n_nodes, size=2)
        if u != v:
            edges.add((min(int(u), int(v)), max(int(u), int(v))))
    uvs = np.array(list(edges), dtype=np.uint64)

    graph = bic.graph.UndirectedGraph.from_edges(n_nodes, uvs)
    weights = rng.standard_normal(graph.number_of_edges).astype(np.float64)
    seeds = np.zeros(n_nodes, dtype=np.uint64)
    seed_nodes = rng.choice(n_nodes, size=5, replace=False)
    for label, node in enumerate(seed_nodes, start=1):
        seeds[node] = label

    actual = bic.graph.edge_weighted_watershed(graph, weights, seeds)
    expected = _reference_kruskal_watershed(graph, weights, seeds)

    np.testing.assert_array_equal(actual, expected)
