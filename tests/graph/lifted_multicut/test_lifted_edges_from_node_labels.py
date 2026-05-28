import numpy as np
import pytest

import bioimage_cpp as bic


def _make_chain(n: int):
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.uint64)
    return bic.graph.UndirectedGraph.from_edges(n, edges)


def _as_pair_set(uvs):
    return set(map(tuple, uvs.tolist()))


def test_chain_depth_1_returns_empty():
    graph = _make_chain(6)
    labels = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint64)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=1, mode="all"
    )
    assert out.shape == (0, 2)
    assert out.dtype == np.uint64


def test_chain_depth_2_pairs_at_distance_two():
    graph = _make_chain(6)
    labels = np.arange(6, dtype=np.uint64)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="all"
    )
    assert _as_pair_set(out) == {(0, 2), (1, 3), (2, 4), (3, 5)}
    # Sorted lexicographically.
    assert out.tolist() == sorted(out.tolist())


def test_chain_depth_3_includes_distance_three():
    graph = _make_chain(6)
    labels = np.arange(6, dtype=np.uint64)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=3, mode="all"
    )
    assert _as_pair_set(out) == {
        (0, 2), (1, 3), (2, 4), (3, 5),       # distance 2
        (0, 3), (1, 4), (2, 5),               # distance 3
    }


def test_mode_same_and_different():
    graph = _make_chain(6)
    # Two label-blocks: nodes 0..2 share label 1, nodes 3..5 share label 2.
    # At depth=2 the only pairs are at distance 2:
    #   (0,2): (1,1) same; (1,3): (1,2) different;
    #   (2,4): (1,2) different; (3,5): (2,2) same.
    labels = np.array([1, 1, 1, 2, 2, 2], dtype=np.uint64)
    same = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="same"
    )
    diff = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="different"
    )
    all_pairs = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="all"
    )

    assert _as_pair_set(same) == {(0, 2), (3, 5)}
    assert _as_pair_set(diff) == {(1, 3), (2, 4)}
    # 'same' + 'different' must partition 'all'.
    assert _as_pair_set(same).isdisjoint(_as_pair_set(diff))
    assert _as_pair_set(same) | _as_pair_set(diff) == _as_pair_set(all_pairs)


def test_ignore_label_drops_pairs_with_that_label():
    graph = _make_chain(6)
    labels = np.array([1, 1, 0, 2, 3, 3], dtype=np.uint64)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="all", ignore_label=0
    )
    # Node 2 has the ignore label, so every pair containing it is dropped:
    # (0,2), (2,4) are gone; (1,3) and (3,5) remain.
    assert _as_pair_set(out) == {(1, 3), (3, 5)}


def test_star_graph_emits_all_leaf_leaf_pairs():
    # Node 0 is the center; nodes 1..4 are leaves connected only to 0.
    edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.uint64)
    graph = bic.graph.UndirectedGraph.from_edges(5, edges)
    labels = np.arange(5, dtype=np.uint64)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="all"
    )
    # Every pair of leaves is at distance 2 via the center. No base edges.
    assert _as_pair_set(out) == {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}


def test_node_zero_is_iterated_as_source():
    # Regression guard: nifty.distributed.liftedNeighborhoodFromNodeLabels
    # silently skips node 0 as a source (off-by-one). bic should not.
    graph = _make_chain(4)
    labels = np.arange(4, dtype=np.uint64)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="all"
    )
    pairs = _as_pair_set(out)
    assert (0, 2) in pairs


def test_disconnected_components():
    edges = np.array([[0, 1], [2, 3]], dtype=np.uint64)
    graph = bic.graph.UndirectedGraph.from_edges(4, edges)
    labels = np.arange(4, dtype=np.uint64)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=5, mode="all"
    )
    # Nothing is at distance >= 2 in either two-node component.
    assert out.shape == (0, 2)


def test_rag_input_accepted():
    # Build a tiny 2D labeled image and use its RAG directly.
    labels_img = np.array(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
            [3, 3, 4, 4, 5, 5],
            [3, 3, 4, 4, 5, 5],
        ],
        dtype=np.uint32,
    )
    rag = bic.graph.region_adjacency_graph(labels_img)
    node_labels = np.array([10, 10, 20, 10, 10, 20], dtype=np.uint64)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        rag, node_labels, graph_depth=2, mode="all"
    )
    assert out.dtype == np.uint64
    assert out.ndim == 2 and out.shape[1] == 2
    # Sanity: every pair is a valid (u < v) and not a base edge.
    for u, v in out.tolist():
        assert u < v
        assert rag.find_edge(int(u), int(v)) == -1


@pytest.mark.parametrize(
    "dtype", [np.uint32, np.uint64, np.int32, np.int64]
)
def test_dtype_variants_match(dtype):
    graph = _make_chain(6)
    labels = np.array([1, 1, 2, 2, 3, 3], dtype=dtype)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="all", ignore_label=0
    )
    # No node has the ignore label; result must match the no-ignore call.
    out_noignore = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="all"
    )
    assert _as_pair_set(out) == _as_pair_set(out_noignore)
    assert out.dtype == np.uint64


def test_round_trip_into_lifted_multicut_objective():
    # The output should plug straight into LiftedMulticutObjective.
    graph = _make_chain(6)
    base_costs = np.ones(5, dtype=np.float64)
    labels = np.array([1, 1, 2, 2, 3, 3], dtype=np.uint64)
    lifted_uvs = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="different"
    )
    lifted_costs = -np.ones(lifted_uvs.shape[0], dtype=np.float64)
    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        graph, base_costs,
        lifted_uvs=lifted_uvs, lifted_costs=lifted_costs,
    )
    assert objective.number_of_lifted_edges == lifted_uvs.shape[0]


def test_error_on_unknown_mode():
    graph = _make_chain(3)
    labels = np.zeros(3, dtype=np.uint64)
    with pytest.raises(ValueError, match="mode"):
        bic.graph.lifted_multicut.lifted_edges_from_node_labels(
            graph, labels, graph_depth=2, mode="not-a-mode"
        )


def test_error_on_zero_graph_depth():
    graph = _make_chain(3)
    labels = np.zeros(3, dtype=np.uint64)
    with pytest.raises(ValueError, match="graph_depth"):
        bic.graph.lifted_multicut.lifted_edges_from_node_labels(
            graph, labels, graph_depth=0, mode="all"
        )


def test_error_on_wrong_ndim():
    graph = _make_chain(3)
    labels = np.zeros((3, 1), dtype=np.uint64)
    with pytest.raises(ValueError, match="1D"):
        bic.graph.lifted_multicut.lifted_edges_from_node_labels(
            graph, labels, graph_depth=2, mode="all"
        )


def test_error_on_length_mismatch():
    graph = _make_chain(3)
    labels = np.zeros(5, dtype=np.uint64)
    with pytest.raises(ValueError, match="number_of_nodes"):
        bic.graph.lifted_multicut.lifted_edges_from_node_labels(
            graph, labels, graph_depth=2, mode="all"
        )


def test_error_on_unsupported_dtype():
    graph = _make_chain(3)
    labels = np.zeros(3, dtype=np.float32)
    with pytest.raises(TypeError, match="dtype"):
        bic.graph.lifted_multicut.lifted_edges_from_node_labels(
            graph, labels, graph_depth=2, mode="all"
        )


def test_threading_produces_same_result():
    graph = _make_chain(10)
    labels = np.arange(10, dtype=np.uint64)
    single = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=3, mode="all", number_of_threads=1
    )
    multi = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=3, mode="all", number_of_threads=4
    )
    assert _as_pair_set(single) == _as_pair_set(multi)
    assert single.tolist() == multi.tolist()  # sorted output is deterministic


def test_empty_graph():
    graph = bic.graph.UndirectedGraph(0)
    labels = np.zeros(0, dtype=np.uint64)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=2, mode="all"
    )
    assert out.shape == (0, 2)


def test_default_threading_is_deterministic_on_large_chain():
    # Regression guard for a data race in the lazy CSR-adjacency rebuild: with
    # default (multi-threaded) execution, every worker used to trigger the
    # not-thread-safe rebuild concurrently on the first node_adjacency() read,
    # corrupting the adjacency. That produced run-to-run varying counts and
    # intermittent segfaults. A graph this size reliably exposes the race
    # (a 10-node chain does not). The result must equal the single-threaded
    # reference on every run.
    n = 2000
    graph = _make_chain(n)  # built via from_edges -> arrives "dirty"
    labels = np.ones(n, dtype=np.uint64)
    reference = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        graph, labels, graph_depth=3, mode="all", number_of_threads=1
    )
    for _ in range(25):
        out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
            graph, labels, graph_depth=3, mode="all"  # default: multi-threaded
        )
        assert out.tolist() == reference.tolist()


def test_default_threading_is_deterministic_on_rag():
    # Same race, reached through the region_adjacency_graph construction path,
    # which also returns a graph with a dirty (not-yet-built) adjacency.
    n = 2000
    segmentation = np.repeat(np.arange(n, dtype=np.uint32), 16).reshape(n, 4, 4)
    rag = bic.graph.region_adjacency_graph(segmentation)
    labels = np.ones(rag.numberOfNodes, dtype=np.uint64)
    reference = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        rag, labels, graph_depth=3, mode="all", number_of_threads=1
    )
    for _ in range(25):
        out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
            rag, labels, graph_depth=3, mode="all"  # default: multi-threaded
        )
        assert out.tolist() == reference.tolist()
