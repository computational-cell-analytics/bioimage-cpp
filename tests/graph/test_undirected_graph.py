import numpy as np
import pytest

import bioimage_cpp as bic


def test_undirected_graph_insert_and_lookup_edges():
    graph = bic.graph.UndirectedGraph(4)

    assert graph.number_of_nodes == 4
    assert graph.number_of_edges == 0
    assert graph.node_id_upper_bound == 3
    assert graph.edge_id_upper_bound == 0

    first = graph.insert_edge(2, 1)
    second = graph.insert_edge(1, 3)
    duplicate = graph.insert_edge(1, 2)

    assert first == 0
    assert second == 1
    assert duplicate == first
    assert graph.number_of_edges == 2
    assert graph.edge_id_upper_bound == 1
    assert graph.find_edge(1, 2) == first
    assert graph.find_edge(2, 1) == first
    assert graph.find_edge(0, 3) == -1
    assert graph.u(first) == 1
    assert graph.v(first) == 2
    assert graph.uv(first) == (1, 2)


def test_undirected_graph_bulk_operations_return_numpy_arrays():
    graph = bic.graph.undirected_graph(5)
    edge_ids = graph.insert_edges([[0, 1], [3, 4], [1, 0]])

    assert edge_ids.dtype == np.uint64
    np.testing.assert_array_equal(edge_ids, np.array([0, 1, 0], dtype=np.uint64))
    np.testing.assert_array_equal(graph.nodes(), np.arange(5, dtype=np.uint64))
    np.testing.assert_array_equal(graph.edges(), np.arange(2, dtype=np.uint64))
    np.testing.assert_array_equal(
        graph.uv_ids(), np.array([[0, 1], [3, 4]], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        graph.find_edges([[1, 0], [0, 4], [4, 3]]),
        np.array([0, -1, 1], dtype=np.int64),
    )


def test_undirected_graph_node_adjacency():
    graph = bic.graph.UndirectedGraph.from_edges(4, np.array([[0, 1], [1, 2], [1, 3]]))

    np.testing.assert_array_equal(
        graph.node_adjacency(1),
        np.array([[0, 0], [2, 1], [3, 2]], dtype=np.uint64),
    )


def test_undirected_graph_serialize_and_deserialize():
    graph = bic.graph.UndirectedGraph.from_edges(3, [[0, 2], [0, 1]])

    serialized = graph.serialize()
    restored = bic.graph.UndirectedGraph.deserialize(serialized)

    assert graph.serialization_size == 6
    np.testing.assert_array_equal(
        serialized, np.array([3, 2, 0, 2, 0, 1], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        restored.uv_ids(), np.array([[0, 2], [0, 1]], dtype=np.uint64)
    )


def test_undirected_graph_extracts_subgraph_edges():
    graph = bic.graph.UndirectedGraph.from_edges(
        5, [[0, 1], [1, 2], [2, 3], [2, 4], [0, 4]]
    )

    inner, outer = graph.extract_subgraph_from_nodes([0, 1, 2])

    np.testing.assert_array_equal(inner, np.array([0, 1], dtype=np.uint64))
    np.testing.assert_array_equal(outer, np.array([4, 2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(graph.edges_from_node_list([0, 1, 2]), inner)


def test_undirected_graph_assign_clears_topology():
    graph = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])

    graph.assign(2)

    assert graph.number_of_nodes == 2
    assert graph.number_of_edges == 0
    np.testing.assert_array_equal(graph.nodes(), np.array([0, 1], dtype=np.uint64))
    np.testing.assert_array_equal(graph.edges(), np.array([], dtype=np.uint64))


def test_undirected_graph_nifty_style_aliases():
    graph = bic.graph.UndirectedGraph(3)

    assert graph.numberOfNodes == 3
    assert graph.nodeIdUpperBound == 2
    assert graph.insertEdge(0, 2) == 0
    assert graph.findEdge(2, 0) == 0
    assert graph.edgeIdUpperBound == 0
    assert graph.serializationSize == 4
    np.testing.assert_array_equal(graph.uvIds(), np.array([[0, 2]], dtype=np.uint64))
    np.testing.assert_array_equal(
        graph.nodeAdjacency(0), np.array([[2, 0]], dtype=np.uint64)
    )


def test_undirected_graph_rejects_invalid_edges():
    graph = bic.graph.UndirectedGraph(2)

    with pytest.raises(ValueError, match="self edges"):
        graph.insert_edge(1, 1)
    with pytest.raises(IndexError, match="node id must be < number_of_nodes"):
        graph.insert_edge(0, 2)
    with pytest.raises(ValueError, match="uvs must have shape"):
        graph.insert_edges([0, 1])


def test_undirected_graph_clone_is_independent_deep_copy():
    original = bic.graph.UndirectedGraph(4)
    original.insert_edge(0, 1)
    original.insert_edge(1, 2)
    original.insert_edge(2, 3)

    copy = original.clone()
    assert copy.number_of_nodes == 4
    assert copy.number_of_edges == 3
    np.testing.assert_array_equal(copy.uv_ids(), original.uv_ids())
    np.testing.assert_array_equal(copy.node_adjacency(1), original.node_adjacency(1))

    # Mutating the copy must not affect the original.
    copy.insert_edge(0, 3)
    assert copy.number_of_edges == 4
    assert original.number_of_edges == 3
    assert original.find_edge(0, 3) == -1
    assert copy.find_edge(0, 3) == 3


def test_undirected_graph_freeze_is_callable_after_inserts_and_after_reads():
    graph = bic.graph.UndirectedGraph(3)
    graph.insert_edge(0, 1)
    graph.insert_edge(1, 2)
    # Eagerly rebuild adjacency before any read.
    graph.freeze()
    np.testing.assert_array_equal(
        graph.node_adjacency(1), np.array([[0, 0], [2, 1]], dtype=np.uint64)
    )
    # Frozen graph still accepts new edges; lazy rebuild fires again on next read.
    graph.insert_edge(0, 2)
    graph.freeze()
    assert graph.number_of_edges == 3
    np.testing.assert_array_equal(
        np.sort(graph.node_adjacency(0)[:, 0]), np.array([1, 2], dtype=np.uint64)
    )
