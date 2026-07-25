import numpy as np
import pytest

import bioimage_cpp as bic


def _triangle():
    return bic.graph.UndirectedGraph.from_edges(
        3,
        np.array([[0, 1], [1, 2], [0, 2]], dtype=np.uint64),
    )


def _parallel_paths():
    return bic.graph.UndirectedGraph.from_edges(
        4,
        np.array([[0, 2], [1, 2], [0, 3], [1, 3]], dtype=np.uint64),
    )


def test_contract_edge_materializes_aligned_values_and_mappings():
    graph = _triangle()
    work = bic.graph.ContractionGraph(graph)
    work.add_node_values(
        "position",
        np.array([[0.0, 2.0], [2.0, 4.0], [10.0, 12.0]], dtype=np.float32),
        reduction="mean",
    )
    work.add_edge_values(
        "weight",
        np.array([100.0, 2.0, 3.0], dtype=np.float64),
        reduction="sum",
    )

    assert work.contract_edge(0, keep_node=0) == 0
    result = work.materialize()

    assert isinstance(result.graph, bic.graph.UndirectedGraph)
    np.testing.assert_array_equal(result.graph.uv_ids(), [[0, 1]])
    np.testing.assert_allclose(
        result.node_values["position"],
        [[1.0, 3.0], [10.0, 12.0]],
    )
    assert result.node_values["position"].dtype == np.float32
    np.testing.assert_array_equal(result.edge_values["weight"], [5.0])
    assert result.edge_values["weight"].dtype == np.float64
    np.testing.assert_array_equal(result.node_mapping, [0, 0, 1])
    np.testing.assert_array_equal(result.edge_mapping, [-1, 0, 0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "reduction, expected",
    [
        ("sum", 8.0),
        ("mean", 4.0),
        ("min", 3.0),
        ("max", 5.0),
    ],
)
def test_edge_reductions_preserve_dtype(dtype, reduction, expected):
    work = bic.graph.ContractionGraph(_triangle())
    work.add_edge_values(
        "value",
        np.array([100.0, 3.0, 5.0], dtype=dtype),
        reduction=reduction,
    )

    work.contract_edge(0, keep_node=0)
    values = work.materialize().edge_values["value"]

    assert values.dtype == np.dtype(dtype)
    np.testing.assert_allclose(values, [expected])


@pytest.mark.parametrize(
    "reduction, expected",
    [
        ("sum", 6.0),
        ("mean", 3.0),
        ("min", 2.0),
        ("max", 4.0),
    ],
)
def test_node_reductions(reduction, expected):
    graph = bic.graph.UndirectedGraph.from_edges(2, [[0, 1]])
    work = bic.graph.ContractionGraph(graph)
    work.add_node_values(
        "value",
        np.array([2.0, 4.0], dtype=np.float32),
        reduction=reduction,
    )

    work.contract_edge(0)

    np.testing.assert_allclose(work.materialize().node_values["value"], [expected])


def test_mean_tracks_the_number_of_represented_edges():
    graph = bic.graph.UndirectedGraph.from_edges(
        4,
        [[0, 2], [1, 2], [0, 1], [0, 3], [2, 3]],
    )
    work = bic.graph.ContractionGraph(graph)
    work.add_edge_values(
        "value",
        np.array([1.0, 3.0, 100.0, 10.0, 100.0], dtype=np.float64),
        reduction="mean",
    )

    work.contract_edge(2, keep_node=0)
    work.contract_edge(4, keep_node=2)

    np.testing.assert_allclose(
        work.materialize().edge_values["value"],
        [(1.0 + 3.0 + 10.0) / 3.0],
    )


def test_suppress_node_reduces_edges_and_drops_node_value():
    graph = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])
    work = bic.graph.ContractionGraph(graph)
    work.add_node_values(
        "node",
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        reduction="sum",
    )
    work.add_edge_values(
        "length",
        np.array([2.0, 3.0], dtype=np.float32),
        reduction="sum",
    )

    assert work.suppress_node(1) == 0
    result = work.materialize()

    np.testing.assert_array_equal(result.graph.uv_ids(), [[0, 1]])
    np.testing.assert_array_equal(result.node_values["node"], [1.0, 3.0])
    np.testing.assert_array_equal(result.edge_values["length"], [5.0])
    np.testing.assert_array_equal(result.node_mapping, [0, -1, 1])
    np.testing.assert_array_equal(result.edge_mapping, [0, 0])


def test_materialized_snapshot_is_independent():
    graph = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])
    work = bic.graph.ContractionGraph(graph)
    work.add_edge_values(
        "length",
        np.array([2.0, 3.0], dtype=np.float64),
        reduction="sum",
    )
    replacement = work.suppress_node(1)
    first = work.materialize()

    work.erase_edge(replacement)
    second = work.materialize()

    assert first.graph.number_of_edges == 1
    np.testing.assert_array_equal(first.edge_values["length"], [5.0])
    assert second.graph.number_of_edges == 0
    assert second.edge_values["length"].shape == (0,)
    np.testing.assert_array_equal(second.edge_mapping, [-1, -1])


def test_active_value_queries_align_with_stable_ids():
    work = bic.graph.ContractionGraph(_triangle())
    work.add_edge_values(
        "value",
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
        reduction="sum",
    )
    work.contract_edge(0, keep_node=0)

    edge_ids, values = work.active_edge_values("value")

    np.testing.assert_array_equal(edge_ids, [2])
    np.testing.assert_array_equal(values, [50.0])
    assert work.edge_value("value", 2).shape == ()
    assert float(work.edge_value("value", 2)) == 50.0


def test_input_graph_is_copied():
    graph = bic.graph.UndirectedGraph.from_edges(3, [[0, 1]])
    work = bic.graph.ContractionGraph(graph)

    graph.insert_edge(1, 2)

    assert work.number_of_edges == 1
    assert work.find_edge(1, 2) == -1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "reduction, expected_active, expected_materialized",
    [
        ("sum", [3.0, 7.0], 10.0),
        ("mean", [1.5, 3.5], 2.5),
        ("min", [1.0, 3.0], 1.0),
        ("max", [2.0, 4.0], 4.0),
    ],
)
def test_keep_parallel_edges_fold_only_in_materialized_snapshot(
    dtype,
    reduction,
    expected_active,
    expected_materialized,
):
    work = bic.graph.ContractionGraph(
        _parallel_paths(),
        parallel_edges="keep",
    )
    work.add_edge_values(
        "value",
        np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype),
        reduction=reduction,
    )

    work.suppress_node(2)
    work.suppress_node(3)

    assert work.parallel_edges == "keep"
    assert work.find_edge(0, 1) == 0
    np.testing.assert_array_equal(work.find_edges(0, 1), [0, 2])
    edge_ids, active_values = work.active_edge_values("value")
    np.testing.assert_array_equal(edge_ids, [0, 2])
    np.testing.assert_allclose(active_values, expected_active)

    result = work.materialize()

    np.testing.assert_array_equal(result.graph.uv_ids(), [[0, 1]])
    np.testing.assert_allclose(
        result.edge_values["value"],
        [expected_materialized],
    )
    assert result.edge_values["value"].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result.node_mapping, [0, 1, -1, -1])
    np.testing.assert_array_equal(result.edge_mapping, [0, 0, 0, 0])

    assert work.number_of_edges == 2
    np.testing.assert_array_equal(work.edges(), [0, 2])
    np.testing.assert_allclose(
        work.active_edge_values("value")[1],
        expected_active,
    )


def test_keep_parallel_edges_rejects_self_edge_from_suppression():
    work = bic.graph.ContractionGraph(_triangle(), parallel_edges="keep")
    work.suppress_node(0)

    with pytest.raises(ValueError, match="self edge"):
        work.suppress_node(1)


def test_parallel_edge_policy_validation_and_public_surface():
    with pytest.raises(ValueError, match="parallel_edges"):
        bic.graph.ContractionGraph(_triangle(), parallel_edges="invalid")

    assert not hasattr(bic.graph.ContractionGraph, "_from_edges")
    assert not hasattr(
        bic.graph.ContractionGraph(_triangle()),
        "_deleted_original_edges",
    )


def test_nan_values_propagate():
    work = bic.graph.ContractionGraph(_triangle())
    work.add_edge_values(
        "value",
        np.array([0.0, np.nan, 1.0], dtype=np.float64),
        reduction="min",
    )
    work.contract_edge(0, keep_node=0)

    assert np.isnan(work.materialize().edge_values["value"][0])


def test_invalid_value_registration_and_inactive_ids():
    work = bic.graph.ContractionGraph(_triangle())

    with pytest.raises(TypeError, match="floating dtype"):
        work.add_edge_values("value", np.ones(3, dtype=np.int64), reduction="sum")
    with pytest.raises(ValueError, match=r"shape\[0\]"):
        work.add_edge_values("value", np.ones(2, dtype=np.float32), reduction="sum")
    with pytest.raises(ValueError, match="reduction"):
        work.add_edge_values("value", np.ones(3, dtype=np.float32), reduction="median")

    work.add_edge_values("value", np.ones(3, dtype=np.float32), reduction="sum")
    with pytest.raises(ValueError, match="already exists"):
        work.add_edge_values("value", np.ones(3, dtype=np.float32), reduction="sum")

    work.contract_edge(0, keep_node=0)
    with pytest.raises(ValueError, match="before the first graph mutation"):
        work.add_node_values("late", np.ones(3, dtype=np.float32), reduction="sum")
    with pytest.raises(ValueError, match="inactive"):
        work.edge_value("value", 0)
    with pytest.raises(ValueError, match="degree 2"):
        work.suppress_node(2)
