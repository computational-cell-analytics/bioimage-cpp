import numpy as np
import pytest

import bioimage_cpp as bic


def _labels_2d():
    return np.array(
        [
            [1, 1, 2],
            [1, 3, 2],
            [4, 3, 3],
        ],
        dtype=np.uint64,
    )


def test_rag_coordinates_basic_2d():
    labels = _labels_2d()
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels)

    assert isinstance(coords, bic.graph.RagCoordinates)
    assert coords.ndim == 2
    assert coords.shape == [3, 3]
    assert coords.number_of_edges == rag.number_of_edges

    # uv_ids: [[1,2],[1,3],[1,4],[2,3],[3,4]]
    # Each contact contributes 2 points (low + high).
    np.testing.assert_array_equal(
        coords.storage_lengths(),
        np.array([2, 4, 2, 4, 2], dtype=np.uint64),
    )


def test_rag_coordinates_edge_coordinates_and_directions():
    labels = _labels_2d()
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels)

    # edge 0 is (1, 2): single contact at (0,1)-(0,2).
    np.testing.assert_array_equal(
        coords.edge_coordinates(0), np.array([[0, 1], [0, 2]])
    )
    np.testing.assert_array_equal(
        coords.edge_coordinates(0, edge_direction=1), np.array([[0, 1]])
    )
    np.testing.assert_array_equal(
        coords.edge_coordinates(0, edge_direction=2), np.array([[0, 2]])
    )

    # edge 1 is (1, 3): two contacts (0,1)-(1,1) and (1,0)-(1,1), in scan order.
    np.testing.assert_array_equal(
        coords.edge_coordinates(1),
        np.array([[0, 1], [1, 1], [1, 0], [1, 1]]),
    )
    np.testing.assert_array_equal(
        coords.edge_coordinates(1, edge_direction=1),
        np.array([[0, 1], [1, 0]]),
    )
    np.testing.assert_array_equal(
        coords.edge_coordinates(1, edge_direction=2),
        np.array([[1, 1], [1, 1]]),
    )


def test_rag_coordinates_edges_to_volume_both():
    labels = _labels_2d()
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels)

    values = np.arange(1, rag.number_of_edges + 1, dtype=np.float32)
    volume = coords.edges_to_volume(values, ignore_value=-1.0)

    assert volume.dtype == np.float32
    assert volume.shape == (3, 3)
    # Higher edge id wins where boundaries coincide.
    np.testing.assert_array_equal(
        volume,
        np.array(
            [
                [-1.0, 2.0, 1.0],
                [3.0, 4.0, 4.0],
                [5.0, 5.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )


def test_rag_coordinates_edges_to_volume_low_side():
    labels = _labels_2d()
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels)

    values = np.arange(1, rag.number_of_edges + 1, dtype=np.float64)
    volume = coords.edges_to_volume(values, edge_direction=1, ignore_value=0.0)

    assert volume.dtype == np.float64
    # Only the lower-coordinate pixel of each contact is painted; (2,1) is only
    # ever a higher-side pixel (for edge 4) and so stays at the ignore value.
    np.testing.assert_array_equal(
        volume,
        np.array(
            [
                [0.0, 2.0, 0.0],
                [3.0, 4.0, 4.0],
                [5.0, 0.0, 0.0],
            ]
        ),
    )


def test_rag_coordinates_edges_to_volume_uint():
    labels = _labels_2d()
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels)

    values = (np.arange(rag.number_of_edges) + 10).astype(np.uint64)
    volume = coords.edges_to_volume(values, ignore_value=0)
    assert volume.dtype == np.uint64
    # Every boundary pixel got some non-zero edge value.
    assert volume.max() > 0


def test_rag_coordinates_3d():
    labels = np.array(
        [
            [[1, 1], [2, 2]],
            [[1, 3], [2, 3]],
        ],
        dtype=np.uint32,
    )
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels, number_of_threads=1)

    assert coords.ndim == 3
    assert coords.shape == [2, 2, 2]
    assert coords.number_of_edges == rag.number_of_edges

    # Every stored coordinate must point to one of the two endpoint labels.
    for edge in range(rag.number_of_edges):
        u, v = rag.uv(edge)
        pts = coords.edge_coordinates(edge)
        seen = {int(labels[tuple(p)]) for p in pts}
        assert seen <= {int(u), int(v)}
        assert seen  # not empty


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64, np.int32, np.int64])
def test_rag_coordinates_label_dtypes(dtype):
    labels = _labels_2d().astype(dtype)
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels)
    np.testing.assert_array_equal(
        coords.storage_lengths(),
        np.array([2, 4, 2, 4, 2], dtype=np.uint64),
    )


def test_rag_coordinates_parallel_matches_single_thread():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 8, size=(16, 17), dtype=np.int64)
    rag = bic.graph.region_adjacency_graph(labels)

    single = bic.graph.rag_coordinates(rag, labels, number_of_threads=1)
    parallel = bic.graph.rag_coordinates(rag, labels, number_of_threads=4)

    np.testing.assert_array_equal(
        single.storage_lengths(), parallel.storage_lengths()
    )
    for edge in range(rag.number_of_edges):
        np.testing.assert_array_equal(
            single.edge_coordinates(edge), parallel.edge_coordinates(edge)
        )


def test_rag_coordinates_storage_lengths_sum():
    labels = _labels_2d()
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels)

    # Each inter-region adjacent pixel pair contributes exactly 2 points.
    expected_contacts = 0
    for axis in range(labels.ndim):
        a = np.take(labels, range(labels.shape[axis] - 1), axis=axis)
        b = np.take(labels, range(1, labels.shape[axis]), axis=axis)
        expected_contacts += int(np.count_nonzero(a != b))
    assert int(coords.storage_lengths().sum()) == 2 * expected_contacts


def test_rag_coordinates_rejects_invalid_input():
    labels = _labels_2d()
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels)

    with pytest.raises(ValueError, match="shape"):
        bic.graph.rag_coordinates(rag, labels[:, :2])

    with pytest.raises(TypeError, match="one of dtypes"):
        bic.graph.rag_coordinates(rag, labels.astype(np.uint8))

    with pytest.raises(ValueError, match="length must match"):
        coords.edges_to_volume(np.ones(rag.number_of_edges + 1, dtype=np.float32))

    with pytest.raises(TypeError, match="one of dtypes"):
        coords.edges_to_volume(np.ones(rag.number_of_edges, dtype=np.int8))

    with pytest.raises(ValueError, match="edge_direction"):
        coords.edge_coordinates(0, edge_direction=3)

    with pytest.raises(ValueError, match="edge_direction"):
        coords.edges_to_volume(
            np.ones(rag.number_of_edges, dtype=np.float32), edge_direction=5
        )


def test_rag_coordinates_camelcase_aliases():
    labels = _labels_2d()
    rag = bic.graph.region_adjacency_graph(labels)
    coords = bic.graph.rag_coordinates(rag, labels)

    np.testing.assert_array_equal(coords.storageLengths(), coords.storage_lengths())
    np.testing.assert_array_equal(
        coords.edgeCoordinates(1), coords.edge_coordinates(1)
    )
    values = np.arange(1, rag.number_of_edges + 1, dtype=np.float32)
    np.testing.assert_array_equal(
        coords.edgesToVolume(values, ignore_value=-1.0),
        coords.edges_to_volume(values, ignore_value=-1.0),
    )
