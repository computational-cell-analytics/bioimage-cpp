import numpy as np
import pytest

import bioimage_cpp as bic


def test_grid_boundary_features_2d_align_to_grid_edges():
    graph = bic.graph.GridGraph2D((2, 3))
    boundary_map = np.array([[0.0, 2.0, 4.0], [10.0, 12.0, 14.0]])

    weights = bic.graph.features.grid_boundary_features(graph, boundary_map)

    assert weights.dtype == np.float64
    np.testing.assert_allclose(
        weights,
        np.array([5.0, 7.0, 9.0, 1.0, 3.0, 11.0, 13.0]),
    )


def test_grid_affinity_features_2d_local_offsets():
    graph = bic.graph.GridGraph2D((2, 3))
    affinities = np.zeros((2, 2, 3), dtype=np.float64)
    affinities[0] = np.array([[10.0, 11.0, 12.0], [0.0, 0.0, 0.0]])
    affinities[1] = np.array([[20.0, 21.0, 0.0], [23.0, 24.0, 0.0]])

    weights, valid = bic.graph.features.grid_affinity_features(
        graph, affinities, offsets=[(1, 0), (0, 1)]
    )

    np.testing.assert_allclose(weights, np.array([10.0, 11.0, 12.0, 20.0, 21.0, 23.0, 24.0]))
    np.testing.assert_array_equal(valid, np.ones(graph.number_of_edges, dtype=bool))


def test_grid_affinity_features_partial_local_offsets():
    graph = bic.graph.GridGraph2D((2, 3))
    affinities = np.arange(6, dtype=np.float64).reshape(1, 2, 3)

    weights, valid = bic.graph.features.grid_affinity_features(
        graph, affinities, offsets=[(1, 0)]
    )

    np.testing.assert_allclose(weights, np.array([0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(valid, np.array([True, True, True, False, False, False, False]))


def test_grid_affinity_features_with_lifted_2d():
    graph = bic.graph.GridGraph2D((3, 3))
    affinities = np.zeros((2, 3, 3), dtype=np.float64)
    affinities[0] = np.arange(9, dtype=np.float64).reshape(3, 3)
    affinities[1] = 100.0 + np.arange(9, dtype=np.float64).reshape(3, 3)

    local_weights, valid, lifted_uvs, lifted_weights, lifted_offset_ids = (
        bic.graph.features.grid_affinity_features_with_lifted(
            graph, affinities, offsets=[(1, 0), (0, 2)]
        )
    )

    np.testing.assert_allclose(
        local_weights,
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        valid,
        np.array([True, True, True, True, True, True, False, False, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        lifted_uvs,
        np.array([[0, 2], [3, 5], [6, 8]], dtype=np.uint64),
    )
    np.testing.assert_allclose(lifted_weights, np.array([100.0, 103.0, 106.0]))
    np.testing.assert_array_equal(lifted_offset_ids, np.array([1, 1, 1], dtype=np.uint64))


def test_grid_affinity_features_with_lifted_3d():
    graph = bic.graph.GridGraph3D((2, 1, 3))
    affinities = np.zeros((2, 2, 1, 3), dtype=np.float64)
    affinities[0, 0, 0, :] = [1.0, 2.0, 3.0]
    affinities[1, 0, 0, 0] = 9.0
    affinities[1, 1, 0, 0] = 10.0

    local_weights, valid, lifted_uvs, lifted_weights, lifted_offset_ids = (
        bic.graph.features.grid_affinity_features_with_lifted(
            graph, affinities, offsets=[(1, 0, 0), (0, 0, 2)]
        )
    )

    np.testing.assert_allclose(local_weights[:3], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(valid[:3], np.array([True, True, True]))
    np.testing.assert_array_equal(
        lifted_uvs,
        np.array([[0, 2], [3, 5]], dtype=np.uint64),
    )
    np.testing.assert_allclose(lifted_weights, np.array([9.0, 10.0]))
    np.testing.assert_array_equal(lifted_offset_ids, np.array([1, 1], dtype=np.uint64))


def test_grid_affinity_features_rejects_long_range_offsets():
    graph = bic.graph.GridGraph2D((3, 3))
    affinities = np.zeros((1, 3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="only local offsets"):
        bic.graph.features.grid_affinity_features(graph, affinities, offsets=[(0, 2)])


def test_grid_affinity_features_rejects_duplicate_local_edges():
    graph = bic.graph.GridGraph2D((2, 3))
    affinities = np.zeros((2, 2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="duplicate local"):
        bic.graph.features.grid_affinity_features(graph, affinities, offsets=[(0, 1), (0, -1)])


def test_grid_affinity_features_rejects_duplicate_lifted_edges():
    graph = bic.graph.GridGraph2D((1, 3))
    affinities = np.zeros((2, 1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="duplicate long-range"):
        bic.graph.features.grid_affinity_features_with_lifted(
            graph, affinities, offsets=[(0, 2), (0, -2)]
        )


def test_grid_boundary_features_preserves_float32_dtype():
    graph = bic.graph.GridGraph2D((2, 3))
    boundary_map = np.array(
        [[0.0, 2.0, 4.0], [10.0, 12.0, 14.0]], dtype=np.float32
    )

    weights = bic.graph.features.grid_boundary_features(graph, boundary_map)

    assert weights.dtype == np.float32
    np.testing.assert_allclose(
        weights, np.array([5.0, 7.0, 9.0, 1.0, 3.0, 11.0, 13.0], dtype=np.float32)
    )


def test_grid_affinity_features_preserves_float32_dtype():
    graph = bic.graph.GridGraph2D((2, 3))
    affinities = np.zeros((2, 2, 3), dtype=np.float32)
    affinities[0] = np.array([[10.0, 11.0, 12.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    affinities[1] = np.array([[20.0, 21.0, 0.0], [23.0, 24.0, 0.0]], dtype=np.float32)

    weights, valid = bic.graph.features.grid_affinity_features(
        graph, affinities, offsets=[(1, 0), (0, 1)]
    )

    assert weights.dtype == np.float32
    np.testing.assert_allclose(
        weights,
        np.array([10.0, 11.0, 12.0, 20.0, 21.0, 23.0, 24.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(valid, np.ones(graph.number_of_edges, dtype=bool))


def test_grid_affinity_features_with_lifted_preserves_float32_dtype():
    graph = bic.graph.GridGraph2D((3, 3))
    affinities = np.zeros((2, 3, 3), dtype=np.float32)
    affinities[0] = np.arange(9, dtype=np.float32).reshape(3, 3)
    affinities[1] = 100.0 + np.arange(9, dtype=np.float32).reshape(3, 3)

    local_weights, valid, lifted_uvs, lifted_weights, _ = (
        bic.graph.features.grid_affinity_features_with_lifted(
            graph, affinities, offsets=[(1, 0), (0, 2)]
        )
    )

    assert local_weights.dtype == np.float32
    assert lifted_weights.dtype == np.float32
    np.testing.assert_allclose(
        local_weights,
        np.array(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        lifted_weights, np.array([100.0, 103.0, 106.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        lifted_uvs, np.array([[0, 2], [3, 5], [6, 8]], dtype=np.uint64)
    )


def test_grid_features_integer_input_promotes_to_float64():
    graph = bic.graph.GridGraph2D((2, 3))
    boundary_map = np.array([[0, 2, 4], [10, 12, 14]], dtype=np.int32)

    weights = bic.graph.features.grid_boundary_features(graph, boundary_map)

    assert weights.dtype == np.float64


def test_grid_feature_validation():
    graph = bic.graph.GridGraph2D((2, 3))

    with pytest.raises(ValueError, match="boundary_map shape"):
        bic.graph.features.grid_boundary_features(graph, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="affinities must have shape"):
        bic.graph.features.grid_affinity_features(graph, np.zeros((2, 3)), offsets=[(0, 1)])
    with pytest.raises(ValueError, match="offsets length"):
        bic.graph.features.grid_affinity_features(graph, np.zeros((2, 2, 3)), offsets=[(0, 1)])
    with pytest.raises(ValueError, match="zero offset"):
        bic.graph.features.grid_affinity_features(graph, np.zeros((1, 2, 3)), offsets=[(0, 0)])
