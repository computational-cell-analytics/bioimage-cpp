import numpy as np
import pytest

import bioimage_cpp as bic


def test_mutex_watershed_2d_attractive_edges_merge_all_pixels():
    affinities = np.ones((2, 3, 4), dtype=np.float32)
    offsets = [[0, 1], [1, 0]]

    labels = bic.segmentation.mutex_watershed(
        affinities, offsets, number_of_attractive_channels=2
    )

    assert labels.dtype == np.uint64
    assert labels.shape == (3, 4)
    np.testing.assert_array_equal(labels, np.ones((3, 4), dtype=np.uint64))


def test_mutex_watershed_2d_mutex_edge_blocks_lower_attractive_edge():
    affinities = np.array([[[0.5, 0.0]], [[0.9, 0.0]]], dtype=np.float64)
    offsets = [[0, 1], [0, 1]]

    labels = bic.segmentation.mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1
    )

    np.testing.assert_array_equal(labels, np.array([[1, 2]], dtype=np.uint64))


def test_mutex_watershed_2d_higher_attractive_edge_wins_before_mutex_edge():
    affinities = np.array([[[0.9, 0.0]], [[0.5, 0.0]]], dtype=np.float32)
    offsets = [[0, 1], [0, 1]]

    labels = bic.segmentation.mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1
    )

    np.testing.assert_array_equal(labels, np.array([[1, 1]], dtype=np.uint64))


def test_mutex_watershed_2d_respects_grid_boundaries():
    affinities = np.ones((1, 2, 3), dtype=np.float32)
    offsets = [[0, 1]]

    labels = bic.segmentation.mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1
    )

    np.testing.assert_array_equal(
        labels, np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint64)
    )


def test_mutex_watershed_2d_accepts_negative_offsets():
    affinities = np.ones((1, 2, 3), dtype=np.float32)
    offsets = [[0, -1]]

    labels = bic.segmentation.mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1
    )

    np.testing.assert_array_equal(
        labels, np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint64)
    )


def test_mutex_watershed_3d_attractive_edges_merge_all_pixels():
    affinities = np.ones((3, 2, 2, 2), dtype=np.float32)
    offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    labels = bic.segmentation.mutex_watershed(
        affinities, offsets, number_of_attractive_channels=3
    )

    assert labels.shape == (2, 2, 2)
    np.testing.assert_array_equal(labels, np.ones((2, 2, 2), dtype=np.uint64))


def test_mutex_watershed_rejects_wrong_affinity_dtype():
    with pytest.raises(TypeError, match="affinities must have one of dtypes"):
        bic.segmentation.mutex_watershed(
            np.ones((1, 2, 2), dtype=np.uint8),
            [[0, 1]],
            number_of_attractive_channels=1,
        )


def test_mutex_watershed_rejects_wrong_offset_length():
    with pytest.raises(ValueError, match="each offset must have length"):
        bic.segmentation.mutex_watershed(
            np.ones((1, 2, 2), dtype=np.float32),
            [[0, 1, 0]],
            number_of_attractive_channels=1,
        )


def test_mutex_watershed_rejects_wrong_number_of_offsets():
    with pytest.raises(ValueError, match="offsets length must match"):
        bic.segmentation.mutex_watershed(
            np.ones((2, 2, 2), dtype=np.float32),
            [[0, 1]],
            number_of_attractive_channels=1,
        )
