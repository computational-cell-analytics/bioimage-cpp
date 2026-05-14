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


def test_mutex_watershed_strides_subsample_mutex_edges():
    affinities = np.zeros((2, 1, 5), dtype=np.float32)
    affinities[0, 0, :] = 0.8
    affinities[1, 0, :] = 0.9
    offsets = [[0, 1], [0, 1]]

    dense = bic.segmentation.mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1
    )
    strided = bic.segmentation.mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1, strides=[1, 2]
    )

    np.testing.assert_array_equal(dense, np.array([[1, 2, 3, 4, 5]], dtype=np.uint64))
    np.testing.assert_array_equal(strided, np.array([[1, 2, 2, 3, 3]], dtype=np.uint64))


def test_mutex_watershed_randomized_strides_use_numpy_random_state():
    affinities = np.zeros((2, 1, 5), dtype=np.float32)
    affinities[0, 0, :] = 0.8
    affinities[1, 0, :] = 0.9
    offsets = [[0, 1], [0, 1]]

    np.random.seed(17)
    first = bic.segmentation.mutex_watershed(
        affinities,
        offsets,
        number_of_attractive_channels=1,
        strides=[1, 2],
        randomized_strides=True,
    )
    np.random.seed(17)
    second = bic.segmentation.mutex_watershed(
        affinities,
        offsets,
        number_of_attractive_channels=1,
        strides=[1, 2],
        randomized_strides=True,
    )

    np.testing.assert_array_equal(first, second)


def test_mutex_watershed_mask_sets_background_zero_and_blocks_edges():
    affinities = np.ones((1, 1, 5), dtype=np.float32)
    offsets = [[0, 1]]
    mask = np.array([[True, True, False, True, True]], dtype=bool)

    labels = bic.segmentation.mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1, mask=mask
    )

    np.testing.assert_array_equal(labels, np.array([[1, 1, 0, 3, 3]], dtype=np.uint64))


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


def test_mutex_watershed_rejects_invalid_strides():
    with pytest.raises(ValueError, match="strides length must match"):
        bic.segmentation.mutex_watershed(
            np.ones((1, 2, 2), dtype=np.float32),
            [[0, 1]],
            number_of_attractive_channels=1,
            strides=[1, 1, 1],
        )


def test_mutex_watershed_rejects_invalid_mask():
    with pytest.raises(TypeError, match="mask must have dtype bool"):
        bic.segmentation.mutex_watershed(
            np.ones((1, 2, 2), dtype=np.float32),
            [[0, 1]],
            number_of_attractive_channels=1,
            mask=np.ones((2, 2), dtype=np.uint8),
        )
