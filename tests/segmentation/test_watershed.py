import numpy as np
import pytest

import bioimage_cpp as bic


def test_watershed_2d_ridge_assigns_each_side_to_its_marker():
    # Strictly distinct heights on each side of the ridge make the result
    # tie-break-free.
    image = np.array([[0.0, 2.0, 9.0, 1.0, 0.0]], dtype=np.float32)
    markers = np.array([[1, 0, 0, 0, 2]], dtype=np.uint32)

    labels = bic.segmentation.watershed(image, markers)

    np.testing.assert_array_equal(labels, np.array([[1, 1, 2, 2, 2]], dtype=np.uint32))
    assert labels.dtype == np.uint32
    assert labels.shape == image.shape


def test_watershed_3d_ridge_assigns_each_side_to_its_marker():
    # 3D version of the 2D ridge test — exercises the 3D neighbour code path
    # while keeping the answer tie-break-free.
    image = np.array([[[0.0, 2.0, 9.0, 1.0, 0.0]]], dtype=np.float64)
    markers = np.array([[[1, 0, 0, 0, 2]]], dtype=np.int64)

    labels = bic.segmentation.watershed(image, markers)

    np.testing.assert_array_equal(
        labels, np.array([[[1, 1, 2, 2, 2]]], dtype=np.int64)
    )
    assert labels.dtype == np.int64
    assert labels.shape == (1, 1, 5)


def test_watershed_3d_exercises_all_axis_neighbours():
    # A single marker at the origin of a (2,2,2) volume should reach every
    # cell when the heightmap is uniform — every axis-aligned neighbour is
    # visited.
    image = np.zeros((2, 2, 2), dtype=np.float32)
    markers = np.zeros((2, 2, 2), dtype=np.uint32)
    markers[0, 0, 0] = 7

    labels = bic.segmentation.watershed(image, markers)

    np.testing.assert_array_equal(labels, np.full((2, 2, 2), 7, dtype=np.uint32))


def test_watershed_returns_zero_where_no_marker_reaches():
    image = np.zeros((1, 3), dtype=np.float32)
    markers = np.zeros((1, 3), dtype=np.uint32)

    labels = bic.segmentation.watershed(image, markers)

    np.testing.assert_array_equal(labels, np.zeros((1, 3), dtype=np.uint32))


def test_watershed_mask_excludes_pixels_from_flooding():
    image = np.zeros((1, 5), dtype=np.float32)
    markers = np.array([[1, 0, 0, 0, 2]], dtype=np.uint64)
    mask = np.array([[True, True, False, True, True]])

    labels = bic.segmentation.watershed(image, markers, mask=mask)

    # The middle column is masked out and stays 0 — it also separates the two
    # flood fronts so each marker stays on its own side.
    np.testing.assert_array_equal(
        labels, np.array([[1, 1, 0, 2, 2]], dtype=np.uint64)
    )


def test_watershed_marker_under_masked_pixel_is_ignored():
    image = np.zeros((1, 3), dtype=np.float32)
    markers = np.array([[0, 7, 0]], dtype=np.int32)
    mask = np.array([[True, False, True]])

    labels = bic.segmentation.watershed(image, markers, mask=mask)

    # The only marker sits under a False mask pixel, so nothing floods.
    np.testing.assert_array_equal(labels, np.zeros((1, 3), dtype=np.int32))


@pytest.mark.parametrize("image_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("marker_dtype", [np.uint32, np.uint64, np.int32, np.int64])
def test_watershed_dtype_matrix(image_dtype, marker_dtype):
    image = np.array([[0.0, 2.0, 9.0, 1.0, 0.0]], dtype=image_dtype)
    markers = np.array([[1, 0, 0, 0, 2]], dtype=marker_dtype)

    labels = bic.segmentation.watershed(image, markers)

    assert labels.dtype == np.dtype(marker_dtype)
    assert labels.shape == image.shape
    np.testing.assert_array_equal(
        labels, np.array([[1, 1, 2, 2, 2]], dtype=marker_dtype)
    )


def test_watershed_is_deterministic():
    rng = np.random.default_rng(0)
    image = rng.random((10, 12), dtype=np.float32)
    markers = np.zeros((10, 12), dtype=np.uint32)
    markers[0, 0] = 1
    markers[9, 11] = 2

    first = bic.segmentation.watershed(image, markers)
    second = bic.segmentation.watershed(image, markers)

    np.testing.assert_array_equal(first, second)


def test_watershed_rejects_1d_image():
    image = np.zeros(5, dtype=np.float32)
    markers = np.zeros(5, dtype=np.uint32)

    with pytest.raises(ValueError, match="ndim"):
        bic.segmentation.watershed(image, markers)


def test_watershed_rejects_mismatched_markers_shape():
    image = np.zeros((3, 3), dtype=np.float32)
    markers = np.zeros((3, 4), dtype=np.uint32)

    with pytest.raises(ValueError, match="markers shape"):
        bic.segmentation.watershed(image, markers)


def test_watershed_rejects_mismatched_mask_shape():
    image = np.zeros((3, 3), dtype=np.float32)
    markers = np.zeros((3, 3), dtype=np.uint32)
    mask = np.ones((3, 4), dtype=bool)

    with pytest.raises(ValueError, match="mask shape"):
        bic.segmentation.watershed(image, markers, mask=mask)


def test_watershed_rejects_non_bool_mask():
    image = np.zeros((3, 3), dtype=np.float32)
    markers = np.zeros((3, 3), dtype=np.uint32)
    mask = np.ones((3, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="mask must have dtype bool"):
        bic.segmentation.watershed(image, markers, mask=mask)


def test_watershed_rejects_unsupported_image_dtype():
    image = np.zeros((3, 3), dtype=np.int16)
    markers = np.zeros((3, 3), dtype=np.uint32)

    with pytest.raises(TypeError, match="image must have one of dtypes"):
        bic.segmentation.watershed(image, markers)


def test_watershed_rejects_unsupported_markers_dtype():
    image = np.zeros((3, 3), dtype=np.float32)
    markers = np.zeros((3, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="markers must have one of dtypes"):
        bic.segmentation.watershed(image, markers)
