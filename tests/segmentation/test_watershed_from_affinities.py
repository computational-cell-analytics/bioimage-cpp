import numpy as np
import pytest

import bioimage_cpp as bic


def _make_2d_ridge_neg():
    # Shape (2, 1, 5); negative-direction offsets [(-1, 0), (0, -1)].
    # Channel 1 (x-axis): aff[1, 0, x] = edge between (0, x-1) and (0, x).
    aff = np.zeros((2, 1, 5), dtype=np.float32)
    aff[1, 0, 1] = 0.9   # edge 0-1: strong
    aff[1, 0, 2] = 0.5   # edge 1-2
    aff[1, 0, 3] = 0.1   # edge 2-3: weak ridge
    aff[1, 0, 4] = 0.9   # edge 3-4: strong
    markers = np.array([[1, 0, 0, 0, 2]], dtype=np.uint32)
    return aff, markers


def _make_2d_ridge_pos():
    # Shape (2, 1, 5); positive-direction offsets [(1, 0), (0, 1)].
    # Channel 1 (x-axis): aff[1, 0, x] = edge between (0, x) and (0, x+1).
    aff = np.zeros((2, 1, 5), dtype=np.float32)
    aff[1, 0, 0] = 0.9   # edge 0-1
    aff[1, 0, 1] = 0.5   # edge 1-2
    aff[1, 0, 2] = 0.1   # edge 2-3
    aff[1, 0, 3] = 0.9   # edge 3-4
    markers = np.array([[1, 0, 0, 0, 2]], dtype=np.uint32)
    return aff, markers


def test_watershed_from_affinities_2d_negative_direction():
    aff, markers = _make_2d_ridge_neg()
    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0), (0, -1)], markers=markers,
    )
    np.testing.assert_array_equal(
        labels, np.array([[1, 1, 1, 2, 2]], dtype=np.uint32)
    )
    assert labels.dtype == np.uint32
    assert labels.shape == (1, 5)


def test_watershed_from_affinities_2d_positive_direction():
    aff, markers = _make_2d_ridge_pos()
    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(1, 0), (0, 1)], markers=markers,
    )
    np.testing.assert_array_equal(
        labels, np.array([[1, 1, 1, 2, 2]], dtype=np.uint32)
    )


def test_watershed_from_affinities_2d_permuted_offsets():
    # Same problem as the negative-direction ridge, but pass offsets in
    # reversed axis order. The function must rebuild the axis-channel
    # mapping; the result is unchanged.
    aff_yx = np.zeros((2, 1, 5), dtype=np.float32)
    aff_yx[1, 0, 1] = 0.9
    aff_yx[1, 0, 2] = 0.5
    aff_yx[1, 0, 3] = 0.1
    aff_yx[1, 0, 4] = 0.9
    markers = np.array([[1, 0, 0, 0, 2]], dtype=np.uint32)

    # Swap channels: channel 0 is now the x-axis edges, channel 1 is the
    # (unused) y-axis edges. Offsets describe the swap explicitly.
    aff_xy = aff_yx[[1, 0]].copy()
    labels = bic.segmentation.watershed_from_affinities(
        aff_xy, offsets=[(0, -1), (-1, 0)], markers=markers,
    )
    np.testing.assert_array_equal(
        labels, np.array([[1, 1, 1, 2, 2]], dtype=np.uint32)
    )


def test_watershed_from_affinities_3d_negative_direction():
    aff = np.zeros((3, 1, 1, 5), dtype=np.float32)
    aff[2, 0, 0, 1] = 0.9
    aff[2, 0, 0, 2] = 0.5
    aff[2, 0, 0, 3] = 0.1
    aff[2, 0, 0, 4] = 0.9
    markers = np.array([[[1, 0, 0, 0, 2]]], dtype=np.int64)

    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0, 0), (0, -1, 0), (0, 0, -1)], markers=markers,
    )
    np.testing.assert_array_equal(
        labels, np.array([[[1, 1, 1, 2, 2]]], dtype=np.int64)
    )
    assert labels.dtype == np.int64


def test_watershed_from_affinities_3d_positive_direction():
    aff = np.zeros((3, 1, 1, 5), dtype=np.float32)
    aff[2, 0, 0, 0] = 0.9
    aff[2, 0, 0, 1] = 0.5
    aff[2, 0, 0, 2] = 0.1
    aff[2, 0, 0, 3] = 0.9
    markers = np.array([[[1, 0, 0, 0, 2]]], dtype=np.uint64)

    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(1, 0, 0), (0, 1, 0), (0, 0, 1)], markers=markers,
    )
    np.testing.assert_array_equal(
        labels, np.array([[[1, 1, 1, 2, 2]]], dtype=np.uint64)
    )


def test_watershed_from_affinities_floods_uniform_region():
    # Single marker in a uniform-affinity volume should claim every pixel.
    aff = np.ones((2, 3, 3), dtype=np.float32)
    markers = np.zeros((3, 3), dtype=np.uint32)
    markers[0, 0] = 5

    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0), (0, -1)], markers=markers,
    )
    np.testing.assert_array_equal(labels, np.full((3, 3), 5, dtype=np.uint32))


def test_watershed_from_affinities_returns_zero_where_no_marker():
    aff = np.ones((2, 1, 3), dtype=np.float32)
    markers = np.zeros((1, 3), dtype=np.uint32)

    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0), (0, -1)], markers=markers,
    )
    np.testing.assert_array_equal(labels, np.zeros((1, 3), dtype=np.uint32))


def test_watershed_from_affinities_mask_excludes_pixels():
    aff = np.ones((2, 1, 5), dtype=np.float32)
    markers = np.array([[1, 0, 0, 0, 2]], dtype=np.uint64)
    mask = np.array([[True, True, False, True, True]])

    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0), (0, -1)], markers=markers, mask=mask,
    )
    np.testing.assert_array_equal(
        labels, np.array([[1, 1, 0, 2, 2]], dtype=np.uint64)
    )


def test_watershed_from_affinities_marker_under_masked_pixel_is_ignored():
    aff = np.ones((2, 1, 3), dtype=np.float32)
    markers = np.array([[0, 7, 0]], dtype=np.int32)
    mask = np.array([[True, False, True]])

    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0), (0, -1)], markers=markers, mask=mask,
    )
    np.testing.assert_array_equal(labels, np.zeros((1, 3), dtype=np.int32))


@pytest.mark.parametrize("aff_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("marker_dtype", [np.uint32, np.uint64, np.int32, np.int64])
def test_watershed_from_affinities_dtype_matrix(aff_dtype, marker_dtype):
    aff = np.zeros((2, 1, 5), dtype=aff_dtype)
    aff[1, 0, 1] = 0.9
    aff[1, 0, 2] = 0.5
    aff[1, 0, 3] = 0.1
    aff[1, 0, 4] = 0.9
    markers = np.array([[1, 0, 0, 0, 2]], dtype=marker_dtype)

    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0), (0, -1)], markers=markers,
    )
    assert labels.dtype == np.dtype(marker_dtype)
    assert labels.shape == (1, 5)
    np.testing.assert_array_equal(
        labels, np.array([[1, 1, 1, 2, 2]], dtype=marker_dtype)
    )


def test_watershed_from_affinities_is_deterministic():
    rng = np.random.default_rng(0)
    aff = rng.random((2, 8, 10), dtype=np.float32)
    markers = np.zeros((8, 10), dtype=np.uint32)
    markers[0, 0] = 1
    markers[7, 9] = 2

    first = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0), (0, -1)], markers=markers,
    )
    second = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0), (0, -1)], markers=markers,
    )
    np.testing.assert_array_equal(first, second)


def test_watershed_from_affinities_rejects_wrong_ndim():
    aff = np.ones((5,), dtype=np.float32)
    markers = np.zeros((5,), dtype=np.uint32)
    with pytest.raises(ValueError, match="ndim"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-1,)], markers=markers,
        )


def test_watershed_from_affinities_rejects_channel_mismatch():
    aff = np.ones((3, 4, 4), dtype=np.float32)  # 3 channels for 2D spatial
    markers = np.zeros((4, 4), dtype=np.uint32)
    with pytest.raises(ValueError, match="channel count"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-1, 0), (0, -1), (-1, 0)], markers=markers,
        )


def test_watershed_from_affinities_rejects_non_nearest_neighbour_offset():
    aff = np.ones((2, 4, 4), dtype=np.float32)
    markers = np.zeros((4, 4), dtype=np.uint32)
    with pytest.raises(ValueError, match="nearest-neighbour"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-2, 0), (0, -1)], markers=markers,
        )


def test_watershed_from_affinities_rejects_mixed_signs():
    aff = np.ones((2, 4, 4), dtype=np.float32)
    markers = np.zeros((4, 4), dtype=np.uint32)
    with pytest.raises(ValueError, match="same sign"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-1, 0), (0, 1)], markers=markers,
        )


def test_watershed_from_affinities_rejects_duplicate_axis():
    aff = np.ones((2, 4, 4), dtype=np.float32)
    markers = np.zeros((4, 4), dtype=np.uint32)
    with pytest.raises(ValueError, match="more than once"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-1, 0), (-1, 0)], markers=markers,
        )


def test_watershed_from_affinities_rejects_markers_shape_mismatch():
    aff = np.ones((2, 4, 4), dtype=np.float32)
    markers = np.zeros((4, 5), dtype=np.uint32)
    with pytest.raises(ValueError, match="markers shape"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-1, 0), (0, -1)], markers=markers,
        )


def test_watershed_from_affinities_rejects_mask_shape_mismatch():
    aff = np.ones((2, 4, 4), dtype=np.float32)
    markers = np.zeros((4, 4), dtype=np.uint32)
    mask = np.ones((4, 5), dtype=bool)
    with pytest.raises(ValueError, match="mask shape"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-1, 0), (0, -1)], markers=markers, mask=mask,
        )


def test_watershed_from_affinities_rejects_non_bool_mask():
    aff = np.ones((2, 4, 4), dtype=np.float32)
    markers = np.zeros((4, 4), dtype=np.uint32)
    mask = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(TypeError, match="mask must have dtype bool"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-1, 0), (0, -1)], markers=markers, mask=mask,
        )


def test_watershed_from_affinities_rejects_unsupported_aff_dtype():
    aff = np.ones((2, 4, 4), dtype=np.int16)
    markers = np.zeros((4, 4), dtype=np.uint32)
    with pytest.raises(TypeError, match="affinities must have one of dtypes"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-1, 0), (0, -1)], markers=markers,
        )


def test_watershed_from_affinities_rejects_unsupported_marker_dtype():
    aff = np.ones((2, 4, 4), dtype=np.float32)
    markers = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(TypeError, match="markers must have one of dtypes"):
        bic.segmentation.watershed_from_affinities(
            aff, offsets=[(-1, 0), (0, -1)], markers=markers,
        )


def test_watershed_from_affinities_full_coverage_on_random_input():
    # Sanity invariant on random input: every pixel ends up with one of the
    # marker labels (no zeros), and the set of output labels is exactly the
    # marker label set. (Exact agreement with the node-based watershed is
    # not expected — the algorithms have different priority semantics.)
    rng = np.random.default_rng(42)
    aff = rng.random((2, 32, 32), dtype=np.float32)
    markers = np.zeros((32, 32), dtype=np.uint32)
    markers[0, 0] = 1
    markers[31, 31] = 2
    markers[0, 31] = 3
    markers[31, 0] = 4

    labels = bic.segmentation.watershed_from_affinities(
        aff, offsets=[(-1, 0), (0, -1)], markers=markers,
    )
    assert set(np.unique(labels).tolist()) == {1, 2, 3, 4}
    assert (labels > 0).all()
