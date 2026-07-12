"""Tests for bioimage_cpp.distance.non_maximum_distance_suppression."""

import numpy as np
import pytest

import bioimage_cpp as bic

nms = bic.distance.non_maximum_distance_suppression


def test_empty_points_returns_empty():
    dm = np.ones((10, 10), dtype=np.float32)
    out = nms(dm, np.zeros((0, 2), dtype=np.int64))
    assert out.shape == (0, 2)
    assert out.dtype == np.int64


def test_single_point_returns_itself():
    dm = np.zeros((11, 11), dtype=np.float32)
    dm[5, 5] = 3.0
    out = nms(dm, np.array([[5, 5]], dtype=np.int64))
    assert out.tolist() == [[5, 5]]


def test_two_close_points_keeps_higher_value():
    # Both points sit within each other's dynamic neighborhood; only the
    # one with the larger distance value survives.
    dm = np.zeros((11, 11), dtype=np.float32)
    dm[5, 5] = 5.0
    dm[5, 6] = 4.0
    pts = np.array([[5, 5], [5, 6]], dtype=np.int64)
    out = nms(dm, pts)
    assert out.tolist() == [[5, 5]]


def test_out_of_range_point_coordinate_raises():
    dm = np.zeros((10, 10), dtype=np.float32)
    pts = np.array([[5, 5], [5, 12]], dtype=np.int64)
    with pytest.raises(ValueError, match="out of bounds"):
        nms(dm, pts)


def test_negative_point_coordinate_raises():
    dm = np.zeros((10, 10), dtype=np.float32)
    pts = np.array([[-1, 5]], dtype=np.int64)
    with pytest.raises(ValueError, match="out of bounds"):
        nms(dm, pts)


def test_two_far_points_both_survive():
    dm = np.zeros((20, 20), dtype=np.float32)
    dm[2, 2] = 1.0
    dm[15, 15] = 1.0
    pts = np.array([[2, 2], [15, 15]], dtype=np.int64)
    out = nms(dm, pts)
    # Far apart relative to their radius of 1.0 -> both kept, original order.
    assert out.tolist() == [[2, 2], [15, 15]]


def test_zero_radius_point_keeps_itself():
    # A point whose distance value is 0 has an empty neighborhood except for
    # itself, so it is always retained.
    dm = np.zeros((10, 10), dtype=np.float32)
    dm[3, 3] = 4.0  # high-value neighbor nearby
    pts = np.array([[3, 4], [3, 3]], dtype=np.int64)  # first has value 0
    out = nms(dm, pts)
    out_set = {tuple(row) for row in out.tolist()}
    assert (3, 4) in out_set  # kept because its own radius is 0
    assert (3, 3) in out_set  # the dominant peak


@pytest.mark.parametrize("shape", [(20, 20), (10, 12, 14)])
def test_subset_and_includes_global_max(shape):
    scipy_ndi = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(0)
    mask = rng.random(shape) > 0.2
    dm = scipy_ndi.distance_transform_edt(mask).astype(np.float32)
    coords = np.argwhere(dm > 1.5).astype(np.int64)
    if len(coords) == 0:
        pytest.skip("no candidate points for this random mask")

    out = nms(dm, coords)
    assert out.ndim == 2
    assert out.shape[1] == len(shape)
    assert out.shape[0] <= coords.shape[0]

    # Every output point must be one of the input points.
    in_set = {tuple(row) for row in coords.tolist()}
    for row in out.tolist():
        assert tuple(row) in in_set

    # The global maximum of the distance map is always its own best point.
    gmax = np.unravel_index(int(np.argmax(dm)), dm.shape)
    if dm[gmax] > 1.5:
        assert list(gmax) in out.tolist()


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.int32, np.uint32])
def test_dtype_dispatch_equivalent(dtype):
    dm = np.zeros((12, 12), dtype=np.float32)
    dm[3, 3] = 5.0
    dm[3, 4] = 4.0
    dm[9, 9] = 2.0
    pts = np.array([[3, 3], [3, 4], [9, 9]], dtype=dtype)
    out = nms(dm, pts)
    assert out.dtype == np.dtype(dtype)
    # (3,3) suppresses (3,4); (9,9) is far and survives.
    assert out.tolist() == [[3, 3], [9, 9]]


def test_distance_map_float64_is_coerced():
    dm = np.zeros((11, 11), dtype=np.float64)
    dm[5, 5] = 5.0
    dm[5, 6] = 4.0
    pts = np.array([[5, 5], [5, 6]], dtype=np.int64)
    out = nms(dm, pts)
    assert out.tolist() == [[5, 5]]


def test_deterministic():
    scipy_ndi = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(7)
    mask = rng.random((40, 40)) > 0.25
    dm = scipy_ndi.distance_transform_edt(mask).astype(np.float32)
    coords = np.argwhere(dm > 1.0).astype(np.int64)
    a = nms(dm, coords)
    b = nms(dm, coords)
    assert np.array_equal(a, b)


def test_thread_counts_are_identical():
    rng = np.random.default_rng(17)
    dm = rng.uniform(0, 5, size=(40, 40)).astype(np.float32)
    points = rng.integers(0, 40, size=(300, 2), dtype=np.int64)
    single = nms(dm, points, number_of_threads=1)
    multi = nms(dm, points, number_of_threads=4)
    np.testing.assert_array_equal(single, multi)


def test_invalid_points_ndim():
    dm = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        nms(dm, np.array([1, 2, 3], dtype=np.int64))


def test_invalid_points_axis_length():
    dm = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        nms(dm, np.array([[1, 2, 3]], dtype=np.int64))


def test_invalid_dtype():
    dm = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(TypeError):
        nms(dm, np.array([[1.0, 2.0]], dtype=np.float32))


def test_out_of_bounds_coordinate_raises():
    dm = np.ones((10, 10), dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError)):
        nms(dm, np.array([[10, 0]], dtype=np.int64))


def test_negative_coordinate_raises():
    dm = np.ones((10, 10), dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError)):
        nms(dm, np.array([[-1, 0]], dtype=np.int64))
