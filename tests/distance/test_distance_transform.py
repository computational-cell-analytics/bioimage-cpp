import numpy as np
import pytest
from scipy import ndimage

import bioimage_cpp as bic


def test_distance_transform_1d_matches_scipy():
    data = np.array([0, 1, 1, 0, 1, 1, 1], dtype=np.uint8)

    got = bic.distance.distance_transform(data)
    ref = ndimage.distance_transform_edt(data).astype(np.float32)

    assert got.dtype == np.float32
    np.testing.assert_allclose(got, ref)


@pytest.mark.parametrize("shape", [(7, 11), (4, 5, 6)])
def test_distance_transform_matches_scipy(shape):
    data = np.ones(shape, dtype=np.uint8)
    center = tuple(axis_size // 2 for axis_size in shape)
    data[center] = 0

    got = bic.distance.distance_transform(data)
    ref = ndimage.distance_transform_edt(data).astype(np.float32)

    assert got.shape == data.shape
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_anisotropic_sampling_matches_scipy():
    data = np.ones((5, 6, 7), dtype=bool)
    data[1, 2, 3] = False
    sampling = (2.0, 0.5, 3.0)

    got = bic.distance.distance_transform(data, sampling=sampling)
    ref = ndimage.distance_transform_edt(data, sampling=sampling).astype(np.float32)

    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_return_indices_matches_scipy_for_unique_nearest_background():
    data = np.ones((5, 6), dtype=np.uint8)
    data[2, 3] = 0

    got_dist, got_idx = bic.distance.distance_transform(data, return_indices=True)
    ref_dist, ref_idx = ndimage.distance_transform_edt(data, return_indices=True)

    assert got_dist.dtype == np.float32
    assert got_idx.dtype == np.int32
    assert got_idx.shape == (2,) + data.shape
    np.testing.assert_allclose(got_dist, ref_dist.astype(np.float32), atol=1e-6)
    np.testing.assert_array_equal(got_idx, ref_idx)


def test_return_indices_only():
    data = np.ones((4, 5), dtype=np.uint8)
    data[1, 2] = 0

    got = bic.distance.distance_transform(
        data, return_distances=False, return_indices=True
    )
    ref = ndimage.distance_transform_edt(
        data, return_distances=False, return_indices=True
    )

    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, ref)


def test_preallocated_outputs_are_filled_and_return_none():
    data = np.ones((4, 5), dtype=np.uint8)
    data[0, 0] = 0
    distances = np.empty(data.shape, dtype=np.float32)
    indices = np.empty((2,) + data.shape, dtype=np.int32)

    result = bic.distance.distance_transform(
        data,
        return_indices=True,
        distances=distances,
        indices=indices,
    )
    ref_dist, ref_idx = ndimage.distance_transform_edt(data, return_indices=True)

    assert result is None
    np.testing.assert_allclose(distances, ref_dist.astype(np.float32), atol=1e-6)
    np.testing.assert_array_equal(indices, ref_idx)


def test_preallocated_distance_returns_requested_indices():
    data = np.ones((4, 5), dtype=np.uint8)
    data[0, 0] = 0
    distances = np.empty(data.shape, dtype=np.float32)

    got_indices = bic.distance.distance_transform(
        data,
        return_indices=True,
        distances=distances,
    )
    ref_dist, ref_idx = ndimage.distance_transform_edt(data, return_indices=True)

    np.testing.assert_allclose(distances, ref_dist.astype(np.float32), atol=1e-6)
    np.testing.assert_array_equal(got_indices, ref_idx)


def test_all_background_and_all_foreground_match_scipy():
    for data in [
        np.zeros((3, 4), dtype=np.uint8),
        np.ones((3, 4), dtype=np.uint8),
    ]:
        got_dist, got_idx = bic.distance.distance_transform(data, return_indices=True)
        ref_dist, ref_idx = ndimage.distance_transform_edt(data, return_indices=True)
        np.testing.assert_allclose(got_dist, ref_dist.astype(np.float32), atol=1e-6)
        np.testing.assert_array_equal(got_idx, ref_idx)


def test_empty_input_matches_scipy():
    data = np.zeros((0, 3), dtype=np.uint8)

    got_dist, got_idx = bic.distance.distance_transform(data, return_indices=True)
    ref_dist, ref_idx = ndimage.distance_transform_edt(data, return_indices=True)

    assert got_dist.shape == ref_dist.shape
    assert got_idx.shape == ref_idx.shape
    assert got_dist.dtype == np.float32
    assert got_idx.dtype == np.int32


def test_non_contiguous_input_is_handled():
    data = np.ones((8, 10), dtype=np.uint8)
    data[2, 4] = 0
    sliced = data[::2, ::2]

    got = bic.distance.distance_transform(sliced)
    ref = ndimage.distance_transform_edt(sliced).astype(np.float32)

    np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("sampling", [(1.0, 1.0, 1.0), (2.5, 1.25, 0.75)])
def test_threaded_outputs_are_exact(sampling):
    zz, yy, xx = np.indices((17, 19, 23))
    data = ((3 * zz + 5 * yy + 7 * xx) % 13 != 0).astype(np.uint8)
    sequential = bic.distance.distance_transform(
        data,
        sampling=sampling,
        return_indices=True,
        return_vectors=True,
        number_of_threads=1,
    )
    threaded = bic.distance.distance_transform(
        data,
        sampling=sampling,
        return_indices=True,
        return_vectors=True,
        number_of_threads=4,
    )
    for got, expected in zip(threaded, sequential):
        np.testing.assert_array_equal(got, expected)

    preallocated = [np.empty_like(expected) for expected in sequential]
    result = bic.distance.distance_transform(
        data,
        sampling=sampling,
        return_indices=True,
        return_vectors=True,
        distances=preallocated[0],
        indices=preallocated[1],
        vectors=preallocated[2],
        number_of_threads=4,
    )
    assert result is None
    for got, expected in zip(preallocated, sequential):
        np.testing.assert_array_equal(got, expected)


def test_vector_difference_transform_unique_target():
    data = np.ones((5, 6), dtype=np.uint8)
    data[2, 3] = 0
    sampling = (2.0, 0.5)

    vectors = bic.distance.vector_difference_transform(data, sampling=sampling)

    assert vectors.shape == data.shape + (2,)
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[2, 3], np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(vectors[0, 0], np.array([4.0, 1.5], dtype=np.float32))
    distances = bic.distance.distance_transform(data, sampling=sampling)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=-1), distances, atol=1e-6)


def test_vector_difference_transform_3d_unique_target():
    data = np.ones((3, 4, 5), dtype=np.uint8)
    data[1, 2, 3] = 0

    vectors = bic.distance.vector_difference_transform(data)

    assert vectors.shape == data.shape + (3,)
    np.testing.assert_array_equal(vectors[0, 0, 0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    distances = bic.distance.distance_transform(data)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=-1), distances, atol=1e-6)


def test_vector_difference_transform_all_foreground_uses_scipy_virtual_background():
    data = np.ones((2, 3), dtype=np.uint8)

    vectors = bic.distance.vector_difference_transform(data)
    expected = np.array(
        [
            [[-1.0, 0.0], [-1.0, -1.0], [-1.0, -2.0]],
            [[-2.0, 0.0], [-2.0, -1.0], [-2.0, -2.0]],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(vectors, expected)
    distances = bic.distance.distance_transform(data)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=-1), distances, atol=1e-6)


def test_invalid_arguments_raise():
    data = np.zeros((3, 4), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="return_distances"):
        bic.distance.distance_transform(data, return_distances=False, return_indices=False)
    with pytest.raises(ValueError, match="sampling"):
        bic.distance.distance_transform(data, sampling=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="positive"):
        bic.distance.distance_transform(data, sampling=(1.0, 0.0))
    with pytest.raises(TypeError, match="float32"):
        bic.distance.distance_transform(data, distances=np.empty(data.shape, dtype=np.float64))
    with pytest.raises(TypeError, match="int32"):
        bic.distance.distance_transform(
            data,
            return_indices=True,
            indices=np.empty((2,) + data.shape, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="shape"):
        bic.distance.distance_transform(data, distances=np.empty((3, 5), dtype=np.float32))
