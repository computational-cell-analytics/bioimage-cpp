import numpy as np
import pytest

import bioimage_cpp as bic


def _numpy_reference(values, offsets, norm):
    """Slow but obvious reference: nested Python loops over voxels."""
    values = np.asarray(values, dtype=np.float64)
    spatial_shape = values.shape[1:]
    n = len(offsets)
    dist = np.zeros((n,) + spatial_shape, dtype=np.float32)
    for oi, offset in enumerate(offsets):
        for coord in np.ndindex(*spatial_shape):
            neighbor = tuple(c + d for c, d in zip(coord, offset))
            if any(nn < 0 or nn >= s for nn, s in zip(neighbor, spatial_shape)):
                continue
            a = values[(slice(None),) + coord]
            b = values[(slice(None),) + neighbor]
            if norm == "l1":
                v = np.sum(np.abs(a - b))
            elif norm == "l2":
                v = np.sqrt(np.sum((a - b) ** 2))
            elif norm == "cosine":
                v = 1.0 - np.dot(a, b) / (
                    np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
                )
            else:
                raise AssertionError(norm)
            dist[(oi,) + coord] = np.float32(v)
    return dist


@pytest.mark.parametrize("norm", ["l1", "l2", "cosine"])
def test_2d_matches_numpy_reference(norm):
    rng = np.random.default_rng(0)
    values = rng.standard_normal(size=(4, 7, 11)).astype(np.float32)
    offsets = [[0, 1], [1, 0], [1, 1], [2, -3], [-1, 2]]

    dist = bic.affinities.compute_embedding_distances(values, offsets, norm=norm)
    ref = _numpy_reference(values, offsets, norm)

    assert dist.dtype == np.float32
    assert dist.shape == (len(offsets), *values.shape[1:])
    np.testing.assert_allclose(dist, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("norm", ["l1", "l2", "cosine"])
def test_3d_matches_numpy_reference(norm):
    rng = np.random.default_rng(1)
    values = rng.standard_normal(size=(3, 4, 5, 6)).astype(np.float32)
    offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [-2, 0, 3]]

    dist = bic.affinities.compute_embedding_distances(values, offsets, norm=norm)
    ref = _numpy_reference(values, offsets, norm)

    assert dist.shape == (len(offsets), *values.shape[1:])
    np.testing.assert_allclose(dist, ref, atol=1e-5, rtol=1e-5)


def test_offset_completely_out_of_bounds_yields_zero():
    values = np.ones((2, 3, 3), dtype=np.float32)
    offsets = [[10, 0]]
    dist = bic.affinities.compute_embedding_distances(values, offsets, norm="l2")
    assert dist.shape == (1, 3, 3)
    assert np.all(dist == 0.0)


def test_threading_does_not_change_output():
    rng = np.random.default_rng(42)
    values = rng.standard_normal(size=(5, 8, 12)).astype(np.float32)
    offsets = [[0, 1], [1, 0], [1, 1], [2, 3], [-1, 2]]

    for norm in ("l1", "l2", "cosine"):
        single = bic.affinities.compute_embedding_distances(
            values, offsets, norm=norm, number_of_threads=1
        )
        multi = bic.affinities.compute_embedding_distances(
            values, offsets, norm=norm, number_of_threads=4
        )
        np.testing.assert_array_equal(single, multi)


def test_non_contiguous_input_is_handled():
    rng = np.random.default_rng(7)
    values = rng.standard_normal(size=(3, 5, 6)).astype(np.float32)
    # Swap the last two axes to produce a non-contiguous view; ascontiguousarray
    # in the wrapper should copy it back.
    values_view = np.swapaxes(values, 1, 2)
    assert not values_view.flags["C_CONTIGUOUS"]

    dist = bic.affinities.compute_embedding_distances(
        values_view, [[0, 1]], norm="l2"
    )
    ref = bic.affinities.compute_embedding_distances(
        np.ascontiguousarray(values_view), [[0, 1]], norm="l2"
    )
    np.testing.assert_array_equal(dist, ref)


def test_default_norm_is_l2():
    rng = np.random.default_rng(99)
    values = rng.standard_normal(size=(3, 4, 4)).astype(np.float32)
    offsets = [[0, 1], [1, 0]]
    default_dist = bic.affinities.compute_embedding_distances(values, offsets)
    explicit_l2 = bic.affinities.compute_embedding_distances(
        values, offsets, norm="l2"
    )
    np.testing.assert_array_equal(default_dist, explicit_l2)


def test_rejects_unsupported_dtype():
    values = np.zeros((2, 4, 4), dtype=np.float64)
    with pytest.raises(TypeError, match="float32"):
        bic.affinities.compute_embedding_distances(values, [[0, 1]])


def test_rejects_input_without_channel_axis():
    values = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="ndim"):
        bic.affinities.compute_embedding_distances(values, [[0, 1]])


def test_rejects_empty_offsets():
    values = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="offsets"):
        bic.affinities.compute_embedding_distances(values, [])


def test_rejects_offset_with_wrong_length():
    values = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="spatial ndim"):
        bic.affinities.compute_embedding_distances(values, [[0, 1, 2]])


def test_rejects_unknown_norm():
    values = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="norm"):
        bic.affinities.compute_embedding_distances(values, [[0, 1]], norm="linf")
