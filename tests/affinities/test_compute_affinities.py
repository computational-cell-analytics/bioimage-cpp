import numpy as np
import pytest

import bioimage_cpp as bic


def test_minimum_int64_offset_is_safely_out_of_bounds():
    labels = np.ones((2, 2), dtype=np.uint32)
    affinities, valid = bic.affinities.compute_affinities(
        labels, [[np.iinfo(np.int64).min, 0]], number_of_threads=1
    )
    assert not affinities.any()
    assert not valid.any()


def test_fractional_offsets_are_rejected():
    with pytest.raises(TypeError, match="integers"):
        bic.affinities.compute_affinities(
            np.ones((2, 2), dtype=np.uint32), [[0.5, 0]]
        )


def _numpy_reference(labels, offsets, ignore_label=None):
    """Slow but obvious reference: nested Python loops over voxels."""
    labels = np.asarray(labels)
    n = len(offsets)
    affs = np.zeros((n,) + labels.shape, dtype=np.float32)
    mask = np.zeros((n,) + labels.shape, dtype=np.uint8)
    for oi, offset in enumerate(offsets):
        it = np.ndindex(*labels.shape)
        for coord in it:
            neighbor = tuple(c + d for c, d in zip(coord, offset))
            if any(n < 0 or n >= s for n, s in zip(neighbor, labels.shape)):
                continue
            a = labels[coord]
            b = labels[neighbor]
            if ignore_label is not None and (a == ignore_label or b == ignore_label):
                continue
            affs[(oi, *coord)] = 1.0 if a == b else 0.0
            mask[(oi, *coord)] = 1
    return affs, mask


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64, np.int32, np.int64])
def test_2d_matches_numpy_reference(dtype):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=(7, 11)).astype(dtype)
    offsets = [[0, 1], [1, 0], [1, 1], [2, -3], [-1, 2]]

    affs, mask = bic.affinities.compute_affinities(labels, offsets)
    ref_affs, ref_mask = _numpy_reference(labels, offsets)

    assert affs.dtype == np.float32
    assert mask.dtype == np.uint8
    assert affs.shape == (len(offsets), *labels.shape)
    np.testing.assert_array_equal(affs, ref_affs)
    np.testing.assert_array_equal(mask, ref_mask)


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64, np.int32, np.int64])
def test_3d_matches_numpy_reference(dtype):
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, size=(4, 5, 6)).astype(dtype)
    offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [-2, 0, 3]]

    affs, mask = bic.affinities.compute_affinities(labels, offsets)
    ref_affs, ref_mask = _numpy_reference(labels, offsets)

    np.testing.assert_array_equal(affs, ref_affs)
    np.testing.assert_array_equal(mask, ref_mask)


def test_ignore_label_masks_pairs_at_ignored_voxels():
    labels = np.array(
        [
            [0, 1, 1, 2],
            [0, 1, 2, 2],
        ],
        dtype=np.uint32,
    )
    offsets = [[0, 1]]

    affs, mask = bic.affinities.compute_affinities(
        labels, offsets, ignore_label=0
    )
    ref_affs, ref_mask = _numpy_reference(labels, offsets, ignore_label=0)
    np.testing.assert_array_equal(affs, ref_affs)
    np.testing.assert_array_equal(mask, ref_mask)
    assert mask[0, 0, 0] == 0  # voxel has ignore label
    assert affs[0, 0, 0] == 0.0


def test_offset_completely_out_of_bounds_yields_zero_mask():
    labels = np.ones((3, 3), dtype=np.uint32)
    offsets = [[10, 0]]  # never in bounds
    affs, mask = bic.affinities.compute_affinities(labels, offsets)
    assert affs.sum() == 0
    assert mask.sum() == 0


def test_return_mask_false_returns_only_affinities():
    labels = np.array([[0, 0], [1, 1]], dtype=np.uint32)
    offsets = [[0, 1], [1, 0]]

    affs = bic.affinities.compute_affinities(labels, offsets, return_mask=False)
    assert isinstance(affs, np.ndarray)
    assert affs.shape == (2, 2, 2)
    assert affs.dtype == np.float32


def test_negative_offsets_work():
    labels = np.array([[1, 2], [1, 2]], dtype=np.uint32)
    offsets = [[0, -1], [-1, 0]]
    affs, mask = bic.affinities.compute_affinities(labels, offsets)
    ref_affs, ref_mask = _numpy_reference(labels, offsets)
    np.testing.assert_array_equal(affs, ref_affs)
    np.testing.assert_array_equal(mask, ref_mask)


def test_threading_does_not_change_output():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 7, size=(8, 12)).astype(np.uint32)
    offsets = [[0, 1], [1, 0], [1, 1], [2, 3], [-1, 2]]

    affs_single, mask_single = bic.affinities.compute_affinities(
        labels, offsets, number_of_threads=1
    )
    affs_multi, mask_multi = bic.affinities.compute_affinities(
        labels, offsets, number_of_threads=4
    )
    np.testing.assert_array_equal(affs_single, affs_multi)
    np.testing.assert_array_equal(mask_single, mask_multi)


def test_non_contiguous_input_is_handled():
    labels = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.uint32)
    # Transpose to produce a non-contiguous view.
    labels_T = labels.T
    assert not labels_T.flags["C_CONTIGUOUS"]
    affs, mask = bic.affinities.compute_affinities(labels_T, [[0, 1]])
    # Should give same result as feeding a contiguous copy.
    ref_affs, ref_mask = bic.affinities.compute_affinities(
        np.ascontiguousarray(labels_T), [[0, 1]]
    )
    np.testing.assert_array_equal(affs, ref_affs)
    np.testing.assert_array_equal(mask, ref_mask)


def test_rejects_unsupported_dtype():
    labels = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="dtypes"):
        bic.affinities.compute_affinities(labels, [[0, 1]])


def test_rejects_1d_input():
    labels = np.zeros(8, dtype=np.uint32)
    with pytest.raises(ValueError, match="2D or 3D"):
        bic.affinities.compute_affinities(labels, [[1]])


def test_rejects_empty_offsets():
    labels = np.zeros((4, 4), dtype=np.uint32)
    with pytest.raises(ValueError, match="offsets"):
        bic.affinities.compute_affinities(labels, [])


def test_rejects_offset_with_wrong_length():
    labels = np.zeros((4, 4), dtype=np.uint32)
    with pytest.raises(ValueError, match="spatial ndim"):
        bic.affinities.compute_affinities(labels, [[0, 1, 2]])
