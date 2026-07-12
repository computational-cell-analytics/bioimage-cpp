import numpy as np
import pytest

import bioimage_cpp as bic


SUPPORTED_DTYPES = [np.uint32, np.uint64, np.int32, np.int64]


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_basic_1d(dtype):
    label_field = np.array([0, 5, 10, 5, 0, 200], dtype=dtype)

    relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(label_field)

    assert relabeled.dtype == label_field.dtype
    assert forward_map.dtype == label_field.dtype
    assert inverse_map.dtype == label_field.dtype
    np.testing.assert_array_equal(relabeled, np.array([0, 1, 2, 1, 0, 3], dtype=dtype))
    # forward_map[old] == new for present labels
    assert forward_map.shape == (201,)
    assert forward_map[0] == 0
    assert forward_map[5] == 1
    assert forward_map[10] == 2
    assert forward_map[200] == 3
    # inverse_map[new] == old
    np.testing.assert_array_equal(inverse_map, np.array([0, 5, 10, 200], dtype=dtype))


def test_default_offset_preserves_zero():
    label_field = np.array([0, 7, 7, 0, 3], dtype=np.uint32)
    relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(label_field)
    # Sorted unique ge offset = {3, 7} -> new labels {1, 2}
    np.testing.assert_array_equal(relabeled, np.array([0, 2, 2, 0, 1], dtype=np.uint32))
    assert forward_map[0] == 0
    assert forward_map[3] == 1
    assert forward_map[7] == 2
    np.testing.assert_array_equal(inverse_map, np.array([0, 3, 7], dtype=np.uint32))


def test_custom_offset_skimage_semantics():
    label_field = np.array([0, 1, 2, 7, 8, 9, 2, 1, 0], dtype=np.uint32)
    relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(
        label_field, offset=5
    )
    # 0 always maps to 0; all non-zero values sorted ({1,2,7,8,9}) -> {5,6,7,8,9}.
    np.testing.assert_array_equal(
        relabeled, np.array([0, 5, 6, 7, 8, 9, 6, 5, 0], dtype=np.uint32)
    )
    assert forward_map[0] == 0
    assert forward_map[1] == 5
    assert forward_map[2] == 6
    assert forward_map[7] == 7
    assert forward_map[8] == 8
    assert forward_map[9] == 9
    # inverse_map size = offset + n_non_zero = 5 + 5 = 10.
    # Only positions 0 and [5..9] are meaningful; positions 1..4 stay 0.
    np.testing.assert_array_equal(
        inverse_map, np.array([0, 0, 0, 0, 0, 1, 2, 7, 8, 9], dtype=np.uint32)
    )


def test_sorted_order_property():
    # Encounter order does not match sorted order.
    label_field = np.array([3, 1, 2, 3, 1, 2], dtype=np.uint32)
    relabeled, _, inverse_map = bic.segmentation.relabel_sequential(label_field)
    # Sorted unique = {1, 2, 3} -> {1, 2, 3} (offset=1)
    np.testing.assert_array_equal(relabeled, np.array([3, 1, 2, 3, 1, 2], dtype=np.uint32))
    np.testing.assert_array_equal(inverse_map, np.array([0, 1, 2, 3], dtype=np.uint32))

    # Gappy labels in non-sorted encounter order
    label_field = np.array([100, 5, 50, 5, 100], dtype=np.uint32)
    relabeled, _, inverse_map = bic.segmentation.relabel_sequential(label_field)
    # Sorted unique = {5, 50, 100} -> {1, 2, 3}
    np.testing.assert_array_equal(relabeled, np.array([3, 1, 2, 1, 3], dtype=np.uint32))
    np.testing.assert_array_equal(inverse_map, np.array([0, 5, 50, 100], dtype=np.uint32))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_2d_shape_preserved(dtype):
    label_field = np.array([[0, 7, 7], [3, 0, 7]], dtype=dtype)
    relabeled, _, _ = bic.segmentation.relabel_sequential(label_field)
    assert relabeled.shape == label_field.shape
    np.testing.assert_array_equal(
        relabeled, np.array([[0, 2, 2], [1, 0, 2]], dtype=dtype)
    )


def test_3d_shape_preserved():
    label_field = np.array(
        [[[0, 5], [10, 5]], [[10, 0], [5, 10]]], dtype=np.int64
    )
    relabeled, _, _ = bic.segmentation.relabel_sequential(label_field)
    assert relabeled.shape == label_field.shape
    # Sorted unique ge offset = {5, 10} -> {1, 2}
    np.testing.assert_array_equal(
        relabeled, np.array([[[0, 1], [2, 1]], [[2, 0], [1, 2]]], dtype=np.int64)
    )


def test_non_contiguous_input_is_copied():
    label_field = np.array([0, 9, 1, 9, 2, 9, 3], dtype=np.uint64)[::2]
    assert not label_field.flags.c_contiguous
    relabeled, _, inverse_map = bic.segmentation.relabel_sequential(label_field)
    assert relabeled.flags.c_contiguous
    np.testing.assert_array_equal(relabeled, np.array([0, 1, 2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(inverse_map, np.array([0, 1, 2, 3], dtype=np.uint64))


def test_all_zero_input():
    label_field = np.zeros((5,), dtype=np.uint32)
    relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(label_field)
    np.testing.assert_array_equal(relabeled, np.zeros(5, dtype=np.uint32))
    # Only value 0 appears; forward_map size = max_value + 1 = 1
    np.testing.assert_array_equal(forward_map, np.array([0], dtype=np.uint32))
    # No labels >= offset; inverse_map only has the offset prefix [0]
    np.testing.assert_array_equal(inverse_map, np.array([0], dtype=np.uint32))


def test_single_non_zero_label():
    label_field = np.array([0, 42, 0, 42], dtype=np.uint32)
    relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(label_field)
    np.testing.assert_array_equal(relabeled, np.array([0, 1, 0, 1], dtype=np.uint32))
    assert forward_map[0] == 0
    assert forward_map[42] == 1
    np.testing.assert_array_equal(inverse_map, np.array([0, 42], dtype=np.uint32))


def test_empty_input():
    label_field = np.array([], dtype=np.uint64)
    relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(label_field)
    assert relabeled.shape == (0,)
    assert forward_map.shape == (0,)
    # inverse_map should have just the offset prefix [0]
    np.testing.assert_array_equal(inverse_map, np.array([0], dtype=np.uint64))


def test_every_value_equals_offset():
    label_field = np.full((4,), 3, dtype=np.uint32)
    relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(
        label_field, offset=3
    )
    np.testing.assert_array_equal(relabeled, np.full(4, 3, dtype=np.uint32))
    assert forward_map[3] == 3
    np.testing.assert_array_equal(inverse_map, np.array([0, 0, 0, 3], dtype=np.uint32))


def test_forward_map_round_trip():
    rng = np.random.default_rng(0)
    label_field = rng.integers(0, 50, size=(20, 20)).astype(np.uint32)
    relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(label_field)
    # Applying forward_map to label_field should recover relabeled
    np.testing.assert_array_equal(forward_map[label_field], relabeled)
    # Applying inverse_map to relabeled should recover label_field
    np.testing.assert_array_equal(inverse_map[relabeled], label_field)


def test_skimage_cross_check():
    skimage_seg = pytest.importorskip("skimage.segmentation")
    rng = np.random.default_rng(42)
    label_field = rng.integers(0, 100, size=(30, 30)).astype(np.uint32)

    bic_relabeled, bic_forward, bic_inverse = bic.segmentation.relabel_sequential(
        label_field
    )
    sk_relabeled, sk_forward, sk_inverse = skimage_seg.relabel_sequential(label_field)

    np.testing.assert_array_equal(bic_relabeled, sk_relabeled)
    # skimage forward_map and inverse_map act like arrays; compare values via indexing.
    unique_old = np.unique(label_field)
    for old in unique_old:
        assert int(bic_forward[int(old)]) == int(sk_forward[int(old)])
    unique_new = np.unique(bic_relabeled)
    for new in unique_new:
        assert int(bic_inverse[int(new)]) == int(sk_inverse[int(new)])


def test_skimage_cross_check_custom_offset():
    skimage_seg = pytest.importorskip("skimage.segmentation")
    rng = np.random.default_rng(7)
    label_field = rng.integers(0, 30, size=(50,)).astype(np.uint32)
    offset = 10

    bic_relabeled, _, _ = bic.segmentation.relabel_sequential(label_field, offset=offset)
    sk_relabeled, _, _ = skimage_seg.relabel_sequential(label_field, offset=offset)
    np.testing.assert_array_equal(bic_relabeled, sk_relabeled)


def test_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="label_field must have one of dtypes"):
        bic.segmentation.relabel_sequential(np.array([1, 2], dtype=np.uint16))


def test_rejects_float_dtype():
    with pytest.raises(TypeError, match="label_field must have one of dtypes"):
        bic.segmentation.relabel_sequential(np.array([1.0, 2.0], dtype=np.float32))


def test_rejects_negative_values():
    with pytest.raises(ValueError, match="must not contain negative values"):
        bic.segmentation.relabel_sequential(np.array([-1, 2], dtype=np.int32))


def test_rejects_zero_offset():
    with pytest.raises(ValueError, match="offset must be >= 1"):
        bic.segmentation.relabel_sequential(np.array([1, 2], dtype=np.uint32), offset=0)


def test_rejects_negative_offset():
    with pytest.raises(ValueError, match="offset must be >= 1"):
        bic.segmentation.relabel_sequential(np.array([1, 2], dtype=np.uint32), offset=-3)


def test_rejects_non_int_offset():
    with pytest.raises(ValueError, match="offset must be a positive integer"):
        bic.segmentation.relabel_sequential(np.array([1, 2], dtype=np.uint32), offset=1.5)


def test_rejects_bool_offset():
    with pytest.raises(ValueError, match="offset must be a positive integer"):
        bic.segmentation.relabel_sequential(np.array([1, 2], dtype=np.uint32), offset=True)


def test_maximal_label_raises_instead_of_segfaulting():
    # max_value + 1 for the dense forward map would wrap size_t to zero; the
    # kernel must reject this rather than write out of bounds.
    labels = np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(OverflowError):
        bic.segmentation.relabel_sequential(labels)


def test_offset_plus_label_count_overflowing_dtype_raises():
    # offset == UINT32_MAX leaves room for exactly one new label; two distinct
    # labels overflow the label dtype and must raise before allocating the
    # multi-gigabyte inverse map.
    labels = np.array([1, 2], dtype=np.uint32)
    with pytest.raises(OverflowError):
        bic.segmentation.relabel_sequential(labels, offset=int(np.iinfo(np.uint32).max))


def test_offset_plus_label_count_overflowing_uint64_raises():
    labels = np.array([1, 2], dtype=np.uint64)
    with pytest.raises(OverflowError):
        bic.segmentation.relabel_sequential(labels, offset=int(np.iinfo(np.uint64).max))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_maps_remain_plain_ndarrays(dtype):
    # The minimal safe fix keeps the dense 3-tuple contract: forward_map and
    # inverse_map are plain ndarrays of the input dtype.
    labels = np.array([0, 5, 10, 5, 0, 200], dtype=dtype)
    relabeled, forward_map, inverse_map = bic.segmentation.relabel_sequential(labels)
    assert isinstance(forward_map, np.ndarray)
    assert isinstance(inverse_map, np.ndarray)
    assert forward_map.dtype == np.dtype(dtype)
    assert inverse_map.dtype == np.dtype(dtype)
    assert relabeled.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(inverse_map[relabeled], labels)
