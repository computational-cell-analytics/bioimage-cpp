import numpy as np
import pytest

import bioimage_cpp as bic


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64, np.int32, np.int64])
def test_take_dict(dtype):
    to_relabel = np.array([1, 3, 2, 1], dtype=dtype)
    relabeling = {
        dtype(1).item(): dtype(10).item(),
        dtype(2).item(): dtype(20).item(),
        dtype(3).item(): dtype(30).item(),
    }

    out = bic.utils.take_dict(relabeling, to_relabel)

    assert out.dtype == to_relabel.dtype
    np.testing.assert_array_equal(out, np.array([10, 30, 20, 10], dtype=dtype))


def test_take_dict_accepts_non_contiguous_input():
    to_relabel = np.array([0, 9, 1, 9, 2], dtype=np.uint64)[::2]

    out = bic.utils.take_dict({0: 5, 1: 6, 2: 7}, to_relabel)

    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, np.array([5, 6, 7], dtype=np.uint64))


def test_take_dict_preserves_2d_shape():
    to_relabel = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.uint32)

    out = bic.utils.take_dict({1: 10, 2: 20, 3: 30}, to_relabel)

    assert out.shape == to_relabel.shape
    assert out.dtype == to_relabel.dtype
    np.testing.assert_array_equal(
        out, np.array([[10, 20, 30], [30, 20, 10]], dtype=np.uint32)
    )


def test_take_dict_preserves_3d_shape():
    to_relabel = np.array([[[1, 2], [3, 1]], [[2, 3], [1, 2]]], dtype=np.int64)

    out = bic.utils.take_dict({1: -1, 2: -2, 3: -3}, to_relabel)

    assert out.shape == to_relabel.shape
    np.testing.assert_array_equal(
        out, np.array([[[-1, -2], [-3, -1]], [[-2, -3], [-1, -2]]], dtype=np.int64)
    )


def test_take_dict_accepts_non_contiguous_2d_input():
    to_relabel = np.array([[1, 9, 2], [3, 9, 1]], dtype=np.uint64)[:, ::2]

    out = bic.utils.take_dict({1: 10, 2: 20, 3: 30}, to_relabel)

    assert out.shape == (2, 2)
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, np.array([[10, 20], [30, 10]], dtype=np.uint64))


def test_take_dict_rejects_missing_key():
    with pytest.raises(IndexError, match="missing key 2"):
        bic.utils.take_dict({1: 10}, np.array([1, 2], dtype=np.uint32))


def test_take_dict_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="to_relabel must have one of dtypes"):
        bic.utils.take_dict({1: 10}, np.array([1, 2], dtype=np.uint16))

def test_take_dict_rejects_non_mapping():
    with pytest.raises(TypeError, match="relabeling must be a mapping"):
        bic.utils.take_dict([(1, 10)], np.array([1], dtype=np.uint64))
