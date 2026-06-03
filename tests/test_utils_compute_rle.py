import numpy as np
import pytest

import bioimage_cpp as bic

SUPPORTED_DTYPES = [
    np.bool_,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int32,
    np.int64,
]


def _rle(values, dtype=np.uint8):
    return bic.utils.compute_rle(np.asarray(values, dtype=dtype))


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 0, 1, 1, 1, 0, 0, 1, 1], [2, 3, 2, 2]),
        ([1, 1, 0, 1], [0, 2, 1, 1]),
        ([1, 1, 1, 1, 1], [0, 5]),
        ([0, 0, 0, 0], [4]),
        ([1], [0, 1]),
        ([0], [1]),
    ],
)
def test_known_cases(values, expected):
    result = _rle(values)
    np.testing.assert_array_equal(result, np.asarray(expected, dtype=np.int64))


def test_empty_input():
    result = _rle([])
    assert result.dtype == np.int64
    assert result.shape == (0,)


def test_output_is_int64_1d():
    result = _rle([0, 1, 1])
    assert result.dtype == np.int64
    assert result.ndim == 1


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_dtype_variants(dtype):
    mask = np.array([0, 1, 1, 0, 0, 1], dtype=dtype)
    result = bic.utils.compute_rle(mask)
    np.testing.assert_array_equal(result, np.array([1, 2, 2, 1], dtype=np.int64))


def _reference_rle(flat):
    # COCO-style: counts of alternating runs starting with zeros.
    counts = []
    current = 0
    run = 0
    for value in flat:
        binary = int(value != 0)
        if binary == current:
            run += 1
        else:
            counts.append(run)
            current = binary
            run = 1
    counts.append(run)
    return np.asarray(counts, dtype=np.int64)


@pytest.mark.parametrize("shape", [(6, 8), (3, 4, 5)])
def test_c_order_flatten(shape):
    rng = np.random.default_rng(0)
    mask = (rng.integers(0, 2, size=shape)).astype(np.uint8)
    result = bic.utils.compute_rle(mask)
    expected = _reference_rle(mask.ravel(order="C"))
    np.testing.assert_array_equal(result, expected)


def test_non_contiguous_input():
    rng = np.random.default_rng(1)
    base = (rng.integers(0, 2, size=(5, 6))).astype(np.uint8)
    view = base.T  # non-contiguous (Fortran-ordered view)
    assert not view.flags["C_CONTIGUOUS"]
    result = bic.utils.compute_rle(view)
    expected = _reference_rle(np.ascontiguousarray(view).ravel(order="C"))
    np.testing.assert_array_equal(result, expected)


def test_nonzero_values_are_binary():
    # Values other than 0/1 are treated as "set".
    mask = np.array([0, 7, 3, 0], dtype=np.int32)
    result = bic.utils.compute_rle(mask)
    np.testing.assert_array_equal(result, np.array([1, 2, 1], dtype=np.int64))


def test_invalid_dtype_raises():
    with pytest.raises(TypeError):
        bic.utils.compute_rle(np.array([0.0, 1.0], dtype=np.float32))
