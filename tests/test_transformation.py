import itertools

import numpy as np
import pytest

import bioimage_cpp as bic


def _matrix(ndim, translation=None):
    matrix = np.zeros((ndim, ndim + 1), dtype=np.float64)
    matrix[:, :ndim] = np.eye(ndim)
    if translation is not None:
        matrix[:, ndim] = translation
    return matrix


def _cubic_weight(x):
    ax = abs(x)
    if ax < 1.0:
        return (1.5 * ax - 2.5) * ax * ax + 1.0
    if ax < 2.0:
        return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0
    return 0.0


def _sample(data, coord, fill_value):
    if any(index < 0 or index >= shape for index, shape in zip(coord, data.shape)):
        return fill_value
    return data[coord]


def _interp_nearest(data, coord, fill_value):
    sampled = tuple(int(np.floor(value + 0.5)) for value in coord)
    return _sample(data, sampled, fill_value)


def _interp_linear(data, coord, fill_value):
    if any(value < 0.0 or value > shape - 1 for value, shape in zip(coord, data.shape)):
        return fill_value
    lower = [int(np.floor(value)) for value in coord]
    frac = [value - lo for value, lo in zip(coord, lower)]
    value = 0.0
    for bits in itertools.product((0, 1), repeat=data.ndim):
        sampled = tuple(lo + bit for lo, bit in zip(lower, bits))
        weight = np.prod([fr if bit else 1.0 - fr for bit, fr in zip(bits, frac)])
        value += weight * _sample(data, sampled, fill_value)
    return value


def _interp_cubic(data, coord, fill_value):
    if any(value < 0.0 or value > shape - 1 for value, shape in zip(coord, data.shape)):
        return fill_value
    bases = [int(np.floor(value)) for value in coord]
    value = 0.0
    for offsets in itertools.product(range(-1, 3), repeat=data.ndim):
        sampled = tuple(base + offset for base, offset in zip(bases, offsets))
        weight = np.prod([
            _cubic_weight(axis_coord - sample)
            for axis_coord, sample in zip(coord, sampled)
        ])
        value += weight * _sample(data, sampled, fill_value)
    return value


def _reference(data, matrix, bounding_box, order, fill_value):
    starts = [item.start for item in bounding_box]
    stops = [item.stop for item in bounding_box]
    shape = tuple(stop - start for start, stop in zip(starts, stops))
    out = np.empty(shape, dtype=data.dtype)
    interpolator = {
        0: _interp_nearest,
        1: _interp_linear,
        3: _interp_cubic,
    }[order]
    for local in np.ndindex(shape):
        output_coord = np.asarray([start + co for start, co in zip(starts, local)])
        input_coord = matrix[:, :-1] @ output_coord + matrix[:, -1]
        out[local] = interpolator(data, input_coord, fill_value)
    return out


@pytest.mark.parametrize("shape", [(5, 7), (4, 5, 6)])
@pytest.mark.parametrize("order", [0, 1, 3])
def test_identity_keeps_full_border(shape, order):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    got = bic.transformation.affine_transform(data, _matrix(len(shape)), order=order, fill_value=-1)
    np.testing.assert_array_equal(got, data)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_2d_translation_matches_reference(order):
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    matrix = _matrix(2, translation=[0.5, 1.25])
    bounding_box = (slice(0, 4), slice(0, 4))
    got = bic.transformation.affine_transform(
        data, matrix, bounding_box=bounding_box, order=order, fill_value=-2
    )
    ref = _reference(data, matrix, bounding_box, order, np.float32(-2))
    np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_3d_bounding_box_matches_reference(order):
    data = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    matrix = _matrix(3, translation=[0.25, -0.5, 1.0])
    bounding_box = (slice(1, 4), slice(0, 3), slice(2, 6))
    got = bic.transformation.affine_transform(
        data, matrix, bounding_box=bounding_box, order=order, fill_value=-7
    )
    ref = _reference(data, matrix, bounding_box, order, np.float64(-7))
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_homogeneous_matrix_is_accepted():
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    matrix = np.eye(3)
    got = bic.transformation.affine_transform(data, matrix, order=1)
    np.testing.assert_array_equal(got, data)


@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_dtype_is_preserved(dtype):
    data = np.arange(16, dtype=dtype).reshape(4, 4)
    got = bic.transformation.affine_transform(data, _matrix(2, [0.5, 0.0]), order=1)
    assert got.dtype == data.dtype


def test_integer_linear_casts_back_to_input_dtype():
    data = np.arange(9, dtype=np.uint8).reshape(3, 3)
    got = bic.transformation.affine_transform(data, _matrix(2, [0.5, 0.0]), order=1)
    assert got[0, 0] == np.uint8(1)


def test_non_contiguous_input_is_handled():
    data = np.arange(100, dtype=np.float32).reshape(10, 10)[::2, ::2]
    got = bic.transformation.affine_transform(data, _matrix(2), order=1)
    np.testing.assert_array_equal(got, data)


def test_invalid_inputs_raise():
    data = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="2D or 3D"):
        bic.transformation.affine_transform(np.zeros((4,), dtype=np.float32), _matrix(1))
    with pytest.raises(ValueError, match="matrix"):
        bic.transformation.affine_transform(data, np.eye(4))
    with pytest.raises(ValueError, match="order"):
        bic.transformation.affine_transform(data, _matrix(2), order=2)
    with pytest.raises(ValueError, match="step"):
        bic.transformation.affine_transform(data, _matrix(2), bounding_box=(slice(None, None, 2), slice(None)))
    with pytest.raises(TypeError, match="dtype"):
        bic.transformation.affine_transform(np.zeros((4, 4), dtype=np.bool_), _matrix(2))
