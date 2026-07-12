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
    if any(value < 0.0 or value > shape - 1 for value, shape in zip(coord, data.shape)):
        return fill_value
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
        value = interpolator(data, input_coord, fill_value)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            value = np.clip(np.round(value), info.min, info.max)
        out[local] = value
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


@pytest.mark.parametrize("order", [0, 1, 3])
def test_2d_rotation_matches_reference(order):
    rng = np.random.default_rng(0)
    data = rng.random((8, 9)).astype(np.float32)
    angle = 0.3
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s, 1.0], [s, c, 0.5]], dtype=np.float64)
    bounding_box = (slice(0, 6), slice(0, 7))
    got = bic.transformation.affine_transform(
        data, matrix, bounding_box=bounding_box, order=order, fill_value=-1.0
    )
    ref = _reference(data, matrix, bounding_box, order, np.float32(-1.0))
    np.testing.assert_allclose(got, ref, atol=1e-5)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_3d_rotation_matches_reference(order):
    rng = np.random.default_rng(1)
    data = rng.random((5, 6, 7)).astype(np.float32)
    # Rotation around the z-axis combined with a small translation.
    angle = 0.2
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.3],
            [0.0, c, -s, 0.7],
            [0.0, s, c, -0.4],
        ],
        dtype=np.float64,
    )
    bounding_box = (slice(0, 5), slice(0, 5), slice(0, 5))
    got = bic.transformation.affine_transform(
        data, matrix, bounding_box=bounding_box, order=order, fill_value=-2.0
    )
    ref = _reference(data, matrix, bounding_box, order, np.float32(-2.0))
    np.testing.assert_allclose(got, ref, atol=1e-5)


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


def test_integer_linear_rounds_to_nearest():
    # data[0,0]=0, data[1,0]=3; linear interp at (0.5, 0) = 1.5; rounds to 2.
    data = np.arange(9, dtype=np.uint8).reshape(3, 3)
    got = bic.transformation.affine_transform(data, _matrix(2, [0.5, 0.0]), order=1)
    assert got[0, 0] == np.uint8(2)


def test_integer_cubic_clamps_to_dtype_range():
    # A sharp step in uint8 makes the Keys cubic kernel overshoot below 0 and
    # above 255 near the discontinuity; the cast must clamp to [0, 255].
    data = np.zeros((6, 6), dtype=np.uint8)
    data[:, 3:] = 255
    matrix = _matrix(2, [0.0, 0.5])
    got = bic.transformation.affine_transform(data, matrix, order=3, fill_value=0)
    assert got.dtype == np.uint8
    assert int(got.min()) >= 0
    assert int(got.max()) <= 255


def test_signed_integer_cubic_clamps_to_dtype_range():
    data = np.zeros((6, 6), dtype=np.int8)
    data[:, 3:] = 127
    matrix = _matrix(2, [0.0, 0.5])
    got = bic.transformation.affine_transform(data, matrix, order=3, fill_value=0)
    assert got.dtype == np.int8
    assert int(got.min()) >= -128
    assert int(got.max()) <= 127


def test_output_entirely_outside_input_yields_fill_value():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    matrix = _matrix(2, translation=[100.0, 100.0])
    got = bic.transformation.affine_transform(data, matrix, order=1, fill_value=-5)
    assert got.shape == data.shape
    assert np.all(got == np.float32(-5))


def test_non_contiguous_input_is_handled():
    # The Python wrapper copies non-contiguous input to a C-contiguous buffer
    # before handing it to the C++ kernel; the kernel itself requires C-contig.
    data = np.arange(100, dtype=np.float32).reshape(10, 10)[::2, ::2]
    got = bic.transformation.affine_transform(data, _matrix(2), order=1)
    np.testing.assert_array_equal(got, data)


def test_out_parameter_writes_in_place():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = np.full((4, 4), 99.0, dtype=np.float32)
    returned = bic.transformation.affine_transform(data, _matrix(2), order=1, out=out)
    assert returned is out
    np.testing.assert_array_equal(out, data)


def test_out_parameter_with_bounding_box():
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    out = np.zeros((3, 3), dtype=np.float32)
    bbox = (slice(0, 3), slice(0, 3))
    returned = bic.transformation.affine_transform(
        data, _matrix(2), bounding_box=bbox, order=1, out=out
    )
    assert returned is out
    np.testing.assert_array_equal(out, data[:3, :3])


def test_out_validates_shape_and_dtype():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    with pytest.raises(ValueError, match="shape"):
        bic.transformation.affine_transform(
            data, _matrix(2), order=1, out=np.zeros((3, 3), dtype=np.float32)
        )
    with pytest.raises(TypeError, match="dtype"):
        bic.transformation.affine_transform(
            data, _matrix(2), order=1, out=np.zeros((4, 4), dtype=np.float64)
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        not_contig = np.zeros((4, 8), dtype=np.float32)[:, ::2]
        bic.transformation.affine_transform(data, _matrix(2), order=1, out=not_contig)
    with pytest.raises(ValueError, match="writable"):
        readonly = np.zeros((4, 4), dtype=np.float32)
        readonly.flags.writeable = False
        bic.transformation.affine_transform(data, _matrix(2), order=1, out=readonly)


def test_negative_bounding_box_start_rejected():
    data = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="start"):
        bic.transformation.affine_transform(
            data, _matrix(2), bounding_box=(slice(-1, 3), slice(0, 4))
        )


def test_compute_anti_aliasing_sigma_rotation():
    # Pure rotation: unit row norms → no smoothing.
    angle = 0.4
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s, 1.0], [s, c, 0.5]])
    sigma = bic.transformation.compute_anti_aliasing_sigma(matrix, 2)
    np.testing.assert_allclose(sigma, [0.0, 0.0], atol=1e-12)


def test_compute_anti_aliasing_sigma_isotropic_downsample():
    # 2x downsample on both axes: sigma = (2 - 1) / 2 = 0.5.
    matrix = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    sigma = bic.transformation.compute_anti_aliasing_sigma(matrix, 2)
    np.testing.assert_allclose(sigma, [0.5, 0.5])


def test_compute_anti_aliasing_sigma_anisotropic():
    # 3x along axis 0, 1x along axis 1.
    matrix = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sigma = bic.transformation.compute_anti_aliasing_sigma(matrix, 2)
    np.testing.assert_allclose(sigma, [1.0, 0.0])


def test_compute_anti_aliasing_sigma_accepts_homogeneous():
    M = np.eye(3)
    M[0, 0] = 4.0
    sigma = bic.transformation.compute_anti_aliasing_sigma(M, 2)
    np.testing.assert_allclose(sigma, [1.5, 0.0])


def test_compute_anti_aliasing_sigma_rejects_bad_shape():
    with pytest.raises(ValueError, match="matrix"):
        bic.transformation.compute_anti_aliasing_sigma(np.zeros((4, 4)), 2)


def test_resample_no_aliasing_matches_affine_transform():
    # For an upsample / identity, no smoothing should be applied and resample
    # should agree with affine_transform exactly.
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    matrix = _matrix(2, translation=[0.0, 0.0])
    direct = bic.transformation.affine_transform(data, matrix, order=1)
    via_resample = bic.transformation.resample(data, matrix, order=1)
    np.testing.assert_array_equal(direct, via_resample)


def test_resample_downsample_low_passes_input():
    # 2x downsampling on random data: direct affine_transform samples every
    # other input pixel and inherits the input's variance; resample first
    # low-passes the input so the output variance is reduced.
    rng = np.random.default_rng(0)
    h = w = 64
    data = rng.random((h, w)).astype(np.float32)
    matrix = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    bbox = (slice(0, h // 2), slice(0, w // 2))
    direct = bic.transformation.affine_transform(data, matrix, bounding_box=bbox, order=1)
    smoothed = bic.transformation.resample(data, matrix, bounding_box=bbox, order=1)
    # Sanity: both outputs are non-trivial.
    assert direct.var() > 0.05
    # Smoothing must materially reduce variance vs. direct sampling.
    assert smoothed.var() < direct.var() * 0.75


def test_resample_with_explicit_sigma_runs_filter():
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    matrix = _matrix(2)
    out = bic.transformation.resample(
        data, matrix, anti_aliasing_sigma=1.0, order=1
    )
    # Output should differ from identity (smoothing was applied).
    assert not np.allclose(out, data)


def test_resample_explicit_sigma_zero_skips_smoothing():
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    matrix = _matrix(2)
    out = bic.transformation.resample(
        data, matrix, anti_aliasing_sigma=0.0, order=1
    )
    np.testing.assert_array_equal(out, data)


def test_resample_rejects_negative_sigma():
    data = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        bic.transformation.resample(
            data, _matrix(2), anti_aliasing_sigma=[-1.0, 1.0]
        )


@pytest.mark.parametrize("order", [2, 4, 5])
@pytest.mark.parametrize("ndim", [2, 3])
def test_bspline_orders_match_scipy_prefilter_false(order, ndim):
    pytest.importorskip("scipy")
    from scipy.ndimage import affine_transform as sp_affine
    rng = np.random.default_rng(0)
    if ndim == 2:
        shape = (12, 13)
        translation = [0.37, -1.13]
    else:
        shape = (7, 8, 9)
        translation = [0.25, -0.7, 0.9]
    data = rng.random(shape).astype(np.float32)
    matrix = _matrix(ndim, translation=translation)
    bbox = tuple(slice(0, s) for s in shape)
    got = bic.transformation.affine_transform(data, matrix, bounding_box=bbox, order=order, fill_value=0)
    lin = matrix[:, :ndim]
    offset = matrix[:, ndim]
    # 'grid-constant' is true constant-fill for out-of-bounds taps; scipy's
    # 'constant' mode implicitly extends the input for B-spline orders.
    ref = sp_affine(data, lin, offset=offset, output_shape=shape,
                    order=order, mode="grid-constant", cval=0, prefilter=False)
    # Drop a border to avoid boundary handling differences in the kernel tap
    # extension (we use `fill_value` for out-of-bounds taps; scipy uses
    # `mode="constant"` with the same `cval`, so they agree, but float noise
    # can creep in at the very edge).
    border = order
    interior = tuple(slice(border, s - border) for s in shape)
    np.testing.assert_allclose(got[interior], ref[interior], atol=1e-5)


def test_invalid_inputs_raise():
    data = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="2D or 3D"):
        bic.transformation.affine_transform(np.zeros((4,), dtype=np.float32), _matrix(1))
    with pytest.raises(ValueError, match="matrix"):
        bic.transformation.affine_transform(data, np.eye(4))
    with pytest.raises(ValueError, match="order"):
        bic.transformation.affine_transform(data, _matrix(2), order=6)
    with pytest.raises(ValueError, match="order"):
        bic.transformation.affine_transform(data, _matrix(2), order=-1)
    with pytest.raises(ValueError, match="step"):
        bic.transformation.affine_transform(data, _matrix(2), bounding_box=(slice(None, None, 2), slice(None)))
    with pytest.raises(TypeError, match="dtype"):
        bic.transformation.affine_transform(np.zeros((4, 4), dtype=np.bool_), _matrix(2))


# ---------------------------------------------------------------------------
# map_coordinates
# ---------------------------------------------------------------------------

_ALL_DTYPES = [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float32, np.float64,
]


def _affine_coords(matrix, shape):
    """The (ndim, *shape) coordinate field that reproduces `matrix` as an explicit deformation."""
    ndim = len(shape)
    grid = np.indices(shape, dtype=np.float64)
    coords = np.empty_like(grid)
    for d in range(ndim):
        acc = np.full(shape, matrix[d, ndim], dtype=np.float64)
        for k in range(ndim):
            acc = acc + matrix[d, k] * grid[k]
        coords[d] = acc
    return coords


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", _ALL_DTYPES)
def test_map_coordinates_matches_affine_2d(order, dtype):
    # A coordinate field built from an affine matrix must reproduce affine_transform exactly: same
    # interpolation backend, so the results are bit-identical for every order and dtype.
    data = (np.arange(5 * 7) % 17).astype(dtype).reshape(5, 7)
    matrix = _matrix(2, translation=[0.5, -1.25])
    coords = _affine_coords(matrix, data.shape)
    got = bic.transformation.map_coordinates(data, coords, order=order, fill_value=0)
    ref = bic.transformation.affine_transform(data, matrix, order=order, fill_value=0)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32, np.float64])
def test_map_coordinates_matches_affine_3d(order, dtype):
    data = (np.arange(4 * 5 * 6) % 13).astype(dtype).reshape(4, 5, 6)
    matrix = _matrix(3, translation=[0.25, -0.5, 1.0])
    coords = _affine_coords(matrix, data.shape)
    got = bic.transformation.map_coordinates(data, coords, order=order, fill_value=0)
    ref = bic.transformation.affine_transform(data, matrix, order=order, fill_value=0)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("ndim", [2, 3])
def test_map_coordinates_matches_scipy(order, ndim):
    sp = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(0)
    shape = (9, 11) if ndim == 2 else (6, 7, 8)
    data = rng.random(shape).astype(np.float64)
    # random source coordinates strictly inside the volume, so boundary handling is not involved.
    coords = np.stack([rng.uniform(1.0, s - 2.0, size=shape) for s in shape])
    got = bic.transformation.map_coordinates(data, coords, order=order)
    ref = sp.map_coordinates(data, coords, order=order, mode="nearest", prefilter=False)
    np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("shape", [(5, 7), (4, 5, 6)])
def test_map_coordinates_identity_round_trip(order, shape):
    # Sampling at the integer grid reproduces the input exactly for the interpolating orders.
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    coords = np.indices(shape, dtype=np.float64)
    got = bic.transformation.map_coordinates(data, coords, order=order, fill_value=-1)
    np.testing.assert_array_equal(got, data)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_map_coordinates_out_of_bounds_uses_fill(order):
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    coords = np.full((2, 3, 3), -50.0)  # every output voxel maps far outside the input
    got = bic.transformation.map_coordinates(data, coords, order=order, fill_value=7.0)
    np.testing.assert_array_equal(got, np.full((3, 3), 7.0, dtype=np.float32))


def test_map_coordinates_writes_into_out():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    coords = np.indices(data.shape, dtype=np.float64)
    out = np.empty(data.shape, dtype=np.float64)
    returned = bic.transformation.map_coordinates(data, coords, order=1, out=out)
    assert returned is out
    np.testing.assert_array_equal(out, data)


def test_map_coordinates_accepts_noncontiguous_coordinates():
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    matrix = _matrix(2, translation=[0.5, 0.5])
    coords = _affine_coords(matrix, data.shape)
    noncontig = np.asfortranarray(coords)  # coerced to contiguous float64 internally
    assert not noncontig.flags.c_contiguous
    got = bic.transformation.map_coordinates(data, noncontig, order=1)
    ref = bic.transformation.affine_transform(data, matrix, order=1)
    np.testing.assert_array_equal(got, ref)


def test_map_coordinates_invalid_inputs_raise():
    data = np.zeros((4, 5), dtype=np.float32)
    good = np.indices(data.shape).astype(np.float64)
    with pytest.raises(ValueError, match="2D or 3D"):
        bic.transformation.map_coordinates(np.zeros((4,), dtype=np.float32), np.zeros((1, 4)))
    with pytest.raises(ValueError, match="order"):
        bic.transformation.map_coordinates(data, good, order=6)
    # coordinates must have ndim == data.ndim + 1
    with pytest.raises(ValueError, match="ndim"):
        bic.transformation.map_coordinates(data, np.zeros((2, 4), dtype=np.float64))
    # coordinates leading axis must equal data ndim
    with pytest.raises(ValueError, match=r"shape\[0\]"):
        bic.transformation.map_coordinates(data, np.zeros((3, 4, 5), dtype=np.float64))
    with pytest.raises(TypeError, match="dtype"):
        bic.transformation.map_coordinates(np.zeros((4, 5), dtype=np.bool_), good)
    with pytest.raises(ValueError, match="shape"):
        bic.transformation.map_coordinates(data, good, out=np.empty((3, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="dtype"):
        bic.transformation.map_coordinates(data, good, out=np.empty((4, 5), dtype=np.float64))


@pytest.mark.parametrize("function", ["affine", "coordinates"])
def test_resampling_supports_output_aliasing_input(function):
    data = np.arange(9, dtype=np.int32).reshape(3, 3)
    matrix = _matrix(2, translation=[-1, 0])
    if function == "affine":
        expected = bic.transformation.affine_transform(
            data, matrix, order=0, fill_value=-1
        )
        returned = bic.transformation.affine_transform(
            data, matrix, order=0, fill_value=-1, out=data
        )
    else:
        coords = _affine_coords(matrix, data.shape)
        expected = bic.transformation.map_coordinates(
            data, coords, order=0, fill_value=-1
        )
        returned = bic.transformation.map_coordinates(
            data, coords, order=0, fill_value=-1, out=data
        )
    assert returned is data
    np.testing.assert_array_equal(data, expected)


def test_non_finite_coordinates_use_fill_and_matrix_is_rejected():
    data = np.arange(9, dtype=np.float32).reshape(3, 3)
    coords = np.indices(data.shape, dtype=np.float64)
    coords[0, 1, 1] = np.nan
    result = bic.transformation.map_coordinates(data, coords, order=0, fill_value=-9)
    assert result[1, 1] == -9
    matrix = _matrix(2)
    matrix[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bic.transformation.affine_transform(data, matrix)
