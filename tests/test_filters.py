"""Tests for ``bioimage_cpp.filters``.

The C++ kernels are validated against ``scipy.ndimage`` reference filters with
``mode="mirror"`` (matching our boundary handling). float32 tolerance is
``atol=1e-3`` for composite filters, looser for eigenvalues because of the
trigonometric closed-form rounding.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy import ndimage

from bioimage_cpp import _core
import bioimage_cpp.filters as bf


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_filters_reject_non_finite_inputs_and_parameters(value):
    image = np.ones((5, 5), dtype=np.float32)
    bad_image = image.copy()
    bad_image[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        bf.gaussian_smoothing(bad_image, 1.0)
    with pytest.raises(ValueError, match="finite"):
        bf.gaussian_smoothing(image, value)
    with pytest.raises(ValueError, match="finite"):
        bf.gaussian_smoothing(image, 1.0, window_size=value)


SHAPES_2D = [(7, 11), (32, 32), (48, 64)]
SHAPES_3D = [(5, 7, 11), (8, 12, 16)]
SIGMAS = [0.7, 1.5, 3.0]


def _random_image(shape, seed=0, dtype=np.float32):
    rng = np.random.RandomState(seed)
    return rng.rand(*shape).astype(dtype)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES_2D + SHAPES_3D)
@pytest.mark.parametrize("sigma", SIGMAS)
def test_gaussian_smoothing_matches_scipy(shape, sigma):
    img = _random_image(shape)
    got = bf.gaussian_smoothing(img, sigma)
    ref = ndimage.gaussian_filter(img, sigma, mode="mirror")
    assert got.shape == ref.shape
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, ref, atol=1e-3)


def test_gaussian_smoothing_anisotropic_sigma():
    img = _random_image((32, 32))
    got = bf.gaussian_smoothing(img, [1.0, 2.5])
    ref = ndimage.gaussian_filter(img, [1.0, 2.5], mode="mirror")
    np.testing.assert_allclose(got, ref, atol=1e-3)


def test_gaussian_smoothing_3d_anisotropic_sigma():
    vol = _random_image((8, 12, 16))
    got = bf.gaussian_smoothing(vol, [0.7, 1.2, 2.1])
    ref = ndimage.gaussian_filter(vol, [0.7, 1.2, 2.1], mode="mirror")
    np.testing.assert_allclose(got, ref, atol=1e-3)


# ---------------------------------------------------------------------------
# Derivative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [[1, 0], [0, 1], [2, 0], [0, 2], [1, 1]])
def test_gaussian_derivative_2d_matches_scipy(order):
    img = _random_image((32, 48))
    got = bf.gaussian_derivative(img, 1.5, order)
    ref = ndimage.gaussian_filter(img, 1.5, order=order, mode="mirror")
    np.testing.assert_allclose(got, ref, atol=1e-3)


@pytest.mark.parametrize("order", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0],
                                    [1, 1, 0], [1, 0, 1]])
def test_gaussian_derivative_3d_matches_scipy(order):
    vol = _random_image((8, 12, 16))
    got = bf.gaussian_derivative(vol, 1.2, order)
    ref = ndimage.gaussian_filter(vol, 1.2, order=order, mode="mirror")
    np.testing.assert_allclose(got, ref, atol=1e-3)


# ---------------------------------------------------------------------------
# Gradient magnitude
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES_2D + SHAPES_3D)
def test_gradient_magnitude_matches_scipy(shape):
    img = _random_image(shape)
    got = bf.gaussian_gradient_magnitude(img, 1.5)
    ref = ndimage.gaussian_gradient_magnitude(img, 1.5, mode="mirror")
    np.testing.assert_allclose(got, ref, atol=1e-3)


# ---------------------------------------------------------------------------
# Laplacian of Gaussian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES_2D + SHAPES_3D)
def test_laplacian_of_gaussian_matches_scipy(shape):
    img = _random_image(shape)
    got = bf.laplacian_of_gaussian(img, 1.5)
    ref = ndimage.gaussian_laplace(img, 1.5, mode="mirror")
    np.testing.assert_allclose(got, ref, atol=1e-3)


# ---------------------------------------------------------------------------
# Hessian eigenvalues
# ---------------------------------------------------------------------------

def _hessian_eigenvalues_reference_2d(img, sigma):
    hyy = ndimage.gaussian_filter(img, sigma, order=[2, 0], mode="mirror")
    hyx = ndimage.gaussian_filter(img, sigma, order=[1, 1], mode="mirror")
    hxx = ndimage.gaussian_filter(img, sigma, order=[0, 2], mode="mirror")
    mat = np.stack(
        [np.stack([hyy, hyx], axis=-1), np.stack([hyx, hxx], axis=-1)],
        axis=-2,
    )
    return np.linalg.eigvalsh(mat)[..., ::-1].astype(np.float32)


def _hessian_eigenvalues_reference_3d(vol, sigma):
    hzz = ndimage.gaussian_filter(vol, sigma, order=[2, 0, 0], mode="mirror")
    hzy = ndimage.gaussian_filter(vol, sigma, order=[1, 1, 0], mode="mirror")
    hzx = ndimage.gaussian_filter(vol, sigma, order=[1, 0, 1], mode="mirror")
    hyy = ndimage.gaussian_filter(vol, sigma, order=[0, 2, 0], mode="mirror")
    hyx = ndimage.gaussian_filter(vol, sigma, order=[0, 1, 1], mode="mirror")
    hxx = ndimage.gaussian_filter(vol, sigma, order=[0, 0, 2], mode="mirror")
    mat = np.stack([
        np.stack([hzz, hzy, hzx], axis=-1),
        np.stack([hzy, hyy, hyx], axis=-1),
        np.stack([hzx, hyx, hxx], axis=-1),
    ], axis=-2)
    return np.linalg.eigvalsh(mat)[..., ::-1].astype(np.float32)


@pytest.mark.parametrize("shape", SHAPES_2D)
def test_hessian_eigenvalues_2d_matches_reference(shape):
    img = _random_image(shape)
    got = bf.hessian_of_gaussian_eigenvalues(img, 1.5)
    ref = _hessian_eigenvalues_reference_2d(img, 1.5)
    assert got.shape == img.shape + (2,)
    np.testing.assert_allclose(got, ref, atol=2e-3)


@pytest.mark.parametrize("shape", SHAPES_3D)
def test_hessian_eigenvalues_3d_matches_reference(shape):
    vol = _random_image(shape)
    got = bf.hessian_of_gaussian_eigenvalues(vol, 1.2)
    ref = _hessian_eigenvalues_reference_3d(vol, 1.2)
    assert got.shape == vol.shape + (3,)
    # 3x3 trig closed-form needs a slightly looser tolerance.
    np.testing.assert_allclose(got, ref, atol=5e-3)


def test_hessian_eigenvalues_descending_order_2d():
    img = _random_image((32, 32))
    got = bf.hessian_of_gaussian_eigenvalues(img, 1.5)
    assert np.all(got[..., 0] >= got[..., 1])


def test_hessian_eigenvalues_descending_order_3d():
    vol = _random_image((8, 16, 16))
    got = bf.hessian_of_gaussian_eigenvalues(vol, 1.2)
    assert np.all(got[..., 0] >= got[..., 1])
    assert np.all(got[..., 1] >= got[..., 2])


# ---------------------------------------------------------------------------
# Structure tensor eigenvalues
# ---------------------------------------------------------------------------

def _structure_tensor_eigenvalues_reference_2d(img, inner, outer):
    gy = ndimage.gaussian_filter(img, inner, order=[1, 0], mode="mirror")
    gx = ndimage.gaussian_filter(img, inner, order=[0, 1], mode="mirror")
    syy = ndimage.gaussian_filter(gy * gy, outer, mode="mirror")
    syx = ndimage.gaussian_filter(gy * gx, outer, mode="mirror")
    sxx = ndimage.gaussian_filter(gx * gx, outer, mode="mirror")
    mat = np.stack(
        [np.stack([syy, syx], axis=-1), np.stack([syx, sxx], axis=-1)],
        axis=-2,
    )
    return np.linalg.eigvalsh(mat)[..., ::-1].astype(np.float32)


def test_structure_tensor_eigenvalues_2d_matches_reference():
    img = _random_image((48, 48))
    got = bf.structure_tensor_eigenvalues(img, 1.0, 2.0)
    ref = _structure_tensor_eigenvalues_reference_2d(img, 1.0, 2.0)
    assert got.shape == img.shape + (2,)
    np.testing.assert_allclose(got, ref, atol=2e-3)


def _structure_tensor_eigenvalues_reference_3d(vol, inner, outer):
    gz = ndimage.gaussian_filter(vol, inner, order=[1, 0, 0], mode="mirror")
    gy = ndimage.gaussian_filter(vol, inner, order=[0, 1, 0], mode="mirror")
    gx = ndimage.gaussian_filter(vol, inner, order=[0, 0, 1], mode="mirror")
    szz = ndimage.gaussian_filter(gz * gz, outer, mode="mirror")
    szy = ndimage.gaussian_filter(gz * gy, outer, mode="mirror")
    szx = ndimage.gaussian_filter(gz * gx, outer, mode="mirror")
    syy = ndimage.gaussian_filter(gy * gy, outer, mode="mirror")
    syx = ndimage.gaussian_filter(gy * gx, outer, mode="mirror")
    sxx = ndimage.gaussian_filter(gx * gx, outer, mode="mirror")
    mat = np.stack([
        np.stack([szz, szy, szx], axis=-1),
        np.stack([szy, syy, syx], axis=-1),
        np.stack([szx, syx, sxx], axis=-1),
    ], axis=-2)
    return np.linalg.eigvalsh(mat)[..., ::-1].astype(np.float32)


def _symmetric_components(matrices):
    return np.ascontiguousarray(
        np.stack(
            [
                matrices[:, 0, 0],
                matrices[:, 0, 1],
                matrices[:, 0, 2],
                matrices[:, 1, 1],
                matrices[:, 1, 2],
                matrices[:, 2, 2],
            ]
        ),
        dtype=np.float32,
    )


def _direct_ev3(components):
    out = np.empty((components.shape[1], 3), dtype=np.float32)
    _core._filters_ev3_symmetric_float32(components, out)
    return out


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 15, 16, 17])
def test_direct_ev3_handles_vector_boundaries(n):
    rng = np.random.default_rng(42 + n)
    matrices = rng.normal(size=(n, 3, 3)).astype(np.float32)
    matrices += matrices.transpose(0, 2, 1)
    got = _direct_ev3(_symmetric_components(matrices))
    ref = np.linalg.eigvalsh(matrices)[..., ::-1].astype(np.float32)
    np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-6)


def test_direct_ev3_matches_numpy_across_scales():
    rng = np.random.default_rng(91)
    n = 1003
    components = rng.normal(size=(6, n)).astype(np.float32)
    scales = np.power(10.0, rng.uniform(-15.0, 15.0, size=n)).astype(np.float32)
    components *= scales

    matrices = np.empty((n, 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = components[0]
    matrices[:, 0, 1] = matrices[:, 1, 0] = components[1]
    matrices[:, 0, 2] = matrices[:, 2, 0] = components[2]
    matrices[:, 1, 1] = components[3]
    matrices[:, 1, 2] = matrices[:, 2, 1] = components[4]
    matrices[:, 2, 2] = components[5]

    got = _direct_ev3(components)
    ref = np.linalg.eigvalsh(matrices)[..., ::-1].astype(np.float32)
    scale = np.maximum(
        np.max(np.abs(ref), axis=1), np.finfo(np.float32).tiny
    )
    scaled_error = np.max(np.abs(got - ref), axis=1) / scale
    assert np.max(scaled_error) < 2e-5
    assert np.all(got[:, :-1] >= got[:, 1:])


def test_direct_ev3_handles_repeated_and_near_repeated_roots():
    rng = np.random.default_rng(23)
    spectra = []
    smallest_subnormal = float(
        np.nextafter(np.float32(0.0), np.float32(1.0))
    )
    for scale in (
        smallest_subnormal,
        np.finfo(np.float32).tiny,
        1e-20,
        1e-8,
        1.0,
        1e8,
        1e20,
    ):
        spectra.extend(
            [
                [0.0, 0.0, 0.0],
                [scale, scale, scale],
                [3.0 * scale, 3.0 * scale, -2.0 * scale],
                [3.0 * scale, -2.0 * scale, -2.0 * scale],
                [scale, scale * (1.0 + 2e-6), -0.5 * scale],
                [scale, scale * (1.0 - 2e-6), -0.5 * scale],
            ]
        )

    matrices = []
    for eigenvalues in spectra:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrices.append(
            (q @ np.diag(eigenvalues) @ q.T).astype(np.float32)
        )
    matrices = np.stack(matrices)

    got = _direct_ev3(_symmetric_components(matrices))
    ref = np.linalg.eigvalsh(matrices)[..., ::-1].astype(np.float32)
    scale = np.maximum(
        np.max(np.abs(ref), axis=1), np.finfo(np.float32).tiny
    )
    scaled_error = np.max(np.abs(got - ref), axis=1) / scale
    assert np.isfinite(got).all()
    assert np.max(scaled_error) < 3e-4
    assert np.all(got[:, :-1] >= got[:, 1:])


def test_direct_ev3_forced_scalar_matches_automatic(monkeypatch):
    monkeypatch.delenv("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR", raising=False)
    if _core._filters_eigenvalue_backend() != "avx2":
        pytest.skip("AVX2 eigenvalue backend is not available")

    rng = np.random.default_rng(7)
    components = rng.normal(size=(6, 1003)).astype(np.float32)
    automatic = _direct_ev3(components)

    monkeypatch.setenv("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR", "1")
    assert _core._filters_eigenvalue_backend() == "scalar"
    scalar = _direct_ev3(components)
    np.testing.assert_allclose(automatic, scalar, rtol=2e-5, atol=3e-5)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_direct_ev3_rejects_non_finite_components(value):
    components = np.zeros((6, 8), dtype=np.float32)
    components[2, 3] = value
    with pytest.raises(ValueError, match="finite"):
        _direct_ev3(components)


def test_direct_ev3_validates_shapes_and_writability():
    with pytest.raises(ValueError, match=r"shape \(6, n\)"):
        _direct_ev3(np.zeros((5, 8), dtype=np.float32))

    components = np.zeros((6, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        _core._filters_ev3_symmetric_float32(
            components, np.empty((8, 2), dtype=np.float32)
        )

    out = np.empty((8, 3), dtype=np.float32)
    out.flags.writeable = False
    with pytest.raises(TypeError):
        _core._filters_ev3_symmetric_float32(components, out)


def test_structure_tensor_eigenvalues_3d_matches_reference():
    vol = _random_image((8, 16, 16))
    got = bf.structure_tensor_eigenvalues(vol, 1.0, 2.0)
    ref = _structure_tensor_eigenvalues_reference_3d(vol, 1.0, 2.0)
    assert got.shape == vol.shape + (3,)
    np.testing.assert_allclose(got, ref, atol=5e-3)
    assert np.all(got[..., 0] >= got[..., 1])
    assert np.all(got[..., 1] >= got[..., 2])
    assert np.all(got >= -1e-6)


@pytest.mark.parametrize("shape", [(1, 5, 7), (2, 3, 4)])
def test_eigenvalue_filters_support_short_axes(shape):
    vol = _random_image(shape)
    got_hessian = bf.hessian_of_gaussian_eigenvalues(vol, 1.0)
    ref_hessian = _hessian_eigenvalues_reference_3d(vol, 1.0)
    np.testing.assert_allclose(got_hessian, ref_hessian, atol=5e-3)

    got_structure = bf.structure_tensor_eigenvalues(vol, 1.0, 1.5)
    ref_structure = _structure_tensor_eigenvalues_reference_3d(vol, 1.0, 1.5)
    np.testing.assert_allclose(got_structure, ref_structure, atol=5e-3)


def test_runtime_radius_fallback_matches_scipy():
    img = _random_image((40, 48))
    got = bf.gaussian_smoothing(img, 5.0, window_size=3.0)
    ref = ndimage.gaussian_filter(img, 5.0, mode="mirror", truncate=3.0)
    np.testing.assert_allclose(got, ref, atol=1e-3)


def test_forced_scalar_matches_automatic_backend(monkeypatch):
    monkeypatch.delenv("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR", raising=False)
    if _core._filters_convolution_backend() != "avx2":
        pytest.skip("AVX2 filter backend is not available")
    assert _core._filters_eigenvalue_backend() == "avx2"

    vol = _random_image((7, 11, 17))
    functions = [
        lambda: bf.gaussian_smoothing(vol, 1.5),
        lambda: bf.gaussian_derivative(vol, 1.5, [1, 0, 0]),
        lambda: bf.gaussian_gradient_magnitude(vol, 1.5),
        lambda: bf.laplacian_of_gaussian(vol, 1.5),
        lambda: bf.hessian_of_gaussian_eigenvalues(vol, 1.5),
        lambda: bf.structure_tensor_eigenvalues(vol, 1.0, 2.0),
    ]
    automatic = [function() for function in functions]

    monkeypatch.setenv("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR", "1")
    assert _core._filters_convolution_backend() == "scalar"
    assert _core._filters_eigenvalue_backend() == "scalar"
    scalar = [function() for function in functions]

    for got, expected in zip(automatic, scalar, strict=True):
        np.testing.assert_allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize("radius", range(1, 13))
def test_specialized_radii_match_scalar_backend(monkeypatch, radius):
    monkeypatch.delenv("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR", raising=False)
    if _core._filters_convolution_backend() != "avx2":
        pytest.skip("AVX2 filter backend is not available")

    image = _random_image((25, 31))
    automatic_smoothing = bf.gaussian_smoothing(
        image, float(radius), window_size=1.0
    )
    automatic_derivative = bf.gaussian_derivative(
        image, float(radius), [1, 0], window_size=1.0
    )

    monkeypatch.setenv("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR", "1")
    scalar_smoothing = bf.gaussian_smoothing(
        image, float(radius), window_size=1.0
    )
    scalar_derivative = bf.gaussian_derivative(
        image, float(radius), [1, 0], window_size=1.0
    )
    np.testing.assert_allclose(automatic_smoothing, scalar_smoothing, atol=1e-5)
    np.testing.assert_allclose(automatic_derivative, scalar_derivative, atol=1e-5)


def test_concurrent_filter_calls_are_deterministic():
    vol = _random_image((8, 12, 16))
    expected = bf.structure_tensor_eigenvalues(vol, 1.0, 2.0)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: bf.structure_tensor_eigenvalues(vol, 1.0, 2.0),
                range(8),
            )
        )
    for result in results:
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(0, 4), (2, 0, 3)])
def test_filters_support_empty_inputs(shape):
    image = np.empty(shape, dtype=np.float32)
    assert bf.gaussian_smoothing(image, 1.0).shape == shape
    assert bf.hessian_of_gaussian_eigenvalues(image, 1.0).shape == shape + (len(shape),)
    assert (
        bf.structure_tensor_eigenvalues(image, 1.0, 2.0).shape
        == shape + (len(shape),)
    )


# ---------------------------------------------------------------------------
# dtype handling
# ---------------------------------------------------------------------------

def test_float64_input_returns_float64():
    img = _random_image((32, 32), dtype=np.float64)
    got = bf.gaussian_smoothing(img, 1.0)
    assert got.dtype == np.float64
    ref = ndimage.gaussian_filter(img.astype(np.float32), 1.0, mode="mirror")
    np.testing.assert_allclose(got, ref.astype(np.float64), atol=1e-3)


def test_uint8_input_returns_float32():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 256, size=(32, 32), dtype=np.uint8)
    got = bf.gaussian_smoothing(img, 1.0)
    assert got.dtype == np.float32
    ref = ndimage.gaussian_filter(img.astype(np.float32), 1.0, mode="mirror")
    # uint8 values can be up to 255; float32 accumulation produces O(0.1)
    # differences at that magnitude. We only care that the result is in the
    # right ballpark.
    np.testing.assert_allclose(got, ref, atol=0.1)


def test_uint16_input_returns_float32():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 4096, size=(32, 32), dtype=np.uint16)
    got = bf.gaussian_smoothing(img, 1.0)
    assert got.dtype == np.float32


def test_non_contiguous_input_is_handled():
    img = _random_image((32, 32))
    sliced = img[::2, ::2]  # non-contiguous strided view
    got = bf.gaussian_smoothing(sliced, 1.0)
    ref = ndimage.gaussian_filter(sliced, 1.0, mode="mirror")
    np.testing.assert_allclose(got, ref, atol=1e-3)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_wrong_ndim_raises():
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        bf.gaussian_smoothing(np.zeros(8, dtype=np.float32), 1.0)
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        bf.gaussian_smoothing(np.zeros((4, 4, 4, 4), dtype=np.float32), 1.0)


def test_unsupported_dtype_raises():
    with pytest.raises(TypeError, match="dtype"):
        bf.gaussian_smoothing(np.zeros((8, 8), dtype=np.int32), 1.0)


def test_non_positive_sigma_raises():
    img = _random_image((16, 16))
    with pytest.raises(Exception):  # noqa: B017 - C++ -> invalid_argument
        bf.gaussian_smoothing(img, 0.0)
    with pytest.raises(Exception):  # noqa: B017
        bf.gaussian_smoothing(img, -1.0)


def test_sigma_length_mismatch_raises():
    img = _random_image((16, 16))
    with pytest.raises(ValueError, match="sigma"):
        bf.gaussian_smoothing(img, [1.0, 2.0, 3.0])


def test_invalid_order_raises():
    img = _random_image((16, 16))
    with pytest.raises(Exception):  # noqa: B017 - C++ -> invalid_argument
        bf.gaussian_derivative(img, 1.0, 3)
    with pytest.raises(Exception):  # noqa: B017
        bf.gaussian_derivative(img, 1.0, -1)
