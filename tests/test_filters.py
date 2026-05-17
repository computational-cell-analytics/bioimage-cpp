"""Tests for ``bioimage_cpp.filters``.

The C++ kernels are validated against ``scipy.ndimage`` reference filters with
``mode="mirror"`` (matching our boundary handling). float32 tolerance is
``atol=1e-3`` for composite filters, looser for eigenvalues because of the
trigonometric closed-form rounding.
"""

import numpy as np
import pytest
from scipy import ndimage

import bioimage_cpp.filters as bf


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


def test_structure_tensor_eigenvalues_3d_shape_and_order():
    vol = _random_image((8, 16, 16))
    got = bf.structure_tensor_eigenvalues(vol, 1.0, 2.0)
    assert got.shape == vol.shape + (3,)
    assert np.all(got[..., 0] >= got[..., 1])
    assert np.all(got[..., 1] >= got[..., 2])
    # All eigenvalues of a positive-semidefinite tensor are >= 0.
    assert np.all(got >= -1e-6)


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
