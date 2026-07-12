"""Python wrappers for the ``_core`` filter bindings.

Conventions
-----------
* Input is a 2D or 3D NumPy array (single channel). Supported dtypes:
  ``float32``, ``float64``, ``uint8``, ``uint16``. Non-``float32`` inputs are
  cast to ``float32`` for the kernel; ``float64`` results are cast back.
* ``sigma`` (and ``order``) accept a scalar or a per-axis sequence of length
  ``image.ndim``. Sigmas must be positive; orders must be in ``{0, 1, 2}``.
* ``window_size`` controls the kernel radius as
  ``radius = ceil(window_size * sigma)`` (per axis), matching the
  ``fastfilters`` / ``vigra`` parameter. ``0`` selects the default
  ``3 + 0.5 * order``.
* Axis order is NumPy native: ``(ny, nx)`` for 2D, ``(nz, ny, nx)`` for 3D.
* Eigenvalue functions return an array with a trailing axis of size
  ``image.ndim``, sorted descending.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core
from .._validation import strict_index

_FLOAT_INPUT_DTYPES = (np.float32, np.float64)
_INT_INPUT_DTYPES = (np.uint8, np.uint16)


def _prepare_input(image: np.ndarray, function: str) -> tuple[np.ndarray, np.dtype]:
    """Validate ndim and cast to contiguous float32. Return (float32 view,
    desired output dtype)."""
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(
            f"{function}: image must be 2D or 3D, got ndim={image.ndim}"
        )
    if image.dtype == np.float32:
        prepared, out_dtype = np.ascontiguousarray(image), np.dtype(np.float32)
    elif image.dtype == np.float64:
        prepared, out_dtype = np.ascontiguousarray(image, dtype=np.float32), np.dtype(np.float64)
    elif image.dtype in (np.dtype(t) for t in _INT_INPUT_DTYPES):
        return np.ascontiguousarray(image, dtype=np.float32), np.dtype(np.float32)
    else:
        raise TypeError(
            f"{function}: image dtype must be one of (float32, float64, uint8, "
            f"uint16), got dtype={image.dtype}"
        )
    # Reject non-finite float inputs without allocating a whole-array boolean
    # temporary: min()/max() are single-pass reductions and propagate NaN/inf
    # (any NaN -> NaN, any +/-inf -> +/-inf), so a non-finite extremum flags a
    # non-finite sample. Integer inputs return above and skip this entirely.
    if prepared.size and not (
        np.isfinite(prepared.min()) and np.isfinite(prepared.max())
    ):
        raise ValueError(f"{function}: image must contain only finite values")
    return prepared, out_dtype


def _broadcast_per_axis(
    value: float | Sequence[float],
    ndim: int,
    name: str,
    function: str,
) -> tuple[float, ...]:
    if np.isscalar(value):
        seq = (float(value),) * ndim
    else:
        seq = tuple(float(v) for v in value)
    if len(seq) != ndim:
        raise ValueError(
            f"{function}: {name} must be a scalar or a sequence of length "
            f"{ndim}, got length {len(seq)}"
        )
    if not all(np.isfinite(v) and v > 0.0 for v in seq):
        raise ValueError(f"{function}: {name} values must be finite and positive")
    return seq


def _broadcast_order(
    value: int | Sequence[int],
    ndim: int,
    function: str,
) -> tuple[int, ...]:
    if np.isscalar(value):
        return (strict_index(value, "order", minimum=0, maximum=2),) * ndim
    seq = tuple(strict_index(v, "order", minimum=0, maximum=2) for v in value)
    if len(seq) != ndim:
        raise ValueError(
            f"{function}: order must be a scalar or a sequence of length "
            f"{ndim}, got length {len(seq)}"
        )
    return seq


def _normalize_window(window_size: float, function: str) -> float:
    value = float(window_size)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{function}: window_size must be finite and non-negative")
    return value


def _finalise(result: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    if result.dtype == out_dtype:
        return result
    return result.astype(out_dtype, copy=False)


def gaussian_smoothing(
    image: np.ndarray,
    sigma: float | Sequence[float],
    *,
    window_size: float = 0.0,
) -> np.ndarray:
    """Gaussian smoothing of a 2D or 3D scalar image."""
    function = "gaussian_smoothing"
    prepared, out_dtype = _prepare_input(image, function)
    sigmas = _broadcast_per_axis(sigma, prepared.ndim, "sigma", function)
    window = _normalize_window(window_size, function)
    if prepared.ndim == 2:
        result = _core._gaussian_smoothing_2d_float32(
            prepared, sigmas[0], sigmas[1], window
        )
    else:
        result = _core._gaussian_smoothing_3d_float32(
            prepared, sigmas[0], sigmas[1], sigmas[2], window
        )
    return _finalise(result, out_dtype)


def gaussian_derivative(
    image: np.ndarray,
    sigma: float | Sequence[float],
    order: int | Sequence[int],
    *,
    window_size: float = 0.0,
) -> np.ndarray:
    """Gaussian derivative of a 2D or 3D scalar image with per-axis order."""
    function = "gaussian_derivative"
    prepared, out_dtype = _prepare_input(image, function)
    sigmas = _broadcast_per_axis(sigma, prepared.ndim, "sigma", function)
    orders = _broadcast_order(order, prepared.ndim, function)
    window = _normalize_window(window_size, function)
    if prepared.ndim == 2:
        result = _core._gaussian_derivative_2d_float32(
            prepared, sigmas[0], sigmas[1],
            orders[0], orders[1], window,
        )
    else:
        result = _core._gaussian_derivative_3d_float32(
            prepared, sigmas[0], sigmas[1], sigmas[2],
            orders[0], orders[1], orders[2], window,
        )
    return _finalise(result, out_dtype)


def gaussian_gradient_magnitude(
    image: np.ndarray,
    sigma: float | Sequence[float],
    *,
    window_size: float = 0.0,
) -> np.ndarray:
    """L2 norm of the Gaussian gradient of a 2D or 3D scalar image."""
    function = "gaussian_gradient_magnitude"
    prepared, out_dtype = _prepare_input(image, function)
    sigmas = _broadcast_per_axis(sigma, prepared.ndim, "sigma", function)
    window = _normalize_window(window_size, function)
    if prepared.ndim == 2:
        result = _core._gaussian_gradient_magnitude_2d_float32(
            prepared, sigmas[0], sigmas[1], window
        )
    else:
        result = _core._gaussian_gradient_magnitude_3d_float32(
            prepared, sigmas[0], sigmas[1], sigmas[2], window
        )
    return _finalise(result, out_dtype)


def laplacian_of_gaussian(
    image: np.ndarray,
    sigma: float | Sequence[float],
    *,
    window_size: float = 0.0,
) -> np.ndarray:
    """Laplacian of Gaussian (sum of second derivatives) of a 2D or 3D scalar
    image."""
    function = "laplacian_of_gaussian"
    prepared, out_dtype = _prepare_input(image, function)
    sigmas = _broadcast_per_axis(sigma, prepared.ndim, "sigma", function)
    window = _normalize_window(window_size, function)
    if prepared.ndim == 2:
        result = _core._laplacian_of_gaussian_2d_float32(
            prepared, sigmas[0], sigmas[1], window
        )
    else:
        result = _core._laplacian_of_gaussian_3d_float32(
            prepared, sigmas[0], sigmas[1], sigmas[2], window
        )
    return _finalise(result, out_dtype)


def hessian_of_gaussian_eigenvalues(
    image: np.ndarray,
    sigma: float | Sequence[float],
    *,
    window_size: float = 0.0,
) -> np.ndarray:
    """Eigenvalues of the Hessian of Gaussian.

    Output shape: ``image.shape + (image.ndim,)``, sorted descending along the
    trailing axis (largest absolute curvature first only when all eigenvalues
    have the same sign; otherwise simply largest signed value first).
    """
    function = "hessian_of_gaussian_eigenvalues"
    prepared, out_dtype = _prepare_input(image, function)
    sigmas = _broadcast_per_axis(sigma, prepared.ndim, "sigma", function)
    window = _normalize_window(window_size, function)
    if prepared.ndim == 2:
        result = _core._hessian_of_gaussian_eigenvalues_2d_float32(
            prepared, sigmas[0], sigmas[1], window
        )
    else:
        result = _core._hessian_of_gaussian_eigenvalues_3d_float32(
            prepared, sigmas[0], sigmas[1], sigmas[2], window
        )
    return _finalise(result, out_dtype)


def structure_tensor_eigenvalues(
    image: np.ndarray,
    inner_sigma: float | Sequence[float],
    outer_sigma: float | Sequence[float],
    *,
    window_size: float = 0.0,
) -> np.ndarray:
    """Eigenvalues of the structure tensor.

    Output shape: ``image.shape + (image.ndim,)``, sorted descending along the
    trailing axis.
    """
    function = "structure_tensor_eigenvalues"
    prepared, out_dtype = _prepare_input(image, function)
    inner = _broadcast_per_axis(inner_sigma, prepared.ndim, "inner_sigma", function)
    outer = _broadcast_per_axis(outer_sigma, prepared.ndim, "outer_sigma", function)
    window = _normalize_window(window_size, function)
    if prepared.ndim == 2:
        result = _core._structure_tensor_eigenvalues_2d_float32(
            prepared,
            inner[0], inner[1],
            outer[0], outer[1],
            window,
        )
    else:
        result = _core._structure_tensor_eigenvalues_3d_float32(
            prepared,
            inner[0], inner[1], inner[2],
            outer[0], outer[1], outer[2],
            window,
        )
    return _finalise(result, out_dtype)
