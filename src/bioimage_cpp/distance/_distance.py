"""Python wrappers for distance transform bindings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core


def _as_binary_input(input_: np.ndarray, function: str) -> np.ndarray:
    array = np.asarray(input_)
    if array.ndim < 1:
        raise ValueError(f"{function}: input must have ndim >= 1, got ndim={array.ndim}")
    return np.ascontiguousarray(array != 0, dtype=np.uint8)


def _normalize_sampling(
    sampling: float | Sequence[float] | None,
    ndim: int,
    function: str,
) -> list[float]:
    if sampling is None:
        values = [1.0] * ndim
    elif np.isscalar(sampling):
        values = [float(sampling)] * ndim
    else:
        values = [float(value) for value in sampling]
        if len(values) != ndim:
            raise ValueError(
                f"{function}: sampling must be a scalar or a sequence of length "
                f"{ndim}, got length {len(values)}"
            )

    for axis, value in enumerate(values):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{function}: sampling values must be positive and finite, "
                f"got sampling[{axis}]={value}"
            )
    return values


def _validate_distances(distances: np.ndarray, shape: tuple[int, ...], function: str) -> np.ndarray:
    array = np.asarray(distances)
    if array.shape != shape:
        raise ValueError(
            f"{function}: distances must have shape {shape}, got shape={array.shape}"
        )
    if array.dtype != np.float32:
        raise TypeError(
            f"{function}: distances must have dtype float32, got dtype={array.dtype}"
        )
    if not array.flags.writeable:
        raise ValueError(f"{function}: distances must be writable")
    return array


def _validate_indices(indices: np.ndarray, shape: tuple[int, ...], function: str) -> np.ndarray:
    array = np.asarray(indices)
    expected = (len(shape),) + shape
    if array.shape != expected:
        raise ValueError(
            f"{function}: indices must have shape {expected}, got shape={array.shape}"
        )
    if array.dtype != np.int32:
        raise TypeError(
            f"{function}: indices must have dtype int32, got dtype={array.dtype}"
        )
    if not array.flags.writeable:
        raise ValueError(f"{function}: indices must be writable")
    return array


def distance_transform(
    input: np.ndarray,
    sampling: float | Sequence[float] | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: np.ndarray | None = None,
    indices: np.ndarray | None = None,
):
    """Exact Euclidean distance transform for binary inputs.

    This follows the interface of ``scipy.ndimage.distance_transform_edt``:
    nonzero input elements are foreground and distances are measured to the
    nearest zero-valued element. Distance outputs use ``float32``.
    """
    function = "distance_transform"
    binary = _as_binary_input(input, function)
    shape = tuple(binary.shape)
    sampling_values = _normalize_sampling(sampling, binary.ndim, function)

    want_distances = bool(return_distances)
    want_indices = bool(return_indices)
    if not want_distances and not want_indices:
        raise RuntimeError("at least one of return_distances/return_indices must be True")

    distances_out = None
    indices_out = None
    if distances is not None:
        distances_out = _validate_distances(distances, shape, function)
        want_distances = True
    if indices is not None:
        indices_out = _validate_indices(indices, shape, function)
        want_indices = True

    computed_distances, computed_indices = _core._distance_transform_uint8(
        binary,
        sampling_values,
        want_distances,
        want_indices,
    )

    returned_distances = computed_distances
    returned_indices = computed_indices
    if distances_out is not None:
        distances_out[...] = computed_distances
        returned_distances = None
    if indices_out is not None:
        indices_out[...] = computed_indices
        returned_indices = None

    if returned_distances is not None and returned_indices is not None:
        return returned_distances, returned_indices
    if returned_distances is not None:
        return returned_distances
    if returned_indices is not None:
        return returned_indices
    return None


def vector_difference_transform(
    input: np.ndarray,
    sampling: float | Sequence[float] | None = None,
) -> np.ndarray:
    """Return sampled vectors from foreground pixels to nearest background.

    The output has shape ``input.shape + (input.ndim,)`` and dtype
    ``float32``. The trailing vector axis follows NumPy axis order.
    """
    function = "vector_difference_transform"
    binary = _as_binary_input(input, function)
    sampling_values = _normalize_sampling(sampling, binary.ndim, function)
    return _core._vector_difference_transform_uint8(binary, sampling_values)
