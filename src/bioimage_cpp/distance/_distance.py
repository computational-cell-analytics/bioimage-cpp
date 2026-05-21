"""Python wrappers for distance transform bindings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core


def _as_binary_input(input_: np.ndarray, function: str) -> np.ndarray:
    array = np.asarray(input_)
    if array.ndim < 1:
        raise ValueError(f"{function}: input must have ndim >= 1, got ndim={array.ndim}")
    if array.dtype == np.uint8 and array.flags.c_contiguous:
        return array
    if array.dtype == bool and array.flags.c_contiguous:
        return array.view(np.uint8)
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


def _validate_buffer(
    buffer: np.ndarray,
    name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
    function: str,
) -> np.ndarray:
    array = np.asarray(buffer)
    if array.shape != expected_shape:
        raise ValueError(
            f"{function}: {name} must have shape {expected_shape}, got shape={array.shape}"
        )
    if array.dtype != expected_dtype:
        raise TypeError(
            f"{function}: {name} must have dtype {expected_dtype}, got dtype={array.dtype}"
        )
    if not array.flags.writeable:
        raise ValueError(f"{function}: {name} must be writable")
    if not array.flags.c_contiguous:
        raise ValueError(f"{function}: {name} must be C-contiguous")
    return array


def _normalize_threads(number_of_threads: int, function: str) -> int:
    if number_of_threads is None:
        return 1
    value = int(number_of_threads)
    if value < 0:
        raise ValueError(
            f"{function}: number_of_threads must be >= 0, got {number_of_threads}"
        )
    return value


def distance_transform(
    input: np.ndarray,
    sampling: float | Sequence[float] | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    return_vectors: bool = False,
    distances: np.ndarray | None = None,
    indices: np.ndarray | None = None,
    vectors: np.ndarray | None = None,
    number_of_threads: int = 1,
):
    """Exact Euclidean distance transform for binary inputs.

    Follows ``scipy.ndimage.distance_transform_edt``: nonzero input elements
    are foreground and distances are measured to the nearest zero-valued
    element. Uses the separable Felzenszwalb–Huttenlocher algorithm with
    O(N * ndim) complexity.

    Parameters
    ----------
    input
        Binary input array of any ndim. Nonzero is foreground.
    sampling
        Per-axis voxel spacing. Scalar or per-axis sequence; default 1.0
        per axis.
    return_distances, return_indices, return_vectors
        Which outputs to compute. Distances are ``float32`` with shape
        ``input.shape``. Indices are ``int32`` with shape
        ``(ndim, *input.shape)`` (one component plane per axis, matches
        SciPy). Vectors are ``float32`` with shape ``(*input.shape, ndim)``
        and report sampled displacements from each pixel to its nearest
        background (``feature_coord - pixel_coord) * sampling`` along each
        axis).
    distances, indices, vectors
        Optional pre-allocated output buffers. Each must be C-contiguous,
        writable, of the documented shape and dtype. Outputs are written
        into the user's buffer in place and excluded from the return value.
    number_of_threads
        ``0`` uses ``hardware_concurrency``; a positive value pins the
        thread count. Default ``1`` (single-threaded).

    Returns
    -------
    Depending on which outputs were requested and which were preallocated:
    a single array (one output requested, no preallocation), a tuple in
    ``(distances, indices, vectors)`` order (multiple outputs requested,
    none preallocated), a tuple with preallocated outputs omitted, or
    ``None`` if every requested output was preallocated.

    Notes
    -----
    For all-foreground inputs SciPy reports distances and indices against a
    virtual background row at axis-0 coordinate ``-1``. This wrapper mirrors
    that convention, so ``indices[0]`` may contain ``-1`` in that case.
    """
    function = "distance_transform"
    binary = _as_binary_input(input, function)
    shape = tuple(binary.shape)
    sampling_values = _normalize_sampling(sampling, binary.ndim, function)
    n_threads = _normalize_threads(number_of_threads, function)

    want_distances = bool(return_distances)
    want_indices = bool(return_indices)
    want_vectors = bool(return_vectors)

    distances_buf = None
    indices_buf = None
    vectors_buf = None
    if distances is not None:
        distances_buf = _validate_buffer(
            distances, "distances", shape, np.dtype(np.float32), function
        )
        want_distances = True
    if indices is not None:
        indices_buf = _validate_buffer(
            indices, "indices", (binary.ndim,) + shape, np.dtype(np.int32), function
        )
        want_indices = True
    if vectors is not None:
        vectors_buf = _validate_buffer(
            vectors, "vectors", shape + (binary.ndim,), np.dtype(np.float32), function
        )
        want_vectors = True

    if not (want_distances or want_indices or want_vectors):
        raise RuntimeError(
            "at least one of return_distances/return_indices/return_vectors must be True"
        )

    computed_distances, computed_indices, computed_vectors = _core._distance_transform_uint8(
        binary,
        sampling_values,
        want_distances,
        want_indices,
        want_vectors,
        distances_buf,
        indices_buf,
        vectors_buf,
        n_threads,
    )

    returned: list[np.ndarray] = []
    if want_distances and distances_buf is None:
        returned.append(computed_distances)
    if want_indices and indices_buf is None:
        returned.append(computed_indices)
    if want_vectors and vectors_buf is None:
        returned.append(computed_vectors)

    if len(returned) == 0:
        return None
    if len(returned) == 1:
        return returned[0]
    return tuple(returned)


def vector_difference_transform(
    input: np.ndarray,
    sampling: float | Sequence[float] | None = None,
    number_of_threads: int = 1,
) -> np.ndarray:
    """Return sampled vectors from foreground pixels to nearest background.

    Thin wrapper around :func:`distance_transform` with
    ``return_vectors=True``. Output has shape ``input.shape + (input.ndim,)``
    and dtype ``float32``. The trailing vector axis follows NumPy axis order.
    """
    return distance_transform(
        input,
        sampling=sampling,
        return_distances=False,
        return_indices=False,
        return_vectors=True,
        number_of_threads=number_of_threads,
    )
