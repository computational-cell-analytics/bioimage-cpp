"""Affine transformations for NumPy arrays."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Number

import numpy as np

from .. import _core


_AFFINE_2D_BY_DTYPE = {
    np.dtype("uint8"): _core._affine_transform_2d_uint8,
    np.dtype("uint16"): _core._affine_transform_2d_uint16,
    np.dtype("uint32"): _core._affine_transform_2d_uint32,
    np.dtype("uint64"): _core._affine_transform_2d_uint64,
    np.dtype("int8"): _core._affine_transform_2d_int8,
    np.dtype("int16"): _core._affine_transform_2d_int16,
    np.dtype("int32"): _core._affine_transform_2d_int32,
    np.dtype("int64"): _core._affine_transform_2d_int64,
    np.dtype("float32"): _core._affine_transform_2d_float32,
    np.dtype("float64"): _core._affine_transform_2d_float64,
}

_AFFINE_3D_BY_DTYPE = {
    np.dtype("uint8"): _core._affine_transform_3d_uint8,
    np.dtype("uint16"): _core._affine_transform_3d_uint16,
    np.dtype("uint32"): _core._affine_transform_3d_uint32,
    np.dtype("uint64"): _core._affine_transform_3d_uint64,
    np.dtype("int8"): _core._affine_transform_3d_int8,
    np.dtype("int16"): _core._affine_transform_3d_int16,
    np.dtype("int32"): _core._affine_transform_3d_int32,
    np.dtype("int64"): _core._affine_transform_3d_int64,
    np.dtype("float32"): _core._affine_transform_3d_float32,
    np.dtype("float64"): _core._affine_transform_3d_float64,
}

_MAP_COORDINATES_2D_BY_DTYPE = {
    np.dtype("uint8"): _core._map_coordinates_2d_uint8,
    np.dtype("uint16"): _core._map_coordinates_2d_uint16,
    np.dtype("uint32"): _core._map_coordinates_2d_uint32,
    np.dtype("uint64"): _core._map_coordinates_2d_uint64,
    np.dtype("int8"): _core._map_coordinates_2d_int8,
    np.dtype("int16"): _core._map_coordinates_2d_int16,
    np.dtype("int32"): _core._map_coordinates_2d_int32,
    np.dtype("int64"): _core._map_coordinates_2d_int64,
    np.dtype("float32"): _core._map_coordinates_2d_float32,
    np.dtype("float64"): _core._map_coordinates_2d_float64,
}

_MAP_COORDINATES_3D_BY_DTYPE = {
    np.dtype("uint8"): _core._map_coordinates_3d_uint8,
    np.dtype("uint16"): _core._map_coordinates_3d_uint16,
    np.dtype("uint32"): _core._map_coordinates_3d_uint32,
    np.dtype("uint64"): _core._map_coordinates_3d_uint64,
    np.dtype("int8"): _core._map_coordinates_3d_int8,
    np.dtype("int16"): _core._map_coordinates_3d_int16,
    np.dtype("int32"): _core._map_coordinates_3d_int32,
    np.dtype("int64"): _core._map_coordinates_3d_int64,
    np.dtype("float32"): _core._map_coordinates_3d_float32,
    np.dtype("float64"): _core._map_coordinates_3d_float64,
}


def _normalize_matrix(matrix, ndim: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape == (ndim, ndim + 1):
        return np.ascontiguousarray(array)
    if array.shape == (ndim + 1, ndim + 1):
        expected_last = np.zeros(ndim + 1, dtype=np.float64)
        expected_last[-1] = 1.0
        if not np.allclose(array[-1], expected_last):
            raise ValueError(
                "homogeneous matrix last row must be "
                f"{expected_last.tolist()}, got {array[-1].tolist()}"
            )
        return np.ascontiguousarray(array[:ndim])
    raise ValueError(
        "matrix must have shape "
        f"({ndim}, {ndim + 1}) or ({ndim + 1}, {ndim + 1}), got {array.shape}"
    )


def _normalize_bounding_box(
    bounding_box: Sequence[slice] | None,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    ndim = len(shape)
    if bounding_box is None:
        starts = np.zeros(ndim, dtype=np.intp)
        output_shape = np.asarray(shape, dtype=np.intp)
        return starts, output_shape

    box = tuple(bounding_box)
    if len(box) != ndim:
        raise ValueError(
            f"bounding_box must contain {ndim} slices, got {len(box)}"
        )

    starts = []
    stops = []
    for axis, item in enumerate(box):
        if not isinstance(item, slice):
            raise TypeError(
                f"bounding_box[{axis}] must be a slice, got {type(item).__name__}"
            )
        if item.step is not None:
            raise ValueError(f"bounding_box[{axis}].step must be None")
        start = 0 if item.start is None else int(item.start)
        stop = shape[axis] if item.stop is None else int(item.stop)
        if start < 0:
            raise ValueError(
                f"bounding_box[{axis}].start must be >= 0, got {start}"
            )
        if stop < start:
            raise ValueError(
                f"bounding_box[{axis}] stop must be >= start, got start={start}, "
                f"stop={stop}"
            )
        starts.append(start)
        stops.append(stop)

    starts_array = np.asarray(starts, dtype=np.intp)
    output_shape = np.asarray(
        [stop - start for start, stop in zip(starts, stops)], dtype=np.intp
    )
    return starts_array, output_shape


def _normalize_fill_value(fill_value: Number, dtype: np.dtype):
    try:
        return dtype.type(fill_value)
    except OverflowError as error:
        raise OverflowError(
            f"fill_value={fill_value!r} cannot be represented as dtype={dtype}"
        ) from error


def _validate_or_allocate_out(
    out: np.ndarray | None,
    output_shape: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    expected_shape = tuple(int(s) for s in output_shape)
    if out is None:
        return np.empty(expected_shape, dtype=dtype)
    if not isinstance(out, np.ndarray):
        raise TypeError(f"out must be a numpy.ndarray, got {type(out).__name__}")
    if out.shape != expected_shape:
        raise ValueError(
            f"out has shape {out.shape}, expected {expected_shape}"
        )
    if out.dtype != dtype:
        raise TypeError(
            f"out has dtype {out.dtype}, expected {dtype} (must match data dtype)"
        )
    if not out.flags.c_contiguous:
        raise ValueError("out must be C-contiguous")
    if not out.flags.writeable:
        raise ValueError("out must be writable")
    return out


def compute_anti_aliasing_sigma(matrix, ndim: int) -> np.ndarray:
    """Per-input-axis Gaussian sigma for anti-aliased resampling.

    Given an affine ``matrix`` that maps output coordinates to input
    coordinates (the convention used by :func:`affine_transform`), returns
    an array of shape ``(ndim,)`` whose entry ``i`` is the recommended
    smoothing sigma along input axis ``i`` *before* sampling.

    The heuristic mirrors ``skimage.transform.resize`` for the
    axis-aligned case: ``sigma_i = max(0, (factor_i - 1) / 2)``, where
    ``factor_i`` is the L2 stretch of input axis ``i`` under the linear
    part of the affine — i.e. ``sqrt(sum_k matrix[i, k]**2)`` over output
    axes ``k``. Pure rotations have unit row norms and yield ``sigma = 0``;
    a uniform 2x downsample yields ``sigma = 0.5`` per axis.

    Accepts ``matrix`` of shape ``(ndim, ndim)``, ``(ndim, ndim + 1)``, or
    the homogeneous ``(ndim + 1, ndim + 1)``.
    """
    M = np.asarray(matrix, dtype=np.float64)
    if M.shape == (ndim, ndim):
        linear = M
    elif M.shape == (ndim, ndim + 1):
        linear = M[:, :ndim]
    elif M.shape == (ndim + 1, ndim + 1):
        linear = M[:ndim, :ndim]
    else:
        raise ValueError(
            f"matrix must have shape ({ndim}, {ndim}), "
            f"({ndim}, {ndim + 1}), or ({ndim + 1}, {ndim + 1}); got {M.shape}"
        )
    factors = np.sqrt(np.sum(linear * linear, axis=1))
    return np.maximum(0.0, (factors - 1.0) / 2.0)


# Smallest sigma we pass to gaussian_smoothing on axes that should not be
# smoothed (the C++ filter rejects sigma == 0; this value produces a kernel
# that is numerically indistinguishable from a delta).
_NO_SMOOTH_SIGMA = 1e-2


def resample(
    data: np.ndarray,
    matrix,
    *,
    bounding_box: Sequence[slice] | None = None,
    order: int = 1,
    fill_value: Number = 0,
    anti_aliasing: bool = True,
    anti_aliasing_sigma=None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Affine resample with optional Gaussian anti-aliasing.

    Equivalent to ``affine_transform(gaussian_smoothing(data, sigma), matrix,
    ...)`` where ``sigma`` is either supplied directly or derived from the
    matrix via :func:`compute_anti_aliasing_sigma`.

    Anti-aliasing is required when the affine map downsamples — i.e. the
    output sample spacing in input coordinates is larger than one input
    pixel. Without it the result aliases the input's high frequencies; with
    it those frequencies are first removed by a Gaussian low-pass before
    sampling. :func:`affine_transform` itself never smooths, so use this
    wrapper for any downscaling pipeline (or compose the two functions
    manually).

    Parameters
    ----------
    anti_aliasing
        If ``True`` (default) and ``anti_aliasing_sigma`` is ``None``, the
        smoothing sigma is computed from ``matrix``. If ``False`` and
        ``anti_aliasing_sigma`` is ``None``, no smoothing is applied. If
        ``anti_aliasing_sigma`` is provided, this flag is ignored.
    anti_aliasing_sigma
        Optional explicit smoothing sigma (scalar or per-input-axis
        sequence). Overrides ``anti_aliasing``.

    Returns
    -------
    numpy.ndarray
        The resampled image, with the same dtype as ``data``.

    See Also
    --------
    affine_transform : The underlying sampler.
    compute_anti_aliasing_sigma : The sigma heuristic used here.
    """
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise ValueError(f"data must be 2D or 3D, got ndim={array.ndim}")

    if anti_aliasing_sigma is None:
        if anti_aliasing:
            sigma = compute_anti_aliasing_sigma(matrix, array.ndim)
        else:
            sigma = np.zeros(array.ndim, dtype=np.float64)
    else:
        sigma = np.asarray(anti_aliasing_sigma, dtype=np.float64)
        if sigma.ndim == 0:
            sigma = np.full(array.ndim, float(sigma))
        if sigma.shape != (array.ndim,):
            raise ValueError(
                f"anti_aliasing_sigma must be a scalar or sequence of length "
                f"{array.ndim}, got shape {sigma.shape}"
            )
        if np.any(sigma < 0):
            raise ValueError(
                f"anti_aliasing_sigma must be non-negative, got {sigma.tolist()}"
            )

    if np.any(sigma > 0):
        from .. import filters
        # Replace zero-sigma axes with a tiny positive value so the C++
        # filter accepts the per-axis tuple; the resulting kernel on those
        # axes is numerically a delta.
        sigma_for_filter = np.maximum(sigma, _NO_SMOOTH_SIGMA)
        smoothed = filters.gaussian_smoothing(array, tuple(sigma_for_filter.tolist()))
        if smoothed.dtype != array.dtype:
            smoothed = smoothed.astype(array.dtype, copy=False)
    else:
        smoothed = array

    return affine_transform(
        smoothed,
        matrix,
        bounding_box=bounding_box,
        order=order,
        fill_value=fill_value,
        out=out,
    )


def affine_transform(
    data: np.ndarray,
    matrix,
    *,
    bounding_box: Sequence[slice] | None = None,
    order: int = 1,
    fill_value: Number = 0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply an affine transformation to a 2D or 3D NumPy array.

    ``matrix`` maps output coordinates to input coordinates. For an output
    coordinate ``o`` in NumPy axis order, the sampled input coordinate is
    ``matrix[:, :-1] @ o + matrix[:, -1]``. ``matrix`` may have shape
    ``(ndim, ndim + 1)`` or homogeneous shape ``(ndim + 1, ndim + 1)``.

    Supported interpolation orders:

    - ``0`` — nearest neighbour.
    - ``1`` — linear (bi-/tri-linear).
    - ``2`` — quadratic B-spline (3 taps per axis).
    - ``3`` — local Keys cubic convolution (4 taps; Catmull-Rom, ``a=-0.5``).
      Interpolating: reproduces input exactly at integer coordinates.
    - ``4`` — quartic B-spline (5 taps per axis).
    - ``5`` — quintic B-spline (6 taps per axis).

    Orders ``2``, ``4``, ``5`` evaluate the cardinal B-spline kernel directly
    on the input samples (no prefilter pass). They match
    ``scipy.ndimage.affine_transform(..., prefilter=False)`` and are *low-pass
    smoothing*, not interpolating — they do not reproduce input samples at
    integer coordinates. Order ``3`` is intentionally different from scipy's
    default cubic (which is a prefiltered cubic B-spline) and reproduces
    integer-coordinate samples. See ``PERFORMANCE_NOTES.md`` for what a
    scipy-compatible prefilter would cost.

    The output preserves the input dtype for all interpolation orders.
    Integer outputs round the interpolated value to the nearest integer
    and clamp to the dtype range so that cubic overshoots are well defined.

    ``bounding_box`` selects the rectangular region of the output frame to
    compute as a tuple of slices with non-negative starts. ``bounding_box``
    of ``None`` is equivalent to ``(slice(0, data.shape[d]) for d in range(ndim))``.

    Pass a pre-allocated, C-contiguous, writable NumPy array as ``out`` to
    write the result in place. ``out`` must have shape matching
    ``bounding_box`` (or ``data.shape`` when ``bounding_box`` is ``None``)
    and dtype equal to ``data.dtype``.
    """
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise ValueError(f"data must be 2D or 3D, got ndim={array.ndim}")

    order = int(order)
    if order not in (0, 1, 2, 3, 4, 5):
        raise ValueError(f"order must be in 0..5, got {order}")

    table = _AFFINE_2D_BY_DTYPE if array.ndim == 2 else _AFFINE_3D_BY_DTYPE
    dtype = array.dtype
    try:
        run = table[dtype]
    except KeyError as error:
        supported = ", ".join(str(name) for name in table)
        raise TypeError(
            f"data must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    normalized_matrix = _normalize_matrix(matrix, array.ndim)
    starts, output_shape = _normalize_bounding_box(bounding_box, array.shape)
    contiguous = np.ascontiguousarray(array)
    typed_fill = _normalize_fill_value(fill_value, dtype)
    output = _validate_or_allocate_out(out, output_shape, dtype)
    run(contiguous, normalized_matrix, starts, output, order, typed_fill)
    return output


def _normalize_coordinates(coordinates, ndim: int) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.ascontiguousarray(coordinates, dtype=np.float64)
    if array.ndim != ndim + 1:
        raise ValueError(
            f"coordinates must have ndim={ndim + 1} (a leading axis of length {ndim} plus the "
            f"output shape), got ndim={array.ndim}"
        )
    if array.shape[0] != ndim:
        raise ValueError(
            f"coordinates.shape[0] must equal the data dimension {ndim}, got {array.shape[0]}"
        )
    output_shape = tuple(int(s) for s in array.shape[1:])
    return array, output_shape


def map_coordinates(
    data: np.ndarray,
    coordinates,
    *,
    order: int = 1,
    fill_value: Number = 0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Resample a 2D or 3D array at explicit, per-output-voxel source coordinates.

    Like ``scipy.ndimage.map_coordinates``, but specialized to a same-rank resampling: for each
    output voxel the source coordinate to sample is read from ``coordinates`` and the input is
    interpolated there. ``coordinates`` has shape ``(ndim, *output_shape)`` in NumPy axis order, so
    ``coordinates[d]`` holds the axis-``d`` source coordinate of every output voxel. The output has
    the same number of dimensions as ``data`` (a 2D or 3D resampling), so ``output_shape`` =
    ``coordinates.shape[1:]`` has ``ndim`` entries. This is the deformation-field counterpart of
    :func:`affine_transform` (which derives the same per-voxel source coordinate from an affine
    matrix) and shares its interpolation backend.

    Supported interpolation orders match :func:`affine_transform`:

    - ``0`` — nearest neighbour.
    - ``1`` — linear (bi-/tri-linear).
    - ``2`` — quadratic B-spline (3 taps per axis).
    - ``3`` — local Keys cubic convolution (4 taps; Catmull-Rom, ``a=-0.5``); reproduces input
      exactly at integer coordinates.
    - ``4`` — quartic B-spline (5 taps per axis).
    - ``5`` — quintic B-spline (6 taps per axis).

    Orders ``2``, ``4``, ``5`` evaluate the cardinal B-spline kernel directly on the input samples
    (no prefilter); they are low-pass smoothing, not interpolating. See :func:`affine_transform`
    for the full discussion.

    The output preserves the input dtype; integer outputs round to nearest and clamp to the dtype
    range. Coordinates that map outside the input contribute ``fill_value``.

    Pass a pre-allocated, C-contiguous, writable NumPy array as ``out`` to write the result in
    place; it must have shape ``coordinates.shape[1:]`` and dtype equal to ``data.dtype``.
    """
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise ValueError(f"data must be 2D or 3D, got ndim={array.ndim}")

    order = int(order)
    if order not in (0, 1, 2, 3, 4, 5):
        raise ValueError(f"order must be in 0..5, got {order}")

    table = _MAP_COORDINATES_2D_BY_DTYPE if array.ndim == 2 else _MAP_COORDINATES_3D_BY_DTYPE
    dtype = array.dtype
    try:
        run = table[dtype]
    except KeyError as error:
        supported = ", ".join(str(name) for name in table)
        raise TypeError(
            f"data must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    coords, output_shape = _normalize_coordinates(coordinates, array.ndim)
    contiguous = np.ascontiguousarray(array)
    typed_fill = _normalize_fill_value(fill_value, dtype)
    output = _validate_or_allocate_out(out, output_shape, dtype)
    run(contiguous, coords, output, order, typed_fill)
    return output
