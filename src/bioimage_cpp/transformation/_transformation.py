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


def affine_transform(
    data: np.ndarray,
    matrix,
    *,
    bounding_box: Sequence[slice] | None = None,
    order: int = 1,
    fill_value: Number = 0,
) -> np.ndarray:
    """Apply an affine transformation to a 2D or 3D NumPy array.

    ``matrix`` maps output coordinates to input coordinates. For output
    coordinate ``o`` in NumPy axis order, the sampled input coordinate is
    ``matrix[:, :-1] @ o + matrix[:, -1]``. Interpolation orders are nearest
    neighbor (``0``), linear (``1``), and local cubic convolution (``3``).
    The output preserves the input dtype for all interpolation orders.
    """
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise ValueError(f"data must be 2D or 3D, got ndim={array.ndim}")

    order = int(order)
    if order not in (0, 1, 3):
        raise ValueError(f"order must be 0, 1 or 3, got {order}")

    table = _AFFINE_2D_BY_DTYPE if array.ndim == 2 else _AFFINE_3D_BY_DTYPE
    dtype = array.dtype
    try:
        run = table[dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in table)
        raise TypeError(
            f"data must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    normalized_matrix = _normalize_matrix(matrix, array.ndim)
    starts, output_shape = _normalize_bounding_box(bounding_box, array.shape)
    contiguous = np.ascontiguousarray(array)
    typed_fill = _normalize_fill_value(fill_value, dtype)
    return run(contiguous, normalized_matrix, starts, output_shape, order, typed_fill)
