"""Segmentation algorithms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core

_MUTEX_WATERSHED_BY_DTYPE = {
    np.dtype("float32"): _core._mutex_watershed_grid_float32,
    np.dtype("float64"): _core._mutex_watershed_grid_float64,
}


def mutex_watershed(
    affinities: np.ndarray,
    offsets: Sequence[Sequence[int]],
    number_of_attractive_channels: int,
) -> np.ndarray:
    """Run mutex watershed on a 2D or 3D image-derived grid graph.

    Parameters
    ----------
    affinities:
        Array with shape ``(channels, y, x)`` for 2D data or
        ``(channels, z, y, x)`` for 3D data. Supported dtypes are
        ``float32`` and ``float64``. Non-contiguous arrays are copied.
    offsets:
        One offset per channel, in NumPy axis order. Each offset has length 2
        for 2D affinities or length 3 for 3D affinities.
    number_of_attractive_channels:
        The first this many affinity channels are attractive merge edges. The
        remaining channels are mutex edges.

    Returns
    -------
    np.ndarray
        Consecutive 1-based ``uint64`` segmentation labels with shape
        ``affinities.shape[1:]``.
    """
    array = np.asarray(affinities)
    if array.ndim not in (3, 4):
        raise ValueError(
            "affinities must have shape (channels, y, x) or "
            f"(channels, z, y, x), got ndim={array.ndim}"
        )

    dtype = array.dtype
    try:
        run = _MUTEX_WATERSHED_BY_DTYPE[dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in _MUTEX_WATERSHED_BY_DTYPE)
        raise TypeError(
            f"affinities must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    normalized_offsets = [tuple(int(value) for value in offset) for offset in offsets]
    spatial_ndim = array.ndim - 1
    if len(normalized_offsets) != array.shape[0]:
        raise ValueError(
            "offsets length must match affinities channel count, got "
            f"offsets length={len(normalized_offsets)}, channels={array.shape[0]}"
        )
    if any(len(offset) != spatial_ndim for offset in normalized_offsets):
        raise ValueError(
            "each offset must have length matching the spatial ndim, got "
            f"spatial ndim={spatial_ndim}"
        )
    if number_of_attractive_channels < 0:
        raise ValueError("number_of_attractive_channels must be non-negative")
    if number_of_attractive_channels > array.shape[0]:
        raise ValueError("number_of_attractive_channels must be <= number of channels")

    contiguous = np.ascontiguousarray(array)
    return run(contiguous, normalized_offsets, number_of_attractive_channels)
