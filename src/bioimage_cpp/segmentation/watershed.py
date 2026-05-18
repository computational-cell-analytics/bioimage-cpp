"""Marker-controlled watershed."""

from __future__ import annotations

import numpy as np

from .. import _core

_WATERSHED_BY_DTYPE: dict[np.dtype, dict[np.dtype, object]] = {
    np.dtype("float32"): {
        np.dtype("uint32"): _core._watershed_float32_uint32,
        np.dtype("uint64"): _core._watershed_float32_uint64,
        np.dtype("int32"): _core._watershed_float32_int32,
        np.dtype("int64"): _core._watershed_float32_int64,
    },
    np.dtype("float64"): {
        np.dtype("uint32"): _core._watershed_float64_uint32,
        np.dtype("uint64"): _core._watershed_float64_uint64,
        np.dtype("int32"): _core._watershed_float64_int32,
        np.dtype("int64"): _core._watershed_float64_int64,
    },
}


def watershed(
    image: np.ndarray,
    markers: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run a marker-controlled watershed on a 2D or 3D heightmap.

    Pixels with non-zero ``markers`` values are treated as seeds. Their
    label is propagated to neighbouring pixels in order of increasing
    ``image`` value, using axis-aligned connectivity (4-neighbours in 2D,
    6-neighbours in 3D). Tie-breaking on equal heights is unspecified.

    Parameters
    ----------
    image:
        Heightmap with shape ``(y, x)`` or ``(z, y, x)``. Supported dtypes
        are ``float32`` and ``float64``. Non-contiguous arrays are copied.
    markers:
        Seed array with the same shape as ``image``. Non-zero entries are
        treated as seeds. Supported dtypes are ``uint32``, ``uint64``,
        ``int32``, ``int64``. The output dtype matches ``markers``.
    mask:
        Optional boolean foreground mask with the same shape as ``image``.
        ``False`` pixels are excluded from the flooding and stay ``0`` in
        the output. A seed under a ``False`` pixel is ignored.

    Returns
    -------
    np.ndarray
        Segmentation labels with the same shape as ``image`` and the same
        dtype as ``markers``. Pixels that the flooding never reaches stay
        at ``0``.
    """
    image_array = np.asarray(image)
    markers_array = np.asarray(markers)

    if image_array.ndim not in (2, 3):
        raise ValueError(
            "image must have ndim 2 or 3, got ndim=" + str(image_array.ndim)
        )
    if markers_array.shape != image_array.shape:
        raise ValueError(
            "markers shape must match image shape, got "
            f"markers shape={markers_array.shape}, image shape={image_array.shape}"
        )

    try:
        by_marker = _WATERSHED_BY_DTYPE[image_array.dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in _WATERSHED_BY_DTYPE)
        raise TypeError(
            f"image must have one of dtypes ({supported}), got dtype={image_array.dtype}"
        ) from error

    try:
        run = by_marker[markers_array.dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in by_marker)
        raise TypeError(
            f"markers must have one of dtypes ({supported}), got dtype={markers_array.dtype}"
        ) from error

    mask_arg = None
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.shape != image_array.shape:
            raise ValueError(
                "mask shape must match image shape, got "
                f"mask shape={mask_array.shape}, image shape={image_array.shape}"
            )
        if mask_array.dtype != np.dtype("bool"):
            raise TypeError(f"mask must have dtype bool, got dtype={mask_array.dtype}")
        mask_arg = np.ascontiguousarray(mask_array.view(np.uint8))

    image_c = np.ascontiguousarray(image_array)
    markers_c = np.ascontiguousarray(markers_array)
    return run(image_c, markers_c, mask_arg)
