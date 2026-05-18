"""Marker-controlled watershed."""

from __future__ import annotations

from collections.abc import Sequence

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

_WATERSHED_FROM_AFFINITIES_BY_DTYPE: dict[np.dtype, dict[np.dtype, object]] = {
    np.dtype("float32"): {
        np.dtype("uint32"): _core._watershed_from_affinities_float32_uint32,
        np.dtype("uint64"): _core._watershed_from_affinities_float32_uint64,
        np.dtype("int32"): _core._watershed_from_affinities_float32_int32,
        np.dtype("int64"): _core._watershed_from_affinities_float32_int64,
    },
    np.dtype("float64"): {
        np.dtype("uint32"): _core._watershed_from_affinities_float64_uint32,
        np.dtype("uint64"): _core._watershed_from_affinities_float64_uint64,
        np.dtype("int32"): _core._watershed_from_affinities_float64_int32,
        np.dtype("int64"): _core._watershed_from_affinities_float64_int64,
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


def watershed_from_affinities(
    affinities: np.ndarray,
    offsets: Sequence[Sequence[int]],
    markers: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run a marker-controlled watershed on nearest-neighbour edge affinities.

    Higher-affinity edges are processed first. Each pixel is claimed by the
    seed that reaches it via the strongest affinity path; ties on equal
    affinities are unspecified.

    Parameters
    ----------
    affinities:
        Affinity tensor of shape ``(C, y, x)`` for 2D data or
        ``(C, z, y, x)`` for 3D, where ``C`` equals the spatial ``ndim``.
        Channel ``c`` holds the affinity of one nearest-neighbour edge per
        pixel, in the direction given by ``offsets[c]``. Supported dtypes
        are ``float32`` and ``float64``. Non-contiguous arrays are copied.
    offsets:
        One nearest-neighbour offset per channel. Each offset is a tuple of
        length ``ndim`` with exactly one entry equal to ``+1`` or ``-1`` and
        the rest zero. All offsets must have the same sign — mixing
        positive and negative directions is not supported. Each spatial
        axis must be covered by exactly one offset.
    markers:
        Seed array with shape ``affinities.shape[1:]``. Non-zero entries
        are seeds. Supported dtypes are ``uint32``, ``uint64``, ``int32``,
        ``int64``; the output dtype matches.
    mask:
        Optional boolean foreground mask with shape
        ``affinities.shape[1:]``. ``False`` pixels stay ``0`` in the
        output; a seed under a ``False`` pixel is ignored.

    Returns
    -------
    np.ndarray
        Segmentation labels with shape ``affinities.shape[1:]`` and the
        same dtype as ``markers``.
    """
    affinities_array = np.asarray(affinities)
    markers_array = np.asarray(markers)

    if affinities_array.ndim not in (3, 4):
        raise ValueError(
            "affinities must have ndim 3 or 4, got ndim=" + str(affinities_array.ndim)
        )
    spatial_ndim = affinities_array.ndim - 1
    n_channels = int(affinities_array.shape[0])
    if n_channels != spatial_ndim:
        raise ValueError(
            "affinities channel count must equal spatial ndim, got "
            f"channels={n_channels}, spatial ndim={spatial_ndim}"
        )
    if markers_array.shape != affinities_array.shape[1:]:
        raise ValueError(
            "markers shape must match affinities spatial shape, got "
            f"markers shape={markers_array.shape}, "
            f"affinities spatial shape={affinities_array.shape[1:]}"
        )

    try:
        by_marker = _WATERSHED_FROM_AFFINITIES_BY_DTYPE[affinities_array.dtype]
    except KeyError as error:
        supported = ", ".join(
            str(dtype) for dtype in _WATERSHED_FROM_AFFINITIES_BY_DTYPE
        )
        raise TypeError(
            "affinities must have one of dtypes "
            f"({supported}), got dtype={affinities_array.dtype}"
        ) from error

    try:
        run = by_marker[markers_array.dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in by_marker)
        raise TypeError(
            f"markers must have one of dtypes ({supported}), got dtype={markers_array.dtype}"
        ) from error

    normalized_offsets = [tuple(int(value) for value in offset) for offset in offsets]
    if len(normalized_offsets) != n_channels:
        raise ValueError(
            "offsets count must equal affinities channel count, got "
            f"offsets={len(normalized_offsets)}, channels={n_channels}"
        )

    sign: int | None = None
    axes_seen: set[int] = set()
    for offset in normalized_offsets:
        if len(offset) != spatial_ndim:
            raise ValueError(
                "each offset must have length matching the spatial ndim, got "
                f"spatial ndim={spatial_ndim}"
            )
        nonzero = [(a, v) for a, v in enumerate(offset) if v != 0]
        if len(nonzero) != 1 or nonzero[0][1] not in (-1, 1):
            raise ValueError(
                "each offset must be a nearest-neighbour offset (one entry of "
                f"value +1 or -1, rest 0), got {offset}"
            )
        axis, value = nonzero[0]
        if sign is None:
            sign = value
        elif sign != value:
            raise ValueError(
                "all offsets must have the same sign (positive or negative "
                "direction); mixing positive and negative is not supported"
            )
        if axis in axes_seen:
            raise ValueError(
                "each spatial axis must be covered by exactly one offset; "
                f"axis {axis} is covered more than once"
            )
        axes_seen.add(axis)
    missing = set(range(spatial_ndim)) - axes_seen
    if missing:
        raise ValueError(
            "each spatial axis must be covered by exactly one offset; "
            f"missing axis {sorted(missing)[0]}"
        )

    mask_arg = None
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.shape != affinities_array.shape[1:]:
            raise ValueError(
                "mask shape must match affinities spatial shape, got "
                f"mask shape={mask_array.shape}, "
                f"affinities spatial shape={affinities_array.shape[1:]}"
            )
        if mask_array.dtype != np.dtype("bool"):
            raise TypeError(f"mask must have dtype bool, got dtype={mask_array.dtype}")
        mask_arg = np.ascontiguousarray(mask_array.view(np.uint8))

    affinities_c = np.ascontiguousarray(affinities_array)
    markers_c = np.ascontiguousarray(markers_array)
    return run(affinities_c, normalized_offsets, markers_c, mask_arg)
