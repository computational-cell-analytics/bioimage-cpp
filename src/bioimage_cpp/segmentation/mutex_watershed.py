"""Segmentation algorithms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core

_MUTEX_WATERSHED_BY_DTYPE = {
    np.dtype("float32"): _core._mutex_watershed_grid_float32,
    np.dtype("float64"): _core._mutex_watershed_grid_float64,
}

_SEMANTIC_MUTEX_WATERSHED_BY_DTYPE = {
    np.dtype("float32"): _core._semantic_mutex_watershed_grid_float32,
    np.dtype("float64"): _core._semantic_mutex_watershed_grid_float64,
}


def mutex_watershed(
    affinities: np.ndarray,
    offsets: Sequence[Sequence[int]],
    number_of_attractive_channels: int,
    *,
    strides: Sequence[int] | None = None,
    randomized_strides: bool = False,
    mask: np.ndarray | None = None,
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
    strides:
        Optional spatial sub-sampling strides for mutex edges. Attractive
        channels are always kept. If given, it must have one positive integer
        per spatial dimension.
    randomized_strides:
        If ``True``, sub-sample mutex edges randomly with probability
        ``1 / np.prod(strides)`` instead of on a regular grid.
    mask:
        Optional boolean foreground mask with shape ``affinities.shape[1:]``.
        Edges touching ``False`` pixels are ignored and ``False`` pixels are
        labeled as background ``0`` in the output.

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

    normalized_strides = _normalize_strides(strides, spatial_ndim, randomized_strides)
    valid_edges = _compute_valid_edges(
        array.shape,
        normalized_offsets,
        number_of_attractive_channels,
        normalized_strides,
        randomized_strides,
        mask,
    )

    contiguous = np.ascontiguousarray(array)
    labels = run(
        contiguous,
        valid_edges,
        normalized_offsets,
        number_of_attractive_channels,
    )
    if mask is not None:
        labels[np.logical_not(np.asarray(mask))] = 0
    return labels


def semantic_mutex_watershed(
    affinities: np.ndarray,
    offsets: Sequence[Sequence[int]],
    number_of_attractive_channels: int,
    *,
    strides: Sequence[int] | None = None,
    randomized_strides: bool = False,
    mask: np.ndarray | None = None,
    mask_label: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run semantic mutex watershed on a 2D or 3D image-derived grid graph.

    The affinity array stacks three groups of channels along axis 0:

    1. ``[0, number_of_attractive_channels)`` — attractive grid edges.
    2. ``[number_of_attractive_channels, len(offsets))`` — mutex grid edges.
    3. ``[len(offsets), affinities.shape[0])`` — one channel per semantic
       class, scoring how likely each pixel belongs to that class.

    The first two groups are the regular mutex watershed input; the third
    extends it: clusters can be tagged with the strongest semantic class
    seen for any of their pixels and two clusters carrying different
    semantic tags are forbidden from merging.

    Parameters
    ----------
    affinities:
        Array with shape ``(channels, y, x)`` or ``(channels, z, y, x)``.
        ``channels`` must be strictly greater than ``len(offsets)`` so that
        at least one semantic class is present; otherwise use
        :func:`mutex_watershed`. Supported dtypes are ``float32`` and
        ``float64``; non-contiguous arrays are copied.
    offsets:
        One offset per spatial channel (attractive + mutex), in NumPy axis
        order. Length must equal ``number_of_attractive_channels +
        number_of_mutex_channels``.
    number_of_attractive_channels:
        The first this many spatial channels are attractive; the remaining
        ``len(offsets) - number_of_attractive_channels`` channels are mutex.
    strides, randomized_strides:
        Same semantics as :func:`mutex_watershed`. Apply to mutex channels
        only.
    mask:
        Optional boolean foreground mask with shape ``affinities.shape[1:]``.
        Spatial edges touching ``False`` pixels are skipped, ``False`` pixels
        are labelled as background ``0`` in ``labels`` and as ``mask_label``
        in ``semantic_labels``.
    mask_label:
        Value written to ``semantic_labels`` for masked-out pixels. Defaults
        to ``0``.

    Returns
    -------
    labels:
        ``uint64`` array with shape ``affinities.shape[1:]``. Consecutive
        1-based segmentation labels.
    semantic_labels:
        ``int64`` array with the same shape. Per-pixel semantic class id, or
        ``-1`` for clusters that received no semantic assignment.
    """
    array = np.asarray(affinities)
    if array.ndim not in (3, 4):
        raise ValueError(
            "affinities must have shape (channels, y, x) or "
            f"(channels, z, y, x), got ndim={array.ndim}"
        )

    dtype = array.dtype
    try:
        run = _SEMANTIC_MUTEX_WATERSHED_BY_DTYPE[dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in _SEMANTIC_MUTEX_WATERSHED_BY_DTYPE)
        raise TypeError(
            f"affinities must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    normalized_offsets = [tuple(int(value) for value in offset) for offset in offsets]
    number_of_offsets = len(normalized_offsets)
    number_of_channels = int(array.shape[0])
    spatial_ndim = array.ndim - 1
    if number_of_offsets == 0:
        raise ValueError("offsets must not be empty")
    if number_of_channels <= number_of_offsets:
        raise ValueError(
            "semantic_mutex_watershed requires at least one semantic-class channel, "
            f"got channels={number_of_channels}, len(offsets)={number_of_offsets}; "
            "use mutex_watershed for the non-semantic case"
        )
    if any(len(offset) != spatial_ndim for offset in normalized_offsets):
        raise ValueError(
            "each offset must have length matching the spatial ndim, got "
            f"spatial ndim={spatial_ndim}"
        )
    if number_of_attractive_channels < 0:
        raise ValueError("number_of_attractive_channels must be non-negative")
    if number_of_attractive_channels > number_of_offsets:
        raise ValueError(
            "number_of_attractive_channels must be <= len(offsets)"
        )

    normalized_strides = _normalize_strides(strides, spatial_ndim, randomized_strides)
    contiguous = np.ascontiguousarray(array)
    valid_edges = _compute_valid_edges_semantic(
        contiguous,
        normalized_offsets,
        number_of_attractive_channels,
        number_of_offsets,
        normalized_strides,
        randomized_strides,
        mask,
    )

    labels, semantic_labels = run(
        contiguous,
        valid_edges,
        normalized_offsets,
        number_of_attractive_channels,
        number_of_offsets,
    )
    if mask is not None:
        invalid = np.logical_not(np.asarray(mask))
        labels[invalid] = 0
        semantic_labels[invalid] = mask_label
    return labels, semantic_labels


def _normalize_strides(
    strides: Sequence[int] | None,
    spatial_ndim: int,
    randomized_strides: bool,
) -> tuple[int, ...] | None:
    if strides is None:
        if randomized_strides:
            raise ValueError("randomized_strides requires strides")
        return None

    normalized = tuple(int(stride) for stride in strides)
    if len(normalized) != spatial_ndim:
        raise ValueError(
            "strides length must match the spatial ndim, got "
            f"strides length={len(normalized)}, spatial ndim={spatial_ndim}"
        )
    if any(stride <= 0 for stride in normalized):
        raise ValueError("strides must contain only positive integers")
    return normalized


def _valid_source_slices(
    image_shape: tuple[int, ...],
    offset: tuple[int, ...],
) -> tuple[slice, ...] | None:
    slices = []
    for axis_size, step in zip(image_shape, offset, strict=True):
        if step > 0:
            if step >= axis_size:
                return None
            slices.append(slice(0, axis_size - step))
        elif step < 0:
            if -step >= axis_size:
                return None
            slices.append(slice(-step, axis_size))
        else:
            slices.append(slice(None))
    return tuple(slices)


def _neighbor_slices(
    image_shape: tuple[int, ...],
    offset: tuple[int, ...],
) -> tuple[slice, ...] | None:
    slices = []
    for axis_size, step in zip(image_shape, offset, strict=True):
        if step > 0:
            if step >= axis_size:
                return None
            slices.append(slice(step, axis_size))
        elif step < 0:
            if -step >= axis_size:
                return None
            slices.append(slice(0, axis_size + step))
        else:
            slices.append(slice(None))
    return tuple(slices)


def _compute_valid_edges(
    affinity_shape: tuple[int, ...],
    offsets: Sequence[tuple[int, ...]],
    number_of_attractive_channels: int,
    strides: tuple[int, ...] | None,
    randomized_strides: bool,
    mask: np.ndarray | None,
) -> np.ndarray:
    image_shape = tuple(int(size) for size in affinity_shape[1:])
    valid_edges = np.zeros(affinity_shape, dtype=bool)

    for channel, offset in enumerate(offsets):
        source_slices = _valid_source_slices(image_shape, offset)
        if source_slices is not None:
            valid_edges[(channel,) + source_slices] = True

    if strides is not None:
        stride_edges = np.zeros_like(valid_edges, dtype=bool)
        stride_edges[:number_of_attractive_channels] = True
        if randomized_strides:
            stride_factor = 1.0 / np.prod(strides)
            stride_edges[number_of_attractive_channels:] = (
                np.random.random(
                    valid_edges[number_of_attractive_channels:].shape
                )
                < stride_factor
            )
        else:
            valid_slice = (slice(number_of_attractive_channels, None),) + tuple(
                slice(None, None, stride) for stride in strides
            )
            stride_edges[valid_slice] = True
        valid_edges &= stride_edges

    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.shape != image_shape:
            raise ValueError(
                "mask shape must match affinities spatial shape, got "
                f"mask shape={mask_array.shape}, spatial shape={image_shape}"
            )
        if mask_array.dtype != np.dtype("bool"):
            raise TypeError(f"mask must have dtype bool, got dtype={mask_array.dtype}")

        invalid = np.logical_not(mask_array)
        for channel, offset in enumerate(offsets):
            source_slices = _valid_source_slices(image_shape, offset)
            neighbor_slices = _neighbor_slices(image_shape, offset)
            if source_slices is None or neighbor_slices is None:
                continue
            touches_invalid = invalid[source_slices] | invalid[neighbor_slices]
            channel_valid = valid_edges[(channel,) + source_slices]
            channel_valid[touches_invalid] = False
            valid_edges[(channel,) + source_slices] = channel_valid

    return np.ascontiguousarray(valid_edges, dtype=np.uint8)


def _compute_valid_edges_semantic(
    affinities: np.ndarray,
    offsets: Sequence[tuple[int, ...]],
    number_of_attractive_channels: int,
    number_of_offsets: int,
    strides: tuple[int, ...] | None,
    randomized_strides: bool,
    mask: np.ndarray | None,
) -> np.ndarray:
    # Reuse the spatial-edge logic, then extend the mask to cover the extra
    # semantic-class channels. The spatial helper allocates the full
    # ``affinities.shape`` mask but only writes into the first
    # ``number_of_offsets`` channels, which is exactly what we want here —
    # everything beyond is initialised to ``False`` and managed below.
    valid_edges_uint = _compute_valid_edges(
        tuple(affinities.shape),
        offsets,
        number_of_attractive_channels,
        strides,
        randomized_strides,
        mask,
    )
    valid_edges = valid_edges_uint.astype(bool, copy=True)

    semantic_channels = affinities[number_of_offsets:]
    if semantic_channels.shape[0] > 0:
        per_pixel_max = semantic_channels.max(axis=0, keepdims=True)
        valid_edges[number_of_offsets:] = semantic_channels == per_pixel_max

    if mask is not None:
        invalid = np.logical_not(np.asarray(mask))
        valid_edges[number_of_offsets:, invalid] = False

    return np.ascontiguousarray(valid_edges, dtype=np.uint8)
