"""Pairwise boolean affinities from a label volume."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import numpy as np

from .. import _core
from .._validation import strict_index, strict_offsets


_COMPUTE_AFFINITIES_2D_BY_DTYPE = {
    np.dtype("uint32"): _core._compute_affinities_2d_uint32,
    np.dtype("uint64"): _core._compute_affinities_2d_uint64,
    np.dtype("int32"): _core._compute_affinities_2d_int32,
    np.dtype("int64"): _core._compute_affinities_2d_int64,
}

_COMPUTE_AFFINITIES_3D_BY_DTYPE = {
    np.dtype("uint32"): _core._compute_affinities_3d_uint32,
    np.dtype("uint64"): _core._compute_affinities_3d_uint64,
    np.dtype("int32"): _core._compute_affinities_3d_int32,
    np.dtype("int64"): _core._compute_affinities_3d_int64,
}


@overload
def compute_affinities(
    labels: np.ndarray,
    offsets: Sequence[Sequence[int]] | np.ndarray,
    *,
    ignore_label: int | None = None,
    return_mask: bool = True,
    number_of_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]: ...


def compute_affinities(
    labels: np.ndarray,
    offsets: Sequence[Sequence[int]] | np.ndarray,
    *,
    ignore_label: int | None = None,
    return_mask: bool = True,
    number_of_threads: int = 1,
):
    """Compute boolean pairwise affinities from a label volume.

    For each spatial coordinate ``c`` and offset index ``oi``,
    ``affinities[oi, c]`` is ``1.0`` if ``labels[c] == labels[c + offsets[oi]]``
    (the two voxels are in the same cluster) and ``0.0`` otherwise.

    Parameters
    ----------
    labels:
        2D or 3D integer label volume. Supported dtypes are ``uint32``,
        ``uint64``, ``int32``, ``int64``. Non-contiguous arrays are copied
        to a C-contiguous buffer first.
    offsets:
        Shape ``(n_offsets, ndim)``. Each offset is a per-axis displacement,
        in NumPy axis order, applied at each voxel to find the neighbor.
    ignore_label:
        If given, any pair where either endpoint has this label produces
        ``affinity = 0`` and ``mask = 0`` (treated as out-of-volume).
    return_mask:
        When ``True`` (default), also return a ``uint8`` validity mask of
        the same shape as the affinities: ``1`` for in-bounds non-ignored
        pairs, ``0`` otherwise. Set to ``False`` to skip the allocation
        when only the affinities are needed.
    number_of_threads:
        Number of threads to parallelize over the offset channels.

    Returns
    -------
    affinities : np.ndarray
        ``float32`` array of shape ``(n_offsets, *labels.shape)``.
    mask : np.ndarray, only if ``return_mask`` is ``True``
        ``uint8`` array of shape ``(n_offsets, *labels.shape)``.
    """
    array = np.ascontiguousarray(labels)
    if array.ndim not in (2, 3):
        raise ValueError(
            "labels must be 2D or 3D, got ndim=" + str(array.ndim)
        )

    table = (
        _COMPUTE_AFFINITIES_2D_BY_DTYPE if array.ndim == 2
        else _COMPUTE_AFFINITIES_3D_BY_DTYPE
    )
    try:
        run = table[array.dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in table)
        raise TypeError(
            f"labels must have one of dtypes ({supported}), got dtype={array.dtype}"
        ) from error

    normalized_offsets = strict_offsets(offsets, array.ndim)
    if len(normalized_offsets) == 0:
        raise ValueError("offsets must not be empty")
    if any(len(offset) != array.ndim for offset in normalized_offsets):
        raise ValueError(
            "each offset must have length matching the spatial ndim, got "
            f"spatial ndim={array.ndim}"
        )

    n_threads = strict_index(number_of_threads, "number_of_threads", minimum=1)

    if ignore_label is None:
        typed_ignore: int | None = None
    else:
        typed_ignore = int(ignore_label)

    affs, mask = run(
        array,
        normalized_offsets,
        typed_ignore,
        bool(return_mask),
        n_threads,
    )
    if return_mask:
        return affs, mask
    return affs
