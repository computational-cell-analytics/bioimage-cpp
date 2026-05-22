"""Pairwise distances between an embedding tensor and itself under offsets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core


_COMPUTE_EMBEDDING_DISTANCES_BY_NDIM = {
    3: _core._compute_embedding_distances_2d,  # (C, H, W) -> spatial ndim 2
    4: _core._compute_embedding_distances_3d,  # (C, D, H, W) -> spatial ndim 3
}

_SUPPORTED_NORMS = ("l1", "l2", "cosine")


def compute_embedding_distances(
    values: np.ndarray,
    offsets: Sequence[Sequence[int]] | np.ndarray,
    *,
    norm: str = "l2",
    number_of_threads: int = 1,
) -> np.ndarray:
    """Compute pairwise distances between an embedding tensor under offsets.

    For each spatial coordinate ``p`` and offset index ``oi``,
    ``distances[oi, p] = norm(values[:, p], values[:, p + offsets[oi]])``.
    Out-of-bounds positions are left at ``0.0``.

    Parameters
    ----------
    values:
        Embedding tensor with shape ``(C, *spatial)`` and dtype ``float32``.
        ``spatial`` must be 2D or 3D. Non-contiguous arrays are copied to a
        C-contiguous buffer first.
    offsets:
        Shape ``(n_offsets, ndim)``. Each offset is a per-axis displacement
        in NumPy axis order applied at each spatial position to find the
        neighbor.
    norm:
        Distance norm. One of:

        - ``"l1"``  : ``sum_c |values[c, p1] - values[c, p2]|``
        - ``"l2"``  : ``sqrt(sum_c (values[c, p1] - values[c, p2])^2)``
        - ``"cosine"`` : ``1 - dot(a, b) / (||a|| * ||b||)``.
          Yields NaN/Inf for zero-norm channel vectors.
    number_of_threads:
        Number of threads to parallelize over the offset channels.

    Returns
    -------
    distances : np.ndarray
        ``float32`` array of shape ``(n_offsets, *spatial)``.
    """
    array = np.ascontiguousarray(values)
    if array.ndim not in _COMPUTE_EMBEDDING_DISTANCES_BY_NDIM:
        raise ValueError(
            "values must have ndim 3 (C, H, W) or 4 (C, D, H, W), got ndim="
            + str(array.ndim)
        )
    if array.dtype != np.float32:
        raise TypeError(
            f"values must have dtype float32, got dtype={array.dtype}"
        )

    spatial_ndim = array.ndim - 1
    normalized_offsets = [
        [int(value) for value in offset] for offset in np.asarray(offsets).tolist()
    ]
    if len(normalized_offsets) == 0:
        raise ValueError("offsets must not be empty")
    if any(len(offset) != spatial_ndim for offset in normalized_offsets):
        raise ValueError(
            "each offset must have length matching the spatial ndim, got "
            f"spatial ndim={spatial_ndim}"
        )

    norm_str = str(norm).lower()
    if norm_str not in _SUPPORTED_NORMS:
        supported = ", ".join(repr(n) for n in _SUPPORTED_NORMS)
        raise ValueError(
            f"norm must be one of ({supported}), got {norm!r}"
        )

    n_threads = int(number_of_threads)
    if n_threads < 1:
        raise ValueError("number_of_threads must be >= 1")

    run = _COMPUTE_EMBEDDING_DISTANCES_BY_NDIM[array.ndim]
    return run(array, normalized_offsets, norm_str, n_threads)
