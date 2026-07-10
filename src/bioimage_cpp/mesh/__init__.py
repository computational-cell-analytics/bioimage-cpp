"""Triangle-mesh extraction from 3-D scalar volumes."""

from __future__ import annotations

from collections.abc import Sequence
import operator

import numpy as np

from .. import _core


def _as_spacing(spacing: Sequence[float]) -> np.ndarray:
    try:
        values = np.asarray(spacing, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("spacing must consist of three floats") from error
    if values.shape != (3,):
        raise ValueError("spacing must consist of three floats")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("spacing entries must be positive and finite")
    return values


def marching_cubes(
    volume: np.ndarray,
    level: float | None = None,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    gradient_direction: str = "descent",
    step_size: int = 1,
    allow_degenerate: bool = True,
    method: str = "lewiner",
    mask: np.ndarray | None = None,
    pad: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract an isosurface from a three-dimensional scalar volume.

    This follows :func:`skimage.measure.marching_cubes` for its volume,
    level, spacing, winding, stride, degenerate-face, method, and mask
    semantics. ``method="lewiner"`` is the topology-resolving Marching Cubes
    33 implementation and is the default; ``method="lorensen"`` uses the
    original classic 256-case lookup table.

    Parameters
    ----------
    volume:
        Three-dimensional numeric array. It is converted to C-contiguous
        ``float32`` before entering the C++ kernel.
    level:
        Iso-value to extract. ``None`` uses the midpoint of the data range.
    spacing:
        Three physical spacings in NumPy axis order ``(z, y, x)``.
    gradient_direction:
        ``"descent"`` (the default) treats objects as values larger than the
        exterior; ``"ascent"`` reverses triangle winding.
    step_size:
        Sampling stride in voxels. Values above one produce a coarser mesh.
    allow_degenerate:
        If false, remove faces with repeated vertex coordinates.
    method:
        ``"lewiner"`` or ``"lorensen"``.
    mask:
        Optional boolean region of the same shape. Cubes are emitted only in
        its true region; this can intentionally produce open surfaces.
    pad:
        If true, add a one-voxel zero-valued halo before extraction and shift
        returned coordinates back to the original volume origin. This closes
        foreground objects that touch the input boundary. The padded halo is
        enabled in a supplied mask so that its boundary cells can be emitted.

    Returns
    -------
    vertices, faces, normals, values:
        Vertices and normals have shape ``(V, 3)`` in NumPy ``(z, y, x)``
        coordinate order; faces has shape ``(F, 3)`` and dtype ``int32``;
    values has shape ``(V,)``. At unit spacing vertices are ``float32``;
    non-unit spacing follows skimage and produces ``float64`` vertices.
    Normals are normalized gradients accumulated from incident cells and
    ``values`` stores the largest local data range seen at each vertex, as in
    scikit-image. ``gradient_direction`` changes face winding only; spacing
    scales vertices but does not transform normals.

    Raises
    ------
    ValueError
        If shapes or options are invalid, or the requested level is outside
        the input data range.
    RuntimeError
        If no surface intersects the chosen level.
    """
    volume_array = np.asarray(volume)
    if volume_array.ndim != 3:
        raise ValueError(f"Input volume should be a 3D numpy array, got ndim={volume_array.ndim}.")
    if any(size < 2 for size in volume_array.shape):
        raise ValueError("Input array must be at least 2x2x2.")
    if not np.issubdtype(volume_array.dtype, np.number) and volume_array.dtype != np.bool_:
        raise TypeError(f"volume must have a numeric dtype, got dtype={volume_array.dtype}")
    if np.issubdtype(volume_array.dtype, np.complexfloating):
        raise TypeError(f"volume must have a real numeric dtype, got dtype={volume_array.dtype}")
    volume_float = np.ascontiguousarray(volume_array, dtype=np.float32)

    volume_min = float(volume_float.min())
    volume_max = float(volume_float.max())
    if level is None:
        level_float = 0.5 * (volume_min + volume_max)
    else:
        level_float = float(level)
    if not np.isfinite(level_float):
        raise ValueError("level must be finite")
    if level_float < volume_min or level_float > volume_max:
        raise ValueError("Surface level must be within volume data range.")

    spacing_array = _as_spacing(spacing)
    try:
        step = operator.index(step_size)
    except TypeError as error:
        raise TypeError("step_size must be an integer") from error
    if step < 1:
        raise ValueError("step_size must be at least one.")
    if method == "lewiner":
        classic = False
    elif method == "lorensen":
        classic = True
    else:
        raise ValueError("method should be either 'lewiner' or 'lorensen'")
    if gradient_direction == "descent":
        descent = True
    elif gradient_direction == "ascent":
        descent = False
    else:
        raise ValueError(
            "Incorrect input gradient_direction, see marching_cubes documentation."
        )

    mask_array: np.ndarray | None = None
    if mask is not None:
        mask_input = np.asarray(mask)
        if mask_input.shape != volume_array.shape:
            raise ValueError("volume and mask must have the same shape.")
        mask_array = np.ascontiguousarray(mask_input != 0, dtype=np.uint8)

    if bool(pad):
        volume_float = np.pad(volume_float, 1, mode="constant", constant_values=0)
        if mask_array is not None:
            mask_array = np.pad(mask_array, 1, mode="constant", constant_values=True)

    vertices, faces, normals, values = _core._marching_cubes_float32(
        volume_float,
        level_float,
        step,
        classic,
        descent,
        mask_array,
        bool(allow_degenerate),
    )

    if np.array_equal(spacing_array, (1.0, 1.0, 1.0)):
        if bool(pad):
            vertices = vertices - np.ones(3, dtype=np.float32)
        return vertices, faces, normals, values

    vertices = vertices.astype(np.float64, copy=False) * spacing_array
    if bool(pad):
        vertices -= spacing_array
    return vertices, faces, normals, values


__all__ = ["marching_cubes"]
