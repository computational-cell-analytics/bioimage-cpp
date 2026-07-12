"""Python wrappers for the geodesic-distance bindings.

Geodesic distance is the shortest-path length constrained to a geometry:

- **masks** (regular grid): distances stay inside the nonzero region and never
  cross background voxels (fast-marching / Eikonal formulation, à la
  scikit-fmm).
- **surfaces** (triangle meshes): first-order Kimmel--Sethian fast marching
  across the mesh surface. This approximates exact MMP geodesics; error grows
  on very obtuse triangulations because obtuse-angle unfolding is not used.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core
from .._validation import strict_integer_array
from ._distance import _as_binary_input, _normalize_sampling, _normalize_threads


def _as_coordinates(points: np.ndarray, ndim: int, function: str, name: str) -> np.ndarray:
    """Coerce a point set to a C-contiguous ``(n, ndim)`` int64 array."""
    array = np.asarray(points)
    if array.ndim == 1 and array.size == ndim:
        # A single coordinate may be given as a flat (ndim,) vector.
        array = array.reshape(1, ndim)
    if array.ndim != 2:
        raise ValueError(
            f"{function}: {name} must have shape (n, {ndim}), got ndim={array.ndim}"
        )
    if array.shape[1] != ndim:
        raise ValueError(
            f"{function}: {name}.shape[1] must equal the mask ndim ({ndim}), "
            f"got {name}.shape[1]={array.shape[1]}"
        )
    return strict_integer_array(array, name, dtype=np.int64, ndim=2)


def _as_vertex_indices(indices: np.ndarray, function: str, name: str) -> np.ndarray:
    """Coerce vertex indices to a C-contiguous 1-D int64 array."""
    array = np.atleast_1d(np.asarray(indices))
    if array.ndim != 1:
        raise ValueError(
            f"{function}: {name} must be 1-D vertex indices, got ndim={array.ndim}"
        )
    return strict_integer_array(array, name, dtype=np.int64, ndim=1)


def _as_mesh(vertices: np.ndarray, faces: np.ndarray, function: str):
    """Coerce a triangle mesh to ``(vertices float64 (V,3), faces int64 (F,3))``."""
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"{function}: vertices must have shape (n_vertices, 3), "
            f"got shape={vertices.shape}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f"{function}: vertices must contain only finite values")
    faces = strict_integer_array(faces, "faces", dtype=np.int64, ndim=2)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"{function}: faces must have shape (n_faces, 3), got shape={faces.shape}"
        )
    return vertices, faces


def _as_speed(
    speed: np.ndarray | None, expected_shape: tuple[int, ...], function: str
) -> np.ndarray | None:
    """Coerce an optional speed field to a C-contiguous float64 array."""
    if speed is None:
        return None
    array = np.ascontiguousarray(speed, dtype=np.float64)
    if array.shape != tuple(expected_shape):
        raise ValueError(
            f"{function}: speed must have shape {tuple(expected_shape)}, "
            f"got shape={array.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{function}: speed values must be finite and strictly positive")
    return array


def _require_foreground_points(
    mask: np.ndarray, points: np.ndarray, function: str, name: str
) -> None:
    for axis, extent in enumerate(mask.shape):
        if points.size and (
            np.any(points[:, axis] < 0) or np.any(points[:, axis] >= extent)
        ):
            raise ValueError(f"{function}: {name} contains an out of bounds coordinate")
    if points.size and np.any(mask[tuple(points.T)] == 0):
        raise ValueError(f"{function}: {name} must lie inside the foreground mask")


def geodesic_distance_field(
    mask: np.ndarray,
    sources: np.ndarray,
    sampling: float | Sequence[float] | None = None,
    speed: np.ndarray | None = None,
    return_gradient: bool = False,
    number_of_threads: int = 1,
):
    """Geodesic distance field within a mask from a set of source coordinates.

    Computes, for every foreground voxel, the shortest-path distance to the
    nearest source, where paths must stay inside the mask (never crossing
    background). A single source is just a one-row ``sources`` array.

    Parameters
    ----------
    mask
        Binary/label array of any ndim. Nonzero is inside the domain; distances
        propagate only through nonzero voxels. Coerced to C-contiguous
        ``uint8``.
    sources
        Integer array of shape ``(n_sources, ndim)`` (or a flat ``(ndim,)``
        vector for a single source); each row is a voxel coordinate in NumPy
        axis order.
    sampling
        Per-axis voxel spacing. Scalar or per-axis sequence; default 1.0.
    speed
        Optional per-voxel speed, same shape as ``mask``. ``None`` gives
        unit-speed geodesic distance; otherwise the result is the weighted
        travel time.
    return_gradient
        If ``True``, also return the per-axis gradient of the field (see
        Returns). Analogous to :func:`vector_difference_transform`.
    number_of_threads
        Retained for API consistency; a single field solve is serial.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        The distance field, a ``float64`` array of shape ``mask.shape``.
        Background voxels and voxels unreachable from any source are ``+inf``.
        If ``return_gradient`` is ``True``, returns ``(field, gradient)`` where
        ``gradient`` is a ``float32`` array of shape ``(*mask.shape, ndim)``
        holding the first-order upwind gradient ``d(field)/d(axis)`` (channel
        last, NumPy axis order). The gradient points **away from the source**
        (direction of increasing distance) with ``norm ~= 1/speed``; negate it
        to trace back toward the source. It is zero at sources, background, and
        unreachable voxels.
    """
    function = "geodesic_distance_field"
    binary = _as_binary_input(mask, function)
    ndim = binary.ndim
    sources_arr = _as_coordinates(sources, ndim, function, "sources")
    _require_foreground_points(binary, sources_arr, function, "sources")
    sampling_values = _normalize_sampling(sampling, ndim, function)
    speed_arr = _as_speed(speed, tuple(binary.shape), function)
    n_threads = _normalize_threads(number_of_threads, function)
    field, gradient = _core._geodesic_distance_field_mask(
        binary, sources_arr, sampling_values, speed_arr, bool(return_gradient), n_threads
    )
    if return_gradient:
        return field, gradient
    return field


def geodesic_gradient_field(
    mask: np.ndarray,
    sources: np.ndarray,
    sampling: float | Sequence[float] | None = None,
    speed: np.ndarray | None = None,
    number_of_threads: int = 1,
) -> np.ndarray:
    """Return the per-axis gradient of the geodesic distance field within a mask.

    Thin wrapper around :func:`geodesic_distance_field` with
    ``return_gradient=True`` that returns only the gradient. Output has shape
    ``mask.shape + (ndim,)`` and dtype ``float32``; the trailing vector axis
    follows NumPy axis order. Each component is ``d(field)/d(axis)`` pointing
    away from the nearest source (``norm ~= 1/speed``); negate to trace toward
    it (e.g. to feed :func:`bioimage_cpp.flow.compute_flow_density`).
    """
    _, gradient = geodesic_distance_field(
        mask,
        sources,
        sampling=sampling,
        speed=speed,
        return_gradient=True,
        number_of_threads=number_of_threads,
    )
    return gradient


def geodesic_distances(
    mask: np.ndarray,
    points: np.ndarray,
    sampling: float | Sequence[float] | None = None,
    speed: np.ndarray | None = None,
    number_of_threads: int = 1,
) -> np.ndarray:
    """Full pairwise geodesic distance matrix between points within a mask.

    Parameters
    ----------
    mask
        Binary/label array of any ndim; nonzero is inside the domain. Coerced
        to C-contiguous ``uint8``.
    points
        Integer array of shape ``(n_points, ndim)``; each row is a voxel
        coordinate in NumPy axis order.
    sampling
        Per-axis voxel spacing. Scalar or per-axis sequence; default 1.0.
    speed
        Optional per-voxel speed, same shape as ``mask``. ``None`` gives
        unit-speed distances.
    number_of_threads
        ``0`` uses ``hardware_concurrency``; a positive value pins the thread
        count. Default ``1``.

    Returns
    -------
    np.ndarray
        Symmetric ``float64`` matrix of shape ``(n_points, n_points)``. Entry
        ``(i, j)`` is the geodesic distance from ``points[i]`` to ``points[j]``
        within the mask, ``+inf`` when they are not connected inside the domain,
        and ``0`` on the diagonal.
    """
    function = "geodesic_distances"
    binary = _as_binary_input(mask, function)
    ndim = binary.ndim
    points_arr = _as_coordinates(points, ndim, function, "points")
    _require_foreground_points(binary, points_arr, function, "points")
    sampling_values = _normalize_sampling(sampling, ndim, function)
    speed_arr = _as_speed(speed, tuple(binary.shape), function)
    n_threads = _normalize_threads(number_of_threads, function)
    return _core._geodesic_distances_mask(
        binary, points_arr, sampling_values, speed_arr, n_threads
    )


def geodesic_distance_field_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    sources: np.ndarray,
    speed: np.ndarray | None = None,
    number_of_threads: int = 1,
) -> np.ndarray:
    """Geodesic distance field on a triangle-mesh surface from source vertices.

    Parameters
    ----------
    vertices
        Vertex positions of shape ``(n_vertices, 3)``. Coerced to
        C-contiguous ``float64``.
    faces
        Triangle vertex indices of shape ``(n_faces, 3)``. Coerced to
        C-contiguous ``int64``.
    sources
        1-D array of source vertex indices (a scalar/one-element array for a
        single source).
    speed
        Optional per-vertex speed of shape ``(n_vertices,)``. ``None`` gives
        unit-speed geodesic distance.
    number_of_threads
        Retained for API consistency; a single field solve is serial.

    Returns
    -------
    np.ndarray
        ``float64`` array of shape ``(n_vertices,)``. Vertices unreachable from
        any source (a disconnected component) are ``+inf``.
    """
    function = "geodesic_distance_field_mesh"
    vertices_arr, faces_arr = _as_mesh(vertices, faces, function)
    sources_arr = _as_vertex_indices(sources, function, "sources")
    speed_arr = _as_speed(speed, (vertices_arr.shape[0],), function)
    n_threads = _normalize_threads(number_of_threads, function)
    return _core._geodesic_distance_field_mesh(
        vertices_arr, faces_arr, sources_arr, speed_arr, n_threads
    )


def geodesic_distances_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    speed: np.ndarray | None = None,
    number_of_threads: int = 1,
) -> np.ndarray:
    """Full pairwise geodesic distance matrix between mesh vertices.

    Parameters
    ----------
    vertices
        Vertex positions of shape ``(n_vertices, 3)``. Coerced to
        C-contiguous ``float64``.
    faces
        Triangle vertex indices of shape ``(n_faces, 3)``. Coerced to
        C-contiguous ``int64``.
    points
        1-D array of vertex indices.
    speed
        Optional per-vertex speed of shape ``(n_vertices,)``. ``None`` gives
        unit-speed distances.
    number_of_threads
        ``0`` uses ``hardware_concurrency``; a positive value pins the thread
        count. Default ``1``.

    Returns
    -------
    np.ndarray
        Symmetric ``float64`` matrix of shape ``(n_points, n_points)``. Entry
        ``(i, j)`` is the surface geodesic distance from ``points[i]`` to
        ``points[j]``, ``+inf`` when they lie in different connected components,
        and ``0`` on the diagonal.
    """
    function = "geodesic_distances_mesh"
    vertices_arr, faces_arr = _as_mesh(vertices, faces, function)
    points_arr = _as_vertex_indices(points, function, "points")
    speed_arr = _as_speed(speed, (vertices_arr.shape[0],), function)
    n_threads = _normalize_threads(number_of_threads, function)
    return _core._geodesic_distances_mesh(
        vertices_arr, faces_arr, points_arr, speed_arr, n_threads
    )
