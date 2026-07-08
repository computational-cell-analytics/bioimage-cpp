"""Python wrappers for the geodesic-distance bindings.

Geodesic distance is the shortest-path length constrained to a geometry:

- **masks** (regular grid): distances stay inside the nonzero region and never
  cross background voxels (fast-marching / Eikonal formulation, à la
  scikit-fmm).
- **surfaces** (triangle meshes): distances are measured across the mesh
  surface (exact MMP geodesics, à la pygeodesic).

.. note::

    The C++ solvers are not implemented yet. These wrappers validate their
    arguments fully and dispatch to the ``_core`` bindings, which currently
    raise ``RuntimeError("... not yet implemented")``. The reference behaviour
    lives in ``development/distance/``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core
from ._distance import _as_binary_input, _normalize_sampling, _normalize_threads


def _as_coordinates(points: np.ndarray, ndim: int, function: str, name: str) -> np.ndarray:
    """Coerce a point set to a C-contiguous ``(n, ndim)`` int64 array."""
    array = np.ascontiguousarray(points)
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
    return np.ascontiguousarray(array, dtype=np.int64)


def _as_vertex_indices(indices: np.ndarray, function: str, name: str) -> np.ndarray:
    """Coerce vertex indices to a C-contiguous 1-D int64 array."""
    array = np.atleast_1d(np.ascontiguousarray(indices))
    if array.ndim != 1:
        raise ValueError(
            f"{function}: {name} must be 1-D vertex indices, got ndim={array.ndim}"
        )
    return np.ascontiguousarray(array, dtype=np.int64)


def _as_mesh(vertices: np.ndarray, faces: np.ndarray, function: str):
    """Coerce a triangle mesh to ``(vertices float64 (V,3), faces int64 (F,3))``."""
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"{function}: vertices must have shape (n_vertices, 3), "
            f"got shape={vertices.shape}"
        )
    faces = np.ascontiguousarray(faces, dtype=np.int64)
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
    return array


def geodesic_distance_field(
    mask: np.ndarray,
    sources: np.ndarray,
    sampling: float | Sequence[float] | None = None,
    speed: np.ndarray | None = None,
    number_of_threads: int = 1,
) -> np.ndarray:
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
    number_of_threads
        ``0`` uses ``hardware_concurrency``; a positive value pins the thread
        count. Default ``1``.

    Returns
    -------
    np.ndarray
        ``float64`` array of shape ``mask.shape``. Background voxels and voxels
        unreachable from any source are ``+inf``.
    """
    function = "geodesic_distance_field"
    binary = _as_binary_input(mask, function)
    ndim = binary.ndim
    sources_arr = _as_coordinates(sources, ndim, function, "sources")
    sampling_values = _normalize_sampling(sampling, ndim, function)
    speed_arr = _as_speed(speed, tuple(binary.shape), function)
    n_threads = _normalize_threads(number_of_threads, function)
    return _core._geodesic_distance_field_mask(
        binary, sources_arr, sampling_values, speed_arr, n_threads
    )


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
        ``0`` uses ``hardware_concurrency``; a positive value pins the thread
        count. Default ``1``.

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
