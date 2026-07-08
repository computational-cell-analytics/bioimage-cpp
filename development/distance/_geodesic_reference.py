"""Reference geodesic-distance implementations for development cross-checks.

These are the oracles that ``bic.distance`` geodesic functions are validated
against. There is one backend per geometry, because scikit-fmm only supports
regular Cartesian grids:

- **masks**  -> scikit-fmm (``skfmm``): fast marching / Eikonal solver.
- **meshes** -> pygeodesic: exact Mitchell-Mount-Papadimitriou (MMP) geodesics.

Not part of the pytest suite; requires ``scikit-fmm`` and/or ``pygeodesic``.
Importing this module is cheap (numpy only); each function imports its backend
lazily and raises a clear ``ImportError`` if the backend is missing.

Note on the scikit-fmm point-source idiom: to get the distance to a set of
seed voxels, we set ``phi = +1`` everywhere and ``phi = -1`` at the seeds and
take ``|skfmm.distance(phi)|``. The zero contour then sits roughly half a cell
away from each seed, so the reference is offset from a "distance is exactly 0 at
the seed" convention by ~0.5 * dx near the sources. Compare with a tolerance,
or exclude the immediate neighbourhood of the seeds, when checking against an
implementation that initialises the seeds to exactly 0.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# masks: scikit-fmm
# --------------------------------------------------------------------------- #


def _skfmm():
    try:
        import skfmm
    except ImportError as error:  # pragma: no cover - dev script
        raise ImportError(
            "scikit-fmm is required for the mask geodesic reference "
            "(`pip install scikit-fmm`)"
        ) from error
    return skfmm


def _normalize_dx(sampling, ndim):
    if sampling is None:
        return 1.0
    if np.isscalar(sampling):
        return float(sampling)
    dx = [float(s) for s in sampling]
    if len(dx) != ndim:
        raise ValueError(f"sampling must have length {ndim}, got {len(dx)}")
    return dx


def reference_geodesic_field_mask(mask, sources, sampling=None, speed=None):
    """Geodesic distance field within a mask from a set of source coordinates.

    Mirrors ``bic.distance.geodesic_distance_field``: for every voxel, the
    shortest-path distance to the nearest source, constrained to the nonzero
    region of ``mask``. Voxels outside the mask (and voxels unreachable from any
    source) are returned as ``+inf``.
    """
    skfmm = _skfmm()
    mask = np.ascontiguousarray(mask) != 0
    sources = np.atleast_2d(np.asarray(sources, dtype=np.int64))
    dx = _normalize_dx(sampling, mask.ndim)

    phi = np.ones(mask.shape, dtype=np.float64)
    phi[tuple(sources.T)] = -1.0
    phi = np.ma.MaskedArray(phi, mask=~mask)

    if speed is None:
        field = np.ma.abs(skfmm.distance(phi, dx=dx))
    else:
        speed = np.ascontiguousarray(speed, dtype=np.float64)
        field = skfmm.travel_time(phi, speed, dx=dx)

    out = np.full(mask.shape, np.inf, dtype=np.float64)
    field_data = np.ma.getdata(field)
    valid = ~np.ma.getmaskarray(field)
    out[valid] = field_data[valid]
    return out


def reference_geodesic_distances_mask(mask, points, sampling=None, speed=None):
    """Full pairwise geodesic distance matrix between points within a mask.

    Runs the field solve once per point and reads the field at the other
    points. The result is symmetrized (per-source solves can differ by a tiny
    numerical amount) with a zero diagonal.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.int64))
    n = len(points)
    out = np.full((n, n), np.inf, dtype=np.float64)
    index = tuple(points.T)
    for i in range(n):
        field = reference_geodesic_field_mask(mask, points[i][None, :], sampling, speed)
        out[i] = field[index]
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    return out


# --------------------------------------------------------------------------- #
# meshes: pygeodesic (exact MMP)
# --------------------------------------------------------------------------- #


def _pygeodesic():
    try:
        import pygeodesic.geodesic as geodesic
    except ImportError as error:  # pragma: no cover - dev script
        raise ImportError(
            "pygeodesic is required for the mesh geodesic reference "
            "(`pip install pygeodesic`)"
        ) from error
    return geodesic


def _mesh_algorithm(vertices, faces):
    geodesic = _pygeodesic()
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    return geodesic.PyGeodesicAlgorithmExact(vertices, faces)


def reference_geodesic_field_mesh(vertices, faces, sources):
    """Geodesic distance field on a triangle mesh from a set of source vertices.

    Mirrors ``bic.distance.geodesic_distance_field_mesh``. Returns a
    ``(n_vertices,)`` array of surface geodesic distance to the nearest source.
    """
    algo = _mesh_algorithm(vertices, faces)
    sources = np.atleast_1d(np.asarray(sources, dtype=np.int32))
    distances, _ = algo.geodesicDistances(sources, None)
    return np.asarray(distances, dtype=np.float64)


def reference_geodesic_distances_mesh(vertices, faces, points):
    """Full pairwise geodesic distance matrix between mesh vertices."""
    algo = _mesh_algorithm(vertices, faces)
    points = np.atleast_1d(np.asarray(points, dtype=np.int32))
    n = len(points)
    out = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n):
        distances, _ = algo.geodesicDistances(points[i : i + 1], points)
        out[i] = np.asarray(distances, dtype=np.float64)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    return out
