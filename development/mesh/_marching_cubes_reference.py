"""scikit-image reference helpers for marching-cubes development checks.

This module is intentionally kept outside the package and test dependency set.
It imports scikit-image lazily, mirrors the public ``pad`` extension, and is
used by the parity and benchmark scripts in this directory.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _skimage_marching_cubes():
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:  # pragma: no cover - development only
        raise ImportError(
            "scikit-image is required for the marching-cubes reference "
            "(`pip install scikit-image`)"
        ) from error
    return marching_cubes


def reference_marching_cubes(
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
):
    """Call scikit-image with the same public contract as ``bic.mesh``."""
    image = np.ascontiguousarray(volume, dtype=np.float32)
    if level is None:
        level = 0.5 * (float(image.min()) + float(image.max()))
    mask_array = None if mask is None else np.ascontiguousarray(np.asarray(mask) != 0)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if pad:
        image = np.pad(image, 1, mode="constant", constant_values=0)
        if mask_array is not None:
            mask_array = np.pad(mask_array, 1, mode="constant", constant_values=True)
    result = _skimage_marching_cubes()(
        image,
        level,
        spacing=spacing_array,
        gradient_direction=gradient_direction,
        step_size=step_size,
        allow_degenerate=allow_degenerate,
        method=method,
        mask=mask_array,
    )
    if pad:
        vertices = result[0] - spacing_array.astype(result[0].dtype, copy=False)
        result = (vertices, *result[1:])
    return result


def _sorted_rows(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(array) == 0:
        return array, np.empty(0, dtype=np.int64)
    order = np.lexsort(tuple(array[:, axis] for axis in range(array.shape[1] - 1, -1, -1)))
    return array[order], order


def _canonical_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    canonical = np.sort(inverse[faces], axis=1)
    canonical, _ = _sorted_rows(canonical)
    return unique_vertices, canonical


def _surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    total = 0.0
    for begin in range(0, len(faces), 100_000):
        triangles = vertices[faces[begin : begin + 100_000]].astype(np.float64, copy=False)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        total += float(np.linalg.norm(cross, axis=1).sum()) * 0.5
    return total


def assert_mesh_matches(actual, reference, *, normal_atol: float = 1e-5) -> None:
    """Assert geometry/topology parity independent of output ordering.

    Normals and local-range values are compared after sorting complete vertex
    records by coordinates. Face winding is tested separately by the package
    tests because canonical triangle comparison intentionally ignores it.
    """
    actual_vertices, actual_faces, actual_normals, actual_values = actual
    reference_vertices, reference_faces, reference_normals, reference_values = reference

    assert actual_vertices.shape == reference_vertices.shape
    assert actual_faces.shape == reference_faces.shape
    assert actual_normals.shape == reference_normals.shape
    assert actual_values.shape == reference_values.shape

    actual_unique, actual_triangles = _canonical_faces(actual_vertices, actual_faces)
    reference_unique, reference_triangles = _canonical_faces(reference_vertices, reference_faces)
    np.testing.assert_allclose(actual_unique, reference_unique, rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(actual_triangles, reference_triangles)
    np.testing.assert_allclose(
        _surface_area(actual_vertices, actual_faces),
        _surface_area(reference_vertices, reference_faces),
        rtol=1e-6,
        atol=1e-8,
    )

    actual_records = np.column_stack(
        (actual_vertices, actual_normals, actual_values)
    ).astype(np.float64, copy=False)
    reference_records = np.column_stack(
        (reference_vertices, reference_normals, reference_values)
    ).astype(np.float64, copy=False)
    actual_records, _ = _sorted_rows(actual_records)
    reference_records, _ = _sorted_rows(reference_records)
    np.testing.assert_allclose(
        actual_records[:, :3], reference_records[:, :3], rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        actual_records[:, 3:6], reference_records[:, 3:6], rtol=0.0, atol=normal_atol
    )
    np.testing.assert_allclose(
        actual_records[:, 6], reference_records[:, 6], rtol=0.0, atol=1e-6
    )
