import numpy as np
import pytest

import bioimage_cpp as bic


def _sphere_mesh(dtype=np.float32):
    n = 22
    center = (n - 1) / 2.0
    z, y, x = np.ogrid[:n, :n, :n]
    volume = (
        (z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2 <= 7.0**2
    ).astype(np.uint8)
    vertices, faces, _, values = bic.mesh.marching_cubes(
        volume, 0.5, allow_degenerate=False
    )
    return vertices.astype(dtype), faces, values


def _edge_counts(faces):
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return counts


def _euler_characteristic(vertices, faces):
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    return len(vertices) - len(np.unique(edges, axis=0)) + len(faces)


def _grid_mesh(size=6, dtype=np.float64):
    y, x = np.mgrid[:size, :size]
    vertices = np.stack([np.zeros_like(x), y, x], axis=-1).reshape(-1, 3).astype(dtype)
    faces = []
    for row in range(size - 1):
        for col in range(size - 1):
            first = row * size + col
            faces.append((first, first + size, first + 1))
            faces.append((first + 1, first + size, first + size + 1))
    return vertices, np.asarray(faces, dtype=np.int64)


def _tetrahedron(dtype=np.float64):
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64
    )
    return vertices, faces


def _reference_normals(vertices, faces):
    triangles = vertices[faces].astype(np.float64)
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals = np.zeros(vertices.shape, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def test_simplifies_closed_marching_cubes_mesh_and_preserves_topology():
    vertices, faces, values = _sphere_mesh()
    target = int(np.ceil(0.5 * len(faces)))

    out_vertices, out_faces, out_normals, out_values = bic.mesh.simplify_mesh(
        vertices, faces, 0.5, values=values
    )

    assert len(out_faces) <= target
    assert len(out_vertices) < len(vertices)
    assert out_vertices.dtype == out_normals.dtype == np.float32
    assert out_faces.dtype == np.int64
    assert out_values.dtype == values.dtype
    assert out_values.shape == (len(out_vertices),)
    assert set(_edge_counts(out_faces)) == {2}
    assert _euler_characteristic(out_vertices, out_faces) == 2
    np.testing.assert_allclose(
        np.linalg.norm(out_normals, axis=1), 1.0, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        out_normals, _reference_normals(out_vertices, out_faces), rtol=2e-6, atol=2e-6
    )
    assert out_values.min() >= values.min()
    assert out_values.max() <= values.max()


def test_open_boundary_remains_a_single_manifold_loop():
    vertices, faces = _grid_mesh()
    out_vertices, out_faces, _, out_values = bic.mesh.simplify_mesh(
        vertices, faces, 0.4
    )

    counts = _edge_counts(out_faces)
    assert set(counts) == {1, 2}
    assert _euler_characteristic(out_vertices, out_faces) == 1
    assert out_values is None

    edges = np.concatenate(
        [out_faces[:, [0, 1]], out_faces[:, [1, 2]], out_faces[:, [2, 0]]],
        axis=0,
    )
    edges.sort(axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[edge_counts == 1]
    degrees = np.bincount(boundary.ravel(), minlength=len(out_vertices))
    np.testing.assert_array_equal(degrees[degrees > 0], 2)


def test_tetrahedron_stops_before_topology_would_change():
    vertices, faces = _tetrahedron()
    output = bic.mesh.simplify_mesh(vertices, faces, 0.9)
    np.testing.assert_array_equal(output[0], vertices)
    np.testing.assert_array_equal(output[1], faces)


@pytest.mark.parametrize("vertex_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value_dtype", [np.float32, np.float64])
def test_vertex_and_value_dtypes_are_preserved(vertex_dtype, value_dtype):
    vertices, faces = _grid_mesh(dtype=vertex_dtype)
    values = np.linspace(0.0, 1.0, len(vertices), dtype=value_dtype)
    out_vertices, out_faces, out_normals, out_values = bic.mesh.simplify_mesh(
        vertices[::1], faces.astype(np.int32), 0.2, values=values
    )
    assert out_vertices.dtype == out_normals.dtype == vertex_dtype
    assert out_faces.dtype == np.int64
    assert out_values.dtype == value_dtype


def test_noncontiguous_inputs_are_normalized_at_python_boundary():
    vertices, faces = _grid_mesh()
    values = np.linspace(0.0, 1.0, len(vertices))
    large_vertices = np.repeat(vertices, 2, axis=0)
    large_values = np.repeat(values, 2)
    large_faces = np.empty((len(faces), 6), dtype=np.int64)
    large_faces[:, ::2] = faces
    vertex_view = large_vertices[::2]
    face_view = large_faces[:, ::2]
    value_view = large_values[::2]
    assert not vertex_view.flags.c_contiguous
    assert not face_view.flags.c_contiguous
    assert not value_view.flags.c_contiguous

    expected = bic.mesh.simplify_mesh(vertices, faces, 0.3, values=values)
    actual = bic.mesh.simplify_mesh(
        vertex_view, face_view, 0.3, values=value_view
    )
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_reduction_zero_is_a_geometry_copy_and_recomputes_normals():
    vertices, faces = _tetrahedron(np.float32)
    out_vertices, out_faces, out_normals, out_values = bic.mesh.simplify_mesh(
        vertices, faces, 0.0
    )
    np.testing.assert_array_equal(out_vertices, vertices)
    np.testing.assert_array_equal(out_faces, faces)
    np.testing.assert_allclose(
        out_normals, _reference_normals(vertices, faces), rtol=2e-6, atol=2e-6
    )
    assert out_values is None


def test_output_is_deterministic():
    vertices, faces, values = _sphere_mesh(np.float64)
    first = bic.mesh.simplify_mesh(
        vertices, faces, 0.35, values=values, feature_angle=30.0, feature_weight=20.0
    )
    second = bic.mesh.simplify_mesh(
        vertices, faces, 0.35, values=values, feature_angle=30.0, feature_weight=20.0
    )
    for actual, expected in zip(first, second, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_soft_feature_weight_improves_fold_preservation():
    size = 9
    y, x = np.mgrid[:size, :size]
    vertices = np.stack([np.abs(x - 4), y, x], axis=-1).reshape(-1, 3).astype(np.float64)
    faces = []
    for row in range(size - 1):
        for col in range(size - 1):
            first = row * size + col
            faces.append((first, first + size, first + 1))
            faces.append((first + 1, first + size, first + size + 1))
    faces = np.asarray(faces, dtype=np.int64)

    unprotected = bic.mesh.simplify_mesh(
        vertices, faces, 0.65, feature_angle=20.0, feature_weight=0.0
    )
    protected = bic.mesh.simplify_mesh(
        vertices, faces, 0.65, feature_angle=20.0, feature_weight=100.0
    )
    def fold_error(points):
        return np.max(np.abs(points[:, 0] - np.abs(points[:, 2] - 4.0)))

    assert not np.array_equal(unprotected[0], protected[0])
    assert fold_error(protected[0]) <= fold_error(unprotected[0])


@pytest.mark.parametrize("reduction", [-0.1, 1.0, np.nan])
def test_rejects_invalid_reduction(reduction):
    vertices, faces = _tetrahedron()
    with pytest.raises(ValueError, match="reduction"):
        bic.mesh.simplify_mesh(vertices, faces, reduction)


def test_rejects_bad_values_and_feature_options():
    vertices, faces = _tetrahedron()
    with pytest.raises(ValueError, match="values must have shape"):
        bic.mesh.simplify_mesh(vertices, faces, 0.2, values=np.zeros(3))
    with pytest.raises(TypeError, match="values must have dtype"):
        bic.mesh.simplify_mesh(vertices, faces, 0.2, values=np.zeros(4, np.int32))
    with pytest.raises(ValueError, match="feature_angle"):
        bic.mesh.simplify_mesh(vertices, faces, 0.2, feature_angle=181)
    with pytest.raises(ValueError, match="feature_weight"):
        bic.mesh.simplify_mesh(vertices, faces, 0.2, feature_weight=-1)


def test_rejects_degenerate_inconsistently_oriented_and_nonmanifold_meshes():
    vertices, faces = _tetrahedron()

    degenerate = faces.copy()
    degenerate[0] = [0, 0, 1]
    with pytest.raises(ValueError, match="repeated vertex"):
        bic.mesh.simplify_mesh(vertices, degenerate, 0.2)

    inconsistent = faces.copy()
    inconsistent[0] = inconsistent[0, ::-1]
    with pytest.raises(ValueError, match="consistent winding"):
        bic.mesh.simplify_mesh(vertices, inconsistent, 0.2)

    nonmanifold_vertices = np.vstack([vertices, [[0.0, -1.0, 0.0]]])
    nonmanifold_faces = np.vstack([faces, [[0, 1, 4]]])
    with pytest.raises(ValueError, match="more than two faces"):
        bic.mesh.simplify_mesh(nonmanifold_vertices, nonmanifold_faces, 0.2)
