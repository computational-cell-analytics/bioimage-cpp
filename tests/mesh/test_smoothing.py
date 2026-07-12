import os
import sys

import numpy as np
import pytest

import bioimage_cpp as bic


def test_public_namespace():
    assert hasattr(bic.mesh, "smooth_mesh")
    assert not hasattr(bic.utils, "smooth_mesh")


def _tetrahedron(dtype):
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    normals = verts.copy() + dtype(0.5)
    faces = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
    )
    return verts, normals, faces


def _octahedron():
    verts = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [0, 5, 2],
            [2, 5, 1],
            [1, 5, 3],
            [3, 5, 0],
        ],
        dtype=np.int64,
    )
    return verts, faces


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tetrahedron_one_iteration_collapses_to_centroid(dtype):
    verts, normals, faces = _tetrahedron(dtype)
    out_verts, out_normals = bic.mesh.smooth_mesh(verts, normals, faces, iterations=1)

    assert out_verts.dtype == verts.dtype
    assert out_normals.dtype == normals.dtype
    assert out_verts.shape == verts.shape
    assert out_normals.shape == normals.shape

    centroid_verts = verts.mean(axis=0, dtype=dtype)
    centroid_normals = normals.mean(axis=0, dtype=dtype)
    rtol = 1e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(out_verts, np.broadcast_to(centroid_verts, verts.shape), rtol=rtol)
    np.testing.assert_allclose(out_normals, np.broadcast_to(centroid_normals, normals.shape), rtol=rtol)


def test_iterations_zero_returns_copy_of_inputs():
    verts, normals, faces = _tetrahedron(np.float64)
    out_verts, out_normals = bic.mesh.smooth_mesh(verts, normals, faces, iterations=0)

    np.testing.assert_array_equal(out_verts, verts)
    np.testing.assert_array_equal(out_normals, normals)
    # Outputs must be independent copies (mutation does not affect inputs).
    out_verts[0, 0] = 42.0
    assert verts[0, 0] != 42.0


def test_octahedron_converges_to_centroid():
    verts, faces = _octahedron()
    normals = verts.copy()
    out_verts, _ = bic.mesh.smooth_mesh(verts, normals, faces, iterations=100)
    centroid = verts.mean(axis=0)
    # Every vertex should be near the centroid after enough iterations.
    np.testing.assert_allclose(out_verts, np.broadcast_to(centroid, verts.shape), atol=1e-6)


@pytest.mark.parametrize("dim", [2, 4])
def test_non_3d_feature_vectors(dim):
    rng = np.random.default_rng(42)
    n_verts = 8
    verts = rng.standard_normal((n_verts, dim)).astype(np.float64)
    normals = rng.standard_normal((n_verts, dim)).astype(np.float64)
    faces = np.array(
        [[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [0, 4, 7], [2, 3, 5]],
        dtype=np.int64,
    )
    out_verts, out_normals = bic.mesh.smooth_mesh(verts, normals, faces, iterations=3)
    assert out_verts.shape == (n_verts, dim)
    assert out_normals.shape == (n_verts, dim)


def test_parity_with_iterations_alternation():
    # Run iterations one-at-a-time and compare against a single multi-iteration call.
    verts, normals, faces = _tetrahedron(np.float64)
    cur_v, cur_n = verts, normals
    for _ in range(5):
        cur_v, cur_n = bic.mesh.smooth_mesh(cur_v, cur_n, faces, iterations=1)
    multi_v, multi_n = bic.mesh.smooth_mesh(verts, normals, faces, iterations=5)
    np.testing.assert_allclose(multi_v, cur_v, rtol=1e-12)
    np.testing.assert_allclose(multi_n, cur_n, rtol=1e-12)


def test_threading_matches_serial():
    rng = np.random.default_rng(7)
    n_verts = 256
    verts = rng.standard_normal((n_verts, 3)).astype(np.float64)
    normals = rng.standard_normal((n_verts, 3)).astype(np.float64)
    # Triangle strip across the vertices.
    faces = np.stack(
        [np.arange(0, n_verts - 2), np.arange(1, n_verts - 1), np.arange(2, n_verts)],
        axis=1,
    ).astype(np.int64)

    serial = bic.mesh.smooth_mesh(verts, normals, faces, iterations=8, n_threads=1)
    parallel = bic.mesh.smooth_mesh(verts, normals, faces, iterations=8, n_threads=4)
    auto = bic.mesh.smooth_mesh(verts, normals, faces, iterations=8, n_threads=0)
    np.testing.assert_array_equal(serial[0], parallel[0])
    np.testing.assert_array_equal(serial[1], parallel[1])
    np.testing.assert_array_equal(serial[0], auto[0])
    np.testing.assert_array_equal(serial[1], auto[1])


def test_non_contiguous_inputs_are_accepted():
    verts, normals, faces = _tetrahedron(np.float64)
    # Make non-contiguous views by replicating and slicing.
    big_verts = np.repeat(verts, 2, axis=0)
    big_normals = np.repeat(normals, 2, axis=0)
    verts_view = big_verts[::2]
    normals_view = big_normals[::2]
    assert not verts_view.flags.c_contiguous

    out_verts, out_normals = bic.mesh.smooth_mesh(verts_view, normals_view, faces, iterations=1)
    expected_v, expected_n = bic.mesh.smooth_mesh(verts, normals, faces, iterations=1)
    np.testing.assert_array_equal(out_verts, expected_v)
    np.testing.assert_array_equal(out_normals, expected_n)


def test_faces_dtype_is_normalised_to_int64():
    verts, normals, faces = _tetrahedron(np.float64)
    out_int64, _ = bic.mesh.smooth_mesh(verts, normals, faces.astype(np.int64), iterations=1)
    out_int32, _ = bic.mesh.smooth_mesh(verts, normals, faces.astype(np.int32), iterations=1)
    out_uint32, _ = bic.mesh.smooth_mesh(verts, normals, faces.astype(np.uint32), iterations=1)
    np.testing.assert_array_equal(out_int64, out_int32)
    np.testing.assert_array_equal(out_int64, out_uint32)


def test_rejects_mismatched_shapes():
    verts = np.zeros((4, 3), dtype=np.float64)
    normals_bad = np.zeros((4, 2), dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="same shape as verts"):
        bic.mesh.smooth_mesh(verts, normals_bad, faces, iterations=1)


def test_rejects_1d_verts():
    verts = np.zeros(4, dtype=np.float64)
    normals = verts.copy()
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="verts must have ndim=2"):
        bic.mesh.smooth_mesh(verts, normals, faces, iterations=1)


def test_rejects_bad_faces_shape():
    verts, normals, _ = _tetrahedron(np.float64)
    bad_faces = np.array([[0, 1, 2, 3]], dtype=np.int64)
    with pytest.raises(ValueError, match=r"faces must have shape"):
        bic.mesh.smooth_mesh(verts, normals, bad_faces, iterations=1)


def test_rejects_out_of_range_faces():
    verts, normals, _ = _tetrahedron(np.float64)
    bad_faces = np.array([[0, 1, 10]], dtype=np.int64)
    with pytest.raises(ValueError, match=r"\[0, n_verts\)"):
        bic.mesh.smooth_mesh(verts, normals, bad_faces, iterations=1)


def test_rejects_negative_iterations():
    verts, normals, faces = _tetrahedron(np.float64)
    with pytest.raises(ValueError, match="iterations must be non-negative"):
        bic.mesh.smooth_mesh(verts, normals, faces, iterations=-1)


def test_rejects_negative_n_threads():
    verts, normals, faces = _tetrahedron(np.float64)
    with pytest.raises(ValueError, match="n_threads must be non-negative"):
        bic.mesh.smooth_mesh(verts, normals, faces, iterations=1, n_threads=-1)


def test_rejects_unsupported_verts_dtype():
    verts = np.zeros((4, 3), dtype=np.int32)
    normals = verts.copy()
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(TypeError, match="verts must have one of dtypes"):
        bic.mesh.smooth_mesh(verts, normals, faces, iterations=1)


def test_rejects_mismatched_verts_normals_dtype():
    verts = np.zeros((4, 3), dtype=np.float32)
    normals = np.zeros((4, 3), dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(TypeError, match="same dtype"):
        bic.mesh.smooth_mesh(verts, normals, faces, iterations=1)


def test_parity_with_python_reference():
    """Compare against the nifty-based Python reference. Skipped if nifty missing.

    Only one iteration is compared. The reference has an aliasing quirk
    (``current_verts = new_verts`` makes them refer to the same buffer) that
    turns iterations 1+ into in-place Gauss-Seidel smoothing, whereas this
    implementation does textbook Jacobi smoothing (independent read/write
    buffers). The two agree exactly at ``iterations=1``.
    """
    pytest.importorskip("nifty")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dev_path = os.path.join(repo_root, "development", "mesh")
    sys.path.insert(0, dev_path)
    try:
        from _mesh_smoothing_reference import smooth_mesh as smooth_mesh_reference
    finally:
        sys.path.remove(dev_path)

    rng = np.random.default_rng(2026)
    scipy_spatial = pytest.importorskip("scipy.spatial")
    n_points = 80
    raw = rng.standard_normal((n_points, 3))
    points = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    hull = scipy_spatial.ConvexHull(points)
    verts = points.astype(np.float64)
    faces = hull.simplices.astype(np.int64)
    normals = verts.copy()

    out_v, out_n = bic.mesh.smooth_mesh(verts, normals, faces, iterations=1)
    ref_v, ref_n = smooth_mesh_reference(verts, normals, faces, 1)
    np.testing.assert_allclose(out_v, ref_v, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out_n, ref_n, rtol=1e-10, atol=1e-12)
