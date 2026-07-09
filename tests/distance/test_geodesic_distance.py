"""Tests for the geodesic-distance solvers (masks + triangle meshes).

The solvers use a first-order fast marching method, so these tests assert the
properties a correct geodesic solver must satisfy (source distance 0,
monotonicity, ``geodesic >= euclidean``, exact behaviour along grid axes, exact
speed scaling, symmetry of pairwise matrices, ``+inf`` for unreachable regions)
rather than tight agreement with an exact Euclidean field (which first-order FMM
overestimates by up to ~20% near diagonals). Tight numeric agreement with the
scikit-fmm / pygeodesic references lives in ``development/distance/`` and in the
optional, dependency-guarded cross-checks at the end of this file.
"""

import numpy as np
import pytest

import bioimage_cpp as bic

INF = np.inf


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def euclidean_from(shape, source, sampling=None):
    """Analytic Euclidean distance from a single source voxel over a full grid."""
    sampling = np.ones(len(shape)) if sampling is None else np.asarray(sampling, float)
    coords = np.indices(shape, dtype=np.float64)
    total = np.zeros(shape, dtype=np.float64)
    for axis, s in enumerate(source):
        total += ((coords[axis] - s) * sampling[axis]) ** 2
    return np.sqrt(total)


def flat_grid_mesh(n):
    """A planar (z=0) triangulated ``n x n`` grid; geodesic == 2D Euclidean."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float64)

    def vid(i, j):
        return i * n + j

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([vid(i, j), vid(i + 1, j), vid(i, j + 1)])
            faces.append([vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    return verts, np.array(faces, np.int64), vid


def sphere_mesh(n_points=300, radius=5.0, seed=0):
    """A closed sphere mesh from the convex hull of points on the sphere."""
    hull = pytest.importorskip("scipy.spatial").ConvexHull
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_points, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius
    faces = hull(pts).simplices.astype(np.int64)
    return np.ascontiguousarray(pts, np.float64), np.ascontiguousarray(faces)


# --------------------------------------------------------------------------- #
# mask: field
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", [(25, 25), (9, 10, 11)])
def test_mask_field_basic_properties(shape):
    mask = np.ones(shape, np.uint8)
    source = tuple(0 for _ in shape)
    field = bic.distance.geodesic_distance_field(mask, np.array([source], np.int64))

    assert field.shape == shape
    assert field.dtype == np.float64
    assert field[source] == 0.0
    assert np.all(np.isfinite(field))
    # geodesic in an obstacle-free domain is >= the straight-line distance
    eucl = euclidean_from(shape, source)
    assert np.all(field >= eucl - 1e-9)


def test_mask_field_axis_aligned_exact():
    mask = np.ones((30, 30), np.uint8)
    field = bic.distance.geodesic_distance_field(mask, np.array([[0, 0]], np.int64))
    np.testing.assert_allclose(field[0, :], np.arange(30), atol=1e-9)
    np.testing.assert_allclose(field[:, 0], np.arange(30), atol=1e-9)


def test_mask_field_farfield_close_to_euclidean():
    mask = np.ones((60, 60), np.uint8)
    field = bic.distance.geodesic_distance_field(mask, np.array([[0, 0]], np.int64))
    eucl = euclidean_from((60, 60), (0, 0))
    far = eucl > 15
    rel = np.abs(field[far] - eucl[far]) / eucl[far]
    assert rel.max() < 0.1  # first-order FMM error is modest in the far field


def test_mask_anisotropic_sampling_axis_exact():
    mask = np.ones((20, 20), np.uint8)
    field = bic.distance.geodesic_distance_field(
        mask, np.array([[0, 0]], np.int64), sampling=(2.0, 0.5)
    )
    np.testing.assert_allclose(field[:, 0], 2.0 * np.arange(20), atol=1e-9)
    np.testing.assert_allclose(field[0, :], 0.5 * np.arange(20), atol=1e-9)


def test_mask_speed_scaling_is_exact():
    mask = np.ones((32, 32), np.uint8)
    src = np.array([[3, 5]], np.int64)
    base = bic.distance.geodesic_distance_field(mask, src)
    scaled = bic.distance.geodesic_distance_field(mask, src, speed=2.0 * np.ones((32, 32)))
    np.testing.assert_allclose(scaled, base / 2.0, atol=1e-9)


def test_mask_multi_source_monotone():
    mask = np.ones((30, 30), np.uint8)
    a = bic.distance.geodesic_distance_field(mask, np.array([[0, 0]], np.int64))
    b = bic.distance.geodesic_distance_field(mask, np.array([[29, 29]], np.int64))
    both = bic.distance.geodesic_distance_field(
        mask, np.array([[0, 0], [29, 29]], np.int64)
    )
    # adding a source can only lower arrival times, so the two-source field is
    # bounded above by the pointwise min of the single-source fields; it dips
    # slightly below near the medial axis where the update stencil mixes fronts.
    assert np.all(both <= np.minimum(a, b) + 1e-9)
    np.testing.assert_allclose(both, np.minimum(a, b), atol=0.5)


def test_mask_single_source_as_flat_vector():
    mask = np.ones((10, 10), np.uint8)
    field = bic.distance.geodesic_distance_field(mask, np.array([2, 3], np.int64))
    assert field[2, 3] == 0.0


def test_mask_obstacle_detour_and_unreachable():
    # Wall across the middle with a gap on the right; source top-left.
    mask = np.ones((21, 21), np.uint8)
    mask[10, :18] = 0  # gap at columns 18, 19, 20
    field = bic.distance.geodesic_distance_field(mask, np.array([[0, 0]], np.int64))

    # wall voxels are background -> +inf
    assert np.isinf(field[10, 0])
    # target directly across the wall must detour around the gap
    target = (20, 0)
    straight = np.hypot(20, 0)
    assert np.isfinite(field[target])
    assert field[target] > straight + 1.0
    # the detour is bounded below by the best two-leg path through a gap cell
    gaps = [(10, c) for c in range(18, 21)]
    ref = min(np.hypot(*np.subtract((0, 0), g)) + np.hypot(*np.subtract(g, target)) for g in gaps)
    assert 0.95 * ref <= field[target] <= 1.25 * ref


def test_mask_isolated_region_is_inf():
    mask = np.zeros((10, 10), np.uint8)
    mask[0:3, 0:3] = 1  # region containing the source
    mask[7:10, 7:10] = 1  # disconnected region
    field = bic.distance.geodesic_distance_field(mask, np.array([[0, 0]], np.int64))
    assert np.all(np.isinf(field[7:10, 7:10]))
    assert np.all(np.isfinite(field[0:3, 0:3]))


# --------------------------------------------------------------------------- #
# mask: pairwise
# --------------------------------------------------------------------------- #


def test_mask_pairwise_symmetric_zero_diagonal():
    mask = np.ones((20, 20), np.uint8)
    points = np.array([[0, 0], [19, 0], [0, 19], [19, 19], [10, 10]], np.int64)
    D = bic.distance.geodesic_distances(mask, points)
    assert D.shape == (5, 5)
    assert D.dtype == np.float64
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
    np.testing.assert_allclose(D, D.T, atol=1e-9)
    assert np.all(D >= 0.0)


def test_mask_pairwise_matches_field_rows():
    mask = np.ones((18, 18), np.uint8)
    points = np.array([[0, 0], [17, 5], [3, 12]], np.int64)
    D = bic.distance.geodesic_distances(mask, points)
    for i, p in enumerate(points):
        field = bic.distance.geodesic_distance_field(mask, p[None, :])
        row = np.array([field[tuple(q)] for q in points])
        np.testing.assert_allclose(D[i], row, atol=1e-9)


# --------------------------------------------------------------------------- #
# mask: validation
# --------------------------------------------------------------------------- #


def test_mask_invalid_arguments():
    mask = np.ones((8, 8), np.uint8)
    with pytest.raises(ValueError, match="out of bounds"):
        bic.distance.geodesic_distance_field(mask, np.array([[8, 0]], np.int64))
    with pytest.raises(ValueError, match="out of bounds"):
        bic.distance.geodesic_distance_field(mask, np.array([[-1, 0]], np.int64))
    with pytest.raises(ValueError, match="speed"):
        bic.distance.geodesic_distance_field(
            mask, np.array([[0, 0]], np.int64), speed=np.ones((3, 3))
        )
    with pytest.raises(ValueError, match="sources.shape"):
        bic.distance.geodesic_distance_field(mask, np.zeros((1, 3), np.int64))


# --------------------------------------------------------------------------- #
# mask: gradient of the field
# --------------------------------------------------------------------------- #


def _interior_mask(shape, source, margin=3, min_radius=8):
    """Voxels away from the source and the array border (smooth-field region)."""
    coords = np.indices(shape, dtype=np.float64)
    eucl = np.sqrt(sum((coords[a] - source[a]) ** 2 for a in range(len(shape))))
    sel = eucl > min_radius
    for a, size in enumerate(shape):
        sel &= (coords[a] > margin) & (coords[a] < size - 1 - margin)
    return sel


def test_mask_gradient_shape_dtype_and_backcompat():
    mask = np.ones((20, 20), np.uint8)
    src = np.array([[0, 0]], np.int64)

    field_only = bic.distance.geodesic_distance_field(mask, src)
    assert isinstance(field_only, np.ndarray)
    assert field_only.dtype == np.float64

    field, grad = bic.distance.geodesic_distance_field(mask, src, return_gradient=True)
    assert np.array_equal(field, field_only)  # field is unchanged by the option
    assert grad.shape == mask.shape + (mask.ndim,)
    assert grad.dtype == np.float32


@pytest.mark.parametrize("shape", [(60, 60), (18, 30, 30)])
def test_mask_gradient_eikonal_norm(shape):
    mask = np.ones(shape, np.uint8)
    source = tuple(0 for _ in shape)
    _, grad = bic.distance.geodesic_distance_field(
        mask, np.array([source], np.int64), return_gradient=True
    )
    norm = np.linalg.norm(grad, axis=-1)
    interior = _interior_mask(shape, source)
    # |grad(T)| = slowness = 1 for unit speed (the Eikonal equation)
    np.testing.assert_allclose(norm[interior], 1.0, atol=0.02)

    # speed = 2 -> |grad| = 0.5
    _, grad2 = bic.distance.geodesic_distance_field(
        mask, np.array([source], np.int64), speed=2.0 * np.ones(shape), return_gradient=True
    )
    np.testing.assert_allclose(np.linalg.norm(grad2, axis=-1)[interior], 0.5, atol=0.02)


def test_mask_gradient_points_away_from_source():
    shape = (60, 60)
    source = (0, 0)
    mask = np.ones(shape, np.uint8)
    _, grad = bic.distance.geodesic_distance_field(
        mask, np.array([source], np.int64), return_gradient=True
    )
    coords = np.indices(shape, dtype=np.float64)
    # dot(grad, position - source) > 0  => grad points away from the source
    dot = grad[..., 0] * coords[0] + grad[..., 1] * coords[1]
    interior = _interior_mask(shape, source)
    assert np.all(dot[interior] > 0)
    # zero at the source itself (a local minimum)
    np.testing.assert_allclose(grad[source], 0.0, atol=1e-6)


def test_mask_gradient_anisotropic_norm():
    shape = (50, 50)
    source = (0, 0)
    mask = np.ones(shape, np.uint8)
    _, grad = bic.distance.geodesic_distance_field(
        mask, np.array([source], np.int64), sampling=(2.0, 0.5), return_gradient=True
    )
    interior = _interior_mask(shape, source)
    np.testing.assert_allclose(np.linalg.norm(grad, axis=-1)[interior], 1.0, atol=0.02)


def test_mask_gradient_field_wrapper_matches():
    mask = np.ones((30, 30), np.uint8)
    src = np.array([[5, 7]], np.int64)
    _, grad = bic.distance.geodesic_distance_field(mask, src, return_gradient=True)
    only = bic.distance.geodesic_gradient_field(mask, src)
    assert np.array_equal(only, grad)
    assert only.dtype == np.float32


def test_mask_gradient_obstacle_and_background_zero():
    mask = np.ones((21, 21), np.uint8)
    mask[10, :18] = 0  # wall with a gap on the right
    field, grad = bic.distance.geodesic_distance_field(
        mask, np.array([[0, 0]], np.int64), return_gradient=True
    )
    # background (wall) voxels have a zero gradient
    assert np.all(grad[mask == 0] == 0.0)
    # gradient is finite everywhere and unit-norm across most of the reachable
    # interior (it may deviate only on the medial axis where fronts meet)
    assert np.all(np.isfinite(grad))
    reachable = np.isfinite(field) & (mask != 0)
    norm = np.linalg.norm(grad, axis=-1)
    interior = reachable & (field > 5)
    frac_unit = np.mean(np.abs(norm[interior] - 1.0) < 0.05)
    assert frac_unit > 0.95


# --------------------------------------------------------------------------- #
# mesh: field
# --------------------------------------------------------------------------- #


def test_mesh_flat_grid_matches_euclidean():
    verts, faces, vid = flat_grid_mesh(21)
    field = bic.distance.geodesic_distance_field_mesh(verts, faces, np.array([vid(0, 0)], np.int64))
    eucl = np.hypot(verts[:, 0], verts[:, 1])

    assert field[vid(0, 0)] == 0.0
    assert np.all(np.isfinite(field))
    assert np.all(field >= eucl - 1e-9)
    # exact along a mesh edge direction
    axis = np.array([field[vid(i, 0)] for i in range(21)])
    np.testing.assert_allclose(axis, np.arange(21), atol=1e-9)
    # close to Euclidean in the far field
    far = eucl > 8
    rel = np.abs(field[far] - eucl[far]) / eucl[far]
    assert rel.max() < 0.12


def test_mesh_field_ge_chord_on_sphere():
    verts, faces = sphere_mesh()
    field = bic.distance.geodesic_distance_field_mesh(verts, faces, np.array([0], np.int64))
    assert field[0] == 0.0
    assert np.all(np.isfinite(field))
    # surface geodesic distance is >= the straight-line (chord) distance
    chord = np.linalg.norm(verts - verts[0], axis=1)
    assert np.all(field >= chord - 1e-6)


def test_mesh_speed_scaling_is_exact():
    verts, faces = sphere_mesh()
    src = np.array([0], np.int64)
    base = bic.distance.geodesic_distance_field_mesh(verts, faces, src)
    scaled = bic.distance.geodesic_distance_field_mesh(
        verts, faces, src, speed=2.0 * np.ones(len(verts))
    )
    np.testing.assert_allclose(scaled, base / 2.0, rtol=1e-9, atol=1e-9)


def test_mesh_multi_source_monotone():
    verts, faces = sphere_mesh()
    a = bic.distance.geodesic_distance_field_mesh(verts, faces, np.array([0], np.int64))
    b = bic.distance.geodesic_distance_field_mesh(verts, faces, np.array([50], np.int64))
    both = bic.distance.geodesic_distance_field_mesh(verts, faces, np.array([0, 50], np.int64))
    assert np.all(both <= np.minimum(a, b) + 1e-9)
    np.testing.assert_allclose(both, np.minimum(a, b), atol=0.3)


def test_mesh_disconnected_components_are_inf():
    # two disjoint triangles (no shared vertices)
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
        np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    field = bic.distance.geodesic_distance_field_mesh(verts, faces, np.array([0], np.int64))
    assert np.all(np.isfinite(field[:3]))
    assert np.all(np.isinf(field[3:]))


# --------------------------------------------------------------------------- #
# mesh: pairwise + validation
# --------------------------------------------------------------------------- #


def test_mesh_pairwise_symmetric_and_matches_field():
    verts, faces = sphere_mesh()
    points = np.array([0, 40, 120, 200], np.int64)
    D = bic.distance.geodesic_distances_mesh(verts, faces, points)
    assert D.shape == (4, 4)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
    # the matrix is symmetrized; single-source FMM on the irregular mesh is
    # slightly direction-dependent, so rows match the raw field only up to that
    # small asymmetry.
    np.testing.assert_allclose(D, D.T, atol=1e-9)
    for i, p in enumerate(points):
        field = bic.distance.geodesic_distance_field_mesh(verts, faces, np.array([p], np.int64))
        np.testing.assert_allclose(D[i], field[points], rtol=0.03, atol=0.2)


def test_mesh_invalid_arguments():
    verts, faces = sphere_mesh(n_points=60)
    with pytest.raises(ValueError, match="out of range"):
        bic.distance.geodesic_distance_field_mesh(
            verts, faces, np.array([len(verts)], np.int64)
        )
    with pytest.raises(ValueError, match="vertices"):
        bic.distance.geodesic_distance_field_mesh(
            np.zeros((5, 2)), faces, np.array([0], np.int64)
        )


# --------------------------------------------------------------------------- #
# optional tight cross-checks against the external references
# --------------------------------------------------------------------------- #


def test_mask_matches_scikit_fmm():
    skfmm = pytest.importorskip("skfmm")
    mask = np.ones((50, 50), np.uint8)
    field = bic.distance.geodesic_distance_field(mask, np.array([[0, 0]], np.int64))
    phi = np.ones((50, 50))
    phi[0, 0] = -1
    ref = np.abs(skfmm.distance(phi, order=1))
    eucl = euclidean_from((50, 50), (0, 0))
    interior = eucl > 3
    # scikit-fmm's level-set seed idiom is offset ~0.5 cell; remove it, then the
    # two first-order schemes must agree to machine precision.
    offset = np.median((field - ref)[interior])
    residual = np.abs((field - ref) - offset)[interior]
    assert residual.max() < 1e-6


def test_mesh_matches_pygeodesic():
    geodesic = pytest.importorskip("pygeodesic.geodesic")
    verts, faces = sphere_mesh(n_points=400)
    ours = bic.distance.geodesic_distance_field_mesh(verts, faces, np.array([0], np.int64))
    algo = geodesic.PyGeodesicAlgorithmExact(verts, np.ascontiguousarray(faces, np.int32))
    ref, _ = algo.geodesicDistances(np.array([0], np.int32), None)
    ref = np.asarray(ref)
    reachable = ref > 1e-9
    rel = np.abs(ours[reachable] - ref[reachable]) / ref[reachable]
    # first-order FMM vs exact MMP: modest error, no obtuse unfolding yet
    assert np.mean(rel) < 0.06
    assert np.percentile(rel, 95) < 0.12
