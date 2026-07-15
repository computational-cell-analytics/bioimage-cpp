import numpy as np
import pytest

import bioimage_cpp as bic
from bioimage_cpp import _core


def _teasar_backend(
    mask, backend, *, spacing=(1.5, 1.0, 1.0), number_of_threads=1
):
    return _core._teasar_uint8_backend(
        np.ascontiguousarray(mask, dtype=np.uint8),
        spacing,
        1.5,
        1.0,
        100000.0,
        4.0,
        backend,
        number_of_threads,
    )


def _assert_valid_tree(mask, vertices, edges, radii, spacing=(1.0, 1.0, 1.0)):
    mask = np.asarray(mask) != 0
    spacing = np.asarray(spacing, dtype=np.float64)

    assert vertices.dtype == np.float64
    assert edges.dtype == np.uint64
    assert radii.dtype == np.float32
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert radii.shape == (vertices.shape[0],)

    n_vertices = vertices.shape[0]
    if n_vertices == 0:
        assert edges.shape == (0, 2)
        return

    assert edges.shape[0] == n_vertices - 1
    assert np.all(edges < n_vertices)
    assert np.all(edges[:, 0] != edges[:, 1])
    assert len({tuple(sorted(map(int, edge))) for edge in edges}) == len(edges)

    voxel_coords_float = vertices / spacing
    voxel_coords = np.rint(voxel_coords_float).astype(np.int64)
    np.testing.assert_allclose(voxel_coords_float, voxel_coords, atol=1e-12)
    assert np.all(voxel_coords >= 0)
    assert np.all(voxel_coords < np.asarray(mask.shape))
    assert np.all(mask[tuple(voxel_coords.T)])
    assert np.unique(voxel_coords, axis=0).shape[0] == n_vertices

    # V - 1 distinct edges plus connectivity proves that the output is a tree.
    adjacency = [[] for _ in range(n_vertices)]
    for u, v in edges:
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))
    reached = {0}
    stack = [0]
    while stack:
        for neighbor in adjacency[stack.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    assert len(reached) == n_vertices

    # TEASAR pads a zero halo so boundary-touching objects have an explicit
    # background feature. Reproduce that independent EDT setup for radii.
    padded = np.pad(mask, 1)
    dbf = bic.distance.distance_transform(padded, sampling=tuple(spacing))
    expected_radii = dbf[tuple((voxel_coords + 1).T)]
    np.testing.assert_allclose(radii, expected_radii, rtol=1e-6, atol=1e-6)


def test_empty_mask_returns_typed_empty_graph():
    vertices, edges, radii = bic.skeleton.teasar(np.zeros((4, 5, 6), bool))
    assert vertices.shape == (0, 3) and vertices.dtype == np.float64
    assert edges.shape == (0, 2) and edges.dtype == np.uint64
    assert radii.shape == (0,) and radii.dtype == np.float32


def test_degenerate_empty_axis_returns_empty_graph():
    result = bic.skeleton.teasar(np.empty((0, 4, 5), np.uint8))
    assert [array.shape for array in result] == [(0, 3), (0, 2), (0,)]


def test_single_voxel():
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[2, 3, 4] = 7
    vertices, edges, radii = bic.skeleton.teasar(mask)
    np.testing.assert_array_equal(vertices, [[2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(edges, np.empty((0, 2), dtype=np.uint64))
    np.testing.assert_array_equal(radii, [1.0])
    _assert_valid_tree(mask, vertices, edges, radii)


def test_straight_filament_is_recovered_exactly():
    mask = np.zeros((9, 9, 11), dtype=bool)
    mask[4, 4, 2:9] = True
    vertices, edges, radii = bic.skeleton.teasar(mask)
    _assert_valid_tree(mask, vertices, edges, radii)
    got = {tuple(coord.astype(int)) for coord in vertices}
    expected = {(4, 4, x) for x in range(2, 9)}
    assert got == expected
    assert np.all(radii == 1.0)


def test_y_branch_has_three_endpoints_and_one_branchpoint():
    mask = np.zeros((9, 11, 10), dtype=np.uint8)
    mask[4, 5, 1:5] = 1
    for step in range(5):
        mask[4, 5 - step, 4 + step] = 1
        mask[4, 5 + step, 4 + step] = 1

    vertices, edges, radii = bic.skeleton.teasar(mask)
    _assert_valid_tree(mask, vertices, edges, radii)
    degree = np.bincount(edges.ravel(), minlength=len(vertices))
    endpoints = {tuple(vertex.astype(int)) for vertex in vertices[degree == 1]}
    assert endpoints == {(4, 5, 1), (4, 1, 8), (4, 9, 8)}
    assert np.count_nonzero(degree == 3) == 1


def test_thick_bent_tube_produces_an_interior_tree():
    zz, yy, xx = np.indices((17, 21, 25))
    first = ((zz - 8) ** 2 + (yy - 7) ** 2 <= 3**2) & (xx >= 3) & (xx <= 16)
    second = ((zz - 8) ** 2 + (xx - 16) ** 2 <= 3**2) & (yy >= 7) & (yy <= 17)
    mask = first | second

    vertices, edges, radii = bic.skeleton.teasar(mask, scale=1.5, constant=1.0)
    _assert_valid_tree(mask, vertices, edges, radii)
    assert len(vertices) >= 2
    assert float(radii.max()) >= 3.0


def test_anisotropic_spacing_scales_vertices_and_radii():
    spacing = (2.0, 1.25, 0.5)
    mask = np.zeros((7, 7, 10), dtype=np.uint8)
    mask[3, 3, 2:8] = 1
    vertices, edges, radii = bic.skeleton.teasar(mask, spacing=spacing)
    _assert_valid_tree(mask, vertices, edges, radii, spacing)
    assert np.all(vertices[:, 0] == 3 * spacing[0])
    assert np.all(vertices[:, 1] == 3 * spacing[1])
    assert radii.min() == spacing[2]
    assert radii.max() == spacing[1]


def test_26_connected_diagonal_is_accepted():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 2, 2] = True
    vertices, edges, radii = bic.skeleton.teasar(mask)
    _assert_valid_tree(mask, vertices, edges, radii)
    assert len(vertices) == 2


def test_multiple_components_are_rejected():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1, 1, 1] = True
    mask[3, 3, 3] = True
    with pytest.raises(ValueError, match="exactly one 26-connected component"):
        bic.skeleton.teasar(mask)


def test_noncontiguous_and_numeric_binary_input_are_accepted():
    base = np.zeros((12, 14, 16), dtype=np.float64)
    base[4, 6, 2:14:2] = 3.5
    mask = base[::2, ::2, ::2]
    assert not mask.flags.c_contiguous
    vertices, edges, radii = bic.skeleton.teasar(mask)
    _assert_valid_tree(mask, vertices, edges, radii)


def test_output_is_deterministic():
    mask = np.zeros((9, 11, 10), dtype=bool)
    mask[4, 5, 1:5] = True
    for step in range(5):
        mask[4, 5 - step, 4 + step] = True
        mask[4, 5 + step, 4 + step] = True
    first = bic.skeleton.teasar(mask)
    second = bic.skeleton.teasar(mask)
    for got, expected in zip(first, second):
        np.testing.assert_array_equal(got, expected)


def test_threaded_output_is_deterministic():
    mask = np.zeros((17, 21, 25), dtype=bool)
    mask[8, 10, 2:18] = True
    mask[8, 4:17, 17] = True
    first = bic.skeleton.teasar(mask, number_of_threads=2)
    second = bic.skeleton.teasar(mask, number_of_threads=4)
    for got, expected in zip(first, second):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (2.5, 1.25, 0.75)])
def test_compact_fp64_backends_have_exact_dense_parity(spacing):
    zz, yy, xx = np.indices((17, 21, 25))
    first = ((zz - 8) ** 2 + (yy - 7) ** 2 <= 3**2) & (xx >= 3) & (xx <= 16)
    second = ((zz - 8) ** 2 + (xx - 16) ** 2 <= 3**2) & (yy >= 7) & (yy <= 17)
    mask = first | second
    dense = _teasar_backend(mask, "dense-fp64", spacing=spacing)
    for backend in ("compact-on-the-fly-fp64", "compact-csr-fp64"):
        compact = _teasar_backend(mask, backend, spacing=spacing)
        for got, expected in zip(compact, dense):
            np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (2.5, 1.25, 0.75)])
def test_cropped_compact_backends_preserve_dense_parity(spacing):
    mask = np.zeros((31, 37, 45), dtype=np.uint8)
    mask[3, 8, 4:30] = 1
    mask[3, 8:22, 29] = 1
    mask[3:17, 21, 29] = 1
    mask[3:12, 8:17, 29] |= np.eye(9, dtype=np.uint8)

    dense = _teasar_backend(mask, "dense-fp64", spacing=spacing)
    for backend in ("compact-on-the-fly-fp64", "compact-csr-fp64"):
        compact = _teasar_backend(
            mask, backend, spacing=spacing, number_of_threads=4
        )
        for got, expected in zip(compact, dense):
            np.testing.assert_array_equal(got, expected)


def test_cropped_compact_backend_preserves_boundary_touching_object():
    mask = np.zeros((13, 17, 21), dtype=np.uint8)
    mask[0, 0, :] = 1
    mask[0, :, -1] = 1
    mask[:, -1, -1] = 1

    dense = _teasar_backend(mask, "dense-fp64")
    compact = _teasar_backend(
        mask, "compact-on-the-fly-fp64", number_of_threads=4
    )
    for got, expected in zip(compact, dense):
        np.testing.assert_array_equal(got, expected)


def test_public_teasar_uses_exact_compact_fp64_backend():
    mask = np.zeros((9, 11, 10), dtype=np.uint8)
    mask[4, 5, 1:5] = 1
    for step in range(5):
        mask[4, 5 - step, 4 + step] = 1
        mask[4, 5 + step, 4 + step] = 1
    public = bic.skeleton.teasar(
        mask, spacing=(1.5, 1.0, 1.0), constant=1.0
    )
    compact = _teasar_backend(mask, "compact-on-the-fly-fp64")
    for got, expected in zip(public, compact):
        np.testing.assert_array_equal(got, expected)


def test_failed_fp32_development_backend_is_not_selectable():
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown TEASAR development backend"):
        _teasar_backend(mask, "compact-csr-fp32")


@pytest.mark.parametrize("shape", [(4, 5), (2, 3, 4, 5)])
def test_rejects_non_3d_input(shape):
    with pytest.raises(ValueError, match="ndim 3"):
        bic.skeleton.teasar(np.ones(shape, dtype=np.uint8))


def test_direct_binding_validates_ndim_and_spacing_before_dispatch():
    parameters = (1.5, 1.0, 100000.0, 4.0, 1)
    with pytest.raises(ValueError, match="mask must have ndim 3, got ndim=2"):
        _core._teasar_uint8(
            np.ones((3, 4), dtype=np.uint8), [1.0, 1.0, 1.0], *parameters
        )
    with pytest.raises(ValueError, match="exactly three values, got 2"):
        _core._teasar_uint8(
            np.ones((3, 4, 5), dtype=np.uint8), [1.0, 1.0], *parameters
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"spacing": (1.0, 2.0)}, "spacing must be a scalar"),
        ({"spacing": (1.0, 0.0, 1.0)}, "positive"),
        ({"scale": -1.0}, "scale"),
        ({"constant": np.inf}, "constant"),
        ({"pdrf_scale": np.nan}, "pdrf_scale"),
        ({"pdrf_exponent": 0.0}, "pdrf_exponent"),
        ({"number_of_threads": -1}, "number_of_threads"),
    ],
)
def test_rejects_invalid_parameters(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        bic.skeleton.teasar(np.ones((3, 3, 3), dtype=bool), **kwargs)
