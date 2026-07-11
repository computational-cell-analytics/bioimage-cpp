import gc

import numpy as np
import pytest

import bioimage_cpp as bic


def _interior_box() -> np.ndarray:
    volume = np.zeros((5, 6, 7), dtype=np.uint8)
    volume[1:4, 1:5, 1:6] = 1
    return volume


def _interior_ball(n: int = 32, radius: float = 11.0) -> np.ndarray:
    center = (n - 1) / 2.0
    z, y, x = np.ogrid[:n, :n, :n]
    return (
        (z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2
        <= radius**2
    ).astype(np.uint8)


def _edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def _zero_area_faces(vertices: np.ndarray, faces: np.ndarray) -> int:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return int(np.count_nonzero(np.linalg.norm(cross, axis=1) <= 1e-12))


def _collapsed_coordinate_faces(vertices: np.ndarray, faces: np.ndarray) -> int:
    triangles = vertices[faces]
    first_second = np.all(triangles[:, 0] == triangles[:, 1], axis=1)
    first_third = np.all(triangles[:, 0] == triangles[:, 2], axis=1)
    second_third = np.all(triangles[:, 1] == triangles[:, 2], axis=1)
    return int(np.count_nonzero(first_second | first_third | second_third))


def _trilinear_sample(volume: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Independently sample a scalar field at NumPy-order (z, y, x) coordinates."""
    image = np.asarray(volume, dtype=np.float64)
    output = np.empty(len(coordinates), dtype=np.float64)
    nz, ny, nx = image.shape
    for index, (z, y, x) in enumerate(coordinates):
        z0, y0, x0 = int(np.floor(z)), int(np.floor(y)), int(np.floor(x))
        z1, y1, x1 = min(z0 + 1, nz - 1), min(y0 + 1, ny - 1), min(x0 + 1, nx - 1)
        fz, fy, fx = z - z0, y - y0, x - x0
        value = 0.0
        for cz, wz in ((z0, 1.0 - fz), (z1, fz)):
            for cy, wy in ((y0, 1.0 - fy), (y1, fy)):
                for cx, wx in ((x0, 1.0 - fx), (x1, fx)):
                    value += wz * wy * wx * image[cz, cy, cx]
        output[index] = value
    return output


def _transitive_degenerate_regression_volume() -> np.ndarray:
    """Reproduce a duplicate-vertex chain that previously yielded face index -1."""
    rng = np.random.default_rng(293841)
    levels = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    for _ in range(16):
        shape = tuple(int(size) for size in rng.integers(3, 9, size=3))
        volume = rng.integers(0, 4, size=shape, dtype=np.uint8)
        volume.flat[0] = 0
        volume.flat[-1] = 3
        level = float(rng.choice(levels))
        method = "lewiner" if rng.random() < 0.7 else "lorensen"
        allow_degenerate = bool(rng.random() < 0.5)
        step_size = 2 if min(shape) >= 5 and rng.random() < 0.25 else 1
        if rng.random() < 0.25:
            rng.random(shape)
    assert shape == (6, 5, 8)
    assert level == 1.0
    assert method == "lorensen"
    assert not allow_degenerate
    assert step_size == 1
    return volume


def test_box_mesh_has_reference_dimensions_and_dtypes():
    vertices, faces, normals, values = bic.mesh.marching_cubes(_interior_box(), 0.5)

    assert vertices.shape == (94, 3)
    assert faces.shape == (184, 3)
    assert normals.shape == vertices.shape
    assert values.shape == (94,)
    assert vertices.dtype == np.float32
    assert faces.dtype == np.int32
    assert normals.dtype == np.float32
    assert values.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=2e-6)
    assert np.all((faces >= 0) & (faces < len(vertices)))
    np.testing.assert_array_equal(vertices.min(axis=0), [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(vertices.max(axis=0), [3.5, 4.5, 5.5])
    np.testing.assert_array_equal(values, np.ones_like(values))


@pytest.mark.parametrize("method", ["lewiner", "lorensen"])
def test_method_output_invariants(method):
    volume = _interior_ball()
    vertices, faces, normals, values = bic.mesh.marching_cubes(
        volume, 0.5, method=method
    )

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert normals.shape == vertices.shape
    assert values.shape == (len(vertices),)
    assert vertices.dtype == normals.dtype == values.dtype == np.float32
    assert faces.dtype == np.int32
    assert np.all((faces >= 0) & (faces < len(vertices)))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
    assert vertices.min() >= 0.0
    for axis, size in enumerate(volume.shape):
        assert vertices[:, axis].max() <= size - 1


def test_lorensen_vertices_lie_on_isosurface():
    rng = np.random.default_rng(0)
    volume = rng.random((12, 13, 14), dtype=np.float32)
    level = 0.5
    vertices, _, _, _ = bic.mesh.marching_cubes(
        volume, level, method="lorensen"
    )
    np.testing.assert_allclose(
        _trilinear_sample(volume, vertices), level, atol=1e-4
    )


def test_output_arrays_are_contiguous_writable_and_survive_collection():
    outputs = bic.mesh.marching_cubes(_interior_box(), 0.5)
    expected = tuple(output.copy() for output in outputs)

    for output in outputs:
        assert output.flags.c_contiguous
        assert output.flags.writeable

    # The binding transfers ownership to NumPy. Exercise the arrays after the
    # C++ result has gone out of scope and after allocation churn/collection.
    for _ in range(8):
        np.empty((1024, 1024), dtype=np.float32)
    gc.collect()
    for output, want in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(output, want)

    outputs[3][0] += 1.0
    assert outputs[3][0] == expected[3][0] + 1.0


def test_lorensen_and_lewiner_choose_different_ambiguous_topologies():
    # The bit pattern 0b00000110 is an ambiguous one-cube configuration.
    volume = np.array([(6 >> bit) & 1 for bit in range(8)], dtype=np.uint8).reshape(2, 2, 2)
    lewiner = bic.mesh.marching_cubes(volume, 0.5, method="lewiner")
    lorensen = bic.mesh.marching_cubes(volume, 0.5, method="lorensen")

    assert lewiner[0].shape == lorensen[0].shape == (6, 3)
    assert lewiner[1].shape == (4, 3)
    assert lorensen[1].shape == (2, 3)


def test_spacing_preserves_mesh_and_uses_float64_vertices():
    volume = _interior_box()
    unit_vertices, unit_faces, unit_normals, unit_values = bic.mesh.marching_cubes(volume, 0.5)
    spacing = (2.0, 0.5, 3.0)
    vertices, faces, normals, values = bic.mesh.marching_cubes(volume, 0.5, spacing=spacing)

    assert vertices.dtype == np.float64
    np.testing.assert_allclose(vertices, unit_vertices * np.asarray(spacing))
    np.testing.assert_array_equal(faces, unit_faces)
    np.testing.assert_array_equal(normals, unit_normals)
    np.testing.assert_array_equal(values, unit_values)


def test_scalar_spacing_is_broadcast_to_all_axes():
    volume = _interior_box()
    scalar = bic.mesh.marching_cubes(volume, 0.5, spacing=2.0)
    explicit = bic.mesh.marching_cubes(volume, 0.5, spacing=(2.0, 2.0, 2.0))
    for actual, expected in zip(scalar, explicit, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert scalar[0].dtype == np.float64


@pytest.mark.parametrize("dtype", [np.bool_, np.uint16, np.float64])
def test_numeric_dtype_conversion_matches_uint8_input(dtype):
    volume = _interior_box()
    if dtype is np.bool_:
        converted = volume.astype(bool)
    else:
        converted = volume.astype(dtype)
    expected = bic.mesh.marching_cubes(volume, 0.5)
    actual = bic.mesh.marching_cubes(converted, 0.5)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_non_contiguous_input_is_normalised_at_the_python_boundary():
    volume = _interior_box()
    repeated = np.repeat(volume, 2, axis=2)
    non_contiguous = repeated[:, :, ::2]
    assert not non_contiguous.flags.c_contiguous

    actual = bic.mesh.marching_cubes(non_contiguous, 0.5)
    expected = bic.mesh.marching_cubes(volume, 0.5)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_gradient_direction_only_reverses_face_winding():
    descent = bic.mesh.marching_cubes(_interior_box(), 0.5, gradient_direction="descent")
    ascent = bic.mesh.marching_cubes(_interior_box(), 0.5, gradient_direction="ascent")

    np.testing.assert_array_equal(descent[0], ascent[0])
    np.testing.assert_array_equal(descent[2], ascent[2])
    np.testing.assert_array_equal(descent[3], ascent[3])
    np.testing.assert_array_equal(descent[1], ascent[1][:, ::-1])


def test_padding_closes_a_boundary_object_and_restores_coordinates():
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[:2, 1:3, 1:3] = 1
    open_vertices, open_faces, _, _ = bic.mesh.marching_cubes(volume, 0.5)
    padded_vertices, padded_faces, _, _ = bic.mesh.marching_cubes(volume, 0.5, pad=True)

    assert open_vertices.shape == (20, 3)
    assert open_faces.shape == (30, 3)
    assert padded_vertices.shape == (24, 3)
    assert padded_faces.shape == (44, 3)
    np.testing.assert_array_equal(open_vertices.min(axis=0), [0.0, 0.5, 0.5])
    np.testing.assert_array_equal(padded_vertices.min(axis=0), [-0.5, 0.5, 0.5])
    assert set(_edge_counts(padded_faces).values()) == {2}


def test_mask_and_step_size_are_honoured():
    rng = np.random.default_rng(9)
    volume = rng.integers(0, 2, size=(8, 9, 10), dtype=np.uint8)
    mask = np.ones_like(volume, dtype=bool)
    mask[:, 0] = False
    vertices, faces, normals, values = bic.mesh.marching_cubes(
        volume, 0.5, method="lorensen", mask=mask, step_size=2
    )

    assert vertices.shape == normals.shape
    assert vertices.shape[1] == 3
    assert len(vertices) > 0
    assert faces.shape[1] == 3
    assert len(faces) > 0
    assert values.shape == (len(vertices),)


def test_mask_reduces_surface_and_step_size_coarsens_mesh():
    rng = np.random.default_rng(4)
    volume = rng.random((20, 22, 24), dtype=np.float32)
    mask = np.ones(volume.shape, dtype=bool)
    mask[:, :, : volume.shape[2] // 2] = False
    fine = bic.mesh.marching_cubes(volume, 0.5)
    masked = bic.mesh.marching_cubes(volume, 0.5, mask=mask)
    coarse = bic.mesh.marching_cubes(volume, 0.5, step_size=2)
    assert len(masked[1]) < len(fine[1])
    assert len(coarse[1]) < len(fine[1])


@pytest.mark.parametrize("method", ["lewiner", "lorensen"])
@pytest.mark.parametrize("step_size", [1, 2, 3])
def test_vertex_cache_with_step_size_and_mask_is_deterministic(method, step_size):
    rng = np.random.default_rng(27)
    volume = rng.random((11, 12, 13), dtype=np.float32)
    mask = np.ones(volume.shape, dtype=bool)
    mask[1::3, 2::4, :] = False
    mask[:, 1::4, 2::3] = False

    first = bic.mesh.marching_cubes(
        volume,
        0.5,
        method=method,
        step_size=step_size,
        mask=mask,
    )
    second = bic.mesh.marching_cubes(
        volume,
        0.5,
        method=method,
        step_size=step_size,
        mask=mask,
    )

    for actual, expected in zip(first, second, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert len(np.unique(first[0], axis=0)) == len(first[0])
    assert np.all((first[1] >= 0) & (first[1] < len(first[0])))


def test_lewiner_closed_sphere_is_watertight_with_euler_characteristic_two():
    vertices, faces, _, _ = bic.mesh.marching_cubes(
        _interior_ball(n=34, radius=12.0), 0.5, method="lewiner"
    )
    edge_counts = _edge_counts(faces)
    assert set(edge_counts.values()) == {2}
    assert len(vertices) - len(edge_counts) + len(faces) == 2


@pytest.mark.parametrize("method", ["lewiner", "lorensen"])
def test_output_is_deterministic(method):
    volume = _interior_ball()
    first = bic.mesh.marching_cubes(volume, 0.5, method=method)
    second = bic.mesh.marching_cubes(volume, 0.5, method=method)
    for actual, expected in zip(first, second, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_allow_degenerate_false_removes_collapsed_faces():
    rng = np.random.default_rng(3)
    volume = rng.integers(0, 3, size=(16, 16, 16), dtype=np.uint8)
    kept = bic.mesh.marching_cubes(volume, 1.0, allow_degenerate=True)
    removed = bic.mesh.marching_cubes(volume, 1.0, allow_degenerate=False)

    assert _collapsed_coordinate_faces(kept[0], kept[1]) > 0
    assert _collapsed_coordinate_faces(removed[0], removed[1]) == 0
    assert _zero_area_faces(removed[0], removed[1]) < _zero_area_faces(kept[0], kept[1])
    assert len(removed[1]) < len(kept[1])
    assert removed[1].dtype == np.int32


def test_transitive_degenerate_vertex_merges_have_valid_indices():
    volume = _transitive_degenerate_regression_volume()
    first = bic.mesh.marching_cubes(
        volume,
        1.0,
        method="lorensen",
        allow_degenerate=False,
    )
    second = bic.mesh.marching_cubes(
        volume,
        1.0,
        method="lorensen",
        allow_degenerate=False,
    )

    assert first[0].shape == (251, 3)
    assert first[1].shape == (360, 3)
    assert np.all((first[1] >= 0) & (first[1] < len(first[0])))
    assert _collapsed_coordinate_faces(first[0], first[1]) == 0
    for actual, expected in zip(first, second, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_default_level_is_midpoint_of_original_volume_when_padding():
    volume = np.full((5, 5, 5), 2.0, dtype=np.float32)
    volume[1:4, 1:4, 1:4] = 4.0
    default = bic.mesh.marching_cubes(volume, pad=True)
    explicit = bic.mesh.marching_cubes(volume, 3.0, pad=True)
    for actual, expected in zip(default, explicit, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_invalid_arguments_and_missing_surface():
    with pytest.raises(ValueError, match="3D"):
        bic.mesh.marching_cubes(np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="at least 2x2x2"):
        bic.mesh.marching_cubes(np.zeros((1, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="Surface level"):
        bic.mesh.marching_cubes(_interior_box(), 2.0)
    with pytest.raises(ValueError, match="method should"):
        bic.mesh.marching_cubes(_interior_box(), 0.5, method="other")
    with pytest.raises(ValueError, match="gradient_direction"):
        bic.mesh.marching_cubes(_interior_box(), 0.5, gradient_direction="other")
    with pytest.raises(ValueError, match="spacing"):
        bic.mesh.marching_cubes(_interior_box(), 0.5, spacing=(1.0, 1.0))
    with pytest.raises(ValueError, match="positive and finite"):
        bic.mesh.marching_cubes(_interior_box(), 0.5, spacing=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="positive and finite"):
        bic.mesh.marching_cubes(_interior_box(), 0.5, spacing=(1.0, np.nan, 1.0))
    with pytest.raises(ValueError, match="step_size"):
        bic.mesh.marching_cubes(_interior_box(), 0.5, step_size=0)
    with pytest.raises(TypeError, match="step_size"):
        bic.mesh.marching_cubes(_interior_box(), 0.5, step_size=1.5)
    with pytest.raises(ValueError, match="finite"):
        bic.mesh.marching_cubes(_interior_box(), np.nan)
    with pytest.raises(ValueError, match="same shape"):
        bic.mesh.marching_cubes(_interior_box(), 0.5, mask=np.ones((3, 3, 3), bool))
    with pytest.raises(RuntimeError, match="No surface"):
        bic.mesh.marching_cubes(np.zeros((3, 3, 3), dtype=np.uint8), 0.0)
    with pytest.raises(TypeError, match="real numeric"):
        bic.mesh.marching_cubes(np.ones((3, 3, 3), dtype=np.complex64), 1.0)
