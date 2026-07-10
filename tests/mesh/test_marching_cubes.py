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
