import sys

import numpy as np
import pytest

import bioimage_cpp as bic
from bioimage_cpp import _core


def _number_of_graph_components(n_vertices, edges):
    if n_vertices == 0:
        return 0
    parents = list(range(n_vertices))

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for u, v in edges:
        first, second = find(int(u)), find(int(v))
        if first != second:
            parents[second] = first
    return len({find(node) for node in range(n_vertices)})


def _assert_valid_forest(
    mask, result, expected_components, *, spacing=(1.0, 1.0, 1.0)
):
    vertices, edges, radii = result
    mask = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=np.float64)

    assert vertices.dtype == np.float64 and vertices.shape[1:] == (3,)
    assert edges.dtype == np.uint64 and edges.shape[1:] == (2,)
    assert radii.dtype == np.float32 and radii.shape == (len(vertices),)
    assert len(edges) == len(vertices) - expected_components
    assert _number_of_graph_components(len(vertices), edges) == expected_components
    if not len(vertices):
        return
    assert np.all(edges < len(vertices))
    assert np.all(edges[:, 0] != edges[:, 1])
    assert len({tuple(sorted(map(int, edge))) for edge in edges}) == len(edges)

    voxel_float = vertices / spacing
    voxels = np.rint(voxel_float).astype(np.int64)
    np.testing.assert_allclose(voxel_float, voxels, atol=1e-12)
    assert np.all(voxels >= 0)
    assert np.all(voxels < np.asarray(mask.shape))
    assert np.all(mask[tuple(voxels.T)])
    assert len(np.unique(voxels, axis=0)) == len(voxels)

    padded = np.pad(mask, 1)
    dbf = bic.distance.distance_transform(padded, sampling=tuple(spacing))
    expected_radii = dbf[tuple((voxels + 1).T)]
    np.testing.assert_allclose(radii, expected_radii, rtol=1e-6, atol=1e-6)


def _components_per_label(labels, background=0):
    dense = bic.segmentation.label(labels, background=background, connectivity=3)
    return {
        int(label): len(np.unique(dense[labels == label]))
        for label in np.unique(labels)
        if int(label) != background
    }


def test_all_background_and_empty_axes_return_empty_dict():
    assert bic.skeleton.teasar_labels(np.zeros((4, 5, 6), np.uint32)) == {}
    assert bic.skeleton.teasar_labels(np.empty((0, 5, 6), np.int64)) == {}


def test_touching_labels_remain_separate():
    labels = np.zeros((5, 5, 8), dtype=np.uint32)
    labels[2, 2, 1:4] = 1
    labels[2, 2, 4:7] = 2
    result = bic.skeleton.teasar_labels(labels)
    assert list(result) == [1, 2]
    for label in result:
        _assert_valid_forest(labels == label, result[label], 1)
        assert np.all(labels[tuple(result[label][0].astype(int).T)] == label)

    binary = bic.skeleton.teasar(labels)
    assert len(binary[0]) == 6
    assert len(binary[1]) == 5


def test_face_edge_and_corner_touching_different_labels_never_merge():
    labels = np.zeros((6, 6, 6), dtype=np.uint16)
    labels[1, 1, 1] = 4
    labels[1, 1, 2] = 9
    labels[3, 3, 3] = 17
    labels[3, 4, 4] = 23
    labels[5, 4, 4] = 31
    labels[4, 5, 5] = 42
    result = bic.skeleton.teasar_labels(labels, number_of_threads=4)
    assert list(result) == [4, 9, 17, 23, 31, 42]
    for label, skeleton in result.items():
        _assert_valid_forest(labels == label, skeleton, 1)


def test_disconnected_repeated_label_returns_one_forest_entry():
    labels = np.zeros((9, 9, 12), dtype=np.int32)
    labels[1, 1, 1:5] = 7
    labels[7, 7, 7:11] = 7
    result = bic.skeleton.teasar_labels(labels, number_of_threads=2)
    assert list(result) == [7]
    _assert_valid_forest(labels == 7, result[7], 2)
    assert len(result[7][0]) == 8


def test_multiple_labels_and_components_have_canonical_order():
    labels = np.zeros((10, 10, 10), dtype=np.int64)
    labels[8, 8, 8] = 3
    labels[1, 1, 1] = -4
    labels[4, 4, 4] = 3
    labels[2, 7, 2] = 12
    labels[7, 2, 7] = -4
    result = bic.skeleton.teasar_labels(labels, number_of_threads=4)
    assert list(result) == [-4, 3, 12]
    expected = _components_per_label(labels)
    for label, skeleton in result.items():
        _assert_valid_forest(labels == label, skeleton, expected[label])
    np.testing.assert_array_equal(
        result[-4][0], [[1.0, 1.0, 1.0], [7.0, 2.0, 7.0]]
    )
    np.testing.assert_array_equal(
        result[3][0], [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]
    )


def test_component_materialization_excludes_other_labels_in_overlapping_boxes():
    labels = np.zeros((9, 9, 9), dtype=np.uint8)
    labels[1, 1, 1] = 5
    labels[7, 7, 7] = 5
    labels[3:6, 3:6, 3:6] = 8
    result = bic.skeleton.teasar_labels(labels)
    _assert_valid_forest(labels == 5, result[5], 2)
    _assert_valid_forest(labels == 8, result[8], 1)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_each_label_is_exactly_equal_to_binary_dispatch(threads):
    labels = np.zeros((17, 21, 25), dtype=np.uint32)
    labels[2, 3, 2:12] = 11
    labels[13, 17, 10:22] = 11
    labels[8, 4:17, 18] = 29
    options = {
        "spacing": (2.0, 1.25, 0.75),
        "constant": 1.0,
        "number_of_threads": threads,
    }
    result = bic.skeleton.teasar_labels(labels, **options)
    for label in (11, 29):
        expected = bic.skeleton.teasar(labels == label, **options)
        for got, wanted in zip(result[label], expected):
            np.testing.assert_array_equal(got, wanted)


@pytest.mark.parametrize(
    "dtype, labels_values",
    [
        (np.uint8, (1, 251)),
        (np.uint16, (2, 60001)),
        (np.uint32, (3, 4_000_000_001)),
        (np.uint64, (4, np.iinfo(np.int64).max + 17)),
        (np.int32, (-2_000_000_000, 17)),
        (np.int64, (np.iinfo(np.int64).min + 5, 23)),
    ],
)
def test_supported_dtypes_preserve_exact_python_keys(dtype, labels_values):
    labels = np.zeros((5, 5, 5), dtype=dtype)
    labels[1, 1, 1] = labels_values[0]
    labels[3, 3, 3] = labels_values[1]
    result = bic.skeleton.teasar_labels(labels)
    assert list(result) == sorted(map(int, labels_values))
    for label, skeleton in result.items():
        _assert_valid_forest(labels == label, skeleton, 1)


def test_nonzero_and_negative_background_values():
    unsigned = np.full((5, 5, 5), 9, dtype=np.uint16)
    unsigned[2, 2, 2] = 3
    assert list(bic.skeleton.teasar_labels(unsigned, background=9)) == [3]

    signed = np.full((5, 5, 5), -7, dtype=np.int32)
    signed[2, 2, 2] = -3
    assert list(bic.skeleton.teasar_labels(signed, background=-7)) == [-3]


def test_noncontiguous_input_is_copied_once_and_preserves_values():
    base = np.zeros((10, 12, 14), dtype=np.uint32)
    base[2, 2, 2:10:2] = 7
    base[8, 8, 4:12:2] = 19
    labels = base[::2, ::2, ::2]
    assert not labels.flags.c_contiguous
    result = bic.skeleton.teasar_labels(labels)
    assert list(result) == [7, 19]


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_thread_counts_preserve_exact_dictionary_and_arrays(threads):
    labels = np.zeros((13, 15, 17), dtype=np.uint16)
    for index, label in enumerate((9, 2, 17, 4)):
        z = 1 + 3 * index
        labels[z, 2 + index, 2:12] = label
    reference = bic.skeleton.teasar_labels(labels, number_of_threads=1)
    result = bic.skeleton.teasar_labels(labels, number_of_threads=threads)
    assert list(result) == list(reference)
    for label in result:
        for got, expected in zip(result[label], reference[label]):
            np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((3, 3, 3), dtype=bool),
        np.zeros((3, 3, 3), dtype=np.int8),
        np.zeros((3, 3, 3), dtype=np.float32),
        np.zeros((3, 3, 3), dtype=np.complex64),
        np.zeros((3, 3, 3), dtype=object),
    ],
)
def test_rejects_unsupported_dtypes(array):
    with pytest.raises(TypeError, match="native-endian dtypes"):
        bic.skeleton.teasar_labels(array)


def test_rejects_non_native_endian_dtype():
    dtype = np.dtype("<u2") if sys.byteorder == "big" else np.dtype(">u2")
    labels = np.zeros((3, 3, 3), dtype=dtype)
    with pytest.raises(TypeError, match="native-endian dtypes"):
        bic.skeleton.teasar_labels(labels)


@pytest.mark.parametrize("background", [1.5, True, "1"])
def test_rejects_non_integer_background(background):
    with pytest.raises(TypeError, match="background must be an integer"):
        bic.skeleton.teasar_labels(
            np.zeros((3, 3, 3), dtype=np.uint8), background=background
        )


@pytest.mark.parametrize("background", [-1, 256])
def test_rejects_out_of_range_background(background):
    with pytest.raises(ValueError, match="outside the range of dtype uint8"):
        bic.skeleton.teasar_labels(
            np.zeros((3, 3, 3), dtype=np.uint8), background=background
        )


@pytest.mark.parametrize("shape", [(4, 5), (2, 3, 4, 5)])
def test_rejects_wrong_dimensionality(shape):
    with pytest.raises(ValueError, match="labels must have ndim 3"):
        bic.skeleton.teasar_labels(np.zeros(shape, dtype=np.uint32))


def test_direct_binding_validates_ndim_and_spacing():
    parameters = (1.5, 1.0, 100000.0, 4.0, 1)
    with pytest.raises(ValueError, match="labels must have ndim 3"):
        _core._teasar_labels_uint32(
            np.zeros((3, 4), np.uint32), 0, [1.0, 1.0, 1.0], *parameters
        )
    with pytest.raises(ValueError, match="exactly three values"):
        _core._teasar_labels_uint32(
            np.zeros((3, 4, 5), np.uint32), 0, [1.0, 1.0], *parameters
        )


def test_small_random_volumes_satisfy_component_properties():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 4, size=(5, 6, 7), dtype=np.uint16)
        expected = _components_per_label(labels)
        result = bic.skeleton.teasar_labels(labels, number_of_threads=3)
        assert set(result) == set(expected)
        for label, skeleton in result.items():
            _assert_valid_forest(
                labels == label, skeleton, expected[label]
            )
