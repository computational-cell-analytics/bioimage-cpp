import numpy as np
import pytest

import bioimage_cpp as bic


def _assert_same_partition(a: np.ndarray, b: np.ndarray) -> None:
    """Assert ``a`` and ``b`` describe the same partition of pixels.

    Exact label integers are allowed to differ; what must match is the
    equivalence relation "same label" pixel-by-pixel. We build the bijection
    between the two label sets greedily and assert it stays consistent for
    every pixel.
    """
    assert a.shape == b.shape, (a.shape, b.shape)
    a_flat = a.ravel()
    b_flat = b.ravel()
    a_to_b: dict[int, int] = {}
    b_to_a: dict[int, int] = {}
    for av, bv in zip(a_flat.tolist(), b_flat.tolist()):
        if av in a_to_b:
            assert a_to_b[av] == bv, (av, bv, a_to_b[av])
        else:
            a_to_b[av] = bv
        if bv in b_to_a:
            assert b_to_a[bv] == av, (av, bv, b_to_a[bv])
        else:
            b_to_a[bv] = av


def test_label_2d_two_disjoint_blobs():
    image = np.array(
        [
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
        ],
        dtype=np.uint8,
    )

    labels = bic.segmentation.label(image, connectivity=1)

    # 4-connectivity: four corner blobs, all disjoint.
    assert labels.dtype == np.uint64
    assert labels.shape == image.shape
    unique = sorted(set(labels.ravel().tolist()))
    assert unique == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(labels == 0, image == 0)


def test_label_2d_diagonal_merges_with_full_connectivity():
    image = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )

    labels4 = bic.segmentation.label(image, connectivity=1)
    labels8 = bic.segmentation.label(image, connectivity=2)

    # 4-conn: every foreground pixel is its own component (5 of them).
    assert sorted(set(labels4.ravel().tolist())) == [0, 1, 2, 3, 4, 5]
    # 8-conn: all five foreground pixels join into one component.
    assert sorted(set(labels8.ravel().tolist())) == [0, 1]


def test_label_2d_first_occurrence_raster_order():
    image = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )

    labels = bic.segmentation.label(image, connectivity=1)

    # Four corner components, labeled 1..4 in raster order.
    expected = np.array(
        [
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(labels, expected)


def test_label_2d_equal_value_required_not_just_non_background():
    # Two adjacent non-background regions with different values: must stay
    # separate. This is the defining contrast with "label any non-zero blob".
    image = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.uint8,
    )

    labels = bic.segmentation.label(image, connectivity=2)

    assert sorted(set(labels.ravel().tolist())) == [1, 2]
    # Pixels with the same input value share a label; the two halves differ.
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 2] == labels[1, 3]
    assert labels[0, 0] != labels[0, 2]


def test_label_2d_non_zero_background():
    image = np.array(
        [
            [5, 5, 1, 5, 5],
            [5, 5, 1, 5, 5],
        ],
        dtype=np.int32,
    )

    labels = bic.segmentation.label(image, background=5, connectivity=1)

    # Only the column of 1s is non-background; it forms one component.
    expected = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(labels, expected)


def test_label_3d_smoke_two_blobs():
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[0, 0, 0] = 1
    image[2, 2, 2] = 1

    labels = bic.segmentation.label(image, connectivity=1)

    assert labels.dtype == np.uint64
    assert labels.shape == image.shape
    assert labels[0, 0, 0] == 1
    assert labels[2, 2, 2] == 2
    assert int((labels > 0).sum()) == 2


def test_label_3d_face_vs_diagonal_connectivity():
    # Two foreground pixels touching only at a corner — connected under
    # 26-conn (connectivity=3) but not under 6-conn (connectivity=1) or
    # 18-conn (connectivity=2).
    image = np.zeros((2, 2, 2), dtype=np.uint8)
    image[0, 0, 0] = 1
    image[1, 1, 1] = 1

    labels1 = bic.segmentation.label(image, connectivity=1)
    labels2 = bic.segmentation.label(image, connectivity=2)
    labels3 = bic.segmentation.label(image, connectivity=3)

    assert sorted(set(labels1.ravel().tolist())) == [0, 1, 2]
    assert sorted(set(labels2.ravel().tolist())) == [0, 1, 2]
    assert sorted(set(labels3.ravel().tolist())) == [0, 1]


def test_label_3d_edge_diagonal_connectivity_18():
    # Two pixels sharing an edge (two equal coordinates differ by 1, one
    # equal): connected under 18-conn and 26-conn, not under 6-conn.
    image = np.zeros((2, 2, 2), dtype=np.uint8)
    image[0, 0, 0] = 1
    image[1, 1, 0] = 1

    labels1 = bic.segmentation.label(image, connectivity=1)
    labels2 = bic.segmentation.label(image, connectivity=2)
    labels3 = bic.segmentation.label(image, connectivity=3)

    assert sorted(set(labels1.ravel().tolist())) == [0, 1, 2]
    assert sorted(set(labels2.ravel().tolist())) == [0, 1]
    assert sorted(set(labels3.ravel().tolist())) == [0, 1]


def test_label_2d_all_background():
    image = np.zeros((4, 5), dtype=np.uint8)
    labels = bic.segmentation.label(image)
    np.testing.assert_array_equal(labels, np.zeros_like(image, dtype=np.uint64))


def test_label_2d_all_foreground_single_value():
    image = np.full((3, 4), 7, dtype=np.uint16)
    labels = bic.segmentation.label(image, connectivity=1)
    # Everything is one component.
    np.testing.assert_array_equal(labels, np.full((3, 4), 1, dtype=np.uint64))


def test_label_2d_empty_shape():
    image = np.zeros((0, 5), dtype=np.uint8)
    labels = bic.segmentation.label(image)
    assert labels.shape == (0, 5)
    assert labels.dtype == np.uint64


def test_label_bool_input_treated_as_uint8():
    image = np.array(
        [
            [True, True, False],
            [False, False, False],
            [True, True, True],
        ]
    )

    labels = bic.segmentation.label(image, connectivity=1)

    # Two components.
    assert sorted(set(labels.ravel().tolist())) == [0, 1, 2]
    assert labels[0, 0] == labels[0, 1]
    assert labels[2, 0] == labels[2, 2]
    assert labels[0, 0] != labels[2, 0]


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.uint32, np.uint64, np.int32, np.int64],
)
def test_label_dtype_matrix(dtype):
    image = np.array([[1, 0, 1], [1, 0, 1]], dtype=dtype)
    labels = bic.segmentation.label(image, connectivity=1)
    assert labels.dtype == np.uint64
    assert sorted(set(labels.ravel().tolist())) == [0, 1, 2]


def test_label_accepts_non_contiguous_input():
    base = np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    # Transposed view is non-contiguous.
    transposed = base.T
    assert not transposed.flags["C_CONTIGUOUS"]

    labels = bic.segmentation.label(transposed, connectivity=1)

    # The transposed image has shape (3, 2); the first row is [1, 1] (one
    # component), the second is [0, 0], the third is [1, 1] (another).
    assert labels.shape == (3, 2)
    assert sorted(set(labels.ravel().tolist())) == [0, 1, 2]
    assert labels[0, 0] == labels[0, 1]
    assert labels[2, 0] == labels[2, 1]
    assert labels[0, 0] != labels[2, 0]


def test_label_rejects_1d_input():
    with pytest.raises(ValueError, match="ndim 2 or 3"):
        bic.segmentation.label(np.zeros(5, dtype=np.uint8))


def test_label_rejects_4d_input():
    with pytest.raises(ValueError, match="ndim 2 or 3"):
        bic.segmentation.label(np.zeros((2, 2, 2, 2), dtype=np.uint8))


def test_label_rejects_float_input():
    with pytest.raises(TypeError, match="dtypes"):
        bic.segmentation.label(np.zeros((3, 3), dtype=np.float32))


def test_label_rejects_invalid_connectivity():
    image = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="connectivity"):
        bic.segmentation.label(image, connectivity=0)
    with pytest.raises(ValueError, match="connectivity"):
        bic.segmentation.label(image, connectivity=3)


def test_label_matches_skimage_on_random_binary_2d():
    skimage_measure = pytest.importorskip("skimage.measure")
    rng = np.random.default_rng(0)
    image = (rng.random((32, 48)) > 0.55).astype(np.uint8)

    for connectivity in (1, 2):
        bic_labels = bic.segmentation.label(image, connectivity=connectivity)
        sk_labels = skimage_measure.label(image, connectivity=connectivity)
        _assert_same_partition(bic_labels, sk_labels)


def test_label_matches_skimage_on_random_binary_3d():
    skimage_measure = pytest.importorskip("skimage.measure")
    rng = np.random.default_rng(1)
    image = (rng.random((8, 12, 16)) > 0.6).astype(np.uint8)

    for connectivity in (1, 2, 3):
        bic_labels = bic.segmentation.label(image, connectivity=connectivity)
        sk_labels = skimage_measure.label(image, connectivity=connectivity)
        _assert_same_partition(bic_labels, sk_labels)


def test_label_matches_skimage_on_multi_value_2d():
    # Multi-valued integer image — components are connected runs of the same
    # value, with 0 as background.
    skimage_measure = pytest.importorskip("skimage.measure")
    rng = np.random.default_rng(2)
    image = rng.integers(low=0, high=4, size=(16, 24), dtype=np.int32)

    bic_labels = bic.segmentation.label(image, connectivity=1)
    sk_labels = skimage_measure.label(image, connectivity=1)
    _assert_same_partition(bic_labels, sk_labels)


def test_label_matches_skimage_with_non_zero_background():
    skimage_measure = pytest.importorskip("skimage.measure")
    rng = np.random.default_rng(3)
    image = rng.integers(low=0, high=3, size=(20, 20), dtype=np.int32)

    bic_labels = bic.segmentation.label(image, background=2, connectivity=2)
    sk_labels = skimage_measure.label(image, background=2, connectivity=2)
    _assert_same_partition(bic_labels, sk_labels)


def test_label_binary_path_matches_multi_value_path_2d():
    # The binary fast path (triggered by bool input) must give the same
    # partition as the multi-value path on the same content.
    rng = np.random.default_rng(42)
    mask = rng.random((37, 53)) > 0.55
    for connectivity in (1, 2):
        binary_labels = bic.segmentation.label(mask, connectivity=connectivity)
        multi_labels = bic.segmentation.label(
            mask.astype(np.uint8), connectivity=connectivity
        )
        _assert_same_partition(binary_labels, multi_labels)


def test_label_binary_path_matches_multi_value_path_3d():
    rng = np.random.default_rng(43)
    mask = rng.random((9, 11, 13)) > 0.6
    for connectivity in (1, 2, 3):
        binary_labels = bic.segmentation.label(mask, connectivity=connectivity)
        multi_labels = bic.segmentation.label(
            mask.astype(np.uint8), connectivity=connectivity
        )
        _assert_same_partition(binary_labels, multi_labels)


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (1, 8), (8, 1), (2, 1), (1, 2), (3, 3)],
)
@pytest.mark.parametrize("connectivity", [1, 2])
def test_label_boundary_shapes_2d(shape, connectivity):
    # Exercise the row/column prologue paths (first row, first column, last
    # column). The expected behaviour is "all foreground == one component".
    skimage_measure = pytest.importorskip("skimage.measure")
    image = np.ones(shape, dtype=np.uint8)
    bic_labels = bic.segmentation.label(image, connectivity=connectivity)
    sk_labels = skimage_measure.label(image, connectivity=connectivity)
    _assert_same_partition(bic_labels, sk_labels)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1), (1, 1, 8), (1, 8, 1), (8, 1, 1), (1, 2, 3), (3, 1, 2), (2, 3, 1)],
)
@pytest.mark.parametrize("connectivity", [1, 2, 3])
def test_label_boundary_shapes_3d(shape, connectivity):
    # Exercise every degenerate 3D shape across all connectivities. Catches
    # off-by-one errors in the slice/row/column prologues.
    skimage_measure = pytest.importorskip("skimage.measure")
    image = np.ones(shape, dtype=np.uint8)
    bic_labels = bic.segmentation.label(image, connectivity=connectivity)
    sk_labels = skimage_measure.label(image, connectivity=connectivity)
    _assert_same_partition(bic_labels, sk_labels)


def test_label_checkerboard_stress_2d_4_conn():
    # Checkerboard: every other pixel is foreground; under 4-connectivity,
    # no two foreground pixels touch, so we expect N/2 distinct components.
    # Stresses LabelUnionFind growth (~500k new_label calls) and uint32
    # headroom.
    size = 1024
    image = np.zeros((size, size), dtype=np.uint8)
    image[::2, ::2] = 1
    image[1::2, 1::2] = 1
    labels = bic.segmentation.label(image, connectivity=1)
    assert labels.dtype == np.uint64
    # Expected components: half the pixels are foreground, none touch via
    # 4-conn.
    assert int(labels.max()) == (size * size) // 2
    assert int((labels > 0).sum()) == (size * size) // 2


def test_label_checkerboard_stress_2d_8_conn():
    # Under 8-connectivity the same checkerboard collapses into two
    # components (top-left pattern and the offset pattern), exercising heavy
    # union-find merges.
    size = 256
    image = np.zeros((size, size), dtype=np.uint8)
    image[::2, ::2] = 1
    image[1::2, 1::2] = 1
    labels = bic.segmentation.label(image, connectivity=2)
    unique_labels = set(labels.ravel().tolist())
    assert unique_labels == {0, 1}, unique_labels


def test_label_binary_mode_matches_skimage_random_3d():
    # Direct skimage parity for the binary fast path in 3D.
    skimage_measure = pytest.importorskip("skimage.measure")
    rng = np.random.default_rng(99)
    mask = rng.random((10, 14, 18)) > 0.6
    for connectivity in (1, 2, 3):
        bic_labels = bic.segmentation.label(mask, connectivity=connectivity)
        sk_labels = skimage_measure.label(mask, connectivity=connectivity)
        _assert_same_partition(bic_labels, sk_labels)
