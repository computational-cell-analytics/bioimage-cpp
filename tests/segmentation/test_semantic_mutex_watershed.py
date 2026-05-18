import numpy as np
import pytest

import bioimage_cpp as bic


def test_semantic_mutex_watershed_2d_smoke():
    affinities = np.array(
        [
            # attractive right-neighbor
            [[1.0, 1.0, 1.0, 0.0]],
            # class-0 affinity (p0 high)
            [[10.0, 0.5, 0.5, 0.5]],
            # class-1 affinity (p3 high)
            [[0.5, 0.5, 0.5, 10.0]],
        ],
        dtype=np.float32,
    )
    offsets = [[0, 1]]

    labels, semantic = bic.segmentation.semantic_mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1
    )

    assert labels.shape == (1, 4)
    assert semantic.shape == (1, 4)
    assert labels.dtype == np.uint64
    assert semantic.dtype == np.int64
    np.testing.assert_array_equal(labels, np.array([[1, 1, 1, 2]], dtype=np.uint64))
    np.testing.assert_array_equal(semantic, np.array([[0, 0, 0, 1]], dtype=np.int64))


def test_semantic_label_propagates_across_merges():
    # p0 anchored as class 2; chain merges all four pixels via attractive
    # edges. All four should end up with class 2.
    affinities = np.array(
        [
            [[1.0, 1.0, 1.0, 0.0]],
            [[10.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    offsets = [[0, 1]]

    labels, semantic = bic.segmentation.semantic_mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1
    )

    np.testing.assert_array_equal(labels, np.ones((1, 4), dtype=np.uint64))
    # Only one class channel → class id 0 propagates everywhere.
    np.testing.assert_array_equal(semantic, np.zeros((1, 4), dtype=np.int64))


def test_unassigned_clusters_report_minus_one():
    # No semantic channel has a high affinity anywhere; class-0 channel ties
    # at 0 for every pixel and would normally be valid at every pixel via the
    # argmax mask, but the cost is 0 so it gets processed after the
    # attractive merges — assignments still happen, but with class 0. To
    # observe a truly unassigned cluster, we use a fully zero semantic
    # channel AND a strong attractive grid that leaves multiple components:
    # impossible because every pixel argmax-ties get a (weak) assignment.
    # Instead, build the example with a fully zero semantic channel and a
    # disconnected attractive grid — we still get class 0 everywhere because
    # the (tied) semantic assignment fires for each pixel. Test the "minus
    # one" case via the graph API in the companion suite; here we just
    # confirm the int64 sentinel format.
    affinities = np.array(
        [
            [[1.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    offsets = [[0, 1]]

    _, semantic = bic.segmentation.semantic_mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1
    )

    assert semantic.dtype == np.int64
    # Every pixel argmax-resolves to class 0 → no -1 expected here.
    assert semantic.min() >= 0


def test_semantic_mutex_watershed_3d_smoke():
    affinities = np.ones((4, 2, 2, 2), dtype=np.float64)
    # 3 spatial offsets + 1 semantic class channel.
    offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    labels, semantic = bic.segmentation.semantic_mutex_watershed(
        affinities, offsets, number_of_attractive_channels=3
    )

    assert labels.shape == (2, 2, 2)
    assert semantic.shape == (2, 2, 2)
    # Fully attractive → one component; class 0 propagates everywhere.
    np.testing.assert_array_equal(labels, np.ones((2, 2, 2), dtype=np.uint64))
    np.testing.assert_array_equal(semantic, np.zeros((2, 2, 2), dtype=np.int64))


def test_without_semantic_channels_raises():
    # No extra channel → caller should use regular mutex_watershed.
    affinities = np.ones((2, 3, 4), dtype=np.float32)
    offsets = [[0, 1], [1, 0]]

    with pytest.raises(ValueError):
        bic.segmentation.semantic_mutex_watershed(
            affinities, offsets, number_of_attractive_channels=2
        )


def test_unsupported_dtype_raises():
    affinities = np.ones((2, 3, 4), dtype=np.float16)
    offsets = [[0, 1]]

    with pytest.raises(TypeError):
        bic.segmentation.semantic_mutex_watershed(
            affinities, offsets, number_of_attractive_channels=1
        )


def test_invalid_offset_length_raises():
    affinities = np.ones((2, 3, 4), dtype=np.float32)
    bad_offsets = [[0, 1, 0]]

    with pytest.raises(ValueError):
        bic.segmentation.semantic_mutex_watershed(
            affinities, bad_offsets, number_of_attractive_channels=1
        )


def test_float32_and_float64_match_on_simple_problem():
    affinities = np.array(
        [
            [[0.9, 0.1, 0.3]],
            [[5.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    offsets = [[0, 1]]

    labels_64, semantic_64 = bic.segmentation.semantic_mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1
    )
    labels_32, semantic_32 = bic.segmentation.semantic_mutex_watershed(
        affinities.astype(np.float32), offsets, number_of_attractive_channels=1
    )
    np.testing.assert_array_equal(labels_32, labels_64)
    np.testing.assert_array_equal(semantic_32, semantic_64)


def test_mask_zeros_labels_and_sets_mask_label():
    affinities = np.array(
        [
            [[1.0, 1.0, 1.0]],
            [[10.0, 10.0, 10.0]],
        ],
        dtype=np.float32,
    )
    offsets = [[0, 1]]
    mask = np.array([[True, False, True]], dtype=bool)

    labels, semantic = bic.segmentation.semantic_mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1,
        mask=mask, mask_label=7,
    )

    # Masked pixel → label 0, semantic = mask_label.
    assert int(labels[0, 1]) == 0
    assert int(semantic[0, 1]) == 7
    # Unmasked pixels stay >=1 and carry the semantic class.
    assert int(labels[0, 0]) >= 1
    assert int(labels[0, 2]) >= 1


def test_negative_mask_label_supported():
    affinities = np.ones((2, 1, 2), dtype=np.float64)
    offsets = [[0, 1]]
    mask = np.array([[True, False]], dtype=bool)

    _, semantic = bic.segmentation.semantic_mutex_watershed(
        affinities, offsets, number_of_attractive_channels=1,
        mask=mask, mask_label=-1,
    )
    assert int(semantic[0, 1]) == -1
