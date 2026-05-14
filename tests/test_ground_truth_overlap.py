import numpy as np
import pytest

import bioimage_cpp as bic


def test_segmentation_overlap_tables_and_counts():
    labels_a = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.uint64)
    labels_b = np.array([[5, 5, 5], [6, 6, 7]], dtype=np.uint32)

    overlap = bic.ground_truth.segmentation_overlap(labels_a, labels_b)

    assert overlap.total_count == 6
    np.testing.assert_array_equal(overlap.labels_a, np.array([1, 2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(overlap.labels_b, np.array([5, 6, 7], dtype=np.uint64))
    assert overlap.count_a(1) == 3
    assert overlap.count_b(5) == 3
    assert overlap.overlap_count(1, 5) == 2
    assert overlap.overlap_count(42, 5) == 0

    counts_a = overlap.counts_a_table()
    assert counts_a.dtype.names == ("label", "count")
    np.testing.assert_array_equal(counts_a["label"], [1, 2, 3])
    np.testing.assert_array_equal(counts_a["count"], [3, 2, 1])

    table = overlap.overlap_table()
    assert table.dtype.names == ("label_a", "label_b", "count")
    np.testing.assert_array_equal(table["label_a"], [1, 1, 2, 2, 3])
    np.testing.assert_array_equal(table["label_b"], [5, 6, 5, 7, 6])
    np.testing.assert_array_equal(table["count"], [2, 1, 1, 1, 1])


def test_segmentation_overlap_normalized_tables():
    labels_a = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.uint64)
    labels_b = np.array([[5, 5, 5], [6, 6, 7]], dtype=np.uint64)
    overlap = bic.ground_truth.segmentation_overlap(labels_a, labels_b)

    by_a = overlap.overlap_table(normalize_by="a")
    assert by_a.dtype.names == ("label_a", "label_b", "count", "fraction")
    np.testing.assert_allclose(by_a["fraction"], [2 / 3, 1 / 3, 1 / 2, 1 / 2, 1.0])

    by_b = overlap.overlap_table(normalize_by="b")
    np.testing.assert_allclose(by_b["fraction"], [2 / 3, 1 / 2, 1 / 3, 1.0, 1 / 2])

    by_total = overlap.overlap_table(normalize_by="total")
    np.testing.assert_allclose(by_total["fraction"], [2 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])

    with pytest.raises(ValueError, match="normalize_by"):
        overlap.overlap_table(normalize_by="bad")


def test_per_label_overlaps_and_best_overlap_are_clear():
    labels_a = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.uint64)
    labels_b = np.array([[5, 5, 5], [6, 6, 7]], dtype=np.uint64)
    overlap = bic.ground_truth.segmentation_overlap(labels_a, labels_b)

    overlaps_a = overlap.overlaps_for_label_a(1, normalize=True)
    assert overlaps_a.dtype.names == ("label", "count", "fraction")
    np.testing.assert_array_equal(overlaps_a["label"], [5, 6])
    np.testing.assert_array_equal(overlaps_a["count"], [2, 1])
    np.testing.assert_allclose(overlaps_a["fraction"], [2 / 3, 1 / 3])

    overlaps_b = overlap.overlaps_for_label_b(5)
    assert overlaps_b.dtype.names == ("label", "count")
    np.testing.assert_array_equal(overlaps_b["label"], [1, 2])
    np.testing.assert_array_equal(overlaps_b["count"], [2, 1])

    best_a = overlap.best_overlap_for_label_a(1)
    assert best_a.label == 5
    assert best_a.count == 2
    assert best_a.fraction == pytest.approx(2 / 3)
    assert best_a.found

    best_b = overlap.best_overlap_for_label_b(5)
    assert best_b.label == 1
    assert best_b.count == 2
    assert best_b.fraction == pytest.approx(2 / 3)
    assert best_b.found

    missing = overlap.best_overlap_for_label_a(99)
    assert missing == bic.ground_truth.BestOverlap(label=0, count=0, fraction=0.0, found=False)


def test_zero_label_handling_and_different_overlap():
    labels_a = np.array([1, 1, 2, 2, 3], dtype=np.uint64)
    labels_b = np.array([0, 5, 0, 6, 6], dtype=np.uint64)
    overlap = bic.ground_truth.segmentation_overlap(labels_a, labels_b)

    assert overlap.is_label_a_overlapping_with_zero(1)
    assert overlap.is_label_b_overlapping_with_zero(0) is False
    assert overlap.best_overlap_for_label_a(1).label == 0
    assert overlap.best_overlap_for_label_a(1, ignore_zero=True).label == 5
    assert overlap.best_overlap_for_label_a(1, ignore_zero=True).found
    assert not overlap.best_overlap_for_label_a(99, ignore_zero=True).found

    assert overlap.different_overlap(1, 2) == pytest.approx(0.75)
    with pytest.raises(IndexError, match="labels must exist"):
        overlap.different_overlap(1, 99)
    with pytest.raises(ValueError, match="non-negative"):
        overlap.count_a(-1)


def test_sparse_large_labels_do_not_require_dense_max_label_storage():
    labels_a = np.array([1, 1_000_000_000_000], dtype=np.uint64)
    labels_b = np.array([7, 8], dtype=np.uint64)

    overlap = bic.ground_truth.segmentation_overlap(labels_a, labels_b)

    np.testing.assert_array_equal(
        overlap.labels_a,
        np.array([1, 1_000_000_000_000], dtype=np.uint64),
    )
    np.testing.assert_array_equal(overlap.overlap_table()["count"], [1, 1])


def test_overlap_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same shape"):
        bic.ground_truth.segmentation_overlap(
            np.ones((2, 2), dtype=np.uint64),
            np.ones((2, 3), dtype=np.uint64),
        )

    with pytest.raises(TypeError, match="integer dtype"):
        bic.ground_truth.segmentation_overlap(
            np.ones((2,), dtype=np.float32),
            np.ones((2,), dtype=np.uint64),
        )

    with pytest.raises(ValueError, match="negative labels"):
        bic.ground_truth.segmentation_overlap(
            np.array([-1, 2], dtype=np.int64),
            np.ones((2,), dtype=np.uint64),
        )

    with pytest.raises(ValueError, match="at least one dimension"):
        bic.ground_truth.segmentation_overlap(
            np.array(1, dtype=np.uint64),
            np.array(1, dtype=np.uint64),
        )
