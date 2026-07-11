import numpy as np

from bioimage_cpp.label_multiset import read_subset


def test_read_subset_basic_merge():
    # Two ranges over a flat ids/counts buffer.
    #  range 0: ids=[1,2], counts=[3,4]
    #  range 1: ids=[2,5], counts=[7,1]
    # merged: {1: 3, 2: 11, 5: 1}
    ids = np.array([1, 2, 2, 5], dtype=np.uint64)
    counts = np.array([3, 4, 7, 1], dtype=np.uint32)
    offsets = np.array([0, 2], dtype=np.uint64)
    sizes = np.array([2, 2], dtype=np.uint64)

    out_ids, out_counts = read_subset(offsets, sizes, ids, counts)
    assert out_ids.tolist() == [1, 2, 5]
    assert out_counts.tolist() == [3, 11, 1]


def test_read_subset_single_range():
    ids = np.array([7, 7, 9], dtype=np.uint64)
    counts = np.array([2, 5, 3], dtype=np.uint32)
    offsets = np.array([0], dtype=np.uint64)
    sizes = np.array([3], dtype=np.uint64)

    out_ids, out_counts = read_subset(offsets, sizes, ids, counts)
    assert out_ids.tolist() == [7, 9]
    assert out_counts.tolist() == [7, 3]


def test_read_subset_no_argsort_returns_same_elements():
    ids = np.array([1, 2, 5], dtype=np.uint64)
    counts = np.array([3, 4, 1], dtype=np.uint32)
    offsets = np.array([0], dtype=np.uint64)
    sizes = np.array([3], dtype=np.uint64)

    sorted_ids, sorted_counts = read_subset(offsets, sizes, ids, counts, argsort=True)
    unsorted_ids, unsorted_counts = read_subset(
        offsets, sizes, ids, counts, argsort=False
    )
    # Same multiset regardless of order.
    assert sorted(sorted_ids.tolist()) == sorted(unsorted_ids.tolist())
    assert sum(sorted_counts.tolist()) == sum(unsorted_counts.tolist())


def test_read_subset_empty_range_list():
    ids = np.array([1, 2], dtype=np.uint64)
    counts = np.array([3, 4], dtype=np.uint32)
    offsets = np.array([], dtype=np.uint64)
    sizes = np.array([], dtype=np.uint64)

    out_ids, out_counts = read_subset(offsets, sizes, ids, counts)
    assert out_ids.shape == (0,)
    assert out_counts.shape == (0,)


def test_read_subset_dtype_promotion():
    # Int Python lists / wrong dtypes should be safely coerced.
    out_ids, out_counts = read_subset([0, 2], [2, 1], [1, 2, 3], [4, 5, 6])
    assert out_ids.tolist() == [1, 2, 3]
    assert out_counts.tolist() == [4, 5, 6]


def test_read_subset_overlapping_ranges_double_count():
    # Same range listed twice should double the counts.
    ids = np.array([1, 2], dtype=np.uint64)
    counts = np.array([3, 4], dtype=np.uint32)
    offsets = np.array([0, 0], dtype=np.uint64)
    sizes = np.array([2, 2], dtype=np.uint64)

    out_ids, out_counts = read_subset(offsets, sizes, ids, counts)
    assert out_ids.tolist() == [1, 2]
    assert out_counts.tolist() == [6, 8]
