import numpy as np

from bioimage_cpp._core import Blocking
from bioimage_cpp.label_multiset import (
    MultisetMerger,
    downsample_multiset,
    multiset_from_labels,
)


def _unique_offsets(ms):
    """For each unique entry, return the offset where it lives in ms.ids."""
    return np.array(
        [int(ms.offsets[np.where(ms.entry_offsets == e)[0][0]])
         for e in range(ms.n_entries)],
        dtype=np.uint64,
    )


def test_merger_init_roundtrips_inputs():
    offsets = np.array([0, 2], dtype=np.uint64)
    entry_sizes = np.array([2, 1], dtype=np.uint64)
    ids = np.array([1, 2, 5], dtype=np.uint64)
    counts = np.array([3, 4, 7], dtype=np.uint32)

    m = MultisetMerger(offsets, entry_sizes, ids, counts)
    assert m.ids.tolist() == [1, 2, 5]
    assert m.counts.tolist() == [3, 4, 7]
    assert m.offsets.tolist() == [0, 2]
    assert m.entry_sizes.tolist() == [2, 1]


def test_merger_deduplicates_identical_entry():
    # Existing storage has entry A: ids=[1,2], counts=[3,4]
    offsets = np.array([0], dtype=np.uint64)
    entry_sizes = np.array([2], dtype=np.uint64)
    ids = np.array([1, 2], dtype=np.uint64)
    counts = np.array([3, 4], dtype=np.uint32)
    m = MultisetMerger(offsets, entry_sizes, ids, counts)

    # New batch: one entry identical to A, one genuinely new.
    batch_unique_offsets = np.array([0, 2], dtype=np.uint64)
    batch_entry_sizes = np.array([2, 1], dtype=np.uint64)
    batch_ids = np.array([1, 2, 9], dtype=np.uint64)
    batch_counts = np.array([3, 4, 11], dtype=np.uint32)
    spatial_offsets = np.array([0, 1], dtype=np.uint64)

    rewritten = m.update(
        batch_unique_offsets, batch_entry_sizes, batch_ids, batch_counts,
        spatial_offsets,
    )
    assert rewritten.tolist() == [0, 2]
    assert m.ids.tolist() == [1, 2, 9]
    assert m.counts.tolist() == [3, 4, 11]
    assert m.offsets.tolist() == [0, 2]
    assert m.entry_sizes.tolist() == [2, 1]


def test_merger_update_with_identical_entries_is_idempotent():
    # Building a merger from a multiset and feeding the same multiset back as
    # an update should leave ids/counts unchanged and rewrite spatial offsets
    # to point at the original byte offsets.
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 4, size=(4, 4), dtype=np.uint64)
    ms = multiset_from_labels(labels, (2, 2))

    m = MultisetMerger.from_multiset(ms)
    spatial_in = ms.entry_offsets.astype(np.uint64).copy()
    rewritten = m.update(
        _unique_offsets(ms),
        ms.entry_sizes.astype(np.uint64),
        ms.ids, ms.counts,
        spatial_in,
    )
    assert m.ids.tolist() == ms.ids.tolist()
    assert m.counts.tolist() == ms.counts.tolist()
    assert rewritten.tolist() == ms.offsets.tolist()


def test_merger_from_multiset_with_real_dedup():
    # Build a multiset that actually deduplicates (all-zero volume → 1 entry,
    # many spatial positions). Ensure from_multiset() correctly extracts the
    # unique offsets and the merger handles the size mismatch (n_spatial >>
    # n_unique).
    labels = np.zeros((8, 8), dtype=np.uint64)
    ms = multiset_from_labels(labels, (2, 2))
    assert ms.n_spatial == 16
    assert ms.n_entries == 1

    m = MultisetMerger.from_multiset(ms)
    assert m.entry_sizes.tolist() == [1]
    assert m.ids.tolist() == [0]
    assert m.counts.tolist() == [4]

    # Update with the same multiset — must remain idempotent.
    spatial = ms.entry_offsets.astype(np.uint64).copy()
    rewritten = m.update(
        _unique_offsets(ms),
        ms.entry_sizes.astype(np.uint64),
        ms.ids, ms.counts,
        spatial,
    )
    assert m.entry_sizes.tolist() == [1]
    assert rewritten.tolist() == [0] * 16


def test_merger_grows_on_disjoint_update():
    # Two non-overlapping multisets fed in sequence — every entry from the
    # second must be appended.
    offsets_a = np.array([0, 1], dtype=np.uint64)
    entry_sizes_a = np.array([1, 1], dtype=np.uint64)
    ids_a = np.array([10, 20], dtype=np.uint64)
    counts_a = np.array([5, 7], dtype=np.uint32)

    m = MultisetMerger(offsets_a, entry_sizes_a, ids_a, counts_a)

    unique_offsets_b = np.array([0, 1], dtype=np.uint64)
    entry_sizes_b = np.array([1, 1], dtype=np.uint64)
    ids_b = np.array([30, 40], dtype=np.uint64)
    counts_b = np.array([9, 11], dtype=np.uint32)
    spatial_b = np.array([0, 1], dtype=np.uint64)

    rewritten = m.update(
        unique_offsets_b, entry_sizes_b, ids_b, counts_b, spatial_b,
    )
    assert m.ids.tolist() == [10, 20, 30, 40]
    assert m.counts.tolist() == [5, 7, 9, 11]
    assert m.offsets.tolist() == [0, 1, 2, 3]
    assert rewritten.tolist() == [2, 3]
