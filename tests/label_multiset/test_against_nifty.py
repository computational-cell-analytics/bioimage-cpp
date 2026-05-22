"""Cross-check against nifty.tools.label_multiset (skipped if nifty missing)."""
import numpy as np
import pytest

nifty_tools = pytest.importorskip("nifty.tools")

from bioimage_cpp._core import Blocking as BicBlocking
from bioimage_cpp.label_multiset import (
    MultisetMerger,
    downsample_multiset,
    multiset_from_labels,
    read_subset,
)


def _bic_to_nifty_counts(counts: np.ndarray) -> np.ndarray:
    return counts.astype(np.int32)


def _entries(ids: np.ndarray, counts: np.ndarray, offsets: np.ndarray,
             entry_sizes: np.ndarray, entry_offsets: np.ndarray):
    """Yield per-spatial-position (ids_tuple, counts_tuple) for comparison."""
    n = offsets.shape[0]
    out = []
    for s in range(n):
        off = int(offsets[s])
        eidx = int(entry_offsets[s])
        sz = int(entry_sizes[eidx])
        out.append(
            (tuple(int(x) for x in ids[off : off + sz]),
             tuple(int(x) for x in counts[off : off + sz]))
        )
    return out


def _entries_from_ms(ms):
    return _entries(ms.ids, ms.counts, ms.offsets, ms.entry_sizes, ms.entry_offsets)


def test_read_subset_matches_nifty():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 10, size=20, dtype=np.uint64)
    counts = rng.integers(1, 5, size=20, dtype=np.uint32)
    offsets = np.array([0, 5, 10, 15], dtype=np.uint64)
    sizes = np.array([5, 5, 5, 5], dtype=np.uint64)

    bic_ids, bic_counts = read_subset(offsets, sizes, ids, counts)

    n_ids, n_counts = nifty_tools.readSubset(
        offsets.astype(np.uint64),
        sizes.astype(np.uint64),
        ids.astype(np.uint64),
        _bic_to_nifty_counts(counts),
        True,
    )
    # Both should be sorted by id and equal element-wise (up to count dtype).
    assert bic_ids.tolist() == n_ids.tolist()
    assert bic_counts.tolist() == n_counts.astype(np.uint32).tolist()


def test_downsample_matches_nifty_3d():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 8, size=(4, 4, 4), dtype=np.uint64)

    # Build the level-0 multiset *both ways* and downsample with both.
    ms0 = multiset_from_labels(labels, (1, 1, 1))

    # nifty's downsampleMultiset signature.
    n_blocking = nifty_tools.blocking(
        roiBegin=[0, 0, 0], roiEnd=list(labels.shape), blockShape=[2, 2, 2]
    )
    bic_blocking = BicBlocking([0, 0, 0], list(labels.shape), [2, 2, 2])

    # Convert our level-0 to nifty types for input.
    n_offsets = ms0.offsets.astype(np.uint64)
    n_entry_sizes = ms0.entry_sizes.astype(np.uint64)
    n_entry_offsets = ms0.entry_offsets.astype(np.uint64)
    n_ids = ms0.ids.astype(np.uint64)
    n_counts = _bic_to_nifty_counts(ms0.counts)

    n_argmax, n_new_offsets, n_new_ids, n_new_counts = nifty_tools.downsampleMultiset(
        n_blocking, n_offsets, n_entry_sizes, n_entry_offsets, n_ids, n_counts,
        restrict_set=-1,
    )

    bic = downsample_multiset(ms0, bic_blocking)

    # Argmax should match.
    assert bic.argmax.tolist() == n_argmax.tolist()

    # Reconstruct entries from nifty's output. nifty returns flat
    # (new_offsets pointing into new_ids/new_counts) but not new_entry_sizes —
    # since nifty's downsample reorders/uniqifies offsets implicitly, we
    # rebuild sizes via np.unique on offsets.
    unique_off, inverse = np.unique(n_new_offsets, return_inverse=True)
    # Sizes are the gaps between consecutive unique offsets, with the last
    # entry extending to len(new_ids).
    sizes_arr = np.diff(
        np.concatenate([unique_off, np.array([n_new_ids.shape[0]], dtype=unique_off.dtype)])
    )
    nifty_entries = _entries(
        n_new_ids,
        n_new_counts.astype(np.uint32),
        n_new_offsets.astype(np.uint64),
        sizes_arr.astype(np.uint64),
        inverse.astype(np.uint64),
    )
    bic_entries = _entries_from_ms(bic)
    assert bic_entries == nifty_entries


def test_multiset_merger_matches_nifty():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 4, size=(4, 4), dtype=np.uint64)
    ms = multiset_from_labels(labels, (2, 2))

    unique_offsets = np.array(
        [int(ms.offsets[np.where(ms.entry_offsets == e)[0][0]])
         for e in range(ms.n_entries)],
        dtype=np.uint64,
    )

    # Both constructors expect (unique_offsets, entry_sizes, ids, counts).
    bic_merger = MultisetMerger.from_multiset(ms)
    n_merger = nifty_tools.MultisetMerger(
        unique_offsets,
        ms.entry_sizes.astype(np.uint64),
        ms.ids.astype(np.uint64),
        _bic_to_nifty_counts(ms.counts),
    )

    # Ingest the same multiset as a batch — everything should be deduped.
    spatial_offsets_bic = ms.entry_offsets.astype(np.uint64).copy()
    spatial_offsets_n = ms.entry_offsets.astype(np.uint64).copy()

    bic_merger.update(
        unique_offsets, ms.entry_sizes.astype(np.uint64),
        ms.ids, ms.counts, spatial_offsets_bic,
    )
    n_merger.update(
        unique_offsets, ms.entry_sizes.astype(np.uint64),
        ms.ids.astype(np.uint64), _bic_to_nifty_counts(ms.counts),
        spatial_offsets_n,
    )

    assert bic_merger.ids.tolist() == ms.ids.tolist()
    assert bic_merger.counts.tolist() == ms.counts.tolist()
    assert n_merger.get_ids().tolist() == ms.ids.tolist()
    assert spatial_offsets_bic.tolist() == spatial_offsets_n.tolist()
