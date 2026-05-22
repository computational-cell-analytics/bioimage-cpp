"""Label multiset data structure.

A label multiset is a sparse, deduplicated representation of label
distributions over a grid of spatial blocks. For each spatial position
(block) it stores a histogram of the labels in the corresponding fine
region as a `(ids, counts)` pair, with identical histograms across
different blocks pointing to the same shared storage.

This is a re-implementation of the label-multiset utilities from
`nifty.tools.label_multiset` with no external C++ dependencies.

Storage layout (mirrors nifty):

- ``offsets``         length ``n_spatial``: spatial position to byte offset into ``ids`` / ``counts``
- ``entry_offsets``   length ``n_spatial``: spatial position to unique-entry index
- ``entry_sizes``     length ``n_unique``: number of ``(id, count)`` pairs per entry
- ``ids``             length ``total_elems``: concatenated label ids (sorted within each entry)
- ``counts``          length ``total_elems``: counts aligned with ``ids``
- ``argmax``          length ``n_spatial``: argmax label per spatial position
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from .. import _core
from .._core import Blocking

_ID_DTYPE = np.dtype(np.uint64)
_COUNT_DTYPE = np.dtype(np.uint32)
_OFFSET_DTYPE = np.dtype(np.uint64)


@dataclass
class LabelMultiset:
    """A deduplicated label-histogram representation over a spatial grid."""

    argmax: np.ndarray  # shape (n_spatial,), dtype uint64
    offsets: np.ndarray  # shape (n_spatial,), dtype uint64
    entry_offsets: np.ndarray  # shape (n_spatial,), dtype uint64
    entry_sizes: np.ndarray  # shape (n_unique,), dtype uint64
    ids: np.ndarray  # shape (total_elems,), dtype uint64
    counts: np.ndarray  # shape (total_elems,), dtype uint32

    @property
    def n_spatial(self) -> int:
        return int(self.offsets.shape[0])

    @property
    def n_entries(self) -> int:
        return int(self.entry_sizes.shape[0])

    def entry(self, spatial_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(ids, counts)`` for the multiset at ``spatial_index``."""
        off = int(self.offsets[spatial_index])
        entry_idx = int(self.entry_offsets[spatial_index])
        size = int(self.entry_sizes[entry_idx])
        return self.ids[off : off + size], self.counts[off : off + size]

    def __getitem__(self, spatial_index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.entry(spatial_index)


def _as_blocking(
    shape: Tuple[int, ...],
    block_shape: Tuple[int, ...],
) -> Blocking:
    if len(shape) != len(block_shape):
        raise ValueError(
            f"shape and block_shape must have the same length, got "
            f"{len(shape)} and {len(block_shape)}"
        )
    roi_begin = [0] * len(shape)
    return Blocking(roi_begin, list(shape), list(block_shape))


def multiset_from_labels(
    labels: np.ndarray,
    block_shape: Tuple[int, ...],
) -> LabelMultiset:
    """Build a level-0 label multiset by aggregating ``labels`` over blocks.

    For each block of shape ``block_shape``, computes the label histogram of
    the contained voxels. Identical histograms are deduplicated.
    """
    labels = np.ascontiguousarray(labels)
    if labels.dtype == np.uint32:
        fn = _core._multiset_from_labels_u32
    elif labels.dtype == np.uint64:
        fn = _core._multiset_from_labels_u64
    else:
        raise TypeError(
            f"labels must have dtype uint32 or uint64, got {labels.dtype}"
        )
    blocking = _as_blocking(tuple(labels.shape), tuple(block_shape))
    argmax, offsets, entry_offsets, entry_sizes, ids, counts = fn(labels, blocking)
    return LabelMultiset(
        argmax=argmax,
        offsets=offsets,
        entry_offsets=entry_offsets,
        entry_sizes=entry_sizes,
        ids=ids,
        counts=counts,
    )


def downsample_multiset(
    multiset: LabelMultiset,
    blocking: Blocking,
    restrict_set: int = -1,
) -> LabelMultiset:
    """Downsample ``multiset`` by aggregating its entries into ``blocking``'s blocks.

    ``blocking`` must be defined over the same spatial extent as the input
    multiset (i.e. ``prod(blocking.roi_end) == multiset.n_spatial``).
    """
    argmax, new_offsets, new_entry_offsets, new_entry_sizes, new_ids, new_counts = (
        _core._downsample_multiset(
            blocking,
            multiset.offsets,
            multiset.entry_sizes,
            multiset.entry_offsets,
            multiset.ids,
            multiset.counts,
            restrict_set,
        )
    )
    return LabelMultiset(
        argmax=argmax,
        offsets=new_offsets,
        entry_offsets=new_entry_offsets,
        entry_sizes=new_entry_sizes,
        ids=new_ids,
        counts=new_counts,
    )


def read_subset(
    offsets: np.ndarray,
    sizes: np.ndarray,
    ids: np.ndarray,
    counts: np.ndarray,
    argsort: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge multisets located at the given ``(offset, size)`` ranges.

    Returns the summed ``(ids, counts)``, sorted by id if ``argsort``.
    """
    offsets = np.ascontiguousarray(offsets, dtype=_OFFSET_DTYPE)
    sizes = np.ascontiguousarray(sizes, dtype=_OFFSET_DTYPE)
    ids = np.ascontiguousarray(ids, dtype=_ID_DTYPE)
    counts = np.ascontiguousarray(counts, dtype=_COUNT_DTYPE)
    return _core._read_subset(offsets, sizes, ids, counts, argsort)


def _unique_offsets_of(multiset: "LabelMultiset") -> np.ndarray:
    """For each unique entry of ``multiset``, return the byte offset into ids/counts."""
    n = multiset.n_entries
    out = np.empty(n, dtype=_OFFSET_DTYPE)
    for e in range(n):
        out[e] = multiset.offsets[np.where(multiset.entry_offsets == e)[0][0]]
    return out


class MultisetMerger:
    """Stateful deduplicating merger for multisets produced in batches.

    The constructor expects one offset *per unique entry* (i.e. arrays of
    length ``n_unique``, not ``n_spatial``). Use :meth:`from_multiset` to
    build one straight from a :class:`LabelMultiset`.

    Call :meth:`update` with subsequent batches; each call extends the
    internal storage with any genuinely new entries and rewrites the
    passed-in ``offsets`` array so each spatial position points at its
    final deduplicated byte offset.
    """

    def __init__(
        self,
        unique_offsets: np.ndarray,
        entry_sizes: np.ndarray,
        ids: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        unique_offsets = np.ascontiguousarray(unique_offsets, dtype=_OFFSET_DTYPE)
        entry_sizes = np.ascontiguousarray(entry_sizes, dtype=_OFFSET_DTYPE)
        if unique_offsets.shape != entry_sizes.shape:
            raise ValueError(
                "unique_offsets and entry_sizes must have the same length "
                "(one entry each). Use MultisetMerger.from_multiset() if you "
                "have a LabelMultiset instead."
            )
        self._impl = _core._MultisetMerger(
            unique_offsets,
            entry_sizes,
            np.ascontiguousarray(ids, dtype=_ID_DTYPE),
            np.ascontiguousarray(counts, dtype=_COUNT_DTYPE),
        )

    @classmethod
    def from_multiset(cls, multiset: "LabelMultiset") -> "MultisetMerger":
        """Build a merger seeded with the unique entries of ``multiset``."""
        return cls(
            _unique_offsets_of(multiset),
            multiset.entry_sizes,
            multiset.ids,
            multiset.counts,
        )

    def update(
        self,
        unique_offsets: np.ndarray,
        entry_sizes: np.ndarray,
        ids: np.ndarray,
        counts: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        """Ingest a batch of entries and rewrite ``offsets`` in-place.

        ``offsets`` is mutated in-place and also returned.
        """
        if offsets.dtype != _OFFSET_DTYPE or not offsets.flags["C_CONTIGUOUS"]:
            raise TypeError(
                "offsets must be a contiguous uint64 array (it is modified in place)"
            )
        return self._impl.update(
            np.ascontiguousarray(unique_offsets, dtype=_OFFSET_DTYPE),
            np.ascontiguousarray(entry_sizes, dtype=_OFFSET_DTYPE),
            np.ascontiguousarray(ids, dtype=_ID_DTYPE),
            np.ascontiguousarray(counts, dtype=_COUNT_DTYPE),
            offsets,
        )

    @property
    def ids(self) -> np.ndarray:
        return self._impl.get_ids()

    @property
    def counts(self) -> np.ndarray:
        return self._impl.get_counts()

    @property
    def offsets(self) -> np.ndarray:
        return self._impl.get_offsets()

    @property
    def entry_sizes(self) -> np.ndarray:
        return self._impl.get_entry_sizes()


__all__ = [
    "LabelMultiset",
    "MultisetMerger",
    "downsample_multiset",
    "multiset_from_labels",
    "read_subset",
]
