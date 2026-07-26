"""Label multiset data structure.

A label multiset is a sparse, deduplicated representation of label
distributions over a grid of spatial blocks. For each spatial position
(block) it stores a histogram of the labels in the corresponding fine
region as a `(ids, counts)` pair, with identical histograms across
different blocks pointing to the same shared storage.

This is a re-implementation of the label-multiset utilities from
`nifty.tools.label_multiset` with no external C++ dependencies.

Storage layout (mirrors nifty):

- ``offsets``         length ``n_spatial``: spatial position to element offset into ``ids`` / ``counts``
- ``entry_offsets``   length ``n_spatial``: spatial position to unique-entry index
- ``entry_sizes``     length ``n_unique``: number of ``(id, count)`` pairs per entry
- ``ids``             length ``total_elems``: concatenated label ids (sorted within each entry)
- ``counts``          length ``total_elems``: counts aligned with ``ids``
- ``argmax``          length ``n_spatial``: argmax label per spatial position
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .. import _core
from .._core import Blocking
from .._validation import strict_integer_array

_ID_DTYPE = np.dtype(np.uint64)
_COUNT_DTYPE = np.dtype(np.uint32)
_OFFSET_DTYPE = np.dtype(np.uint64)


def _as_integer_metadata(values, name: str, dtype: np.dtype) -> np.ndarray:
    untyped = np.asarray(values)
    if untyped.size == 0 and untyped.ndim == 1:
        return np.ascontiguousarray(untyped, dtype=dtype)
    return strict_integer_array(
        values,
        name,
        dtype=dtype,
        ndim=1,
        non_negative=True,
    )


def _validate_multiset_arrays(
    argmax,
    offsets,
    entry_offsets,
    entry_sizes,
    ids,
    counts,
) -> tuple[np.ndarray, ...]:
    arrays = (
        _as_integer_metadata(argmax, "argmax", _ID_DTYPE),
        _as_integer_metadata(offsets, "offsets", _OFFSET_DTYPE),
        _as_integer_metadata(entry_offsets, "entry_offsets", _OFFSET_DTYPE),
        _as_integer_metadata(entry_sizes, "entry_sizes", _OFFSET_DTYPE),
        _as_integer_metadata(ids, "ids", _ID_DTYPE),
        _as_integer_metadata(counts, "counts", _COUNT_DTYPE),
    )
    argmax_array, offset_array, entry_offset_array, size_array, id_array, count_array = arrays

    if id_array.size != count_array.size:
        raise ValueError(
            "ids and counts must have the same length, got "
            f"{id_array.size} and {count_array.size}"
        )
    n_spatial = offset_array.size
    if argmax_array.size != n_spatial or entry_offset_array.size != n_spatial:
        raise ValueError(
            "argmax, offsets, and entry_offsets must have the same length, got "
            f"{argmax_array.size}, {n_spatial}, and {entry_offset_array.size}"
        )
    if n_spatial == 0:
        if size_array.size != 0 or id_array.size != 0:
            raise ValueError(
                "an empty LabelMultiset must have empty entry_sizes, ids, and counts"
            )
        return arrays
    if size_array.size == 0:
        raise ValueError(
            "entry_sizes must not be empty when spatial entries are present"
        )
    if np.any(size_array == 0):
        raise ValueError("entry_sizes must contain only positive values")
    if np.any(entry_offset_array >= size_array.size):
        raise ValueError(
            "entry_offsets values must be less than the length of entry_sizes"
        )
    referenced = np.bincount(
        entry_offset_array.astype(np.intp, copy=False),
        minlength=size_array.size,
    )
    if np.any(referenced == 0):
        raise ValueError("every entry_sizes element must be referenced by entry_offsets")
    if np.any(offset_array > id_array.size):
        raise ValueError("offsets values must not exceed the flat storage length")
    selected_sizes = size_array[entry_offset_array.astype(np.intp, copy=False)]
    remaining = id_array.size - offset_array
    if np.any(selected_sizes > remaining):
        raise ValueError("an entry range exceeds the flat storage length")
    return arrays


def _validated_multiset(multiset: "LabelMultiset") -> tuple[np.ndarray, ...]:
    if not isinstance(multiset, LabelMultiset):
        raise TypeError("multiset must be a LabelMultiset")
    return _validate_multiset_arrays(
        multiset.argmax,
        multiset.offsets,
        multiset.entry_offsets,
        multiset.entry_sizes,
        multiset.ids,
        multiset.counts,
    )


@dataclass
class LabelMultiset:
    """A validated, deduplicated label histogram over a spatial grid.

    Construction converts integer inputs to the storage dtypes. Native
    operations validate the arrays again because the dataclass is mutable.
    """

    argmax: np.ndarray  # shape (n_spatial,), dtype uint64
    offsets: np.ndarray  # shape (n_spatial,), dtype uint64
    entry_offsets: np.ndarray  # shape (n_spatial,), dtype uint64
    entry_sizes: np.ndarray  # shape (n_unique,), dtype uint64
    ids: np.ndarray  # shape (total_elems,), dtype uint64
    counts: np.ndarray  # shape (total_elems,), dtype uint32

    def __post_init__(self) -> None:
        (
            self.argmax,
            self.offsets,
            self.entry_offsets,
            self.entry_sizes,
            self.ids,
            self.counts,
        ) = _validate_multiset_arrays(
            self.argmax,
            self.offsets,
            self.entry_offsets,
            self.entry_sizes,
            self.ids,
            self.counts,
        )

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
    (
        _,
        offsets,
        entry_offsets,
        entry_sizes,
        ids,
        counts,
    ) = _validated_multiset(multiset)
    argmax, new_offsets, new_entry_offsets, new_entry_sizes, new_ids, new_counts = (
        _core._downsample_multiset(
            blocking,
            offsets,
            entry_sizes,
            entry_offsets,
            ids,
            counts,
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
    offsets = _as_integer_metadata(offsets, "offsets", _OFFSET_DTYPE)
    sizes = _as_integer_metadata(sizes, "sizes", _OFFSET_DTYPE)
    ids = _as_integer_metadata(ids, "ids", _ID_DTYPE)
    counts = _as_integer_metadata(counts, "counts", _COUNT_DTYPE)
    return _core._read_subset(offsets, sizes, ids, counts, argsort)


def _unique_offsets_of(
    offsets: np.ndarray,
    entry_offsets: np.ndarray,
    n_entries: int,
) -> np.ndarray:
    """Return one element offset for each unique multiset entry."""
    n = int(n_entries)
    out = np.empty(n, dtype=_OFFSET_DTYPE)
    for e in range(n):
        out[e] = offsets[np.flatnonzero(entry_offsets == e)[0]]
    return out


class MultisetMerger:
    """Stateful deduplicating merger for multisets produced in batches.

    The constructor expects one offset *per unique entry* (i.e. arrays of
    length ``n_unique``, not ``n_spatial``). Use :meth:`from_multiset` to
    build one straight from a :class:`LabelMultiset`.

    Call :meth:`update` with subsequent batches; each call extends the
    internal storage with any genuinely new entries and rewrites the
    passed-in ``offsets`` array so each spatial position points at its
    final deduplicated element offset.
    """

    def __init__(
        self,
        unique_offsets: np.ndarray,
        entry_sizes: np.ndarray,
        ids: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        unique_offsets = _as_integer_metadata(
            unique_offsets, "unique_offsets", _OFFSET_DTYPE
        )
        entry_sizes = _as_integer_metadata(
            entry_sizes, "entry_sizes", _OFFSET_DTYPE
        )
        if unique_offsets.shape != entry_sizes.shape:
            raise ValueError(
                "unique_offsets and entry_sizes must have the same length "
                "(one entry each). Use MultisetMerger.from_multiset() if you "
                "have a LabelMultiset instead."
            )
        self._impl = _core._MultisetMerger(
            unique_offsets,
            entry_sizes,
            _as_integer_metadata(ids, "ids", _ID_DTYPE),
            _as_integer_metadata(counts, "counts", _COUNT_DTYPE),
        )

    @classmethod
    def from_multiset(cls, multiset: "LabelMultiset") -> "MultisetMerger":
        """Build a merger seeded with the unique entries of ``multiset``."""
        _, offsets, entry_offsets, entry_sizes, ids, counts = _validated_multiset(
            multiset
        )
        return cls(
            _unique_offsets_of(offsets, entry_offsets, entry_sizes.size),
            entry_sizes,
            ids,
            counts,
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
        if (
            not isinstance(offsets, np.ndarray)
            or offsets.dtype != _OFFSET_DTYPE
            or offsets.ndim != 1
            or not offsets.flags["C_CONTIGUOUS"]
            or not offsets.flags["WRITEABLE"]
        ):
            raise TypeError(
                "offsets must be a writable contiguous 1D uint64 array"
            )
        return self._impl.update(
            _as_integer_metadata(unique_offsets, "unique_offsets", _OFFSET_DTYPE),
            _as_integer_metadata(entry_sizes, "entry_sizes", _OFFSET_DTYPE),
            _as_integer_metadata(ids, "ids", _ID_DTYPE),
            _as_integer_metadata(counts, "counts", _COUNT_DTYPE),
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
