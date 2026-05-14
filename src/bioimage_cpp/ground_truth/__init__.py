"""Ground-truth comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .. import _core

_COUNT_TABLE_DTYPE = np.dtype([("label", np.uint64), ("count", np.uint64)])
_OVERLAP_TABLE_DTYPE = np.dtype(
    [("label_a", np.uint64), ("label_b", np.uint64), ("count", np.uint64)]
)
_OVERLAP_FRACTION_TABLE_DTYPE = np.dtype(
    [
        ("label_a", np.uint64),
        ("label_b", np.uint64),
        ("count", np.uint64),
        ("fraction", np.float64),
    ]
)
_LABEL_OVERLAP_TABLE_DTYPE = np.dtype(
    [("label", np.uint64), ("count", np.uint64)]
)
_LABEL_OVERLAP_FRACTION_TABLE_DTYPE = np.dtype(
    [("label", np.uint64), ("count", np.uint64), ("fraction", np.float64)]
)


@dataclass(frozen=True)
class BestOverlap:
    """Best overlap result for one queried label."""

    label: int
    count: int
    fraction: float
    found: bool


class SegmentationOverlap:
    """Sparse overlap counts between two segmentation label arrays.

    Use :func:`segmentation_overlap` to construct this object. Labels from the
    first input are called ``label_a`` and labels from the second input are
    called ``label_b`` in all tables.
    """

    def __init__(self, core_overlap):
        self._core_overlap = core_overlap

    @property
    def total_count(self) -> int:
        """Total number of pixels or voxels compared."""
        return int(self._core_overlap.total_count)

    @property
    def labels_a(self) -> np.ndarray:
        """Sorted labels present in the first segmentation."""
        return np.asarray(self._core_overlap.labels_a(), dtype=np.uint64)

    @property
    def labels_b(self) -> np.ndarray:
        """Sorted labels present in the second segmentation."""
        return np.asarray(self._core_overlap.labels_b(), dtype=np.uint64)

    def count_a(self, label: int) -> int:
        """Return the size of a label in the first segmentation."""
        return int(self._core_overlap.count_a(_normalize_label(label)))

    def count_b(self, label: int) -> int:
        """Return the size of a label in the second segmentation."""
        return int(self._core_overlap.count_b(_normalize_label(label)))

    def overlap_count(self, label_a: int, label_b: int) -> int:
        """Return the number of pixels where ``label_a`` and ``label_b`` overlap."""
        return int(
            self._core_overlap.overlap_count(
                _normalize_label(label_a, "label_a"),
                _normalize_label(label_b, "label_b"),
            )
        )

    def counts_a_table(self) -> np.ndarray:
        """Return a structured array with fields ``label`` and ``count``."""
        return _label_count_table(self._core_overlap.counts_a())

    def counts_b_table(self) -> np.ndarray:
        """Return a structured array with fields ``label`` and ``count``."""
        return _label_count_table(self._core_overlap.counts_b())

    def overlap_table(
        self,
        *,
        normalize_by: Literal["a", "b", "total"] | None = None,
    ) -> np.ndarray:
        """Return all non-zero overlaps as a structured array.

        Without normalization, the fields are ``label_a``, ``label_b`` and
        ``count``. With ``normalize_by="a"``, ``"b"`` or ``"total"``, a
        ``fraction`` field is added.
        """
        rows = self._core_overlap.overlap_pairs()
        if normalize_by is None:
            table = np.empty(len(rows), dtype=_OVERLAP_TABLE_DTYPE)
            for index, row in enumerate(rows):
                table[index] = (row.label_a, row.label_b, row.count)
            return table

        _validate_normalize_by(normalize_by)
        table = np.empty(len(rows), dtype=_OVERLAP_FRACTION_TABLE_DTYPE)
        for index, row in enumerate(rows):
            table[index] = (
                row.label_a,
                row.label_b,
                row.count,
                self._fraction(row.label_a, row.label_b, row.count, normalize_by),
            )
        return table

    def overlaps_for_label_a(
        self,
        label: int,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Return labels from B overlapping one label from A."""
        normalized_label = _normalize_label(label)
        rows = self._core_overlap.overlaps_for_label_a(normalized_label)
        denominator = self.count_a(label)
        return _label_overlap_table(rows, normalize=normalize, denominator=denominator)

    def overlaps_for_label_b(
        self,
        label: int,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Return labels from A overlapping one label from B."""
        normalized_label = _normalize_label(label)
        rows = self._core_overlap.overlaps_for_label_b(normalized_label)
        denominator = self.count_b(label)
        return _label_overlap_table(rows, normalize=normalize, denominator=denominator)

    def best_overlap_for_label_a(
        self,
        label: int,
        *,
        ignore_zero: bool = False,
    ) -> BestOverlap:
        """Return the best matching label in B for one label in A."""
        normalized_label = _normalize_label(label)
        best_label, count = self._core_overlap.best_overlap_for_label_a(
            normalized_label, bool(ignore_zero)
        )
        denominator = self.count_a(label)
        return _best_overlap(best_label, count, denominator)

    def best_overlap_for_label_b(
        self,
        label: int,
        *,
        ignore_zero: bool = False,
    ) -> BestOverlap:
        """Return the best matching label in A for one label in B."""
        normalized_label = _normalize_label(label)
        best_label, count = self._core_overlap.best_overlap_for_label_b(
            normalized_label, bool(ignore_zero)
        )
        denominator = self.count_b(label)
        return _best_overlap(best_label, count, denominator)

    def is_label_a_overlapping_with_zero(self, label: int) -> bool:
        """Return whether a label from A overlaps label ``0`` in B."""
        return bool(
            self._core_overlap.is_label_a_overlapping_with_zero(_normalize_label(label))
        )

    def is_label_b_overlapping_with_zero(self, label: int) -> bool:
        """Return whether a label from B overlaps label ``0`` in A."""
        return bool(
            self._core_overlap.is_label_b_overlapping_with_zero(_normalize_label(label))
        )

    def different_overlap(self, label_a_u: int, label_a_v: int) -> float:
        """Return the probability that two A labels overlap different B labels."""
        return float(
            self._core_overlap.different_overlap(
                _normalize_label(label_a_u, "label_a_u"),
                _normalize_label(label_a_v, "label_a_v"),
            )
        )

    def _fraction(
        self,
        label_a: int,
        label_b: int,
        count: int,
        normalize_by: str,
    ) -> float:
        if normalize_by == "a":
            denominator = self.count_a(label_a)
        elif normalize_by == "b":
            denominator = self.count_b(label_b)
        else:
            denominator = self.total_count
        return 0.0 if denominator == 0 else float(count) / float(denominator)


def segmentation_overlap(labels_a: np.ndarray, labels_b: np.ndarray) -> SegmentationOverlap:
    """Compute sparse overlap counts between two segmentations.

    Parameters
    ----------
    labels_a, labels_b:
        Integer NumPy arrays with identical shape. Supported dtypes are
        unsigned integer dtypes and signed integer dtypes with non-negative
        values. Inputs are converted to contiguous ``uint64`` arrays before
        entering C++.

    Returns
    -------
    SegmentationOverlap
        Object exposing named tables and query methods for overlap counts.
    """
    array_a = _normalize_labels(labels_a, "labels_a")
    array_b = _normalize_labels(labels_b, "labels_b")
    if array_a.shape != array_b.shape:
        raise ValueError(
            "labels_a and labels_b must have the same shape, got "
            f"labels_a shape={array_a.shape}, labels_b shape={array_b.shape}"
        )
    if array_a.ndim == 0:
        raise ValueError("labels_a and labels_b must have at least one dimension")

    return SegmentationOverlap(_core._segmentation_overlap_uint64(array_a, array_b))


def _normalize_labels(labels: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(labels)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype, got dtype={array.dtype}")
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
        raise ValueError(f"{name} must not contain negative labels")
    return np.ascontiguousarray(array, dtype=np.uint64)


def _normalize_label(label: int, name: str = "label") -> int:
    label = int(label)
    if label < 0:
        raise ValueError(f"{name} must be non-negative")
    return label


def _label_count_table(rows) -> np.ndarray:
    table = np.empty(len(rows), dtype=_COUNT_TABLE_DTYPE)
    for index, (label, count) in enumerate(rows):
        table[index] = (label, count)
    return table


def _label_overlap_table(rows, *, normalize: bool, denominator: int) -> np.ndarray:
    if not normalize:
        table = np.empty(len(rows), dtype=_LABEL_OVERLAP_TABLE_DTYPE)
        for index, (label, count) in enumerate(rows):
            table[index] = (label, count)
        return table

    table = np.empty(len(rows), dtype=_LABEL_OVERLAP_FRACTION_TABLE_DTYPE)
    for index, (label, count) in enumerate(rows):
        fraction = 0.0 if denominator == 0 else float(count) / float(denominator)
        table[index] = (label, count, fraction)
    return table


def _best_overlap(label: int, count: int, denominator: int) -> BestOverlap:
    fraction = 0.0 if denominator == 0 else float(count) / float(denominator)
    return BestOverlap(
        label=int(label),
        count=int(count),
        fraction=fraction,
        found=bool(count),
    )


def _validate_normalize_by(normalize_by: str) -> None:
    if normalize_by not in ("a", "b", "total"):
        raise ValueError("normalize_by must be one of None, 'a', 'b', or 'total'")


__all__ = [
    "BestOverlap",
    "SegmentationOverlap",
    "segmentation_overlap",
]
