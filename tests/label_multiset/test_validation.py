from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from bioimage_cpp._core import Blocking
from bioimage_cpp.label_multiset import (
    LabelMultiset,
    MultisetMerger,
    downsample_multiset,
    read_subset,
)


def _valid_multiset() -> LabelMultiset:
    return LabelMultiset(
        argmax=[5, 7],
        offsets=[0, 1],
        entry_offsets=[0, 1],
        entry_sizes=[1, 1],
        ids=[5, 7],
        counts=[2, 3],
    )


@pytest.mark.parametrize(
    ("offsets", "sizes", "match"),
    [
        ([2], [1], "offsets"),
        ([1], [2], "range"),
        ([np.iinfo(np.uint64).max], [2], "offsets"),
    ],
)
def test_read_subset_rejects_invalid_ranges(offsets, sizes, match):
    with pytest.raises(ValueError, match=match):
        read_subset(offsets, sizes, [1], [1])


def test_read_subset_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="offsets and sizes"):
        read_subset([0, 1], [1], [1, 2], [1, 1])
    with pytest.raises(ValueError, match="ids and counts"):
        read_subset([0], [1], [1, 2], [1])


@pytest.mark.parametrize(
    ("argument", "values", "match"),
    [
        ("offsets", [-1], "non-negative"),
        ("offsets", [2**64], "fit dtype uint64"),
        ("counts", [2**32], "fit dtype uint32"),
    ],
)
def test_read_subset_checks_unsigned_conversion(argument, values, match):
    kwargs = {
        "offsets": [0],
        "sizes": [1],
        "ids": [1],
        "counts": [1],
    }
    kwargs[argument] = values
    with pytest.raises(ValueError, match=match):
        read_subset(**kwargs)


def test_read_subset_accepts_last_storage_element():
    ids, counts = read_subset([1], [1], [4, 9], [2, 7])
    np.testing.assert_array_equal(ids, [9])
    np.testing.assert_array_equal(counts, [7])


@pytest.mark.parametrize(
    "replacement",
    [
        {"counts": [1]},
        {"argmax": [5]},
        {"entry_offsets": [0, 2]},
        {"entry_sizes": [3, 1]},
        {"entry_sizes": [1, 0]},
    ],
)
def test_label_multiset_rejects_invalid_structure(replacement):
    values = {
        "argmax": [5, 7],
        "offsets": [0, 1],
        "entry_offsets": [0, 1],
        "entry_sizes": [1, 1],
        "ids": [5, 7],
        "counts": [2, 3],
    }
    values.update(replacement)
    with pytest.raises(ValueError):
        LabelMultiset(**values)


def test_label_multiset_accepts_empty_top_level():
    multiset = LabelMultiset(
        argmax=[],
        offsets=[],
        entry_offsets=[],
        entry_sizes=[],
        ids=[],
        counts=[],
    )
    assert multiset.n_spatial == 0
    assert multiset.n_entries == 0
    assert multiset.ids.dtype == np.uint64
    assert multiset.counts.dtype == np.uint32


def test_downsample_revalidates_mutated_multiset():
    multiset = _valid_multiset()
    multiset.offsets[1] = 99
    blocking = Blocking([0], [2], [2])
    with pytest.raises(ValueError, match="offsets"):
        downsample_multiset(multiset, blocking)


def test_downsample_rejects_blocking_extent_mismatch():
    multiset = _valid_multiset()
    blocking = Blocking([0], [3], [2])
    with pytest.raises(ValueError, match=r"product\(blocking.roi_end\)"):
        downsample_multiset(multiset, blocking)


def test_merger_rejects_update_length_and_entry_index():
    merger = MultisetMerger([0], [1], [5], [2])
    output_offsets = np.array([0], dtype=np.uint64)
    with pytest.raises(ValueError, match="offsets and sizes"):
        merger.update([0, 1], [1], [5, 7], [2, 3], output_offsets)
    with pytest.raises(ValueError, match="batch entries"):
        merger.update([0], [1], [7], [3], np.array([1], dtype=np.uint64))


def test_merger_from_multiset_revalidates_mutation():
    multiset = _valid_multiset()
    multiset.entry_offsets[1] = 7
    with pytest.raises(ValueError, match="entry_offsets"):
        MultisetMerger.from_multiset(multiset)


@pytest.mark.parametrize(
    "code",
    [
        """
import numpy as np
from bioimage_cpp.label_multiset import MultisetMerger
try:
    MultisetMerger(
        np.array([0], dtype=np.uint64),
        np.array([0], dtype=np.uint64),
        np.array([], dtype=np.uint64),
        np.array([], dtype=np.uint32),
    )
except ValueError:
    raise SystemExit(0)
raise SystemExit(3)
""",
        """
import numpy as np
from bioimage_cpp.label_multiset import read_subset
try:
    read_subset(
        np.array([99], dtype=np.uint64),
        np.array([1], dtype=np.uint64),
        np.array([1], dtype=np.uint64),
        np.array([1], dtype=np.uint32),
    )
except ValueError:
    raise SystemExit(0)
raise SystemExit(3)
""",
    ],
)
def test_former_native_crash_cases_raise_in_subprocess(code):
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0
