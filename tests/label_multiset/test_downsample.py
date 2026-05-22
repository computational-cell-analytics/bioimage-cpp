import numpy as np
import pytest

from bioimage_cpp._core import Blocking
from bioimage_cpp.label_multiset import (
    LabelMultiset,
    downsample_multiset,
    multiset_from_labels,
)


def _python_block_histograms(labels: np.ndarray, block_shape):
    """Compute per-block (sorted_ids, counts) histograms in pure Python."""
    shape = labels.shape
    ndim = labels.ndim
    blocks_per_axis = [
        (shape[a] + block_shape[a] - 1) // block_shape[a] for a in range(ndim)
    ]
    blocks = []
    if ndim == 2:
        for by in range(blocks_per_axis[0]):
            for bx in range(blocks_per_axis[1]):
                y0, y1 = by * block_shape[0], min((by + 1) * block_shape[0], shape[0])
                x0, x1 = bx * block_shape[1], min((bx + 1) * block_shape[1], shape[1])
                vals, counts = np.unique(labels[y0:y1, x0:x1], return_counts=True)
                blocks.append((tuple(int(v) for v in vals), tuple(int(c) for c in counts)))
    elif ndim == 3:
        for bz in range(blocks_per_axis[0]):
            for by in range(blocks_per_axis[1]):
                for bx in range(blocks_per_axis[2]):
                    z0 = bz * block_shape[0]; z1 = min((bz + 1) * block_shape[0], shape[0])
                    y0 = by * block_shape[1]; y1 = min((by + 1) * block_shape[1], shape[1])
                    x0 = bx * block_shape[2]; x1 = min((bx + 1) * block_shape[2], shape[2])
                    vals, counts = np.unique(
                        labels[z0:z1, y0:y1, x0:x1], return_counts=True
                    )
                    blocks.append(
                        (tuple(int(v) for v in vals), tuple(int(c) for c in counts))
                    )
    else:
        raise ValueError("only 2D and 3D in test")
    return blocks


def _entries_as_dict(ms: LabelMultiset):
    return [
        (tuple(int(i) for i in ms.entry(s)[0]), tuple(int(c) for c in ms.entry(s)[1]))
        for s in range(ms.n_spatial)
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multiset_from_labels_2d_matches_python(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 7, size=(6, 8), dtype=np.uint64)
    block_shape = (2, 2)

    ms = multiset_from_labels(labels, block_shape)
    expected = _python_block_histograms(labels, block_shape)
    actual = _entries_as_dict(ms)
    assert actual == expected


def test_multiset_from_labels_3d_matches_python():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 5, size=(4, 6, 8), dtype=np.uint32)
    block_shape = (2, 2, 2)

    ms = multiset_from_labels(labels, block_shape)
    expected = _python_block_histograms(labels, block_shape)
    actual = _entries_as_dict(ms)
    assert actual == expected


def test_downsample_equivalent_to_block_aggregation():
    # Building level-0 with block_shape (1,1) and then downsampling by (2,2)
    # must give the same result as building directly with block_shape (2,2).
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 4, size=(8, 6), dtype=np.uint64)
    ms0 = multiset_from_labels(labels, (1, 1))
    blocking = Blocking([0, 0], list(labels.shape), [2, 2])
    ms1 = downsample_multiset(ms0, blocking)
    ms_direct = multiset_from_labels(labels, (2, 2))

    assert _entries_as_dict(ms1) == _entries_as_dict(ms_direct)
    assert ms1.argmax.tolist() == ms_direct.argmax.tolist()


def test_downsample_3d_two_levels():
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 6, size=(8, 8, 8), dtype=np.uint64)
    ms0 = multiset_from_labels(labels, (1, 1, 1))

    b1 = Blocking([0, 0, 0], [8, 8, 8], [2, 2, 2])
    ms1 = downsample_multiset(ms0, b1)
    expected1 = _python_block_histograms(labels, (2, 2, 2))
    assert _entries_as_dict(ms1) == expected1

    b2 = Blocking([0, 0, 0], [4, 4, 4], [2, 2, 2])
    ms2 = downsample_multiset(ms1, b2)
    expected2 = _python_block_histograms(labels, (4, 4, 4))
    assert _entries_as_dict(ms2) == expected2


def test_downsample_dedup_reduces_entries():
    # All-zero volume — every block is identical, so n_entries must collapse to 1.
    labels = np.zeros((8, 8), dtype=np.uint64)
    ms = multiset_from_labels(labels, (2, 2))
    assert ms.n_spatial == 16
    assert ms.n_entries == 1
    assert ms.argmax.tolist() == [0] * 16


def test_downsample_restrict_set_keeps_top_k():
    # 4x4 block with labels [0]*15 + [1]; restrict_set=1 should keep only label 0.
    labels = np.zeros((4, 4), dtype=np.uint64)
    labels[3, 3] = 1
    ms_full = multiset_from_labels(labels, (1, 1))
    b = Blocking([0, 0], [4, 4], [4, 4])
    ms_top1 = downsample_multiset(ms_full, b, restrict_set=1)
    assert ms_top1.n_spatial == 1
    ids, counts = ms_top1.entry(0)
    assert ids.tolist() == [0]
    assert counts.tolist() == [15]
    assert ms_top1.argmax.tolist() == [0]
