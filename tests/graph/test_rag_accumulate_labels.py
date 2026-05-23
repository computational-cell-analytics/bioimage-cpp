import numpy as np
import pytest

import bioimage_cpp as bic


LABEL_DTYPES = [np.uint32, np.uint64, np.int32, np.int64]


def _reference_majority(rag, labels, other_labels, ignore_value=None):
    out = np.zeros(int(rag.number_of_nodes), dtype=other_labels.dtype)
    flat_labels = np.asarray(labels).ravel()
    flat_other = np.asarray(other_labels).ravel()
    histograms = [dict() for _ in range(int(rag.number_of_nodes))]
    for node, other in zip(flat_labels, flat_other):
        if ignore_value is not None and int(other) == int(ignore_value):
            continue
        histograms[int(node)][int(other)] = histograms[int(node)].get(int(other), 0) + 1
    for node, hist in enumerate(histograms):
        if not hist:
            out[node] = 0
            continue
        # Sort by (-count, label) for argmax with smallest-label tie-break.
        best_label, _ = min(hist.items(), key=lambda kv: (-kv[1], kv[0]))
        out[node] = best_label
    return out


def test_accumulate_labels_2d_hand_computed():
    labels = np.array([[1, 1, 2], [3, 2, 2]], dtype=np.uint64)
    other = np.array([[7, 7, 8], [9, 8, 8]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)

    actual = bic.graph.features.accumulate_labels(rag, labels, other)

    assert actual.dtype == np.uint64
    np.testing.assert_array_equal(actual, np.array([0, 7, 8, 9], dtype=np.uint64))


def test_accumulate_labels_3d_hand_computed():
    labels = np.array(
        [
            [[1, 1], [2, 2]],
            [[1, 3], [3, 2]],
        ],
        dtype=np.uint32,
    )
    other = np.array(
        [
            [[5, 5], [6, 7]],
            [[5, 8], [8, 7]],
        ],
        dtype=np.uint64,
    )
    rag = bic.graph.region_adjacency_graph(labels)

    actual = bic.graph.features.accumulate_labels(rag, labels, other)

    # node 0: no pixels
    # node 1: other = {5, 5, 5} -> 5
    # node 2: other = {6, 7, 7} -> 7
    # node 3: other = {8, 8}    -> 8
    np.testing.assert_array_equal(actual, np.array([0, 5, 7, 8], dtype=np.uint64))


@pytest.mark.parametrize("labels_dtype", LABEL_DTYPES)
@pytest.mark.parametrize("other_dtype", LABEL_DTYPES)
def test_accumulate_labels_dtype_combinations(labels_dtype, other_dtype):
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 6, size=(5, 7)).astype(labels_dtype)
    other = rng.integers(1, 4, size=(5, 7)).astype(other_dtype)
    rag = bic.graph.region_adjacency_graph(labels)

    actual = bic.graph.features.accumulate_labels(rag, labels, other)
    expected = _reference_majority(rag, labels, other)

    assert actual.dtype == np.dtype(other_dtype)
    np.testing.assert_array_equal(actual, expected)


def test_accumulate_labels_ignore_value():
    labels = np.array([[1, 1, 2], [3, 2, 2]], dtype=np.uint64)
    other = np.array([[0, 0, 0], [9, 8, 8]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)

    with_ignore = bic.graph.features.accumulate_labels(
        rag, labels, other, ignore_value=0
    )
    without_ignore = bic.graph.features.accumulate_labels(rag, labels, other)

    # node 0: empty -> 0
    # node 1: pixels {0, 0}; all ignored -> 0
    # node 2: pixels {0, 8, 8}; 0 ignored -> 8
    # node 3: pixels {9} -> 9
    np.testing.assert_array_equal(
        with_ignore, np.array([0, 0, 8, 9], dtype=np.uint64)
    )
    # Without ignoring, node 1 has only zeros -> 0, node 2 has {0,8,8} -> 8.
    np.testing.assert_array_equal(
        without_ignore, np.array([0, 0, 8, 9], dtype=np.uint64)
    )


def test_accumulate_labels_tie_break_smaller_wins():
    labels = np.array([[1, 1]], dtype=np.uint64)
    other = np.array([[5, 3]], dtype=np.uint64)  # tie between 3 and 5
    rag = bic.graph.region_adjacency_graph(labels)

    actual = bic.graph.features.accumulate_labels(rag, labels, other)
    # Both node 1's pixels are distinct labels with count 1 -> smaller wins.
    assert int(actual[1]) == 3


def test_accumulate_labels_ignore_value_negative_with_signed_dtype():
    labels = np.array([[1, 1, 2]], dtype=np.uint64)
    other = np.array([[-1, 7, 7]], dtype=np.int32)
    rag = bic.graph.region_adjacency_graph(labels)

    actual = bic.graph.features.accumulate_labels(
        rag, labels, other, ignore_value=-1
    )
    # node 1: {-1, 7} with -1 ignored -> 7
    # node 2: {7} -> 7
    np.testing.assert_array_equal(actual, np.array([0, 7, 7], dtype=np.int32))


def test_accumulate_labels_parallel_matches_single_thread():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 20, size=(8, 12, 11)).astype(np.uint32)
    other = rng.integers(0, 50, size=(8, 12, 11)).astype(np.uint64)
    rag = bic.graph.region_adjacency_graph(labels, number_of_threads=2)

    single = bic.graph.features.accumulate_labels(
        rag, labels, other, number_of_threads=1
    )
    parallel = bic.graph.features.accumulate_labels(
        rag, labels, other, number_of_threads=4
    )

    np.testing.assert_array_equal(single, parallel)


# Nifty's gridRagAccumulateLabels iterates a std::unordered_map for argmax
# (see nifty/graph/rag/grid_rag_features.hxx), so its tie-breaking depends
# on hash-table iteration order and is not portable. bic resolves ties
# deterministically (smaller label id wins). The cross-checks below pick
# inputs where each node has a strict majority, so both implementations
# must agree.

def _tie_free_other(labels, n_other, rng):
    """Build `other` so that each node has a strict majority (no ties)."""
    other = rng.integers(0, n_other, size=labels.shape).astype(np.uint32)
    # For every node, overwrite half of its pixels with a distinguished
    # label to guarantee a strict winner.
    flat_labels = labels.ravel()
    flat_other = other.ravel()
    for node in np.unique(flat_labels):
        mask = np.flatnonzero(flat_labels == node)
        # Force a strict majority of label `node % n_other` for this node.
        forced = mask[: (mask.size + 1) // 2 + 1]
        flat_other[forced] = int(node) % n_other
    return flat_other.reshape(labels.shape)


def test_accumulate_labels_matches_nifty():
    nrag = pytest.importorskip("nifty.graph.rag")

    rng = np.random.default_rng(123)
    labels = rng.integers(0, 12, size=(6, 7)).astype(np.uint32)
    other = _tie_free_other(labels, 5, rng)
    rag = bic.graph.region_adjacency_graph(labels)
    nifty_rag = nrag.gridRag(labels)

    actual = bic.graph.features.accumulate_labels(rag, labels, other)
    expected = nrag.gridRagAccumulateLabels(nifty_rag, other)
    np.testing.assert_array_equal(actual, expected)


def test_accumulate_labels_matches_nifty_3d():
    nrag = pytest.importorskip("nifty.graph.rag")

    rng = np.random.default_rng(7)
    labels = rng.integers(0, 8, size=(4, 5, 6)).astype(np.uint32)
    other = _tie_free_other(labels, 4, rng)
    rag = bic.graph.region_adjacency_graph(labels)
    nifty_rag = nrag.gridRag(labels)

    actual = bic.graph.features.accumulate_labels(rag, labels, other)
    expected = nrag.gridRagAccumulateLabels(nifty_rag, other)
    np.testing.assert_array_equal(actual, expected)


def test_accumulate_labels_matches_nifty_ignore_background():
    nrag = pytest.importorskip("nifty.graph.rag")

    rng = np.random.default_rng(31)
    labels = rng.integers(0, 8, size=(5, 6)).astype(np.uint32)
    # Most pixels get label 0 (the "background"), a tie-free minority is
    # the distinguishing label per node.
    other = np.zeros(labels.shape, dtype=np.uint32)
    flat_labels = labels.ravel()
    flat_other = other.ravel()
    for node in np.unique(flat_labels):
        mask = np.flatnonzero(flat_labels == node)
        forced = mask[: (mask.size + 1) // 2 + 1]
        flat_other[forced] = int(node) + 1  # non-zero, unique per node
    other = flat_other.reshape(labels.shape)

    rag = bic.graph.region_adjacency_graph(labels)
    nifty_rag = nrag.gridRag(labels)

    actual = bic.graph.features.accumulate_labels(
        rag, labels, other, ignore_value=0
    )
    expected = nrag.gridRagAccumulateLabels(
        nifty_rag, other, ignoreBackground=True
    )
    np.testing.assert_array_equal(actual, expected)


def test_accumulate_labels_validation():
    labels = np.array([[0, 1]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)

    with pytest.raises(ValueError, match="other_labels shape"):
        bic.graph.features.accumulate_labels(
            rag, labels, np.zeros((2, 2), dtype=np.uint64)
        )

    other_labels = np.zeros_like(labels, dtype=np.float32)
    with pytest.raises(TypeError, match="other_labels must have one of dtypes"):
        bic.graph.features.accumulate_labels(rag, labels, other_labels)

    other_labels = np.zeros_like(labels, dtype=np.uint32)
    with pytest.raises(ValueError, match="ignore_value=-1 is not representable"):
        bic.graph.features.accumulate_labels(
            rag, labels, other_labels, ignore_value=-1
        )

    other_labels_other_shape = np.zeros((2, 2), dtype=np.uint64)
    other_labels = np.zeros_like(other_labels_other_shape)
    with pytest.raises(ValueError, match="rag shape"):
        bic.graph.features.accumulate_labels(
            rag, other_labels_other_shape, other_labels
        )
