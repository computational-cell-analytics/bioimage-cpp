from __future__ import annotations

import numpy as np
import pytest

from development.graph.agglomeration import _compatibility


def test_variation_of_information_identical_partitions():
    labels = np.array([10, 10, 4, 4, 9])
    assert _compatibility.variation_of_information(labels, labels) == pytest.approx(0.0)


def test_variation_of_information_one_cluster_equals_other_entropy():
    one_cluster = np.zeros(4, dtype=np.int64)
    balanced = np.array([0, 0, 1, 1])
    assert _compatibility.variation_of_information(
        one_cluster, balanced
    ) == pytest.approx(np.log(2.0))


def test_variation_of_information_independent_binary_partitions():
    first = np.array([0, 0, 1, 1])
    second = np.array([0, 1, 0, 1])
    assert _compatibility.variation_of_information(first, second) == pytest.approx(
        2.0 * np.log(2.0)
    )


def test_variation_of_information_is_invariant_to_label_values():
    first = np.array([0, 0, 1, 2, 2, 2])
    second = np.array([4, 4, 8, 8, 9, 9])
    renumbered_first = np.array([30, 30, 10, 90, 90, 90])
    renumbered_second = np.array([2, 2, 7, 7, 5, 5])
    expected = _compatibility.variation_of_information(first, second)
    assert _compatibility.variation_of_information(
        renumbered_first, renumbered_second
    ) == pytest.approx(expected)


def test_variation_of_information_empty_and_mismatched_inputs():
    assert _compatibility.variation_of_information([], []) == 0.0
    with pytest.raises(ValueError, match="same number of elements"):
        _compatibility.variation_of_information([0], [0, 1])


def test_variation_of_information_does_not_allocate_dense_contingency(monkeypatch):
    original_zeros = np.zeros

    def reject_matrix(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2:
            raise AssertionError("dense contingency allocation")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(_compatibility.np, "zeros", reject_matrix)
    first = np.arange(5000, dtype=np.int64)
    second = first[::-1]
    assert _compatibility.variation_of_information(first, second) == pytest.approx(0.0)


def test_variation_of_information_matches_sklearn_when_available():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(42)
    for _ in range(5):
        first = rng.integers(0, 6, size=100)
        second = rng.integers(0, 8, size=100)
        first_probabilities = np.unique(first, return_counts=True)[1] / first.size
        second_probabilities = np.unique(second, return_counts=True)[1] / second.size
        ha = -np.sum(first_probabilities * np.log(first_probabilities))
        hb = -np.sum(second_probabilities * np.log(second_probabilities))
        mutual_information = sklearn_metrics.mutual_info_score(first, second)
        expected = ha + hb - 2.0 * mutual_information
        assert _compatibility.variation_of_information(
            first, second
        ) == pytest.approx(expected)
