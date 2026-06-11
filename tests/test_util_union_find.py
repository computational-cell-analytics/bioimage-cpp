import numpy as np
import pytest

import bioimage_cpp as bic


def test_union_find_is_exposed_from_utils_namespace():
    assert not hasattr(bic, "util")
    assert bic.utils.UnionFind(1).size == 1


def test_initial_state_is_all_singletons():
    uf = bic.utils.UnionFind(5)
    assert uf.size == 5
    for i in range(5):
        assert uf.find(i) == i


def test_size_zero():
    uf = bic.utils.UnionFind(0)
    assert uf.size == 0
    labels = uf.element_labeling()
    assert labels.shape == (0,)
    assert labels.dtype == np.uint64


def test_scalar_merge_joins_two_elements():
    uf = bic.utils.UnionFind(4)
    root = uf.merge(0, 1)
    assert uf.find(0) == root
    assert uf.find(1) == root
    assert uf.find(2) == 2
    assert uf.find(3) == 3


def test_scalar_merge_is_transitive():
    uf = bic.utils.UnionFind(5)
    uf.merge(0, 1)
    uf.merge(2, 3)
    uf.merge(1, 2)
    assert uf.find(0) == uf.find(3)
    assert uf.find(0) != uf.find(4)


def test_merge_to_forces_stable_root():
    uf = bic.utils.UnionFind(4)
    root = uf.merge_to(0, 1)
    assert root == 0
    assert uf.find(0) == 0
    assert uf.find(1) == 0


def test_bulk_merge_from_edges():
    uf = bic.utils.UnionFind(6)
    edges = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.uint64)
    uf.merge(edges)
    assert uf.find(0) == uf.find(1) == uf.find(2)
    assert uf.find(3) == uf.find(4)
    assert uf.find(0) != uf.find(3)
    assert uf.find(5) == 5


def test_bulk_merge_empty_edges():
    uf = bic.utils.UnionFind(3)
    edges = np.empty((0, 2), dtype=np.uint64)
    uf.merge(edges)
    assert uf.find(0) == 0
    assert uf.find(1) == 1
    assert uf.find(2) == 2


def test_bulk_find_returns_array_of_roots():
    uf = bic.utils.UnionFind(5)
    uf.merge(0, 1)
    uf.merge(2, 3)

    nodes = np.array([0, 1, 2, 3, 4], dtype=np.uint64)
    roots = uf.find(nodes)

    assert roots.dtype == np.uint64
    assert roots.shape == (5,)
    assert roots[0] == roots[1]
    assert roots[2] == roots[3]
    assert roots[4] == 4
    for i in range(5):
        assert roots[i] == uf.find(int(i))


def test_bulk_find_empty_array():
    uf = bic.utils.UnionFind(3)
    roots = uf.find(np.empty((0,), dtype=np.uint64))
    assert roots.shape == (0,)
    assert roots.dtype == np.uint64


def test_element_labeling_matches_scalar_find():
    uf = bic.utils.UnionFind(6)
    edges = np.array([[0, 1], [2, 3], [3, 4]], dtype=np.uint64)
    uf.merge(edges)

    labels = uf.element_labeling()

    assert labels.dtype == np.uint64
    assert labels.shape == (6,)
    for i in range(6):
        assert labels[i] == uf.find(i)


def test_element_labeling_equivalence_classes():
    uf = bic.utils.UnionFind(5)
    uf.merge(np.array([[0, 2], [1, 3]], dtype=np.uint64))

    labels = uf.element_labeling()

    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]
    assert labels[4] != labels[0]
    assert labels[4] != labels[1]


def test_reset_reinitialises_to_singletons():
    uf = bic.utils.UnionFind(3)
    uf.merge(0, 1)
    uf.merge(1, 2)

    uf.reset(4)

    assert uf.size == 4
    for i in range(4):
        assert uf.find(i) == i


def test_bulk_merge_rejects_wrong_shape():
    uf = bic.utils.UnionFind(3)
    with pytest.raises(Exception, match=r"\(N, 2\)"):
        uf.merge(np.zeros((4,), dtype=np.uint64))


def test_bulk_merge_rejects_three_columns():
    uf = bic.utils.UnionFind(3)
    with pytest.raises(Exception, match=r"\(N, 2\)"):
        uf.merge(np.zeros((2, 3), dtype=np.uint64))


def test_scalar_find_rejects_out_of_range_node():
    uf = bic.utils.UnionFind(3)
    with pytest.raises(ValueError, match="out of range"):
        uf.find(3)


def test_scalar_merge_rejects_out_of_range_node():
    uf = bic.utils.UnionFind(3)
    with pytest.raises(ValueError, match="out of range"):
        uf.merge(0, 5)


def test_merge_to_rejects_out_of_range_node():
    uf = bic.utils.UnionFind(3)
    with pytest.raises(ValueError, match="out of range"):
        uf.merge_to(7, 0)


def test_bulk_find_rejects_out_of_range_node():
    uf = bic.utils.UnionFind(3)
    with pytest.raises(ValueError, match="out of range"):
        uf.find(np.array([0, 1, 9], dtype=np.uint64))


def test_bulk_merge_rejects_out_of_range_node():
    uf = bic.utils.UnionFind(3)
    with pytest.raises(ValueError, match="out of range"):
        uf.merge(np.array([[0, 1], [2, 8]], dtype=np.uint64))
