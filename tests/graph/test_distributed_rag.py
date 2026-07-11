"""Tests for the distributed region-adjacency-graph and edge-feature primitives.

The core property is that computing per block (with a halo) and merging
reproduces the whole-volume result: exactly for the region graph and for the
``size/min/max`` feature columns, and to floating-point tolerance for
``mean/std``. Blocking uses :class:`bioimage_cpp.utils.Blocking` to tile the
array; a block owns the pixels of its inner (non-halo) box.
"""

import numpy as np
import pytest

import bioimage_cpp as bic

dist = bic.graph.distributed

LABEL_DTYPES = [np.uint32, np.uint64, np.int32, np.int64]

# In-core complex feature columns are
# (mean, median, std, min, max, p5, p10, p25, p75, p90, p95, size); the
# distributed complex output is the moment subset (mean, std, min, max, size).
_COMPLEX_MOMENT_COLUMNS = [0, 2, 3, 4, 11]


def _tile_owned(labels, block_shape, halo):
    """Yield ``(sub, slices, own_begin, own_shape)`` for every haloed block."""
    ndim = labels.ndim
    blocking = bic.utils.Blocking([0] * ndim, list(labels.shape), list(block_shape))
    for block_id in range(blocking.number_of_blocks):
        bwh = blocking.get_block_with_halo(block_id, list(halo))
        outer_begin, outer_end = bwh.outer_block.begin, bwh.outer_block.end
        slices = tuple(slice(int(outer_begin[a]), int(outer_end[a])) for a in range(ndim))
        inner_begin, inner_end = bwh.inner_block_local.begin, bwh.inner_block_local.end
        own_begin = [int(inner_begin[a]) for a in range(ndim)]
        own_shape = [int(inner_end[a]) - int(inner_begin[a]) for a in range(ndim)]
        yield np.ascontiguousarray(labels[slices]), slices, own_begin, own_shape


def _blocked_edges(labels, block_shape, halo, *, number_of_threads=0):
    edges = [
        dist.block_region_adjacency_edges(
            sub, own_begin, own_shape, number_of_threads=number_of_threads
        )
        for (sub, _, own_begin, own_shape) in _tile_owned(labels, block_shape, halo)
    ]
    return dist.merge_edges(edges)


def _global_graph(rag, merged_edges):
    return bic.graph.UndirectedGraph.from_unique_edges(
        int(rag.number_of_nodes), merged_edges
    )


def _reduce_stats(global_graph, per_block):
    acc = dist.empty_edge_stats(global_graph.number_of_edges)
    for block_edges, block_stats in per_block:
        acc = dist.merge_block_edge_stats(global_graph, acc, block_edges, block_stats)
    return acc


# --------------------------------------------------------------------------
# Region graph: block extraction + merge
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", LABEL_DTYPES)
def test_single_block_equals_whole_2d(dtype):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 6, size=(9, 11)).astype(dtype)
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()
    single = dist.block_region_adjacency_edges(labels, [0, 0], list(labels.shape))
    np.testing.assert_array_equal(dist.merge_edges(single), whole)


def test_single_block_equals_whole_3d():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 5, size=(5, 6, 7)).astype(np.uint64)
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()
    single = dist.block_region_adjacency_edges(labels, [0, 0, 0], list(labels.shape))
    np.testing.assert_array_equal(dist.merge_edges(single), whole)


@pytest.mark.parametrize("dtype", LABEL_DTYPES)
def test_blocked_vs_whole_graph_2d(dtype):
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 7, size=(10, 13)).astype(dtype)
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()
    merged = _blocked_edges(labels, [4, 5], [1, 1])
    np.testing.assert_array_equal(merged, whole)


def test_blocked_vs_whole_graph_3d():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 5, size=(7, 8, 9)).astype(np.uint32)
    whole = bic.graph.region_adjacency_graph(labels).uv_ids()
    merged = _blocked_edges(labels, [3, 4, 5], [1, 1, 1])
    np.testing.assert_array_equal(merged, whole)


def test_blocked_graph_thread_determinism():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 8, size=(12, 15)).astype(np.uint32)
    one = _blocked_edges(labels, [5, 6], [1, 1], number_of_threads=1)
    many = _blocked_edges(labels, [5, 6], [1, 1], number_of_threads=4)
    np.testing.assert_array_equal(one, many)


def test_block_edges_are_sorted_unique_and_ordered():
    # A single block's edges match the in-core RAG ordering (sorted, u < v).
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 6, size=(8, 8)).astype(np.uint32)
    edges = dist.block_region_adjacency_edges(labels, [0, 0], list(labels.shape))
    assert np.all(edges[:, 0] < edges[:, 1])
    # sorted lexicographically
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    np.testing.assert_array_equal(edges, edges[order])


# --------------------------------------------------------------------------
# Edge-map features
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint32, np.int64])
def test_edge_map_features_blocked_vs_whole_2d(dtype):
    rng = np.random.default_rng(10)
    labels = rng.integers(0, 6, size=(10, 13)).astype(dtype)
    edge_map = rng.standard_normal((10, 13)).astype(np.float64)

    rag = bic.graph.region_adjacency_graph(labels)
    merged = _blocked_edges(labels, [4, 5], [1, 1])
    np.testing.assert_array_equal(merged, rag.uv_ids())
    graph = _global_graph(rag, merged)

    per_block = [
        dist.block_edge_map_stats(
            sub, np.ascontiguousarray(edge_map[slices]), own_begin, own_shape
        )
        for (sub, slices, own_begin, own_shape) in _tile_owned(labels, [4, 5], [1, 1])
    ]
    acc = _reduce_stats(graph, per_block)

    simple = dist.finalize_edge_features(acc, compute_complex_features=False)
    complex_ = dist.finalize_edge_features(acc, compute_complex_features=True)

    whole_simple = bic.graph.features.edge_map_features(rag, labels, edge_map)
    whole_complex = bic.graph.features.edge_map_features_complex(rag, labels, edge_map)

    np.testing.assert_array_equal(simple[:, 1], whole_simple[:, 1])
    np.testing.assert_allclose(simple[:, 0], whole_simple[:, 0], rtol=1e-6, atol=1e-9)
    # complex moment subset
    np.testing.assert_array_equal(complex_[:, 2], whole_complex[:, 3])  # min
    np.testing.assert_array_equal(complex_[:, 3], whole_complex[:, 4])  # max
    np.testing.assert_array_equal(complex_[:, 4], whole_complex[:, 11])  # size
    np.testing.assert_allclose(complex_[:, 0], whole_complex[:, 0], rtol=1e-6, atol=1e-9)  # mean
    np.testing.assert_allclose(complex_[:, 1], whole_complex[:, 2], rtol=1e-6, atol=1e-9)  # std


def test_edge_map_features_blocked_vs_whole_3d():
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 5, size=(6, 7, 8)).astype(np.uint64)
    edge_map = rng.standard_normal((6, 7, 8)).astype(np.float64)

    rag = bic.graph.region_adjacency_graph(labels)
    merged = _blocked_edges(labels, [3, 4, 4], [1, 1, 1])
    graph = _global_graph(rag, merged)

    per_block = [
        dist.block_edge_map_stats(
            sub, np.ascontiguousarray(edge_map[slices]), own_begin, own_shape
        )
        for (sub, slices, own_begin, own_shape) in _tile_owned(labels, [3, 4, 4], [1, 1, 1])
    ]
    acc = _reduce_stats(graph, per_block)
    complex_ = dist.finalize_edge_features(acc, compute_complex_features=True)
    whole_complex = bic.graph.features.edge_map_features_complex(rag, labels, edge_map)
    for dist_col, whole_col in zip(range(5), _COMPLEX_MOMENT_COLUMNS):
        np.testing.assert_allclose(
            complex_[:, dist_col], whole_complex[:, whole_col], rtol=1e-6, atol=1e-9
        )


# --------------------------------------------------------------------------
# Affinity features
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offsets, halo",
    [
        ([[1, 0], [0, 1]], [1, 1]),
        ([[1, 0], [0, 1], [3, 0], [0, 3]], [3, 3]),
    ],
)
def test_affinity_features_blocked_vs_whole_2d(offsets, halo):
    rng = np.random.default_rng(20)
    labels = rng.integers(0, 6, size=(11, 12)).astype(np.uint32)
    affinities = rng.standard_normal((len(offsets),) + labels.shape).astype(np.float64)

    rag = bic.graph.region_adjacency_graph(labels)
    merged = _blocked_edges(labels, [5, 5], halo)
    graph = _global_graph(rag, merged)

    per_block = []
    for sub, slices, own_begin, own_shape in _tile_owned(labels, [5, 5], halo):
        sub_aff = np.ascontiguousarray(affinities[(slice(None),) + slices])
        per_block.append(dist.block_affinity_stats(sub, sub_aff, offsets, own_begin, own_shape))
    acc = _reduce_stats(graph, per_block)

    complex_ = dist.finalize_edge_features(acc, compute_complex_features=True)
    whole_complex = bic.graph.features.affinity_features_complex(rag, labels, affinities, offsets)
    for dist_col, whole_col in zip(range(5), _COMPLEX_MOMENT_COLUMNS):
        np.testing.assert_allclose(
            complex_[:, dist_col], whole_complex[:, whole_col], rtol=1e-6, atol=1e-9
        )


def test_affinity_features_single_block_equals_whole_3d():
    rng = np.random.default_rng(21)
    labels = rng.integers(0, 5, size=(5, 6, 7)).astype(np.uint64)
    offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 2, 0]]
    affinities = rng.standard_normal((len(offsets),) + labels.shape).astype(np.float64)

    rag = bic.graph.region_adjacency_graph(labels)
    graph = _global_graph(rag, rag.uv_ids())
    be, bs = dist.block_affinity_stats(
        labels, affinities, offsets, [0, 0, 0], list(labels.shape)
    )
    acc = dist.merge_block_edge_stats(graph, dist.empty_edge_stats(graph.number_of_edges), be, bs)
    complex_ = dist.finalize_edge_features(acc, compute_complex_features=True)
    whole_complex = bic.graph.features.affinity_features_complex(rag, labels, affinities, offsets)
    for dist_col, whole_col in zip(range(5), _COMPLEX_MOMENT_COLUMNS):
        np.testing.assert_allclose(
            complex_[:, dist_col], whole_complex[:, whole_col], rtol=1e-6, atol=1e-9
        )


# --------------------------------------------------------------------------
# merge_edges
# --------------------------------------------------------------------------


def test_merge_edges_dedup_canonicalize_and_self_loops():
    edges = np.array(
        [[3, 1], [1, 3], [0, 2], [2, 0], [4, 4], [5, 6]], dtype=np.uint64
    )
    merged = dist.merge_edges(edges)
    expected = np.array([[0, 2], [1, 3], [5, 6]], dtype=np.uint64)
    np.testing.assert_array_equal(merged, expected)


def test_merge_edges_accepts_list_and_builds_graph():
    a = np.array([[0, 1], [1, 2]], dtype=np.uint64)
    b = np.array([[1, 2], [2, 3]], dtype=np.uint64)
    merged = dist.merge_edges([a, b])
    np.testing.assert_array_equal(merged, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint64))
    graph = bic.graph.UndirectedGraph.from_unique_edges(4, merged)
    assert graph.number_of_edges == 3
    assert graph.find_edge(2, 1) == 1  # find_edge canonicalizes


def test_merge_edges_empty():
    assert dist.merge_edges([]).shape == (0, 2)
    empty = np.empty((0, 2), dtype=np.uint64)
    assert dist.merge_edges(empty).shape == (0, 2)


# --------------------------------------------------------------------------
# merge_block_edge_stats / finalize / empty_edge_stats
# --------------------------------------------------------------------------


def test_empty_edge_stats():
    stats = dist.empty_edge_stats(4)
    assert stats.shape == (4, 5)
    assert stats.dtype == np.float64
    assert np.all(stats == 0.0)


def test_merge_block_edge_stats_reduction_and_skipping():
    graph = bic.graph.UndirectedGraph.from_unique_edges(
        3, np.array([[0, 1], [1, 2]], dtype=np.uint64)
    )
    acc = dist.empty_edge_stats(graph.number_of_edges)

    # Block A touches edge (0,1) with values {1, 2}: count=2, mean=1.5, M2=0.5.
    edges_a = np.array([[0, 1]], dtype=np.uint64)
    stats_a = np.array([[2.0, 1.5, 0.5, 1.0, 2.0]], dtype=np.float64)
    acc = dist.merge_block_edge_stats(graph, acc, edges_a, stats_a)

    # Block B touches edge (0,1) again with value {10} plus (0,2) which is
    # absent from the graph -> skipped.
    edges_b = np.array([[0, 1], [0, 2]], dtype=np.uint64)
    stats_b = np.array(
        [[1.0, 10.0, 0.0, 10.0, 10.0], [5.0, 1.0, 5.0, -1.0, 9.0]], dtype=np.float64
    )
    acc = dist.merge_block_edge_stats(graph, acc, edges_b, stats_b)

    # edge (0,1) now describes {1, 2, 10}: count=3, mean=13/3,
    # M2 = sum((x - mean)^2) = 438/9, min(1,10)=1, max(2,10)=10
    values = np.array([1.0, 2.0, 10.0])
    np.testing.assert_allclose(
        acc[0], [3.0, values.mean(), ((values - values.mean()) ** 2).sum(), 1.0, 10.0]
    )
    # edge (1,2): never touched
    np.testing.assert_array_equal(acc[1], [0.0, 0.0, 0.0, 0.0, 0.0])


def test_finalize_edge_features_formulas_and_zero_count():
    # edge 0: count=4, mean=2, M2=8 -> var = 8/4 = 2 -> std sqrt(2)
    # edge 1: empty
    stats = np.array(
        [[4.0, 2.0, 8.0, -1.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64
    )
    simple = dist.finalize_edge_features(stats, compute_complex_features=False)
    np.testing.assert_allclose(simple[0], [2.0, 4.0])
    np.testing.assert_array_equal(simple[1], [0.0, 0.0])

    complex_ = dist.finalize_edge_features(stats, compute_complex_features=True)
    np.testing.assert_allclose(complex_[0], [2.0, np.sqrt(2.0), -1.0, 5.0, 4.0])
    np.testing.assert_array_equal(complex_[1], [0.0, 0.0, 0.0, 0.0, 0.0])


def test_std_is_stable_for_large_baseline():
    # Values 1e8 and 1e8 + 1 have std 0.5; the naive sum-of-squares formula
    # returns 0.0 here due to catastrophic cancellation.
    labels = np.array([[0, 1], [0, 1]], dtype=np.uint32)
    edge_map = np.array([[1e8, 1e8], [1e8 + 1, 1e8 + 1]], dtype=np.float64)

    graph = bic.graph.UndirectedGraph.from_unique_edges(
        2, np.array([[0, 1]], dtype=np.uint64)
    )

    # Whole array as one block, and split into two one-row blocks so the
    # cross-block Chan combine is exercised too.
    single = dist.block_edge_map_stats(labels, edge_map, [0, 0], [2, 2])
    blocked = [
        dist.block_edge_map_stats(labels, edge_map, [0, 0], [1, 2]),
        dist.block_edge_map_stats(labels, edge_map, [1, 0], [1, 2]),
    ]
    for per_block in ([single], blocked):
        acc = dist.empty_edge_stats(graph.number_of_edges)
        for be, bs in per_block:
            acc = dist.merge_block_edge_stats(graph, acc, be, bs)
        features = dist.finalize_edge_features(acc, compute_complex_features=True)
        np.testing.assert_allclose(features[0, 1], 0.5)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_owned_box_out_of_bounds_raises():
    labels = np.zeros((6, 6), dtype=np.uint32)
    with pytest.raises(ValueError, match="owned box must lie within the block"):
        dist.block_region_adjacency_edges(labels, [0, 0], [7, 6])


def test_owned_box_nonpositive_shape_raises():
    labels = np.zeros((6, 6), dtype=np.uint32)
    with pytest.raises(ValueError, match="own_shape values must be positive"):
        dist.block_region_adjacency_edges(labels, [0, 0], [0, 6])


def test_owned_box_wrong_length_raises():
    labels = np.zeros((6, 6), dtype=np.uint32)
    with pytest.raises(ValueError, match="own_begin must be a 1D sequence of length 2"):
        dist.block_region_adjacency_edges(labels, [0, 0, 0], [3, 3])


def test_edge_map_shape_mismatch_raises():
    labels = np.zeros((6, 6), dtype=np.uint32)
    edge_map = np.zeros((6, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="edge_map shape must match labels shape"):
        dist.block_edge_map_stats(labels, edge_map, [0, 0], [6, 6])


def test_affinity_offsets_length_mismatch_raises():
    labels = np.zeros((6, 6), dtype=np.uint32)
    affinities = np.zeros((2, 6, 6), dtype=np.float64)
    with pytest.raises(ValueError, match="offsets length must match affinities channel count"):
        dist.block_affinity_stats(labels, affinities, [[1, 0]], [0, 0], [6, 6])


def test_merge_block_edge_stats_row_mismatch_raises():
    graph = bic.graph.UndirectedGraph.from_unique_edges(
        2, np.array([[0, 1]], dtype=np.uint64)
    )
    acc = dist.empty_edge_stats(1)
    with pytest.raises(ValueError, match="same number of rows"):
        dist.merge_block_edge_stats(
            graph,
            acc,
            np.array([[0, 1]], dtype=np.uint64),
            np.zeros((2, 5), dtype=np.float64),
        )


def test_finalize_wrong_stats_shape_raises():
    with pytest.raises(ValueError, match=r"stats must have shape \(number_of_edges, 5\)"):
        dist.finalize_edge_features(np.zeros((3, 4), dtype=np.float64))


def test_unsupported_label_dtype_raises():
    labels = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(TypeError):
        dist.block_region_adjacency_edges(labels, [0, 0], [5, 5])


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("number_of_threads", [1, 4])
def test_negative_labels_raise(dtype, number_of_threads):
    # A negative label must raise a normal Python exception from every block
    # scanner, also when the check fires inside a worker thread (this used to
    # terminate the process via an unhandled exception in std::thread).
    labels = np.zeros((8, 8), dtype=dtype)
    labels[3, 3] = -1
    edge_map = np.zeros((8, 8), dtype=np.float64)
    affinities = np.zeros((2, 8, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="negative"):
        dist.block_region_adjacency_edges(
            labels, [0, 0], [8, 8], number_of_threads=number_of_threads
        )
    with pytest.raises(ValueError, match="negative"):
        dist.block_edge_map_stats(
            labels, edge_map, [0, 0], [8, 8], number_of_threads=number_of_threads
        )
    with pytest.raises(ValueError, match="negative"):
        dist.block_affinity_stats(
            labels, affinities, [[1, 0], [0, 1]], [0, 0], [8, 8],
            number_of_threads=number_of_threads,
        )
