import numpy as np
import pytest

import bioimage_cpp as bic


def test_edge_map_features_simple():
    labels = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.uint64)
    edge_map = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
    rag = bic.graph.region_adjacency_graph(labels)

    features = bic.graph.features.edge_map_features(rag, labels, edge_map)

    assert tuple(bic.graph.features.SIMPLE_EDGE_FEATURE_NAMES) == ("mean", "size")
    np.testing.assert_allclose(
        features,
        np.array(
            [
                [6.0, 1.0],
                [3.25, 2.0],
                [12.0, 1.0],
            ]
        ),
    )


def test_edge_map_features_complex():
    labels = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.uint32)
    edge_map = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
    rag = bic.graph.region_adjacency_graph(labels)

    features = bic.graph.features.edge_map_features_complex(rag, labels, edge_map)

    assert tuple(bic.graph.features.COMPLEX_EDGE_FEATURE_NAMES) == (
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p5",
        "p10",
        "p25",
        "p75",
        "p90",
        "p95",
        "size",
    )
    np.testing.assert_allclose(
        features,
        np.array(
            [
                [
                    6.0, 6.0, 0.0, 6.0, 6.0, 6.0,
                    6.0, 6.0, 6.0, 6.0, 6.0, 1.0,
                ],
                [
                    3.25, 3.25, 0.25, 3.0, 3.5, 3.025,
                    3.05, 3.125, 3.375, 3.45, 3.475, 2.0,
                ],
                [
                    12.0, 12.0, 0.0, 12.0, 12.0, 12.0,
                    12.0, 12.0, 12.0, 12.0, 12.0, 1.0,
                ],
            ]
        ),
    )


def test_edge_map_features_complex_std_stable_for_large_baseline():
    # Values 1e8 and 1e8 + 1 have std 0.5; the naive sum-of-squares formula
    # returns 0.0 here due to catastrophic cancellation.
    labels = np.array([[0, 1], [0, 1]], dtype=np.uint32)
    edge_map = np.array([[1e8, 1e8], [1e8 + 1, 1e8 + 1]], dtype=np.float64)
    rag = bic.graph.region_adjacency_graph(labels)
    features = bic.graph.features.edge_map_features_complex(rag, labels, edge_map)
    np.testing.assert_allclose(features[0, 2], 0.5)


def test_affinity_features_simple():
    labels = np.array([[1, 1, 2], [1, 3, 2]], dtype=np.int64)
    rag = bic.graph.region_adjacency_graph(labels)
    affinities = np.zeros((2, 2, 3), dtype=np.float32)
    affinities[0] = np.array([[0.0, 6.0, 0.0], [7.0, 8.0, 0.0]])
    affinities[1] = np.array([[0.0, 9.0, 0.0], [0.0, 0.0, 0.0]])

    features = bic.graph.features.affinity_features(
        rag, labels, affinities, offsets=[[0, 1], [1, 0]]
    )

    np.testing.assert_allclose(
        features,
        np.array(
            [
                [6.0, 1.0],
                [8.0, 2.0],
                [8.0, 1.0],
            ]
        ),
    )


def _reference_affinity_features(labels, rag, affinities, offsets):
    """Trivial Python reference: per-channel sweep with explicit bounds checks."""
    uv_ids = np.asarray(rag.uv_ids(), dtype=np.uint64)
    edge_lookup = {(int(u), int(v)): i for i, (u, v) in enumerate(uv_ids)}
    out = np.zeros((len(edge_lookup), 2), dtype=np.float64)
    shape = labels.shape
    for channel, offset in enumerate(offsets):
        for index in np.ndindex(*shape):
            target = tuple(int(c) + int(d) for c, d in zip(index, offset))
            if any(t < 0 or t >= s for t, s in zip(target, shape)):
                continue
            u, v = int(labels[index]), int(labels[target])
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            edge = edge_lookup.get(key)
            if edge is None:
                # Pair (u,v) doesn't exist as a RAG edge (e.g. when the offset
                # reaches further than direct neighbors). The kernel skips
                # these via find_edge -> -1; the reference must match.
                continue
            out[edge, 0] += float(affinities[(channel,) + index])
            out[edge, 1] += 1.0
    for edge in range(len(edge_lookup)):
        if out[edge, 1] > 0:
            out[edge, 0] /= out[edge, 1]
    return out


@pytest.mark.parametrize(
    "offsets",
    [
        # axis-aligned positive offsets (the only case the previous tests covered)
        [[0, 1], [1, 0]],
        # negative offsets — must still hit the valid box correctly
        [[0, -1], [-1, 0]],
        # large magnitude that crops most of the image
        [[0, 3], [3, 0]],
        # diagonal / multi-axis nonzero — exercises the new valid-box math
        [[1, 1], [1, -1], [-1, 1]],
        # offset larger than an axis — must produce zero contributions
        [[10, 0], [0, 10]],
    ],
)
def test_affinity_features_2d_offsets_match_reference(offsets):
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 5, size=(5, 7)).astype(np.uint32)
    rag = bic.graph.region_adjacency_graph(labels)
    affinities = rng.standard_normal(
        (len(offsets),) + labels.shape
    ).astype(np.float32)

    expected = _reference_affinity_features(labels, rag, affinities, offsets)
    got = bic.graph.features.affinity_features(rag, labels, affinities, offsets=offsets)

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "offsets",
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        # 3D diagonal
        [[1, 1, 1], [1, -1, 1], [-1, 1, -1]],
        # mixed magnitudes
        [[2, 0, 0], [0, 0, -3]],
    ],
)
def test_affinity_features_3d_offsets_match_reference(offsets):
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 4, size=(4, 5, 6)).astype(np.uint32)
    rag = bic.graph.region_adjacency_graph(labels)
    affinities = rng.standard_normal(
        (len(offsets),) + labels.shape
    ).astype(np.float32)

    expected = _reference_affinity_features(labels, rag, affinities, offsets)
    got = bic.graph.features.affinity_features(rag, labels, affinities, offsets=offsets)

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_affinity_features_complex_parallel_matches_single_thread():
    labels = np.array([[0, 1, 1], [2, 2, 3], [2, 4, 3]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)
    affinities = np.arange(2 * labels.size, dtype=np.float64).reshape((2,) + labels.shape)
    offsets = [[0, 1], [1, 0]]

    single_threaded = bic.graph.features.affinity_features_complex(
        rag, labels, affinities, offsets, number_of_threads=1
    )
    parallel = bic.graph.features.affinity_features_complex(
        rag, labels, affinities, offsets, number_of_threads=3
    )

    np.testing.assert_allclose(parallel, single_threaded)


def test_rag_feature_validation():
    labels = np.array([[1, 2]], dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(labels)

    with pytest.raises(ValueError, match="edge_map shape"):
        bic.graph.features.edge_map_features(rag, labels, np.ones((2, 2)))

    with pytest.raises(ValueError, match="affinities must have shape"):
        bic.graph.features.affinity_features(rag, labels, np.ones((2, 2)), offsets=[[0, 1]])

    with pytest.raises(ValueError, match="offsets length"):
        bic.graph.features.affinity_features(
            rag, labels, np.ones((2, 1, 2)), offsets=[[0, 1]]
        )
