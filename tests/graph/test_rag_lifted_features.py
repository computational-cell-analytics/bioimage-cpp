import numpy as np
import pytest

import bioimage_cpp as bic


@pytest.fixture
def four_quadrant_labels_2d():
    """4x4 image with four 2x2 quadrants labelled 0/1/2/3."""
    labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.uint32,
    )
    return labels


@pytest.fixture
def four_quadrant_rag_2d(four_quadrant_labels_2d):
    return bic.graph.region_adjacency_graph(four_quadrant_labels_2d)


def test_lifted_edges_discovered_via_diagonal_offset(
    four_quadrant_labels_2d, four_quadrant_rag_2d
):
    # Diagonal offset (2, 2): every hit connects label 0 (top-left) with
    # label 3 (bottom-right). (0, 3) is not in the RAG -> lifted edge.
    lifted = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d, four_quadrant_labels_2d, [(2, 2)]
    )
    assert lifted.dtype == np.uint64
    np.testing.assert_array_equal(lifted, np.array([[0, 3]], dtype=np.uint64))


def test_lifted_edges_anti_diagonal_offset(
    four_quadrant_labels_2d, four_quadrant_rag_2d
):
    # (2, -2): connects label 1 (top-right) with label 2 (bottom-left).
    lifted = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d, four_quadrant_labels_2d, [(2, -2)]
    )
    np.testing.assert_array_equal(lifted, np.array([[1, 2]], dtype=np.uint64))


def test_lifted_edges_skip_one_hop_offsets(
    four_quadrant_labels_2d, four_quadrant_rag_2d
):
    # 1-hop offsets only hit pairs that already exist in the RAG. The
    # function must skip them and return an empty result.
    lifted = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d,
        four_quadrant_labels_2d,
        [(0, 1), (1, 0), (0, -1), (-1, 0)],
    )
    assert lifted.shape == (0, 2)


def test_lifted_edges_mixed_offsets(four_quadrant_labels_2d, four_quadrant_rag_2d):
    # Mixing a long-range diagonal with 1-hop offsets: only the long-range
    # contribution produces lifted edges.
    lifted = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d,
        four_quadrant_labels_2d,
        [(0, 1), (2, 2), (1, 0), (2, -2)],
    )
    np.testing.assert_array_equal(
        lifted, np.array([[0, 3], [1, 2]], dtype=np.uint64)
    )


def test_lifted_edges_pair_already_in_rag_skipped():
    # 3x3 with two segments: long-range offset reaches a pair that is
    # already a local edge -> not reported as lifted.
    labels = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint32,
    )
    rag = bic.graph.region_adjacency_graph(labels)
    lifted = bic.graph.lifted_edges_from_affinities(
        rag, labels, [(2, 0), (0, 2), (2, 2)]
    )
    assert lifted.shape == (0, 2)


def test_lifted_edges_empty_offsets(four_quadrant_labels_2d, four_quadrant_rag_2d):
    lifted = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d, four_quadrant_labels_2d, []
    )
    assert lifted.shape == (0, 2)


def test_lifted_edges_3d():
    # 3D label volume with two slabs across z and two columns across x.
    labels = np.zeros((4, 1, 4), dtype=np.uint32)
    labels[0, 0, :2] = 0
    labels[0, 0, 2:] = 1
    labels[1, 0, :2] = 0
    labels[1, 0, 2:] = 1
    labels[2, 0, :2] = 2
    labels[2, 0, 2:] = 3
    labels[3, 0, :2] = 2
    labels[3, 0, 2:] = 3
    rag = bic.graph.region_adjacency_graph(labels)

    # Diagonal (z, y, x) = (2, 0, 2) connects (0,*,0..1) with (2,*,2..3)
    # i.e. label 0 with label 3 -> lifted (0, 3) not in RAG.
    lifted = bic.graph.lifted_edges_from_affinities(rag, labels, [(2, 0, 2)])
    np.testing.assert_array_equal(lifted, np.array([[0, 3]], dtype=np.uint64))


def test_lifted_edges_dedup_across_offsets(
    four_quadrant_labels_2d, four_quadrant_rag_2d
):
    # (2, 2) and (-2, -2) discover the same (0, 3) pair from opposite ends.
    lifted = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d, four_quadrant_labels_2d, [(2, 2), (-2, -2)]
    )
    np.testing.assert_array_equal(lifted, np.array([[0, 3]], dtype=np.uint64))


def test_lifted_edges_threads(four_quadrant_labels_2d, four_quadrant_rag_2d):
    lifted_single = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d,
        four_quadrant_labels_2d,
        [(2, 2), (2, -2)],
        number_of_threads=1,
    )
    lifted_multi = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d,
        four_quadrant_labels_2d,
        [(2, 2), (2, -2)],
        number_of_threads=4,
    )
    np.testing.assert_array_equal(lifted_single, lifted_multi)


def test_lifted_affinity_features_basic(
    four_quadrant_labels_2d, four_quadrant_rag_2d
):
    # Make a (1, 4, 4) affinity volume with known values. The diagonal
    # (2, 2) offset has 4 valid hits in a 4x4 grid: (0,0), (0,1), (1,0),
    # (1,1) all hitting the (0, 3) lifted edge.
    affinities = np.zeros((1, 4, 4), dtype=np.float64)
    affinities[0, 0, 0] = 0.1
    affinities[0, 0, 1] = 0.2
    affinities[0, 1, 0] = 0.3
    affinities[0, 1, 1] = 0.4

    lifted_uvs = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d, four_quadrant_labels_2d, [(2, 2)]
    )
    features = bic.graph.lifted_affinity_features(
        four_quadrant_labels_2d, affinities, [(2, 2)], lifted_uvs
    )
    assert features.shape == (1, 2)
    assert features[0, 0] == pytest.approx(0.25)  # mean of [0.1, 0.2, 0.3, 0.4]
    assert features[0, 1] == pytest.approx(4.0)


def test_lifted_affinity_features_complex_columns(
    four_quadrant_labels_2d, four_quadrant_rag_2d
):
    affinities = np.full((1, 4, 4), 0.5, dtype=np.float64)
    lifted_uvs = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d, four_quadrant_labels_2d, [(2, 2)]
    )
    features = bic.graph.lifted_affinity_features_complex(
        four_quadrant_labels_2d, affinities, [(2, 2)], lifted_uvs
    )
    assert features.shape == (1, 12)
    # mean
    assert features[0, 0] == pytest.approx(0.5)
    # median, min, max
    assert features[0, 1] == pytest.approx(0.5)
    assert features[0, 3] == pytest.approx(0.5)
    assert features[0, 4] == pytest.approx(0.5)
    # count
    assert features[0, -1] == pytest.approx(4.0)


def test_lifted_affinity_features_skips_one_hop(
    four_quadrant_labels_2d, four_quadrant_rag_2d
):
    # A 1-hop offset would contribute lots of affinity hits to local edges,
    # but the lifted accumulator must not bin any of those onto our lifted
    # edge.
    affinities = np.full((2, 4, 4), 0.7, dtype=np.float64)
    # Channel 0 is 1-hop (should be skipped); channel 1 is the long-range
    # diagonal that defines the lifted edge.
    offsets = [(0, 1), (2, 2)]
    lifted_uvs = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d, four_quadrant_labels_2d, offsets
    )
    features = bic.graph.lifted_affinity_features(
        four_quadrant_labels_2d, affinities, offsets, lifted_uvs
    )
    # Only the 4 diagonal hits count.
    assert features[0, 1] == pytest.approx(4.0)
    assert features[0, 0] == pytest.approx(0.7)


def test_lifted_affinity_features_skips_local_hits():
    # If a long-range offset happens to connect a pair that is also a local
    # edge, the lifted accumulator must not bin the affinity onto a lifted
    # edge (the local edge is not in the lifted set).
    labels = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint32,
    )
    rag = bic.graph.region_adjacency_graph(labels)
    # Offset (2, 2): in a 3x3 grid only (0, 0) -> (2, 2), both label 0.
    # So no diff, no accumulation. We pass a lifted edge that doesn't exist
    # in the data to make sure no accumulation happens.
    lifted_uvs = np.array([[0, 1]], dtype=np.uint64)  # actually a local edge!
    affinities = np.full((1, 3, 3), 0.9, dtype=np.float64)
    features = bic.graph.lifted_affinity_features(
        labels, affinities, [(2, 2)], lifted_uvs
    )
    # The (0,0)->(2,2) hit has matching labels, so no count.
    assert features[0, 1] == pytest.approx(0.0)


def test_lifted_affinity_features_empty_lifted_uvs(four_quadrant_labels_2d):
    affinities = np.full((1, 4, 4), 0.3, dtype=np.float64)
    empty_uvs = np.zeros((0, 2), dtype=np.uint64)
    features = bic.graph.lifted_affinity_features(
        four_quadrant_labels_2d, affinities, [(2, 2)], empty_uvs
    )
    assert features.shape == (0, 2)


def test_lifted_affinity_features_threads(
    four_quadrant_labels_2d, four_quadrant_rag_2d
):
    affinities = np.linspace(0.0, 1.0, num=16).reshape(1, 4, 4).astype(np.float64)
    lifted_uvs = bic.graph.lifted_edges_from_affinities(
        four_quadrant_rag_2d, four_quadrant_labels_2d, [(2, 2)]
    )
    single = bic.graph.lifted_affinity_features(
        four_quadrant_labels_2d, affinities, [(2, 2)], lifted_uvs,
        number_of_threads=1,
    )
    multi = bic.graph.lifted_affinity_features(
        four_quadrant_labels_2d, affinities, [(2, 2)], lifted_uvs,
        number_of_threads=4,
    )
    np.testing.assert_array_almost_equal(single, multi)


def test_lifted_affinity_features_3d():
    labels = np.zeros((4, 1, 4), dtype=np.uint32)
    labels[0, 0, :2] = 0
    labels[0, 0, 2:] = 1
    labels[1, 0, :2] = 0
    labels[1, 0, 2:] = 1
    labels[2, 0, :2] = 2
    labels[2, 0, 2:] = 3
    labels[3, 0, :2] = 2
    labels[3, 0, 2:] = 3
    rag = bic.graph.region_adjacency_graph(labels)

    affinities = np.full((1, 4, 1, 4), 0.42, dtype=np.float64)
    lifted_uvs = bic.graph.lifted_edges_from_affinities(rag, labels, [(2, 0, 2)])
    features = bic.graph.lifted_affinity_features(
        labels, affinities, [(2, 0, 2)], lifted_uvs
    )
    # (0,0,0)->(2,0,2): label 0 -> 3, hit
    # (0,0,1)->(2,0,3): label 0 -> 3, hit
    # (1,0,0)->(3,0,2): label 0 -> 3, hit
    # (1,0,1)->(3,0,3): label 0 -> 3, hit
    assert features[0, 0] == pytest.approx(0.42)
    assert features[0, 1] == pytest.approx(4.0)


def test_lifted_affinity_features_validation():
    labels = np.zeros((4, 4), dtype=np.uint32)
    affinities = np.zeros((1, 4, 4), dtype=np.float64)
    bad_affinities = np.zeros((2, 4, 4), dtype=np.float64)
    lifted_uvs = np.zeros((1, 2), dtype=np.uint64)

    with pytest.raises(ValueError, match="channel count"):
        bic.graph.lifted_affinity_features(
            labels, bad_affinities, [(2, 2)], lifted_uvs
        )
    with pytest.raises(ValueError, match="ndim"):
        bic.graph.lifted_affinity_features(
            labels, affinities, [(2, 2, 0)], lifted_uvs
        )


def test_lifted_edges_validation(four_quadrant_labels_2d, four_quadrant_rag_2d):
    with pytest.raises(ValueError, match="ndim"):
        bic.graph.lifted_edges_from_affinities(
            four_quadrant_rag_2d, four_quadrant_labels_2d, [(2, 2, 0)]
        )
