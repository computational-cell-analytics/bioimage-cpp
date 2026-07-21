import numpy as np
import pytest

import bioimage_cpp as bic
from bioimage_cpp.skeleton.postprocessing import split_degree3, split_degree4


def _graph(vertices, edges):
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    return vertices, edges, bic.skeleton.skeleton_to_graph(vertices, edges)


def _component_count(vertices, edges):
    labels = bic.graph.connected_components(bic.skeleton.skeleton_to_graph(vertices, edges))
    return len(np.unique(labels))


def test_split_degree3_splits_odd_arm():
    # Node 0 is a T: a straight through-pair along x (nodes 1, 2) plus an odd arm along y (node 3).
    vertices, _, graph = _graph(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0, 1], [0, 2], [0, 3]],
    )
    assert split_degree3(0, graph, vertices, direction_span=1, min_branch_angle=30.0) == [2]


def test_split_degree3_keeps_collinear_arm():
    # The third arm now runs almost parallel to the through pair, below min_branch_angle.
    vertices, _, graph = _graph(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 10.0]],
        [[0, 1], [0, 2], [0, 3]],
    )
    assert split_degree3(0, graph, vertices, direction_span=1, min_branch_angle=30.0) is None


def test_split_degree4_splits_through_pair():
    # A '+' crossing: edges 0,1 are the x through-pair, edges 2,3 the y through-pair.
    vertices, _, graph = _graph(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        [[0, 1], [0, 2], [0, 3], [0, 4]],
    )
    result = split_degree4(0, graph, vertices, direction_span=1, min_through_angle=170.0)
    assert set(result) in ({0, 1}, {2, 3})


def test_split_degree4_keeps_non_collinear_crossing():
    # Four arms at 0, 60, 120, 180 degrees: no pair is collinear enough to split.
    angle = np.deg2rad([0.0, 60.0, 120.0, 180.0])
    arms = np.stack([np.zeros_like(angle), np.sin(angle), np.cos(angle)], axis=1)
    vertices, _, graph = _graph(
        np.concatenate([[[0.0, 0.0, 0.0]], arms]),
        [[0, 1], [0, 2], [0, 3], [0, 4]],
    )
    assert split_degree4(0, graph, vertices, direction_span=1, min_through_angle=170.0) is None


@pytest.mark.parametrize(
    "tick_length, n_vertices, n_edges",
    [(1.5, 5, 4), (0.5, 6, 5)],
)
def test_remove_ticks_prunes_spurs_below_threshold(tick_length, n_vertices, n_edges):
    # Backbone 0-1-2-3-4 along x with a length-1 spur (node 5) off the branch node 2.
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0],
         [0.0, 0.0, 4.0], [0.0, 1.0, 2.0]],
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [2, 5]], dtype=np.int64)

    out_vertices, out_edges, _ = bic.skeleton.remove_ticks(vertices, edges, tick_length=tick_length)

    assert len(out_vertices) == n_vertices
    assert len(out_edges) == n_edges


def test_remove_ticks_keeps_standalone_path():
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    out_vertices, out_edges, _ = bic.skeleton.remove_ticks(vertices, edges, tick_length=100.0)

    assert len(out_vertices) == 3
    assert len(out_edges) == 2


def test_remove_ticks_empty_edges():
    vertices = np.zeros((3, 3), dtype=np.float64)
    edges = np.empty((0, 2), dtype=np.int64)

    out_vertices, out_edges, _ = bic.skeleton.remove_ticks(vertices, edges, tick_length=10.0)

    assert len(out_vertices) == 3
    assert len(out_edges) == 0


def test_join_close_components_links_collinear_endpoints():
    # Two collinear fragments along x with a gap of 2 between node 1 and node 2.
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]])
    edges = np.array([[0, 1], [2, 3]], dtype=np.int64)

    _, out_edges, _ = bic.skeleton.join_close_components(vertices, edges, dist=2.5)

    assert len(out_edges) == 3
    assert (1, 2) in {tuple(sorted(map(int, edge))) for edge in out_edges}


def test_join_close_components_ignores_beyond_dist():
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]])
    edges = np.array([[0, 1], [2, 3]], dtype=np.int64)

    _, out_edges, _ = bic.skeleton.join_close_components(vertices, edges, dist=1.5)

    assert len(out_edges) == 2


def test_join_close_components_ignores_non_collinear():
    # Two fragments offset in y so the gap bends away from each endpoint tangent.
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 3.0, 2.0], [0.0, 4.0, 2.0]])
    edges = np.array([[0, 1], [2, 3]], dtype=np.int64)

    _, out_edges, _ = bic.skeleton.join_close_components(vertices, edges, dist=3.5)

    assert len(out_edges) == 2


def test_join_close_components_skips_same_component():
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    _, out_edges, _ = bic.skeleton.join_close_components(vertices, edges, dist=5.0)

    assert len(out_edges) == 2


def test_clean_graph_splits_crossing():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
    )
    edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int64)
    radii = np.arange(len(vertices), dtype=np.float64)

    assert _component_count(vertices, edges) == 1

    out_vertices, out_edges, out_radii = bic.skeleton.clean_graph(
        vertices, edges, radii=radii, direction_span=1, tick_length=0.0, join_dist=0.0,
    )

    assert _component_count(out_vertices, out_edges) == 2
    assert len(out_radii) == len(out_vertices)


def test_draw_instances_labels_edge():
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    edges = np.array([[0, 1]], dtype=np.int64)
    labels = np.array([2, 2])

    volume = bic.skeleton.draw_instances(vertices, edges, labels, (1, 1, 6), radius=0)

    np.testing.assert_array_equal(volume[0, 0], [3, 3, 3, 3, 0, 0])
    assert set(np.unique(volume)) == {0, 3}


def test_draw_instances_dilates_by_radius():
    vertices = np.array([[2.0, 2.0, 0.0], [2.0, 2.0, 4.0]])
    edges = np.array([[0, 1]], dtype=np.int64)
    labels = np.array([0, 0])

    thin = bic.skeleton.draw_instances(vertices, edges, labels, (5, 5, 5), radius=0)
    thick = bic.skeleton.draw_instances(vertices, edges, labels, (5, 5, 5), radius=1)

    assert thick.astype(bool).sum() > thin.astype(bool).sum()


def test_draw_instances_empty_edges():
    vertices = np.zeros((2, 3), dtype=np.float64)
    edges = np.empty((0, 2), dtype=np.int64)
    labels = np.zeros(2, dtype=np.int64)

    volume = bic.skeleton.draw_instances(vertices, edges, labels, (2, 3, 4))

    assert volume.shape == (2, 3, 4)
    assert volume.sum() == 0
