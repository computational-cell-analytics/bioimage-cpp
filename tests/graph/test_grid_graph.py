import numpy as np
import pytest

import bioimage_cpp as bic


def test_grid_graph_2d_topology_and_coordinates():
    graph = bic.graph.GridGraph2D((2, 3))

    assert graph.number_of_nodes == 6
    assert graph.number_of_edges == 7
    assert graph.ndim == 2
    np.testing.assert_array_equal(graph.shape, np.array([2, 3], dtype=np.uint64))
    np.testing.assert_array_equal(graph.strides, np.array([3, 1], dtype=np.uint64))
    np.testing.assert_array_equal(
        graph.uv_ids(),
        np.array(
            [[0, 3], [1, 4], [2, 5], [0, 1], [1, 2], [3, 4], [4, 5]],
            dtype=np.uint64,
        ),
    )

    assert graph.node_id((1, 2)) == 5
    np.testing.assert_array_equal(graph.coordinates(4), np.array([1, 1], dtype=np.uint64))
    assert graph.find_edge(0, 3) == 0
    assert graph.find_edge(2, 5) == 2
    assert graph.find_edge(0, 1) == 3
    assert graph.find_edge(0, 2) == -1
    assert graph.edge_axis(0) == 0
    assert graph.edge_axis(3) == 1
    np.testing.assert_array_equal(
        graph.edge_coordinates(6), np.array([1, 1], dtype=np.uint64)
    )


def test_grid_graph_3d_topology_and_offset_targets():
    graph = bic.graph.GridGraph3D((2, 2, 2))

    assert graph.number_of_nodes == 8
    assert graph.number_of_edges == 12
    np.testing.assert_array_equal(graph.shape, np.array([2, 2, 2], dtype=np.uint64))
    np.testing.assert_array_equal(graph.strides, np.array([4, 2, 1], dtype=np.uint64))
    np.testing.assert_array_equal(
        graph.uv_ids(),
        np.array(
            [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [0, 2],
                [1, 3],
                [4, 6],
                [5, 7],
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
            ],
            dtype=np.uint64,
        ),
    )

    assert graph.node_id((1, 1, 1)) == 7
    np.testing.assert_array_equal(
        graph.coordinates(6), np.array([1, 1, 0], dtype=np.uint64)
    )
    assert graph.offset_target(0, (1, 0, 0)) == 4
    assert graph.offset_target(3, (0, 0, -1)) == 2
    assert graph.offset_target(3, (0, 0, 1)) == -1


def test_grid_graph_factory_and_algorithm_interop():
    graph = bic.graph.grid_graph((3, 3))

    assert isinstance(graph, bic.graph.GridGraph2D)
    components = bic.graph.connected_components(graph)
    np.testing.assert_array_equal(components, np.zeros(9, dtype=np.uint64))

    nodes, distances = bic.graph.breadth_first_search(
        graph, 0, max_distance=1, include_source=True
    )
    np.testing.assert_array_equal(nodes, np.array([0, 3, 1], dtype=np.uint64))
    np.testing.assert_array_equal(distances, np.array([0, 1, 1], dtype=np.uint64))


def _adjacency_pairs(adjacency: np.ndarray) -> set[tuple[int, int]]:
    return {(int(adjacency[i, 0]), int(adjacency[i, 1])) for i in range(adjacency.shape[0])}


def test_grid_graph_node_adjacency_returns_grid_neighbors():
    # Triggers the lazy CSR rebuild path: GridGraph::build_edges only emits
    # edges, the first node_adjacency call populates the CSR buffer.
    graph = bic.graph.GridGraph2D((3, 3))

    # Corner (0, 0): two neighbors — (1, 0)=node 3 via edge 0, (0, 1)=node 1 via edge 6.
    assert _adjacency_pairs(graph.node_adjacency(0)) == {(3, 0), (1, 6)}
    # Center (1, 1)=node 4: degree 4.
    assert _adjacency_pairs(graph.node_adjacency(4)) == {(1, 1), (7, 4), (3, 8), (5, 9)}
    # Opposite corner (2, 2)=node 8.
    assert _adjacency_pairs(graph.node_adjacency(8)) == {(5, 5), (7, 11)}


def test_grid_graph_3d_node_adjacency_returns_six_neighbors_for_interior_node():
    graph = bic.graph.GridGraph3D((3, 3, 3))
    # Center node (1, 1, 1) — flat id = 1*9 + 1*3 + 1 = 13. Should have 6 neighbors.
    adj = graph.node_adjacency(13)
    assert adj.shape == (6, 2)
    neighbors = {int(adj[i, 0]) for i in range(adj.shape[0])}
    # Expected neighbors: 4 (z-), 22 (z+), 10 (y-), 16 (y+), 12 (x-), 14 (x+).
    assert neighbors == {4, 22, 10, 16, 12, 14}


def test_grid_graph_adjacency_calls_are_idempotent():
    # The lazy rebuild must run exactly once and survive subsequent reads.
    graph = bic.graph.GridGraph2D((3, 3))
    first = _adjacency_pairs(graph.node_adjacency(4))
    second = _adjacency_pairs(graph.node_adjacency(4))
    assert first == second
    # Reading a different node after the rebuild must still work.
    assert _adjacency_pairs(graph.node_adjacency(0)) == {(3, 0), (1, 6)}


def test_grid_graph_breadth_first_search_with_distance_2():
    graph = bic.graph.GridGraph2D((3, 3))
    nodes, distances = bic.graph.breadth_first_search(
        graph, 4, max_distance=2, include_source=True
    )
    # BFS from center reaches every node within distance 2.
    assert set(int(n) for n in nodes) == set(range(9))
    # Center has distance 0; the four cardinal neighbors have distance 1;
    # the four diagonals have distance 2 (no diagonal grid edges).
    distance_of_node = {int(n): int(d) for n, d in zip(nodes, distances)}
    assert distance_of_node[4] == 0
    for neighbor in (1, 3, 5, 7):
        assert distance_of_node[neighbor] == 1
    for corner in (0, 2, 6, 8):
        assert distance_of_node[corner] == 2


def test_grid_graph_connected_components_with_edge_mask():
    # Mask out the axis-0 edges that bridge row 1 ↔ row 2, splitting the
    # grid into two components.
    graph = bic.graph.GridGraph2D((3, 3))
    edge_mask = np.ones(graph.number_of_edges, dtype=bool)
    edge_mask[3] = False  # (3, 6)
    edge_mask[4] = False  # (4, 7)
    edge_mask[5] = False  # (5, 8)
    labels = bic.graph.connected_components(graph, edge_mask=edge_mask)
    assert labels[0] == labels[5]  # rows 0..1 form one component
    assert labels[6] == labels[8]  # row 2 forms another
    assert labels[0] != labels[6]


def test_grid_graph_extract_subgraph_from_nodes():
    # Pick the top-left 2x2 block of a 3x3 grid.
    graph = bic.graph.GridGraph2D((3, 3))
    nodes = np.array([0, 1, 3, 4], dtype=np.uint64)
    inner, outer = graph.extract_subgraph_from_nodes(nodes)
    # Inner edges: (0,1), (0,3), (1,4), (3,4) — ids 6, 0, 1, 8.
    assert set(int(e) for e in inner) == {0, 1, 6, 8}
    # Outer edges: edges crossing the boundary of the 2x2 block.
    # (1,2)=7, (3,6)=3, (4,5)=9, (4,7)=4.
    assert set(int(e) for e in outer) == {3, 4, 7, 9}


def test_grid_graph_freeze_is_idempotent_and_safe_to_call_repeatedly():
    graph = bic.graph.GridGraph2D((3, 3))
    graph.freeze()  # eager rebuild
    graph.freeze()  # already frozen; no-op
    # Adjacency must still be correct after explicit freeze.
    assert _adjacency_pairs(graph.node_adjacency(4)) == {(1, 1), (7, 4), (3, 8), (5, 9)}


def _bounds_ok(coord, offset, shape):
    return all(0 <= coord[d] + offset[d] < shape[d] for d in range(len(shape)))


def test_project_edge_ids_to_pixels_2d_writes_each_edge_at_its_pivot():
    graph = bic.graph.GridGraph2D((2, 3))
    out = graph.project_edge_ids_to_pixels()

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.int64
    # Exactly graph.number_of_edges valid entries; the rest stay -1.
    assert int((out != -1).sum()) == graph.number_of_edges
    # Every edge id appears exactly once, at the slot
    # (edge_axis, *edge_coordinates(edge)).
    for edge in range(graph.number_of_edges):
        axis = graph.edge_axis(edge)
        pivot = tuple(int(c) for c in graph.edge_coordinates(edge))
        assert out[(axis,) + pivot] == edge
    # Slots on the "missing" row/column for each spanning axis stay -1.
    np.testing.assert_array_equal(out[0, 1, :], -1)
    np.testing.assert_array_equal(out[1, :, 2], -1)


def test_project_edge_ids_to_pixels_3d_round_trip_through_edge_coordinates():
    graph = bic.graph.GridGraph3D((2, 2, 2))
    out = graph.project_edge_ids_to_pixels()

    assert out.shape == (3, 2, 2, 2)
    assert int((out != -1).sum()) == graph.number_of_edges
    for edge in range(graph.number_of_edges):
        axis = graph.edge_axis(edge)
        pivot = tuple(int(c) for c in graph.edge_coordinates(edge))
        assert out[(axis,) + pivot] == edge


def test_project_edge_ids_to_pixels_with_offsets_2d_numbers_in_bounds_targets():
    graph = bic.graph.GridGraph2D((2, 3))
    offsets = [(1, 0), (0, 1), (2, 0), (0, -1)]
    out, n_valid = graph.project_edge_ids_to_pixels_with_offsets(offsets)

    assert out.shape == (4, 2, 3)
    assert out.dtype == np.int64

    # Walk (off_idx, *coord) in C-order, predict counter, compare to output.
    expected = np.full((4, 2, 3), -1, dtype=np.int64)
    counter = 0
    for off_idx, offset in enumerate(offsets):
        for r in range(2):
            for c in range(3):
                if _bounds_ok((r, c), offset, (2, 3)):
                    expected[off_idx, r, c] = counter
                    counter += 1
    np.testing.assert_array_equal(out, expected)
    assert n_valid == counter
    # Offset (2, 0) on a height-2 grid is never in bounds.
    np.testing.assert_array_equal(out[2], -1)


def test_project_edge_ids_to_pixels_with_offsets_3d_long_range_offset_partially_out_of_bounds():
    graph = bic.graph.GridGraph3D((3, 2, 2))
    offsets = [(2, 0, 0), (0, 1, 0)]
    out, n_valid = graph.project_edge_ids_to_pixels_with_offsets(offsets)

    assert out.shape == (2, 3, 2, 2)
    # Offset (2, 0, 0): valid only when z + 2 < 3, i.e. z == 0 — that's 4 entries.
    # Offset (0, 1, 0): valid when y + 1 < 2, i.e. y == 0 — that's 3 * 1 * 2 = 6 entries.
    assert n_valid == 4 + 6
    # Sequential numbering: entries in C-order are 0..n_valid-1, rest are -1.
    valid = out[out != -1]
    np.testing.assert_array_equal(valid, np.arange(n_valid, dtype=np.int64))


def test_project_edge_ids_to_pixels_with_offsets_strides_filter_keeps_aligned_coords():
    graph = bic.graph.GridGraph2D((4, 4))
    offsets = [(1, 0), (0, 1)]
    out, n_valid = graph.project_edge_ids_to_pixels_with_offsets(
        offsets, strides=(2, 2)
    )

    # Strides (2, 2) keep only (r, c) with r%2 == c%2 == 0: (0,0), (0,2),
    # (2,0), (2,2). All four are in bounds for both offsets — 8 entries.
    assert n_valid == 8
    valid = out[out != -1]
    np.testing.assert_array_equal(valid, np.arange(8, dtype=np.int64))
    # Non-aligned coords are -1.
    for r in range(4):
        for c in range(4):
            if r % 2 != 0 or c % 2 != 0:
                assert out[0, r, c] == -1
                assert out[1, r, c] == -1


def test_project_edge_ids_to_pixels_with_offsets_mask_filter_keeps_true_positions():
    graph = bic.graph.GridGraph2D((2, 3))
    offsets = [(1, 0), (0, 1)]
    mask = np.zeros((2, 2, 3), dtype=bool)
    mask[0, 0, 1] = True  # (off=0, coord=(0,1)) — target (1,1) in bounds
    mask[1, 1, 0] = True  # (off=1, coord=(1,0)) — target (1,1) in bounds
    mask[1, 0, 2] = True  # (off=1, coord=(0,2)) — target (0,3) OUT OF BOUNDS
    out, n_valid = graph.project_edge_ids_to_pixels_with_offsets(
        offsets, mask=mask
    )

    # The out-of-bounds entry is suppressed even though the mask says true.
    assert n_valid == 2
    expected = np.full((2, 2, 3), -1, dtype=np.int64)
    expected[0, 0, 1] = 0
    expected[1, 1, 0] = 1
    np.testing.assert_array_equal(out, expected)


def test_project_edge_ids_to_pixels_with_offsets_rejects_strides_and_mask_together():
    graph = bic.graph.GridGraph2D((2, 3))
    mask = np.zeros((1, 2, 3), dtype=bool)
    with pytest.raises(ValueError, match="strides and mask"):
        graph.project_edge_ids_to_pixels_with_offsets(
            [(1, 0)], strides=(1, 1), mask=mask
        )


def test_project_edge_ids_to_pixels_with_offsets_validates_shapes():
    graph = bic.graph.GridGraph2D((2, 3))
    with pytest.raises(ValueError, match=r"offsets\[0\] must have length 2"):
        graph.project_edge_ids_to_pixels_with_offsets([(1, 0, 0)])
    with pytest.raises(ValueError, match="strides must have length 2"):
        graph.project_edge_ids_to_pixels_with_offsets([(1, 0)], strides=(1,))
    with pytest.raises(ValueError, match="strides must be positive"):
        graph.project_edge_ids_to_pixels_with_offsets([(1, 0)], strides=(0, 1))
    with pytest.raises(ValueError, match="mask shape must be"):
        bad_mask = np.zeros((1, 2, 4), dtype=bool)
        graph.project_edge_ids_to_pixels_with_offsets([(1, 0)], mask=bad_mask)


def test_grid_graph_rejects_invalid_shapes_and_coordinates():
    with pytest.raises(ValueError, match="length 2"):
        bic.graph.GridGraph2D((2, 3, 4))
    with pytest.raises(ValueError, match="greater than zero"):
        bic.graph.GridGraph3D((2, 0, 4))
    with pytest.raises(ValueError, match="length 2 or 3"):
        bic.graph.grid_graph((5,))

    graph = bic.graph.GridGraph2D((2, 3))
    with pytest.raises(ValueError, match="coordinate must be"):
        graph.node_id((1, 2, 3))
    with pytest.raises(IndexError, match="coordinate\\[0\\]"):
        graph.node_id((2, 0))
    with pytest.raises(ValueError, match="offset must be"):
        graph.offset_target(0, (1, 0, 0))
