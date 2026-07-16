import numpy as np
import pytest

import bioimage_cpp as bic


dist = bic.skeleton.distributed


def _contains(vertices, coordinate):
    return np.any(np.all(vertices == np.asarray(coordinate), axis=1))


def _number_of_components(number_of_vertices, edges):
    parents = list(range(number_of_vertices))

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for first, second in edges:
        first_root, second_root = find(int(first)), find(int(second))
        if first_root != second_root:
            parents[second_root] = first_root
    return len({find(node) for node in range(number_of_vertices)})


def test_neighboring_faces_select_the_same_global_target():
    volume = np.zeros((7, 9, 11), dtype=np.uint8)
    volume[1:6, 2:8, 1:10] = 1
    left = np.ascontiguousarray(volume[:, :, :6])
    right = np.ascontiguousarray(volume[:, :, 5:])

    left_target = dist.block_border_targets(
        left, [(2, "high")], origin=(10, 20, 30), spacing=(2.0, 1.5, 1.0)
    )
    right_target = dist.block_border_targets(
        right, [(2, "low")], origin=(10, 20, 35), spacing=(2.0, 1.5, 1.0)
    )
    np.testing.assert_array_equal(left_target, right_target)
    np.testing.assert_array_equal(left_target, np.array([[13, 24, 35]]))


def test_border_targets_binary_and_label_semantics_and_corner_deduplication():
    labels = np.zeros((4, 4, 4), dtype=np.int64)
    labels[1:3, 1, -1] = -7
    labels[1:3, 2, -1] = 2**40
    labels[1, 1, -2:] = -7  # same voxel is visible on two requested faces

    labeled = dist.block_border_targets_labels(
        labels, [(1, "low"), (2, "high"), (2, "high")], background=0
    )
    assert set(labeled) == {-7, 2**40}
    assert all(values.dtype == np.int64 for values in labeled.values())

    binary = dist.block_border_targets(labels, [(2, "high")])
    assert binary.shape[0] == 1  # touching nonzero labels are one binary patch
    assert sum(len(values) for values in labeled.values()) == 2


def test_block_teasar_required_target_and_existing_teasar_equivalence():
    mask = np.zeros((9, 9, 13), dtype=np.uint8)
    mask[2:7, 2:7, 1:12] = 1
    target = np.array([[106, 206, 310]], dtype=np.int64)

    block = dist.block_teasar(
        mask, open_faces=(), origin=(100, 200, 300), required_targets=target,
        spacing=(2.0, 1.0, 0.5)
    )
    assert _contains(block[0], target[0])
    assert block[0].dtype == np.int64

    unconstrained = dist.block_teasar(
        mask, open_faces=(), origin=(0, 0, 0), spacing=(2, 1, 0.5)
    )
    physical = dist.lattice_to_physical(unconstrained, spacing=(2, 1, 0.5))
    ordinary = bic.skeleton.teasar(mask, spacing=(2, 1, 0.5))
    for actual, expected in zip(physical, ordinary):
        np.testing.assert_array_equal(actual, expected)


def test_required_target_order_and_thread_determinism():
    mask = np.zeros((11, 11, 15), dtype=np.uint8)
    mask[2:9, 2:9, 1:14] = 1
    targets = np.array([[2, 2, 2], [8, 8, 13], [2, 8, 7]], dtype=np.int64)
    first = dist.block_teasar(
        mask, open_faces=(), required_targets=targets, number_of_threads=1
    )
    second = dist.block_teasar(
        mask, open_faces=(), required_targets=targets[::-1].copy(),
        number_of_threads=4
    )
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)
    assert all(_contains(first[0], target) for target in targets)


def test_open_face_preserves_interface_radius_and_never_emits_ghosts():
    mask = np.zeros((9, 9, 13), dtype=np.uint8)
    mask[2:7, 2:7, 1:] = 1
    target = dist.block_border_targets(mask, [(2, "high")])
    assert target.shape == (1, 3)

    closed = dist.block_teasar(
        mask, open_faces=(), required_targets=target
    )
    opened = dist.block_teasar(
        mask, open_faces=[(2, "high")], required_targets=target
    )

    def target_radius(graph):
        index = np.flatnonzero(np.all(graph[0] == target[0], axis=1))
        assert index.shape == (1,)
        return float(graph[2][index[0]])

    assert target_radius(closed) == pytest.approx(1.0)
    assert target_radius(opened) > 2.0
    np.testing.assert_array_equal(opened[0][0], target[0])  # interface root
    assert np.all(opened[0] >= 0)
    assert np.all(opened[0] < np.asarray(mask.shape))
    assert np.all(mask[tuple(opened[0].T)] != 0)


def test_lexicographically_greatest_open_target_is_deterministic_root():
    mask = np.zeros((7, 7, 11), dtype=np.uint8)
    mask[2:5, 2:5, :] = 1
    targets = np.array([[3, 3, 0], [3, 3, 10]], dtype=np.int64)
    first = dist.block_teasar(
        mask,
        open_faces=[(2, "low"), (2, "high")],
        required_targets=targets,
        number_of_threads=1,
    )
    second = dist.block_teasar(
        mask,
        open_faces=[(2, "high"), (2, "low"), (2, "high")],
        required_targets=targets[::-1],
        number_of_threads=4,
    )
    np.testing.assert_array_equal(first[0][0], targets[1])
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)


def test_all_foreground_block_uses_finite_closed_boundary_fallback():
    mask = np.ones((5, 6, 7), dtype=np.uint8)
    target = np.array([[4, 5, 6]], dtype=np.int64)
    graph = dist.block_teasar(
        mask,
        open_faces=[
            (0, "low"), (0, "high"),
            (1, "low"), (1, "high"),
            (2, "low"), (2, "high"),
        ],
        required_targets=target,
    )
    assert np.all(np.isfinite(graph[2]))
    assert np.all(graph[2] >= 1.0)
    assert np.all(graph[0] >= 0)
    assert np.all(graph[0] < np.asarray(mask.shape))


def test_labeled_required_targets_preserve_labels_and_reject_mismatch():
    labels = np.zeros((7, 7, 9), dtype=np.uint64)
    labels[1:4, 1:4, 1:8] = 5
    labels[4:6, 3:6, 1:8] = 2**63 + 9
    targets = {
        5: np.array([[2, 2, 7]], dtype=np.int64),
        2**63 + 9: np.array([[5, 4, 7]], dtype=np.int64),
    }
    result = dist.block_teasar_labels(
        labels, open_faces=(), required_targets=targets
    )
    assert list(result) == [5, 2**63 + 9]
    for label, target in targets.items():
        assert _contains(result[label][0], target[0])

    with pytest.raises(ValueError, match="does not match"):
        dist.block_teasar_labels(
            labels, open_faces=(),
            required_targets={5: np.array([[5, 4, 7]], dtype=np.int64)}
        )


def test_labeled_open_face_uses_label_specific_distance_boundary():
    labels = np.zeros((9, 9, 13), dtype=np.int64)
    labels[2:7, 2:7, 1:] = -5
    targets = dist.block_border_targets_labels(
        labels, [(2, "high")], background=0
    )
    result = dist.block_teasar_labels(
        labels,
        open_faces=[(2, "high")],
        required_targets=targets,
        background=0,
    )
    graph = result[-5]
    target = targets[-5][0]
    np.testing.assert_array_equal(graph[0][0], target)
    index = np.flatnonzero(np.all(graph[0] == target, axis=1))
    assert float(graph[2][index[0]]) > 2.0
    assert np.all(labels[tuple(graph[0].T)] == -5)


@pytest.mark.parametrize(
    "targets, message",
    [
        (np.array([[-1, 0, 0]], dtype=np.int64), "below the block origin"),
        (np.array([[99, 0, 0]], dtype=np.int64), "outside the block"),
        (np.array([[0, 0, 0]], dtype=np.int64), "foreground"),
    ],
)
def test_required_target_validation(targets, message):
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = 1
    with pytest.raises(ValueError, match=message):
        dist.block_teasar(mask, open_faces=(), required_targets=targets)


def test_open_faces_is_explicit_and_validated():
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="open_faces"):
        dist.block_teasar(mask)
    with pytest.raises(ValueError, match="side"):
        dist.block_teasar(mask, open_faces=[(0, "outside")])


def test_exact_merge_reduces_vertices_radii_and_edges():
    first = (
        np.array([[0, 0, 0], [0, 0, 1], [9, 9, 9]], dtype=np.int64),
        np.array([[0, 1]], dtype=np.uint64),
        np.array([1.0, 2.0, 0.5], dtype=np.float32),
    )
    second = (
        np.array([[0, 0, 1], [0, 0, 2]], dtype=np.int64),
        np.array([[1, 0], [0, 0]], dtype=np.uint64),
        np.array([3.0, 4.0], dtype=np.float32),
    )
    merged = dist.merge_block_skeletons([first, second])
    np.testing.assert_array_equal(
        merged[0], np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [9, 9, 9]])
    )
    np.testing.assert_array_equal(merged[1], np.array([[0, 1], [1, 2]]))
    np.testing.assert_array_equal(merged[2], np.array([1, 3, 4, 0.5], np.float32))


def test_merge_is_associative_commutative_and_idempotent():
    parts = []
    for offset in range(3):
        parts.append(
            (
                np.array([[0, 0, offset], [0, 0, offset + 1]], np.int64),
                np.array([[0, 1]], np.uint64),
                np.array([offset + 1, offset + 2], np.float32),
            )
        )
    direct = dist.merge_block_skeletons(parts)
    grouped = dist.merge_block_skeletons(
        [dist.merge_block_skeletons(parts[:2]), parts[2]]
    )
    reversed_ = dist.merge_block_skeletons(parts[::-1])
    duplicate = dist.merge_block_skeletons([direct, direct])
    for candidate in (grouped, reversed_, duplicate):
        for actual, expected in zip(candidate, direct):
            np.testing.assert_array_equal(actual, expected)


def test_labeled_merge_keeps_equal_coordinates_separate():
    fragment = (
        np.array([[1, 2, 3]], np.int64),
        np.empty((0, 2), np.uint64),
        np.array([1], np.float32),
    )
    merged = dist.merge_block_skeleton_maps(
        [{-4: fragment}, {2**63 + 2: fragment, -4: fragment}]
    )
    assert list(merged) == [-4, 2**63 + 2]
    assert merged[-4][0].shape == (1, 3)
    assert merged[2**63 + 2][0].shape == (1, 3)


def test_minimum_spanning_forest_is_deterministic_and_preserves_vertices():
    graph = (
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [9, 9, 9]], np.int64),
        np.array([[2, 3], [1, 3], [0, 2], [0, 1]], np.uint64),
        np.ones(5, np.float32),
    )
    forest = dist.minimum_spanning_forest(graph)
    np.testing.assert_array_equal(forest[0], graph[0])
    np.testing.assert_array_equal(forest[2], graph[2])
    np.testing.assert_array_equal(
        forest[1], np.array([[0, 1], [0, 2], [1, 3]], np.uint64)
    )
    again = dist.minimum_spanning_forest(forest)
    for actual, expected in zip(again, forest):
        np.testing.assert_array_equal(actual, expected)
    assert _number_of_components(len(forest[0]), forest[1]) == 2


def test_two_block_binary_pipeline_stitches_shared_target():
    mask = np.zeros((11, 11, 19), dtype=np.uint8)
    mask[5, 5, 1:18] = 1
    left = np.ascontiguousarray(mask[:, :, :11])
    right = np.ascontiguousarray(mask[:, :, 10:])
    left_targets = dist.block_border_targets(
        left, [(2, "high")], origin=(0, 0, 0)
    )
    right_targets = dist.block_border_targets(
        right, [(2, "low")], origin=(0, 0, 10)
    )
    np.testing.assert_array_equal(left_targets, right_targets)
    fragments = [
        dist.block_teasar(
            left,
            open_faces=[(2, "high")],
            origin=(0, 0, 0),
            required_targets=left_targets,
        ),
        dist.block_teasar(
            right,
            open_faces=[(2, "low")],
            origin=(0, 0, 10),
            required_targets=right_targets,
        ),
    ]
    graph = dist.minimum_spanning_forest(
        dist.merge_block_skeletons(fragments)
    )
    assert graph[0].shape[0] > 0
    assert _number_of_components(len(graph[0]), graph[1]) == 1
    assert len(graph[1]) == len(graph[0]) - 1


def test_two_block_labeled_pipeline_keeps_touching_labels_separate():
    labels = np.zeros((9, 9, 17), dtype=np.int64)
    labels[3, 4, 1:16] = -3
    labels[4, 4, 1:16] = 8
    left = np.ascontiguousarray(labels[:, :, :10])
    right = np.ascontiguousarray(labels[:, :, 9:])
    left_targets = dist.block_border_targets_labels(
        left, [(2, "high")], origin=(0, 0, 0)
    )
    right_targets = dist.block_border_targets_labels(
        right, [(2, "low")], origin=(0, 0, 9)
    )
    assert left_targets.keys() == right_targets.keys()
    for label in left_targets:
        np.testing.assert_array_equal(left_targets[label], right_targets[label])
    fragments = [
        dist.block_teasar_labels(
            left,
            open_faces=[(2, "high")],
            origin=(0, 0, 0),
            required_targets=left_targets,
        ),
        dist.block_teasar_labels(
            right,
            open_faces=[(2, "low")],
            origin=(0, 0, 9),
            required_targets=right_targets,
        ),
    ]
    graphs = {
        label: dist.minimum_spanning_forest(graph)
        for label, graph in dist.merge_block_skeleton_maps(fragments).items()
    }
    assert list(graphs) == [-3, 8]
    for graph in graphs.values():
        assert _number_of_components(len(graph[0]), graph[1]) == 1
        assert len(graph[1]) == len(graph[0]) - 1


def test_empty_merge_and_invalid_fragment():
    empty = dist.merge_block_skeletons([])
    assert empty[0].shape == (0, 3)
    assert empty[1].shape == (0, 2)
    assert empty[2].shape == (0,)

    invalid = (
        np.array([[0, 0, 0]], np.int64),
        np.array([[0, 1]], np.uint64),
        np.array([1], np.float32),
    )
    with pytest.raises(ValueError, match="outside its vertex range"):
        dist.merge_block_skeletons([invalid])
