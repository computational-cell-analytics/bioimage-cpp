import numpy as np

import bioimage_cpp as bic


def _same_partition(labels, expected):
    labels = np.asarray(labels)
    expected = np.asarray(expected)
    assert labels.shape == expected.shape
    np.testing.assert_array_equal(
        labels[:, None] == labels[None, :],
        expected[:, None] == expected[None, :],
    )


def _node_labels_from_grid_partition(partition):
    return np.asarray(partition, dtype=np.uint64).reshape(-1)


def _affinities_from_partition(partition, offsets, *, high=0.95, low=0.05):
    partition = np.asarray(partition)
    affinities = np.zeros((len(offsets),) + partition.shape, dtype=np.float64)
    for channel, offset in enumerate(offsets):
        offset = tuple(int(v) for v in offset)
        for source in np.ndindex(partition.shape):
            target = tuple(source[axis] + offset[axis] for axis in range(partition.ndim))
            if any(
                target[axis] < 0 or target[axis] >= partition.shape[axis]
                for axis in range(partition.ndim)
            ):
                continue
            affinities[(channel,) + source] = (
                high if partition[source] == partition[target] else low
            )
    return affinities


def test_grid_local_affinities_drive_multicut_partition():
    partition = np.zeros((4, 6), dtype=np.uint64)
    partition[:, 3:] = 1
    graph = bic.graph.GridGraph2D(partition.shape)
    offsets = [(1, 0), (0, 1)]
    affinities = _affinities_from_partition(partition, offsets)

    weights, valid_edges = bic.graph.features.grid_affinity_features(graph, affinities, offsets)
    np.testing.assert_array_equal(valid_edges, np.ones(graph.number_of_edges, dtype=bool))
    edge_costs = weights - 0.5

    objective = bic.graph.multicut.MulticutObjective(graph, edge_costs)
    labels = bic.graph.multicut.ChainedMulticutSolvers(
        [
            bic.graph.multicut.GreedyAdditiveMulticut(),
            bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
        ]
    ).optimize(objective)

    _same_partition(labels, _node_labels_from_grid_partition(partition))
    assert np.unique(labels).size == 2


def test_grid_long_range_affinities_drive_lifted_multicut_partition():
    partition = np.zeros((4, 8), dtype=np.uint64)
    partition[:, 4:] = 1
    graph = bic.graph.GridGraph2D(partition.shape)
    offsets = [(1, 0), (0, 1), (0, 2), (0, 3)]
    affinities = _affinities_from_partition(partition, offsets)

    local_weights, valid_edges, lifted_uvs, lifted_weights, lifted_offset_ids = (
        bic.graph.features.grid_affinity_features_with_lifted(graph, affinities, offsets)
    )
    np.testing.assert_array_equal(valid_edges, np.ones(graph.number_of_edges, dtype=bool))
    assert lifted_uvs.shape[0] > 0
    assert set(np.unique(lifted_offset_ids).tolist()) == {2, 3}

    objective = bic.graph.lifted_multicut.LiftedMulticutObjective(
        graph,
        local_weights - 0.5,
        lifted_uvs=lifted_uvs,
        lifted_costs=lifted_weights - 0.5,
    )
    labels = bic.graph.lifted_multicut.LiftedChainedSolvers(
        [
            bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut(),
            bic.graph.lifted_multicut.LiftedKernighanLinMulticut(number_of_outer_iterations=5),
        ]
    ).optimize(objective)

    _same_partition(labels, _node_labels_from_grid_partition(partition))
    assert np.unique(labels).size == 2
