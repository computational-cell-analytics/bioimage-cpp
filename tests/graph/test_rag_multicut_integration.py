import numpy as np

import bioimage_cpp as bic


def _dense_by_first_occurrence(labels):
    labels = np.asarray(labels)
    dense = np.empty(labels.shape, dtype=np.uint64)
    mapping = {}
    next_label = 0
    for index in np.ndindex(labels.shape):
        label = int(labels[index])
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        dense[index] = mapping[label]
    return dense


def test_rag_feature_multicut_projection_on_realistic_oversegmentation():
    shape = (48, 64)
    block_shape = (4, 4)
    true_segmentation = np.zeros(shape, dtype=np.uint64)
    true_segmentation[8:40, 8:32] = 1
    true_segmentation[12:44, 36:60] = 2

    oversegmentation = np.empty(shape, dtype=np.uint64)
    next_label = 0
    for y in range(0, shape[0], block_shape[0]):
        for x in range(0, shape[1], block_shape[1]):
            yy = slice(y, y + block_shape[0])
            xx = slice(x, x + block_shape[1])
            oversegmentation[yy, xx] = next_label
            next_label += 1

    boundary_map = np.full(shape, 0.08, dtype=np.float64)
    boundary_pixels = np.zeros(shape, dtype=bool)
    boundary_pixels[:, :-1] |= true_segmentation[:, :-1] != true_segmentation[:, 1:]
    boundary_pixels[:, 1:] |= true_segmentation[:, :-1] != true_segmentation[:, 1:]
    boundary_pixels[:-1, :] |= true_segmentation[:-1, :] != true_segmentation[1:, :]
    boundary_pixels[1:, :] |= true_segmentation[:-1, :] != true_segmentation[1:, :]
    boundary_map[boundary_pixels] = 0.92

    rag = bic.graph.region_adjacency_graph(oversegmentation, number_of_threads=3)
    features = bic.graph.features.edge_map_features(
        rag, oversegmentation, boundary_map, number_of_threads=3
    )
    edge_costs = 0.6 - features[:, 0]

    objective = bic.graph.multicut.MulticutObjective(rag, edge_costs)
    node_labels = bic.graph.multicut.ChainedMulticutSolvers(
        [
            bic.graph.multicut.GreedyAdditiveMulticut(),
            bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
        ]
    ).optimize(objective)
    projected = bic.graph.project_node_labels_to_pixels(
        rag, oversegmentation, node_labels, number_of_threads=3
    )

    expected = _dense_by_first_occurrence(true_segmentation)
    actual = _dense_by_first_occurrence(projected)
    np.testing.assert_array_equal(actual, expected)
    assert np.unique(projected).size == 3
