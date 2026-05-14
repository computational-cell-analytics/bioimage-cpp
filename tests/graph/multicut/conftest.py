import numpy as np
import pytest

import bioimage_cpp as bic


@pytest.fixture
def external_toy_problem():
    graph = bic.graph.UndirectedGraph.from_edges(
        6,
        [
            [0, 1],
            [0, 3],
            [1, 2],
            [1, 4],
            [2, 5],
            [3, 4],
            [4, 5],
        ],
    )
    costs = np.array([5, -20, 5, 5, -20, 5, 5], dtype=np.float64)
    expected_cut_edges = np.array([False, True, False, False, True, False, False])
    return graph, costs, expected_cut_edges


@pytest.fixture
def chain_problem():
    graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [1, 2], [2, 3]])
    costs = np.array([1.0, 2.0, -5.0], dtype=np.float64)
    return graph, costs


@pytest.fixture
def frustrated_triangle():
    graph = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2], [0, 2]])
    costs = np.array([2.0, 2.0, -5.0], dtype=np.float64)
    return graph, costs


@pytest.fixture
def grid_problem():
    edges = []
    costs = []
    shape = (5, 5)
    for y in range(shape[0]):
        for x in range(shape[1]):
            node = y * shape[1] + x
            if x + 1 < shape[1]:
                edges.append([node, node + 1])
                costs.append(1.5 if x != 2 else -4.0)
            if y + 1 < shape[0]:
                edges.append([node, node + shape[1]])
                costs.append(1.0 if y != 2 else -3.0)
    graph = bic.graph.UndirectedGraph.from_edges(shape[0] * shape[1], edges)
    return graph, np.array(costs, dtype=np.float64)
