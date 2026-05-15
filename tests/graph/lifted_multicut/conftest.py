import numpy as np
import pytest

import bioimage_cpp as bic


@pytest.fixture
def chain_with_lifted():
    """4-node chain 0-1-2-3 with attractive base edges and a single
    repulsive lifted edge across (0, 3). The repulsive lifted edge should
    cause the chain to be split somewhere along the way."""
    base = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [1, 2], [2, 3]])
    base_costs = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    lifted_uvs = np.array([[0, 3]], dtype=np.uint64)
    lifted_costs = np.array([-10.0], dtype=np.float64)
    return base, base_costs, lifted_uvs, lifted_costs


@pytest.fixture
def disjoint_clusters_with_attractive_lifted():
    """Two two-node base components (0-1) and (2-3) with attractive base edges.
    A single strongly attractive lifted edge across (0, 2) should NOT cause
    the two components to be reported as one cluster — the solvers keep every
    cluster base-graph connected."""
    base = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [2, 3]])
    base_costs = np.array([1.0, 1.0], dtype=np.float64)
    lifted_uvs = np.array([[0, 2]], dtype=np.uint64)
    lifted_costs = np.array([20.0], dtype=np.float64)
    return base, base_costs, lifted_uvs, lifted_costs


@pytest.fixture
def triangle_with_lifted():
    """3-node triangle. Base edges all attractive; one lifted edge (0, 1) doubles
    up over the base (0, 1) edge — exercising the dedup-and-accumulate path."""
    base = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2], [0, 2]])
    base_costs = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    lifted_uvs = np.array([[0, 1]], dtype=np.uint64)
    lifted_costs = np.array([5.0], dtype=np.float64)
    return base, base_costs, lifted_uvs, lifted_costs


@pytest.fixture
def small_chain_bfs_problem():
    """Long chain where BFS-based lifted-edge construction yields a known
    edge count: 5 nodes, chain edges, bfs_distance=2 should add lifted edges
    for every node-pair within 2 hops that isn't already a base edge."""
    base = bic.graph.UndirectedGraph.from_edges(
        5, [[0, 1], [1, 2], [2, 3], [3, 4]]
    )
    base_costs = np.ones(4, dtype=np.float64)
    return base, base_costs
