import numpy as np
import pytest

import bioimage_cpp as bic


def test_objective_no_lifted_edges_matches_base(chain_with_lifted):
    base, base_costs, _, _ = chain_with_lifted
    objective = bic.graph.LiftedMulticutObjective(base, base_costs)

    # Default labels (singleton): every base edge is cut.
    assert objective.number_of_base_edges == int(base.number_of_edges)
    assert objective.number_of_lifted_edges == 0
    assert objective.energy() == pytest.approx(float(base_costs.sum()))

    # Same labeling under a multicut objective should yield the same energy.
    mc_objective = bic.graph.MulticutObjective(base, base_costs)
    assert objective.energy() == pytest.approx(mc_objective.energy())


def test_objective_with_lifted_edges(chain_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = chain_with_lifted
    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )
    assert objective.number_of_base_edges == 3
    assert objective.number_of_lifted_edges == 1

    # All singleton labels: every edge is cut -> 2+2+2+(-10) = -4
    assert objective.energy() == pytest.approx(-4.0)

    # Two-cluster split {0,1,2} | {3}: cuts 2-3 base edge (cost 2)
    # and 0-3 lifted edge (cost -10).
    assert objective.energy([0, 0, 0, 1]) == pytest.approx(-8.0)

    # Three-cluster {0,1}|{2}|{3}: cuts 1-2 base (2), 2-3 base (2),
    # 0-3 lifted (-10) = -6.
    assert objective.energy([0, 0, 1, 2]) == pytest.approx(-6.0)


def test_objective_lifted_edge_over_base_accumulates(triangle_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = triangle_with_lifted
    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )

    # The lifted (0, 1) overlaps the base (0, 1): no new edge in the lifted
    # graph, but the base edge weight should now be 1 + 5 = 6.
    assert objective.number_of_lifted_edges == 0
    # Cut everything (singleton labels): 6 + 1 + 1 = 8
    assert objective.energy() == pytest.approx(8.0)


def test_objective_lifted_edge_over_base_overwrite(triangle_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = triangle_with_lifted
    objective = bic.graph.LiftedMulticutObjective(
        base,
        base_costs,
        lifted_uvs=lifted_uvs,
        lifted_costs=lifted_costs,
        overwrite_existing=True,
    )

    # With overwrite, the (0, 1) base weight becomes 5 (the lifted value),
    # not 1 + 5 = 6.
    assert objective.energy() == pytest.approx(7.0)


def test_objective_set_cost_inserts_and_accumulates(chain_with_lifted):
    base, base_costs, _, _ = chain_with_lifted
    objective = bic.graph.LiftedMulticutObjective(base, base_costs)

    edge, is_new = objective.set_cost(0, 3, -5.0)
    assert is_new
    assert objective.number_of_lifted_edges == 1
    assert int(objective.lifted_graph.find_edge(0, 3)) == edge

    edge2, is_new2 = objective.set_cost(0, 3, -3.0)
    assert not is_new2
    assert edge2 == edge
    assert objective.weights[edge] == pytest.approx(-8.0)

    edge3, is_new3 = objective.set_cost(0, 3, 1.0, overwrite=True)
    assert not is_new3
    assert objective.weights[edge3] == pytest.approx(1.0)


def test_objective_bfs_distance_inserts_within_k_hops(small_chain_bfs_problem):
    base, base_costs = small_chain_bfs_problem

    objective = bic.graph.LiftedMulticutObjective(base, base_costs, bfs_distance=2)
    # 5-node chain at distance 2: lifted edges are (0,2), (1,3), (2,4) — 3 edges.
    assert objective.number_of_lifted_edges == 3
    lifted_uvs = objective.lifted_graph.uv_ids()[objective.number_of_base_edges:]
    pairs = {tuple(sorted(map(int, uv))) for uv in lifted_uvs}
    assert pairs == {(0, 2), (1, 3), (2, 4)}

    # Default lifted weights are zero so the energy equals the multicut energy.
    expected = bic.graph.MulticutObjective(base, base_costs).energy()
    assert objective.energy() == pytest.approx(expected)


def test_objective_bfs_distance_combines_with_explicit_lifted_costs(small_chain_bfs_problem):
    base, base_costs = small_chain_bfs_problem
    lifted_uvs = np.array([[0, 4]], dtype=np.uint64)
    lifted_costs = np.array([-3.0], dtype=np.float64)
    objective = bic.graph.LiftedMulticutObjective(
        base,
        base_costs,
        bfs_distance=2,
        lifted_uvs=lifted_uvs,
        lifted_costs=lifted_costs,
    )
    # BFS adds 3 edges; (0, 4) is new (distance 4) → 4 lifted edges total.
    assert objective.number_of_lifted_edges == 4
    assert objective.energy([0, 0, 0, 0, 1]) == pytest.approx(1.0 - 3.0)


def test_objective_labels_and_reset(chain_with_lifted):
    base, base_costs, lifted_uvs, lifted_costs = chain_with_lifted
    objective = bic.graph.LiftedMulticutObjective(
        base, base_costs, lifted_uvs=lifted_uvs, lifted_costs=lifted_costs
    )
    objective.labels = [0, 0, 1, 1]
    np.testing.assert_array_equal(objective.labels, np.array([0, 0, 1, 1], dtype=np.uint64))
    objective.reset_labels()
    np.testing.assert_array_equal(
        objective.labels, np.arange(4, dtype=np.uint64)
    )


def test_objective_validation_errors():
    base = bic.graph.UndirectedGraph.from_edges(3, [[0, 1], [1, 2]])
    base_costs = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="edge_costs"):
        bic.graph.LiftedMulticutObjective(base, np.array([1.0], dtype=np.float64))

    obj = bic.graph.LiftedMulticutObjective(base, base_costs)
    with pytest.raises(ValueError, match="labels"):
        obj.labels = [0, 1]

    with pytest.raises(ValueError, match="lifted_uvs and lifted_costs"):
        bic.graph.LiftedMulticutObjective(
            base, base_costs, lifted_uvs=np.array([[0, 2]], dtype=np.uint64)
        )

    with pytest.raises(ValueError, match="same length"):
        bic.graph.LiftedMulticutObjective(
            base,
            base_costs,
            lifted_uvs=np.array([[0, 2]], dtype=np.uint64),
            lifted_costs=np.array([1.0, 2.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="bfs_distance"):
        bic.graph.LiftedMulticutObjective(base, base_costs, bfs_distance=0)
