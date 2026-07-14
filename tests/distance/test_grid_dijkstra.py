import heapq
import itertools

import numpy as np
import pytest

import bioimage_cpp as bic


def _reference(mask, sources, connectivity, spacing, costs=None, mode="physical"):
    mask = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    ndim = mask.ndim
    offsets = [
        off
        for off in itertools.product((-1, 0, 1), repeat=ndim)
        if 0 < np.count_nonzero(off) <= connectivity
    ]
    dist = np.full(mask.shape, np.inf, dtype=np.float64)
    heap = []
    for source in sorted(tuple(map(int, src)) for src in np.atleast_2d(sources)):
        dist[source] = 0.0
        heapq.heappush(heap, (0.0, np.ravel_multi_index(source, mask.shape), source))
    settled = np.zeros(mask.shape, dtype=bool)
    while heap:
        value, _, node = heapq.heappop(heap)
        if settled[node]:
            continue
        settled[node] = True
        for offset in offsets:
            target = tuple(node[d] + offset[d] for d in range(ndim))
            if any(target[d] < 0 or target[d] >= mask.shape[d] for d in range(ndim)):
                continue
            if not mask[target] or settled[target]:
                continue
            length = float(np.linalg.norm(np.asarray(offset) * spacing))
            if mode == "physical":
                edge = length
            elif mode == "node":
                edge = float(costs[target])
            else:
                edge = float(costs[target]) * length
            candidate = value + edge
            if candidate < dist[target]:
                dist[target] = candidate
                flat = np.ravel_multi_index(target, mask.shape)
                heapq.heappush(heap, (candidate, flat, target))
    return dist


@pytest.mark.parametrize("shape", [(7, 8), (5, 6, 7)])
def test_physical_field_matches_reference(shape):
    rng = np.random.default_rng(42)
    mask = rng.random(shape) > 0.2
    mask[(0,) * len(shape)] = True
    source = np.zeros((1, len(shape)), dtype=np.int64)
    spacing = tuple(np.linspace(0.7, 1.5, len(shape)))
    got = bic.distance.dijkstra_distance_field(mask, source, spacing=spacing)
    expected = _reference(mask, source, len(shape), spacing)
    np.testing.assert_allclose(got, expected)


def test_connectivity_controls_diagonal_steps():
    mask = np.ones((5, 5), dtype=bool)
    source = np.array([0, 0], dtype=np.int64)
    direct = bic.distance.dijkstra_distance_field(mask, source, connectivity=1)
    diagonal = bic.distance.dijkstra_distance_field(mask, source, connectivity=2)
    assert direct[4, 4] == 8.0
    assert diagonal[4, 4] == pytest.approx(4.0 * np.sqrt(2.0))


@pytest.mark.parametrize("mode", ["node", "node_times_physical"])
def test_weighted_modes_match_reference(mode):
    rng = np.random.default_rng(7)
    mask = np.ones((6, 7, 5), dtype=bool)
    mask[2:5, 3, 2] = False
    costs = rng.uniform(0.0, 3.0, size=mask.shape)
    source = np.array([[0, 0, 0]], dtype=np.int64)
    kwargs = {"cost_mode": mode, "costs": costs}
    spacing = (0.5, 1.0, 2.0)
    if mode == "node_times_physical":
        kwargs["spacing"] = spacing
    got = bic.distance.dijkstra_distance_field(mask, source, **kwargs)
    expected = _reference(
        mask,
        source,
        3,
        spacing if mode == "node_times_physical" else (1.0, 1.0, 1.0),
        costs,
        mode,
    )
    np.testing.assert_allclose(got, expected)


def test_node_costs_are_directed_destination_costs():
    mask = np.ones((1, 4), dtype=bool)
    costs = np.array([[10.0, 2.0, 3.0, 7.0]])
    forward = bic.distance.dijkstra_distance_field(
        mask, [0, 0], costs=costs, cost_mode="node"
    )
    backward = bic.distance.dijkstra_distance_field(
        mask, [0, 3], costs=costs, cost_mode="node"
    )
    assert forward[0, 3] == 12.0  # enter nodes with costs 2 + 3 + 7
    assert backward[0, 0] == 15.0  # enter nodes with costs 3 + 2 + 10


def test_multisource_unreachable_and_predecessors():
    mask = np.ones((5, 8), dtype=bool)
    mask[:, 4] = False
    sources = np.array([[0, 0], [4, 3]], dtype=np.int64)
    field, parents = bic.distance.dijkstra_distance_field(
        mask, sources, return_predecessors=True
    )
    assert field.dtype == np.float64
    assert parents.dtype == np.int64
    assert np.all(np.isfinite(field[:, :4]))
    assert np.all(np.isinf(field[:, 4:]))
    assert np.all(parents[:, 4:] == -1)
    for source in sources:
        coord = tuple(source)
        assert parents[coord] == np.ravel_multi_index(coord, mask.shape)


def test_predecessor_chain_has_the_reported_physical_length():
    mask = np.ones((6, 7), dtype=bool)
    mask[1:5, 3] = False
    source = (2, 1)
    target = (2, 5)
    spacing = np.array((2.0, 0.75))
    field, parents = bic.distance.dijkstra_distance_field(
        mask, source, spacing=spacing, return_predecessors=True
    )

    node = np.ravel_multi_index(target, mask.shape)
    source_flat = np.ravel_multi_index(source, mask.shape)
    length = 0.0
    while node != source_flat:
        parent = int(parents.flat[node])
        assert parent >= 0
        node_coord = np.asarray(np.unravel_index(node, mask.shape))
        parent_coord = np.asarray(np.unravel_index(parent, mask.shape))
        length += np.linalg.norm((node_coord - parent_coord) * spacing)
        node = parent
    assert length == pytest.approx(field[target])


def test_path_matches_field_and_chooses_cheapest_target():
    mask = np.ones((8, 9), dtype=bool)
    mask[2:7, 4] = False
    source = np.array([4, 1], dtype=np.int64)
    targets = np.array([[0, 8], [7, 8]], dtype=np.int64)
    path = bic.distance.dijkstra_path(mask, source, targets)
    field = bic.distance.dijkstra_distance_field(mask, source)
    reached = tuple(path[-1])
    expected = min((tuple(t) for t in targets), key=lambda t: (field[t], t))
    assert reached == expected
    np.testing.assert_array_equal(path[0], source)
    assert all(mask[tuple(point)] for point in path)
    steps = np.diff(path, axis=0)
    length = np.linalg.norm(steps, axis=1).sum()
    assert length == pytest.approx(field[reached])


def test_zero_cost_path_is_deterministic():
    mask = np.ones((5, 5), dtype=bool)
    costs = np.zeros(mask.shape, dtype=np.float64)
    kwargs = {"costs": costs, "cost_mode": "node"}
    first = bic.distance.dijkstra_path(mask, [4, 4], [[0, 0], [0, 4]], **kwargs)
    second = bic.distance.dijkstra_path(mask, [4, 4], [[0, 0], [0, 4]], **kwargs)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[-1], [0, 0])


def test_node_weighted_path_avoids_expensive_voxels():
    mask = np.ones((5, 7), dtype=bool)
    costs = np.ones(mask.shape, dtype=np.float64)
    costs[2, 1:6] = 100.0
    source = (2, 0)
    target = (2, 6)
    path = bic.distance.dijkstra_path(
        mask,
        source,
        target,
        connectivity=1,
        costs=costs,
        cost_mode="node",
    )
    field = bic.distance.dijkstra_distance_field(
        mask,
        source,
        connectivity=1,
        costs=costs,
        cost_mode="node",
    )
    assert not any(tuple(point) in {(2, x) for x in range(1, 6)} for point in path)
    entered_cost = sum(costs[tuple(point)] for point in path[1:])
    assert entered_cost == pytest.approx(field[target])


def test_source_target_path_has_one_vertex():
    path = bic.distance.dijkstra_path(np.ones((3, 3), bool), [1, 1], [1, 1])
    np.testing.assert_array_equal(path, [[1, 1]])


def test_unreachable_path_raises():
    mask = np.zeros((3, 5), dtype=bool)
    mask[1, 0] = True
    mask[1, 4] = True
    with pytest.raises(RuntimeError, match="no target is reachable"):
        bic.distance.dijkstra_path(mask, [1, 0], [1, 4])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"cost_mode": "bad"}, "cost_mode"),
        ({"costs": np.ones((3, 3))}, "costs must be None"),
        ({"cost_mode": "node"}, "costs are required"),
        (
            {"cost_mode": "node", "costs": np.ones((3, 3)), "spacing": 2},
            "spacing must be None",
        ),
        ({"connectivity": 3}, "connectivity"),
    ],
)
def test_invalid_options(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        bic.distance.dijkstra_distance_field(np.ones((3, 3)), [0, 0], **kwargs)


def test_rejects_invalid_points_and_costs():
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    with pytest.raises(ValueError, match="foreground"):
        bic.distance.dijkstra_distance_field(mask, [1, 1])
    with pytest.raises(ValueError, match="at least one"):
        bic.distance.dijkstra_distance_field(mask, np.empty((0, 2), np.int64))
    costs = np.ones(mask.shape)
    costs[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite non-negative"):
        bic.distance.dijkstra_distance_field(
            mask, [0, 0], costs=costs, cost_mode="node"
        )


def test_non_contiguous_mask_and_costs_are_accepted():
    mask = np.ones((8, 9), dtype=np.uint8)[::2, ::2]
    costs = np.ones((8, 9), dtype=np.float32)[::2, ::2]
    result = bic.distance.dijkstra_distance_field(
        mask, [0, 0], costs=costs, cost_mode="node"
    )
    assert result.shape == mask.shape
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("mode", ["physical", "node", "node_times_physical"])
def test_threaded_fields_match_sequential_exactly(mode):
    shape = (24, 40, 40)  # Above the parallel-backend workload threshold.
    rng = np.random.default_rng(123)
    mask = np.ones(shape, dtype=bool)
    mask[:, 20, 4:36] = False
    mask[:, 20, 18:22] = True
    costs = rng.uniform(0.0, 4.0, size=shape)
    yy, xx = np.indices(shape[1:])
    sources = np.stack(
        [np.zeros(yy.size, dtype=np.int64), yy.ravel(), xx.ravel()], axis=1
    )
    sources = sources[mask[0].ravel()]
    kwargs = {"cost_mode": mode}
    if mode != "physical":
        kwargs["costs"] = costs
    if mode != "node":
        kwargs["spacing"] = (1.5, 0.75, 1.25)

    expected = bic.distance.dijkstra_distance_field(
        mask, sources, number_of_threads=1, **kwargs
    )
    got, predecessors = bic.distance.dijkstra_distance_field(
        mask,
        sources,
        number_of_threads=4,
        return_predecessors=True,
        **kwargs,
    )
    got_two, predecessors_two = bic.distance.dijkstra_distance_field(
        mask,
        sources,
        number_of_threads=2,
        return_predecessors=True,
        **kwargs,
    )
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got_two, got)
    np.testing.assert_array_equal(predecessors_two, predecessors)

    for coordinate in ((22, 38, 38), (8, 10, 30), (20, 30, 4)):
        node = np.ravel_multi_index(coordinate, shape)
        steps = 0
        while int(predecessors.flat[node]) != node:
            parent = int(predecessors.flat[node])
            assert parent >= 0
            node = parent
            steps += 1
            assert steps <= np.prod(shape)
        assert np.unravel_index(node, shape)[0] == 0


def test_threaded_zero_cost_path_is_deterministic_across_thread_counts():
    mask = np.ones((24, 40, 40), dtype=bool)
    costs = np.zeros(mask.shape, dtype=np.float64)
    kwargs = {
        "costs": costs,
        "cost_mode": "node_times_physical",
        "number_of_threads": 2,
    }
    first = bic.distance.dijkstra_path(
        mask, [23, 39, 39], [[0, 0, 0], [0, 39, 39]], **kwargs
    )
    second = bic.distance.dijkstra_path(
        mask,
        [23, 39, 39],
        [[0, 0, 0], [0, 39, 39]],
        **{**kwargs, "number_of_threads": 4},
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[-1], [0, 0, 0])


def test_rejects_negative_dijkstra_thread_count():
    with pytest.raises(ValueError, match="number_of_threads"):
        bic.distance.dijkstra_distance_field(
            np.ones((3, 3), dtype=bool), [0, 0], number_of_threads=-1
        )
