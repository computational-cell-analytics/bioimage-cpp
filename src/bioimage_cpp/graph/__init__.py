"""Graph data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .. import _core
from ._external import (
    DEFAULT_EXTERNAL_MULTICUT_PROBLEM_PATH,
    EXTERNAL_MULTICUT_PROBLEM_URL,
    LiftedMulticutProblem,
    external_multicut_problem_path,
    lifted_multicut_problem_path,
    load_external_multicut_problem,
    load_external_multicut_problem_data,
    load_lifted_multicut_problem,
    load_multicut_problem,
    load_multicut_problem_data,
    multicut_problem_path,
)

_REGION_ADJACENCY_GRAPH_BY_DTYPE = {
    np.dtype("uint32"): _core._region_adjacency_graph_uint32,
    np.dtype("uint64"): _core._region_adjacency_graph_uint64,
    np.dtype("int32"): _core._region_adjacency_graph_int32,
    np.dtype("int64"): _core._region_adjacency_graph_int64,
}

_EDGE_MAP_FEATURES_BY_DTYPE = {
    np.dtype("uint32"): _core._accumulate_edge_map_features_uint32,
    np.dtype("uint64"): _core._accumulate_edge_map_features_uint64,
    np.dtype("int32"): _core._accumulate_edge_map_features_int32,
    np.dtype("int64"): _core._accumulate_edge_map_features_int64,
}

_AFFINITY_FEATURES_BY_DTYPE = {
    np.dtype("uint32"): _core._accumulate_affinity_features_uint32,
    np.dtype("uint64"): _core._accumulate_affinity_features_uint64,
    np.dtype("int32"): _core._accumulate_affinity_features_int32,
    np.dtype("int64"): _core._accumulate_affinity_features_int64,
}

_LIFTED_EDGES_FROM_AFFINITIES_BY_DTYPE = {
    np.dtype("uint32"): _core._lifted_edges_from_affinities_uint32,
    np.dtype("uint64"): _core._lifted_edges_from_affinities_uint64,
    np.dtype("int32"): _core._lifted_edges_from_affinities_int32,
    np.dtype("int64"): _core._lifted_edges_from_affinities_int64,
}

_LIFTED_AFFINITY_FEATURES_BY_DTYPE = {
    np.dtype("uint32"): _core._accumulate_lifted_affinity_features_uint32,
    np.dtype("uint64"): _core._accumulate_lifted_affinity_features_uint64,
    np.dtype("int32"): _core._accumulate_lifted_affinity_features_int32,
    np.dtype("int64"): _core._accumulate_lifted_affinity_features_int64,
}

_EDGE_WEIGHTED_WATERSHED_BY_DTYPE = {
    (np.dtype("float32"), np.dtype("uint32")): _core._edge_weighted_watershed_float32_uint32,
    (np.dtype("float32"), np.dtype("uint64")): _core._edge_weighted_watershed_float32_uint64,
    (np.dtype("float32"), np.dtype("int32")): _core._edge_weighted_watershed_float32_int32,
    (np.dtype("float32"), np.dtype("int64")): _core._edge_weighted_watershed_float32_int64,
    (np.dtype("float64"), np.dtype("uint32")): _core._edge_weighted_watershed_float64_uint32,
    (np.dtype("float64"), np.dtype("uint64")): _core._edge_weighted_watershed_float64_uint64,
    (np.dtype("float64"), np.dtype("int32")): _core._edge_weighted_watershed_float64_int32,
    (np.dtype("float64"), np.dtype("int64")): _core._edge_weighted_watershed_float64_int64,
}

_PROJECT_NODE_LABELS_TO_PIXELS_BY_DTYPE = {
    np.dtype("uint32"): _core._project_node_labels_to_pixels_uint32,
    np.dtype("uint64"): _core._project_node_labels_to_pixels_uint64,
    np.dtype("int32"): _core._project_node_labels_to_pixels_int32,
    np.dtype("int64"): _core._project_node_labels_to_pixels_int64,
}

SIMPLE_EDGE_FEATURE_NAMES = ("mean", "size")
COMPLEX_EDGE_FEATURE_NAMES = (
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p5",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
    "size",
)


class UndirectedGraph(_core.UndirectedGraph):
    """Undirected graph with consecutive node and edge ids.

    Nodes are fixed at construction and addressed by ids
    ``0 .. number_of_nodes - 1``. Edges are inserted lazily and receive
    consecutive ids in insertion order. Re-inserting an existing undirected edge
    returns the existing edge id.
    """

    def insert_edges(self, uvs):
        return super().insert_edges(_as_uv_array(uvs, "uvs"))

    def find_edges(self, uvs):
        return super().find_edges(_as_uv_array(uvs, "uvs"))

    def insertEdges(self, uvs):
        return self.insert_edges(uvs)

    def findEdges(self, uvs):
        return self.find_edges(uvs)

    def extract_subgraph_from_nodes(self, nodes):
        return super().extract_subgraph_from_nodes(_as_node_array(nodes, "nodes"))

    def edges_from_node_list(self, nodes):
        return super().edges_from_node_list(_as_node_array(nodes, "nodes"))

    def extractSubgraphFromNodes(self, nodes):
        return self.extract_subgraph_from_nodes(nodes)

    def edgesFromNodeList(self, nodes):
        return self.edges_from_node_list(nodes)

    @classmethod
    def from_edges(cls, number_of_nodes: int, uvs):
        graph = cls(number_of_nodes)
        graph.insert_edges(uvs)
        return graph

    @classmethod
    def deserialize(cls, serialization):
        serialization = _as_serialization_array(serialization)
        number_of_nodes = int(serialization[0])
        number_of_edges = int(serialization[1])
        uvs = serialization[2:].reshape(number_of_edges, 2)
        return cls.from_edges(number_of_nodes, uvs)


def _as_uv_array(uvs, name: str) -> np.ndarray:
    array = np.asarray(uvs, dtype=np.uint64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_edges, 2)")
    return np.ascontiguousarray(array)


def _as_node_array(nodes, name: str) -> np.ndarray:
    array = np.asarray(nodes, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return np.ascontiguousarray(array)


def _as_serialization_array(serialization) -> np.ndarray:
    array = np.asarray(serialization, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError("serialization must be a 1D array")
    if array.size < 2:
        raise ValueError("serialization must have at least two entries")
    number_of_edges = int(array[1])
    if array.size != 2 + 2 * number_of_edges:
        raise ValueError("serialization size must be 2 + 2 * number_of_edges")
    return np.ascontiguousarray(array)


def _copy_graph(graph: UndirectedGraph | RegionAdjacencyGraph) -> _core.UndirectedGraph:
    # `uv_ids()` always returns a unique list (graphs deduplicate on insert),
    # so we can use the bulk constructor that skips per-edge hash dedup —
    # significantly faster than `insert_edges` for large graphs. The result
    # is a ``_core.UndirectedGraph``; downstream code (objectives, solvers,
    # validators) uses base-class methods that work identically.
    if graph.number_of_edges == 0:
        return _core.UndirectedGraph(int(graph.number_of_nodes))
    return _core.UndirectedGraph.from_unique_edges(
        int(graph.number_of_nodes), graph.uv_ids()
    )


def _as_edge_costs(edge_costs, graph: UndirectedGraph | RegionAdjacencyGraph) -> np.ndarray:
    array = np.asarray(edge_costs, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("edge_costs must be a 1D array")
    if array.shape[0] != graph.number_of_edges:
        raise ValueError("edge_costs length must match graph number_of_edges")
    return np.ascontiguousarray(array)


def _as_node_labels(labels, graph: UndirectedGraph | RegionAdjacencyGraph) -> np.ndarray:
    array = np.asarray(labels, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if array.shape[0] != graph.number_of_nodes:
        raise ValueError("labels length must match graph number_of_nodes")
    return np.ascontiguousarray(array)


def _as_1d_array(values, dtype, name: str, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if array.shape[0] != expected_size:
        raise ValueError(
            f"{name} length must be {expected_size}, got {array.shape[0]}"
        )
    return np.ascontiguousarray(array)


def _dense_labels(labels) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint64)
    _, dense = np.unique(labels, return_inverse=True)
    return np.ascontiguousarray(dense.astype(np.uint64, copy=False))


def _subproblem_from_edges(number_of_nodes: int, nodes, uvs, edge_costs):
    local_ids = np.full(int(number_of_nodes), -1, dtype=np.int64)
    local_ids[nodes] = np.arange(nodes.size, dtype=np.int64)
    local_uvs = local_ids[np.asarray(uvs, dtype=np.uint64)]
    sub_graph = UndirectedGraph(int(nodes.size), int(len(edge_costs)))
    if local_uvs.size:
        sub_graph.insert_edges(np.ascontiguousarray(local_uvs.astype(np.uint64, copy=False)))
    return sub_graph, np.ascontiguousarray(np.asarray(edge_costs, dtype=np.float64))


def undirected_graph(number_of_nodes: int) -> UndirectedGraph:
    """Create an :class:`UndirectedGraph`."""
    return UndirectedGraph(number_of_nodes)


RegionAdjacencyGraph = _core.RegionAdjacencyGraph


def connected_components(
    graph: UndirectedGraph | RegionAdjacencyGraph,
    edge_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute dense connected-component labels for graph nodes.

    If ``edge_mask`` is given, only edges with a true mask value contribute to
    the connected components.
    """
    if edge_mask is None:
        return _core._connected_components(graph)

    mask = np.asarray(edge_mask)
    if mask.dtype != np.dtype("bool"):
        raise TypeError(f"edge_mask must have dtype bool, got dtype={mask.dtype}")
    if mask.ndim != 1:
        raise ValueError("edge_mask must be a 1D array")
    if mask.shape[0] != graph.number_of_edges:
        raise ValueError("edge_mask length must match graph number_of_edges")
    return _core._connected_components_masked(
        graph, np.ascontiguousarray(mask.astype(np.uint8, copy=False))
    )


def breadth_first_search(
    graph: UndirectedGraph | RegionAdjacencyGraph,
    source: int,
    *,
    max_distance: int | None = None,
    include_source: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Breadth-first search from ``source`` on ``graph``.

    Returns ``(nodes, distances)`` — two 1D ``uint64`` arrays of equal length,
    listing every reachable node within ``max_distance`` hops (inclusive) in
    BFS order along with its hop distance from the source.

    Parameters
    ----------
    graph:
        :class:`UndirectedGraph` or :class:`RegionAdjacencyGraph`.
    source:
        Source node id.
    max_distance:
        Maximum hop distance from ``source`` to report. ``None`` (default)
        means no limit — the search expands until the entire connected
        component of ``source`` is visited.
    include_source:
        If ``True`` (default), the source itself is reported with distance 0.
        Set to ``False`` for "nodes within k hops, excluding self" queries.
    """
    if int(source) < 0 or int(source) >= int(graph.number_of_nodes):
        raise ValueError(
            f"source must be in [0, number_of_nodes), got source={source}, "
            f"number_of_nodes={int(graph.number_of_nodes)}"
        )
    if max_distance is None:
        limit = (1 << 64) - 1
    else:
        if int(max_distance) < 0:
            raise ValueError("max_distance must be non-negative")
        limit = int(max_distance)
    return _core._breadth_first_search(
        graph, int(source), limit, bool(include_source)
    )


def edge_weighted_watershed(
    graph: UndirectedGraph | RegionAdjacencyGraph,
    edge_weights,
    seeds,
) -> np.ndarray:
    """Kruskal-style edge-weighted seeded watershed on an undirected graph.

    Edges are visited in ascending weight order. Two distinct components are
    merged iff at least one of them is unlabeled (seed label ``0``); the
    non-zero seed label then propagates. Two distinct already-labeled
    components are never merged, so seed boundaries are preserved.

    Parameters
    ----------
    graph:
        :class:`UndirectedGraph` or :class:`RegionAdjacencyGraph`.
    edge_weights:
        1D array of length ``graph.number_of_edges``. Supported dtypes are
        ``float32`` and ``float64``. Other floating dtypes are cast to
        ``float32`` (matches nifty); other dtypes raise ``TypeError``.
    seeds:
        1D array of length ``graph.number_of_nodes``. Supported dtypes are
        ``uint32``, ``uint64``, ``int32``, ``int64``. ``0`` marks unlabeled
        nodes; positive ids are seed labels and propagate along low-weight
        paths. Signed seed arrays must not contain negative values.

    Returns
    -------
    np.ndarray
        1D array of length ``graph.number_of_nodes`` with the same dtype as
        ``seeds``. Nodes reachable from a seed receive that seed's label;
        unreachable nodes remain ``0``. Seed label values are preserved (no
        dense relabeling).
    """
    weight_array = np.asarray(edge_weights)
    if weight_array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        if np.issubdtype(weight_array.dtype, np.floating):
            weight_array = weight_array.astype(np.float32, copy=False)
        else:
            raise TypeError(
                "edge_weights must have dtype float32 or float64, got "
                f"dtype={weight_array.dtype}"
            )

    seed_array = np.asarray(seeds)
    if seed_array.dtype not in (
        np.dtype("uint32"),
        np.dtype("uint64"),
        np.dtype("int32"),
        np.dtype("int64"),
    ):
        raise TypeError(
            "seeds must have dtype uint32, uint64, int32, or int64, got "
            f"dtype={seed_array.dtype}"
        )

    weight_array = _as_1d_array(
        weight_array, weight_array.dtype, "edge_weights", int(graph.number_of_edges)
    )
    seed_array = _as_1d_array(
        seed_array, seed_array.dtype, "seeds", int(graph.number_of_nodes)
    )

    run = _EDGE_WEIGHTED_WATERSHED_BY_DTYPE[(weight_array.dtype, seed_array.dtype)]
    return run(graph, weight_array, seed_array)


class MulticutObjective:
    """Multicut objective for an undirected graph and edge costs."""

    def __init__(
        self,
        graph: UndirectedGraph | RegionAdjacencyGraph,
        edge_costs,
        initial_labels=None,
    ):
        self._graph = _copy_graph(graph)
        self._edge_costs = _as_edge_costs(edge_costs, self._graph)
        if initial_labels is None:
            self._labels = np.arange(self._graph.number_of_nodes, dtype=np.uint64)
        else:
            self._labels = _as_node_labels(initial_labels, self._graph)

    @property
    def graph(self) -> UndirectedGraph:
        return self._graph

    @property
    def edge_costs(self) -> np.ndarray:
        return self._edge_costs

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @labels.setter
    def labels(self, labels) -> None:
        self._labels = _as_node_labels(labels, self._graph)

    def set_labels(self, labels) -> None:
        self.labels = labels

    def reset_labels(self) -> None:
        self._labels = np.arange(self._graph.number_of_nodes, dtype=np.uint64)

    def energy(self, labels=None) -> float:
        label_array = self._labels if labels is None else _as_node_labels(labels, self._graph)
        return float(_core._multicut_energy(self._graph, self._edge_costs, label_array))


class MulticutSolver(ABC):
    """Base class for multicut solvers."""

    @abstractmethod
    def optimize(self, objective: MulticutObjective) -> np.ndarray:
        """Optimize ``objective`` and return the node labeling."""


class GreedyAdditiveMulticut(MulticutSolver):
    def __init__(
        self,
        *,
        weight_stop: float = 0.0,
        node_num_stop: float = -1.0,
        add_noise: bool = False,
        seed: int = 42,
        sigma: float = 1.0,
    ):
        self.weight_stop = float(weight_stop)
        self.node_num_stop = float(node_num_stop)
        self.add_noise = bool(add_noise)
        self.seed = int(seed)
        self.sigma = float(sigma)

    def optimize(self, objective: MulticutObjective) -> np.ndarray:
        labels = _core._multicut_greedy_additive(
            objective.graph,
            objective.edge_costs,
            self.weight_stop,
            self.node_num_stop,
            self.add_noise,
            self.seed,
            self.sigma,
        )
        objective.labels = labels
        return objective.labels

    def _build_cpp_sub_solver(self):
        return _core._GreedyAdditiveMulticutSubSolver(
            weight_stop=self.weight_stop,
            node_num_stop=self.node_num_stop,
            add_noise=self.add_noise,
            seed=self.seed,
            sigma=self.sigma,
        )


class GreedyFixationMulticut(MulticutSolver):
    def __init__(self, *, weight_stop: float = 0.0, node_num_stop: float = -1.0):
        self.weight_stop = float(weight_stop)
        self.node_num_stop = float(node_num_stop)

    def optimize(self, objective: MulticutObjective) -> np.ndarray:
        labels = _core._multicut_greedy_fixation(
            objective.graph,
            objective.edge_costs,
            self.weight_stop,
            self.node_num_stop,
        )
        objective.labels = labels
        return objective.labels

    def _build_cpp_sub_solver(self):
        return _core._GreedyFixationMulticutSubSolver(
            weight_stop=self.weight_stop,
            node_num_stop=self.node_num_stop,
        )


class KernighanLinMulticut(MulticutSolver):
    def __init__(
        self,
        *,
        number_of_outer_iterations: int = 100,
        number_of_inner_iterations: int | None = None,
        epsilon: float = 1.0e-6,
    ):
        self.number_of_outer_iterations = int(number_of_outer_iterations)
        if self.number_of_outer_iterations < 0:
            raise ValueError("number_of_outer_iterations must be non-negative")
        self.number_of_inner_iterations = (
            None if number_of_inner_iterations is None else int(number_of_inner_iterations)
        )
        self.epsilon = float(epsilon)

    def optimize(self, objective: MulticutObjective) -> np.ndarray:
        initial_labels = objective.labels
        if np.array_equal(
            initial_labels,
            np.arange(objective.graph.number_of_nodes, dtype=np.uint64),
        ):
            initial_labels = _core._multicut_greedy_additive(
                objective.graph,
                objective.edge_costs,
                0.0,
                -1.0,
                False,
                42,
                1.0,
            )
        labels = _core._multicut_kernighan_lin(
            objective.graph,
            objective.edge_costs,
            initial_labels,
            self.number_of_outer_iterations,
            self.epsilon,
        )
        objective.labels = labels
        return objective.labels

    def _build_cpp_sub_solver(self):
        return _core._KernighanLinMulticutSubSolver(
            number_of_outer_iterations=self.number_of_outer_iterations,
            epsilon=self.epsilon,
        )


class ProposalGenerator(ABC):
    """Base class for fusion-move proposal generators.

    Concrete generators carry settings on the Python side. ``_build_for_thread``
    constructs an independent underlying C++ proposal-generator object whose
    seed is offset by ``seed_offset`` so that parallel proposal slots produce
    distinct, reproducible streams.
    """

    @abstractmethod
    def _build_for_thread(
        self,
        graph: UndirectedGraph | RegionAdjacencyGraph,
        edge_costs: np.ndarray,
        seed_offset: int,
    ):
        """Construct the underlying C++ proposal generator with a seed offset."""


class WatershedProposalGenerator(ProposalGenerator):
    """Watershed proposal generator (nifty's fusion-move workhorse).

    Per call: add Gaussian noise to edge costs, drop random seeds at the
    endpoints of negative-cost edges, run the edge-weighted watershed.
    """

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        n_seeds_fraction: float = 0.1,
        seed: int = 0,
    ):
        self.sigma = float(sigma)
        self.n_seeds_fraction = float(n_seeds_fraction)
        self.seed = int(seed)

    def _build_for_thread(self, graph, edge_costs, seed_offset):
        return _core._WatershedProposalGenerator(
            graph,
            edge_costs,
            sigma=self.sigma,
            n_seeds_fraction=self.n_seeds_fraction,
            seed=self.seed + int(seed_offset),
        )


class GreedyAdditiveProposalGenerator(ProposalGenerator):
    """Greedy-additive multicut proposal generator.

    Per call: run the greedy-additive multicut solver with noisy edge weights;
    the seed advances every call so successive proposals differ.
    """

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        weight_stop: float = 0.0,
        node_num_stop: float = -1.0,
        seed: int = 0,
    ):
        self.sigma = float(sigma)
        self.weight_stop = float(weight_stop)
        self.node_num_stop = float(node_num_stop)
        self.seed = int(seed)

    def _build_for_thread(self, graph, edge_costs, seed_offset):
        return _core._GreedyAdditiveMulticutProposalGenerator(
            graph,
            edge_costs,
            sigma=self.sigma,
            weight_stop=self.weight_stop,
            node_num_stop=self.node_num_stop,
            seed=self.seed + int(seed_offset),
        )


class FusionMoveMulticut(MulticutSolver):
    """Fusion-move multicut solver.

    Iteratively generates proposals via ``proposal_generator``, fuses them
    with the current best labeling, and accepts improvements. The fuse step
    solves a contracted multicut subproblem with ``sub_solver``; if omitted,
    the default sub-solver is :class:`GreedyAdditiveMulticut`.

    If the objective's current labels are the trivial singleton labeling, the
    driver warm-starts with one greedy-additive pass before the proposal loop.
    The best-of safety net guarantees energy never increases across iterations.

    Threading: ``number_of_threads > 1`` runs ``number_of_parallel_proposals``
    proposal generators in parallel within each iteration. Each parallel slot
    uses an independent proposal generator with seed ``proposal_generator.seed
    + slot_index``. By default ``number_of_parallel_proposals`` is ``2`` when
    ``number_of_threads == 1`` and ``number_of_threads`` otherwise; pass it
    explicitly to override.

    Multi-proposal fuse: when at least two parallel pairwise fuses fail to
    improve on the current best, a joint multi-proposal fuse is run over the
    surviving fused candidates (matches nifty's ``ccFusionMoveBased`` stage-2
    behaviour). With ``number_of_parallel_proposals == 2`` this stage rarely
    triggers; it becomes useful as ``number_of_parallel_proposals`` grows.
    """

    def __init__(
        self,
        *,
        proposal_generator: ProposalGenerator,
        sub_solver: MulticutSolver | None = None,
        number_of_iterations: int = 10,
        stop_if_no_improvement: int = 4,
        number_of_threads: int = 1,
        number_of_parallel_proposals: int | None = None,
    ):
        if not isinstance(proposal_generator, ProposalGenerator):
            raise TypeError("proposal_generator must inherit from ProposalGenerator")
        if sub_solver is not None and not isinstance(sub_solver, MulticutSolver):
            raise TypeError("sub_solver must inherit from MulticutSolver")
        if sub_solver is not None and not hasattr(sub_solver, "_build_cpp_sub_solver"):
            raise TypeError(
                "sub_solver must be a built-in multicut solver "
                "(custom Python solvers are not supported as fusion-move sub-solvers)"
            )
        n_threads = int(number_of_threads)
        if n_threads < 1:
            raise ValueError("number_of_threads must be >= 1")
        if number_of_parallel_proposals is None:
            n_parallel = 2 if n_threads == 1 else n_threads
        else:
            n_parallel = int(number_of_parallel_proposals)
        if n_parallel < 1:
            raise ValueError("number_of_parallel_proposals must be >= 1")

        self.proposal_generator = proposal_generator
        self.sub_solver = sub_solver
        self.number_of_iterations = int(number_of_iterations)
        self.stop_if_no_improvement = int(stop_if_no_improvement)
        self.number_of_threads = n_threads
        self.number_of_parallel_proposals = n_parallel
        if self.number_of_iterations < 0:
            raise ValueError("number_of_iterations must be non-negative")
        if self.stop_if_no_improvement < 1:
            raise ValueError("stop_if_no_improvement must be >= 1")

    def optimize(self, objective: MulticutObjective) -> np.ndarray:
        # Build one C++ proposal generator per parallel slot, each with a
        # distinct seed offset, so parallel streams are independent and
        # reproducible.
        cpp_pgens = [
            self.proposal_generator._build_for_thread(
                objective.graph, objective.edge_costs, slot
            )
            for slot in range(self.number_of_parallel_proposals)
        ]
        cpp_sub_solver = (
            None if self.sub_solver is None else self.sub_solver._build_cpp_sub_solver()
        )
        labels = _core._multicut_fusion_move(
            objective.graph,
            objective.edge_costs,
            objective.labels,
            cpp_pgens,
            cpp_sub_solver,
            self.number_of_iterations,
            self.stop_if_no_improvement,
            self.number_of_threads,
            self.number_of_parallel_proposals,
        )
        objective.labels = labels
        return objective.labels


class ChainedMulticutSolvers(MulticutSolver):
    def __init__(self, solvers):
        self.solvers = list(solvers)
        if len(self.solvers) == 0:
            raise ValueError("solvers must contain at least one solver")
        if not all(isinstance(solver, MulticutSolver) for solver in self.solvers):
            raise TypeError("all solvers must inherit from MulticutSolver")

    def optimize(self, objective: MulticutObjective) -> np.ndarray:
        labels = objective.labels
        for solver in self.solvers:
            labels = solver.optimize(objective)
        return labels


class MulticutDecomposer(MulticutSolver):
    def __init__(
        self,
        sub_solver: MulticutSolver,
        *,
        fallthrough_solver: MulticutSolver | None = None,
        number_of_threads: int = 0,
    ):
        if not isinstance(sub_solver, MulticutSolver):
            raise TypeError("sub_solver must inherit from MulticutSolver")
        if fallthrough_solver is not None and not isinstance(fallthrough_solver, MulticutSolver):
            raise TypeError("fallthrough_solver must inherit from MulticutSolver")
        self.sub_solver = sub_solver
        self.fallthrough_solver = fallthrough_solver
        self.number_of_threads = _normalize_number_of_threads(number_of_threads)

    def optimize(self, objective: MulticutObjective) -> np.ndarray:
        if self.fallthrough_solver is None and isinstance(self.sub_solver, GreedyAdditiveMulticut):
            return self.sub_solver.optimize(objective)

        component_labels = connected_components(
            objective.graph,
            edge_mask=objective.edge_costs > 0.0,
        )
        number_of_components = int(component_labels.max()) + 1 if component_labels.size else 0
        if number_of_components <= 1:
            solver = self.fallthrough_solver or self.sub_solver
            return solver.optimize(objective)

        global_labels = np.empty(objective.graph.number_of_nodes, dtype=np.uint64)
        label_offset = 0
        all_uvs = objective.graph.uv_ids()
        for component in range(number_of_components):
            nodes = np.flatnonzero(component_labels == component).astype(np.uint64)
            if nodes.size == 1:
                global_labels[int(nodes[0])] = label_offset
                label_offset += 1
                continue

            edge_ids = objective.graph.edges_from_node_list(nodes)
            sub_graph, sub_costs = _subproblem_from_edges(
                objective.graph.number_of_nodes,
                nodes,
                all_uvs[edge_ids],
                objective.edge_costs[edge_ids],
            )
            sub_objective = MulticutObjective(sub_graph, sub_costs)
            sub_labels = self.sub_solver.optimize(sub_objective)
            sub_labels = _dense_labels(sub_labels)
            global_labels[nodes] = sub_labels + label_offset
            label_offset += int(sub_labels.max()) + 1

        objective.labels = _dense_labels(global_labels)
        return objective.labels


class LiftedMulticutObjective:
    """Lifted multicut objective.

    Stores a base graph + base edge costs together with an internal *lifted
    graph* that is a superset of the base graph (base edges occupy ids
    ``0 .. base.number_of_edges - 1``; lifted edges follow). The energy of a
    node labeling is the sum of base + lifted edge weights across cut edges.

    The lifted edges can be supplied either as explicit ``(uvs, costs)``
    arrays, via a ``bfs_distance=k`` constructor argument that inserts a
    zero-weight lifted edge for every pair of nodes within ``k`` hops of each
    other in the base graph, or by calling :meth:`set_cost` after construction.
    """

    def __init__(
        self,
        graph: UndirectedGraph | RegionAdjacencyGraph,
        edge_costs,
        *,
        lifted_uvs=None,
        lifted_costs=None,
        bfs_distance: int | None = None,
        overwrite_existing: bool = False,
        initial_labels=None,
    ):
        # The objective holds a reference to the user's base graph (no
        # defensive copy — the C++ ``Objective`` already keeps a const
        # reference). The user is expected to treat the input graph as
        # immutable while the objective is alive; mutations are visible to
        # the objective and may produce undefined behaviour.
        base_graph = graph
        base_costs = _as_edge_costs(edge_costs, base_graph)

        # Use the bulk constructor for the lifted graph's base portion to
        # bypass the per-edge hash dedup that ``insert_edges`` performs.
        if int(base_graph.number_of_edges) > 0:
            lifted_graph = _core.UndirectedGraph.from_unique_edges(
                int(base_graph.number_of_nodes), base_graph.uv_ids()
            )
        else:
            lifted_graph = _core.UndirectedGraph(int(base_graph.number_of_nodes))

        weights_list = [base_costs.copy()]

        if bfs_distance is not None:
            distance = int(bfs_distance)
            if distance < 1:
                raise ValueError("bfs_distance must be >= 1")
            bfs_uvs = []
            for source in range(int(base_graph.number_of_nodes)):
                nodes, _ = breadth_first_search(
                    base_graph,
                    source,
                    max_distance=distance,
                    include_source=False,
                )
                if nodes.size == 0:
                    continue
                tail = nodes[nodes > source]
                if tail.size == 0:
                    continue
                source_column = np.full(tail.size, source, dtype=np.uint64)
                bfs_uvs.append(np.stack([source_column, tail], axis=1))
            if bfs_uvs:
                bfs_uv_array = np.ascontiguousarray(np.concatenate(bfs_uvs, axis=0))
                _add_lifted_edges(
                    lifted_graph,
                    weights_list,
                    bfs_uv_array,
                    np.zeros(bfs_uv_array.shape[0], dtype=np.float64),
                    overwrite_existing=overwrite_existing,
                )

        if lifted_uvs is not None or lifted_costs is not None:
            if lifted_uvs is None or lifted_costs is None:
                raise ValueError(
                    "lifted_uvs and lifted_costs must be provided together"
                )
            uv_array = _as_uv_array(lifted_uvs, "lifted_uvs")
            cost_array = np.asarray(lifted_costs, dtype=np.float64)
            if cost_array.ndim != 1:
                raise ValueError("lifted_costs must be a 1D array")
            if cost_array.shape[0] != uv_array.shape[0]:
                raise ValueError(
                    "lifted_uvs and lifted_costs must have the same length, got "
                    f"lifted_uvs.shape[0]={uv_array.shape[0]}, "
                    f"lifted_costs.shape[0]={cost_array.shape[0]}"
                )
            _add_lifted_edges(
                lifted_graph,
                weights_list,
                uv_array,
                np.ascontiguousarray(cost_array),
                overwrite_existing=overwrite_existing,
            )

        self._base_graph = base_graph
        self._lifted_graph = lifted_graph
        self._n_base_edges = int(base_graph.number_of_edges)
        self._weights = np.ascontiguousarray(np.concatenate(weights_list)) \
            if len(weights_list) > 1 else weights_list[0]
        if initial_labels is None:
            self._labels = np.arange(base_graph.number_of_nodes, dtype=np.uint64)
        else:
            self._labels = _as_node_labels(initial_labels, base_graph)

    @property
    def graph(self) -> UndirectedGraph:
        return self._base_graph

    @property
    def lifted_graph(self) -> UndirectedGraph:
        return self._lifted_graph

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def number_of_base_edges(self) -> int:
        return self._n_base_edges

    @property
    def number_of_lifted_edges(self) -> int:
        return int(self._lifted_graph.number_of_edges) - self._n_base_edges

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @labels.setter
    def labels(self, labels) -> None:
        self._labels = _as_node_labels(labels, self._base_graph)

    def set_labels(self, labels) -> None:
        self.labels = labels

    def reset_labels(self) -> None:
        self._labels = np.arange(self._base_graph.number_of_nodes, dtype=np.uint64)

    def set_cost(
        self,
        u: int,
        v: int,
        weight: float,
        *,
        overwrite: bool = False,
    ) -> tuple[int, bool]:
        """Insert or update a single lifted edge weight.

        Returns ``(edge_id, is_new)`` — the lifted-graph edge id and whether a
        new edge was inserted. If the edge already exists (as a base edge or a
        previously inserted lifted edge), the weight is accumulated unless
        ``overwrite=True``.
        """
        pre = int(self._lifted_graph.number_of_edges)
        edge = int(self._lifted_graph.insert_edge(int(u), int(v)))
        if int(self._lifted_graph.number_of_edges) > pre:
            self._weights = np.concatenate(
                [self._weights, np.asarray([float(weight)], dtype=np.float64)]
            )
            return edge, True
        if overwrite:
            self._weights[edge] = float(weight)
        else:
            self._weights[edge] = self._weights[edge] + float(weight)
        return edge, False

    def energy(self, labels=None) -> float:
        label_array = (
            self._labels if labels is None else _as_node_labels(labels, self._base_graph)
        )
        return float(
            _core._lifted_multicut_energy(self._lifted_graph, self._weights, label_array)
        )


def _add_lifted_edges(
    lifted_graph: UndirectedGraph,
    weights_list: list[np.ndarray],
    lifted_uvs: np.ndarray,
    lifted_costs: np.ndarray,
    *,
    overwrite_existing: bool,
) -> None:
    """Insert lifted edges into an UndirectedGraph and update the weights list.

    Mirrors the C++ ``build_lifted_graph`` semantics: brand-new edges append
    their cost to ``weights_list``; existing edges (duplicates of a base edge
    or of a previously inserted lifted edge) either overwrite or accumulate
    their weight in place. ``weights_list`` is expected to hold the current
    weights array as its single element on entry; on exit it contains either
    one or two ndarray entries (the existing weights plus the new tail).
    """
    if lifted_uvs.shape[0] == 0:
        return
    # In-place updates require a single flat working buffer; coalesce first.
    if len(weights_list) > 1:
        weights_list[:] = [np.ascontiguousarray(np.concatenate(weights_list))]

    # Fast path: bulk-insert and detect uniqueness from the row count delta.
    # For the typical case — ``lifted_uvs`` produced by
    # ``lifted_edges_from_affinities`` or by the BFS constructor — every row
    # is a brand-new edge, so the delta equals the input length and we can
    # append the weights array directly. No ``find_edges`` calls are needed.
    if not overwrite_existing:
        pre_count = int(lifted_graph.number_of_edges)
        lifted_graph.insert_edges(lifted_uvs)
        post_count = int(lifted_graph.number_of_edges)
        n_new = post_count - pre_count

        if n_new == lifted_uvs.shape[0]:
            weights_list.append(
                np.ascontiguousarray(lifted_costs.astype(np.float64, copy=False))
            )
            return

        # Some rows collided with existing edges or with each other. Use
        # find_edges to recover the per-row edge id (insertion is already
        # done; this is just a lookup).
        edge_ids = np.asarray(lifted_graph.find_edges(lifted_uvs))
        lifted_costs_f64 = lifted_costs.astype(np.float64, copy=False)
        working = weights_list[0]

        collision_mask = edge_ids < pre_count
        if collision_mask.any():
            np.add.at(
                working,
                edge_ids[collision_mask].astype(np.intp, copy=False),
                lifted_costs_f64[collision_mask],
            )

        new_mask = ~collision_mask
        if new_mask.any():
            slot = (edge_ids[new_mask] - pre_count).astype(np.int64, copy=False)
            new_weights = np.bincount(
                slot, weights=lifted_costs_f64[new_mask], minlength=n_new
            ).astype(np.float64, copy=False)
        else:
            new_weights = np.zeros(n_new, dtype=np.float64)

        weights_list[0] = working
        if n_new > 0:
            weights_list.append(new_weights)
        return

    # Slow path: per-row Python loop for ``overwrite_existing=True`` (rare).
    # Order-sensitive last-write-wins semantics on collisions.
    working = weights_list[0]
    new_costs: list[float] = []
    for index in range(lifted_uvs.shape[0]):
        u = int(lifted_uvs[index, 0])
        v = int(lifted_uvs[index, 1])
        weight = float(lifted_costs[index])
        pre = int(lifted_graph.number_of_edges)
        edge = int(lifted_graph.insert_edge(u, v))
        if int(lifted_graph.number_of_edges) > pre:
            new_costs.append(weight)
        else:
            working[edge] = weight
    if new_costs:
        weights_list[0] = working
        weights_list.append(np.asarray(new_costs, dtype=np.float64))
    else:
        weights_list[0] = working


class LiftedMulticutSolver(ABC):
    """Base class for lifted multicut solvers."""

    @abstractmethod
    def optimize(self, objective: LiftedMulticutObjective) -> np.ndarray:
        """Optimize ``objective`` and return the node labeling."""


class LiftedGreedyAdditiveMulticut(LiftedMulticutSolver):
    def __init__(
        self,
        *,
        weight_stop: float = 0.0,
        node_num_stop: float = -1.0,
        add_noise: bool = False,
        seed: int = 42,
        sigma: float = 1.0,
    ):
        self.weight_stop = float(weight_stop)
        self.node_num_stop = float(node_num_stop)
        self.add_noise = bool(add_noise)
        self.seed = int(seed)
        self.sigma = float(sigma)

    def optimize(self, objective: LiftedMulticutObjective) -> np.ndarray:
        labels = _core._lifted_multicut_greedy_additive(
            objective.lifted_graph,
            objective.weights,
            objective.number_of_base_edges,
            self.weight_stop,
            self.node_num_stop,
            self.add_noise,
            self.seed,
            self.sigma,
        )
        objective.labels = labels
        return objective.labels

    def _build_cpp_sub_solver(self):
        return _core._GreedyAdditiveLiftedMulticutSubSolver(
            weight_stop=self.weight_stop,
            node_num_stop=self.node_num_stop,
            add_noise=self.add_noise,
            seed=self.seed,
            sigma=self.sigma,
        )


class LiftedKernighanLinMulticut(LiftedMulticutSolver):
    def __init__(
        self,
        *,
        number_of_outer_iterations: int = 100,
        epsilon: float = 1.0e-6,
    ):
        self.number_of_outer_iterations = int(number_of_outer_iterations)
        if self.number_of_outer_iterations < 0:
            raise ValueError("number_of_outer_iterations must be non-negative")
        self.epsilon = float(epsilon)

    def optimize(self, objective: LiftedMulticutObjective) -> np.ndarray:
        initial_labels = objective.labels
        if np.array_equal(
            initial_labels,
            np.arange(objective.graph.number_of_nodes, dtype=np.uint64),
        ):
            initial_labels = _core._lifted_multicut_greedy_additive(
                objective.lifted_graph,
                objective.weights,
                objective.number_of_base_edges,
                0.0,
                -1.0,
                False,
                42,
                1.0,
            )
        labels = _core._lifted_multicut_kernighan_lin(
            objective.graph,
            objective.lifted_graph,
            objective.weights,
            objective.number_of_base_edges,
            initial_labels,
            self.number_of_outer_iterations,
            self.epsilon,
        )
        objective.labels = labels
        return objective.labels

    def _build_cpp_sub_solver(self):
        return _core._KernighanLinLiftedMulticutSubSolver(
            number_of_outer_iterations=self.number_of_outer_iterations,
            epsilon=self.epsilon,
        )


class LiftedChainedSolvers(LiftedMulticutSolver):
    """Chain of lifted multicut solvers run in sequence on the same objective.

    Each solver's output labeling is fed to the next via the shared
    :class:`LiftedMulticutObjective`. Typical use: ``[LiftedGreedyAdditiveMulticut(),
    LiftedKernighanLinMulticut(...)]`` for a fast warm-start followed by a
    local refinement.
    """

    def __init__(self, solvers):
        self.solvers = list(solvers)
        if len(self.solvers) == 0:
            raise ValueError("solvers must contain at least one solver")
        if not all(isinstance(solver, LiftedMulticutSolver) for solver in self.solvers):
            raise TypeError("all solvers must inherit from LiftedMulticutSolver")

    def optimize(self, objective: LiftedMulticutObjective) -> np.ndarray:
        labels = objective.labels
        for solver in self.solvers:
            labels = solver.optimize(objective)
        return labels


def region_adjacency_graph(
    labels: np.ndarray,
    *,
    number_of_threads: int = 0,
) -> RegionAdjacencyGraph:
    """Build a region adjacency graph from a 2D or 3D label image.

    Nodes correspond to label ids from ``0`` to ``labels.max()``. Undirected
    edges connect different labels that touch along the pixel or voxel grid's
    direct neighborhood. The edge ids are deterministic and sorted
    lexicographically by their endpoint ids.
    """
    array = np.asarray(labels)
    if array.ndim not in (2, 3):
        raise ValueError(f"labels must be a 2D or 3D array, got ndim={array.ndim}")
    if any(size == 0 for size in array.shape):
        raise ValueError("labels must not have empty dimensions")

    dtype = array.dtype
    try:
        run = _REGION_ADJACENCY_GRAPH_BY_DTYPE[dtype]
    except KeyError as error:
        supported = ", ".join(
            str(dtype) for dtype in _REGION_ADJACENCY_GRAPH_BY_DTYPE
        )
        raise TypeError(
            f"labels must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    number_of_threads = int(number_of_threads)
    if number_of_threads < 0:
        raise ValueError("number_of_threads must be non-negative")
    return run(np.ascontiguousarray(array), number_of_threads)


def edge_map_features(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    edge_map: np.ndarray,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute mean and size features for edge-map values on RAG boundaries."""
    return _accumulate_edge_map_features(
        rag,
        labels,
        edge_map,
        compute_complex_features=False,
        number_of_threads=number_of_threads,
    )


def edge_map_features_complex(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    edge_map: np.ndarray,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute complex edge-map features on RAG boundaries.

    The output columns are given by ``COMPLEX_EDGE_FEATURE_NAMES``.
    """
    return _accumulate_edge_map_features(
        rag,
        labels,
        edge_map,
        compute_complex_features=True,
        number_of_threads=number_of_threads,
    )


def affinity_features(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute mean and size features for affinity links crossing RAG edges."""
    return _accumulate_affinity_features(
        rag,
        labels,
        affinities,
        offsets,
        compute_complex_features=False,
        number_of_threads=number_of_threads,
    )


def affinity_features_complex(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute complex affinity features for links crossing RAG edges.

    The output columns are given by ``COMPLEX_EDGE_FEATURE_NAMES``.
    """
    return _accumulate_affinity_features(
        rag,
        labels,
        affinities,
        offsets,
        compute_complex_features=True,
        number_of_threads=number_of_threads,
    )


def lifted_edges_from_affinities(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    offsets,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Discover lifted edges implied by long-range affinity offsets.

    Walks every grid coordinate together with each long-range offset (1-hop
    offsets are skipped automatically). When the labels at ``(p, p + offset)``
    differ and ``(labels[p], labels[p + offset])`` is not already a local
    edge of ``rag``, the pair is recorded as a lifted edge.

    Parameters
    ----------
    rag:
        :class:`RegionAdjacencyGraph` built from ``labels``.
    labels:
        2D or 3D label array. Supported dtypes: ``uint32``, ``uint64``,
        ``int32``, ``int64``.
    offsets:
        Sequence of per-channel offsets. Each offset must have length equal
        to ``labels.ndim``. Offsets with L1 norm ``<= 1`` are skipped, so
        callers can pass the full offset list of an affinity volume without
        pre-filtering.

    Returns
    -------
    np.ndarray
        ``(n_lifted, 2)`` ``uint64`` array of ``(u, v)`` pairs with
        ``u < v``, sorted lexicographically.
    """
    label_array = _normalize_labels(labels)
    if tuple(int(size) for size in rag.shape) != label_array.shape:
        raise ValueError(
            "rag shape must match labels shape, got "
            f"rag shape={tuple(rag.shape)}, labels shape={label_array.shape}"
        )

    normalized_offsets = [tuple(int(value) for value in offset) for offset in offsets]
    if any(len(offset) != label_array.ndim for offset in normalized_offsets):
        raise ValueError("each offset must have length matching labels ndim")

    run = _LIFTED_EDGES_FROM_AFFINITIES_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        normalized_offsets,
        _normalize_number_of_threads(number_of_threads),
    )


def lifted_affinity_features(
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    lifted_uvs,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute mean and size features for affinity links across lifted edges.

    Affinity values at pixel pairs ``(p, p + offset)`` whose labels match a
    row of ``lifted_uvs`` are binned into that lifted edge. Pixel pairs that
    fall on a non-lifted edge (or no edge at all) are silently skipped, so
    a local edge that is also reachable by a long-range offset is not
    contaminated by long-range affinities.

    1-hop offsets are skipped automatically.

    The returned array has shape ``(len(lifted_uvs), 2)`` with columns
    ``SIMPLE_EDGE_FEATURE_NAMES`` (``mean``, ``size``).
    """
    return _accumulate_lifted_affinity_features(
        labels,
        affinities,
        offsets,
        lifted_uvs,
        compute_complex_features=False,
        number_of_threads=number_of_threads,
    )


def lifted_affinity_features_complex(
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    lifted_uvs,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Complex affinity features for links across lifted edges.

    Output columns: ``COMPLEX_EDGE_FEATURE_NAMES``.
    """
    return _accumulate_lifted_affinity_features(
        labels,
        affinities,
        offsets,
        lifted_uvs,
        compute_complex_features=True,
        number_of_threads=number_of_threads,
    )


def project_node_labels_to_pixels(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    node_labels,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Map RAG node labels back to a pixel-wise segmentation.

    ``labels`` is the over-segmentation used to construct ``rag``. Each pixel
    value is interpreted as a RAG node id and replaced by the corresponding
    entry in the 1D ``node_labels`` array. The returned segmentation has the
    same shape as ``labels`` and dtype ``uint64``.
    """
    label_array = _normalize_labels(labels)
    if tuple(int(size) for size in rag.shape) != label_array.shape:
        raise ValueError(
            "rag shape must match labels shape, got "
            f"rag shape={tuple(rag.shape)}, labels shape={label_array.shape}"
        )

    node_label_array = np.asarray(node_labels, dtype=np.uint64)
    if node_label_array.ndim != 1:
        raise ValueError("node_labels must be a 1D array")
    if node_label_array.shape[0] != rag.number_of_nodes:
        raise ValueError("node_labels length must match rag number_of_nodes")

    run = _PROJECT_NODE_LABELS_TO_PIXELS_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        np.ascontiguousarray(node_label_array),
        _normalize_number_of_threads(number_of_threads),
    )


def _accumulate_edge_map_features(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    edge_map: np.ndarray,
    *,
    compute_complex_features: bool,
    number_of_threads: int,
) -> np.ndarray:
    label_array = _normalize_labels(labels)
    edge_map_array = np.asarray(edge_map, dtype=np.float64)
    if edge_map_array.shape != label_array.shape:
        raise ValueError(
            "edge_map shape must match labels shape, got "
            f"edge_map shape={edge_map_array.shape}, labels shape={label_array.shape}"
        )
    run = _EDGE_MAP_FEATURES_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        np.ascontiguousarray(edge_map_array),
        bool(compute_complex_features),
        _normalize_number_of_threads(number_of_threads),
    )


def _accumulate_affinity_features(
    rag: RegionAdjacencyGraph,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    *,
    compute_complex_features: bool,
    number_of_threads: int,
) -> np.ndarray:
    label_array = _normalize_labels(labels)
    affinity_array = np.asarray(affinities, dtype=np.float64)
    if affinity_array.ndim != label_array.ndim + 1:
        raise ValueError("affinities must have shape (channels, *labels.shape)")
    if affinity_array.shape[1:] != label_array.shape:
        raise ValueError(
            "affinities spatial shape must match labels shape, got "
            f"affinities shape={affinity_array.shape}, labels shape={label_array.shape}"
        )

    normalized_offsets = [tuple(int(value) for value in offset) for offset in offsets]
    if len(normalized_offsets) != affinity_array.shape[0]:
        raise ValueError(
            "offsets length must match affinities channel count, got "
            f"offsets length={len(normalized_offsets)}, channels={affinity_array.shape[0]}"
        )
    if any(len(offset) != label_array.ndim for offset in normalized_offsets):
        raise ValueError("each offset must have length matching labels ndim")

    run = _AFFINITY_FEATURES_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        np.ascontiguousarray(affinity_array),
        normalized_offsets,
        bool(compute_complex_features),
        _normalize_number_of_threads(number_of_threads),
    )


def _accumulate_lifted_affinity_features(
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    lifted_uvs,
    *,
    compute_complex_features: bool,
    number_of_threads: int,
) -> np.ndarray:
    label_array = _normalize_labels(labels)
    affinity_array = np.asarray(affinities, dtype=np.float64)
    if affinity_array.ndim != label_array.ndim + 1:
        raise ValueError("affinities must have shape (channels, *labels.shape)")
    if affinity_array.shape[1:] != label_array.shape:
        raise ValueError(
            "affinities spatial shape must match labels shape, got "
            f"affinities shape={affinity_array.shape}, labels shape={label_array.shape}"
        )

    normalized_offsets = [tuple(int(value) for value in offset) for offset in offsets]
    if len(normalized_offsets) != affinity_array.shape[0]:
        raise ValueError(
            "offsets length must match affinities channel count, got "
            f"offsets length={len(normalized_offsets)}, channels={affinity_array.shape[0]}"
        )
    if any(len(offset) != label_array.ndim for offset in normalized_offsets):
        raise ValueError("each offset must have length matching labels ndim")

    lifted_uv_array = _as_uv_array(lifted_uvs, "lifted_uvs")

    run = _LIFTED_AFFINITY_FEATURES_BY_DTYPE[label_array.dtype]
    return run(
        label_array,
        np.ascontiguousarray(affinity_array),
        normalized_offsets,
        lifted_uv_array,
        bool(compute_complex_features),
        _normalize_number_of_threads(number_of_threads),
    )


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim not in (2, 3):
        raise ValueError(f"labels must be a 2D or 3D array, got ndim={array.ndim}")
    try:
        _REGION_ADJACENCY_GRAPH_BY_DTYPE[array.dtype]
    except KeyError as error:
        supported = ", ".join(
            str(dtype) for dtype in _REGION_ADJACENCY_GRAPH_BY_DTYPE
        )
        raise TypeError(
            f"labels must have one of dtypes ({supported}), got dtype={array.dtype}"
        ) from error
    return np.ascontiguousarray(array)


def _normalize_number_of_threads(number_of_threads: int) -> int:
    number_of_threads = int(number_of_threads)
    if number_of_threads < 0:
        raise ValueError("number_of_threads must be non-negative")
    return number_of_threads


__all__ = [
    "ChainedMulticutSolvers",
    "COMPLEX_EDGE_FEATURE_NAMES",
    "DEFAULT_EXTERNAL_MULTICUT_PROBLEM_PATH",
    "EXTERNAL_MULTICUT_PROBLEM_URL",
    "FusionMoveMulticut",
    "GreedyAdditiveMulticut",
    "GreedyAdditiveProposalGenerator",
    "GreedyFixationMulticut",
    "KernighanLinMulticut",
    "LiftedChainedSolvers",
    "LiftedGreedyAdditiveMulticut",
    "LiftedKernighanLinMulticut",
    "LiftedMulticutObjective",
    "LiftedMulticutProblem",
    "LiftedMulticutSolver",
    "MulticutDecomposer",
    "MulticutObjective",
    "MulticutSolver",
    "ProposalGenerator",
    "RegionAdjacencyGraph",
    "SIMPLE_EDGE_FEATURE_NAMES",
    "UndirectedGraph",
    "WatershedProposalGenerator",
    "affinity_features",
    "affinity_features_complex",
    "breadth_first_search",
    "connected_components",
    "edge_map_features",
    "edge_map_features_complex",
    "edge_weighted_watershed",
    "lifted_affinity_features",
    "lifted_affinity_features_complex",
    "lifted_edges_from_affinities",
    "external_multicut_problem_path",
    "lifted_multicut_problem_path",
    "load_external_multicut_problem",
    "load_external_multicut_problem_data",
    "load_lifted_multicut_problem",
    "load_multicut_problem",
    "load_multicut_problem_data",
    "multicut_problem_path",
    "project_node_labels_to_pixels",
    "region_adjacency_graph",
    "undirected_graph",
]
