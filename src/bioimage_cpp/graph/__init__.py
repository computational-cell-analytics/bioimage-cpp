"""Graph data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .. import _core
from ._external import (
    DEFAULT_EXTERNAL_MULTICUT_PROBLEM_PATH,
    EXTERNAL_MULTICUT_PROBLEM_URL,
    external_multicut_problem_path,
    load_external_multicut_problem,
    load_external_multicut_problem_data,
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


def _copy_graph(graph: UndirectedGraph | RegionAdjacencyGraph) -> UndirectedGraph:
    copied = UndirectedGraph(int(graph.number_of_nodes), int(graph.number_of_edges))
    if graph.number_of_edges:
        copied.insert_edges(graph.uv_ids())
    return copied


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
    "GreedyAdditiveMulticut",
    "GreedyFixationMulticut",
    "KernighanLinMulticut",
    "MulticutDecomposer",
    "MulticutObjective",
    "MulticutSolver",
    "RegionAdjacencyGraph",
    "SIMPLE_EDGE_FEATURE_NAMES",
    "UndirectedGraph",
    "affinity_features",
    "affinity_features_complex",
    "connected_components",
    "edge_map_features",
    "edge_map_features_complex",
    "external_multicut_problem_path",
    "load_external_multicut_problem",
    "load_external_multicut_problem_data",
    "project_node_labels_to_pixels",
    "region_adjacency_graph",
    "undirected_graph",
]
