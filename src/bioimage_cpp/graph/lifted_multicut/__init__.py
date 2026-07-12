"""Lifted multicut objective and solvers.

The lifted multicut problem extends the standard multicut with a second set
of *lifted* edges that do not have to be present in the base graph but
contribute to the energy via their cut status. This submodule exposes:

- :class:`LiftedMulticutObjective` — base graph + base costs + lifted edges +
  working labeling.
- :class:`LiftedMulticutSolver` and the concrete solvers
  :class:`LiftedGreedyAdditiveMulticut`, :class:`LiftedKernighanLinMulticut`,
  :class:`FusionMoveLiftedMulticut`, :class:`LiftedChainedSolvers`.
- :class:`LiftedMulticutProblem` and loaders for the benchmark lifted multicut
  problem instances.
- :class:`ProposalGenerator`, :class:`WatershedProposalGenerator`,
  :class:`GreedyAdditiveProposalGenerator` re-exported from
  :mod:`bioimage_cpp.graph.multicut` (the lifted fusion-move solver consumes
  them).
- :func:`lifted_edges_from_node_labels` — discover lifted edges by combining
  a BFS neighborhood on the base graph with a per-node label predicate
  (port of ``nifty.distributed.liftedNeighborhoodFromNodeLabels``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .. import _core
from .._external import (
    LiftedMulticutProblem,
    lifted_multicut_problem_path,
    load_lifted_multicut_problem,
)
from .._shared import (
    _as_edge_costs,
    _as_node_labels,
    _as_uv_array,
    _normalize_number_of_threads,
)


_LIFTED_EDGES_FROM_NODE_LABELS_BY_DTYPE = {
    np.dtype("uint32"): _core._lifted_edges_from_node_labels_uint32,
    np.dtype("uint64"): _core._lifted_edges_from_node_labels_uint64,
    np.dtype("int32"): _core._lifted_edges_from_node_labels_int32,
    np.dtype("int64"): _core._lifted_edges_from_node_labels_int64,
}


def lifted_edges_from_node_labels(
    graph,
    node_labels,
    graph_depth: int,
    *,
    mode: str = "all",
    ignore_label: int | None = None,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Discover lifted edges from a BFS neighborhood plus per-node labels.

    For every source node ``u`` the BFS reports each node ``v`` reached within
    ``graph_depth`` hops. The pair ``(u, v)`` (with ``u < v``) becomes a lifted
    edge iff:

    - the BFS hop distance is in ``[2, graph_depth]`` — base-graph edges
      (distance 1) are always excluded;
    - neither ``node_labels[u]`` nor ``node_labels[v]`` equals ``ignore_label``
      (when ``ignore_label`` is not ``None``);
    - the ``mode`` predicate is satisfied: ``'all'`` keeps every pair,
      ``'same'`` keeps pairs with matching labels, ``'different'`` keeps the
      complement.

    Mirrors ``nifty.distributed.liftedNeighborhoodFromNodeLabels`` with the
    following intentional differences: snake-case parameter names,
    ``ignore_label`` defaults to ``None`` (no filtering), and node ``0`` is
    iterated as a source (nifty's distributed variant skips it).

    Parameters
    ----------
    graph:
        :class:`bioimage_cpp.graph.UndirectedGraph` or
        :class:`bioimage_cpp.graph.RegionAdjacencyGraph`.
    node_labels:
        1D array of length ``graph.number_of_nodes``. Supported dtypes:
        ``uint32``, ``uint64``, ``int32``, ``int64``.
    graph_depth:
        Maximum BFS hop distance (inclusive). Must be ``>= 1``;
        ``graph_depth == 1`` returns an empty array because base edges are
        excluded by construction.
    mode:
        ``'all'``, ``'same'``, or ``'different'``.
    ignore_label:
        If set, drop every pair where either endpoint label equals this value.
    number_of_threads:
        ``0`` (default) selects the bioimage-cpp default thread count.

    Returns
    -------
    np.ndarray
        ``(n_lifted, 2)`` ``uint64`` array of ``(u, v)`` pairs with
        ``u < v``, sorted lexicographically.
    """
    if mode not in ("all", "same", "different"):
        raise ValueError(
            f"mode must be one of 'all', 'same', 'different', got {mode!r}"
        )
    depth = int(graph_depth)
    if depth < 1:
        raise ValueError(f"graph_depth must be >= 1, got {depth}")

    label_array = np.ascontiguousarray(np.asarray(node_labels))
    if label_array.ndim != 1:
        raise ValueError(
            f"node_labels must be a 1D array, got ndim={label_array.ndim}"
        )
    if label_array.shape[0] != int(graph.number_of_nodes):
        raise ValueError(
            "node_labels length must match graph number_of_nodes, got "
            f"node_labels length={label_array.shape[0]}, "
            f"number_of_nodes={int(graph.number_of_nodes)}"
        )

    try:
        run = _LIFTED_EDGES_FROM_NODE_LABELS_BY_DTYPE[label_array.dtype]
    except KeyError as error:
        supported = ", ".join(
            str(dtype) for dtype in _LIFTED_EDGES_FROM_NODE_LABELS_BY_DTYPE
        )
        raise TypeError(
            f"node_labels must have one of dtypes ({supported}), got "
            f"dtype={label_array.dtype}"
        ) from error

    ignore_arg = None if ignore_label is None else int(ignore_label)
    return run(
        graph,
        label_array,
        depth,
        mode,
        ignore_arg,
        _normalize_number_of_threads(number_of_threads),
    )
from ..multicut import (
    GreedyAdditiveProposalGenerator,
    ProposalGenerator,
    WatershedProposalGenerator,
)


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
        graph,
        edge_costs,
        *,
        lifted_uvs=None,
        lifted_costs=None,
        bfs_distance: int | None = None,
        overwrite_existing: bool = False,
        initial_labels=None,
    ):
        # Local import to avoid a circular dependency at module-load time
        # (``breadth_first_search`` lives in the parent ``bioimage_cpp.graph``
        # namespace which imports this submodule).
        from .. import breadth_first_search

        # The objective holds a reference to the user's base graph (no
        # defensive copy — the C++ ``Objective`` already keeps a const
        # reference). The user is expected to treat the input graph as
        # immutable while the objective is alive; mutations are visible to
        # the objective and may produce undefined behaviour.
        base_graph = graph
        base_costs = _as_edge_costs(edge_costs, base_graph)

        # Preserve the base graph's edge IDs: base costs are indexed in that
        # order, whereas the sorted bulk constructor is only valid when its
        # input already has lexicographic edge order.
        lifted_graph = base_graph.clone()

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
    def graph(self):
        return self._base_graph

    @property
    def lifted_graph(self):
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
    lifted_graph,
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
    """Greedy additive edge contraction (GAEC) lifted multicut solver.

    Introduced in "An efficient fusion move algorithm for the minimum cost
    lifted multicut problem":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/1939997197/beier_16_efficient.pdf
    """

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
    """Kernighan-Lin lifted multicut solver.

    Introduced in "Efficient decomposition of image and mesh graphs by lifted
    multicuts":
    http://openaccess.thecvf.com/content_iccv_2015/papers/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.pdf
    """

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


class FusionMoveLiftedMulticut(LiftedMulticutSolver):
    """Fusion-move lifted multicut solver.

    Introduced in "An efficient fusion move algorithm for the minimum cost
    lifted multicut problem":
    https://hci.iwr.uni-heidelberg.de/sites/default/files/publications/files/1939997197/beier_16_efficient.pdf

    Iteratively generates proposals via ``proposal_generator`` (which sees the
    *base* graph and base edge costs), fuses them with the current best
    labeling, and accepts improvements. Each fuse contracts the base graph by
    agreement across the proposals, aggregates base + lifted weights onto the
    contracted lifted-multicut subproblem, and dispatches to ``sub_solver``.
    If ``sub_solver`` is omitted, the default sub-solver is
    :class:`LiftedGreedyAdditiveMulticut`.

    If the objective's current labels are the trivial singleton labeling, the
    driver warm-starts with one lifted greedy-additive pass before the proposal
    loop. The best-of safety net guarantees energy never increases across
    iterations.

    Threading: ``number_of_threads > 1`` runs ``number_of_parallel_proposals``
    proposal generators in parallel within each iteration. Each parallel slot
    uses an independent proposal generator with seed ``proposal_generator.seed
    + slot_index``. By default ``number_of_parallel_proposals`` is ``2`` when
    ``number_of_threads == 1`` and ``number_of_threads`` otherwise; pass it
    explicitly to override.
    """

    def __init__(
        self,
        *,
        proposal_generator: ProposalGenerator,
        sub_solver: LiftedMulticutSolver | None = None,
        number_of_iterations: int = 10,
        stop_if_no_improvement: int = 4,
        number_of_threads: int = 1,
        number_of_parallel_proposals: int | None = None,
    ):
        if not isinstance(proposal_generator, ProposalGenerator):
            raise TypeError("proposal_generator must inherit from ProposalGenerator")
        if sub_solver is not None and not isinstance(sub_solver, LiftedMulticutSolver):
            raise TypeError("sub_solver must inherit from LiftedMulticutSolver")
        if sub_solver is not None and not hasattr(sub_solver, "_build_cpp_sub_solver"):
            raise TypeError(
                "sub_solver must be a built-in lifted multicut solver "
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

    def optimize(self, objective: LiftedMulticutObjective) -> np.ndarray:
        n_base = objective.number_of_base_edges
        # The base costs back the proposal generators (the lifted weights
        # cannot drive base-graph contraction or watershed segmentation).
        base_costs = np.ascontiguousarray(objective.weights[:n_base])
        cpp_pgens = [
            self.proposal_generator._build_for_thread(
                objective.graph, base_costs, slot
            )
            for slot in range(self.number_of_parallel_proposals)
        ]
        cpp_sub_solver = (
            None if self.sub_solver is None else self.sub_solver._build_cpp_sub_solver()
        )
        labels = _core._lifted_multicut_fusion_move(
            objective.graph,
            objective.lifted_graph,
            objective.weights,
            n_base,
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


__all__ = [
    "FusionMoveLiftedMulticut",
    "GreedyAdditiveProposalGenerator",
    "LiftedChainedSolvers",
    "LiftedGreedyAdditiveMulticut",
    "LiftedKernighanLinMulticut",
    "LiftedMulticutObjective",
    "LiftedMulticutProblem",
    "LiftedMulticutSolver",
    "ProposalGenerator",
    "WatershedProposalGenerator",
    "lifted_edges_from_node_labels",
    "lifted_multicut_problem_path",
    "load_lifted_multicut_problem",
]
