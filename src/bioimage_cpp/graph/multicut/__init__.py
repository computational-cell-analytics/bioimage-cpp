"""Multicut objective and solvers on undirected graphs.

Public surface:

- :class:`MulticutObjective` — base graph + edge costs + working labeling.
- :class:`MulticutSolver` — abstract base for solvers; concrete subclasses
  are :class:`GreedyAdditiveMulticut`, :class:`GreedyFixationMulticut`,
  :class:`KernighanLinMulticut`, :class:`FusionMoveMulticut`,
  :class:`ChainedMulticutSolvers`, :class:`MulticutDecomposer`.
- :class:`ProposalGenerator` and the concrete
  :class:`WatershedProposalGenerator`,
  :class:`GreedyAdditiveProposalGenerator` — proposal generators consumed by
  fusion-move solvers (also re-exported from
  :mod:`bioimage_cpp.graph.lifted_multicut`).
- Loaders for the benchmark multicut problem instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .. import _core
from .._external import (
    DEFAULT_EXTERNAL_MULTICUT_PROBLEM_PATH,
    EXTERNAL_MULTICUT_PROBLEM_URL,
    external_multicut_problem_path,
    load_external_multicut_problem,
    load_external_multicut_problem_data,
    load_multicut_problem,
    load_multicut_problem_data,
    multicut_problem_path,
)
from .._shared import (
    _as_edge_costs,
    _as_node_labels,
    _copy_graph,
    _dense_labels,
    _normalize_number_of_threads,
    _subproblem_from_edges,
)


class MulticutObjective:
    """Multicut objective for an undirected graph and edge costs."""

    def __init__(
        self,
        graph,
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
    def graph(self):
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
        graph,
        edge_costs: np.ndarray,
        seed_offset: int,
    ):
        """Construct the underlying C++ proposal generator with a seed offset."""


class WatershedProposalGenerator(ProposalGenerator):
    """Watershed proposal generator (nifty's fusion-move workhorse).

    Per call: add Gaussian noise to edge costs, drop random seeds at the
    endpoints of negative-cost edges, run the edge-weighted watershed.

    ``n_seeds_fraction`` is the target *total seed count*: ``<= 1.0`` is a
    fraction of ``number_of_nodes``, otherwise an absolute count. The
    seeding loop places two seeds per iteration and runs ``n_seeds / 2``
    times, matching nifty's ``WatershedProposalGenerator`` so the same
    parameter value yields the same proposal density on both sides.
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
        # Local import to avoid a circular dependency at module-load time:
        # ``connected_components`` lives in the top-level ``bioimage_cpp.graph``
        # namespace, which itself imports this submodule.
        from .. import connected_components

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


__all__ = [
    "ChainedMulticutSolvers",
    "DEFAULT_EXTERNAL_MULTICUT_PROBLEM_PATH",
    "EXTERNAL_MULTICUT_PROBLEM_URL",
    "FusionMoveMulticut",
    "GreedyAdditiveMulticut",
    "GreedyAdditiveProposalGenerator",
    "GreedyFixationMulticut",
    "KernighanLinMulticut",
    "MulticutDecomposer",
    "MulticutObjective",
    "MulticutSolver",
    "ProposalGenerator",
    "WatershedProposalGenerator",
    "external_multicut_problem_path",
    "load_external_multicut_problem",
    "load_external_multicut_problem_data",
    "load_multicut_problem",
    "load_multicut_problem_data",
    "multicut_problem_path",
]
