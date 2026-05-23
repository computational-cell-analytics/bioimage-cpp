"""Hierarchical agglomerative cluster policies on undirected graphs.

Equivalent of ``nifty.graph.agglo`` cluster policies. Each class encapsulates
the priority computation, merge rule, and stopping criterion of one
agglomeration scheme; calling :meth:`optimize` runs the heap-driven
contraction loop on the supplied graph and returns dense node labels.

All policies operate on an :class:`bioimage_cpp.graph.UndirectedGraph` or a
subclass (``RegionAdjacencyGraph``, ``GridGraph2D``/``GridGraph3D``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .. import _core
from ._shared import (
    _as_1d_array,
    _resolve_weight_dtype,
)


_EDGE_WEIGHTED_BY_DTYPE = {
    np.dtype("float32"): _core._agglo_edge_weighted_float32,
    np.dtype("float64"): _core._agglo_edge_weighted_float64,
}

_NODE_AND_EDGE_WEIGHTED_BY_DTYPE = {
    np.dtype("float32"): _core._agglo_node_and_edge_weighted_float32,
    np.dtype("float64"): _core._agglo_node_and_edge_weighted_float64,
}

_MALA_BY_DTYPE = {
    np.dtype("float32"): _core._agglo_mala_float32,
    np.dtype("float64"): _core._agglo_mala_float64,
}

_GASP_BY_DTYPE = {
    np.dtype("float32"): _core._agglo_gasp_float32,
    np.dtype("float64"): _core._agglo_gasp_float64,
}


class ClusterPolicy(ABC):
    """Abstract base for agglomerative cluster policies."""

    @abstractmethod
    def optimize(self, graph, *args, **kwargs) -> np.ndarray:
        """Run the agglomeration on ``graph`` and return dense node labels."""


def _ensure_edge_array(values, name, n_edges, dtype):
    if values is None:
        return np.ones(int(n_edges), dtype=dtype)
    return _as_1d_array(values, dtype, name, int(n_edges))


def _ensure_node_array(values, name, n_nodes, dtype):
    if values is None:
        return np.ones(int(n_nodes), dtype=dtype)
    return _as_1d_array(values, dtype, name, int(n_nodes))


class EdgeWeightedClusterPolicy(ClusterPolicy):
    """Hierarchical edge-weighted agglomerative clustering.

    Equivalent of ``nifty.graph.agglo.edgeWeightedClusterPolicy``. The heap
    priority of an edge is ``edge_indicator * sFac`` where
    ``sFac = 2 / (1 / sizeU ** sr + 1 / sizeV ** sr)`` is a harmonic-mean
    size regulariser. Each merge combines edge indicators by their size
    -weighted average and adds the node sizes.

    Parameters
    ----------
    num_clusters_stop:
        Stop when the number of remaining clusters reaches this value.
    size_regularizer:
        Exponent ``sr`` controlling the harmonic-mean size factor. Set to
        ``0.0`` to disable size regularisation entirely (priority becomes
        the raw edge indicator).
    """

    def __init__(self, *, num_clusters_stop: int = 1, size_regularizer: float = 1.0):
        self.num_clusters_stop = int(num_clusters_stop)
        self.size_regularizer = float(size_regularizer)

    def optimize(
        self,
        graph,
        edge_indicators,
        *,
        edge_sizes=None,
        node_sizes=None,
    ) -> np.ndarray:
        indicator_array = _resolve_weight_dtype(edge_indicators, "edge_indicators")
        dtype = indicator_array.dtype
        indicator_array = _as_1d_array(
            indicator_array, dtype, "edge_indicators", int(graph.number_of_edges)
        )
        edge_size_array = _ensure_edge_array(
            edge_sizes, "edge_sizes", graph.number_of_edges, dtype
        )
        node_size_array = _ensure_node_array(
            node_sizes, "node_sizes", graph.number_of_nodes, dtype
        )
        run = _EDGE_WEIGHTED_BY_DTYPE[dtype]
        return run(
            graph,
            indicator_array,
            edge_size_array,
            node_size_array,
            int(self.num_clusters_stop),
            float(self.size_regularizer),
        )


class NodeAndEdgeWeightedClusterPolicy(ClusterPolicy):
    """Agglomeration blending edge indicators with a node-feature distance.

    Equivalent of ``nifty.graph.agglo.nodeAndEdgeWeightedClusterPolicy``.
    The priority is ``(beta * ||featU - featV|| + (1 - beta) * indicator)
    * sFac``. Node features aggregate as a size-weighted mean on merge.

    Parameters
    ----------
    num_clusters_stop:
        Stop when this many clusters remain.
    size_regularizer:
        Exponent of the harmonic-mean size factor (see
        :class:`EdgeWeightedClusterPolicy`).
    beta:
        Blend factor in ``[0, 1]``. ``beta=0`` reproduces
        :class:`EdgeWeightedClusterPolicy`; ``beta=1`` is pure feature
        distance.
    """

    def __init__(
        self,
        *,
        num_clusters_stop: int = 1,
        size_regularizer: float = 1.0,
        beta: float = 0.5,
    ):
        self.num_clusters_stop = int(num_clusters_stop)
        self.size_regularizer = float(size_regularizer)
        self.beta = float(beta)

    def optimize(
        self,
        graph,
        edge_indicators,
        node_features,
        *,
        edge_sizes=None,
        node_sizes=None,
    ) -> np.ndarray:
        indicator_array = _resolve_weight_dtype(edge_indicators, "edge_indicators")
        feature_array = _resolve_weight_dtype(node_features, "node_features")
        if indicator_array.dtype != feature_array.dtype:
            indicator_array = indicator_array.astype(np.float64, copy=False)
            feature_array = feature_array.astype(np.float64, copy=False)
        dtype = indicator_array.dtype
        indicator_array = _as_1d_array(
            indicator_array, dtype, "edge_indicators", int(graph.number_of_edges)
        )
        edge_size_array = _ensure_edge_array(
            edge_sizes, "edge_sizes", graph.number_of_edges, dtype
        )
        node_size_array = _ensure_node_array(
            node_sizes, "node_sizes", graph.number_of_nodes, dtype
        )
        feature_array = np.ascontiguousarray(feature_array)
        if feature_array.ndim != 2 or feature_array.shape[0] != int(graph.number_of_nodes):
            raise ValueError(
                "node_features must have shape (number_of_nodes, n_channels), got "
                f"shape={feature_array.shape}, number_of_nodes={int(graph.number_of_nodes)}"
            )
        run = _NODE_AND_EDGE_WEIGHTED_BY_DTYPE[dtype]
        return run(
            graph,
            indicator_array,
            edge_size_array,
            node_size_array,
            feature_array,
            int(self.num_clusters_stop),
            float(self.size_regularizer),
            float(self.beta),
        )


class MalaClusterPolicy(ClusterPolicy):
    """Histogram-based MALA cluster policy.

    Equivalent of ``nifty.graph.agglo.malaClusterPolicy``. Each edge holds a
    running histogram of its indicator values; the heap priority is the
    histogram's median, and the agglomeration terminates when the next
    candidate edge would exceed ``threshold``.

    Parameters
    ----------
    num_bins:
        Number of histogram bins covering ``[bin_min, bin_max]``.
    bin_min, bin_max:
        Range covered by the histogram. Values outside the range fall into
        the boundary bins.
    num_clusters_stop:
        Stop when at most this many clusters remain. ``0`` disables this
        criterion.
    num_edges_stop:
        Stop when at most this many active edges remain. ``0`` disables
        this criterion.
    threshold:
        Stop when the heap-top priority (the running median of an edge)
        first reaches ``threshold``.
    """

    def __init__(
        self,
        *,
        num_bins: int = 40,
        bin_min: float = 0.0,
        bin_max: float = 1.0,
        num_clusters_stop: int = 1,
        num_edges_stop: int = 0,
        threshold: float = 0.5,
    ):
        self.num_bins = int(num_bins)
        self.bin_min = float(bin_min)
        self.bin_max = float(bin_max)
        self.num_clusters_stop = int(num_clusters_stop)
        self.num_edges_stop = int(num_edges_stop)
        self.threshold = float(threshold)

    def optimize(self, graph, edge_indicators) -> np.ndarray:
        indicator_array = _resolve_weight_dtype(edge_indicators, "edge_indicators")
        dtype = indicator_array.dtype
        indicator_array = _as_1d_array(
            indicator_array, dtype, "edge_indicators", int(graph.number_of_edges)
        )
        run = _MALA_BY_DTYPE[dtype]
        return run(
            graph,
            indicator_array,
            int(self.num_bins),
            float(self.bin_min),
            float(self.bin_max),
            int(self.num_clusters_stop),
            int(self.num_edges_stop),
            float(self.threshold),
        )


class GaspClusterPolicy(ClusterPolicy):
    """GASP signed-graph agglomerative clustering (Bailoni et al.).

    Equivalent of nifty's ``gaspClusterPolicy``. Edge weights are signed
    (positive = attractive, negative = repulsive); the heap is ordered by
    ``|weight|``. The selected ``linkage`` controls how parallel edges
    combine on merge.

    Parameters
    ----------
    num_clusters_stop:
        Stop when at most this many clusters remain.
    linkage:
        One of ``"sum"``, ``"mean"``, ``"max"``, ``"min"``, ``"abs_max"``,
        or ``"mutex_watershed"``. The ``mutex_watershed`` linkage treats a
        negative heap-top weight as a cannot-link constraint (matching the
        mutex-watershed algorithm on a single edge list).

    Notes
    -----
    The optional ``is_mergeable`` mask, when supplied to :meth:`optimize`,
    marks edges that may never trigger a merge; those edges are processed
    in priority order to install permanent cannot-link constraints
    between the clusters they connect.
    """

    _LINKAGE = {
        "sum": 0,
        "mean": 1,
        "max": 2,
        "min": 3,
        "abs_max": 4,
        "mutex_watershed": 5,
    }

    def __init__(self, *, num_clusters_stop: int = 1, linkage: str = "mean"):
        self.num_clusters_stop = int(num_clusters_stop)
        if linkage not in self._LINKAGE:
            raise ValueError(
                f"linkage must be one of {sorted(self._LINKAGE)!r}, got {linkage!r}"
            )
        self.linkage = linkage

    def optimize(
        self,
        graph,
        edge_weights,
        *,
        edge_sizes=None,
        is_mergeable=None,
    ) -> np.ndarray:
        weight_array = _resolve_weight_dtype(edge_weights, "edge_weights")
        dtype = weight_array.dtype
        weight_array = _as_1d_array(
            weight_array, dtype, "edge_weights", int(graph.number_of_edges)
        )
        edge_size_array = _ensure_edge_array(
            edge_sizes, "edge_sizes", graph.number_of_edges, dtype
        )
        if is_mergeable is None:
            mergeable_array = np.empty(0, dtype=np.uint8)
        else:
            mergeable_array = np.asarray(is_mergeable)
            if mergeable_array.dtype != np.dtype("bool") and not np.issubdtype(
                mergeable_array.dtype, np.integer
            ):
                raise TypeError(
                    "is_mergeable must have a boolean or integer dtype, got "
                    f"dtype={mergeable_array.dtype}"
                )
            mergeable_array = _as_1d_array(
                mergeable_array.astype(np.uint8, copy=False),
                np.uint8,
                "is_mergeable",
                int(graph.number_of_edges),
            )
        run = _GASP_BY_DTYPE[dtype]
        return run(
            graph,
            weight_array,
            edge_size_array,
            mergeable_array,
            int(self.num_clusters_stop),
            int(self._LINKAGE[self.linkage]),
        )


__all__ = [
    "ClusterPolicy",
    "EdgeWeightedClusterPolicy",
    "GaspClusterPolicy",
    "MalaClusterPolicy",
    "NodeAndEdgeWeightedClusterPolicy",
]
