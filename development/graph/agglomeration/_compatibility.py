"""Benchmark scaffolding for comparing bioimage-cpp agglomeration policies
against the corresponding ``nifty.graph.agglo`` implementations.

Loads the external multicut problem (a generic edge list + costs) and
reinterprets the costs as boundary indicators (after a sigmoid) so the
policies have something realistic to chew on. Reports median runtime over
``--repeats`` invocations and partition agreement (variation of information
and adjusted Rand index) between the two implementations.
"""

from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np


def parser(description: str) -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description=description)
    arg_parser.add_argument(
        "--sample",
        default="A",
        choices=["A", "B", "C"],
        help="Multicut problem sample to load.",
    )
    arg_parser.add_argument(
        "--size",
        default="small",
        choices=["small", "medium"],
        help="Multicut problem size to load (small ~ 60k nodes, medium ~ 700k).",
    )
    arg_parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repeats per implementation.",
    )
    arg_parser.add_argument(
        "--num-clusters-stop",
        type=int,
        default=200,
        help="Stop when this many clusters remain (must be > 1 to keep both "
        "implementations from collapsing the whole graph).",
    )
    arg_parser.add_argument(
        "--size-regularizer",
        type=float,
        default=0.5,
        help="Size regulariser exponent for the edge-weighted policies.",
    )
    arg_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for the MALA policy.",
    )
    arg_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Download timeout in seconds if the external problem is not cached.",
    )
    return arg_parser


def load_problem(sample: str = "A", size: str = "small", *, timeout: float = 120.0):
    """Load a multicut problem and derive indicator / weight arrays.

    Returns ``(bic_graph, nifty_graph, indicators, signed_weights, uv_ids)``
    where ``indicators`` are in ``[0, 1]`` (boundary strength) and
    ``signed_weights`` keeps the original multicut sign (positive = attract).
    """
    import bioimage_cpp as bic
    import nifty.graph as ng

    uv_ids, costs = bic.graph.multicut.load_multicut_problem_data(
        sample, size, timeout=timeout
    )
    n_nodes = int(uv_ids.max()) + 1

    bic_graph = bic.graph.UndirectedGraph.from_edges(n_nodes, uv_ids)
    nifty_graph = ng.undirectedGraph(n_nodes)
    nifty_graph.insertEdges(uv_ids)

    # Multicut costs are signed log-odds (positive = attractive, large
    # magnitude = certain). Map to a boundary-strength indicator in [0, 1]
    # via a sigmoid of the negated cost so 'large positive cost' becomes
    # 'small indicator' (weak boundary), matching nifty's convention.
    indicators = 1.0 / (1.0 + np.exp(np.asarray(costs, dtype=np.float64)))
    indicators = np.ascontiguousarray(indicators.astype(np.float64))
    # Signed weights for GASP: keep the multicut sign directly.
    signed_weights = np.ascontiguousarray(np.asarray(costs, dtype=np.float64))
    return bic_graph, nifty_graph, indicators, signed_weights, uv_ids


def time_call(function: Callable[[], np.ndarray], repeats: int):
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        timings.append(perf_counter() - start)
    assert result is not None
    return timings, result


def variation_of_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    labels_a = np.asarray(labels_a).astype(np.int64)
    labels_b = np.asarray(labels_b).astype(np.int64)
    n = labels_a.size
    if n == 0:
        return 0.0
    _, a_inv, a_counts = np.unique(labels_a, return_inverse=True, return_counts=True)
    _, b_inv, b_counts = np.unique(labels_b, return_inverse=True, return_counts=True)
    pa = a_counts / n
    pb = b_counts / n
    contingency = np.zeros((a_counts.size, b_counts.size), dtype=np.float64)
    np.add.at(contingency, (a_inv, b_inv), 1.0)
    contingency /= n
    with np.errstate(divide="ignore", invalid="ignore"):
        ha = -np.sum(pa * np.log(pa, where=pa > 0))
        hb = -np.sum(pb * np.log(pb, where=pb > 0))
        joint = -np.sum(
            contingency * np.log(contingency, where=contingency > 0)
        )
    mutual_info = ha + hb - joint
    return float(2.0 * joint - ha - hb - 2.0 * mutual_info + ha + hb)


def adjusted_rand(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return float("nan")
    return float(adjusted_rand_score(labels_a, labels_b))


def report(
    name: str,
    bic_timings,
    nifty_timings,
    bic_labels,
    nifty_labels,
    n_nodes,
    n_edges,
    *,
    sample: str | None = None,
    size: str | None = None,
):
    vi = variation_of_information(bic_labels, nifty_labels)
    ari = adjusted_rand(bic_labels, nifty_labels)
    bic_clusters = int(np.unique(bic_labels).size)
    nifty_clusters = int(np.unique(nifty_labels).size)
    bic_med = median(bic_timings)
    nifty_med = median(nifty_timings)
    speedup = nifty_med / bic_med if bic_med > 0 else float("nan")
    suffix = ""
    if sample is not None and size is not None:
        suffix = f" [sample {sample} / {size}]"
    print(f"policy: {name}{suffix}")
    print(f"nodes: {n_nodes}, edges: {n_edges}")
    print(f"bioimage_cpp clusters: {bic_clusters}")
    print(f"nifty clusters:        {nifty_clusters}")
    print(f"bioimage_cpp median runtime [s]: {bic_med:.6f}")
    print(f"nifty median runtime [s]:        {nifty_med:.6f}")
    print(f"speedup (nifty / bioimage_cpp):  {speedup:.2f}x")
    print(f"variation of information: {vi:.6f}")
    print(f"adjusted Rand index:      {ari:.6f}")
