from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np


def parser(description: str) -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description=description)
    arg_parser.add_argument(
        "--path",
        default=None,
        help="Path to the external multicut problem. Defaults to the package cache.",
    )
    arg_parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repeats per implementation.",
    )
    arg_parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for solvers that support it.",
    )
    arg_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Download timeout in seconds if the external problem is not cached.",
    )
    arg_parser.add_argument(
        "--energy-bound",
        type=float,
        default=-76900.0,
        help="Maximum accepted energy for both implementations.",
    )
    return arg_parser


def load_problem(path: str | None, *, timeout: float):
    import bioimage_cpp as bic
    import nifty

    uv_ids, costs = bic.graph.load_external_multicut_problem_data(
        path,
        timeout=timeout,
    )
    bic_graph = bic.graph.UndirectedGraph.from_edges(int(uv_ids.max()) + 1, uv_ids)
    nifty_graph = nifty.graph.undirectedGraph(int(uv_ids.max()) + 1)
    nifty_graph.insertEdges(uv_ids)
    return bic_graph, nifty_graph, costs


def time_call(function: Callable[[], np.ndarray], repeats: int):
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        timings.append(perf_counter() - start)
    assert result is not None
    return timings, result


def optimize_bic_solver(make_bic_solver, objective):
    objective.reset_labels()
    return make_bic_solver().optimize(objective)


def bic_energy(graph, costs: np.ndarray, labels: np.ndarray) -> float:
    import bioimage_cpp as bic

    return bic.graph.MulticutObjective(graph, costs).energy(labels)


def nifty_energy(graph, costs: np.ndarray, labels: np.ndarray) -> float:
    import nifty.graph.opt.multicut as nmc

    return float(nmc.multicutObjective(graph, costs).evalNodeLabels(labels))


def run_comparison(
    name: str,
    make_bic_solver,
    make_nifty_solver,
    args: argparse.Namespace,
) -> dict[str, float]:
    import bioimage_cpp as bic
    import nifty.graph.opt.multicut as nmc

    bic_graph, nifty_graph, costs = load_problem(args.path, timeout=args.timeout)
    bic_objective = bic.graph.MulticutObjective(bic_graph, costs)
    nifty_objective = nmc.multicutObjective(nifty_graph, costs)

    bic_timings, bic_labels = time_call(
        lambda: optimize_bic_solver(make_bic_solver, bic_objective),
        args.repeats,
    )
    nifty_timings, nifty_labels = time_call(
        lambda: make_nifty_solver(nifty_objective).create(nifty_objective).optimize(),
        args.repeats,
    )

    bic_score = bic_energy(bic_graph, costs, bic_labels)
    nifty_score = nifty_energy(nifty_graph, costs, nifty_labels)
    if bic_score > args.energy_bound:
        raise AssertionError(
            f"bioimage-cpp {name} energy {bic_score:.6f} exceeds bound {args.energy_bound:.6f}"
        )
    if nifty_score > args.energy_bound:
        raise AssertionError(
            f"nifty {name} energy {nifty_score:.6f} exceeds bound {args.energy_bound:.6f}"
        )

    result = {
        "bioimage_cpp_energy": bic_score,
        "nifty_energy": nifty_score,
        "energy_difference": bic_score - nifty_score,
        "bioimage_cpp_median_runtime": median(bic_timings),
        "nifty_median_runtime": median(nifty_timings),
    }
    print(f"solver: {name}")
    print(f"nodes: {bic_graph.number_of_nodes}, edges: {bic_graph.number_of_edges}")
    print(f"bioimage-cpp energy: {bic_score:.6f}")
    print(f"nifty energy:        {nifty_score:.6f}")
    print(f"energy difference:   {bic_score - nifty_score:.6f}")
    print(f"bioimage-cpp median runtime [s]: {median(bic_timings):.6f}")
    print(f"nifty median runtime [s]:        {median(nifty_timings):.6f}")
    return result
