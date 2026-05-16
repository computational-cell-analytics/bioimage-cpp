from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np


def parser(description: str) -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description=description)
    arg_parser.add_argument(
        "--size",
        choices=("2d", "3d", "grid"),
        default="3d",
        help="Lifted multicut problem instance to load (default: 3d).",
    )
    arg_parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repeats per implementation.",
    )
    arg_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Download timeout in seconds if the lifted problem is not cached.",
    )
    arg_parser.add_argument(
        "--energy-bound",
        type=float,
        default=None,
        help=(
            "Optional maximum accepted energy for both implementations. If "
            "omitted, energies are reported but not asserted."
        ),
    )
    return arg_parser


def load_problem(size: str, *, timeout: float):
    import bioimage_cpp as bic
    import nifty
    import nifty.graph.opt.lifted_multicut as nlmc

    problem = bic.graph.load_lifted_multicut_problem(size, timeout=timeout)

    bic_graph = bic.graph.UndirectedGraph.from_edges(
        problem.n_nodes, problem.local_uvs
    )

    nifty_graph = nifty.graph.undirectedGraph(int(problem.n_nodes))
    nifty_graph.insertEdges(problem.local_uvs.astype(np.uint64, copy=False))
    nifty_objective = nlmc.liftedMulticutObjective(nifty_graph)
    nifty_objective.setGraphEdgesCosts(problem.local_costs)
    if problem.lifted_uvs.shape[0] > 0:
        nifty_objective.setCosts(
            problem.lifted_uvs.astype(np.uint64, copy=False),
            problem.lifted_costs.astype(np.float64, copy=False),
        )

    return bic_graph, nifty_objective, problem


def time_call(function: Callable[[], np.ndarray], repeats: int):
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        timings.append(perf_counter() - start)
    assert result is not None
    return timings, result


def optimize_bic_solver(make_bic_solver, bic_graph, problem):
    import bioimage_cpp as bic

    objective = bic.graph.LiftedMulticutObjective(
        bic_graph,
        problem.local_costs,
        lifted_uvs=problem.lifted_uvs,
        lifted_costs=problem.lifted_costs,
    )
    return make_bic_solver().optimize(objective)


def bic_energy(bic_graph, problem, labels: np.ndarray) -> float:
    import bioimage_cpp as bic

    objective = bic.graph.LiftedMulticutObjective(
        bic_graph,
        problem.local_costs,
        lifted_uvs=problem.lifted_uvs,
        lifted_costs=problem.lifted_costs,
    )
    return float(objective.energy(labels))


def nifty_energy(nifty_objective, labels: np.ndarray) -> float:
    return float(nifty_objective.evalNodeLabels(labels.astype(np.uint64, copy=False)))


def run_comparison(
    name: str,
    make_bic_solver,
    make_nifty_solver,
    args: argparse.Namespace,
) -> dict[str, float]:
    bic_graph, nifty_objective, problem = load_problem(args.size, timeout=args.timeout)

    bic_timings, bic_labels = time_call(
        lambda: optimize_bic_solver(make_bic_solver, bic_graph, problem),
        args.repeats,
    )
    nifty_timings, nifty_labels = time_call(
        lambda: make_nifty_solver(nifty_objective).create(nifty_objective).optimize(),
        args.repeats,
    )

    bic_score = bic_energy(bic_graph, problem, bic_labels)
    nifty_score = nifty_energy(nifty_objective, np.asarray(nifty_labels))

    if args.energy_bound is not None:
        if bic_score > args.energy_bound:
            raise AssertionError(
                f"bioimage-cpp {name} energy {bic_score:.6f} exceeds bound "
                f"{args.energy_bound:.6f}"
            )
        if nifty_score > args.energy_bound:
            raise AssertionError(
                f"nifty {name} energy {nifty_score:.6f} exceeds bound "
                f"{args.energy_bound:.6f}"
            )

    result = {
        "bioimage_cpp_energy": bic_score,
        "nifty_energy": nifty_score,
        "energy_difference": bic_score - nifty_score,
        "bioimage_cpp_median_runtime": median(bic_timings),
        "nifty_median_runtime": median(nifty_timings),
    }
    print(f"solver: {name}")
    print(
        f"problem: size={args.size}, nodes={problem.n_nodes}, "
        f"local edges={problem.local_uvs.shape[0]}, "
        f"lifted edges={problem.lifted_uvs.shape[0]}"
    )
    print(f"bioimage-cpp energy: {bic_score:.6f}")
    print(f"nifty energy:        {nifty_score:.6f}")
    print(f"energy difference:   {bic_score - nifty_score:.6f}")
    print(f"bioimage-cpp median runtime [s]: {median(bic_timings):.6f}")
    print(f"nifty median runtime [s]:        {median(nifty_timings):.6f}")
    return result
