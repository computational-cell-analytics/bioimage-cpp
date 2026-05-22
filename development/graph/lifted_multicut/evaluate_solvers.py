from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Callable

import numpy as np


PROBLEMS = ("2d", "3d", "grid")


@dataclass(frozen=True)
class SolverConfig:
    make_bic_solver: Callable[[], object]
    make_nifty_factory: Callable[[object], object]


def solver_configs():
    import bioimage_cpp as bic

    # Fair single-threaded comparison: nifty's lifted fusion-move backend is
    # single-threaded, so we pin the bic side to threads=1 and one parallel
    # proposal per iteration (matching nifty's "generate one, fuse" loop) and
    # chain greedy-additive in front of nifty to mirror bic's auto warm-start.
    # Watershed seeding strategy is forced to SEED_FROM_LOCAL on the nifty
    # side because the bic proposal generator only sees the base graph.
    return {
        "lifted_greedy_additive": SolverConfig(
            make_bic_solver=lambda: bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut(),
            make_nifty_factory=lambda objective: objective.liftedMulticutGreedyAdditiveFactory(),
        ),
        "lifted_kernighan_lin": SolverConfig(
            make_bic_solver=lambda: bic.graph.lifted_multicut.LiftedKernighanLinMulticut(
                number_of_outer_iterations=10
            ),
            make_nifty_factory=lambda objective: objective.chainedSolversFactory(
                [
                    objective.liftedMulticutGreedyAdditiveFactory(),
                    objective.liftedMulticutKernighanLinFactory(
                        numberOfOuterIterations=10
                    ),
                ]
            ),
        ),
        "lifted_fusion_move": SolverConfig(
            make_bic_solver=lambda: bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
                proposal_generator=bic.graph.lifted_multicut.WatershedProposalGenerator(),
                number_of_iterations=10,
                stop_if_no_improvement=4,
                number_of_threads=1,
                number_of_parallel_proposals=1,
            ),
            make_nifty_factory=lambda objective: objective.chainedSolversFactory(
                [
                    objective.liftedMulticutGreedyAdditiveFactory(),
                    objective.fusionMoveBasedFactory(
                        proposalGenerator=objective.watershedProposalGenerator(
                            seedingStrategy="SEED_FROM_LOCAL",
                        ),
                        numberOfIterations=10,
                        stopIfNoImprovement=4,
                        numberOfThreads=1,
                    ),
                ]
            ),
        ),
    }


def load_problem(size: str, *, timeout: float):
    import bioimage_cpp as bic
    import nifty
    import nifty.graph.opt.lifted_multicut as nlmc

    problem = bic.graph.lifted_multicut.load_lifted_multicut_problem(size, timeout=timeout)
    bic_graph = bic.graph.UndirectedGraph.from_edges(problem.n_nodes, problem.local_uvs)
    nifty_graph = nifty.graph.undirectedGraph(int(problem.n_nodes))
    nifty_graph.insertEdges(problem.local_uvs.astype(np.uint64, copy=False))

    def make_nifty_objective():
        objective = nlmc.liftedMulticutObjective(nifty_graph)
        objective.setGraphEdgesCosts(problem.local_costs)
        if problem.lifted_uvs.shape[0] > 0:
            objective.setCosts(
                problem.lifted_uvs.astype(np.uint64, copy=False),
                problem.lifted_costs.astype(np.float64, copy=False),
            )
        return objective

    return bic_graph, nifty_graph, make_nifty_objective, problem


def make_bic_objective(bic_graph, problem):
    import bioimage_cpp as bic

    return bic.graph.lifted_multicut.LiftedMulticutObjective(
        bic_graph,
        problem.local_costs,
        lifted_uvs=problem.lifted_uvs,
        lifted_costs=problem.lifted_costs,
    )


def bic_energy(bic_graph, problem, labels: np.ndarray) -> float:
    return float(make_bic_objective(bic_graph, problem).energy(labels))


def nifty_energy(nifty_objective, labels: np.ndarray) -> float:
    return float(nifty_objective.evalNodeLabels(labels.astype(np.uint64, copy=False)))


def evaluate(problem_name: str, solver_name: str, config: SolverConfig, args):
    bic_graph, _, make_nifty_objective, problem = load_problem(
        problem_name, timeout=args.timeout
    )
    bic_energies = []
    nifty_energies = []
    bic_runtimes = []
    nifty_runtimes = []

    for _ in range(args.n_repeats):
        bic_objective = make_bic_objective(bic_graph, problem)
        start = perf_counter()
        bic_labels = config.make_bic_solver().optimize(bic_objective)
        bic_runtimes.append(perf_counter() - start)
        bic_energies.append(bic_energy(bic_graph, problem, bic_labels))

        nifty_objective = make_nifty_objective()
        start = perf_counter()
        nifty_labels = (
            config.make_nifty_factory(nifty_objective)
            .create(nifty_objective)
            .optimize()
        )
        nifty_runtimes.append(perf_counter() - start)
        nifty_energies.append(nifty_energy(nifty_objective, np.asarray(nifty_labels)))

    bic_runtime = mean(bic_runtimes)
    nifty_runtime = mean(nifty_runtimes)
    return {
        "problem": problem_name,
        "solver": solver_name,
        "nodes": int(problem.n_nodes),
        "local_edges": int(problem.local_uvs.shape[0]),
        "lifted_edges": int(problem.lifted_uvs.shape[0]),
        "bic_energy": mean(bic_energies),
        "nifty_energy": mean(nifty_energies),
        "energy_diff": mean(bic_energies) - mean(nifty_energies),
        "bic_runtime_s": bic_runtime,
        "nifty_runtime_s": nifty_runtime,
        "runtime_ratio": nifty_runtime / bic_runtime if bic_runtime > 0 else float("inf"),
    }


def format_float(value: float) -> str:
    return f"{value:.6g}"


def print_progress(row: dict) -> None:
    print(
        (
            f"[done] {row['problem']} / {row['solver']}: "
            f"bic_energy={format_float(row['bic_energy'])}, "
            f"nifty_energy={format_float(row['nifty_energy'])}, "
            f"delta={format_float(row['energy_diff'])}, "
            f"bic_runtime={format_float(row['bic_runtime_s'])}s, "
            f"nifty_runtime={format_float(row['nifty_runtime_s'])}s, "
            f"ratio={format_float(row['runtime_ratio'])}"
        ),
        file=sys.stderr,
        flush=True,
    )


def print_markdown_table(rows: list[dict]) -> None:
    headers = [
        "problem",
        "solver",
        "nodes",
        "local_edges",
        "lifted_edges",
        "bic_energy",
        "nifty_energy",
        "energy_diff",
        "bic_runtime_s",
        "nifty_runtime_s",
        "runtime_ratio_nifty_over_bic",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            row["problem"],
            row["solver"],
            str(row["nodes"]),
            str(row["local_edges"]),
            str(row["lifted_edges"]),
            format_float(row["bic_energy"]),
            format_float(row["nifty_energy"]),
            format_float(row["energy_diff"]),
            format_float(row["bic_runtime_s"]),
            format_float(row["nifty_runtime_s"]),
            format_float(row["runtime_ratio"]),
        ]
        print("| " + " | ".join(values) + " |")


def main() -> None:
    configs = solver_configs()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate matched bioimage-cpp and nifty lifted multicut solvers "
            "on registered lifted multicut problems."
        )
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=tuple(configs.keys()),
        default=tuple(configs.keys()),
        help="Solvers to evaluate. Defaults to all.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=PROBLEMS,
        default=PROBLEMS,
        help="Problems to evaluate. Defaults to all.",
    )
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1")

    rows = []
    for problem_name in args.problems:
        for solver_name in args.solvers:
            row = evaluate(problem_name, solver_name, configs[solver_name], args)
            rows.append(row)
            print_progress(row)
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
