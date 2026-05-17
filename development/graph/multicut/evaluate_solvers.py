from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Callable

import numpy as np


PROBLEMS = tuple(
    f"{sample}_{size}"
    for sample in ("A", "B", "C")
    for size in ("small", "medium")
)


@dataclass(frozen=True)
class SolverConfig:
    make_bic_solver: Callable[[int], object]
    make_nifty_factory: Callable[[object, int], object]


def solver_configs():
    import bioimage_cpp as bic

    return {
        "greedy_additive": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.GreedyAdditiveMulticut(),
            make_nifty_factory=lambda objective, threads: objective.greedyAdditiveFactory(),
        ),
        "kernighan_lin": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.KernighanLinMulticut(
                number_of_outer_iterations=5
            ),
            make_nifty_factory=lambda objective, threads: objective.kernighanLinFactory(
                warmStartGreedy=True,
                numberOfOuterIterations=5,
            ),
        ),
        "greedy_fixation": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.GreedyFixationMulticut(),
            make_nifty_factory=lambda objective, threads: objective.greedyFixationFactory(),
        ),
        "chained": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.ChainedMulticutSolvers(
                [
                    bic.graph.GreedyAdditiveMulticut(),
                    bic.graph.KernighanLinMulticut(number_of_outer_iterations=5),
                ]
            ),
            make_nifty_factory=lambda objective, threads: objective.chainedSolversFactory(
                [
                    objective.greedyAdditiveFactory(),
                    objective.kernighanLinFactory(numberOfOuterIterations=5),
                ]
            ),
        ),
        "decomposer": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.MulticutDecomposer(
                bic.graph.GreedyAdditiveMulticut()
            ),
            make_nifty_factory=lambda objective, threads: objective.multicutDecomposerFactory(
                submodelFactory=objective.greedyAdditiveFactory(),
                fallthroughFactory=objective.greedyAdditiveFactory(),
                numberOfThreads=threads,
            ),
        ),
        "fusion_move": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.FusionMoveMulticut(
                proposal_generator=bic.graph.WatershedProposalGenerator(),
                number_of_threads=threads,
                number_of_parallel_proposals=threads,
            ),
            make_nifty_factory=lambda objective, threads: objective.ccFusionMoveBasedFactory(
                proposalGenerator=objective.watershedCcProposals(),
                fusionMove=objective.fusionMoveSettings(
                    mcFactory=objective.greedyAdditiveFactory(),
                ),
                numberOfIterations=10,
                stopIfNoImprovement=4,
                numberOfThreads=threads,
                warmStartGreedy=True,
            ),
        ),
    }


def parse_problem_name(name: str) -> tuple[str, str]:
    try:
        sample, size = name.split("_", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"problem must be SAMPLE_SIZE, got {name!r}"
        ) from error
    if name not in PROBLEMS:
        raise argparse.ArgumentTypeError(
            f"unknown problem {name!r}; available: {', '.join(PROBLEMS)}"
        )
    return sample, size


def load_problem(problem_name: str, *, timeout: float):
    import bioimage_cpp as bic
    import nifty

    sample, size = parse_problem_name(problem_name)
    uv_ids, costs = bic.graph.load_multicut_problem_data(
        sample=sample,
        size=size,
        timeout=timeout,
    )
    n_nodes = int(uv_ids.max()) + 1
    bic_graph = bic.graph.UndirectedGraph.from_edges(n_nodes, uv_ids)
    nifty_graph = nifty.graph.undirectedGraph(n_nodes)
    nifty_graph.insertEdges(uv_ids.astype(np.uint64, copy=False))
    return bic_graph, nifty_graph, costs


def bic_energy(graph, costs: np.ndarray, labels: np.ndarray) -> float:
    import bioimage_cpp as bic

    return float(bic.graph.MulticutObjective(graph, costs).energy(labels))


def nifty_energy(graph, costs: np.ndarray, labels: np.ndarray) -> float:
    import nifty.graph.opt.multicut as nmc

    return float(nmc.multicutObjective(graph, costs).evalNodeLabels(labels))


def evaluate(problem_name: str, solver_name: str, config: SolverConfig, args):
    import bioimage_cpp as bic
    import nifty.graph.opt.multicut as nmc

    bic_graph, nifty_graph, costs = load_problem(problem_name, timeout=args.timeout)
    bic_energies = []
    nifty_energies = []
    bic_runtimes = []
    nifty_runtimes = []

    for _ in range(args.n_repeats):
        if args.backend in ("both", "bic"):
            bic_objective = bic.graph.MulticutObjective(bic_graph, costs)
            start = perf_counter()
            bic_labels = config.make_bic_solver(args.threads).optimize(bic_objective)
            bic_runtimes.append(perf_counter() - start)
            bic_energies.append(bic_energy(bic_graph, costs, bic_labels))

        if args.backend in ("both", "nifty"):
            nifty_objective = nmc.multicutObjective(nifty_graph, costs)
            start = perf_counter()
            nifty_labels = (
                config.make_nifty_factory(nifty_objective, args.threads)
                .create(nifty_objective)
                .optimize()
            )
            nifty_runtimes.append(perf_counter() - start)
            nifty_energies.append(nifty_energy(nifty_graph, costs, nifty_labels))

    bic_runtime = mean(bic_runtimes) if bic_runtimes else None
    nifty_runtime = mean(nifty_runtimes) if nifty_runtimes else None
    bic_energy_value = mean(bic_energies) if bic_energies else None
    nifty_energy_value = mean(nifty_energies) if nifty_energies else None
    energy_diff = (
        bic_energy_value - nifty_energy_value
        if bic_energy_value is not None and nifty_energy_value is not None
        else None
    )
    return {
        "problem": problem_name,
        "solver": solver_name,
        "nodes": int(bic_graph.number_of_nodes),
        "edges": int(bic_graph.number_of_edges),
        "bic_energy": bic_energy_value,
        "nifty_energy": nifty_energy_value,
        "energy_diff": energy_diff,
        "bic_runtime_s": bic_runtime,
        "nifty_runtime_s": nifty_runtime,
        "runtime_ratio": (
            nifty_runtime / bic_runtime
            if bic_runtime is not None and nifty_runtime is not None and bic_runtime > 0
            else None
        ),
    }


def format_float(value: float) -> str:
    if value is None:
        return ""
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


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        json.dump(row, f, sort_keys=True)
        f.write("\n")
        f.flush()


def print_markdown_table(rows: list[dict]) -> None:
    headers = [
        "problem",
        "solver",
        "nodes",
        "edges",
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
            str(row["edges"]),
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
            "Evaluate matched bioimage-cpp and nifty multicut solvers on "
            "registered multicut problems."
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
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--backend",
        choices=("both", "bic", "nifty"),
        default="both",
        help="Which implementation to run. Defaults to both.",
    )
    parser.add_argument(
        "--results-jsonl",
        help="Append each completed row to this JSONL file.",
    )
    args = parser.parse_args()

    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    rows = []
    for problem_name in args.problems:
        for solver_name in args.solvers:
            row = evaluate(problem_name, solver_name, configs[solver_name], args)
            rows.append(row)
            print_progress(row)
            if args.results_jsonl is not None:
                append_jsonl(args.results_jsonl, row)
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
