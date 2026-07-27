from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
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
            make_bic_solver=lambda threads: bic.graph.multicut.GreedyAdditiveMulticut(),
            make_nifty_factory=lambda objective, threads: objective.greedyAdditiveFactory(),
        ),
        "kernighan_lin": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.multicut.KernighanLinMulticut(
                number_of_outer_iterations=5
            ),
            make_nifty_factory=lambda objective, threads: objective.kernighanLinFactory(
                warmStartGreedy=True,
                numberOfOuterIterations=5,
            ),
        ),
        "greedy_fixation": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.multicut.GreedyFixationMulticut(),
            make_nifty_factory=lambda objective, threads: objective.greedyFixationFactory(),
        ),
        "chained": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.multicut.ChainedMulticutSolvers(
                [
                    bic.graph.multicut.GreedyAdditiveMulticut(),
                    bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
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
            make_bic_solver=lambda threads: bic.graph.multicut.MulticutDecomposer(
                bic.graph.multicut.GreedyAdditiveMulticut(),
                fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
                number_of_threads=threads,
            ),
            make_nifty_factory=lambda objective, threads: objective.multicutDecomposerFactory(
                submodelFactory=objective.greedyAdditiveFactory(),
                fallthroughFactory=objective.greedyAdditiveFactory(),
                numberOfThreads=threads,
            ),
        ),
        "fusion_move": SolverConfig(
            make_bic_solver=lambda threads: bic.graph.multicut.FusionMoveMulticut(
                proposal_generator=bic.graph.multicut.WatershedProposalGenerator(),
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


def load_problem(problem_name: str, *, timeout: float, load_nifty: bool):
    import bioimage_cpp as bic

    sample, size = parse_problem_name(problem_name)
    uv_ids, costs = bic.graph.multicut.load_multicut_problem_data(
        sample=sample,
        size=size,
        timeout=timeout,
    )
    n_nodes = int(uv_ids.max()) + 1
    bic_graph = bic.graph.UndirectedGraph.from_edges(n_nodes, uv_ids)
    nifty_graph = None
    if load_nifty:
        import nifty

        nifty_graph = nifty.graph.undirectedGraph(n_nodes)
        nifty_graph.insertEdges(uv_ids.astype(np.uint64, copy=False))
    return bic_graph, nifty_graph, costs


def bic_energy(graph, costs: np.ndarray, labels: np.ndarray) -> float:
    import bioimage_cpp as bic

    return float(bic.graph.multicut.MulticutObjective(graph, costs).energy(labels))


def nifty_energy(graph, costs: np.ndarray, labels: np.ndarray) -> float:
    import nifty.graph.opt.multicut as nmc

    return float(nmc.multicutObjective(graph, costs).evalNodeLabels(labels))


def label_digest(labels: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(labels)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.uint64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout.strip() or completed.stderr.strip()
    return output or None


def cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def collect_environment(build_command: str | None) -> dict:
    import bioimage_cpp as bic

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler_version = command_output([*compiler, "--version"]) if compiler else None
    tracked_status = command_output(
        ["git", "status", "--short", "--untracked-files=no"]
    )
    return {
        "command": shlex.join([sys.executable, *sys.argv]),
        "build_command": build_command,
        "source_commit": command_output(["git", "rev-parse", "HEAD"]),
        "tracked_dirty": bool(tracked_status),
        "tracked_status": tracked_status,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "compiler_command": compiler,
        "compiler_version": compiler_version,
        "packages": {
            "bioimage-cpp": package_version("bioimage-cpp"),
            "numpy": np.__version__,
            "nifty": package_version("nifty"),
        },
        "module_paths": {
            "bioimage_cpp": str(Path(bic.__file__).resolve()),
            "nifty": (
                importlib.util.find_spec("nifty").origin
                if importlib.util.find_spec("nifty") is not None
                else None
            ),
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CXX",
                "CXXFLAGS",
                "CMAKE_ARGS",
                "SKBUILD_CMAKE_ARGS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def evaluate(
    problem_name: str,
    solver_name: str,
    config: SolverConfig,
    args,
    *,
    run_bic: bool,
    run_nifty: bool,
):
    import bioimage_cpp as bic

    nmc = None
    if run_nifty:
        import nifty.graph.opt.multicut as nmc

    bic_graph, nifty_graph, costs = load_problem(
        problem_name,
        timeout=args.timeout,
        load_nifty=run_nifty,
    )
    if solver_name == "decomposer":
        component_labels = bic.graph.connected_components(
            bic_graph,
            edge_mask=costs > 0.0,
        )
        component_sizes = np.bincount(component_labels.astype(np.intp, copy=False))
        if np.count_nonzero(component_sizes > 1) < 2:
            raise ValueError(
                "decomposer benchmark requires at least two non-singleton "
                "positive-cost components"
            )
    bic_energies = []
    nifty_energies = []
    bic_runtimes = []
    nifty_runtimes = []
    bic_label_digests = []
    nifty_label_digests = []

    for repeat in range(args.n_repeats):
        order = ("bic", "nifty") if repeat % 2 == 0 else ("nifty", "bic")
        for backend in order:
            if backend == "bic" and run_bic:
                bic_objective = bic.graph.multicut.MulticutObjective(bic_graph, costs)
                start = perf_counter()
                bic_labels = config.make_bic_solver(args.threads).optimize(bic_objective)
                bic_runtimes.append(perf_counter() - start)
                bic_energies.append(bic_energy(bic_graph, costs, bic_labels))
                bic_label_digests.append(label_digest(bic_labels))
            elif backend == "nifty" and run_nifty:
                nifty_objective = nmc.multicutObjective(nifty_graph, costs)
                start = perf_counter()
                nifty_labels = (
                    config.make_nifty_factory(nifty_objective, args.threads)
                    .create(nifty_objective)
                    .optimize()
                )
                nifty_runtimes.append(perf_counter() - start)
                nifty_energies.append(nifty_energy(nifty_graph, costs, nifty_labels))
                nifty_label_digests.append(label_digest(nifty_labels))

    bic_runtime = median(bic_runtimes) if bic_runtimes else None
    nifty_runtime = median(nifty_runtimes) if nifty_runtimes else None
    bic_energy_value = median(bic_energies) if bic_energies else None
    nifty_energy_value = median(nifty_energies) if nifty_energies else None
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
        "bic_energies": bic_energies,
        "nifty_energies": nifty_energies,
        "bic_label_digests": bic_label_digests,
        "nifty_label_digests": nifty_label_digests,
        "bic_runtimes_s": bic_runtimes,
        "nifty_runtimes_s": nifty_runtimes,
        "bic_runtime_median_s": bic_runtime,
        "nifty_runtime_median_s": nifty_runtime,
        "bic_runtime_min_s": min(bic_runtimes) if bic_runtimes else None,
        "nifty_runtime_min_s": min(nifty_runtimes) if nifty_runtimes else None,
        "runtime_ratio_median": (
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
            f"bic_median={format_float(row['bic_runtime_median_s'])}s, "
            f"nifty_median={format_float(row['nifty_runtime_median_s'])}s, "
            f"ratio={format_float(row['runtime_ratio_median'])}"
        ),
        file=sys.stderr,
        flush=True,
    )


def append_jsonl(path: str, row: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
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
        "bic_runtime_median_s",
        "bic_runtime_min_s",
        "nifty_runtime_median_s",
        "nifty_runtime_min_s",
        "runtime_ratio_median_nifty_over_bic",
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
            format_float(row["bic_runtime_median_s"]),
            format_float(row["bic_runtime_min_s"]),
            format_float(row["nifty_runtime_median_s"]),
            format_float(row["nifty_runtime_min_s"]),
            format_float(row["runtime_ratio_median"]),
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
    parser.add_argument(
        "--build-command",
        help="Build command to store with a JSONL benchmark run.",
    )
    parser.add_argument(
        "--require-reference",
        action="append",
        choices=("nifty",),
        default=[],
        help="Fail if the named reference is unavailable.",
    )
    args = parser.parse_args()

    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.results_jsonl is not None and args.build_command is None:
        raise ValueError("--build-command is required with --results-jsonl")

    nifty_available = importlib.util.find_spec("nifty") is not None
    if "nifty" in args.require_reference and not nifty_available:
        raise RuntimeError("required reference 'nifty' is not available")
    run_bic = args.backend in ("both", "bic")
    run_nifty = args.backend in ("both", "nifty") and nifty_available
    reference_status = (
        "OK"
        if run_nifty
        else "NO-REF"
        if args.backend in ("both", "nifty")
        else "NOT_REQUESTED"
    )

    run_id = datetime.now(timezone.utc).isoformat()
    if args.results_jsonl is not None:
        append_jsonl(
            args.results_jsonl,
            {
                "record_type": "metadata",
                "schema_version": 2,
                "run_id": run_id,
                "created_utc": run_id,
                "reference_status": {"nifty": reference_status},
                "arguments": vars(args),
                "environment": collect_environment(args.build_command),
            },
        )

    rows = []
    for problem_name in args.problems:
        for solver_name in args.solvers:
            row = evaluate(
                problem_name,
                solver_name,
                configs[solver_name],
                args,
                run_bic=run_bic,
                run_nifty=run_nifty,
            )
            row["record_type"] = "result"
            row["schema_version"] = 2
            row["run_id"] = run_id
            row["reference_status"] = reference_status
            rows.append(row)
            print_progress(row)
            if args.results_jsonl is not None:
                append_jsonl(args.results_jsonl, row)
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
