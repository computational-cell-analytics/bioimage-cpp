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


PROBLEMS = ("2d", "3d", "grid")


@dataclass(frozen=True)
class SolverConfig:
    make_bic_solver: Callable[[int], object]
    make_nifty_factory: Callable[[object, int], object]


def solver_configs():
    import bioimage_cpp as bic

    # Nifty runs lifted fusion move with one thread. Keep its side at one
    # thread when the bioimage-cpp benchmark requests more threads.
    return {
        "lifted_greedy_additive": SolverConfig(
            make_bic_solver=lambda threads: (
                bic.graph.lifted_multicut.LiftedGreedyAdditiveMulticut()
            ),
            make_nifty_factory=lambda objective, threads: (
                objective.liftedMulticutGreedyAdditiveFactory()
            ),
        ),
        "lifted_kernighan_lin": SolverConfig(
            make_bic_solver=lambda threads: (
                bic.graph.lifted_multicut.LiftedKernighanLinMulticut(
                    number_of_outer_iterations=10
                )
            ),
            make_nifty_factory=lambda objective, threads: (
                objective.chainedSolversFactory(
                    [
                        objective.liftedMulticutGreedyAdditiveFactory(),
                        objective.liftedMulticutKernighanLinFactory(
                            numberOfOuterIterations=10
                        ),
                    ]
                )
            ),
        ),
        "lifted_fusion_move": SolverConfig(
            make_bic_solver=lambda threads: (
                bic.graph.lifted_multicut.FusionMoveLiftedMulticut(
                    proposal_generator=(
                        bic.graph.lifted_multicut.WatershedProposalGenerator()
                    ),
                    number_of_iterations=10,
                    stop_if_no_improvement=4,
                    number_of_threads=threads,
                    number_of_parallel_proposals=threads,
                )
            ),
            make_nifty_factory=lambda objective, threads: (
                objective.chainedSolversFactory(
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
                )
            ),
        ),
    }


def load_problem(size: str, *, timeout: float, load_nifty: bool):
    import bioimage_cpp as bic

    problem = bic.graph.lifted_multicut.load_lifted_multicut_problem(
        size, timeout=timeout
    )
    bic_graph = bic.graph.UndirectedGraph.from_edges(
        problem.n_nodes, problem.local_uvs
    )

    if not load_nifty:
        return bic_graph, None, problem

    import nifty
    import nifty.graph.opt.lifted_multicut as nlmc

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

    return bic_graph, make_nifty_objective, problem


def make_bic_objective(bic_graph, problem):
    import bioimage_cpp as bic

    return bic.graph.lifted_multicut.LiftedMulticutObjective(
        bic_graph,
        problem.local_costs,
        lifted_uvs=problem.lifted_uvs,
        lifted_costs=problem.lifted_costs,
    )


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
    bic_graph, make_nifty_objective, problem = load_problem(
        problem_name,
        timeout=args.timeout,
        load_nifty=run_nifty,
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
                objective = make_bic_objective(bic_graph, problem)
                start = perf_counter()
                labels = config.make_bic_solver(args.threads).optimize(objective)
                bic_runtimes.append(perf_counter() - start)
                bic_energies.append(float(objective.energy(labels)))
                bic_label_digests.append(label_digest(labels))
            elif backend == "nifty" and run_nifty:
                objective = make_nifty_objective()
                start = perf_counter()
                labels = (
                    config.make_nifty_factory(objective, args.threads)
                    .create(objective)
                    .optimize()
                )
                nifty_runtimes.append(perf_counter() - start)
                labels = np.asarray(labels)
                nifty_energies.append(
                    float(
                        objective.evalNodeLabels(
                            labels.astype(np.uint64, copy=False)
                        )
                    )
                )
                nifty_label_digests.append(label_digest(labels))

    bic_runtime = median(bic_runtimes) if bic_runtimes else None
    nifty_runtime = median(nifty_runtimes) if nifty_runtimes else None
    bic_energy = median(bic_energies) if bic_energies else None
    nifty_energy = median(nifty_energies) if nifty_energies else None
    energy_diff = (
        bic_energy - nifty_energy
        if bic_energy is not None and nifty_energy is not None
        else None
    )
    return {
        "problem": problem_name,
        "solver": solver_name,
        "nodes": int(problem.n_nodes),
        "local_edges": int(problem.local_uvs.shape[0]),
        "lifted_edges": int(problem.lifted_uvs.shape[0]),
        "bic_energy": bic_energy,
        "nifty_energy": nifty_energy,
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
            if bic_runtime is not None
            and nifty_runtime is not None
            and bic_runtime > 0
            else None
        ),
    }


def format_float(value: float | None) -> str:
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
    with output_path.open("a", encoding="utf-8") as file:
        json.dump(row, file, sort_keys=True)
        file.write("\n")
        file.flush()


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
            str(row["local_edges"]),
            str(row["lifted_edges"]),
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
            "Evaluate matched bioimage-cpp and nifty lifted multicut solvers."
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
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--backend",
        choices=("both", "bic", "nifty"),
        default="both",
        help="Implementation to run. Defaults to both.",
    )
    parser.add_argument(
        "--results-jsonl",
        help="Append metadata and completed rows to this JSONL file.",
    )
    parser.add_argument(
        "--build-command",
        help="Build command to store with the benchmark metadata.",
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
