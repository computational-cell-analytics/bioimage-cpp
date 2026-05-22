"""Compare bioimage-cpp `mutex_watershed_clustering` against affogato's
`compute_mws_clustering` reference, using the three registered lifted multicut
problems as inputs (`local_uvs`/`local_costs` as attractive edges,
`lifted_uvs`/`lifted_costs` as mutex edges).

Two modes:

* ``check`` (default): run a single problem (``--size 2d|3d|grid``), report
  partition equivalence + runtimes.
* ``evaluate``: run all registered sizes and print a markdown comparison
  table — mirrors the layout of
  ``development/graph/multicut/evaluate_solvers.py`` and
  ``development/graph/lifted_multicut/evaluate_solvers.py``.

Not part of the pytest suite (per AGENTS.md). Run manually with affogato
installed.
"""

from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np


PROBLEMS = ("2d", "3d", "grid")
DTYPES = ("float32", "float64")


def load_problem(size: str, *, timeout: float):
    import bioimage_cpp as bic

    problem = bic.graph.lifted_multicut.load_lifted_multicut_problem(size, timeout=timeout)
    bic_graph = bic.graph.UndirectedGraph.from_edges(problem.n_nodes, problem.local_uvs)
    return bic_graph, problem


def run_bioimage_cpp(bic_graph, problem, *, dtype: np.dtype) -> np.ndarray:
    import bioimage_cpp as bic

    return bic.graph.mutex_watershed.mutex_watershed_clustering(
        bic_graph,
        problem.local_costs.astype(dtype, copy=False),
        problem.lifted_uvs,
        problem.lifted_costs.astype(dtype, copy=False),
    )


def run_affogato_reference(problem) -> np.ndarray:
    # affogato's `compute_mws_clustering` uses float32 weights.
    from affogato.segmentation import compute_mws_clustering

    return compute_mws_clustering(
        int(problem.n_nodes),
        problem.local_uvs.astype(np.uint64, copy=False),
        problem.lifted_uvs.astype(np.uint64, copy=False),
        problem.local_costs.astype(np.float32, copy=False),
        problem.lifted_costs.astype(np.float32, copy=False),
    )


def _load_validation_metrics():
    try:
        from elf.validation import rand_index, variation_of_information

        return "elf.validation", rand_index, variation_of_information
    except ImportError:
        from elf.evaluation import rand_index, variation_of_information

        return "elf.evaluation", rand_index, variation_of_information


def _canonical_labels(labels: np.ndarray) -> np.ndarray:
    # Map to dense ids in first-occurrence order, so two partitions compare
    # equal iff they induce the same node grouping (independent of which
    # integer happened to be assigned to which cluster).
    array = np.asarray(labels)
    _, first_index, inverse = np.unique(
        array, return_index=True, return_inverse=True
    )
    order = np.argsort(first_index)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    return remap[inverse].astype(np.uint64, copy=False)


def compare_partitions(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict:
    source, rand_index, variation_of_information = _load_validation_metrics()
    vi_split, vi_merge = variation_of_information(candidate, reference)
    adapted_rand_error, ri = rand_index(candidate, reference)
    partition_equal = bool(
        np.array_equal(_canonical_labels(candidate), _canonical_labels(reference))
    )
    return {
        "validation_source": source,
        "vi_split": float(vi_split),
        "vi_merge": float(vi_merge),
        "adapted_rand_error": float(adapted_rand_error),
        "rand_index": float(ri),
        "partition_equal": partition_equal,
        "n_clusters_bic": int(np.unique(candidate).size),
        "n_clusters_reference": int(np.unique(reference).size),
    }


def time_function_interleaved(
    bic_run: Callable[[], np.ndarray],
    reference_run: Callable[[], np.ndarray],
    repeats: int,
) -> tuple[list[float], np.ndarray, list[float], np.ndarray]:
    # Warm up both implementations so JIT / first-call allocation costs do
    # not contaminate the timed runs.
    bic_result = bic_run()
    ref_result = reference_run()

    bic_timings: list[float] = []
    ref_timings: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            start = perf_counter()
            bic_result = bic_run()
            bic_timings.append(perf_counter() - start)
            start = perf_counter()
            ref_result = reference_run()
            ref_timings.append(perf_counter() - start)
        else:
            start = perf_counter()
            ref_result = reference_run()
            ref_timings.append(perf_counter() - start)
            start = perf_counter()
            bic_result = bic_run()
            bic_timings.append(perf_counter() - start)
    return bic_timings, bic_result, ref_timings, ref_result


def run_size(
    size: str, *, repeats: int, timeout: float, dtype: np.dtype
) -> dict:
    bic_graph, problem = load_problem(size, timeout=timeout)

    bic_timings, bic_labels, ref_timings, ref_labels = time_function_interleaved(
        lambda: run_bioimage_cpp(bic_graph, problem, dtype=dtype),
        lambda: run_affogato_reference(problem),
        repeats,
    )

    metrics = compare_partitions(bic_labels, ref_labels)
    bic_median = median(bic_timings)
    ref_median = median(ref_timings)
    return {
        "problem": size,
        "dtype": np.dtype(dtype).name,
        "nodes": int(problem.n_nodes),
        "local_edges": int(problem.local_uvs.shape[0]),
        "lifted_edges": int(problem.lifted_uvs.shape[0]),
        "bic_runtime_s": bic_median,
        "affogato_runtime_s": ref_median,
        "runtime_ratio": ref_median / bic_median if bic_median > 0 else float("inf"),
        **metrics,
    }


def print_check_report(result: dict) -> None:
    print(f"problem: size={result['problem']}, dtype={result['dtype']}, "
          f"nodes={result['nodes']}, local edges={result['local_edges']}, "
          f"lifted edges={result['lifted_edges']}")
    print(f"validation metrics: {result['validation_source']}")
    print(
        "VI split/merge: "
        f"{result['vi_split']:.6g} / {result['vi_merge']:.6g}"
    )
    print(
        "adapted rand error / rand index: "
        f"{result['adapted_rand_error']:.6g} / {result['rand_index']:.12g}"
    )
    print(f"partition equality (after canonical relabel): {result['partition_equal']}")
    print(f"clusters (bic / affogato): "
          f"{result['n_clusters_bic']} / {result['n_clusters_reference']}")
    print(f"bioimage-cpp median runtime [s]:    {result['bic_runtime_s']:.6f}")
    print(f"affogato median runtime [s]:        {result['affogato_runtime_s']:.6f}")
    print(f"affogato / bioimage-cpp runtime ratio: {result['runtime_ratio']:.3f}x")


def format_float(value: float) -> str:
    return f"{value:.6g}"


def print_markdown_table(rows: list[dict]) -> None:
    headers = [
        "problem",
        "dtype",
        "nodes",
        "local_edges",
        "lifted_edges",
        "n_clusters_bic",
        "n_clusters_affogato",
        "vi_split",
        "vi_merge",
        "adapted_rand_error",
        "rand_index",
        "partition_equal",
        "bic_runtime_s",
        "affogato_runtime_s",
        "runtime_ratio_affogato_over_bic",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            row["problem"],
            row["dtype"],
            str(row["nodes"]),
            str(row["local_edges"]),
            str(row["lifted_edges"]),
            str(row["n_clusters_bic"]),
            str(row["n_clusters_reference"]),
            format_float(row["vi_split"]),
            format_float(row["vi_merge"]),
            format_float(row["adapted_rand_error"]),
            format_float(row["rand_index"]),
            str(row["partition_equal"]),
            format_float(row["bic_runtime_s"]),
            format_float(row["affogato_runtime_s"]),
            format_float(row["runtime_ratio"]),
        ]
        print("| " + " | ".join(values) + " |")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare bioimage-cpp `mutex_watershed_clustering` against "
            "affogato's `compute_mws_clustering` on the registered lifted "
            "multicut problems."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=False)

    check_parser = subparsers.add_parser(
        "check",
        help=(
            "Run a single problem and print a partition-equivalence + "
            "runtime report (default mode)."
        ),
    )
    check_parser.add_argument(
        "--size",
        choices=PROBLEMS,
        default="3d",
        help="Lifted multicut problem instance to load (default: 3d).",
    )
    check_parser.add_argument("--repeats", type=int, default=3)
    check_parser.add_argument("--timeout", type=float, default=60.0)
    check_parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default="float32",
        help=(
            "Weight dtype for the bioimage-cpp call (default: float32, "
            "matching the precision affogato's reference uses internally)."
        ),
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help=(
            "Run all registered problem sizes and print a markdown "
            "comparison table."
        ),
    )
    evaluate_parser.add_argument(
        "--problems",
        nargs="+",
        choices=PROBLEMS,
        default=PROBLEMS,
        help="Problems to evaluate. Defaults to all.",
    )
    evaluate_parser.add_argument("--n-repeats", type=int, default=1)
    evaluate_parser.add_argument("--timeout", type=float, default=60.0)
    evaluate_parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DTYPES,
        default=("float32",),
        help=(
            "Weight dtype(s) for the bioimage-cpp call. Each (problem, "
            "dtype) pair becomes a row. Defaults to float32 (matches "
            "affogato's internal precision)."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Default to `check` if no subcommand was passed, matching the existing
    # `check_*` scripts in this directory.
    if args.mode is None or args.mode == "check":
        if args.mode is None:
            args.size = "3d"
            args.repeats = 3
            args.timeout = 60.0
            args.dtype = "float32"
        if args.repeats < 1:
            raise ValueError("--repeats must be at least 1")
        result = run_size(
            args.size,
            repeats=args.repeats,
            timeout=args.timeout,
            dtype=np.dtype(args.dtype),
        )
        print_check_report(result)
        return

    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1")
    rows = []
    for problem in args.problems:
        for dtype in args.dtypes:
            rows.append(
                run_size(
                    problem,
                    repeats=args.n_repeats,
                    timeout=args.timeout,
                    dtype=np.dtype(dtype),
                )
            )
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
