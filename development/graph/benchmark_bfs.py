"""Benchmark the workspace-reused BFS path (``lifted_edges_from_node_labels``).

The code review changed ``BfsWorkspace::reset`` from an O(N) clear of the
visited/distance buffers to an O(1) generation-stamp bump. That only pays off
when one workspace is reset across many sources, which is exactly what
``lifted_edges_from_node_labels`` does (one workspace per chunk, reset per
source). The single-call ``breadth_first_search`` builds a fresh workspace each
call and would NOT show the change, so we benchmark the lifted-edge path.

Single-threaded so the per-source reset cost is not hidden by parallelism. The
node count is swept to expose the previous O(N^2)-of-memset behavior.

Not part of the test suite. Run::

    python development/graph/benchmark_bfs.py --repeats 5
"""
from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic


def _timeit(fn, repeats: int, warmup: int = 1) -> dict:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        timings.append(perf_counter() - t0)
    return {"median": median(timings), "min": min(timings), "n": repeats}


def run(repeats: int = 5, depth: int = 2) -> dict:
    shapes = [(100, 100), (160, 160), (220, 220)]
    results: dict[str, dict] = {}
    for shape in shapes:
        n_nodes = int(np.prod(shape))
        graph = bic.graph.grid_graph(shape)
        rng = np.random.default_rng(0)
        node_labels = rng.integers(0, 50, size=(n_nodes,), dtype=np.uint64)

        def fn(g=graph, nl=node_labels, d=depth):
            bic.graph.lifted_multicut.lifted_edges_from_node_labels(
                g, nl, graph_depth=d, number_of_threads=1
            )

        key = f"bfs_lifted_{n_nodes}nodes_d{depth}"
        results[key] = _timeit(fn, repeats)
        results[key]["meta"] = {"n_nodes": n_nodes, "shape": shape, "depth": depth}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    results = run(repeats=args.repeats, depth=args.depth)
    for name, r in results.items():
        print(f"{name:<28} median={r['median'] * 1e3:9.3f} ms  min={r['min'] * 1e3:9.3f} ms")


if __name__ == "__main__":
    main()
