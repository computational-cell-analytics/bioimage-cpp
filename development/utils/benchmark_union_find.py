"""Micro-benchmark for the UnionFind Python bindings.

Times the bulk ``merge((N, 2) edges)`` and ``find(node_array)`` entry points,
which the code review changed to validate every node id against ``uf.size``
before touching the C++ structure (an extra O(N) pre-pass). This script lets us
A/B that pre-pass against a pre-review build.

Not part of the test suite. Run::

    python development/utils/benchmark_union_find.py --repeats 11
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


def run(repeats: int = 11, n_nodes: int = 1_000_000, n_edges: int = 2_000_000) -> dict:
    rng = np.random.default_rng(0)
    edges = rng.integers(0, n_nodes, size=(n_edges, 2), dtype=np.uint64)
    query = rng.integers(0, n_nodes, size=(n_edges,), dtype=np.uint64)

    # Pre-merged structure for the find benchmark (find does not mutate the
    # partition, so it can be reused across repeats).
    merged = bic.utils.UnionFind(n_nodes)
    merged.merge(edges)

    def bulk_merge():
        uf = bic.utils.UnionFind(n_nodes)
        uf.merge(edges)

    def bulk_find():
        merged.find(query)

    results = {
        "uf_bulk_merge": _timeit(bulk_merge, repeats),
        "uf_bulk_find": _timeit(bulk_find, repeats),
    }
    for r in results.values():
        r["meta"] = {"n_nodes": n_nodes, "n_edges": n_edges}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    results = run(repeats=args.repeats)
    for name, r in results.items():
        print(f"{name:<16} median={r['median'] * 1e3:8.3f} ms  min={r['min'] * 1e3:8.3f} ms")


if __name__ == "__main__":
    main()
