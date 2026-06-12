"""Benchmark RAG label accumulation (``accumulate_labels``).

The code review replaced the per-thread ``n_threads x n_nodes`` map-of-maps with
a single combined-key ``(node, other)`` histogram per thread, and made the
per-node argmax single-threaded. This script times ``accumulate_labels`` across
thread counts (to confirm the allocation win and rule out a high-thread-count
regression from the now-sequential argmax) and across ``other``-label
cardinalities.

Inputs: a deterministic block-wise over-segmentation (each node is a contiguous
block, as in a real over-segmentation), so the RAG adjacency is local.

Not part of the test suite. Run::

    python development/graph/benchmark_accumulate_labels.py --repeats 11
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


def _make_labels(shape=(40, 256, 256), coarsen=(2, 4, 4)) -> np.ndarray:
    coarse_shape = tuple(s // c for s, c in zip(shape, coarsen))
    n_nodes = int(np.prod(coarse_shape))
    coarse = np.arange(n_nodes, dtype=np.uint64).reshape(coarse_shape)
    block = np.ones(coarsen, dtype=np.uint64)
    return np.ascontiguousarray(np.kron(coarse, block))


def run(repeats: int = 11) -> dict:
    labels = _make_labels()
    rag = bic.graph.region_adjacency_graph(labels)
    n_nodes = int(rag.number_of_nodes)
    rng = np.random.default_rng(0)

    results: dict[str, dict] = {}
    for n_other, tag in ((8, "dense8"), (2000, "sparse2000")):
        other = rng.integers(0, n_other, size=labels.shape).astype(np.uint64)
        other = np.ascontiguousarray(other)
        for threads in (1, 4, 8):
            def fn(t=threads, o=other):
                bic.graph.features.accumulate_labels(rag, labels, o, number_of_threads=t)

            key = f"accumulate_labels_{tag}_t{threads}"
            results[key] = _timeit(fn, repeats)
            results[key]["meta"] = {
                "n_nodes": n_nodes,
                "n_pixels": int(labels.size),
                "n_other": n_other,
                "threads": threads,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    results = run(repeats=args.repeats)
    for name, r in results.items():
        print(f"{name:<32} median={r['median'] * 1e3:8.3f} ms  min={r['min'] * 1e3:8.3f} ms")


if __name__ == "__main__":
    main()
