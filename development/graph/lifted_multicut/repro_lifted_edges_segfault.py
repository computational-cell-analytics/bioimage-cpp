#!/usr/bin/env python
"""Self-contained reproduction of the bug in
``bioimage_cpp.graph.lifted_multicut.lifted_edges_from_node_labels``.

The function misbehaves once the graph grows past a few hundred nodes. The bug is
*nondeterministic* and shows up in two ways on the very same (deterministic) input:

  1. It intermittently SIGSEGVs.
  2. When it does return, the number of lifted edges varies from run to run and disagrees
     with the reference ``nifty.distributed.liftedNeighborhoodFromNodeLabels`` (which is
     stable). E.g. for a 2000-node chain at depth 3 we have seen 3288, 3837, ... while nifty
     consistently returns 3993.

A varying result for fixed input plus occasional crashes is the classic signature of a memory
error (out-of-bounds read/write) in the C++ implementation. It reproduces with a trivial chain
graph -- no RAG or production-scale data required -- and is independent of node-label values,
``graph_depth`` and ``mode``. The RegionAdjacencyGraph path tends to crash the most reliably.

Minimal trigger (run on its own a few times: some runs crash, others print a different count):

    import numpy as np
    import bioimage_cpp as bic

    n = 2000
    uv = np.array([(i, i + 1) for i in range(n - 1)], dtype="uint64")  # a simple chain
    g = bic.graph.UndirectedGraph.from_edges(n, uv)
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        g, np.zeros(n, "uint64"), graph_depth=3, mode="all")
    print(len(out))   # -> Segmentation fault, or a different number each run

Each configuration below is run several times, each in its own child process, so a crash does
not abort the sweep and the run-to-run variation is visible.

Run it with:

    python repro_lifted_edges_segfault.py
"""
import multiprocessing as mp
import queue as _queue
import signal

import numpy as np
import bioimage_cpp as bic

try:
    import nifty.distributed as ndist
except ImportError:
    ndist = None

GRAPH_DEPTH = 3
MODE = "all"
NODE_LADDER = (100, 500, 1000, 2000)
REPS = 5


def _chain_edges(n_nodes):
    """A simple connected chain 0-1-2-...-(n-1); enough to trigger the bug."""
    return np.array([(i, i + 1) for i in range(n_nodes - 1)], dtype="uint64")


def _chain_segmentation(n_nodes):
    """A labeled volume whose region adjacency graph is the same chain of ``n_nodes`` nodes."""
    return np.repeat(np.arange(n_nodes, dtype="uint32"), 16).reshape(n_nodes, 4, 4)


# --- workers (each invocation runs in its own process) -------------------------------------
def _bic_undirected(n_nodes, q):
    g = bic.graph.UndirectedGraph.from_edges(n_nodes, _chain_edges(n_nodes))
    node_labels = np.ones(n_nodes, dtype="uint64")  # label values are irrelevant to the bug
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        g, node_labels, graph_depth=GRAPH_DEPTH, mode=MODE)
    q.put(len(out))


def _bic_rag(n_nodes, q):
    rag = bic.graph.region_adjacency_graph(_chain_segmentation(n_nodes))
    node_labels = np.ones(rag.numberOfNodes, dtype="uint64")
    out = bic.graph.lifted_multicut.lifted_edges_from_node_labels(
        rag, node_labels, graph_depth=GRAPH_DEPTH, mode=MODE)
    q.put(len(out))


def _nifty_undirected(n_nodes, q):
    g = ndist.Graph(_chain_edges(n_nodes))
    node_labels = np.ones(n_nodes, dtype="uint64")  # non-zero so ignoreLabel=0 keeps every pair
    out = ndist.liftedNeighborhoodFromNodeLabels(
        g, node_labels, GRAPH_DEPTH, mode=MODE, numberOfThreads=1, ignoreLabel=0)
    q.put(len(out))


def _run_once(worker, n_nodes, timeout=120):
    """Run ``worker(n_nodes, queue)`` in a child process; return its lifted count or a status."""
    q = mp.Queue()
    p = mp.Process(target=worker, args=(n_nodes, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return "timeout"
    if p.exitcode == -signal.SIGSEGV:
        return "segfault"
    if p.exitcode == -signal.SIGABRT:
        return "abort"  # glibc "double free or corruption"
    if p.exitcode != 0:
        return f"exit {p.exitcode}"
    try:
        return q.get(timeout=5)
    except _queue.Empty:
        return "no-result"


def _summary(worker, n_nodes):
    """Run ``worker`` REPS times and summarise crashes and the distinct lifted-edge counts."""
    results = [_run_once(worker, n_nodes) for _ in range(REPS)]
    crash_labels = ("segfault", "abort")
    crashes = sum(1 for r in results if r in crash_labels)
    counts = sorted({r for r in results if isinstance(r, int)})
    others = [r for r in results if not isinstance(r, int) and r not in crash_labels]

    parts = []
    if crashes:
        kinds = "/".join(sorted({r.upper() for r in results if r in crash_labels}))
        parts.append(f"{crashes}/{REPS} {kinds}")
    if counts:
        flag = "  <- NONDETERMINISTIC" if len(counts) > 1 else ""
        parts.append("lifted=" + ",".join(map(str, counts)) + flag)
    parts.extend(sorted(set(others)))
    return "; ".join(parts) if parts else "no output"


def main():
    print("Reproducing the lifted_edges_from_node_labels bug "
          f"(graph_depth={GRAPH_DEPTH}, mode={MODE!r}, {REPS} runs per case).\n")

    workers = [("bic  UndirectedGraph     ", _bic_undirected),
               ("bic  RegionAdjacencyGraph", _bic_rag)]
    if ndist is not None:
        workers.append(("nifty (reference)        ", _nifty_undirected))

    for n in NODE_LADDER:
        print(f"=== {n} nodes ===")
        for name, worker in workers:
            print(f"  {name} : {_summary(worker, n)}")
        print()

    print("Reference (nifty) returns a single stable count; bic varies and/or crashes -> "
          "memory corruption in lifted_edges_from_node_labels.")


if __name__ == "__main__":
    # "spawn" gives each case a fresh interpreter, so a crash is cleanly attributed.
    mp.set_start_method("spawn", force=True)
    main()
