"""Diagnose divergences between bioimage-cpp and nifty agglomeration policies.

Targeted at the cases where the benchmark sweep showed ARI < 0.90:

* mala on A/B/C small and C medium,
* edge_weighted on C medium,
* gasp_max on A/B small.

For each case, run both implementations, compare partitions, and (for the
hypothesised root cause) print enough state to confirm or deny it.
"""

from __future__ import annotations

import argparse
import numpy as np

import bioimage_cpp as bic

from _compatibility import load_problem


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(a, b))
    except ImportError:
        return float("nan")


def _cluster_sizes(labels: np.ndarray, top: int = 8) -> str:
    _, counts = np.unique(labels, return_counts=True)
    counts = np.sort(counts)[::-1]
    head = counts[:top].tolist()
    return f"n_clusters={len(counts)} top{top}={head} max={int(counts[0])} median={int(np.median(counts))}"


def diagnose_mala(sample: str, size: str) -> None:
    print(f"\n=== MALA  sample={sample} size={size} ===")
    bic_graph, nifty_graph, indicators, _, _ = load_problem(sample, size, timeout=120.0)
    n = int(bic_graph.number_of_nodes)
    e = int(bic_graph.number_of_edges)
    print(f"  nodes={n} edges={e}")

    bic_labels = bic.graph.agglomeration.MalaClusterPolicy(
        num_bins=40, bin_min=0.0, bin_max=1.0,
        num_clusters_stop=1000, threshold=0.5,
    ).optimize(bic_graph, indicators)

    import nifty.graph.agglo as nagglo
    policy = nagglo.malaClusterPolicy(
        graph=nifty_graph,
        edgeIndicators=indicators.astype(np.float32),
        nodeSizes=np.ones(n, dtype=np.float32),
        edgeSizes=np.ones(e, dtype=np.float32),
        threshold=0.5, numberOfNodesStop=1000,
    )
    clustering = nagglo.agglomerativeClustering(policy)
    clustering.run()
    nifty_labels = np.asarray(clustering.result(), dtype=np.uint64)

    print(f"  bic   : {_cluster_sizes(bic_labels)}")
    print(f"  nifty : {_cluster_sizes(nifty_labels)}")
    print(f"  ARI   : {_ari(bic_labels, nifty_labels):.4f}")
    # Sample a handful of indicators near 0.5 — the threshold — to highlight
    # how bin-center vs interpolated median changes the stop decision.
    mid = indicators[np.abs(indicators - 0.5) < 0.1]
    print(f"  #indicators within 0.1 of threshold=0.5: {len(mid)} "
          f"(out of {len(indicators)})")


def diagnose_edge_weighted(sample: str, size: str) -> None:
    print(f"\n=== EDGE_WEIGHTED  sample={sample} size={size} ===")
    bic_graph, nifty_graph, indicators, _, _ = load_problem(sample, size, timeout=120.0)
    n = int(bic_graph.number_of_nodes)
    e = int(bic_graph.number_of_edges)
    print(f"  nodes={n} edges={e}")

    edge_sizes = np.ones(e, dtype=np.float64)
    node_sizes = np.ones(n, dtype=np.float64)

    bic_labels = bic.graph.agglomeration.EdgeWeightedClusterPolicy(
        num_clusters_stop=1000, size_regularizer=0.5,
    ).optimize(bic_graph, indicators, edge_sizes=edge_sizes, node_sizes=node_sizes)

    import nifty.graph.agglo as nagglo
    policy = nagglo.edgeWeightedClusterPolicy(
        graph=nifty_graph,
        edgeIndicators=indicators.astype(np.float32),
        edgeSizes=edge_sizes.astype(np.float32),
        nodeSizes=node_sizes.astype(np.float32),
        numberOfNodesStop=1000, sizeRegularizer=0.5,
    )
    clustering = nagglo.agglomerativeClustering(policy)
    clustering.run()
    nifty_labels = np.asarray(clustering.result(), dtype=np.uint64)

    print(f"  bic   : {_cluster_sizes(bic_labels)}")
    print(f"  nifty : {_cluster_sizes(nifty_labels)}")
    print(f"  ARI   : {_ari(bic_labels, nifty_labels):.4f}")

    # How many edges share the smallest-bucket priority? (tie-breaking
    # signal: large equal-priority cohorts let the two impls diverge.)
    p = np.round(indicators, 6)
    _, counts = np.unique(p, return_counts=True)
    top = np.sort(counts)[::-1][:5].tolist()
    print(f"  unique-priorities up to 6 dp: {len(counts)}  top-5 counts: {top}")


def diagnose_gasp_max(sample: str, size: str) -> None:
    print(f"\n=== GASP max  sample={sample} size={size} ===")
    bic_graph, nifty_graph, _, signed_weights, _ = load_problem(sample, size, timeout=120.0)
    n = int(bic_graph.number_of_nodes)
    e = int(bic_graph.number_of_edges)
    print(f"  nodes={n} edges={e}")
    print(f"  signed_weights: min={signed_weights.min():.3f} max={signed_weights.max():.3f} "
          f"positive_fraction={(signed_weights > 0).mean():.3f}")

    edge_sizes = np.ones(e, dtype=np.float64)

    bic_labels = bic.graph.agglomeration.GaspClusterPolicy(
        num_clusters_stop=1000, linkage="max",
    ).optimize(bic_graph, signed_weights, edge_sizes=edge_sizes)

    import nifty.graph.agglo as nagglo
    policy = nagglo.gaspClusterPolicy(
        graph=nifty_graph,
        signedWeights=signed_weights.astype(np.float64),
        isMergeEdge=np.ones(e, dtype=np.uint8),
        edgeSizes=edge_sizes.astype(np.float64),
        nodeSizes=np.ones(n, dtype=np.float64),
        updateRule0=nagglo.MaxSettings(),
        numberOfNodesStop=1000,
    )
    clustering = nagglo.agglomerativeClustering(policy)
    clustering.run()
    nifty_labels = np.asarray(clustering.result(), dtype=np.uint64)

    print(f"  bic   : {_cluster_sizes(bic_labels)}")
    print(f"  nifty : {_cluster_sizes(nifty_labels)}")
    print(f"  ARI   : {_ari(bic_labels, nifty_labels):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["mala", "edge_weighted", "gasp_max", "all"],
                        default="all")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--size", default=None)
    args = parser.parse_args()

    cases_mala = [("A", "small"), ("B", "small"), ("C", "small"), ("C", "medium")]
    cases_ew = [("C", "medium")]
    cases_gm = [("A", "small"), ("B", "small")]

    if args.sample and args.size:
        cases_mala = [(args.sample, args.size)]
        cases_ew = [(args.sample, args.size)]
        cases_gm = [(args.sample, args.size)]

    if args.policy in ("mala", "all"):
        for s, z in cases_mala:
            diagnose_mala(s, z)
    if args.policy in ("edge_weighted", "all"):
        for s, z in cases_ew:
            diagnose_edge_weighted(s, z)
    if args.policy in ("gasp_max", "all"):
        for s, z in cases_gm:
            diagnose_gasp_max(s, z)


if __name__ == "__main__":
    main()
