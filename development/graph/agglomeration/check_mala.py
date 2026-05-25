"""Compare bioimage-cpp and nifty MALA agglomerative clustering."""

from __future__ import annotations

import numpy as np

import bioimage_cpp as bic

from _compatibility import load_problem, parser, report, time_call


def main() -> None:
    args = parser(__doc__ or "").parse_args()
    bic_graph, nifty_graph, indicators, _, _ = load_problem(
        args.sample, args.size, timeout=args.timeout
    )
    n_edges = int(bic_graph.number_of_edges)
    n_nodes = int(bic_graph.number_of_nodes)

    import nifty.graph.agglo as nagglo

    def run_bic() -> np.ndarray:
        return bic.graph.agglomeration.MalaClusterPolicy(
            num_bins=40,
            bin_min=0.0,
            bin_max=1.0,
            num_clusters_stop=args.num_clusters_stop,
            threshold=args.threshold,
        ).optimize(bic_graph, indicators)

    def run_nifty() -> np.ndarray:
        policy = nagglo.malaClusterPolicy(
            graph=nifty_graph,
            edgeIndicators=indicators.astype(np.float32),
            nodeSizes=np.ones(n_nodes, dtype=np.float32),
            edgeSizes=np.ones(n_edges, dtype=np.float32),
            threshold=args.threshold,
            numberOfNodesStop=args.num_clusters_stop,
        )
        clustering = nagglo.agglomerativeClustering(policy)
        clustering.run()
        return np.asarray(clustering.result(), dtype=np.uint64)

    bic_timings, bic_labels = time_call(run_bic, args.repeats)
    nifty_timings, nifty_labels = time_call(run_nifty, args.repeats)

    report(
        "mala",
        bic_timings,
        nifty_timings,
        bic_labels,
        nifty_labels,
        n_nodes,
        n_edges,
        sample=args.sample,
        size=args.size,
    )


if __name__ == "__main__":
    main()
