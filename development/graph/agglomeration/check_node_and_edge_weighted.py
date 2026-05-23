"""Compare bioimage-cpp and nifty node-and-edge-weighted agglomerative clustering."""

from __future__ import annotations

import numpy as np

import bioimage_cpp as bic

from _compatibility import load_problem, parser, report, time_call


def main() -> None:
    args = parser(__doc__ or "").parse_args()
    bic_graph, nifty_graph, indicators, _, uv_ids = load_problem(
        args.sample, args.size, timeout=args.timeout
    )
    n_edges = int(bic_graph.number_of_edges)
    n_nodes = int(bic_graph.number_of_nodes)
    edge_sizes = np.ones(n_edges, dtype=np.float64)
    node_sizes = np.ones(n_nodes, dtype=np.float64)

    # No node features come with the external multicut problem, so synthesise
    # a small 4-channel embedding deterministically.
    rng = np.random.RandomState(0)
    node_features = rng.rand(n_nodes, 4).astype(np.float64)

    import nifty.graph.agglo as nagglo

    def run_bic() -> np.ndarray:
        return bic.graph.agglomeration.NodeAndEdgeWeightedClusterPolicy(
            num_clusters_stop=args.num_clusters_stop,
            size_regularizer=args.size_regularizer,
            beta=0.5,
        ).optimize(
            bic_graph,
            indicators,
            node_features,
            edge_sizes=edge_sizes,
            node_sizes=node_sizes,
        )

    def run_nifty() -> np.ndarray:
        policy = nagglo.nodeAndEdgeWeightedClusterPolicy(
            graph=nifty_graph,
            edgeIndicators=indicators.astype(np.float32),
            edgeSizes=edge_sizes.astype(np.float32),
            nodeSizes=node_sizes.astype(np.float32),
            nodeFeatures=node_features.astype(np.float32),
            beta=0.5,
            numberOfNodesStop=args.num_clusters_stop,
            sizeRegularizer=args.size_regularizer,
        )
        clustering = nagglo.agglomerativeClustering(policy)
        clustering.run()
        return np.asarray(clustering.result(), dtype=np.uint64)

    bic_timings, bic_labels = time_call(run_bic, args.repeats)
    nifty_timings, nifty_labels = time_call(run_nifty, args.repeats)

    report(
        "node_and_edge_weighted",
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
