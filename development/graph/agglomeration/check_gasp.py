"""Compare bioimage-cpp and nifty GASP (signed-graph) agglomerative clustering."""

from __future__ import annotations

import argparse

import numpy as np

import bioimage_cpp as bic

from _compatibility import load_problem, parser as base_parser, report, time_call


def make_parser() -> argparse.ArgumentParser:
    arg_parser = base_parser(__doc__ or "")
    # ``abs_max`` is intentionally not compared against nifty: nifty has no
    # direct sign-aware absolute-maximum linkage. The closest match is
    # ``MutexWatershedSettings`` but it additionally installs cannot-link
    # constraints, so the comparison is apples-to-oranges. Run the unit
    # tests for ``abs_max`` coverage instead.
    arg_parser.add_argument(
        "--linkage",
        default="mean",
        choices=["sum", "mean", "max", "min", "mutex_watershed"],
        help="GASP linkage rule.",
    )
    return arg_parser


def main() -> None:
    args = make_parser().parse_args()
    bic_graph, nifty_graph, _, signed_weights, _ = load_problem(
        args.sample, args.size, timeout=args.timeout
    )
    n_edges = int(bic_graph.number_of_edges)
    n_nodes = int(bic_graph.number_of_nodes)
    edge_sizes = np.ones(n_edges, dtype=np.float64)

    import nifty.graph.agglo as nagglo

    nifty_settings_cls = {
        "mean": nagglo.ArithmeticMeanSettings,
        "sum": nagglo.SumSettings,
        "max": nagglo.MaxSettings,
        "min": nagglo.MinSettings,
        "mutex_watershed": nagglo.MutexWatershedSettings,
    }[args.linkage]

    def run_bic() -> np.ndarray:
        return bic.graph.agglomeration.GaspClusterPolicy(
            num_clusters_stop=args.num_clusters_stop,
            linkage=args.linkage,
        ).optimize(bic_graph, signed_weights, edge_sizes=edge_sizes)

    def run_nifty() -> np.ndarray:
        policy = nagglo.gaspClusterPolicy(
            graph=nifty_graph,
            signedWeights=signed_weights.astype(np.float64),
            isMergeEdge=np.ones(n_edges, dtype=np.uint8),
            edgeSizes=edge_sizes.astype(np.float64),
            nodeSizes=np.ones(n_nodes, dtype=np.float64),
            updateRule0=nifty_settings_cls(),
            numberOfNodesStop=args.num_clusters_stop,
        )
        clustering = nagglo.agglomerativeClustering(policy)
        clustering.run()
        return np.asarray(clustering.result(), dtype=np.uint64)

    bic_timings, bic_labels = time_call(run_bic, args.repeats)
    nifty_timings, nifty_labels = time_call(run_nifty, args.repeats)

    report(
        f"gasp_{args.linkage}",
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
