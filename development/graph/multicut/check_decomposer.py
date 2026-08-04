from __future__ import annotations

import bioimage_cpp as bic

from _compatibility import parser, run_comparison


def main() -> None:
    args = parser("Compare bioimage-cpp and nifty decomposer multicut.").parse_args()
    run_comparison(
        "decomposer",
        lambda: bic.graph.multicut.MulticutDecomposer(
            bic.graph.multicut.GreedyAdditiveMulticut(),
            fallthrough_solver=bic.graph.multicut.GreedyAdditiveMulticut(),
            number_of_threads=args.threads,
        ),
        lambda objective: objective.multicutDecomposerFactory(
            submodelFactory=objective.greedyAdditiveFactory(),
            fallthroughFactory=objective.greedyAdditiveFactory(),
            numberOfThreads=args.threads,
        ),
        args,
    )


if __name__ == "__main__":
    main()
