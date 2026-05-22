from __future__ import annotations

import bioimage_cpp as bic

from _compatibility import parser, run_comparison


def main() -> None:
    args = parser("Compare bioimage-cpp and nifty chained multicut solvers.").parse_args()
    run_comparison(
        "chained",
        lambda: bic.graph.multicut.ChainedMulticutSolvers(
            [
                bic.graph.multicut.GreedyAdditiveMulticut(),
                bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
            ]
        ),
        lambda objective: objective.chainedSolversFactory(
            [
                objective.greedyAdditiveFactory(),
                objective.kernighanLinFactory(numberOfOuterIterations=5),
            ]
        ),
        args,
    )


if __name__ == "__main__":
    main()
