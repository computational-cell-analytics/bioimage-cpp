from __future__ import annotations

import bioimage_cpp as bic

from _compatibility import parser, run_comparison


def main() -> None:
    args = parser(
        "Compare bioimage-cpp and nifty Kernighan-Lin lifted multicut."
    ).parse_args()
    # On the bioimage-cpp side, LiftedKernighanLinMulticut auto-warm-starts
    # from a greedy-additive pass when the objective's current labels are the
    # trivial singleton labeling. On the nifty side we make the equivalent
    # warm-start explicit via the chained-solvers factory so the two
    # implementations are doing the same work.
    run_comparison(
        "lifted_kernighan_lin",
        lambda: bic.graph.lifted_multicut.LiftedKernighanLinMulticut(number_of_outer_iterations=10),
        lambda objective: objective.chainedSolversFactory(
            [
                objective.liftedMulticutGreedyAdditiveFactory(),
                objective.liftedMulticutKernighanLinFactory(
                    numberOfOuterIterations=10
                ),
            ]
        ),
        args,
    )


if __name__ == "__main__":
    main()
