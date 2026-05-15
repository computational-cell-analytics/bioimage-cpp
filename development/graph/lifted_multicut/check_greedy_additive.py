from __future__ import annotations

import bioimage_cpp as bic

from _compatibility import parser, run_comparison


def main() -> None:
    args = parser(
        "Compare bioimage-cpp and nifty greedy-additive lifted multicut."
    ).parse_args()
    run_comparison(
        "lifted_greedy_additive",
        lambda: bic.graph.LiftedGreedyAdditiveMulticut(),
        lambda objective: objective.liftedMulticutGreedyAdditiveFactory(),
        args,
    )


if __name__ == "__main__":
    main()
