from __future__ import annotations

import bioimage_cpp as bic

from _compatibility import parser, run_comparison


def main() -> None:
    args = parser("Compare bioimage-cpp and nifty greedy-fixation multicut.").parse_args()
    run_comparison(
        "greedy_fixation",
        lambda: bic.graph.GreedyFixationMulticut(),
        lambda objective: objective.greedyFixationFactory(),
        args,
    )


if __name__ == "__main__":
    main()
