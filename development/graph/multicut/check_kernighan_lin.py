from __future__ import annotations

import bioimage_cpp as bic

from _compatibility import parser, run_comparison


def main() -> None:
    args = parser("Compare bioimage-cpp and nifty Kernighan-Lin multicut.").parse_args()
    run_comparison(
        "kernighan_lin",
        lambda: bic.graph.multicut.KernighanLinMulticut(number_of_outer_iterations=5),
        lambda objective: objective.kernighanLinFactory(
            warmStartGreedy=True,
            numberOfOuterIterations=5,
        ),
        args,
    )


if __name__ == "__main__":
    main()
