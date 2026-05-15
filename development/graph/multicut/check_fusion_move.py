from __future__ import annotations

import bioimage_cpp as bic

from _compatibility import parser, run_comparison


def main() -> None:
    args = parser(
        "Compare bioimage-cpp and nifty fusion-move multicut at matched settings."
    ).parse_args()
    # Match settings on both sides:
    #   - threads / parallel-proposals = `--threads N` (default 1). We set
    #     P = T explicitly on both sides so the comparison is apples-to-apples
    #     regardless of either library's API default for P.
    #   - greedy-additive warm-start (we do it automatically when initial
    #     labels are the trivial singleton; nifty exposes it as
    #     `warmStartGreedy=True`).
    #   - greedy-additive sub-solver.
    threads = int(args.threads)
    run_comparison(
        "fusion_move",
        lambda: bic.graph.FusionMoveMulticut(
            proposal_generator=bic.graph.WatershedProposalGenerator(),
            number_of_threads=threads,
            number_of_parallel_proposals=threads,
        ),
        lambda objective: objective.ccFusionMoveBasedFactory(
            proposalGenerator=objective.watershedCcProposals(),
            fusionMove=objective.fusionMoveSettings(
                mcFactory=objective.greedyAdditiveFactory(),
            ),
            numberOfIterations=10,
            stopIfNoImprovement=4,
            numberOfThreads=threads,
            warmStartGreedy=True,
        ),
        args,
    )


if __name__ == "__main__":
    main()
