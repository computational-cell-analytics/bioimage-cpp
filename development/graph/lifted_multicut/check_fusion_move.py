from __future__ import annotations

import bioimage_cpp as bic

from _compatibility import parser, run_comparison


def main() -> None:
    args = parser(
        "Compare bioimage-cpp and nifty fusion-move lifted multicut at matched "
        "settings."
    ).parse_args()
    # Match settings on both sides:
    #   - threads / parallel-proposals = `--threads N` (default 1). We pin
    #     P = T explicitly on the bic side so the comparison is
    #     apples-to-apples regardless of either library's default for P.
    #   - greedy-additive warm-start: we do it automatically when the
    #     objective's labels are the trivial singleton labeling. nifty's
    #     `fusionMoveBasedFactory` has no warm-start parameter, so we make it
    #     explicit on the nifty side by chaining greedy-additive in front.
    #   - greedy-additive sub-solver (the implicit default on both sides).
    #   - watershed proposal generator seeded from local edges. The bic
    #     proposal generator only sees the base graph, so seeding from
    #     "local" mirrors that exactly; the nifty default
    #     `SEED_FROM_LIFTED` would put seeds where bic cannot.
    threads = int(args.threads)
    # nifty's lifted-multicut fusion-move backend only implements single-
    # threaded execution; passing numberOfThreads > 1 raises at solve time.
    # Cap the nifty side at 1 so multi-threaded bic benchmarks still run, and
    # surface the asymmetry in the printed output.
    nifty_threads = 1
    if threads != nifty_threads:
        print(
            f"note: nifty lifted fusion-move runs single-threaded; comparing "
            f"bic threads={threads} vs nifty threads={nifty_threads}."
        )
    run_comparison(
        "lifted_fusion_move",
        lambda: bic.graph.FusionMoveLiftedMulticut(
            proposal_generator=bic.graph.WatershedProposalGenerator(),
            number_of_iterations=10,
            stop_if_no_improvement=4,
            number_of_threads=threads,
            number_of_parallel_proposals=threads,
        ),
        lambda objective: objective.chainedSolversFactory(
            [
                objective.liftedMulticutGreedyAdditiveFactory(),
                objective.fusionMoveBasedFactory(
                    proposalGenerator=objective.watershedProposalGenerator(
                        seedingStrategy="SEED_FROM_LOCAL",
                    ),
                    numberOfIterations=10,
                    stopIfNoImprovement=4,
                    numberOfThreads=nifty_threads,
                ),
            ]
        ),
        args,
    )


if __name__ == "__main__":
    main()
