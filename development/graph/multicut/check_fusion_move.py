from __future__ import annotations

import bioimage_cpp as bic

from _compatibility import parser, run_comparison


def main() -> None:
    args = parser(
        "Compare bioimage-cpp and nifty fusion-move multicut at default settings."
    ).parse_args()
    # bioimage-cpp warm-starts from the trivial singleton labeling with a
    # greedy-additive pass before the proposal loop. Nifty's ccFusionMoveBased
    # exposes the same behaviour via the `warmStartGreedy=True` flag (which
    # internally chains a greedyAdditiveFactory in front of the fusion-move
    # factory). Both sides therefore enter the proposal loop from the same
    # starting point.
    run_comparison(
        "fusion_move",
        lambda: bic.graph.FusionMoveMulticut(
            proposal_generator=bic.graph.WatershedProposalGenerator(),
        ),
        lambda objective: objective.ccFusionMoveBasedFactory(
            proposalGenerator=objective.watershedCcProposals(),
            fusionMove=objective.fusionMoveSettings(
                mcFactory=objective.greedyAdditiveFactory(),
            ),
            numberOfIterations=10,
            stopIfNoImprovement=4,
            numberOfThreads=1,
            warmStartGreedy=True,
        ),
        args,
    )


if __name__ == "__main__":
    main()
