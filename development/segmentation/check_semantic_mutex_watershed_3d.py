from __future__ import annotations

import argparse

from _semantic_mws_equivalence import add_common_arguments, run_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bioimage-cpp and affogato semantic mutex watershed in 3D."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=(8, 448, 448),
        help="Spatial crop shape for the 3D check.",
    )
    parser.add_argument(
        "--z-start",
        type=int,
        default=0,
        help="Z offset into the (cropped) volume.",
    )
    args = parser.parse_args()

    run_check(
        ndim=3,
        repeats=args.repeats,
        z=0,
        yx_shape=(0, 0),
        zyx_shape=tuple(args.shape),
        seed=args.seed,
        z_start=args.z_start,
    )


if __name__ == "__main__":
    main()
