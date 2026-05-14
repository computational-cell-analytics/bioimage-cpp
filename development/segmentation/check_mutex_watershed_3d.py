from __future__ import annotations

import argparse

from _mutex_watershed_equivalence import add_common_arguments, run_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bioimage-cpp and affogato mutex watershed in 3D."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=(6, 96, 96),
        help="Spatial crop shape for the 3D check.",
    )
    args = parser.parse_args()

    run_check(
        ndim=3,
        repeats=args.repeats,
        data_prefix=args.data_prefix,
        z=0,
        yx_shape=(0, 0),
        zyx_shape=tuple(args.shape),
    )


if __name__ == "__main__":
    main()
