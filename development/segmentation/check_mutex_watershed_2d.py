from __future__ import annotations

import argparse

from _mutex_watershed_equivalence import add_common_arguments, run_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bioimage-cpp and affogato mutex watershed in 2D."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--z",
        type=int,
        default=0,
        help="Z slice from the example affinities used for the 2D check.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("Y", "X"),
        default=(512, 512),
        help="Spatial crop shape for the 2D check.",
    )
    args = parser.parse_args()

    run_check(
        ndim=2,
        repeats=args.repeats,
        data_prefix=args.data_prefix,
        z=args.z,
        yx_shape=tuple(args.shape),
        zyx_shape=(0, 0, 0),
    )


if __name__ == "__main__":
    main()
