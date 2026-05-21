"""Compare bioimage_cpp.segmentation.label against skimage and vigra in 2D.

Example::

    python development/segmentation/check_label_2d.py --size 1024 --repeats 5
    python development/segmentation/check_label_2d.py --connectivity 2 --problem multi
"""

from __future__ import annotations

import argparse

from _label_equivalence import add_common_arguments, run_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bioimage-cpp and skimage/vigra label() in 2D."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Side length of the square 2D image.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 = 4-connectivity, 2 = 8-connectivity.",
    )
    args = parser.parse_args()

    run_check(
        ndim=2,
        size=args.size,
        connectivity=args.connectivity,
        repeats=args.repeats,
        density=args.density,
        problem_kind=args.problem,
    )


if __name__ == "__main__":
    main()
