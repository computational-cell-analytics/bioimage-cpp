"""Compare bioimage_cpp.segmentation.label against skimage and vigra in 3D.

Example::

    python development/segmentation/check_label_3d.py --size 128 --repeats 3
    python development/segmentation/check_label_3d.py --connectivity 3
"""

from __future__ import annotations

import argparse

from _label_equivalence import add_common_arguments, run_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bioimage-cpp and skimage/vigra label() in 3D."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Side length of the cubic 3D volume.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="1 = 6-connectivity, 2 = 18-connectivity, 3 = 26-connectivity.",
    )
    args = parser.parse_args()

    run_check(
        ndim=3,
        size=args.size,
        connectivity=args.connectivity,
        repeats=args.repeats,
        density=args.density,
        problem_kind=args.problem,
    )


if __name__ == "__main__":
    main()
