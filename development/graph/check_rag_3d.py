from __future__ import annotations

import argparse

from _rag_compatibility import add_common_arguments, run_compatibility_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bioimage-cpp and Nifty RAG functionality in 3D."
    )
    add_common_arguments(parser)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("Z", "Y", "X"), default=(6, 96, 96))
    args = parser.parse_args()

    run_compatibility_check(
        ndim=3,
        repeats=args.repeats,
        threads=args.threads,
        z=0,
        yx_shape=(0, 0),
        zyx_shape=tuple(args.shape),
        watershed_min_distance=args.watershed_min_distance,
        watershed_grid_spacing=args.watershed_grid_spacing,
        max_markers=args.max_markers,
    )


if __name__ == "__main__":
    main()
