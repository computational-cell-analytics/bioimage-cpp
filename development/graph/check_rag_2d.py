from __future__ import annotations

import argparse

from _rag_compatibility import add_common_arguments, run_compatibility_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bioimage-cpp and Nifty RAG functionality in 2D."
    )
    add_common_arguments(parser)
    parser.add_argument("--z", type=int, default=0)
    parser.add_argument("--shape", type=int, nargs=2, metavar=("Y", "X"), default=(128, 128))
    args = parser.parse_args()

    run_compatibility_check(
        ndim=2,
        repeats=args.repeats,
        threads=args.threads,
        data_prefix=args.data_prefix,
        z=args.z,
        yx_shape=tuple(args.shape),
        zyx_shape=(0, 0, 0),
        watershed_min_distance=args.watershed_min_distance,
        watershed_grid_spacing=args.watershed_grid_spacing,
        max_markers=args.max_markers,
    )


if __name__ == "__main__":
    main()
