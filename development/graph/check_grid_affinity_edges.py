from __future__ import annotations

import argparse

from _grid_affinity_compatibility import (
    add_common_arguments,
    affogato_edges,
    bioimage_cpp_local,
    compare_edge_sets,
    load_problem,
    nifty_local,
    prepare_2d_problem,
    prepare_3d_problem,
    print_timing,
    select_local_offsets,
    time_call,
)


def run_check(args: argparse.Namespace) -> None:
    affinities, offsets = load_problem(args.data_prefix)
    if args.ndim == 2:
        affinities, offsets = prepare_2d_problem(
            affinities, offsets, z=args.z, yx_shape=tuple(args.yx_shape)
        )
    else:
        affinities, offsets = prepare_3d_problem(
            affinities, offsets, zyx_shape=tuple(args.zyx_shape)
        )
    affinities, offsets = select_local_offsets(affinities, offsets)

    bic_timings, (bic_uvs, bic_weights) = time_call(
        lambda: bioimage_cpp_local(affinities, offsets), args.repeats
    )
    nifty_timings, (nifty_uvs, nifty_weights) = time_call(
        lambda: nifty_local(affinities, offsets), args.repeats
    )
    affogato_timings, (affogato_uvs, affogato_weights) = time_call(
        lambda: affogato_edges(affinities, offsets), args.repeats
    )

    nifty_summary = compare_edge_sets(
        "nifty local", bic_uvs, bic_weights, nifty_uvs, nifty_weights
    )
    affogato_summary = compare_edge_sets(
        "affogato local", bic_uvs, bic_weights, affogato_uvs, affogato_weights
    )

    print(f"Grid local affinity edge check ({args.ndim}D)")
    print(f"affinities shape: {affinities.shape}, offsets: {offsets}")
    print(
        f"nifty edges: {nifty_summary['number_of_edges']}, "
        f"max abs weight diff: {nifty_summary['max_abs_weight_diff']:.6g}"
    )
    print(
        f"affogato edges: {affogato_summary['number_of_edges']}, "
        f"max abs weight diff: {affogato_summary['max_abs_weight_diff']:.6g}"
    )
    print_timing("local edges", "bioimage-cpp", bic_timings, "nifty", nifty_timings)
    print_timing("local edges", "bioimage-cpp", bic_timings, "affogato", affogato_timings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local grid affinity topology and weights."
    )
    add_common_arguments(parser)
    run_check(parser.parse_args())


if __name__ == "__main__":
    main()
