from __future__ import annotations

import argparse

from _grid_affinity_compatibility import (
    add_common_arguments,
    affogato_edges,
    bioimage_cpp_lifted,
    compare_edge_sets,
    load_problem,
    prepare_2d_problem,
    prepare_3d_problem,
    print_timing,
    select_mixed_offsets,
    split_affogato_edges,
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
    affinities, offsets = select_mixed_offsets(affinities, offsets)

    bic_timings, (graph, bic_local_uvs, bic_local_weights, bic_lifted_uvs, bic_lifted_weights) = (
        time_call(lambda: bioimage_cpp_lifted(affinities, offsets), args.repeats)
    )
    affogato_timings, (affogato_uvs, affogato_weights) = time_call(
        lambda: affogato_edges(affinities, offsets), args.repeats
    )
    affogato_local_uvs, affogato_local_weights, affogato_lifted_uvs, affogato_lifted_weights = (
        split_affogato_edges(affogato_uvs, affogato_weights, graph)
    )

    local_summary = compare_edge_sets(
        "affogato local",
        bic_local_uvs,
        bic_local_weights,
        affogato_local_uvs,
        affogato_local_weights,
    )
    lifted_summary = compare_edge_sets(
        "affogato lifted",
        bic_lifted_uvs,
        bic_lifted_weights,
        affogato_lifted_uvs,
        affogato_lifted_weights,
    )

    print(f"Grid lifted affinity edge check ({args.ndim}D)")
    print(f"affinities shape: {affinities.shape}, offsets: {offsets}")
    print(
        f"local edges: {local_summary['number_of_edges']}, "
        f"max abs weight diff: {local_summary['max_abs_weight_diff']:.6g}"
    )
    print(
        f"lifted edges: {lifted_summary['number_of_edges']}, "
        f"max abs weight diff: {lifted_summary['max_abs_weight_diff']:.6g}"
    )
    print_timing("local+lifted edges", "bioimage-cpp", bic_timings, "affogato", affogato_timings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local and lifted grid affinity topology and weights."
    )
    add_common_arguments(parser)
    run_check(parser.parse_args())


if __name__ == "__main__":
    main()
