from __future__ import annotations

import argparse

from _grid_affinity_compatibility import (
    add_common_arguments,
    affogato_edges,
    affogato_edges_on_graph,
    bioimage_cpp_local,
    bioimage_cpp_local_on_graph,
    compare_edge_sets,
    load_problem,
    nifty_local,
    nifty_local_on_graph,
    prepare_2d_problem,
    prepare_3d_problem,
    print_timing,
    select_local_offsets,
    time_call,
)

import numpy as np


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

    import bioimage_cpp as bic
    import nifty.graph as ng
    from affogato.segmentation import MWSGridGraph

    bic_graph = bic.graph.grid_graph(affinities.shape[1:])
    bic_affinities = np.ascontiguousarray(affinities, dtype=np.float64)
    nifty_graph = ng.undirectedGridGraph(list(affinities.shape[1:]))
    affogato_graph = MWSGridGraph(list(affinities.shape[1:]))
    bic_feature_timings, _ = time_call(
        lambda: bioimage_cpp_local_on_graph(bic_graph, bic_affinities, offsets),
        args.repeats,
    )
    nifty_feature_timings, _ = time_call(
        lambda: nifty_local_on_graph(nifty_graph, affinities, offsets),
        args.repeats,
    )
    affogato_feature_timings, _ = time_call(
        lambda: affogato_edges_on_graph(affogato_graph, affinities, offsets),
        args.repeats,
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
    print_timing("local edges total", "bioimage-cpp", bic_timings, "nifty", nifty_timings)
    print_timing("local edges total", "bioimage-cpp", bic_timings, "affogato", affogato_timings)
    print_timing(
        "local edges prebuilt",
        "bioimage-cpp",
        bic_feature_timings,
        "nifty",
        nifty_feature_timings,
    )
    print_timing(
        "local edges prebuilt",
        "bioimage-cpp",
        bic_feature_timings,
        "affogato",
        affogato_feature_timings,
    )
    print("prebuilt bioimage-cpp timing excludes float32 -> float64 conversion")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local grid affinity topology and weights."
    )
    add_common_arguments(parser)
    run_check(parser.parse_args())


if __name__ == "__main__":
    main()
