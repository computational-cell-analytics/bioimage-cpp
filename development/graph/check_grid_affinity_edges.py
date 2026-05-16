from __future__ import annotations

import argparse

from _grid_affinity_compatibility import (
    add_common_arguments,
    affogato_edges,
    affogato_edges_on_graph,
    assert_local_offsets_cover_all_edges,
    bioimage_cpp_local,
    bioimage_cpp_local_weights_only,
    bioimage_cpp_local_with_uvs,
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


# Notes on remaining apples-to-apples caveats (left in-code rather than fixed):
# * affogato's `MWSGridGraph` carries MWS-specific state and is heavier to
#   construct than a pure undirected grid graph. The "total" timings include
#   that cost on the affogato side. This isn't a bug in the comparison, it
#   reflects affogato's intended workload.
# * nifty / affogato accept the chosen dtype directly (verified separately);
#   feeding all three libraries the same dtype removes the previous implicit
#   float32 -> float64 copy that was charged only to bioimage-cpp.
# * `time_call` does an untimed warm-up call before the measured loop so the
#   first sample doesn't carry one-shot init costs.


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
    # One conversion to the chosen common dtype, BEFORE any timed work.
    # bioimage-cpp, nifty, and affogato all consume this same array.
    affinities = np.ascontiguousarray(affinities, dtype=np.dtype(args.dtype))

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
    nifty_graph = ng.undirectedGridGraph(list(affinities.shape[1:]))
    affogato_graph = MWSGridGraph(list(affinities.shape[1:]))
    # One-shot correctness check, OUTSIDE the timing loop.
    assert_local_offsets_cover_all_edges(bic_graph, affinities, offsets)

    # Apples-to-apples timing #1: each library returns (uvs, weights) for a
    # pre-built graph. nifty and affogato bundle uvs in their return value,
    # bioimage-cpp materializes them via graph.uv_ids().
    bic_with_uvs_timings, _ = time_call(
        lambda: bioimage_cpp_local_with_uvs(bic_graph, affinities, offsets),
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
    # Apples-to-apples timing #2: bioimage-cpp ONLY computes weights (no
    # uvs materialization). This isolates the cost of the feature kernel
    # itself. There is no nifty/affogato equivalent that returns just
    # weights, so this is reported on its own.
    bic_weights_only_timings, _ = time_call(
        lambda: bioimage_cpp_local_weights_only(bic_graph, affinities, offsets),
        args.repeats,
    )

    nifty_summary = compare_edge_sets(
        "nifty local", bic_uvs, bic_weights, nifty_uvs, nifty_weights
    )
    affogato_summary = compare_edge_sets(
        "affogato local", bic_uvs, bic_weights, affogato_uvs, affogato_weights
    )

    print(f"Grid local affinity edge check ({args.ndim}D)")
    print(
        f"affinities shape: {affinities.shape}, dtype: {affinities.dtype}, "
        f"size: {affinities.nbytes / 1e6:.2f} MB, offsets: {len(offsets)}, "
        f"repeats: {args.repeats}"
    )
    print(f"offsets: {offsets}")
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
        "local edges prebuilt (with uvs)",
        "bioimage-cpp",
        bic_with_uvs_timings,
        "nifty",
        nifty_feature_timings,
    )
    print_timing(
        "local edges prebuilt (with uvs)",
        "bioimage-cpp",
        bic_with_uvs_timings,
        "affogato",
        affogato_feature_timings,
    )
    # Weights-only kernel timing (no comparator — informational).
    from statistics import median

    print(
        "local edges prebuilt (weights only, no uvs materialization) "
        f"bioimage-cpp median runtime: {median(bic_weights_only_timings):.6f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local grid affinity topology and weights."
    )
    add_common_arguments(parser)
    run_check(parser.parse_args())


if __name__ == "__main__":
    main()
