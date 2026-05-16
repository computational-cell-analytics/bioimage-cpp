from __future__ import annotations

import argparse

from _grid_affinity_compatibility import (
    add_common_arguments,
    affogato_edges,
    affogato_edges_on_graph,
    bioimage_cpp_lifted,
    bioimage_cpp_lifted_features_only,
    bioimage_cpp_lifted_with_uvs,
    compare_edge_sets,
    load_problem,
    prepare_2d_problem,
    prepare_3d_problem,
    print_timing,
    select_mixed_offsets,
    split_affogato_edges,
    time_call,
)

import numpy as np


# Notes on remaining apples-to-apples caveats (left in-code rather than fixed):
# * affogato bundles local + lifted edges into a single (uvs, weights) array,
#   bioimage-cpp returns local and lifted in separate arrays. The two output
#   formats are intrinsic to each library's design; the bookkeeping cost of
#   producing them is included in the respective timings.
# * affogato's `MWSGridGraph` is heavier to construct than a plain grid graph
#   (MWS-specific state). The "total" timings include that cost.
# * `time_call` does an untimed warm-up before the measured loop.


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
    # One conversion to the chosen common dtype, BEFORE any timed work.
    affinities = np.ascontiguousarray(affinities, dtype=np.dtype(args.dtype))

    bic_timings, (graph, bic_local_uvs, bic_local_weights, bic_lifted_uvs, bic_lifted_weights) = (
        time_call(lambda: bioimage_cpp_lifted(affinities, offsets), args.repeats)
    )
    affogato_timings, (affogato_uvs, affogato_weights) = time_call(
        lambda: affogato_edges(affinities, offsets), args.repeats
    )

    import bioimage_cpp as bic
    from affogato.segmentation import MWSGridGraph

    bic_graph = bic.graph.grid_graph(affinities.shape[1:])
    affogato_graph = MWSGridGraph(list(affinities.shape[1:]))
    # Apples-to-apples #1: both return (uvs, local_weights, lifted_uvs,
    # lifted_weights) — bundle of arrays sized for downstream multicut.
    bic_with_uvs_timings, _ = time_call(
        lambda: bioimage_cpp_lifted_with_uvs(bic_graph, affinities, offsets),
        args.repeats,
    )
    affogato_feature_timings, _ = time_call(
        lambda: affogato_edges_on_graph(affogato_graph, affinities, offsets),
        args.repeats,
    )
    # Apples-to-apples #2: bioimage-cpp ONLY returns weight arrays and the
    # lifted uvs (which must be returned because they aren't grid-indexed).
    # Isolates the cost of the feature kernel from local uv_ids() materialization.
    bic_features_only_timings, _ = time_call(
        lambda: bioimage_cpp_lifted_features_only(bic_graph, affinities, offsets),
        args.repeats,
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
    print(
        f"affinities shape: {affinities.shape}, dtype: {affinities.dtype}, "
        f"size: {affinities.nbytes / 1e6:.2f} MB, offsets: {len(offsets)}, "
        f"repeats: {args.repeats}"
    )
    print(f"offsets: {offsets}")
    print(
        f"local edges: {local_summary['number_of_edges']}, "
        f"max abs weight diff: {local_summary['max_abs_weight_diff']:.6g}"
    )
    print(
        f"lifted edges: {lifted_summary['number_of_edges']}, "
        f"max abs weight diff: {lifted_summary['max_abs_weight_diff']:.6g}"
    )
    print_timing(
        "local+lifted edges total",
        "bioimage-cpp",
        bic_timings,
        "affogato",
        affogato_timings,
    )
    print_timing(
        "local+lifted edges prebuilt (with uvs)",
        "bioimage-cpp",
        bic_with_uvs_timings,
        "affogato",
        affogato_feature_timings,
    )
    from statistics import median

    print(
        "local+lifted edges prebuilt (features only, no local uvs materialization) "
        f"bioimage-cpp median runtime: {median(bic_features_only_timings):.6f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local and lifted grid affinity topology and weights."
    )
    add_common_arguments(parser)
    run_check(parser.parse_args())


if __name__ == "__main__":
    main()
