from __future__ import annotations

import argparse

import napari
from skimage.segmentation import find_boundaries

import bioimage_cpp as bic
from bioimage_cpp._data import load_isbi_raw

from _lifted_problem import build_lifted_problem, load_affinity_problem


def load_raw(ndim: int, z_slice: int):
    raw = load_isbi_raw()
    return raw[z_slice] if ndim == 2 else raw


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run watershed oversegmentation + RAG lifted multicut on the ISBI "
            "affinity example."
        )
    )
    parser.add_argument("--ndim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--z-slice", type=int, default=0)
    parser.add_argument("--local-threshold", type=float, default=0.1)
    parser.add_argument("--lifted-threshold", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--watershed-min-distance", type=int, default=5)
    parser.add_argument("--watershed-grid-spacing", type=int, default=12)
    parser.add_argument("--max-markers", type=int, default=2048)
    parser.add_argument("--kl-outer-iterations", type=int, default=10)
    args = parser.parse_args()

    affinity_problem = load_affinity_problem(args.ndim, args.z_slice)
    raw = load_raw(args.ndim, args.z_slice)

    lifted_problem = build_lifted_problem(
        affinity_problem,
        local_threshold=args.local_threshold,
        lifted_threshold=args.lifted_threshold,
        number_of_threads=args.threads,
        watershed_min_distance=args.watershed_min_distance,
        watershed_grid_spacing=args.watershed_grid_spacing,
        max_markers=args.max_markers,
    )
    print(
        f"Built RAG: {lifted_problem.number_of_nodes} nodes, "
        f"{lifted_problem.local_costs.size} local edges, "
        f"{lifted_problem.lifted_uvs.shape[0]} lifted edges"
    )

    objective = bic.graph.LiftedMulticutObjective(
        lifted_problem.rag,
        lifted_problem.local_costs,
        lifted_uvs=lifted_problem.lifted_uvs,
        lifted_costs=lifted_problem.lifted_costs,
    )
    solver = bic.graph.LiftedChainedSolvers(
        [
            bic.graph.LiftedGreedyAdditiveMulticut(),
            bic.graph.LiftedKernighanLinMulticut(
                number_of_outer_iterations=args.kl_outer_iterations
            ),
        ]
    )
    node_labels = solver.optimize(objective)
    print(
        f"Lifted multicut energy: {objective.energy(node_labels):.3f}, "
        f"{int(node_labels.max()) + 1} segments"
    )

    segmentation = bic.graph.project_node_labels_to_pixels(
        lifted_problem.rag,
        lifted_problem.oversegmentation,
        node_labels,
        number_of_threads=args.threads,
    )

    viewer = napari.Viewer()
    viewer.add_image(raw, name="raw")
    viewer.add_image(affinity_problem.direct_affinities, name="direct affinities")
    viewer.add_image(
        affinity_problem.long_range_affinities, name="long-range affinities"
    )
    viewer.add_image(lifted_problem.heightmap, name="watershed heightmap")
    viewer.add_labels(
        lifted_problem.oversegmentation, name="watershed oversegmentation"
    )
    viewer.add_labels(segmentation, name="lifted multicut segmentation")
    viewer.add_labels(find_boundaries(segmentation), name="lifted multicut boundaries")
    napari.run()


if __name__ == "__main__":
    main()
