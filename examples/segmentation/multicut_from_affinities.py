from __future__ import annotations

import argparse
from pathlib import Path

import napari
import numpy as np
from elf.io import open_file
from elf.segmentation.utils import load_mutex_watershed_problem
from skimage.feature import peak_local_max
from skimage.measure import label as label_components
from skimage.segmentation import find_boundaries, watershed

import bioimage_cpp as bic


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PREFIX = THIS_DIR / "isbi-data-"


def load_problem(data_prefix: Path, ndim: int, z_slice: int):
    affinities, offsets = load_mutex_watershed_problem(prefix=str(data_prefix))
    offsets = [tuple(int(v) for v in offset) for offset in offsets]
    if ndim == 2:
        channels_2d = [index for index, offset in enumerate(offsets) if offset[0] == 0]
        affinities = affinities[channels_2d, z_slice]
        offsets = [offsets[index][1:] for index in channels_2d]
    elif ndim != 3:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    direct_channels = [
        index for index, offset in enumerate(offsets) if sum(abs(v) for v in offset) == 1
    ]
    direct_affinities = np.ascontiguousarray(affinities[direct_channels], dtype=np.float32)
    direct_offsets = [offsets[index] for index in direct_channels]
    return direct_affinities, direct_offsets


def load_raw(data_prefix: Path, ndim: int, z_slice: int):
    data_path = data_prefix.with_name(data_prefix.name + "test.h5")
    with open_file(data_path, "r") as f:
        raw = f["raw"][z_slice] if ndim == 2 else f["raw"][:]
    return raw


def make_heightmap(affinities: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.mean(affinities, axis=0), dtype=np.float32)


def make_watershed_oversegmentation(
    heightmap: np.ndarray,
    *,
    min_distance: int,
    grid_spacing: int,
    max_markers: int,
) -> np.ndarray:
    coordinates = peak_local_max(
        -heightmap,
        min_distance=min_distance,
        exclude_border=False,
        num_peaks=max_markers,
    )
    marker_mask = np.zeros(heightmap.shape, dtype=bool)
    if len(coordinates) > 0:
        marker_mask[tuple(coordinates.T)] = True
    markers = label_components(marker_mask).astype(np.int32, copy=False)

    if int(markers.max()) < 2:
        markers = np.zeros(heightmap.shape, dtype=np.int32)
        slices = tuple(slice(None, None, grid_spacing) for _ in heightmap.shape)
        marker_coordinates = np.argwhere(np.ones(heightmap.shape, dtype=bool)[slices])
        marker_coordinates *= grid_spacing
        for marker_id, coord in enumerate(marker_coordinates, start=1):
            markers[tuple(coord)] = marker_id

    return watershed(heightmap, markers=markers).astype(np.uint32, copy=False)


def multicut_from_affinities(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    *,
    threshold: float,
    number_of_threads: int,
    watershed_min_distance: int,
    watershed_grid_spacing: int,
    max_markers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heightmap = make_heightmap(affinities)
    oversegmentation = make_watershed_oversegmentation(
        heightmap,
        min_distance=watershed_min_distance,
        grid_spacing=watershed_grid_spacing,
        max_markers=max_markers,
    )

    rag = bic.graph.region_adjacency_graph(
        oversegmentation, number_of_threads=number_of_threads
    )
    features = bic.graph.affinity_features(
        rag,
        oversegmentation,
        affinities,
        offsets,
        number_of_threads=number_of_threads,
    )
    edge_costs = threshold - features[:, 0]

    objective = bic.graph.MulticutObjective(rag, edge_costs)
    node_labels = bic.graph.ChainedMulticutSolvers(
        [
            bic.graph.GreedyAdditiveMulticut(),
            bic.graph.KernighanLinMulticut(number_of_outer_iterations=10),
        ]
    ).optimize(objective)
    segmentation = bic.graph.project_node_labels_to_pixels(
        rag,
        oversegmentation,
        node_labels,
        number_of_threads=number_of_threads,
    )
    return heightmap, oversegmentation, segmentation


def main():
    parser = argparse.ArgumentParser(
        description="Run watershed oversegmentation + RAG multicut on the ISBI affinity example."
    )
    parser.add_argument("--ndim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--z-slice", type=int, default=0)
    parser.add_argument("--data-prefix", type=Path, default=DEFAULT_DATA_PREFIX)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--watershed-min-distance", type=int, default=5)
    parser.add_argument("--watershed-grid-spacing", type=int, default=12)
    parser.add_argument("--max-markers", type=int, default=2048)
    args = parser.parse_args()

    affinities, offsets = load_problem(args.data_prefix, args.ndim, args.z_slice)
    raw = load_raw(args.data_prefix, args.ndim, args.z_slice)
    heightmap, oversegmentation, segmentation = multicut_from_affinities(
        affinities,
        offsets,
        threshold=args.threshold,
        number_of_threads=args.threads,
        watershed_min_distance=args.watershed_min_distance,
        watershed_grid_spacing=args.watershed_grid_spacing,
        max_markers=args.max_markers,
    )

    viewer = napari.Viewer()
    viewer.add_image(raw, name="raw")
    viewer.add_image(affinities, name="direct affinities")
    viewer.add_image(heightmap, name="watershed heightmap")
    viewer.add_labels(oversegmentation, name="watershed oversegmentation")
    viewer.add_labels(segmentation, name="multicut segmentation")
    viewer.add_labels(find_boundaries(segmentation), name="multicut boundaries")
    napari.run()


if __name__ == "__main__":
    main()
