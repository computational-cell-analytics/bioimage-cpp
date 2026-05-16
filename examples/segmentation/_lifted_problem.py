"""Shared lifted-multicut-from-affinities pipeline.

Used by `lifted_multicut_from_affinities.py` (visualization) and
`serialize_lifted_problem.py` (writes the problem to disk for the development
comparison scripts). Both scripts share the heightmap + watershed +
RAG + lifted-edge construction so they stay in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from skimage.feature import peak_local_max
from skimage.measure import label as label_components
from skimage.segmentation import watershed

import bioimage_cpp as bic
from bioimage_cpp._data import load_isbi_affinities


@dataclass
class AffinityProblem:
    """Affinity volume split into direct / long-range channels."""

    full_affinities: np.ndarray
    full_offsets: list[tuple[int, ...]]
    direct_affinities: np.ndarray
    direct_offsets: list[tuple[int, ...]]
    long_range_affinities: np.ndarray
    long_range_offsets: list[tuple[int, ...]]


@dataclass
class LiftedProblem:
    """Built lifted-multicut problem ready for solvers or serialization."""

    rag: bic.graph.RegionAdjacencyGraph
    oversegmentation: np.ndarray
    heightmap: np.ndarray
    local_costs: np.ndarray
    lifted_uvs: np.ndarray
    lifted_costs: np.ndarray

    @property
    def number_of_nodes(self) -> int:
        return int(self.rag.number_of_nodes)

    @property
    def local_uvs(self) -> np.ndarray:
        return self.rag.uv_ids()


def load_affinity_problem(
    ndim: int,
    z_slice: int,
) -> AffinityProblem:
    affinities, offsets = load_isbi_affinities()
    offsets = [tuple(int(v) for v in offset) for offset in offsets]
    if ndim == 2:
        channels_2d = [index for index, offset in enumerate(offsets) if offset[0] == 0]
        affinities = affinities[channels_2d, z_slice]
        offsets = [offsets[index][1:] for index in channels_2d]
    elif ndim != 3:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    direct_channels = [
        index for index, offset in enumerate(offsets)
        if sum(abs(v) for v in offset) == 1
    ]
    long_range_channels = [
        index for index in range(len(offsets)) if index not in direct_channels
    ]
    return AffinityProblem(
        full_affinities=np.ascontiguousarray(affinities, dtype=np.float32),
        full_offsets=offsets,
        direct_affinities=np.ascontiguousarray(
            affinities[direct_channels], dtype=np.float32
        ),
        direct_offsets=[offsets[index] for index in direct_channels],
        long_range_affinities=np.ascontiguousarray(
            affinities[long_range_channels], dtype=np.float32
        ),
        long_range_offsets=[offsets[index] for index in long_range_channels],
    )


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


def build_lifted_problem(
    affinity_problem: AffinityProblem,
    *,
    local_threshold: float,
    lifted_threshold: float,
    number_of_threads: int,
    watershed_min_distance: int,
    watershed_grid_spacing: int,
    max_markers: int,
) -> LiftedProblem:
    heightmap = make_heightmap(affinity_problem.direct_affinities)
    oversegmentation = make_watershed_oversegmentation(
        heightmap,
        min_distance=watershed_min_distance,
        grid_spacing=watershed_grid_spacing,
        max_markers=max_markers,
    )

    rag = bic.graph.region_adjacency_graph(
        oversegmentation, number_of_threads=number_of_threads
    )

    local_features = bic.graph.affinity_features(
        rag,
        oversegmentation,
        affinity_problem.direct_affinities,
        affinity_problem.direct_offsets,
        number_of_threads=number_of_threads,
    )
    local_costs = (local_threshold - local_features[:, 0]).astype(np.float64, copy=False)

    lifted_uvs = bic.graph.lifted_edges_from_affinities(
        rag,
        oversegmentation,
        affinity_problem.long_range_offsets,
        number_of_threads=number_of_threads,
    )
    if lifted_uvs.shape[0] == 0:
        lifted_costs = np.zeros(0, dtype=np.float64)
    else:
        lifted_features = bic.graph.lifted_affinity_features(
            oversegmentation,
            affinity_problem.long_range_affinities,
            affinity_problem.long_range_offsets,
            lifted_uvs,
            number_of_threads=number_of_threads,
        )
        lifted_costs = (lifted_threshold - lifted_features[:, 0]).astype(
            np.float64, copy=False
        )

    return LiftedProblem(
        rag=rag,
        oversegmentation=oversegmentation,
        heightmap=heightmap,
        local_costs=np.ascontiguousarray(local_costs),
        lifted_uvs=np.ascontiguousarray(lifted_uvs),
        lifted_costs=np.ascontiguousarray(lifted_costs),
    )
