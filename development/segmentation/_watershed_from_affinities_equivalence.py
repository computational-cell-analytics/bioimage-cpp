"""Shared logic for the affinity-watershed comparison scripts.

Runs ``bioimage_cpp.segmentation.watershed_from_affinities`` directly on the
nearest-neighbour ISBI affinity channels and compares against
``bioimage_cpp.segmentation.watershed`` running on the heightmap derived
from those same channels (``1 − mean(NN affinities)``). Both algorithms see
identical markers (local minima of the smoothed heightmap). Partition
agreement is reported but expected to be partial — the two algorithms have
deliberately different priority semantics (edge-keyed vs node-keyed) and the
purpose of this script is to surface that difference, not to gate it.
"""

from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np

# Reuse the existing equivalence helpers — heightmap + marker generation are
# identical to the node-watershed comparison.
from _watershed_equivalence import (
    _load_validation_metrics,
    load_problem,
    make_heightmap,
    make_markers,
)


def _nearest_neighbour_channels(offsets):
    return [
        i
        for i, offset in enumerate(offsets)
        if sum(1 for v in offset if v != 0) == 1
        and all(abs(v) <= 1 for v in offset)
    ]


def prepare_2d_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    *,
    z: int,
    yx_shape: tuple[int, int],
    smoothing_sigma: float,
) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray, np.ndarray]:
    """Return (nn_affinities_2d, nn_offsets_2d, heightmap, markers)."""
    channels_2d = [i for i, offset in enumerate(offsets) if offset[0] == 0]
    offsets_2d = [offsets[i][1:] for i in channels_2d]
    # Keep only nearest-neighbour channels for the affinity watershed input.
    nn_idx_within_2d = _nearest_neighbour_channels(offsets_2d)
    nn_channels_in_full = [channels_2d[i] for i in nn_idx_within_2d]
    nn_offsets_2d = [offsets_2d[i] for i in nn_idx_within_2d]
    y, x = yx_shape

    nn_affinities = np.ascontiguousarray(
        affinities[nn_channels_in_full, z, :y, :x]
    )

    # Reuse the existing heightmap/marker pipeline so the markers are
    # identical to the node-watershed scripts'.
    full_2d_affs = np.ascontiguousarray(affinities[channels_2d, z, :y, :x])
    heightmap = make_heightmap(full_2d_affs, offsets_2d)
    markers = make_markers(heightmap, smoothing_sigma=smoothing_sigma)
    return nn_affinities, nn_offsets_2d, heightmap, markers


def prepare_3d_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    *,
    zyx_shape: tuple[int, int, int],
    smoothing_sigma: float,
) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray, np.ndarray]:
    z, y, x = zyx_shape
    nn_channels = _nearest_neighbour_channels(offsets)
    nn_affinities = np.ascontiguousarray(
        affinities[nn_channels, :z, :y, :x]
    )
    nn_offsets = [offsets[i] for i in nn_channels]
    full_3d_affs = np.ascontiguousarray(affinities[:, :z, :y, :x])
    heightmap = make_heightmap(full_3d_affs, offsets)
    markers = make_markers(heightmap, smoothing_sigma=smoothing_sigma)
    return nn_affinities, nn_offsets, heightmap, markers


def run_from_affinities(
    nn_affinities: np.ndarray,
    nn_offsets: list[tuple[int, ...]],
    markers: np.ndarray,
) -> np.ndarray:
    import bioimage_cpp as bic

    return bic.segmentation.watershed_from_affinities(
        nn_affinities, nn_offsets, markers,
    )


def run_node_watershed(heightmap: np.ndarray, markers: np.ndarray) -> np.ndarray:
    import bioimage_cpp as bic

    return bic.segmentation.watershed(heightmap, markers)


def compare_segmentations(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float | str | bool]:
    source, rand_index, variation_of_information = _load_validation_metrics()

    vi_split, vi_merge = variation_of_information(candidate, reference)
    adapted_rand_error, ri = rand_index(candidate, reference)
    exact_equal = bool(np.array_equal(candidate, reference))
    return {
        "validation_source": source,
        "vi_split": float(vi_split),
        "vi_merge": float(vi_merge),
        "adapted_rand_error": float(adapted_rand_error),
        "rand_index": float(ri),
        "exact_label_equality": exact_equal,
    }


def time_runs(
    fn: Callable[[], np.ndarray],
    repeats: int,
) -> tuple[list[float], np.ndarray]:
    fn()  # warmup
    timings: list[float] = []
    last_result: np.ndarray | None = None
    for _ in range(repeats):
        start = perf_counter()
        last_result = fn()
        timings.append(perf_counter() - start)
    assert last_result is not None
    return timings, last_result


def print_report(
    *,
    ndim: int,
    nn_affinities: np.ndarray,
    markers: np.ndarray,
    metrics: dict[str, float | str | bool],
    aff_timings: list[float],
    node_timings: list[float],
):
    aff_median = median(aff_timings)
    node_median = median(node_timings)
    ratio = node_median / aff_median if aff_median > 0 else float("inf")
    n_markers = int(markers.max())

    print(f"Watershed-from-affinities {ndim}D comparison")
    print(
        f"NN-affinity shape: {nn_affinities.shape}, dtype: {nn_affinities.dtype}"
    )
    print(f"markers: {n_markers} seeds")
    print(f"validation metrics: {metrics['validation_source']}")
    print(
        "VI split/merge: "
        f"{metrics['vi_split']:.6g} / {metrics['vi_merge']:.6g}"
    )
    print(
        "adapted rand error / rand index: "
        f"{metrics['adapted_rand_error']:.6g} / {metrics['rand_index']:.6g}"
    )
    print(f"watershed_from_affinities median runtime: {aff_median:.6f} s")
    print(f"watershed (node-based) median runtime: {node_median:.6f} s")
    print(f"node-based / affinity-based runtime ratio: {ratio:.3f}x")


def run_check(
    *,
    ndim: int,
    repeats: int,
    z: int,
    yx_shape: tuple[int, int],
    zyx_shape: tuple[int, int, int],
    smoothing_sigma: float,
):
    affinities, offsets = load_problem()
    if ndim == 2:
        nn_affinities, nn_offsets, heightmap, markers = prepare_2d_problem(
            affinities, offsets, z=z, yx_shape=yx_shape,
            smoothing_sigma=smoothing_sigma,
        )
    elif ndim == 3:
        nn_affinities, nn_offsets, heightmap, markers = prepare_3d_problem(
            affinities, offsets, zyx_shape=zyx_shape,
            smoothing_sigma=smoothing_sigma,
        )
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    aff_timings, aff_labels = time_runs(
        lambda: run_from_affinities(nn_affinities, nn_offsets, markers),
        repeats,
    )
    node_timings, node_labels = time_runs(
        lambda: run_node_watershed(heightmap, markers),
        repeats,
    )

    metrics = compare_segmentations(aff_labels, node_labels)
    print_report(
        ndim=ndim,
        nn_affinities=nn_affinities,
        markers=markers,
        metrics=metrics,
        aff_timings=aff_timings,
        node_timings=node_timings,
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of timed runs for each implementation.",
    )
    parser.add_argument(
        "--smoothing-sigma", type=float, default=1.5,
        help="Gaussian sigma applied to the heightmap before finding local minima.",
    )
