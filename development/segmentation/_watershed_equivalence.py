"""Shared logic for the watershed correctness + runtime comparisons.

Builds a node-heightmap and seed markers from the cached ISBI affinity
volume, then runs ``bioimage_cpp.segmentation.watershed`` against
``skimage.segmentation.watershed`` (connectivity=1). Correctness uses
partition-comparison metrics (VI, rand index) rather than exact label
equality — tie-breaking differs between the two implementations.
"""

from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np


def load_problem():
    from bioimage_cpp._data import load_isbi_affinities

    affinities, offsets = load_isbi_affinities()
    return np.ascontiguousarray(affinities), [tuple(offset) for offset in offsets]


def _nearest_neighbour_channels(offsets):
    """Return indices of channels whose offset moves one step along a single axis."""
    return [
        i
        for i, offset in enumerate(offsets)
        if sum(1 for v in offset if v != 0) == 1
        and all(abs(v) <= 1 for v in offset)
    ]


def make_heightmap(affinities: np.ndarray, offsets: list[tuple[int, ...]]) -> np.ndarray:
    """Build a per-pixel heightmap from the nearest-neighbour affinity channels.

    Affinity ~ 1 means "neighbours are in the same object". Inverting and
    averaging the nearest-neighbour channels gives a smooth boundary map
    suitable as a watershed heightmap.
    """
    nn_channels = _nearest_neighbour_channels(offsets)
    if not nn_channels:
        raise ValueError("no nearest-neighbour affinity channels found")
    mean_aff = affinities[nn_channels].mean(axis=0).astype(np.float32, copy=True)
    return np.ascontiguousarray(1.0 - mean_aff)


def make_markers(heightmap: np.ndarray, *, smoothing_sigma: float = 1.5) -> np.ndarray:
    """Build seed markers as labelled connected components of the heightmap's local minima.

    A small Gaussian smoothing is applied before detecting minima so that
    affinity noise doesn't produce thousands of single-pixel seeds.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import local_minima
    from skimage.measure import label

    if smoothing_sigma > 0:
        smoothed = ndi.gaussian_filter(heightmap, sigma=smoothing_sigma)
    else:
        smoothed = heightmap
    minima = local_minima(smoothed)
    markers = label(minima, connectivity=1).astype(np.int32, copy=False)
    return np.ascontiguousarray(markers)


def prepare_2d_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    *,
    z: int,
    yx_shape: tuple[int, int],
    smoothing_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    channels_2d = [i for i, offset in enumerate(offsets) if offset[0] == 0]
    y, x = yx_shape
    cropped = affinities[channels_2d, z, :y, :x]
    offsets_2d = [offsets[i][1:] for i in channels_2d]
    heightmap = make_heightmap(np.ascontiguousarray(cropped), offsets_2d)
    markers = make_markers(heightmap, smoothing_sigma=smoothing_sigma)
    return heightmap, markers


def prepare_3d_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    *,
    zyx_shape: tuple[int, int, int],
    smoothing_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    z, y, x = zyx_shape
    cropped = affinities[:, :z, :y, :x]
    heightmap = make_heightmap(np.ascontiguousarray(cropped), offsets)
    markers = make_markers(heightmap, smoothing_sigma=smoothing_sigma)
    return heightmap, markers


def run_bioimage_cpp(heightmap: np.ndarray, markers: np.ndarray) -> np.ndarray:
    import bioimage_cpp as bic

    return bic.segmentation.watershed(heightmap, markers)


def run_skimage_reference(heightmap: np.ndarray, markers: np.ndarray) -> np.ndarray:
    from skimage.segmentation import watershed as sk_watershed

    return sk_watershed(heightmap, markers=markers, connectivity=1)


def _load_validation_metrics():
    try:
        from elf.validation import rand_index, variation_of_information

        return "elf.validation", rand_index, variation_of_information
    except ImportError:
        from elf.evaluation import rand_index, variation_of_information

        return "elf.evaluation", rand_index, variation_of_information


def compare_segmentations(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    min_rand_index: float = 0.99,
) -> dict[str, float | str | bool]:
    """Partition-style comparison.

    Exact label equality is not expected — tie-breaking on equal heights is
    implementation-defined for both watersheds, so boundary pixels around
    every region can move by 1–2 cells. We use Rand Index as the primary
    "do these partitions agree" check (which is the metric that copes
    gracefully with boundary jitter); VI and ARE are reported for context
    but not gated.
    """
    source, rand_index, variation_of_information = _load_validation_metrics()

    vi_split, vi_merge = variation_of_information(candidate, reference)
    adapted_rand_error, ri = rand_index(candidate, reference)
    exact_equal = bool(np.array_equal(candidate, reference))
    equivalent = ri >= min_rand_index
    metrics: dict[str, float | str | bool] = {
        "validation_source": source,
        "vi_split": float(vi_split),
        "vi_merge": float(vi_merge),
        "adapted_rand_error": float(adapted_rand_error),
        "rand_index": float(ri),
        "exact_label_equality": exact_equal,
        "equivalent": equivalent,
    }
    if not equivalent:
        print(
            f"WARNING: rand index {ri:.6g} below threshold {min_rand_index:.6g} — "
            "watershed partitions disagree substantially"
        )
    return metrics


def time_functions_interleaved(
    first: Callable[[np.ndarray, np.ndarray], np.ndarray],
    second: Callable[[np.ndarray, np.ndarray], np.ndarray],
    heightmap: np.ndarray,
    markers: np.ndarray,
    repeats: int,
) -> tuple[list[float], np.ndarray, list[float], np.ndarray]:
    def timed_call(run):
        start = perf_counter()
        result = run(heightmap, markers)
        return perf_counter() - start, result

    # Warm up imports, JIT/Cython compilation, allocator caches.
    first(heightmap, markers)
    second(heightmap, markers)

    first_timings: list[float] = []
    second_timings: list[float] = []
    first_result = None
    second_result = None
    for repeat in range(repeats):
        if repeat % 2 == 0:
            first_time, first_result = timed_call(first)
            second_time, second_result = timed_call(second)
        else:
            second_time, second_result = timed_call(second)
            first_time, first_result = timed_call(first)
        first_timings.append(first_time)
        second_timings.append(second_time)

    assert first_result is not None
    assert second_result is not None
    return first_timings, first_result, second_timings, second_result


def print_report(
    *,
    ndim: int,
    heightmap: np.ndarray,
    markers: np.ndarray,
    metrics: dict[str, float | str | bool],
    bic_timings: list[float],
    ref_timings: list[float],
):
    bic_median = median(bic_timings)
    ref_median = median(ref_timings)
    speedup = ref_median / bic_median if bic_median > 0 else float("inf")
    n_markers = int(markers.max())

    print(f"Watershed {ndim}D comparison")
    print(f"heightmap shape: {heightmap.shape}, dtype: {heightmap.dtype}")
    print(f"markers: {n_markers} seeds (dtype={markers.dtype})")
    print(f"validation metrics: {metrics['validation_source']}")
    print(
        "VI split/merge: "
        f"{metrics['vi_split']:.6g} / {metrics['vi_merge']:.6g}"
    )
    print(
        "adapted rand error / rand index: "
        f"{metrics['adapted_rand_error']:.6g} / {metrics['rand_index']:.6g}"
    )
    print(f"exact label equality: {metrics['exact_label_equality']}")
    print(f"within thresholds: {metrics['equivalent']}")
    print(f"bioimage-cpp median runtime: {bic_median:.6f} s")
    print(f"skimage reference median runtime: {ref_median:.6f} s")
    print(f"reference / bioimage-cpp runtime ratio: {speedup:.3f}x")


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
        heightmap, markers = prepare_2d_problem(
            affinities,
            offsets,
            z=z,
            yx_shape=yx_shape,
            smoothing_sigma=smoothing_sigma,
        )
    elif ndim == 3:
        heightmap, markers = prepare_3d_problem(
            affinities,
            offsets,
            zyx_shape=zyx_shape,
            smoothing_sigma=smoothing_sigma,
        )
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    ref_timings, ref_seg, bic_timings, bic_seg = time_functions_interleaved(
        run_skimage_reference,
        run_bioimage_cpp,
        heightmap,
        markers,
        repeats,
    )
    metrics = compare_segmentations(bic_seg, ref_seg)
    print_report(
        ndim=ndim,
        heightmap=heightmap,
        markers=markers,
        metrics=metrics,
        bic_timings=bic_timings,
        ref_timings=ref_timings,
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs for each implementation.",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma applied to the heightmap before finding local minima.",
    )
