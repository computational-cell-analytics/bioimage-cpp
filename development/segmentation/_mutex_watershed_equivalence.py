from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PREFIX = PROJECT_ROOT / "examples" / "segmentation" / "isbi-data-"


def load_problem(data_prefix: Path | str = DEFAULT_DATA_PREFIX):
    from elf.segmentation.utils import load_mutex_watershed_problem

    affinities, offsets = load_mutex_watershed_problem(prefix=str(data_prefix))
    return np.ascontiguousarray(affinities), [tuple(offset) for offset in offsets]


def prepare_2d_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    z: int,
    yx_shape: tuple[int, int],
):
    channels_2d = [i for i, offset in enumerate(offsets) if offset[0] == 0]
    y, x = yx_shape
    cropped = affinities[channels_2d, z, :y, :x]
    offsets_2d = [offsets[i][1:] for i in channels_2d]
    return np.ascontiguousarray(cropped), offsets_2d, 2


def prepare_3d_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    zyx_shape: tuple[int, int, int],
):
    z, y, x = zyx_shape
    cropped = affinities[:, :z, :y, :x]
    return np.ascontiguousarray(cropped), offsets, 3


def run_bioimage_cpp(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    number_of_attractive_channels: int,
) -> np.ndarray:
    import bioimage_cpp as bic

    affs = affinities.copy()
    affs[:number_of_attractive_channels] *= -1
    affs[:number_of_attractive_channels] += 1
    return bic.segmentation.mutex_watershed(
        affs,
        offsets,
        number_of_attractive_channels=number_of_attractive_channels,
    )


def run_affogato_reference(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    number_of_attractive_channels: int,
) -> np.ndarray:
    from elf.segmentation.mutex_watershed import mutex_watershed

    spatial_ndim = len(offsets[0])
    if number_of_attractive_channels != spatial_ndim:
        raise ValueError(
            "the elf mutex_watershed wrapper assumes one attractive channel per "
            f"spatial axis, got ndim={spatial_ndim}, attractive channels="
            f"{number_of_attractive_channels}"
        )
    return mutex_watershed(affinities.copy(), offsets, strides=[1] * spatial_ndim)


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
    max_vi: float = 1.0e-10,
    max_are: float = 1.0e-10,
    min_rand_index: float = 1.0 - 1.0e-10,
) -> dict[str, float | str | bool]:
    source, rand_index, variation_of_information = _load_validation_metrics()

    vi_split, vi_merge = variation_of_information(candidate, reference)
    adapted_rand_error, ri = rand_index(candidate, reference)
    exact_equal = bool(np.array_equal(candidate, reference))
    equivalent = (
        vi_split <= max_vi
        and vi_merge <= max_vi
        and adapted_rand_error <= max_are
        and ri >= min_rand_index
    )
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
        raise AssertionError(
            "mutex watershed results differ: "
            f"VI split={vi_split:.6g}, VI merge={vi_merge:.6g}, "
            f"adapted rand error={adapted_rand_error:.6g}, "
            f"rand index={ri:.12g}, exact labels={exact_equal}"
        )
    return metrics


def time_function(
    run: Callable[[np.ndarray, list[tuple[int, ...]], int], np.ndarray],
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    number_of_attractive_channels: int,
    repeats: int,
) -> tuple[list[float], np.ndarray]:
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = run(affinities, offsets, number_of_attractive_channels)
        timings.append(perf_counter() - start)
    assert result is not None
    return timings, result


def print_report(
    *,
    ndim: int,
    affinities: np.ndarray,
    metrics: dict[str, float | str | bool],
    bic_timings: list[float],
    ref_timings: list[float],
):
    bic_median = median(bic_timings)
    ref_median = median(ref_timings)
    speedup = ref_median / bic_median if bic_median > 0 else float("inf")

    print(f"Mutex watershed {ndim}D equivalence check")
    print(f"affinities shape: {affinities.shape}, dtype: {affinities.dtype}")
    print(f"validation metrics: {metrics['validation_source']}")
    print(
        "VI split/merge: "
        f"{metrics['vi_split']:.6g} / {metrics['vi_merge']:.6g}"
    )
    print(
        "adapted rand error / rand index: "
        f"{metrics['adapted_rand_error']:.6g} / {metrics['rand_index']:.12g}"
    )
    print(f"exact label equality: {metrics['exact_label_equality']}")
    print(f"bioimage-cpp median runtime: {bic_median:.6f} s")
    print(f"affogato reference median runtime: {ref_median:.6f} s")
    print(f"reference / bioimage-cpp runtime ratio: {speedup:.3f}x")


def run_check(
    *,
    ndim: int,
    repeats: int,
    data_prefix: Path | str,
    z: int,
    yx_shape: tuple[int, int],
    zyx_shape: tuple[int, int, int],
):
    affinities, offsets = load_problem(data_prefix)
    if ndim == 2:
        affs, used_offsets, attractive_channels = prepare_2d_problem(
            affinities, offsets, z=z, yx_shape=yx_shape
        )
    elif ndim == 3:
        affs, used_offsets, attractive_channels = prepare_3d_problem(
            affinities, offsets, zyx_shape=zyx_shape
        )
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    ref_timings, ref_seg = time_function(
        run_affogato_reference, affs, used_offsets, attractive_channels, repeats
    )
    bic_timings, bic_seg = time_function(
        run_bioimage_cpp, affs, used_offsets, attractive_channels, repeats
    )
    metrics = compare_segmentations(bic_seg, ref_seg)
    print_report(
        ndim=ndim,
        affinities=affs,
        metrics=metrics,
        bic_timings=bic_timings,
        ref_timings=ref_timings,
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-prefix",
        type=Path,
        default=DEFAULT_DATA_PREFIX,
        help=(
            "Path prefix for the ISBI mutex watershed data. The loader expects "
            "'test.h5' and 'train.h5' suffixes."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs for each implementation.",
    )
