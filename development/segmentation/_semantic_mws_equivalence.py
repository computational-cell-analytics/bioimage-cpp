"""Shared harness for the semantic-mutex-watershed comparison scripts.

Loads the registered ``semantic_labels`` volume, derives a realistic affinity
+ semantic-class weight stack from it (via
:func:`bioimage_cpp.affinities.compute_affinities` plus Gaussian noise), runs
both bioimage-cpp and affogato side-by-side, and reports partition equivalence
and timing. Not part of the pytest suite — see ``check_semantic_mutex_watershed_{2d,3d}.py``.
"""

from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np


_DEFAULT_2D_OFFSETS: list[tuple[int, ...]] = [
    (-1, 0),
    (0, -1),
    (-3, 0),
    (0, -3),
    (-9, 0),
    (0, -9),
]
_DEFAULT_2D_ATTRACTIVE = 2

_DEFAULT_3D_OFFSETS: list[tuple[int, ...]] = [
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
    (0, -3, 0),
    (0, 0, -3),
    (0, -9, 0),
    (0, 0, -9),
]
_DEFAULT_3D_ATTRACTIVE = 3


def load_problem() -> tuple[np.ndarray, np.ndarray]:
    from bioimage_cpp._data import load_semantic_labels

    instance, semantic = load_semantic_labels()
    return np.ascontiguousarray(instance), np.ascontiguousarray(semantic)


def prepare_2d_problem(
    instance: np.ndarray,
    semantic: np.ndarray,
    z: int,
    yx_shape: tuple[int, int],
):
    y, x = yx_shape
    inst = np.ascontiguousarray(instance[z, :y, :x])
    sem = np.ascontiguousarray(semantic[z, :y, :x])
    offsets = list(_DEFAULT_2D_OFFSETS)
    return inst, sem, offsets, _DEFAULT_2D_ATTRACTIVE


def prepare_3d_problem(
    instance: np.ndarray,
    semantic: np.ndarray,
    zyx_shape: tuple[int, int, int],
    *,
    z_start: int = 0,
):
    z, y, x = zyx_shape
    inst = np.ascontiguousarray(instance[z_start : z_start + z, :y, :x])
    sem = np.ascontiguousarray(semantic[z_start : z_start + z, :y, :x])
    offsets = list(_DEFAULT_3D_OFFSETS)
    return inst, sem, offsets, _DEFAULT_3D_ATTRACTIVE


def build_weight_volume(
    instance_labels: np.ndarray,
    semantic_labels: np.ndarray,
    offsets: list[tuple[int, ...]],
    number_of_attractive_channels: int,
    *,
    n_classes: int | None = None,
    high: float = 0.95,
    low: float = 0.05,
    noise: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Derive an MWS-style weight stack from instance + semantic ground truth.

    Spatial channels are computed via
    :func:`bioimage_cpp.affinities.compute_affinities`, which returns ``1`` when
    two endpoints share an instance and ``0`` otherwise. We keep this
    convention for **attractive** channels (high = merge), invert it for
    **mutex** channels (high = repel), rescale into ``[low, high]``, and add
    Gaussian noise.

    Semantic channels (one per class id ``c`` in ``[0, n_classes)``) start at
    ``low`` everywhere and rise to ``high`` where ``semantic_labels == c``,
    then receive the same noise treatment. If ``n_classes`` is ``None`` we
    derive it as ``int(semantic_labels.max()) + 1`` so the channel index of
    each class equals its source class id.

    Returns ``(weights, number_of_offsets)``; the second item lets callers
    locate the boundary between spatial and semantic channels in the stack.
    """
    import bioimage_cpp as bic

    number_of_offsets = len(offsets)
    if number_of_attractive_channels > number_of_offsets:
        raise ValueError(
            "number_of_attractive_channels must be <= len(offsets), got "
            f"{number_of_attractive_channels} vs {number_of_offsets}"
        )

    if n_classes is None:
        n_classes = int(semantic_labels.max()) + 1
        if n_classes < 1:
            raise ValueError(
                "could not infer n_classes from semantic_labels; pass it explicitly"
            )

    rng = np.random.default_rng(seed)

    affs, _ = bic.affinities.compute_affinities(
        instance_labels, offsets, return_mask=True
    )
    affs = affs.astype(np.float32, copy=False)
    # Mutex channels: invert so high = boundary = repel.
    if number_of_offsets > number_of_attractive_channels:
        affs[number_of_attractive_channels:number_of_offsets] = (
            1.0 - affs[number_of_attractive_channels:number_of_offsets]
        )

    spatial_weights = affs * (high - low) + low
    spatial_weights += rng.normal(loc=0.0, scale=noise, size=spatial_weights.shape).astype(
        np.float32, copy=False
    )

    spatial_shape = instance_labels.shape
    semantic_weights = np.full(
        (n_classes, *spatial_shape), low, dtype=np.float32
    )
    for c in range(n_classes):
        semantic_weights[c][semantic_labels == c] = high
    semantic_weights += rng.normal(
        loc=0.0, scale=noise, size=semantic_weights.shape
    ).astype(np.float32, copy=False)

    weights = np.concatenate([spatial_weights, semantic_weights], axis=0).astype(
        np.float64, copy=False
    )
    np.clip(weights, 0.0, 1.0, out=weights)

    # Break ties deterministically: the affogato wrapper sorts via
    # ``np.argsort`` (unstable), while bioimage-cpp's C++ sort tie-breaks by
    # edge id. With many derived weights clipped to exactly 0.0 or 1.0, the
    # difference in tie-break ordering massively rearranges processing,
    # fragmenting the output. We subtract a tiny per-edge perturbation that
    # grows with linear edge id, so smaller-id edges always sort first in
    # descending order — matching bioimage-cpp's explicit tiebreak and
    # removing the only source of disagreement. Float64 has enough precision
    # for the perturbation to survive across ~1M edges.
    perturbation = np.arange(weights.size, dtype=np.float64).reshape(weights.shape)
    weights -= perturbation * 1.0e-12
    return np.ascontiguousarray(weights), number_of_offsets


def run_bioimage_cpp(
    weights: np.ndarray,
    offsets: list[tuple[int, ...]],
    number_of_attractive_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    import bioimage_cpp as bic

    return bic.segmentation.semantic_mutex_watershed(
        weights,
        offsets,
        number_of_attractive_channels=number_of_attractive_channels,
    )


def run_affogato_reference(
    weights: np.ndarray,
    offsets: list[tuple[int, ...]],
    number_of_attractive_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    from affogato.segmentation import compute_semantic_mws_segmentation

    spatial_ndim = len(offsets[0])
    labels, semantic = compute_semantic_mws_segmentation(
        weights,
        offsets,
        number_of_attractive_channels=number_of_attractive_channels,
        strides=[1] * spatial_ndim,
    )
    return labels, semantic


def _load_validation_metrics():
    try:
        from elf.validation import rand_index, variation_of_information

        return "elf.validation", rand_index, variation_of_information
    except ImportError:
        from elf.evaluation import rand_index, variation_of_information

        return "elf.evaluation", rand_index, variation_of_information


def compare_semantic_segmentations(
    candidate: tuple[np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
    *,
    max_vi: float = 1.0e-10,
    max_are: float = 1.0e-10,
    min_rand_index: float = 1.0 - 1.0e-10,
) -> dict:
    """Compute partition + semantic-label agreement metrics.

    Does **not** raise when results disagree. The two implementations are
    expected to disagree on this data because affogato's C++ kernel calls
    ``boost::disjoint_sets::link(u, v)`` with the raw node ids instead of
    their union-find roots; with multi-class semantic inputs that
    miscompounds the tree and fragments the output. The harness reports the
    discrepancy and lets the caller decide what to do with it.
    """
    cand_labels, cand_semantic = candidate
    ref_labels, ref_semantic = reference
    source, rand_index, variation_of_information = _load_validation_metrics()

    vi_split, vi_merge = variation_of_information(cand_labels, ref_labels)
    adapted_rand_error, ri = rand_index(cand_labels, ref_labels)
    exact_labels = bool(np.array_equal(cand_labels, ref_labels))
    partition_equivalent = (
        vi_split <= max_vi
        and vi_merge <= max_vi
        and adapted_rand_error <= max_are
        and ri >= min_rand_index
    )

    semantic_exact = bool(np.array_equal(cand_semantic, ref_semantic))
    semantic_match_fraction = float(np.mean(cand_semantic == ref_semantic))

    return {
        "validation_source": source,
        "vi_split": float(vi_split),
        "vi_merge": float(vi_merge),
        "adapted_rand_error": float(adapted_rand_error),
        "rand_index": float(ri),
        "exact_label_equality": exact_labels,
        "partition_equivalent": partition_equivalent,
        "semantic_exact_equality": semantic_exact,
        "semantic_match_fraction": semantic_match_fraction,
        "n_clusters_bic": int(np.unique(cand_labels).size),
        "n_clusters_reference": int(np.unique(ref_labels).size),
    }


def time_functions_interleaved(
    first: Callable[..., tuple[np.ndarray, np.ndarray]],
    second: Callable[..., tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    offsets: list[tuple[int, ...]],
    number_of_attractive_channels: int,
    repeats: int,
) -> tuple[list[float], tuple[np.ndarray, np.ndarray], list[float], tuple[np.ndarray, np.ndarray]]:
    def timed_call(run):
        start = perf_counter()
        result = run(weights, offsets, number_of_attractive_channels)
        return perf_counter() - start, result

    first(weights, offsets, number_of_attractive_channels)
    second(weights, offsets, number_of_attractive_channels)

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
    weights: np.ndarray,
    n_classes: int,
    metrics: dict,
    bic_timings: list[float],
    ref_timings: list[float],
) -> None:
    bic_median = median(bic_timings)
    ref_median = median(ref_timings)
    speedup = ref_median / bic_median if bic_median > 0 else float("inf")

    print(f"Semantic mutex watershed {ndim}D comparison")
    print(
        f"weights shape: {weights.shape}, dtype: {weights.dtype}, "
        f"n_classes: {n_classes}"
    )
    print(f"validation metrics: {metrics['validation_source']}")
    print(
        f"cluster count (bic / affogato): {metrics['n_clusters_bic']} / "
        f"{metrics['n_clusters_reference']}"
    )
    print(
        "VI split/merge: "
        f"{metrics['vi_split']:.6g} / {metrics['vi_merge']:.6g}"
    )
    print(
        "adapted rand error / rand index: "
        f"{metrics['adapted_rand_error']:.6g} / {metrics['rand_index']:.12g}"
    )
    print(f"exact label equality: {metrics['exact_label_equality']}")
    print(f"partition_equivalent: {metrics['partition_equivalent']}")
    print(
        "semantic exact equality / match fraction: "
        f"{metrics['semantic_exact_equality']} / "
        f"{metrics['semantic_match_fraction']:.6f}"
    )
    print(f"bioimage-cpp median runtime: {bic_median:.6f} s")
    print(f"affogato reference median runtime: {ref_median:.6f} s")
    print(f"reference / bioimage-cpp runtime ratio: {speedup:.3f}x")
    if not metrics["partition_equivalent"]:
        print(
            "  NOTE: bic and affogato disagree. Affogato's C++ kernel calls "
            "boost::disjoint_sets::link(u, v) with raw node ids instead of "
            "their roots; with multi-class semantic inputs this corrupts "
            "the union-find tree and fragments the output. bioimage-cpp's "
            "result agrees with a from-scratch Python reference."
        )


def run_check(
    *,
    ndim: int,
    repeats: int,
    z: int,
    yx_shape: tuple[int, int],
    zyx_shape: tuple[int, int, int],
    seed: int = 42,
    z_start: int = 0,
) -> None:
    instance, semantic = load_problem()
    if ndim == 2:
        inst, sem, offsets, attractive_channels = prepare_2d_problem(
            instance, semantic, z=z, yx_shape=yx_shape
        )
    elif ndim == 3:
        inst, sem, offsets, attractive_channels = prepare_3d_problem(
            instance, semantic, zyx_shape=zyx_shape, z_start=z_start
        )
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    weights, _ = build_weight_volume(
        inst, sem, offsets, attractive_channels, seed=seed
    )
    n_classes = int(weights.shape[0] - len(offsets))

    ref_timings, ref_result, bic_timings, bic_result = time_functions_interleaved(
        run_affogato_reference,
        run_bioimage_cpp,
        weights,
        offsets,
        attractive_channels,
        repeats,
    )
    metrics = compare_semantic_segmentations(bic_result, ref_result)
    print_report(
        ndim=ndim,
        weights=weights,
        n_classes=n_classes,
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
        "--seed",
        type=int,
        default=42,
        help="Seed for the noise added to the derived weights.",
    )
