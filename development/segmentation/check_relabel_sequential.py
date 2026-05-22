"""Benchmark and equivalence check for relabel_sequential.

Compares bioimage_cpp.segmentation.relabel_sequential against:
- skimage.segmentation.relabel_sequential
- vigra.analysis.relabelConsecutive

Not part of the pytest suite. Run from the repository root, e.g.:

    python development/segmentation/check_relabel_sequential.py --repeats 5
"""

from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np

import bioimage_cpp as bic


def make_problem(shape: tuple[int, ...], n_labels: int, *, seed: int) -> np.ndarray:
    """Build a random label field with roughly ``n_labels`` distinct non-zero labels.

    The label values themselves are sparse — drawn from a range larger than
    ``n_labels`` — to exercise the sorted-unique remapping logic. About 10% of
    pixels are background (0).
    """
    rng = np.random.default_rng(seed)
    value_range = max(n_labels * 4, 16)
    label_field = rng.integers(1, value_range, size=shape, dtype=np.int64).astype(
        np.uint32
    )
    background_mask = rng.random(size=shape) < 0.1
    label_field[background_mask] = 0
    return np.ascontiguousarray(label_field)


def run_bioimage_cpp(label_field: np.ndarray) -> np.ndarray:
    relabeled, _, _ = bic.segmentation.relabel_sequential(label_field)
    return relabeled


def run_skimage(label_field: np.ndarray) -> np.ndarray:
    from skimage.segmentation import relabel_sequential as sk_relabel

    relabeled, _, _ = sk_relabel(label_field)
    return np.asarray(relabeled)


def run_vigra(label_field: np.ndarray) -> np.ndarray:
    import vigra

    # vigra.analysis.relabelConsecutive(labels, start_label=1, keep_zeros=True)
    # returns (relabeled, max_new_label, mapping_dict). The relabeled array is
    # the first element. We pass start_label=1 and keep_zeros=True to match
    # skimage's default behavior.
    relabeled, _, _ = vigra.analysis.relabelConsecutive(
        label_field.astype(np.uint32), start_label=1, keep_zeros=True
    )
    return np.asarray(relabeled)


def time_calls(
    fn: Callable[[np.ndarray], np.ndarray],
    label_field: np.ndarray,
    repeats: int,
) -> tuple[list[float], np.ndarray]:
    # warm-up
    result = fn(label_field)
    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        result = fn(label_field)
        timings.append(perf_counter() - start)
    return timings, result


def check_consecutive(relabeled: np.ndarray, *, offset: int = 1) -> bool:
    unique = np.unique(relabeled)
    non_pass = unique[unique >= offset]
    if non_pass.size == 0:
        return True
    return bool(np.array_equal(non_pass, np.arange(offset, offset + non_pass.size)))


def report_one(
    name: str,
    label_field: np.ndarray,
    repeats: int,
    bic_timings: list[float],
    bic_result: np.ndarray,
) -> None:
    print(f"\n=== {name} ===")
    print(
        f"shape={label_field.shape}, dtype={label_field.dtype}, "
        f"distinct labels in input={int(np.unique(label_field).size)}"
    )
    bic_median = median(bic_timings)
    print(f"bioimage-cpp median runtime: {bic_median * 1000:.3f} ms")
    print(
        f"bioimage-cpp produces consecutive labels: "
        f"{check_consecutive(bic_result)}"
    )

    try:
        sk_timings, sk_result = time_calls(run_skimage, label_field, repeats)
        sk_median = median(sk_timings)
        agrees = bool(np.array_equal(sk_result, bic_result))
        print(f"skimage median runtime: {sk_median * 1000:.3f} ms")
        print(f"skimage / bioimage-cpp ratio: {sk_median / bic_median:.3f}x")
        print(f"skimage and bioimage-cpp agree on relabeled array: {agrees}")
    except ImportError:
        print("skimage not available, skipping skimage comparison")

    try:
        vi_timings, vi_result = time_calls(run_vigra, label_field, repeats)
        vi_median = median(vi_timings)
        # vigra preserves first-occurrence order, not sorted order; we expect
        # disagreement on the specific label values but agreement on the
        # partition the labels induce.
        agrees = bool(np.array_equal(vi_result, bic_result))
        same_partition = bool(
            np.unique(vi_result).size == np.unique(bic_result).size
        )
        print(f"vigra median runtime: {vi_median * 1000:.3f} ms")
        print(f"vigra / bioimage-cpp ratio: {vi_median / bic_median:.3f}x")
        print(f"vigra and bioimage-cpp produce identical relabeled array: {agrees}")
        print(
            f"vigra and bioimage-cpp produce same number of unique labels: "
            f"{same_partition}"
        )
    except ImportError:
        print("vigra not available, skipping vigra comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cases = [
        ("2D small label set (1024x1024, ~100 labels)", (1024, 1024), 100),
        ("2D large label set (1024x1024, ~100000 labels)", (1024, 1024), 100_000),
        ("3D small label set (128x128x128, ~100 labels)", (128, 128, 128), 100),
        (
            "3D large label set (128x128x128, ~100000 labels)",
            (128, 128, 128),
            100_000,
        ),
    ]

    for name, shape, n_labels in cases:
        label_field = make_problem(shape, n_labels, seed=args.seed)
        bic_timings, bic_result = time_calls(
            run_bioimage_cpp, label_field, args.repeats
        )
        report_one(name, label_field, args.repeats, bic_timings, bic_result)


if __name__ == "__main__":
    main()
