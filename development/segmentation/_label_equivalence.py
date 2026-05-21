"""Shared logic for the connected-components correctness + runtime comparisons.

Builds a binary or integer label image, then runs
``bioimage_cpp.segmentation.label`` against ``skimage.measure.label`` and,
when available, ``vigra.analysis.labelMultiArrayWithBackground``. Correctness
uses partition equality: exact label integers may differ across
implementations, but the equivalence relation "do these two pixels share a
component" must agree everywhere.
"""

from __future__ import annotations

import argparse
import importlib.util
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #


def make_binary_problem(
    *,
    ndim: int,
    size: int,
    seed: int = 0,
    density: float = 0.5,
) -> np.ndarray:
    """Generate a binary mask from ``skimage.data.binary_blobs``."""
    from skimage.data import binary_blobs

    shape = (size,) * ndim
    image = binary_blobs(
        length=size,
        n_dim=ndim,
        volume_fraction=density,
        rng=seed,
    )
    assert image.shape == shape
    return image.astype(np.uint8, copy=True)


def make_multi_value_problem(
    *,
    ndim: int,
    size: int,
    seed: int = 1,
    n_values: int = 4,
) -> np.ndarray:
    """Generate a small-cardinality integer image (0..n_values-1)."""
    rng = np.random.default_rng(seed)
    shape = (size,) * ndim
    return rng.integers(low=0, high=n_values, size=shape, dtype=np.int32)


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


def run_bioimage_cpp(image: np.ndarray, *, connectivity: int) -> np.ndarray:
    import bioimage_cpp as bic

    return bic.segmentation.label(image, connectivity=connectivity)


def run_skimage(image: np.ndarray, *, connectivity: int) -> np.ndarray:
    from skimage.measure import label as sk_label

    return sk_label(image, connectivity=connectivity)


def vigra_available() -> bool:
    return importlib.util.find_spec("vigra") is not None


def _vigra_neighborhood(ndim: int, connectivity: int) -> str:
    """Map a ``connectivity`` value to vigra's neighborhood string.

    vigra exposes two neighborhood modes: ``"direct"`` (axis-aligned only)
    and ``"indirect"`` (all diagonals). It does not support the intermediate
    18-connectivity setting, so ``connectivity=2`` in 3D is approximated by
    falling back to ``"indirect"`` here — call sites that need a faithful
    comparison should restrict to ``connectivity`` in ``{1, ndim}``.
    """
    if connectivity == 1:
        return "direct"
    if connectivity == ndim:
        return "indirect"
    return "indirect"


def run_vigra(image: np.ndarray, *, connectivity: int) -> np.ndarray:
    import vigra

    neighborhood = _vigra_neighborhood(image.ndim, connectivity)
    # vigra wants a uint32 view; conversion is done outside the timed region
    # by the caller (see ``prepare_vigra_input``). This adapter assumes the
    # input is already uint32.
    return vigra.analysis.labelMultiArrayWithBackground(
        image, neighborhood=neighborhood, background_value=0
    )


def prepare_vigra_input(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image.astype(np.uint32, copy=False))


# --------------------------------------------------------------------------- #
# Partition-equality check
# --------------------------------------------------------------------------- #


def assert_same_partition(a: np.ndarray, b: np.ndarray) -> None:
    """Assert that ``a`` and ``b`` describe the same pixel partition.

    Raises ``AssertionError`` if a pair of pixels share a label in ``a`` but
    not in ``b`` (or vice versa). Exact label integers are allowed to differ.
    """
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    a_flat = a.ravel().tolist()
    b_flat = b.ravel().tolist()
    a_to_b: dict[int, int] = {}
    b_to_a: dict[int, int] = {}
    for av, bv in zip(a_flat, b_flat):
        if av in a_to_b:
            if a_to_b[av] != bv:
                raise AssertionError(
                    f"label {av} in a maps to multiple labels in b "
                    f"({a_to_b[av]} and {bv})"
                )
        else:
            a_to_b[av] = bv
        if bv in b_to_a:
            if b_to_a[bv] != av:
                raise AssertionError(
                    f"label {bv} in b maps to multiple labels in a "
                    f"({b_to_a[bv]} and {av})"
                )
        else:
            b_to_a[bv] = av


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def time_runs_interleaved(
    runners: dict[str, Callable[[], np.ndarray]],
    repeats: int,
) -> tuple[dict[str, list[float]], dict[str, np.ndarray]]:
    """Run each callable ``repeats`` times, interleaved, returning timings.

    Each callable is invoked once outside the timed loop as a warm-up.
    """
    for run in runners.values():
        run()

    timings: dict[str, list[float]] = {name: [] for name in runners}
    last_result: dict[str, np.ndarray] = {}
    names = list(runners)
    for repeat in range(repeats):
        # Alternate the order each repeat to spread caching effects.
        order = names if repeat % 2 == 0 else list(reversed(names))
        for name in order:
            start = perf_counter()
            result = runners[name]()
            elapsed = perf_counter() - start
            timings[name].append(elapsed)
            last_result[name] = result
    return timings, last_result


def print_timing_table(
    timings: dict[str, list[float]],
    reference_name: str = "skimage",
) -> None:
    print()
    print(f"{'implementation':<16}  {'median (ms)':>12}  {'speedup vs ref':>16}")
    ref_median = median(timings[reference_name]) if reference_name in timings else None
    for name, runs in timings.items():
        med_ms = median(runs) * 1000.0
        if ref_median is None or ref_median == 0:
            ratio_str = "-"
        else:
            ratio = ref_median / median(runs) if median(runs) > 0 else float("inf")
            ratio_str = f"{ratio:.3f}x"
        print(f"{name:<16}  {med_ms:>12.3f}  {ratio_str:>16}")


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def run_check(
    *,
    ndim: int,
    size: int,
    connectivity: int,
    repeats: int,
    density: float,
    problem_kind: str,
) -> None:
    if problem_kind == "binary":
        image = make_binary_problem(ndim=ndim, size=size, density=density)
    elif problem_kind == "multi":
        image = make_multi_value_problem(ndim=ndim, size=size)
    else:
        raise ValueError(f"unknown problem kind: {problem_kind}")

    print(
        f"Connected components {ndim}D comparison "
        f"(problem={problem_kind}, shape={image.shape}, dtype={image.dtype}, "
        f"connectivity={connectivity})"
    )

    bic_labels = run_bioimage_cpp(image, connectivity=connectivity)
    sk_labels = run_skimage(image, connectivity=connectivity)
    assert_same_partition(bic_labels, sk_labels)
    print(f"  partition matches skimage: yes ({int(bic_labels.max())} components)")

    runners: dict[str, Callable[[], np.ndarray]] = {
        "bioimage_cpp": lambda: run_bioimage_cpp(image, connectivity=connectivity),
        "skimage": lambda: run_skimage(image, connectivity=connectivity),
    }

    if vigra_available():
        # vigra's labelMultiArrayWithBackground groups together every
        # non-background pixel regardless of value, so it only matches
        # skimage/bioimage_cpp on binary problems. For multi-value images we
        # skip the vigra timing rather than report numbers that come from a
        # different problem definition. The supported connectivities are
        # "direct" (1) and "indirect" (ndim) — intermediate values fall back
        # to "indirect" inside the adapter and are skipped here too.
        if problem_kind == "binary" and connectivity in (1, ndim):
            vigra_input = prepare_vigra_input(image)
            vigra_labels = run_vigra(vigra_input, connectivity=connectivity)
            try:
                assert_same_partition(np.asarray(vigra_labels), sk_labels)
                print("  partition matches vigra:   yes")
                runners["vigra"] = lambda: run_vigra(
                    vigra_input, connectivity=connectivity
                )
            except AssertionError as error:
                print(f"  partition matches vigra:   NO ({error}) — vigra excluded from timing")
        else:
            print("  vigra excluded from timing for this problem/connectivity")
    else:
        print("  vigra not installed — skipping vigra timing")

    timings, _ = time_runs_interleaved(runners, repeats)
    print_timing_table(timings, reference_name="skimage")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed runs per implementation.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="Foreground volume fraction for the binary-blobs generator.",
    )
    parser.add_argument(
        "--problem",
        choices=("binary", "multi"),
        default="binary",
        help="Binary mask (skimage.data.binary_blobs) or multi-value integer image.",
    )
