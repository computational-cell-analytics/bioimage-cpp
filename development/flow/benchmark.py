"""Benchmark C++ flow tracing against the Python reference implementation.

Both implementations use the same sample data and parameters as
``development/flow/create_test_data.py``: ``n_iter=100``, ``dt=0.1`` and no
smoothing. The C++ implementation accepts a flow field, so this script passes
``-dist`` to it. The reference implementation accepts directed distances and
negates internally, so it receives the stored ``dist`` unchanged.

Run::

    python development/flow/benchmark.py --dim 2 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np

import bioimage_cpp as bic
from bioimage_cpp._data import load_flow_data

from _reference_impl import _compute_flow_density


N_ITER = 100
DT = 0.1
SIGMA = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark bioimage_cpp flow tracing vs the Python reference."
    )
    parser.add_argument("--dim", choices=("2", "3", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--small",
        action="store_true",
        help="Crop to a smaller array for quick smoke runs. Cropped results are not compared to stored density.",
    )
    return parser.parse_args()


def _dims(selection: str) -> list[int]:
    if selection == "both":
        return [2, 3]
    return [int(selection)]


def _crop_inputs(
    dist: np.ndarray,
    fg: np.ndarray,
    density: np.ndarray,
    ndim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ndim == 2:
        spatial = (slice(0, 160), slice(0, 160))
    else:
        spatial = (slice(0, 16), slice(0, 96), slice(0, 96))
    return (
        np.ascontiguousarray(dist[(slice(None),) + spatial]),
        np.ascontiguousarray(fg[spatial]),
        np.ascontiguousarray(density[spatial]),
    )


def _time_call(fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    start = perf_counter()
    result = fn()
    return result, perf_counter() - start


def _time_interleaved(
    ours_fn: Callable[[], np.ndarray],
    reference_fn: Callable[[], np.ndarray],
    repeats: int,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    ours = None
    reference = None
    for _ in range(warmup):
        ours = ours_fn()
        reference = reference_fn()

    ours_times = []
    ref_times = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            ours, elapsed = _time_call(ours_fn)
            ours_times.append(elapsed)
            reference, elapsed = _time_call(reference_fn)
            ref_times.append(elapsed)
        else:
            reference, elapsed = _time_call(reference_fn)
            ref_times.append(elapsed)
            ours, elapsed = _time_call(ours_fn)
            ours_times.append(elapsed)

    assert ours is not None
    assert reference is not None
    return ours, reference, ours_times, ref_times


def _benchmark_dim(ndim: int, repeats: int, warmup: int, timeout: float, small: bool) -> None:
    dist, fg, density = load_flow_data(ndim, timeout=timeout)
    if small:
        dist, fg, density = _crop_inputs(dist, fg, density, ndim)

    dist = np.ascontiguousarray(dist, dtype=np.float32)
    flow = np.ascontiguousarray(-dist, dtype=np.float32)
    mask = np.ascontiguousarray(fg > 0.5)

    def run_ours() -> np.ndarray:
        return bic.flow.compute_flow_density(flow, mask, n_iter=N_ITER, dt=DT, sigma=SIGMA)

    def run_reference() -> np.ndarray:
        return _compute_flow_density(dist, mask, n_iter=N_ITER, dt=DT, sigma=SIGMA, verbose=False)

    ours, reference, ours_times, ref_times = _time_interleaved(
        run_ours, run_reference, repeats, warmup
    )

    diff = np.abs(ours.astype(np.float64) - reference.astype(np.float64))
    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0

    ours_med = median(ours_times)
    ref_med = median(ref_times)
    ratio = ours_med / ref_med if ref_med > 0 else float("nan")

    print(f"\n== {ndim}D ==")
    print(f"shape={mask.shape}, foreground={int(mask.sum())}, repeats={repeats}, warmup={warmup}")
    print(f"bioimage_cpp median={ours_med:.4f}s min={min(ours_times):.4f}s")
    print(f"reference    median={ref_med:.4f}s min={min(ref_times):.4f}s")
    print(f"bioimage_cpp/reference median ratio={ratio:.3f}")
    print(f"max_abs_diff_vs_reference={max_diff:.6g}, mean_abs_diff_vs_reference={mean_diff:.6g}")
    if small:
        print("stored-density comparison skipped for cropped smoke run")
    else:
        stored_diff = np.abs(ours.astype(np.float64) - density.astype(np.float64))
        stored_max_diff = float(stored_diff.max()) if stored_diff.size else 0.0
        print(f"max_abs_diff_vs_stored_density={stored_max_diff:.6g}")


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("--warmup must be >= 0", file=sys.stderr)
        return 2

    for ndim in _dims(args.dim):
        _benchmark_dim(ndim, args.repeats, args.warmup, args.timeout, args.small)
    return 0


if __name__ == "__main__":
    sys.exit(main())
