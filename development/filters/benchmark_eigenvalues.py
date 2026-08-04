"""Benchmark the scalar and automatic 3 x 3 eigensolver backends.

Run this script after an editable build:

    python development/filters/benchmark_eigenvalues.py
"""

from __future__ import annotations

import argparse
import csv
import os
from statistics import median
from time import perf_counter

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the internal batch eigensolver."
    )
    parser.add_argument("--matrices", type=int, default=60 * 256 * 256)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--minimum-speedup", type=float, default=3.0)
    parser.add_argument("--csv", default=None)
    return parser.parse_args()


def set_scalar_backend(force_scalar: bool) -> None:
    if force_scalar:
        os.environ["BIOIMAGE_CPP_FILTERS_FORCE_SCALAR"] = "1"
    else:
        os.environ.pop("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR", None)


def run_once(components: np.ndarray, out: np.ndarray) -> None:
    from bioimage_cpp import _core

    _core._filters_ev3_symmetric_float32(components, out)


def measure(
    components: np.ndarray, out: np.ndarray, repeats: int, force_scalar: bool
) -> tuple[str, list[float], np.ndarray]:
    from bioimage_cpp import _core

    set_scalar_backend(force_scalar)
    backend = _core._filters_eigenvalue_backend()
    run_once(components, out)
    times = []
    for _ in range(repeats):
        start = perf_counter()
        run_once(components, out)
        times.append(perf_counter() - start)
    return backend, times, out.copy()


def main() -> int:
    args = parse_args()
    if args.matrices < 8:
        raise ValueError("--matrices must be at least 8")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    rng = np.random.default_rng(args.seed)
    components = rng.normal(size=(6, args.matrices)).astype(np.float32)
    out = np.empty((args.matrices, 3), dtype=np.float32)

    original_force_scalar = os.environ.get("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR")
    try:
        automatic_backend, automatic_times, automatic = measure(
            components, out, args.repeats, False
        )
        scalar_backend, scalar_times, scalar = measure(
            components, out, args.repeats, True
        )
    finally:
        if original_force_scalar is None:
            os.environ.pop("BIOIMAGE_CPP_FILTERS_FORCE_SCALAR", None)
        else:
            os.environ["BIOIMAGE_CPP_FILTERS_FORCE_SCALAR"] = original_force_scalar

    automatic_median = median(automatic_times)
    scalar_median = median(scalar_times)
    speedup = scalar_median / automatic_median
    max_difference = float(np.max(np.abs(automatic - scalar)))
    scale = np.maximum(
        np.max(np.abs(scalar), axis=1), np.finfo(np.float32).tiny
    )
    max_scaled_difference = float(
        np.max(np.max(np.abs(automatic - scalar), axis=1) / scale)
    )

    print(
        f"matrices={args.matrices}, repeats={args.repeats}, "
        f"automatic_backend={automatic_backend}"
    )
    print(
        f"automatic: median={automatic_median * 1e3:.3f} ms, "
        f"min={min(automatic_times) * 1e3:.3f} ms"
    )
    print(
        f"{scalar_backend}: median={scalar_median * 1e3:.3f} ms, "
        f"min={min(scalar_times) * 1e3:.3f} ms"
    )
    print(f"scalar / automatic speedup: {speedup:.3f}x")
    print(f"maximum absolute difference: {max_difference:.3e}")
    print(f"maximum scaled difference: {max_scaled_difference:.3e}")

    if args.csv is not None:
        with open(args.csv, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "backend",
                    "matrices",
                    "median_s",
                    "min_s",
                    "repeats",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "backend": automatic_backend,
                    "matrices": args.matrices,
                    "median_s": automatic_median,
                    "min_s": min(automatic_times),
                    "repeats": args.repeats,
                }
            )
            writer.writerow(
                {
                    "backend": scalar_backend,
                    "matrices": args.matrices,
                    "median_s": scalar_median,
                    "min_s": min(scalar_times),
                    "repeats": args.repeats,
                }
            )
        print(f"wrote {args.csv}")

    passed = (
        np.isfinite(automatic).all()
        and np.isfinite(scalar).all()
        and max_scaled_difference < 5e-5
    )
    if automatic_backend == "avx2":
        passed = passed and speedup >= args.minimum_speedup
    else:
        print("The automatic build does not provide the AVX2 backend.")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
