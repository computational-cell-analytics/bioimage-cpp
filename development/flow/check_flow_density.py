"""Validate bioimage_cpp flow tracing on the registered sample data.

The sample files store directed distances under ``dist``. They point toward
boundaries, so this script negates them before calling
``bioimage_cpp.flow.compute_flow_density``.

Run::

    python development/flow/check_flow_density.py --dim 2
"""

from __future__ import annotations

import argparse
import sys
from statistics import median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic
from bioimage_cpp._data import load_flow_data


N_ITER = 100
DT = 0.1
SIGMA = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check bioimage_cpp.flow.compute_flow_density against stored reference density."
    )
    parser.add_argument("--dim", choices=("2", "3", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def _dims(selection: str) -> list[int]:
    if selection == "both":
        return [2, 3]
    return [int(selection)]


def _time_ours(flow: np.ndarray, mask: np.ndarray, repeats: int) -> tuple[np.ndarray, list[float]]:
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = bic.flow.compute_flow_density(flow, mask, n_iter=N_ITER, dt=DT, sigma=SIGMA)
        timings.append(perf_counter() - start)
    assert result is not None
    return result, timings


def _check_dim(ndim: int, repeats: int, atol: float, timeout: float) -> bool:
    dist, fg, reference = load_flow_data(ndim, timeout=timeout)
    flow = np.ascontiguousarray(-dist, dtype=np.float32)
    mask = np.ascontiguousarray(fg > 0.5)
    result, timings = _time_ours(flow, mask, repeats)

    diff = np.abs(result.astype(np.float64) - reference.astype(np.float64))
    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0
    exact_fraction = float(np.mean(result == reference)) if result.size else 1.0
    ok = max_diff <= atol

    print(f"\n== {ndim}D ==")
    print(f"shape={reference.shape}, foreground={int(mask.sum())}, repeats={repeats}")
    print(f"time median={median(timings):.4f}s min={min(timings):.4f}s")
    print(
        f"max_abs_diff={max_diff:.6g}, mean_abs_diff={mean_diff:.6g}, "
        f"exact_fraction={exact_fraction:.6f}, atol={atol:g}"
    )
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2

    all_ok = True
    for ndim in _dims(args.dim):
        try:
            ok = _check_dim(ndim, args.repeats, args.atol, args.timeout)
        except (FileNotFoundError, ModuleNotFoundError, RuntimeError) as error:
            print(f"{ndim}D SKIP: {error}", file=sys.stderr)
            ok = False
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
