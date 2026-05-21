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


SIGMA = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check bioimage_cpp.flow.compute_flow_density against stored reference density."
    )
    parser.add_argument("--dim", choices=("2", "3", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.15,
        help="Pass gate on mean(|ours-ref|)/mean(ref).",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--method", choices=("euler", "rk2"), default=None)
    parser.add_argument(
        "--restrict-to-mask",
        dest="restrict_to_mask",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _dims(selection: str) -> list[int]:
    if selection == "both":
        return [2, 3]
    return [int(selection)]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size == 0:
        return 1.0
    a -= a.mean()
    b -= b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den == 0.0:
        return 1.0
    return float((a * b).sum() / den)


def _time_ours(
    flow: np.ndarray,
    mask: np.ndarray,
    repeats: int,
    kwargs: dict,
) -> tuple[np.ndarray, list[float]]:
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = bic.flow.compute_flow_density(flow, mask, sigma=SIGMA, **kwargs)
        timings.append(perf_counter() - start)
    assert result is not None
    return result, timings


def _check_dim(ndim: int, args: argparse.Namespace) -> bool:
    dist, fg, reference = load_flow_data(ndim, timeout=args.timeout)
    flow = np.ascontiguousarray(-dist, dtype=np.float32)
    mask = np.ascontiguousarray(fg > 0.5)

    kwargs: dict = {"number_of_threads": args.threads}
    for name in ("n_iter", "dt", "tol", "method", "restrict_to_mask"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value

    result, timings = _time_ours(flow, mask, args.repeats, kwargs)

    diff = np.abs(result.astype(np.float64) - reference.astype(np.float64))
    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0
    ref_mean = float(reference.astype(np.float64).mean()) if reference.size else 0.0
    rel_diff = mean_diff / ref_mean if ref_mean > 0 else float("nan")
    pearson = _pearson(result, reference)
    ok = rel_diff <= args.rel_tol

    print(f"\n== {ndim}D ==  kwargs={kwargs}")
    print(f"shape={reference.shape}, foreground={int(mask.sum())}, repeats={args.repeats}")
    print(f"time median={median(timings):.4f}s min={min(timings):.4f}s")
    print(
        f"max_abs_diff={max_diff:.6g}, mean_abs_diff={mean_diff:.6g}, "
        f"rel_diff={rel_diff:.4f} (gate {args.rel_tol:g}), "
        f"pearson={pearson:.4f}"
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
            ok = _check_dim(ndim, args)
        except (FileNotFoundError, ModuleNotFoundError, RuntimeError) as error:
            print(f"{ndim}D SKIP: {error}", file=sys.stderr)
            ok = False
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
