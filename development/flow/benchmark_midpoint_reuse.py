"""Benchmark the RK2 midpoint cell-reuse against adversarial flow magnitudes.

The RK2 midpoint moves ``0.5 * dt * |flow|`` voxels from the current position.
With the default ``dt=0.2`` the tracer's same-cell reuse branch is nearly
always taken for unit-magnitude flows (scale 1), becomes unpredictable around
scale 10 (midpoint displacement ~1 voxel), and is nearly always a reload for
larger scales. This script sweeps that range with synthetic random flows and
also times the registered fixtures, so the reuse can be compared against the
previous kernel by running the script under both builds.

Run::

    python development/flow/benchmark_midpoint_reuse.py --repeats 3
"""

from __future__ import annotations

import argparse
import sys
from statistics import median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic
from bioimage_cpp._data import load_flow_data


SCALES = (1.0, 5.0, 10.0, 20.0, 40.0)
SHAPE_3D = (32, 192, 192)
SHAPE_2D = (512, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RK2 midpoint cell-reuse on adversarial flow scales."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--skip-fixture",
        action="store_true",
        help="Only run the synthetic sweep (no registered-fixture download).",
    )
    return parser.parse_args()


def _time(flow: np.ndarray, mask: np.ndarray, repeats: int, threads: int) -> tuple[np.ndarray, list[float]]:
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = bic.flow.compute_flow_density(
            flow, mask, sigma=None, number_of_threads=threads
        )
        timings.append(perf_counter() - start)
    assert result is not None
    return result, timings


def _report(label: str, result: np.ndarray, timings: list[float]) -> None:
    print(
        f"{label}: median={median(timings):.4f}s min={min(timings):.4f}s "
        f"particles={int(result.sum())}"
    )


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2

    if not args.skip_fixture:
        for ndim in (2, 3):
            dist, fg, _ = load_flow_data(ndim, timeout=args.timeout)
            flow = np.ascontiguousarray(-dist, dtype=np.float32)
            mask = np.ascontiguousarray(fg > 0.5)
            result, timings = _time(flow, mask, args.repeats, args.threads)
            _report(f"fixture {ndim}D threads={args.threads}", result, timings)

    rng = np.random.default_rng(0)
    mask_3d = np.ones(SHAPE_3D, dtype=bool)
    mask_2d = np.ones(SHAPE_2D, dtype=bool)
    for scale in SCALES:
        flow = rng.normal(scale=scale, size=(3,) + SHAPE_3D).astype(np.float32)
        result, timings = _time(flow, mask_3d, args.repeats, args.threads)
        _report(f"sweep 3D scale={scale:g} threads={args.threads}", result, timings)

    flow = rng.normal(scale=10.0, size=(2,) + SHAPE_2D).astype(np.float32)
    result, timings = _time(flow, mask_2d, args.repeats, args.threads)
    _report(f"sweep 2D scale=10 threads={args.threads}", result, timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
