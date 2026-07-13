"""Benchmark the lockstep trajectory interleave against adversarial inputs.

The RK2 tracer processes K=3 particles per worker in lockstep
(``trace_particle_block``), so the out-of-order core can overlap the
independent trajectories' dependent chains. Two input families stress it:

- ``stripes``: lane-divergence worst case. Zero flow everywhere except every
  fourth x-column, which carries an in-plane swirl of magnitude
  ``2*tol/dt`` — those particles never converge and never leave the mask,
  while their group neighbours converge on the first step, so a lockstep
  group idles most of its lanes for all 50 iterations.
- ``random``: irregular trajectories with unpredictable per-lane branching
  (same generator as ``benchmark_midpoint_reuse.py``).

Compare kernels by running this script under two installed builds (swap the
prebuilt ``_core*.so`` between runs — rebuild rounds drift thermally).

Run::

    python development/flow/benchmark_interleave.py --repeats 3
"""

from __future__ import annotations

import argparse
import sys
from statistics import median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic
from bioimage_cpp._data import load_flow_data


SHAPE_3D = (32, 192, 192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the lockstep trajectory interleave."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--skip-fixture",
        action="store_true",
        help="Only run the synthetic cases (no registered-fixture download).",
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


def _stripe_flow(n_orbiter_columns_of_4: int) -> np.ndarray:
    """Swirling flow on every n-of-4 x-columns, zero elsewhere."""
    zz, yy, xx = np.indices(SHAPE_3D, dtype=np.float32)
    cy = (SHAPE_3D[1] - 1) / 2.0
    cx = (SHAPE_3D[2] - 1) / 2.0
    dy, dx = yy - cy, xx - cx
    radius = np.hypot(dy, dx)
    radius[radius == 0] = 1.0
    # |dt * step| = 0.01 >= tol = 0.005 at the defaults: orbiters never
    # converge; the rotation keeps them inside the mask for all iterations.
    speed = 0.05
    flow = np.stack(
        [np.zeros(SHAPE_3D, np.float32), -dx / radius * speed, dy / radius * speed]
    ).astype(np.float32)
    orbiter = (xx.astype(np.int64) % 4) < n_orbiter_columns_of_4
    flow *= orbiter[None]
    return flow


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

    mask = np.ones(SHAPE_3D, dtype=bool)
    for n_orbiters in (1, 3):
        flow = _stripe_flow(n_orbiters)
        result, timings = _time(flow, mask, args.repeats, args.threads)
        _report(
            f"stripes {n_orbiters}/4 long-lived threads={args.threads}",
            result,
            timings,
        )

    rng = np.random.default_rng(0)
    for scale in (1.0, 10.0):
        flow = rng.normal(scale=scale, size=(3,) + SHAPE_3D).astype(np.float32)
        result, timings = _time(flow, mask, args.repeats, args.threads)
        _report(f"random scale={scale:g} threads={args.threads}", result, timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
