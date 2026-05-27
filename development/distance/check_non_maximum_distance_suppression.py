"""Cross-check bioimage-cpp's non_maximum_distance_suppression against nifty.

Builds random binary masks, computes their Euclidean distance transform, picks
candidate points by thresholding the distance map, and compares
``bic.distance.non_maximum_distance_suppression`` against
``nifty.filters.nonMaximumDistanceSuppression`` for 2D and 3D inputs. Reports
both correctness (set + row order) and per-call runtime.

Not part of the pytest suite; requires nifty and scipy.
"""

from __future__ import annotations

import argparse
import sys
from statistics import median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic

try:
    from nifty.filters import nonMaximumDistanceSuppression
except ImportError as error:  # pragma: no cover - dev script
    sys.stderr.write(f"nifty not installed: {error}\n")
    sys.exit(1)

try:
    from scipy.ndimage import distance_transform_edt
except ImportError as error:  # pragma: no cover - dev script
    sys.stderr.write(f"scipy not installed: {error}\n")
    sys.exit(1)


CASES = [
    # (name, shape, foreground_fraction, threshold)
    ("2d_small", (60, 60), 0.85, 2.0),
    ("2d_large", (256, 256), 0.9, 3.0),
    ("3d_small", (25, 25, 25), 0.85, 2.0),
    ("3d_large", (40, 40, 40), 0.9, 3.0),
]


def time_call(fn, repeats):
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = fn()
        timings.append(perf_counter() - start)
    return median(timings), result


def run_case(name, shape, fg_fraction, threshold, n_trials, repeats, rng):
    rows = []
    for trial in range(n_trials):
        mask = rng.random(shape) < fg_fraction
        dm = distance_transform_edt(mask).astype(np.float32)
        coords = np.argwhere(dm > threshold).astype(np.uint64)
        if len(coords) == 0:
            continue

        ref_s, ref = time_call(
            lambda: nonMaximumDistanceSuppression(dm, coords), repeats
        )
        ours_s, ours = time_call(
            lambda: bic.distance.non_maximum_distance_suppression(dm, coords), repeats
        )

        exact = ref.shape == ours.shape and np.array_equal(ref, ours)
        same_set = {tuple(r) for r in ref.tolist()} == {tuple(r) for r in ours.tolist()}
        rows.append(
            {
                "case": name,
                "trial": trial,
                "n_points": len(coords),
                "n_ref": len(ref),
                "n_ours": len(ours),
                "set_ok": same_set,
                "order_ok": exact,
                "ref_s": ref_s,
                "ours_s": ours_s,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    all_rows = []
    for name, shape, fg, thr in CASES:
        all_rows.extend(
            run_case(name, shape, fg, thr, args.trials, args.repeats, rng)
        )

    header = (
        f"{'case':>10} {'trial':>5} {'n_pts':>7} {'n_ref':>6} {'n_ours':>6}"
        f" {'set':>5} {'order':>6} {'nifty_ms':>9} {'bic_ms':>9} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    all_ok = True
    speedups = []
    for r in all_rows:
        speedup = r["ref_s"] / r["ours_s"] if r["ours_s"] > 0 else float("nan")
        speedups.append(speedup)
        print(
            f"{r['case']:>10} {r['trial']:>5d} {r['n_points']:>7d}"
            f" {r['n_ref']:>6d} {r['n_ours']:>6d}"
            f" {'OK' if r['set_ok'] else 'FAIL':>5}"
            f" {'OK' if r['order_ok'] else 'FAIL':>6}"
            f" {r['ref_s'] * 1e3:>9.3f} {r['ours_s'] * 1e3:>9.3f}"
            f" {speedup:>7.2f}x"
        )
        all_ok = all_ok and r["set_ok"] and r["order_ok"]

    finite = [s for s in speedups if np.isfinite(s)]
    if finite:
        geo_mean = float(np.exp(np.mean(np.log(finite))))
        print(
            f"\nSpeedup (bic vs nifty): min {min(finite):.2f}x, "
            f"max {max(finite):.2f}x, geo-mean {geo_mean:.2f}x"
        )

    if not all_ok:
        print("\nFAIL: output mismatch vs nifty", file=sys.stderr)
        sys.exit(1)
    print("All cases match nifty (set and row order).")


if __name__ == "__main__":
    main()
