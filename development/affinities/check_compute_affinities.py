"""Cross-check bioimage-cpp's compute_affinities against affogato.

Benchmarks on the registered ISBI ground-truth segmentation volume
(30 × 512 × 512 = 7.86 M voxels, ~660 distinct labels) using the same
17-channel offset configuration that elf uses for mutex-watershed on this
data. Small enough to fit in memory, big enough that initialization and
allocation effects don't dominate.

Not part of the pytest suite (per CLAUDE.md: external-library comparisons
live under ``development/``, not under ``tests/``).
"""

from __future__ import annotations

import argparse
import sys
from statistics import mean, median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic
from bioimage_cpp._data import ISBI_AFFINITY_OFFSETS, load_isbi_gt_segmentation

try:
    import affogato.affinities as affo
except ImportError as error:  # pragma: no cover - dev script
    sys.stderr.write(f"affogato not installed: {error}\n")
    sys.exit(1)


# Subsets of ISBI_AFFINITY_OFFSETS. The "nearest neighbours" subset is the
# typical multicut input (3 channels); the "full 17" subset matches
# elf's mutex-watershed proposal generator and exercises long-range offsets.
OFFSET_SUBSETS = {
    "nearest": [(-1, 0, 0), (0, -1, 0), (0, 0, -1)],
    "full17": list(ISBI_AFFINITY_OFFSETS),
}


def time_call(fn, repeats):
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = fn()
        timings.append(perf_counter() - start)
    return timings, result


def run_case(labels, offsets, *, repeats, ignore_label=None):
    offsets_list = [list(offset) for offset in offsets]

    bic_timings, (bic_affs, bic_mask) = time_call(
        lambda: bic.affinities.compute_affinities(
            labels,
            offsets_list,
            ignore_label=ignore_label,
            return_mask=True,
            number_of_threads=1,
        ),
        repeats,
    )
    affo_timings, (affo_affs, affo_mask) = time_call(
        lambda: affo.compute_affinities(
            labels,
            offsets_list,
            have_ignore_label=ignore_label is not None,
            ignore_label=ignore_label if ignore_label is not None else 0,
        ),
        repeats,
    )

    return {
        "n_offsets": len(offsets_list),
        "ignore": ignore_label,
        "ok_affs": np.array_equal(bic_affs, affo_affs),
        "ok_mask": np.array_equal(bic_mask, affo_mask),
        "bic_median_s": median(bic_timings),
        "affo_median_s": median(affo_timings),
        "bic_mean_s": mean(bic_timings),
        "affo_mean_s": mean(affo_timings),
        "max_abs_diff": float(np.max(np.abs(bic_affs - affo_affs))) if bic_affs.shape == affo_affs.shape else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    labels = load_isbi_gt_segmentation(timeout=args.timeout)
    n_labels = int(labels.max()) + 1
    n_voxels = int(np.prod(labels.shape))
    print(
        f"labels: shape={labels.shape}, dtype={labels.dtype}, "
        f"n_voxels={n_voxels:,}, n_labels={n_labels}",
        flush=True,
    )

    rows = []
    for name, offsets in OFFSET_SUBSETS.items():
        for ig in (None, 0):
            row = run_case(labels, offsets, repeats=args.repeats, ignore_label=ig)
            row["offsets_name"] = name
            rows.append(row)

    print()
    print(
        f"{'offsets':>10} {'n':>3} {'ignore':>6} {'affs':>5} {'mask':>5}"
        f" {'bic_s':>9} {'affo_s':>9} {'speedup':>8}"
    )
    print("-" * 68)
    all_ok = True
    for r in rows:
        speedup = r["affo_median_s"] / r["bic_median_s"] if r["bic_median_s"] > 0 else float("inf")
        print(
            f"{r['offsets_name']:>10} {r['n_offsets']:>3d} {str(r['ignore']):>6}"
            f" {'OK' if r['ok_affs'] else 'FAIL':>5}"
            f" {'OK' if r['ok_mask'] else 'FAIL':>5}"
            f" {r['bic_median_s']:>9.4f} {r['affo_median_s']:>9.4f}"
            f" {speedup:>7.2f}x"
        )
        all_ok = all_ok and r["ok_affs"] and r["ok_mask"]

    if not all_ok:
        print("\nFAIL: output mismatch", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
