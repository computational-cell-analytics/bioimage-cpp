"""Throughput benchmark of bioimage_cpp.filters against fastfilters, vigra,
and scipy.ndimage on 2D and 3D test data from ``skimage.data``.

Run::

    python development/filters/check_parity.py    # gate: run me first
    python development/filters/benchmark.py [--small] [--sigma 1.5] [--repeats 5]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from statistics import geometric_mean

from _bench_utils import (
    BenchConfig,
    FILTERS,
    LIBRARIES,
    build_adapters,
    format_results_table,
    load_2d,
    load_3d,
    time_interleaved,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark bioimage_cpp.filters vs fastfilters / vigra / scipy."
    )
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--inner-sigma", type=float, default=1.0)
    parser.add_argument("--outer-sigma", type=float, default=2.0)
    parser.add_argument("--window-size", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--small", action="store_true",
                        help="Crop to fast sizes for a smoke run.")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--no-2d", action="store_true")
    parser.add_argument(
        "--filters", default=",".join(FILTERS),
        help="Comma-separated subset of filters to benchmark.",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Optional path to write per-(filter, dim, library) median+min "
             "timings as CSV.",
    )
    return parser.parse_args()


def _load_test_data(args) -> list[tuple[str, "np.ndarray"]]:
    crop_2d = (128, 128) if args.small else None
    crop_3d = (32, 64, 64) if args.small else None
    targets = []
    if not args.no_2d:
        targets.append(("2D", load_2d(crop=crop_2d)))
    if not args.no_3d:
        targets.append(("3D", load_3d(crop=crop_3d)))
    return targets


def main() -> int:
    args = parse_args()
    cfg = BenchConfig(
        sigma=args.sigma,
        inner_sigma=args.inner_sigma,
        outer_sigma=args.outer_sigma,
        window_size=args.window_size,
    )
    requested = [f.strip() for f in args.filters.split(",") if f.strip()]
    unknown = [f for f in requested if f not in FILTERS]
    if unknown:
        print(f"unknown filter(s): {unknown}", file=sys.stderr)
        return 2

    targets = _load_test_data(args)

    print(
        f"sigma={cfg.sigma}, inner={cfg.inner_sigma}, outer={cfg.outer_sigma}, "
        f"window_size={cfg.window_size}, repeats={args.repeats}"
    )

    rows = []
    csv_rows = []
    for dim_label, image in targets:
        for filter_name in requested:
            adapters = build_adapters(filter_name, cfg)
            timed = {lib: fn for lib, fn in adapters.items() if fn is not None}
            if not timed:
                continue
            results = time_interleaved(timed, image, repeats=args.repeats)
            full_results = {lib: results.get(lib) for lib in LIBRARIES}
            row = {
                "filter": filter_name,
                "dim": dim_label,
                "shape": str(tuple(image.shape)),
                "results": full_results,
            }
            rows.append(row)
            for lib, r in full_results.items():
                if r is None:
                    continue
                csv_rows.append({
                    "filter": filter_name,
                    "dim": dim_label,
                    "shape": tuple(image.shape),
                    "library": lib,
                    "median_s": r["median"],
                    "min_s": r["min"],
                    "repeats": args.repeats,
                })

    print()
    print(format_results_table(rows))

    # Headline ratios across all benched (filter, dim) combinations.
    print()
    _print_headline_ratios(rows)

    if args.csv is not None:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["filter", "dim", "shape", "library", "median_s", "min_s", "repeats"]
            )
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        print(f"wrote {args.csv}")

    return 0


def _print_headline_ratios(rows: list[dict]) -> None:
    """Print geometric-mean wall-time ratios.

    Each ratio is ``bioimage_cpp.median / other.median``: values above 1.0
    mean we are slower than the other library; values below 1.0 mean we are
    faster.
    """

    def gm_ratio(other: str) -> tuple[float | None, int]:
        ratios = []
        for row in rows:
            ours = row["results"].get("bioimage_cpp")
            them = row["results"].get(other)
            if ours and them and them["median"] > 0:
                ratios.append(ours["median"] / them["median"])
        return (geometric_mean(ratios) if ratios else None), len(ratios)

    others = ["fastfilters", "vigra", "scipy"]
    print("speedup summary (geomean of bioimage_cpp.median / other.median; "
          ">1.0 means bioimage_cpp slower, <1.0 means faster):")
    for other in others:
        gm, n = gm_ratio(other)
        if gm is None:
            continue
        print(f"  bioimage_cpp / {other:<12s}  geomean = {gm:.3f}  (n={n})")


if __name__ == "__main__":
    sys.exit(main())
