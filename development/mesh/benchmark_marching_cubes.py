"""Benchmark bioimage-cpp marching cubes against scikit-image.

The benchmark runs a direct parity preflight before timing every workload.

Examples
--------
python development/mesh/benchmark_marching_cubes.py --size medium --repeats 7
python development/mesh/benchmark_marching_cubes.py --size large --method lorensen
"""

from __future__ import annotations

import argparse
import json
import platform
from statistics import median
import sys
from time import perf_counter

import numpy as np
import skimage

import bioimage_cpp as bic

from _marching_cubes_reference import assert_mesh_matches, reference_marching_cubes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--method", choices=("lewiner", "lorensen", "all"), default="all")
    parser.add_argument(
        "--workload",
        choices=("binary_sphere", "dense_binary_mask", "scalar_field", "all"),
        default="all",
    )
    parser.add_argument(
        "--backend", choices=("both", "bic", "skimage"), default="both"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--json", default="", help="optional JSON result path")
    parser.add_argument("--baseline", default="", help="optional prior benchmark JSON for bic relative changes")
    return parser.parse_args()


def shape_for(size: str) -> tuple[int, int, int]:
    return {"small": (48, 48, 48), "medium": (96, 96, 96), "large": (160, 160, 160)}[size]


def workloads(shape: tuple[int, int, int], seed: int):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = np.asarray(shape, dtype=np.float32) / 2.0
    radius = min(shape) * 0.28
    sphere = ((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2 < radius**2).astype(np.uint8)
    rng = np.random.default_rng(seed)
    dense_mask = (rng.random(shape) < 0.10).astype(np.uint8)
    zf = z.astype(np.float32) / max(shape[0] - 1, 1)
    yf = y.astype(np.float32) / max(shape[1] - 1, 1)
    xf = x.astype(np.float32) / max(shape[2] - 1, 1)
    scalar = (
        np.sin(4.0 * np.pi * zf)
        + np.cos(6.0 * np.pi * yf)
        + np.sin(5.0 * np.pi * xf)
    ).astype(np.float32)
    return [
        ("binary_sphere", sphere, 0.5),
        ("dense_binary_mask", dense_mask, 0.5),
        ("scalar_field", scalar, 0.0),
    ]


def time_call(function, repeats: int, warmup: int, batches: int) -> dict[str, object]:
    raw_timings = []
    batch_medians = []
    for _ in range(batches):
        for _ in range(warmup):
            function()
        timings = []
        for _ in range(repeats):
            start = perf_counter()
            function()
            timings.append(perf_counter() - start)
        raw_timings.extend(timings)
        batch_medians.append(median(timings))
    return {
        "raw_s": raw_timings,
        "batch_medians_s": batch_medians,
        "median_s": median(batch_medians),
        "p10_s": float(np.percentile(raw_timings, 10)),
        "p90_s": float(np.percentile(raw_timings, 90)),
        "min_s": min(raw_timings),
    }


def load_baseline(path: str) -> dict[tuple[str, str, tuple[int, int, int]], float]:
    if not path:
        return {}
    with open(path) as file:
        rows = json.load(file)["results"]
    return {
        (row["workload"], row["method"], tuple(row["shape"])): row["bic_median_s"]
        for row in rows
    }


def main() -> int:
    args = parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.batches < 1:
        raise SystemExit("repeats and batches must be >= 1 and warmup must be >= 0")
    methods = ("lewiner", "lorensen") if args.method == "all" else (args.method,)
    shape = shape_for(args.size)
    n_voxels = int(np.prod(shape))
    baseline = load_baseline(args.baseline)
    rows = []

    print(f"shape={shape} voxels={n_voxels} repeats={args.repeats} batches={args.batches}")
    print(f"{'workload/method':<28} {'V':>8} {'F':>8} {'bic ms':>11} {'p10-p90 ms':>15} {'skimage ms':>12} {'speed':>9} {'Mvox/s':>10} {'delta':>9}")
    print("-" * 125)
    selected_workloads = workloads(shape, args.seed)
    if args.workload != "all":
        selected_workloads = [row for row in selected_workloads if row[0] == args.workload]
    for workload, volume, level in selected_workloads:
        for method in methods:
            kwargs = {"method": method}
            start = perf_counter()
            actual = bic.mesh.marching_cubes(volume, level, **kwargs)
            bic_first_call = perf_counter() - start
            start = perf_counter()
            reference = reference_marching_cubes(volume, level, **kwargs)
            reference_first_call = perf_counter() - start
            assert_mesh_matches(actual, reference)
            bic_times = None
            if args.backend in ("both", "bic"):
                bic_times = time_call(
                    lambda: bic.mesh.marching_cubes(volume, level, **kwargs),
                    args.repeats,
                    args.warmup,
                    args.batches,
                )
            reference_times = None
            if args.backend in ("both", "skimage"):
                reference_times = time_call(
                    lambda: reference_marching_cubes(volume, level, **kwargs),
                    args.repeats,
                    args.warmup,
                    args.batches,
                )
            bic_median = None if bic_times is None else bic_times["median_s"]
            reference_median = (
                None if reference_times is None else reference_times["median_s"]
            )
            baseline_time = baseline.get((workload, method, shape))
            relative_change = (
                None
                if baseline_time is None or bic_median is None
                else bic_median / baseline_time - 1.0
            )
            speedup = (
                None
                if bic_median is None or reference_median is None
                else reference_median / bic_median
            )
            row = {
                "workload": workload,
                "method": method,
                "shape": shape,
                "vertices": len(actual[0]),
                "faces": len(actual[1]),
                "bic_first_call_s": bic_first_call,
                "reference_first_call_s": reference_first_call,
                "bic_raw_s": None if bic_times is None else bic_times["raw_s"],
                "bic_batch_medians_s": None if bic_times is None else bic_times["batch_medians_s"],
                "bic_median_s": bic_median,
                "bic_min_s": None if bic_times is None else bic_times["min_s"],
                "bic_p10_s": None if bic_times is None else bic_times["p10_s"],
                "bic_p90_s": None if bic_times is None else bic_times["p90_s"],
                "reference_raw_s": None if reference_times is None else reference_times["raw_s"],
                "reference_batch_medians_s": None if reference_times is None else reference_times["batch_medians_s"],
                "reference_median_s": reference_median,
                "reference_min_s": None if reference_times is None else reference_times["min_s"],
                "speedup": speedup,
                "mvox_per_s": None if bic_median is None else n_voxels / bic_median / 1e6,
                "baseline_bic_median_s": baseline_time,
                "relative_change": relative_change,
            }
            rows.append(row)
            delta = "-" if row["relative_change"] is None else f"{100.0 * row['relative_change']:+.1f}%"
            bic_text = "-" if bic_median is None else f"{bic_median * 1e3:>11.2f}"
            spread_text = (
                "      -      "
                if bic_times is None
                else f"{bic_times['p10_s'] * 1e3:>6.2f}-{bic_times['p90_s'] * 1e3:>6.2f}"
            )
            reference_text = (
                "-" if reference_median is None else f"{reference_median * 1e3:>12.2f}"
            )
            speed_text = "-" if speedup is None else f"{speedup:>8.2f}x"
            throughput_text = (
                "-" if row["mvox_per_s"] is None else f"{row['mvox_per_s']:>10.2f}"
            )
            print(
                f"{workload + '/' + method:<28} {row['vertices']:>8} {row['faces']:>8} "
                f"{bic_text:>11} {spread_text:>15} {reference_text:>12} "
                f"{speed_text:>9} {throughput_text:>10} {delta:>9}"
            )
    if args.json:
        with open(args.json, "w") as file:
            json.dump(
                {
                    "shape": shape,
                    "repeats": args.repeats,
                    "warmup": args.warmup,
                    "batches": args.batches,
                    "seed": args.seed,
                    "workload": args.workload,
                    "backend": args.backend,
                    "environment": {
                        "python": sys.version,
                        "platform": platform.platform(),
                        "numpy": np.__version__,
                        "scikit_image": skimage.__version__,
                    },
                    "results": rows,
                },
                file,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
