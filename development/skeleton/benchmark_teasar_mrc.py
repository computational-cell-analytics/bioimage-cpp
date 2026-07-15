from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter

import mrcfile
import numpy as np

import bioimage_cpp as bic


def centered_crop(shape: tuple[int, ...], fraction: float):
    shape_array = np.asarray(shape, dtype=np.int64)
    crop_shape = np.maximum(1, np.floor(shape_array * fraction).astype(np.int64))
    begin = (shape_array - crop_shape) // 2
    end = begin + crop_shape
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(begin, end))
    return slices, tuple(int(value) for value in begin)


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def exact_result(first, second):
    return all(np.array_equal(a, b) for a, b in zip(first, second))


def time_threads(mask, threads, scale, repeats, warmup):
    results = {}
    samples = {thread: [] for thread in threads}
    for _ in range(warmup):
        for thread in threads:
            results[thread] = bic.skeleton.teasar(
                mask,
                scale=scale,
                number_of_threads=thread,
            )
    rng = random.Random(20260715)
    for _ in range(repeats):
        order = list(threads)
        rng.shuffle(order)
        for thread in order:
            start = perf_counter()
            results[thread] = bic.skeleton.teasar(
                mask,
                scale=scale,
                number_of_threads=thread,
            )
            samples[thread].append(perf_counter() - start)
    reference_thread = threads[0]
    for thread in threads[1:]:
        if not exact_result(results[reference_thread], results[thread]):
            raise RuntimeError(
                f"thread count {thread} changed the skeleton compared with "
                f"thread count {reference_thread}"
            )
    return samples, results


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "examples" / "skeleton" / "00004_gt_mask.mrc",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.5],
    )
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--threads", type=int, nargs="+", default=[8])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be >= 1 and --warmup must be >= 0")
    if not args.threads or any(thread < 1 for thread in args.threads):
        parser.error("--threads must contain positive values")
    if any(not 0.0 < fraction <= 1.0 for fraction in args.fractions):
        parser.error("--fractions values must be in (0, 1]")
    fractions = set(args.fractions)
    if args.include_full:
        fractions.add(1.0)
    args.fractions = sorted(fractions)
    args.threads = list(dict.fromkeys(args.threads))
    return args


def main() -> int:
    args = parse_args()
    rows = []
    header = (
        f"{'fraction':>8} {'threads':>7} {'shape':>18} {'foreground':>11} "
        f"{'components':>10} {'vertices':>10} {'median s':>10} {'min s':>10}"
    )
    print(header)
    print("-" * len(header))
    with mrcfile.mmap(args.input, mode="r", permissive=True) as mrc:
        source = mrc.data
        source_shape = tuple(int(value) for value in source.shape)
        for fraction in args.fractions:
            slices, origin = centered_crop(source_shape, fraction)
            mask = np.array(source[slices], dtype=np.uint8, copy=True)
            foreground = int(np.count_nonzero(mask))
            samples, results = time_threads(
                mask,
                args.threads,
                args.scale,
                args.repeats,
                args.warmup,
            )
            for thread in args.threads:
                vertices, edges, _ = results[thread]
                components = len(vertices) - len(edges)
                row = {
                    "fraction": fraction,
                    "origin": list(origin),
                    "shape": list(mask.shape),
                    "full_voxels": int(mask.size),
                    "foreground_voxels": foreground,
                    "number_of_threads": thread,
                    "components": components,
                    "vertices": len(vertices),
                    "edges": len(edges),
                    "samples_s": samples[thread],
                    "median_s": median(samples[thread]),
                    "min_s": min(samples[thread]),
                }
                rows.append(row)
                print(
                    f"{fraction:8.3f} {thread:7d} {str(mask.shape):>18} "
                    f"{foreground:11d} {components:10d} {len(vertices):10d} "
                    f"{row['median_s']:10.3f} {row['min_s']:10.3f}"
                )
            del results
            del mask
    if args.json is not None:
        payload = {
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "numpy": np.__version__,
                "bioimage_cpp": package_version("bioimage-cpp"),
                "mrcfile": package_version("mrcfile"),
            },
            "input": str(args.input.resolve()),
            "source_shape": list(source_shape),
            "fractions": args.fractions,
            "threads": args.threads,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "scale": args.scale,
            "results": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
