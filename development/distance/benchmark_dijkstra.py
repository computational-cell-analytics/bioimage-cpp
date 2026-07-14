"""Benchmark the public grid-Dijkstra primitives independently of TEASAR.

The benchmark includes full physical and node-weighted fields, a predecessor
field, and early-stopping paths on 2D and 3D masks. ``--include-geodesic`` also
times the existing first-order fast-marching field on the same masks. The two
fields approximate different metrics, so that row is a performance comparison,
not a numerical reference comparison.

Examples
--------
python development/distance/benchmark_dijkstra.py --small --repeats 3
python development/distance/benchmark_dijkstra.py --repeats 10 --include-geodesic
python development/distance/benchmark_dijkstra.py --large --repeats 3 \
    --include-geodesic --json /tmp/dijkstra.json
"""

from __future__ import annotations

import argparse
import json
from statistics import median
import sys
from time import perf_counter

import numpy as np

import bioimage_cpp as bic


def make_mask(shape: tuple[int, ...]) -> np.ndarray:
    """Make one connected domain with structured obstacles and narrow doors."""
    mask = np.ones(shape, dtype=np.uint8)
    axis = len(shape) - 1
    extent = shape[axis]
    for wall_fraction, door_fraction in ((1 / 3, 0.25), (2 / 3, 0.75)):
        position = int(wall_fraction * extent)
        plane = [slice(None)] * len(shape)
        plane[axis] = position
        mask[tuple(plane)] = 0

        # Alternating doors keep the foreground connected while forcing a
        # substantial detour between the benchmark source and target.
        door = list(plane)
        for other_axis, other_extent in enumerate(shape):
            if other_axis == axis:
                continue
            center = int(door_fraction * (other_extent - 1))
            half_width = max(1, other_extent // 32)
            door[other_axis] = slice(
                max(0, center - half_width), min(other_extent, center + half_width + 1)
            )
        mask[tuple(door)] = 1
    return mask


def time_call(function, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return samples


def build_cases(shape: tuple[int, ...], include_geodesic: bool, number_of_threads: int):
    mask = make_mask(shape)
    source = np.full(len(shape), 2, dtype=np.int64)
    target = np.asarray(shape, dtype=np.int64) - 3
    assert mask[tuple(source)] and mask[tuple(target)]
    spacing = tuple(np.linspace(0.7, 1.4, len(shape)))
    zz = np.indices(shape, sparse=True)
    costs = np.ones(shape, dtype=np.float64)
    for axis, coordinates in enumerate(zz):
        costs += (axis + 1) * coordinates / max(1, shape[axis] - 1)

    cases = [
        (
            "dijkstra/physical-field",
            lambda: bic.distance.dijkstra_distance_field(
                mask, source, spacing=spacing, number_of_threads=number_of_threads
            ),
        ),
        (
            "dijkstra/field+parents",
            lambda: bic.distance.dijkstra_distance_field(
                mask,
                source,
                spacing=spacing,
                return_predecessors=True,
                number_of_threads=number_of_threads,
            ),
        ),
        (
            "dijkstra/node-field",
            lambda: bic.distance.dijkstra_distance_field(
                mask,
                source,
                costs=costs,
                cost_mode="node",
                number_of_threads=number_of_threads,
            ),
        ),
        (
            "dijkstra/node-times-physical-field",
            lambda: bic.distance.dijkstra_distance_field(
                mask,
                source,
                costs=costs,
                cost_mode="node_times_physical",
                spacing=spacing,
                number_of_threads=number_of_threads,
            ),
        ),
        (
            "dijkstra/early-path",
            lambda: bic.distance.dijkstra_path(
                mask,
                source,
                target,
                spacing=spacing,
                number_of_threads=number_of_threads,
            ),
        ),
    ]
    if include_geodesic:
        cases.append(
            (
                "geodesic/FMM-field",
                lambda: bic.distance.geodesic_distance_field(
                    mask,
                    source,
                    sampling=spacing,
                    number_of_threads=number_of_threads,
                ),
            )
        )
    return mask, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--small", action="store_true")
    sizes.add_argument("--large", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--include-geodesic", action="store_true")
    parser.add_argument(
        "--threads", type=int, nargs="+", default=[1],
        help="thread counts to benchmark (0 uses hardware concurrency)",
    )
    parser.add_argument("--json", default="", help="optional JSON result path")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be >= 1 and --warmup must be >= 0")

    if args.small:
        shapes = ((256, 256), (32, 96, 96))
    elif args.large:
        shapes = ((2048, 2048), (128, 256, 256))
    else:
        shapes = ((1024, 1024), (64, 160, 160))

    rows = []
    header = f"{'case':>36} {'threads':>7} {'shape':>16} {'median ms':>11} {'min ms':>10} {'ns/fg voxel':>13}"
    print(header)
    print("-" * len(header))
    for shape in shapes:
        for number_of_threads in args.threads:
            mask, cases = build_cases(
                shape, args.include_geodesic, number_of_threads
            )
            foreground = int(np.count_nonzero(mask))
            for name, function in cases:
                samples = time_call(function, args.repeats, args.warmup)
                median_s = median(samples)
                row = {
                    "case": name,
                    "number_of_threads": number_of_threads,
                    "shape": list(shape),
                    "foreground_voxels": foreground,
                    "samples_s": samples,
                    "median_s": median_s,
                    "min_s": min(samples),
                    "ns_per_foreground_voxel": median_s * 1e9 / foreground,
                }
                rows.append(row)
                print(
                    f"{name:>36} {number_of_threads:7d} {str(shape):>16} "
                    f"{median_s * 1e3:11.2f} {min(samples) * 1e3:10.2f} "
                    f"{row['ns_per_foreground_voxel']:13.1f}"
                )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump({"repeats": args.repeats, "results": rows}, file, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
