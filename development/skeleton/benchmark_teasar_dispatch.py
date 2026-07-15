"""Benchmark the compact TEASAR shortest-path dispatch around its size gate.

The cases deliberately include compact domains at and above the former
``1 << 20`` foreground-node delta-stepping threshold, plus the established
branching-tube control below it.  The script has no optional dependencies and
requires array-exact output across worker counts.

Run from the repository root::

    python development/skeleton/benchmark_teasar_dispatch.py \
        --threads 1 2 4 --repeats 5 --warmup 1 \
        --json /tmp/teasar_dispatch.json
"""

from __future__ import annotations

import argparse
import json
from statistics import median
import sys

import numpy as np

import bioimage_cpp as bic

from benchmark_teasar import count_bic, make_branching_tube, time_backends


def dispatch_cases():
    return (
        ("solid-cube-102", np.ones((102, 102, 102), dtype=np.uint8)),
        (
            "solid-cuboid-64x128x128",
            np.ones((64, 128, 128), dtype=np.uint8),
        ),
        ("branching-tube-256", make_branching_tube(256, 9)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be >= 1 and --warmup must be >= 0")
    if not args.threads or any(thread < 1 for thread in args.threads):
        parser.error("--threads must contain positive worker counts")
    if 1 not in args.threads:
        parser.error("--threads must include 1 for exact-output comparison")

    spacing = (1.5, 1.0, 1.0)
    parameters = {
        "scale": 1.5,
        "constant": 1.0,
        "pdrf_scale": 100000.0,
        "pdrf_exponent": 4.0,
    }
    backends = [
        (
            f"bioimage-cpp/t{thread}",
            lambda mask, thread=thread: bic.skeleton.teasar(
                mask,
                spacing=spacing,
                number_of_threads=thread,
                **parameters,
            ),
            count_bic,
        )
        for thread in args.threads
    ]

    rows = []
    header = (
        f"{'case':>28} {'workers':>7} {'shape':>18} {'foreground':>11} "
        f"{'vertices':>9} {'median ms':>11} {'min ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for case_name, mask in dispatch_cases():
        samples, results = time_backends(mask, backends, args.repeats, args.warmup)
        reference = results["bioimage-cpp/t1"]
        foreground = int(np.count_nonzero(mask))
        for thread in args.threads:
            backend = f"bioimage-cpp/t{thread}"
            result = results[backend]
            if not all(
                np.array_equal(got, expected)
                for got, expected in zip(result, reference)
            ):
                raise RuntimeError(
                    f"{case_name}: thread count {thread} changed TEASAR output"
                )
            n_vertices, n_edges = count_bic(result)
            row = {
                "case": case_name,
                "backend": backend,
                "number_of_threads": thread,
                "shape": list(mask.shape),
                "foreground_voxels": foreground,
                "vertices": n_vertices,
                "edges": n_edges,
                "samples_s": samples[backend],
                "median_s": median(samples[backend]),
                "min_s": min(samples[backend]),
            }
            rows.append(row)
            print(
                f"{case_name:>28} {thread:7d} {str(mask.shape):>18} "
                f"{foreground:11d} {n_vertices:9d} "
                f"{row['median_s'] * 1e3:11.2f} {row['min_s'] * 1e3:10.2f}"
            )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "repeats": args.repeats,
                    "warmup": args.warmup,
                    "results": rows,
                },
                file,
                indent=2,
            )
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
