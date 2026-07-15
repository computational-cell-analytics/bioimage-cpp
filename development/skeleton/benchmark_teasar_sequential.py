"""Benchmark the sequential TEASAR design matrix beyond the size sweep.

This complements ``benchmark_teasar.py --sequential-backends`` with sparse and
relatively dense domains, isotropic/anisotropic spacing, and low/default/high
PDRF regimes. It intentionally uses the private development backend selector;
the public API remains backend-free.

Run from the repository root::

    python development/skeleton/benchmark_teasar_sequential.py --repeats 5 \
        --json /tmp/teasar_sequential_cases.json
"""

from __future__ import annotations

import argparse
import json
from statistics import median

import numpy as np

from benchmark_teasar import (
    bic_backend_call,
    count_bic,
    draw_ball,
    make_branching_tube,
    time_backends,
)


BACKENDS = (
    "dense-fp64",
    "compact-on-the-fly-fp64",
    "compact-csr-fp64",
)


def make_dense_ball(size: int) -> np.ndarray:
    mask = np.zeros((size, size, size), dtype=np.uint8)
    draw_ball(mask, np.full(3, (size - 1) / 2.0), int(0.36 * size))
    return mask


def cases():
    default = {
        "scale": 1.5,
        "constant": 1.0,
        "pdrf_scale": 100000.0,
        "pdrf_exponent": 4.0,
    }
    return (
        (
            "sparse-embedded",
            make_branching_tube(192, 2),
            (1.5, 1.0, 1.0),
            default,
        ),
        (
            "relatively-dense-ball",
            make_dense_ball(96),
            (1.5, 1.0, 1.0),
            default,
        ),
        (
            "isotropic",
            make_branching_tube(128, 5),
            (1.0, 1.0, 1.0),
            default,
        ),
        (
            "anisotropic",
            make_branching_tube(128, 5),
            (2.5, 1.25, 0.75),
            default,
        ),
        (
            "low-pdrf",
            make_branching_tube(128, 5),
            (1.5, 1.0, 1.0),
            {**default, "pdrf_scale": 100.0, "pdrf_exponent": 2.0},
        ),
        (
            "high-pdrf",
            make_branching_tube(128, 5),
            (1.5, 1.0, 1.0),
            {**default, "pdrf_scale": 10000000.0, "pdrf_exponent": 6.0},
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be >= 1 and --warmup must be >= 0")

    rows = []
    header = (
        f"{'case':>23} {'backend':>27} {'shape':>14} {'foreground':>11} "
        f"{'vertices':>9} {'median ms':>11}"
    )
    print(header)
    print("-" * len(header))
    for case_name, mask, spacing, parameters in cases():
        backends = [
            (
                backend,
                lambda input_mask, backend=backend: bic_backend_call(
                    input_mask, spacing, parameters, backend
                ),
                count_bic,
            )
            for backend in BACKENDS
        ]
        samples, results = time_backends(mask, backends, args.repeats, args.warmup)
        for backend in BACKENDS:
            result = results[backend]
            n_vertices, n_edges = count_bic(result)
            row = {
                "case": case_name,
                "backend": backend,
                "shape": list(mask.shape),
                "foreground_voxels": int(np.count_nonzero(mask)),
                "vertices": n_vertices,
                "edges": n_edges,
                "samples_s": samples[backend],
                "median_s": median(samples[backend]),
            }
            rows.append(row)
            print(
                f"{case_name:>23} {backend:>27} {str(mask.shape):>14} "
                f"{row['foreground_voxels']:11d} {n_vertices:9d} "
                f"{row['median_s'] * 1e3:11.2f}"
            )

        for backend in ("compact-on-the-fly-fp64", "compact-csr-fp64"):
            if not all(
                np.array_equal(got, expected)
                for got, expected in zip(results[backend], results["dense-fp64"])
            ):
                raise RuntimeError(f"{case_name}: {backend} lost exact dense parity")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump({"repeats": args.repeats, "results": rows}, file, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
