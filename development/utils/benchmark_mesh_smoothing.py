"""Benchmark Laplacian mesh smoothing: bioimage_cpp vs the nifty-based reference.

The mesh is built by triangulating ``n`` random points sampled on the unit
sphere via ``scipy.spatial.ConvexHull`` — a closed manifold of ``2n - 4``
triangles is a fair workload for the algorithm. The Python reference is
imported from ``_mesh_smoothing_reference.py`` and uses ``nifty`` for graph
adjacency; if either library is missing, the corresponding column is skipped.

Run::

    python development/utils/benchmark_mesh_smoothing.py --n-vertices 5000 --iterations 5 --repeats 3
    python development/utils/benchmark_mesh_smoothing.py --n-vertices 20000 --threads 1,2,4,0

Notes
-----
The reference does in-place Gauss-Seidel smoothing after the first iteration
(an aliasing accident in the Python source); bioimage_cpp does textbook Jacobi
smoothing. Correctness against the reference is only checked for
``iterations=1``, which is where the two implementations agree exactly.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-vertices", type=int, default=5000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--threads",
        type=str,
        default="1,0",
        help="Comma-separated list of n_threads values to benchmark (0=auto).",
    )
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    return parser.parse_args()


@dataclass(frozen=True)
class Mesh:
    verts: np.ndarray
    normals: np.ndarray
    faces: np.ndarray


def build_sphere_mesh(n_vertices: int, seed: int) -> Mesh:
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_vertices, 3))
    points = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float64)
    hull = ConvexHull(points)
    return Mesh(verts=points, normals=points.copy(), faces=hull.simplices.astype(np.int64))


def time_call(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        times.append(perf_counter() - t0)
    return times


def import_reference():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import nifty  # noqa: F401
    except ImportError:
        return None
    try:
        from _mesh_smoothing_reference import smooth_mesh as ref
    except ImportError:
        return None
    return ref


def main() -> int:
    args = parse_args()
    thread_values = [int(t) for t in args.threads.split(",") if t.strip()]

    try:
        mesh = build_sphere_mesh(args.n_vertices, args.seed)
    except ImportError:
        print("scipy is required to build the benchmark mesh.")
        return 1

    n_faces = mesh.faces.shape[0]
    print(
        f"Mesh: {args.n_vertices} vertices, {n_faces} faces "
        f"({args.iterations} smoothing iterations, {args.repeats} repeats)"
    )

    rows: list[dict] = []

    # bioimage_cpp variants (one per --threads value).
    for n_threads in thread_values:
        label = f"bioimage_cpp[n_threads={n_threads}]"

        def run(verts=mesh.verts, normals=mesh.normals, faces=mesh.faces, n=n_threads):
            bic.utils.smooth_mesh(verts, normals, faces, iterations=args.iterations, n_threads=n)

        times = time_call(run, args.repeats, args.warmup)
        rows.append(
            {"impl": label, "median_s": median(times), "best_s": min(times)}
        )

    # Reference (single-threaded by definition).
    reference = import_reference()
    if reference is not None:
        def run_ref():
            reference(mesh.verts, mesh.normals, mesh.faces, args.iterations)

        times = time_call(run_ref, args.repeats, args.warmup)
        rows.append(
            {"impl": "reference[nifty]", "median_s": median(times), "best_s": min(times)}
        )
    else:
        print("(reference skipped: nifty not installed)")

    # Correctness check against the reference at iterations=1 (only point of
    # exact agreement; see module docstring).
    if reference is not None:
        ref_v, ref_n = reference(mesh.verts, mesh.normals, mesh.faces, 1)
        ours_v, ours_n = bic.utils.smooth_mesh(mesh.verts, mesh.normals, mesh.faces, iterations=1)
        v_max_diff = float(np.max(np.abs(ours_v - ref_v)))
        n_max_diff = float(np.max(np.abs(ours_n - ref_n)))
        print(f"Correctness vs reference (iterations=1): "
              f"max|Δverts|={v_max_diff:.2e}, max|Δnormals|={n_max_diff:.2e}")

    # Print result table.
    print()
    print(f"{'impl':<32} {'median (s)':>12} {'best (s)':>12} {'speedup':>10}")
    baseline = None
    for row in rows:
        if row["impl"] == "reference[nifty]":
            baseline = row["median_s"]
            break
    for row in rows:
        speedup = baseline / row["median_s"] if baseline else float("nan")
        print(
            f"{row['impl']:<32} {row['median_s']:>12.4f} {row['best_s']:>12.4f} "
            f"{speedup:>10.2f}x"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["impl", "median_s", "best_s"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\nWrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
