"""Benchmark geodesic distances: bioimage-cpp vs scikit-fmm / pygeodesic.

Times the ``bic.distance`` geodesic entry points against their reference oracles
(scikit-fmm for masks, pygeodesic exact for meshes) on a range of mask grids and
sphere meshes, reporting per-call statistics and (optionally) the speedup versus
the reference. Reference columns are skipped when the backend is not installed or
when ``--no-ref`` is passed.

For each case it reports the min, median and spread (p75-p25) of the timed calls
plus ``ns/unit`` — nanoseconds per element processed (voxels for a field solve,
sources x voxels for a pairwise matrix), a size-independent throughput number that
is directly comparable across sizes and dimensionalities and is the right metric
for before/after optimization comparisons.

For rigorous before/after measurement use a high ``--repeats`` (>=15), pin
``--threads 1`` on the field cases, run on a quiet machine, and dump ``--json`` for
each build so the two runs can be diffed. Interleave baseline/optimized builds to
cancel slow thermal/scheduler drift.

Not part of the pytest suite; requires scikit-fmm, pygeodesic and scipy (the last
only for the sphere-mesh construction; ``--no-ref`` still needs scipy for meshes).

Run::

    python benchmark_geodesic.py --small
    python benchmark_geodesic.py --large --threads 0
    python benchmark_geodesic.py --xlarge --only field --threads 1 --repeats 20 \\
        --no-ref --json /tmp/baseline.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from statistics import median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic

from _geodesic_reference import (
    reference_geodesic_distances_mask,
    reference_geodesic_distances_mesh,
    reference_geodesic_field_mask,
    reference_geodesic_field_mesh,
)


def available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def time_call(fn, repeats: int, warmup: int = 1) -> list[float]:
    """Return the list of per-call wall-clock times (seconds), after warmup."""
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        timings.append(perf_counter() - start)
    return timings


def stats(timings: list[float]) -> dict:
    ordered = sorted(timings)
    n = len(ordered)
    p25 = ordered[max(0, (n - 1) // 4)]
    p75 = ordered[min(n - 1, (3 * (n - 1)) // 4)]
    return {
        "min": ordered[0],
        "median": median(ordered),
        "max": ordered[-1],
        "spread": p75 - p25,
    }


def make_sphere(n_points: int, radius: float = 5.0, seed: int = 0):
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_points, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius
    verts = np.ascontiguousarray(pts, dtype=np.float64)
    faces = np.ascontiguousarray(ConvexHull(pts).simplices, dtype=np.int64)
    return verts, faces


def scattered_points(n_axis_points: int, ndim: int, extent: int) -> np.ndarray:
    coords = np.linspace(1, extent - 2, num=n_axis_points, dtype=np.int64)
    grids = np.meshgrid(*([coords] * ndim), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1).astype(np.int64)


def build_cases(args):
    """Return a list of (name, bic_fn, ref_fn, n_units) tuples for the size.

    ``n_units`` is the number of elements processed by one call: the voxel/vertex
    count for a field solve, or ``n_sources * count`` for a pairwise matrix. It is
    the denominator for the ns/unit throughput metric.
    """
    if args.small:
        mask2d_shape, mask3d_shape, mesh_n = (256, 256), (24, 48, 48), 1500
    elif args.large:
        mask2d_shape, mask3d_shape, mesh_n = (1024, 1024), (64, 128, 128), 20000
    elif args.xlarge:
        mask2d_shape, mask3d_shape, mesh_n = (1536, 1536), (128, 128, 128), 40000
    else:
        mask2d_shape, mask3d_shape, mesh_n = (512, 512), (32, 96, 96), 5000

    def prod(shape):
        n = 1
        for s in shape:
            n *= s
        return n

    cases = []

    mask2d = np.ones(mask2d_shape, np.uint8)
    src2d = np.array([[mask2d_shape[0] // 2, mask2d_shape[1] // 2]], np.int64)
    pts2d = scattered_points(args.n_pairwise, 2, min(mask2d_shape))
    cases.append((
        f"mask2d/field {mask2d_shape}",
        lambda: bic.distance.geodesic_distance_field(mask2d, src2d, number_of_threads=args.threads),
        lambda: reference_geodesic_field_mask(mask2d, src2d),
        prod(mask2d_shape),
    ))
    cases.append((
        f"mask2d/pairwise N={len(pts2d)}",
        lambda: bic.distance.geodesic_distances(mask2d, pts2d, number_of_threads=args.threads),
        lambda: reference_geodesic_distances_mask(mask2d, pts2d),
        len(pts2d) * prod(mask2d_shape),
    ))

    mask3d = np.ones(mask3d_shape, np.uint8)
    src3d = np.array([[s // 2 for s in mask3d_shape]], np.int64)
    pts3d = scattered_points(args.n_pairwise, 3, min(mask3d_shape))
    cases.append((
        f"mask3d/field {mask3d_shape}",
        lambda: bic.distance.geodesic_distance_field(mask3d, src3d, number_of_threads=args.threads),
        lambda: reference_geodesic_field_mask(mask3d, src3d),
        prod(mask3d_shape),
    ))
    cases.append((
        f"mask3d/pairwise N={len(pts3d)}",
        lambda: bic.distance.geodesic_distances(mask3d, pts3d, number_of_threads=args.threads),
        lambda: reference_geodesic_distances_mask(mask3d, pts3d),
        len(pts3d) * prod(mask3d_shape),
    ))

    verts, faces = make_sphere(mesh_n, seed=args.seed)
    src_v = np.array([0], np.int64)
    pts_v = np.linspace(0, len(verts) - 1, num=args.n_pairwise, dtype=np.int64)
    cases.append((
        f"mesh/field V={len(verts)}",
        lambda: bic.distance.geodesic_distance_field_mesh(verts, faces, src_v, number_of_threads=args.threads),
        lambda: reference_geodesic_field_mesh(verts, faces, src_v),
        len(verts),
    ))
    cases.append((
        f"mesh/pairwise N={len(pts_v)}",
        lambda: bic.distance.geodesic_distances_mesh(verts, faces, pts_v, number_of_threads=args.threads),
        lambda: reference_geodesic_distances_mesh(verts, faces, pts_v),
        len(pts_v) * len(verts),
    ))

    if args.only:
        cases = [c for c in cases if args.only in c[0]]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--xlarge", action="store_true")
    parser.add_argument("--threads", type=int, default=1, help="0 = hardware concurrency")
    parser.add_argument("--n-pairwise", type=int, default=8, dest="n_pairwise",
                        help="points per axis (mask) / total points (mesh) for pairwise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ref", action="store_true", help="skip reference timings")
    parser.add_argument("--only", type=str, default="",
                        help="substring filter on case name, e.g. 'field' or 'mask3d'")
    parser.add_argument("--json", type=str, default="", help="dump results to this JSON path")
    args = parser.parse_args()
    n_sizes = sum(bool(x) for x in (args.small, args.large, args.xlarge))
    if n_sizes > 1:
        print("choose at most one of --small / --large / --xlarge", file=sys.stderr)
        return 2

    have_mask_ref = available("skfmm") and not args.no_ref
    have_mesh_ref = available("pygeodesic") and not args.no_ref
    if not args.no_ref:
        if not available("skfmm"):
            print("scikit-fmm not installed: mask reference column skipped", file=sys.stderr)
        if not available("pygeodesic"):
            print("pygeodesic not installed: mesh reference column skipped", file=sys.stderr)

    cases = build_cases(args)

    header = (f"{'case':>26} {'bic_min':>10} {'bic_med':>10} {'spread':>9} "
              f"{'ns/unit':>9} {'ref_med':>10} {'speedup':>9}")
    print(f"threads={args.threads} repeats={args.repeats}")
    print(header)
    print("-" * len(header))
    results = []
    for name, bic_fn, ref_fn, n_units in cases:
        bic_t = time_call(bic_fn, args.repeats)
        bs = stats(bic_t)
        ns_per_unit = bs["median"] / n_units * 1e9
        is_mesh = name.startswith("mesh")
        has_ref = have_mesh_ref if is_mesh else have_mask_ref
        entry = {"case": name, "n_units": n_units, "threads": args.threads,
                 "bic_s": bs, "ns_per_unit": ns_per_unit}
        if has_ref:
            ref_t = time_call(ref_fn, args.repeats)
            rs = stats(ref_t)
            speed = rs["median"] / bs["median"]
            entry["ref_s"] = rs
            entry["speedup"] = speed
            ref_col, speed_col = f"{rs['median']*1e3:10.2f}", f"{speed:6.2f}x"
        else:
            ref_col, speed_col = f"{'n/a':>10}", f"{'n/a':>9}"
        print(f"{name:>26} {bs['min']*1e3:10.2f} {bs['median']*1e3:10.2f} "
              f"{bs['spread']*1e3:9.2f} {ns_per_unit:9.1f} {ref_col} {speed_col:>9}")
        results.append(entry)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"threads": args.threads, "repeats": args.repeats,
                       "results": results}, fh, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
