"""Benchmark binary 3D TEASAR on deterministic synthetic branching tubes.

The default benchmark has no optional dependencies and measures only
``bioimage_cpp.skeleton.teasar``. Pass ``--kimimaro`` to add kimimaro as an
external reference backend when it is installed. The implementations do not
have identical production heuristics, so compare topology and timings rather
than expecting vertex-for-vertex equality.

Examples
--------
python development/skeleton/benchmark_teasar.py --small --repeats 3
python development/skeleton/benchmark_teasar.py --repeats 5
python development/skeleton/benchmark_teasar.py --large --sequential-backends
python development/skeleton/benchmark_teasar.py --large --kimimaro --repeats 3 \
    --json /tmp/teasar.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from statistics import median
import sys
from time import perf_counter

import numpy as np

import bioimage_cpp as bic
from bioimage_cpp import _core


def draw_ball(mask: np.ndarray, center: np.ndarray, radius: int) -> None:
    center = np.rint(center).astype(int)
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius + 1, mask.shape)
    slices = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    coords = np.ogrid[tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))]
    squared_distance = sum((coords[d] - center[d]) ** 2 for d in range(3))
    mask[slices] |= squared_distance <= radius**2


def draw_segment(
    mask: np.ndarray, start: tuple[float, ...], stop: tuple[float, ...], radius: int
) -> None:
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    n_steps = int(np.ceil(np.linalg.norm(stop - start))) + 1
    for point in np.linspace(start, stop, n_steps):
        draw_ball(mask, point, radius)


def make_branching_tube(size: int, radius: int) -> np.ndarray:
    mask = np.zeros((size, size, size), dtype=np.uint8)
    c = (size - 1) / 2.0
    margin = max(radius + 3, int(0.10 * size))
    fork = (c, c, 0.54 * size)
    draw_segment(mask, (c, c, margin), fork, radius)
    draw_segment(mask, fork, (c, 0.22 * size, size - margin), radius)
    draw_segment(mask, fork, (0.24 * size, 0.76 * size, size - margin), radius)
    draw_segment(mask, fork, (0.76 * size, 0.76 * size, size - margin), radius)
    return mask


def time_backends(mask, backends, repeats: int, warmup: int):
    """Time backends in a deterministic shuffled order for every repeat."""
    samples = {name: [] for name, _, _ in backends}
    results = {}
    for _ in range(warmup):
        for name, function, _ in backends:
            results[name] = function(mask)

    rng = random.Random(20260714)
    for _ in range(repeats):
        order = list(backends)
        rng.shuffle(order)
        for name, function, _ in order:
            start = perf_counter()
            results[name] = function(mask)
            samples[name].append(perf_counter() - start)
    return samples, results


def count_bic(result) -> tuple[int, int]:
    vertices, edges, _ = result
    return len(vertices), len(edges)


def compare_skeletons(reference, candidate, spacing):
    """Return the FP32 acceptance metrics used by OPTIM_DIJKSTRA.md."""
    ref_vertices, ref_edges, _ = reference
    got_vertices, got_edges, _ = candidate

    def degree_signature(edges, n_vertices):
        degrees = np.bincount(edges.ravel().astype(np.int64), minlength=n_vertices)
        return tuple(sorted(int(degree) for degree in degrees if degree != 2))

    ref_voxels = ref_vertices / np.asarray(spacing, dtype=np.float64)
    got_voxels = got_vertices / np.asarray(spacing, dtype=np.float64)
    pairwise = np.linalg.norm(
        ref_voxels[:, None, :] - got_voxels[None, :, :], axis=2
    )
    ref_to_got = pairwise.min(axis=1)
    got_to_ref = pairwise.min(axis=0)

    def total_length(vertices, edges):
        if len(edges) == 0:
            return 0.0
        return float(
            np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).sum()
        )

    ref_length = total_length(ref_vertices, ref_edges)
    got_length = total_length(got_vertices, got_edges)
    return {
        "topology_match": degree_signature(ref_edges, len(ref_vertices))
        == degree_signature(got_edges, len(got_vertices)),
        "bidirectional_p95_voxels": float(
            max(np.percentile(ref_to_got, 95), np.percentile(got_to_ref, 95))
        ),
        "hausdorff_voxels": float(max(ref_to_got.max(), got_to_ref.max())),
        "relative_length_error": abs(got_length - ref_length) / max(ref_length, 1e-300),
    }


def kimimaro_call(mask, spacing, scale, constant, pdrf_scale, pdrf_exponent):
    import kimimaro

    skeletons = kimimaro.skeletonize(
        mask,
        object_ids=[1],
        teasar_params={
            "scale": scale,
            "const": constant,
            "pdrf_scale": pdrf_scale,
            "pdrf_exponent": int(pdrf_exponent),
            "soma_detection_threshold": float("inf"),
            "soma_acceptance_threshold": float("inf"),
            "soma_invalidation_scale": 1.0,
            "soma_invalidation_const": 0.0,
        },
        anisotropy=spacing,
        dust_threshold=0,
        progress=False,
        fix_branching=True,
        fix_borders=False,
        fill_holes=False,
        parallel=1,
    )
    return skeletons[1]


def bic_backend_call(mask, spacing, parameters, backend):
    """Call a development-only C++ backend without changing the public API."""
    return _core._teasar_uint8_backend(
        mask,
        spacing,
        parameters["scale"],
        parameters["constant"],
        parameters["pdrf_scale"],
        parameters["pdrf_exponent"],
        backend,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--small", action="store_true")
    sizes.add_argument("--large", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--kimimaro", action="store_true")
    parser.add_argument(
        "--sequential-backends",
        action="store_true",
        help="compare dense FP64, compact on-the-fly/CSR FP64, and CSR FP32",
    )
    parser.add_argument("--json", default="", help="optional JSON result path")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be >= 1 and --warmup must be >= 0")

    if args.kimimaro and importlib.util.find_spec("kimimaro") is None:
        parser.error("--kimimaro requested, but kimimaro is not installed")

    if args.small:
        specs = ((64, 3),)
    elif args.large:
        specs = ((128, 5), (192, 7), (256, 9))
    else:
        specs = ((96, 4), (128, 5), (192, 7))

    spacing = (1.5, 1.0, 1.0)
    parameters = {
        "scale": 1.5,
        "constant": 1.0,
        "pdrf_scale": 100000.0,
        "pdrf_exponent": 4.0,
    }
    if args.sequential_backends:
        backends = [
            (
                backend,
                lambda mask, backend=backend: bic_backend_call(
                    mask, spacing, parameters, backend
                ),
                count_bic,
            )
            for backend in (
                "dense-fp64",
                "compact-on-the-fly-fp64",
                "compact-csr-fp64",
                "compact-csr-fp32",
            )
        ]
    else:
        backends = [
            (
                "bioimage-cpp",
                lambda mask: bic.skeleton.teasar(mask, spacing=spacing, **parameters),
                count_bic,
            )
        ]
    if args.kimimaro:
        backends.append(
            (
                "kimimaro",
                lambda mask: kimimaro_call(
                    mask, spacing, parameters["scale"], parameters["constant"],
                    parameters["pdrf_scale"], parameters["pdrf_exponent"]
                ),
                lambda skeleton: (len(skeleton.vertices), len(skeleton.edges)),
            )
        )

    rows = []
    quality_rows = []
    header = f"{'backend':>14} {'shape':>14} {'foreground':>11} {'vertices':>9} {'median ms':>11} {'min ms':>10}"
    print(header)
    print("-" * len(header))
    for size, radius in specs:
        mask = make_branching_tube(size, radius)
        foreground = int(np.count_nonzero(mask))
        samples_by_backend, results_by_backend = time_backends(
            mask, backends, args.repeats, args.warmup
        )
        case_rows = []
        for name, function, count_result in backends:
            samples = samples_by_backend[name]
            result = results_by_backend[name]
            vertices, edges = count_result(result)
            median_s = median(samples)
            row = {
                "backend": name,
                "shape": list(mask.shape),
                "foreground_voxels": foreground,
                "vertices": vertices,
                "edges": edges,
                "samples_s": samples,
                "median_s": median_s,
                "min_s": min(samples),
            }
            rows.append(row)
            case_rows.append(row)
            print(
                f"{name:>14} {str(mask.shape):>14} {foreground:11d} {vertices:9d} "
                f"{median_s * 1e3:11.2f} {min(samples) * 1e3:10.2f}"
            )
        if args.sequential_backends:
            dense_result = results_by_backend["dense-fp64"]
            for exact_backend in (
                "compact-on-the-fly-fp64", "compact-csr-fp64"
            ):
                exact = all(
                    np.array_equal(got, expected)
                    for got, expected in zip(
                        results_by_backend[exact_backend], dense_result
                    )
                )
                print(f"  {exact_backend} exact dense parity: {exact}")
            quality = compare_skeletons(
                results_by_backend["compact-csr-fp64"],
                results_by_backend["compact-csr-fp32"],
                spacing,
            )
            quality_rows.append({"shape": list(mask.shape), **quality})
            print(
                "  FP32 quality: "
                f"topology={quality['topology_match']} "
                f"p95={quality['bidirectional_p95_voxels']:.3f} vox "
                f"hausdorff={quality['hausdorff_voxels']:.3f} vox "
                f"length_error={quality['relative_length_error']:.3%}"
            )
        if len(case_rows) == 2 and {row["backend"] for row in case_rows} == {
            "bioimage-cpp", "kimimaro"
        }:
            by_name = {row["backend"]: row for row in case_rows}
            ratio = (
                by_name["bioimage-cpp"]["median_s"]
                / by_name["kimimaro"]["median_s"]
            )
            print(f"{'bioimage-cpp / kimimaro':>54}: {ratio:.2f}x")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(
                {"repeats": args.repeats, "results": rows, "quality": quality_rows},
                file,
                indent=2,
            )
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
