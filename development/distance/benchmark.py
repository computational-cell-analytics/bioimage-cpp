"""Benchmark distance transforms on skimage-derived 2D and 3D masks.

The current bioimage-cpp implementation is correctness-first and brute-force:
runtime is proportional to ``number_of_pixels * number_of_background_sites``.
To keep this benchmark runnable before the optimized EDT implementation lands,
the default data uses real skimage images/volumes to choose a sparse set of
background sites. Every library receives the same contiguous float32 mask:
nonzero foreground, zero background.

Run::

    python development/distance/benchmark.py --small --repeats 3
    python development/distance/benchmark.py --repeats 5 --csv distance.csv

Compared libraries:

* bioimage_cpp.distance.distance_transform
* bioimage_cpp.distance.vector_difference_transform
* vigra.filters.distanceTransform / vectorDistanceTransform
* scipy.ndimage.distance_transform_edt

SciPy has no direct vector distance transform. The SciPy vector baseline uses
``return_indices=True`` and converts feature indices to sampled difference
vectors, which is the closest SciPy-only equivalent.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import geometric_mean, median
from time import perf_counter

import numpy as np


LIBRARIES = ("bioimage_cpp", "vigra", "scipy")
OPERATIONS = ("distance_transform", "vector_difference_transform")


@dataclass(frozen=True)
class DataSpec:
    label: str
    shape: tuple[int, ...]
    n_targets: int


@dataclass(frozen=True)
class BenchConfig:
    sampling: tuple[float, ...] | None

    def sampling_for(self, ndim: int) -> tuple[float, ...]:
        if self.sampling is None:
            return (1.0,) * ndim
        if len(self.sampling) == 1:
            return self.sampling * ndim
        if len(self.sampling) != ndim:
            raise ValueError(
                f"sampling has length {len(self.sampling)}, but data has ndim={ndim}"
            )
        return self.sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark bioimage_cpp.distance vs vigra and scipy."
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--small", action="store_true", help="Fast smoke benchmark.")
    parser.add_argument("--large", action="store_true", help="Larger benchmark sizes.")
    parser.add_argument("--no-2d", action="store_true")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument(
        "--operations",
        default=",".join(OPERATIONS),
        help="Comma-separated subset of operations to benchmark.",
    )
    parser.add_argument(
        "--sampling",
        default="1.0",
        help=(
            "Scalar or comma-separated per-axis sampling. Default 1.0 keeps "
            "VIGRA vector outputs directly comparable."
        ),
    )
    parser.add_argument(
        "--targets-2d",
        type=int,
        default=None,
        help="Number of zero-valued background sites in the 2D mask.",
    )
    parser.add_argument(
        "--targets-3d",
        type=int,
        default=None,
        help="Number of zero-valued background sites in the 3D mask.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write per-(operation, dim, library) timings.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip one-shot result checks before timing.",
    )
    return parser.parse_args()


def _parse_sampling(text: str) -> tuple[float, ...] | None:
    if text.strip().lower() in {"none", ""}:
        return None
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        return None
    for value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"sampling values must be positive and finite, got {values}")
    return values


def _import_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _quiet_vigra_matplotlib_cache() -> None:
    # VIGRA imports matplotlib in some environments. Point matplotlib's cache
    # at /tmp to avoid noisy warnings when the user's config dir is read-only.
    path = os.path.join(tempfile.gettempdir(), "bioimage_cpp_matplotlib_cache")
    os.makedirs(path, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", path)


# ---------------------------------------------------------------------------
# skimage-derived benchmark data
# ---------------------------------------------------------------------------

def _grid_shape(shape: tuple[int, ...], n_blocks: int) -> tuple[int, ...]:
    dims = [1] * len(shape)
    while math.prod(dims) < n_blocks:
        axis = max(range(len(shape)), key=lambda ax: shape[ax] / dims[ax])
        dims[axis] += 1
    return tuple(dims)


def _dark_landmarks(image: np.ndarray, n_targets: int) -> np.ndarray:
    """Pick spatially spread low-intensity coordinates from `image`.

    The image is partitioned into approximately `n_targets` blocks, and the
    darkest voxel/pixel from each block is used as a background site.
    """
    block_shape = _grid_shape(tuple(image.shape), n_targets)
    ranges = [
        np.array_split(np.arange(axis_size), n_splits)
        for axis_size, n_splits in zip(image.shape, block_shape)
    ]
    coords = []
    for block_indices in np.ndindex(*block_shape):
        slices = tuple(
            slice(int(ranges[axis][block_indices[axis]][0]),
                  int(ranges[axis][block_indices[axis]][-1]) + 1)
            for axis in range(image.ndim)
        )
        block = image[slices]
        local = np.unravel_index(int(np.argmin(block)), block.shape)
        coords.append(tuple(local[axis] + slices[axis].start for axis in range(image.ndim)))
    coords = sorted(coords, key=lambda coord: float(image[coord]))
    return np.asarray(coords[:n_targets], dtype=np.int64)


def _mask_from_image(image: np.ndarray, n_targets: int) -> np.ndarray:
    coords = _dark_landmarks(image, n_targets)
    mask = np.ones(image.shape, dtype=np.float32)
    mask[tuple(coords.T)] = 0.0
    return np.ascontiguousarray(mask)


def load_2d(spec: DataSpec) -> np.ndarray:
    from skimage import data

    image = data.camera().astype(np.float32)
    image = image[: spec.shape[0], : spec.shape[1]]
    return _mask_from_image(image, spec.n_targets)


def load_3d(spec: DataSpec) -> np.ndarray:
    from skimage import data

    volume = data.cells3d()[:, 1].astype(np.float32)
    volume = volume[: spec.shape[0], : spec.shape[1], : spec.shape[2]]
    return _mask_from_image(volume, spec.n_targets)


def build_specs(args: argparse.Namespace) -> list[DataSpec]:
    if args.small and args.large:
        raise ValueError("--small and --large are mutually exclusive")

    if args.small:
        shape_2d, targets_2d = (64, 64), 16
        shape_3d, targets_3d = (12, 32, 32), 16
    elif args.large:
        shape_2d, targets_2d = (256, 256), 128
        shape_3d, targets_3d = (32, 96, 96), 128
    else:
        shape_2d, targets_2d = (128, 128), 64
        shape_3d, targets_3d = (20, 64, 64), 64

    if args.targets_2d is not None:
        targets_2d = args.targets_2d
    if args.targets_3d is not None:
        targets_3d = args.targets_3d

    specs = []
    if not args.no_2d:
        specs.append(DataSpec("2D", shape_2d, targets_2d))
    if not args.no_3d:
        specs.append(DataSpec("3D", shape_3d, targets_3d))
    return specs


# ---------------------------------------------------------------------------
# Library adapters
# ---------------------------------------------------------------------------

def _bic_distance(sampling: tuple[float, ...]):
    from bioimage_cpp import distance

    def fn(mask: np.ndarray) -> np.ndarray:
        return distance.distance_transform(mask, sampling=sampling)

    return fn


def _bic_vector(sampling: tuple[float, ...]):
    from bioimage_cpp import distance

    def fn(mask: np.ndarray) -> np.ndarray:
        return distance.vector_difference_transform(mask, sampling=sampling)

    return fn


def _vigra_distance(sampling: tuple[float, ...]):
    _quiet_vigra_matplotlib_cache()
    import vigra.filters as vf

    def fn(mask: np.ndarray) -> np.ndarray:
        return np.asarray(
            vf.distanceTransform(mask, background=False, pixel_pitch=sampling)
        )

    return fn


def _vigra_vector(sampling: tuple[float, ...]):
    _quiet_vigra_matplotlib_cache()
    import vigra.filters as vf
    sampling_array = np.asarray(sampling, dtype=np.float32)
    needs_scaling = not np.allclose(sampling_array, 1.0)

    def fn(mask: np.ndarray) -> np.ndarray:
        vectors = np.asarray(
            vf.vectorDistanceTransform(mask, background=False, pixel_pitch=sampling)
        )
        if needs_scaling:
            vectors = vectors * sampling_array.reshape((1,) * mask.ndim + (-1,))
        return vectors

    return fn


def _scipy_distance(sampling: tuple[float, ...]):
    from scipy import ndimage

    def fn(mask: np.ndarray) -> np.ndarray:
        return ndimage.distance_transform_edt(mask, sampling=sampling)

    return fn


def _scipy_vector(sampling: tuple[float, ...]):
    from scipy import ndimage

    coords_cache: dict[tuple[int, ...], np.ndarray] = {}
    sampling_array = np.asarray(sampling, dtype=np.float64)

    def fn(mask: np.ndarray) -> np.ndarray:
        _, indices = ndimage.distance_transform_edt(
            mask, sampling=sampling, return_indices=True
        )
        coords = coords_cache.get(mask.shape)
        if coords is None:
            coords = np.indices(mask.shape, dtype=np.int32)
            coords_cache[mask.shape] = coords
        vectors = np.moveaxis(indices - coords, 0, -1)
        vectors = vectors * sampling_array.reshape((1,) * mask.ndim + (-1,))
        return vectors

    return fn


def build_adapters(operation: str, sampling: tuple[float, ...]) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    builders = {
        "distance_transform": {
            "bioimage_cpp": _bic_distance,
            "vigra": _vigra_distance,
            "scipy": _scipy_distance,
        },
        "vector_difference_transform": {
            "bioimage_cpp": _bic_vector,
            "vigra": _vigra_vector,
            "scipy": _scipy_vector,
        },
    }[operation]

    adapters = {}
    for library, builder in builders.items():
        if library == "vigra" and not _import_available("vigra"):
            continue
        if library == "scipy" and not _import_available("scipy"):
            continue
        adapters[library] = builder(sampling)
    return adapters


# ---------------------------------------------------------------------------
# Timing and checks
# ---------------------------------------------------------------------------

def time_interleaved(
    callables: dict[str, Callable[[np.ndarray], np.ndarray]],
    mask: np.ndarray,
    repeats: int,
) -> dict[str, dict]:
    libs = list(callables)
    for fn in callables.values():
        fn(mask)

    timings = {lib: [] for lib in libs}
    last_result = {}
    for repeat in range(repeats):
        rotation = repeat % len(libs)
        order = libs[rotation:] + libs[:rotation]
        for lib in order:
            t0 = perf_counter()
            result = callables[lib](mask)
            timings[lib].append(perf_counter() - t0)
            last_result[lib] = np.asarray(result)

    return {
        lib: {
            "timings": timings[lib],
            "median": median(timings[lib]),
            "min": min(timings[lib]),
            "result": last_result[lib],
        }
        for lib in libs
    }


def check_results(
    operation: str,
    mask: np.ndarray,
    sampling: tuple[float, ...],
    adapters: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> dict[str, float]:
    if "scipy" not in adapters:
        return {}

    reference_distance = _scipy_distance(sampling)(mask).astype(np.float32, copy=False)
    errors = {}
    for library, fn in adapters.items():
        result = np.asarray(fn(mask))
        if operation == "distance_transform":
            errors[library] = float(np.max(np.abs(result.astype(np.float32) - reference_distance)))
        else:
            # Equidistant feature-index ties can choose different nearest
            # targets. Vector magnitudes must still match the distance map.
            magnitudes = np.linalg.norm(result.astype(np.float32), axis=-1)
            errors[library] = float(np.max(np.abs(magnitudes - reference_distance)))
    return errors


def format_results_table(rows: list[dict]) -> str:
    headers = ["operation", "dim", "shape", "targets"]
    for lib in LIBRARIES:
        headers.extend([f"{lib} ms", "x ours"])

    table_rows = []
    for row in rows:
        ours = row["results"].get("bioimage_cpp")
        ours_median = ours["median"] if ours else None
        line = [
            row["operation"],
            row["dim"],
            row["shape"],
            str(row["targets"]),
        ]
        for lib in LIBRARIES:
            result = row["results"].get(lib)
            if result is None:
                line.extend(["n/a", "n/a"])
            else:
                line.append(f"{result['median'] * 1e3:.2f}")
                if lib == "bioimage_cpp":
                    line.append("1.00")
                elif ours_median is None:
                    line.append("-")
                else:
                    line.append(f"{ours_median / result['median']:.2f}")
        table_rows.append(line)

    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def render(values: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [render(headers), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in table_rows)
    return "\n".join(lines)


def print_headline_ratios(rows: list[dict]) -> None:
    print()
    print("speedup summary (geomean of bioimage_cpp.median / other.median; "
          ">1.0 means bioimage_cpp slower):")
    for other in ("vigra", "scipy"):
        ratios = []
        for row in rows:
            ours = row["results"].get("bioimage_cpp")
            theirs = row["results"].get(other)
            if ours and theirs and theirs["median"] > 0:
                ratios.append(ours["median"] / theirs["median"])
        if ratios:
            print(f"  bioimage_cpp / {other:<5s}  geomean = {geometric_mean(ratios):.3f}  (n={len(ratios)})")


def main() -> int:
    args = parse_args()
    requested = tuple(op.strip() for op in args.operations.split(",") if op.strip())
    unknown = [op for op in requested if op not in OPERATIONS]
    if unknown:
        print(f"unknown operation(s): {unknown}", file=sys.stderr)
        return 2
    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2

    try:
        cfg = BenchConfig(sampling=_parse_sampling(args.sampling))
        specs = build_specs(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not _import_available("skimage"):
        print("skimage is required for benchmark data", file=sys.stderr)
        return 2

    print(f"repeats={args.repeats}, sampling={args.sampling}")
    print("data: skimage camera/cells3d, sparse zero-valued landmarks from dark blocks")

    rows = []
    csv_rows = []
    for spec in specs:
        mask = load_2d(spec) if spec.label == "2D" else load_3d(spec)
        sampling = cfg.sampling_for(mask.ndim)
        print(
            f"{spec.label}: shape={mask.shape}, targets={int(np.count_nonzero(mask == 0))}, "
            f"foreground={int(np.count_nonzero(mask != 0))}"
        )
        for operation in requested:
            adapters = build_adapters(operation, sampling)
            if not adapters:
                continue
            errors = {} if args.skip_checks else check_results(operation, mask, sampling, adapters)
            results = time_interleaved(adapters, mask, args.repeats)
            full_results = {lib: results.get(lib) for lib in LIBRARIES}
            rows.append({
                "operation": operation,
                "dim": spec.label,
                "shape": str(tuple(mask.shape)),
                "targets": int(np.count_nonzero(mask == 0)),
                "results": full_results,
                "errors": errors,
            })
            for lib, result in full_results.items():
                if result is None:
                    continue
                csv_rows.append({
                    "operation": operation,
                    "dim": spec.label,
                    "shape": tuple(mask.shape),
                    "targets": int(np.count_nonzero(mask == 0)),
                    "library": lib,
                    "median_s": result["median"],
                    "min_s": result["min"],
                    "max_abs_error": errors.get(lib, ""),
                    "repeats": args.repeats,
                })

    print()
    print(format_results_table(rows))
    if not args.skip_checks:
        print()
        print("max abs error vs scipy distance map (vector rows compare vector norm):")
        for row in rows:
            formatted = ", ".join(
                f"{lib}={err:.3g}" for lib, err in row["errors"].items()
            )
            print(f"  {row['operation']} {row['dim']}: {formatted}")
    print_headline_ratios(rows)

    if args.csv is not None:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "operation",
                    "dim",
                    "shape",
                    "targets",
                    "library",
                    "median_s",
                    "min_s",
                    "max_abs_error",
                    "repeats",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"wrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
