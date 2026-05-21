"""Benchmark distance transforms on skimage-derived 2D and 3D masks.

bioimage-cpp's distance transform uses the separable Felzenszwalb–Huttenlocher
algorithm (O(N * ndim)), which is the same algorithmic class as vigra and
scipy. Masks are derived by thresholding real skimage images/volumes at a
controllable quantile, giving a realistic foreground/background split.

Each library is fed a pre-built mask in its preferred dtype, allocated once
outside the timing loop so per-call dtype conversion does not show up in the
measurement of a particular library:

* bioimage_cpp.distance.distance_transform        — uint8 mask
* bioimage_cpp.distance.vector_difference_transform — uint8 mask
* vigra.filters.distanceTransform / vectorDistanceTransform — float32 mask
* scipy.ndimage.distance_transform_edt            — float32 mask

SciPy has no direct vector distance transform. The SciPy vector baseline uses
``return_indices=True`` and converts feature indices to sampled difference
vectors, which is the closest SciPy-only equivalent.

Run::

    python development/distance/benchmark.py --small --repeats 3
    python development/distance/benchmark.py --repeats 5 --csv distance.csv
    python development/distance/benchmark.py --large --threads 0
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
    fraction_background: float


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


@dataclass
class Adapter:
    fn: Callable[[np.ndarray], np.ndarray]
    mask: np.ndarray


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
        "--density",
        type=float,
        default=0.5,
        help=(
            "Fraction of pixels classified as background (0.0–1.0). The mask "
            "is built by thresholding the source image at this quantile."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "Thread count for bioimage_cpp. 0 = hardware concurrency. "
            "vigra and scipy are single-threaded."
        ),
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

def _quantile_mask(image: np.ndarray, fraction_background: float) -> np.ndarray:
    """Threshold `image` so roughly `fraction_background` of pixels are zero.

    Returns a bool array where True is foreground (the brighter pixels). Both
    extremes are clamped so degenerate all-zero / all-one masks are avoided.
    """
    fraction = float(np.clip(fraction_background, 0.0, 1.0))
    if fraction <= 0.0:
        return np.ones(image.shape, dtype=bool)
    if fraction >= 1.0:
        return np.zeros(image.shape, dtype=bool)
    threshold = np.quantile(image, fraction)
    return image > threshold


def load_2d(spec: DataSpec) -> np.ndarray:
    from skimage import data

    image = data.camera().astype(np.float32)
    h, w = spec.shape
    if h > image.shape[0] or w > image.shape[1]:
        reps_h = math.ceil(h / image.shape[0])
        reps_w = math.ceil(w / image.shape[1])
        image = np.tile(image, (reps_h, reps_w))
    image = image[:h, :w]
    return np.ascontiguousarray(_quantile_mask(image, spec.fraction_background))


def load_3d(spec: DataSpec) -> np.ndarray:
    from skimage import data

    volume = data.cells3d()[:, 1].astype(np.float32)
    z, h, w = spec.shape
    reps_z = math.ceil(z / volume.shape[0]) if z > volume.shape[0] else 1
    reps_h = math.ceil(h / volume.shape[1]) if h > volume.shape[1] else 1
    reps_w = math.ceil(w / volume.shape[2]) if w > volume.shape[2] else 1
    if reps_z * reps_h * reps_w > 1:
        volume = np.tile(volume, (reps_z, reps_h, reps_w))
    volume = volume[:z, :h, :w]
    return np.ascontiguousarray(_quantile_mask(volume, spec.fraction_background))


def build_specs(args: argparse.Namespace) -> list[DataSpec]:
    if args.small and args.large:
        raise ValueError("--small and --large are mutually exclusive")

    if args.small:
        shape_2d = (256, 256)
        shape_3d = (16, 64, 64)
    elif args.large:
        shape_2d = (2048, 2048)
        shape_3d = (64, 512, 512)
    else:
        shape_2d = (512, 512)
        shape_3d = (32, 256, 256)

    specs = []
    if not args.no_2d:
        specs.append(DataSpec("2D", shape_2d, args.density))
    if not args.no_3d:
        specs.append(DataSpec("3D", shape_3d, args.density))
    return specs


# ---------------------------------------------------------------------------
# Library adapters
# ---------------------------------------------------------------------------

def _bic_distance(sampling: tuple[float, ...], n_threads: int):
    from bioimage_cpp import distance

    def fn(mask: np.ndarray) -> np.ndarray:
        return distance.distance_transform(
            mask, sampling=sampling, number_of_threads=n_threads
        )

    return fn


def _bic_vector(sampling: tuple[float, ...], n_threads: int):
    from bioimage_cpp import distance

    def fn(mask: np.ndarray) -> np.ndarray:
        return distance.vector_difference_transform(
            mask, sampling=sampling, number_of_threads=n_threads
        )

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

    sampling_array = np.asarray(sampling, dtype=np.float64)

    def fn(mask: np.ndarray) -> np.ndarray:
        _, indices = ndimage.distance_transform_edt(
            mask, sampling=sampling, return_indices=True
        )
        coords = np.indices(mask.shape, dtype=np.int32)
        vectors = np.moveaxis(indices - coords, 0, -1)
        return vectors * sampling_array.reshape((1,) * mask.ndim + (-1,))

    return fn


def _prepare_mask(library: str, base_mask: np.ndarray) -> np.ndarray:
    if library == "bioimage_cpp":
        # The Python wrapper fast-paths uint8 C-contiguous input.
        return np.ascontiguousarray(base_mask, dtype=np.uint8)
    if library in ("vigra", "scipy"):
        # vigra accepts float32 / uint32; scipy accepts any nonzero-tested
        # array. Match the conventional float32 mask both libraries are
        # typically called with.
        return np.ascontiguousarray(base_mask, dtype=np.float32)
    raise ValueError(f"unknown library: {library}")


def build_adapters(
    operation: str,
    sampling: tuple[float, ...],
    n_threads: int,
    base_mask: np.ndarray,
) -> dict[str, Adapter]:
    builders = {
        "distance_transform": {
            "bioimage_cpp": lambda: _bic_distance(sampling, n_threads),
            "vigra": lambda: _vigra_distance(sampling),
            "scipy": lambda: _scipy_distance(sampling),
        },
        "vector_difference_transform": {
            "bioimage_cpp": lambda: _bic_vector(sampling, n_threads),
            "vigra": lambda: _vigra_vector(sampling),
            "scipy": lambda: _scipy_vector(sampling),
        },
    }[operation]

    adapters: dict[str, Adapter] = {}
    for library, builder in builders.items():
        if library == "vigra" and not _import_available("vigra"):
            continue
        if library == "scipy" and not _import_available("scipy"):
            continue
        adapters[library] = Adapter(
            fn=builder(),
            mask=_prepare_mask(library, base_mask),
        )
    return adapters


# ---------------------------------------------------------------------------
# Timing and checks
# ---------------------------------------------------------------------------

def time_interleaved(
    adapters: dict[str, Adapter],
    repeats: int,
) -> dict[str, dict]:
    libs = list(adapters)
    for adapter in adapters.values():
        adapter.fn(adapter.mask)

    timings = {lib: [] for lib in libs}
    last_result = {}
    for repeat in range(repeats):
        rotation = repeat % len(libs)
        order = libs[rotation:] + libs[:rotation]
        for lib in order:
            adapter = adapters[lib]
            t0 = perf_counter()
            result = adapter.fn(adapter.mask)
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
    sampling: tuple[float, ...],
    adapters: dict[str, Adapter],
) -> dict[str, float]:
    if "scipy" not in adapters:
        return {}

    scipy_adapter = adapters["scipy"]
    reference_distance = _scipy_distance(sampling)(scipy_adapter.mask).astype(
        np.float32, copy=False
    )
    errors = {}
    for library, adapter in adapters.items():
        result = np.asarray(adapter.fn(adapter.mask))
        if operation == "distance_transform":
            errors[library] = float(
                np.max(np.abs(result.astype(np.float32) - reference_distance))
            )
        else:
            # Equidistant feature-index ties can choose different nearest
            # targets. Vector magnitudes must still match the distance map.
            magnitudes = np.linalg.norm(result.astype(np.float32), axis=-1)
            errors[library] = float(np.max(np.abs(magnitudes - reference_distance)))
    return errors


def format_results_table(rows: list[dict]) -> str:
    headers = ["operation", "dim", "shape", "bg %"]
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
            f"{row['fraction_background'] * 100:.1f}",
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
    if args.threads < 0:
        print("--threads must be >= 0", file=sys.stderr)
        return 2
    if not (0.0 <= args.density <= 1.0):
        print("--density must be in [0.0, 1.0]", file=sys.stderr)
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

    print(
        f"repeats={args.repeats}, sampling={args.sampling}, density={args.density}, "
        f"bioimage_cpp threads={args.threads}"
    )
    print("data: quantile-thresholded skimage camera/cells3d (tiled if smaller than the requested shape)")

    rows = []
    csv_rows = []
    for spec in specs:
        base_mask = load_2d(spec) if spec.label == "2D" else load_3d(spec)
        sampling = cfg.sampling_for(base_mask.ndim)
        actual_bg = float(np.count_nonzero(base_mask == 0)) / base_mask.size
        print(
            f"{spec.label}: shape={base_mask.shape}, background={int(np.count_nonzero(base_mask == 0))} "
            f"({actual_bg * 100:.1f}%), foreground={int(np.count_nonzero(base_mask != 0))}"
        )
        for operation in requested:
            adapters = build_adapters(operation, sampling, args.threads, base_mask)
            if not adapters:
                continue
            errors = (
                {} if args.skip_checks else check_results(operation, sampling, adapters)
            )
            results = time_interleaved(adapters, args.repeats)
            full_results = {lib: results.get(lib) for lib in LIBRARIES}
            rows.append({
                "operation": operation,
                "dim": spec.label,
                "shape": str(tuple(base_mask.shape)),
                "fraction_background": actual_bg,
                "results": full_results,
                "errors": errors,
            })
            for lib, result in full_results.items():
                if result is None:
                    continue
                csv_rows.append({
                    "operation": operation,
                    "dim": spec.label,
                    "shape": tuple(base_mask.shape),
                    "fraction_background": actual_bg,
                    "library": lib,
                    "median_s": result["median"],
                    "min_s": result["min"],
                    "max_abs_error": errors.get(lib, ""),
                    "repeats": args.repeats,
                    "threads": args.threads if lib == "bioimage_cpp" else 1,
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
                    "fraction_background",
                    "library",
                    "median_s",
                    "min_s",
                    "max_abs_error",
                    "repeats",
                    "threads",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"wrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
