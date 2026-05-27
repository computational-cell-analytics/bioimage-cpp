"""Benchmark local-maxima detection: vigra vs. skimage on smoothed 2D/3D data.

This script exists to answer a single question: does bioimage-cpp need to
reimplement local-maxima / peak detection at all? If
``skimage.feature.peak_local_max`` performs comparably to
``vigra.analysis.localMaxima`` there is no reason to reimplement it — users can
just call skimage, and bioimage-cpp focuses on algorithms that are genuinely
hard to install.

The two APIs differ in shape but agree on the underlying detection:

* ``vigra.analysis.localMaxima`` (2D) / ``localMaxima3D`` (3D) return a *marked
  image* of the input shape, with maxima set to ``marker``. Plain
  neighborhood comparison (4/8 connectivity in 2D, 6/26 in 3D); no native
  minimum-distance suppression.
* ``skimage.feature.peak_local_max`` returns a *coordinate list* and additionally
  supports ``min_distance`` suppression (peaks separated by at least
  ``min_distance``).

With matched settings the two detect identical maxima at ``min_distance=1``:

* vigra 2D: ``localMaxima(img, neighborhood=8, allowAtBorder=True, allowPlateaus=True)``
* vigra 3D: ``localMaxima3D(vol, neighborhood=26, allowAtBorder=True, allowPlateaus=True)``
* skimage : ``peak_local_max(img, min_distance=1, exclude_border=False, threshold_abs=None)``

Both libraries are fed the same Gaussian-smoothed ``float32`` image, allocated
once outside the timing loop. ``min_distance == 1`` is the apples-to-apples
case (both libraries timed; identical maxima). ``min_distance > 1`` is the
realistic seed-detection case (skimage only, since vigra has no native
distance suppression).

Run::

    python development/local_maxima/benchmark.py --small --repeats 3
    python development/local_maxima/benchmark.py --repeats 5 --csv local_maxima.csv
    python development/local_maxima/benchmark.py --min-distance 3,5,10

Findings (sigma=2.0, repeats=5; vigra 1.11, skimage 0.25.2; single run, your
numbers will vary with hardware):

    dim  shape           min_dist  n_maxima  vigra ms  skimage ms  skimage/vigra
    2D   (512, 512)      1         1228      12.0      4.8         0.40
    2D   (512, 512)      5          572      n/a      12.0         n/a
    2D   (512, 512)      10         208      n/a       9.9         n/a
    3D   (32, 256, 256)  1          857     284.6     55.0         0.19
    3D   (32, 256, 256)  5          271      n/a      69.1         n/a
    3D   (32, 256, 256)  10          79      n/a      72.0         n/a

* At ``min_distance=1`` (apples-to-apples) the two libraries detect *identical*
  maxima (the agreement check reports 0 mismatches in both 2D and 3D), and
  ``skimage.peak_local_max`` is ~2.5x faster in 2D and ~5x faster in 3D than
  ``vigra.localMaxima`` (geomean skimage/vigra ~= 0.28).
* With realistic ``min_distance > 1`` suppression — which vigra has no native
  equivalent for — skimage stays at or below vigra's ``min_distance=1`` time.

Conclusion: ``peak_local_max`` is not just comparable to vigra but faster, and
additionally supports ``min_distance`` separation that vigra lacks. There is no
performance case for reimplementing local-maxima detection in bioimage-cpp;
users should call ``skimage.feature.peak_local_max`` directly.
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


LIBRARIES = ("vigra", "skimage")


@dataclass(frozen=True)
class DataSpec:
    label: str
    shape: tuple[int, ...]


@dataclass
class Adapter:
    fn: Callable[[np.ndarray], np.ndarray]
    image: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vigra.analysis.localMaxima vs skimage.feature.peak_local_max."
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--small", action="store_true", help="Fast smoke benchmark.")
    parser.add_argument("--large", action="store_true", help="Larger benchmark sizes.")
    parser.add_argument("--no-2d", action="store_true")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help=(
            "Gaussian smoothing sigma applied to the source image/volume. "
            "Smoothing yields realistic, well-separated peaks (0 = no smoothing)."
        ),
    )
    parser.add_argument(
        "--min-distance",
        default="1,5",
        help=(
            "Comma-separated skimage min_distance values to sweep. vigra is only "
            "comparable (and timed) at min_distance=1; larger values are skimage-only."
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write per-(dim, min_distance, library) timings.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip the one-shot vigra/skimage agreement check at min_distance=1.",
    )
    return parser.parse_args()


def _parse_min_distances(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError(f"min_distance values must be >= 1, got {value}")
        values.append(value)
    if not values:
        raise ValueError("--min-distance must contain at least one value")
    # Preserve order, drop duplicates.
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


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

def _smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.ascontiguousarray(image, dtype=np.float32)
    from scipy import ndimage

    smoothed = ndimage.gaussian_filter(image.astype(np.float32), sigma)
    return np.ascontiguousarray(smoothed, dtype=np.float32)


def load_2d(spec: DataSpec, sigma: float) -> np.ndarray:
    from skimage import data

    image = data.camera().astype(np.float32)
    h, w = spec.shape
    if h > image.shape[0] or w > image.shape[1]:
        reps_h = math.ceil(h / image.shape[0])
        reps_w = math.ceil(w / image.shape[1])
        image = np.tile(image, (reps_h, reps_w))
    image = image[:h, :w]
    return _smooth(image, sigma)


def load_3d(spec: DataSpec, sigma: float) -> np.ndarray:
    from skimage import data

    volume = data.cells3d()[:, 1].astype(np.float32)
    z, h, w = spec.shape
    reps_z = math.ceil(z / volume.shape[0]) if z > volume.shape[0] else 1
    reps_h = math.ceil(h / volume.shape[1]) if h > volume.shape[1] else 1
    reps_w = math.ceil(w / volume.shape[2]) if w > volume.shape[2] else 1
    if reps_z * reps_h * reps_w > 1:
        volume = np.tile(volume, (reps_z, reps_h, reps_w))
    volume = volume[:z, :h, :w]
    return _smooth(volume, sigma)


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
        specs.append(DataSpec("2D", shape_2d))
    if not args.no_3d:
        specs.append(DataSpec("3D", shape_3d))
    return specs


# ---------------------------------------------------------------------------
# Library adapters
# ---------------------------------------------------------------------------

def _skimage_peaks(min_distance: int):
    from skimage.feature import peak_local_max

    def fn(image: np.ndarray) -> np.ndarray:
        return peak_local_max(
            image,
            min_distance=min_distance,
            exclude_border=False,
            threshold_abs=None,
        )

    return fn


def _vigra_peaks():
    _quiet_vigra_matplotlib_cache()
    import vigra.analysis as va

    def fn(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            marked = va.localMaxima(
                image, neighborhood=8, allowAtBorder=True, allowPlateaus=True
            )
        elif image.ndim == 3:
            marked = va.localMaxima3D(
                image, neighborhood=26, allowAtBorder=True, allowPlateaus=True
            )
        else:
            raise ValueError(f"vigra local maxima supports 2D/3D, got ndim={image.ndim}")
        return np.asarray(marked)

    return fn


def _vigra_marked_to_coords(marked: np.ndarray) -> np.ndarray:
    return np.argwhere(marked > 0)


def build_adapters(min_distance: int, image: np.ndarray) -> dict[str, Adapter]:
    adapters: dict[str, Adapter] = {}
    # vigra has no native min_distance suppression, so it is only timed in the
    # apples-to-apples case (min_distance == 1).
    if min_distance == 1 and _import_available("vigra"):
        adapters["vigra"] = Adapter(fn=_vigra_peaks(), image=image)
    if _import_available("skimage"):
        adapters["skimage"] = Adapter(fn=_skimage_peaks(min_distance), image=image)
    return adapters


# ---------------------------------------------------------------------------
# Timing and checks
# ---------------------------------------------------------------------------

def _count_maxima(library: str, result: np.ndarray) -> int:
    # vigra returns a marked image; skimage returns a coordinate list.
    if library == "vigra":
        return int(np.count_nonzero(result > 0))
    return int(result.shape[0])


def time_interleaved(
    adapters: dict[str, Adapter],
    repeats: int,
) -> dict[str, dict]:
    libs = list(adapters)
    for adapter in adapters.values():
        adapter.fn(adapter.image)

    timings = {lib: [] for lib in libs}
    last_result = {}
    for repeat in range(repeats):
        rotation = repeat % len(libs)
        order = libs[rotation:] + libs[:rotation]
        for lib in order:
            adapter = adapters[lib]
            t0 = perf_counter()
            result = adapter.fn(adapter.image)
            timings[lib].append(perf_counter() - t0)
            last_result[lib] = np.asarray(result)

    return {
        lib: {
            "timings": timings[lib],
            "median": median(timings[lib]),
            "min": min(timings[lib]),
            "n_maxima": _count_maxima(lib, last_result[lib]),
            "result": last_result[lib],
        }
        for lib in libs
    }


def check_agreement(adapters: dict[str, Adapter]) -> dict | None:
    """Confirm vigra and skimage detect the same maxima (min_distance == 1)."""
    if "vigra" not in adapters or "skimage" not in adapters:
        return None

    marked = np.asarray(adapters["vigra"].fn(adapters["vigra"].image))
    vigra_coords = _vigra_marked_to_coords(marked)
    skimage_coords = np.asarray(adapters["skimage"].fn(adapters["skimage"].image))

    vigra_set = {tuple(int(c) for c in row) for row in vigra_coords}
    skimage_set = {tuple(int(c) for c in row) for row in skimage_coords}
    return {
        "n_vigra": len(vigra_set),
        "n_skimage": len(skimage_set),
        "mismatched": len(vigra_set ^ skimage_set),
    }


def format_results_table(rows: list[dict]) -> str:
    headers = [
        "dim", "shape", "sigma", "min_dist", "n_maxima",
        "vigra ms", "skimage ms", "skimage/vigra",
    ]

    table_rows = []
    for row in rows:
        vigra = row["results"].get("vigra")
        skimage = row["results"].get("skimage")
        n_maxima = (skimage or vigra)["n_maxima"] if (skimage or vigra) else 0
        vigra_ms = f"{vigra['median'] * 1e3:.3f}" if vigra else "n/a"
        skimage_ms = f"{skimage['median'] * 1e3:.3f}" if skimage else "n/a"
        if vigra and skimage and vigra["median"] > 0:
            ratio = f"{skimage['median'] / vigra['median']:.2f}"
        else:
            ratio = "n/a"
        table_rows.append([
            row["dim"],
            row["shape"],
            f"{row['sigma']:.1f}",
            str(row["min_distance"]),
            str(n_maxima),
            vigra_ms,
            skimage_ms,
            ratio,
        ])

    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def render(values: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [render(headers), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in table_rows)
    return "\n".join(lines)


def print_headline_ratio(rows: list[dict]) -> None:
    ratios = []
    for row in rows:
        if row["min_distance"] != 1:
            continue
        vigra = row["results"].get("vigra")
        skimage = row["results"].get("skimage")
        if vigra and skimage and vigra["median"] > 0:
            ratios.append(skimage["median"] / vigra["median"])
    print()
    if ratios:
        print(
            "speedup summary (geomean of skimage.median / vigra.median at "
            "min_distance=1; >1.0 means skimage slower):"
        )
        print(f"  skimage / vigra  geomean = {geometric_mean(ratios):.3f}  (n={len(ratios)})")
    else:
        print("no comparable (min_distance=1) rows with both libraries available.")


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2
    if args.sigma < 0.0:
        print("--sigma must be >= 0.0", file=sys.stderr)
        return 2

    try:
        min_distances = _parse_min_distances(args.min_distance)
        specs = build_specs(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not _import_available("skimage"):
        print("skimage is required for benchmark data and the skimage baseline", file=sys.stderr)
        return 2
    if args.sigma > 0.0 and not _import_available("scipy"):
        print("scipy is required for Gaussian smoothing (--sigma > 0)", file=sys.stderr)
        return 2
    if not _import_available("vigra"):
        print("warning: vigra not available; timing skimage only", file=sys.stderr)

    print(
        f"repeats={args.repeats}, sigma={args.sigma}, "
        f"min_distance={','.join(str(d) for d in min_distances)}"
    )
    print("data: Gaussian-smoothed skimage camera/cells3d (tiled if smaller than the requested shape)")

    rows = []
    csv_rows = []
    for spec in specs:
        image = load_2d(spec, args.sigma) if spec.label == "2D" else load_3d(spec, args.sigma)
        print(f"{spec.label}: shape={image.shape}, dtype={image.dtype}")
        for min_distance in min_distances:
            adapters = build_adapters(min_distance, image)
            if not adapters:
                continue
            agreement = None
            if not args.skip_checks and min_distance == 1:
                agreement = check_agreement(adapters)
            results = time_interleaved(adapters, args.repeats)
            full_results = {lib: results.get(lib) for lib in LIBRARIES}
            rows.append({
                "dim": spec.label,
                "shape": str(tuple(image.shape)),
                "sigma": args.sigma,
                "min_distance": min_distance,
                "results": full_results,
                "agreement": agreement,
            })
            for lib, result in full_results.items():
                if result is None:
                    continue
                csv_rows.append({
                    "dim": spec.label,
                    "shape": tuple(image.shape),
                    "sigma": args.sigma,
                    "min_distance": min_distance,
                    "library": lib,
                    "n_maxima": result["n_maxima"],
                    "median_s": result["median"],
                    "min_s": result["min"],
                    "repeats": args.repeats,
                })

    print()
    print(format_results_table(rows))

    if not args.skip_checks:
        agreement_rows = [row for row in rows if row.get("agreement")]
        if agreement_rows:
            print()
            print("vigra/skimage agreement at min_distance=1 (mismatched should be 0):")
            for row in agreement_rows:
                a = row["agreement"]
                print(
                    f"  {row['dim']}: n_vigra={a['n_vigra']}, "
                    f"n_skimage={a['n_skimage']}, mismatched={a['mismatched']}"
                )

    print_headline_ratio(rows)

    if args.csv is not None:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dim",
                    "shape",
                    "sigma",
                    "min_distance",
                    "library",
                    "n_maxima",
                    "median_s",
                    "min_s",
                    "repeats",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"wrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
