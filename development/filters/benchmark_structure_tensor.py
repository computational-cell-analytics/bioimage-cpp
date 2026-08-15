"""Benchmark structure tensor components and selected eigenvectors.

The SciPy baseline implements the same two-scale Gaussian-derivative tensor as
bioimage-cpp. The native skimage baseline pre-smooths the image, then uses
Sobel derivatives through ``skimage.feature.structure_tensor``. These two
baselines do not produce identical tensors.

Run::

    python development/filters/benchmark_structure_tensor.py --small
    python development/filters/benchmark_structure_tensor.py --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations_with_replacement

import numpy as np
from scipy import ndimage
from skimage.feature import structure_tensor as skimage_structure_tensor

from _bench_utils import (
    BenchConfig,
    format_results_table,
    load_2d,
    load_3d,
    time_interleaved,
)


LIBRARIES = ("bioimage_cpp", "scipy_numpy", "skimage_numpy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark structure tensor components and selected eigenvectors."
    )
    parser.add_argument("--inner-sigma", type=float, default=1.0)
    parser.add_argument("--outer-sigma", type=float, default=2.0)
    parser.add_argument("--window-size", type=float, default=3.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--no-2d", action="store_true")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--csv")
    return parser.parse_args()


def scipy_components(image: np.ndarray, cfg: BenchConfig) -> np.ndarray:
    unit_orders = np.eye(image.ndim, dtype=int)
    gradients = [
        ndimage.gaussian_filter(
            image,
            cfg.inner_sigma,
            order=unit_orders[axis],
            mode="mirror",
            truncate=cfg.truncate,
        )
        for axis in range(image.ndim)
    ]
    return np.stack(
        [
            ndimage.gaussian_filter(
                gradients[row] * gradients[column],
                cfg.outer_sigma,
                mode="mirror",
                truncate=cfg.truncate,
            )
            for row, column in combinations_with_replacement(
                range(image.ndim), 2
            )
        ]
    )


def native_skimage_components(image: np.ndarray, cfg: BenchConfig) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(
        image,
        cfg.inner_sigma,
        mode="mirror",
        truncate=cfg.truncate,
    )
    return np.ascontiguousarray(
        skimage_structure_tensor(
            smoothed,
            sigma=cfg.outer_sigma,
            mode="mirror",
            order="rc",
        )
    )


def matrices_at_mask(components: np.ndarray, mask: np.ndarray) -> np.ndarray:
    matrix_dimension = 2 if components.shape[0] == 3 else 3
    coordinates = np.nonzero(mask)
    matrices = np.empty(
        (coordinates[0].size, matrix_dimension, matrix_dimension),
        dtype=components.dtype,
    )
    for component, (row, column) in zip(
        components,
        combinations_with_replacement(range(matrix_dimension), 2),
        strict=True,
    ):
        values = component[coordinates]
        matrices[:, row, column] = values
        matrices[:, column, row] = values
    return matrices


def numpy_smallest_eigenvector(
    components: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    matrix_dimension = 2 if components.shape[0] == 3 else 3
    matrices = matrices_at_mask(components, mask)
    _, eigenvectors = np.linalg.eigh(matrices)
    output = np.zeros(mask.shape + (matrix_dimension,), dtype=components.dtype)
    output[mask] = eigenvectors[..., 0]
    return output


def validate_exact_path(
    image: np.ndarray, mask: np.ndarray, cfg: BenchConfig
) -> None:
    from bioimage_cpp import filters as bf

    components = bf.structure_tensor(
        image,
        cfg.inner_sigma,
        cfg.outer_sigma,
        window_size=cfg.window_size,
    )
    reference_components = scipy_components(image, cfg)
    maximum_component_error = float(
        np.max(np.abs(components - reference_components), initial=0.0)
    )
    if maximum_component_error > 2e-3:
        raise RuntimeError(
            "structure tensor parity failed: maximum component error="
            f"{maximum_component_error:.3e}"
        )

    selected = bf.symmetric_eigenvector(
        components, image.ndim - 1, mask=mask
    )
    matrices = matrices_at_mask(components, mask)
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    got = selected[mask]
    expected = eigenvectors[..., 0]
    dots = np.abs(np.sum(got * expected, axis=-1))
    scale = np.maximum(
        np.linalg.norm(matrices, axis=(-2, -1)),
        np.finfo(components.dtype).tiny,
    )
    residual = np.linalg.norm(
        np.einsum("...ij,...j->...i", matrices, got)
        - eigenvalues[..., 0, None] * got,
        axis=-1,
    ) / scale
    if np.max(residual, initial=0.0) > 2e-5:
        raise RuntimeError(
            "selected eigenvector residual failed: maximum residual="
            f"{np.max(residual):.3e}"
        )

    if matrices.shape[0] != 0:
        gap = eigenvalues[..., 1] - eigenvalues[..., 0]
        unique = gap > 1e-4 * scale
        if np.any(unique) and np.min(dots[unique]) < 1.0 - 2e-4:
            raise RuntimeError(
                "selected eigenvector direction failed: minimum absolute dot="
                f"{np.min(dots[unique]):.6f}"
            )
    print(
        f"  parity: max component error={maximum_component_error:.3e}, "
        f"max eigenvector residual={np.max(residual, initial=0.0):.3e}"
    )


def summarize(results: dict[str, dict]) -> dict[str, dict]:
    return {
        library: {"median": result["median"], "min": result["min"]}
        for library, result in results.items()
    }


def benchmark_target(
    label: str,
    image: np.ndarray,
    cfg: BenchConfig,
    threshold: float,
    repeats: int,
) -> list[dict]:
    from bioimage_cpp import filters as bf

    mask = np.ascontiguousarray(image > threshold)
    selected_index = image.ndim - 1
    print(
        f"\n== {label}: shape={image.shape}, foreground={mask.mean():.1%} =="
    )
    validate_exact_path(image, mask, cfg)

    exact_components = scipy_components(image, cfg)
    stages = {
        "structure_tensor": {
            "bioimage_cpp": lambda value: bf.structure_tensor(
                value,
                cfg.inner_sigma,
                cfg.outer_sigma,
                window_size=cfg.window_size,
            ),
            "scipy_numpy": lambda value: scipy_components(value, cfg),
            "skimage_numpy": lambda value: native_skimage_components(value, cfg),
        },
        "smallest_eigenvector": {
            "bioimage_cpp": lambda _: bf.symmetric_eigenvector(
                exact_components, selected_index, mask=mask
            ),
            "scipy_numpy": lambda _: numpy_smallest_eigenvector(
                exact_components, mask
            ),
        },
        "end_to_end": {
            "bioimage_cpp": lambda value: bf.symmetric_eigenvector(
                bf.structure_tensor(
                    value,
                    cfg.inner_sigma,
                    cfg.outer_sigma,
                    window_size=cfg.window_size,
                ),
                selected_index,
                mask=mask,
            ),
            "scipy_numpy": lambda value: numpy_smallest_eigenvector(
                scipy_components(value, cfg), mask
            ),
            "skimage_numpy": lambda value: numpy_smallest_eigenvector(
                native_skimage_components(value, cfg), mask
            ),
        },
    }

    rows = []
    for stage, callables in stages.items():
        results = time_interleaved(callables, image, repeats)
        rows.append(
            {
                "filter": stage,
                "dim": label,
                "shape": str(tuple(image.shape)),
                "results": summarize(results),
            }
        )
    return rows


def write_csv(path: str, rows: list[dict], repeats: int) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "stage", "dim", "shape", "library", "median_s", "min_s", "repeats",
            ],
        )
        writer.writeheader()
        for row in rows:
            for library, result in row["results"].items():
                writer.writerow(
                    {
                        "stage": row["filter"],
                        "dim": row["dim"],
                        "shape": row["shape"],
                        "library": library,
                        "median_s": result["median"],
                        "min_s": result["min"],
                        "repeats": repeats,
                    }
                )


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    cfg = BenchConfig(
        inner_sigma=args.inner_sigma,
        outer_sigma=args.outer_sigma,
        window_size=args.window_size,
    )

    targets = []
    if not args.no_2d:
        targets.append(
            ("2D", load_2d(crop=(128, 128) if args.small else None))
        )
    if not args.no_3d:
        targets.append(
            ("3D", load_3d(crop=(16, 64, 64) if args.small else None))
        )
    if not targets:
        raise ValueError("enable at least one of the 2D and 3D benchmarks")

    print(
        f"inner_sigma={cfg.inner_sigma}, outer_sigma={cfg.outer_sigma}, "
        f"window_size={cfg.window_size}, threshold={args.mask_threshold}, "
        f"repeats={args.repeats}"
    )
    print(
        "skimage_numpy uses Sobel derivatives and native skimage Gaussian "
        "support; it is not a numerical-equivalence baseline."
    )

    rows = []
    for label, image in targets:
        rows.extend(
            benchmark_target(
                label,
                image,
                cfg,
                args.mask_threshold,
                args.repeats,
            )
        )
    print()
    print(format_results_table(rows, libraries=LIBRARIES))
    if args.csv:
        write_csv(args.csv, rows, args.repeats)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
