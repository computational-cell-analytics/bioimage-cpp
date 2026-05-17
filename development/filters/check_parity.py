"""Equivalence check across scipy / vigra / fastfilters / bioimage_cpp filters.

Compares every library's output against ``bioimage_cpp`` on the interior of
the image (a margin of ``2 * ceil(window_size * sigma)`` pixels is dropped on
every axis to avoid boundary-mode differences between libraries).

Run::

    python development/filters/check_parity.py [--sigma 1.5] [--no-3d]

Exits non-zero on any tolerance failure.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from _bench_utils import (
    ADAPTERS,
    BenchConfig,
    FILTERS,
    LIBRARIES,
    build_adapters,
    interior_slice,
    load_2d,
    load_3d,
    parity_atol_for_filter,
    parity_border_for_filter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Equivalence check across filter implementations."
    )
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--inner-sigma", type=float, default=1.0)
    parser.add_argument("--outer-sigma", type=float, default=2.0)
    parser.add_argument("--window-size", type=float, default=3.0)
    parser.add_argument("--atol", type=float, default=None,
                        help="Override per-filter tolerance.")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--no-2d", action="store_true")
    parser.add_argument(
        "--filters", default=",".join(FILTERS),
        help="Comma-separated subset of filters to check.",
    )
    return parser.parse_args()


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _run_one(filter_name: str, image: np.ndarray, cfg: BenchConfig, atol: float) -> bool:
    adapters = build_adapters(filter_name, cfg)
    border = parity_border_for_filter(filter_name, cfg)
    spatial = image.shape
    spatial_slice = interior_slice(spatial, border)

    ref_fn = adapters["bioimage_cpp"]
    ref = np.asarray(ref_fn(image))
    # The slice for the trailing eigenvalue axis (if any) is full.
    full_slice = spatial_slice + (slice(None),) * (ref.ndim - len(spatial))
    ref_interior = ref[full_slice]

    all_ok = True
    for lib in LIBRARIES:
        if lib == "bioimage_cpp":
            continue
        fn = adapters[lib]
        if fn is None:
            print(f"  {filter_name:<32s} {lib:<12s}  SKIP (not supported)")
            continue
        got = np.asarray(fn(image))
        if got.shape != ref.shape:
            print(
                f"  {filter_name:<32s} {lib:<12s}  FAIL "
                f"(shape mismatch: {got.shape} vs {ref.shape})"
            )
            all_ok = False
            continue
        got_interior = got[full_slice]
        diff = _max_abs_diff(ref_interior, got_interior)
        ok = diff <= atol
        status = "PASS" if ok else "FAIL"
        print(
            f"  {filter_name:<32s} {lib:<12s}  {status}  max|diff|={diff:.3e}"
            f"  (atol={atol:.1e}, border={border})"
        )
        if not ok:
            all_ok = False
    return all_ok


def main() -> int:
    args = parse_args()
    cfg = BenchConfig(
        sigma=args.sigma,
        inner_sigma=args.inner_sigma,
        outer_sigma=args.outer_sigma,
        window_size=args.window_size,
    )
    requested = [f.strip() for f in args.filters.split(",") if f.strip()]
    unknown = [f for f in requested if f not in FILTERS]
    if unknown:
        print(f"unknown filter(s): {unknown}", file=sys.stderr)
        return 2

    targets = []
    if not args.no_2d:
        targets.append(("2D", load_2d()))
    if not args.no_3d:
        targets.append(("3D", load_3d()))

    any_failure = False
    for dim_label, image in targets:
        print(f"\n== {dim_label} parity (shape={image.shape}, dtype={image.dtype}) ==")
        for filter_name in requested:
            atol = args.atol if args.atol is not None else parity_atol_for_filter(filter_name)
            ok = _run_one(filter_name, image, cfg, atol)
            if not ok:
                any_failure = True

    print("\n" + ("FAILURE" if any_failure else "OK"))
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
