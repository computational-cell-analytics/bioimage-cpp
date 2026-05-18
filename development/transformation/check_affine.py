"""Correctness + runtime comparison for ``bic.transformation.affine_transform``.

Compares ``bioimage_cpp`` against:

- ``nifty.transformation.affineTransformation`` (orders 0 and 1 only; cubic is
  not supported by nifty's NumPy affine path),
- ``scipy.ndimage.affine_transform`` (orders 0, 1, and 3).

For each ``(image, matrix, bounding_box, order)`` case the script measures the
maximum absolute interior difference and the median + min wall-clock runtime
across several interleaved repeats.

Run::

    python development/transformation/check_affine.py
    python development/transformation/check_affine.py --small --repeats 3
    python development/transformation/check_affine.py --no-3d --orders 0,1

Notes:

- nifty only supports orders 0 (nearest) and 1 (linear); cubic cases are
  measured against scipy only.
- ``scipy.ndimage`` ``order=3`` is a (prefiltered) cubic B-spline.
  ``bioimage_cpp`` ``order=3`` is local Keys cubic convolution
  (a = -0.5). They disagree by design — the script reports the
  difference but does not gate on it.
- nifty's NumPy affine path treats the last index along each axis as
  out-of-bounds. To avoid that single-pixel rim biasing the diff, the
  comparison drops a configurable border on every axis before computing
  ``max|diff|``.

This script is part of the development tooling and is not exercised by the
test suite.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np


LIBRARIES: tuple[str, ...] = ("bioimage_cpp", "nifty", "scipy")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_2d(shape: tuple[int, int]) -> np.ndarray:
    """``skimage.data.camera`` cropped to ``shape``, float32 in [0, 1]."""
    from skimage import data
    arr = data.camera()
    arr = arr[: shape[0], : shape[1]]
    return np.ascontiguousarray(arr.astype(np.float32) / 255.0)


def _load_3d(shape: tuple[int, int, int]) -> np.ndarray:
    """``skimage.data.cells3d`` nuclei channel, float32 in [0, 1]."""
    from skimage import data
    vol = data.cells3d()[:, 1]
    vol = vol[: shape[0], : shape[1], : shape[2]]
    arr = vol.astype(np.float32)
    arr /= float(arr.max() if arr.max() > 0 else 1.0)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def _identity(ndim: int) -> np.ndarray:
    matrix = np.zeros((ndim, ndim + 1), dtype=np.float64)
    matrix[:, :ndim] = np.eye(ndim)
    return matrix


def _translation(ndim: int, t: list[float]) -> np.ndarray:
    matrix = _identity(ndim)
    matrix[:, ndim] = t
    return matrix


def _rotation_2d(angle: float, center: tuple[float, float]) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    cy, cx = center
    # Rotate around `center`: input = R @ (out - center) + center.
    translation = np.array([cy, cx]) - np.array([[c, -s], [s, c]]) @ np.array([cy, cx])
    return np.array(
        [[c, -s, translation[0]], [s, c, translation[1]]],
        dtype=np.float64,
    )


def _rotation_3d_about_z(angle: float, center: tuple[float, float, float]) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    cz, cy, cx = center
    lin = np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )
    translation = np.array([cz, cy, cx]) - lin @ np.array([cz, cy, cx])
    return np.hstack([lin, translation[:, None]])


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    label: str
    ndim: int
    image: np.ndarray
    matrix: np.ndarray
    bounding_box: tuple[slice, ...]
    order: int
    fill_value: float = 0.0


def _bic_run(case: Case) -> Callable[[], np.ndarray] | None:
    from bioimage_cpp.transformation import affine_transform
    image = case.image
    matrix = case.matrix
    bbox = case.bounding_box
    order = case.order
    fill = case.fill_value

    def fn():
        return affine_transform(
            image, matrix, bounding_box=bbox, order=order, fill_value=fill
        )

    return fn


def _nifty_run(case: Case) -> Callable[[], np.ndarray] | None:
    try:
        import nifty.transformation as nt
    except ImportError:
        return None
    if case.order not in (0, 1):
        return None  # nifty only supports orders 0 and 1
    image = case.image
    matrix = case.matrix
    bbox = case.bounding_box
    order = case.order
    fill = case.fill_value

    def fn():
        return nt.affineTransformation(image, matrix, order, bbox, fill)

    return fn


def _scipy_run(case: Case) -> Callable[[], np.ndarray] | None:
    try:
        from scipy.ndimage import affine_transform as sp_affine
    except ImportError:
        return None
    image = case.image
    M = case.matrix
    ndim = case.ndim
    lin = np.ascontiguousarray(M[:, :ndim])
    starts = np.array([bb.start for bb in case.bounding_box], dtype=np.float64)
    # Our convention: input_coord = lin @ (starts + out_coord) + M[:, -1].
    # scipy's:        input_coord = lin @  out_coord + offset.
    offset = lin @ starts + M[:, ndim]
    out_shape = tuple(int(bb.stop - bb.start) for bb in case.bounding_box)
    order = case.order
    fill = case.fill_value
    # We don't implement the spline prefilter (see PERFORMANCE_NOTES.md). Use
    # prefilter=False so scipy evaluates the raw B-spline kernel — that is the
    # variant we match for orders 2, 4, and 5. (Order 3 still differs by
    # design: we use Keys cubic, scipy uses cubic B-spline regardless of
    # prefilter.)
    prefilter = False

    # scipy has two "constant" modes: 'constant' implicitly extends the input
    # for tap evaluation and treats coords fully past the boundary as fill;
    # 'grid-constant' uses cval for every out-of-bounds tap. Our impl matches
    # the per-order semantic:
    #   * interpolating kernels (nearest, linear, Keys cubic) early-exit to
    #     fill_value when the coord is outside [0, shape-1] → scipy 'constant'.
    #   * B-spline smoothing kernels evaluate the kernel everywhere and pull
    #     in fill for out-of-bounds taps → scipy 'grid-constant'.
    if order in (2, 4, 5):
        mode = "grid-constant"
    else:
        mode = "constant"

    def fn():
        return sp_affine(
            image,
            lin,
            offset=offset,
            output_shape=out_shape,
            order=order,
            mode=mode,
            cval=fill,
            prefilter=prefilter,
        )

    return fn


ADAPTERS: dict[str, Callable[[Case], Callable[[], np.ndarray] | None]] = {
    "bioimage_cpp": _bic_run,
    "nifty": _nifty_run,
    "scipy": _scipy_run,
}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _build_cases(
    image_2d: np.ndarray | None,
    image_3d: np.ndarray | None,
    orders: list[int],
) -> list[Case]:
    cases: list[Case] = []

    if image_2d is not None:
        h, w = image_2d.shape
        full_bbox = (slice(0, h), slice(0, w))
        crop_bbox = (slice(h // 8, h - h // 8), slice(w // 8, w - w // 8))
        for order in orders:
            cases.append(Case("identity",   2, image_2d, _identity(2),                       full_bbox, order))
            cases.append(Case("translate",  2, image_2d, _translation(2, [3.25, -1.13]),     full_bbox, order))
            cases.append(Case("rotate30",   2, image_2d, _rotation_2d(np.deg2rad(30.0),
                                                                     (h / 2.0, w / 2.0)),    full_bbox, order))
            cases.append(Case("crop",       2, image_2d, _identity(2),                       crop_bbox, order))

    if image_3d is not None:
        d, h, w = image_3d.shape
        full_bbox = (slice(0, d), slice(0, h), slice(0, w))
        for order in orders:
            cases.append(Case("identity",   3, image_3d, _identity(3),                       full_bbox, order))
            cases.append(Case("translate",  3, image_3d, _translation(3, [0.37, -2.25, 0.91]), full_bbox, order))
            cases.append(Case("rotateZ20",  3, image_3d, _rotation_3d_about_z(np.deg2rad(20.0),
                                                                              (d / 2.0, h / 2.0, w / 2.0)),
                              full_bbox, order))

    return cases


# ---------------------------------------------------------------------------
# Correctness + timing
# ---------------------------------------------------------------------------

def _interior_slice(shape: tuple[int, ...], border: int) -> tuple[slice, ...]:
    return tuple(slice(min(border, dim), max(border, dim - border)) for dim in shape)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _time_interleaved(
    callables: dict[str, Callable[[], np.ndarray]],
    repeats: int,
) -> dict[str, dict]:
    libs = list(callables.keys())
    # One untimed warmup per library covers lazy init.
    for fn in callables.values():
        fn()

    timings: dict[str, list[float]] = {lib: [] for lib in libs}
    last_result: dict[str, np.ndarray] = {}
    n = len(libs)
    for r in range(repeats):
        rotation = r % n
        order = libs[rotation:] + libs[:rotation]
        for lib in order:
            fn = callables[lib]
            t0 = perf_counter()
            result = fn()
            t1 = perf_counter()
            timings[lib].append(t1 - t0)
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_ms(s: float | None) -> str:
    return "n/a" if s is None else f"{s * 1000.0:9.3f}"


def _format_report(rows: list[dict]) -> str:
    headers = ["case", "ndim", "order"]
    for lib in LIBRARIES:
        headers.append(f"{lib} ms")
        headers.append("x ours")
    headers.append("max|diff| (interior)")

    str_rows: list[list[str]] = []
    for row in rows:
        ref = row["results"].get("bioimage_cpp")
        line = [row["case"], str(row["ndim"]), str(row["order"])]
        for lib in LIBRARIES:
            r = row["results"].get(lib)
            if r is None:
                line.append("    skip")
                line.append("   -")
                continue
            line.append(_format_ms(r["median"]))
            if ref is None or lib == "bioimage_cpp" or r["median"] == 0.0:
                line.append("   -")
            else:
                # ours.median / lib.median: > 1 means lib is faster than us.
                line.append(f"{ref['median'] / r['median']:5.2f}")
        diff_parts = []
        for lib in LIBRARIES:
            if lib == "bioimage_cpp":
                continue
            if lib not in row["diffs"]:
                continue
            diff_parts.append(f"{lib}={row['diffs'][lib]:.2e}")
        line.append(", ".join(diff_parts) if diff_parts else "-")
        str_rows.append(line)

    widths = [max(len(h), *(len(r[i]) for r in str_rows)) for i, h in enumerate(headers)]
    sep = "  "
    out = [sep.join(h.ljust(w) for h, w in zip(headers, widths))]
    out.append(sep.join("-" * w for w in widths))
    for r in str_rows:
        out.append(sep.join(c.ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Equivalence + runtime comparison for affine_transform.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--small", action="store_true",
        help="Use small images for a fast smoke run.",
    )
    parser.add_argument("--no-2d", action="store_true")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument(
        "--orders", default="0,1,2,3,4,5",
        help="Comma-separated interpolation orders to compare.",
    )
    parser.add_argument(
        "--border", type=int, default=2,
        help="Number of pixels per axis to exclude when measuring max|diff| "
             "(default: 2). Hides boundary-handling differences between "
             "libraries — in particular, nifty's last-index-OOB bug.",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4,
        help="Interior tolerance for parity gating on order 1 (linear). "
             "Order 0 (nearest) diffs and order 3 (cubic) diffs vs scipy "
             "are reported but not gated: nearest tie-breaking conventions "
             "differ harmlessly between us and scipy/nifty, and our Keys "
             "cubic intentionally differs from scipy's spline cubic.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        orders = [int(o.strip()) for o in args.orders.split(",") if o.strip()]
    except ValueError:
        print(f"could not parse --orders={args.orders!r}", file=sys.stderr)
        return 2
    valid_orders = {0, 1, 2, 3, 4, 5}
    bad = [o for o in orders if o not in valid_orders]
    if bad:
        print(f"invalid orders: {bad}; valid: {sorted(valid_orders)}", file=sys.stderr)
        return 2

    shape_2d = (256, 256) if args.small else (512, 512)
    shape_3d = (16, 64, 64) if args.small else (40, 128, 128)

    image_2d = None if args.no_2d else _load_2d(shape_2d)
    image_3d = None if args.no_3d else _load_3d(shape_3d)

    cases = _build_cases(image_2d, image_3d, orders)
    if not cases:
        print("No cases to run (both --no-2d and --no-3d set?).", file=sys.stderr)
        return 0

    print("Affine transform comparison")
    print(f"  repeats={args.repeats}, border={args.border}, atol={args.atol}")
    if image_2d is not None:
        print(f"  2D shape: {tuple(image_2d.shape)} (dtype={image_2d.dtype})")
    if image_3d is not None:
        print(f"  3D shape: {tuple(image_3d.shape)} (dtype={image_3d.dtype})")
    print()

    rows: list[dict] = []
    any_failure = False
    for case in cases:
        callables = {
            lib: fn for lib, builder in ADAPTERS.items()
            if (fn := builder(case)) is not None
        }
        if "bioimage_cpp" not in callables:
            continue
        results = _time_interleaved(callables, args.repeats)

        ref = results["bioimage_cpp"]["result"]
        spatial_slice = _interior_slice(ref.shape, args.border)
        diffs: dict[str, float] = {}
        for lib in LIBRARIES:
            if lib == "bioimage_cpp" or lib not in results:
                continue
            res = results[lib]["result"]
            if res.shape != ref.shape:
                print(
                    f"  shape mismatch for case={case.label} order={case.order} "
                    f"lib={lib}: {res.shape} vs {ref.shape}",
                    file=sys.stderr,
                )
                diffs[lib] = float("inf")
                any_failure = True
                continue
            d = _max_abs_diff(ref[spatial_slice], res[spatial_slice])
            diffs[lib] = d
            # Parity gate:
            #   * order 1 (linear) — gated against both libraries.
            #   * orders 2, 4, 5 (B-spline) — gated against scipy
            #     (with prefilter=False). nifty doesn't implement these.
            #   * order 0 — not gated: nearest-neighbor tie-breaking
            #     convention differs harmlessly between libraries.
            #   * order 3 — not gated vs scipy: we use Keys cubic, scipy
            #     uses B-spline cubic.
            gate = False
            if case.order == 1:
                gate = True
            elif case.order in (2, 4, 5) and lib == "scipy":
                gate = True
            if gate and d > args.atol:
                any_failure = True

        rows.append({
            "case": case.label,
            "ndim": case.ndim,
            "order": case.order,
            "results": {lib: results.get(lib) for lib in LIBRARIES},
            "diffs": diffs,
        })

    print(_format_report(rows))
    print()
    print(
        "Note: scipy is invoked with prefilter=False so that its B-spline "
        "orders match ours (we do not implement the IIR prefilter that "
        "scipy's default prefilter=True applies to orders >= 2). See "
        "PERFORMANCE_NOTES.md."
    )
    print(
        "Note: order=0 (nearest) disagreements at non-zero levels come from "
        "tie-breaking: we use round-half-up, scipy/nifty use round-half-to-even. "
        "Order=0 diffs are reported but not gated."
    )
    print(
        "Note: order=3 is intentionally different from scipy: we use Keys "
        "cubic (a=-0.5, interpolating); scipy uses cubic B-spline. The cubic "
        "diff is expected to be non-zero and is not gated."
    )
    print(
        "Note: 'x ours' = our median / library median. Values > 1 mean the "
        "library is faster than us; values < 1 mean it is slower."
    )
    print()
    print("FAILURE" if any_failure else "OK")
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
