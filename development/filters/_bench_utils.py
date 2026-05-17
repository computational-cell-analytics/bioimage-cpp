"""Shared helpers for the filters benchmark scripts.

This module is intentionally not part of the test suite. It provides:

* data loaders that prepare the same float32 array for every library
* one adapter per (library, filter) that hides parameter-name and output-shape
  differences so the harness can call any adapter as ``fn(image)``
* a small interleaved timing harness and a fixed-width report formatter
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np


LIBRARIES: tuple[str, ...] = ("bioimage_cpp", "fastfilters", "vigra", "scipy")
FILTERS: tuple[str, ...] = (
    "gaussian_smoothing",
    "gaussian_derivative",
    "gaussian_gradient_magnitude",
    "laplacian_of_gaussian",
    "hessian_of_gaussian_eigenvalues",
    "structure_tensor_eigenvalues",
)

# Per-axis order tuple used for gaussian_derivative. Convention: first
# derivative along the trailing spatial axis ("d/dx"), zero elsewhere.
DERIVATIVE_ORDER_2D = (0, 1)
DERIVATIVE_ORDER_3D = (0, 0, 1)

NOT_APPLICABLE = "n/a"


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def load_2d(*, crop: tuple[int, int] | None = None) -> np.ndarray:
    """skimage.data.camera() as contiguous float32 in [0, 1]."""
    from skimage import data
    arr = data.camera()
    if crop is not None:
        arr = arr[: crop[0], : crop[1]]
    return np.ascontiguousarray(arr.astype(np.float32) / 255.0)


def load_3d(*, crop: tuple[int, int, int] | None = None) -> np.ndarray:
    """skimage.data.cells3d() nuclei channel as contiguous float32 in [0, 1]."""
    from skimage import data
    vol = data.cells3d()[:, 1]  # (60, 256, 256), uint16
    if crop is not None:
        vol = vol[: crop[0], : crop[1], : crop[2]]
    arr = vol.astype(np.float32)
    arr /= float(arr.max() if arr.max() > 0 else 1.0)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Bench config: knobs every adapter must respect for fair comparison.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchConfig:
    sigma: float = 1.5
    inner_sigma: float = 1.0
    outer_sigma: float = 2.0
    # window_size in vigra / fastfilters / bioimage_cpp and truncate in scipy
    # both mean "kernel half-width / sigma". Fixed to make kernel sizes match
    # across libraries.
    window_size: float = 3.0

    @property
    def truncate(self) -> float:
        return self.window_size

    def derivative_order(self, ndim: int) -> tuple[int, ...]:
        return DERIVATIVE_ORDER_2D if ndim == 2 else DERIVATIVE_ORDER_3D


# ---------------------------------------------------------------------------
# Adapters: each builder returns a callable fn(image) -> np.ndarray.
# A None return value indicates the library does not support that filter.
# ---------------------------------------------------------------------------

# ---- bioimage_cpp adapters ------------------------------------------------

def _bic_smoothing(cfg: BenchConfig):
    from bioimage_cpp import filters as bf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return bf.gaussian_smoothing(image, sigma, window_size=ws)
    return fn


def _bic_derivative(cfg: BenchConfig):
    from bioimage_cpp import filters as bf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        order = cfg.derivative_order(image.ndim)
        return bf.gaussian_derivative(image, sigma, order, window_size=ws)
    return fn


def _bic_grad_magnitude(cfg: BenchConfig):
    from bioimage_cpp import filters as bf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return bf.gaussian_gradient_magnitude(image, sigma, window_size=ws)
    return fn


def _bic_log(cfg: BenchConfig):
    from bioimage_cpp import filters as bf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return bf.laplacian_of_gaussian(image, sigma, window_size=ws)
    return fn


def _bic_hessian_ev(cfg: BenchConfig):
    from bioimage_cpp import filters as bf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return bf.hessian_of_gaussian_eigenvalues(image, sigma, window_size=ws)
    return fn


def _bic_st_ev(cfg: BenchConfig):
    from bioimage_cpp import filters as bf
    inner, outer, ws = cfg.inner_sigma, cfg.outer_sigma, cfg.window_size
    def fn(image):
        return bf.structure_tensor_eigenvalues(image, inner, outer, window_size=ws)
    return fn


# ---- fastfilters adapters -------------------------------------------------

def _ff_smoothing(cfg: BenchConfig):
    import fastfilters as ff
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return ff.gaussianSmoothing(image, sigma, window_size=ws)
    return fn


def _ff_derivative(cfg: BenchConfig):
    # fastfilters only supports uniform per-axis order; our derivative
    # benchmark is order=1 along the last axis only, so fastfilters cannot
    # match the operation. Mark as not applicable.
    return None


def _ff_grad_magnitude(cfg: BenchConfig):
    import fastfilters as ff
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return ff.gaussianGradientMagnitude(image, sigma, window_size=ws)
    return fn


def _ff_log(cfg: BenchConfig):
    import fastfilters as ff
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return ff.laplacianOfGaussian(image, sigma, window_size=ws)
    return fn


def _ff_hessian_ev(cfg: BenchConfig):
    import fastfilters as ff
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return ff.hessianOfGaussianEigenvalues(image, sigma, window_size=ws)
    return fn


def _ff_st_ev(cfg: BenchConfig):
    import fastfilters as ff
    inner, outer, ws = cfg.inner_sigma, cfg.outer_sigma, cfg.window_size
    # fastfilters' Python wrapper calls the C function as
    #   fastfilters_fir_structure_tensor2d(in, sigma_inner, sigma_outer, ...)
    # but the C signature is (sigma_outer, sigma_inner) — see src/python/core.cxx
    # line 328 vs src/library/fir_filters.c line 156 in the fastfilters source.
    # The two scales are therefore swapped at the Python boundary relative to
    # vigra / scipy / bioimage_cpp. Swap them back so this adapter computes
    # the same operation as the rest.
    def fn(image):
        return ff.structureTensorEigenvalues(image, innerScale=outer, outerScale=inner, window_size=ws)
    return fn


# ---- vigra adapters -------------------------------------------------------

def _vigra_smoothing(cfg: BenchConfig):
    import vigra.filters as vf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return np.asarray(vf.gaussianSmoothing(image, sigma, window_size=ws))
    return fn


def _vigra_derivative(cfg: BenchConfig):
    import vigra.filters as vf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        order = list(cfg.derivative_order(image.ndim))
        return np.asarray(vf.gaussianDerivative(image, sigma, order, window_size=ws))
    return fn


def _vigra_grad_magnitude(cfg: BenchConfig):
    import vigra.filters as vf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return np.asarray(vf.gaussianGradientMagnitude(image, sigma, window_size=ws))
    return fn


def _vigra_log(cfg: BenchConfig):
    import vigra.filters as vf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return np.asarray(vf.laplacianOfGaussian(image, scale=sigma, window_size=ws))
    return fn


def _vigra_hessian_ev(cfg: BenchConfig):
    import vigra.filters as vf
    sigma, ws = cfg.sigma, cfg.window_size
    def fn(image):
        return np.asarray(vf.hessianOfGaussianEigenvalues(image, scale=sigma, window_size=ws))
    return fn


def _vigra_st_ev(cfg: BenchConfig):
    import vigra.filters as vf
    inner, outer, ws = cfg.inner_sigma, cfg.outer_sigma, cfg.window_size
    def fn(image):
        return np.asarray(
            vf.structureTensorEigenvalues(
                image, innerScale=inner, outerScale=outer, window_size=ws,
            )
        )
    return fn


# ---- scipy / numpy adapters ----------------------------------------------
#
# The two eigenvalue filters do not have a direct scipy entry point; we hand-
# build the components using scipy.ndimage gaussian filters with matching
# kernel support and use numpy.linalg.eigvalsh per pixel. This is what a
# scipy user would write and represents the realistic scipy-only baseline.

def _scipy_smoothing(cfg: BenchConfig):
    from scipy import ndimage
    sigma, tr = cfg.sigma, cfg.truncate
    def fn(image):
        return ndimage.gaussian_filter(image, sigma, mode="mirror", truncate=tr)
    return fn


def _scipy_derivative(cfg: BenchConfig):
    from scipy import ndimage
    sigma, tr = cfg.sigma, cfg.truncate
    def fn(image):
        order = list(cfg.derivative_order(image.ndim))
        return ndimage.gaussian_filter(image, sigma, order=order, mode="mirror", truncate=tr)
    return fn


def _scipy_grad_magnitude(cfg: BenchConfig):
    from scipy import ndimage
    sigma, tr = cfg.sigma, cfg.truncate
    def fn(image):
        return ndimage.gaussian_gradient_magnitude(
            image, sigma, mode="mirror", truncate=tr,
        )
    return fn


def _scipy_log(cfg: BenchConfig):
    from scipy import ndimage
    sigma, tr = cfg.sigma, cfg.truncate
    def fn(image):
        return ndimage.gaussian_laplace(image, sigma, mode="mirror", truncate=tr)
    return fn


def _scipy_hessian_ev(cfg: BenchConfig):
    from scipy import ndimage
    sigma, tr = cfg.sigma, cfg.truncate

    def _components_2d(image):
        hyy = ndimage.gaussian_filter(image, sigma, order=[2, 0], mode="mirror", truncate=tr)
        hyx = ndimage.gaussian_filter(image, sigma, order=[1, 1], mode="mirror", truncate=tr)
        hxx = ndimage.gaussian_filter(image, sigma, order=[0, 2], mode="mirror", truncate=tr)
        return hyy, hyx, hxx

    def _components_3d(image):
        return (
            ndimage.gaussian_filter(image, sigma, order=[2, 0, 0], mode="mirror", truncate=tr),
            ndimage.gaussian_filter(image, sigma, order=[1, 1, 0], mode="mirror", truncate=tr),
            ndimage.gaussian_filter(image, sigma, order=[1, 0, 1], mode="mirror", truncate=tr),
            ndimage.gaussian_filter(image, sigma, order=[0, 2, 0], mode="mirror", truncate=tr),
            ndimage.gaussian_filter(image, sigma, order=[0, 1, 1], mode="mirror", truncate=tr),
            ndimage.gaussian_filter(image, sigma, order=[0, 0, 2], mode="mirror", truncate=tr),
        )

    def fn(image):
        if image.ndim == 2:
            hyy, hyx, hxx = _components_2d(image)
            mat = np.stack(
                [np.stack([hyy, hyx], axis=-1), np.stack([hyx, hxx], axis=-1)],
                axis=-2,
            )
        else:
            hzz, hzy, hzx, hyy, hyx, hxx = _components_3d(image)
            mat = np.stack([
                np.stack([hzz, hzy, hzx], axis=-1),
                np.stack([hzy, hyy, hyx], axis=-1),
                np.stack([hzx, hyx, hxx], axis=-1),
            ], axis=-2)
        evs = np.linalg.eigvalsh(mat)[..., ::-1]
        return evs.astype(np.float32, copy=False)

    return fn


def _scipy_st_ev(cfg: BenchConfig):
    from scipy import ndimage
    inner, outer, tr = cfg.inner_sigma, cfg.outer_sigma, cfg.truncate

    def _grads(image):
        unit = np.eye(image.ndim, dtype=int)
        return [
            ndimage.gaussian_filter(image, inner, order=list(unit[i]),
                                     mode="mirror", truncate=tr)
            for i in range(image.ndim)
        ]

    def fn(image):
        grads = _grads(image)
        comps = {}
        for i in range(image.ndim):
            for j in range(i, image.ndim):
                comps[(i, j)] = ndimage.gaussian_filter(
                    grads[i] * grads[j], outer, mode="mirror", truncate=tr,
                )
        if image.ndim == 2:
            mat = np.stack([
                np.stack([comps[(0, 0)], comps[(0, 1)]], axis=-1),
                np.stack([comps[(0, 1)], comps[(1, 1)]], axis=-1),
            ], axis=-2)
        else:
            mat = np.stack([
                np.stack([comps[(0, 0)], comps[(0, 1)], comps[(0, 2)]], axis=-1),
                np.stack([comps[(0, 1)], comps[(1, 1)], comps[(1, 2)]], axis=-1),
                np.stack([comps[(0, 2)], comps[(1, 2)], comps[(2, 2)]], axis=-1),
            ], axis=-2)
        evs = np.linalg.eigvalsh(mat)[..., ::-1]
        return evs.astype(np.float32, copy=False)

    return fn


# ---------------------------------------------------------------------------
# Adapter table
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, dict[str, Callable[[BenchConfig], Callable[[np.ndarray], np.ndarray] | None]]] = {
    "gaussian_smoothing": {
        "bioimage_cpp": _bic_smoothing,
        "fastfilters": _ff_smoothing,
        "vigra": _vigra_smoothing,
        "scipy": _scipy_smoothing,
    },
    "gaussian_derivative": {
        "bioimage_cpp": _bic_derivative,
        "fastfilters": _ff_derivative,
        "vigra": _vigra_derivative,
        "scipy": _scipy_derivative,
    },
    "gaussian_gradient_magnitude": {
        "bioimage_cpp": _bic_grad_magnitude,
        "fastfilters": _ff_grad_magnitude,
        "vigra": _vigra_grad_magnitude,
        "scipy": _scipy_grad_magnitude,
    },
    "laplacian_of_gaussian": {
        "bioimage_cpp": _bic_log,
        "fastfilters": _ff_log,
        "vigra": _vigra_log,
        "scipy": _scipy_log,
    },
    "hessian_of_gaussian_eigenvalues": {
        "bioimage_cpp": _bic_hessian_ev,
        "fastfilters": _ff_hessian_ev,
        "vigra": _vigra_hessian_ev,
        "scipy": _scipy_hessian_ev,
    },
    "structure_tensor_eigenvalues": {
        "bioimage_cpp": _bic_st_ev,
        "fastfilters": _ff_st_ev,
        "vigra": _vigra_st_ev,
        "scipy": _scipy_st_ev,
    },
}


def build_adapters(filter_name: str, cfg: BenchConfig) -> dict[str, Callable | None]:
    """Build a dict {library: callable_or_None} for one filter."""
    return {lib: builder(cfg) for lib, builder in ADAPTERS[filter_name].items()}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_interleaved(
    callables: dict[str, Callable[[np.ndarray], np.ndarray]],
    image: np.ndarray,
    repeats: int,
) -> dict[str, dict]:
    """Run each callable ``repeats`` times in round-robin order.

    Returns {library: {"timings": [s, ...], "median": s, "min": s,
                        "result": last_result_array}}.
    The order in which callables are timed is rotated by repeat index so that
    no library is systematically advantaged by cache state.
    """
    libs = list(callables.keys())
    # One untimed warmup call per library covers lazy init.
    for fn in callables.values():
        fn(image)

    timings: dict[str, list[float]] = {lib: [] for lib in libs}
    last_result: dict[str, np.ndarray] = {}

    n = len(libs)
    for r in range(repeats):
        rotation = r % n
        order = libs[rotation:] + libs[:rotation]
        for lib in order:
            fn = callables[lib]
            t0 = perf_counter()
            result = fn(image)
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

def format_results_table(
    rows: list[dict],
    *,
    reference_library: str = "bioimage_cpp",
    libraries: Sequence[str] = LIBRARIES,
) -> str:
    """Render a fixed-width text table.

    Each row is a dict with at least:
      - "filter": str
      - "dim": str (e.g. "2D" or "3D")
      - "shape": str (e.g. "(512, 512)")
      - "results": dict {lib: {"median": s, "min": s} or None}
    """
    headers = ["filter", "dim", "shape"]
    for lib in libraries:
        headers.append(f"{lib} ms")
        # "x ours" reads as "speedup factor over bioimage_cpp": ours_time /
        # this_lib_time, so >1 means this lib is faster than ours.
        headers.append("x ours")

    str_rows: list[list[str]] = []
    for row in rows:
        ref = row["results"].get(reference_library) or {}
        ref_median = ref.get("median") if ref else None
        line = [row["filter"], row["dim"], row["shape"]]
        for lib in libraries:
            r = row["results"].get(lib)
            if r is None:
                line.append(NOT_APPLICABLE)
                line.append(NOT_APPLICABLE)
            else:
                line.append(f"{r['median'] * 1e3:.2f}")
                if ref_median is None or lib == reference_library:
                    line.append("1.00" if lib == reference_library else "-")
                else:
                    speedup = ref_median / r["median"]
                    line.append(f"{speedup:.2f}")
        str_rows.append(line)

    # Column widths
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def render_row(values):
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    out = [render_row(headers), render_row(["-" * w for w in widths])]
    out.extend(render_row(r) for r in str_rows)
    return "\n".join(out)


def interior_slice(image_shape: tuple[int, ...], border: int) -> tuple[slice, ...]:
    """Slice that drops `border` pixels on each side of every spatial axis.
    If the array has more axes than `len(image_shape)` (e.g. trailing
    eigenvalue axis), those are passed through unsliced."""
    base = tuple(slice(border, size - border) for size in image_shape)
    return base


def parity_atol_for_filter(filter_name: str) -> float:
    # Per-filter float32 tolerance budgets:
    # - smoothing / derivative / gradient_magnitude: dominated by single
    #   separable-Gaussian rounding; 2e-3 is comfortable.
    # - LoG: sum of two second-derivatives accumulates a touch more noise;
    #   scipy's slightly different truncation rule pushes us over 2e-3.
    # - eigenvalues: sqrt/acos/trig + sort tie-breaking near plateaus,
    #   plus two convolutions for the structure tensor.
    if filter_name == "laplacian_of_gaussian":
        return 5e-3
    if filter_name.endswith("_eigenvalues"):
        return 5e-3
    return 2e-3


def parity_border_for_filter(filter_name: str, cfg: "BenchConfig") -> int:
    """How many pixels to drop on each side before comparing.

    The structure-tensor pipeline applies *two* Gaussians (gradient at
    ``inner_sigma``, smoothing at ``outer_sigma``), so its boundary footprint
    is the sum of both radii. Other filters only see ``sigma`` once.
    """
    ws = cfg.window_size
    if filter_name == "structure_tensor_eigenvalues":
        return int(math.ceil(ws * (cfg.inner_sigma + cfg.outer_sigma)))
    return int(math.ceil(ws * cfg.sigma))


# Imports kept at the bottom for `parity_border_for_filter` annotation.
import math  # noqa: E402
