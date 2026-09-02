"""Shared inputs and helpers for the flow-density development harnesses.

Used by ``perf_kernel.py``, ``paired_bench.py`` and ``differential_check.py``.
Not part of the package or the test suite.

``preload_core`` lets a harness load ``bioimage_cpp._core`` from an explicit
``.so`` path before ``import bioimage_cpp`` runs. This is how two prebuilt
kernels are compared without rebuilding between A and B (rebuild rounds drift
thermally and the editable install otherwise loads ``_core`` from
site-packages).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULTS: dict = {
    "n_iter": 50,
    "dt": 0.2,
    "tol": 0.005,
    "method": "rk2",
    "restrict_to_mask": True,
}


def preload_core(so_path: str | os.PathLike | None) -> None:
    """Pre-seed ``sys.modules['bioimage_cpp._core']`` from ``so_path``.

    Must be called before ``bioimage_cpp`` is imported. A ``None`` path is a
    no-op (the installed extension is used).
    """
    if so_path is None:
        return
    if "bioimage_cpp" in sys.modules or "bioimage_cpp._core" in sys.modules:
        raise RuntimeError("preload_core must run before bioimage_cpp is imported")
    resolved = str(Path(so_path).resolve())
    spec = importlib.util.spec_from_file_location("bioimage_cpp._core", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create an import spec for {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["bioimage_cpp._core"] = module
    spec.loader.exec_module(module)
    import bioimage_cpp

    # A pre-seeded sys.modules entry is not bound as a package attribute by
    # the import system, so do it explicitly for `bioimage_cpp._core` users.
    bioimage_cpp._core = module
    loaded = sys.modules["bioimage_cpp._core"].__file__
    if loaded != resolved:
        raise RuntimeError(f"expected _core from {resolved}, got {loaded}")


def pin_cpus(cpus: str | None) -> list[int]:
    """Pin the current process to ``cpus`` ('36,37') before allocating data."""
    if not cpus:
        return sorted(os.sched_getaffinity(0))
    selected = {int(c) for c in cpus.split(",") if c}
    os.sched_setaffinity(0, selected)
    return sorted(selected)


def load_fixture(ndim: int, timeout: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Registered fixture as ``(flow float32, mask bool)`` (see check_flow_density.py)."""
    from bioimage_cpp._data import load_flow_data

    dist, fg, _ = load_flow_data(ndim, timeout=timeout)
    flow = np.ascontiguousarray(-dist, dtype=np.float32)
    mask = np.ascontiguousarray(fg > 0.5)
    return flow, mask


def _stripes(n_orbiter_columns_of_4: int) -> tuple[np.ndarray, np.ndarray]:
    from benchmark_interleave import SHAPE_3D, _stripe_flow

    return _stripe_flow(n_orbiter_columns_of_4), np.ones(SHAPE_3D, dtype=bool)


def _random3d(scale: float) -> tuple[np.ndarray, np.ndarray]:
    from benchmark_interleave import SHAPE_3D

    rng = np.random.default_rng(0)
    flow = rng.normal(scale=scale, size=(3,) + SHAPE_3D).astype(np.float32)
    return flow, np.ones(SHAPE_3D, dtype=bool)


def _random2d(scale: float) -> tuple[np.ndarray, np.ndarray]:
    from benchmark_midpoint_reuse import SHAPE_2D

    rng = np.random.default_rng(0)
    flow = rng.normal(scale=scale, size=(2,) + SHAPE_2D).astype(np.float32)
    return flow, np.ones(SHAPE_2D, dtype=bool)


# name -> zero-argument builder returning (flow, mask). The random cases use a
# fresh default_rng(0) each, so they are deterministic but not bit-identical to
# the sequentially drawn arrays inside benchmark_interleave/midpoint_reuse.
CASES: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {
    "fixture3d": lambda: load_fixture(3),
    "fixture2d": lambda: load_fixture(2),
    "stripes1": lambda: _stripes(1),
    "stripes3": lambda: _stripes(3),
    "random1": lambda: _random3d(1.0),
    "random10": lambda: _random3d(10.0),
    "sweep5": lambda: _random3d(5.0),
    "sweep20": lambda: _random3d(20.0),
    "sweep40": lambda: _random3d(40.0),
    "sweep2d10": lambda: _random2d(10.0),
}

FIXTURE_CASES = ("fixture3d", "fixture2d")
ADVERSARIAL_CASES = (
    "stripes1", "stripes3", "random1", "random10",
    "sweep5", "sweep20", "sweep40", "sweep2d10",
)


def build_case(name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        builder = CASES[name]
    except KeyError as error:
        raise SystemExit(f"unknown case {name!r}; choose from {', '.join(CASES)}") from error
    return builder()


def make_runner(bare: bool, ndim: int) -> Callable:
    """Return ``run(flow, mask_or_u8, threads, **params) -> density``.

    ``bare`` calls the binding directly (mask must be uint8), bypassing the
    Python wrapper's conversions; otherwise the public wrapper is used.
    """
    import bioimage_cpp as bic

    if not bare:
        def run(flow, mask, threads, **params):
            return bic.flow.compute_flow_density(
                flow, mask, sigma=None, number_of_threads=threads, **params
            )
        return run

    core = sys.modules["bioimage_cpp._core"]
    fn = getattr(core, f"_compute_flow_density_{ndim}d_float32")

    def run_bare(flow, mask_u8, threads, **params):
        p = {**DEFAULTS, **params}
        return fn(
            flow, mask_u8, p["n_iter"], p["dt"], p["tol"], p["method"],
            p["restrict_to_mask"], threads,
        )
    return run_bare


def add_param_args(parser) -> None:
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--method", choices=("euler", "rk2"), default=None)
    import argparse

    parser.add_argument(
        "--restrict-to-mask", action=argparse.BooleanOptionalAction, default=None
    )


def params_from_args(args) -> dict:
    params = {}
    for name in ("n_iter", "dt", "tol", "method", "restrict_to_mask"):
        value = getattr(args, name, None)
        if value is not None:
            params[name] = value
    return params
