"""External graph problems used for development and regression checks.

Downloads go through the shared pooch registry in :mod:`bioimage_cpp._data`
(cached under ``~/.cache/bioimage-cpp/``). The module exposes:

- :func:`load_multicut_problem` / :func:`load_multicut_problem_data` /
  :func:`multicut_problem_path` for the 6 multicut problems (3 samples × 2
  sizes from ``elf.segmentation.utils.load_multicut_problem``).
- :func:`load_lifted_multicut_problem` /
  :func:`lifted_multicut_problem_path` for the 2D and 3D lifted multicut
  problems built by ``examples/segmentation/serialize_lifted_problem.py``.

A legacy compatibility layer (``load_external_multicut_problem`` and friends)
delegates to sample A, size small, which is the problem the regression test
uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from bioimage_cpp._data import cache_dir, fetch


VALID_SAMPLES = ("A", "B", "C")
VALID_SIZES = ("small", "medium")
VALID_LIFTED_SIZES = ("2d", "3d")


# Legacy constants. The URL points at the canonical sample-A-small download
# used since the first public version. The default path now resolves to the
# new shared cache location instead of /tmp.
EXTERNAL_MULTICUT_PROBLEM_URL = "https://oc.embl.de/index.php/s/yVKwyQ8VoPXYkft/download"
DEFAULT_EXTERNAL_MULTICUT_PROBLEM_PATH = cache_dir() / "multicut_problem_A_small.txt"


@dataclass
class LiftedMulticutProblem:
    """Raw arrays describing a lifted multicut problem."""

    n_nodes: int
    local_uvs: np.ndarray
    local_costs: np.ndarray
    lifted_uvs: np.ndarray
    lifted_costs: np.ndarray


# Multicut helpers --------------------------------------------------------


def _validate_sample_size(sample: str, size: str) -> None:
    if sample not in VALID_SAMPLES:
        raise ValueError(
            f"sample must be one of {VALID_SAMPLES}, got {sample!r}"
        )
    if size not in VALID_SIZES:
        raise ValueError(f"size must be one of {VALID_SIZES}, got {size!r}")


def multicut_problem_path(
    sample: str = "A",
    size: str = "small",
    *,
    timeout: Optional[float] = None,
) -> Path:
    """Return the cached path to a multicut problem, downloading if needed."""
    _validate_sample_size(sample, size)
    return fetch(f"multicut_problem_{sample}_{size}.txt", timeout=timeout)


def load_multicut_problem_data(
    sample: str = "A",
    size: str = "small",
    *,
    timeout: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(uv_ids, costs)`` for one of the multicut problems."""
    path = multicut_problem_path(sample, size, timeout=timeout)
    problem = np.genfromtxt(path)
    uv_ids = np.ascontiguousarray(problem[:, :2].astype(np.uint64, copy=False))
    costs = np.ascontiguousarray(problem[:, -1].astype(np.float64, copy=False))
    return uv_ids, costs


def load_multicut_problem(
    sample: str = "A",
    size: str = "small",
    *,
    timeout: Optional[float] = None,
):
    """Load a multicut problem as a bioimage-cpp graph and edge-cost vector."""
    from . import UndirectedGraph

    uv_ids, costs = load_multicut_problem_data(sample, size, timeout=timeout)
    graph = UndirectedGraph.from_edges(int(uv_ids.max()) + 1, uv_ids)
    return graph, costs


# Lifted multicut helpers ------------------------------------------------


def _validate_lifted_size(size: str) -> None:
    if size not in VALID_LIFTED_SIZES:
        raise ValueError(
            f"size must be one of {VALID_LIFTED_SIZES}, got {size!r}"
        )


def lifted_multicut_problem_path(
    size: str = "2d",
    *,
    timeout: Optional[float] = None,
) -> Path:
    """Return the cached path to the lifted multicut problem file."""
    _validate_lifted_size(size)
    return fetch(f"lifted_multicut_problem_{size}.npz", timeout=timeout)


def load_lifted_multicut_problem(
    size: str = "2d",
    *,
    timeout: Optional[float] = None,
) -> LiftedMulticutProblem:
    """Load a lifted multicut problem as :class:`LiftedMulticutProblem`.

    Parameters
    ----------
    size:
        ``"2d"`` for the small ISBI 2D slice (~756 nodes, fast, used by the
        regression test) or ``"3d"`` for the full ISBI volume (~18k nodes,
        used by the development comparison scripts).
    timeout:
        Optional HTTP timeout in seconds for the download.
    """
    path = lifted_multicut_problem_path(size, timeout=timeout)
    data = np.load(path)
    return LiftedMulticutProblem(
        n_nodes=int(data["n_nodes"]),
        local_uvs=np.ascontiguousarray(
            data["local_uvs"].astype(np.uint64, copy=False)
        ),
        local_costs=np.ascontiguousarray(
            data["local_costs"].astype(np.float64, copy=False)
        ),
        lifted_uvs=np.ascontiguousarray(
            data["lifted_uvs"].astype(np.uint64, copy=False)
        ),
        lifted_costs=np.ascontiguousarray(
            data["lifted_costs"].astype(np.float64, copy=False)
        ),
    )


# Legacy shims (defaults to sample A small) ------------------------------


def external_multicut_problem_path(
    path: str | os.PathLike | None = None,
    *,
    download: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Return the local path for the default multicut problem (A, small).

    Honors ``BIOIMAGE_CPP_EXTERNAL_MULTICUT_PATH`` (explicit existing file)
    and ``BIOIMAGE_CPP_EXTERNAL_MULTICUT_CACHE`` (explicit cache path) for
    backwards compatibility; otherwise routes through the shared pooch
    cache via :func:`multicut_problem_path`.
    """
    explicit_path = path or os.environ.get("BIOIMAGE_CPP_EXTERNAL_MULTICUT_PATH")
    if explicit_path is not None:
        resolved = Path(explicit_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"external multicut problem does not exist: {resolved}"
            )
        return resolved

    legacy_cache = os.environ.get("BIOIMAGE_CPP_EXTERNAL_MULTICUT_CACHE")
    if legacy_cache is not None:
        legacy_path = Path(legacy_cache)
        if legacy_path.exists():
            return legacy_path
        if not download:
            raise FileNotFoundError(
                f"external multicut problem does not exist: {legacy_path}"
            )

    if not download:
        candidate = DEFAULT_EXTERNAL_MULTICUT_PROBLEM_PATH
        if not candidate.exists():
            raise FileNotFoundError(
                f"external multicut problem does not exist: {candidate}"
            )
        return candidate

    return multicut_problem_path(timeout=timeout)


def load_external_multicut_problem_data(
    path: str | os.PathLike | None = None,
    *,
    download: bool = True,
    timeout: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load default multicut problem edge ids and costs (sample A, small)."""
    problem_path = external_multicut_problem_path(
        path, download=download, timeout=timeout
    )
    problem = np.genfromtxt(problem_path)
    uv_ids = np.ascontiguousarray(problem[:, :2].astype(np.uint64, copy=False))
    costs = np.ascontiguousarray(problem[:, -1].astype(np.float64, copy=False))
    return uv_ids, costs


def load_external_multicut_problem(
    path: str | os.PathLike | None = None,
    *,
    download: bool = True,
    timeout: float = 30.0,
):
    """Load the default multicut problem as a graph + cost vector."""
    from . import UndirectedGraph

    uv_ids, costs = load_external_multicut_problem_data(
        path, download=download, timeout=timeout
    )
    graph = UndirectedGraph.from_edges(int(uv_ids.max()) + 1, uv_ids)
    return graph, costs
