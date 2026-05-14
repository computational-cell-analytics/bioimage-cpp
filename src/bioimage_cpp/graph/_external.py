"""External graph problems used for development and regression checks."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from urllib.error import URLError

import numpy as np


EXTERNAL_MULTICUT_PROBLEM_URL = "https://oc.embl.de/index.php/s/yVKwyQ8VoPXYkft/download"
DEFAULT_EXTERNAL_MULTICUT_PROBLEM_PATH = Path("/tmp/bioimage_cpp_external_multicut_problem.txt")


def external_multicut_problem_path(
    path: str | os.PathLike | None = None,
    *,
    download: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Return the local path for the external multicut problem.

    The path can be supplied explicitly, via
    ``BIOIMAGE_CPP_EXTERNAL_MULTICUT_PATH``, or via the cache path
    ``BIOIMAGE_CPP_EXTERNAL_MULTICUT_CACHE``. If no existing file is found and
    ``download`` is true, the problem is downloaded into the cache path.
    """
    explicit_path = path or os.environ.get("BIOIMAGE_CPP_EXTERNAL_MULTICUT_PATH")
    if explicit_path is not None:
        resolved = Path(explicit_path)
        if not resolved.exists():
            raise FileNotFoundError(f"external multicut problem does not exist: {resolved}")
        return resolved

    cache_path = Path(
        os.environ.get(
            "BIOIMAGE_CPP_EXTERNAL_MULTICUT_CACHE",
            str(DEFAULT_EXTERNAL_MULTICUT_PROBLEM_PATH),
        )
    )
    if cache_path.exists():
        return cache_path
    if not download:
        raise FileNotFoundError(f"external multicut problem does not exist: {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(EXTERNAL_MULTICUT_PROBLEM_URL, timeout=timeout) as response:
            cache_path.write_bytes(response.read())
    except URLError as error:
        raise RuntimeError(f"could not download external multicut problem: {error}") from error
    return cache_path


def load_external_multicut_problem_data(
    path: str | os.PathLike | None = None,
    *,
    download: bool = True,
    timeout: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load external multicut problem edge ids and costs."""
    problem_path = external_multicut_problem_path(
        path,
        download=download,
        timeout=timeout,
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
    """Load the external multicut problem as a bioimage-cpp graph and costs."""
    from . import UndirectedGraph

    uv_ids, costs = load_external_multicut_problem_data(
        path,
        download=download,
        timeout=timeout,
    )
    graph = UndirectedGraph.from_edges(int(uv_ids.max()) + 1, uv_ids)
    return graph, costs
