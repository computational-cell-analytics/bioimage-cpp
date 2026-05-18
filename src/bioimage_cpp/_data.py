"""Pooch-based registry for downloadable problem instances.

Caches files under ``~/.cache/bioimage-cpp/`` by default, overridable with
``BIOIMAGE_CPP_CACHE``. Pooch is imported lazily so the runtime does not gain
a hard dependency on it; ``fetch`` raises a clear error if pooch is missing.

Registered files
----------------

Multicut problems (text files with rows ``u v cost``; originate from
``elf.segmentation.utils.load_multicut_problem``):

- ``multicut_problem_A_small.txt`` ... ``multicut_problem_C_medium.txt``
  (3 samples × 2 sizes = 6 problems).

Lifted multicut problems (``.npz`` files written by
``examples/segmentation/serialize_lifted_problem.py``):

- ``lifted_multicut_problem_2d.npz`` — 2D ISBI slice (small, ~756 nodes).
- ``lifted_multicut_problem_3d.npz`` — full 3D ISBI volume (medium, ~18k nodes).
- ``lifted_multicut_problem_grid.npz`` — lifted multicut problem from grid graph (large, ~260k nodes).

Affinities:
- ``affinities`` — HDF5 file with sample affinities from the ISBI volume.
  Contains affinities under key ``affinities``.

Semantic labels:
- ``semantic_labels`` — HDF5 file with paired instance and semantic ground
  truth on a single 3D volume. Contains ``labels/instances``,
  ``labels/semantic`` and ``raw``. The on-disk arrays are padded with ``-1``
  outside the labelled region; loaders crop to the labelled content slab.
  Used by the semantic-mutex-watershed comparison scripts under
  ``development/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bioimage-cpp"
CACHE_ENV_VAR = "BIOIMAGE_CPP_CACHE"
ISBI_AFFINITY_FILENAME = "affinities"
SEMANTIC_LABELS_FILENAME = "semantic_labels"
ISBI_AFFINITY_OFFSETS = (
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
    (-1, -1, -1),
    (-1, 1, 1),
    (-1, -1, 1),
    (-1, 1, -1),
    (0, -9, 0),
    (0, 0, -9),
    (0, -9, -9),
    (0, 9, -9),
    (0, -9, -4),
    (0, -4, -9),
    (0, 4, -9),
    (0, 9, -4),
    (0, -27, 0),
    (0, 0, -27),
)


# Each entry is filename -> (url, sha256). To refresh a hash, delete the
# corresponding file under :func:`cache_dir`, re-download it, and run
# ``sha256sum`` on the cached file.
_REGISTRY: dict[str, tuple[str, Optional[str]]] = {
    "multicut_problem_A_small.txt": (
        "https://oc.embl.de/index.php/s/yVKwyQ8VoPXYkft/download",
        "eeb1083557a20f7ce1ece28f5c613cc8ce5bf6231cd74aadbeb8a5012c6f8ef0",
    ),
    "multicut_problem_A_medium.txt": (
        "https://oc.embl.de/index.php/s/ztnwjmv0bmd3mnS/download",
        "a8cdd23fcd911ad62b1b859b242bac28d16e7cdc3920137116b05672c4a6ec8a",
    ),
    "multicut_problem_B_small.txt": (
        "https://oc.embl.de/index.php/s/QKYA2EoMXqxQuO4/download",
        "abd2c040234f20b107cc237b2c87120058d78e2c5e3ba2b95bc12b3b4d433aa5",
    ),
    "multicut_problem_B_medium.txt": (
        "https://oc.embl.de/index.php/s/yuk7VwCvgZC017q/download",
        "6a8406c774553753e49103531945c32170587cc0d20d0459c866b47de5b014ec",
    ),
    "multicut_problem_C_small.txt": (
        "https://oc.embl.de/index.php/s/eDZprDwT2cXFAe0/download",
        "6db8336c0ba3f75e3f9432628ac13b156fb9e43f75307cdda11469927ed1a108",
    ),
    "multicut_problem_C_medium.txt": (
        "https://oc.embl.de/index.php/s/hGyqlkenHfsq5P4/download",
        "130d1be14d69f8bfb5d20d1375452291db7ba620e2f03bf9ffbe52d1f577f0dc",
    ),
    "lifted_multicut_problem_2d.npz": (
        "https://owncloud.gwdg.de/index.php/s/QikYgJzbVxD5q8q/download",
        "27f10d9b7b2405cf64fab49c9065291455f2f1364224bb94a255c4cc72798240",
    ),
    "lifted_multicut_problem_3d.npz": (
        "https://owncloud.gwdg.de/index.php/s/ZVzDy8Xb0Dr2Ell/download",
        "269ce644e2b9f8259f7f2ff827d5808ac5c9bfe6ca0444e298290f23867dce8a",
    ),
    "lifted_multicut_problem_grid.npz": (
        "https://owncloud.gwdg.de/index.php/s/YWNZSYsBd1VwSX1/download",
        "20583b2000838ed0942f8f1c343b84287d8bf218d19d77a8b5627924661c5aa3",
    ),
    "affinities": (
        "https://owncloud.gwdg.de/index.php/s/aAyF2ekzsW7DFJo/download",
        "6472ad0fcf3c57a4ae345fda68c3cbb6072ee3e8db67b423502746b46d8cd5e5",
    ),
    "semantic_labels": (
        "https://owncloud.gwdg.de/index.php/s/Ah7IGuYH7uuomQV/download",
        "6232fe2fd58fdbd3def978798143fdcc65a2af118b4d9ee177b5c942173ece26",
    ),
}


def cache_dir() -> Path:
    """Return the cache directory used for downloaded problem instances."""
    override = os.environ.get(CACHE_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_CACHE_DIR


def registered_files() -> list[str]:
    """List every filename available via :func:`fetch`."""
    return sorted(_REGISTRY.keys())


def fetch(filename: str, *, timeout: Optional[float] = None) -> Path:
    """Return the local path to a registered file, downloading on first call.

    Parameters
    ----------
    filename:
        Filename as listed by :func:`registered_files`.
    timeout:
        Optional HTTP timeout in seconds, forwarded to pooch's downloader.

    Raises
    ------
    FileNotFoundError
        If ``filename`` is not registered.
    ModuleNotFoundError
        If ``pooch`` is not installed. Install with ``pip install pooch``.
    RuntimeError
        If the download fails. The underlying ``HTTPError`` is chained.
    """
    if filename not in _REGISTRY:
        registered = registered_files()
        raise FileNotFoundError(
            f"{filename!r} is not in the bioimage-cpp data registry. "
            f"Available: {registered}"
        )
    try:
        import pooch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "pooch is required to download bioimage-cpp problem instances. "
            "Install it with `pip install pooch`."
        ) from error

    url, sha256 = _REGISTRY[filename]
    fetcher = pooch.create(
        path=cache_dir(),
        base_url="",
        registry={filename: sha256},
        urls={filename: url},
    )
    downloader = None
    if timeout is not None:
        downloader = pooch.HTTPDownloader(timeout=float(timeout))
    try:
        local_path = fetcher.fetch(filename, downloader=downloader)
    except Exception as error:
        raise RuntimeError(
            f"could not download {filename} from {url}: {error}"
        ) from error
    return Path(local_path)


def affinity_path(*, timeout: Optional[float] = None) -> Path:
    """Return the cached path to the registered ISBI affinity HDF5 file."""
    return fetch(ISBI_AFFINITY_FILENAME, timeout=timeout)


def load_isbi_affinities(
    *,
    timeout: Optional[float] = None,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Load the registered ISBI affinity volume and its offsets.

    The offsets are the fixed channel offsets used by
    ``elf.segmentation.utils.load_mutex_watershed_problem`` for this data.
    """
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "h5py is required to load the registered ISBI affinity file. "
            "Install it with `pip install h5py`."
        ) from error

    with h5py.File(affinity_path(timeout=timeout), "r") as f:
        affinities = f["affinities"][:]
    return np.ascontiguousarray(affinities), list(ISBI_AFFINITY_OFFSETS)


def load_isbi_raw(
    *,
    timeout: Optional[float] = None,
) -> np.ndarray:
    """Load the registered ISBI raw volume.

    The raw data is stored in the same HDF5 file as the affinities under key
    ``raw``.
    """
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "h5py is required to load the registered ISBI raw file. "
            "Install it with `pip install h5py`."
        ) from error

    with h5py.File(affinity_path(timeout=timeout), "r") as f:
        raw = f["raw"][:]
    return np.ascontiguousarray(raw)


def load_isbi_gt_segmentation(
    *,
    timeout: Optional[float] = None,
) -> np.ndarray:
    """Load the registered ISBI ground-truth segmentation volume.

    The labels are stored in the same HDF5 file as the affinities under key
    ``labels/gt_segmentation``. Returned as a C-contiguous ``uint64`` array
    with shape ``(30, 512, 512)`` (~7.9 M voxels).
    """
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "h5py is required to load the registered ISBI segmentation file. "
            "Install it with `pip install h5py`."
        ) from error

    with h5py.File(affinity_path(timeout=timeout), "r") as f:
        labels = f["labels/gt_segmentation"][:]
    return np.ascontiguousarray(labels)


def semantic_labels_path(*, timeout: Optional[float] = None) -> Path:
    """Return the cached path to the registered semantic-labels HDF5 file."""
    return fetch(SEMANTIC_LABELS_FILENAME, timeout=timeout)


# Bounding box of the labelled content in the on-disk volume. Outside this
# slab both label volumes are uniformly ``-1``; cropping here keeps callers
# from having to special-case the padding.
SEMANTIC_LABELS_CROP = (slice(16, 32), slice(64, 512), slice(64, 512))


def load_semantic_labels(
    *,
    timeout: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the registered (instance, semantic) ground-truth volumes.

    Both volumes share the same spatial shape and live in a single HDF5 file
    under keys ``labels/instances`` and ``labels/semantic``. The on-disk
    arrays are padded with ``-1`` outside the labelled slab; this loader
    returns the cropped labelled region (``(16, 448, 448)``).

    Returns
    -------
    instance_labels:
        Integer instance-segmentation volume.
    semantic_labels:
        Integer semantic-class volume (one class id per voxel).
    """
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "h5py is required to load the registered semantic-labels file. "
            "Install it with `pip install h5py`."
        ) from error

    with h5py.File(semantic_labels_path(timeout=timeout), "r") as f:
        instance = f["labels/instances"][SEMANTIC_LABELS_CROP]
        semantic = f["labels/semantic"][SEMANTIC_LABELS_CROP]
    return np.ascontiguousarray(instance), np.ascontiguousarray(semantic)


def load_semantic_raw(
    *,
    timeout: Optional[float] = None,
) -> np.ndarray:
    """Load the raw volume paired with the registered semantic labels.

    Stored alongside ``labels/instances`` / ``labels/semantic`` in the same
    HDF5 file under key ``raw``. Cropped consistently with
    :func:`load_semantic_labels`.
    """
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "h5py is required to load the registered semantic-raw file. "
            "Install it with `pip install h5py`."
        ) from error

    with h5py.File(semantic_labels_path(timeout=timeout), "r") as f:
        raw = f["raw"][SEMANTIC_LABELS_CROP]
    return np.ascontiguousarray(raw)
