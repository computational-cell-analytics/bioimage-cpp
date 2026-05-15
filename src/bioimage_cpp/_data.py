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
- ``lifted_multicut_problem_3d.npz`` — full 3D ISBI volume (large, ~18k nodes).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bioimage-cpp"
CACHE_ENV_VAR = "BIOIMAGE_CPP_CACHE"


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
