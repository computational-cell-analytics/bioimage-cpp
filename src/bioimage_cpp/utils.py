"""Utility algorithms for array relabeling and related operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _core

_TAKE_DICT_BY_DTYPE = {
    np.dtype("uint32"): _core._take_dict_uint32,
    np.dtype("uint64"): _core._take_dict_uint64,
    np.dtype("int32"): _core._take_dict_int32,
    np.dtype("int64"): _core._take_dict_int64,
}


def take_dict(relabeling: Mapping[Any, Any], to_relabel: np.ndarray) -> np.ndarray:
    """Map an integer array through a dictionary.

    Parameters
    ----------
    relabeling:
        Mapping from input labels to output labels. All keys present in
        ``to_relabel`` must exist in the mapping.
    to_relabel:
        NumPy array with dtype ``uint32``, ``uint64``, ``int32``, or
        ``int64``. Non-contiguous inputs are copied to a contiguous array
        before calling the C++ kernel.

    Returns
    -------
    np.ndarray
        Array with the same shape and dtype as ``to_relabel``.

    Raises
    ------
    TypeError
        If ``relabeling`` is not a mapping or ``to_relabel`` has an
        unsupported dtype.
    IndexError
        If a value in ``to_relabel`` is missing from ``relabeling``.
    """
    if not isinstance(relabeling, Mapping):
        raise TypeError("relabeling must be a mapping")

    array = np.asarray(to_relabel)
    dtype = array.dtype
    try:
        take = _TAKE_DICT_BY_DTYPE[dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in _TAKE_DICT_BY_DTYPE)
        raise TypeError(
            f"to_relabel must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    contiguous = np.ascontiguousarray(array)
    try:
        return take(dict(relabeling), contiguous)
    except IndexError:
        raise
    except RuntimeError as error:
        # nanobind translates std::out_of_range to IndexError in recent
        # versions, but older builds may surface it as RuntimeError.
        if "missing key" in str(error):
            raise IndexError(str(error)) from error
        raise
