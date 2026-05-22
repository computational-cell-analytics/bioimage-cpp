"""Relabel an integer array to consecutive labels."""

from __future__ import annotations

import numpy as np

from .. import _core

_RELABEL_SEQUENTIAL_BY_DTYPE = {
    np.dtype("uint32"): _core._relabel_sequential_uint32,
    np.dtype("uint64"): _core._relabel_sequential_uint64,
    np.dtype("int32"): _core._relabel_sequential_int32,
    np.dtype("int64"): _core._relabel_sequential_int64,
}


def relabel_sequential(
    label_field: np.ndarray,
    offset: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relabel an integer label array so non-zero labels are consecutive.

    Mirrors :func:`skimage.segmentation.relabel_sequential`. Label ``0`` is
    treated as background and always maps to ``0``. All other distinct labels
    are mapped to consecutive integers starting at ``offset``, in sorted
    order of the original label values.

    Parameters
    ----------
    label_field:
        Integer NumPy array with dtype ``uint32``, ``uint64``, ``int32``, or
        ``int64``. Non-contiguous inputs are copied before entering C++.
        Signed dtypes must not contain negative values.
    offset:
        The first new label assigned to non-zero values. Must be a positive
        integer.

    Returns
    -------
    relabeled:
        Array with the same shape and dtype as ``label_field``.
    forward_map:
        1-D array indexed by old label, returning the new label.
        Length ``max(label_field) + 1`` (or ``0`` for empty input).
        Entries for labels that do not appear in ``label_field`` are ``0``.
    inverse_map:
        1-D array indexed by new label, returning the old label.
        Length ``offset + N`` where ``N`` is the number of distinct non-zero
        labels in ``label_field``. Entries below ``offset`` (other than
        ``inverse_map[0]``) are ``0``.

    Raises
    ------
    TypeError
        If ``label_field`` has an unsupported dtype.
    ValueError
        If ``offset`` is not a positive integer, or ``label_field`` contains
        negative values.
    """
    if not isinstance(offset, (int, np.integer)) or isinstance(offset, bool):
        raise ValueError(f"offset must be a positive integer, got {offset!r}")
    offset_int = int(offset)
    if offset_int < 1:
        raise ValueError(f"offset must be >= 1, got {offset_int}")

    array = np.asarray(label_field)
    dtype = array.dtype
    try:
        relabel = _RELABEL_SEQUENTIAL_BY_DTYPE[dtype]
    except KeyError as error:
        supported = ", ".join(str(d) for d in _RELABEL_SEQUENTIAL_BY_DTYPE)
        raise TypeError(
            f"label_field must have one of dtypes ({supported}), got dtype={dtype}"
        ) from error

    if np.issubdtype(dtype, np.signedinteger) and array.size > 0 and array.min() < 0:
        raise ValueError(
            f"label_field must not contain negative values, got min={int(array.min())}"
        )

    offset_value = dtype.type(offset_int)
    if int(offset_value) != offset_int:
        raise ValueError(
            f"offset={offset_int} is not representable in dtype {dtype}"
        )

    contiguous = np.ascontiguousarray(array)
    relabeled, forward_map, inverse_map = relabel(contiguous, offset_value)
    return relabeled, forward_map, inverse_map
