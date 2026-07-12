"""Small, dependency-free validators shared by public Python wrappers."""

from __future__ import annotations

import operator

import numpy as np


def strict_index(value, name: str, *, minimum: int | None = None,
                 maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {value!r}") from error
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {result}")
    return result


def strict_integer_array(values, name: str, *, dtype: np.dtype,
                         ndim: int | None = None,
                         shape: tuple[int | None, ...] | None = None,
                         non_negative: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain integers, got dtype={array.dtype}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got ndim={array.ndim}")
    if shape is not None:
        if array.ndim != len(shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(array.shape, shape)
        ):
            raise ValueError(f"{name} must have shape {shape}, got shape={array.shape}")
    if non_negative and np.issubdtype(array.dtype, np.signedinteger):
        if array.size and np.any(array < 0):
            raise ValueError(f"{name} must contain non-negative integers")

    target = np.dtype(dtype)
    # Fast path: a lossless (safe) cast is representable for every value by
    # definition, so skip the O(n) min()/max() range scan. This keeps hot
    # callers such as `objective.energy(labels)` (uint64->uint64,
    # uint32->uint64) as cheap as a plain `np.asarray`. The range scan only
    # runs for genuinely narrowing or sign-changing casts.
    if array.size and not np.can_cast(array.dtype, target):
        info = np.iinfo(target)
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < info.min or maximum > info.max:
            raise ValueError(
                f"{name} values must fit dtype {target}, got range [{minimum}, {maximum}]"
            )
    return np.ascontiguousarray(array, dtype=target)


def strict_offsets(offsets, ndim: int, name: str = "offsets") -> list[tuple[int, ...]]:
    untyped = np.asarray(offsets)
    if untyped.size == 0:
        raise ValueError(f"{name} must not be empty")
    array = strict_integer_array(offsets, name, dtype=np.int64, ndim=2)
    if array.shape[1] != ndim:
        raise ValueError(
            f"{name}[0] must have length {ndim}; each offset must have "
            f"length {ndim} to match the spatial ndim={ndim}, "
            f"got shape={array.shape}"
        )
    return [tuple(int(v) for v in row) for row in array]
