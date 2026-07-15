"""Skeletonization algorithms for binary image volumes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core
from ..distance._distance import _as_binary_input, _normalize_sampling, _normalize_threads


def _finite_parameter(value, name: str, *, positive: bool) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number, got {value!r}") from error
    valid = np.isfinite(result) and (result > 0.0 if positive else result >= 0.0)
    if not valid:
        requirement = "positive and finite" if positive else "finite and non-negative"
        raise ValueError(f"{name} must be {requirement}, got {result}")
    return result


def teasar(
    mask: np.ndarray,
    *,
    spacing: float | Sequence[float] | None = (1.0, 1.0, 1.0),
    scale: float = 1.5,
    constant: float = 0.0,
    pdrf_scale: float = 100000.0,
    pdrf_exponent: float = 4.0,
    number_of_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skeletonize one 26-connected binary object with 3D TEASAR.

    This is a correctness-first implementation of the core TEASAR procedure:
    distance-to-boundary and distance-from-root fields guide repeated penalized
    Dijkstra paths, and a rolling physical invalidation cube determines when
    the object has been covered. Production kimimaro heuristics such as soma
    handling, border stitching, hole filling, and manual targets are not part
    of this function.

    Parameters
    ----------
    mask:
        Three-dimensional binary input. Nonzero values are foreground. A
        nonempty mask must contain exactly one 26-connected component.
    spacing:
        Positive scalar or three physical voxel spacings in NumPy ``(z, y, x)``
        order.
    scale, constant:
        The invalidation radius at a path vertex is
        ``scale * distance_to_boundary + constant`` in physical units.
    pdrf_scale, pdrf_exponent:
        Scale and exponent of the boundary-avoidance term in the penalized
        distance-from-root field.
    number_of_threads:
        Thread budget for the exact distance transform. Compact Dijkstra root
        and rail solves remain sequential because their wavefronts benchmarked
        faster on the optimized heap. ``0`` uses hardware concurrency; default
        ``1`` is sequential.

    Returns
    -------
    vertices, edges, radii:
        ``vertices`` is ``float64`` with shape ``(V, 3)`` and physical
        ``(z, y, x)`` coordinates. ``edges`` is ``uint64`` with shape
        ``(E, 2)`` and indexes the vertices. ``radii`` is ``float32`` with
        shape ``(V,)`` and contains physical distance-to-boundary values.
        An empty mask returns correctly typed empty arrays.

    Raises
    ------
    ValueError
        If the input is not 3D, parameters are invalid, or foreground has more
        than one 26-connected component.
    """
    function = "teasar"
    binary = _as_binary_input(mask, function)
    if binary.ndim != 3:
        raise ValueError(f"{function}: mask must have ndim 3, got ndim={binary.ndim}")
    spacing_values = _normalize_sampling(spacing, 3, function, name="spacing")
    scale_value = _finite_parameter(scale, "scale", positive=False)
    constant_value = _finite_parameter(constant, "constant", positive=False)
    pdrf_scale_value = _finite_parameter(pdrf_scale, "pdrf_scale", positive=False)
    pdrf_exponent_value = _finite_parameter(
        pdrf_exponent, "pdrf_exponent", positive=True
    )
    n_threads = _normalize_threads(number_of_threads, function)
    return _core._teasar_uint8(
        binary,
        spacing_values,
        scale_value,
        constant_value,
        pdrf_scale_value,
        pdrf_exponent_value,
        n_threads,
    )


__all__ = ["teasar"]
