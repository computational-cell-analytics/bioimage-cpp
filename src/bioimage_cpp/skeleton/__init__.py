"""Skeletonization algorithms for binary and labeled image volumes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core
from .._validation import strict_index
from ..distance._distance import _as_binary_input, _normalize_sampling, _normalize_threads
from ._graph import skeleton_to_graph
from .postprocessing import clean_graph, draw_instances, join_close_components, remove_ticks


_TEASAR_LABELS_BY_DTYPE = {
    np.dtype("uint8"): _core._teasar_labels_uint8,
    np.dtype("uint16"): _core._teasar_labels_uint16,
    np.dtype("uint32"): _core._teasar_labels_uint32,
    np.dtype("uint64"): _core._teasar_labels_uint64,
    np.dtype("int32"): _core._teasar_labels_int32,
    np.dtype("int64"): _core._teasar_labels_int64,
}


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


def _normalize_teasar_options(
    function: str,
    spacing: float | Sequence[float] | None,
    scale: float,
    constant: float,
    pdrf_scale: float,
    pdrf_exponent: float,
    number_of_threads: int,
) -> tuple[list[float], float, float, float, float, int]:
    spacing_values = _normalize_sampling(spacing, 3, function, name="spacing")
    scale_value = _finite_parameter(scale, "scale", positive=False)
    constant_value = _finite_parameter(constant, "constant", positive=False)
    pdrf_scale_value = _finite_parameter(pdrf_scale, "pdrf_scale", positive=False)
    pdrf_exponent_value = _finite_parameter(
        pdrf_exponent, "pdrf_exponent", positive=True
    )
    n_threads = _normalize_threads(number_of_threads, function)
    return (
        spacing_values,
        scale_value,
        constant_value,
        pdrf_scale_value,
        pdrf_exponent_value,
        n_threads,
    )


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
    """Skeletonize a binary volume with 3D TEASAR.

    This is a correctness-first implementation of the core TEASAR procedure:
    distance-to-boundary and distance-from-root fields guide repeated penalized
    Dijkstra paths, and a rolling physical invalidation cube determines when
    the object has been covered. Production kimimaro heuristics such as soma
    handling, border stitching, hole filling, and manual targets are not part
    of this function.

    Parameters
    ----------
    mask:
        Three-dimensional binary input. All nonzero values are one foreground
        class. Each 26-connected component is skeletonized independently.
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
        With several components the tuple represents their skeleton forest in
        first-voxel C-order. An empty mask returns correctly typed empty arrays.
        Use :func:`skeleton_to_graph` to convert the vertex and edge arrays to
        an undirected graph; graph node ids continue to index ``vertices`` and
        ``radii``.

    Raises
    ------
    ValueError
        If the input is not 3D or parameters are invalid.
    """
    function = "teasar"
    binary = _as_binary_input(mask, function)
    if binary.ndim != 3:
        raise ValueError(f"{function}: mask must have ndim 3, got ndim={binary.ndim}")
    options = _normalize_teasar_options(
        function, spacing, scale, constant, pdrf_scale, pdrf_exponent,
        number_of_threads
    )
    return _core._teasar_uint8(
        binary,
        *options,
    )


def teasar_labels(
    labels: np.ndarray,
    *,
    background: int = 0,
    spacing: float | Sequence[float] | None = (1.0, 1.0, 1.0),
    scale: float = 1.5,
    constant: float = 0.0,
    pdrf_scale: float = 100000.0,
    pdrf_exponent: float = 4.0,
    number_of_threads: int = 1,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Skeletonize every semantic label in a 3D integer volume.

    Every non-background value is preserved as a result key. Connected
    components within one label are skeletonized independently and combined
    into one deterministic forest. Different labels remain separate even when
    their voxels touch.

    Parameters
    ----------
    labels:
        Three-dimensional native-endian integer array with dtype ``uint8``,
        ``uint16``, ``uint32``, ``uint64``, ``int32``, or ``int64``.
        Non-contiguous inputs are copied once.
    background:
        Integer value excluded from skeletonization. It must fit the input
        dtype. Defaults to ``0``.
    spacing, scale, constant, pdrf_scale, pdrf_exponent, number_of_threads:
        The same TEASAR parameters and shared thread budget as :func:`teasar`.

    Returns
    -------
    dict
        Maps original label values, inserted in ascending numeric order, to
        ``(vertices, edges, radii)`` forest tuples.
    """
    function = "teasar_labels"
    labels_array = np.asarray(labels)
    if labels_array.ndim != 3:
        raise ValueError(
            f"{function}: labels must have ndim 3, got ndim={labels_array.ndim}"
        )
    try:
        run = _TEASAR_LABELS_BY_DTYPE[labels_array.dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in _TEASAR_LABELS_BY_DTYPE)
        if labels_array.dtype == np.dtype("bool"):
            detail = "; use teasar for binary input"
        else:
            detail = ""
        raise TypeError(
            f"{function}: labels must have one of native-endian dtypes "
            f"({supported}), got dtype={labels_array.dtype}{detail}"
        ) from error

    background_value = strict_index(background, "background")
    info = np.iinfo(labels_array.dtype)
    if background_value < info.min or background_value > info.max:
        raise ValueError(
            f"background={background_value} is outside the range of dtype "
            f"{labels_array.dtype}"
        )
    labels_c = np.ascontiguousarray(labels_array)
    options = _normalize_teasar_options(
        function, spacing, scale, constant, pdrf_scale, pdrf_exponent,
        number_of_threads
    )
    return run(labels_c, labels_array.dtype.type(background_value), *options)


__all__ = ["clean_graph", "draw_instances", "join_close_components", "remove_ticks", "skeleton_to_graph", "teasar", "teasar_labels"]
from . import distributed


__all__ = ["distributed", "skeleton_to_graph", "teasar", "teasar_labels"]
