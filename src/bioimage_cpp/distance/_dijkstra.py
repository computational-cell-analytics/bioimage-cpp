"""Discrete shortest paths on masked 2D and 3D grids."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import _core
from .._validation import strict_index
from ._distance import _as_binary_input, _normalize_sampling
from ._geodesic import _as_coordinates, _require_foreground_points


_COST_MODES = {
    "physical": 0,
    "node": 1,
    "node_times_physical": 2,
}


def _normalize_dijkstra_inputs(
    mask,
    connectivity,
    spacing,
    costs,
    cost_mode,
    function: str,
):
    binary = _as_binary_input(mask, function)
    if binary.ndim not in (2, 3):
        raise ValueError(f"{function}: mask must have ndim 2 or 3, got ndim={binary.ndim}")

    if connectivity is None:
        connectivity_value = binary.ndim
    else:
        connectivity_value = strict_index(
            connectivity, "connectivity", minimum=1, maximum=binary.ndim
        )

    try:
        mode_value = _COST_MODES[cost_mode]
    except (KeyError, TypeError) as error:
        supported = ", ".join(repr(mode) for mode in _COST_MODES)
        raise ValueError(
            f"{function}: cost_mode must be one of ({supported}), got {cost_mode!r}"
        ) from error

    if cost_mode == "node" and spacing is not None:
        raise ValueError(f"{function}: spacing must be None for cost_mode='node'")
    spacing_values = (
        []
        if cost_mode == "node"
        else _normalize_sampling(spacing, binary.ndim, function, name="spacing")
    )

    costs_array = None
    if cost_mode == "physical":
        if costs is not None:
            raise ValueError(f"{function}: costs must be None for cost_mode='physical'")
    else:
        if costs is None:
            raise ValueError(f"{function}: costs are required for cost_mode={cost_mode!r}")
        try:
            costs_array = np.ascontiguousarray(costs, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{function}: costs must be a real numeric array") from error
        if costs_array.shape != binary.shape:
            raise ValueError(
                f"{function}: costs must have shape {binary.shape}, got shape={costs_array.shape}"
            )

    return binary, connectivity_value, spacing_values, costs_array, mode_value


def dijkstra_distance_field(
    mask: np.ndarray,
    sources: np.ndarray,
    *,
    connectivity: int | None = None,
    spacing: float | Sequence[float] | None = None,
    costs: np.ndarray | None = None,
    cost_mode: str = "physical",
    return_predecessors: bool = False,
):
    """Compute a multi-source Dijkstra distance field on a masked grid.

    The mask may be two- or three-dimensional. Nonzero voxels are traversable;
    background and unreachable voxels receive distance ``+inf``.

    ``cost_mode`` defines the directed edge cost from a voxel ``u`` to its
    neighbour ``v``:

    - ``"physical"``: the Euclidean neighbour-step length under ``spacing``;
      ``costs`` must be omitted.
    - ``"node"``: ``costs[v]``; ``spacing`` must be omitted.
    - ``"node_times_physical"``: ``costs[v]`` times the physical step length.

    Parameters
    ----------
    mask:
        Binary 2D or 3D array; nonzero values define the traversable domain.
    sources:
        Integer coordinates with shape ``(n_sources, mask.ndim)``. A flat
        ``(mask.ndim,)`` coordinate is accepted for one source.
    connectivity:
        ``1`` to ``mask.ndim``. ``None`` selects full 8-/26-connectivity.
    spacing:
        Positive scalar or one value per axis. Used by the physical modes.
    costs:
        Finite non-negative array matching ``mask.shape`` for node-cost modes.
    cost_mode:
        ``"physical"``, ``"node"``, or ``"node_times_physical"``.
    return_predecessors:
        Also return a flat-index predecessor field when true.

    Returns
    -------
    distances or (distances, predecessors):
        Distances are ``float64`` with the mask shape. Predecessors are
        ``int64`` with the same shape: sources contain their own flat C-order
        index, reachable voxels contain their predecessor's flat index, and
        background/unreachable voxels contain ``-1``.
    """
    function = "dijkstra_distance_field"
    binary, connectivity_value, spacing_values, costs_array, mode_value = (
        _normalize_dijkstra_inputs(
            mask, connectivity, spacing, costs, cost_mode, function
        )
    )
    sources_array = _as_coordinates(sources, binary.ndim, function, "sources")
    if sources_array.shape[0] == 0:
        raise ValueError(f"{function}: sources must contain at least one coordinate")
    _require_foreground_points(binary, sources_array, function, "sources")

    distances, predecessors = _core._dijkstra_distance_field_mask(
        binary,
        sources_array,
        connectivity_value,
        spacing_values,
        costs_array,
        mode_value,
        bool(return_predecessors),
    )
    if return_predecessors:
        return distances, predecessors
    return distances


def dijkstra_path(
    mask: np.ndarray,
    source: np.ndarray,
    targets: np.ndarray,
    *,
    connectivity: int | None = None,
    spacing: float | Sequence[float] | None = None,
    costs: np.ndarray | None = None,
    cost_mode: str = "physical",
) -> np.ndarray:
    """Return the cheapest path from one source to any supplied target.

    The returned ``int64`` coordinates have shape ``(n_path, mask.ndim)`` and
    are ordered from the source to the reached target. The solve stops when
    the cheapest target is settled. Equal-cost targets are resolved by their
    flat C-order index.
    """
    function = "dijkstra_path"
    binary, connectivity_value, spacing_values, costs_array, mode_value = (
        _normalize_dijkstra_inputs(
            mask, connectivity, spacing, costs, cost_mode, function
        )
    )
    source_array = _as_coordinates(source, binary.ndim, function, "source")
    if source_array.shape[0] != 1:
        raise ValueError(
            f"{function}: source must contain exactly one coordinate, "
            f"got {source_array.shape[0]}"
        )
    targets_array = _as_coordinates(targets, binary.ndim, function, "targets")
    if targets_array.shape[0] == 0:
        raise ValueError(f"{function}: targets must contain at least one coordinate")
    _require_foreground_points(binary, source_array, function, "source")
    _require_foreground_points(binary, targets_array, function, "targets")

    return _core._dijkstra_path_mask(
        binary,
        source_array,
        targets_array,
        connectivity_value,
        spacing_values,
        costs_array,
        mode_value,
    )
