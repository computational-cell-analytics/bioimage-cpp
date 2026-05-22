"""Shared internal helpers for the ``bioimage_cpp.graph`` submodules.

This module is private (every name is ``_``-prefixed). It holds the validators,
shape helpers, and small graph utilities that are reused by more than one of
``graph.multicut``, ``graph.lifted_multicut``, ``graph.mutex_watershed``,
``graph.features``, and the core ``graph`` namespace.
"""

from __future__ import annotations

import numpy as np

from .. import _core


_REGION_ADJACENCY_GRAPH_BY_DTYPE = {
    np.dtype("uint32"): _core._region_adjacency_graph_uint32,
    np.dtype("uint64"): _core._region_adjacency_graph_uint64,
    np.dtype("int32"): _core._region_adjacency_graph_int32,
    np.dtype("int64"): _core._region_adjacency_graph_int64,
}


_GRID_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _as_shape(shape, ndim: int, name: str = "shape") -> list[int]:
    array = np.asarray(shape)
    if array.ndim != 1 or array.shape[0] != ndim:
        raise ValueError(f"{name} must be a 1D sequence of length {ndim}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if np.any(array <= 0):
        raise ValueError(f"{name} dimensions must be greater than zero")
    return [int(axis_size) for axis_size in array]


def _as_coordinate_array(coordinate, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(coordinate, dtype=np.uint64)
    if array.ndim != 1 or array.shape[0] != ndim:
        raise ValueError(f"{name} must be a 1D sequence of length {ndim}")
    return np.ascontiguousarray(array)


def _as_offset_array(offset, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(offset, dtype=np.int64)
    if array.ndim != 1 or array.shape[0] != ndim:
        raise ValueError(f"{name} must be a 1D sequence of length {ndim}")
    return np.ascontiguousarray(array)


def _as_uv_array(uvs, name: str) -> np.ndarray:
    array = np.asarray(uvs, dtype=np.uint64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_edges, 2)")
    return np.ascontiguousarray(array)


def _as_node_array(nodes, name: str) -> np.ndarray:
    array = np.asarray(nodes, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return np.ascontiguousarray(array)


def _as_serialization_array(serialization) -> np.ndarray:
    array = np.asarray(serialization, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError("serialization must be a 1D array")
    if array.size < 2:
        raise ValueError("serialization must have at least two entries")
    number_of_edges = int(array[1])
    if array.size != 2 + 2 * number_of_edges:
        raise ValueError("serialization size must be 2 + 2 * number_of_edges")
    return np.ascontiguousarray(array)


def _copy_graph(graph) -> _core.UndirectedGraph:
    # `uv_ids()` always returns a unique list (graphs deduplicate on insert),
    # so we can use the bulk constructor that skips per-edge hash dedup —
    # significantly faster than `insert_edges` for large graphs. The result
    # is a ``_core.UndirectedGraph``; downstream code (objectives, solvers,
    # validators) uses base-class methods that work identically.
    if graph.number_of_edges == 0:
        return _core.UndirectedGraph(int(graph.number_of_nodes))
    return _core.UndirectedGraph.from_unique_edges(
        int(graph.number_of_nodes), graph.uv_ids()
    )


def _as_edge_costs(edge_costs, graph) -> np.ndarray:
    array = np.asarray(edge_costs, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("edge_costs must be a 1D array")
    if array.shape[0] != graph.number_of_edges:
        raise ValueError("edge_costs length must match graph number_of_edges")
    return np.ascontiguousarray(array)


def _as_node_labels(labels, graph) -> np.ndarray:
    array = np.asarray(labels, dtype=np.uint64)
    if array.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if array.shape[0] != graph.number_of_nodes:
        raise ValueError("labels length must match graph number_of_nodes")
    return np.ascontiguousarray(array)


def _as_1d_array(values, dtype, name: str, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if array.shape[0] != expected_size:
        raise ValueError(
            f"{name} length must be {expected_size}, got {array.shape[0]}"
        )
    return np.ascontiguousarray(array)


def _dense_labels(labels) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint64)
    _, dense = np.unique(labels, return_inverse=True)
    return np.ascontiguousarray(dense.astype(np.uint64, copy=False))


def _subproblem_from_edges(number_of_nodes: int, nodes, uvs, edge_costs):
    # Local import to avoid a circular dependency with the multicut submodule
    # at module-load time (this helper is only called from the decomposer).
    from . import UndirectedGraph

    local_ids = np.full(int(number_of_nodes), -1, dtype=np.int64)
    local_ids[nodes] = np.arange(nodes.size, dtype=np.int64)
    local_uvs = local_ids[np.asarray(uvs, dtype=np.uint64)]
    sub_graph = UndirectedGraph(int(nodes.size), int(len(edge_costs)))
    if local_uvs.size:
        sub_graph.insert_edges(np.ascontiguousarray(local_uvs.astype(np.uint64, copy=False)))
    return sub_graph, np.ascontiguousarray(np.asarray(edge_costs, dtype=np.float64))


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim not in (2, 3):
        raise ValueError(f"labels must be a 2D or 3D array, got ndim={array.ndim}")
    try:
        _REGION_ADJACENCY_GRAPH_BY_DTYPE[array.dtype]
    except KeyError as error:
        supported = ", ".join(
            str(dtype) for dtype in _REGION_ADJACENCY_GRAPH_BY_DTYPE
        )
        raise TypeError(
            f"labels must have one of dtypes ({supported}), got dtype={array.dtype}"
        ) from error
    return np.ascontiguousarray(array)


def _normalize_number_of_threads(number_of_threads: int) -> int:
    number_of_threads = int(number_of_threads)
    if number_of_threads < 0:
        raise ValueError("number_of_threads must be non-negative")
    return number_of_threads


def _resolve_weight_dtype(array, name: str) -> np.ndarray:
    """Coerce a weights input to a supported floating dtype.

    ``float32`` and ``float64`` pass through unchanged. Other floating
    dtypes are cast to ``float32`` (matches the convention used by
    :func:`bioimage_cpp.graph.edge_weighted_watershed`). Non-floating dtypes
    raise.
    """
    array = np.asarray(array)
    if array.dtype in (np.dtype("float32"), np.dtype("float64")):
        return array
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32, copy=False)
    raise TypeError(
        f"{name} must have a floating dtype (float32 or float64), got "
        f"dtype={array.dtype}"
    )


def _grid_ndim(graph) -> int:
    # Local import to avoid a circular dependency at module-load time.
    from . import GridGraph2D, GridGraph3D

    if isinstance(graph, GridGraph2D):
        return 2
    if isinstance(graph, GridGraph3D):
        return 3
    raise TypeError("graph must be a GridGraph2D or GridGraph3D")


def _grid_shape(graph) -> tuple[int, ...]:
    return tuple(int(size) for size in graph.shape)


def _as_grid_data(values, graph, name: str, *, with_channels: bool) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype not in _GRID_FLOAT_DTYPES:
        # Integer / non-float input falls back to float64 — the previous default.
        # float32 and float64 inputs are passed through end-to-end, no copy.
        array = array.astype(np.float64)
    shape = _grid_shape(graph)
    if with_channels:
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
            raise ValueError(
                f"{name} must have shape (channels, *graph.shape), got "
                f"shape={array.shape}, graph shape={shape}"
            )
    elif array.shape != shape:
        raise ValueError(
            f"{name} shape must match graph shape, got "
            f"{name} shape={array.shape}, graph shape={shape}"
        )
    return np.ascontiguousarray(array)


def _normalize_grid_offsets(offsets, ndim: int, n_channels: int) -> list[tuple[int, ...]]:
    normalized = [tuple(int(value) for value in offset) for offset in offsets]
    if len(normalized) != n_channels:
        raise ValueError(
            "offsets length must match affinities channel count, got "
            f"offsets length={len(normalized)}, channels={n_channels}"
        )
    if any(len(offset) != ndim for offset in normalized):
        raise ValueError("each offset must have length matching graph ndim")
    if any(all(value == 0 for value in offset) for offset in normalized):
        raise ValueError("offsets must not contain the zero offset")
    return normalized


def _grid_dtype_suffix(array: np.ndarray) -> str:
    if array.dtype == np.float32:
        return "float32"
    return "float64"
