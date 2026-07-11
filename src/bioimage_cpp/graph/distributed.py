"""Low-level primitives for distributed region-adjacency graphs and features.

These functions compute a region adjacency graph (RAG) and its edge features
one block at a time and merge the per-block results into a whole-volume result.
They are the building blocks for processing volumes too large to hold in memory;
the surrounding orchestration — iterating blocks, sizing halos, serializing and
reading back the per-block artifacts (e.g. to zarr/N5/HDF5), and scheduling the
merges — lives outside this module and is **not** provided here.

The pipeline has three stages:

1. **Per block** (with a halo): :func:`block_region_adjacency_edges` extracts the
   edges the block owns; :func:`block_edge_map_stats` / :func:`block_affinity_stats`
   extract the partial edge statistics it owns. A block owns the pixel-pairs
   whose reference pixel lies in its inner (non-halo) box ``[own_begin,
   own_begin + own_shape)``; the neighbor pixel is read from the passed (outer,
   haloed) block. Because inner boxes tile the volume, every contribution is
   counted exactly once. The caller must supply a halo large enough to reach the
   neighbors (≥1 on the forward faces for nearest neighbors; ≥ ``max |offset|``
   per side for affinities) — too small a halo silently drops owned pairs.

2. **Merge the graph**: :func:`merge_edges` unions the per-block edges into the
   whole-volume edge set; build the global graph with
   :meth:`bioimage_cpp.graph.UndirectedGraph.from_unique_edges`.

3. **Merge the features**: fold each block's partial statistics onto the global
   edges with :func:`merge_block_edge_stats` (starting from
   :func:`empty_edge_stats`; the accumulator is updated in place), then convert
   to features with :func:`finalize_edge_features`.

Exactly-recoverable features are ``size``, ``mean``, ``std``, ``min`` and
``max``. Median and percentiles cannot be reconstructed from block partials, so
the distributed complex output is the moment subset ``[mean, std, min, max,
size]`` — it equals the corresponding columns of the in-core complex features.
"""

from __future__ import annotations

import numpy as np

from .. import _core
from ._shared import _normalize_labels, _normalize_number_of_threads


_BLOCK_REGION_ADJACENCY_EDGES_BY_DTYPE = {
    np.dtype("uint32"): _core._block_region_adjacency_edges_uint32,
    np.dtype("uint64"): _core._block_region_adjacency_edges_uint64,
    np.dtype("int32"): _core._block_region_adjacency_edges_int32,
    np.dtype("int64"): _core._block_region_adjacency_edges_int64,
}

_BLOCK_EDGE_MAP_STATS_BY_DTYPE = {
    np.dtype("uint32"): _core._block_edge_map_stats_uint32,
    np.dtype("uint64"): _core._block_edge_map_stats_uint64,
    np.dtype("int32"): _core._block_edge_map_stats_int32,
    np.dtype("int64"): _core._block_edge_map_stats_int64,
}

_BLOCK_AFFINITY_STATS_BY_DTYPE = {
    np.dtype("uint32"): _core._block_affinity_stats_uint32,
    np.dtype("uint64"): _core._block_affinity_stats_uint64,
    np.dtype("int32"): _core._block_affinity_stats_int32,
    np.dtype("int64"): _core._block_affinity_stats_int64,
}


def _as_index_vector(values, ndim: int, name: str) -> list[int]:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != ndim:
        raise ValueError(f"{name} must be a 1D sequence of length {ndim}")
    return [int(value) for value in array]


def _as_stats_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError(f"{name} must have shape (number_of_edges, 5)")
    return np.ascontiguousarray(array)


def _dispatch_labels(labels, table, name: str):
    label_array = _normalize_labels(labels)
    try:
        run = table[label_array.dtype]
    except KeyError as error:
        supported = ", ".join(str(dtype) for dtype in table)
        raise TypeError(
            f"{name} labels must have one of dtypes ({supported}), "
            f"got dtype={label_array.dtype}"
        ) from error
    return label_array, run


def block_region_adjacency_edges(
    labels: np.ndarray,
    own_begin,
    own_shape,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Extract the region-adjacency edges a block owns.

    ``labels`` is a (haloed) 2D or 3D label block with global label ids;
    ``own_begin`` / ``own_shape`` delimit the owned inner box in block-local
    coordinates. Returns an ``(n, 2)`` ``uint64`` array of ``(u, v)`` edges with
    ``u < v``, sorted lexicographically. Concatenate these across blocks and
    pass them to :func:`merge_edges` to obtain the whole-volume edge set.
    """
    label_array, run = _dispatch_labels(
        labels, _BLOCK_REGION_ADJACENCY_EDGES_BY_DTYPE, "region adjacency"
    )
    own_begin = _as_index_vector(own_begin, label_array.ndim, "own_begin")
    own_shape = _as_index_vector(own_shape, label_array.ndim, "own_shape")
    return run(
        label_array, own_begin, own_shape, _normalize_number_of_threads(number_of_threads)
    )


def block_edge_map_stats(
    labels: np.ndarray,
    edge_map: np.ndarray,
    own_begin,
    own_shape,
    *,
    number_of_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate the partial edge-map statistics a block owns.

    The value on an edge is the average of the two endpoint pixel values,
    matching :func:`bioimage_cpp.graph.features.edge_map_features`. ``edge_map``
    must have the same shape as ``labels``. Returns ``(edges, stats)`` where
    ``edges`` is ``(n, 2)`` ``uint64`` (matching
    :func:`block_region_adjacency_edges` for the same block) and ``stats`` is
    ``(n, 5)`` ``float64`` with columns ``[count, mean, M2, min, max]`` aligned
    row-by-row to ``edges`` (``M2`` is the sum of squared deviations from the
    mean, as in Welford's algorithm — numerically stable when merged).
    """
    label_array, run = _dispatch_labels(
        labels, _BLOCK_EDGE_MAP_STATS_BY_DTYPE, "edge-map"
    )
    edge_map_array = np.asarray(edge_map, dtype=np.float64)
    if edge_map_array.shape != label_array.shape:
        raise ValueError(
            "edge_map shape must match labels shape, got "
            f"edge_map shape={edge_map_array.shape}, labels shape={label_array.shape}"
        )
    own_begin = _as_index_vector(own_begin, label_array.ndim, "own_begin")
    own_shape = _as_index_vector(own_shape, label_array.ndim, "own_shape")
    return run(
        label_array,
        np.ascontiguousarray(edge_map_array),
        own_begin,
        own_shape,
        _normalize_number_of_threads(number_of_threads),
    )


def block_affinity_stats(
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    own_begin,
    own_shape,
    *,
    number_of_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate the partial affinity statistics a block owns.

    ``affinities`` has shape ``(len(offsets), *labels.shape)``; the value on an
    edge is the affinity stored at the reference node, matching
    :func:`bioimage_cpp.graph.features.affinity_features`. Values from all
    offsets are aggregated per ``(u, v)``. Returns ``(edges, stats)`` as in
    :func:`block_edge_map_stats`; ``edges`` may include long-range-only pairs,
    which are dropped at merge time if absent from the global graph.
    """
    label_array, run = _dispatch_labels(
        labels, _BLOCK_AFFINITY_STATS_BY_DTYPE, "affinity"
    )
    affinity_array = np.asarray(affinities, dtype=np.float64)
    if affinity_array.ndim != label_array.ndim + 1:
        raise ValueError("affinities must have shape (channels, *labels.shape)")
    if affinity_array.shape[1:] != label_array.shape:
        raise ValueError(
            "affinities spatial shape must match labels shape, got "
            f"affinities shape={affinity_array.shape}, labels shape={label_array.shape}"
        )

    normalized_offsets = [tuple(int(value) for value in offset) for offset in offsets]
    if len(normalized_offsets) != affinity_array.shape[0]:
        raise ValueError(
            "offsets length must match affinities channel count, got "
            f"offsets length={len(normalized_offsets)}, channels={affinity_array.shape[0]}"
        )
    if any(len(offset) != label_array.ndim for offset in normalized_offsets):
        raise ValueError("each offset must have length matching labels ndim")

    own_begin = _as_index_vector(own_begin, label_array.ndim, "own_begin")
    own_shape = _as_index_vector(own_shape, label_array.ndim, "own_shape")
    return run(
        label_array,
        np.ascontiguousarray(affinity_array),
        normalized_offsets,
        own_begin,
        own_shape,
        _normalize_number_of_threads(number_of_threads),
    )


def merge_edges(edges) -> np.ndarray:
    """Merge per-block edge arrays into the whole-volume edge set.

    ``edges`` is either a single ``(N, 2)`` array or a sequence of ``(n_i, 2)``
    arrays (which are concatenated). Edges are canonicalized to ``u < v``,
    self-edges are dropped, and the result is deduplicated and sorted
    lexicographically. The returned ``(E, 2)`` ``uint64`` array feeds
    :meth:`bioimage_cpp.graph.UndirectedGraph.from_unique_edges` directly.
    """
    if isinstance(edges, np.ndarray):
        stacked = edges
    else:
        parts = [np.asarray(part, dtype=np.uint64) for part in edges]
        if not parts:
            return np.empty((0, 2), dtype=np.uint64)
        stacked = np.concatenate(parts, axis=0)

    stacked = np.asarray(stacked, dtype=np.uint64)
    if stacked.ndim != 2 or stacked.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2)")
    return _core._merge_edges(np.ascontiguousarray(stacked))


def empty_edge_stats(number_of_edges: int) -> np.ndarray:
    """Return a zero-initialized ``(number_of_edges, 5)`` ``float64`` accumulator.

    Use this as the starting ``current_stats`` for :func:`merge_block_edge_stats`.
    """
    return np.zeros((int(number_of_edges), 5), dtype=np.float64)


def merge_block_edge_stats(
    global_graph,
    current_stats: np.ndarray,
    block_edges: np.ndarray,
    block_stats: np.ndarray,
) -> np.ndarray:
    """Fold one block's partial statistics onto the global edges.

    ``current_stats`` is the running ``(E, 5)`` accumulator (rows aligned to
    ``global_graph`` edge ids; start from :func:`empty_edge_stats`). **It is
    updated in place and returned**, so one merge costs O(block edges)
    regardless of the global graph size; it must be a C-contiguous, writable
    ``float64`` array (as produced by :func:`empty_edge_stats`).
    ``block_edges`` / ``block_stats`` are a block extraction's ``(n, 2)`` /
    ``(n, 5)`` outputs. Each block edge is mapped to its global edge id via
    ``global_graph.find_edge``; edges absent from the graph are skipped.
    ``count`` adds, ``mean/M2`` combine via the Chan formula, and ``min/max``
    reduce.

    Block edge endpoints must be valid node ids of ``global_graph``.
    """
    # Do not coerce/copy `current_stats` — the merge mutates it in place, and a
    # silent copy would discard the update.
    if not isinstance(current_stats, np.ndarray) or current_stats.dtype != np.float64:
        raise TypeError(
            "current_stats must be a float64 numpy array, got "
            f"{type(current_stats).__name__}"
            + (f" with dtype={current_stats.dtype}" if isinstance(current_stats, np.ndarray) else "")
        )
    if current_stats.ndim != 2 or current_stats.shape[1] != 5:
        raise ValueError("current_stats must have shape (number_of_edges, 5)")
    if not current_stats.flags.c_contiguous:
        raise ValueError("current_stats must be C-contiguous (it is updated in place)")
    if not current_stats.flags.writeable:
        raise ValueError("current_stats must be writable (it is updated in place)")
    if int(current_stats.shape[0]) != int(global_graph.number_of_edges):
        raise ValueError("current_stats rows must match global_graph number_of_edges")

    block_edge_array = np.asarray(block_edges, dtype=np.uint64)
    if block_edge_array.ndim != 2 or block_edge_array.shape[1] != 2:
        raise ValueError("block_edges must have shape (n_edges, 2)")
    block_stat_array = _as_stats_array(block_stats, "block_stats")
    if block_stat_array.shape[0] != block_edge_array.shape[0]:
        raise ValueError("block_edges and block_stats must have the same number of rows")

    _core._merge_block_edge_stats(
        global_graph,
        current_stats,
        np.ascontiguousarray(block_edge_array),
        block_stat_array,
    )
    return current_stats


def finalize_edge_features(
    stats: np.ndarray,
    *,
    compute_complex_features: bool = False,
) -> np.ndarray:
    """Convert accumulated partial statistics into edge features.

    ``stats`` is an ``(E, 5)`` accumulator (from :func:`merge_block_edge_stats`).
    With ``compute_complex_features=False`` returns ``(E, 2)`` columns
    ``[mean, size]``; with ``True`` returns ``(E, 5)`` columns
    ``[mean, std, min, max, size]``. Edges with zero count give all-zero rows.
    """
    stats_array = _as_stats_array(stats, "stats")
    return _core._finalize_edge_features(stats_array, bool(compute_complex_features))


__all__ = [
    "block_affinity_stats",
    "block_edge_map_stats",
    "block_region_adjacency_edges",
    "empty_edge_stats",
    "finalize_edge_features",
    "merge_block_edge_stats",
    "merge_edges",
]
