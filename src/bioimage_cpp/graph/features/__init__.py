"""Edge-feature accumulation on graphs.

Functions in this submodule compute edge-aligned feature vectors:

- :func:`edge_map_features` / :func:`edge_map_features_complex` — features
  derived from a scalar edge map on RAG boundaries.
- :func:`affinity_features` / :func:`affinity_features_complex` — features
  derived from affinity channels on RAG edges.
- :func:`lifted_edges_from_affinities`, :func:`lifted_affinity_features`,
  :func:`lifted_affinity_features_complex` — features for long-range
  (lifted) edges discovered from affinity offsets.
- :func:`grid_boundary_features`, :func:`grid_affinity_features`,
  :func:`grid_affinity_features_with_lifted` — weights for nearest-neighbor
  grid graphs (and optional long-range edges) computed directly from
  boundary maps / affinity channels.
- :func:`accumulate_labels` — majority-vote of a second label volume per
  RAG node (equivalent to nifty's ``gridRagAccumulateLabels``).
"""

from __future__ import annotations

import numpy as np

from .. import _core
from ..._validation import strict_offsets
from .._shared import (
    _as_grid_data,
    _as_uv_array,
    _grid_dtype_suffix,
    _grid_ndim,
    _normalize_grid_offsets,
    _normalize_labels,
    _normalize_number_of_threads,
)


SIMPLE_EDGE_FEATURE_NAMES = ("mean", "size")
COMPLEX_EDGE_FEATURE_NAMES = (
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p5",
    "p10",
    "p25",
    "p75",
    "p90",
    "p95",
    "size",
)


_EDGE_MAP_FEATURES_BY_DTYPE = {
    np.dtype("uint32"): _core._accumulate_edge_map_features_uint32,
    np.dtype("uint64"): _core._accumulate_edge_map_features_uint64,
    np.dtype("int32"): _core._accumulate_edge_map_features_int32,
    np.dtype("int64"): _core._accumulate_edge_map_features_int64,
}

_AFFINITY_FEATURES_BY_DTYPE = {
    np.dtype("uint32"): _core._accumulate_affinity_features_uint32,
    np.dtype("uint64"): _core._accumulate_affinity_features_uint64,
    np.dtype("int32"): _core._accumulate_affinity_features_int32,
    np.dtype("int64"): _core._accumulate_affinity_features_int64,
}

_LIFTED_EDGES_FROM_AFFINITIES_BY_DTYPE = {
    np.dtype("uint32"): _core._lifted_edges_from_affinities_uint32,
    np.dtype("uint64"): _core._lifted_edges_from_affinities_uint64,
    np.dtype("int32"): _core._lifted_edges_from_affinities_int32,
    np.dtype("int64"): _core._lifted_edges_from_affinities_int64,
}

_LIFTED_AFFINITY_FEATURES_BY_DTYPE = {
    np.dtype("uint32"): _core._accumulate_lifted_affinity_features_uint32,
    np.dtype("uint64"): _core._accumulate_lifted_affinity_features_uint64,
    np.dtype("int32"): _core._accumulate_lifted_affinity_features_int32,
    np.dtype("int64"): _core._accumulate_lifted_affinity_features_int64,
}

_GRID_BOUNDARY_DISPATCH = {
    (2, "float32"): _core._grid_boundary_features_2d_float32,
    (2, "float64"): _core._grid_boundary_features_2d_float64,
    (3, "float32"): _core._grid_boundary_features_3d_float32,
    (3, "float64"): _core._grid_boundary_features_3d_float64,
}
_GRID_AFFINITY_DISPATCH = {
    (2, "float32"): _core._grid_affinity_features_2d_float32,
    (2, "float64"): _core._grid_affinity_features_2d_float64,
    (3, "float32"): _core._grid_affinity_features_3d_float32,
    (3, "float64"): _core._grid_affinity_features_3d_float64,
}
_GRID_AFFINITY_LIFTED_DISPATCH = {
    (2, "float32"): _core._grid_affinity_features_with_lifted_2d_float32,
    (2, "float64"): _core._grid_affinity_features_with_lifted_2d_float64,
    (3, "float32"): _core._grid_affinity_features_with_lifted_3d_float32,
    (3, "float64"): _core._grid_affinity_features_with_lifted_3d_float64,
}


_LABEL_DTYPES = (
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("int32"),
    np.dtype("int64"),
)


def _accumulate_labels_dispatch():
    suffix = {
        np.dtype("uint32"): "uint32",
        np.dtype("uint64"): "uint64",
        np.dtype("int32"): "int32",
        np.dtype("int64"): "int64",
    }
    dispatch = {}
    for labels_dtype in _LABEL_DTYPES:
        for other_dtype in _LABEL_DTYPES:
            name = (
                f"_accumulate_labels_{suffix[labels_dtype]}_{suffix[other_dtype]}"
            )
            dispatch[(labels_dtype, other_dtype)] = getattr(_core, name)
    return dispatch


_ACCUMULATE_LABELS_DISPATCH = _accumulate_labels_dispatch()


def edge_map_features(
    rag,
    labels: np.ndarray,
    edge_map: np.ndarray,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute mean and size features for edge-map values on RAG boundaries."""
    return _accumulate_edge_map_features(
        rag,
        labels,
        edge_map,
        compute_complex_features=False,
        number_of_threads=number_of_threads,
    )


def edge_map_features_complex(
    rag,
    labels: np.ndarray,
    edge_map: np.ndarray,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute complex edge-map features on RAG boundaries.

    The output columns are given by :data:`COMPLEX_EDGE_FEATURE_NAMES`.
    """
    return _accumulate_edge_map_features(
        rag,
        labels,
        edge_map,
        compute_complex_features=True,
        number_of_threads=number_of_threads,
    )


def affinity_features(
    rag,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute mean and size features for affinity links crossing RAG edges."""
    return _accumulate_affinity_features(
        rag,
        labels,
        affinities,
        offsets,
        compute_complex_features=False,
        number_of_threads=number_of_threads,
    )


def affinity_features_complex(
    rag,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute complex affinity features for links crossing RAG edges.

    The output columns are given by :data:`COMPLEX_EDGE_FEATURE_NAMES`.
    """
    return _accumulate_affinity_features(
        rag,
        labels,
        affinities,
        offsets,
        compute_complex_features=True,
        number_of_threads=number_of_threads,
    )


def lifted_edges_from_affinities(
    rag,
    labels: np.ndarray,
    offsets,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Discover lifted edges implied by long-range affinity offsets.

    Walks every grid coordinate together with each long-range offset (1-hop
    offsets are skipped automatically). When the labels at ``(p, p + offset)``
    differ and ``(labels[p], labels[p + offset])`` is not already a local
    edge of ``rag``, the pair is recorded as a lifted edge.

    Parameters
    ----------
    rag:
        :class:`bioimage_cpp.graph.RegionAdjacencyGraph` built from ``labels``.
    labels:
        2D or 3D label array. Supported dtypes: ``uint32``, ``uint64``,
        ``int32``, ``int64``.
    offsets:
        Sequence of per-channel offsets. Each offset must have length equal
        to ``labels.ndim``. Offsets with L1 norm ``<= 1`` are skipped, so
        callers can pass the full offset list of an affinity volume without
        pre-filtering.

    Returns
    -------
    np.ndarray
        ``(n_lifted, 2)`` ``uint64`` array of ``(u, v)`` pairs with
        ``u < v``, sorted lexicographically.
    """
    label_array = _normalize_labels(labels)
    if tuple(int(size) for size in rag.shape) != label_array.shape:
        raise ValueError(
            "rag shape must match labels shape, got "
            f"rag shape={tuple(rag.shape)}, labels shape={label_array.shape}"
        )

    if np.asarray(offsets).size == 0:
        return np.empty((0, 2), dtype=np.uint64)
    normalized_offsets = strict_offsets(offsets, label_array.ndim)
    if any(len(offset) != label_array.ndim for offset in normalized_offsets):
        raise ValueError("each offset must have length matching labels ndim")

    run = _LIFTED_EDGES_FROM_AFFINITIES_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        normalized_offsets,
        _normalize_number_of_threads(number_of_threads),
    )


def lifted_affinity_features(
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    lifted_uvs,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Compute mean and size features for affinity links across lifted edges.

    Affinity values at pixel pairs ``(p, p + offset)`` whose labels match a
    row of ``lifted_uvs`` are binned into that lifted edge. Pixel pairs that
    fall on a non-lifted edge (or no edge at all) are silently skipped, so
    a local edge that is also reachable by a long-range offset is not
    contaminated by long-range affinities.

    1-hop offsets are skipped automatically.

    The returned array has shape ``(len(lifted_uvs), 2)`` with columns
    :data:`SIMPLE_EDGE_FEATURE_NAMES` (``mean``, ``size``).
    """
    return _accumulate_lifted_affinity_features(
        labels,
        affinities,
        offsets,
        lifted_uvs,
        compute_complex_features=False,
        number_of_threads=number_of_threads,
    )


def lifted_affinity_features_complex(
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    lifted_uvs,
    *,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Complex affinity features for links across lifted edges.

    Output columns: :data:`COMPLEX_EDGE_FEATURE_NAMES`.
    """
    return _accumulate_lifted_affinity_features(
        labels,
        affinities,
        offsets,
        lifted_uvs,
        compute_complex_features=True,
        number_of_threads=number_of_threads,
    )


def grid_boundary_features(graph, boundary_map) -> np.ndarray:
    """Compute scalar nearest-neighbor grid edge weights from a boundary map.

    The output is a 1D array aligned to ``graph.edges()``. Output dtype matches
    the input: ``float32`` and ``float64`` inputs are processed without copying,
    other dtypes are promoted to ``float64``. Each edge receives the average of
    the two endpoint boundary-map values.
    """
    ndim = _grid_ndim(graph)
    boundary_array = _as_grid_data(
        boundary_map, graph, "boundary_map", with_channels=False
    )
    return _GRID_BOUNDARY_DISPATCH[(ndim, _grid_dtype_suffix(boundary_array))](
        graph, boundary_array
    )


def grid_affinity_features(graph, affinities, offsets) -> tuple[np.ndarray, np.ndarray]:
    """Map local affinity channels to grid graph edge weights.

    Only nearest-neighbor offsets with L1 norm 1 are accepted. The returned
    tuple is ``(edge_weights, valid_edges)``, both aligned to ``graph.edges()``.
    ``edge_weights`` has the same dtype as ``affinities`` (``float32`` or
    ``float64``; other dtypes are promoted to ``float64``). ``valid_edges`` is
    boolean and marks local graph edges covered by offsets.
    """
    ndim = _grid_ndim(graph)
    affinity_array = _as_grid_data(
        affinities, graph, "affinities", with_channels=True
    )
    normalized_offsets = _normalize_grid_offsets(
        offsets, ndim, int(affinity_array.shape[0])
    )
    weights, valid = _GRID_AFFINITY_DISPATCH[
        (ndim, _grid_dtype_suffix(affinity_array))
    ](graph, affinity_array, normalized_offsets)
    return weights, valid.astype(bool, copy=False)


def grid_affinity_features_with_lifted(
    graph,
    affinities,
    offsets,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map local affinities and emit explicit long-range grid edges.

    Returns ``(local_weights, valid_local_edges, lifted_uvs, lifted_weights,
    lifted_offset_ids)``. Local arrays are aligned to ``graph.edges()``.
    Long-range arrays have one row/value per valid offset hit and are suitable
    for lifted multicut or mutex-watershed style callers. Weight arrays share
    the dtype of ``affinities`` (``float32`` or ``float64``; other dtypes are
    promoted to ``float64``).
    """
    ndim = _grid_ndim(graph)
    affinity_array = _as_grid_data(
        affinities, graph, "affinities", with_channels=True
    )
    normalized_offsets = _normalize_grid_offsets(
        offsets, ndim, int(affinity_array.shape[0])
    )
    local_weights, valid, lifted_uvs, lifted_weights, lifted_offset_ids = (
        _GRID_AFFINITY_LIFTED_DISPATCH[
            (ndim, _grid_dtype_suffix(affinity_array))
        ](graph, affinity_array, normalized_offsets)
    )
    return (
        local_weights,
        valid.astype(bool, copy=False),
        lifted_uvs,
        lifted_weights,
        lifted_offset_ids,
    )


def accumulate_labels(
    rag,
    labels: np.ndarray,
    other_labels: np.ndarray,
    *,
    ignore_value: int | None = None,
    number_of_threads: int = 0,
) -> np.ndarray:
    """Majority-vote of ``other_labels`` per RAG node.

    For each node in ``rag``, returns the most frequent value of
    ``other_labels`` across the pixels assigned to that node by the
    over-segmentation ``labels``. This is the equivalent of nifty's
    ``gridRagAccumulateLabels``.

    Parameters
    ----------
    rag:
        :class:`bioimage_cpp.graph.RegionAdjacencyGraph` built from ``labels``.
    labels:
        2D or 3D over-segmentation used to construct ``rag``. Supported
        dtypes: ``uint32``, ``uint64``, ``int32``, ``int64``.
    other_labels:
        Secondary label volume with the same shape as ``labels``. Supported
        dtypes: ``uint32``, ``uint64``, ``int32``, ``int64``.
    ignore_value:
        Optional. Pixels where ``other_labels`` equals this value are
        excluded from the histogram. Set to ``0`` to reproduce nifty's
        ``ignoreBackground=True``.
    number_of_threads:
        ``0`` (default) uses the library default; positive integers fix
        the thread count.

    Returns
    -------
    np.ndarray
        1D array of length ``rag.number_of_nodes`` with the same dtype as
        ``other_labels``. Nodes whose pixels are all ignored (or for which
        no pixel contributes) get ``0``. Ties are broken by smaller label
        id (deterministic).
    """
    label_array = _normalize_labels(labels)
    if tuple(int(size) for size in rag.shape) != label_array.shape:
        raise ValueError(
            "rag shape must match labels shape, got "
            f"rag shape={tuple(rag.shape)}, labels shape={label_array.shape}"
        )

    other_array = np.asarray(other_labels)
    if other_array.dtype not in _LABEL_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _LABEL_DTYPES)
        raise TypeError(
            f"other_labels must have one of dtypes ({supported}), got "
            f"dtype={other_array.dtype}"
        )
    if other_array.shape != label_array.shape:
        raise ValueError(
            "other_labels shape must match labels shape, got "
            f"other_labels shape={other_array.shape}, "
            f"labels shape={label_array.shape}"
        )
    other_array = np.ascontiguousarray(other_array)

    if ignore_value is None:
        has_ignore_value = False
        ignore_scalar = other_array.dtype.type(0)
    else:
        info = np.iinfo(other_array.dtype)
        ignore_int = int(ignore_value)
        if ignore_int < info.min or ignore_int > info.max:
            raise ValueError(
                f"ignore_value={ignore_int} is not representable in "
                f"other_labels dtype {other_array.dtype}"
            )
        has_ignore_value = True
        ignore_scalar = other_array.dtype.type(ignore_int)

    run = _ACCUMULATE_LABELS_DISPATCH[(label_array.dtype, other_array.dtype)]
    return run(
        rag,
        label_array,
        other_array,
        bool(has_ignore_value),
        ignore_scalar,
        _normalize_number_of_threads(number_of_threads),
    )


def _accumulate_edge_map_features(
    rag,
    labels: np.ndarray,
    edge_map: np.ndarray,
    *,
    compute_complex_features: bool,
    number_of_threads: int,
) -> np.ndarray:
    label_array = _normalize_labels(labels)
    edge_map_array = np.asarray(edge_map, dtype=np.float64)
    if edge_map_array.shape != label_array.shape:
        raise ValueError(
            "edge_map shape must match labels shape, got "
            f"edge_map shape={edge_map_array.shape}, labels shape={label_array.shape}"
        )
    run = _EDGE_MAP_FEATURES_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        np.ascontiguousarray(edge_map_array),
        bool(compute_complex_features),
        _normalize_number_of_threads(number_of_threads),
    )


def _accumulate_affinity_features(
    rag,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    *,
    compute_complex_features: bool,
    number_of_threads: int,
) -> np.ndarray:
    label_array = _normalize_labels(labels)
    affinity_array = np.asarray(affinities, dtype=np.float64)
    if affinity_array.ndim != label_array.ndim + 1:
        raise ValueError("affinities must have shape (channels, *labels.shape)")
    if affinity_array.shape[1:] != label_array.shape:
        raise ValueError(
            "affinities spatial shape must match labels shape, got "
            f"affinities shape={affinity_array.shape}, labels shape={label_array.shape}"
        )

    normalized_offsets = strict_offsets(offsets, label_array.ndim)
    if len(normalized_offsets) != affinity_array.shape[0]:
        raise ValueError(
            "offsets length must match affinities channel count, got "
            f"offsets length={len(normalized_offsets)}, channels={affinity_array.shape[0]}"
        )
    if any(len(offset) != label_array.ndim for offset in normalized_offsets):
        raise ValueError("each offset must have length matching labels ndim")

    run = _AFFINITY_FEATURES_BY_DTYPE[label_array.dtype]
    return run(
        rag,
        label_array,
        np.ascontiguousarray(affinity_array),
        normalized_offsets,
        bool(compute_complex_features),
        _normalize_number_of_threads(number_of_threads),
    )


def _accumulate_lifted_affinity_features(
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets,
    lifted_uvs,
    *,
    compute_complex_features: bool,
    number_of_threads: int,
) -> np.ndarray:
    label_array = _normalize_labels(labels)
    affinity_array = np.asarray(affinities, dtype=np.float64)
    if affinity_array.ndim != label_array.ndim + 1:
        raise ValueError("affinities must have shape (channels, *labels.shape)")
    if affinity_array.shape[1:] != label_array.shape:
        raise ValueError(
            "affinities spatial shape must match labels shape, got "
            f"affinities shape={affinity_array.shape}, labels shape={label_array.shape}"
        )

    normalized_offsets = strict_offsets(offsets, label_array.ndim)
    if len(normalized_offsets) != affinity_array.shape[0]:
        raise ValueError(
            "offsets length must match affinities channel count, got "
            f"offsets length={len(normalized_offsets)}, channels={affinity_array.shape[0]}"
        )
    if any(len(offset) != label_array.ndim for offset in normalized_offsets):
        raise ValueError("each offset must have length matching labels ndim")

    lifted_uv_array = _as_uv_array(lifted_uvs, "lifted_uvs")

    run = _LIFTED_AFFINITY_FEATURES_BY_DTYPE[label_array.dtype]
    return run(
        label_array,
        np.ascontiguousarray(affinity_array),
        normalized_offsets,
        lifted_uv_array,
        bool(compute_complex_features),
        _normalize_number_of_threads(number_of_threads),
    )


__all__ = [
    "COMPLEX_EDGE_FEATURE_NAMES",
    "SIMPLE_EDGE_FEATURE_NAMES",
    "accumulate_labels",
    "affinity_features",
    "affinity_features_complex",
    "edge_map_features",
    "edge_map_features_complex",
    "grid_affinity_features",
    "grid_affinity_features_with_lifted",
    "grid_boundary_features",
    "lifted_affinity_features",
    "lifted_affinity_features_complex",
    "lifted_edges_from_affinities",
]
