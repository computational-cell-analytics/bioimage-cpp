"""Low-level primitives for blockwise TEASAR skeletonization and stitching.

The caller owns block iteration, storage, scheduling, retries, and reduction
orchestration. A processing block should contain its non-overlapping core plus
one voxel on every high side that has a neighbor. The resulting shared plane is
passed to :func:`block_border_targets`; its global targets are then mandatory
rails for :func:`block_teasar` in both neighboring blocks. The same faces must
be supplied as ``open_faces`` so their artificial cuts do not collapse the
distance-to-boundary radius.

Block artifacts are ``(vertices, edges, radii)`` tuples. Unlike ordinary
TEASAR output, ``vertices`` contains global ``int64`` lattice coordinates.
This makes :func:`merge_block_skeletons` an exact canonical set union and lets
its output feed another hierarchical merge without floating-point recovery.
Call :func:`lattice_to_physical` only after the final merge and optional cycle
removal.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np

from .. import _core
from .._validation import strict_index
from ..distance._distance import _as_binary_input, _normalize_sampling, _normalize_threads
from . import _normalize_teasar_options


SkeletonFragment: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]

_LABEL_DTYPES = (
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("int32"),
    np.dtype("int64"),
)

_BORDER_TARGET_LABELS = {
    dtype: getattr(_core, f"_block_border_targets_labels_{dtype.name}")
    for dtype in _LABEL_DTYPES
}
_BLOCK_TEASAR_LABELS = {
    dtype: getattr(_core, f"_block_teasar_labels_{dtype.name}")
    for dtype in _LABEL_DTYPES
}


def _normalize_origin(origin) -> list[int]:
    try:
        values = [operator.index(value) for value in origin]
    except TypeError as error:
        raise TypeError("origin must contain exactly three integers") from error
    if len(values) != 3:
        raise ValueError(f"origin must contain exactly three values, got {len(values)}")
    info = np.iinfo(np.int64)
    if any(value < info.min or value > info.max for value in values):
        raise ValueError("origin values must fit int64")
    return values


def _normalize_faces(faces) -> tuple[list[int], list[int]]:
    axes: list[int] = []
    high: list[int] = []
    try:
        entries = list(faces)
    except TypeError as error:
        raise TypeError("faces must be a sequence of (axis, side) pairs") from error
    for index, entry in enumerate(entries):
        try:
            axis, side = entry
        except (TypeError, ValueError) as error:
            raise ValueError(f"faces[{index}] must be an (axis, side) pair") from error
        try:
            axis_value = operator.index(axis)
        except TypeError as error:
            raise TypeError(f"faces[{index}] axis must be an integer") from error
        if axis_value < 0 or axis_value >= 3:
            raise ValueError(f"faces[{index}] axis must be in [0, 3)")
        if side not in ("low", "high"):
            raise ValueError(f"faces[{index}] side must be 'low' or 'high'")
        axes.append(axis_value)
        high.append(side == "high")
    return axes, high


def _normalize_integer_coordinates(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        if not (
            (array.ndim == 1 and array.shape == (0,))
            or (array.ndim == 2 and array.shape[1] == 3)
        ):
            raise ValueError(f"{name} must have shape (n, 3), got shape={array.shape}")
        return np.empty((0, 3), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got shape={array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers, got dtype={array.dtype}")
    info = np.iinfo(np.int64)
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if np.any(array > info.max):
            raise ValueError(f"{name} values must fit int64")
    elif np.any(array < info.min) or np.any(array > info.max):
        raise ValueError(f"{name} values must fit int64")
    return np.ascontiguousarray(array, dtype=np.int64)


def _normalize_labels(labels, function: str):
    array = np.asarray(labels)
    if array.ndim != 3:
        raise ValueError(f"{function}: labels must have ndim 3, got ndim={array.ndim}")
    if array.dtype not in _LABEL_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _LABEL_DTYPES)
        raise TypeError(
            f"{function}: labels must have one of native-endian dtypes "
            f"({supported}), got dtype={array.dtype}"
        )
    return np.ascontiguousarray(array)


def _normalize_background(background, dtype: np.dtype):
    value = strict_index(background, "background")
    info = np.iinfo(dtype)
    if value < info.min or value > info.max:
        raise ValueError(f"background={value} is outside the range of dtype {dtype}")
    return dtype.type(value)


def block_border_targets(
    mask: np.ndarray,
    faces,
    *,
    origin=(0, 0, 0),
    spacing: float | Sequence[float] | None = (1.0, 1.0, 1.0),
    number_of_threads: int = 1,
) -> np.ndarray:
    """Select deterministic global stitch targets on requested block faces.

    ``faces`` contains ``(axis, "low"|"high")`` pairs. One target is selected
    for every 8-connected foreground patch using an anisotropic 2D distance
    transform. Plateaus are resolved by component centroid, face centre,
    corner, edge, and finally global coordinate. The sorted unique result has
    shape ``(T, 3)`` and dtype ``int64``.
    """
    function = "block_border_targets"
    binary = _as_binary_input(mask, function)
    if binary.ndim != 3:
        raise ValueError(f"{function}: mask must have ndim 3, got ndim={binary.ndim}")
    axes, high = _normalize_faces(faces)
    return _core._block_border_targets_uint8(
        binary,
        axes,
        high,
        _normalize_origin(origin),
        _normalize_sampling(spacing, 3, function, name="spacing"),
        _normalize_threads(number_of_threads, function),
    )


def block_border_targets_labels(
    labels: np.ndarray,
    faces,
    *,
    origin=(0, 0, 0),
    background: int = 0,
    spacing: float | Sequence[float] | None = (1.0, 1.0, 1.0),
    number_of_threads: int = 1,
) -> dict[int, np.ndarray]:
    """Select deterministic global stitch targets per semantic label.

    Face connectivity requires equal non-background label values. The returned
    dictionary preserves original integer labels and stores a sorted global
    ``int64 (T, 3)`` coordinate array for each label.
    """
    function = "block_border_targets_labels"
    array = _normalize_labels(labels, function)
    axes, high = _normalize_faces(faces)
    return _BORDER_TARGET_LABELS[array.dtype](
        array,
        _normalize_background(background, array.dtype),
        axes,
        high,
        _normalize_origin(origin),
        _normalize_sampling(spacing, 3, function, name="spacing"),
        _normalize_threads(number_of_threads, function),
    )


def block_teasar(
    mask: np.ndarray,
    *,
    open_faces,
    origin=(0, 0, 0),
    required_targets=None,
    spacing: float | Sequence[float] | None = (1.0, 1.0, 1.0),
    scale: float = 1.5,
    constant: float = 0.0,
    pdrf_scale: float = 100000.0,
    pdrf_exponent: float = 4.0,
    number_of_threads: int = 1,
) -> SkeletonFragment:
    """Skeletonize a binary processing block into a global lattice fragment.

    ``open_faces`` declares artificial processing-block cuts. Their foreground
    is extended only for the distance-to-boundary transform; paths remain
    confined to real input voxels. ``required_targets`` contains global
    coordinates and is normally the union of these faces' targets. One target
    on an open face becomes a deterministic component root and every other
    target is forced onto a rail.
    """
    function = "block_teasar"
    binary = _as_binary_input(mask, function)
    if binary.ndim != 3:
        raise ValueError(f"{function}: mask must have ndim 3, got ndim={binary.ndim}")
    targets = _normalize_integer_coordinates(
        [] if required_targets is None else required_targets,
        "required_targets",
    )
    open_axes, open_high = _normalize_faces(open_faces)
    options = _normalize_teasar_options(
        function, spacing, scale, constant, pdrf_scale, pdrf_exponent,
        number_of_threads,
    )
    return _core._block_teasar_uint8(
        binary, targets, open_axes, open_high, _normalize_origin(origin), *options
    )


def block_teasar_labels(
    labels: np.ndarray,
    *,
    open_faces,
    origin=(0, 0, 0),
    required_targets: Mapping[int, np.ndarray] | None = None,
    background: int = 0,
    spacing: float | Sequence[float] | None = (1.0, 1.0, 1.0),
    scale: float = 1.5,
    constant: float = 0.0,
    pdrf_scale: float = 100000.0,
    pdrf_exponent: float = 4.0,
    number_of_threads: int = 1,
) -> dict[int, SkeletonFragment]:
    """Skeletonize a labeled processing block into global lattice fragments.

    ``open_faces`` declares artificial processing-block cuts for DBF purposes.
    ``required_targets`` maps original labels to global coordinate arrays. A
    coordinate must contain exactly its mapping key in ``labels``. One open-face
    target roots each affected component. The result has one global lattice
    forest per original non-background label.
    """
    function = "block_teasar_labels"
    array = _normalize_labels(labels, function)
    if required_targets is None:
        target_map = {}
    elif not isinstance(required_targets, Mapping):
        raise TypeError("required_targets must be a mapping from labels to coordinates")
    else:
        target_map = {}
        info = np.iinfo(array.dtype)
        for key, coordinates in required_targets.items():
            label = strict_index(key, "required target label")
            if label < info.min or label > info.max:
                raise ValueError(
                    f"required target label {label} is outside dtype {array.dtype}"
                )
            target_map[int(label)] = _normalize_integer_coordinates(
                coordinates, f"required_targets[{label}]"
            )
    options = _normalize_teasar_options(
        function, spacing, scale, constant, pdrf_scale, pdrf_exponent,
        number_of_threads,
    )
    open_axes, open_high = _normalize_faces(open_faces)
    return _BLOCK_TEASAR_LABELS[array.dtype](
        array,
        _normalize_background(background, array.dtype),
        target_map,
        open_axes,
        open_high,
        _normalize_origin(origin),
        *options,
    )


def _normalize_fragment(fragment, name: str = "fragment") -> SkeletonFragment:
    try:
        vertices, edges, radii = fragment
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a (vertices, edges, radii) tuple") from error
    vertices_array = _normalize_integer_coordinates(vertices, f"{name} vertices")

    edges_array = np.asarray(edges)
    if edges_array.size == 0:
        if not (
            (edges_array.ndim == 1 and edges_array.shape == (0,))
            or (edges_array.ndim == 2 and edges_array.shape[1] == 2)
        ):
            raise ValueError(f"{name} edges must have shape (n, 2)")
        edges_array = np.empty((0, 2), dtype=np.uint64)
    else:
        if edges_array.ndim != 2 or edges_array.shape[1] != 2:
            raise ValueError(f"{name} edges must have shape (n, 2)")
        if not np.issubdtype(edges_array.dtype, np.integer):
            raise TypeError(f"{name} edges must contain integers")
        if np.issubdtype(edges_array.dtype, np.signedinteger) and np.any(edges_array < 0):
            raise ValueError(f"{name} edges must be non-negative")
        edges_array = np.ascontiguousarray(edges_array, dtype=np.uint64)

    radii_array = np.asarray(radii)
    if radii_array.ndim != 1 or radii_array.shape[0] != vertices_array.shape[0]:
        raise ValueError(f"{name} radii must have shape (n_vertices,)")
    if not np.issubdtype(radii_array.dtype, np.floating):
        raise TypeError(f"{name} radii must have a floating dtype")
    radii_array = np.ascontiguousarray(radii_array, dtype=np.float32)
    return vertices_array, edges_array, radii_array


def merge_block_skeletons(fragments) -> SkeletonFragment:
    """Exactly consolidate global lattice fragments for one semantic object.

    Equal coordinates are unified, duplicate radii reduce by maximum, and
    remapped edges are canonicalized, sorted, and deduplicated. Isolated
    vertices are retained. The operation is associative, commutative, and
    idempotent, so its output is directly reusable in a reduction tree.
    """
    normalized = [
        _normalize_fragment(fragment, f"fragments[{index}]")
        for index, fragment in enumerate(fragments)
    ]
    return _core._merge_block_skeletons(normalized)


def merge_block_skeleton_maps(
    fragment_maps,
) -> dict[int, SkeletonFragment]:
    """Consolidate block fragment dictionaries without mixing semantic labels."""
    maps = list(fragment_maps)
    if any(not isinstance(mapping, Mapping) for mapping in maps):
        raise TypeError("fragment_maps must contain mappings")
    labels = sorted({strict_index(label, "fragment label") for mapping in maps for label in mapping})
    return {
        label: merge_block_skeletons(
            [mapping[label] for mapping in maps if label in mapping]
        )
        for label in labels
    }


def minimum_spanning_forest(
    fragment: SkeletonFragment,
    *,
    spacing: float | Sequence[float] | None = (1.0, 1.0, 1.0),
) -> SkeletonFragment:
    """Remove cycles with a deterministic minimum-physical-length forest.

    This explicit postprocessor preserves all vertices and graph connected
    components. Exact consolidation never removes cycles on its own.
    """
    return _core._minimum_spanning_forest(
        _normalize_fragment(fragment),
        _normalize_sampling(spacing, 3, "minimum_spanning_forest", name="spacing"),
    )


def lattice_to_physical(
    fragment: SkeletonFragment,
    *,
    spacing: float | Sequence[float] | None = (1.0, 1.0, 1.0),
) -> SkeletonFragment:
    """Finalize a global lattice fragment as a physical-coordinate skeleton.

    The returned tuple matches ordinary TEASAR dtypes: float64 physical
    vertices, uint64 edges, and float32 radii.
    """
    return _core._lattice_to_physical(
        _normalize_fragment(fragment),
        _normalize_sampling(spacing, 3, "lattice_to_physical", name="spacing"),
    )


__all__ = [
    "block_border_targets",
    "block_border_targets_labels",
    "block_teasar",
    "block_teasar_labels",
    "lattice_to_physical",
    "merge_block_skeleton_maps",
    "merge_block_skeletons",
    "minimum_spanning_forest",
]
