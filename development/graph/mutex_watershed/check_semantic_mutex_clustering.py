"""Compare bioimage-cpp ``semantic_mutex_watershed_clustering`` against
affogato's ``compute_semantic_mws_clustering`` reference.

Pipeline (single mode, no subcommands):

1. Load the registered ``semantic_labels`` volume (instance + semantic ground
   truth + raw, cropped to the labelled slab inside ``_data.py``).
2. Derive an MWS-style affinity stack from the instance + semantic labels via
   the helper from ``../segmentation/_semantic_mws_equivalence.py``
   (``bic.affinities.compute_affinities`` + Gaussian noise + tiebreak
   perturbation).
3. Oversegment with a simple grid-marker watershed on
   ``1 - mean(attractive affinities)``.
4. Build a RAG from the oversegmentation and reduce weights to per-edge
   attractive costs (``bic.graph.affinity_features`` mean), to long-range
   mutex edges (``bic.graph.lifted_edges_from_affinities`` +
   ``bic.graph.lifted_affinity_features``), and to per-(node, class)
   semantic costs via ``scipy.ndimage.mean``.
5. Run bioimage-cpp + affogato side-by-side, time both, and report partition
   + semantic agreement.

The two implementations will not in general agree exactly — see the script's
output for the reason. Not part of the pytest suite (per AGENTS.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np

# Reuse the array-version helper for deriving weights, so both flavours of the
# comparison build their inputs the same way.
_SEGMENTATION_DEV = Path(__file__).resolve().parent.parent / "segmentation"
if str(_SEGMENTATION_DEV) not in sys.path:
    sys.path.insert(0, str(_SEGMENTATION_DEV))
from _semantic_mws_equivalence import build_weight_volume  # noqa: E402


_DEFAULT_2D_OFFSETS: list[tuple[int, ...]] = [
    (-1, 0),
    (0, -1),
    (-3, 0),
    (0, -3),
    (-9, 0),
    (0, -9),
]
_DEFAULT_2D_ATTRACTIVE = 2

_DEFAULT_3D_OFFSETS: list[tuple[int, ...]] = [
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
    (0, -3, 0),
    (0, 0, -3),
    (0, -9, 0),
    (0, 0, -9),
]
_DEFAULT_3D_ATTRACTIVE = 3


def _load_validation_metrics():
    try:
        from elf.validation import rand_index, variation_of_information

        return "elf.validation", rand_index, variation_of_information
    except ImportError:
        from elf.evaluation import rand_index, variation_of_information

        return "elf.evaluation", rand_index, variation_of_information


def load_volume(ndim: int, z_start: int, shape: tuple[int, ...]):
    from bioimage_cpp._data import load_semantic_labels

    instance, semantic = load_semantic_labels()
    if ndim == 2:
        y, x = shape
        z = z_start
        inst = np.ascontiguousarray(instance[z, :y, :x])
        sem = np.ascontiguousarray(semantic[z, :y, :x])
        offsets = list(_DEFAULT_2D_OFFSETS)
        attractive = _DEFAULT_2D_ATTRACTIVE
    elif ndim == 3:
        z, y, x = shape
        inst = np.ascontiguousarray(instance[z_start : z_start + z, :y, :x])
        sem = np.ascontiguousarray(semantic[z_start : z_start + z, :y, :x])
        offsets = list(_DEFAULT_3D_OFFSETS)
        attractive = _DEFAULT_3D_ATTRACTIVE
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    return inst, sem, offsets, attractive


def oversegment(
    weights: np.ndarray,
    number_of_attractive_channels: int,
    seed_spacing: int,
) -> np.ndarray:
    """Stride-marker watershed on the inverted mean attractive affinity."""
    from skimage.segmentation import watershed

    attr = weights[:number_of_attractive_channels]
    heightmap = np.ascontiguousarray(
        1.0 - attr.mean(axis=0), dtype=np.float32
    )
    markers = np.zeros(heightmap.shape, dtype=np.int32)
    slices = tuple(slice(None, None, seed_spacing) for _ in heightmap.shape)
    coord_grid = np.argwhere(np.ones(heightmap.shape, dtype=bool)[slices]) * seed_spacing
    for marker_id, coord in enumerate(coord_grid, start=1):
        markers[tuple(coord)] = marker_id
    return watershed(heightmap, markers=markers).astype(np.uint64, copy=False)


def build_costs(
    over_seg: np.ndarray,
    weights: np.ndarray,
    offsets: list[tuple[int, ...]],
    number_of_attractive_channels: int,
    number_of_offsets: int,
    n_classes: int,
    *,
    rag_threads: int,
):
    """Reduce the per-pixel weight stack to RAG-level costs."""
    import bioimage_cpp as bic
    from scipy.ndimage import mean as ndi_mean

    over_seg = np.ascontiguousarray(over_seg, dtype=np.uint64)
    rag = bic.graph.region_adjacency_graph(over_seg, number_of_threads=rag_threads)

    # Attractive edges: mean affinity along each RAG edge.
    attr_w = np.ascontiguousarray(
        weights[:number_of_attractive_channels].astype(np.float32, copy=False)
    )
    attr_offsets = offsets[:number_of_attractive_channels]
    attr_features = bic.graph.affinity_features(
        rag, over_seg, attr_w, attr_offsets, number_of_threads=rag_threads
    )
    edge_costs = np.ascontiguousarray(attr_features[:, 0], dtype=np.float32)

    # Mutex (long-range) edges discovered from the labelling at long-range
    # offsets; cost = mean of inverted-mutex weights over those pairs.
    mutex_offsets = offsets[number_of_attractive_channels:number_of_offsets]
    mutex_w = np.ascontiguousarray(
        weights[number_of_attractive_channels:number_of_offsets].astype(
            np.float32, copy=False
        )
    )
    mutex_uvs = bic.graph.lifted_edges_from_affinities(
        rag, over_seg, mutex_offsets, number_of_threads=rag_threads
    )
    mutex_features = bic.graph.lifted_affinity_features(
        over_seg, mutex_w, mutex_offsets, mutex_uvs, number_of_threads=rag_threads
    )
    # mutex_features columns are (mean, size); use the mean as the edge cost.
    mutex_costs = np.ascontiguousarray(mutex_features[:, 0], dtype=np.float32)

    # Per-(segment, class) semantic costs: emit one entry per class per
    # segment that actually contains pixels (skimage's watershed labels
    # start at 1, so segment 0 is empty in the RAG even though
    # ``rag.number_of_nodes`` includes it).
    present_segments = np.unique(over_seg).astype(np.uint64, copy=False)
    semantic_node_classes_list: list[np.ndarray] = []
    semantic_costs_list: list[np.ndarray] = []
    for c in range(n_classes):
        per_segment = ndi_mean(
            weights[number_of_offsets + c],
            labels=over_seg,
            index=present_segments.astype(np.int64),
        )
        per_segment = np.asarray(per_segment, dtype=np.float32)
        nodes_col = present_segments.reshape(-1, 1)
        class_col = np.full((present_segments.size, 1), c, dtype=np.uint64)
        semantic_node_classes_list.append(np.concatenate([nodes_col, class_col], axis=1))
        semantic_costs_list.append(per_segment)

    semantic_node_classes = np.ascontiguousarray(
        np.concatenate(semantic_node_classes_list, axis=0), dtype=np.uint64
    )
    semantic_costs = np.ascontiguousarray(
        np.concatenate(semantic_costs_list, axis=0), dtype=np.float32
    )
    return rag, edge_costs, mutex_uvs, mutex_costs, semantic_node_classes, semantic_costs


def run_bioimage_cpp(
    rag,
    edge_costs,
    mutex_uvs,
    mutex_costs,
    semantic_node_classes,
    semantic_costs,
) -> tuple[np.ndarray, np.ndarray]:
    import bioimage_cpp as bic

    return bic.graph.semantic_mutex_watershed_clustering(
        rag,
        edge_costs,
        mutex_uvs,
        mutex_costs,
        semantic_node_classes,
        semantic_costs,
    )


def run_affogato_reference(
    n_nodes: int,
    uvs: np.ndarray,
    mutex_uvs: np.ndarray,
    semantic_node_classes: np.ndarray,
    edge_costs: np.ndarray,
    mutex_costs: np.ndarray,
    semantic_costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from affogato.segmentation import compute_semantic_mws_clustering

    return compute_semantic_mws_clustering(
        int(n_nodes),
        np.ascontiguousarray(uvs, dtype=np.uint64),
        np.ascontiguousarray(mutex_uvs, dtype=np.uint64),
        np.ascontiguousarray(semantic_node_classes, dtype=np.uint64),
        np.ascontiguousarray(edge_costs, dtype=np.float32),
        np.ascontiguousarray(mutex_costs, dtype=np.float32),
        np.ascontiguousarray(semantic_costs, dtype=np.float32),
    )


def _canonical_labels(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    _, first_index, inverse = np.unique(
        array, return_index=True, return_inverse=True
    )
    order = np.argsort(first_index)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    return remap[inverse].astype(np.uint64, copy=False)


def compare(
    bic_result: tuple[np.ndarray, np.ndarray],
    aff_result: tuple[np.ndarray, np.ndarray],
) -> dict:
    bic_lab, bic_sem = bic_result
    aff_lab, aff_sem = aff_result
    source, rand_index, variation_of_information = _load_validation_metrics()

    vi_split, vi_merge = variation_of_information(bic_lab, aff_lab)
    are, ri = rand_index(bic_lab, aff_lab)
    partition_equal = bool(
        np.array_equal(_canonical_labels(bic_lab), _canonical_labels(aff_lab))
    )
    semantic_match = float(np.mean(bic_sem == aff_sem))
    return {
        "validation_source": source,
        "vi_split": float(vi_split),
        "vi_merge": float(vi_merge),
        "adapted_rand_error": float(are),
        "rand_index": float(ri),
        "partition_equal": partition_equal,
        "semantic_match_fraction": semantic_match,
        "n_clusters_bic": int(np.unique(bic_lab).size),
        "n_clusters_reference": int(np.unique(aff_lab).size),
    }


def time_interleaved(
    bic_run: Callable[[], tuple[np.ndarray, np.ndarray]],
    aff_run: Callable[[], tuple[np.ndarray, np.ndarray]],
    repeats: int,
):
    bic_result = bic_run()
    aff_result = aff_run()
    bic_timings: list[float] = []
    aff_timings: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            t = perf_counter()
            bic_result = bic_run()
            bic_timings.append(perf_counter() - t)
            t = perf_counter()
            aff_result = aff_run()
            aff_timings.append(perf_counter() - t)
        else:
            t = perf_counter()
            aff_result = aff_run()
            aff_timings.append(perf_counter() - t)
            t = perf_counter()
            bic_result = bic_run()
            bic_timings.append(perf_counter() - t)
    return bic_timings, bic_result, aff_timings, aff_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare bioimage-cpp and affogato semantic-mutex-watershed "
            "clustering on a RAG built from a simple watershed oversegmentation "
            "of the registered semantic-labels volume."
        )
    )
    parser.add_argument("--ndim", type=int, default=2, choices=(2, 3))
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Spatial crop. Default: (448, 448) for ndim=2, (8, 448, 448) for "
            "ndim=3."
        ),
    )
    parser.add_argument(
        "--z-start",
        type=int,
        default=0,
        help="Z offset into the (cropped) volume.",
    )
    parser.add_argument(
        "--seed-spacing",
        type=int,
        default=4,
        help="Grid spacing (in pixels) for the watershed marker grid.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rag-threads",
        type=int,
        default=1,
        help="Thread count for RAG construction and feature accumulation.",
    )
    args = parser.parse_args()

    shape = tuple(args.shape) if args.shape else (
        (448, 448) if args.ndim == 2 else (8, 448, 448)
    )

    inst, sem, offsets, attr = load_volume(args.ndim, args.z_start, shape)
    weights, number_of_offsets = build_weight_volume(
        inst, sem, offsets, attr, seed=args.seed
    )
    weights = weights.astype(np.float32, copy=False)
    n_classes = int(weights.shape[0] - number_of_offsets)
    print(
        f"loaded: weights={weights.shape} dtype={weights.dtype} "
        f"n_classes={n_classes}"
    )

    over_seg = oversegment(weights, attr, args.seed_spacing)
    n_segments = int(over_seg.max()) + 1
    print(
        f"oversegmentation: shape={over_seg.shape} n_segments~{n_segments} "
        f"(seed_spacing={args.seed_spacing})"
    )

    rag, edge_costs, mutex_uvs, mutex_costs, semantic_node_classes, semantic_costs = build_costs(
        over_seg, weights, offsets, attr, number_of_offsets, n_classes,
        rag_threads=args.rag_threads,
    )
    print(
        f"rag: nodes={int(rag.number_of_nodes)} edges={int(rag.number_of_edges)} "
        f"mutex={mutex_uvs.shape[0]} semantic_pairs={semantic_node_classes.shape[0]}"
    )

    uvs_array = np.ascontiguousarray(rag.uv_ids(), dtype=np.uint64)

    bic_timings, bic_result, aff_timings, aff_result = time_interleaved(
        lambda: run_bioimage_cpp(
            rag, edge_costs, mutex_uvs, mutex_costs,
            semantic_node_classes, semantic_costs,
        ),
        lambda: run_affogato_reference(
            int(rag.number_of_nodes), uvs_array, mutex_uvs,
            semantic_node_classes, edge_costs, mutex_costs, semantic_costs,
        ),
        args.repeats,
    )

    metrics = compare(bic_result, aff_result)
    bic_median = median(bic_timings)
    aff_median = median(aff_timings)
    speedup = aff_median / bic_median if bic_median > 0 else float("inf")

    print(f"Semantic mutex watershed clustering comparison (ndim={args.ndim})")
    print(f"validation metrics: {metrics['validation_source']}")
    print(
        f"cluster count (bic / affogato): {metrics['n_clusters_bic']} / "
        f"{metrics['n_clusters_reference']}"
    )
    print(
        "VI split/merge: "
        f"{metrics['vi_split']:.6g} / {metrics['vi_merge']:.6g}"
    )
    print(
        "adapted rand error / rand index: "
        f"{metrics['adapted_rand_error']:.6g} / {metrics['rand_index']:.12g}"
    )
    print(f"partition equal (canonical): {metrics['partition_equal']}")
    print(f"semantic match fraction: {metrics['semantic_match_fraction']:.6f}")
    print(f"bioimage-cpp median runtime: {bic_median:.6f} s")
    print(f"affogato reference median runtime: {aff_median:.6f} s")
    print(f"reference / bioimage-cpp runtime ratio: {speedup:.3f}x")
    if not metrics["partition_equal"]:
        print(
            "  NOTE: bic and affogato may disagree. affogato's "
            "compute_semantic_mws_clustering omits the merge_semantic_labels "
            "call on attractive merges (bioimage-cpp's port fixes this) AND "
            "calls boost::disjoint_sets::link with raw node ids instead of "
            "their roots. Either can shift the partition. Report the metrics."
        )


if __name__ == "__main__":
    main()
