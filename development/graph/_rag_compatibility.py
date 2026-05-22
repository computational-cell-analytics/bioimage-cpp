from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter

import numpy as np
from skimage.feature import peak_local_max
from skimage.measure import label as label_components
from skimage.segmentation import watershed

def load_problem():
    from bioimage_cpp._data import load_isbi_affinities

    affinities, offsets = load_isbi_affinities()
    return np.ascontiguousarray(affinities), [tuple(offset) for offset in offsets]


def prepare_2d_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    z: int,
    yx_shape: tuple[int, int],
):
    channels_2d = [index for index, offset in enumerate(offsets) if offset[0] == 0]
    y, x = yx_shape
    affinities_2d = affinities[channels_2d, z, :y, :x]
    offsets_2d = [offsets[index][1:] for index in channels_2d]
    return _select_direct_affinity_channels(affinities_2d, offsets_2d)


def prepare_3d_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    zyx_shape: tuple[int, int, int],
):
    z, y, x = zyx_shape
    cropped = affinities[:, :z, :y, :x]
    return _select_direct_affinity_channels(cropped, offsets)


def _select_direct_affinity_channels(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
):
    direct_channels = [
        index for index, offset in enumerate(offsets) if sum(abs(v) for v in offset) == 1
    ]
    direct_affinities = np.ascontiguousarray(affinities[direct_channels])
    direct_offsets = [tuple(offsets[index]) for index in direct_channels]
    return direct_affinities, direct_offsets


def heightmap_from_affinities(affinities: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(1.0 - np.mean(affinities, axis=0), dtype=np.float32)


def make_watershed_labels(
    heightmap: np.ndarray,
    *,
    min_distance: int,
    grid_spacing: int,
    max_markers: int,
) -> np.ndarray:
    coordinates = peak_local_max(
        -heightmap,
        min_distance=min_distance,
        exclude_border=False,
        num_peaks=max_markers,
    )
    marker_mask = np.zeros(heightmap.shape, dtype=bool)
    if len(coordinates) > 0:
        marker_mask[tuple(coordinates.T)] = True
    markers = label_components(marker_mask).astype(np.int32, copy=False)

    if int(markers.max()) < 2:
        markers = np.zeros(heightmap.shape, dtype=np.int32)
        slices = tuple(slice(None, None, grid_spacing) for _ in heightmap.shape)
        marker_coordinates = np.argwhere(np.ones(heightmap.shape, dtype=bool)[slices])
        marker_coordinates *= grid_spacing
        for marker_id, coord in enumerate(marker_coordinates, start=1):
            markers[tuple(coord)] = marker_id

    return watershed(heightmap, markers=markers).astype(np.uint32, copy=False)


def time_call(function, repeats: int):
    # One untimed warm-up call before the measured loop so the first sample
    # doesn't carry nanobind tuple-shape caching, numpy ufunc init, code-page
    # faults, etc. Mirrors the grid-affinity helper for consistency.
    function()
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        timings.append(perf_counter() - start)
    assert result is not None
    return timings, result


def sorted_uv_ids(uv_ids: np.ndarray) -> np.ndarray:
    uv_ids = np.asarray(uv_ids, dtype=np.uint64)
    if len(uv_ids) == 0:
        return uv_ids.reshape(0, 2)
    order = np.lexsort((uv_ids[:, 1], uv_ids[:, 0]))
    return uv_ids[order]


def feature_order(source_uv_ids: np.ndarray, target_uv_ids: np.ndarray) -> np.ndarray:
    source_map = {tuple(map(int, uv)): index for index, uv in enumerate(source_uv_ids)}
    return np.array([source_map[tuple(map(int, uv))] for uv in target_uv_ids], dtype=np.int64)


def compare_graphs(bic_rag, nifty_rag) -> dict[str, int]:
    bic_uv = sorted_uv_ids(bic_rag.uv_ids())
    nifty_uv = sorted_uv_ids(nifty_rag.uvIds())

    np.testing.assert_array_equal(bic_uv, nifty_uv)
    if bic_rag.number_of_nodes != nifty_rag.numberOfNodes:
        raise AssertionError(
            "number of graph nodes differs: "
            f"bioimage-cpp={bic_rag.number_of_nodes}, nifty={nifty_rag.numberOfNodes}"
        )
    return {
        "number_of_nodes": int(bic_rag.number_of_nodes),
        "number_of_edges": int(bic_rag.number_of_edges),
    }


def compare_boundary_features(
    bic_rag,
    nifty_rag,
    labels: np.ndarray,
    boundary_map: np.ndarray,
    *,
    threads: int,
    repeats: int,
):
    import bioimage_cpp as bic
    import nifty.graph.rag as nrag

    # Don't force blockShape on the nifty side — its default block layout is
    # what its parallelism is tuned for, and forcing a single block (== labels
    # shape) starves nifty's worker pool.
    bic_timings, bic_features = time_call(
        lambda: bic.graph.features.edge_map_features(
            bic_rag, labels, boundary_map, number_of_threads=threads
        ),
        repeats,
    )
    nifty_timings, nifty_features = time_call(
        lambda: nrag.accumulateEdgeMeanAndLength(
            nifty_rag,
            np.ascontiguousarray(boundary_map, dtype=np.float32),
            numberOfThreads=threads,
        ),
        repeats,
    )

    order = feature_order(np.asarray(bic_rag.uv_ids()), np.asarray(nifty_rag.uvIds()))
    aligned_bic = bic_features[order]
    np.testing.assert_allclose(aligned_bic[:, 0], nifty_features[:, 0], rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(2.0 * aligned_bic[:, 1], nifty_features[:, 1], rtol=1.0e-5, atol=1.0e-6)
    return bic_timings, nifty_timings, {
        "max_mean_abs_diff": float(np.max(np.abs(aligned_bic[:, 0] - nifty_features[:, 0]))),
        "size_convention": "nifty counts both boundary pixels/voxels; bioimage-cpp counts boundary contacts",
    }


def compare_affinity_features(
    bic_rag,
    nifty_rag,
    labels: np.ndarray,
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    *,
    threads: int,
    repeats: int,
):
    import bioimage_cpp as bic
    import nifty.graph.rag as nrag

    offsets_for_nifty = [list(offset) for offset in offsets]
    affs_float32 = np.ascontiguousarray(affinities, dtype=np.float32)
    min_val = float(np.min(affs_float32))
    max_val = float(np.max(affs_float32))
    if min_val == max_val:
        max_val = min_val + 1.0

    bic_timings, bic_features = time_call(
        lambda: bic.graph.features.affinity_features(
            bic_rag,
            labels,
            affs_float32,
            offsets,
            number_of_threads=threads,
        ),
        repeats,
    )
    # nifty's accumulateAffinityStandartFeatures crashes inside vigra's
    # UserRangeHistogram when numberOfThreads=1 (the accumulators never get
    # setMinMax). For threads >= 2 we can pass the value through; for the
    # single-thread case we fall back to nifty's default (-1, all cores) and
    # surface the unfairness in the printed report.
    nifty_threads_effective = threads if threads != 1 else -1
    nifty_timings, nifty_features_full = time_call(
        lambda: nrag.accumulateAffinityStandartFeatures(
            nifty_rag,
            affs_float32,
            offsets_for_nifty,
            min_val,
            max_val,
            numberOfThreads=nifty_threads_effective,
        ),
        repeats,
    )
    nifty_features = nifty_features_full[:, [0, -1]]

    order = feature_order(np.asarray(bic_rag.uv_ids()), np.asarray(nifty_rag.uvIds()))
    aligned_bic = bic_features[order]
    np.testing.assert_allclose(aligned_bic, nifty_features, rtol=1.0e-5, atol=1.0e-6)
    return bic_timings, nifty_timings, {
        "max_abs_diff": float(np.max(np.abs(aligned_bic - nifty_features))),
        "nifty_threads_effective": nifty_threads_effective,
        "nifty_threads_requested": threads,
    }


def run_compatibility_check(
    *,
    ndim: int,
    repeats: int,
    threads: int,
    z: int,
    yx_shape: tuple[int, int],
    zyx_shape: tuple[int, int, int],
    watershed_min_distance: int,
    watershed_grid_spacing: int,
    max_markers: int,
):
    import bioimage_cpp as bic
    import nifty.graph.rag as nrag

    affinities, offsets = load_problem()
    if ndim == 2:
        direct_affinities, direct_offsets = prepare_2d_problem(affinities, offsets, z, yx_shape)
    elif ndim == 3:
        direct_affinities, direct_offsets = prepare_3d_problem(affinities, offsets, zyx_shape)
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    heightmap = heightmap_from_affinities(direct_affinities)
    watershed_timings, labels = time_call(
        lambda: make_watershed_labels(
            heightmap,
            min_distance=watershed_min_distance,
            grid_spacing=watershed_grid_spacing,
            max_markers=max_markers,
        ),
        repeats,
    )
    labels = np.ascontiguousarray(labels, dtype=np.uint32)
    boundary_map = heightmap

    bic_graph_timings, bic_rag = time_call(
        lambda: bic.graph.region_adjacency_graph(labels, number_of_threads=threads),
        repeats,
    )
    nifty_graph_timings, nifty_rag = time_call(
        lambda: nrag.gridRag(labels, numberOfThreads=threads),
        repeats,
    )
    graph_summary = compare_graphs(bic_rag, nifty_rag)

    boundary_bic_timings, boundary_nifty_timings, boundary_summary = compare_boundary_features(
        bic_rag,
        nifty_rag,
        labels,
        boundary_map,
        threads=threads,
        repeats=repeats,
    )
    affinity_bic_timings, affinity_nifty_timings, affinity_summary = compare_affinity_features(
        bic_rag,
        nifty_rag,
        labels,
        direct_affinities,
        direct_offsets,
        threads=threads,
        repeats=repeats,
    )

    print(f"RAG compatibility check ({ndim}D)")
    print(f"labels shape: {labels.shape}, labels: {labels.max()} regions")
    print(f"graph nodes / edges: {graph_summary['number_of_nodes']} / {graph_summary['number_of_edges']}")
    print(f"watershed median runtime: {median(watershed_timings):.6f} s")
    _print_timing("graph creation", bic_graph_timings, nifty_graph_timings)
    _print_timing("boundary-map features", boundary_bic_timings, boundary_nifty_timings)
    print(f"boundary-map max mean abs diff: {boundary_summary['max_mean_abs_diff']:.6g}")
    print(f"boundary-map size convention: {boundary_summary['size_convention']}")
    _print_timing("affinity features", affinity_bic_timings, affinity_nifty_timings)
    print(f"affinity feature max abs diff: {affinity_summary['max_abs_diff']:.6g}")
    requested = affinity_summary["nifty_threads_requested"]
    effective = affinity_summary["nifty_threads_effective"]
    if requested != effective:
        print(
            f"WARNING: affinity features — requested {requested} thread(s) but "
            f"nifty was called with numberOfThreads={effective} "
            "(its single-thread path crashes inside vigra's UserRangeHistogram). "
            f"bioimage-cpp used {requested} thread(s); the timing comparison is NOT apples-to-apples."
        )


def _print_timing(name: str, bic_timings: list[float], nifty_timings: list[float]):
    bic_median = median(bic_timings)
    nifty_median = median(nifty_timings)
    ratio = nifty_median / bic_median if bic_median > 0 else float("inf")
    print(f"{name} bioimage-cpp median runtime: {bic_median:.6f} s")
    print(f"{name} nifty median runtime: {nifty_median:.6f} s")
    print(f"{name} nifty / bioimage-cpp runtime ratio: {ratio:.3f}x")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    # 5 matches the grid-affinity helper; median of 3 is noisy because one
    # GC stall in the middle sample becomes the median.
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--watershed-min-distance", type=int, default=5)
    parser.add_argument("--watershed-grid-spacing", type=int, default=12)
    parser.add_argument("--max-markers", type=int, default=512)
