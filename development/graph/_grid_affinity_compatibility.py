from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PREFIX = PROJECT_ROOT / "examples" / "segmentation" / "isbi-data-"


def load_problem(data_prefix: Path | str = DEFAULT_DATA_PREFIX):
    from elf.segmentation.utils import load_mutex_watershed_problem

    affinities, offsets = load_mutex_watershed_problem(prefix=str(data_prefix))
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
    return np.ascontiguousarray(affinities_2d, dtype=np.float32), offsets_2d


def prepare_3d_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    zyx_shape: tuple[int, int, int],
):
    z, y, x = zyx_shape
    cropped = affinities[:, :z, :y, :x]
    return np.ascontiguousarray(cropped, dtype=np.float32), offsets


def select_local_offsets(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
):
    local_channels = [
        index for index, offset in enumerate(offsets)
        if sum(abs(value) for value in offset) == 1
    ]
    local_affinities = np.ascontiguousarray(affinities[local_channels], dtype=np.float32)
    local_offsets = [tuple(offsets[index]) for index in local_channels]
    return local_affinities, local_offsets


def select_mixed_offsets(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
):
    selected = [
        index for index, offset in enumerate(offsets)
        if sum(abs(value) for value in offset) >= 1
    ]
    mixed_affinities = np.ascontiguousarray(affinities[selected], dtype=np.float32)
    mixed_offsets = [tuple(offsets[index]) for index in selected]
    return mixed_affinities, mixed_offsets


def time_call(function: Callable[[], tuple[np.ndarray, np.ndarray]], repeats: int):
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        timings.append(perf_counter() - start)
    assert result is not None
    return timings, result


def sorted_edges_and_weights(uvs: np.ndarray, weights: np.ndarray):
    uvs = np.asarray(uvs, dtype=np.uint64)
    weights = np.asarray(weights, dtype=np.float64)
    if uvs.shape[0] == 0:
        return uvs.reshape(0, 2), weights.reshape(0)
    normalized = np.sort(uvs, axis=1)
    order = np.lexsort((normalized[:, 1], normalized[:, 0]))
    return normalized[order], weights[order]


def split_affogato_edges(uvs: np.ndarray, weights: np.ndarray, graph):
    local_mask = np.asarray(graph.find_edges(uvs), dtype=np.int64) >= 0
    return (
        uvs[local_mask],
        weights[local_mask],
        uvs[~local_mask],
        weights[~local_mask],
    )


def bioimage_cpp_local(affinities: np.ndarray, offsets: list[tuple[int, ...]]):
    import bioimage_cpp as bic

    graph = bic.graph.grid_graph(affinities.shape[1:])
    return bioimage_cpp_local_on_graph(graph, affinities, offsets)


def bioimage_cpp_local_on_graph(graph, affinities: np.ndarray, offsets: list[tuple[int, ...]]):
    import bioimage_cpp as bic

    weights, valid_edges = bic.graph.grid_affinity_features(graph, affinities, offsets)
    if not np.all(valid_edges):
        raise AssertionError("local offsets did not cover all grid edges")
    return graph.uv_ids(), weights


def bioimage_cpp_lifted(affinities: np.ndarray, offsets: list[tuple[int, ...]]):
    import bioimage_cpp as bic

    graph = bic.graph.grid_graph(affinities.shape[1:])
    return bioimage_cpp_lifted_on_graph(graph, affinities, offsets)


def bioimage_cpp_lifted_on_graph(graph, affinities: np.ndarray, offsets: list[tuple[int, ...]]):
    import bioimage_cpp as bic

    local_weights, valid_edges, lifted_uvs, lifted_weights, _ = (
        bic.graph.grid_affinity_features_with_lifted(graph, affinities, offsets)
    )
    if not np.all(valid_edges):
        raise AssertionError("local offsets did not cover all grid edges")
    return graph, graph.uv_ids(), local_weights, lifted_uvs, lifted_weights


def nifty_local(affinities: np.ndarray, offsets: list[tuple[int, ...]]):
    import nifty.graph as ng

    graph = ng.undirectedGridGraph(list(affinities.shape[1:]))
    return nifty_local_on_graph(graph, affinities, offsets)


def nifty_local_on_graph(graph, affinities: np.ndarray, offsets: list[tuple[int, ...]]):
    n_edges, uvs, weights = graph.affinitiesToEdgeMapWithOffsets(
        affinities,
        [list(offset) for offset in offsets],
    )
    return np.asarray(uvs[:n_edges], dtype=np.uint64), np.asarray(weights[:n_edges])


def affogato_edges(affinities: np.ndarray, offsets: list[tuple[int, ...]]):
    from affogato.segmentation import MWSGridGraph

    graph = MWSGridGraph(list(affinities.shape[1:]))
    return affogato_edges_on_graph(graph, affinities, offsets)


def affogato_edges_on_graph(graph, affinities: np.ndarray, offsets: list[tuple[int, ...]]):
    uvs, weights = graph.compute_nh_and_weights(
        affinities,
        [list(offset) for offset in offsets],
        strides=[1] * (affinities.ndim - 1),
        randomize_strides=False,
    )
    return np.asarray(uvs, dtype=np.uint64), np.asarray(weights)


def compare_edge_sets(
    name: str,
    candidate_uvs: np.ndarray,
    candidate_weights: np.ndarray,
    reference_uvs: np.ndarray,
    reference_weights: np.ndarray,
):
    candidate_uvs, candidate_weights = sorted_edges_and_weights(
        candidate_uvs, candidate_weights
    )
    reference_uvs, reference_weights = sorted_edges_and_weights(
        reference_uvs, reference_weights
    )
    np.testing.assert_array_equal(candidate_uvs, reference_uvs)
    np.testing.assert_allclose(candidate_weights, reference_weights, rtol=1.0e-6, atol=1.0e-6)
    max_abs_diff = (
        float(np.max(np.abs(candidate_weights - reference_weights)))
        if candidate_weights.size
        else 0.0
    )
    return {
        "name": name,
        "number_of_edges": int(candidate_uvs.shape[0]),
        "max_abs_weight_diff": max_abs_diff,
    }


def print_timing(name: str, first_name: str, first_timings: list[float],
                 second_name: str, second_timings: list[float]):
    first_median = median(first_timings)
    second_median = median(second_timings)
    ratio = second_median / first_median if first_median > 0 else float("inf")
    print(f"{name} {first_name} median runtime: {first_median:.6f} s")
    print(f"{name} {second_name} median runtime: {second_median:.6f} s")
    print(f"{name} {second_name} / {first_name} runtime ratio: {ratio:.3f}x")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ndim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--data-prefix", type=Path, default=DEFAULT_DATA_PREFIX)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--z", type=int, default=20)
    parser.add_argument("--yx-shape", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--zyx-shape", type=int, nargs=3, default=(16, 512, 512))
