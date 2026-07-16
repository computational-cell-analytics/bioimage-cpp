"""Rigorous serial comparison of blockwise TEASAR against Kimimaro.

This workflow is deliberately development-only. It uses Kimimaro and SciPy as
reference dependencies, fixes every backend to one thread, applies the same
exact lattice merge and minimum-spanning forest to both block implementations,
and can enforce the quality gates recorded in ``SKEL-PARALLEL.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
from statistics import median
import sys
from time import perf_counter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.spatial import cKDTree

import bioimage_cpp as bic

from benchmark_teasar import draw_segment, make_branching_tube
from blockwise_stitching import (
    _assert_matching_interfaces,
    _processing_blocks,
    _unique_coordinates,
)


dist = bic.skeleton.distributed
SPACING = (1.0, 1.0, 1.0)
PARAMETERS = {
    "scale": 1.5,
    "constant": 0.0,
    "pdrf_scale": 100000.0,
    "pdrf_exponent": 4.0,
}


@dataclass(frozen=True)
class Workload:
    name: str
    volume: np.ndarray
    block_shape: tuple[int, int, int]
    spacing: tuple[float, float, float] = SPACING
    timing_gate: bool = True


@dataclass
class BlockResult:
    raw: tuple[np.ndarray, np.ndarray, np.ndarray]
    forest: tuple[np.ndarray, np.ndarray, np.ndarray]
    phases: dict[str, float]
    fragments: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    targets: list[np.ndarray]


def _kimimaro_parameters():
    return {
        "scale": PARAMETERS["scale"],
        "const": PARAMETERS["constant"],
        "pdrf_scale": PARAMETERS["pdrf_scale"],
        "pdrf_exponent": int(PARAMETERS["pdrf_exponent"]),
        "soma_detection_threshold": float("inf"),
        "soma_acceptance_threshold": float("inf"),
        "soma_invalidation_scale": 1.0,
        "soma_invalidation_const": 0.0,
    }


def _kimimaro_call(
    volume, *, spacing, fix_borders, extra_targets_before=()
):
    import kimimaro

    return kimimaro.skeletonize(
        volume,
        teasar_params=_kimimaro_parameters(),
        anisotropy=spacing,
        object_ids=[1],
        dust_threshold=0,
        progress=False,
        fix_branching=True,
        fix_borders=fix_borders,
        fill_holes=False,
        parallel=1,
        extra_targets_before=list(extra_targets_before),
    )


def _physical_to_lattice(vertices, spacing, backend):
    scaled = np.asarray(vertices, dtype=float) / np.asarray(spacing, dtype=float)
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, rtol=0.0, atol=1e-5):
        error = float(np.max(np.abs(scaled - rounded)))
        raise RuntimeError(
            f"{backend} returned vertices off the voxel lattice by {error:.6g}"
        )
    return rounded.astype(np.int64)


def _kimimaro_graph(result, spacing, origin=(0, 0, 0)):
    if 1 not in result:
        return (
            np.empty((0, 3), np.int64),
            np.empty((0, 2), np.uint64),
            np.empty((0,), np.float32),
        )
    skeleton = result[1]
    vertices = _physical_to_lattice(
        skeleton.vertices, spacing, "Kimimaro"
    )
    vertices += np.asarray(origin, dtype=np.int64)
    return (
        vertices,
        np.asarray(skeleton.edges, dtype=np.uint64),
        np.asarray(skeleton.radius, dtype=np.float32),
    )


def run_bioimage_block(volume, block_shape, spacing) -> BlockResult:
    phases = {name: 0.0 for name in ("targets", "teasar", "merge", "forest")}
    fragments = []
    block_targets = []
    per_face = {}
    blocking = None
    for blocking, block_id, block, origin, faces in _processing_blocks(
        volume, block_shape
    ):
        start = perf_counter()
        face_targets = []
        for axis, side in faces:
            targets = dist.block_border_targets(
                block,
                [(axis, side)],
                origin=origin,
                spacing=spacing,
                number_of_threads=1,
            )
            per_face[(block_id, axis, side)] = targets
            face_targets.append(targets)
        targets = _unique_coordinates(face_targets)
        phases["targets"] += perf_counter() - start

        start = perf_counter()
        fragments.append(
            dist.block_teasar(
                block,
                open_faces=faces,
                origin=origin,
                required_targets=targets,
                spacing=spacing,
                number_of_threads=1,
                **PARAMETERS,
            )
        )
        phases["teasar"] += perf_counter() - start
        block_targets.append(targets)
    if blocking is None:
        raise RuntimeError("comparison volume produced no blocks")
    _assert_matching_interfaces(blocking, per_face)

    start = perf_counter()
    raw = dist.merge_block_skeletons(fragments)
    phases["merge"] += perf_counter() - start
    start = perf_counter()
    forest = dist.minimum_spanning_forest(raw, spacing=spacing)
    phases["forest"] += perf_counter() - start
    return BlockResult(raw, forest, phases, fragments, block_targets)


def run_kimimaro_block(volume, block_shape, spacing) -> BlockResult:
    phases = {name: 0.0 for name in ("targets", "teasar", "merge", "forest")}
    fragments = []
    for _, _, block, origin, _ in _processing_blocks(volume, block_shape):
        start = perf_counter()
        fragments.append(
            _kimimaro_graph(
                _kimimaro_call(
                    block, spacing=spacing, fix_borders=True
                ),
                spacing,
                origin=origin,
            )
        )
        phases["teasar"] += perf_counter() - start
    start = perf_counter()
    raw = dist.merge_block_skeletons(fragments)
    phases["merge"] += perf_counter() - start
    start = perf_counter()
    forest = dist.minimum_spanning_forest(raw, spacing=spacing)
    phases["forest"] += perf_counter() - start
    return BlockResult(raw, forest, phases, fragments, [])


def run_bioimage_whole(volume, spacing):
    vertices, edges, radii = bic.skeleton.teasar(
        volume,
        spacing=spacing,
        number_of_threads=1,
        **PARAMETERS,
    )
    lattice_vertices = _physical_to_lattice(
        vertices, spacing, "bioimage-cpp"
    )
    return lattice_vertices, edges, radii


def run_kimimaro_whole(volume, spacing):
    return _kimimaro_graph(
        _kimimaro_call(volume, spacing=spacing, fix_borders=False), spacing
    )


def _number_of_components(number_of_vertices, edges):
    parent = np.arange(number_of_vertices, dtype=np.int64)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for first, second in np.asarray(edges, dtype=np.int64):
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root != second_root:
            parent[second_root] = first_root
    return len({find(node) for node in range(number_of_vertices)})


def graph_stats(graph, spacing):
    vertices, edges, _ = graph
    edge_ids = np.asarray(edges, dtype=np.int64)
    components = _number_of_components(len(vertices), edge_ids)
    degree = np.bincount(edge_ids.ravel(), minlength=len(vertices)) if len(edges) else np.zeros(len(vertices), dtype=np.int64)
    length = 0.0
    if len(edge_ids):
        edge_vectors = (
            vertices[edge_ids[:, 0]] - vertices[edge_ids[:, 1]]
        ) * np.asarray(spacing)
        length = float(np.linalg.norm(edge_vectors, axis=1).sum())
    return {
        "vertices": int(len(vertices)),
        "edges": int(len(edges)),
        "components": int(components),
        "cycle_rank": int(len(edges) - len(vertices) + components),
        "endpoints": int(np.count_nonzero(degree == 1)),
        "branchpoints": int(np.count_nonzero(degree >= 3)),
        "contracted_degree_signature": sorted(
            int(value) for value in degree if value != 2
        ),
        "cable_length": length,
    }


def distance_metrics(first, second, spacing):
    first_vertices = (
        np.asarray(first[0], dtype=float) * np.asarray(spacing)
    )
    second_vertices = (
        np.asarray(second[0], dtype=float) * np.asarray(spacing)
    )
    if len(first_vertices) == 0 or len(second_vertices) == 0:
        return {"median": float("inf"), "p95": float("inf"), "hausdorff": float("inf")}
    first_to_second = cKDTree(second_vertices).query(first_vertices)[0]
    second_to_first = cKDTree(first_vertices).query(second_vertices)[0]
    both = np.concatenate([first_to_second, second_to_first])
    return {
        "median": float(np.median(both)),
        "p95": float(np.quantile(both, 0.95)),
        "hausdorff": float(np.max(both)),
        "first_exact_fraction": float(np.mean(first_to_second == 0)),
        "second_exact_fraction": float(np.mean(second_to_first == 0)),
    }


def critical_point_distances(first, second, predicate, spacing):
    def points(graph):
        vertices, edges, _ = graph
        degree = np.bincount(
            np.asarray(edges, dtype=np.int64).ravel(), minlength=len(vertices)
        ) if len(edges) else np.zeros(len(vertices), dtype=np.int64)
        return (
            np.asarray(vertices, dtype=float)[predicate(degree)]
            * np.asarray(spacing)
        )

    first_points = points(first)
    second_points = points(second)
    if len(first_points) == 0 or len(second_points) == 0:
        return {"p95": None, "hausdorff": None}
    distances = np.concatenate([
        cKDTree(second_points).query(first_points)[0],
        cKDTree(first_points).query(second_points)[0],
    ])
    return {
        "p95": float(np.quantile(distances, 0.95)),
        "hausdorff": float(np.max(distances)),
    }


def radius_metrics(first, second):
    first_map = {
        tuple(coordinate): float(radius)
        for coordinate, radius in zip(first[0], first[2])
    }
    second_map = {
        tuple(coordinate): float(radius)
        for coordinate, radius in zip(second[0], second[2])
    }
    shared = sorted(first_map.keys() & second_map.keys())
    errors = np.asarray([
        abs(first_map[coordinate] - second_map[coordinate])
        for coordinate in shared
    ])
    return {
        "shared_vertices": len(shared),
        "mean_absolute_error": float(errors.mean()) if len(errors) else None,
        "maximum_absolute_error": float(errors.max()) if len(errors) else None,
    }


def anchor_radius_errors(volume, block_shape, spacing, bio_result):
    errors = []
    compared = 0
    expected = 0
    for (_, _, block, origin, _), fragment, targets in zip(
        _processing_blocks(volume, block_shape),
        bio_result.fragments,
        bio_result.targets,
    ):
        if len(targets) == 0:
            continue
        expected += len(targets)
        local_targets = targets - np.asarray(origin, dtype=np.int64)
        reference = _kimimaro_graph(
            _kimimaro_call(
                block,
                spacing=spacing,
                fix_borders=False,
                extra_targets_before=local_targets.tolist(),
            ),
            spacing,
            origin=origin,
        )
        bio_radii = {
            tuple(coordinate): float(radius)
            for coordinate, radius in zip(fragment[0], fragment[2])
        }
        reference_radii = {
            tuple(coordinate): float(radius)
            for coordinate, radius in zip(reference[0], reference[2])
        }
        for target in targets:
            coordinate = tuple(target)
            if coordinate not in reference_radii or coordinate not in bio_radii:
                continue
            compared += 1
            errors.append(abs(bio_radii[coordinate] - reference_radii[coordinate]))
    return {
        "compared": compared,
        "expected": expected,
        "maximum_absolute_error": max(errors, default=None),
        "mean_absolute_error": float(np.mean(errors)) if errors else None,
    }


def _canonical_graph(graph):
    vertices, edges, radii = graph
    radius = {tuple(v): float(r) for v, r in zip(vertices, radii)}
    coordinate_edges = set()
    for first, second in np.asarray(edges, dtype=np.int64):
        uv = sorted((tuple(vertices[first]), tuple(vertices[second])))
        coordinate_edges.add(tuple(uv))
    return radius, coordinate_edges


def graphs_exact(first, second):
    first_radius, first_edges = _canonical_graph(first)
    second_radius, second_edges = _canonical_graph(second)
    return first_radius == second_radius and first_edges == second_edges


def timed(backends, warmup, repeats):
    results = {}
    for _ in range(warmup):
        for name, function in backends.items():
            results[name] = function()
    samples = {name: [] for name in backends}
    rng = random.Random(20260716)
    gc.disable()
    try:
        for _ in range(repeats):
            names = list(backends)
            rng.shuffle(names)
            for name in names:
                start = perf_counter()
                results[name] = backends[name]()
                samples[name].append(perf_counter() - start)
    finally:
        gc.enable()
    return results, samples


def workloads(quick):
    line = np.zeros((17, 17, 25), dtype=np.uint8)
    line[8, 8, 2:23] = 1
    output = [
        Workload("thin-line", line, (9, 9, 8), timing_gate=False)
    ]
    oblique = np.zeros((64, 64, 64), dtype=np.uint8)
    draw_segment(oblique, (7, 12, 5), (56, 40, 50), 1)
    output.append(
        Workload(
            "oblique-anisotropic-64-8",
            oblique,
            (32, 32, 32),
            spacing=(1.5, 1.0, 1.0),
        )
    )
    specs = [
        ("tube-64-8", 64, 3, (32, 32, 32)),
        ("tube-96-8", 96, 4, (48, 48, 48)),
    ]
    if not quick:
        specs.extend([
            ("tube-128-aligned-8", 128, 5, (64, 64, 64)),
            ("tube-128-27", 128, 5, (48, 48, 48)),
        ])
    output.extend(
        Workload(name, make_branching_tube(size, radius), block_shape)
        for name, size, radius, block_shape in specs
    )
    return output


def compare_workload(workload, warmup, repeats):
    backends = {
        "bioimage_whole": lambda: run_bioimage_whole(
            workload.volume, workload.spacing
        ),
        "bioimage_block": lambda: run_bioimage_block(
            workload.volume, workload.block_shape, workload.spacing
        ),
        "kimimaro_whole": lambda: run_kimimaro_whole(
            workload.volume, workload.spacing
        ),
        "kimimaro_block": lambda: run_kimimaro_block(
            workload.volume, workload.block_shape, workload.spacing
        ),
    }
    results, samples = timed(backends, warmup, repeats)
    bio_whole = results["bioimage_whole"]
    bio_block_result = results["bioimage_block"]
    kimi_whole = results["kimimaro_whole"]
    kimi_block_result = results["kimimaro_block"]
    bio_block = bio_block_result.forest
    kimi_block = kimi_block_result.forest
    graphs = {
        "bioimage_whole": bio_whole,
        "bioimage_block_raw": bio_block_result.raw,
        "bioimage_block": bio_block,
        "kimimaro_whole": kimi_whole,
        "kimimaro_block_raw": kimi_block_result.raw,
        "kimimaro_block": kimi_block,
    }
    output = {
        "name": workload.name,
        "shape": list(workload.volume.shape),
        "block_shape": list(workload.block_shape),
        "spacing": list(workload.spacing),
        "number_of_blocks": int(np.prod(np.ceil(
            np.asarray(workload.volume.shape) / np.asarray(workload.block_shape)
        ))),
        "timing_gate": workload.timing_gate,
        "timing_seconds": {
            name: {
                "median": median(values),
                "samples": values,
            }
            for name, values in samples.items()
        },
        "bioimage_block_phases_seconds": bio_block_result.phases,
        "kimimaro_block_phases_seconds": kimi_block_result.phases,
        "graphs": {
            name: graph_stats(graph, workload.spacing)
            for name, graph in graphs.items()
        },
        "distances": {
            "bioimage_whole_to_kimimaro_whole": distance_metrics(
                bio_whole, kimi_whole, workload.spacing
            ),
            "bioimage_block_to_kimimaro_block": distance_metrics(
                bio_block, kimi_block, workload.spacing
            ),
            "bioimage_block_to_kimimaro_whole": distance_metrics(
                bio_block, kimi_whole, workload.spacing
            ),
            "kimimaro_block_to_kimimaro_whole": distance_metrics(
                kimi_block, kimi_whole, workload.spacing
            ),
        },
        "critical_points": {
            "endpoints": critical_point_distances(
                bio_block,
                kimi_block,
                lambda degree: degree == 1,
                workload.spacing,
            ),
            "branchpoints": critical_point_distances(
                bio_block,
                kimi_block,
                lambda degree: degree >= 3,
                workload.spacing,
            ),
        },
        "radii": {
            "merged_common_vertices": radius_metrics(bio_block, kimi_block),
            "identical_anchor_coordinates": anchor_radius_errors(
                workload.volume,
                workload.block_shape,
                workload.spacing,
                bio_block_result,
            ),
        },
        "exact_graph": {
            "bioimage_block_to_kimimaro_block": graphs_exact(
                bio_block, kimi_block
            ),
            "bioimage_block_to_kimimaro_whole": graphs_exact(
                bio_block, kimi_whole
            ),
        },
    }
    return output


def gate_failures(result):
    failures = []
    name = result["name"]
    graphs = result["graphs"]
    bio = graphs["bioimage_block"]
    kimi = graphs["kimimaro_block"]
    whole = graphs["kimimaro_whole"]
    if bio["components"] != whole["components"]:
        failures.append(f"{name}: bioimage block component count differs from whole")
    if bio["cycle_rank"] != 0:
        failures.append(f"{name}: final bioimage graph contains cycles")
    if (
        name == "thin-line" and
        not result["exact_graph"]["bioimage_block_to_kimimaro_whole"]
    ):
        failures.append(f"{name}: expected exact graph agreement")
    if (
        name == "oblique-anisotropic-64-8" and
        not result["exact_graph"]["bioimage_block_to_kimimaro_block"]
    ):
        failures.append(f"{name}: expected exact blocked graph agreement")
    anchor_error = result["radii"]["identical_anchor_coordinates"]
    if anchor_error["compared"] != anchor_error["expected"]:
        failures.append(
            f"{name}: compared radii at {anchor_error['compared']}/"
            f"{anchor_error['expected']} interface anchors"
        )
    if anchor_error["compared"] and anchor_error["maximum_absolute_error"] > 1e-5:
        failures.append(
            f"{name}: interface radius error {anchor_error['maximum_absolute_error']:.6g}"
        )
    if name != "thin-line":
        cable_ratio = bio["cable_length"] / kimi["cable_length"]
        if abs(cable_ratio - 1.0) > 0.10:
            failures.append(
                f"{name}: blocked cable ratio to Kimimaro is {cable_ratio:.4f}"
            )
        direct = result["distances"]["bioimage_block_to_kimimaro_block"]
        if direct["p95"] > 2.0 or direct["hausdorff"] > 5.0:
            failures.append(
                f"{name}: direct p95/Hausdorff is "
                f"{direct['p95']:.3f}/{direct['hausdorff']:.3f}"
            )
        bio_whole_error = abs(bio["cable_length"] / whole["cable_length"] - 1.0)
        kimi_whole_error = abs(kimi["cable_length"] / whole["cable_length"] - 1.0)
        if bio_whole_error > kimi_whole_error + 0.10:
            failures.append(f"{name}: cable error exceeds layered reference gate")
        bio_to_whole = result["distances"]["bioimage_block_to_kimimaro_whole"]
        kimi_to_whole = result["distances"]["kimimaro_block_to_kimimaro_whole"]
        if bio_to_whole["p95"] > kimi_to_whole["p95"] + 1.0:
            failures.append(f"{name}: p95 error exceeds layered reference gate")
        bio_endpoint_error = abs(bio["endpoints"] - whole["endpoints"])
        kimi_endpoint_error = abs(kimi["endpoints"] - whole["endpoints"])
        if bio_endpoint_error > kimi_endpoint_error + 2:
            failures.append(f"{name}: endpoint error exceeds layered reference gate")
    if result["timing_gate"]:
        timings = result["timing_seconds"]
        bio_time = timings["bioimage_block"]["median"]
        if bio_time > timings["kimimaro_whole"]["median"]:
            failures.append(f"{name}: blocked bioimage is slower than whole Kimimaro")
        if bio_time > timings["kimimaro_block"]["median"]:
            failures.append(f"{name}: blocked bioimage is slower than blocked Kimimaro")
        phases = result["bioimage_block_phases_seconds"]
        if (phases["merge"] + phases["forest"]) > 0.10 * sum(phases.values()):
            failures.append(f"{name}: merge plus forest exceeds 10% of block time")
    return failures


def print_summary(result):
    timing = result["timing_seconds"]
    graphs = result["graphs"]
    direct = result["distances"]["bioimage_block_to_kimimaro_block"]
    bio = graphs["bioimage_block"]
    kimi = graphs["kimimaro_block"]
    whole = graphs["kimimaro_whole"]
    anchor = result["radii"]["identical_anchor_coordinates"]
    print(
        f"{result['name']}: "
        f"bio-block={1000 * timing['bioimage_block']['median']:.2f} ms, "
        f"kimi-whole={1000 * timing['kimimaro_whole']['median']:.2f} ms, "
        f"kimi-block={1000 * timing['kimimaro_block']['median']:.2f} ms"
    )
    print(
        f"  length bio/kimi-block/whole="
        f"{bio['cable_length']:.2f}/{kimi['cable_length']:.2f}/"
        f"{whole['cable_length']:.2f}; endpoints="
        f"{bio['endpoints']}/{kimi['endpoints']}/{whole['endpoints']}; "
        f"direct p95/H={direct['p95']:.2f}/{direct['hausdorff']:.2f}; "
        f"anchor max radius error={anchor['maximum_absolute_error']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats must be >= 1")

    versions = {}
    for package in ("bioimage-cpp", "kimimaro", "edt", "numpy", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "versions": versions,
        "python": sys.version,
        "default_spacing": SPACING,
        "parameters": PARAMETERS,
        "threads": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "workloads": [],
    }
    failures = []
    for workload in workloads(args.quick):
        result = compare_workload(workload, args.warmup, args.repeats)
        report["workloads"].append(result)
        print_summary(result)
        failures.extend(gate_failures(result))
    report["gate_failures"] = failures
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        print("Quality gate failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        if args.check:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
