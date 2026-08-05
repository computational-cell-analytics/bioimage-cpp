"""Benchmark TEASAR settings on a real binary MRC volume.

The benchmark supports the four bioimage-cpp combinations of invalidation
geometry and branching behavior. It can also run both Kimimaro branching
modes. Timing excludes MRC loading, crop extraction, and graph statistics.

Examples
--------
Run the comparison used by ``TEASAR_OPTIONS_COMPARISON.md``::

    python development/skeleton/benchmark_teasar_mrc.py \
        --crop-size 200 --pixel-size 10 --scale 0 --constant 70 \
        --ball-constant 140 --all-settings --kimimaro \
        --threads 1 8 --repeats 7 --warmup 2 \
        --json /tmp/teasar_options_mrc.json

Run the original size sweep with the default TEASAR behavior::

    python development/skeleton/benchmark_teasar_mrc.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
from statistics import median
import subprocess
import sys
import time
from time import perf_counter

import mrcfile
import numpy as np

import bioimage_cpp as bic
from bioimage_cpp import _core
from benchmark_teasar import proc_tree_rss_kib

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None


@dataclass(frozen=True)
class Setting:
    name: str
    implementation: str
    invalidation: str
    fix_branching: bool


@dataclass(frozen=True)
class Backend:
    key: str
    setting: Setting
    number_of_threads: int
    constant: float
    function: Callable[[np.ndarray], object]


BIOIMAGE_SETTINGS = {
    "cube-fix": Setting("bioimage-cpp/cube/fix", "bioimage-cpp", "cube", True),
    "cube-parental": Setting(
        "bioimage-cpp/cube/parental", "bioimage-cpp", "cube", False
    ),
    "ball-fix": Setting("bioimage-cpp/ball/fix", "bioimage-cpp", "ball", True),
    "ball-parental": Setting(
        "bioimage-cpp/ball/parental", "bioimage-cpp", "ball", False
    ),
}

KIMIMARO_SETTINGS = (
    Setting("kimimaro/ball/fix", "kimimaro", "ball", True),
    Setting("kimimaro/ball/parental", "kimimaro", "ball", False),
)

MEMORY_SETTINGS = {
    **BIOIMAGE_SETTINGS,
    "kimimaro-fix": KIMIMARO_SETTINGS[0],
    "kimimaro-parental": KIMIMARO_SETTINGS[1],
}


def centered_fraction_crop(shape: tuple[int, ...], fraction: float):
    shape_array = np.asarray(shape, dtype=np.int64)
    crop_shape = np.maximum(1, np.floor(shape_array * fraction).astype(np.int64))
    begin = (shape_array - crop_shape) // 2
    end = begin + crop_shape
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(begin, end))
    return slices, tuple(int(value) for value in begin)


def centered_size_crop(shape: tuple[int, ...], size: int):
    if any(size > extent for extent in shape):
        raise ValueError(f"crop size {size} exceeds source shape {shape}")
    begin = tuple((extent - size) // 2 for extent in shape)
    slices = tuple(slice(lo, lo + size) for lo in begin)
    return slices, begin


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "bioimage_cpp": package_version("bioimage-cpp"),
        "kimimaro": package_version("kimimaro"),
        "edt": package_version("edt"),
        "mrcfile": package_version("mrcfile"),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def exact_result(first, second):
    return all(np.array_equal(a, b) for a, b in zip(first, second))


def kimimaro_parameters(parameters):
    return {
        "scale": parameters["scale"],
        "const": parameters["constant"],
        "pdrf_scale": parameters["pdrf_scale"],
        "pdrf_exponent": int(parameters["pdrf_exponent"]),
        "soma_detection_threshold": float("inf"),
        "soma_acceptance_threshold": float("inf"),
        "soma_invalidation_scale": 1.0,
        "soma_invalidation_const": 0.0,
    }


def kimimaro_call(
    mask,
    spacing,
    parameters,
    fix_branching,
    number_of_threads,
):
    import kimimaro

    return kimimaro.skeletonize(
        mask,
        teasar_params=kimimaro_parameters(parameters),
        anisotropy=spacing,
        object_ids=[1],
        dust_threshold=0,
        progress=False,
        fix_branching=fix_branching,
        fix_borders=False,
        fill_holes=False,
        parallel=number_of_threads,
    )


def bioimage_cpp_call(
    mask,
    spacing,
    parameters,
    invalidation,
    fix_branching,
    number_of_threads,
):
    return bic.skeleton.teasar(
        mask,
        spacing=spacing,
        invalidation=invalidation,
        fix_branching=fix_branching,
        number_of_threads=number_of_threads,
        **parameters,
    )


def setting_constant(setting: Setting, constant: float, ball_constant: float):
    return ball_constant if setting.invalidation == "ball" else constant


def make_backends(args, spacing):
    selected = [BIOIMAGE_SETTINGS[name] for name in args.settings]
    if args.kimimaro:
        selected.extend(KIMIMARO_SETTINGS)

    backends = []
    for setting in selected:
        constant = setting_constant(setting, args.constant, args.ball_constant)
        parameters = {
            "scale": args.scale,
            "constant": constant,
            "pdrf_scale": args.pdrf_scale,
            "pdrf_exponent": args.pdrf_exponent,
        }
        for threads in args.threads:
            key = f"{setting.name}/t{threads}"
            if setting.implementation == "bioimage-cpp":
                function = partial(
                    bioimage_cpp_call,
                    spacing=spacing,
                    parameters=parameters,
                    invalidation=setting.invalidation,
                    fix_branching=setting.fix_branching,
                    number_of_threads=threads,
                )
            else:
                function = partial(
                    kimimaro_call,
                    spacing=spacing,
                    parameters=parameters,
                    fix_branching=setting.fix_branching,
                    number_of_threads=threads,
                )
            backends.append(Backend(key, setting, threads, constant, function))
    return backends


def time_backends(mask, backends, repeats, warmup):
    results = {}
    samples = {backend.key: [] for backend in backends}
    for _ in range(warmup):
        for backend in backends:
            results[backend.key] = backend.function(mask)
    rng = random.Random(20260804)
    for _ in range(repeats):
        order = list(backends)
        rng.shuffle(order)
        for backend in order:
            start = perf_counter()
            results[backend.key] = backend.function(mask)
            samples[backend.key].append(perf_counter() - start)
    return samples, results


def flatten_result(result, implementation):
    if implementation == "bioimage-cpp":
        vertices, edges, _ = result
        return np.asarray(vertices, dtype=np.float64), np.asarray(edges, dtype=np.int64)

    skeletons = result.values() if isinstance(result, dict) else (result,)
    vertices = []
    edges = []
    offset = 0
    for skeleton in skeletons:
        skeleton_vertices = np.asarray(skeleton.vertices, dtype=np.float64)
        skeleton_edges = np.asarray(skeleton.edges, dtype=np.int64).reshape(-1, 2)
        vertices.append(skeleton_vertices)
        edges.append(skeleton_edges + offset)
        offset += len(skeleton_vertices)
    if not vertices:
        return np.empty((0, 3), np.float64), np.empty((0, 2), np.int64)
    return np.concatenate(vertices), np.concatenate(edges)


def build_adjacency(number_of_vertices, edges):
    if not len(edges):
        return (
            np.zeros(number_of_vertices + 1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.zeros(number_of_vertices, dtype=np.int64),
        )
    sources = np.concatenate([edges[:, 0], edges[:, 1]])
    targets = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(sources, kind="stable")
    targets = targets[order]
    degrees = np.bincount(sources, minlength=number_of_vertices)
    offsets = np.zeros(number_of_vertices + 1, dtype=np.int64)
    np.cumsum(degrees, out=offsets[1:])
    return offsets, targets, degrees


def component_count(number_of_vertices, edges):
    parents = np.arange(number_of_vertices, dtype=np.int64)

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = int(parents[node])
        return node

    for first, second in edges:
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root != second_root:
            parents[second_root] = first_root
    return len({find(node) for node in range(number_of_vertices)})


def walk_arm(node, first, offsets, targets, degrees, vertices):
    previous = node
    current = int(first)
    length = float(np.linalg.norm(vertices[current] - vertices[node]))
    while degrees[current] == 2:
        neighbors = targets[offsets[current]:offsets[current + 1]]
        next_node = int(neighbors[0])
        if next_node == previous:
            next_node = int(neighbors[1])
        length += float(np.linalg.norm(vertices[next_node] - vertices[current]))
        previous, current = current, next_node
    return current, length


def graph_statistics(vertices, edges, spur_length):
    offsets, targets, degrees = build_adjacency(len(vertices), edges)
    if len(edges):
        segment_lengths = np.linalg.norm(
            vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
        )
    else:
        segment_lengths = np.empty(0, dtype=np.float64)
    degree_three = np.flatnonzero(degrees == 3)
    shortest_arms = []
    spurs = 0
    real_junctions = 0
    for node in degree_three:
        arms = [
            walk_arm(
                int(node), targets[index], offsets, targets, degrees, vertices
            )
            for index in range(offsets[node], offsets[node + 1])
        ]
        lengths = [length for _, length in arms]
        shortest_arms.append(min(lengths))
        if min(lengths) > spur_length:
            real_junctions += 1
        if any(
            degrees[end] == 1 and length <= spur_length
            for end, length in arms
        ):
            spurs += 1

    shortest_arms = np.asarray(shortest_arms, dtype=np.float64)
    output = {
        "vertices": len(vertices),
        "edges": len(edges),
        "length_physical": float(segment_lengths.sum()),
        "components": component_count(len(vertices), edges),
        "degree_1": int(np.count_nonzero(degrees == 1)),
        "degree_3": len(degree_three),
        "degree_4": int(np.count_nonzero(degrees == 4)),
        "spurs": spurs,
        "real_junctions": real_junctions,
    }
    for percentile in (25, 50, 75, 90):
        output[f"shortest_arm_p{percentile}"] = (
            float(np.percentile(shortest_arms, percentile))
            if shortest_arms.size
            else None
        )
    output["spur_percent"] = (
        100.0 * spurs / len(degree_three) if len(degree_three) else None
    )
    output["real_junction_percent"] = (
        100.0 * real_junctions / len(degree_three)
        if len(degree_three)
        else None
    )
    return output


def crop_specs(args, source_shape):
    specs = []
    if args.crop_size is not None:
        slices, origin = centered_size_crop(source_shape, args.crop_size)
        specs.append((f"crop-{args.crop_size}", None, slices, origin))
        if args.include_full:
            slices, origin = centered_fraction_crop(source_shape, 1.0)
            specs.append(("fraction-1.000", 1.0, slices, origin))
    else:
        fractions = set(args.fractions)
        if args.include_full:
            fractions.add(1.0)
        for fraction in sorted(fractions):
            slices, origin = centered_fraction_crop(source_shape, fraction)
            specs.append((f"fraction-{fraction:.3f}", fraction, slices, origin))
    return specs


def memory_worker(args):
    if resource is None:
        raise RuntimeError("memory worker requires the Unix resource module")
    setting = MEMORY_SETTINGS[args.memory_setting]
    with mrcfile.mmap(args.input, mode="r", permissive=True) as mrc:
        source_shape = tuple(int(value) for value in mrc.data.shape)
        specs = crop_specs(args, source_shape)
        _, _, slices, _ = specs[args.memory_crop_index]
        mask = np.array(mrc.data[slices] > 0, dtype=np.uint8, copy=True)

    spacing = (args.pixel_size,) * 3
    constant = setting_constant(setting, args.constant, args.ball_constant)
    parameters = {
        "scale": args.scale,
        "constant": constant,
        "pdrf_scale": args.pdrf_scale,
        "pdrf_exponent": args.pdrf_exponent,
    }
    if setting.implementation == "kimimaro":
        importlib.import_module("kimimaro")
    gc.collect()
    self_peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("READY", flush=True)
    sys.stdin.readline()
    start = perf_counter()
    if setting.implementation == "bioimage-cpp":
        if args.memory_edt_strategy == "auto":
            result = bioimage_cpp_call(
                mask, spacing, parameters, setting.invalidation,
                setting.fix_branching, args.memory_thread,
            )
        else:
            result = _core._teasar_uint8_edt_backend(
                mask,
                spacing,
                parameters["scale"],
                parameters["constant"],
                parameters["pdrf_scale"],
                parameters["pdrf_exponent"],
                setting.invalidation == "ball",
                setting.fix_branching,
                args.memory_edt_strategy,
                256 * 1024 * 1024,
                args.memory_thread,
            )[:3]
        vertices = len(result[0])
        edges = len(result[1])
    else:
        result = kimimaro_call(
            mask, spacing, parameters, setting.fix_branching,
            args.memory_thread,
        )
        skeletons = result.values() if isinstance(result, dict) else (result,)
        vertices = sum(len(skeleton.vertices) for skeleton in skeletons)
        edges = sum(len(skeleton.edges) for skeleton in skeletons)
    print(json.dumps({
        "setting": setting.name,
        "edt_strategy": args.memory_edt_strategy,
        "implementation": setting.implementation,
        "threads": args.memory_thread,
        "elapsed_s": perf_counter() - start,
        "input_nbytes": mask.nbytes,
        "vertices": vertices,
        "edges": edges,
        "self_peak_before_kib": self_peak_before,
        "self_peak_after_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }), flush=True)
    return 0


def run_memory_probe(args, crop_index, crop_name, setting_key, threads, repeat):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--memory-worker",
        "--memory-setting", setting_key,
        "--memory-crop-index", str(crop_index),
        "--memory-thread", str(threads),
        "--memory-edt-strategy", args.memory_edt_strategy,
        "--input", str(args.input),
        "--pixel-size", str(args.pixel_size),
        "--scale", str(args.scale),
        "--constant", str(args.constant),
        "--ball-constant", str(args.ball_constant),
        "--pdrf-scale", str(args.pdrf_scale),
        "--pdrf-exponent", str(args.pdrf_exponent),
    ]
    if args.crop_size is not None:
        command.extend(("--crop-size", str(args.crop_size)))
    else:
        command.append("--fractions")
        command.extend(str(value) for value in args.fractions)
    if args.include_full:
        command.append("--include-full")

    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    ready = process.stdout.readline().strip()
    if ready != "READY":
        _, stderr = process.communicate()
        raise RuntimeError(f"memory worker failed before READY: {stderr}")
    baseline = proc_tree_rss_kib(process.pid)
    peak = baseline
    process.stdin.write("go\n")
    process.stdin.flush()
    while process.poll() is None:
        peak = max(peak, proc_tree_rss_kib(process.pid))
        time.sleep(0.002)
    peak = max(peak, proc_tree_rss_kib(process.pid))
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"memory worker failed: {stderr}")
    payload = json.loads(stdout.strip().splitlines()[-1])
    self_incremental = max(
        0, payload["self_peak_after_kib"] - payload["self_peak_before_kib"]
    )
    peak = max(peak, baseline + self_incremental)
    payload.update({
        "crop": crop_name,
        "repeat": repeat,
        "baseline_process_tree_rss_kib": baseline,
        "peak_process_tree_rss_kib": peak,
        "incremental_peak_kib": max(0, peak - baseline),
    })
    return payload


def run_memory_probes(args, specs):
    if platform.system() != "Linux" or not os.path.isdir("/proc"):
        raise RuntimeError("--memory requires Linux /proc")
    setting_keys = list(args.settings)
    if args.kimimaro:
        setting_keys.extend(("kimimaro-fix", "kimimaro-parental"))
    rows = []
    for crop_index, (crop_name, _, _, _) in enumerate(specs):
        for setting_key in setting_keys:
            for threads in args.memory_threads:
                for repeat in range(args.memory_repeats):
                    rows.append(run_memory_probe(
                        args, crop_index, crop_name, setting_key, threads, repeat
                    ))
    return rows


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "examples" / "skeleton" / "00004_gt_mask.mrc",
    )
    crops = parser.add_mutually_exclusive_group()
    crops.add_argument("--fractions", type=float, nargs="+")
    crops.add_argument("--crop-size", type=int)
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--threads", type=int, nargs="+", default=[8])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--constant", type=float, default=0.0)
    parser.add_argument(
        "--ball-constant",
        type=float,
        help="constant for ball modes; defaults to --constant",
    )
    parser.add_argument("--pdrf-scale", type=float, default=100000.0)
    parser.add_argument("--pdrf-exponent", type=float, default=4.0)
    parser.add_argument("--spur-length", type=float, default=200.0)
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=tuple(BIOIMAGE_SETTINGS),
        help="bioimage-cpp settings; defaults to cube-fix",
    )
    parser.add_argument("--all-settings", action="store_true")
    parser.add_argument("--kimimaro", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--memory-threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--memory-repeats", type=int, default=3)
    parser.add_argument(
        "--memory-edt-strategy", choices=("auto", "local", "shared"),
        default="auto",
    )
    parser.add_argument("--memory-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--memory-setting", choices=tuple(MEMORY_SETTINGS), help=argparse.SUPPRESS
    )
    parser.add_argument("--memory-crop-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--memory-thread", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be >= 1 and --warmup must be >= 0")
    if not args.threads or any(thread < 1 for thread in args.threads):
        parser.error("--threads must contain positive values")
    if not args.memory_threads or any(thread < 1 for thread in args.memory_threads):
        parser.error("--memory-threads must contain positive values")
    if args.memory_repeats < 1:
        parser.error("--memory-repeats must be positive")
    if args.fractions is not None and any(
        not 0.0 < fraction <= 1.0 for fraction in args.fractions
    ):
        parser.error("--fractions values must be in (0, 1]")
    if args.crop_size is not None and args.crop_size < 1:
        parser.error("--crop-size must be positive")
    if args.all_settings and args.settings is not None:
        parser.error("--all-settings and --settings cannot be combined")
    if args.kimimaro and importlib.util.find_spec("kimimaro") is None:
        parser.error("--kimimaro requested, but kimimaro is not installed")
    for name in (
        "pixel_size",
        "pdrf_exponent",
        "spur_length",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("scale", "constant", "pdrf_scale"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.ball_constant is None:
        args.ball_constant = args.constant
    elif not np.isfinite(args.ball_constant) or args.ball_constant < 0.0:
        parser.error("--ball-constant must be finite and non-negative")

    if args.fractions is None and args.crop_size is None:
        args.fractions = [0.125, 0.25, 0.5]
    args.settings = (
        list(BIOIMAGE_SETTINGS)
        if args.all_settings
        else list(dict.fromkeys(args.settings or ["cube-fix"]))
    )
    args.threads = list(dict.fromkeys(args.threads))
    return args


def main() -> int:
    args = parse_args()
    if args.memory_worker:
        return memory_worker(args)
    rows = []
    spacing = (args.pixel_size,) * 3
    backends = make_backends(args, spacing)
    header = (
        f"{'crop':>14} {'backend':>30} {'t':>2} {'const':>7} "
        f"{'shape':>17} {'foreground':>10} {'comp':>5} {'vertices':>8} "
        f"{'deg3':>5} {'spurs':>5} {'real':>5} {'median s':>9} {'min s':>9}"
    )
    print(header)
    print("-" * len(header))
    with mrcfile.mmap(args.input, mode="r", permissive=True) as mrc:
        source = mrc.data
        source_shape = tuple(int(value) for value in source.shape)
        specs = crop_specs(args, source_shape)
        for crop_name, fraction, slices, origin in specs:
            mask = np.array(source[slices] > 0, dtype=np.uint8, copy=True)
            foreground = int(np.count_nonzero(mask))
            samples, results = time_backends(
                mask,
                backends,
                args.repeats,
                args.warmup,
            )

            for setting_name in args.settings:
                setting = BIOIMAGE_SETTINGS[setting_name]
                reference_key = f"{setting.name}/t{args.threads[0]}"
                for threads in args.threads[1:]:
                    candidate_key = f"{setting.name}/t{threads}"
                    if not exact_result(results[reference_key], results[candidate_key]):
                        raise RuntimeError(
                            f"{setting.name}: thread count {threads} changed the skeleton"
                        )

            for backend in backends:
                vertices, edges = flatten_result(
                    results[backend.key], backend.setting.implementation
                )
                statistics = graph_statistics(vertices, edges, args.spur_length)
                samples_s = samples[backend.key]
                row = {
                    "crop": crop_name,
                    "fraction": fraction,
                    "origin": list(origin),
                    "shape": list(mask.shape),
                    "full_voxels": int(mask.size),
                    "foreground_voxels": foreground,
                    "backend": backend.setting.name,
                    "implementation": backend.setting.implementation,
                    "invalidation": backend.setting.invalidation,
                    "fix_branching": backend.setting.fix_branching,
                    "number_of_threads": backend.number_of_threads,
                    "constant": backend.constant,
                    "samples_s": samples_s,
                    "median_s": median(samples_s),
                    "min_s": min(samples_s),
                    **statistics,
                }
                rows.append(row)
                print(
                    f"{crop_name:>14} {backend.setting.name:>30} "
                    f"{backend.number_of_threads:2d} {backend.constant:7.1f} "
                    f"{str(mask.shape):>17} {foreground:10d} "
                    f"{statistics['components']:5d} {statistics['vertices']:8d} "
                    f"{statistics['degree_3']:5d} {statistics['spurs']:5d} "
                    f"{statistics['real_junctions']:5d} "
                    f"{row['median_s']:9.3f} {row['min_s']:9.3f}"
                )
            del results
            del mask

    memory_rows = run_memory_probes(args, specs) if args.memory else []
    if memory_rows:
        print("\npeak process-tree RSS")
        for row in memory_rows:
            print(
                f"  {row['crop']:>14} {row['setting']:>30}/t{row['threads']} "
                f"run {row['repeat'] + 1}: "
                f"{row['incremental_peak_kib'] / 1024:.1f} MiB incremental"
            )

    if args.json is not None:
        payload = {
            "environment": environment(),
            "input": str(args.input.resolve()),
            "source_shape": list(source_shape),
            "crop_size": args.crop_size,
            "fractions": args.fractions,
            "threads": args.threads,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "spacing": list(spacing),
            "parameters": {
                "scale": args.scale,
                "cube_constant": args.constant,
                "ball_constant": args.ball_constant,
                "pdrf_scale": args.pdrf_scale,
                "pdrf_exponent": args.pdrf_exponent,
                "spur_length": args.spur_length,
            },
            "settings": args.settings,
            "kimimaro": args.kimimaro,
            "results": rows,
            "memory": memory_rows,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
