"""Benchmark binary and multi-label 3D TEASAR against kimimaro.

The binary suite measures ``bioimage_cpp.skeleton.teasar`` against kimimaro's
``object_ids=[1]`` path. The labels suite measures native semantic-label
dispatch, kimimaro's labeled-volume path, and a deliberately naive Python loop
that constructs one full-volume binary mask per label. The paired packed cases
use identical foreground geometry in binary and labeled representations.

Examples
--------
python development/skeleton/benchmark_teasar.py --small --suite all --kimimaro
python development/skeleton/benchmark_teasar.py --large --suite all --kimimaro \
    --threads 1 2 4 8 --repeats 5 --json /tmp/teasar_dispatch.json
python development/skeleton/benchmark_teasar.py --large --suite all --kimimaro \
    --threads 1 4 --memory --memory-threads 1 4 --stress
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
from statistics import median
import subprocess
import sys
import time
from time import perf_counter

import numpy as np

import bioimage_cpp as bic
from bioimage_cpp import _core

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None


@dataclass(frozen=True)
class Workload:
    name: str
    scenario: str
    volume: np.ndarray
    semantic_labels: int
    components: int


def draw_ball(mask: np.ndarray, center: np.ndarray, radius: int) -> None:
    center = np.rint(center).astype(int)
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius + 1, mask.shape)
    slices = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    coords = np.ogrid[tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))]
    squared_distance = sum((coords[d] - center[d]) ** 2 for d in range(3))
    mask[slices] |= squared_distance <= radius**2


def draw_segment(
    mask: np.ndarray, start: tuple[float, ...], stop: tuple[float, ...], radius: int
) -> None:
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    n_steps = int(np.ceil(np.linalg.norm(stop - start))) + 1
    for point in np.linspace(start, stop, n_steps):
        draw_ball(mask, point, radius)


def make_branching_tube(size: int, radius: int) -> np.ndarray:
    mask = np.zeros((size, size, size), dtype=np.uint8)
    c = (size - 1) / 2.0
    margin = max(radius + 3, int(0.10 * size))
    fork = (c, c, 0.54 * size)
    draw_segment(mask, (c, c, margin), fork, radius)
    draw_segment(mask, fork, (c, 0.22 * size, size - margin), radius)
    draw_segment(mask, fork, (0.24 * size, 0.76 * size, size - margin), radius)
    draw_segment(mask, fork, (0.76 * size, 0.76 * size, size - margin), radius)
    return mask


def packed_labels(tile_size: int, grid_size: int, radius: int, same_label: bool):
    gutter = 4
    size = gutter + grid_size * (tile_size + gutter)
    labels = np.zeros((size, size, size), dtype=np.uint32)
    tube = make_branching_tube(tile_size, radius).astype(bool)
    label = 1
    for z in range(grid_size):
        for y in range(grid_size):
            for x in range(grid_size):
                begin = np.array([z, y, x]) * (tile_size + gutter) + gutter
                slices = tuple(slice(int(v), int(v + tile_size)) for v in begin)
                view = labels[slices]
                view[tube] = 7 if same_label else label
                label += 1
    return labels


def make_imbalanced_labels(size: int, tube_size: int, n_filaments: int):
    labels = np.zeros((size, size, size), dtype=np.uint32)
    tube = make_branching_tube(tube_size, max(2, tube_size // 24)).astype(bool)
    labels[:tube_size, :tube_size, :tube_size][tube] = 1
    grid = int(np.ceil(n_filaments ** (1.0 / 3.0)))
    available = size - tube_size - 4
    step = max(4, available // max(1, grid))
    for index in range(n_filaments):
        z = tube_size + 2 + (index // (grid * grid)) * step
        y = 2 + ((index // grid) % grid) * max(4, size // grid)
        x = 2 + (index % grid) * max(4, (size - 6) // grid)
        z = min(z, size - 2)
        y = min(y, size - 2)
        x = min(x, size - 4)
        labels[z, y, x:x + 3] = index + 2
    return labels


def make_dense_ball_labels(tile_size: int = 40):
    gutter = 4
    size = gutter + 2 * (tile_size + gutter)
    labels = np.zeros((size, size, size), dtype=np.uint32)
    ball = np.zeros((tile_size,) * 3, dtype=np.uint8)
    draw_ball(ball, np.full(3, (tile_size - 1) / 2.0), int(0.34 * tile_size))
    label = 1
    for z in range(2):
        for y in range(2):
            for x in range(2):
                begin = np.array([z, y, x]) * (tile_size + gutter) + gutter
                slices = tuple(slice(int(v), int(v + tile_size)) for v in begin)
                view = labels[slices]
                view[ball != 0] = label
                label += 1
    return labels


def tier_parameters(tier: str):
    if tier == "small":
        return (20, 2, 1), (80, 48, 8)
    if tier == "large":
        return (48, 3, 2), (192, 128, 64)
    return (32, 2, 2), (128, 80, 27)


def workloads(tier: str, suite: str, stress: bool) -> list[Workload]:
    output = []
    packed_spec, imbalanced_spec = tier_parameters(tier)
    tile, grid, radius = packed_spec
    distinct = packed_labels(tile, grid, radius, False)
    n_packed = grid**3
    if suite in ("binary", "all"):
        if tier == "small":
            specs = ((64, 3),)
        elif tier == "large":
            specs = ((128, 5), (192, 7), (256, 9))
        else:
            specs = ((96, 4), (128, 5), (192, 7))
        output.extend(
            Workload(
                f"branching-tube-{size}", "binary",
                make_branching_tube(size, tube_radius), 1, 1
            )
            for size, tube_radius in specs
        )
        output.append(Workload(
            "packed-binary", "binary", (distinct != 0).astype(np.uint8),
            1, n_packed
        ))
    if suite in ("labels", "all"):
        output.append(Workload(
            "packed-distinct", "labels", distinct, n_packed, n_packed
        ))
        output.append(Workload(
            "packed-fragmented", "labels",
            packed_labels(tile, grid, radius, True), 1, n_packed
        ))
        size, tube_size, n_filaments = imbalanced_spec
        output.append(Workload(
            "imbalanced", "labels",
            make_imbalanced_labels(size, tube_size, n_filaments),
            n_filaments + 1, n_filaments + 1
        ))
        dense = make_dense_ball_labels(28 if tier == "small" else 40)
        output.append(Workload("dense-balls", "labels", dense, 8, 8))
        if stress:
            checker_size = 24 if tier == "small" else 48
            checker = 1 + (np.indices((checker_size,) * 3).sum(axis=0) & 1)
            output.append(Workload(
                "alternating-labels", "labels",
                checker.astype(np.uint32), 2, 2
            ))
    return output


def time_backends(volume, backends, repeats: int, warmup: int):
    """Time backends in a deterministic shuffled order for every repeat."""
    samples = {name: [] for name, _, _ in backends}
    results = {}
    for _ in range(warmup):
        for name, function, _ in backends:
            results[name] = function(volume)
    rng = random.Random(20260714)
    for _ in range(repeats):
        order = list(backends)
        rng.shuffle(order)
        for name, function, _ in order:
            start = perf_counter()
            results[name] = function(volume)
            samples[name].append(perf_counter() - start)
    return samples, results


def count_bic(result) -> tuple[int, int]:
    vertices, edges, _ = result
    return len(vertices), len(edges)


def count_label_dict(result) -> tuple[int, int]:
    return (
        sum(len(skeleton[0]) for skeleton in result.values()),
        sum(len(skeleton[1]) for skeleton in result.values()),
    )


def count_kimimaro_dict(result) -> tuple[int, int]:
    return (
        sum(len(skeleton.vertices) for skeleton in result.values()),
        sum(len(skeleton.edges) for skeleton in result.values()),
    )


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


def kimimaro_call(volume, scenario, spacing, parameters, number_of_threads=1):
    import kimimaro

    kwargs = {}
    if scenario == "binary":
        kwargs["object_ids"] = [1]
    return kimimaro.skeletonize(
        volume,
        teasar_params=kimimaro_parameters(parameters),
        anisotropy=spacing,
        dust_threshold=0,
        progress=False,
        fix_branching=True,
        fix_borders=False,
        fill_holes=False,
        parallel=number_of_threads,
        **kwargs,
    )


def python_label_loop(volume, spacing, parameters, number_of_threads):
    output = {}
    for label in np.unique(volume):
        if label == 0:
            continue
        output[int(label)] = bic.skeleton.teasar(
            volume == label,
            spacing=spacing,
            number_of_threads=number_of_threads,
            **parameters,
        )
    return output


def bic_backend_call(mask, spacing, parameters, backend, number_of_threads=1):
    """Call a development-only C++ backend without changing the public API."""
    return _core._teasar_uint8_backend(
        mask,
        spacing,
        parameters["scale"],
        parameters["constant"],
        parameters["pdrf_scale"],
        parameters["pdrf_exponent"],
        backend,
        number_of_threads,
    )


def number_of_runs(volume, scenario):
    if volume.shape[2] == 0:
        return 0
    if scenario == "binary":
        foreground = volume != 0
        return int(np.count_nonzero(foreground[..., :1])) + int(
            np.count_nonzero(foreground[..., 1:] & ~foreground[..., :-1])
        )
    non_background = volume != 0
    return int(np.count_nonzero(non_background[..., :1])) + int(np.count_nonzero(
        non_background[..., 1:] & (volume[..., 1:] != volume[..., :-1])
    ))


def exact_bic_result(first, second, scenario):
    if scenario == "binary":
        return all(np.array_equal(a, b) for a, b in zip(first, second))
    if list(first) != list(second):
        return False
    return all(
        np.array_equal(a, b)
        for label in first
        for a, b in zip(first[label], second[label])
    )


def validate_result(workload, backend, result, counts):
    vertices, edges = counts
    if edges != vertices - workload.components:
        raise RuntimeError(
            f"{workload.name}/{backend}: expected E=V-C, got "
            f"V={vertices}, E={edges}, C={workload.components}"
        )
    if workload.scenario == "labels":
        expected = set(int(value) for value in np.unique(workload.volume) if value != 0)
        if set(result) != expected:
            raise RuntimeError(
                f"{workload.name}/{backend}: result keys do not match input labels"
            )


def make_backends(workload, args, spacing, parameters):
    if args.sequential_backends:
        return [
            (
                backend,
                lambda mask, backend=backend: bic_backend_call(
                    mask, spacing, parameters, backend
                ),
                count_bic,
            )
            for backend in (
                "dense-fp64", "compact-on-the-fly-fp64", "compact-csr-fp64"
            )
        ]
    output = []
    for threads in args.threads:
        if workload.scenario == "binary":
            function = lambda volume, threads=threads: bic.skeleton.teasar(
                volume, spacing=spacing, number_of_threads=threads, **parameters
            )
            counter = count_bic
        else:
            function = lambda volume, threads=threads: bic.skeleton.teasar_labels(
                volume, spacing=spacing, number_of_threads=threads, **parameters
            )
            counter = count_label_dict
        output.append((f"bioimage-cpp/t{threads}", function, counter))
    if workload.scenario == "labels" and args.python_loop:
        output.extend(
            (
                f"python-label-loop/t{threads}",
                lambda volume, threads=threads: python_label_loop(
                    volume, spacing, parameters, threads
                ),
                count_label_dict,
            )
            for threads in args.threads
        )
    if args.kimimaro:
        output.extend(
            (
                f"kimimaro/t{threads}",
                lambda volume, threads=threads: kimimaro_call(
                    volume, workload.scenario, spacing, parameters, threads
                ),
                count_kimimaro_dict,
            )
            for threads in args.threads
        )
    return output


def package_version(name):
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
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def proc_tree_rss_kib(root_pid):
    processes = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            values = {}
            with open(f"/proc/{entry}/status", encoding="utf-8") as file:
                for line in file:
                    if line.startswith(("PPid:", "VmRSS:")):
                        key, value = line.split(":", 1)
                        values[key] = int(value.split()[0])
            processes[int(entry)] = (values.get("PPid", -1), values.get("VmRSS", 0))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in processes.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(processes.get(pid, (-1, 0))[1] for pid in descendants)


def memory_worker(args):
    if resource is None:
        raise RuntimeError("memory worker requires the Unix resource module")
    selected = next(
        item for item in workloads(args.tier, "all", True)
        if item.name == args.memory_case and item.scenario == args.memory_scenario
    )
    spacing = (1.5, 1.0, 1.0)
    parameters = {
        "scale": 1.5, "constant": 1.0,
        "pdrf_scale": 100000.0, "pdrf_exponent": 4.0,
    }
    if args.memory_backend == "kimimaro":
        importlib.import_module("kimimaro")  # import before the baseline sample
    gc.collect()
    self_peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("READY", flush=True)
    sys.stdin.readline()
    start = perf_counter()
    if args.memory_backend == "bioimage-cpp":
        if selected.scenario == "binary":
            result = bic.skeleton.teasar(
                selected.volume, spacing=spacing,
                number_of_threads=args.memory_thread, **parameters
            )
            counts = count_bic(result)
        else:
            result = bic.skeleton.teasar_labels(
                selected.volume, spacing=spacing,
                number_of_threads=args.memory_thread, **parameters
            )
            counts = count_label_dict(result)
    else:
        result = kimimaro_call(
            selected.volume, selected.scenario, spacing, parameters,
            args.memory_thread
        )
        counts = count_kimimaro_dict(result)
    payload = {
        "backend": args.memory_backend,
        "scenario": selected.scenario,
        "case": selected.name,
        "threads": args.memory_thread,
        "elapsed_s": perf_counter() - start,
        "input_nbytes": selected.volume.nbytes,
        "vertices": counts[0],
        "edges": counts[1],
        "self_peak_before_kib": self_peak_before,
        "self_peak_after_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    print(json.dumps(payload), flush=True)
    return 0


def run_memory_probe(script, workload, backend, threads, tier):
    command = [
        sys.executable, script,
        "--memory-worker",
        "--memory-backend", backend,
        "--memory-case", workload.name,
        "--memory-scenario", workload.scenario,
        "--memory-thread", str(threads),
        "--tier", tier,
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True
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
        "baseline_process_tree_rss_kib": baseline,
        "peak_process_tree_rss_kib": peak,
        "incremental_peak_kib": max(0, peak - baseline),
    })
    return payload


def run_memory_probes(args, selected_workloads):
    if platform.system() != "Linux" or not os.path.isdir("/proc"):
        raise RuntimeError("--memory requires Linux /proc")
    names = {"packed-distinct", "packed-fragmented", "imbalanced"}
    binary = [item for item in selected_workloads if item.scenario == "binary"]
    selected = [item for item in selected_workloads if item.name in names]
    if binary:
        selected.insert(0, binary[-2] if binary[-1].name == "packed-binary" else binary[-1])
    if args.stress:
        selected.extend(item for item in selected_workloads if item.name == "alternating-labels")
    script = os.path.abspath(__file__)
    rows = []
    for workload in selected:
        for backend in (("bioimage-cpp", "kimimaro") if args.kimimaro else ("bioimage-cpp",)):
            for threads in args.memory_threads:
                rows.append(run_memory_probe(
                    script, workload, backend, threads, args.tier
                ))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--small", action="store_true")
    sizes.add_argument("--large", action="store_true")
    parser.add_argument("--tier", choices=("small", "default", "large"), help=argparse.SUPPRESS)
    parser.add_argument("--suite", choices=("binary", "labels", "all"), default="binary")
    parser.add_argument(
        "--case", action="append", dest="cases",
        help="run only the named case (repeatable)"
    )
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--kimimaro", action="store_true")
    parser.add_argument("--python-loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sequential-backends", action="store_true")
    parser.add_argument("--json", default="", help="optional JSON result path")
    parser.add_argument("--threads", type=int, nargs="+", default=[1])
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--memory-threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--memory-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--memory-backend", choices=("bioimage-cpp", "kimimaro"), help=argparse.SUPPRESS)
    parser.add_argument("--memory-case", help=argparse.SUPPRESS)
    parser.add_argument("--memory-scenario", choices=("binary", "labels"), help=argparse.SUPPRESS)
    parser.add_argument("--memory-thread", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.tier is None:
        args.tier = "small" if args.small else "large" if args.large else "default"
    return args


def main() -> int:
    args = parse_args()
    if args.memory_worker:
        return memory_worker(args)
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats must be >= 1 and --warmup must be >= 0")
    if not args.threads or any(thread < 1 for thread in args.threads):
        raise SystemExit("--threads must contain positive worker counts")
    if args.sequential_backends and args.suite != "binary":
        raise SystemExit("--sequential-backends requires --suite binary")
    if args.kimimaro and importlib.util.find_spec("kimimaro") is None:
        raise SystemExit("--kimimaro requested, but kimimaro is not installed")

    spacing = (1.5, 1.0, 1.0)
    parameters = {
        "scale": 1.5,
        "constant": 1.0,
        "pdrf_scale": 100000.0,
        "pdrf_exponent": 4.0,
    }
    selected_workloads = workloads(args.tier, args.suite, args.stress)
    if args.sequential_backends:
        selected_workloads = [
            item for item in selected_workloads if item.components == 1
        ]
    if args.cases:
        requested = set(args.cases)
        selected_workloads = [
            item for item in selected_workloads if item.name in requested
        ]
        missing = requested - {item.name for item in selected_workloads}
        if missing:
            raise SystemExit("unknown or disabled cases: " + ", ".join(sorted(missing)))
    memory_rows = run_memory_probes(args, selected_workloads) if args.memory else []
    rows = []
    stored_results = {}
    header = (
        f"{'scenario':>8} {'case':>24} {'backend':>25} {'shape':>15} "
        f"{'fg':>10} {'labels':>6} {'comp':>5} {'runs':>9} "
        f"{'vertices':>9} {'median ms':>11} {'min ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for workload in selected_workloads:
        backends = make_backends(workload, args, spacing, parameters)
        samples, results = time_backends(
            workload.volume, backends, args.repeats, args.warmup
        )
        if args.sequential_backends:
            dense = results["dense-fp64"]
            for backend in ("compact-on-the-fly-fp64", "compact-csr-fp64"):
                if not all(
                    np.array_equal(got, expected)
                    for got, expected in zip(results[backend], dense)
                ):
                    raise RuntimeError(
                        f"{workload.name}: {backend} lost exact dense parity"
                    )
        foreground = int(np.count_nonzero(workload.volume))
        runs = number_of_runs(workload.volume, workload.scenario)
        if not args.sequential_backends:
            reference = results[f"bioimage-cpp/t{args.threads[0]}"]
            for threads in args.threads[1:]:
                candidate = results[f"bioimage-cpp/t{threads}"]
                if not exact_bic_result(reference, candidate, workload.scenario):
                    raise RuntimeError(
                        f"{workload.name}: worker count {threads} changed output"
                    )
        for name, _, count_result in backends:
            result = results[name]
            counts = count_result(result)
            validate_result(workload, name, result, counts)
            samples_s = samples[name]
            row = {
                "scenario": workload.scenario,
                "case": workload.name,
                "backend": name,
                "number_of_threads": int(name.rsplit("/t", 1)[1]) if "/t" in name else 1,
                "shape": list(workload.volume.shape),
                "full_voxels": int(workload.volume.size),
                "foreground_voxels": foreground,
                "semantic_labels": workload.semantic_labels,
                "components": workload.components,
                "runs": runs,
                "vertices": counts[0],
                "edges": counts[1],
                "samples_s": samples_s,
                "median_s": median(samples_s),
                "min_s": min(samples_s),
            }
            rows.append(row)
            print(
                f"{workload.scenario:>8} {workload.name:>24} {name:>25} "
                f"{str(workload.volume.shape):>15} {foreground:10d} "
                f"{workload.semantic_labels:6d} {workload.components:5d} "
                f"{runs:9d} {counts[0]:9d} {row['median_s'] * 1e3:11.2f} "
                f"{row['min_s'] * 1e3:10.2f}"
            )
            stored_results[(workload.name, name)] = result

        if args.kimimaro and not args.sequential_backends:
            by_name = {row["backend"]: row for row in rows if row["case"] == workload.name}
            for threads in args.threads:
                bio = by_name[f"bioimage-cpp/t{threads}"]["median_s"]
                kimi = by_name[f"kimimaro/t{threads}"]["median_s"]
                print(f"  bioimage-cpp/t{threads} / kimimaro/t{threads}: {bio / kimi:.2f}x")

    if args.suite == "all" and not args.sequential_backends:
        for threads in args.threads:
            binary = stored_results[("packed-binary", f"bioimage-cpp/t{threads}")]
            labeled = stored_results[("packed-distinct", f"bioimage-cpp/t{threads}")]
            flattened_vertices = []
            flattened_radii = []
            flattened_edges = []
            offset = 0
            for skeleton in labeled.values():
                flattened_vertices.append(skeleton[0])
                flattened_radii.append(skeleton[2])
                flattened_edges.append(skeleton[1] + offset)
                offset += len(skeleton[0])
            flattened = (
                np.concatenate(flattened_vertices),
                np.concatenate(flattened_edges),
                np.concatenate(flattened_radii),
            )
            if not all(np.array_equal(a, b) for a, b in zip(binary, flattened)):
                raise RuntimeError("paired binary and labeled dispatch lost exact parity")
        print("paired packed binary/multi-label exact parity: True")

    payload = {
        "environment": environment(),
        "tier": args.tier,
        "suite": args.suite,
        "stress": args.stress,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "spacing": spacing,
        "parameters": parameters,
        "results": rows,
        "memory": memory_rows,
    }
    if memory_rows:
        print("\npeak process-tree RSS")
        for row in memory_rows:
            print(
                f"  {row['scenario']:>6} {row['case']:>22} "
                f"{row['backend']:>12}/t{row['threads']}: "
                f"{row['incremental_peak_kib'] / 1024:.1f} MiB incremental"
            )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
