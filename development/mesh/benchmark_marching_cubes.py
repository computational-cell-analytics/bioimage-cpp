"""Benchmark bioimage-cpp marching cubes against scikit-image.

The default ``single`` suite preserves the historical one-size benchmark. The
``scaling`` suite reproduces the independent comparison matrix used to select
the implementation: full Lewiner scaling, representative Lorensen cases, and
optional fresh-process peak-memory measurements.

Examples
--------
python development/mesh/benchmark_marching_cubes.py --size medium
python development/mesh/benchmark_marching_cubes.py --suite scaling --repeats 5 --batches 1
python development/mesh/benchmark_marching_cubes.py --suite scaling --memory --json /tmp/mc.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
from statistics import median
import subprocess
import sys
from time import perf_counter

import numpy as np
import skimage

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None

import bioimage_cpp as bic

from _marching_cubes_reference import assert_mesh_matches, reference_marching_cubes


WORKLOADS = ("binary_sphere", "dense_binary_mask", "scalar_field")
METHODS = ("lewiner", "lorensen")
MEMORY_CASES = (
    ("binary_sphere", 512),
    ("dense_binary_mask", 256),
    ("scalar_field", 512),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("single", "scaling"), default="single")
    parser.add_argument("--size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--workload", choices=(*WORKLOADS, "all"), default="all")
    parser.add_argument("--backend", choices=("both", "bic", "skimage"), default="both")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--memory", action="store_true", help="run fresh-process largest-case RSS probes")
    parser.add_argument("--memory-only", action="store_true", help="run only the fresh-process RSS probes")
    parser.add_argument("--json", default="", help="optional JSON result path")
    parser.add_argument("--baseline", default="", help="optional prior JSON for bic relative changes")
    # Private subprocess interface used by --memory.
    parser.add_argument("--memory-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--memory-backend", choices=("bic", "skimage"), help=argparse.SUPPRESS)
    parser.add_argument("--memory-workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    parser.add_argument("--memory-size", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def shape_for(size: str) -> tuple[int, int, int]:
    return {"small": (48, 48, 48), "medium": (96, 96, 96), "large": (160, 160, 160)}[size]


def make_workload(name: str, n: int, seed: int) -> tuple[np.ndarray, float]:
    shape = (n, n, n)
    if name == "binary_sphere":
        # Build slice-wise so the 512^3 memory probe is not polluted by large
        # temporary broadcast arrays before the measured call.
        y, x = np.ogrid[:n, :n]
        center = np.float32((n - 1) / 2.0)
        radius = np.float32(0.28 * n)
        plane_distance = (y - center) ** 2 + (x - center) ** 2
        volume = np.empty(shape, dtype=np.uint8)
        for z in range(n):
            volume[z] = (
                (np.float32(z) - center) ** 2 + plane_distance <= radius**2
            )
        return volume, 0.5
    if name == "dense_binary_mask":
        rng = np.random.default_rng(seed + n)
        return (
            (rng.random(shape, dtype=np.float32) < np.float32(0.10)).astype(np.uint8),
            0.5,
        )
    if name == "scalar_field":
        q = np.linspace(0.0, 1.0, n, dtype=np.float32)
        z = np.sin(np.float32(4.0 * np.pi) * q)[:, None, None]
        y = np.cos(np.float32(6.0 * np.pi) * q)[None, :, None]
        x = np.sin(np.float32(5.0 * np.pi) * q)[None, None, :]
        return np.asarray(z + y + x, dtype=np.float32, order="C"), 0.0
    raise ValueError(f"unknown workload: {name}")


def case_specs(args: argparse.Namespace) -> list[tuple[str, int, str]]:
    workloads = WORKLOADS if args.workload == "all" else (args.workload,)
    methods = METHODS if args.method == "all" else (args.method,)
    if args.suite == "single":
        n = shape_for(args.size)[0]
        return [(workload, n, method) for workload in workloads for method in methods]

    specs = []
    if "lewiner" in methods:
        for workload in workloads:
            sizes = (64, 128, 192, 256) if workload == "dense_binary_mask" else (64, 128, 256, 512)
            specs.extend((workload, n, "lewiner") for n in sizes)
    if "lorensen" in methods:
        specs.extend((workload, 128, "lorensen") for workload in workloads)
    return specs


def call_bic(volume: np.ndarray, level: float, method: str):
    return bic.mesh.marching_cubes(volume, level, method=method)


def call_skimage(volume: np.ndarray, level: float, method: str):
    return reference_marching_cubes(volume, level, method=method)


def time_once(function, volume, level, method) -> float:
    start = perf_counter()
    result = function(volume, level, method)
    elapsed = perf_counter() - start
    del result
    return elapsed


def statistics(samples: list[float], batch_medians: list[float]) -> dict[str, object]:
    return {
        "raw_s": samples,
        "batch_medians_s": batch_medians,
        "median_s": median(batch_medians),
        "min_s": min(samples),
        "q25_s": float(np.percentile(samples, 25)),
        "q75_s": float(np.percentile(samples, 75)),
        "p10_s": float(np.percentile(samples, 10)),
        "p90_s": float(np.percentile(samples, 90)),
    }


def time_backends(
    volume: np.ndarray,
    level: float,
    method: str,
    backend: str,
    repeats: int,
    warmup: int,
    batches: int,
    seed: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    functions = {}
    if backend in ("both", "bic"):
        functions["bic"] = call_bic
    if backend in ("both", "skimage"):
        functions["skimage"] = call_skimage
    samples = {name: [] for name in functions}
    batch_values = {name: [] for name in functions}
    rng = random.Random(seed)
    gc.collect()
    gc.disable()
    try:
        for batch in range(batches):
            for _ in range(warmup):
                for name, function in functions.items():
                    time_once(function, volume, level, method)
            current = {name: [] for name in functions}
            for _ in range(repeats):
                order = list(functions)
                rng.shuffle(order)
                for name in order:
                    elapsed = time_once(functions[name], volume, level, method)
                    samples[name].append(elapsed)
                    current[name].append(elapsed)
            for name in functions:
                batch_values[name].append(median(current[name]))
    finally:
        gc.enable()
    return (
        None if "bic" not in functions else statistics(samples["bic"], batch_values["bic"]),
        None if "skimage" not in functions else statistics(samples["skimage"], batch_values["skimage"]),
    )


def load_baseline(path: str) -> dict[tuple[str, str, tuple[int, int, int]], float]:
    if not path:
        return {}
    with open(path) as file:
        rows = json.load(file)["results"]
    return {
        (row["workload"], row["method"], tuple(row["shape"])): row["bic_median_s"]
        for row in rows
        if row.get("bic_median_s") is not None
    }


def validate_preflights(specs: list[tuple[str, int, str]], seed: int) -> None:
    combinations = sorted({(workload, method) for workload, _, method in specs})
    for workload, method in combinations:
        volume, level = make_workload(workload, 64, seed)
        actual = call_bic(volume, level, method)
        reference = call_skimage(volume, level, method)
        assert_mesh_matches(actual, reference)


def scaling_exponents(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    combinations = sorted({(row["workload"], row["method"]) for row in rows})
    for workload, method in combinations:
        selected = sorted(
            (row for row in rows if row["workload"] == workload and row["method"] == method),
            key=lambda row: row["voxel_count"],
        )
        if len(selected) < 3:
            continue
        selected = selected[-3:]
        x = np.log([row["voxel_count"] for row in selected])
        entry = {"workload": workload, "method": method, "sizes": [row["shape"][0] for row in selected]}
        for backend, field in (("bic", "bic_median_s"), ("skimage", "reference_median_s")):
            values = [row[field] for row in selected]
            entry[f"{backend}_exponent"] = None if any(value is None for value in values) else float(np.polyfit(x, np.log(values), 1)[0])
        output.append(entry)
    return output


def current_rss_kib() -> int:
    status = "/proc/self/status"
    if not os.path.exists(status):
        return 0
    with open(status) as file:
        for line in file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def memory_worker(args: argparse.Namespace) -> int:
    if resource is None:
        raise RuntimeError("memory worker requires the Unix resource module")
    if args.memory_backend is None or args.memory_workload is None or args.memory_size is None:
        raise SystemExit("memory worker requires backend, workload, and size")
    volume, level = make_workload(args.memory_workload, args.memory_size, args.seed)
    gc.collect()
    rss_before = current_rss_kib()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    function = call_bic if args.memory_backend == "bic" else call_skimage
    start = perf_counter()
    result = function(volume, level, "lewiner")
    elapsed = perf_counter() - start
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "backend": args.memory_backend,
        "workload": args.memory_workload,
        "method": "lewiner",
        "shape": list(volume.shape),
        "elapsed_s": elapsed,
        "input_nbytes": volume.nbytes,
        "output_nbytes": sum(np.asarray(array).nbytes for array in result),
        "vertices": len(result[0]),
        "faces": len(result[1]),
        "rss_before_kib": rss_before,
        "rss_after_kib": current_rss_kib(),
        "peak_before_kib": peak_before,
        "peak_after_kib": peak_after,
        "incremental_peak_kib": max(0, peak_after - peak_before),
    }
    print(json.dumps(payload))
    return 0


def run_memory_probes(args: argparse.Namespace) -> list[dict[str, object]]:
    if platform.system() != "Linux" or resource is None:
        raise RuntimeError("--memory currently requires Linux /proc and ru_maxrss semantics")
    backends = ("bic", "skimage") if args.backend == "both" else (args.backend,)
    rows = []
    script = os.path.abspath(__file__)
    for workload, size in MEMORY_CASES:
        if args.workload != "all" and args.workload != workload:
            continue
        for backend in backends:
            command = [
                sys.executable,
                script,
                "--memory-worker",
                "--memory-backend", backend,
                "--memory-workload", workload,
                "--memory-size", str(size),
                "--seed", str(args.seed),
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            rows.append(json.loads(completed.stdout))
    return rows


def environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_image": skimage.__version__,
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def main() -> int:
    args = parse_args()
    if args.memory_worker:
        return memory_worker(args)
    if args.repeats < 1 or args.warmup < 0 or args.batches < 1:
        raise SystemExit("repeats and batches must be >= 1 and warmup must be >= 0")
    if args.memory_only and not args.memory:
        raise SystemExit("--memory-only requires --memory")
    # Launch memory workers before this process has allocated any large benchmark
    # volume: Linux preserves the parent's RSS high-water mark across fork/exec.
    memory_rows = run_memory_probes(args) if args.memory else []
    if args.memory_only:
        payload = {
            "suite": args.suite,
            "seed": args.seed,
            "backend": args.backend,
            "environment": environment(),
            "results": [],
            "scaling": [],
            "memory": memory_rows,
        }
        for row in memory_rows:
            print(
                f"{row['workload']}/{row['backend']}/{row['shape'][0]}^3: "
                f"peak={row['peak_after_kib'] / 1024:.1f} MiB "
                f"increment={row['incremental_peak_kib'] / 1024:.1f} MiB"
            )
        if args.json:
            with open(args.json, "w") as file:
                json.dump(payload, file, indent=2)
        return 0
    specs = case_specs(args)
    baseline = load_baseline(args.baseline)
    validate_preflights(specs, args.seed)
    print(
        f"suite={args.suite} cases={len(specs)} repeats={args.repeats} "
        f"batches={args.batches} preflight=OK"
    )
    print(
        f"{'workload/method/size':<39} {'V':>10} {'F':>11} {'bic ms':>11} "
        f"{'IQR ms':>9} {'skimage ms':>12} {'speed':>9} {'Mvox/s':>10} {'delta':>9}"
    )
    print("-" * 128)
    rows = []
    for case_index, (workload, size, method) in enumerate(specs):
        volume, level = make_workload(workload, size, args.seed)
        start = perf_counter()
        actual = call_bic(volume, level, method)
        bic_first_call = perf_counter() - start
        start = perf_counter()
        reference = call_skimage(volume, level, method)
        reference_first_call = perf_counter() - start
        counts_match = len(actual[0]) == len(reference[0]) and len(actual[1]) == len(reference[1])
        valid_faces = bool(np.all((actual[1] >= 0) & (actual[1] < len(actual[0]))))
        finite = all(np.all(np.isfinite(array)) for array in (actual[0], actual[2], actual[3]))
        if not counts_match or not valid_faces or not finite:
            raise AssertionError(
                f"large-case validation failed for {workload}/{method}/{size}: "
                f"counts={counts_match}, faces={valid_faces}, finite={finite}"
            )
        bic_times, reference_times = time_backends(
            volume,
            level,
            method,
            args.backend,
            args.repeats,
            args.warmup,
            args.batches,
            args.seed + case_index,
        )
        bic_median = None if bic_times is None else bic_times["median_s"]
        reference_median = None if reference_times is None else reference_times["median_s"]
        baseline_time = baseline.get((workload, method, tuple(volume.shape)))
        relative_change = None if baseline_time is None or bic_median is None else bic_median / baseline_time - 1.0
        speedup = None if bic_median is None or reference_median is None else reference_median / bic_median
        mvox = None if bic_median is None else volume.size / bic_median / 1e6
        row = {
            "workload": workload,
            "method": method,
            "shape": list(volume.shape),
            "voxel_count": int(volume.size),
            "vertices": len(actual[0]),
            "faces": len(actual[1]),
            "counts_match": counts_match,
            "valid_faces": valid_faces,
            "finite_outputs": finite,
            "bic_first_call_s": bic_first_call,
            "reference_first_call_s": reference_first_call,
            "bic": bic_times,
            "skimage": reference_times,
            "bic_median_s": bic_median,
            "reference_median_s": reference_median,
            "speedup": speedup,
            "mvox_per_s": mvox,
            "baseline_bic_median_s": baseline_time,
            "relative_change": relative_change,
        }
        rows.append(row)
        bic_text = "-" if bic_median is None else f"{bic_median * 1e3:.2f}"
        iqr_text = "-" if bic_times is None else f"{(bic_times['q75_s'] - bic_times['q25_s']) * 1e3:.2f}"
        reference_text = "-" if reference_median is None else f"{reference_median * 1e3:.2f}"
        speed_text = "-" if speedup is None else f"{speedup:.2f}x"
        throughput_text = "-" if mvox is None else f"{mvox:.2f}"
        delta_text = "-" if relative_change is None else f"{relative_change * 100:+.1f}%"
        name = f"{workload}/{method}/{size}^3"
        print(
            f"{name:<39} {len(actual[0]):>10} {len(actual[1]):>11} {bic_text:>11} "
            f"{iqr_text:>9} {reference_text:>12} {speed_text:>9} "
            f"{throughput_text:>10} {delta_text:>9}"
        )
        del actual, reference, volume

    exponents = scaling_exponents(rows)
    if exponents:
        print("\nscaling exponents (time ~ voxel_count^p, largest three sizes)")
        for entry in exponents:
            bic_exponent = entry["bic_exponent"]
            reference_exponent = entry["skimage_exponent"]
            bic_text = "-" if bic_exponent is None else f"{bic_exponent:.3f}"
            reference_text = "-" if reference_exponent is None else f"{reference_exponent:.3f}"
            print(
                f"  {entry['workload']}/{entry['method']}: "
                f"bic={bic_text} skimage={reference_text}"
            )
    if memory_rows:
        print("\npeak memory")
        for row in memory_rows:
            print(
                f"  {row['workload']}/{row['backend']}/{row['shape'][0]}^3: "
                f"peak={row['peak_after_kib'] / 1024:.1f} MiB "
                f"increment={row['incremental_peak_kib'] / 1024:.1f} MiB"
            )
    payload = {
        "suite": args.suite,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "batches": args.batches,
        "seed": args.seed,
        "backend": args.backend,
        "environment": environment(),
        "results": rows,
        "scaling": exponents,
        "memory": memory_rows,
    }
    if args.json:
        with open(args.json, "w") as file:
            json.dump(payload, file, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
