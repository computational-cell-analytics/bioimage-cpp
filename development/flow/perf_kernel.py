"""Kernel-only timing and hardware-counter harness for compute_flow_density.

Loads one case once, pre-converts the inputs, warms the kernel, then times N
calls. With ``--perf-group`` it attaches ``perf stat -p <pid>`` around only the
timed calls (one attach per event group, five events per group because
``nmi_watchdog=1`` leaves five programmable counters on Zen 3), so process
start-up, HDF5 loading and NumPy conversions are excluded from the counts.

Examples::

    taskset -c 36 python development/flow/perf_kernel.py --case fixture3d --perf-group all
    python development/flow/perf_kernel.py --case fixture3d --overhead
    python development/flow/perf_kernel.py --so /path/to/_core.so --case fixture2d --steps 4200000

``--steps S`` (the executed particle-step count printed by a
``BIOIMAGE_PROFILE=ON`` build) converts totals into per-step budgets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from statistics import median

import numpy as np

from _flow_cases import (
    CASES,
    DEFAULTS,
    add_param_args,
    build_case,
    make_runner,
    params_from_args,
    pin_cpus,
    preload_core,
)

PERF_GROUPS: dict[str, list[str]] = {
    "core": ["cycles", "instructions", "branches", "branch-misses", "ls_dc_accesses"],
    "l1l2": [
        "cycles", "L1-dcache-load-misses", "l2_cache_misses_from_dc_misses",
        "ls_dmnd_fills_from_sys.lcl_l2", "ls_dmnd_fills_from_sys.int_cache",
    ],
    "far": [
        "cycles", "ls_dmnd_fills_from_sys.ext_cache_local",
        "ls_dmnd_fills_from_sys.mem_io_local", "ls_dmnd_fills_from_sys.mem_io_remote",
        "ls_pref_instr_disp",
    ],
    "fp": [
        "cycles", "fp_ret_sse_avx_ops.all", "fp_ret_sse_avx_ops.mac_flops",
        "fp_ret_sse_avx_ops.add_sub_flops", "fp_ret_sse_avx_ops.mult_flops",
    ],
    "tlb": [
        "cycles", "fp_disp_faults.xmm_fill_fault", "ls_l1_d_tlb_miss.all",
        "ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit", "ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit",
    ],
    "stall": [
        "cycles", "stalled-cycles-backend", "stalled-cycles-frontend",
        "de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall",
        "de_dis_dispatch_token_stalls1.load_queue_rsrc_stall",
    ],
    "misal": ["cycles", "ls_misal_loads.ma64", "ls_misal_loads.ma4k", "ls_dc_accesses", "instructions"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="fixture3d", choices=sorted(CASES))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warm", type=int, default=1)
    parser.add_argument("--n-timed", type=int, default=8)
    parser.add_argument("--bare", action="store_true", default=True, help="call _core directly (default)")
    parser.add_argument("--wrapper", action="store_true", help="time the public Python wrapper instead")
    parser.add_argument("--so", default=None, help="load bioimage_cpp._core from this path")
    parser.add_argument("--cpu", default=None, help="pin to these CPUs, e.g. 36 or 36,37")
    parser.add_argument("--perf-group", default=None, help="comma list of groups or 'all'")
    parser.add_argument("--perf-events", default=None, help="explicit comma list of events (one group)")
    parser.add_argument("--perf-out", default=None, help="directory for perf CSV files")
    parser.add_argument("--perf-cpu", default=None, help="CPU to pin the perf process to")
    parser.add_argument("--steps", type=float, default=None, help="executed particle-steps per call")
    parser.add_argument("--json", default=None, help="write results to this JSON file")
    parser.add_argument("--overhead", action="store_true", help="measure wrapper/binding overheads")
    parser.add_argument("--attach-wait", type=float, default=0.0, help="print pid and sleep before timing")
    parser.add_argument("--timeout", type=float, default=60.0)
    add_param_args(parser)
    return parser.parse_args()


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _time_calls(run, flow, mask, threads, params, n: int) -> tuple[list[float], np.ndarray]:
    times = []
    result = None
    for _ in range(n):
        start = time.perf_counter()
        result = run(flow, mask, threads, **params)
        times.append(time.perf_counter() - start)
    return times, result


def _parse_perf_csv(path: str) -> dict[str, float | None]:
    counts: dict[str, float | None] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")
            value, event = fields[0], fields[2] if len(fields) > 2 else fields[-1]
            event = event.removesuffix(":u")
            try:
                counts[event] = float(value)
            except ValueError:
                counts[event] = None  # <not counted> / <not supported>
    return counts


# The fp_ret_sse_avx_ops.* umasks cannot be scheduled as one hardware group on
# this PMU (every member reports <not counted>), so that set is attached
# ungrouped and multiplexed instead.
UNGROUPED = {"fp"}


def _perf_attach(events: list[str], out_csv: str, perf_cpu: str | None, grouped: bool = True) -> subprocess.Popen:
    if grouped:
        group = "{" + ",".join(f"{e}:u" for e in events) + "}"
    else:
        group = ",".join(f"{e}:u" for e in events)
    cmd = ["perf", "stat", "-x", ",", "-o", out_csv, "-p", str(os.getpid()), "-e", group]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if perf_cpu:
        try:
            os.sched_setaffinity(proc.pid, {int(c) for c in perf_cpu.split(",")})
        except OSError:
            pass
    time.sleep(1.0)  # let perf open the counters before the timed calls
    return proc


def _perf_detach(proc: subprocess.Popen) -> str:
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(timeout=30)
    return err.decode(errors="replace")


def _derived(counts: dict[str, float | None], n_calls: int, steps: float | None) -> dict[str, float]:
    out: dict[str, float] = {}
    c = counts.get("cycles")
    i = counts.get("instructions")
    if c and i:
        out["ipc"] = i / c
    loads = counts.get("ls_dc_accesses")
    if loads and i:
        out["loads_per_instr"] = loads / i
    l1m = counts.get("L1-dcache-load-misses")
    if loads and l1m is not None:
        out["l1d_miss_pct"] = 100.0 * l1m / loads
    l2m = counts.get("l2_cache_misses_from_dc_misses")
    if l1m and l2m is not None:
        out["l2_miss_pct_of_l1_misses"] = 100.0 * l2m / l1m
    br = counts.get("branches")
    brm = counts.get("branch-misses")
    if br and brm is not None:
        out["branch_miss_pct"] = 100.0 * brm / br
    if steps:
        denom = n_calls * steps
        for key, value in counts.items():
            if value is not None:
                out[f"{key}/step"] = value / denom
    return out


def run_overhead(args, flow, mask, mask_u8, ndim, params) -> dict:
    import bioimage_cpp as bic

    def best(fn, n=10):
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return min(ts)

    bare = make_runner(True, ndim)
    wrapper = make_runner(False, ndim)
    threads = args.threads
    n_iter0 = {**params, "n_iter": 0}
    results = {
        "wrapper_s": best(lambda: wrapper(flow, mask, threads, **params), 3),
        "bare_s": best(lambda: bare(flow, mask_u8, threads, **params), 3),
        "bare_n_iter0_s": best(lambda: bare(flow, mask_u8, threads, **n_iter0)),
        "np_isfinite_all_s": best(lambda: np.isfinite(flow).all()),
        "mask_astype_u8_s": best(lambda: mask.astype(np.uint8)),
        "flow_ascontiguous_noop_s": best(lambda: np.ascontiguousarray(flow, dtype=np.float32)),
        "flow_bytes": int(flow.nbytes),
    }
    _ = bic
    return results


def main() -> int:
    args = parse_args()
    cpus = pin_cpus(args.cpu)
    preload_core(args.so)
    import bioimage_cpp as bic

    flow, mask = build_case(args.case)
    ndim = flow.shape[0]
    mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
    params = params_from_args(args)
    bare = not args.wrapper
    run = make_runner(bare, ndim)
    mask_arg = mask_u8 if bare else mask

    info = {
        "case": args.case,
        "shape": list(flow.shape[1:]),
        "particles_in_mask": int(mask.sum()),
        "threads": args.threads,
        "cpus": cpus,
        "bare": bare,
        "params": {**DEFAULTS, **params},
        "so": sys.modules["bioimage_cpp._core"].__file__,
        "backend": getattr(sys.modules["bioimage_cpp._core"], "_flow_trace_backend", lambda: "unknown")(),
        "pid": os.getpid(),
    }
    print(json.dumps(info), flush=True)

    if args.overhead:
        results = run_overhead(args, flow, mask, mask_u8, ndim, params)
        info["overhead"] = results
        for key, value in results.items():
            print(f"  {key:28s} {value:.6f}" if isinstance(value, float) else f"  {key:28s} {value}")
        if args.json:
            with open(args.json, "w") as handle:
                json.dump(info, handle, indent=2)
        return 0

    gc.collect()
    gc.disable()
    if args.warm:
        _time_calls(run, flow, mask_arg, args.threads, params, args.warm)
    if args.attach_wait > 0:
        print(f"pid {os.getpid()} sleeping {args.attach_wait}s for manual attach", flush=True)
        time.sleep(args.attach_wait)

    groups: list[tuple[str, list[str]]] = []
    if args.perf_events:
        groups.append(("custom", args.perf_events.split(",")))
    if args.perf_group:
        names = list(PERF_GROUPS) if args.perf_group == "all" else args.perf_group.split(",")
        if args.perf_group == "all":
            names = [n for n in names if n != "misal"]
        groups.extend((n, PERF_GROUPS[n]) for n in names)

    all_times: list[float] = []
    result = None
    counters: dict[str, dict] = {}
    if not groups:
        all_times, result = _time_calls(run, flow, mask_arg, args.threads, params, args.n_timed)
    else:
        out_dir = args.perf_out or "."
        os.makedirs(out_dir, exist_ok=True)
        for name, events in groups:
            csv_path = os.path.join(out_dir, f"{args.case}_{args.threads}T_{name}.csv")
            proc = _perf_attach(events, csv_path, args.perf_cpu, grouped=name not in UNGROUPED)
            times, result = _time_calls(run, flow, mask_arg, args.threads, params, args.n_timed)
            err = _perf_detach(proc)
            all_times.extend(times)
            counts = _parse_perf_csv(csv_path)
            counters[name] = {"counts": counts, "derived": _derived(counts, args.n_timed, args.steps),
                              "wall_min": min(times), "wall_median": median(times)}
            if err.strip() and "Warning" not in err:
                counters[name]["perf_stderr"] = err.strip()[:500]
    gc.enable()

    assert result is not None
    info.update({
        "n_timed": args.n_timed,
        "wall_min": min(all_times),
        "wall_median": median(all_times),
        "particles_out": float(result.sum()),
        "density_sha256": sha256(result),
        "counters": counters,
    })
    print(f"{args.case} threads={args.threads}: min={min(all_times):.4f}s median={median(all_times):.4f}s "
          f"particles={info['particles_out']:.0f} sha={info['density_sha256']}")
    for name, block in counters.items():
        print(f"  [{name}] wall_min={block['wall_min']:.4f}s")
        for event, value in block["counts"].items():
            per_call = "" if value is None else f"  ({value / args.n_timed:,.0f}/call)"
            print(f"    {event:52s} {'<n/a>' if value is None else f'{value:,.0f}'}{per_call}")
        for key, value in block["derived"].items():
            print(f"    -> {key:48s} {value:,.4f}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(info, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
