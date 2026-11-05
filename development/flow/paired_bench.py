"""Paired A/B benchmark of two prebuilt ``_core`` extension modules.

Every measurement runs in a fresh, CPU-pinned subprocess that loads ``_core``
from the requested ``.so`` (see ``_flow_cases.preload_core``), builds the case,
makes one warm call and ``--inner`` timed calls, and reports its min/median.
Per repeat the order is A B B A, so slow drift cancels. The verdict compares
the best-of-all-subprocess minimum of A and B against the per-side noise
(spread of the per-subprocess minima).

Calibrate first with ``--a X.so --b X.so``: every verdict must read ``noise``.

Example::

    python development/flow/paired_bench.py --a base.so --b cand.so \
        --cases fixture3d,fixture2d --threads 1 --cpu 36 --repeats 4 --inner 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from statistics import median

import numpy as np

from _flow_cases import (
    ADVERSARIAL_CASES,
    CASES,
    FIXTURE_CASES,
    add_param_args,
    build_case,
    make_runner,
    params_from_args,
    pin_cpus,
    preload_core,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", help="baseline _core .so")
    parser.add_argument("--b", help="candidate _core .so")
    parser.add_argument("--env-a", action="append", default=[], help="VAR=VALUE for side A")
    parser.add_argument("--env-b", action="append", default=[], help="VAR=VALUE for side B")
    parser.add_argument("--cases", default="fixture3d,fixture2d",
                        help="comma list, or 'fixtures', 'adversarial', 'all'")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu", default=None, help="CPUs for the workers, e.g. 36 or 36,37")
    parser.add_argument("--repeats", type=int, default=4, help="ABBA rounds per case")
    parser.add_argument("--inner", type=int, default=3, help="timed calls per subprocess")
    parser.add_argument("--wrapper", action="store_true", help="time the Python wrapper, not _core")
    parser.add_argument("--json", default=None)
    parser.add_argument("--noise-margin", type=float, default=0.5, help="extra %% added to 2*noise")
    add_param_args(parser)
    # worker mode
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--so", default=None)
    parser.add_argument("--case", default=None)
    parser.add_argument("--params", default="{}")
    return parser.parse_args()


def worker(args) -> int:
    pin_cpus(args.cpu)
    preload_core(args.so)
    import bioimage_cpp as bic

    params = json.loads(args.params)
    flow, mask = build_case(args.case)
    ndim = flow.shape[0]
    bare = not args.wrapper
    run = make_runner(bare, ndim)
    mask_arg = np.ascontiguousarray(mask, dtype=np.uint8) if bare else mask
    import gc

    gc.collect()
    gc.disable()
    run(flow, mask_arg, args.threads, **params)  # warm
    times = []
    result = None
    for _ in range(args.inner):
        start = time.perf_counter()
        result = run(flow, mask_arg, args.threads, **params)
        times.append(time.perf_counter() - start)
    assert result is not None
    print(json.dumps({
        "case": args.case,
        "so": sys.modules["bioimage_cpp._core"].__file__,
        "times": times,
        "min": min(times),
        "median": median(times),
        "particles": float(result.sum()),
        "sha256": hashlib.sha256(result.tobytes()).hexdigest()[:16],
        "backend": getattr(sys.modules["bioimage_cpp._core"], "_flow_trace_backend", lambda: "unknown")(),
    }))
    return 0


def _spawn(args, so: str, env_extra: list[str], case: str, params: dict) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--so", so, "--case", case,
           "--threads", str(args.threads), "--inner", str(args.inner), "--params", json.dumps(params)]
    if args.cpu:
        cmd = ["taskset", "-c", args.cpu] + cmd + ["--cpu", args.cpu]
    if args.wrapper:
        cmd.append("--wrapper")
    env = dict(os.environ)
    for item in env_extra:
        key, _, value = item.partition("=")
        env[key] = value
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if out.returncode != 0:
        raise SystemExit(f"worker failed for {case} ({so}):\n{out.stderr}")
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("{")][-1]
    return json.loads(line)


def _resolve_cases(spec: str) -> list[str]:
    if spec == "fixtures":
        return list(FIXTURE_CASES)
    if spec == "adversarial":
        return list(ADVERSARIAL_CASES)
    if spec == "all":
        return list(FIXTURE_CASES) + list(ADVERSARIAL_CASES)
    names = [c for c in spec.split(",") if c]
    for name in names:
        if name not in CASES:
            raise SystemExit(f"unknown case {name!r}")
    return names


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)
    if not args.a or not args.b:
        raise SystemExit("--a and --b are required")
    params = params_from_args(args)
    cases = _resolve_cases(args.cases)
    sides = {"A": (args.a, args.env_a), "B": (args.b, args.env_b)}
    print(f"A = {args.a} {args.env_a or ''}\nB = {args.b} {args.env_b or ''}")
    print(f"threads={args.threads} cpu={args.cpu} repeats={args.repeats} (ABBA) inner={args.inner} "
          f"{'wrapper' if args.wrapper else 'bare'} params={params}")
    header = (f"{'case':12s} {'thr':>3s} {'A_min':>9s} {'B_min':>9s} {'dMin%':>7s} "
              f"{'A_med':>9s} {'B_med':>9s} {'dMed%':>7s} {'nzA%':>5s} {'nzB%':>5s} par verdict")
    print(header)
    print("-" * len(header))
    report = []
    for case in cases:
        runs: dict[str, list[dict]] = {"A": [], "B": []}
        for _ in range(args.repeats):
            for side in ("A", "B", "B", "A"):
                so, env_extra = sides[side]
                runs[side].append(_spawn(args, so, env_extra, case, params))
        mins = {s: [r["min"] for r in runs[s]] for s in runs}
        meds = {s: [r["median"] for r in runs[s]] for s in runs}
        a_min, b_min = min(mins["A"]), min(mins["B"])
        a_med, b_med = median(meds["A"]), median(meds["B"])
        noise = {s: 100.0 * (max(mins[s]) - min(mins[s])) / min(mins[s]) for s in mins}
        d_min = 100.0 * (b_min - a_min) / a_min
        d_med = 100.0 * (b_med - a_med) / a_med
        shas = {s: {r["sha256"] for r in runs[s]} for s in runs}
        parity = "=" if shas["A"] == shas["B"] and len(shas["A"]) == 1 else ("~" if len(shas["A"]) == 1 and len(shas["B"]) == 1 else "!")
        threshold = 2.0 * max(noise.values()) + args.noise_margin
        verdict = "significant" if abs(d_min) > threshold else "noise"
        row = {"case": case, "threads": args.threads, "a_min": a_min, "b_min": b_min, "d_min_pct": d_min,
               "a_med": a_med, "b_med": b_med, "d_med_pct": d_med, "noise_a_pct": noise["A"],
               "noise_b_pct": noise["B"], "parity": parity, "verdict": verdict,
               "particles_a": runs["A"][0]["particles"], "particles_b": runs["B"][0]["particles"],
               "backend_a": runs["A"][0]["backend"], "backend_b": runs["B"][0]["backend"]}
        report.append(row)
        print(f"{case:12s} {args.threads:3d} {a_min:9.4f} {b_min:9.4f} {d_min:+7.2f} "
              f"{a_med:9.4f} {b_med:9.4f} {d_med:+7.2f} {noise['A']:5.2f} {noise['B']:5.2f}  {parity}  {verdict}",
              flush=True)
    print("parity: '=' identical densities, '~' each side deterministic but different, '!' nondeterministic")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"a": args.a, "b": args.b, "threads": args.threads, "cpu": args.cpu,
                       "repeats": args.repeats, "inner": args.inner, "params": params, "rows": report},
                      handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
