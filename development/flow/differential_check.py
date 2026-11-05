"""Randomized differential check of two prebuilt ``_core`` modules.

Both sides regenerate the same deterministic list of small cases (2D/3D,
length-1 axes, tiny and medium grids, several flow and mask families, Euler
and RK2, tol/dt/n_iter variants, mask restriction on/off, 1 and 4 threads),
compute densities and save them; the parent compares them case by case.

Gates (accuracy bar, not bitwise identity): the density sum (particle count)
must be equal in every case, and the fraction of differing voxels must stay
below ``--max-diff-frac`` (default 1 %; the generator includes chaotic
scale-50 random flows where FMA-vs-non-FMA contraction moved 0.6 % of the
voxels by up to two counts). The report also states how many cases were
bitwise identical and the worst max |delta|.

Examples::

    python development/flow/differential_check.py --a base.so --b cand.so --n-cases 400
    python development/flow/differential_check.py --a base.so --b base.so \
        --env-b BIOIMAGE_CPP_FLOW_FORCE_SCALAR=1        # FMA vs scalar path
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from _flow_cases import pin_cpus, preload_core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--env-a", action="append", default=[])
    parser.add_argument("--env-b", action="append", default=[])
    parser.add_argument("--n-cases", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-diff-frac", type=float, default=1e-2,
                        help="max fraction of differing voxels per case (default 1%%)")
    parser.add_argument("--cpu", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--so", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


SHAPES_2D = [(1, 7), (5, 1), (1, 1), (2, 2), (3, 4), (17, 9), (64, 64), (40, 96)]
SHAPES_3D = [(1, 4, 6), (3, 1, 5), (2, 3, 1), (1, 1, 3), (2, 2, 2), (3, 4, 5), (12, 40, 40), (8, 32, 32)]
FLOW_KINDS = ("zero", "int", "half", "normal0.05", "normal0.5", "normal5", "normal50", "swirl")
MASK_KINDS = ("ones", "p0.3", "sparse", "zeros")


def make_case(index: int, seed: int) -> dict:
    rng = np.random.default_rng(seed * 100003 + index)
    ndim = int(rng.integers(2, 4))
    shape = tuple(SHAPES_2D[int(rng.integers(len(SHAPES_2D)))] if ndim == 2
                  else SHAPES_3D[int(rng.integers(len(SHAPES_3D)))])
    flow_kind = FLOW_KINDS[int(rng.integers(len(FLOW_KINDS)))]
    mask_kind = MASK_KINDS[int(rng.integers(len(MASK_KINDS)))]
    return {
        "index": index,
        "ndim": ndim,
        "shape": shape,
        "flow": flow_kind,
        "mask": mask_kind,
        "method": ("euler", "rk2")[int(rng.integers(2))],
        "tol": (0.0, 0.005, 0.3)[int(rng.integers(3))],
        "dt": (0.0, 0.05, 0.2, 1.0)[int(rng.integers(4))],
        "n_iter": (0, 1, 7, 50)[int(rng.integers(4))],
        "restrict_to_mask": bool(rng.integers(2)),
        "threads": (1, 4)[int(rng.integers(2))],
    }


def build_arrays(case: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed * 7919 + case["index"] + 1)
    ndim, shape = case["ndim"], tuple(case["shape"])
    full = (ndim,) + shape
    kind = case["flow"]
    if kind == "zero":
        flow = np.zeros(full, np.float32)
    elif kind == "int":
        flow = rng.integers(-3, 4, size=full).astype(np.float32)
    elif kind == "half":
        flow = rng.integers(-3, 4, size=full).astype(np.float32) + 0.5
    elif kind.startswith("normal"):
        flow = rng.normal(scale=float(kind[len("normal"):]), size=full).astype(np.float32)
    else:  # swirl toward the center with rotation
        grids = np.indices(shape, dtype=np.float32)
        centers = [(s - 1) / 2.0 for s in shape]
        d = [g - c for g, c in zip(grids, centers)]
        r = np.sqrt(sum(x * x for x in d)) + 1e-3
        flow = np.stack([-x / r for x in d]).astype(np.float32)
        flow[-1] += 0.3 * d[-2] / r
        flow[-2] -= 0.3 * d[-1] / r
    mk = case["mask"]
    if mk == "ones":
        mask = np.ones(shape, bool)
    elif mk == "p0.3":
        mask = rng.random(shape) > 0.3
    elif mk == "sparse":
        mask = rng.random(shape) > 0.9
    else:
        mask = np.zeros(shape, bool)
    return np.ascontiguousarray(flow), mask


def worker(args) -> int:
    pin_cpus(args.cpu)
    preload_core(args.so)
    import bioimage_cpp as bic

    densities = {}
    meta = []
    for index in range(args.n_cases):
        case = make_case(index, args.seed)
        flow, mask = build_arrays(case, args.seed)
        density = bic.flow.compute_flow_density(
            flow, mask, n_iter=case["n_iter"], dt=case["dt"], tol=case["tol"], method=case["method"],
            restrict_to_mask=case["restrict_to_mask"], sigma=None, number_of_threads=case["threads"],
        )
        densities[f"c{index}"] = density
        meta.append({**case, "sum": float(density.sum())})
    np.savez(args.out, **densities)
    with open(args.out + ".json", "w") as handle:
        json.dump({"so": sys.modules["bioimage_cpp._core"].__file__, "cases": meta,
                   "backend": getattr(sys.modules["bioimage_cpp._core"], "_flow_trace_backend", lambda: "unknown")()}, handle)
    return 0


def _spawn(args, so: str, env_extra: list[str], out: str) -> None:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--so", so, "--n-cases", str(args.n_cases),
           "--seed", str(args.seed), "--out", out]
    if args.cpu:
        cmd += ["--cpu", args.cpu]
    env = dict(os.environ)
    for item in env_extra:
        key, _, value = item.partition("=")
        env[key] = value
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if res.returncode != 0:
        raise SystemExit(f"worker failed for {so}:\n{res.stderr}")


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)
    if not args.a or not args.b:
        raise SystemExit("--a and --b are required")
    with tempfile.TemporaryDirectory(prefix="flowdiff_", dir=os.environ.get("SCRATCH")) as tmp:
        out_a, out_b = os.path.join(tmp, "a.npz"), os.path.join(tmp, "b.npz")
        _spawn(args, args.a, args.env_a, out_a)
        _spawn(args, args.b, args.env_b, out_b)
        da, db = np.load(out_a), np.load(out_b)
        with open(out_a + ".json") as fa, open(out_b + ".json") as fb:
            ma, mb = json.load(fa), json.load(fb)
    print(f"A = {ma['so']} backend={ma['backend']} {args.env_a or ''}")
    print(f"B = {mb['so']} backend={mb['backend']} {args.env_b or ''}")
    n_identical = 0
    failures = []
    worst_frac, worst_delta = 0.0, 0.0
    rows = []
    for ca, cb in zip(ma["cases"], mb["cases"], strict=True):
        a, b = da[f"c{ca['index']}"], db[f"c{cb['index']}"]
        n_diff = int(np.count_nonzero(a != b))
        frac = n_diff / max(a.size, 1)
        delta = float(np.abs(a - b).max()) if a.size else 0.0
        identical = n_diff == 0
        n_identical += identical
        worst_frac, worst_delta = max(worst_frac, frac), max(worst_delta, delta)
        sums_equal = ca["sum"] == cb["sum"]
        ok = sums_equal and frac <= args.max_diff_frac
        rows.append({**ca, "n_diff": n_diff, "diff_frac": frac, "max_delta": delta, "sums_equal": sums_equal, "ok": ok})
        if not ok:
            failures.append(rows[-1])
    n = len(rows)
    print(f"cases={n} bitwise_identical={n_identical} ({100.0 * n_identical / n:.1f}%) "
          f"worst_diff_frac={worst_frac:.5f} worst_max_delta={worst_delta:g} failures={len(failures)}")
    for row in failures[:10]:
        print("  FAIL", json.dumps({k: row[k] for k in ("index", "ndim", "shape", "flow", "mask", "method", "tol", "dt",
                                                        "n_iter", "restrict_to_mask", "threads", "n_diff",
                                                        "diff_frac", "max_delta", "sums_equal")}))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"a": ma, "b": mb, "rows": rows, "n_identical": n_identical}, handle, indent=1)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
