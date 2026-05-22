"""Cross-check bioimage-cpp's compute_embedding_distances against affogato.

Affogato exposes ``l2`` and ``cosine`` distances over an embedding tensor
``(C, *spatial)`` with a list of spatial offsets. bioimage-cpp additionally
supports ``l1``; that norm is benchmarked here but only compared against a
NumPy reference (affogato has no L1 path).

Benchmarks use synthetic standard-normal float32 embeddings sized to
exercise both 2D and 3D kernels. Not part of the pytest suite.
"""

from __future__ import annotations

import argparse
import sys
from statistics import mean, median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic

try:
    import affogato.affinities as affo
except ImportError as error:  # pragma: no cover - dev script
    sys.stderr.write(f"affogato not installed: {error}\n")
    sys.exit(1)


CASES = [
    # (name, values shape (C, *spatial), offsets)
    (
        "2d_small",
        (12, 256, 256),
        [(-1, 0), (0, -1), (-3, 0), (0, -3), (-9, 0), (0, -9)],
    ),
    (
        "2d_large",
        (16, 512, 512),
        [(-1, 0), (0, -1), (-3, 0), (0, -3), (-9, 0), (0, -9), (-27, 0), (0, -27)],
    ),
    (
        "3d_small",
        (12, 16, 128, 128),
        [(-1, 0, 0), (0, -1, 0), (0, 0, -1),
         (-3, 0, 0), (0, -3, 0), (0, 0, -3)],
    ),
    (
        "3d_large",
        (16, 32, 256, 256),
        [(-1, 0, 0), (0, -1, 0), (0, 0, -1),
         (-3, 0, 0), (0, -3, 0), (0, 0, -3),
         (0, -9, 0), (0, 0, -9)],
    ),
]


def time_call(fn, repeats):
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = fn()
        timings.append(perf_counter() - start)
    return timings, result


def numpy_reference_l1(values, offsets):
    """Vectorized NumPy reference for the L1 norm (no affogato analogue)."""
    spatial = values.shape[1:]
    out = np.zeros((len(offsets),) + spatial, dtype=np.float32)
    for oi, offset in enumerate(offsets):
        src_slices = []
        dst_slices = []
        ok = True
        for d, extent in zip(offset, spatial):
            if d > 0:
                src_slices.append(slice(0, extent - d))
                dst_slices.append(slice(d, extent))
            elif d < 0:
                src_slices.append(slice(-d, extent))
                dst_slices.append(slice(0, extent + d))
            else:
                src_slices.append(slice(0, extent))
                dst_slices.append(slice(0, extent))
            if src_slices[-1].stop <= src_slices[-1].start:
                ok = False
                break
        if not ok:
            continue
        a = values[(slice(None),) + tuple(src_slices)]
        b = values[(slice(None),) + tuple(dst_slices)]
        out_slices = [slice(0, extent) for extent in spatial]
        # Output position corresponds to the "source" position (p, with
        # neighbor p + offset). Negative-offset entries fill from the high
        # end of the spatial axis; positive offsets fill from the low end.
        for axis, d in enumerate(offset):
            extent = spatial[axis]
            if d > 0:
                out_slices[axis] = slice(0, extent - d)
            elif d < 0:
                out_slices[axis] = slice(-d, extent)
            else:
                out_slices[axis] = slice(0, extent)
        out[(oi,) + tuple(out_slices)] = np.sum(np.abs(a - b), axis=0).astype(
            np.float32
        )
    return out


def run_case(name, shape, offsets, repeats, rng):
    values = rng.standard_normal(size=shape).astype(np.float32)
    n_threads = 1
    offsets_list = [list(o) for o in offsets]
    n_voxels = int(np.prod(shape[1:]))

    rows = []
    for norm in ("l1", "l2", "cosine"):
        bic_timings, bic_out = time_call(
            lambda norm=norm: bic.affinities.compute_embedding_distances(
                values,
                offsets_list,
                norm=norm,
                number_of_threads=n_threads,
            ),
            repeats,
        )

        if norm == "l1":
            ref_label = "numpy"
            ref_timings = [float("nan")]
            ref_out = numpy_reference_l1(values, offsets_list)
        else:
            ref_label = "affogato"
            ref_timings, ref_out = time_call(
                lambda norm=norm: affo.compute_embedding_distances(
                    values,
                    offsets_list,
                    norm=norm,
                ),
                repeats,
            )

        if bic_out.shape != ref_out.shape:
            ok = False
            max_abs = float("nan")
        else:
            max_abs = float(np.max(np.abs(bic_out - ref_out)))
            # Allow small floating-point drift from differing accumulation
            # orders. atol scaled by sqrt(C) since errors compound across
            # channels.
            atol = 1e-4 * np.sqrt(shape[0])
            ok = np.allclose(bic_out, ref_out, atol=atol, rtol=1e-4)

        rows.append({
            "case": name,
            "shape": shape,
            "n_offsets": len(offsets_list),
            "n_voxels": n_voxels,
            "norm": norm,
            "ref": ref_label,
            "ok": ok,
            "max_abs_diff": max_abs,
            "bic_median_s": median(bic_timings),
            "bic_mean_s": mean(bic_timings),
            "ref_median_s": median(ref_timings),
            "ref_mean_s": mean(ref_timings),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    all_rows = []
    for name, shape, offsets in CASES:
        all_rows.extend(run_case(name, shape, offsets, args.repeats, rng))

    print()
    header = (
        f"{'case':>10} {'norm':>6} {'ref':>8} {'n_off':>5} {'n_vox':>11}"
        f" {'check':>5} {'max_abs':>10} {'bic_s':>9} {'ref_s':>9} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    all_ok = True
    for r in all_rows:
        if r["ref_median_s"] > 0 and not np.isnan(r["ref_median_s"]):
            speedup = r["ref_median_s"] / r["bic_median_s"]
            speedup_str = f"{speedup:>7.2f}x"
        else:
            speedup_str = f"{'n/a':>8}"
        ref_s_str = (
            f"{r['ref_median_s']:>9.4f}"
            if not np.isnan(r["ref_median_s"])
            else f"{'n/a':>9}"
        )
        print(
            f"{r['case']:>10} {r['norm']:>6} {r['ref']:>8} {r['n_offsets']:>5d}"
            f" {r['n_voxels']:>11,d}"
            f" {'OK' if r['ok'] else 'FAIL':>5}"
            f" {r['max_abs_diff']:>10.2e}"
            f" {r['bic_median_s']:>9.4f}"
            f" {ref_s_str}"
            f" {speedup_str}"
        )
        all_ok = all_ok and r["ok"]

    if not all_ok:
        print("\nFAIL: output mismatch", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
