"""Benchmark the mutex-watershed grid kernel (bic only).

The code review routed the neighbor computation in ``mutex_watershed_grid``
through ``detail::valid_offset_target`` (per-axis bounds check) instead of a
single precomputed flat-offset add. This script times ``mutex_watershed`` on the
ISBI affinities for 2D and 3D so we can A/B that inner-loop change against a
pre-review build. No external (affogato) dependency — bic only.

Not part of the test suite. Run::

    python development/segmentation/benchmark_mutex_watershed.py --repeats 5
"""
from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter

import numpy as np

import bioimage_cpp as bic
from bioimage_cpp._data import load_isbi_affinities


def _timeit(fn, repeats: int, warmup: int = 1) -> dict:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        timings.append(perf_counter() - t0)
    return {"median": median(timings), "min": min(timings), "n": repeats}


def _attractive_flip(affs: np.ndarray, n_attractive: int) -> np.ndarray:
    # Match the convention in the equivalence checker: attractive channels are
    # turned into merge affinities (1 - aff).
    out = affs.copy()
    out[:n_attractive] *= -1
    out[:n_attractive] += 1
    return out


def run(repeats: int = 5) -> dict:
    affinities, offsets = load_isbi_affinities()
    affinities = np.ascontiguousarray(affinities)
    offsets = [tuple(o) for o in offsets]
    results: dict[str, dict] = {}

    # --- 2D: in-plane channels of a single z slice ---
    channels_2d = [i for i, o in enumerate(offsets) if o[0] == 0]
    aff2d = np.ascontiguousarray(affinities[channels_2d, 0, :256, :256])
    offsets_2d = [offsets[i][1:] for i in channels_2d]
    aff2d_flipped = _attractive_flip(aff2d, 2)

    def mws_2d():
        bic.segmentation.mutex_watershed(
            aff2d_flipped, offsets_2d, number_of_attractive_channels=2
        )

    results["mws_2d_256x256"] = _timeit(mws_2d, repeats)
    results["mws_2d_256x256"]["meta"] = {"shape": list(aff2d.shape)}

    # --- 3D: small crop, all offsets ---
    aff3d = np.ascontiguousarray(affinities[:, :6, :256, :256])
    aff3d_flipped = _attractive_flip(aff3d, 3)

    def mws_3d():
        bic.segmentation.mutex_watershed(
            aff3d_flipped, offsets, number_of_attractive_channels=3
        )

    results["mws_3d_6x256x256"] = _timeit(mws_3d, repeats)
    results["mws_3d_6x256x256"]["meta"] = {"shape": list(aff3d.shape)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    results = run(repeats=args.repeats)
    for name, r in results.items():
        print(f"{name:<20} median={r['median'] * 1e3:9.3f} ms  min={r['min'] * 1e3:9.3f} ms")


if __name__ == "__main__":
    main()
