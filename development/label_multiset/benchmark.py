"""Benchmark the bioimage-cpp label_multiset implementation against nifty.

Runs:
  - multiset_from_labels (bioimage-cpp only — nifty has no direct equivalent;
    upstream code typically writes the level-0 multiset out manually)
  - downsampleMultiset
  - readSubset
  - MultisetMerger.update

on a deterministic 3D label volume. Prints a comparison table and verifies
that the results agree numerically.

Run with:
    python development/label_multiset/benchmark.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from bioimage_cpp._core import Blocking as BicBlocking
from bioimage_cpp.label_multiset import (
    MultisetMerger,
    downsample_multiset,
    multiset_from_labels,
    read_subset,
)

try:
    import nifty.tools as nt

    HAVE_NIFTY = True
except ImportError:
    HAVE_NIFTY = False


SHAPE = (256, 256, 256)
N_LABELS = 2000
COARSEN = 2  # voxels per coarse cell (smaller -> more variety per downsample block)
DOWN_BLOCK = (2, 2, 2)
N_SUBSET_QUERIES = 5000
SUBSET_RANGE = 64
N_REPEATS = 3


def make_labels(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse_shape = tuple(s // COARSEN for s in SHAPE)
    coarse = rng.integers(0, N_LABELS, size=coarse_shape, dtype=np.uint64)
    return np.kron(coarse, np.ones((COARSEN,) * len(SHAPE), dtype=np.uint64))


@dataclass
class TimedResult:
    seconds: float
    value: object


def time_best(fn: Callable[[], object], repeats: int = N_REPEATS) -> TimedResult:
    times = []
    value = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return TimedResult(min(times), value)


def fmt_row(name: str, ours: float, theirs: float | None) -> str:
    if theirs is None:
        return f"  {name:<32} ours: {ours*1000:8.2f} ms   nifty: (n/a)"
    ratio = ours / theirs if theirs > 0 else float("inf")
    return (
        f"  {name:<32} ours: {ours*1000:8.2f} ms   nifty: {theirs*1000:8.2f} ms"
        f"   ratio: {ratio:5.2f}x"
    )


def bench_from_labels(labels: np.ndarray) -> TimedResult:
    return time_best(lambda: multiset_from_labels(labels, (1, 1, 1)))


def bench_downsample_ours(ms) -> Tuple[TimedResult, object]:
    blocking = BicBlocking([0, 0, 0], list(SHAPE), list(DOWN_BLOCK))
    res = time_best(lambda: downsample_multiset(ms, blocking))
    return res, blocking


def bench_downsample_nifty(ms) -> TimedResult | None:
    if not HAVE_NIFTY:
        return None
    n_blocking = nt.blocking(
        roiBegin=[0, 0, 0], roiEnd=list(SHAPE), blockShape=list(DOWN_BLOCK)
    )
    n_offsets = ms.offsets.astype(np.uint64)
    n_entry_sizes = ms.entry_sizes.astype(np.uint64)
    n_entry_offsets = ms.entry_offsets.astype(np.uint64)
    n_ids = ms.ids.astype(np.uint64)
    n_counts = ms.counts.astype(np.int32)

    return time_best(
        lambda: nt.downsampleMultiset(
            n_blocking, n_offsets, n_entry_sizes, n_entry_offsets,
            n_ids, n_counts, restrict_set=-1,
        )
    )


def bench_read_subset(ms) -> Tuple[TimedResult, TimedResult | None]:
    rng = np.random.default_rng(1)
    # Random sub-multisets: pick N_SUBSET_QUERIES random spatial positions and
    # gather their (offset, size) ranges.
    n_spatial = ms.n_spatial
    positions = rng.integers(0, n_spatial, size=N_SUBSET_QUERIES)
    range_size = SUBSET_RANGE  # how many entries to merge per query
    offsets_list = []
    sizes_list = []
    for p in positions:
        start = max(0, p - range_size // 2)
        end = min(n_spatial, start + range_size)
        offsets_list.append(ms.offsets[start:end].astype(np.uint64))
        sizes_list.append(
            ms.entry_sizes[ms.entry_offsets[start:end]].astype(np.uint64)
        )
    flat_offsets = np.concatenate(offsets_list)
    flat_sizes = np.concatenate(sizes_list)

    ours = time_best(lambda: read_subset(flat_offsets, flat_sizes, ms.ids, ms.counts))

    if HAVE_NIFTY:
        ids_int32 = ms.ids.astype(np.uint64)
        counts_int32 = ms.counts.astype(np.int32)
        theirs = time_best(
            lambda: nt.readSubset(
                flat_offsets, flat_sizes, ids_int32, counts_int32, True
            )
        )
    else:
        theirs = None
    return ours, theirs


def bench_merger(ms_downsampled) -> Tuple[TimedResult, TimedResult | None]:
    # Build a merger from the downsampled multiset, then update with itself.
    # The constructor expects one offset per unique entry (length n_unique).
    entry_sizes = ms_downsampled.entry_sizes.astype(np.uint64)
    ids = ms_downsampled.ids.astype(np.uint64)
    counts = ms_downsampled.counts.astype(np.uint32)
    counts_i32 = ms_downsampled.counts.astype(np.int32)

    unique_off = np.array(
        [int(ms_downsampled.offsets[
            np.where(ms_downsampled.entry_offsets == e)[0][0]])
         for e in range(ms_downsampled.n_entries)],
        dtype=np.uint64,
    )

    def run_ours():
        m = MultisetMerger(unique_off, entry_sizes, ids, counts)
        spatial = ms_downsampled.entry_offsets.astype(np.uint64).copy()
        m.update(unique_off, entry_sizes, ids, counts, spatial)
        return m

    ours = time_best(run_ours, repeats=N_REPEATS)

    if HAVE_NIFTY:
        def run_nifty():
            m = nt.MultisetMerger(unique_off, entry_sizes, ids, counts_i32)
            spatial = ms_downsampled.entry_offsets.astype(np.uint64).copy()
            m.update(unique_off, entry_sizes, ids, counts_i32, spatial)
            return m
        theirs = time_best(run_nifty, repeats=N_REPEATS)
    else:
        theirs = None
    return ours, theirs


def main() -> None:
    print(f"shape={SHAPE}  labels={N_LABELS}  downsample={DOWN_BLOCK}  repeats={N_REPEATS}")
    print(f"nifty available: {HAVE_NIFTY}")
    print()
    print("Generating label volume...")
    labels = make_labels()
    print(f"  unique labels in volume: {len(np.unique(labels))}")
    print()

    print("Benchmarks (best of N):")
    t_from = bench_from_labels(labels)
    print(fmt_row("multiset_from_labels (1,1,1)", t_from.seconds, None))
    ms0 = t_from.value

    t_down, blocking = bench_downsample_ours(ms0)
    t_down_nifty = bench_downsample_nifty(ms0)
    print(fmt_row(
        f"downsample_multiset {DOWN_BLOCK}",
        t_down.seconds,
        t_down_nifty.seconds if t_down_nifty else None,
    ))
    ms1 = t_down.value
    print(f"    -> level-1: n_spatial={ms1.n_spatial}  n_entries={ms1.n_entries}")

    t_read_ours, t_read_nifty = bench_read_subset(ms0)
    print(fmt_row(
        f"read_subset (x{N_SUBSET_QUERIES})",
        t_read_ours.seconds,
        t_read_nifty.seconds if t_read_nifty else None,
    ))

    t_merger_ours, t_merger_nifty = bench_merger(ms1)
    print(fmt_row(
        "MultisetMerger.update",
        t_merger_ours.seconds,
        t_merger_nifty.seconds if t_merger_nifty else None,
    ))

    # Correctness cross-check on downsample.
    if HAVE_NIFTY and t_down_nifty is not None:
        bic_argmax = ms1.argmax
        n_argmax = t_down_nifty.value[0]
        assert bic_argmax.tolist() == n_argmax.tolist(), "argmax mismatch vs nifty!"
        print("\nargmax(downsample) cross-check vs nifty: OK")


if __name__ == "__main__":
    main()
