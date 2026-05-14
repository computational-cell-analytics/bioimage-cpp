# AGENTS.md — bioimage-cpp

A small C++/Python package for bioimage-analysis algorithms that are otherwise hard to install. Goal: reliable PyPI wheels with a minimal dependency footprint. Not a compatibility layer for `nifty`/`vigra`/`affogato`, and not a full reimplementation — a focused implementation of selected functionality.

## Core principles

- Easy to build, easy to ship on PyPI.
- C++20 standard library only. Avoid template-heavy libraries and external binary dependencies.
- NumPy at the Python boundary; lightweight C++ array views internally.
- Keep I/O outside the C++ core (TIFF/HDF5/zarr/N5/OME-NGFF belong in Python).
- Keep the C++ algorithmic core independent of Python, NumPy, and nanobind — nanobind only in the binding layer.
- Explicit, narrow APIs. No compatibility namespaces for older libraries.

## Build & test

```bash
pip install -e . --no-build-isolation   # editable build (rebuilds C++)
python -m pytest tests/ -q              # full Python test suite
```

Required in the environment: `scikit-build-core`, `cmake>=3.21`, `ninja`, `nanobind`, `numpy`, `pytest`. The build produces one extension module: `bioimage_cpp._core`. Development scripts under `development/` compare against external references (nifty, affogato); they are not part of the test suite and must not be required by `pytest`.

## Reuse existing helpers — do not duplicate

Before introducing union-finds, priority queues, edge hashing, stride math, threading helpers, or label relabeling, check `include/bioimage_cpp/detail/`:

- `detail/union_find.hxx` — `UnionFind` (path compression + union by rank). `find`, `merge`, `merge_to`, `unite_roots`.
- `detail/indexed_heap.hxx` — addressable max-heap with mutable priorities. `DenseIndexedHeap` (vector-backed locator, integer keys in `[0, N)`) and `SparseIndexedHeap` (hashmap-backed, arbitrary keys).
- `detail/edge_hash.hxx` — `Edge`, `edge_key`, `EdgeHash` for hashing unordered node pairs.
- `detail/grid.hxx` — `c_order_strides`, `valid_offset_target`, `is_valid_grid_edge` for row-major grid offsets.
- `detail/relabel.hxx` — `dense_relabel` to map arbitrary labels onto `[0, k)` preserving first-occurrence order.
- `detail/threading.hxx` — `normalize_thread_count`, `parallel_for_chunks(n_threads, n_items, chunk)`.

If a needed helper does not exist but is generally useful, add it to `detail/` as a header-only utility with a focused API rather than inlining it into one algorithm. Keep the contract small and well-documented; over time `detail/` is what lets multiple modules stay small and consistent.

## Dependencies

**Allowed**: C++20 stdlib, `nanobind`, `numpy`, `scikit-build-core`, `cmake`, `pytest`, `cibuildwheel` (CI only).

**Avoid in the core package** (require strong justification + wheel impact analysis): `xtensor`, `Boost.Python`, `pybind11`, `vigra`, `nifty`, HDF5, z5/N5, libtiff/libpng/libjpeg, OpenEXR, FFTW, heavy solver libraries, conda-only packages, and unguarded OpenMP. Isolate optional heavy features behind separate packages.

## Repository layout

```
include/bioimage_cpp/         public/semi-public C++ headers (header-only algorithms)
include/bioimage_cpp/detail/  shared header-only utilities (see above)
src/cpp/                      non-templated C++ implementation files
src/bindings/                 nanobind binding code only (module.cxx + per-module .cxx)
src/bioimage_cpp/             Python package
tests/                        pytest suite, runs against the installed package
development/                  reference comparisons — not required by tests
```

Build system is `scikit-build-core` + CMake. Build one extension module (`bioimage_cpp._core`). Do not use `setup.py`-style builds. Do not enable nanobind `STABLE_ABI` while Python <3.12 is supported.

## CMake guidelines

- C++20 throughout (`cxx_std_20`).
- Do not assume conda or system HDF5/TIFF/Boost.
- Do not enable `-march=native` in wheel builds.
- Use hidden symbol visibility where appropriate.
- Avoid platform-specific flags unless carefully guarded.

## Wheel building

`cibuildwheel` in GitHub Actions. Targets: Linux x86_64 (manylinux), macOS x86_64 / arm64, Windows x86_64. Linux aarch64 once core wheels are stable.

- No conda dependency. No compiler required at install time.
- No external shared libraries unless bundled correctly and legally.
- Test the installed wheel — importability and core functions on small arrays.
- No `-march=native`, no unguarded OpenMP, no platform-specific assumptions.

## Python API

Prefer functions: `bic.some_function(input_array, labels, ...)`. Avoid exposing complex C++ classes unless genuinely needed; if exposed, lifetime semantics must be obvious. No `vigra`/`nifty` compatibility namespace.

## Array handling

- NumPy at the Python boundary, `ArrayView<T>` / `ConstArrayView<T>` internally. The view carries `data`, `shape`, `strides`, `ndim()` only.
- Algorithm code must not depend on Python, NumPy, or nanobind.
- C-contiguous in C++ kernels; Python wrappers convert with `np.ascontiguousarray` when copying is acceptable. Support for arbitrary strides, if any, must be intentional and tested.
- NumPy axis order throughout. For shape `(z, y, x)`, coordinates are reported as `(z, y, x)` unless explicitly documented otherwise.
- Define supported dtypes explicitly per function. Typical: label arrays `uint32/uint64/int32/int64`; image arrays `uint8/uint16/uint32/float32/float64`; stats outputs `float64`; index outputs `int64/uint64`. Use explicit dtype dispatch in the binding layer; avoid large template instantiation matrices.

### Memory ownership

- Do not store raw pointers to array data past the call. Do not return views into temporaries.
- Prefer returning newly allocated arrays. For in-place outputs, require writable arrays and document mutation.
- If nanobind wraps memory owned by C++, attach an explicit capsule/owner so lifetimes are correct.

### GIL

Release the GIL only after all Python-object validation and metadata extraction:

```cpp
{ nb::gil_scoped_release release; run_algorithm(...); }
```

Do not touch Python objects while the GIL is released.

### Error handling

Throw `std::invalid_argument` / `std::runtime_error` from C++; nanobind translates them to Python exceptions. Messages must name the argument and report expected vs. actual shape/dtype. Example: `labels must have ndim >= 2, got ndim=1`.

## Binding layer

The binding layer (and only the binding layer) validates `nanobind::ndarray` inputs — dimensionality, dtype (or convert), writability for outputs, shape compatibility, C-contiguity when required — converts metadata into `ArrayView`, allocates outputs, dispatches to the right template instantiation, and releases the GIL. Do not pass `nanobind::ndarray` or other nanobind types into algorithm code. Use `namespace nb = nanobind;` at the top of binding files. Prefer readable, explicit binding code over overly generic templates.

## Testing

Tests run against the installed Python package. For each public function, cover: small 2D / small 3D arrays, nontrivial labels, background-label handling, dtype variants, non-contiguous inputs (if accepted), invalid shapes, invalid dtypes, empty/degenerate inputs, and deterministic behavior. Reference comparisons against `nifty`/`affogato` are useful during development but must not be required by the default test suite, wheel tests, or source builds.

## Coding style

**C++**: clear, boring C++20. Small functions over template-heavy abstractions. `std::vector`/`std::array`/`std::span` where appropriate. Explicit integer types when overflow matters; careful with signed/unsigned conversions for shapes, strides, labels. Avoid clever metaprogramming.

**Python**: thin wrappers. Validate user-facing arguments in Python when this improves error messages. Use NumPy for lightweight preprocessing only. Avoid dependencies for small conveniences.

## Performance

Correct first, fast second. Benchmark before adding complexity. Prefer algorithmic improvements over build-level tweaks. Release the GIL for expensive kernels. Add threading only after the single-threaded implementation is stable; it must be portable and user-controllable.

## Documentation

`README.md` covers: what `bioimage-cpp` is and isn't, install/build, minimal examples, design philosophy. Public functions need concise docstrings documenting input shapes, supported dtypes, output shapes/dtypes, copy behavior, background-label behavior (if relevant), and axis/coordinate conventions.

Update `MIGRATION_GUIDE.md` whenever public functionality changes, so users migrating from `nifty`/`affogato` see the corresponding `bioimage-cpp` API, behavioral differences, and intentional improvements.

## Checklist when modifying code

1. Reuse existing `detail/` helpers; add new shared ones rather than duplicating.
2. Keep the build dependency-light. No external C++ dependencies without strong justification.
3. Algorithm code stays separate from binding code. nanobind only in the binding layer.
4. Validate arrays at the binding boundary.
5. Use internal array views — not external multidim-array libraries.
6. Tests for new behavior; tests run against the installed package.
7. Preserve PyPI wheel portability — no conda, no system libraries, no compiler at install time.
8. Clear error messages over permissive behavior.
9. No I/O format support in the C++ core.
10. No compatibility namespaces for older libraries.

When in doubt, choose the simpler design that is easier to build, test, and ship as wheels.
