# AGENTS.md — bioimage-cpp

This repository implements `bioimage-cpp`: a small C++/Python package for bioimage-analysis algorithms that are currently hard to install from existing C++-heavy libraries. The primary goal is to provide reliable PyPI wheels with a minimal dependency footprint.

The package should be designed as a focused implementation of selected bioimage-analysis functionality, not as a compatibility layer for older libraries and not as a full reimplementation of large libraries such as `nifty` or `vigra`.

## Core principles

- Keep the package easy to build and easy to ship on PyPI.
- Prefer simple, dependency-light C++20 over large template libraries or external binary dependencies.
- Avoid dependencies that complicate wheels unless they are absolutely necessary.
- Do not introduce a C++ image container or multidimensional-array framework as a public dependency.
- Treat NumPy arrays as the Python-side data model and lightweight C++ array views as the C++-side data model.
- Keep I/O outside the C++ core. File formats such as TIFF, HDF5, zarr, N5, OME-NGFF, etc. should be handled by existing Python libraries, not by this package.
- Prefer explicit, narrow APIs over broad compatibility with old libraries.
- Keep the C++ algorithmic core independent of Python, NumPy, and nanobind. Nanobind should only be used in the binding layer.

## Dependencies

### Preferred dependencies

- C++20 standard library.
- `nanobind` for Python bindings.
- `numpy` for array inputs and outputs.
- `scikit-build-core` as the Python build backend.
- `cmake` for configuring the C++ build.
- `pytest` for tests.
- `cibuildwheel` for wheel builds in CI.

### Avoid in the core package

Do not add these dependencies to the core package unless there is a strong reason and the wheel-building consequences have been considered carefully:

- `xtensor`
- `Boost.Python`
- `pybind11`
- `vigra`
- `nifty`
- HDF5
- z5 / N5 libraries
- libtiff, libpng, libjpeg, OpenEXR
- FFTW
- solver libraries with nontrivial binary dependencies
- conda-only dependencies
- OpenMP, unless the packaging implications are handled explicitly

If optional functionality really requires a heavy dependency, isolate it behind an optional extension or a separate package. The default PyPI wheel should remain small and robust.

## Suggested repository layout

Use a `src` layout for the Python package and keep C++ headers, C++ implementation files, and Python bindings clearly separated.

```text
bioimage-cpp/
├── AGENTS.md
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── include/
│   └── bioimage_cpp/
│       ├── array_view.hxx
│       ├── dtype_dispatch.hxx
│       ├── errors.hxx
│       └── ...
├── src/
│   ├── bioimage_cpp/
│   │   ├── __init__.py
│   │   ├── _version.py
│   │   └── ...
│   ├── cpp/
│   │   └── ...
│   └── bindings/
│       ├── module.cxx
│       ├── array_conversions.cxx
│       └── ...
├── tests/
│   ├── test_*.py
│   └── reference/
│       └── ...
└── .github/
    └── workflows/
        ├── tests.yml
        └── wheels.yml
```

The exact file names may change, but the separation should remain:

- `include/bioimage_cpp/`: public or semi-public C++ headers.
- `src/cpp/`: C++ implementations.
- `src/bindings/`: Python binding code only.
- `src/bioimage_cpp/`: Python package code.
- `tests/`: Python-level tests and, where useful, C++ behavior tests exposed through Python.

## Build system

Use `scikit-build-core` with CMake. Avoid `setup.py`-based builds.

A minimal `pyproject.toml` should look conceptually like this:

```toml
[build-system]
requires = ["scikit-build-core", "nanobind"]
build-backend = "scikit_build_core.build"

[project]
name = "bioimage-cpp"
requires-python = ">=3.10"
dependencies = ["numpy"]

[tool.scikit-build]
cmake.version = ">=3.21"
wheel.packages = ["src/bioimage_cpp"]
```

The package name on PyPI should be `bioimage-cpp`. The import name should be `bioimage_cpp`.

If support for Python 3.10 and 3.11 is dropped later, the project may switch to Python >=3.12 and nanobind stable-ABI wheels. Do not enable nanobind `STABLE_ABI` while the package still supports Python <3.12.

## CMake guidelines

- Build one main Python extension module, for example `bioimage_cpp._core`.
- Use C++20 consistently.
- Use nanobind, not pybind11.
- Keep CMake simple and portable.
- Avoid platform-specific compiler flags unless they are guarded carefully.
- Do not assume conda is available.
- Do not assume system libraries such as HDF5, TIFF, or Boost are available.
- Do not enable native architecture flags such as `-march=native` in wheel builds.
- Use hidden symbol visibility where appropriate.
- Keep the extension self-contained.

A typical CMake structure should be:

```cmake
cmake_minimum_required(VERSION 3.21)
project(bioimage_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core
    NB_STATIC
    src/bindings/module.cxx
    src/bindings/array_conversions.cxx
    src/cpp/some_algorithm.cxx
)

target_include_directories(_core PRIVATE include)
target_compile_features(_core PRIVATE cxx_std_20)
```

If the minimum Python version is raised to 3.12 or later, stable-ABI builds can be considered:

```cmake
nanobind_add_module(_core
    STABLE_ABI
    NB_STATIC
    src/bindings/module.cxx
    src/bindings/array_conversions.cxx
    src/cpp/some_algorithm.cxx
)
```

Adjust this as the project grows, but avoid turning the build into a large dependency-discovery system.

## Wheel building

Use `cibuildwheel` in GitHub Actions for binary wheels.

The initial target platforms should be:

- Linux x86_64, using manylinux wheels.
- macOS x86_64.
- macOS arm64.
- Windows x86_64.

Later targets such as Linux aarch64 can be added once the core wheels are stable.

Important wheel-building rules:

- Wheels must not depend on conda.
- Wheels must not require users to have a local compiler at install time.
- Avoid external shared libraries unless they are bundled correctly and legally.
- Test the installed wheel, not only the source tree.
- Test importability with `python -c "import bioimage_cpp"`.
- Test core functions on small arrays after wheel installation.
- Do not use `-march=native`, unguarded OpenMP, or platform-specific compiler assumptions in release wheels.

## Python API design

The public Python API should be explicit and simple. Prefer functions such as:

```python
import bioimage_cpp as bic

result = bic.some_function(input_array, labels, ...)
```

Avoid exposing complex C++ classes unless they are genuinely needed. If a C++ object is exposed, its lifetime and ownership semantics must be obvious from Python.

Do not implement a `vigra` or `nifty` compatibility namespace. The API should be designed for `bioimage-cpp` directly.

## Multidimensional-array handling

### General rule

Use NumPy arrays at the Python boundary and lightweight C++ views internally.

Do not use `xtensor`, `vigra::MultiArray`, or another multidimensional-array framework as a core dependency.

### C++ array view

Use a simple internal view type that stores:

- data pointer
- shape
- strides
- number of dimensions
- dtype handled by template dispatch

For example:

```cpp
#include <cstddef>
#include <vector>

template <class T>
struct ArrayView {
    T* data = nullptr;
    std::vector<std::ptrdiff_t> shape;
    std::vector<std::ptrdiff_t> strides;

    std::ptrdiff_t ndim() const {
        return static_cast<std::ptrdiff_t>(shape.size());
    }
};
```

For read-only inputs, use `const T*` or a separate `ConstArrayView<T>`.

The C++ algorithm implementation must not depend directly on Python, NumPy, or nanobind.

### Binding-layer responsibilities

The Python binding layer is responsible for:

- accepting `nanobind::ndarray` inputs
- checking dimensionality
- checking or converting dtype
- checking writability for output or in-place arguments
- checking shape compatibility between inputs
- checking C-contiguity if the C++ implementation requires it
- converting nanobind array metadata into `ArrayView`
- allocating output arrays
- dispatching to the correct C++ template instantiation

Use nanobind only at this boundary. Do not pass `nanobind::ndarray` or other nanobind types into the algorithmic core.

A typical binding file should use a local namespace alias:

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
```

Example pattern:

```cpp
nb::ndarray<nb::numpy, std::uint64_t, nb::shape<-1, -1>, nb::c_contig>
some_function(
    nb::ndarray<nb::numpy, const std::uint64_t, nb::ndim<2>, nb::c_contig> labels
) {
    // Validate Python-facing arguments here.
    // Convert labels to ConstArrayView<std::uint64_t>.
    // Release the GIL.
    // Run the C++ implementation.
    // Return a newly allocated output array.
}
```

Prefer readable, explicit binding code over overly generic binding templates.

### Contiguity policy

Prefer a two-level policy:

1. Fast C++ kernels assume C-contiguous arrays.
2. Python wrappers convert inputs with `np.ascontiguousarray` when copying is acceptable.

For large arrays, avoid hidden expensive copies where possible. Public Python functions should make the copy policy clear, for example with an argument such as:

```python
def some_function(array, *, copy=True):
    ...
```

or by documenting that inputs are converted to contiguous arrays.

If an algorithm supports arbitrary strides, this should be intentional and tested. Do not accidentally support some strided cases while failing on others.

### Shape and axis conventions

Use NumPy conventions throughout:

- arrays are indexed in row-major order
- dimensions are ordered as they appear in the NumPy array
- spatial axes should not be silently reordered
- coordinate outputs should be in NumPy axis order

For an array with shape `(z, y, x)`, coordinates should be reported as `(z, y, x)` unless a function explicitly documents otherwise.

### Dtype policy

Do not silently accept every dtype. Each function should define supported dtypes explicitly.

Common policies:

- label arrays: `uint32`, `uint64`, `int32`, `int64`
- image arrays: `uint8`, `uint16`, `uint32`, `float32`, `float64`
- output statistics: usually `float64` for numeric stability, unless there is a reason to preserve dtype
- index arrays: prefer `int64` or `uint64` at the Python boundary

Use explicit dtype dispatch in the binding layer. Avoid large, uncontrolled template instantiation matrices.

### Memory ownership and lifetime

- Never store raw pointers to array data beyond the duration of the function call unless the owning Python object is kept alive explicitly.
- Do not return C++ views into temporary arrays.
- Do not return references to internal buffers unless ownership is clear and tested.
- Prefer returning newly allocated arrays.
- For in-place operations, require writable arrays and document mutation clearly.
- If a nanobind array wraps memory owned by C++, attach an explicit owner/capsule so the lifetime is correct.

### GIL handling

For long-running C++ kernels, release the Python GIL in the binding layer:

```cpp
{
    nb::gil_scoped_release release;
    run_algorithm(...);
}
```

Only release the GIL after all Python-object validation and array metadata extraction have completed. Do not touch Python objects while the GIL is released.

### Error handling

C++ code should throw standard exceptions such as `std::invalid_argument` or `std::runtime_error`. The binding layer should allow nanobind to translate these into Python exceptions.

Error messages should include:

- the offending argument name
- expected dimensionality or dtype
- actual dimensionality or dtype
- expected shape relationship, if applicable

Example:

```text
labels must have ndim >= 2, got ndim=1
```

## Testing guidelines

Tests should run against the installed Python package.

For each public function, test:

- small 2D arrays
- small 3D arrays where applicable
- nontrivial labels or regions
- background label behavior
- dtype variants
- non-contiguous inputs, if accepted
- incorrect shapes
- incorrect dtypes
- empty or degenerate inputs
- deterministic behavior

Reference tests against older libraries may be useful during development, but they are not part of the public API contract. Do not require `vigra` or `nifty` for the default test suite, wheel tests, or source builds.

## Coding style

### C++

- Use clear, boring C++20.
- Prefer small functions over large template-heavy abstractions.
- Keep algorithm code independent of nanobind.
- Use `std::vector`, `std::array`, `std::span` where available and appropriate.
- Avoid clever metaprogramming unless it clearly reduces maintenance burden.
- Use explicit integer types where overflow matters.
- Be careful with signed/unsigned conversions for shapes, strides, and labels.

### Python

- Keep Python wrappers thin.
- Validate user-facing arguments in Python when this improves error messages.
- Use NumPy for lightweight preprocessing only.
- Avoid adding dependencies for small convenience features.

## Performance guidelines

- Start with correct, portable implementations.
- Benchmark before introducing complexity.
- Prefer algorithmic improvements over build-level optimizations.
- Avoid `-march=native` in distributed wheels.
- Consider releasing the GIL for expensive kernels.
- Consider optional threading only after the single-threaded implementation is stable.
- If threading is added, make it portable and controllable by the user.

## Documentation expectations

The README should explain:

- what `bioimage-cpp` is
- what it intentionally is not
- how to install it from PyPI
- how to build it from source
- minimal examples for the public API
- the dependency-light design philosophy

Public functions should have concise docstrings documenting:

- input shapes
- supported dtypes
- output shapes and dtypes
- copy behavior
- background-label behavior, if relevant
- axis and coordinate conventions

## What an agent should do when modifying this repository

When adding or changing code:

1. Keep the build dependency-light.
2. Do not add external C++ dependencies without strong justification.
3. Keep algorithm code separate from binding code.
4. Validate arrays at the Python/binding boundary.
5. Use nanobind only in the binding layer.
6. Use internal C++ array views rather than external multidimensional-array libraries.
7. Add tests for new behavior.
8. Preserve PyPI wheel portability.
9. Prefer clear error messages over permissive behavior.
10. Do not add I/O format support to the C++ core.
11. Do not add compatibility namespaces for older libraries.

When unsure, choose the simpler design that is easier to build, test, and ship as wheels.
