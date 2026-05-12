# bioimage-cpp

Stand-alone implementation of image analysis and segmentaton functionality in C++ with minimal python bindings.

`bioimage-cpp` is a small C++/Python package for dependency-light bioimage-analysis algorithms that are difficult to install from larger C++ libraries. It exposes NumPy-based Python APIs backed by a focused C++20 core.

## Install

```bash
python -m pip install bioimage-cpp
```

## Build from source

```bash
python -m pip install -v ".[test]"
pytest -q
```

## Example

```python
import numpy as np
import bioimage_cpp as bic

labels = np.array([1, 3, 2, 1], dtype=np.uint64)
relabeling = {1: 10, 2: 20, 3: 30}

out = bic.utils.take_dict(relabeling, labels)
# array([10, 30, 20, 10], dtype=uint64)
```

## Scope

The project is not a compatibility layer for `nifty`, `vigra`, or other large libraries. It keeps I/O and heavy dependencies out of the C++ core; callers should use existing Python packages for file formats and pass NumPy arrays into `bioimage-cpp`.
