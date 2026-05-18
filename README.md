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

```python
affinities = np.ones((2, 4, 4), dtype=np.float32)
offsets = [[0, 1], [1, 0]]

segmentation = bic.segmentation.mutex_watershed(
    affinities,
    offsets,
    number_of_attractive_channels=2,
)
```

```python
image = np.arange(64, dtype=np.float32).reshape(8, 8)
matrix = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -1.0]])

transformed = bic.transformation.affine_transform(
    image,
    matrix,
    order=3,
    fill_value=0,
)
```

```python
graph = bic.graph.UndirectedGraph.from_edges(4, [[0, 1], [1, 2], [2, 3]])
graph.find_edge(2, 1)
# 1

grid = bic.graph.grid_graph((64, 64))
grid.number_of_edges
# 8064

edge_weights = bic.graph.grid_boundary_features(grid, boundary_map)
local_weights, valid_edges, lifted_uvs, lifted_weights, offset_ids = (
    bic.graph.grid_affinity_features_with_lifted(grid, affinities, offsets)
)
```

## Scope

The project is not a compatibility layer for `nifty`, `vigra`, or other large libraries. It keeps I/O and heavy dependencies out of the C++ core; callers should use existing Python packages for file formats and pass NumPy arrays into `bioimage-cpp`.
