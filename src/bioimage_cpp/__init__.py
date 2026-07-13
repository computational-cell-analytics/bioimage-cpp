"""`bioimage_cpp` implements image processing and segmentation functionality in C++.
It generates light-weight python bindings with nanobind with minimal dependencies to enable distribution via pip.

The main goal of this library is to provide functionality that is missing from scipy or scikit-image,
or to provide more performant versions of functionality from these libraries.

The functionality implemented here bundles and improves algorithms etc. from:
- [affogato](https://github.com/constantinpape/affogato)
- [fastfilters](https://github.com/sciai-lab/fastfilters)
- [nifty](https://github.com/DerThorsten/nifty)
- [vigra](https://github.com/ukoethe/vigra)

The goal is to provide the functionality within a single library and via pip as well as conda.

**Warning:** This library was written mainly by coding agents (claude code and openai codex).
It is not very thoroughly tested and may contain bugs.

## Installation

The `bioimage_cpp` library is available on PyPI and can be installed via pip:
```bash
pip install bioimage-cpp
```

Or via conda-forge:
```bash
conda install -c conda-forge bioimage-cpp
```

Additional dependencies for tests / data downloads can be installed via `pip install bioimage-cpp[test]` /  `pip install bioimage-cpp[data]` respectively.

You can also install this library from source. The build requires C++20 (GCC >= 10 or Clang >= 13).

If your system already has a C++20-capable compiler, clone and build directly:
```bash
git clone https://github.com/computational-cell-analytics/bioimage-cpp
cd bioimage-cpp
pip install -e .
```

On systems with an older compiler (e.g. many HPC clusters ship GCC 8), install a modern one from conda-forge alongside the build dependencies, and point `CC`/`CXX` at it:
```bash
conda install gcc_linux-64 gxx_linux-64 scikit-build-core nanobind -c conda-forge -y
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
pip install --no-build-isolation -e .
```

The `--no-build-isolation` flag reuses the `scikit-build-core` and `nanobind` already installed in your environment.
Re-run the same command after any C++ source change to rebuild.

## Functionality

This library provides the following functionality:
- `affinities`: functionality for deriving affinities from segmentations.
- `distance`: distance transform functionality.
- `filters`: efficient implementation of convolutional image filters.
- `graph`: graph creation and graph (partitioning) algorithms.
- `mesh`: triangle-mesh extraction and processing.
- `segmentation`: image segmentation functionality.
- `transformation`: affine transformations.
- `utils`: misc utility functionality.

## Example

Below is a simple example for creating and partitioning a graph with this library.
For more realistic use-cases check out [the migration guide](#migration-guide).

```python
import numpy as np
import bioimage_cpp as bic

# Create a graph with 50 nodes.
graph = bic.graph.undirected_graph(number_of_nodes=50)

# Insert a bunch of edges forming a chain.
graph.insert_edges(
    np.concatenate([np.arange(0, 49)[:, None], np.arange(1, 50)[:, None]], axis=1)
)

# Create edge weights in [-1, 1].
weights = 2 * np.random.rand(graph.number_of_edges) - 1

# Partition the graph via multicut (greedy solver).
objective = bic.graph.MulticutObjective(graph, weights)
solver = bic.graph.GreedyAdditiveMulticut()
partition = solver.optimize(objective)
print("Partitioned into", len(np.unique(partition)), "elements")
```

.. include:: ../../MIGRATION_GUIDE.md
"""

from ._version import __version__
from . import affinities
from . import distance
from . import filters
from . import flow
from . import graph
from . import label_multiset
from . import mesh
from . import segmentation
from . import transformation
from . import utils

__all__ = [
    "__version__",
    "affinities",
    "distance",
    "filters",
    "flow",
    "graph",
    "label_multiset",
    "mesh",
    "segmentation",
    "transformation",
    "utils",
]
