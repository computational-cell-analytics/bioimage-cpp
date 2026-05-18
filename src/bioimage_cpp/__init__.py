"""`bioimage_cpp` implements functionality needed for image procesing in C++.
It generates light-weight python bindings with nanobind with minimal dependencies to enable distribution via pip.

The main goal of this library is to provide functionality that is missing from scipy or scikit-image or
to provide more performant versions of functionality from these libraries.

The functionality implemented here bundles and improves algorithms etc. from:
- [affogato](https://github.com/constantinpape/affogato)
- [fastfilters](https://github.com/sciai-lab/fastfilters)
- [nifty](https://github.com/DerThorsten/nifty)
- [vigra](https://github.com/ukoethe/vigra)

The goal is to provide the functionality within a single library and via pip as well as conda.

**Warning:** This library was written mainly by coding agents (claude code and openai codex).
It is not very thoroughly tested and may contain bugs.

## Installation

TODO: document once on pip / conda

.. include:: ../../MIGRATION_GUIDE.md
"""

from ._version import __version__
from . import affinities
from . import filters
from . import graph
from . import segmentation
from . import transformation
from . import utils

__all__ = [
    "__version__",
    "affinities",
    "filters",
    "graph",
    "segmentation",
    "transformation",
    "utils",
]
