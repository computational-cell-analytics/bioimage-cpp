# bioimage-cpp

[![Documentation](https://img.shields.io/badge/docs-pdoc-blue)](https://computational-cell-analytics.github.io/bioimage-cpp/)
[![Build Status](https://github.com/computational-cell-analytics/bioimage-cpp/actions/workflows/tests.yml/badge.svg)](https://github.com/computational-cell-analytics/bioimage-cpp/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/bioimage-cpp)](https://pypi.org/project/bioimage-cpp/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/bioimage-cpp)](https://anaconda.org/conda-forge/bioimage-cpp)

Image processing and segmentation functionality in C++ with light-weight python bindings through nanobind and minimal dependencies to enable distribution via pip.

The package includes dependency-free triangle-mesh extraction from 3D volumes
and segmentation masks, plus Laplacian mesh smoothing, under `bioimage_cpp.mesh`.
It also includes exact masked-grid Dijkstra paths under `bioimage_cpp.distance`
and binary-forest plus semantic multi-label 3D TEASAR skeletonization under
`bioimage_cpp.skeleton`.

The `bioimage_cpp` python library can be installed via pip:
```bash
pip install bioimage-cpp
```

Or via conda-forge:
```bash
conda install -c conda-forge bioimage-cpp
```

Please refer to [the documentation](https://computational-cell-analytics.github.io/bioimage-cpp/) for details.
