"""Segmentation algorithms."""

from .label import label
from .mutex_watershed import mutex_watershed, semantic_mutex_watershed
from .watershed import watershed, watershed_from_affinities

__all__ = [
    "label",
    "mutex_watershed",
    "semantic_mutex_watershed",
    "watershed",
    "watershed_from_affinities",
]
