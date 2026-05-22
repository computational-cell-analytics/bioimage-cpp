from typing import Tuple

import nifty
import numpy as np


def smooth_mesh(
    verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, iterations: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth mesh surface via laplacian smoothing.

    Args:
        verts: The mesh vertices.
        normals: The mesh normals.
        faces: The mesh faces.
        iterations: The number of smoothing iterations.

    Returns:
        The vertices after smoothing.
        The normals after smoothing.
    """
    n_verts = len(verts)
    g = nifty.graph.undirectedGraph(n_verts)

    edges = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, ::2]], axis=0)
    g.insertEdges(edges)

    current_verts = verts
    current_normals = normals
    new_verts = np.zeros_like(verts, dtype=verts.dtype)
    new_normals = np.zeros_like(normals, dtype=normals.dtype)

    # Implement this directly in nifty for speed up?
    for it in range(iterations):
        for vert in range(n_verts):
            nbrs = np.array([vert] + [nbr[0] for nbr in g.nodeAdjacency(vert)], dtype="int")
            new_verts[vert] = np.mean(current_verts[nbrs], axis=0)
            new_normals[vert] = np.mean(current_normals[nbrs], axis=0)
        current_verts = new_verts
        current_normals = new_normals

    return new_verts, new_normals
