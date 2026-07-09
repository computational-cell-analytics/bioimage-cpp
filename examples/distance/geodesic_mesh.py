import napari
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from bioimage_cpp.distance import geodesic_distance_field_mesh
from shape import make_shape

# Create the mesh based on our sample shape.
mask = make_shape()
verts, faces, _, _ = marching_cubes(mask, level=0.5)

# Create some points in the volume as sources for the distances.
c = mask.shape[0] / 2.0
angles = np.linspace(0.0, 2.0 * np.pi, num=5, endpoint=False)
points = np.array(
    [[c, c + 33.0 * np.sin(a), c + 33.0 * np.cos(a)] for a in angles]
)

# Snap the points to the closest mesh vertex.
_, idx = cKDTree(verts).query(points)
sources = np.unique(idx)

# Compute the distance field on the mesh.
field = geodesic_distance_field_mesh(verts, faces, sources)

# Alternatively, you can also compute the pairwise geodesic distance
# between all the points like this:
# pairwise_distances = geodesic_distances_mesh(verts, faces, sources, number_of_threads=1)

# Display the mesh and points in napari.
# The mesh gets the distance field as values.
v = napari.Viewer()
v.add_surface((verts, faces, field))
v.add_points(points)
napari.run()
