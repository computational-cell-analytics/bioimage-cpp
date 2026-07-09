import napari
import numpy as np

from bioimage_cpp.distance import geodesic_distance_field, distance_transform
from shape import make_shape

mask = make_shape()
max_point = np.argmax(distance_transform(mask).ravel())
max_point = np.unravel_index(max_point, mask.shape)

sources = np.array([max_point])
distance_field, gradients = geodesic_distance_field(mask, sources, return_gradient=True)
gradients = gradients.transpose((3, 0, 1, 2))

v = napari.Viewer()
v.add_image(distance_field)
v.add_image(gradients)
v.add_labels(mask)
v.add_points([max_point])
napari.run()
