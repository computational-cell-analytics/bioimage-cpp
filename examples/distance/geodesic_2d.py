import imageio.v3 as imageio
import napari
import numpy as np

from bioimage_cpp.distance import geodesic_distance_field, distance_transform

mask = (imageio.imread("./complex-shape.tif") == 2).astype("uint8")
# compute the most central point --- distance transform maximum
max_point = np.argmax(distance_transform(mask).ravel())
max_point = np.unravel_index(max_point, mask.shape)

sources = np.array([max_point])
distance_field, gradients = geodesic_distance_field(mask, sources, return_gradient=True)
gradients = gradients.transpose((2, 0, 1))

v = napari.Viewer()
v.add_image(distance_field)
v.add_image(gradients)
v.add_labels(mask)
v.add_points([max_point])
napari.run()
