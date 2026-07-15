import time
from pathlib import Path

import mrcfile
import napari
import numpy as np

from bioimage_cpp.skeleton import teasar


def main():
    path = Path(__file__).with_name("00004_gt_mask.mrc")

    full_crop = True
    if full_crop:
        bb = np.s_[:]
    else:
        bb = np.s_[100:200, 1024:1700, 1024:1700]

    print("Load data ...")
    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        mask = np.array(
            mrc.data[bb],
            dtype=np.uint8,
            copy=True,
        )
    print("... done")
    print(mask.shape)

    t0 = time.time()
    print("Start skeletonization ...")
    vertices, edges, _ = teasar(mask, scale=3.0, number_of_threads=8)
    print("... done in:", time.time() - t0, "s")

    # degrees = np.bincount(edges.ravel(), minlength=len(vertices))
    # endpoints = degrees <= 1
    # branch_points = degrees > 2

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_labels(mask, name="mask", opacity=0.6)
    viewer.add_shapes(
        vertices[edges],
        shape_type="line",
        name="skeleton edges",
        edge_color="#00d7ff",
        edge_width=1,
        opacity=0.9,
        blending="translucent_no_depth",
    )
    # viewer.add_points(
    #     vertices[endpoints],
    #     name="endpoints",
    #     size=3,
    #     face_color="#ffd166",
    #     border_color="#1b1b1b",
    #     border_width=0.15,
    #     n_dimensional=True,
    #     blending="translucent_no_depth",
    # )
    # viewer.add_points(
    #     vertices[branch_points],
    #     name="branch points",
    #     size=5,
    #     face_color="#ff4d6d",
    #     border_color="#1b1b1b",
    #     border_width=0.15,
    #     n_dimensional=True,
    #     blending="translucent_no_depth",
    # )
    napari.run()


if __name__ == "__main__":
    main()
