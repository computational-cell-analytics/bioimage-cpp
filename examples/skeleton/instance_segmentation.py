#!/usr/bin/env python
"""Split a skeletonized mask into filament instances with bioimage_cpp.

Skeletonize a binary mask with TEASAR, clean the skeleton graph so each filament
becomes its own connected component, and save the labeled instances.
"""

import argparse
from pathlib import Path

import numpy as np

from bioimage_cpp.graph import connected_components
from bioimage_cpp.skeleton import clean_filament_graph, draw_instances, skeleton_to_graph, teasar


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mask_path", required=True,
                        help="Binary mask to skeletonize; an .h5 or MRC volume.")
    parser.add_argument("--seg_key", default=None,
                        help="HDF5 dataset holding the mask; required for .h5 inputs.")
    parser.add_argument("--output_dir", default=".",
                        help="Directory for the skeleton and instance .npz files.")
    parser.add_argument("--pixel_size", type=float, default=10.0,
                        help="Isotropic voxel spacing for teasar; graph coordinates use these units.")
    parser.add_argument("--teasar_scale", type=float, default=0.0,
                        help="teasar invalidation-radius scale.")
    parser.add_argument("--teasar_const", type=float, default=70.0,
                        help="teasar invalidation-radius constant.")
    parser.add_argument("--number_of_threads", type=int, default=8,
                        help="Threads for teasar.")
    parser.add_argument("--direction_span", type=int, default=10,
                        help="Nodes walked along each arm to estimate its direction at a junction.")
    parser.add_argument("--min_through_angle", type=float, default=170.0,
                        help="Min through-pair angle (degrees) for a degree-4 crossing to split.")
    parser.add_argument("--min_branch_angle", type=float, default=30.0,
                        help="Min branch angle (degrees) for a degree-3 odd arm to be separated.")
    parser.add_argument("--tick_length", type=float, default=50.0,
                        help="Prune dead-end branches shorter than this distance; 0 disables.")
    parser.add_argument("--join_dist", type=float, default=50.0,
                        help="Join collinear endpoints across gaps up to this distance; 0 disables.")
    parser.add_argument("--min_join_angle", type=float, default=175.0,
                        help="Min straightness (degrees) for a join; 180 is collinear.")
    parser.add_argument("--circle_size", type=float, default=70.0,
                        help="Instance tube diameter for the --view render, in --pixel_size units.")
    parser.add_argument("--view", action="store_true",
                        help="Open the binary mask and instances in napari.")
    return parser.parse_args()


def load_mask(mask_path, seg_key):
    path = Path(mask_path)
    if path.suffix in (".h5", ".hdf5"):
        import h5py

        if seg_key is None:
            raise ValueError("--seg_key is required for HDF5 inputs")
        with h5py.File(path, "r") as f:
            if seg_key not in f:
                raise KeyError(f"{seg_key} not in {path}")
            return np.asarray(f[seg_key][:])

    import mrcfile

    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        return np.asarray(mrc.data)


def main():
    args = parse_args()
    mask = load_mask(args.mask_path, args.seg_key)
    binary = (mask > 0).astype(np.uint8)

    spacing = (args.pixel_size,) * 3
    raw_vertices, raw_edges, raw_radii = teasar(
        binary, spacing=spacing, scale=args.teasar_scale, constant=args.teasar_const,
        number_of_threads=args.number_of_threads,
    )

    vertices, edges, radii = clean_filament_graph(
        raw_vertices, raw_edges, radii=raw_radii,
        direction_span=args.direction_span, min_through_angle=args.min_through_angle,
        min_branch_angle=args.min_branch_angle, tick_length=args.tick_length,
        join_dist=args.join_dist, min_join_angle=args.min_join_angle,
    )
    labels = connected_components(skeleton_to_graph(vertices, edges))

    name = Path(args.mask_path).stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton = dict(vertices=raw_vertices, edges=raw_edges)
    if raw_radii is not None:
        skeleton["radii"] = raw_radii
    instances = dict(vertices=vertices, edges=edges, labels=labels)
    if radii is not None:
        instances["radii"] = radii
    np.savez_compressed(output_dir / f"{name}_skeleton.npz", **skeleton)
    np.savez_compressed(output_dir / f"{name}_instances.npz", **instances)
    print(f"{name}: {len(np.unique(labels))} instances -> {output_dir}")

    if args.view:
        import napari

        radius = (args.circle_size / 2) / args.pixel_size
        volume = draw_instances(vertices / args.pixel_size, edges, labels, binary.shape, radius)
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_labels(binary, name="mask", opacity=0.4)
        viewer.add_labels(volume, name="instances")
        napari.run()


if __name__ == "__main__":
    main()
