import h5py

from bioimage_cpp.flow._reference_impl import _compute_flow_density


def _compute_test_data(path, check=True):
    with h5py.File(path, mode="r") as f:
        dist = f["dist"][:]
        mask = (f["fg"][:] > 0.5)
    density = _compute_flow_density(dist, mask, n_iter=100, dt=0.1, sigma=None, verbose=True)
    with h5py.File(path, mode="a") as f:
        ds = f.require_dataset("density", shape=density.shape, dtype=density.dtype, compression="gzip")
        ds[:] = density

    if check:
        import napari
        v = napari.Viewer()
        v.add_image(mask)
        v.add_image(dist, visible=False)
        v.add_image(density)
        napari.run()


def compute_test_data_2d():
    path = "flow_data_2d.h5"
    _compute_test_data(path)


def compute_test_data_3d():
    path = "flow_data_3d.h5"
    _compute_test_data(path)


def main():
    compute_test_data_2d()
    compute_test_data_3d()


if __name__ == "__main__":
    main()
