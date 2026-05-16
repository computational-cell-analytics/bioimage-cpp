import napari

from bioimage_cpp._data import load_isbi_affinities, load_isbi_raw


def mws_bic(affinities, offsets):
    import bioimage_cpp as bic
    print("Start MWS ...")
    affs = affinities.copy()
    ndim = len(offsets[0])
    affs[:ndim] *= -1
    affs[:ndim] += 1
    segmentation = bic.segmentation.mutex_watershed(
        affs, offsets, number_of_attractive_channels=ndim
    )
    print("done ...")
    return segmentation


def _filter_2d(affinities, offsets):
    chans_2d = [i for i, off in enumerate(offsets) if off[0] == 0]
    affinities = affinities[chans_2d][:, 0]
    offsets = [off[1:] for i, off in enumerate(offsets) if i in chans_2d]
    return affinities, offsets


def main():
    affinities, offsets = load_isbi_affinities()
    raw = load_isbi_raw()

    check_2d = True
    if check_2d:
        affinities, offsets = _filter_2d(affinities, offsets)
        raw = raw[0]
    segmentation = mws_bic(affinities, offsets)

    viewer = napari.Viewer()
    viewer.add_image(raw, name="raw")
    viewer.add_image(affinities, name="affinities")
    viewer.add_labels(segmentation, name="mws-segmentation")
    napari.run()


if __name__ == "__main__":
    main()
