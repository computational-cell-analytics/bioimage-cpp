"""Apply both watershed variants to the cached ISBI volume and visualize.

Runs ``bic.segmentation.watershed`` on a node heightmap derived from the
nearest-neighbour affinity channels (``1 - mean(NN affs)``) and
``bic.segmentation.watershed_from_affinities`` directly on those same
channels. Markers come from labelled local minima of the smoothed
heightmap, so both algorithms see the same seeds and the two segmentations
can be compared side-by-side in napari.
"""

import napari
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label

import bioimage_cpp as bic
from bioimage_cpp._data import load_isbi_affinities, load_isbi_raw


def _nearest_neighbour_channels(offsets):
    return [
        i
        for i, offset in enumerate(offsets)
        if sum(1 for v in offset if v != 0) == 1
        and all(abs(v) <= 1 for v in offset)
    ]


def _filter_2d(affinities, offsets, z=0):
    channels_2d = [i for i, offset in enumerate(offsets) if offset[0] == 0]
    affinities_2d = np.ascontiguousarray(affinities[channels_2d, z])
    offsets_2d = [tuple(offset[1:]) for offset in (offsets[i] for i in channels_2d)]
    return affinities_2d, offsets_2d


def build_inputs(
    affinities,
    offsets,
    *,
    smoothing_sigma=4.0,
    min_distance=8,
):
    """Return (nn_affinities, nn_offsets, heightmap, markers).

    Seeds are local maxima of the smoothed inverted-boundary score (i.e.
    the "interior-ness" of each pixel). Heavily smoothing the noisy
    per-pixel NN-affinity map before peak-finding suppresses the spurious
    minima on membranes and within cells that one gets from raw
    ``local_minima``, while ``min_distance`` enforces one seed per cell.
    """
    nn_channels = _nearest_neighbour_channels(offsets)
    nn_affinities = np.ascontiguousarray(affinities[nn_channels])
    nn_offsets = [tuple(offsets[i]) for i in nn_channels]

    # Boundary probability heightmap: ~0 inside cells, ~1 on membranes.
    heightmap = (1.0 - nn_affinities.mean(axis=0)).astype(np.float32)

    interior_score = 1.0 - ndi.gaussian_filter(heightmap, sigma=smoothing_sigma)
    peak_coords = peak_local_max(
        interior_score,
        min_distance=min_distance,
        exclude_border=False,
    )
    seed_mask = np.zeros_like(interior_score, dtype=bool)
    if peak_coords.size > 0:
        seed_mask[tuple(peak_coords.T)] = True
    markers = label(seed_mask, connectivity=1).astype(np.int32, copy=False)

    return nn_affinities, nn_offsets, heightmap, np.ascontiguousarray(markers)


def main():
    affinities, offsets = load_isbi_affinities()
    raw = load_isbi_raw()

    run_2d = True
    if run_2d:
        affinities, offsets = _filter_2d(affinities, offsets, z=0)
        raw = raw[0]

    nn_affinities, nn_offsets, heightmap, markers = build_inputs(affinities, offsets)

    print(f"Heightmap shape: {heightmap.shape}, markers: {int(markers.max())} seeds")

    print("Running watershed on heightmap...")
    seg_heightmap = bic.segmentation.watershed(heightmap, markers)
    print("Running watershed_from_affinities...")
    seg_affinity = bic.segmentation.watershed_from_affinities(
        nn_affinities, nn_offsets, markers,
    )

    viewer = napari.Viewer()
    viewer.add_image(raw, name="raw")
    viewer.add_image(heightmap, name="heightmap (1 - mean NN affinities)")
    viewer.add_image(nn_affinities, name="NN affinities", channel_axis=0)
    viewer.add_labels(markers, name="markers (distance-transform peaks)")
    viewer.add_labels(seg_heightmap, name="watershed (heightmap)")
    viewer.add_labels(seg_affinity, name="watershed_from_affinities")
    napari.run()


if __name__ == "__main__":
    main()
