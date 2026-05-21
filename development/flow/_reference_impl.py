"""Naive reference implementation for flow tracing.

Lives under ``development/`` and depends on ``scipy``, ``scikit-image``, and
``tqdm`` — none of which are required by the installed ``bioimage_cpp``
package. Only used by the comparison/benchmark scripts in this directory.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates
from skimage.filters import gaussian
from tqdm import trange


def _compute_flow_density(
    directed_distances: np.ndarray,
    fg_mask: np.ndarray,
    n_iter: int,
    dt: float,
    sigma: Optional[float] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    verbose: bool = False
) -> np.ndarray:
    """Compute density from tracing a flow field derived from directed distances.

    Args:
        directed_distances: The directed distances predictions, one channel per spatial dimension.
        fg_mask: The predicted foreground mask; densities should only be computed within the mask.
        n_iter: The number of iterations for flow tracing.
        dt: The step size.
        sigma: Optinal sigma value for smoothing the result.
        spacing: The pixel / voxel size of the data.
        verbose: Whether to print progress.

    Returns:
        The density map from tracing the flow field. Contains integer values that record how
        many pixels were traced to this pixel.
    """
    shape, ndim = fg_mask.shape, fg_mask.ndim

    # Negate: directed_distances point toward boundary; -directed_distances point toward center.
    flow = (-directed_distances).astype(np.float32)

    fg_coords = np.stack(np.where(fg_mask), axis=1).astype(np.float32)  # (N, ndim)
    if len(fg_coords) == 0:
        return np.zeros(shape, dtype="uint32")

    positions = fg_coords.copy()

    for _ in trange(n_iter, disable=not verbose):
        # Clip positions to valid index range.
        for d in range(ndim):
            positions[:, d] = np.clip(positions[:, d], 0, shape[d] - 1)

        coords_list = [positions[:, d] for d in range(ndim)]

        # Sample the flow field at current (sub-pixel) positions.
        step = np.stack(
            [map_coordinates(flow[d], coords_list, order=1, mode="nearest") for d in range(ndim)],
            axis=1,
        )

        positions += dt * step

    # Clip final positions.
    for d in range(ndim):
        positions[:, d] = np.clip(positions[:, d], 0, shape[d] - 1)

    # Build convergence density: count how many pixels converged near each location.
    final_pos = np.round(positions).astype(np.int32)
    density = np.zeros(shape, dtype="float32")
    np.add.at(density, tuple(final_pos[:, d] for d in range(ndim)), 1.0)

    if sigma is not None:
        # Smooth the density to merge nearby convergence peaks into single seeds.
        sigma = sigma
        if spacing is not None and ndim == 3:
            sp = np.array(spacing, dtype="float32")
            sigma = (sigma / sp).tolist()  # physical-space isotropic smoothing
        density = gaussian(density, sigma=sigma)

    # Make sure the mask is empty.
    density *= fg_mask
    return density
