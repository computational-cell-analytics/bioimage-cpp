"""Connected-components labeling."""

from __future__ import annotations

import numpy as np

from .. import _core

_LABEL_BY_DTYPE: dict[np.dtype, object] = {
    np.dtype("uint8"): _core._label_uint8,
    np.dtype("uint16"): _core._label_uint16,
    np.dtype("uint32"): _core._label_uint32,
    np.dtype("uint64"): _core._label_uint64,
    np.dtype("int32"): _core._label_int32,
    np.dtype("int64"): _core._label_int64,
}


def label(
    image: np.ndarray,
    background: int = 0,
    connectivity: int | None = None,
) -> np.ndarray:
    """Label connected components of equal-valued pixels.

    Mirrors :func:`skimage.measure.label`: two non-background pixels share a
    component iff there is a path of ``connectivity``-neighbour steps between
    them along which the input value is constant. Output labels are dense,
    start at ``1``, and are assigned in row-major first-occurrence order.

    Parameters
    ----------
    image:
        2D or 3D integer (or boolean) array. Supported dtypes are ``bool``,
        ``uint8``, ``uint16``, ``uint32``, ``uint64``, ``int32``, ``int64``.
        Floating-point inputs are rejected. Non-contiguous arrays are copied.
        Passing a ``bool`` array enables an internal fast path that skips
        per-pixel value-equality compares; convert to ``bool`` first if your
        mask is currently stored as ``uint8`` with values ``{0, 1}``.
    background:
        Pixel value treated as background. Background pixels stay ``0`` in the
        output. Defaults to ``0``.
    connectivity:
        Integer in ``[1, image.ndim]``. ``1`` means orthogonal neighbours only
        (4-connectivity in 2D, 6-connectivity in 3D); ``image.ndim`` enables
        full diagonal connectivity (8-connectivity in 2D, 26-connectivity in
        3D). Defaults to ``image.ndim`` (same as ``skimage.measure.label``).

    Returns
    -------
    np.ndarray
        ``uint64`` label array with the same shape as ``image``.
    """
    image_array = np.asarray(image)

    if image_array.ndim not in (2, 3):
        raise ValueError(
            "image must have ndim 2 or 3, got ndim=" + str(image_array.ndim)
        )

    if connectivity is None:
        connectivity = image_array.ndim
    connectivity = int(connectivity)
    if not 1 <= connectivity <= image_array.ndim:
        raise ValueError(
            "connectivity must be in [1, image.ndim], got "
            f"connectivity={connectivity} for image.ndim={image_array.ndim}"
        )

    if image_array.dtype == np.dtype("bool"):
        image_view = image_array.view(np.uint8)
        binary_mode = True
    else:
        image_view = image_array
        binary_mode = False

    try:
        run = _LABEL_BY_DTYPE[image_view.dtype]
    except KeyError as error:
        supported = "bool, " + ", ".join(str(dtype) for dtype in _LABEL_BY_DTYPE)
        raise TypeError(
            f"image must have one of dtypes ({supported}), "
            f"got dtype={image_array.dtype}"
        ) from error

    image_c = np.ascontiguousarray(image_view)
    background_value = image_view.dtype.type(background)
    return run(image_c, background_value, connectivity, binary_mode)
