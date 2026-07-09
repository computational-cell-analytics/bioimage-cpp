import numpy as np


def make_shape(size=80):
    R, r = 24.0, 9.0
    c = size / 2.0
    zz, yy, xx = np.ogrid[:size, :size, :size]
    q = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) - R
    mask = (q ** 2 + (zz - c) ** 2 <= r ** 2).astype("uint8")
    return mask
