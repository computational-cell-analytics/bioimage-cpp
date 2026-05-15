import numpy as np


def same_partition(labels, expected):
    labels = np.asarray(labels)
    expected = np.asarray(expected)
    assert labels.shape == expected.shape
    np.testing.assert_array_equal(
        labels[:, None] == labels[None, :],
        expected[:, None] == expected[None, :],
    )
