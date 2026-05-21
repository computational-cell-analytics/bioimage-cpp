import numpy as np
import pytest

import bioimage_cpp as bic


def test_n_iter_zero_counts_foreground_pixels():
    flow = np.zeros((2, 3, 4), dtype=np.float32)
    mask = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    )

    density = bic.flow.compute_flow_density(flow, mask, n_iter=0, dt=0.1)

    assert density.dtype == np.float32
    assert density.shape == mask.shape
    np.testing.assert_array_equal(density, mask.astype(np.float32))


def test_2d_flow_converges_to_center():
    shape = (5, 5)
    yy, xx = np.indices(shape, dtype=np.float32)
    flow = np.stack([2.0 - yy, 2.0 - xx]).astype(np.float32)
    mask = np.ones(shape, dtype=bool)

    density = bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=1.0)

    expected = np.zeros(shape, dtype=np.float32)
    expected[2, 2] = np.prod(shape)
    np.testing.assert_array_equal(density, expected)


def test_3d_flow_converges_to_center():
    shape = (3, 5, 5)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    flow = np.stack([1.0 - zz, 2.0 - yy, 2.0 - xx]).astype(np.float32)
    mask = np.ones(shape, dtype=bool)

    density = bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=1.0)

    expected = np.zeros(shape, dtype=np.float32)
    expected[1, 2, 2] = np.prod(shape)
    np.testing.assert_array_equal(density, expected)


def test_non_contiguous_inputs_are_handled():
    rng = np.random.default_rng(42)
    flow = rng.normal(size=(2, 6, 7)).astype(np.float32)
    mask = rng.random((6, 7)) > 0.4
    flow_view = flow[:, ::-1, :]
    mask_view = mask[::-1, :]

    got = bic.flow.compute_flow_density(flow_view, mask_view, n_iter=3, dt=0.2)
    expected = bic.flow.compute_flow_density(
        np.ascontiguousarray(flow_view),
        np.ascontiguousarray(mask_view),
        n_iter=3,
        dt=0.2,
    )

    np.testing.assert_array_equal(got, expected)


def test_smoothing_keeps_density_zero_outside_mask():
    flow = np.zeros((2, 5, 5), dtype=np.float32)
    mask = np.ones((5, 5), dtype=bool)
    mask[0, :] = False
    mask[:, 0] = False

    density = bic.flow.compute_flow_density(flow, mask, n_iter=0, dt=0.0, sigma=1.0)

    assert density.dtype == np.float32
    assert density.shape == mask.shape
    assert np.all(density[~mask] == 0.0)


def test_rejects_invalid_flow_ndim():
    with pytest.raises(ValueError, match="2D or 3D"):
        bic.flow.compute_flow_density(np.zeros((2, 4), dtype=np.float32), np.zeros(4), n_iter=1, dt=0.1)


def test_rejects_wrong_channel_count():
    flow = np.zeros((3, 5, 5), dtype=np.float32)
    mask = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="first axis"):
        bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=0.1)


def test_rejects_mismatched_mask_shape():
    flow = np.zeros((2, 5, 5), dtype=np.float32)
    mask = np.ones((5, 4), dtype=bool)
    with pytest.raises(ValueError, match="fg_mask"):
        bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=0.1)


def test_rejects_invalid_parameters():
    flow = np.zeros((2, 5, 5), dtype=np.float32)
    mask = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="n_iter"):
        bic.flow.compute_flow_density(flow, mask, n_iter=-1, dt=0.1)
    with pytest.raises(ValueError, match="dt"):
        bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=-0.1)
    with pytest.raises(ValueError, match="dt"):
        bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=np.inf)
    with pytest.raises(ValueError, match="number_of_threads"):
        bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=0.1, number_of_threads=0)


def test_multithreaded_matches_single_threaded():
    rng = np.random.default_rng(1234)
    flow = rng.normal(scale=0.5, size=(3, 6, 24, 24)).astype(np.float32)
    mask = (rng.random((6, 24, 24)) > 0.3).astype(bool)

    single = bic.flow.compute_flow_density(flow, mask, n_iter=20, dt=0.1, number_of_threads=1)
    multi = bic.flow.compute_flow_density(flow, mask, n_iter=20, dt=0.1, number_of_threads=4)

    np.testing.assert_array_equal(single, multi)


def test_sigma_with_spacing_runs_for_3d():
    flow = np.zeros((3, 4, 5, 6), dtype=np.float32)
    mask = np.ones((4, 5, 6), dtype=bool)

    density = bic.flow.compute_flow_density(
        flow, mask, n_iter=0, dt=0.0, sigma=1.0, spacing=(2.0, 1.0, 1.0)
    )

    assert density.dtype == np.float32
    assert density.shape == mask.shape
