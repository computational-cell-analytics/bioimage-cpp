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

    density = bic.flow.compute_flow_density(
        flow, mask, n_iter=1, dt=1.0, tol=0.0, method="euler", restrict_to_mask=False
    )

    expected = np.zeros(shape, dtype=np.float32)
    expected[2, 2] = np.prod(shape)
    np.testing.assert_array_equal(density, expected)


def test_3d_flow_converges_to_center():
    shape = (3, 5, 5)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    flow = np.stack([1.0 - zz, 2.0 - yy, 2.0 - xx]).astype(np.float32)
    mask = np.ones(shape, dtype=bool)

    density = bic.flow.compute_flow_density(
        flow, mask, n_iter=1, dt=1.0, tol=0.0, method="euler", restrict_to_mask=False
    )

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
    with pytest.raises(ValueError, match="tol"):
        bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=0.1, tol=-0.1)
    with pytest.raises(ValueError, match="method"):
        bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=0.1, method="bogus")
    with pytest.raises(ValueError, match="number_of_threads"):
        bic.flow.compute_flow_density(flow, mask, n_iter=1, dt=0.1, number_of_threads=0)


def test_multithreaded_matches_single_threaded():
    rng = np.random.default_rng(1234)
    flow = rng.normal(scale=0.5, size=(3, 6, 24, 24)).astype(np.float32)
    mask = (rng.random((6, 24, 24)) > 0.3).astype(bool)

    kwargs = dict(n_iter=20, dt=0.1, tol=0.0, method="euler", restrict_to_mask=False)
    single = bic.flow.compute_flow_density(flow, mask, number_of_threads=1, **kwargs)
    multi = bic.flow.compute_flow_density(flow, mask, number_of_threads=4, **kwargs)

    np.testing.assert_array_equal(single, multi)


def test_n_iter_beyond_convergence_is_stable():
    flow = np.zeros((2, 8, 8), dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)

    short = bic.flow.compute_flow_density(flow, mask, n_iter=5, dt=0.1, tol=0.01)
    long = bic.flow.compute_flow_density(flow, mask, n_iter=500, dt=0.1, tol=0.01)

    np.testing.assert_array_equal(short, long)


def test_restrict_to_mask_keeps_density_inside_mask():
    rng = np.random.default_rng(7)
    flow = rng.normal(scale=0.4, size=(2, 12, 12)).astype(np.float32)
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3:9] = True

    density = bic.flow.compute_flow_density(
        flow, mask, n_iter=20, dt=0.1, tol=0.0, method="rk2", restrict_to_mask=True
    )
    assert (density[~mask] == 0).all()


@pytest.mark.parametrize("method", ["euler", "rk2"])
def test_restrict_to_mask_freezes_at_last_valid_position(method):
    flow = np.zeros((2, 3, 3), dtype=np.float32)
    flow[1] = 1.0
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    density = bic.flow.compute_flow_density(
        flow, mask, n_iter=3, dt=1.0, tol=0.0,
        method=method, restrict_to_mask=True,
    )
    expected = np.zeros((3, 3), dtype=np.float32)
    expected[1, 1] = 1.0
    np.testing.assert_array_equal(density, expected)


def test_rk2_midpoint_cell_crossing_is_exact():
    # Constant flow of 3 px along x: the RK2 midpoint always lands outside the
    # interpolation cell of the current position, while the constant flow keeps
    # the exact trajectory independent of which cell is sampled.
    shape = (3, 16)
    flow = np.zeros((2,) + shape, dtype=np.float32)
    flow[1] = 3.0
    mask = np.ones(shape, dtype=bool)

    density = bic.flow.compute_flow_density(
        flow, mask, n_iter=2, dt=1.0, tol=0.0, method="rk2", restrict_to_mask=False
    )

    # Each particle advances 3 px per iteration and is clipped at x=15, so
    # x_final = min(x + 6, 15): starts 0..8 land on 6..14, starts 9..15 pile
    # up on the boundary column.
    expected = np.zeros(shape, dtype=np.float32)
    expected[:, 6:15] = 1.0
    expected[:, 15] = 7.0
    np.testing.assert_array_equal(density, expected)


def test_rk2_large_flow_multithreaded_matches_single_threaded():
    # Large flow steps make the RK2 midpoint cross interpolation cells; the
    # result must stay independent of the thread count.
    rng = np.random.default_rng(99)
    flow = rng.normal(scale=8.0, size=(3, 6, 20, 20)).astype(np.float32)
    mask = rng.random((6, 20, 20)) > 0.3

    kwargs = dict(n_iter=20, dt=0.2, tol=0.005, method="rk2", restrict_to_mask=True)
    single = bic.flow.compute_flow_density(flow, mask, number_of_threads=1, **kwargs)
    multi = bic.flow.compute_flow_density(flow, mask, number_of_threads=4, **kwargs)

    np.testing.assert_array_equal(single, multi)


def test_rk2_mixed_lifetimes_thread_equality():
    # 65 foreground particles (not divisible by 2/3/4) with strongly divergent
    # lifetimes: zero-flow columns converge on the first step while drift
    # columns run until frozen at the mask border. Different thread counts
    # split the particle range differently, so equality across them exercises
    # the lockstep block/remainder boundaries of the interleaved RK2 tracer.
    shape = (5, 13)
    flow = np.zeros((2,) + shape, dtype=np.float32)
    flow[1, :, 1::2] = 2.0
    mask = np.ones(shape, dtype=bool)
    kwargs = dict(n_iter=50, dt=0.2, tol=0.005, method="rk2", restrict_to_mask=True)

    results = [
        bic.flow.compute_flow_density(flow, mask, number_of_threads=t, **kwargs)
        for t in (1, 2, 3, 4, 5)
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(results[0], other)
    # restrict_to_mask freezes particles instead of dropping them
    assert results[0].sum() == mask.sum()
    # zero-flow columns converge in place and keep their particle
    assert (results[0][:, 0::2] >= 1.0).all()


def test_flow_rejects_non_finite_values():
    flow = np.zeros((2, 3, 3), dtype=np.float32)
    flow[0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bic.flow.compute_flow_density(flow, np.ones((3, 3), bool))


def test_rk2_runs():
    rng = np.random.default_rng(0)
    flow = rng.normal(scale=0.3, size=(2, 12, 12)).astype(np.float32)
    mask = np.ones((12, 12), dtype=bool)

    density = bic.flow.compute_flow_density(
        flow, mask, n_iter=30, dt=0.1, method="rk2", tol=0.0, restrict_to_mask=False
    )
    assert density.dtype == np.float32
    assert density.shape == mask.shape
    # Particle count is preserved when restrict_to_mask=False and the mask
    # covers the whole array.
    assert density.sum() == float(mask.size)


def test_sigma_with_spacing_runs_for_3d():
    flow = np.zeros((3, 4, 5, 6), dtype=np.float32)
    mask = np.ones((4, 5, 6), dtype=bool)

    density = bic.flow.compute_flow_density(
        flow, mask, n_iter=0, dt=0.0, sigma=1.0, spacing=(2.0, 1.0, 1.0)
    )

    assert density.dtype == np.float32
    assert density.shape == mask.shape
