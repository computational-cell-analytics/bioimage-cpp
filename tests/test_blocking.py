import pytest

import bioimage_cpp as bic


def test_block_shape_and_halo_local_coordinates():
    block = bic.Block([2, 3], [7, 11])

    assert block.begin == [2, 3]
    assert block.end == [7, 11]
    assert block.shape == [5, 8]
    assert block.ndim == 2

    with_halo = bic.BlockWithHalo(bic.Block([0, 1], [9, 12]), block)
    assert with_halo.outer_block.begin == [0, 1]
    assert with_halo.outer_block.end == [9, 12]
    assert with_halo.inner_block.begin == [2, 3]
    assert with_halo.inner_block_local.begin == [2, 2]
    assert with_halo.inner_block_local.end == [7, 10]


def test_blocking_grid_and_blocks_match_nifty_layout():
    blocking = bic.Blocking([0, 0], [10, 7], [4, 3])

    assert blocking.roi_begin == [0, 0]
    assert blocking.roi_end == [10, 7]
    assert blocking.block_shape == [4, 3]
    assert blocking.block_shift == [0, 0]
    assert blocking.blocks_per_axis == [3, 3]
    assert blocking.number_of_blocks == 9
    assert blocking.ndim == 2

    assert blocking.block_grid_position(0) == [0, 0]
    assert blocking.block_grid_position(5) == [1, 2]
    assert blocking.get_block(0).begin == [0, 0]
    assert blocking.get_block(0).end == [4, 3]
    assert blocking.get_block(5).begin == [4, 6]
    assert blocking.get_block(5).end == [8, 7]
    assert blocking.get_block(8).begin == [8, 6]
    assert blocking.get_block(8).end == [10, 7]


def test_neighbor_ids():
    blocking = bic.Blocking([0, 0], [10, 7], [4, 3])

    assert blocking.get_neighbor_id(4, axis=0, lower=True) == 1
    assert blocking.get_neighbor_id(4, axis=0, lower=False) == 7
    assert blocking.get_neighbor_id(4, axis=1, lower=True) == 3
    assert blocking.get_neighbor_id(4, axis=1, lower=False) == 5
    assert blocking.get_neighbor_id(1, axis=0, lower=True) == -1
    assert blocking.get_neighbor_id(7, axis=0, lower=False) == -1
    assert blocking.get_neighbor_id(3, axis=1, lower=True) == -1
    assert blocking.get_neighbor_id(5, axis=1, lower=False) == -1


def test_shifted_roi_coordinates_to_block_id_and_overlapping_box():
    blocking = bic.Blocking([10, 20], [20, 31], [4, 5], [2, 1])

    assert blocking.blocks_per_axis == [3, 3]
    assert blocking.get_block(0).begin == [10, 20]
    assert blocking.get_block(0).end == [12, 24]
    assert blocking.get_block(4).begin == [12, 24]
    assert blocking.get_block(4).end == [16, 29]

    assert blocking.coordinates_to_block_id([10, 20]) == 0
    assert blocking.coordinates_to_block_id([12, 24]) == 4
    assert blocking.coordinates_to_block_id([19, 30]) == 8

    assert blocking.get_block_ids_overlapping_bounding_box([11, 23], [17, 30]) == list(
        range(9)
    )


def test_block_ids_in_bounding_box_requires_full_enclosure():
    blocking = bic.Blocking([0, 0], [10, 10], [4, 4])

    assert blocking.get_block_ids_in_bounding_box([4, 4], [10, 10]) == [4, 5, 7, 8]
    assert blocking.get_block_ids_in_bounding_box([5, 5], [10, 10]) == [8]


def test_overlapping_bounding_box_is_dimension_independent():
    blocking = bic.Blocking([0, 0, 0, 0], [6, 6, 6, 6], [3, 3, 3, 3])

    assert blocking.blocks_per_axis == [2, 2, 2, 2]
    assert blocking.get_block_ids_overlapping_bounding_box(
        [2, 2, 2, 2], [4, 4, 4, 4]
    ) == list(range(16))
    assert blocking.get_block_ids_overlapping_bounding_box(
        [0, 0, 0, 0], [3, 3, 3, 3]
    ) == [0]


def test_block_with_halo_and_add_halo_clip_to_roi():
    blocking = bic.Blocking([0, 0], [10, 10], [4, 4])

    block = blocking.get_block_with_halo(0, [2, 1])
    assert block.inner_block.begin == [0, 0]
    assert block.inner_block.end == [4, 4]
    assert block.outer_block.begin == [0, 0]
    assert block.outer_block.end == [6, 5]
    assert block.inner_block_local.begin == [0, 0]
    assert block.inner_block_local.end == [4, 4]

    asymmetric = blocking.add_halo(blocking.get_block(4), [1, 2], [3, 4])
    assert asymmetric.outer_block.begin == [3, 2]
    assert asymmetric.outer_block.end == [10, 10]
    assert asymmetric.inner_block_local.begin == [1, 2]
    assert asymmetric.inner_block_local.end == [5, 6]


def test_local_overlaps_return_local_coordinates_or_none():
    blocking = bic.Blocking([0, 0], [10, 10], [5, 5])

    overlaps = blocking.get_local_overlaps(0, 1, [1, 1])
    assert overlaps == ([0, 4], [6, 6], [0, 0], [6, 2])

    assert blocking.get_local_overlaps(0, 3, [0, 0]) is None


def test_block_ids_in_slice_uses_first_axis_and_halo():
    blocking = bic.Blocking([0, 0, 0], [6, 4, 4], [3, 2, 2])

    assert blocking.get_block_ids_in_slice(3, [0, 0, 0]) == list(range(4, 8))
    assert blocking.get_block_ids_in_slice(3, [1, 0, 0]) == list(range(8))


def test_invalid_inputs_raise_clear_errors():
    with pytest.raises(ValueError, match="block_shape values must be positive"):
        bic.Blocking([0, 0], [10, 10], [4, 0])

    blocking = bic.Blocking([0, 0], [10, 10], [4, 4])
    with pytest.raises(IndexError, match="block_id is out of range"):
        blocking.get_block(9)
    with pytest.raises(IndexError, match="coordinates must lie inside"):
        blocking.coordinates_to_block_id([10, 0])
    with pytest.raises(ValueError, match="box_end must be >= box_begin"):
        blocking.get_block_ids_overlapping_bounding_box([5, 0], [4, 1])
