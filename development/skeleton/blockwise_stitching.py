"""Minimal serial harness for blockwise TEASAR and exact stitching.

This is deliberately development-only orchestration. It performs no I/O,
parallel scheduling, retries, or artifact serialization; those responsibilities
belong to downstream block-processing systems.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import bioimage_cpp as bic


dist = bic.skeleton.distributed


def _processing_blocks(volume: np.ndarray, block_shape: Sequence[int]):
    blocking = bic.utils.Blocking([0, 0, 0], list(volume.shape), list(block_shape))
    for block_id in range(blocking.number_of_blocks):
        core = blocking.get_block(block_id)
        begin = [int(value) for value in core.begin]
        end = [int(value) for value in core.end]
        faces = []
        for axis in range(3):
            if blocking.get_neighbor_id(block_id, axis, lower=True) >= 0:
                faces.append((axis, "low"))
            if blocking.get_neighbor_id(block_id, axis, lower=False) >= 0:
                faces.append((axis, "high"))
                end[axis] += 1
        slices = tuple(slice(begin[axis], end[axis]) for axis in range(3))
        yield blocking, block_id, np.ascontiguousarray(volume[slices]), begin, faces


def _unique_coordinates(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.concatenate(parts, axis=0), axis=0)


def _unique_target_map(parts: list[dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    labels = sorted({label for part in parts for label in part})
    return {
        label: _unique_coordinates([part[label] for part in parts if label in part])
        for label in labels
    }


def _assert_matching_interfaces(blocking, per_face) -> None:
    for block_id in range(blocking.number_of_blocks):
        for axis in range(3):
            neighbor = blocking.get_neighbor_id(block_id, axis, lower=False)
            if neighbor < 0:
                continue
            left = per_face[(block_id, axis, "high")]
            right = per_face[(neighbor, axis, "low")]
            if isinstance(left, dict):
                if left.keys() != right.keys():
                    raise AssertionError(
                        f"target labels disagree across blocks {block_id}/{neighbor}"
                    )
                for label in left:
                    np.testing.assert_array_equal(left[label], right[label])
            else:
                np.testing.assert_array_equal(left, right)


def run_blockwise_binary(
    mask: np.ndarray,
    block_shape: Sequence[int],
    *,
    spacing=(1.0, 1.0, 1.0),
    remove_cycles: bool = False,
):
    """Run the complete binary block pipeline serially and return a lattice graph."""
    fragments = []
    per_face = {}
    blocking = None
    for blocking, block_id, block, origin, faces in _processing_blocks(
        np.asarray(mask), block_shape
    ):
        face_targets = []
        for axis, side in faces:
            targets = dist.block_border_targets(
                block,
                [(axis, side)],
                origin=origin,
                spacing=spacing,
                number_of_threads=1,
            )
            per_face[(block_id, axis, side)] = targets
            face_targets.append(targets)
        targets = _unique_coordinates(face_targets)
        fragments.append(
            dist.block_teasar(
                block,
                open_faces=faces,
                origin=origin,
                required_targets=targets,
                spacing=spacing,
                number_of_threads=1,
            )
        )
    if blocking is None:
        raise ValueError("mask must be a three-dimensional array")
    _assert_matching_interfaces(blocking, per_face)
    merged = dist.merge_block_skeletons(fragments)
    if remove_cycles:
        merged = dist.minimum_spanning_forest(merged, spacing=spacing)
    return merged


def run_blockwise_labels(
    labels: np.ndarray,
    block_shape: Sequence[int],
    *,
    background: int = 0,
    spacing=(1.0, 1.0, 1.0),
    remove_cycles: bool = False,
):
    """Run the labeled block pipeline serially and return lattice graphs by label."""
    fragment_maps = []
    per_face = {}
    blocking = None
    for blocking, block_id, block, origin, faces in _processing_blocks(
        np.asarray(labels), block_shape
    ):
        face_targets = []
        for axis, side in faces:
            targets = dist.block_border_targets_labels(
                block,
                [(axis, side)],
                origin=origin,
                background=background,
                spacing=spacing,
                number_of_threads=1,
            )
            per_face[(block_id, axis, side)] = targets
            face_targets.append(targets)
        targets = _unique_target_map(face_targets)
        fragment_maps.append(
            dist.block_teasar_labels(
                block,
                open_faces=faces,
                origin=origin,
                required_targets=targets,
                background=background,
                spacing=spacing,
                number_of_threads=1,
            )
        )
    if blocking is None:
        raise ValueError("labels must be a three-dimensional array")
    _assert_matching_interfaces(blocking, per_face)
    merged = dist.merge_block_skeleton_maps(fragment_maps)
    if remove_cycles:
        merged = {
            label: dist.minimum_spanning_forest(fragment, spacing=spacing)
            for label, fragment in merged.items()
        }
    return merged


if __name__ == "__main__":
    example = np.zeros((17, 17, 25), dtype=np.uint8)
    example[8, 8, 2:23] = 1
    result = run_blockwise_binary(example, (9, 9, 8), remove_cycles=True)
    print(
        f"stitched {len(result[0])} lattice vertices and {len(result[1])} edges "
        "from serial processing blocks"
    )
