"""Compare ``bic.mesh.marching_cubes`` to ``skimage.measure.marching_cubes``.

Run from the repository root:

    python development/mesh/check_marching_cubes.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import bioimage_cpp as bic

from _marching_cubes_reference import assert_mesh_matches, reference_marching_cubes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-cases",
        type=int,
        default=1024,
        help="number of deterministic random scalar one-cube configurations to compare (default: 1024)",
    )
    parser.add_argument(
        "--random-volumes",
        type=int,
        default=1000,
        help="number of deterministic small multi-cube stress cases (default: 1000)",
    )
    return parser.parse_args()


def cases():
    box = np.zeros((7, 8, 9), dtype=np.uint8)
    box[1:6, 2:7, 1:8] = 1

    boundary = np.zeros((6, 7, 8), dtype=np.uint8)
    boundary[:3, 2:6, 2:7] = 1

    rng = np.random.default_rng(20260709)
    scalar = rng.normal(size=(8, 9, 10)).astype(np.float32)
    binary = rng.integers(0, 2, size=(8, 9, 10), dtype=np.uint8)
    roi = np.ones_like(binary, dtype=bool)
    roi[:, :2] = False
    degenerate = rng.integers(0, 3, size=(8, 9, 10), dtype=np.uint8)

    positive_background = np.full((6, 7, 8), 2.0, dtype=np.float32)
    positive_background[1:5, 2:6, 2:7] = 4.0

    ambiguous = np.array([(6 >> bit) & 1 for bit in range(8)], dtype=np.uint8).reshape(2, 2, 2)
    return [
        ("box/lewiner", box, 0.5, {"method": "lewiner"}),
        ("box/lorensen", box, 0.5, {"method": "lorensen"}),
        ("scalar/lewiner", scalar, 0.0, {"method": "lewiner"}),
        ("binary/lorensen-step", binary, 0.5, {"method": "lorensen", "step_size": 2}),
        ("masked/lewiner", binary, 0.5, {"mask": roi}),
        ("masked/lewiner-step3", binary, 0.5, {"mask": roi, "step_size": 3}),
        ("degenerate/removed", degenerate, 1.0, {"allow_degenerate": False}),
        ("anisotropic/ascent", box, 0.5, {"spacing": (2.0, 0.5, 3.0), "gradient_direction": "ascent"}),
        ("ambiguous/lewiner", ambiguous, 0.5, {"method": "lewiner"}),
        ("ambiguous/lorensen", ambiguous, 0.5, {"method": "lorensen"}),
        ("boundary/padded", boundary, 0.5, {"pad": True}),
        ("padded/default-level", positive_background, None, {"pad": True}),
    ]


def assert_case(volume: np.ndarray, level: float | None, kwargs: dict[str, object]) -> None:
    actual = bic.mesh.marching_cubes(volume, level, **kwargs)
    reference = reference_marching_cubes(volume, level, **kwargs)
    assert_mesh_matches(actual, reference)


def check_all_binary_cube_configurations() -> None:
    for configuration in range(1, 255):
        volume = np.array(
            [(configuration >> bit) & 1 for bit in range(8)], dtype=np.uint8
        ).reshape(2, 2, 2)
        for method in ("lewiner", "lorensen"):
            assert_case(volume, 0.5, {"method": method})


def check_random_scalar_cubes(n_cases: int) -> None:
    rng = np.random.default_rng(20260710)
    checked = 0
    while checked < n_cases:
        volume = rng.normal(size=(2, 2, 2)).astype(np.float32)
        if np.all(volume <= 0.0) or np.all(volume > 0.0):
            continue
        for method in ("lewiner", "lorensen"):
            assert_case(volume, 0.0, {"method": method})
        checked += 1


def collapsed_coordinate_faces(vertices: np.ndarray, faces: np.ndarray) -> int:
    triangles = vertices[faces]
    collapsed = (
        np.all(triangles[:, 0] == triangles[:, 1], axis=1)
        | np.all(triangles[:, 0] == triangles[:, 2], axis=1)
        | np.all(triangles[:, 1] == triangles[:, 2], axis=1)
    )
    return int(np.count_nonzero(collapsed))


def check_random_volumes(n_cases: int) -> int:
    """Stress masks, strides, and exact-level degeneracies on small volumes.

    Robust transitive duplicate merging intentionally differs from a rare
    scikit-image negative-index remapping quirk. Such differences are counted
    and reported, while invalid indices or collapsed output faces always fail.
    """
    rng = np.random.default_rng(293841)
    levels = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5])
    parity_differences = 0
    for _ in range(n_cases):
        shape = tuple(int(size) for size in rng.integers(3, 9, size=3))
        volume = rng.integers(0, 4, size=shape, dtype=np.uint8)
        volume.flat[0] = 0
        volume.flat[-1] = 3
        level = float(rng.choice(levels))
        kwargs: dict[str, object] = {
            "method": "lewiner" if rng.random() < 0.7 else "lorensen",
            "allow_degenerate": bool(rng.random() < 0.5),
            "step_size": 2 if min(shape) >= 5 and rng.random() < 0.25 else 1,
        }
        if rng.random() < 0.25:
            kwargs["mask"] = rng.random(shape) < 0.75

        try:
            actual = bic.mesh.marching_cubes(volume, level, **kwargs)
        except RuntimeError:
            try:
                reference_marching_cubes(volume, level, **kwargs)
            except RuntimeError:
                continue
            raise AssertionError("bioimage-cpp found no surface but scikit-image did")
        reference = reference_marching_cubes(volume, level, **kwargs)
        if not np.all((actual[1] >= 0) & (actual[1] < len(actual[0]))):
            raise AssertionError("marching cubes returned an invalid face index")
        if not kwargs["allow_degenerate"]:
            if collapsed_coordinate_faces(actual[0], actual[1]) != 0:
                raise AssertionError("degenerate removal left a collapsed-coordinate face")
            try:
                assert_mesh_matches(actual, reference)
            except AssertionError:
                parity_differences += 1
        else:
            assert_mesh_matches(actual, reference)
    return parity_differences


def main() -> int:
    args = parse_args()
    if args.random_cases < 0 or args.random_volumes < 0:
        raise SystemExit("random-cases and random-volumes must be >= 0")
    failed = False
    print(f"{'case':<26} {'vertices':>10} {'faces':>10} {'status':>8}")
    print("-" * 60)
    for name, volume, level, kwargs in cases():
        try:
            actual = bic.mesh.marching_cubes(volume, level, **kwargs)
            reference = reference_marching_cubes(volume, level, **kwargs)
            assert_mesh_matches(actual, reference)
        except Exception as error:
            print(f"{name:<26} {'-':>10} {'-':>10} {'FAIL':>8}  {error}")
            failed = True
            continue
        print(f"{name:<26} {len(actual[0]):>10} {len(actual[1]):>10} {'OK':>8}")
    if not failed:
        try:
            check_all_binary_cube_configurations()
            print(f"{'all binary one-cube cases':<26} {'254':>10} {'508':>10} {'OK':>8}")
            check_random_scalar_cubes(args.random_cases)
            print(f"{'random scalar one-cube cases':<26} {args.random_cases:>10} {2 * args.random_cases:>10} {'OK':>8}")
            differences = check_random_volumes(args.random_volumes)
            detail = f"{differences} intentional reference differences"
            print(
                f"{'random multi-cube cases':<26} {args.random_volumes:>10} "
                f"{differences:>10} {'OK':>8}  {detail}"
            )
        except Exception as error:
            print(f"{'randomized parity':<26} {'-':>10} {'-':>10} {'FAIL':>8}  {error}")
            failed = True
    if failed:
        print("marching-cubes parity check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
