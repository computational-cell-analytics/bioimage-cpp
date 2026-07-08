"""Cross-check bioimage-cpp geodesic distances against scikit-fmm / pygeodesic.

Builds small mask (grid) and triangle-mesh test geometries, computes the
reference geodesic fields and pairwise matrices with the oracles in
``_geodesic_reference`` (scikit-fmm for masks, pygeodesic for meshes), and
compares them against ``bic.distance``:

    geodesic_distance_field        <-> skfmm             (mask field)
    geodesic_distances             <-> skfmm             (mask pairwise)
    geodesic_distance_field_mesh   <-> pygeodesic exact  (mesh field)
    geodesic_distances_mesh        <-> pygeodesic exact  (mesh pairwise)

The bioimage-cpp solvers are not implemented yet, so the bic calls currently
raise ``RuntimeError("... not yet implemented")``. Those cases are reported as
PENDING (not a failure); the script is the acceptance check that turns green
once the algorithms land. Cases whose reference backend is not installed are
reported as NO-REF and also skipped.

Not part of the pytest suite; requires scikit-fmm, pygeodesic and scikit-image.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import bioimage_cpp as bic

from _geodesic_reference import (
    reference_geodesic_distances_mask,
    reference_geodesic_distances_mesh,
    reference_geodesic_field_mask,
    reference_geodesic_field_mesh,
)


# --------------------------------------------------------------------------- #
# test geometries
# --------------------------------------------------------------------------- #


def make_mask_2d():
    """40x40 domain with a horizontal wall + gap; geodesic paths must detour."""
    mask = np.ones((40, 40), dtype=np.uint8)
    mask[20, :34] = 0  # wall across most of the row, gap near the right edge
    sources = np.array([[5, 5]], dtype=np.int64)
    points = np.array([[5, 5], [35, 5], [35, 35], [5, 35]], dtype=np.int64)
    return mask, sources, points


def make_mask_3d():
    """Solid ball on a 30^3 grid."""
    n, c, r = 30, 15, 12
    zz, yy, xx = np.ogrid[:n, :n, :n]
    mask = ((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 <= r * r).astype(np.uint8)
    sources = np.array([[c - r + 1, c, c]], dtype=np.int64)
    points = np.array(
        [[c - r + 1, c, c], [c + r - 1, c, c], [c, c - r + 1, c], [c, c, c + r - 1]],
        dtype=np.int64,
    )
    return mask, sources, points


def make_mesh():
    """Surface of a ball as a triangle mesh via marching cubes."""
    from skimage.measure import marching_cubes

    n, c, r = 32, 16, 12
    zz, yy, xx = np.ogrid[:n, :n, :n]
    vol = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2
    verts, faces, _, _ = marching_cubes(vol.astype(np.float64), level=float(r * r))
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    n_verts = len(verts)
    # a spread of vertex indices for sources / pairwise points
    sources = np.array([0], dtype=np.int64)
    points = np.linspace(0, n_verts - 1, num=6, dtype=np.int64)
    return verts, faces, sources, points


# --------------------------------------------------------------------------- #
# comparison harness
# --------------------------------------------------------------------------- #


def compare(name, reference_fn, bic_fn, atol, rtol):
    try:
        ref = reference_fn()
    except ImportError as error:  # backend missing
        return {"name": name, "status": "NO-REF", "detail": str(error).split("(")[0]}

    ref = np.asarray(ref, dtype=np.float64)
    row = {"name": name, "ref_shape": ref.shape}
    finite = np.isfinite(ref)
    if finite.any():
        row["ref_range"] = (float(ref[finite].min()), float(ref[finite].max()))

    try:
        got = np.asarray(bic_fn(), dtype=np.float64)
    except RuntimeError as error:
        if "not yet implemented" in str(error):
            row["status"] = "PENDING"
            return row
        raise

    # Compare where both are finite. NOTE: near the seeds the scikit-fmm
    # level-set idiom is offset by ~0.5 cell (see _geodesic_reference); widen
    # atol or mask the seed neighbourhood here once the solver lands.
    both_finite = finite & np.isfinite(got)
    inf_match = np.array_equal(~finite, ~np.isfinite(got))
    if both_finite.any():
        diff = np.abs(ref[both_finite] - got[both_finite])
        row["max_abs"] = float(diff.max())
        close = np.allclose(ref[both_finite], got[both_finite], atol=atol, rtol=rtol)
    else:
        row["max_abs"] = 0.0
        close = True
    row["status"] = "OK" if (close and inf_match) else "FAIL"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    mask2d, src2d, pts2d = make_mask_2d()
    mask3d, src3d, pts3d = make_mask_3d()
    try:
        verts, faces, src_v, pts_v = make_mesh()
        mesh_ok = True
    except ImportError as error:
        sys.stderr.write(f"scikit-image not installed, skipping mesh cases: {error}\n")
        mesh_ok = False

    cases = [
        (
            "mask2d/field",
            lambda: reference_geodesic_field_mask(mask2d, src2d),
            lambda: bic.distance.geodesic_distance_field(mask2d, src2d),
        ),
        (
            "mask2d/pairwise",
            lambda: reference_geodesic_distances_mask(mask2d, pts2d),
            lambda: bic.distance.geodesic_distances(mask2d, pts2d),
        ),
        (
            "mask3d/field",
            lambda: reference_geodesic_field_mask(mask3d, src3d),
            lambda: bic.distance.geodesic_distance_field(mask3d, src3d),
        ),
        (
            "mask3d/pairwise",
            lambda: reference_geodesic_distances_mask(mask3d, pts3d),
            lambda: bic.distance.geodesic_distances(mask3d, pts3d),
        ),
    ]
    if mesh_ok:
        cases += [
            (
                "mesh/field",
                lambda: reference_geodesic_field_mesh(verts, faces, src_v),
                lambda: bic.distance.geodesic_distance_field_mesh(verts, faces, src_v),
            ),
            (
                "mesh/pairwise",
                lambda: reference_geodesic_distances_mesh(verts, faces, pts_v),
                lambda: bic.distance.geodesic_distances_mesh(verts, faces, pts_v),
            ),
        ]

    header = f"{'case':>16} {'status':>8} {'ref_shape':>14} {'ref_range':>22} {'max_abs':>10}"
    print(header)
    print("-" * len(header))
    any_fail = False
    for name, ref_fn, bic_fn in cases:
        r = compare(name, ref_fn, bic_fn, args.atol, args.rtol)
        rng = r.get("ref_range")
        rng_s = f"[{rng[0]:.3f}, {rng[1]:.3f}]" if rng else "-"
        shape_s = str(r.get("ref_shape", "-"))
        max_abs_s = f"{r['max_abs']:.3e}" if "max_abs" in r else "-"
        print(f"{name:>16} {r['status']:>8} {shape_s:>14} {rng_s:>22} {max_abs_s:>10}")
        if r["status"] == "FAIL":
            any_fail = True

    print()
    if any_fail:
        print("FAIL: at least one case disagrees with the reference.", file=sys.stderr)
        sys.exit(1)
    print(
        "No mismatches. (PENDING = bic solver not implemented yet; "
        "NO-REF = reference backend not installed.)"
    )


if __name__ == "__main__":
    main()
