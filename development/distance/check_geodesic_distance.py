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

Not part of the pytest suite; requires scikit-fmm, pygeodesic and scipy.
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


def make_mesh(n_points=600, radius=5.0, seed=0):
    """A closed sphere mesh from the convex hull of points on the sphere.

    scipy's ConvexHull gives a clean manifold triangulation (pygeodesic can
    overflow on the degenerate faces marching_cubes produces).
    """
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_points, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius
    verts = np.ascontiguousarray(pts, dtype=np.float64)
    faces = np.ascontiguousarray(ConvexHull(pts).simplices, dtype=np.int64)
    n_verts = len(verts)
    sources = np.array([0], dtype=np.int64)
    points = np.linspace(0, n_verts - 1, num=6, dtype=np.int64)
    return verts, faces, sources, points


# --------------------------------------------------------------------------- #
# comparison harness
# --------------------------------------------------------------------------- #


def compare(case, atol, rtol, mesh_rtol):
    """Compare one case; ``case`` is ``(name, kind, reference_fn, bic_fn)``.

    Masks use the same first-order scheme as scikit-fmm, so after removing the
    ~0.5-cell level-set seed offset (see ``_geodesic_reference``) the residual is
    tiny. Meshes compare our first-order FMM to the exact pygeodesic distances
    via relative error (looser: first-order + no obtuse unfolding yet).
    """
    name, kind, reference_fn, bic_fn = case
    try:
        ref = np.asarray(reference_fn(), dtype=np.float64)
    except ImportError as error:  # backend missing
        return {"name": name, "status": "NO-REF", "metric": str(error).split("(")[0].strip()}

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

    inf_match = np.array_equal(~finite, ~np.isfinite(got))
    both = finite & np.isfinite(got)
    if not both.any():
        row["status"] = "OK" if inf_match else "FAIL"
        row["metric"] = "no finite overlap"
        return row

    if kind == "mask":
        away = both & (ref > 1.0)  # skip the seed neighbourhood / matrix diagonal
        sel = away if away.any() else both
        diff = got[sel] - ref[sel]
        offset = float(np.median(diff))
        max_resid = float(np.abs(diff - offset).max())
        scale = float(ref[sel].max())
        ok = inf_match and max_resid < (rtol * scale + atol)
        row["metric"] = f"offset={offset:+.3f} resid_max={max_resid:.2e}"
    else:  # mesh, vs exact pygeodesic
        reach = both & (ref > 1e-9)  # exclude the sources (ref == 0)
        rel = np.abs(got[reach] - ref[reach]) / ref[reach]
        mean_rel, p95, mx = float(rel.mean()), float(np.percentile(rel, 95)), float(rel.max())
        ok = inf_match and mean_rel < 0.06 and p95 < mesh_rtol
        row["metric"] = f"rel mean={mean_rel:.3f} p95={p95:.3f} max={mx:.3f}"

    row["status"] = "OK" if ok else "FAIL"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-2, help="mask absolute residual floor")
    parser.add_argument("--rtol", type=float, default=2e-2, help="mask residual vs max distance")
    parser.add_argument("--mesh-rtol", type=float, default=0.12, help="mesh p95 relative error")
    args = parser.parse_args()

    mask2d, src2d, pts2d = make_mask_2d()
    mask3d, src3d, pts3d = make_mask_3d()
    try:
        verts, faces, src_v, pts_v = make_mesh()
        mesh_ok = True
    except ImportError as error:
        sys.stderr.write(f"scipy not installed, skipping mesh cases: {error}\n")
        mesh_ok = False

    cases = [
        (
            "mask2d/field", "mask",
            lambda: reference_geodesic_field_mask(mask2d, src2d),
            lambda: bic.distance.geodesic_distance_field(mask2d, src2d),
        ),
        (
            "mask2d/pairwise", "mask",
            lambda: reference_geodesic_distances_mask(mask2d, pts2d),
            lambda: bic.distance.geodesic_distances(mask2d, pts2d),
        ),
        (
            "mask3d/field", "mask",
            lambda: reference_geodesic_field_mask(mask3d, src3d),
            lambda: bic.distance.geodesic_distance_field(mask3d, src3d),
        ),
        (
            "mask3d/pairwise", "mask",
            lambda: reference_geodesic_distances_mask(mask3d, pts3d),
            lambda: bic.distance.geodesic_distances(mask3d, pts3d),
        ),
    ]
    if mesh_ok:
        cases += [
            (
                "mesh/field", "mesh",
                lambda: reference_geodesic_field_mesh(verts, faces, src_v),
                lambda: bic.distance.geodesic_distance_field_mesh(verts, faces, src_v),
            ),
            (
                "mesh/pairwise", "mesh",
                lambda: reference_geodesic_distances_mesh(verts, faces, pts_v),
                lambda: bic.distance.geodesic_distances_mesh(verts, faces, pts_v),
            ),
        ]

    header = f"{'case':>16} {'status':>8} {'ref_shape':>14} {'ref_range':>20} {'agreement':>34}"
    print(header)
    print("-" * len(header))
    any_fail = False
    for case in cases:
        r = compare(case, args.atol, args.rtol, args.mesh_rtol)
        rng = r.get("ref_range")
        rng_s = f"[{rng[0]:.3f}, {rng[1]:.3f}]" if rng else "-"
        shape_s = str(r.get("ref_shape", "-"))
        metric_s = r.get("metric", "-")
        print(f"{r['name']:>16} {r['status']:>8} {shape_s:>14} {rng_s:>20} {metric_s:>34}")
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
