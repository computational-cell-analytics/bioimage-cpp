"""Serialize a grid-graph ISBI lifted-multicut problem to a .npz file.

This builds a lifted multicut problem directly on the regular pixel/voxel grid
from the ISBI example affinities. It uses ``grid_graph`` plus
``grid_affinity_features_with_lifted`` and writes the same fields as
``serialize_lifted_problem.py``:

    n_nodes      : scalar uint64
    local_uvs    : (n_local, 2)  uint64
    local_costs  : (n_local,)    float64
    lifted_uvs   : (n_lifted, 2) uint64
    lifted_costs : (n_lifted,)   float64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bioimage_cpp as bic
from bioimage_cpp._data import load_isbi_affinities


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = THIS_DIR / "grid_lifted_multicut_problem.npz"


def load_affinities(
    ndim: int,
    spatial_shape: tuple[int, ...],
    z_slice: int,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    affinities, offsets = load_isbi_affinities()
    offsets = [tuple(int(v) for v in offset) for offset in offsets]

    if ndim == 2:
        y, x = spatial_shape
        channels_2d = [index for index, offset in enumerate(offsets) if offset[0] == 0]
        affinities = affinities[channels_2d, z_slice, :y, :x]
        offsets = [offsets[index][1:] for index in channels_2d]
    elif ndim == 3:
        z, y, x = spatial_shape
        affinities = affinities[:, :z, :y, :x]
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    if affinities.shape[1:] != spatial_shape:
        raise ValueError(
            f"requested spatial shape {spatial_shape} exceeds available data; "
            f"extracted shape is {affinities.shape[1:]}"
        )

    return np.ascontiguousarray(affinities, dtype=np.float32), offsets


def parse_spatial_shape(values: list[int] | None, ndim: int) -> tuple[int, ...]:
    if values is None:
        return (256, 256) if ndim == 2 else (16, 256, 256)
    if len(values) != ndim:
        raise ValueError(
            f"--spatial-shape must contain {ndim} values for {ndim}D, "
            f"got {len(values)}"
        )
    if any(value <= 0 for value in values):
        raise ValueError("--spatial-shape values must be positive")
    return tuple(values)


def build_grid_lifted_problem(
    affinities: np.ndarray,
    offsets: list[tuple[int, ...]],
    *,
    local_threshold: float,
    lifted_threshold: float,
):
    graph = bic.graph.grid_graph(affinities.shape[1:])
    local_weights, valid_edges, lifted_uvs, lifted_weights, _ = (
        bic.graph.grid_affinity_features_with_lifted(graph, affinities, offsets)
    )
    if not np.all(valid_edges):
        invalid = int(valid_edges.size - np.count_nonzero(valid_edges))
        raise RuntimeError(
            "local affinity offsets did not cover all grid graph edges; "
            f"{invalid} edges are missing"
        )

    local_costs = (local_threshold - local_weights).astype(np.float64, copy=False)
    lifted_costs = (lifted_threshold - lifted_weights).astype(np.float64, copy=False)
    return (
        int(graph.number_of_nodes),
        graph.uv_ids(),
        np.ascontiguousarray(local_costs),
        np.ascontiguousarray(lifted_uvs.astype(np.uint64, copy=False)),
        np.ascontiguousarray(lifted_costs),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build an ISBI lifted multicut problem directly on a regular grid "
            "graph and serialize it to a .npz file."
        )
    )
    parser.add_argument("--ndim", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--spatial-shape",
        type=int,
        nargs="+",
        default=None,
        metavar=("Y", "X"),
        help=(
            "Spatial crop shape. Pass Y X for 2D or Z Y X for 3D. "
            "Defaults to 256 256 for 2D and 16 256 256 for 3D."
        ),
    )
    parser.add_argument(
        "--z-slice",
        type=int,
        default=0,
        help="Z slice used for 2D extraction.",
    )
    parser.add_argument("--local-threshold", type=float, default=0.1)
    parser.add_argument("--lifted-threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    spatial_shape = parse_spatial_shape(args.spatial_shape, args.ndim)
    affinities, offsets = load_affinities(
        args.ndim, spatial_shape, args.z_slice
    )
    n_nodes, local_uvs, local_costs, lifted_uvs, lifted_costs = (
        build_grid_lifted_problem(
            affinities,
            offsets,
            local_threshold=args.local_threshold,
            lifted_threshold=args.lifted_threshold,
        )
    )

    local_uvs = np.ascontiguousarray(local_uvs.astype(np.uint64, copy=False))
    n_nodes_array = np.uint64(n_nodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        n_nodes=n_nodes_array,
        local_uvs=local_uvs,
        local_costs=local_costs,
        lifted_uvs=lifted_uvs,
        lifted_costs=lifted_costs,
    )

    print(f"Wrote grid lifted multicut problem to {args.output}")
    print(f"  ndim:                   {args.ndim}")
    print(f"  spatial shape:          {spatial_shape}")
    print(f"  number of nodes:        {n_nodes}")
    print(f"  number of local edges:  {local_uvs.shape[0]}")
    print(f"  number of lifted edges: {lifted_uvs.shape[0]}")
    if local_costs.size:
        print(
            f"  local cost range:       [{float(local_costs.min()):+.3f}, "
            f"{float(local_costs.max()):+.3f}]"
        )
    if lifted_costs.size:
        print(
            f"  lifted cost range:      [{float(lifted_costs.min()):+.3f}, "
            f"{float(lifted_costs.max()):+.3f}]"
        )


if __name__ == "__main__":
    main()
