"""Serialize the ISBI lifted-multicut problem to a .npz file.

Runs the same build pipeline as `lifted_multicut_from_affinities.py` up to
the LiftedMulticutObjective inputs, then writes them to a single `.npz` file
with fields:

    n_nodes      : scalar uint64
    local_uvs    : (n_local, 2)  uint64
    local_costs  : (n_local,)    float64
    lifted_uvs   : (n_lifted, 2) uint64
    lifted_costs : (n_lifted,)   float64

The resulting file is what the development comparison scripts in
`development/graph/lifted_multicut/` load.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _lifted_problem import build_lifted_problem, load_affinity_problem


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = THIS_DIR / "lifted_multicut_problem.npz"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the ISBI lifted multicut problem (RAG + local + lifted edges + "
            "costs) and serialize it to a .npz file."
        )
    )
    parser.add_argument("--ndim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--z-slice", type=int, default=0)
    parser.add_argument("--local-threshold", type=float, default=0.1)
    parser.add_argument("--lifted-threshold", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--watershed-min-distance", type=int, default=5)
    parser.add_argument("--watershed-grid-spacing", type=int, default=12)
    parser.add_argument("--max-markers", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    affinity_problem = load_affinity_problem(args.ndim, args.z_slice)
    lifted_problem = build_lifted_problem(
        affinity_problem,
        local_threshold=args.local_threshold,
        lifted_threshold=args.lifted_threshold,
        number_of_threads=args.threads,
        watershed_min_distance=args.watershed_min_distance,
        watershed_grid_spacing=args.watershed_grid_spacing,
        max_markers=args.max_markers,
    )

    local_uvs = np.ascontiguousarray(lifted_problem.local_uvs.astype(np.uint64, copy=False))
    local_costs = np.ascontiguousarray(
        lifted_problem.local_costs.astype(np.float64, copy=False)
    )
    lifted_uvs = np.ascontiguousarray(lifted_problem.lifted_uvs.astype(np.uint64, copy=False))
    lifted_costs = np.ascontiguousarray(
        lifted_problem.lifted_costs.astype(np.float64, copy=False)
    )

    n_nodes = np.uint64(lifted_problem.number_of_nodes)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        n_nodes=n_nodes,
        local_uvs=local_uvs,
        local_costs=local_costs,
        lifted_uvs=lifted_uvs,
        lifted_costs=lifted_costs,
    )

    print(f"Wrote lifted multicut problem to {output}")
    print(f"  number of nodes:        {int(n_nodes)}")
    print(f"  number of local edges:  {local_uvs.shape[0]}")
    print(f"  number of lifted edges: {lifted_uvs.shape[0]}")
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
