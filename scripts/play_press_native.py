#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.native_press_player import run_press_player


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the native PySide/PyVista press process player.")
    parser.add_argument("sample", type=Path, help="Sample .npz, metadata/GT JSON, run directory, or sample stem.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional search directory for sample stems.")
    parser.add_argument("--surface-resolution", type=int, default=96, help="Resolution of the deforming visual surface.")
    parser.add_argument("--start-row", type=int, default=None)
    parser.add_argument("--start-col", type=int, default=None)
    parser.add_argument(
        "--mesh-style",
        choices=["continuous", "discrete", "both"],
        default="continuous",
        help="3D scene style: continuous proxy, FEM/discrete mesh layers, or both.",
    )
    parser.add_argument(
        "--tet-stride",
        type=int,
        default=64,
        help="Show every Nth tetrahedron in the discrete wire layer. Use 1 for all tets.",
    )
    parser.add_argument(
        "--vertex-stride",
        type=int,
        default=16,
        help="Show every Nth mesh vertex in the discrete vertex layer. Use 1 for all vertices.",
    )
    args = parser.parse_args()
    run_press_player(
        args.sample,
        data_dir=args.data_dir,
        surface_resolution=args.surface_resolution,
        start_row=args.start_row,
        start_col=args.start_col,
        mesh_style=args.mesh_style,
        tet_stride=args.tet_stride,
        vertex_stride=args.vertex_stride,
    )


if __name__ == "__main__":
    main()
