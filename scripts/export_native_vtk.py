#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.native_data import resolve_sample_or_metadata
from palpation_sim.vtk_exports import write_sample_vtk_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export palpation samples to native VTK/ParaView files.")
    parser.add_argument("sample", type=Path, help="Sample .npz, metadata/GT JSON, run directory, or sample stem.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional search directory for sample stems.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: <sample-dir>/vtk.")
    parser.add_argument("--no-tet-mesh", action="store_true", help="Skip .vtu tet mesh export.")
    parser.add_argument("--extract-surface", action="store_true", help="Also write extracted tet boundary surface .vtp.")
    parser.add_argument("--no-scan-points", action="store_true", help="Skip scan point cloud .vtp export.")
    parser.add_argument("--analytic-resolution", type=int, default=56, help="Analytic lump surface resolution.")
    parser.add_argument("--timeseries-row", type=int, default=None, help="Press row for visual surface time-series export.")
    parser.add_argument("--timeseries-col", type=int, default=None, help="Press col for visual surface time-series export.")
    parser.add_argument("--timeseries-stride", type=int, default=1, help="Write every Nth press timestep.")
    parser.add_argument("--timeseries-surface-resolution", type=int, default=96)
    args = parser.parse_args()

    sample_path, metadata_path = resolve_sample_or_metadata(args.sample, args.data_dir)
    if sample_path is None:
        raise SystemExit("A .npz sample is required for VTK export.")
    out_dir = args.out_dir or (sample_path.parent / "vtk")
    if (args.timeseries_row is None) ^ (args.timeseries_col is None):
        raise SystemExit("--timeseries-row and --timeseries-col must be supplied together.")
    timeseries_press = None
    if args.timeseries_row is not None and args.timeseries_col is not None:
        timeseries_press = (int(args.timeseries_row), int(args.timeseries_col))

    outputs = write_sample_vtk_bundle(
        sample_path,
        out_dir,
        metadata_path=metadata_path,
        include_tet_mesh=not args.no_tet_mesh,
        extract_surface=args.extract_surface,
        include_scan_points=not args.no_scan_points,
        analytic_lump_resolution=args.analytic_resolution,
        timeseries_press=timeseries_press,
        timeseries_stride=args.timeseries_stride,
        timeseries_surface_resolution=args.timeseries_surface_resolution,
    )
    print(f"wrote VTK bundle: {out_dir}")
    for key, value in outputs.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} files")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
