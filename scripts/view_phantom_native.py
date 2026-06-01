#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.native_data import (
    load_phantom_scan_material_lumps,
    resolve_sample_or_metadata,
    sample_id_from_paths,
)
from palpation_sim.visual_geometry import (
    LUMP_COLORS,
    analytic_lump_polydata,
    phantom_box_polydata,
    require_pyvista,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a native PyVista viewer for analytic phantom attributes.")
    parser.add_argument("phantom", type=Path, help="Sample .npz, *_gt.json, metadata.json, run directory, or sample stem.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional search directory for sample stems.")
    parser.add_argument("--resolution", type=int, default=56, help="Marching-cubes resolution for analytic lump surfaces.")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Visual z scale.")
    parser.add_argument("--screenshot", type=Path, default=None, help="Write a screenshot instead of only opening the viewer.")
    parser.add_argument("--off-screen", action="store_true", help="Render off-screen; useful with --screenshot.")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw lump labels.")
    args = parser.parse_args()

    pv = require_pyvista()
    sample_path, metadata_path = resolve_sample_or_metadata(args.phantom, args.data_dir)
    phantom, _scan, _material, lumps, _metadata = load_phantom_scan_material_lumps(
        sample_path=sample_path,
        metadata_path=metadata_path,
    )
    title = sample_id_from_paths(sample_path, metadata_path)

    plotter = pv.Plotter(off_screen=args.off_screen)
    plotter.set_background("#101316")
    box = phantom_box_polydata(phantom)
    plotter.add_mesh(box, color="#8fc7e8", opacity=0.12, show_edges=True, edge_color="#bcd8e8", label="phantom")

    label_points = []
    label_text = []
    for idx, lump in enumerate(lumps):
        mesh = analytic_lump_polydata(lump, resolution=args.resolution)
        color = LUMP_COLORS[idx % len(LUMP_COLORS)]
        plotter.add_mesh(
            mesh,
            color=color,
            opacity=0.74,
            smooth_shading=True,
            specular=0.22,
            label=f"{idx}: {lump.shape} {lump.stiffness_multiplier:.1f}x",
        )
        plotter.add_mesh(mesh.extract_feature_edges(), color="#111111", line_width=1, opacity=0.28)
        label_points.append(lump.center)
        label_text.append(f"{idx} {lump.shape}\n{lump.stiffness_multiplier:.1f}x")

    if label_points and not args.no_labels:
        plotter.add_point_labels(
            np.asarray(label_points, dtype=np.float32),
            label_text,
            point_size=8,
            font_size=12,
            text_color="white",
            shape_opacity=0.35,
        )

    plotter.add_axes()
    plotter.show_grid(color="#68727a")
    plotter.set_scale(zscale=float(args.z_scale))
    plotter.add_text(
        f"{title}: analytic inclusion surfaces",
        position="upper_left",
        color="white",
        font_size=11,
    )
    radius = max(phantom.size_x, phantom.size_y, phantom.height)
    plotter.camera_position = [
        (radius * 0.95, -radius * 1.45, radius * 0.9),
        (0.0, 0.0, phantom.height * 0.45),
        (0.0, 0.0, 1.0),
    ]
    plotter.show(screenshot=str(args.screenshot) if args.screenshot else None)


if __name__ == "__main__":
    main()
