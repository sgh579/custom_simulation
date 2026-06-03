from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.config import MaterialConfig, PhantomConfig, ScanConfig
from palpation_sim.exports import (
    _add_material,
    _add_mesh_node,
    _compact_indexed_geometry,
    _lump_color,
    _selected_tet_boundary_faces,
    _selected_tet_wire_geometry,
    _surface_wire_geometry,
    build_ground_truth_metadata,
    write_ground_truth_metadata,
    write_press_records,
)
from palpation_sim.features import extract_feature_map
from palpation_sim.newton_vbd import NewtonVBDPalpationSimulator
from palpation_sim.phantom import LumpSpec, create_structured_tet_mesh, material_arrays_for_lumps


MM = 1.0e-3
PHANTOM_SIZE_M = (80.0 * MM, 80.0 * MM, 25.0 * MM)
CYLINDER_RADIUS_M = 10.0 * MM
CYLINDER_HALF_HEIGHT_M = 2.5 * MM
STIFFNESS_MULTIPLIER = 100.0
FIXED_LUMP_CENTERS_M = (
    (-20.0 * MM, 20.0 * MM, 6.5 * MM),
    (20.0 * MM, 20.0 * MM, 10.5 * MM),
    (-20.0 * MM, -20.0 * MM, 14.5 * MM),
    (20.0 * MM, -20.0 * MM, 18.5 * MM),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed four-cylinder palpation POC.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/fixed_four_cylinder_poc"))
    parser.add_argument("--allow-newton-missing", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use tiny meshes/grids for a quick script check.")
    parser.add_argument("--no-press-records", action="store_true")
    parser.add_argument("--no-visualization", action="store_true")

    parser.add_argument("--newton-cells-xy", type=int, default=32)
    parser.add_argument("--newton-cells-z", type=int, default=10)
    parser.add_argument("--newton-grid", type=int, default=5)
    parser.add_argument("--newton-substeps", type=int, default=3)
    parser.add_argument("--newton-vbd-iterations", type=int, default=5)
    parser.add_argument("--newton-root", type=Path, default=Path("/home/guoheng/newton"))
    parser.add_argument("--newton-device", type=str, default="auto")
    parser.add_argument("--newton-k-mu", type=float, default=MaterialConfig.k_mu)
    parser.add_argument("--newton-k-lambda", type=float, default=MaterialConfig.k_lambda)
    parser.add_argument("--newton-soft-contact-ke", type=float, default=MaterialConfig.soft_contact_ke)
    parser.add_argument("--newton-soft-contact-margin-mm", type=float, default=1.0)

    parser.add_argument("--press-steps", type=int, default=64)
    parser.add_argument("--max-indentation-mm", type=float, default=16.0)
    parser.add_argument("--probe-diameter-mm", type=float, default=8.0)
    parser.add_argument("--edge-margin-mm", type=float, default=4.0)
    args = parser.parse_args()
    if args.smoke:
        _apply_smoke_overrides(args)

    start = time.time()
    material = MaterialConfig()
    lumps = _fixed_lumps()
    _validate_lumps(lumps)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, object]] = []
    try:
        run_summaries.append(_run_newton(args, material, lumps))
    except Exception as exc:
        if not args.allow_newton_missing:
            raise
        failure = {
            "run": "newton_poc",
            "status": "skipped",
            "reason": f"{type(exc).__name__}: {exc}",
        }
        run_summaries.append(failure)
        _write_json(args.out_dir / "newton_poc_skipped.json", failure)

    _write_json(
        args.out_dir / "summary.json",
        {
            "schema_version": 1,
            "description": "Fixed four-cylinder Newton/VBD phantom palpation proof of concept.",
            "elapsed_seconds": time.time() - start,
            "units": "meters",
            "phantom_size_m": list(PHANTOM_SIZE_M),
            "cylinder_radius_m": CYLINDER_RADIUS_M,
            "cylinder_half_height_m": CYLINDER_HALF_HEIGHT_M,
            "lump_stiffness_multiplier": STIFFNESS_MULTIPLIER,
            "lumps": [lump.to_dict(_phantom(1, 1, 1)) for lump in lumps],
            "runs": run_summaries,
        },
    )
    print(f"done: outputs in {args.out_dir}")


def _apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.out_dir = args.out_dir / "smoke"
    args.newton_cells_xy = min(args.newton_cells_xy, 8)
    args.newton_cells_z = min(args.newton_cells_z, 5)
    args.newton_grid = min(args.newton_grid, 2)
    args.press_steps = min(args.press_steps, 5)


def _run_newton(
    args: argparse.Namespace,
    material: MaterialConfig,
    lumps: Sequence[LumpSpec],
) -> dict[str, object]:
    run_dir = args.out_dir / "newton_poc"
    newton_material = _newton_material(args, material)
    phantom = _phantom(args.newton_cells_xy, args.newton_cells_xy, args.newton_cells_z)
    scan = _scan(
        grid_h=args.newton_grid,
        grid_w=args.newton_grid,
        press_steps=args.press_steps,
        max_indentation_mm=args.max_indentation_mm,
        probe_diameter_mm=args.probe_diameter_mm,
        edge_margin_mm=args.edge_margin_mm,
        soft_contact_margin_mm=args.newton_soft_contact_margin_mm,
        substeps=args.newton_substeps,
        vbd_iterations=args.newton_vbd_iterations,
    )
    print("building Newton POC mesh...", flush=True)
    mesh = create_structured_tet_mesh(phantom)
    _, _, _, tet_lump_mask, tet_lump_id = material_arrays_for_lumps(mesh, newton_material, lumps)
    _require_nonempty_lump_tets(tet_lump_id, len(lumps), "newton_poc")

    print("generating Newton/VBD neo-Hookean-like response...", flush=True)
    simulator = NewtonVBDPalpationSimulator(
        phantom,
        newton_material,
        scan,
        newton_root=args.newton_root,
        device=args.newton_device,
    )
    sample = simulator.run_sample(lumps)
    return _write_run_outputs(
        run_dir=run_dir,
        sample_id="fixed_four_cylinder_newton_poc",
        backend_label="newton_vbd",
        sample=sample,
        phantom=phantom,
        material=newton_material,
        scan=scan,
        lumps=lumps,
        mesh=mesh,
        tet_lump_mask=tet_lump_mask,
        tet_lump_id=tet_lump_id,
        args=args,
    )


def _write_run_outputs(
    *,
    run_dir: Path,
    sample_id: str,
    backend_label: str,
    sample: dict[str, object],
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ScanConfig,
    lumps: Sequence[LumpSpec],
    mesh: object,
    tet_lump_mask: np.ndarray,
    tet_lump_id: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    run_start = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    npz_path = run_dir / "fixed_four_cylinder_sample.npz"
    metadata_path = run_dir / "metadata.json"
    gltf_path = None if args.no_visualization else run_dir / "phantom_mesh.gltf"
    animation_path = None if args.no_visualization else run_dir / "scan_animation.html"
    press_dir = None if args.no_press_records else run_dir / "press_records"

    sample["features"] = extract_feature_map(np.asarray(sample["presses"], dtype=np.float32))
    sample["tet_lump_mask"] = np.asarray(tet_lump_mask, dtype=np.uint8)
    sample["tet_lump_id"] = np.asarray(tet_lump_id, dtype=np.int32)
    sample["mesh_vertices"] = np.asarray(mesh.vertices, dtype=np.float32)
    sample["mesh_tets"] = np.asarray(mesh.tets, dtype=np.int32)
    sample["phantom_json"] = json.dumps(phantom.to_dict())
    sample["material_json"] = json.dumps(material.to_dict())
    sample["scan_json"] = json.dumps(scan.to_dict())
    sample["mesh_json"] = json.dumps(_mesh_summary(phantom, mesh, tet_lump_mask, tet_lump_id, len(lumps)))

    print(f"writing {npz_path}...", flush=True)
    np.savez_compressed(npz_path, **sample)

    if gltf_path is not None:
        print(f"writing {gltf_path}...", flush=True)
        _write_fast_phantom_gltf(gltf_path, phantom, lumps, material, mesh, tet_lump_id)
    if animation_path is not None:
        print(f"writing {animation_path}...", flush=True)
        _write_surface_scan_animation_html(animation_path, phantom, scan, sample, lumps, mesh, tet_lump_id)
    if press_dir is not None:
        print(f"writing press records in {press_dir}...", flush=True)
        write_press_records(press_dir, sample, sample_id=sample_id, split=backend_label)

    curve_summary_path = run_dir / "curve_summary.csv"
    _write_curve_summary(curve_summary_path, sample)
    metadata = build_ground_truth_metadata(
        sample_id=sample_id,
        split=backend_label,
        phantom=phantom,
        material=material,
        scan=scan,
        lumps=lumps,
        sample=sample,
        npz_path=npz_path,
        gltf_path=gltf_path,
        press_records_dir=press_dir,
        scan_animation_path=animation_path,
    )
    metadata["poc"] = {
        "name": "fixed_four_cylinder",
        "units": "meters",
        "requested_phantom_size_mm": [80.0, 80.0, 25.0],
        "requested_lump_shape": "cylinder",
        "requested_lump_diameter_mm": 20.0,
        "requested_lump_height_mm": 5.0,
        "requested_probe_diameter_mm": float(args.probe_diameter_mm),
        "requested_max_indentation_mm": float(args.max_indentation_mm),
        "requested_press_steps": int(args.press_steps),
    }
    metadata["mesh"] = _mesh_summary(phantom, mesh, tet_lump_mask, tet_lump_id, len(lumps))
    write_ground_truth_metadata(metadata_path, metadata)

    return {
        "run": run_dir.name,
        "backend": backend_label,
        "elapsed_seconds": time.time() - run_start,
        "npz": str(npz_path),
        "metadata": str(metadata_path),
        "phantom_mesh": str(gltf_path) if gltf_path is not None else None,
        "scan_animation": str(animation_path) if animation_path is not None else None,
        "press_records": str(press_dir) if press_dir is not None else None,
        "curve_summary": str(curve_summary_path),
        "mesh": _mesh_summary(phantom, mesh, tet_lump_mask, tet_lump_id, len(lumps)),
        "presses_shape": list(np.asarray(sample["presses"]).shape),
        "fz_shape": list(np.asarray(sample["fz"]).shape),
    }


def _fixed_lumps() -> list[LumpSpec]:
    return [
        LumpSpec(
            shape="cylinder",
            center=tuple(float(v) for v in center),
            radii=(CYLINDER_RADIUS_M, CYLINDER_RADIUS_M, CYLINDER_HALF_HEIGHT_M),
            stiffness_multiplier=STIFFNESS_MULTIPLIER,
            yaw=0.0,
        )
        for center in FIXED_LUMP_CENTERS_M
    ]


def _validate_lumps(lumps: Sequence[LumpSpec]) -> None:
    sx, sy, sz = PHANTOM_SIZE_M
    for idx, lump in enumerate(lumps):
        cx, cy, cz = lump.center
        rx, ry, rz = lump.radii
        inside = (
            -0.5 * sx <= cx - rx
            and cx + rx <= 0.5 * sx
            and -0.5 * sy <= cy - ry
            and cy + ry <= 0.5 * sy
            and 0.0 <= cz - rz
            and cz + rz <= sz
        )
        if not inside:
            raise ValueError(f"fixed lump {idx} is outside the phantom: {lump}")


def _require_nonempty_lump_tets(tet_lump_id: np.ndarray, lump_count: int, label: str) -> None:
    missing = [idx for idx in range(lump_count) if int(np.count_nonzero(tet_lump_id == idx)) == 0]
    if missing:
        raise RuntimeError(f"{label} mesh has no assigned tetrahedra for lumps {missing}")


def _phantom(cells_x: int, cells_y: int, cells_z: int) -> PhantomConfig:
    sx, sy, sz = PHANTOM_SIZE_M
    min_cell = min(sx / float(cells_x), sy / float(cells_y), sz / float(cells_z))
    return PhantomConfig(
        size_x=sx,
        size_y=sy,
        height=sz,
        cells_x=int(cells_x),
        cells_y=int(cells_y),
        cells_z=int(cells_z),
        particle_radius=0.45 * min_cell,
    )


def _scan(
    *,
    grid_h: int,
    grid_w: int,
    press_steps: int,
    max_indentation_mm: float,
    probe_diameter_mm: float,
    edge_margin_mm: float,
    soft_contact_margin_mm: float | None = None,
    substeps: int = 3,
    vbd_iterations: int = 5,
) -> ScanConfig:
    return ScanConfig(
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        edge_margin=float(edge_margin_mm) * MM,
        probe_radius=0.5 * float(probe_diameter_mm) * MM,
        max_indentation=float(max_indentation_mm) * MM,
        press_steps=int(press_steps),
        sim_substeps_per_depth=int(substeps),
        vbd_iterations=int(vbd_iterations),
        soft_contact_margin=(
            ScanConfig.soft_contact_margin
            if soft_contact_margin_mm is None
            else float(soft_contact_margin_mm) * MM
        ),
    )


def _newton_material(args: argparse.Namespace, base: MaterialConfig) -> MaterialConfig:
    return MaterialConfig(
        k_mu=float(args.newton_k_mu),
        k_lambda=float(args.newton_k_lambda),
        k_damp=base.k_damp,
        soft_contact_ke=float(args.newton_soft_contact_ke),
        soft_contact_kd=base.soft_contact_kd,
        soft_contact_mu=base.soft_contact_mu,
        probe_contact_mu=base.probe_contact_mu,
        lump_stiffness_min=base.lump_stiffness_min,
        lump_stiffness_max=base.lump_stiffness_max,
    )


def _mesh_summary(
    phantom: PhantomConfig,
    mesh: object,
    tet_lump_mask: np.ndarray,
    tet_lump_id: np.ndarray,
    lump_count: int,
) -> dict[str, object]:
    return {
        "cells": [phantom.cells_x, phantom.cells_y, phantom.cells_z],
        "vertex_count": int(mesh.vertices.shape[0]),
        "tet_count": int(mesh.tets.shape[0]),
        "tet_lump_count": int(np.count_nonzero(tet_lump_mask)),
        "tet_count_by_lump": [int(np.count_nonzero(tet_lump_id == idx)) for idx in range(lump_count)],
        "particle_radius_m": float(phantom.particle_radius),
    }


def _write_curve_summary(path: Path, sample: dict[str, object]) -> None:
    xy = np.asarray(sample["xy"], dtype=np.float32)
    depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
    fz = np.asarray(sample["fz"], dtype=np.float32)
    ratio = np.asarray(sample["nonlinearity_ratio"], dtype=np.float32) if "nonlinearity_ratio" in sample else None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row",
                "col",
                "x_m",
                "y_m",
                "peak_force_n",
                "final_force_n",
                "endpoint_stiffness_n_per_m",
                "late_early_slope_ratio",
            ],
        )
        writer.writeheader()
        for row in range(fz.shape[0]):
            for col in range(fz.shape[1]):
                z = depth[row, col]
                force = fz[row, col]
                dz = float(z[-1] - z[0]) if z.size else 0.0
                df = float(force[-1] - force[0]) if force.size else 0.0
                writer.writerow(
                    {
                        "row": row,
                        "col": col,
                        "x_m": f"{float(xy[row, col, 0]):.10g}",
                        "y_m": f"{float(xy[row, col, 1]):.10g}",
                        "peak_force_n": f"{float(np.max(force)):.10g}",
                        "final_force_n": f"{float(force[-1]):.10g}",
                        "endpoint_stiffness_n_per_m": f"{(df / dz) if dz > 0.0 else 0.0:.10g}",
                        "late_early_slope_ratio": f"{float(ratio[row, col]):.10g}" if ratio is not None else "",
                    }
                )


def _write_surface_scan_animation_html(
    path: Path,
    phantom: PhantomConfig,
    scan: ScanConfig,
    sample: dict[str, object],
    lumps: Sequence[LumpSpec],
    mesh: object,
    tet_lump_id: np.ndarray,
) -> None:
    normal_faces = _outer_boundary_faces(phantom)
    normal_vertices, normal_indices = _compact_indexed_geometry(mesh.vertices, normal_faces)
    _, normal_edges = _surface_wire_geometry(normal_vertices, normal_indices)
    lump_surfaces = []
    for idx, lump in enumerate(lumps):
        faces = _selected_tet_boundary_faces(mesh.tets, tet_lump_id == idx)
        vertices, indices = _compact_indexed_geometry(mesh.vertices, faces)
        _, edges = _surface_wire_geometry(vertices, indices)
        lump_surfaces.append(
            {
                "id": idx,
                "shape": lump.shape,
                "vertices": _flat_rounded(vertices),
                "triangles": np.asarray(indices, dtype=np.uint32).reshape(-1).tolist(),
                "edges": np.asarray(edges, dtype=np.uint32).reshape(-1).tolist(),
                "triangle_count": int(indices.shape[0]),
            }
        )
    data = {
        "sample_id": path.stem,
        "phantom": phantom.to_dict(),
        "scan": scan.to_dict(),
        "xy": np.round(np.asarray(sample["xy"], dtype=np.float32), 7).tolist(),
        "indentation": np.round(np.asarray(sample["indentation_depth"], dtype=np.float32), 7).tolist(),
        "force_z": np.round(np.asarray(sample["fz"], dtype=np.float32), 6).tolist(),
        "lumps": [lump.to_dict(phantom) for lump in lumps],
        "mesh": {
            "cells": [phantom.cells_x, phantom.cells_y, phantom.cells_z],
            "vertex_count": int(mesh.vertices.shape[0]),
            "tet_count": int(mesh.tets.shape[0]),
            "surface_vertex_count": int(normal_vertices.shape[0]),
            "normal": {
                "vertices": _flat_rounded(normal_vertices),
                "triangles": np.asarray(normal_indices, dtype=np.uint32).reshape(-1).tolist(),
                "edges": np.asarray(normal_edges, dtype=np.uint32).reshape(-1).tolist(),
            },
            "lump_surfaces": lump_surfaces,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_surface_animation_html(json.dumps(data, separators=(",", ":")).replace("</", "<\\/")), encoding="utf-8")


def _write_fast_phantom_gltf(
    path: Path,
    phantom: PhantomConfig,
    lumps: Sequence[LumpSpec],
    material: MaterialConfig,
    mesh: object,
    tet_lump_id: np.ndarray,
) -> None:
    gltf: dict[str, object] = {
        "asset": {"version": "2.0", "generator": "custom_simulation.scripts.run_fixed_four_cylinder_poc"},
        "scene": 0,
        "scenes": [{"name": "fixed_four_cylinder_scene", "nodes": []}],
        "materials": [],
        "meshes": [],
        "nodes": [],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
        "extras": {
            "units": "meters",
            "geometry_source": "structured_tetrahedral_mesh_outer_surface_plus_lump_tet_surfaces",
            "phantom": phantom.to_dict(),
            "tet_count": int(mesh.tets.shape[0]),
            "vertex_count": int(mesh.vertices.shape[0]),
            "num_lumps": len(lumps),
            "lumps": [lump.to_dict(phantom) for lump in lumps],
        },
    }
    normal_material = _add_material(
        gltf,
        name="normal_tissue_outer_surface",
        color=(0.62, 0.78, 1.0, 0.12),
    )
    normal_wire_material = _add_material(
        gltf,
        name="normal_tissue_outer_surface_wire",
        color=(0.18, 0.26, 0.30, 0.30),
    )
    normal_vertices, normal_indices = _compact_indexed_geometry(mesh.vertices, _outer_boundary_faces(phantom))
    _add_mesh_node(gltf, "normal_tissue_outer_surface", normal_vertices, normal_indices, normal_material)
    _, normal_wire_indices = _surface_wire_geometry(normal_vertices, normal_indices)
    _add_mesh_node(
        gltf,
        "normal_tissue_outer_surface_wire",
        normal_vertices,
        normal_wire_indices,
        normal_wire_material,
        mode=1,
    )

    for idx, lump in enumerate(lumps):
        faces = _selected_tet_boundary_faces(mesh.tets, tet_lump_id == idx)
        if faces.size == 0:
            continue
        lump_material = _add_material(
            gltf,
            name=f"lump_{idx:02d}_{lump.shape}_tet_surface",
            color=_lump_color(idx),
        )
        lump_wire_material = _add_material(
            gltf,
            name=f"lump_{idx:02d}_{lump.shape}_tet_wire",
            color=(0.05, 0.05, 0.05, 0.86),
        )
        vertices, indices = _compact_indexed_geometry(mesh.vertices, faces)
        label = f"lump_{idx:02d}_{lump.shape}_tet_stiffness_{lump.stiffness_multiplier:.3f}"
        _add_mesh_node(gltf, label, vertices, indices, lump_material)
        wire_vertices, wire_indices = _selected_tet_wire_geometry(mesh.vertices, mesh.tets, tet_lump_id == idx)
        if wire_indices.size > 0:
            _add_mesh_node(gltf, f"{label}_wire", wire_vertices, wire_indices, lump_wire_material, mode=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(gltf, f, indent=2)


def _outer_boundary_faces(phantom: PhantomConfig) -> np.ndarray:
    nx, ny, nz = int(phantom.cells_x), int(phantom.cells_y), int(phantom.cells_z)
    faces: list[tuple[int, int, int]] = []

    def idx(x: int, y: int, z: int) -> int:
        return z * ((ny + 1) * (nx + 1)) + y * (nx + 1) + x

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        faces.append((a, b, c))
        faces.append((a, c, d))

    for z in (0, nz):
        for y in range(ny):
            for x in range(nx):
                add_quad(idx(x, y, z), idx(x + 1, y, z), idx(x + 1, y + 1, z), idx(x, y + 1, z))
    for y in (0, ny):
        for z in range(nz):
            for x in range(nx):
                add_quad(idx(x, y, z), idx(x + 1, y, z), idx(x + 1, y, z + 1), idx(x, y, z + 1))
    for x in (0, nx):
        for z in range(nz):
            for y in range(ny):
                add_quad(idx(x, y, z), idx(x, y + 1, z), idx(x, y + 1, z + 1), idx(x, y, z + 1))
    return np.asarray(faces, dtype=np.uint32)


def _flat_rounded(values: np.ndarray, decimals: int = 7) -> list[float]:
    return np.round(np.asarray(values, dtype=np.float32), decimals).reshape(-1).tolist()


def _surface_animation_html(data_json: str) -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fixed Four-Cylinder Scan Animation</title>
  <style>
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #101214; color: #f3f1ea; font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }
    #viewport { position: fixed; inset: 0; display: block; }
    .panel { position: fixed; top: 12px; left: 12px; right: 12px; z-index: 2; display: grid; grid-template-columns: minmax(200px, 1fr) auto; gap: 10px; align-items: center; padding: 8px 10px; border: 1px solid rgba(255,255,255,0.16); border-radius: 8px; background: rgba(16,18,20,0.74); backdrop-filter: blur(8px); }
    .title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 650; }
    .controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    label { display: inline-flex; align-items: center; gap: 5px; user-select: none; }
    button { border: 1px solid rgba(255,255,255,0.28); border-radius: 6px; background: #f4f0e8; color: #17191b; padding: 5px 9px; font: inherit; cursor: pointer; }
    input[type="range"] { width: 130px; }
    .status { position: fixed; left: 12px; bottom: 12px; z-index: 2; max-width: min(820px, calc(100vw - 24px)); padding: 7px 9px; border: 1px solid rgba(255,255,255,0.16); border-radius: 8px; background: rgba(16,18,20,0.74); color: #dedbd1; backdrop-filter: blur(8px); white-space: pre-wrap; }
    @media (max-width: 760px) { .panel { grid-template-columns: 1fr; } .controls { justify-content: flex-start; } }
  </style>
</head>
<body>
  <canvas id="viewport"></canvas>
  <div class="panel">
    <div id="title" class="title">fixed four-cylinder scan animation</div>
    <div class="controls">
      <button id="playButton" type="button">Pause</button>
      <button id="resetButton" type="button">Reset</button>
      <label>speed <input id="speedRange" type="range" min="1" max="120" value="36"></label>
      <label><input id="autoViewToggle" type="checkbox" checked> auto view</label>
      <label><input id="surfaceToggle" type="checkbox" checked> tissue</label>
      <label><input id="lumpToggle" type="checkbox" checked> lumps</label>
      <label><input id="wireToggle" type="checkbox" checked> surface wire</label>
    </div>
  </div>
  <div id="status" class="status">Loading...</div>
  <script type="importmap">
    { "imports": { "three": "https://unpkg.com/three@0.165.0/build/three.module.js", "three/addons/": "https://unpkg.com/three@0.165.0/examples/jsm/" } }
  </script>
  <script type="module">
    import * as THREE from "three";
    import { OrbitControls } from "three/addons/controls/OrbitControls.js";

    const scanData = __DATA_JSON__;
    const canvas = document.getElementById("viewport");
    const title = document.getElementById("title");
    const status = document.getElementById("status");
    const playButton = document.getElementById("playButton");
    const resetButton = document.getElementById("resetButton");
    const speedRange = document.getElementById("speedRange");
    const autoViewToggle = document.getElementById("autoViewToggle");
    const surfaceToggle = document.getElementById("surfaceToggle");
    const lumpToggle = document.getElementById("lumpToggle");
    const wireToggle = document.getElementById("wireToggle");

    const phantom = scanData.phantom;
    const scan = scanData.scan;
    const rows = scanData.xy.length;
    const cols = scanData.xy[0].length;
    const steps = scanData.indentation[0][0].length;
    const totalFrames = rows * cols * steps;
    title.textContent = `${scanData.sample_id}: ${scanData.mesh.cells.join(" x ")} cells, ${scanData.mesh.tet_count.toLocaleString()} tets`;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101214);
    const camera = new THREE.PerspectiveCamera(46, window.innerWidth / window.innerHeight, 0.001, 20);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.55;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x4b555d, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(0.25, -0.45, 0.8);
    scene.add(keyLight);

    const tissueGroup = new THREE.Group();
    const lumpGroup = new THREE.Group();
    const wireGroup = new THREE.Group();
    scene.add(tissueGroup, lumpGroup, wireGroup);

    const deformTargets = [];
    addSurface(scanData.mesh.normal, tissueGroup, wireGroup, 0x8dc4e6, 0x24404d, 0.25);
    const colors = [0xe53d34, 0x1692e6, 0xf2a20b, 0x40a85a];
    scanData.mesh.lump_surfaces.forEach((surface, idx) => {
      addSurface(surface, lumpGroup, wireGroup, colors[idx % colors.length], 0x050505, 0.72);
    });

    const pathLine = createScanPath();
    scene.add(pathLine);
    const probe = new THREE.Mesh(
      new THREE.SphereGeometry(scan.probe_radius, 32, 16),
      new THREE.MeshStandardMaterial({ color: 0xf2f0e7, metalness: 0.05, roughness: 0.35 })
    );
    scene.add(probe);
    const contactRing = new THREE.Mesh(
      new THREE.RingGeometry(scan.probe_radius * 0.75, scan.probe_radius * 1.15, 48),
      new THREE.MeshBasicMaterial({ color: 0xffd15c, transparent: true, opacity: 0.9, side: THREE.DoubleSide })
    );
    scene.add(contactRing);

    let playing = true;
    let frame = 0;
    let lastFrame = -1;
    let lastTime = performance.now();
    resetCamera();

    playButton.addEventListener("click", () => { playing = !playing; playButton.textContent = playing ? "Pause" : "Play"; });
    resetButton.addEventListener("click", () => { frame = 0; lastFrame = -1; resetCamera(); updateFrame(); });
    autoViewToggle.addEventListener("change", () => { controls.autoRotate = autoViewToggle.checked; });
    surfaceToggle.addEventListener("change", () => { tissueGroup.visible = surfaceToggle.checked; });
    lumpToggle.addEventListener("change", () => { lumpGroup.visible = lumpToggle.checked; });
    wireToggle.addEventListener("change", () => { wireGroup.visible = wireToggle.checked; });
    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function addSurface(surface, surfaceGroup, wireParent, color, wireColor, opacity) {
      const base = new Float32Array(surface.vertices);
      const current = new Float32Array(base);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(current, 3).setUsage(THREE.DynamicDrawUsage));
      geom.setIndex(new THREE.BufferAttribute(new Uint32Array(surface.triangles), 1));
      geom.computeVertexNormals();
      const mesh = new THREE.Mesh(
        geom,
        new THREE.MeshStandardMaterial({ color, transparent: true, opacity, roughness: 0.58, side: THREE.DoubleSide })
      );
      surfaceGroup.add(mesh);
      if (surface.edges && surface.edges.length) {
        const wireGeom = new THREE.BufferGeometry();
        wireGeom.setAttribute("position", geom.getAttribute("position"));
        wireGeom.setIndex(new THREE.BufferAttribute(new Uint32Array(surface.edges), 1));
        wireParent.add(new THREE.LineSegments(wireGeom, new THREE.LineBasicMaterial({ color: wireColor, transparent: true, opacity: 0.45 })));
      }
      deformTargets.push({ base, current, geom, mesh });
    }

    function createScanPath() {
      const points = [];
      for (let r = 0; r < rows; r++) {
        const colRange = r % 2 === 0 ? [...Array(cols).keys()] : [...Array(cols).keys()].reverse();
        for (const c of colRange) {
          const xy = scanData.xy[r][c];
          points.push(new THREE.Vector3(xy[0], xy[1], phantom.height + 0.0015));
        }
      }
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xffd15c, transparent: true, opacity: 0.72 }));
    }

    function updateSurfaceVertices(px, py, depth) {
      const sigma = Math.max(scan.probe_radius * 1.45, 0.001);
      const sigma2 = sigma * sigma;
      const deform = Math.min(depth, phantom.height * 0.65);
      const depthSigma = Math.max(phantom.height * 0.42, 0.001);
      for (const target of deformTargets) {
        const base = target.base;
        const current = target.current;
        for (let i = 0; i < base.length; i += 3) {
          const x = base[i];
          const y = base[i + 1];
          const z = base[i + 2];
          const dx = x - px;
          const dy = y - py;
          const lateral = Math.exp(-0.5 * (dx * dx + dy * dy) / sigma2);
          const belowSurface = Math.max(phantom.height - z, 0.0);
          const depthFalloff = Math.exp(-belowSurface / depthSigma);
          current[i] = x;
          current[i + 1] = y;
          current[i + 2] = Math.max(0.0, z - deform * lateral * depthFalloff);
        }
        target.geom.attributes.position.needsUpdate = true;
        target.geom.computeVertexNormals();
      }
    }

    function updateFrame() {
      if (frame === lastFrame) return;
      lastFrame = frame;
      const step = frame % steps;
      const pressIndex = Math.floor(frame / steps);
      const row = Math.floor(pressIndex / cols);
      const col = pressIndex % cols;
      const xy = scanData.xy[row][col];
      const depth = scanData.indentation[row][col][step];
      const force = scanData.force_z[row][col][step];
      const pz = phantom.height + scan.probe_radius + (scan.preload_gap || 0.0) - depth;
      probe.position.set(xy[0], xy[1], pz);
      contactRing.position.set(xy[0], xy[1], phantom.height + 0.0004);
      updateSurfaceVertices(xy[0], xy[1], depth);
      status.textContent =
        `${scanData.sample_id}\\nframe ${frame + 1} / ${totalFrames}   press r${row} c${col}   step ${step + 1} / ${steps}\\n` +
        `x=${xy[0].toFixed(4)} m  y=${xy[1].toFixed(4)} m  indentation=${depth.toFixed(5)} m  Fz=${force.toFixed(4)} N`;
    }

    function resetCamera() {
      const radius = Math.max(phantom.size_x, phantom.size_y, phantom.height);
      camera.position.set(radius * 0.9, -radius * 1.35, radius * 0.9);
      controls.target.set(0, 0, phantom.height * 0.45);
      camera.near = Math.max(radius / 1000, 0.0001);
      camera.far = radius * 100;
      camera.updateProjectionMatrix();
      controls.update();
    }

    function animate(now) {
      const dt = Math.max((now - lastTime) / 1000, 0.0);
      lastTime = now;
      if (playing) {
        const framesToAdvance = Math.max(1, Math.floor(Number(speedRange.value) * dt));
        frame = (frame + framesToAdvance) % totalFrames;
      }
      updateFrame();
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }

    updateFrame();
    requestAnimationFrame(animate);
  </script>
</body>
</html>
""".replace("__DATA_JSON__", data_json)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
