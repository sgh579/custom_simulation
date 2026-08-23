#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.native_data import load_phantom_scan_material_lumps
from palpation_sim.visual_geometry import phantom_box_polydata, require_pyvista


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GT phantom inclusions as printable STL meshes.")
    parser.add_argument("data_dir", type=Path, help="Directory containing sample_*.npz and *_gt.json files.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Optional selected_candidates.csv for sample order and Dice metadata.",
    )
    parser.add_argument("--resolution", type=int, default=96, help="Implicit-surface resolution per inclusion.")
    parser.add_argument(
        "--centered",
        action="store_true",
        help="Keep GT-centered x/y coordinates instead of shifting the phantom to positive x/y.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = _load_selection(args.selection_csv)
    sample_paths = _ordered_samples(data_dir, selection_rows)
    if not sample_paths:
        raise SystemExit(f"No sample_*.npz files found in {data_dir}")

    manifest: dict[str, object] = {
        "source_data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "units": "millimeters",
        "coordinate_mode": "gt_centered_xy" if args.centered else "positive_xy_print_frame",
        "coordinate_note": (
            "x/y/z are GT meters converted to millimeters."
            if args.centered
            else "x/y are GT meters converted to millimeters and shifted by half the phantom width/height; z is bottom-up millimeters."
        ),
        "samples": [],
    }
    summary_rows: list[dict[str, object]] = []

    for sample_path in sample_paths:
        sample_key = sample_path.name
        sample_stem = sample_path.stem
        gt_path = sample_path.with_name(f"{sample_stem}_gt.json")
        phantom, _scan, _material, lumps, metadata = load_phantom_scan_material_lumps(
            sample_path=sample_path,
            metadata_path=gt_path if gt_path.exists() else None,
        )
        sample_out = out_dir / sample_stem
        sample_out.mkdir(parents=True, exist_ok=True)
        shift = (0.0, 0.0, 0.0) if args.centered else (0.5 * phantom.size_x, 0.5 * phantom.size_y, 0.0)

        box_mesh = _to_export_frame(phantom_box_polydata(phantom), shift=shift)
        box_path = sample_out / f"{sample_stem}_phantom_box_reference.stl"
        _save_stl(box_mesh, box_path)

        lump_paths: list[str] = []
        lump_meshes = []
        for idx, lump in enumerate(lumps):
            mesh = _lump_mesh(lump, resolution=args.resolution)
            mesh["lump_id"] = np.full(mesh.n_points, idx, dtype=np.int32)
            export_mesh = _to_export_frame(mesh, shift=shift)
            lump_path = sample_out / f"{sample_stem}_lump_{idx:02d}_{lump.shape}.stl"
            _save_stl(export_mesh, lump_path)
            lump_paths.append(str(lump_path))
            lump_meshes.append(export_mesh)

            row = _lump_row(
                sample_path=sample_path,
                selection=selection_rows.get(sample_key, {}),
                phantom=phantom,
                metadata=metadata,
                lump_index=idx,
                lump=lump,
                shift=shift,
            )
            summary_rows.append(row)

        combined_path = sample_out / f"{sample_stem}_inclusions_combined.stl"
        if lump_meshes:
            combined = lump_meshes[0].copy(deep=True)
            for mesh in lump_meshes[1:]:
                combined = combined.merge(mesh, merge_points=False)
            _save_stl(combined, combined_path)

        (sample_out / f"{sample_stem}_solidworks_dimensions.csv").write_text(
            _rows_to_csv_text([row for row in summary_rows if row["sample"] == sample_key]),
            encoding="utf-8",
        )
        manifest["samples"].append(
            {
                "sample": sample_key,
                "rank": selection_rows.get(sample_key, {}).get("rank"),
                "val_selected_dice": selection_rows.get(sample_key, {}).get("val_selected_dice"),
                "base_phantom_id": metadata.get("base_phantom_id"),
                "phantom_box_reference_stl": str(box_path),
                "inclusions_combined_stl": str(combined_path),
                "lump_stls": lump_paths,
                "solidworks_dimensions_csv": str(sample_out / f"{sample_stem}_solidworks_dimensions.csv"),
            }
        )

    (out_dir / "solidworks_dimensions_all.csv").write_text(_rows_to_csv_text(summary_rows), encoding="utf-8")
    (out_dir / "stl_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme_text(manifest), encoding="utf-8")
    print(f"Exported {len(sample_paths)} sample(s) to {out_dir}")


def _load_selection(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["sample"]: row for row in csv.DictReader(f) if row.get("sample")}


def _ordered_samples(data_dir: Path, selection_rows: dict[str, dict[str, str]]) -> list[Path]:
    by_name = {path.name: path for path in data_dir.glob("sample_*.npz")}
    ordered: list[Path] = []
    rows = sorted(selection_rows.values(), key=lambda row: (_float_or_inf(row.get("rank")), row.get("sample", "")))
    for row in rows:
        path = by_name.get(row.get("sample", ""))
        if path is not None:
            ordered.append(path)
    ordered.extend(path for path in sorted(by_name.values()) if path not in ordered)
    return ordered


def _to_export_frame(mesh, *, shift: tuple[float, float, float]):
    out = mesh.copy(deep=True)
    points = np.asarray(out.points, dtype=np.float64)
    points = (points + np.asarray(shift, dtype=np.float64)) * 1000.0
    out.points = points.astype(np.float32)
    return out


def _lump_mesh(lump, *, resolution: int):
    pv = require_pyvista()
    resolution = max(int(resolution), 16)
    center = np.asarray(lump.center, dtype=np.float64)
    radii = np.asarray(lump.radii, dtype=np.float64)
    yaw = float(lump.yaw)

    if lump.shape in {"sphere", "ellipsoid"}:
        mesh = pv.Sphere(
            radius=1.0,
            theta_resolution=resolution,
            phi_resolution=max(resolution // 2, 16),
        )
        points = np.asarray(mesh.points, dtype=np.float64) * radii
        mesh.points = _local_to_world(points, center, yaw).astype(np.float32)
        return mesh

    if lump.shape == "box":
        return _box_mesh(pv, center=center, radii=radii, yaw=yaw)

    if lump.shape == "cylinder":
        return _cylinder_mesh(pv, center=center, radii=radii, yaw=yaw, resolution=resolution)

    if lump.shape == "capsule":
        radius = float(radii[0])
        half_axis = float(radii[2])
        mesh = pv.Sphere(
            radius=radius,
            theta_resolution=resolution,
            phi_resolution=max(resolution // 2, 16),
        )
        points = np.asarray(mesh.points, dtype=np.float64)
        if half_axis > 0.0:
            points[:, 2] += np.where(points[:, 2] >= 0.0, half_axis, -half_axis)
        mesh.points = _local_to_world(points, center, yaw).astype(np.float32)
        return mesh

    raise ValueError(f"Unsupported lump shape: {lump.shape}")


def _box_mesh(pv, *, center: np.ndarray, radii: np.ndarray, yaw: float):
    rx, ry, rz = radii
    local = np.asarray(
        [
            [-rx, -ry, -rz],
            [rx, -ry, -rz],
            [rx, ry, -rz],
            [-rx, ry, -rz],
            [-rx, -ry, rz],
            [rx, -ry, rz],
            [rx, ry, rz],
            [-rx, ry, rz],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [4, 0, 1, 2, 3],
            [4, 4, 7, 6, 5],
            [4, 0, 4, 5, 1],
            [4, 1, 5, 6, 2],
            [4, 2, 6, 7, 3],
            [4, 3, 7, 4, 0],
        ],
        dtype=np.int64,
    )
    return pv.PolyData(_local_to_world(local, center, yaw).astype(np.float32), faces.reshape(-1))


def _cylinder_mesh(pv, *, center: np.ndarray, radii: np.ndarray, yaw: float, resolution: int):
    rx, ry, rz = radii
    n = max(int(resolution), 16)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
    bottom = np.column_stack([ring, np.full(n, -rz)])
    top = np.column_stack([ring, np.full(n, rz)])
    vertices = np.vstack([bottom, top, [[0.0, 0.0, -rz], [0.0, 0.0, rz]]])
    bottom_center = 2 * n
    top_center = 2 * n + 1

    faces: list[int] = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([4, i, j, n + j, n + i])
        faces.extend([3, bottom_center, j, i])
        faces.extend([3, top_center, n + i, n + j])
    return pv.PolyData(_local_to_world(vertices, center, yaw).astype(np.float32), np.asarray(faces, dtype=np.int64))


def _local_to_world(points: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    out = np.array(points, dtype=np.float64, copy=True)
    x = out[:, 0].copy()
    y = out[:, 1].copy()
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    out += center
    return out


def _save_stl(mesh, path: Path) -> None:
    mesh = mesh.triangulate().clean()
    mesh.save(path)


def _lump_row(*, sample_path: Path, selection: dict[str, str], phantom, metadata: dict, lump_index: int, lump, shift):
    center_m = np.asarray(lump.center, dtype=np.float64)
    radii_m = np.asarray(lump.radii, dtype=np.float64)
    center_export_mm = (center_m + np.asarray(shift, dtype=np.float64)) * 1000.0
    center_gt_mm = center_m * 1000.0
    radii_mm = radii_m * 1000.0
    z_extent = float(_z_extent(lump)) * 1000.0
    height_mm = float(phantom.height) * 1000.0
    return {
        "rank": selection.get("rank", ""),
        "sample": sample_path.name,
        "val_selected_dice": selection.get("val_selected_dice", ""),
        "fixed_dice": selection.get("fixed_dice", ""),
        "base_phantom_id": metadata.get("base_phantom_id", ""),
        "lump_index": lump_index,
        "shape": lump.shape,
        "stiffness_multiplier": float(lump.stiffness_multiplier),
        "center_x_print_mm": center_export_mm[0],
        "center_y_print_mm": center_export_mm[1],
        "center_z_bottom_up_mm": center_export_mm[2],
        "center_x_gt_mm": center_gt_mm[0],
        "center_y_gt_mm": center_gt_mm[1],
        "center_z_gt_mm": center_gt_mm[2],
        "center_depth_from_top_mm": height_mm - center_gt_mm[2],
        "top_depth_from_top_mm": max(height_mm - center_gt_mm[2] - z_extent, 0.0),
        "bottom_depth_from_top_mm": max(height_mm - center_gt_mm[2] + z_extent, 0.0),
        "radius_x_mm": radii_mm[0],
        "radius_y_mm": radii_mm[1],
        "radius_z_or_half_axis_mm": radii_mm[2],
        "effective_z_extent_mm": z_extent,
        "yaw_rad": float(lump.yaw),
        "yaw_deg": float(np.degrees(lump.yaw)),
    }


def _z_extent(lump) -> float:
    if lump.shape == "capsule":
        return float(lump.radii[2] + lump.radii[0])
    return float(lump.radii[2])


def _rows_to_csv_text(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _readme_text(manifest: dict[str, object]) -> str:
    return f"""# Printable Phantom GT STL Export

Units: millimeters.

Coordinate mode: `{manifest["coordinate_mode"]}`.

For the default positive print frame, the phantom block is `0..180 mm` in X, `0..180 mm` in Y, and `0..80 mm` in Z. The original GT coordinate system is centered in X/Y and bottom-up in Z, so the conversion is:

```text
x_print_mm = x_gt_m * 1000 + 90
y_print_mm = y_gt_m * 1000 + 90
z_print_mm = z_gt_m * 1000
depth_from_top_mm = 80 - z_print_mm
```

Each sample folder contains:

- `*_phantom_box_reference.stl`: the 180 x 180 x 80 mm reference block.
- `*_lump_##_<shape>.stl`: one inclusion per STL.
- `*_inclusions_combined.stl`: all inclusions in one STL as separate solids.
- `*_solidworks_dimensions.csv`: dimensions for manual SolidWorks recreation.

The inclusion STLs are direct GT geometry exports. For a physical phantom workflow, these are best treated as inclusion positives or CAD references. A printable mold with cavities, registration pins, pouring gates, and split planes should be designed as a second step around the fabrication method.
"""


def _float_or_inf(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else float("inf")
    except ValueError:
        return float("inf")


if __name__ == "__main__":
    main()
