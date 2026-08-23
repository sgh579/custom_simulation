#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from export_phantom_gt_stl import _lump_mesh
from palpation_sim.native_data import load_phantom_scan_material_lumps
from palpation_sim.phantom import LumpSpec
from palpation_sim.visual_geometry import require_pyvista


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a first-pass fabrication mold and hanging insert STL.")
    parser.add_argument("sample", type=Path, help="GT sample .npz to use for the hanging insert demo.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--outer-size-mm", type=float, default=90.0)
    parser.add_argument("--inner-size-mm", type=float, default=80.0)
    parser.add_argument("--phantom-height-mm", type=float, default=80.0)
    parser.add_argument("--bottom-thickness-mm", type=float, default=5.0)
    parser.add_argument("--slot-length-mm", type=float, default=9.9, help="Slot/key length along the wall tangent.")
    parser.add_argument("--slot-depth-mm", type=float, default=8.0, help="Vertical depth cut down from the wall top.")
    parser.add_argument("--support-radius-mm", type=float, default=2.5)
    parser.add_argument("--top-rod-radius-mm", type=float, default=2.5)
    parser.add_argument(
        "--watertight-voxel-mm",
        type=float,
        default=0.5,
        help="Voxel size for implicit-union watertight hanging assembly export.",
    )
    parser.add_argument(
        "--reference-wall-stl",
        type=Path,
        default=None,
        help="Optional existing boxWall STL to include instead of the rough generated mold.",
    )
    parser.add_argument("--resolution", type=int, default=96)
    args = parser.parse_args()

    sample_path = args.sample.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = sample_path.with_name(f"{sample_path.stem}_gt.json")
    phantom, _scan, _material, lumps, metadata = load_phantom_scan_material_lumps(
        sample_path=sample_path,
        metadata_path=gt_path if gt_path.exists() else None,
    )

    params = FabricationParams(
        outer_size_mm=float(args.outer_size_mm),
        inner_size_mm=float(args.inner_size_mm),
        phantom_height_mm=float(args.phantom_height_mm),
        bottom_thickness_mm=float(args.bottom_thickness_mm),
        slot_length_mm=float(args.slot_length_mm),
        slot_depth_mm=float(args.slot_depth_mm),
        support_radius_mm=float(args.support_radius_mm),
        top_rod_radius_mm=float(args.top_rod_radius_mm),
    )
    wall_mm = 0.5 * (params.outer_size_mm - params.inner_size_mm)
    if wall_mm <= 0:
        raise SystemExit("--outer-size-mm must be larger than --inner-size-mm")
    mold_top_z = params.bottom_thickness_mm + params.phantom_height_mm

    scaled_lumps = _scale_lumps_to_inner(lumps, phantom_size_x_mm=float(phantom.size_x) * 1000.0, params=params)
    if args.reference_wall_stl is not None:
        reference_wall_path = args.reference_wall_stl.expanduser().resolve()
        mold_path = out_dir / f"boxWall_reference_outer{params.outer_size_mm:g}_inner{params.inner_size_mm:g}.stl"
        shutil.copy2(reference_wall_path, mold_path)
        mold_mesh = None
    else:
        mold_mesh = _mold_mesh(params)
        mold_path = out_dir / f"mold_outer{params.outer_size_mm:g}_inner{params.inner_size_mm:g}_slot{params.slot_length_mm:g}.stl"
        _save_stl(mold_mesh, mold_path)

    hanging_mesh = _hanging_assembly_mesh(
        scaled_lumps,
        params=params,
        mold_top_z=mold_top_z,
        resolution=int(args.resolution),
    )
    hanging_path = out_dir / f"{sample_path.stem}_hanging_assembly_d5_support.stl"
    _save_stl(hanging_mesh, hanging_path)

    watertight_mesh = _hanging_assembly_watertight_mesh(
        scaled_lumps,
        params=params,
        mold_top_z=mold_top_z,
        voxel_mm=float(args.watertight_voxel_mm),
    )
    watertight_path = out_dir / f"{sample_path.stem}_hanging_assembly_d5_support_watertight.stl"
    _save_stl(watertight_mesh, watertight_path)

    lumps_mesh = _combined_lumps_mesh(scaled_lumps, params=params, resolution=int(args.resolution))
    lumps_path = out_dir / f"{sample_path.stem}_scaled_lumps_only.stl"
    _save_stl(lumps_mesh, lumps_path)

    preview_path = None
    if mold_mesh is not None:
        preview_mesh = mold_mesh.merge(hanging_mesh, merge_points=False)
        preview_path = out_dir / f"{sample_path.stem}_mold_plus_hanging_preview.stl"
        _save_stl(preview_mesh, preview_path)

    rows = _dimension_rows(sample_path, metadata, scaled_lumps, params=params)
    dims_path = out_dir / f"{sample_path.stem}_fabrication_dimensions.csv"
    dims_path.write_text(_rows_to_csv_text(rows), encoding="utf-8")
    manifest = {
        "sample": str(sample_path),
        "base_phantom_id": metadata.get("base_phantom_id"),
        "assumptions": {
            "outer_size_mm": params.outer_size_mm,
            "inner_size_mm": params.inner_size_mm,
            "wall_thickness_mm": wall_mm,
            "phantom_height_mm": params.phantom_height_mm,
            "bottom_thickness_mm": params.bottom_thickness_mm,
            "slot_length_along_wall_mm": params.slot_length_mm,
            "slot_cut_down_from_top_mm": params.slot_depth_mm,
            "slot_count": 4,
            "slot_position": "midpoint of each wall, through wall thickness, open from top edge",
            "support_diameter_mm": 2.0 * params.support_radius_mm,
            "top_rod_diameter_mm": 2.0 * params.top_rod_radius_mm,
            "reference_wall_stl": str(args.reference_wall_stl.expanduser().resolve()) if args.reference_wall_stl else None,
            "watertight_voxel_mm": float(args.watertight_voxel_mm),
            "watertight_note": "The watertight hanging STL is an implicit union of keys, rods, supports, and lumps; use it for slicing/printing.",
            "xy_scaling": "GT phantom x/y were scaled from 180 mm simulation width to the 80 mm physical cavity.",
            "z_scaling": "GT z was kept at 80 mm physical height, then shifted up by the bottom plate thickness.",
        },
        "outputs": {
            "mold_stl": str(mold_path),
            "hanging_assembly_stl": str(hanging_path),
            "hanging_assembly_watertight_stl": str(watertight_path),
            "scaled_lumps_only_stl": str(lumps_path),
            "mold_plus_hanging_preview_stl": str(preview_path) if preview_path is not None else None,
            "dimensions_csv": str(dims_path),
        },
    }
    (out_dir / "fabrication_demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(manifest), encoding="utf-8")
    print(f"Wrote fabrication demo STL files to {out_dir}")


class FabricationParams:
    def __init__(
        self,
        *,
        outer_size_mm: float,
        inner_size_mm: float,
        phantom_height_mm: float,
        bottom_thickness_mm: float,
        slot_length_mm: float,
        slot_depth_mm: float,
        support_radius_mm: float,
        top_rod_radius_mm: float,
    ) -> None:
        self.outer_size_mm = outer_size_mm
        self.inner_size_mm = inner_size_mm
        self.phantom_height_mm = phantom_height_mm
        self.bottom_thickness_mm = bottom_thickness_mm
        self.slot_length_mm = slot_length_mm
        self.slot_depth_mm = slot_depth_mm
        self.support_radius_mm = support_radius_mm
        self.top_rod_radius_mm = top_rod_radius_mm

    @property
    def wall_mm(self) -> float:
        return 0.5 * (self.outer_size_mm - self.inner_size_mm)

    @property
    def cavity_center_mm(self) -> float:
        return 0.5 * self.outer_size_mm

    @property
    def mold_top_z_mm(self) -> float:
        return self.bottom_thickness_mm + self.phantom_height_mm


def _mold_mesh(params: FabricationParams):
    pv = require_pyvista()
    o = params.outer_size_mm
    w = params.wall_mm
    h = params.mold_top_z_mm
    b = params.bottom_thickness_mm
    s = params.slot_length_mm
    slot_bottom_z = max(h - params.slot_depth_mm, b)
    mid = 0.5 * o
    half = 0.5 * s

    if not all(float(value).is_integer() for value in (o, w, h, b, s, slot_bottom_z)):
        raise ValueError("The first-pass voxel mold exporter expects integer millimeter dimensions.")
    nx, ny, nz = int(o), int(o), int(h)
    solid = np.zeros((nx, ny, nz), dtype=bool)
    solid[:, :, : int(b)] = True
    solid[: int(w), int(w) : int(o - w), int(b) :] = True
    solid[int(o - w) :, int(w) : int(o - w), int(b) :] = True
    solid[int(w) : int(o - w), : int(w), int(b) :] = True
    solid[int(w) : int(o - w), int(o - w) :, int(b) :] = True

    z0 = int(slot_bottom_z)
    lo = int(mid - half)
    hi = int(mid + half)
    solid[: int(w), lo:hi, z0:] = False
    solid[int(o - w) :, lo:hi, z0:] = False
    solid[lo:hi, : int(w), z0:] = False
    solid[lo:hi, int(o - w) :, z0:] = False
    return _occupancy_surface_mesh(pv, solid)


def _hanging_assembly_mesh(lumps: list[LumpSpec], *, params: FabricationParams, mold_top_z: float, resolution: int):
    pv = require_pyvista()
    c = params.cavity_center_mm
    w = params.wall_mm
    o = params.outer_size_mm
    z_top = mold_top_z + 5.0
    z_key_low = mold_top_z - params.slot_depth_mm
    z_key_high = z_top + params.top_rod_radius_mm
    rod_r = params.top_rod_radius_mm
    support_r = params.support_radius_mm

    parts = [
        _cylinder_between(pv, (0.0, c, z_top), (o, c, z_top), rod_r),
        _cylinder_between(pv, (c, 0.0, z_top), (c, o, z_top), rod_r),
        _sphere(pv, (c, c, z_top), 2.0),
    ]

    half = 0.5 * params.slot_length_mm
    # Four rectangular keys that sit in the four wall slots.
    parts.extend(
        [
            _box(pv, c - half, c + half, 0.0, w, z_key_low, z_key_high),
            _box(pv, c - half, c + half, o - w, o, z_key_low, z_key_high),
            _box(pv, 0.0, w, c - half, c + half, z_key_low, z_key_high),
            _box(pv, o - w, o, c - half, c + half, z_key_low, z_key_high),
        ]
    )

    for idx, lump in enumerate(lumps):
        lump_mesh = _lump_mesh(lump, resolution=resolution)
        lump_mesh.points = _meters_to_physical_mm(np.asarray(lump_mesh.points), params=params)
        parts.append(lump_mesh)

        center = _center_to_physical_mm(lump.center, params=params)
        top_z = _lump_top_mm(lump, params=params)
        attach_z = max(top_z - 1.5, params.bottom_thickness_mm)
        parts.append(_cylinder_between(pv, (center[0], center[1], z_top), (center[0], center[1], attach_z), support_r))
        parts.append(_sphere(pv, (center[0], center[1], z_top), 1.65))

        # Thin top arm from the central cross to this support, so each vertical rod is connected to the ceiling piece.
        parts.append(_cylinder_between(pv, (c, center[1], z_top), (center[0], center[1], z_top), rod_r))
        parts.append(_cylinder_between(pv, (c, c, z_top), (c, center[1], z_top), rod_r))

    return _merge(parts)


def _hanging_assembly_watertight_mesh(
    lumps: list[LumpSpec],
    *,
    params: FabricationParams,
    mold_top_z: float,
    voxel_mm: float,
):
    try:
        from skimage.measure import marching_cubes
    except ModuleNotFoundError as exc:
        raise RuntimeError("Watertight export requires scikit-image.") from exc

    pv = require_pyvista()
    c = params.cavity_center_mm
    o = params.outer_size_mm
    w = params.wall_mm
    rod_r = params.top_rod_radius_mm
    support_r = params.support_radius_mm
    z_top = mold_top_z + 5.0
    z_key_low = mold_top_z - params.slot_depth_mm
    z_key_high = z_top + rod_r
    half_slot = 0.5 * params.slot_length_mm
    voxel = max(float(voxel_mm), 0.25)
    padding = max(2.0 * voxel, params.support_radius_mm, params.top_rod_radius_mm, 2.5) + 2.0

    xmin, xmax = -padding, o + padding
    ymin, ymax = -padding, o + padding
    zmin = min(_lump_bottom_mm(lump, params=params) for lump in lumps) - padding
    zmax = z_key_high + padding
    xs = np.arange(xmin, xmax + voxel, voxel, dtype=np.float32)
    ys = np.arange(ymin, ymax + voxel, voxel, dtype=np.float32)
    zs = np.arange(zmin, zmax + voxel, voxel, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1)
    sdf = np.full(xx.shape, np.inf, dtype=np.float32)

    def union(values: np.ndarray) -> None:
        nonlocal sdf
        sdf = np.minimum(sdf, values.astype(np.float32))

    # Cross rods pass through the four slot keys, ensuring key-to-ceiling connectivity.
    union(_segment_cylinder_sdf(points, (0.0, c, z_top), (o, c, z_top), rod_r))
    union(_segment_cylinder_sdf(points, (c, 0.0, z_top), (c, o, z_top), rod_r))
    union(_sphere_sdf(points, (c, c, z_top), rod_r))

    # Four alignment keys sit in the wall slots and extend upward into the rods.
    union(_box_sdf(points, (c, 0.5 * w, 0.5 * (z_key_low + z_key_high)), (half_slot, 0.5 * w, 0.5 * (z_key_high - z_key_low))))
    union(_box_sdf(points, (c, o - 0.5 * w, 0.5 * (z_key_low + z_key_high)), (half_slot, 0.5 * w, 0.5 * (z_key_high - z_key_low))))
    union(_box_sdf(points, (0.5 * w, c, 0.5 * (z_key_low + z_key_high)), (0.5 * w, half_slot, 0.5 * (z_key_high - z_key_low))))
    union(_box_sdf(points, (o - 0.5 * w, c, 0.5 * (z_key_low + z_key_high)), (0.5 * w, half_slot, 0.5 * (z_key_high - z_key_low))))

    for lump in lumps:
        center = _center_to_physical_mm(lump.center, params=params)
        top_z = _lump_top_mm(lump, params=params)
        attach_z = max(top_z - 1.5, params.bottom_thickness_mm)
        union(_lump_sdf_mm(points, lump, params=params))
        union(_segment_cylinder_sdf(points, (center[0], center[1], z_top), (center[0], center[1], attach_z), support_r))
        union(_segment_cylinder_sdf(points, (c, center[1], z_top), (center[0], center[1], z_top), rod_r))
        union(_segment_cylinder_sdf(points, (c, c, z_top), (c, center[1], z_top), rod_r))

    if float(sdf.min()) > 0.0 or float(sdf.max()) < 0.0:
        raise RuntimeError("Implicit hanging assembly did not cross the zero level.")
    verts, faces, _normals, _values = marching_cubes(sdf, level=0.0, spacing=(voxel, voxel, voxel))
    verts += np.asarray([xmin, ymin, zmin], dtype=np.float32)
    vtk_faces = np.empty((faces.shape[0], 4), dtype=np.int64)
    vtk_faces[:, 0] = 3
    vtk_faces[:, 1:] = faces.astype(np.int64)
    return pv.PolyData(verts.astype(np.float32), vtk_faces.reshape(-1)).clean(tolerance=1e-6)


def _combined_lumps_mesh(lumps: list[LumpSpec], *, params: FabricationParams, resolution: int):
    parts = []
    for lump in lumps:
        mesh = _lump_mesh(lump, resolution=resolution)
        mesh.points = _meters_to_physical_mm(np.asarray(mesh.points), params=params)
        parts.append(mesh)
    return _merge(parts)


def _scale_lumps_to_inner(lumps: list[LumpSpec], *, phantom_size_x_mm: float, params: FabricationParams) -> list[LumpSpec]:
    scale_xy = params.inner_size_mm / phantom_size_x_mm
    scaled = []
    for lump in lumps:
        cx, cy, cz = lump.center
        rx, ry, rz = lump.radii
        scaled.append(
            replace(
                lump,
                center=(cx * scale_xy, cy * scale_xy, cz),
                radii=(rx * scale_xy, ry * scale_xy, rz),
            )
        )
    return scaled


def _meters_to_physical_mm(points: np.ndarray, *, params: FabricationParams) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64) * 1000.0
    out[:, 0] += params.cavity_center_mm
    out[:, 1] += params.cavity_center_mm
    out[:, 2] += params.bottom_thickness_mm
    return out.astype(np.float32)


def _center_to_physical_mm(center_m: tuple[float, float, float], *, params: FabricationParams) -> np.ndarray:
    center = np.asarray(center_m, dtype=np.float64).reshape(1, 3)
    return _meters_to_physical_mm(center, params=params)[0]


def _lump_top_mm(lump: LumpSpec, *, params: FabricationParams) -> float:
    z_extent = lump.radii[2] + lump.radii[0] if lump.shape == "capsule" else lump.radii[2]
    return params.bottom_thickness_mm + 1000.0 * (float(lump.center[2]) + float(z_extent))


def _lump_bottom_mm(lump: LumpSpec, *, params: FabricationParams) -> float:
    z_extent = lump.radii[2] + lump.radii[0] if lump.shape == "capsule" else lump.radii[2]
    return params.bottom_thickness_mm + 1000.0 * (float(lump.center[2]) - float(z_extent))


def _segment_cylinder_sdf(points: np.ndarray, p0, p1, radius: float) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    pa = points - p0
    ba = p1 - p0
    denom = max(float(np.dot(ba, ba)), 1e-9)
    h = np.clip(np.sum(pa * ba, axis=-1) / denom, 0.0, 1.0)
    nearest = p0 + h[..., None] * ba
    return np.linalg.norm(points - nearest, axis=-1) - float(radius)


def _sphere_sdf(points: np.ndarray, center, radius: float) -> np.ndarray:
    return np.linalg.norm(points - np.asarray(center, dtype=np.float32), axis=-1) - float(radius)


def _box_sdf(points: np.ndarray, center, half_size) -> np.ndarray:
    q = np.abs(points - np.asarray(center, dtype=np.float32)) - np.asarray(half_size, dtype=np.float32)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def _lump_sdf_mm(points: np.ndarray, lump: LumpSpec, *, params: FabricationParams) -> np.ndarray:
    center = _center_to_physical_mm(lump.center, params=params).astype(np.float32)
    radii = (np.asarray(lump.radii, dtype=np.float32) * 1000.0).clip(min=1e-6)
    rel = points - center
    rel = _rotate_xy(rel, -float(lump.yaw))

    if lump.shape in {"sphere", "ellipsoid"}:
        return (np.sqrt(np.sum((rel / radii) ** 2, axis=-1)) - 1.0) * float(np.min(radii))
    if lump.shape == "box":
        q = np.abs(rel) - radii
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside
    if lump.shape == "cylinder":
        radial = np.sqrt((rel[..., 0] / radii[0]) ** 2 + (rel[..., 1] / radii[1]) ** 2) - 1.0
        axial = np.abs(rel[..., 2]) / radii[2] - 1.0
        return np.maximum(radial, axial) * float(np.min(radii))
    if lump.shape == "capsule":
        radius = float(radii[0])
        half_axis = float(radii[2])
        nearest_z = np.clip(rel[..., 2], -half_axis, half_axis)
        dx = rel[..., 0]
        dy = rel[..., 1]
        dz = rel[..., 2] - nearest_z
        return np.sqrt(dx * dx + dy * dy + dz * dz) - radius
    raise ValueError(f"Unsupported lump shape: {lump.shape}")


def _rotate_xy(points: np.ndarray, yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    out = np.array(points, copy=True)
    x = points[..., 0]
    y = points[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def _box(pv, xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float):
    return pv.Box(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))


def _sphere(pv, center: tuple[float, float, float], radius: float):
    return pv.Sphere(radius=radius, center=center, theta_resolution=32, phi_resolution=16)


def _cylinder_between(pv, p0, p1, radius: float):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return _sphere(pv, tuple(p0), radius)
    axis = delta / length
    helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis, helper))) > 0.95:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    n = 32
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle = np.asarray([np.cos(angles), np.sin(angles)]).T
    bottom = p0 + float(radius) * (circle[:, 0:1] * u + circle[:, 1:2] * v)
    top = p1 + float(radius) * (circle[:, 0:1] * u + circle[:, 1:2] * v)
    vertices = np.vstack([bottom, top, p0.reshape(1, 3), p1.reshape(1, 3)])
    bottom_center = 2 * n
    top_center = 2 * n + 1
    faces: list[int] = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([3, i, j, n + j])
        faces.extend([3, i, n + j, n + i])
        faces.extend([3, bottom_center, j, i])
        faces.extend([3, top_center, n + i, n + j])
    return pv.PolyData(vertices.astype(np.float32), np.asarray(faces, dtype=np.int64))


def _merge(parts):
    if not parts:
        return require_pyvista().PolyData()
    merged = parts[0].copy(deep=True)
    for part in parts[1:]:
        merged = merged.merge(part, merge_points=False)
    return merged


def _occupancy_surface_mesh(pv, solid: np.ndarray):
    vertices: list[tuple[float, float, float]] = []
    faces: list[int] = []
    vertex_index: dict[tuple[int, int, int], int] = {}

    def add_vertex(coord: tuple[int, int, int]) -> int:
        if coord not in vertex_index:
            vertex_index[coord] = len(vertices)
            vertices.append((float(coord[0]), float(coord[1]), float(coord[2])))
        return vertex_index[coord]

    def add_face(coords: list[tuple[int, int, int]]) -> None:
        faces.extend([4, *(add_vertex(coord) for coord in coords)])

    nx, ny, nz = solid.shape
    for x, y, z in np.argwhere(solid):
        if x == 0 or not solid[x - 1, y, z]:
            add_face([(x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z)])
        if x == nx - 1 or not solid[x + 1, y, z]:
            add_face([(x + 1, y, z), (x + 1, y + 1, z), (x + 1, y + 1, z + 1), (x + 1, y, z + 1)])
        if y == 0 or not solid[x, y - 1, z]:
            add_face([(x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1)])
        if y == ny - 1 or not solid[x, y + 1, z]:
            add_face([(x, y + 1, z), (x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y + 1, z)])
        if z == 0 or not solid[x, y, z - 1]:
            add_face([(x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z)])
        if z == nz - 1 or not solid[x, y, z + 1]:
            add_face([(x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)])

    return pv.PolyData(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64))


def _save_stl(mesh, path: Path) -> None:
    mesh.triangulate().save(path)


def _dimension_rows(sample_path: Path, metadata: dict, lumps: list[LumpSpec], *, params: FabricationParams):
    rows = []
    for idx, lump in enumerate(lumps):
        center = _center_to_physical_mm(lump.center, params=params)
        rx, ry, rz = np.asarray(lump.radii, dtype=np.float64) * 1000.0
        top_z = _lump_top_mm(lump, params=params)
        rows.append(
            {
                "sample": sample_path.name,
                "base_phantom_id": metadata.get("base_phantom_id", ""),
                "lump_index": idx,
                "shape": lump.shape,
                "center_x_mm_in_mold_frame": center[0],
                "center_y_mm_in_mold_frame": center[1],
                "center_z_mm_from_mold_bottom": center[2],
                "center_depth_from_ecoflex_top_mm": params.mold_top_z_mm - center[2],
                "top_z_mm_from_mold_bottom": top_z,
                "top_depth_from_ecoflex_top_mm": params.mold_top_z_mm - top_z,
                "radius_x_mm": rx,
                "radius_y_mm": ry,
                "radius_z_or_half_axis_mm": rz,
                "yaw_deg": float(np.degrees(lump.yaw)),
            }
        )
    return rows


def _rows_to_csv_text(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _readme(manifest: dict) -> str:
    assumptions = manifest["assumptions"]
    return f"""# Fabrication Demo Mold

This is a first-pass STL interpretation for review, not a final manufacturing release.

Assumptions in this export:

- Outer square: {assumptions["outer_size_mm"]} mm x {assumptions["outer_size_mm"]} mm
- Inner cavity: {assumptions["inner_size_mm"]} mm x {assumptions["inner_size_mm"]} mm
- Wall thickness: {assumptions["wall_thickness_mm"]} mm
- Ecoflex height: {assumptions["phantom_height_mm"]} mm
- Bottom plate thickness: {assumptions["bottom_thickness_mm"]} mm
- Four top alignment slots: {assumptions["slot_length_along_wall_mm"]} mm along the wall, through wall thickness, cut {assumptions["slot_cut_down_from_top_mm"]} mm down from the top edge.
- Hanging vertical support diameter: {assumptions["support_diameter_mm"]} mm
- Hanging top rod diameter: {assumptions["top_rod_diameter_mm"]} mm
- Reference wall STL: {assumptions["reference_wall_stl"]}
- Watertight implicit-union voxel size: {assumptions["watertight_voxel_mm"]} mm

The hanging assembly includes the GT lumps, a top cross support, four alignment keys for the wall slots, and vertical support rods that penetrate slightly into the top of each lump.

Use `*_hanging_assembly_d5_support_watertight.stl` for slicing/printing. The non-watertight `*_hanging_assembly_d5_support.stl` is kept only as an exact component-layout preview.
"""


if __name__ == "__main__":
    main()
