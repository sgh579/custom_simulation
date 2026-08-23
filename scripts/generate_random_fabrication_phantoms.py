#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from export_fabrication_demo_mold import (  # noqa: E402
    FabricationParams,
    _combined_lumps_mesh,
    _hanging_assembly_mesh,
    _hanging_assembly_watertight_mesh,
    _mold_mesh,
    _save_stl,
    _rows_to_csv_text,
)
from palpation_sim.phantom import LumpSpec  # noqa: E402


SMOOTH_SHAPES = ("sphere", "ellipsoid", "cylinder", "capsule")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random fabrication-first phantom configurations and hanging insert STLs."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-configs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--count-mode", choices=["balanced", "uniform"], default="balanced")
    parser.add_argument("--min-lumps", type=int, default=1)
    parser.add_argument("--max-lumps", type=int, default=4)
    parser.add_argument("--shapes", type=str, default=",".join(SMOOTH_SHAPES))

    parser.add_argument("--outer-size-mm", type=float, default=90.0)
    parser.add_argument("--inner-size-mm", type=float, default=80.0)
    parser.add_argument("--phantom-height-mm", type=float, default=80.0)
    parser.add_argument("--bottom-thickness-mm", type=float, default=5.0)
    parser.add_argument("--slot-length-mm", type=float, default=9.9)
    parser.add_argument("--slot-depth-mm", type=float, default=8.0)
    parser.add_argument(
        "--slot-reference-z-mm",
        type=float,
        default=None,
        help=(
            "Z level used as the lower reference for the top hanging assembly. "
            "By default this is the wall inner top, outer_size - wall_thickness, so the ceiling does not move "
            "when the cast phantom height is reduced."
        ),
    )
    parser.add_argument("--support-radius-mm", type=float, default=2.5)
    parser.add_argument("--top-rod-radius-mm", type=float, default=2.5)

    parser.add_argument("--min-radius-mm", type=float, default=6.0)
    parser.add_argument("--max-radius-mm", type=float, default=12.0)
    parser.add_argument("--min-z-extent-mm", type=float, default=6.0)
    parser.add_argument("--max-z-extent-mm", type=float, default=16.0)
    parser.add_argument("--boundary-clearance-mm", type=float, default=3.0)
    parser.add_argument("--lump-clearance-mm", type=float, default=3.0)
    parser.add_argument("--support-clearance-mm", type=float, default=1.5)

    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--watertight-voxel-mm", type=float, default=0.5)
    parser.add_argument("--no-component-preview", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--reference-wall-stl",
        type=Path,
        default=Path("/Users/goodmansun/Downloads/202605_mouldSetV3_boxWall.STL"),
        help="Optional existing boxWall STL to copy into the output folder.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    shapes = _parse_shapes(args.shapes)
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
    constraints = PlacementConstraints(
        min_radius_mm=float(args.min_radius_mm),
        max_radius_mm=float(args.max_radius_mm),
        min_z_extent_mm=float(args.min_z_extent_mm),
        max_z_extent_mm=float(args.max_z_extent_mm),
        boundary_clearance_mm=float(args.boundary_clearance_mm),
        lump_clearance_mm=float(args.lump_clearance_mm),
        support_clearance_mm=float(args.support_clearance_mm),
    )

    slot_reference_z = (
        float(args.slot_reference_z_mm)
        if args.slot_reference_z_mm is not None
        else float(params.outer_size_mm - params.wall_mm)
    )
    hanging_top_center_z = slot_reference_z + 5.0
    wall_path = None
    if args.reference_wall_stl and args.reference_wall_stl.expanduser().exists():
        src = args.reference_wall_stl.expanduser().resolve()
        wall_path = out_dir / f"boxWall_reference_outer{params.outer_size_mm:g}_inner{params.inner_size_mm:g}.stl"
        shutil.copy2(src, wall_path)
    else:
        wall_path = out_dir / f"mold_outer{params.outer_size_mm:g}_inner{params.inner_size_mm:g}_slot{params.slot_length_mm:g}.stl"
        _save_stl(_mold_mesh(params), wall_path)

    counts = _lump_counts(
        num_configs=int(args.num_configs),
        min_lumps=int(args.min_lumps),
        max_lumps=int(args.max_lumps),
        mode=args.count_mode,
        rng=rng,
    )

    all_rows: list[dict[str, object]] = []
    configs: list[dict[str, object]] = []
    for config_index, lump_count in enumerate(counts):
        config_id = f"random_phantom_{config_index:03d}"
        config_dir = out_dir / config_id
        config_dir.mkdir(exist_ok=True)
        lumps_mm = sample_lumps_mm(
            rng,
            lump_count=int(lump_count),
            params=params,
            constraints=constraints,
            shapes=shapes,
        )
        lumps = [_lump_mm_to_spec(lump) for lump in lumps_mm]
        quality = _quality_summary(lumps_mm, params=params, constraints=constraints)

        hanging_mesh = _hanging_assembly_mesh(
            lumps,
            params=params,
            mold_top_z=slot_reference_z,
            resolution=int(args.resolution),
        )
        hanging_path = config_dir / f"{config_id}_hanging_assembly_d5_support.stl"
        if not args.no_component_preview:
            _save_stl(hanging_mesh, hanging_path)
        else:
            hanging_path = None

        watertight_mesh = _hanging_assembly_watertight_mesh(
            lumps,
            params=params,
            mold_top_z=slot_reference_z,
            voxel_mm=float(args.watertight_voxel_mm),
        )
        watertight_path = config_dir / f"{config_id}_hanging_assembly_d5_support_watertight.stl"
        _save_stl(watertight_mesh, watertight_path)

        lumps_path = config_dir / f"{config_id}_lumps_only.stl"
        _save_stl(_combined_lumps_mesh(lumps, params=params, resolution=int(args.resolution)), lumps_path)

        rows = _dimension_rows(config_id, lumps_mm, params=params)
        dims_path = config_dir / f"{config_id}_dimensions.csv"
        dims_path.write_text(_rows_to_csv_text(rows), encoding="utf-8")
        all_rows.extend(rows)

        config_manifest = {
            "config_id": config_id,
            "seed": int(args.seed),
            "config_index": config_index,
            "lump_count": int(lump_count),
            "units": "mm",
            "frame": {
                "x_y": (
                    f"mold frame, 0..{params.outer_size_mm:g} mm outer wall; "
                    f"cavity is {params.wall_mm:g}..{params.outer_size_mm - params.wall_mm:g} mm"
                ),
                "z": (
                    f"mold bottom-up; Ecoflex is {params.bottom_thickness_mm:g}.."
                    f"{params.mold_top_z_mm:g} mm"
                ),
            },
            "placement_constraints": constraints.to_dict(),
            "fabrication_params": _fabrication_params_dict(params),
            "hanging_layout": {
                "slot_reference_z_mm": slot_reference_z,
                "top_rod_center_z_mm": hanging_top_center_z,
                "top_rod_max_z_mm": hanging_top_center_z + params.top_rod_radius_mm,
                "note": "Cast phantom height and hanging ceiling height are intentionally decoupled.",
            },
            "quality": quality,
            "lumps": [lump.to_json_dict(params=params) for lump in lumps_mm],
            "outputs": {
                "hanging_assembly_watertight_stl": str(watertight_path),
                "hanging_assembly_component_preview_stl": str(hanging_path) if hanging_path is not None else None,
                "lumps_only_stl": str(lumps_path),
                "dimensions_csv": str(dims_path),
            },
        }
        config_json = config_dir / f"{config_id}_config.json"
        config_json.write_text(json.dumps(config_manifest, indent=2), encoding="utf-8")
        configs.append(
            {
                "config_id": config_id,
                "lump_count": int(lump_count),
                "directory": str(config_dir),
                "config_json": str(config_json),
                "hanging_assembly_watertight_stl": str(watertight_path),
                "lumps_only_stl": str(lumps_path),
                "min_boundary_margin_mm": quality["min_boundary_margin_mm"],
                "min_pair_sphere_gap_mm": quality["min_pair_sphere_gap_mm"],
                "min_support_xy_gap_mm": quality["min_support_xy_gap_mm"],
            }
        )
        print(f"wrote {config_id}: {lump_count} lumps -> {watertight_path}", flush=True)

    all_dims = out_dir / "all_random_fabrication_dimensions.csv"
    all_dims.write_text(_rows_to_csv_text(all_rows), encoding="utf-8")
    config_table = out_dir / "random_fabrication_configurations.csv"
    config_table.write_text(_rows_to_csv_text(configs), encoding="utf-8")

    manifest = {
        "description": "Fabrication-first random phantom configurations; not matched to synthetic training data.",
        "seed": int(args.seed),
        "num_configs": len(configs),
        "count_mode": args.count_mode,
        "shapes": shapes,
        "units": "mm",
        "fabrication_params": _fabrication_params_dict(params),
        "hanging_layout": {
            "slot_reference_z_mm": slot_reference_z,
            "top_rod_center_z_mm": hanging_top_center_z,
            "top_rod_max_z_mm": hanging_top_center_z + params.top_rod_radius_mm,
            "note": "Cast phantom remains at the bottom; the hanging ceiling remains near the wall top.",
        },
        "placement_constraints": constraints.to_dict(),
        "reference_wall_stl": str(wall_path),
        "outputs": {
            "all_dimensions_csv": str(all_dims),
            "configurations_csv": str(config_table),
        },
        "configurations": configs,
    }
    manifest_path = out_dir / "random_fabrication_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(manifest), encoding="utf-8")

    if not args.no_zip:
        zip_path = out_dir.with_suffix(".zip")
        _zip_dir(out_dir, zip_path)
        print(f"zipped output -> {zip_path}", flush=True)
    print(f"Wrote random fabrication phantom batch to {out_dir}")


class PlacementConstraints:
    def __init__(
        self,
        *,
        min_radius_mm: float,
        max_radius_mm: float,
        min_z_extent_mm: float,
        max_z_extent_mm: float,
        boundary_clearance_mm: float,
        lump_clearance_mm: float,
        support_clearance_mm: float,
    ) -> None:
        if min_radius_mm <= 0 or max_radius_mm <= min_radius_mm:
            raise ValueError("Expected 0 < min_radius_mm < max_radius_mm.")
        if min_z_extent_mm <= 0 or max_z_extent_mm <= min_z_extent_mm:
            raise ValueError("Expected 0 < min_z_extent_mm < max_z_extent_mm.")
        self.min_radius_mm = min_radius_mm
        self.max_radius_mm = max_radius_mm
        self.min_z_extent_mm = min_z_extent_mm
        self.max_z_extent_mm = max_z_extent_mm
        self.boundary_clearance_mm = boundary_clearance_mm
        self.lump_clearance_mm = lump_clearance_mm
        self.support_clearance_mm = support_clearance_mm

    def to_dict(self) -> dict[str, float]:
        return {
            "min_radius_mm": self.min_radius_mm,
            "max_radius_mm": self.max_radius_mm,
            "min_z_extent_mm": self.min_z_extent_mm,
            "max_z_extent_mm": self.max_z_extent_mm,
            "boundary_clearance_mm": self.boundary_clearance_mm,
            "lump_clearance_mm": self.lump_clearance_mm,
            "support_clearance_mm": self.support_clearance_mm,
        }


class LumpMM:
    def __init__(
        self,
        *,
        shape: str,
        center_mm: tuple[float, float, float],
        radii_mm: tuple[float, float, float],
        yaw_rad: float,
    ) -> None:
        self.shape = shape
        self.center_mm = center_mm
        self.radii_mm = radii_mm
        self.yaw_rad = yaw_rad

    @property
    def z_extent_mm(self) -> float:
        return _z_extent_mm(self.shape, self.radii_mm)

    @property
    def xy_footprint_radius_mm(self) -> float:
        return max(float(self.radii_mm[0]), float(self.radii_mm[1]))

    @property
    def bounding_radius_mm(self) -> float:
        hx, hy, hz = _conservative_extents_mm(self.shape, self.radii_mm)
        return float(math.sqrt(hx * hx + hy * hy + hz * hz))

    def to_json_dict(self, *, params: FabricationParams) -> dict[str, object]:
        cx, cy, cz = self.center_mm
        rx, ry, rz = self.radii_mm
        return {
            "shape": self.shape,
            "center_x_mm_in_mold_frame": cx + params.cavity_center_mm,
            "center_y_mm_in_mold_frame": cy + params.cavity_center_mm,
            "center_z_mm_from_mold_bottom": cz + params.bottom_thickness_mm,
            "center_x_mm_in_cavity_frame": cx,
            "center_y_mm_in_cavity_frame": cy,
            "center_z_mm_from_ecoflex_bottom": cz,
            "center_depth_from_ecoflex_top_mm": params.phantom_height_mm - cz,
            "radii_mm": [rx, ry, rz],
            "z_extent_mm": self.z_extent_mm,
            "top_z_mm_from_mold_bottom": params.bottom_thickness_mm + cz + self.z_extent_mm,
            "top_depth_from_ecoflex_top_mm": params.phantom_height_mm - cz - self.z_extent_mm,
            "bottom_z_mm_from_mold_bottom": params.bottom_thickness_mm + cz - self.z_extent_mm,
            "bottom_depth_from_ecoflex_top_mm": params.phantom_height_mm - cz + self.z_extent_mm,
            "yaw_deg": math.degrees(self.yaw_rad),
        }


def sample_lumps_mm(
    rng: np.random.Generator,
    *,
    lump_count: int,
    params: FabricationParams,
    constraints: PlacementConstraints,
    shapes: tuple[str, ...],
) -> list[LumpMM]:
    max_layout_attempts = 500
    max_candidate_attempts = 2000
    for _layout_attempt in range(max_layout_attempts):
        placed: list[LumpMM] = []
        for _idx in range(lump_count):
            candidate = None
            for _candidate_attempt in range(max_candidate_attempts):
                maybe = _sample_candidate_mm(rng, shapes=shapes, constraints=constraints)
                center = _sample_center_mm(rng, maybe, params=params, constraints=constraints)
                if center is None:
                    continue
                maybe.center_mm = center
                if _placement_ok(maybe, placed, params=params, constraints=constraints):
                    candidate = maybe
                    break
            if candidate is None:
                break
            placed.append(candidate)
        if len(placed) == lump_count:
            return placed
    raise RuntimeError(
        f"Could not place {lump_count} non-overlapping lumps in {params.inner_size_mm:g} x "
        f"{params.inner_size_mm:g} x {params.phantom_height_mm:g} mm. "
        "Reduce size limits, clearances, or lump count."
    )


def _sample_candidate_mm(
    rng: np.random.Generator,
    *,
    shapes: tuple[str, ...],
    constraints: PlacementConstraints,
) -> LumpMM:
    shape = str(rng.choice(shapes))
    r_min = constraints.min_radius_mm
    r_max = constraints.max_radius_mm
    z_min = constraints.min_z_extent_mm
    z_max = constraints.max_z_extent_mm

    if shape == "sphere":
        radius = float(rng.uniform(r_min, min(r_max, z_max)))
        radii = (radius, radius, radius)
    elif shape == "ellipsoid":
        base = float(rng.uniform(r_min, r_max))
        rx = float(np.clip(base * rng.uniform(0.75, 1.25), r_min, r_max))
        ry = float(np.clip(base * rng.uniform(0.75, 1.25), r_min, r_max))
        rz = float(np.clip(base * rng.uniform(0.70, 1.35), z_min, z_max))
        radii = (rx, ry, rz)
    elif shape == "cylinder":
        base = float(rng.uniform(r_min, 0.92 * r_max))
        rx = float(np.clip(base * rng.uniform(0.85, 1.15), r_min, r_max))
        ry = float(np.clip(base * rng.uniform(0.85, 1.15), r_min, r_max))
        rz = float(rng.uniform(z_min, 0.85 * z_max))
        radii = (rx, ry, rz)
    elif shape == "capsule":
        radius_hi = min(0.82 * r_max, 0.72 * z_max)
        radius = float(rng.uniform(r_min, radius_hi))
        half_axis_hi = max(z_max - radius, 0.0)
        half_axis_lo = max(z_min - radius, 0.0)
        half_axis = float(rng.uniform(half_axis_lo, half_axis_hi))
        radii = (radius, radius, half_axis)
    elif shape == "box":
        base = float(rng.uniform(r_min, 0.9 * r_max))
        rx = float(np.clip(base * rng.uniform(0.8, 1.2), r_min, r_max))
        ry = float(np.clip(base * rng.uniform(0.8, 1.2), r_min, r_max))
        rz = float(rng.uniform(z_min, 0.8 * z_max))
        radii = (rx, ry, rz)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return LumpMM(shape=shape, center_mm=(0.0, 0.0, 0.0), radii_mm=radii, yaw_rad=float(rng.uniform(-math.pi, math.pi)))


def _sample_center_mm(
    rng: np.random.Generator,
    lump: LumpMM,
    *,
    params: FabricationParams,
    constraints: PlacementConstraints,
) -> tuple[float, float, float] | None:
    hx, hy, hz = _conservative_extents_mm(lump.shape, lump.radii_mm)
    half = 0.5 * params.inner_size_mm
    c = constraints.boundary_clearance_mm
    x_lo, x_hi = -half + hx + c, half - hx - c
    y_lo, y_hi = -half + hy + c, half - hy - c
    z_lo, z_hi = hz + c, params.phantom_height_mm - hz - c
    if x_hi < x_lo or y_hi < y_lo or z_hi < z_lo:
        return None
    return (
        float(rng.uniform(x_lo, x_hi)),
        float(rng.uniform(y_lo, y_hi)),
        float(rng.uniform(z_lo, z_hi)),
    )


def _placement_ok(
    candidate: LumpMM,
    placed: list[LumpMM],
    *,
    params: FabricationParams,
    constraints: PlacementConstraints,
) -> bool:
    cc = np.asarray(candidate.center_mm, dtype=np.float64)
    for lump in placed:
        lc = np.asarray(lump.center_mm, dtype=np.float64)
        gap = float(np.linalg.norm(cc - lc)) - candidate.bounding_radius_mm - lump.bounding_radius_mm
        if gap < constraints.lump_clearance_mm:
            return False
        xy_gap = (
            float(np.linalg.norm(cc[:2] - lc[:2]))
            - candidate.xy_footprint_radius_mm
            - lump.xy_footprint_radius_mm
        )
        if xy_gap < params.support_radius_mm + constraints.support_clearance_mm:
            return False
    return True


def _quality_summary(
    lumps: list[LumpMM],
    *,
    params: FabricationParams,
    constraints: PlacementConstraints,
) -> dict[str, object]:
    boundary_margins = []
    half = 0.5 * params.inner_size_mm
    for lump in lumps:
        hx, hy, hz = _conservative_extents_mm(lump.shape, lump.radii_mm)
        cx, cy, cz = lump.center_mm
        boundary_margins.extend(
            [
                cx - (-half) - hx,
                half - cx - hx,
                cy - (-half) - hy,
                half - cy - hy,
                cz - hz,
                params.phantom_height_mm - cz - hz,
            ]
        )

    pair_sphere_gaps = []
    support_xy_gaps = []
    for i, a in enumerate(lumps):
        ac = np.asarray(a.center_mm, dtype=np.float64)
        for b in lumps[i + 1 :]:
            bc = np.asarray(b.center_mm, dtype=np.float64)
            pair_sphere_gaps.append(float(np.linalg.norm(ac - bc)) - a.bounding_radius_mm - b.bounding_radius_mm)
            support_xy_gaps.append(
                float(np.linalg.norm(ac[:2] - bc[:2])) - a.xy_footprint_radius_mm - b.xy_footprint_radius_mm
            )

    return {
        "min_boundary_margin_mm": float(min(boundary_margins)) if boundary_margins else None,
        "boundary_clearance_requested_mm": constraints.boundary_clearance_mm,
        "min_pair_sphere_gap_mm": float(min(pair_sphere_gaps)) if pair_sphere_gaps else None,
        "lump_clearance_requested_mm": constraints.lump_clearance_mm,
        "min_support_xy_gap_mm": float(min(support_xy_gaps)) if support_xy_gaps else None,
        "support_radius_mm": params.support_radius_mm,
        "note": "Pair gaps use conservative bounding spheres; positive values guarantee no volume overlap under that bound.",
    }


def _lump_mm_to_spec(lump: LumpMM) -> LumpSpec:
    cx, cy, cz = lump.center_mm
    rx, ry, rz = lump.radii_mm
    return LumpSpec(
        shape=lump.shape,  # type: ignore[arg-type]
        center=(cx / 1000.0, cy / 1000.0, cz / 1000.0),
        radii=(rx / 1000.0, ry / 1000.0, rz / 1000.0),
        stiffness_multiplier=20.0,
        yaw=lump.yaw_rad,
    )


def _dimension_rows(config_id: str, lumps: list[LumpMM], *, params: FabricationParams) -> list[dict[str, object]]:
    rows = []
    for idx, lump in enumerate(lumps):
        info = lump.to_json_dict(params=params)
        rows.append(
            {
                "config_id": config_id,
                "lump_index": idx,
                "shape": lump.shape,
                "center_x_mm_in_mold_frame": info["center_x_mm_in_mold_frame"],
                "center_y_mm_in_mold_frame": info["center_y_mm_in_mold_frame"],
                "center_z_mm_from_mold_bottom": info["center_z_mm_from_mold_bottom"],
                "center_x_mm_in_cavity_frame": info["center_x_mm_in_cavity_frame"],
                "center_y_mm_in_cavity_frame": info["center_y_mm_in_cavity_frame"],
                "center_z_mm_from_ecoflex_bottom": info["center_z_mm_from_ecoflex_bottom"],
                "center_depth_from_ecoflex_top_mm": info["center_depth_from_ecoflex_top_mm"],
                "top_z_mm_from_mold_bottom": info["top_z_mm_from_mold_bottom"],
                "top_depth_from_ecoflex_top_mm": info["top_depth_from_ecoflex_top_mm"],
                "bottom_z_mm_from_mold_bottom": info["bottom_z_mm_from_mold_bottom"],
                "bottom_depth_from_ecoflex_top_mm": info["bottom_depth_from_ecoflex_top_mm"],
                "radius_x_mm": lump.radii_mm[0],
                "radius_y_mm": lump.radii_mm[1],
                "radius_z_or_half_axis_mm": lump.radii_mm[2],
                "effective_z_extent_mm": lump.z_extent_mm,
                "yaw_deg": info["yaw_deg"],
            }
        )
    return rows


def _conservative_extents_mm(shape: str, radii_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    rx, ry, rz = (float(v) for v in radii_mm)
    if shape == "capsule":
        return rx, ry, rx + rz
    xy = max(rx, ry)
    return xy, xy, rz


def _z_extent_mm(shape: str, radii_mm: tuple[float, float, float]) -> float:
    if shape == "capsule":
        return float(radii_mm[0] + radii_mm[2])
    return float(radii_mm[2])


def _lump_counts(
    *,
    num_configs: int,
    min_lumps: int,
    max_lumps: int,
    mode: str,
    rng: np.random.Generator,
) -> list[int]:
    if min_lumps < 1 or max_lumps < min_lumps:
        raise ValueError("Expected 1 <= min_lumps <= max_lumps.")
    if mode == "uniform":
        return [int(rng.integers(min_lumps, max_lumps + 1)) for _ in range(num_configs)]
    base = list(range(min_lumps, max_lumps + 1))
    counts = [base[i % len(base)] for i in range(num_configs)]
    rng.shuffle(counts)
    return counts


def _parse_shapes(text: str) -> tuple[str, ...]:
    shapes = tuple(part.strip() for part in text.split(",") if part.strip())
    allowed = set(SMOOTH_SHAPES) | {"box"}
    bad = [shape for shape in shapes if shape not in allowed]
    if bad:
        raise ValueError(f"Unsupported shapes: {bad}; allowed: {sorted(allowed)}")
    if not shapes:
        raise ValueError("At least one shape is required.")
    return shapes


def _fabrication_params_dict(params: FabricationParams) -> dict[str, float]:
    return {
        "outer_size_mm": params.outer_size_mm,
        "inner_size_mm": params.inner_size_mm,
        "wall_thickness_mm": params.wall_mm,
        "phantom_height_mm": params.phantom_height_mm,
        "bottom_thickness_mm": params.bottom_thickness_mm,
        "ecoflex_bottom_z_mm": params.bottom_thickness_mm,
        "ecoflex_top_z_mm": params.mold_top_z_mm,
        "slot_length_mm": params.slot_length_mm,
        "slot_depth_mm": params.slot_depth_mm,
        "support_diameter_mm": 2.0 * params.support_radius_mm,
        "top_rod_diameter_mm": 2.0 * params.top_rod_radius_mm,
    }


def _readme(manifest: dict[str, object]) -> str:
    params = manifest["fabrication_params"]
    constraints = manifest["placement_constraints"]
    hanging = manifest["hanging_layout"]
    return f"""# Random Fabrication Phantom Batch

These configurations are newly randomized for physical manufacturing. They are not intended to match the old synthetic dataset geometry.

Units: millimeters.

Frame:
- Mold frame X/Y: outer wall is 0..{params["outer_size_mm"]} mm; cavity is {params["wall_thickness_mm"]}..{params["outer_size_mm"] - params["wall_thickness_mm"]} mm.
- Mold frame Z: Ecoflex occupies {params["ecoflex_bottom_z_mm"]}..{params["ecoflex_top_z_mm"]} mm.
- Hanging ceiling: top rod center is at z={hanging["top_rod_center_z_mm"]} mm; this stays near the wall top even when the cast phantom height is only {params["phantom_height_mm"]} mm.

Sampling constraints:
- Lump count range: 1..4, count mode `{manifest["count_mode"]}`.
- Shapes: {", ".join(manifest["shapes"])}.
- Radius / half-axis range: {constraints["min_radius_mm"]}..{constraints["max_radius_mm"]} mm.
- Effective z extent range: {constraints["min_z_extent_mm"]}..{constraints["max_z_extent_mm"]} mm.
- Boundary clearance: at least {constraints["boundary_clearance_mm"]} mm.
- Conservative inter-lump clearance: at least {constraints["lump_clearance_mm"]} mm.

Use each `*_hanging_assembly_d5_support_watertight.stl` for slicing/printing. The component-preview STL is kept only for visual inspection.
"""


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))


if __name__ == "__main__":
    main()
