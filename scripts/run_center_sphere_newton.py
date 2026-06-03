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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.config import MaterialConfig, PhantomConfig, ScanConfig
from palpation_sim.exports import build_ground_truth_metadata, write_ground_truth_metadata
from palpation_sim.features import extract_feature_map
from palpation_sim.newton_vbd import NewtonVBDPalpationSimulator
from palpation_sim.phantom import (
    LumpSpec,
    create_structured_tet_mesh,
    material_arrays_for_lumps,
    mask_for_scan_grid,
)
from palpation_sim.workflow import DEFAULT_NEWTON_ROOT, REQUIRED_NEWTON_DEVICE, require_runtime_environment


MM = 1.0e-3
DEFAULT_SIZE_M = (80.0 * MM, 80.0 * MM, 25.0 * MM)


class ChunkScanConfig(ScanConfig):
    def __init__(self, *args, x_values_override: np.ndarray, y_values_override: np.ndarray, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._x_values_override = np.asarray(x_values_override, dtype=np.float32)
        self._y_values_override = np.asarray(y_values_override, dtype=np.float32)

    def x_values(self, phantom: PhantomConfig) -> list[float]:
        del phantom
        return [float(v) for v in self._x_values_override]

    def y_values(self, phantom: PhantomConfig) -> list[float]:
        del phantom
        return [float(v) for v in self._y_values_override]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a centered-sphere Newton/VBD palpation sample.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/center_sphere_newton_vbd_48x48x32_scan20x20"))
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT, help="Pinned Newton source root.")
    parser.add_argument("--device", type=str, default=REQUIRED_NEWTON_DEVICE, help="Pinned Warp/Newton CUDA device.")
    parser.add_argument("--row-chunk-size", type=int, default=1)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-end", type=int, default=None)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--no-assemble", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-features", action="store_true")

    parser.add_argument("--cells-x", type=int, default=48)
    parser.add_argument("--cells-y", type=int, default=48)
    parser.add_argument("--cells-z", type=int, default=32)
    parser.add_argument("--grid-h", type=int, default=20)
    parser.add_argument("--grid-w", type=int, default=20)
    parser.add_argument("--edge-margin-mm", type=float, default=4.0)
    parser.add_argument("--probe-diameter-mm", type=float, default=8.0)
    parser.add_argument("--max-indentation-mm", type=float, default=8.0)
    parser.add_argument("--press-steps", type=int, default=64)
    parser.add_argument("--soft-contact-margin-mm", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=3)
    parser.add_argument("--vbd-iterations", type=int, default=5)
    parser.add_argument("--k-mu", type=float, default=2.0e5)
    parser.add_argument("--k-lambda", type=float, default=2.0e5)
    parser.add_argument("--soft-contact-ke", type=float, default=2.0e6)
    parser.add_argument("--sphere-radius-mm", type=float, default=10.0)
    parser.add_argument("--stiffness-multiplier", type=float, default=100.0)
    args = parser.parse_args()
    require_runtime_environment(require_newton=True, newton_root=args.newton_root)

    phantom = _phantom(args)
    material = _material(args)
    scan = _scan(args)
    lump = _center_sphere(phantom, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = args.out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(args.out_dir / "run_config.json", args, phantom, material, scan, [lump])

    if not args.assemble_only:
        _run_chunks(args, phantom, material, scan, [lump], chunk_dir)
    if not args.no_assemble:
        _assemble(args, phantom, material, scan, [lump], chunk_dir)


def _run_chunks(
    args: argparse.Namespace,
    phantom: PhantomConfig,
    material: MaterialConfig,
    full_scan: ScanConfig,
    lumps: Sequence[LumpSpec],
    chunk_dir: Path,
) -> None:
    xs = np.asarray(full_scan.x_values(phantom), dtype=np.float32)
    ys = np.asarray(full_scan.y_values(phantom), dtype=np.float32)
    row_end = len(ys) if args.row_end is None else min(int(args.row_end), len(ys))
    row_start = max(int(args.row_start), 0)
    chunk_size = max(int(args.row_chunk_size), 1)

    for start in range(row_start, row_end, chunk_size):
        end = min(start + chunk_size, row_end)
        chunk_path = _chunk_path(chunk_dir, start, end)
        if args.resume and chunk_path.exists():
            print(f"skip existing {chunk_path}", flush=True)
            continue

        chunk_scan = ChunkScanConfig(
            grid_h=end - start,
            grid_w=full_scan.grid_w,
            edge_margin=full_scan.edge_margin,
            probe_radius=full_scan.probe_radius,
            max_indentation=full_scan.max_indentation,
            press_steps=full_scan.press_steps,
            sim_substeps_per_depth=full_scan.sim_substeps_per_depth,
            sim_dt=full_scan.sim_dt,
            vbd_iterations=full_scan.vbd_iterations,
            soft_contact_margin=full_scan.soft_contact_margin,
            preload_gap=full_scan.preload_gap,
            reset_between_points=full_scan.reset_between_points,
            x_values_override=xs,
            y_values_override=ys[start:end],
        )
        print(f"running rows {start}:{end} -> {chunk_path}", flush=True)
        started = time.time()
        simulator = NewtonVBDPalpationSimulator(
            phantom,
            material,
            chunk_scan,
            newton_root=args.newton_root,
            device=args.device,
        )
        sample = simulator.run_sample(lumps)
        sample["chunk_rows"] = np.asarray([start, end], dtype=np.int32)
        sample["elapsed_seconds"] = np.asarray(time.time() - started, dtype=np.float32)
        np.savez_compressed(chunk_path, **sample)
        print(f"wrote {chunk_path} in {time.time() - started:.1f}s", flush=True)


def _assemble(
    args: argparse.Namespace,
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ScanConfig,
    lumps: Sequence[LumpSpec],
    chunk_dir: Path,
) -> None:
    mesh = create_structured_tet_mesh(phantom)
    _k_mu, _k_lambda, _k_damp, tet_lump_mask, tet_lump_id = material_arrays_for_lumps(mesh, material, lumps)

    xs = np.asarray(scan.x_values(phantom), dtype=np.float32)
    ys = np.asarray(scan.y_values(phantom), dtype=np.float32)
    h, w, t = scan.grid_h, scan.grid_w, scan.press_steps
    sample: dict[str, object] = {
        "presses": np.zeros((h, w, t, 2), dtype=np.float32),
        "probe_pose": np.zeros((h, w, t, 7), dtype=np.float32),
        "indentation_depth": np.zeros((h, w, t), dtype=np.float32),
        "fz": np.zeros((h, w, t), dtype=np.float32),
        "contact_features": np.zeros((h, w, t, 5), dtype=np.float32),
        "xy": np.stack(np.meshgrid(xs, ys), axis=-1).astype(np.float32),
        "mask": mask_for_scan_grid(scan, phantom, lumps),
        "tet_lump_mask": np.asarray(tet_lump_mask, dtype=np.uint8),
        "tet_lump_id": np.asarray(tet_lump_id, dtype=np.int32),
        "mesh_vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "mesh_tets": np.asarray(mesh.tets, dtype=np.int32),
        "phantom_json": json.dumps(phantom.to_dict()),
        "material_json": json.dumps(material.to_dict()),
        "scan_json": json.dumps(scan.to_dict()),
        "lump_json": json.dumps(lumps[0].to_dict(phantom)) if lumps else "{}",
        "lumps_json": json.dumps([lump.to_dict(phantom) for lump in lumps]),
        "num_lumps": np.asarray(len(lumps), dtype=np.int32),
        "backend": np.asarray("newton_vbd"),
    }
    filled_rows = np.zeros(h, dtype=bool)
    chunk_elapsed: list[float] = []

    for path in sorted(chunk_dir.glob("rows_*.npz")):
        with np.load(path, allow_pickle=True) as chunk:
            start, end = [int(v) for v in chunk["chunk_rows"]]
            if end <= 0 or start >= h:
                continue
            start_clip = max(start, 0)
            end_clip = min(end, h)
            local_start = start_clip - start
            local_end = local_start + (end_clip - start_clip)
            for key in ("presses", "probe_pose", "indentation_depth", "fz", "contact_features"):
                np.asarray(sample[key])[start_clip:end_clip] = chunk[key][local_start:local_end]
            filled_rows[start_clip:end_clip] = True
            if "elapsed_seconds" in chunk:
                chunk_elapsed.append(float(chunk["elapsed_seconds"]))

    missing = np.flatnonzero(~filled_rows)
    if missing.size:
        raise RuntimeError(f"Cannot assemble final sample: missing row chunks {missing.tolist()}")

    sample["nonlinearity_ratio"] = _nonlinearity_ratio_map(
        np.asarray(sample["indentation_depth"], dtype=np.float32),
        np.asarray(sample["fz"], dtype=np.float32),
    )
    if not args.no_features:
        sample["features"] = extract_feature_map(np.asarray(sample["presses"], dtype=np.float32))

    npz_path = args.out_dir / "center_sphere_newton_sample.npz"
    metadata_path = args.out_dir / "metadata.json"
    np.savez_compressed(npz_path, **sample)
    metadata = build_ground_truth_metadata(
        sample_id="center_sphere_newton_vbd",
        split="single",
        phantom=phantom,
        material=material,
        scan=scan,
        lumps=lumps,
        sample=sample,
        npz_path=npz_path,
        metadata_path=metadata_path,
    )
    metadata["center_sphere_run"] = {
        "requested_backend": "newton_vbd",
        "requested_mesh": [int(args.cells_x), int(args.cells_y), int(args.cells_z)],
        "requested_scan_grid": [int(args.grid_h), int(args.grid_w)],
        "requested_probe_diameter_mm": float(args.probe_diameter_mm),
        "requested_max_indentation_mm": float(args.max_indentation_mm),
        "requested_press_steps": int(args.press_steps),
        "requested_soft_contact_margin_mm": float(args.soft_contact_margin_mm),
        "requested_inclusion": "center sphere",
        "sphere_radius_mm_assumption": float(args.sphere_radius_mm),
        "stiffness_multiplier": float(args.stiffness_multiplier),
        "chunk_elapsed_seconds": chunk_elapsed,
    }
    write_ground_truth_metadata(metadata_path, metadata)
    _write_curve_summary(args.out_dir / "curve_summary.csv", sample)
    _write_summary(args.out_dir / "summary.json", sample, npz_path, metadata_path, chunk_elapsed)
    print(f"assembled {npz_path}", flush=True)


def _phantom(args: argparse.Namespace) -> PhantomConfig:
    sx, sy, sz = DEFAULT_SIZE_M
    min_cell = min(sx / float(args.cells_x), sy / float(args.cells_y), sz / float(args.cells_z))
    return PhantomConfig(
        size_x=sx,
        size_y=sy,
        height=sz,
        cells_x=int(args.cells_x),
        cells_y=int(args.cells_y),
        cells_z=int(args.cells_z),
        particle_radius=0.45 * min_cell,
    )


def _material(args: argparse.Namespace) -> MaterialConfig:
    return MaterialConfig(
        k_mu=float(args.k_mu),
        k_lambda=float(args.k_lambda),
        soft_contact_ke=float(args.soft_contact_ke),
    )


def _scan(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        grid_h=int(args.grid_h),
        grid_w=int(args.grid_w),
        edge_margin=float(args.edge_margin_mm) * MM,
        probe_radius=0.5 * float(args.probe_diameter_mm) * MM,
        max_indentation=float(args.max_indentation_mm) * MM,
        press_steps=int(args.press_steps),
        sim_substeps_per_depth=int(args.substeps),
        vbd_iterations=int(args.vbd_iterations),
        soft_contact_margin=float(args.soft_contact_margin_mm) * MM,
    )


def _center_sphere(phantom: PhantomConfig, args: argparse.Namespace) -> LumpSpec:
    radius = float(args.sphere_radius_mm) * MM
    center = (0.0, 0.0, 0.5 * float(phantom.height))
    if center[2] - radius < 0.0 or center[2] + radius > phantom.height:
        raise ValueError("Center sphere radius does not fit inside the phantom height.")
    return LumpSpec(
        shape="sphere",
        center=center,
        radii=(radius, radius, radius),
        stiffness_multiplier=float(args.stiffness_multiplier),
    )


def _chunk_path(chunk_dir: Path, start: int, end: int) -> Path:
    return chunk_dir / f"rows_{start:03d}_{end:03d}.npz"


def _write_run_config(
    path: Path,
    args: argparse.Namespace,
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ScanConfig,
    lumps: Sequence[LumpSpec],
) -> None:
    data = {
        "schema_version": 1,
        "description": "Centered 100x sphere Newton/VBD run, chunked by scan rows.",
        "args": vars(args),
        "phantom": phantom.to_dict(),
        "material": material.to_dict(),
        "scan": scan.to_dict(),
        "scan_x_values": scan.x_values(phantom),
        "scan_y_values": scan.y_values(phantom),
        "indentation_values": scan.indentation_values(),
        "lumps": [lump.to_dict(phantom) for lump in lumps],
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_curve_summary(path: Path, sample: dict[str, object]) -> None:
    xy = np.asarray(sample["xy"], dtype=np.float32)
    depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
    fz = np.asarray(sample["fz"], dtype=np.float32)
    ratio = np.asarray(sample["nonlinearity_ratio"], dtype=np.float32)
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
                        "late_early_slope_ratio": f"{float(ratio[row, col]):.10g}",
                    }
                )


def _write_summary(
    path: Path,
    sample: dict[str, object],
    npz_path: Path,
    metadata_path: Path,
    chunk_elapsed: Sequence[float],
) -> None:
    fz = np.asarray(sample["fz"], dtype=np.float32)
    depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
    df = np.diff(fz, axis=-1)
    dz = np.diff(depth, axis=-1)
    slopes = np.divide(df, dz, out=np.zeros_like(df), where=np.abs(dz) > 1.0e-12)
    summary = {
        "schema_version": 1,
        "npz": str(npz_path),
        "metadata": str(metadata_path),
        "curve_shape": list(fz.shape),
        "peak_force_median_n": float(np.nanmedian(np.nanmax(fz, axis=-1))),
        "peak_force_max_n": float(np.nanmax(fz)),
        "negative_step_fraction": float(np.mean(df < -1.0e-4)),
        "monotone_curve_fraction": float(np.mean(np.all(df >= -1.0e-4, axis=-1))),
        "median_convex_fraction": float(np.median(np.mean(np.diff(slopes, axis=-1) >= -1.0e-5, axis=-1))),
        "chunk_count": len(chunk_elapsed),
        "chunk_elapsed_seconds": [float(v) for v in chunk_elapsed],
        "total_chunk_elapsed_seconds": float(sum(chunk_elapsed)),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _nonlinearity_ratio_map(depth: np.ndarray, fz: np.ndarray) -> np.ndarray:
    ratios = np.zeros(fz.shape[:2], dtype=np.float32)
    for row in range(fz.shape[0]):
        for col in range(fz.shape[1]):
            early = _segment_slope(depth[row, col], fz[row, col], 0.10, 0.35)
            late = _segment_slope(depth[row, col], fz[row, col], 0.65, 0.90)
            ratios[row, col] = np.float32(late / early) if early > 1.0e-12 else np.float32(0.0)
    return ratios


def _segment_slope(z: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    span = float(np.max(z) - np.min(z))
    if span <= 1.0e-12:
        return 0.0
    keep = (z >= float(np.min(z)) + lo * span) & (z <= float(np.min(z)) + hi * span)
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    zz = z[keep].astype(np.float64)
    ff = f[keep].astype(np.float64)
    zc = zz - float(np.mean(zz))
    denom = float(np.sum(zc * zc))
    if denom <= 1.0e-18:
        return 0.0
    return float(np.sum(zc * (ff - float(np.mean(ff)))) / denom)


if __name__ == "__main__":
    main()
