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
from palpation_sim.curve_plots import draw_sample_curves
from palpation_sim.exports import build_ground_truth_metadata, write_metadata_with_resource_usage, write_visualization_command
from palpation_sim.features import extract_feature_map
from palpation_sim.newton_vbd import NewtonVBDPalpationSimulator
from palpation_sim.phantom import LumpSpec, create_structured_tet_mesh, material_arrays_for_lumps, mask_for_scan_grid
from palpation_sim.workflow import DEFAULT_NEWTON_ROOT, REQUIRED_NEWTON_DEVICE, ResourceMonitor, require_runtime_environment


MM = 1.0e-3
DEFAULT_SOURCE_METADATA = Path(
    "data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/"
    "random_trajectory/data/test/sample_0147_gt.json"
)
DEFAULT_OUT_DIR = Path("runs/real_phantom_sample0147_rigid_ecoflex0010_probe8_depth10_highres")


class ExplicitGridScanConfig(ScanConfig):
    def __init__(self, *args, x_values_override: Sequence[float], y_values_override: Sequence[float], **kwargs) -> None:
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
    parser = argparse.ArgumentParser(
        description="Regenerate a selected real-phantom candidate with high-precision Newton/VBD settings."
    )
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT)
    parser.add_argument("--device", type=str, default=REQUIRED_NEWTON_DEVICE)

    parser.add_argument("--row-chunk-size", type=int, default=1)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-end", type=int, default=None)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--no-assemble", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--no-draw-curves", action="store_true")

    parser.add_argument("--cells-x", type=int, default=96)
    parser.add_argument("--cells-y", type=int, default=96)
    parser.add_argument("--cells-z", type=int, default=32)
    parser.add_argument("--particle-radius-mm", type=float, default=None)
    parser.add_argument("--probe-diameter-mm", type=float, default=8.0)
    parser.add_argument("--max-indentation-mm", type=float, default=10.0)
    parser.add_argument("--press-steps", type=int, default=160)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--vbd-iterations", type=int, default=32)
    parser.add_argument("--soft-contact-margin-mm", type=float, default=1.0)

    parser.add_argument("--normal-k-mu", type=float, default=1.0e4)
    parser.add_argument("--normal-k-lambda", type=float, default=1.0e4)
    parser.add_argument("--normal-k-damp", type=float, default=1.0e-4)
    parser.add_argument("--soft-contact-ke", type=float, default=2.0e6)
    parser.add_argument("--soft-contact-kd", type=float, default=1.0e-7)
    parser.add_argument("--soft-contact-mu", type=float, default=0.5)
    parser.add_argument("--probe-contact-mu", type=float, default=0.8)
    parser.add_argument("--rigid-stiffness-multiplier", type=float, default=10000.0)
    args = parser.parse_args()

    require_runtime_environment(require_newton=True, newton_root=args.newton_root)
    source_metadata = _read_source_metadata(args.source_metadata)
    phantom = _phantom_from_source(source_metadata, args)
    material = _material_from_args(args)
    scan = _scan_from_source(source_metadata, args, phantom)
    lumps = _lumps_from_source(source_metadata, args.rigid_stiffness_multiplier)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = args.out_dir / "chunks"
    logs_dir = args.out_dir / "logs"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = args.out_dir / "run_config.json"
    if not (args.resume and run_config_path.exists()):
        _write_run_config(run_config_path, args, source_metadata, phantom, material, scan, lumps)
    print(f"output dir: {args.out_dir}", flush=True)

    if args.init_only:
        print(f"initialized {args.out_dir}", flush=True)
        return

    monitor = ResourceMonitor(device=args.device).start() if not args.no_assemble else None
    if not args.assemble_only:
        _run_chunks(args, phantom, material, scan, lumps, chunk_dir)
    if not args.no_assemble:
        assert monitor is not None
        _assemble(args, source_metadata, phantom, material, scan, lumps, chunk_dir, monitor)


def _read_source_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"source metadata not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _phantom_from_source(source: dict[str, object], args: argparse.Namespace) -> PhantomConfig:
    src = dict(source["phantom"])  # type: ignore[index,arg-type]
    size_x = float(src["size_x"])
    size_y = float(src["size_y"])
    height = float(src["height"])
    min_cell = min(size_x / float(args.cells_x), size_y / float(args.cells_y), height / float(args.cells_z))
    particle_radius = 0.45 * min_cell if args.particle_radius_mm is None else float(args.particle_radius_mm) * MM
    return PhantomConfig(
        size_x=size_x,
        size_y=size_y,
        height=height,
        cells_x=int(args.cells_x),
        cells_y=int(args.cells_y),
        cells_z=int(args.cells_z),
        density=float(src.get("density", 500.0)),
        particle_radius=float(particle_radius),
    )


def _material_from_args(args: argparse.Namespace) -> MaterialConfig:
    return MaterialConfig(
        k_mu=float(args.normal_k_mu),
        k_lambda=float(args.normal_k_lambda),
        k_damp=float(args.normal_k_damp),
        soft_contact_ke=float(args.soft_contact_ke),
        soft_contact_kd=float(args.soft_contact_kd),
        soft_contact_mu=float(args.soft_contact_mu),
        probe_contact_mu=float(args.probe_contact_mu),
        lump_stiffness_min=float(args.rigid_stiffness_multiplier),
        lump_stiffness_max=float(args.rigid_stiffness_multiplier),
    )


def _scan_from_source(source: dict[str, object], args: argparse.Namespace, phantom: PhantomConfig) -> ExplicitGridScanConfig:
    src = dict(source["scan"])  # type: ignore[index,arg-type]
    x_values = src.get("x_values")
    y_values = src.get("y_values")
    if x_values is None:
        x_values = ScanConfig(grid_w=int(src["grid_w"]), edge_margin=float(src["edge_margin"])).x_values(phantom)
    if y_values is None:
        y_values = ScanConfig(grid_h=int(src["grid_h"]), edge_margin=float(src["edge_margin"])).y_values(phantom)
    return ExplicitGridScanConfig(
        grid_h=int(src["grid_h"]),
        grid_w=int(src["grid_w"]),
        edge_margin=float(src.get("edge_margin", 0.015)),
        probe_radius=0.5 * float(args.probe_diameter_mm) * MM,
        max_indentation=float(args.max_indentation_mm) * MM,
        press_steps=int(args.press_steps),
        sim_substeps_per_depth=int(args.substeps),
        sim_dt=float(src.get("sim_dt", 1.0 / 600.0)),
        vbd_iterations=int(args.vbd_iterations),
        soft_contact_margin=float(args.soft_contact_margin_mm) * MM,
        preload_gap=float(src.get("preload_gap", 0.0005)),
        reset_between_points=bool(src.get("reset_between_points", True)),
        x_values_override=[float(v) for v in x_values],
        y_values_override=[float(v) for v in y_values],
    )


def _lumps_from_source(source: dict[str, object], multiplier: float) -> list[LumpSpec]:
    lumps = []
    for record in source.get("lumps", []):  # type: ignore[union-attr]
        item = dict(record)
        lumps.append(
            LumpSpec(
                shape=item["shape"],
                center=tuple(float(v) for v in item["center"]),
                radii=tuple(float(v) for v in item["radii"]),
                stiffness_multiplier=float(multiplier),
                yaw=float(item.get("yaw", 0.0)),
            )
        )
    if not lumps:
        raise ValueError("source metadata contains no lumps")
    return lumps


def _run_chunks(
    args: argparse.Namespace,
    phantom: PhantomConfig,
    material: MaterialConfig,
    full_scan: ExplicitGridScanConfig,
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

        chunk_scan = ExplicitGridScanConfig(
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
    source_metadata: dict[str, object],
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ExplicitGridScanConfig,
    lumps: Sequence[LumpSpec],
    chunk_dir: Path,
    monitor: ResourceMonitor,
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
        "probe_force": np.zeros((h, w, t, 3), dtype=np.float32),
        "probe_torque": np.zeros((h, w, t, 3), dtype=np.float32),
        "probe_wrench": np.zeros((h, w, t, 6), dtype=np.float32),
        "contact_features": np.zeros((h, w, t, 5), dtype=np.float32),
        "trajectory_xy_offset": np.zeros((h, w, t, 2), dtype=np.float32),
        "trajectory_json": np.asarray(json.dumps({"mode": "straight", "source": "real_phantom_test_grid"})),
        "xy": np.stack(np.meshgrid(xs, ys), axis=-1).astype(np.float32),
        "mask": mask_for_scan_grid(scan, phantom, lumps),
        "tet_lump_mask": np.asarray(tet_lump_mask, dtype=np.uint8),
        "tet_lump_id": np.asarray(tet_lump_id, dtype=np.int32),
        "mesh_vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "mesh_tets": np.asarray(mesh.tets, dtype=np.int32),
        "phantom_json": json.dumps(phantom.to_dict()),
        "material_json": json.dumps(material.to_dict()),
        "scan_json": json.dumps(scan.to_dict()),
        "lump_json": json.dumps(lumps[0].to_dict(phantom)),
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
            for key in (
                "presses",
                "probe_pose",
                "indentation_depth",
                "fz",
                "probe_force",
                "probe_torque",
                "probe_wrench",
                "contact_features",
                "trajectory_xy_offset",
            ):
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

    npz_path = args.out_dir / "real_phantom_sample0147_rigid_ecoflex0010_probe8_depth10_highres.npz"
    metadata_path = args.out_dir / "metadata.json"
    curve_plot_path = None if args.no_draw_curves else args.out_dir / "fz_curves.png"
    np.savez_compressed(npz_path, **sample)
    visualization_command_path = write_visualization_command(npz_path, project_root=PROJECT_ROOT)
    if curve_plot_path is not None:
        draw_sample_curves(sample, curve_plot_path, title=args.out_dir.name)
    metadata = build_ground_truth_metadata(
        sample_id="real_phantom_sample0147_rigid_ecoflex0010_probe8_depth10_highres",
        split="single",
        phantom=phantom,
        material=material,
        scan=scan,
        lumps=lumps,
        sample=sample,
        npz_path=npz_path,
        metadata_path=metadata_path,
    )
    metadata["files"]["curve_plot"] = curve_plot_path.name if curve_plot_path is not None else None
    metadata["files"]["visualization_command"] = visualization_command_path.name
    metadata["source_real_phantom_candidate"] = {
        "source_metadata": str(args.source_metadata),
        "source_sample_id": source_metadata.get("sample_id"),
        "base_phantom_id": source_metadata.get("base_phantom_id"),
        "original_probe_radius_m": dict(source_metadata["scan"]).get("probe_radius"),  # type: ignore[index,arg-type]
        "original_max_indentation_m": dict(source_metadata["scan"]).get("max_indentation"),  # type: ignore[index,arg-type]
    }
    metadata["high_precision_run"] = _run_details(args, scan, chunk_elapsed)
    _write_curve_summary(args.out_dir / "curve_summary.csv", sample)
    resource_usage = monitor.finish(storage_root=args.out_dir)
    _write_summary(args.out_dir / "summary.json", sample, npz_path, metadata_path, curve_plot_path, chunk_elapsed, resource_usage)
    write_metadata_with_resource_usage(metadata_path, metadata, resource_usage, storage_root=args.out_dir)
    print(f"assembled {npz_path}", flush=True)


def _chunk_path(chunk_dir: Path, start: int, end: int) -> Path:
    return chunk_dir / f"rows_{start:03d}_{end:03d}.npz"


def _write_run_config(
    path: Path,
    args: argparse.Namespace,
    source_metadata: dict[str, object],
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ExplicitGridScanConfig,
    lumps: Sequence[LumpSpec],
) -> None:
    args_data = vars(args).copy()
    args_data["init_only"] = False
    data = {
        "schema_version": 1,
        "description": (
            "High-precision Newton/VBD regeneration of real-phantom candidate sample_0147: "
            "same 20x20 real-phantom grid/geometry, 8 mm probe, 10 mm indentation, "
            "Ecoflex 00-10 approximate normal tissue, near-rigid inclusions."
        ),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in args_data.items()},
        "source_real_phantom_candidate": {
            "source_metadata": str(args.source_metadata),
            "source_sample_id": source_metadata.get("sample_id"),
            "base_phantom_id": source_metadata.get("base_phantom_id"),
        },
        "material_interpretation": {
            "normal_tissue": "Ecoflex 00-10 approximate low-kPa stiffness scale in this Newton/VBD parameterization.",
            "near_rigid_inclusions": "Implemented as a large Neo-Hookean tet stiffness multiplier, not as analytic rigid bodies.",
            "effective_lump_k_mu": float(material.k_mu * lumps[0].stiffness_multiplier),
            "effective_lump_k_lambda": float(material.k_lambda * lumps[0].stiffness_multiplier),
        },
        "force_estimator": {
            "name": "bottom_support_reaction",
            "description": (
                "Probe Fz is estimated from the internal elastic support reaction at the fixed bottom boundary "
                "after each Newton/VBD solve, rather than from post-solve contact penetration residuals."
            ),
        },
        "phantom": phantom.to_dict(),
        "material": material.to_dict(),
        "scan": scan.to_dict(),
        "scan_x_values": scan.x_values(phantom),
        "scan_y_values": scan.y_values(phantom),
        "indentation_values": scan.indentation_values(),
        "lumps": [lump.to_dict(phantom) for lump in lumps],
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _run_details(args: argparse.Namespace, scan: ExplicitGridScanConfig, chunk_elapsed: Sequence[float]) -> dict[str, object]:
    return {
        "requested_probe_diameter_mm": float(args.probe_diameter_mm),
        "requested_max_indentation_mm": float(args.max_indentation_mm),
        "requested_press_steps": int(args.press_steps),
        "requested_substeps": int(args.substeps),
        "requested_vbd_iterations": int(args.vbd_iterations),
        "requested_soft_contact_margin_mm": float(args.soft_contact_margin_mm),
        "requested_scan_grid": [int(scan.grid_h), int(scan.grid_w)],
        "requested_mesh": [int(args.cells_x), int(args.cells_y), int(args.cells_z)],
        "normal_k_mu": float(args.normal_k_mu),
        "normal_k_lambda": float(args.normal_k_lambda),
        "rigid_stiffness_multiplier": float(args.rigid_stiffness_multiplier),
        "force_estimator": "bottom_support_reaction",
        "chunk_elapsed_seconds": [float(v) for v in chunk_elapsed],
    }


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
    curve_plot_path: Path | None,
    chunk_elapsed: Sequence[float],
    resource_usage: dict[str, object],
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
        "curve_plot": str(curve_plot_path) if curve_plot_path is not None else None,
        "curve_shape": list(fz.shape),
        "force_estimator": "bottom_support_reaction",
        "peak_force_median_n": float(np.nanmedian(np.nanmax(fz, axis=-1))),
        "peak_force_max_n": float(np.nanmax(fz)),
        "negative_step_fraction": float(np.mean(df < -1.0e-4)),
        "monotone_curve_fraction": float(np.mean(np.all(df >= -1.0e-4, axis=-1))),
        "median_convex_fraction": float(np.median(np.mean(np.diff(slopes, axis=-1) >= -1.0e-5, axis=-1))),
        "chunk_count": len(chunk_elapsed),
        "chunk_elapsed_seconds": [float(v) for v in chunk_elapsed],
        "total_chunk_elapsed_seconds": float(sum(chunk_elapsed)),
        "resource_usage": resource_usage,
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
