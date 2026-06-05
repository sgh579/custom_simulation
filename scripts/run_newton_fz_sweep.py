from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.config import MaterialConfig, PhantomConfig, ScanConfig
from palpation_sim.newton_vbd import NewtonVBDPalpationSimulator
from palpation_sim.phantom import LumpSpec
from palpation_sim.exports import write_metadata_with_resource_usage, write_visualization_command
from palpation_sim.workflow import (
    DEFAULT_NEWTON_ROOT,
    REQUIRED_NEWTON_DEVICE,
    ResourceMonitor,
    metadata_contract,
    require_runtime_environment,
    run_output_metadata,
    runtime_metadata,
    with_run_date_prefix,
)


MM = 1.0e-3
PHANTOM_SIZE_M = (80.0 * MM, 80.0 * MM, 25.0 * MM)
CYLINDER_RADIUS_M = 10.0 * MM
CYLINDER_HALF_HEIGHT_M = 2.5 * MM
FIXED_LUMP_CENTERS_M = (
    (-20.0 * MM, 20.0 * MM, 6.5 * MM),
    (20.0 * MM, 20.0 * MM, 10.5 * MM),
    (-20.0 * MM, -20.0 * MM, 14.5 * MM),
    (20.0 * MM, -20.0 * MM, 18.5 * MM),
)

TARGET_POINTS_MM = {
    "center_gap": (0.0, 0.0),
    "top_lump": (20.0, -20.0),
    "mid_lump": (20.0, 20.0),
    "bottom_lump": (-20.0, 20.0),
}


class PointScanConfig(ScanConfig):
    def __init__(self, *args, point_x: float, point_y: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.point_x = float(point_x)
        self.point_y = float(point_y)

    def x_values(self, phantom: PhantomConfig) -> list[float]:
        del phantom
        return [self.point_x]

    def y_values(self, phantom: PhantomConfig) -> list[float]:
        del phantom
        return [self.point_y]


@dataclass(frozen=True)
class SweepCase:
    label: str
    probe_diameter_mm: float = 12.0
    k_mu: float = 2.0e5
    k_lambda: float = 2.0e5
    soft_contact_ke: float = 2.0e6
    press_steps: int | None = None
    substeps: int | None = None
    iterations: int | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused Newton/VBD F-z sweeps on the fixed-cylinder phantom.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/newton_fz_sweeps/fixed_four"))
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT, help="Pinned Newton source root.")
    parser.add_argument("--device", type=str, default=REQUIRED_NEWTON_DEVICE, help="Pinned Warp/Newton CUDA device.")
    parser.add_argument("--target", choices=[*TARGET_POINTS_MM.keys(), "all"], default="top_lump")
    parser.add_argument("--case-preset", choices=["quick", "probe", "stiffness", "convergence"], default="quick")

    parser.add_argument("--cells-xy", type=int, default=24)
    parser.add_argument("--cells-z", type=int, default=8)
    parser.add_argument("--press-steps", type=int, default=24)
    parser.add_argument("--max-indentation-mm", type=float, default=16.0)
    parser.add_argument("--soft-contact-margin-mm", type=float, default=1.0)
    parser.add_argument("--substeps", type=int, default=6)
    parser.add_argument("--vbd-iterations", type=int, default=10)
    parser.add_argument("--lump-stiffness-multiplier", type=float, default=100.0)
    parser.add_argument("--no-save-samples", action="store_true")
    args = parser.parse_args()
    require_runtime_environment(require_newton=True, newton_root=args.newton_root)
    args.out_dir = with_run_date_prefix(args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {args.out_dir}", flush=True)
    cases = _preset_cases(args.case_preset)
    targets = TARGET_POINTS_MM.items() if args.target == "all" else [(args.target, TARGET_POINTS_MM[args.target])]

    summaries: list[dict[str, object]] = []
    run_monitor = ResourceMonitor(device=args.device).start()
    for target_name, point_mm in targets:
        for case in cases:
            print(f"running {target_name}/{case.label}...", flush=True)
            summaries.append(_run_case(args, target_name, point_mm, case))

    _write_summary(args.out_dir / "summary.csv", summaries)
    _write_json(args.out_dir / "summary.json", {"cases": summaries})
    _plot_curves(args.out_dir / "fz_curves.png", summaries)
    run_resource_usage = run_monitor.finish(storage_root=args.out_dir)
    metadata = {
        "schema_name": "newton_fz_sweep_metadata",
        "schema_version": 2,
        "data_contract": metadata_contract(),
        "runtime": runtime_metadata(newton_root=args.newton_root, device=args.device),
        "run": run_output_metadata(args.out_dir),
        "backend": "newton_vbd",
        "phantom_design": "fixed_four_cylinder",
        "files": {
            "summary_json": "summary.json",
            "summary_csv": "summary.csv",
            "plot": "fz_curves.png",
            "sample_npz_pattern": "{target}/{case_label}.npz",
            "sample_visualization_command_pattern": "{target}/{case_label}_visualization_command.md",
        },
        "cases": summaries,
    }
    write_metadata_with_resource_usage(args.out_dir / "metadata.json", metadata, run_resource_usage, storage_root=args.out_dir)
    print(f"done: wrote {len(summaries)} cases to {args.out_dir}", flush=True)


def _run_case(
    args: argparse.Namespace,
    target_name: str,
    point_mm: tuple[float, float],
    case: SweepCase,
) -> dict[str, object]:
    phantom = _phantom(args.cells_xy, args.cells_xy, args.cells_z)
    scan = PointScanConfig(
        grid_h=1,
        grid_w=1,
        edge_margin=4.0 * MM,
        probe_radius=0.5 * float(case.probe_diameter_mm) * MM,
        max_indentation=float(args.max_indentation_mm) * MM,
        press_steps=int(case.press_steps or args.press_steps),
        sim_substeps_per_depth=int(case.substeps or args.substeps),
        sim_dt=1.0 / 600.0,
        vbd_iterations=int(case.iterations or args.vbd_iterations),
        soft_contact_margin=float(args.soft_contact_margin_mm) * MM,
        preload_gap=0.0005,
        reset_between_points=True,
        point_x=float(point_mm[0]) * MM,
        point_y=float(point_mm[1]) * MM,
    )
    material = MaterialConfig(
        k_mu=float(case.k_mu),
        k_lambda=float(case.k_lambda),
        soft_contact_ke=float(case.soft_contact_ke),
    )
    simulator = NewtonVBDPalpationSimulator(
        phantom,
        material,
        scan,
        newton_root=args.newton_root,
        device=args.device,
    )

    monitor = ResourceMonitor(device=args.device).start()
    sample = simulator.run_sample(_fixed_lumps(float(args.lump_stiffness_multiplier)))
    depth = np.asarray(sample["indentation_depth"][0, 0], dtype=np.float32)
    force = np.asarray(sample["fz"][0, 0], dtype=np.float32)
    metrics = _curve_metrics(depth, force)

    sample_path: Path | None = None
    visualization_command_path: Path | None = None
    if not args.no_save_samples:
        sample_dir = args.out_dir / target_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / f"{case.label}.npz"
        np.savez_compressed(sample_path, **sample)
        visualization_command_path = write_visualization_command(sample_path, project_root=PROJECT_ROOT)
    storage_root = (sample_path, visualization_command_path) if sample_path is not None else args.out_dir
    resource_usage = monitor.finish(storage_root=storage_root)

    return {
        "target": target_name,
        "point_mm": [float(point_mm[0]), float(point_mm[1])],
        "label": case.label,
        "elapsed_seconds": float(resource_usage["elapsed_seconds"]),
        "resource_usage": resource_usage,
        "sample": str(sample_path) if sample_path is not None else None,
        "visualization_command": str(visualization_command_path) if visualization_command_path is not None else None,
        "cells": [int(phantom.cells_x), int(phantom.cells_y), int(phantom.cells_z)],
        "particle_radius_mm": float(phantom.particle_radius * 1000.0),
        "probe_diameter_mm": float(case.probe_diameter_mm),
        "k_mu": float(case.k_mu),
        "k_lambda": float(case.k_lambda),
        "soft_contact_ke": float(case.soft_contact_ke),
        "soft_contact_margin_mm": float(args.soft_contact_margin_mm),
        "press_steps": int(scan.press_steps),
        "substeps": int(scan.sim_substeps_per_depth),
        "vbd_iterations": int(scan.vbd_iterations),
        "peak_force_n": metrics["peak_force_n"],
        "final_force_n": metrics["final_force_n"],
        "late_early_slope_ratio": metrics["late_early_slope_ratio"],
        "monotone_fraction": metrics["monotone_fraction"],
        "convex_fraction": metrics["convex_fraction"],
        "depth_mm": (depth * 1000.0).round(6).tolist(),
        "force_n": force.round(6).tolist(),
    }


def _preset_cases(name: str) -> list[SweepCase]:
    if name == "probe":
        return [
            SweepCase("probe8_k2e5", probe_diameter_mm=8.0),
            SweepCase("probe12_k2e5", probe_diameter_mm=12.0),
            SweepCase("probe16_k2e5", probe_diameter_mm=16.0),
        ]
    if name == "stiffness":
        return [
            SweepCase("probe12_k2e5", probe_diameter_mm=12.0, k_mu=2.0e5, k_lambda=2.0e5),
            SweepCase("probe12_k5e5", probe_diameter_mm=12.0, k_mu=5.0e5, k_lambda=5.0e5),
            SweepCase("probe12_k1e6", probe_diameter_mm=12.0, k_mu=1.0e6, k_lambda=1.0e6),
        ]
    if name == "convergence":
        return [
            SweepCase("probe12_base", probe_diameter_mm=12.0),
            SweepCase("probe12_converged", probe_diameter_mm=12.0, press_steps=32, substeps=8, iterations=16),
        ]
    return [
        SweepCase("probe8_k2e5", probe_diameter_mm=8.0),
        SweepCase("probe12_k2e5", probe_diameter_mm=12.0),
        SweepCase("probe12_k5e5", probe_diameter_mm=12.0, k_mu=5.0e5, k_lambda=5.0e5),
    ]


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


def _fixed_lumps(stiffness_multiplier: float) -> list[LumpSpec]:
    return [
        LumpSpec(
            shape="cylinder",
            center=tuple(float(v) for v in center),
            radii=(CYLINDER_RADIUS_M, CYLINDER_RADIUS_M, CYLINDER_HALF_HEIGHT_M),
            stiffness_multiplier=float(stiffness_multiplier),
            yaw=0.0,
        )
        for center in FIXED_LUMP_CENTERS_M
    ]


def _curve_metrics(depth: np.ndarray, force: np.ndarray) -> dict[str, float]:
    slopes = np.diff(force) / np.maximum(np.diff(depth), np.float32(1.0e-12))
    early = _segment_slope(depth, force, 0.10, 0.35)
    late = _segment_slope(depth, force, 0.65, 0.90)
    return {
        "peak_force_n": float(np.max(force)),
        "final_force_n": float(force[-1]),
        "late_early_slope_ratio": float(late / early) if early > 1.0e-12 else 0.0,
        "monotone_fraction": float(np.mean(np.diff(force) >= -1.0e-4)) if force.size > 1 else 1.0,
        "convex_fraction": float(np.mean(np.diff(slopes) >= -1.0e-5)) if slopes.size > 1 else 1.0,
    }


def _segment_slope(depth: np.ndarray, force: np.ndarray, lo: float, hi: float) -> float:
    span = float(np.max(depth) - np.min(depth))
    if span <= 1.0e-12:
        return 0.0
    keep = (depth >= np.min(depth) + lo * span) & (depth <= np.min(depth) + hi * span)
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    z = depth[keep].astype(np.float64)
    f = force[keep].astype(np.float64)
    zc = z - float(np.mean(z))
    denom = float(np.sum(zc * zc))
    if denom <= 1.0e-18:
        return 0.0
    return float(np.sum(zc * (f - float(np.mean(f)))) / denom)


def _write_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "target",
        "label",
        "elapsed_seconds",
        "sample",
        "visualization_command",
        "cells",
        "particle_radius_mm",
        "point_mm",
        "probe_diameter_mm",
        "k_mu",
        "k_lambda",
        "soft_contact_ke",
        "soft_contact_margin_mm",
        "press_steps",
        "substeps",
        "vbd_iterations",
        "peak_force_n",
        "final_force_n",
        "late_early_slope_ratio",
        "monotone_fraction",
        "convex_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return value


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _plot_curves(path: Path, rows: Sequence[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for row in rows:
        depth = np.asarray(row["depth_mm"], dtype=np.float32)
        force = np.asarray(row["force_n"], dtype=np.float32)
        label = f"{row['target']} / {row['label']}"
        ax.plot(depth, force, linewidth=2.0, label=label)
    ax.set_xlabel("indentation [mm]")
    ax.set_ylabel("Fz [N]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
