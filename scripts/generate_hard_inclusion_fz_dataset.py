#!/usr/bin/env python3
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
from palpation_sim.exports import build_ground_truth_metadata, write_metadata_with_resource_usage
from palpation_sim.features import extract_feature_map
from palpation_sim.newton_vbd import NewtonVBDPalpationSimulator
from palpation_sim.phantom import LumpSpec, create_structured_tet_mesh, material_arrays_for_lumps
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
SCAN_X_MM = (-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0)


class ExplicitGridScanConfig(ScanConfig):
    def __init__(
        self,
        *args,
        x_values_override: Sequence[float],
        y_values_override: Sequence[float],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._x_values_override = [float(v) for v in x_values_override]
        self._y_values_override = [float(v) for v in y_values_override]

    def x_values(self, phantom: PhantomConfig) -> list[float]:
        del phantom
        return list(self._x_values_override)

    def y_values(self, phantom: PhantomConfig) -> list[float]:
        del phantom
        return list(self._y_values_override)


@dataclass(frozen=True)
class InclusionCase:
    label: str
    shape: str | None
    center_mm: tuple[float, float, float] | None
    radii_mm: tuple[float, float, float] | None
    stiffness_multiplier: float
    yaw: float = 0.0

    def lumps(self) -> list[LumpSpec]:
        if self.shape is None or self.center_mm is None or self.radii_mm is None:
            return []
        return [
            LumpSpec(
                shape=self.shape,  # type: ignore[arg-type]
                center=tuple(float(v) * MM for v in self.center_mm),
                radii=tuple(float(v) * MM for v in self.radii_mm),
                stiffness_multiplier=float(self.stiffness_multiplier),
                yaw=float(self.yaw),
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Newton/VBD F-z curves for hard-inclusion contrast checks.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/hard_inclusion_fz_dataset"))
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT, help="Pinned Newton source root.")
    parser.add_argument("--device", type=str, default=REQUIRED_NEWTON_DEVICE, help="Pinned Warp/Newton CUDA device.")
    parser.add_argument("--cells-xy", type=int, default=20)
    parser.add_argument("--cells-z", type=int, default=8)
    parser.add_argument("--press-steps", type=int, default=18)
    parser.add_argument("--max-indentation-mm", type=float, default=10.0)
    parser.add_argument("--probe-diameter-mm", type=float, default=8.0)
    parser.add_argument("--substeps", type=int, default=3)
    parser.add_argument("--vbd-iterations", type=int, default=6)
    parser.add_argument("--k-mu", type=float, default=1.0e5)
    parser.add_argument("--k-lambda", type=float, default=1.0e5)
    parser.add_argument("--soft-contact-ke", type=float, default=2.0e6)
    parser.add_argument("--edge-margin-mm", type=float, default=4.0)
    parser.add_argument("--scan-mode", choices=["line", "plane"], default="line")
    parser.add_argument("--scan-step-mm", type=float, default=1.0, help="XY spacing for --scan-mode plane.")
    args = parser.parse_args()
    require_runtime_environment(require_newton=True, newton_root=args.newton_root)
    args.out_dir = with_run_date_prefix(args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {args.out_dir}", flush=True)
    phantom = _phantom(args)
    material = _material(args)
    scan = _scan(args)
    cases = _cases()

    all_rows: list[dict[str, object]] = []
    case_resource_usage: list[dict[str, object]] = []
    dataset_monitor = ResourceMonitor(device=args.device).start()
    for case in cases:
        print(f"running {case.label}...", flush=True)
        rows, resource_usage = _run_case(args, phantom, material, scan, case)
        all_rows.extend(rows)
        case_resource_usage.append({"case": case.label, "resource_usage": resource_usage})

    _write_csv(args.out_dir / "all_curve_summary.csv", all_rows)
    _plot_all_curves(args.out_dir / "all_fz_curves.png", args.out_dir, cases)
    _plot_slope_vs_distance(args.out_dir / "slope_vs_distance.png", all_rows)
    dataset_resource_usage = dataset_monitor.finish(storage_root=args.out_dir)
    _write_dataset_metadata(
        args.out_dir / "metadata.json",
        args,
        cases,
        all_rows,
        case_resource_usage=case_resource_usage,
        resource_usage=dataset_resource_usage,
    )
    print(f"done: dataset in {args.out_dir}", flush=True)


def _cases() -> list[InclusionCase]:
    return [
        InclusionCase("control_no_inclusion", None, None, None, 1.0),
        InclusionCase("shallow_sphere_30x", "sphere", (0.0, 0.0, 17.5), (6.0, 6.0, 6.0), 30.0),
        InclusionCase("shallow_sphere_120x", "sphere", (0.0, 0.0, 17.5), (6.0, 6.0, 6.0), 120.0),
        InclusionCase("deep_sphere_120x", "sphere", (0.0, 0.0, 11.5), (6.0, 6.0, 6.0), 120.0),
        InclusionCase("shallow_large_sphere_120x", "sphere", (0.0, 0.0, 16.5), (8.0, 8.0, 8.0), 120.0),
        InclusionCase("shallow_ellipsoid_120x", "ellipsoid", (0.0, 0.0, 19.5), (10.0, 5.0, 4.0), 120.0),
        InclusionCase("shallow_cylinder_120x", "cylinder", (0.0, 0.0, 20.5), (7.0, 7.0, 2.5), 120.0),
    ]


def _run_case(
    args: argparse.Namespace,
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ExplicitGridScanConfig,
    case: InclusionCase,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    case_dir = args.out_dir / case.label
    case_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(device=args.device).start()
    lumps = case.lumps()
    mesh = create_structured_tet_mesh(phantom)
    _k_mu, _k_lambda, _k_damp, tet_lump_mask, tet_lump_id = material_arrays_for_lumps(mesh, material, lumps)

    simulator = NewtonVBDPalpationSimulator(
        phantom,
        material,
        scan,
        newton_root=args.newton_root,
        device=args.device,
    )
    sample = simulator.run_sample(lumps)
    sample["features"] = extract_feature_map(np.asarray(sample["presses"], dtype=np.float32))
    sample["mesh_vertices"] = np.asarray(mesh.vertices, dtype=np.float32)
    sample["mesh_tets"] = np.asarray(mesh.tets, dtype=np.int32)
    sample["tet_lump_mask"] = np.asarray(tet_lump_mask, dtype=np.uint8)
    sample["tet_lump_id"] = np.asarray(tet_lump_id, dtype=np.int32)
    sample["phantom_json"] = json.dumps(phantom.to_dict())
    sample["material_json"] = json.dumps(material.to_dict())
    sample["scan_json"] = json.dumps(scan.to_dict())

    npz_path = case_dir / "sample.npz"
    metadata_path = case_dir / "metadata.json"
    np.savez_compressed(npz_path, **sample)

    rows = _curve_rows(case, sample, elapsed=monitor.elapsed_seconds())
    _write_csv(case_dir / "curve_summary.csv", rows)
    _plot_case_curves(case_dir / "fz_curves.png", case, sample)
    resource_usage = monitor.finish(storage_root=case_dir)
    rows = _curve_rows(case, sample, elapsed=float(resource_usage["elapsed_seconds"]))
    _write_csv(case_dir / "curve_summary.csv", rows)

    metadata = build_ground_truth_metadata(
        sample_id=case.label,
        split="hard_inclusion_fz",
        phantom=phantom,
        material=material,
        scan=scan,
        lumps=lumps,
        sample=sample,
        npz_path=npz_path,
        metadata_path=metadata_path,
    )
    metadata["case"] = _case_metadata(case)
    metadata["mesh"] = {
        "vertices": int(mesh.vertices.shape[0]),
        "tets": int(mesh.tets.shape[0]),
        "assigned_lump_tets": int(np.count_nonzero(tet_lump_mask)),
    }
    write_metadata_with_resource_usage(metadata_path, metadata, resource_usage, storage_root=case_dir)
    print(f"  wrote {npz_path} ({rows[0]['elapsed_seconds']:.1f}s)", flush=True)
    return rows, resource_usage


def _curve_rows(case: InclusionCase, sample: dict[str, object], *, elapsed: float) -> list[dict[str, object]]:
    xy = np.asarray(sample["xy"], dtype=np.float32)
    depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
    fz = np.asarray(sample["fz"], dtype=np.float32)
    ratio = np.asarray(sample["nonlinearity_ratio"], dtype=np.float32)
    rows: list[dict[str, object]] = []
    for row in range(fz.shape[0]):
        for col in range(fz.shape[1]):
            z = depth[row, col]
            force = fz[row, col]
            early = _segment_slope(z, force, 0.10, 0.35)
            late = _segment_slope(z, force, 0.65, 0.95)
            rows.append(
                {
                    "case": case.label,
                    "shape": case.shape or "none",
                    "stiffness_multiplier": float(case.stiffness_multiplier),
                    "center_depth_mm": _center_depth_mm(case),
                    "top_depth_mm": _top_depth_mm(case),
                    "row": row,
                    "col": col,
                    "x_mm": float(xy[row, col, 0] / MM),
                    "y_mm": float(xy[row, col, 1] / MM),
                    "xy_distance_to_inclusion_mm": _xy_distance_mm(case, xy[row, col]),
                    "final_force_n": float(force[-1]),
                    "peak_force_n": float(np.max(force)),
                    "early_slope_n_per_m": float(early),
                    "late_slope_n_per_m": float(late),
                    "late_early_slope_ratio": float(ratio[row, col]),
                    "endpoint_stiffness_n_per_m": _endpoint_stiffness(z, force),
                    "elapsed_seconds": float(elapsed),
                }
            )
    return rows


def _phantom(args: argparse.Namespace) -> PhantomConfig:
    sx, sy, sz = PHANTOM_SIZE_M
    min_cell = min(sx / float(args.cells_xy), sy / float(args.cells_xy), sz / float(args.cells_z))
    return PhantomConfig(
        size_x=sx,
        size_y=sy,
        height=sz,
        cells_x=int(args.cells_xy),
        cells_y=int(args.cells_xy),
        cells_z=int(args.cells_z),
        particle_radius=0.45 * min_cell,
    )


def _material(args: argparse.Namespace) -> MaterialConfig:
    return MaterialConfig(
        k_mu=float(args.k_mu),
        k_lambda=float(args.k_lambda),
        soft_contact_ke=float(args.soft_contact_ke),
    )


def _scan(args: argparse.Namespace) -> ExplicitGridScanConfig:
    edge_margin = float(args.edge_margin_mm) * MM
    if args.scan_mode == "plane":
        x_values = _axis_values_for_step(
            side_m=PHANTOM_SIZE_M[0],
            edge_margin_m=edge_margin,
            step_mm=float(args.scan_step_mm),
        )
        y_values = _axis_values_for_step(
            side_m=PHANTOM_SIZE_M[1],
            edge_margin_m=edge_margin,
            step_mm=float(args.scan_step_mm),
        )
    else:
        x_values = [v * MM for v in SCAN_X_MM]
        y_values = [0.0]
    return ExplicitGridScanConfig(
        grid_h=len(y_values),
        grid_w=len(x_values),
        edge_margin=edge_margin,
        probe_radius=0.5 * float(args.probe_diameter_mm) * MM,
        max_indentation=float(args.max_indentation_mm) * MM,
        press_steps=int(args.press_steps),
        sim_substeps_per_depth=int(args.substeps),
        vbd_iterations=int(args.vbd_iterations),
        soft_contact_margin=1.0 * MM,
        preload_gap=0.5 * MM,
        reset_between_points=True,
        x_values_override=x_values,
        y_values_override=y_values,
    )


def _axis_values_for_step(*, side_m: float, edge_margin_m: float, step_mm: float) -> list[float]:
    if step_mm <= 0.0:
        raise ValueError("--scan-step-mm must be positive.")
    side_mm = float(side_m / MM)
    margin_mm = float(edge_margin_m / MM)
    lo = -0.5 * side_mm + margin_mm
    hi = 0.5 * side_mm - margin_mm
    step = float(step_mm)
    count = int(np.floor((hi - lo) / step + 1.0e-9)) + 1
    values = (lo + step * np.arange(max(count, 1), dtype=np.float64)).tolist()
    if values[-1] < hi - 1.0e-6:
        values.append(hi)
    return [float(v * MM) for v in values]


def _write_dataset_metadata(
    path: Path,
    args: argparse.Namespace,
    cases: Sequence[InclusionCase],
    rows: Sequence[dict[str, object]],
    *,
    case_resource_usage: Sequence[dict[str, object]],
    resource_usage: dict[str, object],
) -> None:
    metadata = {
        "schema_name": "hard_inclusion_fz_dataset_metadata",
        "schema_version": 2,
        "data_contract": metadata_contract(),
        "runtime": runtime_metadata(newton_root=args.newton_root, device=args.device),
        "run": run_output_metadata(args.out_dir),
        "backend": "newton_vbd",
        "description": "Line-scan F-z curves designed to show larger slope near hard inclusions and variation with inclusion properties.",
        "files": {
            "metadata": "metadata.json",
            "all_curve_summary": "all_curve_summary.csv",
            "all_fz_curves": "all_fz_curves.png",
            "slope_vs_distance": "slope_vs_distance.png",
            "case_sample_pattern": "{case}/sample.npz",
            "case_metadata_pattern": "{case}/metadata.json",
            "case_curve_summary_pattern": "{case}/curve_summary.csv",
            "case_curve_plot_pattern": "{case}/fz_curves.png",
        },
        "scan_mode": str(args.scan_mode),
        "scan_step_mm": float(args.scan_step_mm) if args.scan_mode == "plane" else None,
        "scan_x_mm": [float(v / MM) for v in _scan(args).x_values(_phantom(args))],
        "scan_y_mm": [float(v / MM) for v in _scan(args).y_values(_phantom(args))],
        "cases": [_case_metadata(case) for case in cases],
        "case_resource_usage": list(case_resource_usage),
        "num_cases": len(cases),
        "num_curves": len(rows),
        "elapsed_seconds": float(resource_usage["elapsed_seconds"]),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_metadata_with_resource_usage(path, metadata, resource_usage, storage_root=args.out_dir)


def _case_metadata(case: InclusionCase) -> dict[str, object]:
    return {
        "label": case.label,
        "shape": case.shape,
        "center_mm": list(case.center_mm) if case.center_mm is not None else None,
        "radii_mm": list(case.radii_mm) if case.radii_mm is not None else None,
        "stiffness_multiplier": float(case.stiffness_multiplier),
        "center_depth_from_top_mm": _center_depth_mm(case),
        "top_depth_from_top_mm": _top_depth_mm(case),
        "yaw": float(case.yaw),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_case_curves(path: Path, case: InclusionCase, sample: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    depth_all = np.asarray(sample["indentation_depth"], dtype=np.float32)
    fz_all = np.asarray(sample["fz"], dtype=np.float32)
    xy_all = np.asarray(sample["xy"], dtype=np.float32)
    row_idx = _center_row_index(xy_all)
    col_indices = _representative_columns(fz_all.shape[1])
    depth = depth_all[row_idx]
    fz = fz_all[row_idx]
    xy = xy_all[row_idx]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for col in col_indices:
        x_mm = float(xy[col, 0] / MM)
        dist = _xy_distance_mm(case, xy[col])
        suffix = "control" if dist is None else f"d={dist:.0f}mm"
        ax.plot(depth[col] / MM, fz[col], linewidth=1.8, label=f"x={x_mm:.0f}mm {suffix}")
    y_mm = float(xy[0, 1] / MM)
    ax.set_title(f"{case.label} center row y={y_mm:.0f}mm")
    ax.set_xlabel("indentation [mm]")
    ax.set_ylabel("Fz [N]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_all_curves(path: Path, out_dir: Path, cases: Sequence[InclusionCase]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), 1, figsize=(8.2, 2.35 * len(cases)), sharex=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        with np.load(out_dir / case.label / "sample.npz", allow_pickle=False) as sample:
            depth_all = np.asarray(sample["indentation_depth"], dtype=np.float32)
            fz_all = np.asarray(sample["fz"], dtype=np.float32)
            xy_all = np.asarray(sample["xy"], dtype=np.float32)
        row_idx = _center_row_index(xy_all)
        col_indices = _representative_columns(fz_all.shape[1])
        depth = depth_all[row_idx]
        fz = fz_all[row_idx]
        xy = xy_all[row_idx]
        for col in col_indices:
            x_mm = float(xy[col, 0] / MM)
            width = 2.8 if abs(x_mm) < 1.0e-4 else 1.4
            ax.plot(depth[col] / MM, fz[col], linewidth=width, label=f"{x_mm:.0f}mm")
        y_mm = float(xy[0, 1] / MM)
        ax.set_title(f"{case.label} center row y={y_mm:.0f}mm", fontsize=10)
        ax.set_ylabel("Fz [N]")
        ax.grid(True, alpha=0.22)
    axes[-1].set_xlabel("indentation [mm]")
    legend_cols = min(8, len(col_indices)) if cases else 1
    axes[0].legend(fontsize=7, ncol=legend_cols, loc="upper left", bbox_to_anchor=(0.0, 1.35))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _center_row_index(xy: np.ndarray) -> int:
    y = np.asarray(xy[:, 0, 1], dtype=np.float32)
    return int(np.argmin(np.abs(y)))


def _representative_columns(count: int, *, max_columns: int = 17) -> list[int]:
    if count <= max_columns:
        return list(range(count))
    return sorted({int(v) for v in np.linspace(0, count - 1, max_columns)})


def _plot_slope_vs_distance(path: Path, rows: Sequence[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["xy_distance_to_inclusion_mm"] is None:
            continue
        grouped.setdefault(str(row["case"]), []).append(row)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for case, case_rows in grouped.items():
        ordered = sorted(case_rows, key=lambda r: float(r["xy_distance_to_inclusion_mm"]))
        ax.plot(
            [float(r["xy_distance_to_inclusion_mm"]) for r in ordered],
            [float(r["late_slope_n_per_m"]) for r in ordered],
            marker="o",
            linewidth=2.0,
            label=case,
        )
    ax.set_xlabel("xy distance to inclusion center [mm]")
    ax.set_ylabel("late F-z slope [N/m]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _segment_slope(depth: np.ndarray, force: np.ndarray, lo: float, hi: float) -> float:
    span = float(np.max(depth) - np.min(depth))
    if span <= 1.0e-12:
        return 0.0
    keep = (depth >= float(np.min(depth)) + lo * span) & (depth <= float(np.min(depth)) + hi * span)
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    z = depth[keep].astype(np.float64)
    f = force[keep].astype(np.float64)
    zc = z - float(np.mean(z))
    denom = float(np.sum(zc * zc))
    if denom <= 1.0e-18:
        return 0.0
    return float(np.sum(zc * (f - float(np.mean(f)))) / denom)


def _endpoint_stiffness(depth: np.ndarray, force: np.ndarray) -> float:
    dz = float(depth[-1] - depth[0])
    if dz <= 1.0e-12:
        return 0.0
    return float((force[-1] - force[0]) / dz)


def _xy_distance_mm(case: InclusionCase, xy_m: np.ndarray) -> float | None:
    if case.center_mm is None:
        return None
    dx = float(xy_m[0] / MM) - float(case.center_mm[0])
    dy = float(xy_m[1] / MM) - float(case.center_mm[1])
    return float(np.sqrt(dx * dx + dy * dy))


def _center_depth_mm(case: InclusionCase) -> float | None:
    if case.center_mm is None:
        return None
    return float(PHANTOM_SIZE_M[2] / MM - case.center_mm[2])


def _top_depth_mm(case: InclusionCase) -> float | None:
    if case.center_mm is None or case.radii_mm is None:
        return None
    z_extent = _z_extent_mm(case)
    return float(max(PHANTOM_SIZE_M[2] / MM - (case.center_mm[2] + z_extent), 0.0))


def _z_extent_mm(case: InclusionCase) -> float:
    if case.radii_mm is None:
        return 0.0
    if case.shape == "capsule":
        return float(case.radii_mm[0] + case.radii_mm[2])
    return float(case.radii_mm[2])


if __name__ == "__main__":
    main()
