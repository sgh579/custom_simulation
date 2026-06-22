from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_palpation_dataset import (  # noqa: E402
    _material_from_args,
    _parse_shapes,
    _remove_sample_artifacts,
    _sample_artifacts_complete,
    _write_npz_atomic,
)
from palpation_sim.analytic import run_analytic_sample  # noqa: E402
from palpation_sim.config import PhantomConfig, ScanConfig  # noqa: E402
from palpation_sim.exports import (  # noqa: E402
    build_dataset_metadata,
    build_ground_truth_metadata,
    write_ground_truth_metadata,
    write_metadata_with_resource_usage,
    write_phantom_gltf,
    write_press_records,
    write_scan_animation_html,
    write_visualization_command,
)
from palpation_sim.features import extract_feature_map  # noqa: E402
from palpation_sim.newton_vbd import NewtonVBDPalpationSimulator  # noqa: E402
from palpation_sim.phantom import mask_for_scan_grid, sample_lumps  # noqa: E402
from palpation_sim.trajectories import PointedEllipseTrajectoryConfig, sample_pointed_ellipse_offsets  # noqa: E402
from palpation_sim.workflow import DEFAULT_NEWTON_ROOT, REQUIRED_NEWTON_DEVICE, ResourceMonitor, require_runtime_environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate palpation data with randomized nonlinear probe trajectories.")
    parser.add_argument("--backend", choices=["newton", "analytic"], default="newton")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-train-phantoms", type=int, default=100)
    parser.add_argument("--num-val-phantoms", type=int, default=40)
    parser.add_argument("--num-test-phantoms", type=int, default=40)
    parser.add_argument("--trajectory-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--no-save-phantom-3d", action="store_true")
    parser.add_argument("--no-save-press-records", action="store_true")
    parser.add_argument("--no-save-scan-animation", action="store_true")

    parser.add_argument("--grid-h", type=int, default=20)
    parser.add_argument("--grid-w", type=int, default=20)
    parser.add_argument("--edge-margin", type=float, default=0.015)
    parser.add_argument("--probe-radius", type=float, default=0.012)
    parser.add_argument("--press-steps", type=int, default=20)
    parser.add_argument("--max-indentation", type=float, default=0.018)
    parser.add_argument("--substeps-per-depth", type=int, default=3)
    parser.add_argument("--vbd-iterations", type=int, default=10)

    parser.add_argument("--cells-x", type=int, default=48)
    parser.add_argument("--cells-y", type=int, default=48)
    parser.add_argument("--cells-z", type=int, default=16)
    parser.add_argument("--size-x", type=float, default=0.18)
    parser.add_argument("--size-y", type=float, default=0.18)
    parser.add_argument("--height", type=float, default=0.08)
    parser.add_argument("--particle-radius", type=float, default=0.004)
    parser.add_argument("--lumps-min", type=int, default=1)
    parser.add_argument("--lumps-max", type=int, default=4)
    parser.add_argument("--lump-shapes", type=str, default="sphere,ellipsoid,box,cylinder,capsule")
    parser.add_argument("--lump-size-scale", type=float, default=1.0)
    parser.add_argument("--lump-stiffness-multiplier", type=float, default=20.0)
    parser.add_argument("--max-lump-radius-fraction", type=float, default=0.2)
    parser.add_argument("--min-center-depth", type=float, default=None)
    parser.add_argument("--max-center-depth", type=float, default=None)
    parser.add_argument("--allow-lump-overlap", action="store_true")
    parser.add_argument("--allow-z-overlap", action="store_true")
    parser.add_argument("--z-gap", type=float, default=0.0)
    parser.add_argument("--allow-empty-mask", action="store_true")

    parser.add_argument("--trajectory-amplitude-min", type=float, default=0.0006)
    parser.add_argument("--trajectory-amplitude-max", type=float, default=0.0030)
    parser.add_argument("--trajectory-aspect-min", type=float, default=0.30)
    parser.add_argument("--trajectory-aspect-max", type=float, default=0.85)
    parser.add_argument("--trajectory-cycles-min", type=float, default=0.75)
    parser.add_argument("--trajectory-cycles-max", type=float, default=1.35)
    parser.add_argument("--trajectory-sharpness-min", type=float, default=1.15)
    parser.add_argument("--trajectory-sharpness-max", type=float, default=2.40)
    parser.add_argument("--trajectory-skew-min", type=float, default=-0.45)
    parser.add_argument("--trajectory-skew-max", type=float, default=0.45)
    parser.add_argument("--trajectory-pointiness-min", type=float, default=0.10)
    parser.add_argument("--trajectory-pointiness-max", type=float, default=0.45)

    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT)
    parser.add_argument("--device", type=str, default=REQUIRED_NEWTON_DEVICE)
    args = parser.parse_args()

    if args.worker_count < 1:
        raise SystemExit("--worker-count must be >= 1.")
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise SystemExit("--worker-index must be in [0, --worker-count).")
    if args.trajectory_repeats < 1:
        raise SystemExit("--trajectory-repeats must be >= 1.")

    require_runtime_environment(require_newton=args.backend == "newton", newton_root=args.newton_root)
    rng = np.random.default_rng(args.seed)
    phantom = PhantomConfig(
        size_x=args.size_x,
        size_y=args.size_y,
        height=args.height,
        cells_x=args.cells_x,
        cells_y=args.cells_y,
        cells_z=args.cells_z,
        particle_radius=args.particle_radius,
    )
    material = _material_from_args(args)
    scan = ScanConfig(
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        edge_margin=args.edge_margin,
        probe_radius=args.probe_radius,
        press_steps=args.press_steps,
        max_indentation=args.max_indentation,
        sim_substeps_per_depth=args.substeps_per_depth,
        vbd_iterations=args.vbd_iterations,
    )
    trajectory_config = PointedEllipseTrajectoryConfig(
        amplitude_min=args.trajectory_amplitude_min,
        amplitude_max=args.trajectory_amplitude_max,
        aspect_min=args.trajectory_aspect_min,
        aspect_max=args.trajectory_aspect_max,
        cycles_min=args.trajectory_cycles_min,
        cycles_max=args.trajectory_cycles_max,
        sharpness_min=args.trajectory_sharpness_min,
        sharpness_max=args.trajectory_sharpness_max,
        skew_min=args.trajectory_skew_min,
        skew_max=args.trajectory_skew_max,
        pointiness_min=args.trajectory_pointiness_min,
        pointiness_max=args.trajectory_pointiness_max,
    )
    shapes = _parse_shapes(args.lump_shapes)
    depth_range = None
    if args.min_center_depth is not None or args.max_center_depth is not None:
        depth_range = (
            0.0 if args.min_center_depth is None else float(args.min_center_depth),
            phantom.height if args.max_center_depth is None else float(args.max_center_depth),
        )

    simulator = None
    if args.backend == "newton":
        simulator = NewtonVBDPalpationSimulator(
            phantom,
            material,
            scan,
            newton_root=args.newton_root,
            device=args.device,
        )

    split_phantom_counts = {
        "train": int(args.num_train_phantoms),
        "val": int(args.num_val_phantoms),
        "test": int(args.num_test_phantoms),
    }
    split_counts = {split: count * int(args.trajectory_repeats) for split, count in split_phantom_counts.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {args.out_dir}", flush=True)

    dataset_monitor = ResourceMonitor(device=args.device if args.backend == "newton" else None).start()
    dataset_metadata = build_dataset_metadata(
        dataset_id=args.out_dir.name,
        backend=args.backend,
        seed=args.seed,
        phantom=phantom,
        material=material,
        scan=scan,
        split_counts=split_counts,
        args=vars(args),
        out_dir=args.out_dir,
    )
    dataset_metadata["split_phantom_counts"] = split_phantom_counts
    dataset_metadata["trajectory"] = {
        **trajectory_config.to_dict(),
        "repeats_per_phantom": int(args.trajectory_repeats),
        "contract": "Each phantom is sampled once, then rescanned with randomized nonlinear trajectories centered at the same 20x20 grid points.",
    }
    write_ground_truth_metadata(args.out_dir / "metadata.json", dataset_metadata)

    global_sample_idx = 0
    for split, phantom_count in split_phantom_counts.items():
        split_dir = args.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for phantom_idx in range(phantom_count):
            lumps = sample_lumps(
                rng,
                phantom,
                material,
                count_min=args.lumps_min,
                count_max=args.lumps_max,
                shapes=shapes,
                size_scale=args.lump_size_scale,
                center_depth_range=depth_range,
                max_radius_fraction=args.max_lump_radius_fraction,
                allow_overlap=args.allow_lump_overlap,
                separate_z=not args.allow_z_overlap,
                z_gap=args.z_gap,
            )
            if not args.allow_empty_mask:
                for _ in range(100):
                    if float(mask_for_scan_grid(scan, phantom, lumps).sum()) > 0.0:
                        break
                    lumps = sample_lumps(
                        rng,
                        phantom,
                        material,
                        count_min=args.lumps_min,
                        count_max=args.lumps_max,
                        shapes=shapes,
                        size_scale=args.lump_size_scale,
                        center_depth_range=depth_range,
                        max_radius_fraction=args.max_lump_radius_fraction,
                        allow_overlap=args.allow_lump_overlap,
                        separate_z=not args.allow_z_overlap,
                        z_gap=args.z_gap,
                    )

            for repeat_idx in range(args.trajectory_repeats):
                sample_idx = phantom_idx * args.trajectory_repeats + repeat_idx
                sample_name = f"sample_{sample_idx:04d}"
                out_path = split_dir / f"{sample_name}.npz"
                gt_path = split_dir / f"{sample_name}_gt.json"
                gltf_path = None if args.no_save_phantom_3d else split_dir / f"{sample_name}_phantom.gltf"
                press_records_dir = None if args.no_save_press_records else split_dir / f"{sample_name}_press_records"
                scan_animation_path = None if args.no_save_scan_animation else split_dir / f"{sample_name}_scan_animation.html"

                offsets, trajectory_metadata = sample_pointed_ellipse_offsets(rng, scan, phantom, trajectory_config)
                trajectory_metadata = {
                    **trajectory_metadata,
                    "split": split,
                    "phantom_index": int(phantom_idx),
                    "repeat_index": int(repeat_idx),
                    "sample_index": int(sample_idx),
                    "global_sample_index": int(global_sample_idx),
                    "base_phantom_id": f"{split}_phantom_{phantom_idx:04d}",
                    "start_end_centered": True,
                }

                assigned_to_worker = global_sample_idx % args.worker_count == args.worker_index
                global_sample_idx += 1
                if not assigned_to_worker:
                    continue
                if args.resume and _trajectory_sample_complete(
                    out_path,
                    gt_path,
                    args.backend,
                    scan,
                    expected_multiplier=args.lump_stiffness_multiplier,
                    count_min=args.lumps_min,
                    count_max=args.lumps_max,
                ):
                    write_visualization_command(out_path, project_root=PROJECT_ROOT)
                    print(f"[{split}] skip existing {out_path}", flush=True)
                    continue
                if args.resume and out_path.exists():
                    _remove_sample_artifacts(out_path, gt_path, gltf_path, press_records_dir, scan_animation_path)
                    print(f"[{split}] regenerating incomplete {out_path}", flush=True)

                monitor = ResourceMonitor(device=args.device if args.backend == "newton" else None).start()
                if args.backend == "newton":
                    assert simulator is not None
                    sample = simulator.run_sample(
                        lumps,
                        trajectory_xy_offsets=offsets,
                        trajectory_metadata=trajectory_metadata,
                    )
                else:
                    sample = run_analytic_sample(phantom, material, scan, lumps, rng)
                    sample["trajectory_xy_offset"] = offsets
                    sample["trajectory_json"] = np.asarray(json.dumps(trajectory_metadata))

                sample["phantom_json"] = json.dumps(phantom.to_dict())
                sample["material_json"] = json.dumps(material.to_dict())
                sample["scan_json"] = json.dumps(scan.to_dict())
                sample["base_phantom_index"] = np.asarray(phantom_idx, dtype=np.int32)
                sample["trajectory_repeat_index"] = np.asarray(repeat_idx, dtype=np.int32)
                sample["global_sample_index"] = np.asarray(trajectory_metadata["global_sample_index"], dtype=np.int32)
                if args.save_features:
                    sample["features"] = extract_feature_map(sample["presses"])  # type: ignore[arg-type]

                metadata = build_ground_truth_metadata(
                    sample_id=sample_name,
                    split=split,
                    phantom=phantom,
                    material=material,
                    scan=scan,
                    lumps=lumps,
                    sample=sample,
                    npz_path=out_path,
                    metadata_path=gt_path,
                    gltf_path=gltf_path,
                    press_records_dir=press_records_dir,
                    scan_animation_path=scan_animation_path,
                )
                metadata["base_phantom_id"] = trajectory_metadata["base_phantom_id"]
                metadata["trajectory"] = trajectory_metadata

                _write_npz_atomic(out_path, sample)
                visualization_command_path = write_visualization_command(out_path, project_root=PROJECT_ROOT)
                metadata["files"]["visualization_command"] = visualization_command_path.name
                if gltf_path is not None:
                    write_phantom_gltf(gltf_path, phantom, lumps, material)
                if scan_animation_path is not None:
                    write_scan_animation_html(scan_animation_path, phantom, scan, sample, lumps)
                if press_records_dir is not None:
                    write_press_records(press_records_dir, sample, sample_id=sample_name, split=split)
                resource_usage = monitor.finish(
                    storage_root=(out_path, gt_path, gltf_path, press_records_dir, scan_animation_path, visualization_command_path)
                )
                write_metadata_with_resource_usage(
                    gt_path,
                    metadata,
                    resource_usage,
                    storage_root=(out_path, gt_path, gltf_path, press_records_dir, scan_animation_path, visualization_command_path),
                )
                print(f"[{split}] wrote {out_path}", flush=True)

    dataset_resource_usage = dataset_monitor.finish(storage_root=args.out_dir)
    write_metadata_with_resource_usage(
        args.out_dir / "metadata.json",
        dataset_metadata,
        dataset_resource_usage,
        storage_root=args.out_dir,
    )


def _trajectory_sample_complete(
    out_path: Path,
    gt_path: Path,
    backend: str,
    scan: ScanConfig,
    *,
    expected_multiplier: float | None,
    count_min: int,
    count_max: int,
) -> bool:
    if not _sample_artifacts_complete(
        out_path,
        gt_path,
        backend,
        scan,
        expected_multiplier=expected_multiplier,
        count_min=count_min,
        count_max=count_max,
    ):
        return False
    try:
        with np.load(out_path) as sample:
            offsets = np.asarray(sample["trajectory_xy_offset"], dtype=np.float32)
            if offsets.shape != (scan.grid_h, scan.grid_w, scan.press_steps, 2):
                raise ValueError(offsets.shape)
            if not np.isfinite(offsets).all():
                raise ValueError("non-finite trajectory offsets")
            if np.max(np.abs(offsets[:, :, 0, :])) > 1.0e-8 or np.max(np.abs(offsets[:, :, -1, :])) > 1.0e-8:
                raise ValueError("trajectory is not centered at start/end")
            expected_force_shape = (scan.grid_h, scan.grid_w, scan.press_steps, 3)
            expected_wrench_shape = (scan.grid_h, scan.grid_w, scan.press_steps, 6)
            probe_force = np.asarray(sample["probe_force"], dtype=np.float32)
            probe_torque = np.asarray(sample["probe_torque"], dtype=np.float32)
            probe_wrench = np.asarray(sample["probe_wrench"], dtype=np.float32)
            if probe_force.shape != expected_force_shape:
                raise ValueError(f"probe_force shape {probe_force.shape} != {expected_force_shape}")
            if probe_torque.shape != expected_force_shape:
                raise ValueError(f"probe_torque shape {probe_torque.shape} != {expected_force_shape}")
            if probe_wrench.shape != expected_wrench_shape:
                raise ValueError(f"probe_wrench shape {probe_wrench.shape} != {expected_wrench_shape}")
            if not np.isfinite(probe_force).all() or not np.isfinite(probe_torque).all() or not np.isfinite(probe_wrench).all():
                raise ValueError("probe wrench arrays contain non-finite values")
            if not np.allclose(probe_wrench[..., :3], probe_force, rtol=1.0e-5, atol=1.0e-6):
                raise ValueError("probe_wrench force channels do not match probe_force")
            if not np.allclose(probe_wrench[..., 3:], probe_torque, rtol=1.0e-5, atol=1.0e-6):
                raise ValueError("probe_wrench torque channels do not match probe_torque")
            if not np.allclose(np.maximum(probe_wrench[..., 2], 0.0), np.asarray(sample["fz"], dtype=np.float32), rtol=1.0e-5, atol=1.0e-6):
                raise ValueError("positive probe_wrench Fz channel does not match fz")
            trajectory_json = sample["trajectory_json"]
            trajectory = json.loads(np.asarray(trajectory_json).reshape(()).item())
            if trajectory.get("mode") != "pointed_ellipse":
                raise ValueError("wrong trajectory mode")
    except Exception:
        return False
    return True


if __name__ == "__main__":
    main()
