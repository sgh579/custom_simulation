from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.analytic import run_analytic_sample
from palpation_sim.config import MaterialConfig, PhantomConfig, ScanConfig
from palpation_sim.exports import (
    build_dataset_metadata,
    build_ground_truth_metadata,
    visualization_command_path,
    write_ground_truth_metadata,
    write_metadata_with_resource_usage,
    write_phantom_gltf,
    write_press_records,
    write_scan_animation_html,
    write_visualization_command,
)
from palpation_sim.features import extract_feature_map
from palpation_sim.newton_vbd import NewtonVBDPalpationSimulator
from palpation_sim.phantom import mask_for_scan_grid, sample_lumps
from palpation_sim.workflow import (
    DEFAULT_NEWTON_ROOT,
    REQUIRED_NEWTON_DEVICE,
    ResourceMonitor,
    require_runtime_environment,
    with_run_date_prefix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic palpation process data.")
    parser.add_argument("--backend", choices=["newton", "analytic"], default="newton")
    parser.add_argument("--out-dir", type=Path, default=Path("data/palpation"))
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-val", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples whose .npz already exists while still advancing the sampler for deterministic continuation.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="Number of deterministic sample-partition workers for parallel generation.",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="Zero-based worker index; this worker generates sample_idx %% worker_count.",
    )
    parser.add_argument("--save-features", action="store_true", help="Also store engineered feature maps.")
    parser.add_argument("--no-save-phantom-3d", action="store_true", help="Do not write per-phantom glTF 3D preview files.")
    parser.add_argument(
        "--no-save-press-records",
        action="store_true",
        help="Do not write per-sample press CSV/F-z plot folders.",
    )
    parser.add_argument(
        "--no-save-scan-animation",
        action="store_true",
        help="Do not write per-sample interactive 3D scan animation HTML files.",
    )

    parser.add_argument("--grid-h", type=int, default=9)
    parser.add_argument("--grid-w", type=int, default=9)
    parser.add_argument("--edge-margin", type=float, default=0.015)
    parser.add_argument("--probe-radius", type=float, default=0.012)
    parser.add_argument("--press-steps", type=int, default=16)
    parser.add_argument("--max-indentation", type=float, default=0.018)
    parser.add_argument("--substeps-per-depth", type=int, default=3)
    parser.add_argument("--vbd-iterations", type=int, default=5)

    parser.add_argument("--cells-x", type=int, default=32)
    parser.add_argument("--cells-y", type=int, default=32)
    parser.add_argument("--cells-z", type=int, default=12)
    parser.add_argument("--size-x", type=float, default=0.18)
    parser.add_argument("--size-y", type=float, default=0.18)
    parser.add_argument("--height", type=float, default=0.08)
    parser.add_argument("--particle-radius", type=float, default=0.004)
    parser.add_argument("--lumps-min", type=int, default=4, help="Minimum number of lumps per phantom.")
    parser.add_argument("--lumps-max", type=int, default=4, help="Maximum number of lumps per phantom.")
    parser.add_argument(
        "--lump-shapes",
        type=str,
        default="sphere,ellipsoid,box,cylinder,capsule",
        help="Comma-separated shape set: sphere,ellipsoid,box,cylinder,capsule.",
    )
    parser.add_argument("--lump-size-scale", type=float, default=1.0, help="Multiplier for sampled lump dimensions.")
    parser.add_argument(
        "--lump-stiffness-multiplier",
        type=float,
        default=None,
        help="When set, force every sampled lump to this stiffness multiplier.",
    )
    parser.add_argument(
        "--max-lump-radius-fraction",
        type=float,
        default=0.2,
        help="Max xy radius/half-axis as a fraction of the corresponding phantom x/y side length.",
    )
    parser.add_argument(
        "--min-center-depth",
        type=float,
        default=None,
        help="Optional min lump center depth from top surface [m].",
    )
    parser.add_argument(
        "--max-center-depth",
        type=float,
        default=None,
        help="Optional max lump center depth from top surface [m].",
    )
    parser.add_argument(
        "--allow-lump-overlap",
        action="store_true",
        help="Allow full 3D lump overlap when --allow-z-overlap is passed. The default z-separated sampler only enforces z-interval separation.",
    )
    parser.add_argument("--allow-z-overlap", action="store_true", help="Allow z intervals of lumps to overlap.")
    parser.add_argument("--z-gap", type=float, default=0.0, help="Required gap between lump z intervals [m].")

    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT, help="Pinned Newton source root.")
    parser.add_argument("--device", type=str, default=REQUIRED_NEWTON_DEVICE, help="Pinned Warp/Newton CUDA device.")
    parser.add_argument("--allow-empty-mask", action="store_true", help="Allow sampled lumps that miss all scan cells.")
    args = parser.parse_args()
    if args.worker_count < 1:
        raise SystemExit("--worker-count must be >= 1.")
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise SystemExit("--worker-index must be in [0, --worker-count).")
    require_runtime_environment(require_newton=args.backend == "newton", newton_root=args.newton_root)
    args.out_dir = with_run_date_prefix(args.out_dir, enabled=not args.resume)

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

    split_counts = {"train": args.num_train, "val": args.num_val}
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
    write_ground_truth_metadata(args.out_dir / "metadata.json", dataset_metadata)
    for split, count in split_counts.items():
        split_dir = args.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(count):
            out_path = split_dir / f"sample_{sample_idx:04d}.npz"
            gt_path = split_dir / f"sample_{sample_idx:04d}_gt.json"
            gltf_path = None if args.no_save_phantom_3d else split_dir / f"sample_{sample_idx:04d}_phantom.gltf"
            press_records_dir = None if args.no_save_press_records else split_dir / f"sample_{sample_idx:04d}_press_records"
            scan_animation_path = (
                None if args.no_save_scan_animation else split_dir / f"sample_{sample_idx:04d}_scan_animation.html"
            )
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
            if sample_idx % args.worker_count != args.worker_index:
                continue
            if args.resume and _sample_artifacts_complete(
                out_path,
                gt_path,
                args.backend,
                scan,
                expected_multiplier=args.lump_stiffness_multiplier,
                count_min=args.lumps_min,
                count_max=args.lumps_max,
            ):
                write_visualization_command(out_path, project_root=PROJECT_ROOT)
                print(f"[{split}] skip existing {out_path}")
                continue
            if args.resume and out_path.exists():
                _remove_sample_artifacts(out_path, gt_path, gltf_path, press_records_dir, scan_animation_path)
                print(f"[{split}] regenerating incomplete {out_path}")
            monitor = ResourceMonitor(device=args.device if args.backend == "newton" else None).start()
            if args.backend == "newton":
                assert simulator is not None
                sample = simulator.run_sample(lumps)
            else:
                sample = run_analytic_sample(phantom, material, scan, lumps, rng)

            sample["phantom_json"] = json.dumps(phantom.to_dict())
            sample["material_json"] = json.dumps(material.to_dict())
            sample["scan_json"] = json.dumps(scan.to_dict())
            if args.save_features:
                sample["features"] = extract_feature_map(sample["presses"])  # type: ignore[arg-type]

            metadata = build_ground_truth_metadata(
                sample_id=f"sample_{sample_idx:04d}",
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
            _write_npz_atomic(out_path, sample)
            visualization_command_path = write_visualization_command(out_path, project_root=PROJECT_ROOT)
            metadata["files"]["visualization_command"] = visualization_command_path.name
            if gltf_path is not None:
                write_phantom_gltf(gltf_path, phantom, lumps, material)
            if scan_animation_path is not None:
                write_scan_animation_html(scan_animation_path, phantom, scan, sample, lumps)
            if press_records_dir is not None:
                write_press_records(
                    press_records_dir,
                    sample,
                    sample_id=f"sample_{sample_idx:04d}",
                    split=split,
                )
            resource_usage = monitor.finish(
                storage_root=(out_path, gt_path, gltf_path, press_records_dir, scan_animation_path, visualization_command_path)
            )
            write_metadata_with_resource_usage(
                gt_path,
                metadata,
                resource_usage,
                storage_root=(out_path, gt_path, gltf_path, press_records_dir, scan_animation_path, visualization_command_path),
            )
            print(f"[{split}] wrote {out_path}")

    dataset_resource_usage = dataset_monitor.finish(storage_root=args.out_dir)
    write_metadata_with_resource_usage(
        args.out_dir / "metadata.json",
        dataset_metadata,
        dataset_resource_usage,
        storage_root=args.out_dir,
    )


def _material_from_args(args: argparse.Namespace) -> MaterialConfig:
    if args.lump_stiffness_multiplier is None:
        return MaterialConfig()
    multiplier = float(args.lump_stiffness_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise SystemExit("--lump-stiffness-multiplier must be a positive finite value.")
    return MaterialConfig(lump_stiffness_min=multiplier, lump_stiffness_max=multiplier)


def _write_npz_atomic(out_path: Path, sample: dict[str, object]) -> None:
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.npz")
    if tmp_path.exists():
        tmp_path.unlink()
    np.savez_compressed(tmp_path, **sample)
    tmp_path.replace(out_path)


def _sample_artifacts_complete(
    out_path: Path,
    gt_path: Path,
    backend: str,
    scan: ScanConfig,
    *,
    expected_multiplier: float | None,
    count_min: int,
    count_max: int,
) -> bool:
    if not out_path.exists() or not gt_path.exists():
        return False
    if not _valid_json_file(gt_path):
        return False
    try:
        with np.load(out_path) as sample:
            _require_npz_array(sample, "fz", (scan.grid_h, scan.grid_w, scan.press_steps))
            _require_npz_array(sample, "presses", (scan.grid_h, scan.grid_w, scan.press_steps, 2))
            _require_npz_array(sample, "mask", (scan.grid_h, scan.grid_w))
            _require_npz_member(sample, "lumps_json")
            _require_npz_member(sample, "num_lumps")
            num_lumps = int(np.asarray(sample["num_lumps"]).reshape(()))
            if num_lumps < int(count_min) or num_lumps > int(count_max):
                raise ValueError(f"num_lumps {num_lumps} outside [{count_min}, {count_max}]")
            lumps = json.loads(_npz_string(sample["lumps_json"]))
            if len(lumps) != num_lumps:
                raise ValueError(f"lumps_json length {len(lumps)} != num_lumps {num_lumps}")
            if expected_multiplier is not None:
                expected = float(expected_multiplier)
                for lump in lumps:
                    multiplier = float(lump.get("stiffness_multiplier", np.nan))
                    if abs(multiplier - expected) > 1.0e-6:
                        raise ValueError(f"stiffness_multiplier {multiplier} != {expected}")
            if backend == "newton":
                _require_npz_member(sample, "tet_lump_mask")
                _require_npz_member(sample, "tet_lump_id")
    except Exception:
        return False
    return True


def _valid_json_file(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return False
    return True


def _require_npz_array(sample: np.lib.npyio.NpzFile, key: str, shape: tuple[int, ...]) -> None:
    _require_npz_member(sample, key)
    value = np.asarray(sample[key])
    if value.shape != shape:
        raise ValueError(f"{key} shape {value.shape} != {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")


def _require_npz_member(sample: np.lib.npyio.NpzFile, key: str) -> None:
    if key not in sample.files:
        raise KeyError(key)


def _npz_string(value: object) -> str:
    item = np.asarray(value).reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _remove_sample_artifacts(
    out_path: Path,
    gt_path: Path,
    gltf_path: Path | None,
    press_records_dir: Path | None,
    scan_animation_path: Path | None,
) -> None:
    candidates = [
        out_path,
        out_path.with_name(f".{out_path.name}.tmp.npz"),
        gt_path,
        gltf_path,
        scan_animation_path,
        visualization_command_path(out_path),
    ]
    for path in candidates:
        if path is not None and path.exists() and path.is_file():
            path.unlink()
    if press_records_dir is not None and press_records_dir.exists():
        shutil.rmtree(press_records_dir)


def _parse_shapes(raw: str) -> tuple[str, ...]:
    allowed = {"sphere", "ellipsoid", "box", "cylinder", "capsule"}
    shapes = tuple(part.strip() for part in raw.split(",") if part.strip())
    bad = sorted(set(shapes) - allowed)
    if bad:
        raise SystemExit(f"Unsupported --lump-shapes values: {bad}. Allowed: {sorted(allowed)}")
    if not shapes:
        raise SystemExit("--lump-shapes must contain at least one shape.")
    return shapes


if __name__ == "__main__":
    main()
