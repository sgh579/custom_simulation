from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.analytic import run_analytic_sample
from palpation_sim.config import MaterialConfig, PhantomConfig, ScanConfig
from palpation_sim.exports import (
    build_ground_truth_metadata,
    write_ground_truth_metadata,
    write_phantom_gltf,
    write_press_records,
    write_scan_animation_html,
)
from palpation_sim.features import extract_feature_map
from palpation_sim.newton_vbd import NewtonVBDPalpationSimulator
from palpation_sim.phantom import mask_for_scan_grid, sample_lumps
from palpation_sim.strain_stiffening import run_strain_stiffening_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic palpation process data.")
    parser.add_argument("--backend", choices=["newton", "analytic", "strain_stiffening"], default="newton")
    parser.add_argument("--out-dir", type=Path, default=Path("data/palpation"))
    parser.add_argument("--num-train", type=int, default=8)
    parser.add_argument("--num-val", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples whose .npz already exists while still advancing the sampler for deterministic continuation.",
    )
    parser.add_argument("--save-features", action="store_true", help="Also store engineered feature maps.")
    parser.add_argument("--no-save-gt-json", action="store_true", help="Do not write per-phantom GT metadata JSON files.")
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
    parser.add_argument(
        "--strain-hardening-b",
        type=float,
        default=1.8,
        help="Fung-like hardening strength for --backend strain_stiffening.",
    )
    parser.add_argument(
        "--strain-noise-std",
        type=float,
        default=0.0,
        help="Relative force noise for --backend strain_stiffening before convex enforcement.",
    )
    parser.add_argument(
        "--no-enforce-convex",
        action="store_true",
        help="Do not post-process strain-stiffening curves into monotone convex loading curves.",
    )

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

    parser.add_argument("--newton-root", type=Path, default=Path("/home/goodmansun/newton"))
    parser.add_argument("--device", type=str, default=None, help="Warp/Newton device, e.g. cpu or cuda:0.")
    parser.add_argument("--allow-empty-mask", action="store_true", help="Allow sampled lumps that miss all scan cells.")
    args = parser.parse_args()

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
    material = MaterialConfig()
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
    for split, count in split_counts.items():
        split_dir = args.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(count):
            out_path = split_dir / f"sample_{sample_idx:04d}.npz"
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
            if args.resume and out_path.exists():
                print(f"[{split}] skip existing {out_path}")
                continue
            if args.backend == "newton":
                assert simulator is not None
                sample = simulator.run_sample(lumps)
            elif args.backend == "strain_stiffening":
                sample = run_strain_stiffening_sample(
                    phantom,
                    material,
                    scan,
                    lumps,
                    rng,
                    hardening_b=args.strain_hardening_b,
                    noise_std=args.strain_noise_std,
                    enforce_convex=not args.no_enforce_convex,
                )
            else:
                sample = run_analytic_sample(phantom, material, scan, lumps, rng)

            sample["phantom_json"] = json.dumps(phantom.to_dict())
            sample["material_json"] = json.dumps(material.to_dict())
            sample["scan_json"] = json.dumps(scan.to_dict())
            if args.save_features:
                sample["features"] = extract_feature_map(sample["presses"])  # type: ignore[arg-type]

            gltf_path = None if args.no_save_phantom_3d else split_dir / f"sample_{sample_idx:04d}_phantom.gltf"
            gt_path = None if args.no_save_gt_json else split_dir / f"sample_{sample_idx:04d}_gt.json"
            press_records_dir = None if args.no_save_press_records else split_dir / f"sample_{sample_idx:04d}_press_records"
            scan_animation_path = (
                None if args.no_save_scan_animation else split_dir / f"sample_{sample_idx:04d}_scan_animation.html"
            )
            metadata = build_ground_truth_metadata(
                sample_id=f"sample_{sample_idx:04d}",
                split=split,
                phantom=phantom,
                material=material,
                scan=scan,
                lumps=lumps,
                sample=sample,
                npz_path=out_path,
                gltf_path=gltf_path,
                press_records_dir=press_records_dir,
                scan_animation_path=scan_animation_path,
            )
            np.savez_compressed(out_path, **sample)
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
            if gt_path is not None:
                write_ground_truth_metadata(gt_path, metadata)
            print(f"[{split}] wrote {out_path}")

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
