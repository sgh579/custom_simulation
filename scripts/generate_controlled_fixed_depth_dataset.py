from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.analytic import run_analytic_sample
from palpation_sim.config import MaterialConfig, PhantomConfig, ScanConfig
from palpation_sim.exports import build_dataset_metadata, build_ground_truth_metadata, write_visualization_command
from palpation_sim.features import extract_feature_map
from palpation_sim.newton_vbd import NewtonVBDPalpationSimulator
from palpation_sim.phantom import LumpSpec, lumps_membership, mask_for_scan_grid, sample_lumps
from palpation_sim.workflow import DEFAULT_NEWTON_ROOT, REQUIRED_NEWTON_DEVICE, ResourceMonitor, json_ready, require_runtime_environment


DEFAULT_OUT_DIR = Path(
    "data/palpation_fixed_depth_36mm_40step_ecoflex0010_linear_2000train_800val_800test_20260629"
)
SPLITS = ("train", "val", "test")
DEFAULT_SPLIT_COUNTS = {"train": 2000, "val": 800, "test": 800}
REFERENCE_LUMP_COUNT_HIST = {
    "train": {1: 253, 2: 267, 3: 212, 4: 268},
    "val": {1: 99, 2: 83, 3: 107, 4: 111},
    "test": {1: 101, 2: 97, 3: 105, 4: 97},
}
ALLOWED_SHAPES = ("sphere", "ellipsoid", "box", "cylinder", "capsule")


@dataclass(frozen=True)
class PlannedSample:
    split: str
    split_index: int
    global_index: int
    sample_id: str
    normal_k_mu: float
    normal_k_lambda: float
    lump_count: int
    lump_stiffness_multipliers: tuple[float, ...]
    geometry_seed: int

    @property
    def npz_name(self) -> str:
        return f"{self.sample_id}.npz"

    @property
    def metadata_name(self) -> str:
        return f"{self.sample_id}_gt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the controlled 36 mm fixed-depth palpation dataset with resumable manifest-driven samples."
    )
    parser.add_argument("--backend", choices=("newton", "analytic"), default="newton")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-train", type=int, default=DEFAULT_SPLIT_COUNTS["train"])
    parser.add_argument("--num-val", type=int, default=DEFAULT_SPLIT_COUNTS["val"])
    parser.add_argument("--num-test", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replace-manifest", action="store_true")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--no-save-phantom-3d", action="store_true")
    parser.add_argument("--no-save-press-records", action="store_true")
    parser.add_argument("--no-save-scan-animation", action="store_true")
    parser.add_argument("--allow-empty-scan-mask", action="store_true")

    parser.add_argument("--grid-h", type=int, default=20)
    parser.add_argument("--grid-w", type=int, default=20)
    parser.add_argument("--edge-margin", type=float, default=0.015)
    parser.add_argument("--probe-radius", type=float, default=0.012)
    parser.add_argument("--press-steps", type=int, default=40)
    parser.add_argument("--max-indentation", type=float, default=0.036)
    parser.add_argument("--substeps-per-depth", type=int, default=3)
    parser.add_argument("--vbd-iterations", type=int, default=10)

    parser.add_argument("--cells-x", type=int, default=48)
    parser.add_argument("--cells-y", type=int, default=48)
    parser.add_argument("--cells-z", type=int, default=16)
    parser.add_argument("--size-x", type=float, default=0.18)
    parser.add_argument("--size-y", type=float, default=0.18)
    parser.add_argument("--height", type=float, default=0.08)
    parser.add_argument("--particle-radius", type=float, default=0.004)
    parser.add_argument("--lump-shapes", type=str, default=",".join(ALLOWED_SHAPES))
    parser.add_argument("--lump-size-scale", type=float, default=1.0)
    parser.add_argument("--max-lump-radius-fraction", type=float, default=0.2)
    parser.add_argument("--min-center-depth", type=float, default=None)
    parser.add_argument("--max-center-depth", type=float, default=None)
    parser.add_argument("--allow-lump-overlap", action="store_true")
    parser.add_argument("--allow-z-overlap", action="store_true")
    parser.add_argument("--z-gap", type=float, default=0.0)

    parser.add_argument("--normal-k-mu-min", type=float, default=8000.0)
    parser.add_argument("--normal-k-mu-max", type=float, default=12000.0)
    parser.add_argument("--normal-k-lambda-min", type=float, default=8000.0)
    parser.add_argument("--normal-k-lambda-max", type=float, default=12000.0)
    parser.add_argument("--lump-stiffness-min", type=float, default=5.0)
    parser.add_argument("--lump-stiffness-max", type=float, default=30.0)

    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON_ROOT)
    parser.add_argument("--device", type=str, default=REQUIRED_NEWTON_DEVICE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_runtime_environment(require_newton=args.backend == "newton", newton_root=args.newton_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("logs", "status"):
        (args.out_dir / dirname).mkdir(exist_ok=True)

    phantom = PhantomConfig(
        size_x=float(args.size_x),
        size_y=float(args.size_y),
        height=float(args.height),
        cells_x=int(args.cells_x),
        cells_y=int(args.cells_y),
        cells_z=int(args.cells_z),
        particle_radius=float(args.particle_radius),
    )
    scan = ScanConfig(
        grid_h=int(args.grid_h),
        grid_w=int(args.grid_w),
        edge_margin=float(args.edge_margin),
        probe_radius=float(args.probe_radius),
        press_steps=int(args.press_steps),
        max_indentation=float(args.max_indentation),
        sim_substeps_per_depth=int(args.substeps_per_depth),
        vbd_iterations=int(args.vbd_iterations),
    )
    shapes = parse_shapes(args.lump_shapes)
    split_counts = {"train": int(args.num_train), "val": int(args.num_val), "test": int(args.num_test)}
    manifest_path = args.out_dir / "planned_manifest.jsonl"
    samples = load_or_create_manifest(args, split_counts, manifest_path)
    write_dataset_metadata(args, split_counts, phantom, scan, samples)

    selected = [
        row
        for row in samples
        if row.global_index % int(args.worker_count) == int(args.worker_index)
    ]
    if args.max_samples is not None:
        selected = selected[: max(int(args.max_samples), 0)]
    status_path = args.out_dir / "status" / f"worker_{args.worker_index:03d}_status.json"
    events_path = args.out_dir / "logs" / f"worker_{args.worker_index:03d}_events.jsonl"
    print(f"output dir: {args.out_dir}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"worker {args.worker_index}/{args.worker_count}: {len(selected)} planned samples", flush=True)

    dataset_started = time.time()
    completed = 0
    skipped = 0
    failed = 0
    for row in selected:
        write_worker_status(status_path, args, row, completed=completed, skipped=skipped, failed=failed, state="running")
        out_path = args.out_dir / row.split / row.npz_name
        gt_path = args.out_dir / row.split / row.metadata_name
        if args.resume and sample_complete(out_path, gt_path, row, scan):
            write_visualization_command(out_path, project_root=PROJECT_ROOT)
            skipped += 1
            append_event(events_path, {"event": "skip", "sample": row.sample_id, "split": row.split})
            continue
        remove_incomplete_artifacts(out_path, gt_path)
        try:
            generate_one(args, row, phantom, scan, shapes, out_path, gt_path)
        except Exception as exc:
            failed += 1
            append_event(
                events_path,
                {"event": "failed", "sample": row.sample_id, "split": row.split, "error": repr(exc)},
            )
            write_worker_status(
                status_path,
                args,
                row,
                completed=completed,
                skipped=skipped,
                failed=failed,
                state="failed",
                error=repr(exc),
            )
            raise
        completed += 1
        append_event(events_path, {"event": "complete", "sample": row.sample_id, "split": row.split})
        print(f"[{row.split}] wrote {out_path}", flush=True)

    write_worker_status(
        status_path,
        args,
        selected[-1] if selected else None,
        completed=completed,
        skipped=skipped,
        failed=failed,
        state="complete",
        elapsed_seconds=time.time() - dataset_started,
    )


def validate_args(args: argparse.Namespace) -> None:
    if int(args.worker_count) < 1:
        raise SystemExit("--worker-count must be >= 1")
    if int(args.worker_index) < 0 or int(args.worker_index) >= int(args.worker_count):
        raise SystemExit("--worker-index must be in [0, --worker-count)")
    for name in ("num_train", "num_val", "num_test", "grid_h", "grid_w", "press_steps"):
        if int(getattr(args, name)) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if not (float(args.normal_k_mu_min) > 0 and float(args.normal_k_mu_max) >= float(args.normal_k_mu_min)):
        raise SystemExit("normal k_mu bounds are invalid")
    if not (float(args.normal_k_lambda_min) > 0 and float(args.normal_k_lambda_max) >= float(args.normal_k_lambda_min)):
        raise SystemExit("normal k_lambda bounds are invalid")
    if not (float(args.lump_stiffness_min) > 0 and float(args.lump_stiffness_max) >= float(args.lump_stiffness_min)):
        raise SystemExit("lump stiffness bounds are invalid")


def load_or_create_manifest(
    args: argparse.Namespace,
    split_counts: dict[str, int],
    manifest_path: Path,
) -> list[PlannedSample]:
    if manifest_path.exists() and not args.replace_manifest:
        return read_manifest(manifest_path)
    if manifest_path.exists():
        backup = manifest_path.with_suffix(f".bak-{int(time.time())}.jsonl")
        manifest_path.replace(backup)
    samples = build_manifest(args, split_counts)
    write_manifest_atomic(manifest_path, samples)
    write_manifest_csv(manifest_path.with_suffix(".csv"), samples)
    return samples


def build_manifest(args: argparse.Namespace, split_counts: dict[str, int]) -> list[PlannedSample]:
    samples: list[PlannedSample] = []
    global_index = 0
    for split in SPLITS:
        count = int(split_counts[split])
        normal_mu = permuted_linspace(
            float(args.normal_k_mu_min),
            float(args.normal_k_mu_max),
            count,
            seed=stable_seed(args.seed, split, "normal_mu"),
        )
        normal_lambda = permuted_linspace(
            float(args.normal_k_lambda_min),
            float(args.normal_k_lambda_max),
            count,
            seed=stable_seed(args.seed, split, "normal_lambda"),
        )
        lump_counts = planned_lump_count_sequence(split, count, seed=stable_seed(args.seed, split, "lump_counts"))
        total_lumps = int(sum(lump_counts))
        lump_ratios = permuted_linspace(
            float(args.lump_stiffness_min),
            float(args.lump_stiffness_max),
            total_lumps,
            seed=stable_seed(args.seed, split, "lump_ratios"),
        )
        cursor = 0
        for split_index, lump_count in enumerate(lump_counts):
            ratios = tuple(float(v) for v in lump_ratios[cursor : cursor + int(lump_count)])
            cursor += int(lump_count)
            samples.append(
                PlannedSample(
                    split=split,
                    split_index=int(split_index),
                    global_index=int(global_index),
                    sample_id=f"sample_{split_index:04d}",
                    normal_k_mu=float(normal_mu[split_index]),
                    normal_k_lambda=float(normal_lambda[split_index]),
                    lump_count=int(lump_count),
                    lump_stiffness_multipliers=ratios,
                    geometry_seed=stable_seed(args.seed, split, split_index, "geometry"),
                )
            )
            global_index += 1
    return samples


def planned_lump_count_sequence(split: str, count: int, *, seed: int) -> np.ndarray:
    hist = REFERENCE_LUMP_COUNT_HIST[split]
    total = float(sum(hist.values()))
    raw = {k: float(v) * float(count) / total for k, v in hist.items()}
    allocated = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = int(count) - sum(allocated.values())
    order = sorted(raw, key=lambda k: (raw[k] - allocated[k], raw[k]), reverse=True)
    for key in order[:remainder]:
        allocated[key] += 1
    values: list[int] = []
    for lump_count, n in sorted(allocated.items()):
        values.extend([int(lump_count)] * int(n))
    array = np.asarray(values, dtype=np.int16)
    rng = np.random.default_rng(seed)
    rng.shuffle(array)
    return array


def generate_one(
    args: argparse.Namespace,
    row: PlannedSample,
    phantom: PhantomConfig,
    scan: ScanConfig,
    shapes: tuple[str, ...],
    out_path: Path,
    gt_path: Path,
) -> None:
    material = material_for_row(args, row)
    rng = np.random.default_rng(row.geometry_seed)
    depth_range = None
    if args.min_center_depth is not None or args.max_center_depth is not None:
        depth_range = (
            0.0 if args.min_center_depth is None else float(args.min_center_depth),
            phantom.height if args.max_center_depth is None else float(args.max_center_depth),
        )
    lumps = sample_controlled_lumps(args, row, phantom, material, scan, shapes, rng, depth_range)
    monitor = ResourceMonitor(device=args.device if args.backend == "newton" else None).start()
    if args.backend == "newton":
        simulator = NewtonVBDPalpationSimulator(
            phantom,
            material,
            scan,
            newton_root=args.newton_root,
            device=args.device,
        )
        sample = simulator.run_sample(lumps)
    else:
        sample = run_analytic_sample(phantom, material, scan, lumps, rng)

    sample["mask"] = full_phantom_xy_mask(phantom, scan.grid_h, scan.grid_w, lumps)
    sample["scan_mask"] = mask_for_scan_grid(scan, phantom, lumps)
    sample["label_xy"] = full_phantom_label_xy(phantom, scan.grid_h, scan.grid_w)
    sample["label_extent_json"] = np.asarray(
        json.dumps(
            {
                "mask_key": "mask",
                "xy_key": "label_xy",
                "extent": "full_phantom_xy",
                "x_min": -0.5 * phantom.size_x,
                "x_max": 0.5 * phantom.size_x,
                "y_min": -0.5 * phantom.size_y,
                "y_max": 0.5 * phantom.size_y,
                "note": "scan xy remains edge-margin limited; GT mask spans the complete phantom XY plane.",
            },
            sort_keys=True,
        )
    )
    sample["controlled_schedule_json"] = np.asarray(json.dumps(schedule_payload(row), sort_keys=True))
    sample["phantom_json"] = json.dumps(phantom.to_dict())
    sample["material_json"] = json.dumps(material.to_dict())
    sample["scan_json"] = json.dumps(scan.to_dict())
    if args.save_features:
        sample["features"] = extract_feature_map(sample["presses"])  # type: ignore[arg-type]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_ground_truth_metadata(
        sample_id=row.sample_id,
        split=row.split,
        phantom=phantom,
        material=material,
        scan=scan,
        lumps=lumps,
        sample=sample,
        npz_path=out_path,
        metadata_path=gt_path,
        gltf_path=None,
        press_records_dir=None,
        scan_animation_path=None,
    )
    metadata["controlled_schedule"] = schedule_payload(row)
    metadata["label"] = {
        "mask_key": "mask",
        "xy_key": "label_xy",
        "extent": "full_phantom_xy",
        "shape": [int(scan.grid_h), int(scan.grid_w)],
        "x_values": full_axis_values(phantom.size_x, scan.grid_w),
        "y_values": full_axis_values(phantom.size_y, scan.grid_h),
    }
    metadata["scan_label_compatibility"] = {
        "scan_xy_key": "xy",
        "scan_mask_key": "scan_mask",
        "scan_extent": "edge_margin_limited",
    }

    write_npz_atomic(out_path, sample)
    viz_path = write_visualization_command(out_path, project_root=PROJECT_ROOT)
    metadata["files"]["visualization_command"] = viz_path.name
    resource_usage = monitor.finish(storage_root=(out_path, viz_path))
    metadata["resource_usage"] = resource_usage
    write_json_atomic(gt_path, metadata)
    if not sample_complete(out_path, gt_path, row, scan):
        raise RuntimeError(f"generated sample failed validation: {out_path}")


def sample_controlled_lumps(
    args: argparse.Namespace,
    row: PlannedSample,
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ScanConfig,
    shapes: tuple[str, ...],
    rng: np.random.Generator,
    depth_range: tuple[float, float] | None,
) -> list[LumpSpec]:
    for _ in range(200):
        lumps = sample_lumps(
            rng,
            phantom,
            material,
            count_min=row.lump_count,
            count_max=row.lump_count,
            shapes=shapes,  # type: ignore[arg-type]
            size_scale=float(args.lump_size_scale),
            center_depth_range=depth_range,
            max_radius_fraction=float(args.max_lump_radius_fraction),
            allow_overlap=bool(args.allow_lump_overlap),
            separate_z=not bool(args.allow_z_overlap),
            z_gap=float(args.z_gap),
        )
        lumps = [
            LumpSpec(
                shape=lump.shape,
                center=lump.center,
                radii=lump.radii,
                stiffness_multiplier=float(row.lump_stiffness_multipliers[idx]),
                yaw=lump.yaw,
            )
            for idx, lump in enumerate(lumps)
        ]
        full_mask = full_phantom_xy_mask(phantom, scan.grid_h, scan.grid_w, lumps)
        scan_mask = mask_for_scan_grid(scan, phantom, lumps)
        if float(full_mask.sum()) <= 0.0:
            continue
        if not args.allow_empty_scan_mask and float(scan_mask.sum()) <= 0.0:
            continue
        return lumps
    raise RuntimeError(f"Could not sample valid lumps for {row.split}/{row.sample_id}")


def material_for_row(args: argparse.Namespace, row: PlannedSample) -> MaterialConfig:
    return MaterialConfig(
        k_mu=float(row.normal_k_mu),
        k_lambda=float(row.normal_k_lambda),
        k_damp=1.0e-4,
        soft_contact_ke=2.0e6,
        soft_contact_kd=1.0e-7,
        soft_contact_mu=0.5,
        probe_contact_mu=0.8,
        lump_stiffness_min=float(args.lump_stiffness_min),
        lump_stiffness_max=float(args.lump_stiffness_max),
    )


def sample_complete(out_path: Path, gt_path: Path, row: PlannedSample, scan: ScanConfig) -> bool:
    if not out_path.exists() or not gt_path.exists():
        return False
    try:
        with np.load(out_path, allow_pickle=False) as sample:
            require_array(sample, "fz", (scan.grid_h, scan.grid_w, scan.press_steps))
            require_array(sample, "presses", (scan.grid_h, scan.grid_w, scan.press_steps, 2))
            require_array(sample, "probe_wrench", (scan.grid_h, scan.grid_w, scan.press_steps, 6))
            require_array(sample, "mask", (scan.grid_h, scan.grid_w))
            require_array(sample, "scan_mask", (scan.grid_h, scan.grid_w))
            require_array(sample, "label_xy", (scan.grid_h, scan.grid_w, 2))
            fz = np.asarray(sample["fz"], dtype=np.float32)
            wrench = np.asarray(sample["probe_wrench"], dtype=np.float32)
            if not np.allclose(fz, np.maximum(wrench[..., 2], 0.0), rtol=1.0e-4, atol=1.0e-4):
                raise ValueError("fz does not match positive probe_wrench z force")
            schedule = json.loads(np.asarray(sample["controlled_schedule_json"]).reshape(()).item())
            if abs(float(schedule["normal_k_mu"]) - row.normal_k_mu) > 1.0e-6:
                raise ValueError("normal_k_mu mismatch")
            if abs(float(schedule["normal_k_lambda"]) - row.normal_k_lambda) > 1.0e-6:
                raise ValueError("normal_k_lambda mismatch")
            ratios = [float(v) for v in schedule["lump_stiffness_multipliers"]]
            if len(ratios) != row.lump_count:
                raise ValueError("lump multiplier count mismatch")
            if np.max(np.abs(np.asarray(ratios) - np.asarray(row.lump_stiffness_multipliers))) > 1.0e-6:
                raise ValueError("lump multiplier mismatch")
            if float(np.asarray(sample["mask"]).sum()) <= 0.0:
                raise ValueError("full phantom mask is empty")
        with gt_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("label", {}).get("extent") != "full_phantom_xy":
            raise ValueError("metadata label extent mismatch")
    except Exception:
        return False
    return True


def require_array(sample: np.lib.npyio.NpzFile, key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in sample.files:
        raise KeyError(key)
    value = np.asarray(sample[key])
    if value.shape != shape:
        raise ValueError(f"{key} shape {value.shape} != {shape}")
    if value.dtype.kind in "fiu" and not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def full_phantom_xy_mask(
    phantom: PhantomConfig,
    label_h: int,
    label_w: int,
    lumps: LumpSpec | Sequence[LumpSpec],
) -> np.ndarray:
    label_xy = full_phantom_label_xy(phantom, label_h, label_w)
    points = np.concatenate([label_xy, np.zeros((*label_xy.shape[:2], 1), dtype=np.float32)], axis=-1)
    return lumps_membership(points, lumps, project_xy=True).astype(np.float32)


def full_phantom_label_xy(phantom: PhantomConfig, label_h: int, label_w: int) -> np.ndarray:
    xs = np.asarray(full_axis_values(phantom.size_x, label_w), dtype=np.float32)
    ys = np.asarray(full_axis_values(phantom.size_y, label_h), dtype=np.float32)
    return np.stack(np.meshgrid(xs, ys), axis=-1).astype(np.float32)


def full_axis_values(side_length: float, count: int) -> list[float]:
    if int(count) <= 1:
        return [0.0]
    return np.linspace(-0.5 * float(side_length), 0.5 * float(side_length), int(count), dtype=np.float64).tolist()


def schedule_payload(row: PlannedSample) -> dict[str, object]:
    return {
        "split": row.split,
        "split_index": row.split_index,
        "global_index": row.global_index,
        "sample_id": row.sample_id,
        "normal_k_mu": row.normal_k_mu,
        "normal_k_lambda": row.normal_k_lambda,
        "lump_count": row.lump_count,
        "lump_stiffness_multipliers": list(row.lump_stiffness_multipliers),
        "geometry_seed": row.geometry_seed,
    }


def write_dataset_metadata(
    args: argparse.Namespace,
    split_counts: dict[str, int],
    phantom: PhantomConfig,
    scan: ScanConfig,
    samples: Sequence[PlannedSample],
) -> None:
    mean_material = MaterialConfig(
        k_mu=0.5 * (float(args.normal_k_mu_min) + float(args.normal_k_mu_max)),
        k_lambda=0.5 * (float(args.normal_k_lambda_min) + float(args.normal_k_lambda_max)),
        lump_stiffness_min=float(args.lump_stiffness_min),
        lump_stiffness_max=float(args.lump_stiffness_max),
    )
    metadata = build_dataset_metadata(
        dataset_id=args.out_dir.name,
        backend=args.backend,
        seed=int(args.seed),
        phantom=phantom,
        material=mean_material,
        scan=scan,
        split_counts=split_counts,
        args=vars(args),
        out_dir=args.out_dir,
    )
    metadata["controlled_schedule"] = {
        "normal_stiffness": {
            "distribution": "linear coverage per split with deterministic permutation",
            "k_mu_min": float(args.normal_k_mu_min),
            "k_mu_max": float(args.normal_k_mu_max),
            "k_lambda_min": float(args.normal_k_lambda_min),
            "k_lambda_max": float(args.normal_k_lambda_max),
        },
        "lump_stiffness_ratio": {
            "distribution": "linear coverage over lump instances per split with deterministic permutation",
            "min": float(args.lump_stiffness_min),
            "max": float(args.lump_stiffness_max),
        },
        "sample_count": len(samples),
        "lump_instance_count": int(sum(row.lump_count for row in samples)),
    }
    metadata["label"] = {
        "mask_key": "mask",
        "xy_key": "label_xy",
        "extent": "full_phantom_xy",
        "shape": [int(scan.grid_h), int(scan.grid_w)],
        "x_values": full_axis_values(phantom.size_x, scan.grid_w),
        "y_values": full_axis_values(phantom.size_y, scan.grid_h),
        "bugfix": "GT spans the complete phantom XY plane, not the edge-margin-limited scan region.",
    }
    write_json_atomic(args.out_dir / "metadata.json", metadata)


def permuted_linspace(start: float, stop: float, count: int, *, seed: int) -> np.ndarray:
    if int(count) <= 0:
        return np.zeros((0,), dtype=np.float64)
    values = np.linspace(float(start), float(stop), int(count), dtype=np.float64)
    rng = np.random.default_rng(seed)
    return values[rng.permutation(int(count))]


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def parse_shapes(raw: str) -> tuple[str, ...]:
    shapes = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(shapes) - set(ALLOWED_SHAPES))
    if unknown:
        raise SystemExit(f"Unsupported lump shapes: {unknown}; allowed: {ALLOWED_SHAPES}")
    if not shapes:
        raise SystemExit("At least one lump shape is required")
    return shapes


def write_manifest_atomic(path: Path, samples: Sequence[PlannedSample]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(schedule_payload(row), sort_keys=True) + "\n")
    tmp.replace(path)


def write_manifest_csv(path: Path, samples: Sequence[PlannedSample]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "split_index",
                "global_index",
                "sample_id",
                "normal_k_mu",
                "normal_k_lambda",
                "lump_count",
                "lump_stiffness_multipliers",
                "geometry_seed",
            ],
        )
        writer.writeheader()
        for row in samples:
            payload = schedule_payload(row)
            payload["lump_stiffness_multipliers"] = json.dumps(payload["lump_stiffness_multipliers"])
            writer.writerow(payload)
    tmp.replace(path)


def read_manifest(path: Path) -> list[PlannedSample]:
    rows: list[PlannedSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                PlannedSample(
                    split=str(item["split"]),
                    split_index=int(item["split_index"]),
                    global_index=int(item["global_index"]),
                    sample_id=str(item["sample_id"]),
                    normal_k_mu=float(item["normal_k_mu"]),
                    normal_k_lambda=float(item["normal_k_lambda"]),
                    lump_count=int(item["lump_count"]),
                    lump_stiffness_multipliers=tuple(float(v) for v in item["lump_stiffness_multipliers"]),
                    geometry_seed=int(item["geometry_seed"]),
                )
            )
    return rows


def write_npz_atomic(path: Path, sample: dict[str, object]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **sample)
    tmp.replace(path)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_event(path: Path, payload: dict[str, object]) -> None:
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(payload), sort_keys=True) + "\n")


def write_worker_status(
    path: Path,
    args: argparse.Namespace,
    row: PlannedSample | None,
    *,
    completed: int,
    skipped: int,
    failed: int,
    state: str,
    elapsed_seconds: float | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "state": state,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "out_dir": str(args.out_dir),
        "worker_index": int(args.worker_index),
        "worker_count": int(args.worker_count),
        "completed": int(completed),
        "skipped": int(skipped),
        "failed": int(failed),
        "current_sample": schedule_payload(row) if row is not None else None,
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = float(elapsed_seconds)
    if error is not None:
        payload["error"] = error
    write_json_atomic(path, payload)


def remove_incomplete_artifacts(out_path: Path, gt_path: Path) -> None:
    candidates = [
        out_path.with_name(f".{out_path.name}.tmp.npz"),
        gt_path.with_name(f".{gt_path.name}.tmp"),
    ]
    if out_path.exists() and not gt_path.exists():
        candidates.append(out_path)
    for path in candidates:
        if path.exists() and path.is_file():
            path.unlink()
    press_records = out_path.with_name(f"{out_path.stem}_press_records")
    if press_records.exists() and press_records.is_dir():
        shutil.rmtree(press_records)


if __name__ == "__main__":
    main()
