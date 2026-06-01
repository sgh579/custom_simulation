from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.config import MaterialConfig, PhantomConfig, ScanConfig
from palpation_sim.phantom import LumpSpec
from palpation_sim.strain_stiffening import run_strain_stiffening_sample
from palpation_sim.xiao2020 import XIAO_DEPTH_CLASSES_MM, xiao_sequence_from_press


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Xiao et al. 2020-style single-palpation LSTM data from simulation."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/xiao_lstm_sim"))
    parser.add_argument("--samples-per-class-train", type=int, default=70)
    parser.add_argument("--samples-per-class-val", type=int, default=15)
    parser.add_argument("--samples-per-class-test", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--depths-mm", type=str, default="0,5,8,10")
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--loading-steps", type=int, default=64)
    parser.add_argument("--min-indentation-mm", type=float, default=4.0)
    parser.add_argument("--max-indentation-mm", type=float, default=10.0)
    parser.add_argument("--force-noise-std", type=float, default=0.025)
    parser.add_argument("--hysteresis-min", type=float, default=0.74)
    parser.add_argument("--hysteresis-max", type=float, default=0.90)
    parser.add_argument("--phantom-size-x-mm", type=float, default=100.0)
    parser.add_argument("--phantom-size-y-mm", type=float, default=50.0)
    parser.add_argument("--phantom-height-mm", type=float, default=20.0)
    parser.add_argument("--probe-radius-mm", type=float, default=5.0)
    parser.add_argument("--lump-radius-mm", type=float, default=3.0)
    parser.add_argument("--xy-jitter-mm", type=float, default=1.5)
    parser.add_argument("--stiffness-min", type=float, default=4.0)
    parser.add_argument("--stiffness-max", type=float, default=9.0)
    parser.add_argument("--hardening-b", type=float, default=1.8)
    parser.add_argument("--resume", action="store_true", help="Skip existing .npz files.")
    args = parser.parse_args()

    depths_mm = _parse_depths(args.depths_mm)
    if tuple(int(depth) for depth in depths_mm) != XIAO_DEPTH_CLASSES_MM:
        print(f"using custom class depths in mm: {depths_mm}")

    rng = np.random.default_rng(args.seed)
    phantom = PhantomConfig(
        size_x=_mm(args.phantom_size_x_mm),
        size_y=_mm(args.phantom_size_y_mm),
        height=_mm(args.phantom_height_mm),
        cells_x=16,
        cells_y=8,
        cells_z=8,
        particle_radius=_mm(1.0),
    )
    material = MaterialConfig()
    split_counts = {
        "train": int(args.samples_per_class_train),
        "val": int(args.samples_per_class_val),
        "test": int(args.samples_per_class_test),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for split, samples_per_class in split_counts.items():
        split_dir = args.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        sample_id = 0
        for label, depth_mm in enumerate(depths_mm):
            for class_sample_idx in range(samples_per_class):
                out_path = split_dir / f"sample_{sample_id:05d}.npz"
                sample_id += 1
                if args.resume and out_path.exists():
                    continue

                sim_sample = _simulate_one(
                    rng,
                    phantom=phantom,
                    material=material,
                    label=label,
                    depth_mm=depth_mm,
                    args=args,
                )
                np.savez_compressed(out_path, **sim_sample)
                rows.append(
                    {
                        "split": split,
                        "file": str(out_path.relative_to(args.out_dir)),
                        "label": label,
                        "depth_mm": depth_mm,
                        "max_indentation_mm": float(sim_sample["max_indentation_mm"]),
                        "stiffness_multiplier": float(sim_sample["stiffness_multiplier"]),
                    }
                )
                print(f"[{split}] wrote {out_path}")

    _write_manifest(args.out_dir, args, depths_mm, phantom, material, rows)


def _simulate_one(
    rng: np.random.Generator,
    *,
    phantom: PhantomConfig,
    material: MaterialConfig,
    label: int,
    depth_mm: float,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | str | float | int]:
    max_indentation_m = _mm(float(rng.uniform(args.min_indentation_mm, args.max_indentation_mm)))
    scan = ScanConfig(
        grid_h=1,
        grid_w=1,
        edge_margin=0.0,
        probe_radius=_mm(args.probe_radius_mm),
        max_indentation=max_indentation_m,
        press_steps=max(int(args.loading_steps), 3),
    )
    stiffness = 1.0
    lumps: list[LumpSpec] = []
    if depth_mm > 0.0:
        stiffness = float(
            np.exp(rng.uniform(np.log(float(args.stiffness_min)), np.log(float(args.stiffness_max))))
        )
        depth_m = _mm(depth_mm)
        radius_m = _safe_lump_radius(
            requested_radius=_mm(args.lump_radius_mm),
            center_depth=depth_m,
            phantom_height=phantom.height,
        )
        center_x = float(
            np.clip(rng.normal(0.0, _mm(args.xy_jitter_mm)), -0.45 * phantom.size_x, 0.45 * phantom.size_x)
        )
        center_y = float(
            np.clip(rng.normal(0.0, _mm(args.xy_jitter_mm)), -0.45 * phantom.size_y, 0.45 * phantom.size_y)
        )
        center_z = float(phantom.height - depth_m)
        lumps.append(
            LumpSpec(
                shape="sphere",
                center=(center_x, center_y, center_z),
                radii=(radius_m, radius_m, radius_m),
                stiffness_multiplier=stiffness,
            )
        )

    sim = run_strain_stiffening_sample(
        phantom,
        material,
        scan,
        lumps,
        rng,
        hardening_b=float(args.hardening_b),
        noise_std=float(args.force_noise_std),
        enforce_convex=True,
    )
    loading_press = np.asarray(sim["presses"], dtype=np.float32)[0, 0, :, :2]
    full_press = _add_unloading_branch(
        loading_press,
        rng,
        hysteresis_min=float(args.hysteresis_min),
        hysteresis_max=float(args.hysteresis_max),
    )
    sequence = xiao_sequence_from_press(full_press, sequence_length=int(args.sequence_length))
    metadata = {
        "backend": "strain_stiffening",
        "depth_definition": "lump center depth from the top surface",
        "class_depths_mm": _parse_depths(args.depths_mm),
        "paper_sequence": "columns are Fz, z, Fz/z after uniform resampling",
        "lumps": [lump.to_dict(phantom) for lump in lumps],
    }
    return {
        "sequence": sequence,
        "press": full_press.astype(np.float32),
        "loading_press": loading_press.astype(np.float32),
        "label": np.asarray(label, dtype=np.int64),
        "depth_mm": np.asarray(float(depth_mm), dtype=np.float32),
        "class_depths_mm": np.asarray(_parse_depths(args.depths_mm), dtype=np.float32),
        "max_indentation_mm": np.asarray(max_indentation_m * 1000.0, dtype=np.float32),
        "stiffness_multiplier": np.asarray(stiffness, dtype=np.float32),
        "phantom_json": json.dumps(phantom.to_dict()),
        "material_json": json.dumps(material.to_dict()),
        "scan_json": json.dumps(scan.to_dict()),
        "metadata_json": json.dumps(metadata),
    }


def _add_unloading_branch(
    loading_press: np.ndarray,
    rng: np.random.Generator,
    *,
    hysteresis_min: float,
    hysteresis_max: float,
) -> np.ndarray:
    loading_press = np.asarray(loading_press, dtype=np.float32)
    if loading_press.shape[0] < 2:
        return loading_press
    loading_press = loading_press.copy()
    loading_press[:, 0] = loading_press[:, 0] - np.float32(loading_press[0, 0])
    loading_press[:, 1] = np.maximum(loading_press[:, 1] - np.float32(loading_press[0, 1]), 0.0)
    loading_press[0, :] = 0.0
    unloading_z = loading_press[-2::-1, 0]
    unloading_f = loading_press[-2::-1, 1]
    max_z = max(float(np.max(loading_press[:, 0])), 1e-9)
    normalized_depth = np.clip(unloading_z / np.float32(max_z), 0.0, 1.0)
    hysteresis = float(rng.uniform(hysteresis_min, hysteresis_max))
    unloading_scale = hysteresis + (1.0 - hysteresis) * normalized_depth
    unloading_f = unloading_f * unloading_scale.astype(np.float32)
    unloading = np.stack([unloading_z, unloading_f], axis=-1).astype(np.float32)
    return np.concatenate([loading_press, unloading], axis=0).astype(np.float32)


def _write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    depths_mm: tuple[float, ...],
    phantom: PhantomConfig,
    material: MaterialConfig,
    rows: list[dict[str, object]],
) -> None:
    metadata = {
        "description": "Xiao et al. 2020 LSTM depth-estimation simulation dataset",
        "class_depths_mm": depths_mm,
        "sequence_columns": ["Fz", "z", "Fz/z"],
        "sequence_length": int(args.sequence_length),
        "phantom": phantom.to_dict(),
        "material": material.to_dict(),
        "args": vars(args) | {"out_dir": str(args.out_dir)},
    }
    with (out_dir / "manifest.json").open("w") as manifest_file:
        json.dump(metadata, manifest_file, indent=2)
    if rows:
        with (out_dir / "samples.csv").open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _parse_depths(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) < 2:
        raise SystemExit("--depths-mm must contain at least two comma-separated classes")
    if values[0] != 0.0:
        raise SystemExit("The first depth class must be 0 for inclusion-free samples")
    return values


def _safe_lump_radius(*, requested_radius: float, center_depth: float, phantom_height: float) -> float:
    margin = 5e-4
    max_radius = max(min(center_depth, phantom_height - center_depth) - margin, 5e-4)
    return float(min(requested_radius, max_radius))


def _mm(value: float) -> float:
    return float(value) / 1000.0


if __name__ == "__main__":
    main()
