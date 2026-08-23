from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_per_sample_norm_transfer_and_real import (  # noqa: E402
    build_model,
    threshold_for,
)
from run_highres_per_sample_norm_unet import (  # noqa: E402
    curve_force_response_scale_per_sample,
    stiffness_log_robust_per_sample,
)
from run_segmentation_accuracy_sweep import predict_neural  # noqa: E402
from segment_real_grid_scan_v2_delta_fz import (  # noqa: E402
    load_real_grid_scan,
    resample_real_response,
)


MODEL_DIRS = {
    "V1": {
        "curve": "runs/per_sample_norm_unet_128_strict_20260624_fixed_depth/"
        "r128_fixed_depth_unet_curve_force_response_scale_per_sample",
        "stiffness": "runs/per_sample_norm_unet_128_strict_20260624_fixed_depth/"
        "r128_fixed_depth_unet_stiffness_log_robust_per_sample",
    },
    "V2": {
        "curve": "runs/per_sample_norm_unet_128_v2_halfdepth_20260709/"
        "r128_unet_curve_force_response_scale_per_sample",
        "stiffness": "runs/per_sample_norm_unet_128_v2_halfdepth_20260709/"
        "r128_unet_stiffness_log_robust_per_sample",
    },
    "V1+V2": {
        "curve": "runs/per_sample_norm_unet_128_v1fixed_v2ref55k_halfdepth_20260709/"
        "r128_unet_curve_force_response_scale_per_sample",
        "stiffness": "runs/per_sample_norm_unet_128_v1fixed_v2ref55k_halfdepth_20260709/"
        "r128_unet_stiffness_log_robust_per_sample",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the six established per-sample-normalized U-Nets on a real 20x20 grid scan."
    )
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-max-depth-mm", type=float, default=9.5)
    parser.add_argument("--curve-steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    real = load_real_grid_scan(args.scan_dir)
    response_hwt = resample_real_response(
        real,
        int(args.curve_steps),
        policy="physical_hold",
        max_depth_m=float(args.target_max_depth_mm) / 1000.0,
    )
    response_bchw = np.moveaxis(response_hwt, -1, 0)[None].astype(np.float32)
    curve_input = curve_force_response_scale_per_sample(response_bchw)

    stiffness_n_per_mm = real["final_response_n"] / np.maximum(real["max_depth_m"] * 1000.0, 1.0e-6)
    stiffness_n_per_m = stiffness_n_per_mm[None, None].astype(np.float32) * 1000.0
    stiffness_input = stiffness_log_robust_per_sample(stiffness_n_per_m)
    inputs = {"curve": curve_input, "stiffness": stiffness_input}

    np.save(args.out_dir / "curve_response_resampled_hwt.npy", response_hwt.astype(np.float32))
    np.save(args.out_dir / "curve_input_normalized_chw.npy", curve_input[0].astype(np.float32))
    np.save(args.out_dir / "stiffness_n_per_mm.npy", stiffness_n_per_mm.astype(np.float32))
    np.save(args.out_dir / "stiffness_input_normalized_chw.npy", stiffness_input[0].astype(np.float32))

    requested_device = str(args.device)
    device = torch.device(requested_device if torch.cuda.is_available() and requested_device.startswith("cuda") else "cpu")
    rows: list[dict[str, Any]] = []
    scores: dict[tuple[str, str], np.ndarray] = {}
    masks: dict[tuple[str, str], np.ndarray] = {}

    for train_name, input_models in MODEL_DIRS.items():
        for input_name, relative_dir in input_models.items():
            method_dir = PROJECT_ROOT / relative_dir
            checkpoint_path = method_dir / "best.pt"
            threshold = threshold_for(method_dir)
            model, checkpoint, base_channels = build_model(method_dir, int(inputs[input_name].shape[1]), device)
            score = predict_neural(model, inputs[input_name], device, batch_size=1)[0].astype(np.float32)
            mask = (score >= threshold).astype(np.uint8)
            key = (train_name, input_name)
            scores[key] = score
            masks[key] = mask
            stem = f"{train_name.replace('+', 'plus').lower()}_{input_name}"
            np.save(args.out_dir / f"{stem}_probability_128x128.npy", score)
            np.save(args.out_dir / f"{stem}_mask_128x128.npy", mask)
            rows.append(
                {
                    "train_data": train_name,
                    "input": input_name,
                    "threshold": threshold,
                    "mask_positive_fraction": float(mask.mean()),
                    "mask_positive_pixels": int(mask.sum()),
                    "prob_min": float(np.min(score)),
                    "prob_p05": float(np.percentile(score, 5)),
                    "prob_median": float(np.median(score)),
                    "prob_mean": float(np.mean(score)),
                    "prob_p95": float(np.percentile(score, 95)),
                    "prob_max": float(np.max(score)),
                    "base_channels": int(base_channels),
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "checkpoint_best_val": checkpoint.get("best_val"),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                }
            )
            print(f"done {train_name} {input_name}", flush=True)

    overlap_rows = pairwise_overlap(masks)
    write_csv(args.out_dir / "six_unet_outputs_summary.csv", rows)
    write_csv(args.out_dir / "six_unet_pairwise_mask_overlap.csv", overlap_rows)
    render_outputs(args.out_dir, scores, masks)
    render_inputs(args.out_dir, response_hwt, stiffness_n_per_mm, real["max_depth_m"])

    payload = {
        "scan_dir": str(args.scan_dir),
        "output_dir": str(args.out_dir),
        "device": str(device),
        "grid": {
            "rows": int(real["rows"]),
            "cols": int(real["cols"]),
            "valid_points": int(real["valid_points"]),
            "image_convention": "origin=lower; grid col/x increases right; grid row/y increases up",
        },
        "preprocessing": {
            "response": "baseline_fz - fz, clipped at zero; baseline is the shallowest 2% median",
            "curve": (
                f"{int(args.curve_steps)} samples from 0 to {float(args.target_max_depth_mm):g} mm, "
                "then divide the full sample by its response p95"
            ),
            "stiffness": (
                "final_response / measured_max_depth in N/mm; convert to N/m, log1p, then median/IQR normalize "
                "within this scan"
            ),
            "threshold": "each checkpoint's validation-selected threshold from metrics_summary.json",
        },
        "input_stats": {
            "curve_normalized": summarize(curve_input),
            "stiffness_n_per_mm": summarize(stiffness_n_per_mm),
            "stiffness_normalized": summarize(stiffness_input),
            "max_depth_mm": summarize(real["max_depth_m"] * 1000.0),
            "final_response_n": summarize(real["final_response_n"]),
        },
        "models": rows,
        "pairwise_mask_overlap": overlap_rows,
        "interpretation_boundary": (
            "No physical ground-truth mask was supplied, so these are model predictions and cross-model consistency "
            "checks, not real-world Dice or accuracy measurements."
        ),
    }
    write_json(args.out_dir / "summary.json", payload)
    print(json.dumps({"out": str(args.out_dir), "models": len(rows)}, indent=2), flush=True)


def pairwise_overlap(masks: dict[tuple[str, str], np.ndarray]) -> list[dict[str, Any]]:
    keys = list(masks)
    rows: list[dict[str, Any]] = []
    for index, key_a in enumerate(keys):
        for key_b in keys[index + 1 :]:
            a = masks[key_a].astype(bool)
            b = masks[key_b].astype(bool)
            intersection = int(np.logical_and(a, b).sum())
            union = int(np.logical_or(a, b).sum())
            denom = int(a.sum() + b.sum())
            rows.append(
                {
                    "model_a": f"{key_a[0]}_{key_a[1]}",
                    "model_b": f"{key_b[0]}_{key_b[1]}",
                    "dice": float(2 * intersection / max(denom, 1)),
                    "iou": float(intersection / max(union, 1)),
                }
            )
    return rows


def render_outputs(
    out_dir: Path,
    scores: dict[tuple[str, str], np.ndarray],
    masks: dict[tuple[str, str], np.ndarray],
) -> None:
    train_order = ["V1", "V2", "V1+V2"]
    input_order = ["curve", "stiffness"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    image = None
    for row_index, input_name in enumerate(input_order):
        for col_index, train_name in enumerate(train_order):
            ax = axes[row_index, col_index]
            score = scores[(train_name, input_name)]
            mask = masks[(train_name, input_name)]
            image = ax.imshow(score, origin="lower", cmap="viridis", vmin=0, vmax=1)
            if mask.any() and not mask.all():
                ax.contour(mask, levels=[0.5], colors="white", linewidths=1.1, origin="lower")
            ax.set_title(f"{train_name} / {input_name}\nFG {mask.mean() * 100:.1f}%")
            ax.set_xticks([])
            ax.set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="P(inclusion)")
    fig.savefig(out_dir / "six_unet_probability_contours.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row_index, input_name in enumerate(input_order):
        for col_index, train_name in enumerate(train_order):
            ax = axes[row_index, col_index]
            mask = masks[(train_name, input_name)]
            ax.imshow(mask, origin="lower", cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{train_name} / {input_name}\nFG {mask.mean() * 100:.1f}%")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(out_dir / "six_unet_masks.png", dpi=220)
    plt.close(fig)


def render_inputs(out_dir: Path, response_hwt: np.ndarray, stiffness: np.ndarray, max_depth_m: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    panels = [
        (response_hwt[..., -1], "response at 9.5 mm (N)", "viridis"),
        (stiffness, "final-response stiffness (N/mm)", "magma"),
        (max_depth_m * 1000.0, "measured max depth (mm)", "cividis"),
    ]
    for ax, (value, title, cmap) in zip(axes, panels):
        image = ax.imshow(value, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("grid col / +x")
        ax.set_ylabel("grid row / +y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_dir / "real_scan_model_inputs.png", dpi=220)
    plt.close(fig)


def summarize(value: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    return {
        "shape": list(value.shape),
        "min": float(np.min(flat)),
        "p05": float(np.percentile(flat, 5)),
        "median": float(np.median(flat)),
        "mean": float(np.mean(flat)),
        "p95": float(np.percentile(flat, 95)),
        "max": float(np.max(flat)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
