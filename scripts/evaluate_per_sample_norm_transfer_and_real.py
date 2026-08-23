from __future__ import annotations

import csv
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

from run_highres_per_sample_norm_unet import (  # noqa: E402
    curve_force_response_scale_per_sample,
    stiffness_log_robust_per_sample,
)
from run_highres_segmentation_sweep import UpsampleLogits, load_split  # noqa: E402
from run_segmentation_accuracy_sweep import FEATURE_NAMES, ValidationUNet, predict_neural  # noqa: E402


def main() -> None:
    root = PROJECT_ROOT
    out = root / "runs" / "real_ur_grid_scan_20260630_six_unet_outputs_per_sample_norm_20260709"
    out.mkdir(parents=True, exist_ok=True)

    real_src = (
        root
        / "runs"
        / "per_sample_norm_unet_128_strict_20260624_fixed_depth"
        / "real_inference_20260630-155255-691171_full20_original_boundary"
    )

    models = {
        "V1": {
            "curve": root
            / "runs/per_sample_norm_unet_128_strict_20260624_fixed_depth/r128_fixed_depth_unet_curve_force_response_scale_per_sample",
            "stiffness": root
            / "runs/per_sample_norm_unet_128_strict_20260624_fixed_depth/r128_fixed_depth_unet_stiffness_log_robust_per_sample",
        },
        "V2": {
            "curve": root / "runs/per_sample_norm_unet_128_v2_halfdepth_20260709/r128_unet_curve_force_response_scale_per_sample",
            "stiffness": root / "runs/per_sample_norm_unet_128_v2_halfdepth_20260709/r128_unet_stiffness_log_robust_per_sample",
        },
        "V1+V2": {
            "curve": root
            / "runs/per_sample_norm_unet_128_v1fixed_v2ref55k_halfdepth_20260709/r128_unet_curve_force_response_scale_per_sample",
            "stiffness": root
            / "runs/per_sample_norm_unet_128_v1fixed_v2ref55k_halfdepth_20260709/r128_unet_stiffness_log_robust_per_sample",
        },
    }

    test_dirs = {
        "V1": root / "data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/fixed_depth/data/test",
        "V2": root / "tmp/v2_delta_fz_highres_package/data/test",
    }

    print("loading test splits", flush=True)
    test_splits = {name: load_split(path, label_size=128, depth_steps=20) for name, path in test_dirs.items()}
    test_inputs: dict[tuple[str, str], np.ndarray] = {}
    for name, split in test_splits.items():
        test_inputs[(name, "curve")] = curve_force_response_scale_per_sample(split.fz)
        stiffness = split.features[:, FEATURE_NAMES.index("equivalent_stiffness")][:, None]
        test_inputs[(name, "stiffness")] = stiffness_log_robust_per_sample(stiffness)

    real_inputs = {
        "curve": np.load(real_src / "today_curve_force_response_scale_chw.npy").astype(np.float32)[None],
        "stiffness": np.load(real_src / "today_stiffness_input_log_robust_chw.npy").astype(np.float32)[None],
    }
    if real_inputs["curve"].shape[1:] != (20, 20, 20):
        raise RuntimeError(f"unexpected real curve shape {real_inputs['curve'].shape}")
    if real_inputs["stiffness"].shape[1:] != (1, 20, 20):
        raise RuntimeError(f"unexpected real stiffness shape {real_inputs['stiffness'].shape}")

    real_input_summary = {key: summarize_array(value) for key, value in real_inputs.items()}
    write_json(out / "real_input_summary.json", real_input_summary)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    cross_rows: list[dict[str, Any]] = []
    real_rows: list[dict[str, Any]] = []
    real_scores: dict[tuple[str, str], np.ndarray] = {}
    real_masks: dict[tuple[str, str], np.ndarray] = {}
    model_info: dict[str, Any] = {}

    for train_name, input_to_dir in models.items():
        for input_name, method_dir in input_to_dir.items():
            threshold = threshold_for(method_dir)
            input_channels = int(real_inputs[input_name].shape[1])
            model, checkpoint, base_channels = build_model(method_dir, input_channels, device)
            model_info[f"{train_name}_{input_name}"] = {
                "method_dir": str(method_dir),
                "threshold": threshold,
                "base_channels": base_channels,
                "checkpoint_best_epoch": checkpoint.get("epoch"),
                "checkpoint_best_val": checkpoint.get("best_val"),
            }

            for test_name, split in test_splits.items():
                scores = predict_neural(model, test_inputs[(test_name, input_name)], device, batch_size=16)
                row_metrics = metrics(scores, split.masks, threshold)
                cross_rows.append(
                    {
                        "input": input_name,
                        "train_data": train_name,
                        "test_data": test_name,
                        "threshold": threshold,
                        "dice": row_metrics["dice"],
                        "iou": row_metrics["iou"],
                        "pred_positive_fraction": row_metrics["pred_positive_fraction"],
                        "gt_positive_fraction": row_metrics["gt_positive_fraction"],
                        "tp": row_metrics["tp"],
                        "fp": row_metrics["fp"],
                        "fn": row_metrics["fn"],
                        "tn": row_metrics["tn"],
                        "method_dir": str(method_dir),
                    }
                )

            score = predict_neural(model, real_inputs[input_name], device, batch_size=1)[0]
            mask = score >= threshold
            real_scores[(train_name, input_name)] = score.astype(np.float32)
            real_masks[(train_name, input_name)] = mask.astype(np.uint8)
            stem_train = train_name.replace("+", "plus").lower()
            np.save(out / f"real_{stem_train}_{input_name}_probability_128x128.npy", score.astype(np.float32))
            np.save(out / f"real_{stem_train}_{input_name}_mask_128x128.npy", mask.astype(np.uint8))
            real_rows.append(
                {
                    "input": input_name,
                    "train_data": train_name,
                    "threshold": threshold,
                    "mask_positive_fraction": float(mask.mean()),
                    "mask_positive_pixels": int(mask.sum()),
                    "prob_min": float(np.nanmin(score)),
                    "prob_p05": float(np.nanpercentile(score, 5)),
                    "prob_median": float(np.nanmedian(score)),
                    "prob_mean": float(np.nanmean(score)),
                    "prob_p95": float(np.nanpercentile(score, 95)),
                    "prob_max": float(np.nanmax(score)),
                    "method_dir": str(method_dir),
                }
            )
            print(f"done {train_name} {input_name}", flush=True)

    write_csv(out / "per_sample_norm_transfer_test_metrics.csv", cross_rows)
    write_csv(out / "real_phantom_six_outputs_summary.csv", real_rows)

    write_json(
        out / "summary.json",
        {
            "normalization": {
                "curve": (
                    "real reuses old V1 input: positive force response divided by this real scan p95; "
                    "synthetic eval uses response=max(fz-fz_at_first_depth,0)/sample_p95"
                ),
                "stiffness": (
                    "real reuses old V1 input: log1p(N/m) robust normalized by this real scan median/IQR; "
                    "synthetic eval uses per-sample median/IQR"
                ),
            },
            "real_source": str(real_src),
            "output_dir": str(out),
            "model_info": model_info,
            "real_input_summary": real_input_summary,
            "cross_rows": cross_rows,
            "real_rows": real_rows,
        },
    )

    write_real_figures(out, real_scores, real_masks)
    write_transfer_heatmap(out, cross_rows)
    print(json.dumps({"out": str(out)}, indent=2), flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def threshold_for(method_dir: Path) -> float:
    return float(read_json(method_dir / "metrics_summary.json")["threshold_sweep_best"]["threshold"])


def infer_base_channels(state: dict[str, torch.Tensor]) -> int:
    for key, value in state.items():
        if key.endswith("weight") and value.ndim == 4:
            return int(value.shape[0])
    return 24


def build_model(method_dir: Path, input_channels: int, device: torch.device):
    checkpoint = torch.load(method_dir / "best.pt", map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    base_channels = infer_base_channels(state)
    model = UpsampleLogits(ValidationUNet(input_channels, base_channels=base_channels), (128, 128))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, checkpoint, base_channels


def metrics(scores: np.ndarray, masks: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = np.asarray(scores >= threshold, dtype=bool)
    gt = np.asarray(masks > 0, dtype=bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_positive_fraction": float(pred.mean()),
        "gt_positive_fraction": float(gt.mean()),
    }


def summarize_array(x: np.ndarray) -> dict[str, float | list[int]]:
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    return {
        "shape": list(x.shape),
        "min": float(np.nanmin(flat)),
        "p05": float(np.nanpercentile(flat, 5)),
        "median": float(np.nanmedian(flat)),
        "mean": float(np.nanmean(flat)),
        "p95": float(np.nanpercentile(flat, 95)),
        "max": float(np.nanmax(flat)),
    }


def write_real_figures(
    out: Path,
    real_scores: dict[tuple[str, str], np.ndarray],
    real_masks: dict[tuple[str, str], np.ndarray],
) -> None:
    train_order = ["V1", "V2", "V1+V2"]
    input_order = ["curve", "stiffness"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    last_image = None
    for row_idx, input_name in enumerate(input_order):
        for col_idx, train_name in enumerate(train_order):
            ax = axes[row_idx, col_idx]
            score = real_scores[(train_name, input_name)]
            mask = real_masks[(train_name, input_name)]
            last_image = ax.imshow(score, origin="lower", cmap="viridis", vmin=0, vmax=1)
            ax.contour(mask, levels=[0.5], colors="white", linewidths=1.1, origin="lower")
            ax.set_title(f"{train_name} / {input_name}\nFG {mask.mean() * 100:.1f}%")
            ax.set_xticks([])
            ax.set_yticks([])
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="P(lump)")
    fig.savefig(out / "real_phantom_six_unet_score_contours.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row_idx, input_name in enumerate(input_order):
        for col_idx, train_name in enumerate(train_order):
            ax = axes[row_idx, col_idx]
            mask = real_masks[(train_name, input_name)]
            ax.imshow(mask, origin="lower", cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{train_name} / {input_name}\nFG {mask.mean() * 100:.1f}%")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(out / "real_phantom_six_unet_masks.png", dpi=220)
    plt.close(fig)


def write_transfer_heatmap(out: Path, rows: list[dict[str, Any]]) -> None:
    train_order = ["V1", "V2", "V1+V2"]
    test_order = ["V1", "V2"]
    input_order = ["curve", "stiffness"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True, constrained_layout=True)
    last_image = None
    for ax, input_name in zip(axes, input_order):
        values = np.asarray(
            [
                [
                    next(
                        row["dice"]
                        for row in rows
                        if row["input"] == input_name and row["train_data"] == train_name and row["test_data"] == test_name
                    )
                    for test_name in test_order
                ]
                for train_name in train_order
            ],
            dtype=np.float32,
        )
        last_image = ax.imshow(values, vmin=0, vmax=1, cmap="magma")
        ax.set_title(input_name)
        ax.set_xticks([0, 1], ["V1 test", "V2 test"])
        ax.set_yticks([0, 1, 2], train_order)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if values[i, j] < 0.65 else "black",
                    fontsize=10,
                )
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03, label="Dice")
    fig.savefig(out / "per_sample_norm_transfer_test_dice_heatmap.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
