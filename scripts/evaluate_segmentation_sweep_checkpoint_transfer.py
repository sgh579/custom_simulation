from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_segmentation_accuracy_sweep import (  # noqa: E402
    FEATURE_NAMES,
    PatchMLP,
    PointCNN1D,
    PointGRU,
    PointMLP,
    PointTemporalTransformer,
    ShallowCNN,
    SpatialTokenTransformer,
    TemporalTransformerSpatialHead,
    ValidationUNet,
    _read_json,
    _stiffness,
    counts_from_prediction,
    load_split,
    metrics_from_counts,
    normalize_maps,
    normalized_stiffness_maps,
    preload_subtract_fz,
    predict_neural,
)
from palpation_sim.workflow import require_runtime_environment  # noqa: E402


THRESHOLD_SCORE_METHODS = {
    "stiffness_global_train_threshold",
    "stiffness_otsu_train",
    "stiffness_kmeans_train",
    "stiffness_gmm_train",
}

SKIP_METHODS = {
    "sanity_all_background": "sanity baseline, not a learned model",
    "sanity_all_foreground": "sanity baseline, not a learned model",
    "sanity_train_foreground_prior": "sanity baseline, not a learned model",
    "stiffness_val_oracle_global_threshold": "oracle threshold uses labels",
    "stiffness_val_oracle_per_sample_threshold": "oracle threshold uses labels",
    "stiffness_morphology_val_oracle": "oracle morphology uses labels",
    "stiffness_morphology_train_selected": "morphology operator is not persisted as a reusable model",
    "stiffness_template_shape_prior": "template prior parameters are not persisted as a reusable model",
    "logistic_mechanical": "sklearn estimator was not persisted",
    "logistic_mechanical_xy": "sklearn estimator was not persisted",
    "random_forest_mechanical": "sklearn estimator was not persisted",
    "random_forest_mechanical_xy": "sklearn estimator was not persisted",
    "hist_gradient_boosting_mechanical": "sklearn estimator was not persisted",
    "hist_gradient_boosting_mechanical_xy": "sklearn estimator was not persisted",
    "rimon_gru_reconstruction_decoder": "requires paired frozen encoder+decoder evaluation; handled separately if needed",
    "rimon_gru_reconstruction_pretrain": "pretraining checkpoint, not a segmentation model",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reusable checkpoints from a segmentation sweep on a target test split."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--source-sweep-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--resample-fz-steps",
        type=int,
        default=None,
        help="Resample target Fz curves to this many depth samples before evaluating old full-curve checkpoints.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    require_runtime_environment()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = args.train_dir or args.data_dir / "train"
    test_dir = args.test_dir or args.data_dir / "test"
    print(f"loading train split: {train_dir}", flush=True)
    train = load_split(train_dir)
    print(f"loading test split: {test_dir}", flush=True)
    test = load_split(test_dir)
    original_fz_steps = int(train.fz.shape[1])
    if args.resample_fz_steps is not None:
        train = replace_fz(train, resample_fz_steps(train.fz, int(args.resample_fz_steps)))
        test = replace_fz(test, resample_fz_steps(test.fz, int(args.resample_fz_steps)))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    source_metrics = sorted(args.source_sweep_dir.glob("*/metrics_summary.json"))
    for metrics_path in source_metrics:
        method = metrics_path.parent.name
        if method in SKIP_METHODS:
            skipped.append({"method": method, "reason": SKIP_METHODS[method]})
            continue
        method_dir = args.out_dir / method
        if method_dir.joinpath("metrics_summary.json").exists() and not args.force:
            rows.append(_read_json(method_dir / "leaderboard_row.json"))
            print(f"skip existing {method}", flush=True)
            continue
        source_summary = _read_json(metrics_path)
        try:
            if method in THRESHOLD_SCORE_METHODS:
                scores = _stiffness(test).astype(np.float32)
                model_source = "source fixed threshold"
            else:
                checkpoint = metrics_path.parent / "best.pt"
                if not checkpoint.exists():
                    skipped.append({"method": method, "reason": "no reusable checkpoint found"})
                    continue
                x_train, x_test = inputs_for_method(method, train, test)
                model = build_model(method, input_channels=int(x_train.shape[1]), spatial_shape=train.masks.shape[-2:])
                scores = predict_checkpoint(model, checkpoint, x=x_test, device=device, batch_size=args.batch_size)
                model_source = str(checkpoint)
        except Exception as exc:
            skipped.append({"method": method, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"skip {method}: {type(exc).__name__}: {exc}", flush=True)
            continue

        row = write_method_result(
            method_dir,
            method=method,
            source_summary=source_summary,
            scores=scores,
            masks=test.masks,
            names=test.names,
            model_source=model_source,
            config={
                "target_train_dir": str(train_dir),
                "target_test_dir": str(test_dir),
                "source_sweep_dir": str(args.source_sweep_dir),
                "target_train_samples": len(train.names),
                "target_test_samples": len(test.names),
                "target_original_fz_steps": original_fz_steps,
                "target_evaluated_fz_steps": int(train.fz.shape[1]),
                "resample_fz_steps": args.resample_fz_steps,
                "threshold_policy": (
                    "primary uses the threshold stored in the source sweep metrics; "
                    "test_oracle_best is reported only as a diagnostic"
                ),
                "normalization_note": (
                    "Dataset-normalized inputs use target train-split statistics because the old sweep did not persist "
                    "its normalization tensors."
                ),
            },
        )
        rows.append(row)
        print(f"{method}: primary_dice={float(row['primary_dice']):.4f}", flush=True)

    rows.sort(key=lambda row: float(row["primary_dice"]), reverse=True)
    if rows:
        write_csv(args.out_dir / "leaderboard.csv", rows, list(rows[0].keys()))
    write_csv(args.out_dir / "skipped_methods.csv", skipped, ["method", "reason"])
    write_report(args.out_dir, rows, skipped)
    print(f"evaluation complete: {args.out_dir}", flush=True)


def replace_fz(split: Any, fz: np.ndarray) -> Any:
    return type(split)(
        names=split.names,
        fz=fz.astype(np.float32),
        masks=split.masks,
        features=split.features,
        xy=split.xy,
    )


def resample_fz_steps(fz: np.ndarray, steps: int) -> np.ndarray:
    fz = np.asarray(fz, dtype=np.float32)
    current = int(fz.shape[1])
    if current == int(steps):
        return fz
    source = np.linspace(0.0, 1.0, current, dtype=np.float32)
    target = np.linspace(0.0, 1.0, int(steps), dtype=np.float32)
    moved = np.moveaxis(fz, 1, -1).reshape(-1, current)
    out = np.empty((moved.shape[0], int(steps)), dtype=np.float32)
    for idx, curve in enumerate(moved):
        out[idx] = np.interp(target, source, curve).astype(np.float32)
    return np.moveaxis(out.reshape(fz.shape[0], fz.shape[2], fz.shape[3], int(steps)), -1, 1)


def inputs_for_method(method: str, train: Any, test: Any) -> tuple[np.ndarray, np.ndarray]:
    if method in {
        "pointwise_mlp_stiffness",
        "patchwise_mlp_stiffness_p3",
        "patchwise_mlp_stiffness_p5",
        "shallow_cnn_stiffness",
        "unet_stiffness",
        "unet_stiffness_aug_focal",
    }:
        return normalized_stiffness_maps(train, test)

    if method in {
        "pixel_mlp_rerun",
        "point_transformer_sample",
        "point_1d_cnn",
        "point_gru",
        "patch_mlp_fz_p3",
        "patch_mlp_fz_p5",
        "shallow_cnn_fz",
    }:
        return normalize_maps(train.fz, test.fz, mode="sample")

    if method in {
        "point_transformer_dataset",
        "temporal_transformer_spatial_head",
        "spatial_token_transformer_fz",
        "unet_fz_norm_dataset",
    }:
        return normalize_maps(train.fz, test.fz, mode="dataset")

    if method == "unet_delta_fz_norm_dataset":
        return normalize_maps(preload_subtract_fz(train.fz), preload_subtract_fz(test.fz), mode="dataset")

    if method == "unet_fz_norm_none":
        return normalize_maps(train.fz, test.fz, mode="none")

    if method in {"spatial_token_transformer_fz_features", "unet_fz_features_dataset_norm", "unet_fz_features_aug_focal"}:
        fz_train, fz_test = normalize_maps(train.fz, test.fz, mode="dataset")
        feat_train, feat_test = normalize_maps(train.features, test.features, mode="dataset")
        return np.concatenate([fz_train, feat_train], axis=1), np.concatenate([fz_test, feat_test], axis=1)

    raise ValueError(f"Unsupported checkpoint method: {method}")


def build_model(method: str, *, input_channels: int, spatial_shape: tuple[int, int]) -> nn.Module:
    if method in {"pointwise_mlp_stiffness", "pixel_mlp_rerun"}:
        return PointMLP(input_channels, hidden_dims=(128, 64))
    if method in {"patchwise_mlp_stiffness_p3", "patch_mlp_fz_p3"}:
        return PatchMLP(input_channels, patch_size=3, hidden_dims=(512, 128))
    if method in {"patchwise_mlp_stiffness_p5", "patch_mlp_fz_p5"}:
        return PatchMLP(input_channels, patch_size=5, hidden_dims=(512, 128))
    if method in {"shallow_cnn_stiffness", "shallow_cnn_fz"}:
        return ShallowCNN(input_channels, base_channels=32)
    if method in {
        "unet_stiffness",
        "unet_stiffness_aug_focal",
        "unet_fz_norm_none",
        "unet_fz_norm_dataset",
        "unet_delta_fz_norm_dataset",
        "unet_fz_features_dataset_norm",
        "unet_fz_features_aug_focal",
    }:
        return ValidationUNet(input_channels, base_channels=24)
    if method in {"point_transformer_sample", "point_transformer_dataset"}:
        return PointTemporalTransformer(input_channels, d_model=48, nhead=4, num_layers=2)
    if method == "temporal_transformer_spatial_head":
        return TemporalTransformerSpatialHead(input_channels, d_model=32, nhead=4, num_layers=2)
    if method in {"spatial_token_transformer_fz", "spatial_token_transformer_fz_features"}:
        return SpatialTokenTransformer(input_channels, spatial_shape=spatial_shape, d_model=64, nhead=4, num_layers=2)
    if method == "point_1d_cnn":
        return PointCNN1D(input_channels)
    if method == "point_gru":
        return PointGRU()
    raise ValueError(f"No model builder for method: {method}")


def predict_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    x: np.ndarray | None = None,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if x is None:
        raise ValueError("x is required")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    return predict_neural(model, x, device, batch_size=batch_size)


def write_method_result(
    method_dir: Path,
    *,
    method: str,
    source_summary: dict[str, Any],
    scores: np.ndarray,
    masks: np.ndarray,
    names: list[str],
    model_source: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    method_dir.mkdir(parents=True, exist_ok=True)
    source_fixed = source_summary.get("fixed_threshold", {})
    source_best = source_summary.get("threshold_sweep_best", {})
    fixed_threshold = float(source_fixed.get("threshold", 0.5))
    selected_threshold = float(source_best.get("threshold", fixed_threshold))
    if method in THRESHOLD_SCORE_METHODS:
        selected_threshold = fixed_threshold
        threshold_source = "source fixed/train threshold"
    else:
        threshold_source = "source validation-selected threshold"

    fixed = aggregate_metrics(scores, masks, fixed_threshold)
    selected = aggregate_metrics(scores, masks, selected_threshold)
    oracle = oracle_best_metrics(scores, masks)
    per_sample_rows = []
    selected_pred = scores >= selected_threshold
    fixed_pred = scores >= fixed_threshold
    for idx, name in enumerate(names):
        fixed_sample = metrics_from_counts(counts_from_prediction(fixed_pred[idx], masks[idx]), threshold=fixed_threshold)
        selected_sample = metrics_from_counts(counts_from_prediction(selected_pred[idx], masks[idx]), threshold=selected_threshold)
        per_sample_rows.append(
            {
                "sample": name,
                "fixed_dice": fixed_sample["dice"],
                "source_selected_dice": selected_sample["dice"],
                "gt_positive": int(masks[idx].sum()),
            }
        )

    summary = {
        "method": method,
        "group": source_summary.get("group", ""),
        "model_source": model_source,
        "fixed_threshold": fixed,
        "source_selected_threshold": selected,
        "source_threshold": selected_threshold,
        "source_threshold_source": threshold_source,
        "test_oracle_best": oracle,
        "num_samples": int(masks.shape[0]),
        "source_validation_summary": source_summary,
        "config": config,
    }
    np.save(method_dir / "test_scores.npy", np.asarray(scores, dtype=np.float32))
    (method_dir / "metrics_summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    write_csv(method_dir / "metrics_per_sample.csv", per_sample_rows, list(per_sample_rows[0].keys()))
    row = {
        "method": method,
        "group": source_summary.get("group", ""),
        "primary_dice": selected["dice"],
        "primary_iou": selected["iou"],
        "primary_precision": selected["precision"],
        "primary_recall": selected["recall"],
        "primary_threshold": selected_threshold,
        "primary_threshold_source": threshold_source,
        "fixed_dice": fixed["dice"],
        "fixed_iou": fixed["iou"],
        "test_oracle_best_dice": oracle["dice"],
        "test_oracle_best_threshold": oracle["threshold"],
        "source_val_best_dice": source_best.get("dice"),
        "source_val_fixed_dice": source_fixed.get("dice"),
        "path": str(method_dir),
    }
    (method_dir / "leaderboard_row.json").write_text(json.dumps(json_ready(row), indent=2), encoding="utf-8")
    return row


def aggregate_metrics(scores: np.ndarray, masks: np.ndarray, threshold: float) -> dict[str, float | int]:
    return metrics_from_counts(counts_from_prediction(np.asarray(scores) >= float(threshold), masks), threshold=float(threshold))


def oracle_best_metrics(scores: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float32)
    if float(np.nanmin(scores)) >= -1e-6 and float(np.nanmax(scores)) <= 1.0 + 1e-6:
        thresholds = np.asarray([round(v, 6) for v in np.arange(0.05, 0.951, 0.05)], dtype=np.float64)
    else:
        flat = scores.reshape(-1)
        thresholds = np.unique(np.quantile(flat, np.linspace(0.0, 1.0, 240))).astype(np.float64)
    rows = [aggregate_metrics(scores, masks, float(threshold)) for threshold in thresholds]
    return max(rows, key=lambda row: float(row["dice"]))


def write_report(out_dir: Path, rows: list[dict[str, Any]], skipped: list[dict[str, str]]) -> None:
    lines = [
        "# Checkpoint Transfer Test Results",
        "",
        "| rank | method | group | primary Dice | primary IoU | fixed Dice | test-oracle Dice | source val Dice |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | `{row['method']}` | {row['group']} | {fmt(row['primary_dice'])} | "
            f"{fmt(row['primary_iou'])} | {fmt(row['fixed_dice'])} | "
            f"{fmt(row['test_oracle_best_dice'])} | {fmt(row['source_val_best_dice'])} |"
        )
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for row in skipped:
            lines.append(f"- `{row['method']}`: {row['reason']}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
