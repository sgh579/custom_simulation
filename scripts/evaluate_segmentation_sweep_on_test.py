from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_segmentation_accuracy_sweep import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    PointMLP,
    ShallowCNN,
    ValidationUNet,
    _stiffness,
    _write_csv,
    _write_json,
    counts_from_prediction,
    load_split,
    metrics_from_counts,
    normalize_maps,
    normalized_stiffness_maps,
    predict_neural,
)
from palpation_sim.workflow import require_runtime_environment  # noqa: E402


DEFAULT_TEST_DIR = DEFAULT_DATA_DIR / "test"
DEFAULT_TEST_OUT_DIR = Path("runs/segmentation_accuracy_sweep_20x_4seed_test_seed71_80")

METHOD_SPECS = [
    {
        "method": "stiffness_kmeans_train",
        "requested_label": "1. stiffness mapping + KMeans",
        "input_family": "stiffness_map",
        "model_family": "kmeans_threshold",
    },
    {
        "method": "pointwise_mlp_stiffness",
        "requested_label": "2.1 stiffness map point MLP",
        "input_family": "stiffness_map",
        "model_family": "point_mlp",
    },
    {
        "method": "shallow_cnn_stiffness",
        "requested_label": "2.2 stiffness map shallow CNN",
        "input_family": "stiffness_map",
        "model_family": "shallow_cnn",
    },
    {
        "method": "unet_stiffness",
        "requested_label": "2.3 stiffness map U-Net",
        "input_family": "stiffness_map",
        "model_family": "unet",
    },
    {
        "method": "pixel_mlp_rerun",
        "requested_label": "3.1 full curve point MLP",
        "input_family": "full_curve_raw_fz",
        "model_family": "point_mlp",
    },
    {
        "method": "shallow_cnn_fz",
        "requested_label": "3.2 full curve shallow CNN",
        "input_family": "full_curve_raw_fz",
        "model_family": "shallow_cnn",
    },
    {
        "method": "unet_fz_norm_dataset",
        "requested_label": "3.3 full curve U-Net",
        "input_family": "full_curve_raw_fz",
        "model_family": "unet",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selected segmentation sweep methods on a held-out test split.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None, help="Validation split used only for provenance in run_config.")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TEST_OUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    require_runtime_environment()
    train_dir = args.train_dir or args.data_dir / "train"
    val_dir = args.val_dir or args.data_dir / "val"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print(f"loading train split: {train_dir}", flush=True)
    train = load_split(train_dir)
    print(f"loading test split: {args.test_dir}", flush=True)
    test = load_split(args.test_dir)
    context = {
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "test_dir": str(args.test_dir),
        "sweep_dir": str(args.sweep_dir),
        "train_samples": len(train.names),
        "test_samples": len(test.names),
        "shape": list(test.masks.shape[-2:]),
        "threshold_policy": "fixed thresholds and validation-selected thresholds only; no test-label threshold tuning",
        "requested_methods": METHOD_SPECS,
    }
    _write_json(args.out_dir / "run_config.json", context)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rows: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        method = str(spec["method"])
        method_dir = args.out_dir / method
        if method_dir.joinpath("metrics_summary.json").exists() and not args.force:
            rows.append(_load_row(method_dir / "leaderboard_row.json"))
            print(f"skip existing {method}", flush=True)
            continue
        print(f"evaluating {method}", flush=True)
        scores = scores_for_method(method, train, test, args.sweep_dir, device=device, batch_size=args.batch_size)
        val_summary = _load_json(args.sweep_dir / method / "metrics_summary.json")
        fixed_threshold = float(val_summary.get("fixed_threshold", {}).get("threshold", 0.5))
        val_best_threshold = float(val_summary.get("threshold_sweep_best", {}).get("threshold", fixed_threshold))
        row = write_test_method(
            method_dir,
            method=method,
            spec=spec,
            scores=scores,
            masks=test.masks,
            names=test.names,
            fixed_threshold=fixed_threshold,
            val_best_threshold=val_best_threshold,
            validation_summary=val_summary,
            config=context,
        )
        rows.append(row)

    rows.sort(key=lambda row: float(row["test_primary_dice"]), reverse=True)
    _write_csv(args.out_dir / "leaderboard.csv", rows, list(rows[0].keys()))
    write_report(args.out_dir, rows, elapsed_seconds=time.perf_counter() - started)
    print(f"test evaluation complete: {args.out_dir}", flush=True)


def scores_for_method(
    method: str,
    train: Any,
    test: Any,
    sweep_dir: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if method == "stiffness_kmeans_train":
        return _stiffness(test).astype(np.float32)
    if method in {"pointwise_mlp_stiffness", "shallow_cnn_stiffness", "unet_stiffness"}:
        x_train, x_test = normalized_stiffness_maps(train, test)
        model = build_model(method, input_channels=x_train.shape[1], spatial_shape=train.masks.shape[-2:])
        return predict_checkpoint(model, sweep_dir / method / "best.pt", x_test, device=device, batch_size=batch_size)
    if method in {"pixel_mlp_rerun", "shallow_cnn_fz"}:
        x_train, x_test = normalize_maps(train.fz, test.fz, mode="sample")
        model = build_model(method, input_channels=x_train.shape[1], spatial_shape=train.masks.shape[-2:])
        return predict_checkpoint(model, sweep_dir / method / "best.pt", x_test, device=device, batch_size=batch_size)
    if method == "unet_fz_norm_dataset":
        x_train, x_test = normalize_maps(train.fz, test.fz, mode="dataset")
        model = build_model(method, input_channels=x_train.shape[1], spatial_shape=train.masks.shape[-2:])
        return predict_checkpoint(model, sweep_dir / method / "best.pt", x_test, device=device, batch_size=batch_size)
    raise ValueError(f"Unsupported method: {method}")


def build_model(method: str, *, input_channels: int, spatial_shape: tuple[int, int]) -> nn.Module:
    del spatial_shape
    if method in {"pointwise_mlp_stiffness", "pixel_mlp_rerun"}:
        return PointMLP(input_channels, hidden_dims=(128, 64))
    if method in {"shallow_cnn_stiffness", "shallow_cnn_fz"}:
        return ShallowCNN(input_channels, base_channels=32)
    if method in {"unet_stiffness", "unet_fz_norm_dataset"}:
        return ValidationUNet(input_channels, base_channels=24)
    raise ValueError(f"No model builder for method: {method}")


def predict_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    return predict_neural(model, x, device, batch_size=batch_size)


def write_test_method(
    method_dir: Path,
    *,
    method: str,
    spec: dict[str, Any],
    scores: np.ndarray,
    masks: np.ndarray,
    names: list[str],
    fixed_threshold: float,
    val_best_threshold: float,
    validation_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    method_dir.mkdir(parents=True, exist_ok=True)
    fixed = aggregate_metrics(scores, masks, fixed_threshold)
    val_selected = aggregate_metrics(scores, masks, val_best_threshold)
    primary_key = "fixed" if method == "stiffness_kmeans_train" else "val_selected"
    primary = fixed if primary_key == "fixed" else val_selected
    primary_threshold_source = (
        "train KMeans centers" if method == "stiffness_kmeans_train" else "validation-selected threshold"
    )
    per_sample_rows = []
    for idx, name in enumerate(names):
        fixed_sample = aggregate_metrics(scores[idx : idx + 1], masks[idx : idx + 1], fixed_threshold)
        val_sample = aggregate_metrics(scores[idx : idx + 1], masks[idx : idx + 1], val_best_threshold)
        per_sample_rows.append(
            {
                "sample": name,
                "fixed_dice": fixed_sample["dice"],
                "fixed_iou": fixed_sample["iou"],
                "val_selected_dice": val_sample["dice"],
                "val_selected_iou": val_sample["iou"],
                "gt_positive": int(masks[idx].sum()),
            }
        )

    summary = {
        "method": method,
        "requested_label": spec["requested_label"],
        "input_family": spec["input_family"],
        "model_family": spec["model_family"],
        "fixed_threshold": fixed,
        "val_selected_threshold": val_selected,
        "primary_threshold": primary,
        "primary_threshold_source": primary_threshold_source,
        "num_samples": int(masks.shape[0]),
        "validation_summary_source": validation_summary,
        "config": config,
    }
    np.save(method_dir / "test_scores.npy", np.asarray(scores, dtype=np.float32))
    _write_json(method_dir / "metrics_summary.json", summary)
    _write_csv(method_dir / "metrics_per_sample.csv", per_sample_rows, list(per_sample_rows[0].keys()))
    row = {
        "method": method,
        "requested_label": spec["requested_label"],
        "input_family": spec["input_family"],
        "model_family": spec["model_family"],
        "test_primary_dice": primary["dice"],
        "test_primary_iou": primary["iou"],
        "test_primary_precision": primary["precision"],
        "test_primary_recall": primary["recall"],
        "test_primary_threshold": primary["threshold"],
        "primary_threshold_source": primary_threshold_source,
        "test_fixed_dice": fixed["dice"],
        "test_fixed_iou": fixed["iou"],
        "test_fixed_precision": fixed["precision"],
        "test_fixed_recall": fixed["recall"],
        "test_fixed_threshold": fixed["threshold"],
        "test_val_selected_dice": val_selected["dice"],
        "test_val_selected_iou": val_selected["iou"],
        "test_val_selected_precision": val_selected["precision"],
        "test_val_selected_recall": val_selected["recall"],
        "test_val_selected_threshold": val_selected["threshold"],
        "validation_fixed_dice": validation_summary.get("fixed_threshold", {}).get("dice"),
        "validation_best_dice": validation_summary.get("threshold_sweep_best", {}).get("dice"),
        "validation_best_threshold": validation_summary.get("threshold_sweep_best", {}).get("threshold"),
        "path": str(method_dir),
    }
    _write_json(method_dir / "leaderboard_row.json", row)
    return row


def aggregate_metrics(scores: np.ndarray, masks: np.ndarray, threshold: float) -> dict[str, float | int]:
    return metrics_from_counts(counts_from_prediction(np.asarray(scores) >= threshold, masks), threshold=threshold)


def write_report(out_dir: Path, rows: list[dict[str, Any]], *, elapsed_seconds: float) -> None:
    lines = [
        "# Held-Out Test Segmentation Results",
        "",
        f"- Methods: {len(rows)}",
        f"- Elapsed seconds: {elapsed_seconds:.2f}",
        "- Threshold policy: fixed threshold plus validation-selected threshold; test labels were not used to select thresholds.",
        "",
        "| rank | method | input | primary source | primary Dice | primary IoU | fixed/KMeans Dice | val-threshold Dice | val Dice |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | `{row['method']}` | {row['input_family']} | "
            f"{row['primary_threshold_source']} | {_fmt(row['test_primary_dice'])} | {_fmt(row['test_primary_iou'])} | "
            f"{_fmt(row['test_fixed_dice'])} | {_fmt(row['test_val_selected_dice'])} | "
            f"{_fmt(row['validation_best_dice'])} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_row(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    if not data:
        raise FileNotFoundError(path)
    return data


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
