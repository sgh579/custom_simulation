from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_highres_segmentation_sweep import UpsampleLogits, load_split  # noqa: E402
from run_segmentation_accuracy_sweep import (  # noqa: E402
    ValidationUNet,
    _load_fz_hwt,
    counts_from_prediction,
    evaluate_scores,
    metrics_from_counts,
    predict_neural,
    preload_subtract_fz,
)
from palpation_sim.workflow import require_runtime_environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a high-res delta-Fz U-Net checkpoint on a transfer split.")
    parser.add_argument("--source-train-dir", type=Path, required=True)
    parser.add_argument("--target-train-dir", type=Path, required=True)
    parser.add_argument("--target-test-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--input-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-source-train", type=int, default=None)
    parser.add_argument("--max-target-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    require_runtime_environment()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_summary = read_json(args.source_metrics)
    source_threshold = float(source_summary.get("threshold_sweep_best", {}).get("threshold", 0.5))
    fixed_threshold = float(source_summary.get("fixed_threshold", {}).get("threshold", 0.5))

    print(f"loading source train Fz: {args.source_train_dir}", flush=True)
    source_train_fz = resample_fz_steps(load_fz_split(args.source_train_dir, max_samples=args.max_source_train), args.input_steps)
    source_train_delta = preload_subtract_fz(source_train_fz)
    source_mean, source_std = dataset_stats(source_train_delta)

    print(f"loading target train Fz: {args.target_train_dir}", flush=True)
    target_train_fz = resample_fz_steps(load_fz_split(args.target_train_dir, max_samples=args.max_target_train), args.input_steps)
    target_train_delta = preload_subtract_fz(target_train_fz)
    target_mean, target_std = dataset_stats(target_train_delta)

    print(f"loading target test labels/Fz: {args.target_test_dir}", flush=True)
    target_test = load_split(
        args.target_test_dir,
        label_size=int(args.resolution),
        max_samples=args.max_test,
        random_pair_seed=7,
    )
    original_steps = int(target_test.fz.shape[1])
    target_test_delta = preload_subtract_fz(resample_fz_steps(target_test.fz, args.input_steps))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = UpsampleLogits(ValidationUNet(int(args.input_steps), base_channels=24), (int(args.resolution), int(args.resolution)))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    context = {
        "source_train_dir": str(args.source_train_dir),
        "target_train_dir": str(args.target_train_dir),
        "target_test_dir": str(args.target_test_dir),
        "checkpoint": str(args.checkpoint),
        "source_metrics": str(args.source_metrics),
        "source_threshold": source_threshold,
        "fixed_threshold": fixed_threshold,
        "target_original_fz_steps": original_steps,
        "target_evaluated_fz_steps": int(args.input_steps),
        "force_preprocess": "resample_to_input_steps_then_subtract_initial_fz_per_curve",
        "normalization_variants": [
            "source_train_delta_fz_stats",
            "target_train_delta_fz_stats",
        ],
        "resolution": int(args.resolution),
        "elapsed_seconds": None,
    }
    write_json(args.out_dir / "run_config.json", context)

    rows = []
    variants = [
        ("source_norm", source_mean, source_std, "source V2 train delta-Fz statistics"),
        ("target_norm", target_mean, target_std, "target V1 train delta-Fz statistics"),
    ]
    for variant, mean, std, policy in variants:
        method = f"r{int(args.resolution)}_delta_fz_unet_{variant}"
        method_dir = args.out_dir / method
        if method_dir.joinpath("metrics_summary.json").exists() and not args.force:
            rows.append(read_json(method_dir / "leaderboard_row.json"))
            print(f"skip existing {method}", flush=True)
            continue
        print(f"predicting {method}", flush=True)
        x_test = apply_stats(target_test_delta, mean, std)
        scores = predict_neural(model, x_test, device, batch_size=int(args.batch_size))
        row = write_result(
            method_dir,
            method=method,
            scores=scores,
            masks=target_test.masks,
            names=target_test.names,
            fixed_threshold=fixed_threshold,
            source_threshold=source_threshold,
            normalization_policy=policy,
            context={**context, "normalization_policy": policy, "elapsed_seconds": time.perf_counter() - started},
        )
        rows.append(row)
        print(f"{method}: source_threshold_dice={float(row['source_threshold_dice']):.4f} oracle={float(row['test_oracle_dice']):.4f}", flush=True)

    rows.sort(key=lambda row: float(row["source_threshold_dice"]), reverse=True)
    if rows:
        write_csv(args.out_dir / "leaderboard.csv", rows, list(rows[0].keys()))
    write_report(args.out_dir, rows, elapsed_seconds=time.perf_counter() - started)
    print(f"transfer evaluation complete: {args.out_dir}", flush=True)


def load_fz_split(split_dir: Path, *, max_samples: int | None) -> np.ndarray:
    files = sorted(split_dir.glob("*.npz"))
    if max_samples is not None:
        files = files[: int(max_samples)]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")
    items = []
    for path in files:
        with np.load(path, allow_pickle=False) as sample:
            items.append(np.moveaxis(_load_fz_hwt(sample), -1, 0).astype(np.float32))
    return np.stack(items).astype(np.float32)


def resample_fz_steps(fz: np.ndarray, steps: int) -> np.ndarray:
    fz = np.asarray(fz, dtype=np.float32)
    current = int(fz.shape[1])
    steps = int(steps)
    if current == steps:
        return fz.astype(np.float32)
    source = np.linspace(0.0, 1.0, current, dtype=np.float32)
    target = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    moved = np.moveaxis(fz, 1, -1).reshape(-1, current)
    out = np.empty((moved.shape[0], steps), dtype=np.float32)
    for idx, curve in enumerate(moved):
        out[idx] = np.interp(target, source, curve).astype(np.float32)
    return np.moveaxis(out.reshape(fz.shape[0], fz.shape[2], fz.shape[3], steps), -1, 1).astype(np.float32)


def dataset_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    return np.nanmean(x, axis=(0, 2, 3), keepdims=True), np.nanstd(x, axis=(0, 2, 3), keepdims=True)


def apply_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (np.asarray(x, dtype=np.float32) - mean) / (std + np.float32(1e-6))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def write_result(
    method_dir: Path,
    *,
    method: str,
    scores: np.ndarray,
    masks: np.ndarray,
    names: list[str],
    fixed_threshold: float,
    source_threshold: float,
    normalization_policy: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    method_dir.mkdir(parents=True, exist_ok=True)
    fixed = metrics_from_counts(counts_from_prediction(scores >= fixed_threshold, masks), threshold=fixed_threshold)
    source_selected = metrics_from_counts(counts_from_prediction(scores >= source_threshold, masks), threshold=source_threshold)
    oracle = evaluate_scores(scores, masks, fixed_threshold=0.5).best
    per_sample_rows = []
    for idx, name in enumerate(names):
        fixed_row = metrics_from_counts(counts_from_prediction(scores[idx] >= fixed_threshold, masks[idx]), threshold=fixed_threshold)
        source_row = metrics_from_counts(counts_from_prediction(scores[idx] >= source_threshold, masks[idx]), threshold=source_threshold)
        per_sample_rows.append(
            {
                "sample": name,
                "fixed_dice": fixed_row["dice"],
                "source_threshold_dice": source_row["dice"],
                "gt_positive": int(masks[idx].sum()),
            }
        )
    np.save(method_dir / "test_scores.npy", np.asarray(scores, dtype=np.float32))
    write_csv(method_dir / "metrics_per_sample.csv", per_sample_rows, list(per_sample_rows[0].keys()))
    summary = {
        "method": method,
        "normalization_policy": normalization_policy,
        "fixed_threshold": fixed,
        "source_selected_threshold": source_selected,
        "source_threshold": source_threshold,
        "test_oracle_best": oracle,
        "num_samples": int(masks.shape[0]),
        "config": context,
    }
    write_json(method_dir / "metrics_summary.json", summary)
    row = {
        "method": method,
        "normalization_policy": normalization_policy,
        "source_threshold_dice": source_selected["dice"],
        "source_threshold_iou": source_selected["iou"],
        "source_threshold_precision": source_selected["precision"],
        "source_threshold_recall": source_selected["recall"],
        "source_threshold": source_threshold,
        "fixed_dice": fixed["dice"],
        "fixed_iou": fixed["iou"],
        "test_oracle_dice": oracle["dice"],
        "test_oracle_threshold": oracle["threshold"],
        "path": str(method_dir),
    }
    write_json(method_dir / "leaderboard_row.json", row)
    return row


def write_report(out_dir: Path, rows: list[dict[str, Any]], *, elapsed_seconds: float) -> None:
    lines = [
        "# High-Res Delta-Fz Transfer Evaluation",
        "",
        f"- Elapsed seconds: {elapsed_seconds:.2f}",
        "- Primary threshold: source validation-selected threshold from the checkpoint run.",
        "",
        "| rank | method | normalization | source-threshold Dice | fixed Dice | oracle Dice | oracle threshold |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | `{row['method']}` | {row['normalization_policy']} | "
            f"{fmt(row['source_threshold_dice'])} | {fmt(row['fixed_dice'])} | "
            f"{fmt(row['test_oracle_dice'])} | {fmt(row['test_oracle_threshold'])} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(data), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
