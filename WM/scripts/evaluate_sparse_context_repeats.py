from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_wrench_world_model_sweep import (  # noqa: E402
    apply_normalization,
    fit_normalization,
    load_world_split,
    make_temporal_cnn_world_model,
    predict_scores,
)


DEFAULT_METHODS = {
    "sparse_unet": "WM/runs/sparse_baseline_core_20260701/r128_wrench_temporal_cnn_sparse_unet",
    "world_unet": "runs/wrench_temporal_cnn_world_20260622/r128_wrench_temporal_cnn_world_unet",
    "world_curve_unet": "runs/wrench_temporal_cnn_world_20260622/r128_wrench_temporal_cnn_world_curve_unet",
}


@dataclass
class MethodBundle:
    label: str
    method_dir: Path
    config: dict[str, Any]
    model: torch.nn.Module


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated sparse-context evaluation with paired bootstrap deltas.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--package-dir", type=Path, default=Path("data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/random_trajectory"))
    parser.add_argument("--out-dir", type=Path, default=Path("WM/reports/sparse_context_repeats"))
    parser.add_argument("--ratios", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--seeds", type=str, default="20260701,20260702,20260703,20260704,20260705")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    root = args.root.resolve()
    package_dir = resolve_path(root, args.package_dir)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = [float(item) for item in args.ratios.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    data_root = package_dir / "data"
    train_raw = load_world_split(data_root / "train", label_size=128, max_samples=None)
    stats = fit_normalization(train_raw)
    val = apply_normalization(load_world_split(data_root / "val", label_size=128, max_samples=None), stats)
    test = apply_normalization(load_world_split(data_root / "test", label_size=128, max_samples=None), stats)
    if val is None or test is None:
        raise RuntimeError("Expected val/test splits")

    methods = [load_method(label, resolve_path(root, Path(path)), test.world_x, test.wrench_x.shape[1], device) for label, path in DEFAULT_METHODS.items()]
    metric_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, float, int], dict[str, Any]] = {}
    for ratio in ratios:
        for seed in seeds:
            for bundle in methods:
                set_seed(seed)
                val_scores = predict_scores(bundle.model, val.world_x, device=device, batch_size=int(args.batch_size), context_ratio=ratio)
                threshold_choice = tune_threshold(val_scores, val.masks)
                set_seed(seed)
                test_scores = predict_scores(bundle.model, test.world_x, device=device, batch_size=int(args.batch_size), context_ratio=ratio)
                fixed = metrics(test_scores >= 0.5, test.masks)
                selected = metrics(test_scores >= float(threshold_choice["threshold"]), test.masks)
                sample_dice = per_sample_dice(test_scores, test.masks, threshold=float(threshold_choice["threshold"]))
                row = {
                    "method": bundle.label,
                    "method_dir": str(bundle.method_dir),
                    "ratio": ratio,
                    "seed": seed,
                    "val_selected_threshold": threshold_choice["threshold"],
                    "val_selected_val_dice": threshold_choice["dice"],
                    "test_fixed_dice": fixed["dice"],
                    "test_val_selected_dice": selected["dice"],
                    "test_mean_sample_dice": float(sample_dice.mean()),
                }
                metric_rows.append(row)
                prediction_cache[(bundle.label, ratio, seed)] = {"sample_dice": sample_dice, "test_dice": selected["dice"]}

    delta_rows = compute_delta_rows(prediction_cache, ratios, seeds, baseline_label="sparse_unet", bootstrap_reps=int(args.bootstrap_reps))
    write_csv(out_dir / "sparse_context_repeated_metrics.csv", metric_rows)
    write_csv(out_dir / "sparse_context_delta_ci.csv", delta_rows)
    write_report(out_dir / "sparse_context_repeats.md", metric_rows, delta_rows)
    print(f"wrote {out_dir / 'sparse_context_repeats.md'}")


def load_method(label: str, method_dir: Path, world_x: np.ndarray, curve_channels: int, device: torch.device) -> MethodBundle:
    config = read_json(method_dir / "config.json")
    checkpoint = torch.load(method_dir / "best.pt", map_location=device)
    state = checkpoint["model_state"]
    decoder_base = int(state.get("mask_decoder.inc.0.weight").shape[0]) if "mask_decoder.inc.0.weight" in state else int(config.get("decoder_base_channels", 32))
    variant = str(config.get("variant", ""))
    ns = SimpleNamespace(
        dim=int(config.get("dim", 96)),
        decoder_base_channels=decoder_base,
        world_target_ratio=float(config.get("world_target_ratio", 0.35)),
    )
    curve_aux = variant == "wrench_temporal_cnn_world_curve_unet"
    model = make_temporal_cnn_world_model(world_x, curve_channels, int(config.get("resolution", 128)), ns, curve_aux=curve_aux)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return MethodBundle(label=label, method_dir=method_dir, config=config, model=model)


def compute_delta_rows(
    cache: dict[tuple[str, float, int], dict[str, Any]],
    ratios: list[float],
    seeds: list[int],
    *,
    baseline_label: str,
    bootstrap_reps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = sorted({key[0] for key in cache if key[0] != baseline_label})
    for ratio in ratios:
        for label in labels:
            seed_deltas: list[float] = []
            sample_deltas: list[np.ndarray] = []
            for seed in seeds:
                base = cache[(baseline_label, ratio, seed)]
                method = cache[(label, ratio, seed)]
                seed_deltas.append(float(method["test_dice"] - base["test_dice"]))
                sample_deltas.append(method["sample_dice"] - base["sample_dice"])
            combined = np.concatenate(sample_deltas, axis=0)
            ci = paired_bootstrap_ci(combined, reps=bootstrap_reps, seed=20260701)
            rows.append(
                {
                    "method": label,
                    "baseline": baseline_label,
                    "ratio": ratio,
                    "mean_aggregate_dice_delta": float(np.mean(seed_deltas)),
                    "min_aggregate_dice_delta": float(np.min(seed_deltas)),
                    "max_aggregate_dice_delta": float(np.max(seed_deltas)),
                    "mean_sample_dice_delta": float(combined.mean()),
                    "sample_bootstrap_ci_low": ci[0],
                    "sample_bootstrap_ci_high": ci[1],
                    "num_seeds": len(seeds),
                    "num_sample_seed_pairs": int(combined.shape[0]),
                }
            )
    return rows


def tune_threshold(scores: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for threshold in np.linspace(0.05, 0.95, 37, dtype=np.float32):
        row = {"threshold": float(threshold), **metrics(scores >= float(threshold), masks)}
        if best is None or row["dice"] > best["dice"]:
            best = row
    assert best is not None
    return best


def metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> dict[str, float]:
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    tp = int(np.logical_and(pred_bool, target_bool).sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    dice = 2.0 * tp / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    return {"dice": float(dice), "iou": float(iou)}


def per_sample_dice(scores: np.ndarray, masks: np.ndarray, *, threshold: float, eps: float = 1e-9) -> np.ndarray:
    pred = np.asarray(scores >= threshold, dtype=bool)
    target = np.asarray(masks, dtype=bool)
    tp = np.logical_and(pred, target).sum(axis=(1, 2)).astype(np.float64)
    fp = np.logical_and(pred, ~target).sum(axis=(1, 2)).astype(np.float64)
    fn = np.logical_and(~pred, target).sum(axis=(1, 2)).astype(np.float64)
    return (2.0 * tp / np.maximum(2.0 * tp + fp + fn, eps)).astype(np.float64)


def paired_bootstrap_ci(values: np.ndarray, *, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = int(values.shape[0])
    draws = np.empty(int(reps), dtype=np.float64)
    for idx in range(int(reps)):
        sample = rng.integers(0, n, size=n)
        draws[idx] = float(values[sample].mean())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_report(path: Path, metric_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Sparse-Context WM Repeated Evaluation",
        "",
        "Validation-selected thresholds are tuned independently for each method, ratio, and random sparse-context seed.",
        "",
        "## Delta CI",
        "",
        "| Method | Ratio | Mean Aggregate Dice Delta | Mean Sample Dice Delta | 95% CI | Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in delta_rows:
        lines.append(
            "| {method} | {ratio:g} | {agg:+.6f} | {sample:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {seeds} |".format(
                method=row["method"],
                ratio=float(row["ratio"]),
                agg=float(row["mean_aggregate_dice_delta"]),
                sample=float(row["mean_sample_dice_delta"]),
                lo=float(row["sample_bootstrap_ci_low"]),
                hi=float(row["sample_bootstrap_ci_high"]),
                seeds=int(row["num_seeds"]),
            )
        )
    lines.extend(["", "## Per-Seed Metrics", "", "| Method | Ratio | Seed | Test Dice | Threshold |", "|---|---:|---:|---:|---:|"])
    for row in metric_rows:
        lines.append(
            f"| {row['method']} | {float(row['ratio']):g} | {int(row['seed'])} | {float(row['test_val_selected_dice']):.6f} | {float(row['val_selected_threshold']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
