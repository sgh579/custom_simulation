from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_sparse_context_repeats import (  # noqa: E402
    DEFAULT_METHODS,
    compute_delta_rows,
    load_method,
    metrics,
    paired_bootstrap_ci,
    per_sample_dice,
    resolve_path,
    set_seed,
    tune_threshold,
    write_csv,
)
from run_wrench_world_model_sweep import (  # noqa: E402
    apply_normalization,
    fit_normalization,
    load_world_split,
)


DEFAULT_RATIOS = "0.1,0.15,0.25,0.35,0.5,0.65,0.75,0.9"
DEFAULT_PATTERNS = "random,space_filling,raster_lines,local_block,missing_block"
DEFAULT_SEEDS = "20260701,20260702,20260703,20260704,20260705"


@dataclass(frozen=True)
class EvalKey:
    pattern: str
    ratio: float
    seed: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse-context deep-dive evaluation across ratios and spatial patterns.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--package-dir", type=Path, default=Path("data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/random_trajectory"))
    parser.add_argument("--out-dir", type=Path, default=Path("WM/reports/sparse_context_deep_dive"))
    parser.add_argument("--ratios", type=str, default=DEFAULT_RATIOS)
    parser.add_argument("--patterns", type=str, default=DEFAULT_PATTERNS)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sparse-dir", type=Path, default=Path(DEFAULT_METHODS["sparse_unet"]))
    parser.add_argument("--world-dir", type=Path, default=Path(DEFAULT_METHODS["world_unet"]))
    parser.add_argument("--world-curve-dir", type=Path, default=Path(DEFAULT_METHODS["world_curve_unet"]))
    args = parser.parse_args()

    root = args.root.resolve()
    package_dir = resolve_path(root, args.package_dir)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = [float(item) for item in args.ratios.split(",") if item.strip()]
    patterns = [item.strip() for item in args.patterns.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    data_root = package_dir / "data"
    train_raw = load_world_split(data_root / "train", label_size=128, max_samples=None)
    stats = fit_normalization(train_raw)
    val = apply_normalization(load_world_split(data_root / "val", label_size=128, max_samples=None), stats)
    test = apply_normalization(load_world_split(data_root / "test", label_size=128, max_samples=None), stats)
    if val is None or test is None:
        raise RuntimeError("Expected val/test splits")

    method_paths = {
        "sparse_unet": args.sparse_dir,
        "world_unet": args.world_dir,
        "world_curve_unet": args.world_curve_dir,
    }
    methods = [load_method(label, resolve_path(root, path), test.world_x, test.wrench_x.shape[1], device) for label, path in method_paths.items()]

    metric_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, float, int], dict[str, Any]] = {}
    for pattern in patterns:
        for ratio in ratios:
            for seed in seeds:
                print(f"evaluating pattern={pattern} ratio={ratio:g} seed={seed}", flush=True)
                val_mask = make_context_masks(len(val.names), val.world_x.shape[-2], val.world_x.shape[-1], ratio, pattern, seed)
                test_mask = make_context_masks(len(test.names), test.world_x.shape[-2], test.world_x.shape[-1], ratio, pattern, seed + 100_000)
                for bundle in methods:
                    set_seed(seed)
                    val_scores = predict_scores_with_masks(
                        bundle.model,
                        val.world_x,
                        val_mask,
                        device=device,
                        batch_size=int(args.batch_size),
                    )
                    threshold_choice = tune_threshold(val_scores, val.masks)
                    set_seed(seed)
                    test_scores = predict_scores_with_masks(
                        bundle.model,
                        test.world_x,
                        test_mask,
                        device=device,
                        batch_size=int(args.batch_size),
                    )
                    fixed = metrics(test_scores >= 0.5, test.masks)
                    selected = metrics(test_scores >= float(threshold_choice["threshold"]), test.masks)
                    sample_dice = per_sample_dice(test_scores, test.masks, threshold=float(threshold_choice["threshold"]))
                    sample_iou = per_sample_iou(test_scores, test.masks, threshold=float(threshold_choice["threshold"]))
                    precision, recall = precision_recall(test_scores >= float(threshold_choice["threshold"]), test.masks)
                    row = {
                        "method": bundle.label,
                        "method_dir": str(bundle.method_dir),
                        "pattern": pattern,
                        "ratio": ratio,
                        "seed": seed,
                        "observed_tokens": int(test_mask.sum(axis=1).mean()),
                        "val_selected_threshold": threshold_choice["threshold"],
                        "val_selected_val_dice": threshold_choice["dice"],
                        "test_fixed_dice": fixed["dice"],
                        "test_val_selected_dice": selected["dice"],
                        "test_iou": selected["iou"],
                        "test_precision": precision,
                        "test_recall": recall,
                        "test_mean_sample_dice": float(sample_dice.mean()),
                        "test_mean_sample_iou": float(sample_iou.mean()),
                    }
                    metric_rows.append(row)
                    cache[(bundle.label, pattern, ratio, seed)] = {
                        "sample_dice": sample_dice,
                        "test_dice": selected["dice"],
                    }
                    for idx, dice in enumerate(sample_dice):
                        sample_rows.append(
                            {
                                "method": bundle.label,
                                "pattern": pattern,
                                "ratio": ratio,
                                "seed": seed,
                                "sample": test.names[idx],
                                "mask_area": int(test.masks[idx].sum()),
                                "area_bin": area_bin(int(test.masks[idx].sum()), test.masks),
                                "sample_dice": float(dice),
                                "sample_iou": float(sample_iou[idx]),
                            }
                        )

    delta_rows = compute_pattern_delta_rows(cache, patterns, ratios, seeds, baseline_label="sparse_unet", bootstrap_reps=int(args.bootstrap_reps))
    area_rows = compute_area_rows(sample_rows, baseline_label="sparse_unet")

    write_csv(out_dir / "sparse_context_deep_metrics.csv", metric_rows)
    write_csv(out_dir / "sparse_context_deep_delta_ci.csv", delta_rows)
    write_csv(out_dir / "sparse_context_deep_samples.csv", sample_rows)
    write_csv(out_dir / "sparse_context_deep_area_bins.csv", area_rows)
    write_report(out_dir / "sparse_context_deep_dive.md", metric_rows, delta_rows, area_rows)
    print(f"wrote {out_dir / 'sparse_context_deep_dive.md'}")


@torch.no_grad()
def predict_scores_with_masks(
    model: torch.nn.Module,
    x: np.ndarray,
    context_masks: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(context_masks.astype(bool))),
        batch_size=batch_size,
        shuffle=False,
    )
    probs: list[np.ndarray] = []
    for batch, keep in loader:
        batch = batch.to(device)
        keep = keep.to(device=device, dtype=torch.bool)
        logits = model(batch, context_keep_mask=keep)
        probs.append(torch.sigmoid(logits)[:, 0].cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


def make_context_masks(n_samples: int, height: int, width: int, ratio: float, pattern: str, seed: int) -> np.ndarray:
    n = int(height) * int(width)
    k = int(round(float(ratio) * n))
    k = max(1, min(n - 1, k))
    masks = np.zeros((int(n_samples), n), dtype=bool)
    base_space_filling = None
    if pattern == "space_filling":
        base_space_filling = farthest_point_indices(height, width, k, np.random.default_rng(int(seed)))
    for idx in range(int(n_samples)):
        rng = np.random.default_rng(int(seed) + idx * 9973)
        if pattern == "random":
            keep = rng.choice(n, size=k, replace=False)
        elif pattern == "space_filling":
            assert base_space_filling is not None
            keep = shift_indices(base_space_filling, height, width, int(rng.integers(0, height)), int(rng.integers(0, width)))
        elif pattern == "raster_lines":
            keep = raster_line_indices(height, width, k, rng)
        elif pattern == "local_block":
            keep = block_indices(height, width, k, rng, keep_inside=True)
        elif pattern == "missing_block":
            drop = block_indices(height, width, n - k, rng, keep_inside=True)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[drop] = False
            keep = np.flatnonzero(keep_mask)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        masks[idx, keep] = True
    return masks


def shift_indices(indices: np.ndarray, height: int, width: int, row_shift: int, col_shift: int) -> np.ndarray:
    rows = indices // int(width)
    cols = indices % int(width)
    shifted_rows = (rows + int(row_shift)) % int(height)
    shifted_cols = (cols + int(col_shift)) % int(width)
    return (shifted_rows * int(width) + shifted_cols).astype(np.int64)


def farthest_point_indices(height: int, width: int, k: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
    first = int(rng.integers(0, coords.shape[0]))
    selected = [first]
    dist2 = np.sum((coords - coords[first]) ** 2, axis=1)
    while len(selected) < int(k):
        # Add tiny jitter only to break exact-distance ties reproducibly.
        next_idx = int(np.argmax(dist2 + rng.random(coords.shape[0]) * 1e-6))
        selected.append(next_idx)
        new_dist2 = np.sum((coords - coords[next_idx]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_dist2)
    return np.asarray(selected, dtype=np.int64)


def raster_line_indices(height: int, width: int, k: int, rng: np.random.Generator) -> np.ndarray:
    rows_needed = max(1, min(height, int(np.ceil(float(k) / float(width)))))
    offset = int(rng.integers(0, max(height, 1)))
    base_rows = [int(round(x)) for x in np.linspace(0, height - 1, num=rows_needed)]
    row_order: list[int] = []
    for row in base_rows:
        shifted = (int(row) + offset) % int(height)
        if shifted not in row_order:
            row_order.append(shifted)
    for row in np.roll(np.arange(height), offset):
        if int(row) not in row_order:
            row_order.append(int(row))
        if len(row_order) >= rows_needed:
            break
    indices: list[int] = []
    reverse = bool(rng.integers(0, 2))
    cols = np.arange(width - 1, -1, -1) if reverse else np.arange(width)
    for row in row_order:
        for col in cols:
            indices.append(int(row) * int(width) + int(col))
            if len(indices) >= int(k):
                return np.asarray(indices, dtype=np.int64)
    return np.asarray(indices[:k], dtype=np.int64)


def block_indices(height: int, width: int, k: int, rng: np.random.Generator, *, keep_inside: bool) -> np.ndarray:
    del keep_inside
    aspect = float(width) / max(float(height), 1.0)
    bh = max(1, min(height, int(round(np.sqrt(k / max(aspect, 1e-6))))))
    bw = max(1, min(width, int(np.ceil(k / bh))))
    while bh * bw < k and bh < height:
        bh += 1
    while bh * bw < k and bw < width:
        bw += 1
    top = int(rng.integers(0, max(1, height - bh + 1)))
    left = int(rng.integers(0, max(1, width - bw + 1)))
    block = [r * width + c for r in range(top, min(top + bh, height)) for c in range(left, min(left + bw, width))]
    if len(block) > k:
        block = list(rng.choice(np.asarray(block, dtype=np.int64), size=k, replace=False))
    elif len(block) < k:
        missing = np.setdiff1d(np.arange(height * width, dtype=np.int64), np.asarray(block, dtype=np.int64), assume_unique=False)
        extra = rng.choice(missing, size=k - len(block), replace=False)
        block.extend([int(x) for x in extra])
    return np.asarray(block, dtype=np.int64)


def per_sample_iou(scores: np.ndarray, masks: np.ndarray, *, threshold: float, eps: float = 1e-9) -> np.ndarray:
    pred = np.asarray(scores >= threshold, dtype=bool)
    target = np.asarray(masks, dtype=bool)
    tp = np.logical_and(pred, target).sum(axis=(1, 2)).astype(np.float64)
    fp = np.logical_and(pred, ~target).sum(axis=(1, 2)).astype(np.float64)
    fn = np.logical_and(~pred, target).sum(axis=(1, 2)).astype(np.float64)
    return (tp / np.maximum(tp + fp + fn, eps)).astype(np.float64)


def precision_recall(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> tuple[float, float]:
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    tp = float(np.logical_and(pred_bool, target_bool).sum())
    fp = float(np.logical_and(pred_bool, ~target_bool).sum())
    fn = float(np.logical_and(~pred_bool, target_bool).sum())
    return tp / max(tp + fp, eps), tp / max(tp + fn, eps)


def area_bin(area: int, masks: np.ndarray) -> str:
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    q1, q2 = np.quantile(areas, [1 / 3, 2 / 3])
    if area <= q1:
        return "small"
    if area <= q2:
        return "medium"
    return "large"


def compute_pattern_delta_rows(
    cache: dict[tuple[str, str, float, int], dict[str, Any]],
    patterns: list[str],
    ratios: list[float],
    seeds: list[int],
    *,
    baseline_label: str,
    bootstrap_reps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = sorted({key[0] for key in cache if key[0] != baseline_label})
    for pattern in patterns:
        for ratio in ratios:
            for label in labels:
                seed_deltas: list[float] = []
                sample_deltas: list[np.ndarray] = []
                for seed in seeds:
                    base = cache[(baseline_label, pattern, ratio, seed)]
                    method = cache[(label, pattern, ratio, seed)]
                    seed_deltas.append(float(method["test_dice"] - base["test_dice"]))
                    sample_deltas.append(method["sample_dice"] - base["sample_dice"])
                combined = np.concatenate(sample_deltas, axis=0)
                ci = paired_bootstrap_ci(combined, reps=bootstrap_reps, seed=20260701)
                rows.append(
                    {
                        "method": label,
                        "baseline": baseline_label,
                        "pattern": pattern,
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


def compute_area_rows(sample_rows: list[dict[str, Any]], *, baseline_label: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, int, str, str], list[float]] = defaultdict(list)
    for row in sample_rows:
        grouped[(row["pattern"], float(row["ratio"]), int(row["seed"]), row["area_bin"], row["method"])].append(float(row["sample_dice"]))
    rows: list[dict[str, Any]] = []
    keys = sorted({key[:4] for key in grouped})
    methods = sorted({key[4] for key in grouped})
    for pattern, ratio, seed, bin_name in keys:
        base_values = grouped.get((pattern, ratio, seed, bin_name, baseline_label), [])
        base_mean = float(np.mean(base_values)) if base_values else np.nan
        for method in methods:
            values = grouped.get((pattern, ratio, seed, bin_name, method), [])
            if not values:
                continue
            mean_dice = float(np.mean(values))
            rows.append(
                {
                    "pattern": pattern,
                    "ratio": ratio,
                    "seed": seed,
                    "area_bin": bin_name,
                    "method": method,
                    "mean_sample_dice": mean_dice,
                    "delta_vs_sparse_unet": mean_dice - base_mean if method != baseline_label and np.isfinite(base_mean) else "",
                    "num_samples": len(values),
                }
            )
    return rows


def write_report(path: Path, metric_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], area_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Sparse-Context Deep Dive",
        "",
        "All rows use 128x128 scan-area GT. Thresholds are selected on validation for each method, sparse ratio, pattern, and seed.",
        "",
        "## Ratio-Pattern Delta Summary",
        "",
        "| Pattern | Ratio | Method | Mean Test Dice | Delta vs No-World | Sample Delta 95% CI |",
        "|---|---:|---|---:|---:|---:|",
    ]
    metric_mean = aggregate_metric_means(metric_rows)
    for row in sorted(delta_rows, key=lambda r: (r["pattern"], float(r["ratio"]), r["method"])):
        method = row["method"]
        pattern = row["pattern"]
        ratio = float(row["ratio"])
        mean_dice = metric_mean[(pattern, ratio, method)]
        lines.append(
            "| {pattern} | {ratio:g} | {method} | {dice:.6f} | {delta:+.6f} | [{lo:+.6f}, {hi:+.6f}] |".format(
                pattern=pattern,
                ratio=ratio,
                method=method,
                dice=mean_dice,
                delta=float(row["mean_aggregate_dice_delta"]),
                lo=float(row["sample_bootstrap_ci_low"]),
                hi=float(row["sample_bootstrap_ci_high"]),
            )
        )
    lines.extend(["", "## No-World Baseline Means", "", "| Pattern | Ratio | Sparse UNet Mean Test Dice |", "|---|---:|---:|"])
    for key, value in sorted(metric_mean.items()):
        pattern, ratio, method = key
        if method == "sparse_unet":
            lines.append(f"| {pattern} | {ratio:g} | {value:.6f} |")
    lines.extend(["", "## Area-Bin Delta Summary", "", "| Pattern | Ratio | Area Bin | Method | Mean Delta vs No-World |", "|---|---:|---|---|---:|"])
    area_summary: dict[tuple[str, float, str, str], list[float]] = defaultdict(list)
    for row in area_rows:
        if row["method"] == "sparse_unet" or row["delta_vs_sparse_unet"] == "":
            continue
        area_summary[(row["pattern"], float(row["ratio"]), row["area_bin"], row["method"])].append(float(row["delta_vs_sparse_unet"]))
    for (pattern, ratio, bin_name, method), values in sorted(area_summary.items()):
        lines.append(f"| {pattern} | {ratio:g} | {bin_name} | {method} | {float(np.mean(values)):+.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_metric_means(metric_rows: list[dict[str, Any]]) -> dict[tuple[str, float, str], float]:
    grouped: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        grouped[(row["pattern"], float(row["ratio"]), row["method"])].append(float(row["test_val_selected_dice"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


if __name__ == "__main__":
    random.seed(20260701)
    main()
