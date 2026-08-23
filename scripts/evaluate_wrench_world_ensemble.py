from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_segmentation_accuracy_sweep import _read_json, _write_csv, _write_json, evaluate_scores, write_score_method
from run_highres_segmentation_sweep import write_test_metrics
from run_wrench_world_model_sweep import DEFAULT_PACKAGE_DIR, load_world_split


DEFAULT_BASELINE_DIR = Path(
    "runs/nonlinear_trajectory_20x_repeats10_seed20260618/highres_wrench_temporal_variants/r128_wrench_temporal_cnn32_unet"
)
DEFAULT_WORLD_DIR = Path("runs/wrench_temporal_cnn_world_20260622/r128_wrench_temporal_cnn_world_unet")
DEFAULT_OUT_DIR = Path("runs/wrench_world_fusion_20260622")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate score-level fusion between the best 6D wrench baseline and a world model.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--world-dir", type=Path, default=DEFAULT_WORLD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--alpha-grid", type=str, default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    alphas = [float(item) for item in args.alpha_grid.split(",") if item.strip()]
    if not alphas:
        raise ValueError("alpha grid is empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    method = args.method or f"{args.world_dir.name}_fusion_with_{args.baseline_dir.name}"
    method_dir = args.out_dir / method
    if method_dir.joinpath("metrics_summary.json").exists() and not args.force:
        print(f"{method}: existing metrics found")
        write_leaderboard(args.out_dir)
        return

    val_scores_base = np.load(args.baseline_dir / "val_scores.npy").astype(np.float32)
    test_scores_base = np.load(args.baseline_dir / "test_scores.npy").astype(np.float32)
    val_scores_world = np.load(args.world_dir / "val_scores.npy").astype(np.float32)
    test_scores_world = np.load(args.world_dir / "test_scores.npy").astype(np.float32)
    val = load_world_split(args.package_dir / "data" / "val", label_size=args.resolution, max_samples=None)
    test = load_world_split(args.package_dir / "data" / "test", label_size=args.resolution, max_samples=None)

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for alpha in alphas:
        val_scores = blend_scores(val_scores_base, val_scores_world, alpha)
        test_scores = blend_scores(test_scores_base, test_scores_world, alpha)
        val_result = evaluate_scores(val_scores, val["masks"], fixed_threshold=0.5)
        test_result = evaluate_scores(test_scores, test["masks"], fixed_threshold=float(val_result.best["threshold"]))
        row = {
            "alpha_world": alpha,
            "val_best_dice": val_result.best["dice"],
            "val_best_threshold": val_result.best["threshold"],
            "test_val_selected_dice": test_result.fixed["dice"],
            "test_oracle_dice": test_result.best["dice"],
        }
        rows.append(row)
        if best_row is None or float(row["val_best_dice"]) > float(best_row["val_best_dice"]):
            best_row = row
    if best_row is None:
        raise RuntimeError("No alpha candidate was evaluated")

    alpha = float(best_row["alpha_world"])
    val_scores = blend_scores(val_scores_base, val_scores_world, alpha)
    test_scores = blend_scores(test_scores_base, test_scores_world, alpha)
    context = {
        "method": method,
        "resolution": int(args.resolution),
        "input": "baseline_probe_wrench_plus_world_model_score_fusion",
        "model": "score_fusion",
        "baseline_dir": str(args.baseline_dir),
        "world_dir": str(args.world_dir),
        "alpha_world": alpha,
        "alpha_selection": "max val threshold-sweep Dice over alpha grid",
        "alpha_grid": alphas,
        "package_dir": str(args.package_dir),
    }
    _write_csv(method_dir / "alpha_search.csv", rows, list(rows[0].keys()))
    _write_json(method_dir / "ensemble_config.json", context)
    write_score_method(
        args.out_dir,
        method,
        "wrench_world_fusion",
        val_scores,
        val["masks"],
        val["names"],
        config=context,
        fixed_threshold=0.5,
        force=True,
    )
    val_summary = _read_json(method_dir / "metrics_summary.json")
    val_best_threshold = float(val_summary.get("threshold_sweep_best", {}).get("threshold", 0.5))
    write_test_metrics(
        method_dir,
        test_scores,
        test["masks"],
        test["names"],
        fixed_threshold=0.5,
        val_best_threshold=val_best_threshold,
        context=context,
        force=True,
    )
    write_leaderboard(args.out_dir)
    print(f"fusion complete: {method_dir}")
    print(f"selected alpha_world={alpha:.3f}")


def blend_scores(base: np.ndarray, world: np.ndarray, alpha_world: float) -> np.ndarray:
    if base.shape != world.shape:
        raise ValueError(f"Score shape mismatch: baseline {base.shape}, world {world.shape}")
    alpha = np.float32(alpha_world)
    return ((np.float32(1.0) - alpha) * base + alpha * world).astype(np.float32)


def write_leaderboard(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out_dir.glob("*/metrics_summary.json")):
        summary = _read_json(metrics_path)
        config = summary.get("config", {})
        fixed = summary.get("fixed_threshold", {})
        best = summary.get("threshold_sweep_best", {})
        test_summary = _read_json(metrics_path.parent / "test_metrics_summary.json")
        test_fixed = test_summary.get("fixed_threshold", {})
        test_val = test_summary.get("val_selected_threshold", {})
        test_oracle = test_summary.get("test_oracle_threshold", {})
        rows.append(
            {
                "method": summary.get("method", metrics_path.parent.name),
                "resolution": config.get("resolution", ""),
                "input": config.get("input", ""),
                "model": config.get("model", ""),
                "alpha_world": config.get("alpha_world", ""),
                "val_best_dice": best.get("dice", ""),
                "val_best_threshold": best.get("threshold", ""),
                "test_fixed_dice": test_fixed.get("dice", ""),
                "test_val_selected_dice": test_val.get("dice", ""),
                "test_val_selected_threshold": test_val.get("threshold", ""),
                "test_oracle_dice": test_oracle.get("dice", ""),
                "baseline_dir": config.get("baseline_dir", ""),
                "world_dir": config.get("world_dir", ""),
                "path": str(metrics_path.parent),
            }
        )
    rows.sort(key=lambda row: safe_float(row.get("test_val_selected_dice")), reverse=True)
    if rows:
        _write_csv(out_dir / "leaderboard.csv", rows, list(rows[0].keys()))


def safe_float(value: Any) -> float:
    try:
        if value == "":
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


if __name__ == "__main__":
    main()
