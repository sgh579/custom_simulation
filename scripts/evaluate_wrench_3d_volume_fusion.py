from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_segmentation_accuracy_sweep import _read_json, _write_csv, _write_json
from run_wrench_3d_volume_prediction import (
    DEFAULT_PACKAGE_DIR,
    depth_group_metric_rows,
    depth_leaderboard_values,
    evaluate_volume_scores,
    load_raw_volume_split,
    parse_volume_shape,
    safe_float,
    summarize_depth_groups,
    write_volume_metrics,
)


DEFAULT_BASELINE_DIR = Path("runs/wrench_3d_volume_prediction_20260622/v32x64x64_wrench_temporal_cnn_volume")
DEFAULT_OUT_DIR = Path("runs/wrench_3d_volume_fusion_20260622")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse supervised and world-model 3D volume scores with val-selected alpha.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--world-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--volume-shape", type=str, default="32,64,64")
    parser.add_argument("--alpha-grid", type=str, default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0")
    parser.add_argument("--selection-objective", choices=("global_dice", "depth_aware"), default="global_dice")
    parser.add_argument("--depth-shallow-weight", type=float, default=0.45)
    parser.add_argument("--depth-mid-weight", type=float, default=0.35)
    parser.add_argument("--depth-deep-weight", type=float, default=0.20)
    parser.add_argument("--deep-fp-penalty", type=float, default=1.0)
    parser.add_argument("--deep-overprediction-penalty", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    volume_shape = parse_volume_shape(args.volume_shape)
    alphas = [float(item.strip()) for item in args.alpha_grid.split(",") if item.strip()]
    if not alphas:
        raise ValueError("alpha grid is empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    method = args.method or f"{args.world_dir.name}_fusion_with_{args.baseline_dir.name}"
    method_dir = args.out_dir / method
    if method_dir.joinpath("metrics_summary.json").exists() and not args.force:
        print(f"{method}: existing metrics found")
        write_leaderboard(args.out_dir)
        return

    val_base = np.load(args.baseline_dir / "scores.npy").astype(np.float32)
    test_base = np.load(args.baseline_dir / "test_scores.npy").astype(np.float32)
    val_world = np.load(args.world_dir / "scores.npy").astype(np.float32)
    test_world = np.load(args.world_dir / "test_scores.npy").astype(np.float32)
    val = load_raw_volume_split(args.package_dir / "data" / "val", volume_shape=volume_shape, max_samples=None)
    test = load_raw_volume_split(args.package_dir / "data" / "test", volume_shape=volume_shape, max_samples=None)

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    for alpha in alphas:
        val_scores = blend_scores(val_base, val_world, alpha)
        test_scores = blend_scores(test_base, test_world, alpha)
        val_result = evaluate_volume_scores(val_scores, val.volumes, fixed_threshold=0.5)
        threshold = float(val_result["threshold_sweep_best"]["threshold"])
        test_result = evaluate_volume_scores(test_scores, test.volumes, fixed_threshold=threshold)
        val_depth = summarize_depth_groups(depth_group_metric_rows(val_scores >= threshold, val.volumes, threshold=threshold))
        test_depth = summarize_depth_groups(depth_group_metric_rows(test_scores >= threshold, test.volumes, threshold=threshold))
        selection_score = fusion_selection_score(args, val_result, val_depth)
        row = {
            "alpha_world": alpha,
            "selection_objective": args.selection_objective,
            "selection_score": selection_score,
            "val_best_dice": val_result["threshold_sweep_best"]["dice"],
            "val_best_threshold": threshold,
            "val_projection_dice": val_result["threshold_sweep_best"].get("projection_dice", ""),
            "val_shallow_dice": val_depth.get("shallow_dice", ""),
            "val_mid_dice": val_depth.get("mid_dice", ""),
            "val_deep_dice": val_depth.get("deep_dice", ""),
            "val_deep_false_positive_rate": val_depth.get("deep_false_positive_rate", ""),
            "val_deep_overprediction_rate": val_depth.get("deep_overprediction_rate", ""),
            "test_val_selected_dice": test_result["fixed_threshold"]["dice"],
            "test_val_selected_projection_dice": test_result["fixed_threshold"].get("projection_dice", ""),
            "test_oracle_dice": test_result["threshold_sweep_best"]["dice"],
            "test_shallow_dice": test_depth.get("shallow_dice", ""),
            "test_mid_dice": test_depth.get("mid_dice", ""),
            "test_deep_dice": test_depth.get("deep_dice", ""),
            "test_deep_false_positive_rate": test_depth.get("deep_false_positive_rate", ""),
            "test_deep_overprediction_rate": test_depth.get("deep_overprediction_rate", ""),
        }
        rows.append(row)
        if best_row is None or float(row["selection_score"]) > float(best_row["selection_score"]):
            best_row = row
    if best_row is None:
        raise RuntimeError("No alpha candidate was evaluated")

    alpha = float(best_row["alpha_world"])
    val_scores = blend_scores(val_base, val_world, alpha)
    test_scores = blend_scores(test_base, test_world, alpha)
    context = {
        "method": method,
        "representation": "scan_aligned_depth_slice_occupancy_score_fusion",
        "volume_shape_dhw": list(volume_shape),
        "model": "score_fusion",
        "baseline_dir": str(args.baseline_dir),
        "world_dir": str(args.world_dir),
        "alpha_world": alpha,
        "alpha_selection": args.selection_objective,
        "alpha_grid": alphas,
        "depth_selection_weights": {
            "shallow": float(args.depth_shallow_weight),
            "mid": float(args.depth_mid_weight),
            "deep": float(args.depth_deep_weight),
            "deep_fp_penalty": float(args.deep_fp_penalty),
            "deep_overprediction_penalty": float(args.deep_overprediction_penalty),
        },
        "package_dir": str(args.package_dir),
    }
    method_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(method_dir / "alpha_search.csv", rows, list(rows[0].keys()))
    _write_json(method_dir / "fusion_config.json", context)
    write_volume_metrics(
        method_dir,
        method,
        val_scores,
        val.volumes,
        val.names,
        context=context,
        fixed_threshold=0.5,
        split="val",
    )
    val_summary = _read_json(method_dir / "metrics_summary.json")
    threshold = float(val_summary.get("threshold_sweep_best", {}).get("threshold", 0.5))
    write_volume_metrics(
        method_dir,
        method,
        test_scores,
        test.volumes,
        test.names,
        context={**context, "val_selected_threshold": threshold},
        fixed_threshold=threshold,
        split="test",
    )
    write_leaderboard(args.out_dir)
    print(f"3D volume fusion complete: {method_dir}")
    print(f"selected alpha_world={alpha:.3f}")


def fusion_selection_score(args: argparse.Namespace, val_result: dict[str, Any], depth_summary: dict[str, Any]) -> float:
    if args.selection_objective == "global_dice":
        return safe_float(val_result.get("threshold_sweep_best", {}).get("dice", ""))
    shallow = safe_float(depth_summary.get("shallow_dice", ""))
    mid = safe_float(depth_summary.get("mid_dice", ""))
    deep = safe_float(depth_summary.get("deep_dice", ""))
    deep_fp = max(safe_float(depth_summary.get("deep_false_positive_rate", "")), 0.0)
    deep_over = max(safe_float(depth_summary.get("deep_overprediction_rate", "")), 0.0)
    return float(
        float(args.depth_shallow_weight) * shallow
        + float(args.depth_mid_weight) * mid
        + float(args.depth_deep_weight) * deep
        - float(args.deep_fp_penalty) * deep_fp
        - float(args.deep_overprediction_penalty) * deep_over
    )


def blend_scores(base: np.ndarray, world: np.ndarray, alpha_world: float) -> np.ndarray:
    if base.shape != world.shape:
        raise ValueError(f"Score shape mismatch: baseline {base.shape}, world {world.shape}")
    alpha = np.float32(alpha_world)
    return ((np.float32(1.0) - alpha) * base + alpha * world).astype(np.float32)


def write_leaderboard(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out_dir.glob("*/metrics_summary.json")):
        summary = _read_json(metrics_path)
        test_summary = _read_json(metrics_path.parent / "test_metrics_summary.json")
        config = summary.get("config", {})
        best = summary.get("threshold_sweep_best", {})
        test_fixed = test_summary.get("fixed_threshold", {})
        test_best = test_summary.get("threshold_sweep_best", {})
        rows.append(
            {
                "method": summary.get("method", metrics_path.parent.name),
                "representation": config.get("representation", ""),
                "volume_shape_dhw": "x".join(str(v) for v in config.get("volume_shape_dhw", [])),
                "alpha_world": config.get("alpha_world", ""),
                "val_best_dice": best.get("dice", ""),
                "val_best_threshold": best.get("threshold", ""),
                "val_projection_dice": best.get("projection_dice", ""),
                "test_val_selected_dice": test_fixed.get("dice", ""),
                "test_val_selected_projection_dice": test_fixed.get("projection_dice", ""),
                "test_oracle_dice": test_best.get("dice", ""),
                **depth_leaderboard_values(summary, test_summary),
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
