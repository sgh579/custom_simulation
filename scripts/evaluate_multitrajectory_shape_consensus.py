from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_highres_segmentation_sweep import load_split  # noqa: E402
from run_segmentation_accuracy_sweep import counts_from_prediction, evaluate_scores, metrics_from_counts  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "runs/nonlinear_trajectory_20x_repeats10_seed20260618/"
    "highres_fz_temporal_variants/r128_fz_features_aug_focal_unet"
)
DEFAULT_DATA_ROOT = Path(
    "data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/"
    "random_trajectory/data"
)
DEFAULT_OUT_DIR = Path("runs/multitrajectory_shape_consensus_20260625")
AGGREGATIONS = ("mean", "median", "logit_mean", "trimmed_mean", "p60", "p70")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-trajectory probability consensus for repeated phantom scans.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--baseline-threshold", type=float, default=None)
    parser.add_argument("--max-visual-samples", type=int, default=10)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_threshold = args.baseline_threshold or read_baseline_threshold(args.run_dir / "test_metrics_summary.json")
    val = load_payload(args.data_root / "val", args.run_dir / "val_scores.npy", args.resolution)
    test = load_payload(args.data_root / "test", args.run_dir / "test_scores.npy", args.resolution)
    baseline = metrics_from_counts(counts_from_prediction(test["scores"] >= baseline_threshold, test["masks"]), threshold=baseline_threshold)

    rows: list[dict[str, Any]] = []
    grouped_scores: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
    for aggregation in AGGREGATIONS:
        val_scores = aggregate_scores(val["scores"], val["base_ids"], aggregation)
        test_scores = aggregate_scores(test["scores"], test["base_ids"], aggregation)
        grouped_scores["val"][aggregation] = val_scores
        grouped_scores["test"][aggregation] = test_scores
        val_best = evaluate_scores(val_scores, val["masks"], fixed_threshold=0.5).best
        threshold = float(val_best["threshold"])
        test_selected = metrics_from_counts(counts_from_prediction(test_scores >= threshold, test["masks"]), threshold=threshold)
        test_oracle = evaluate_scores(test_scores, test["masks"], fixed_threshold=0.5).best
        rows.append(
            {
                "aggregation": aggregation,
                "val_threshold": threshold,
                "val_dice": val_best["dice"],
                "val_iou": val_best["iou"],
                "test_dice": test_selected["dice"],
                "test_iou": test_selected["iou"],
                "test_precision": test_selected["precision"],
                "test_recall": test_selected["recall"],
                "test_oracle_threshold": test_oracle["threshold"],
                "test_oracle_dice": test_oracle["dice"],
                "test_delta_dice": test_selected["dice"] - baseline["dice"],
            }
        )
    rows.sort(key=lambda row: float(row["val_dice"]), reverse=True)
    selected = rows[0]
    selected_aggregation = str(selected["aggregation"])
    selected_scores = grouped_scores["test"][selected_aggregation]
    selected_threshold = float(selected["val_threshold"])
    selected_metrics = metrics_from_counts(
        counts_from_prediction(selected_scores >= selected_threshold, test["masks"]),
        threshold=selected_threshold,
    )

    np.save(args.out_dir / "test_selected_consensus_scores.npy", selected_scores.astype(np.float32))
    np.save(args.out_dir / "val_selected_consensus_scores.npy", grouped_scores["val"][selected_aggregation].astype(np.float32))
    write_csv(args.out_dir / "aggregation_leaderboard.csv", rows)
    per_sample_rows = write_per_sample(
        args.out_dir / "metrics_per_sample.csv",
        test["names"],
        test["scores"],
        selected_scores,
        test["masks"],
        baseline_threshold=baseline_threshold,
        selected_threshold=selected_threshold,
    )
    summary = {
        "experiment": "multi_trajectory_shape_consensus",
        "interpretation": (
            "This setting assumes repeated trajectories for the same base phantom are available. "
            "Scores are aggregated within base_phantom_index groups, validation selects aggregation and threshold, "
            "and test labels are used only for reporting."
        ),
        "run_dir": str(args.run_dir),
        "data_root": str(args.data_root),
        "baseline_threshold": baseline_threshold,
        "num_val_samples": len(val["names"]),
        "num_test_samples": len(test["names"]),
        "num_val_groups": int(len(set(val["base_ids"].tolist()))),
        "num_test_groups": int(len(set(test["base_ids"].tolist()))),
        "group_sizes_test": sorted(set(np.bincount(test["base_ids"])[np.bincount(test["base_ids"]) > 0].astype(int).tolist())),
        "baseline": baseline,
        "selected": selected,
        "selected_test_metrics": selected_metrics,
        "delta_vs_baseline": {
            "dice": selected_metrics["dice"] - baseline["dice"],
            "iou": selected_metrics["iou"] - baseline["iou"],
        },
        "leaderboard": rows,
    }
    write_json(args.out_dir / "metrics_summary.json", summary)
    write_report(args.out_dir / "REPORT.md", summary, per_sample_rows)
    write_visuals(args.out_dir, test, selected_scores, per_sample_rows, baseline_threshold, selected_threshold, args.max_visual_samples)
    print(json.dumps(summary, indent=2), flush=True)


def load_payload(data_dir: Path, score_path: Path, resolution: int) -> dict[str, Any]:
    split = load_split(data_dir, label_size=resolution)
    scores = np.asarray(np.load(score_path), dtype=np.float32)
    base_ids = []
    for name in split.names:
        with np.load(data_dir / name) as sample:
            base_ids.append(int(np.asarray(sample["base_phantom_index"]).reshape(())))
    return {"names": split.names, "scores": scores, "masks": split.masks, "base_ids": np.asarray(base_ids, dtype=np.int64)}


def aggregate_scores(scores: np.ndarray, base_ids: np.ndarray, aggregation: str) -> np.ndarray:
    out = np.empty_like(scores)
    for base_id in sorted(set(base_ids.tolist())):
        idx = np.flatnonzero(base_ids == int(base_id))
        stack = scores[idx]
        if aggregation == "mean":
            aggregate = stack.mean(axis=0)
        elif aggregation == "median":
            aggregate = np.median(stack, axis=0)
        elif aggregation == "logit_mean":
            aggregate = sigmoid_np(logit_np(stack).mean(axis=0))
        elif aggregation == "trimmed_mean":
            ordered = np.sort(stack, axis=0)
            aggregate = ordered[1:-1].mean(axis=0) if ordered.shape[0] > 2 else ordered.mean(axis=0)
        elif aggregation == "p60":
            aggregate = np.quantile(stack, 0.60, axis=0)
        elif aggregation == "p70":
            aggregate = np.quantile(stack, 0.70, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        out[idx] = aggregate.astype(np.float32)
    return out.astype(np.float32)


def logit_np(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(scores, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(scores / (1.0 - scores)).astype(np.float32)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float32)))).astype(np.float32)


def write_per_sample(
    path: Path,
    names: Sequence[str],
    baseline_scores: np.ndarray,
    selected_scores: np.ndarray,
    masks: np.ndarray,
    *,
    baseline_threshold: float,
    selected_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for idx, name in enumerate(names):
        baseline = metrics_from_counts(
            counts_from_prediction(baseline_scores[idx] >= baseline_threshold, masks[idx]),
            threshold=baseline_threshold,
        )
        selected = metrics_from_counts(
            counts_from_prediction(selected_scores[idx] >= selected_threshold, masks[idx]),
            threshold=selected_threshold,
        )
        rows.append(
            {
                "sample": name,
                "baseline_dice": baseline["dice"],
                "consensus_dice": selected["dice"],
                "delta_dice": selected["dice"] - baseline["dice"],
                "baseline_iou": baseline["iou"],
                "consensus_iou": selected["iou"],
                "delta_iou": selected["iou"] - baseline["iou"],
                "gt_positive": int(masks[idx].sum()),
            }
        )
    write_csv(path, rows)
    return rows


def write_visuals(
    out_dir: Path,
    test: dict[str, Any],
    consensus_scores: np.ndarray,
    rows: Sequence[dict[str, Any]],
    baseline_threshold: float,
    selected_threshold: float,
    max_visual_samples: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_dice = metrics_from_counts(
        counts_from_prediction(test["scores"] >= baseline_threshold, test["masks"]),
        threshold=baseline_threshold,
    )["dice"]
    consensus_dice = metrics_from_counts(
        counts_from_prediction(consensus_scores >= selected_threshold, test["masks"]),
        threshold=selected_threshold,
    )["dice"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(["baseline", "consensus"], [baseline_dice, consensus_dice], color=["#3b82f6", "#f97316"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice")
    ax.set_title("Multi-Trajectory Shape Consensus")
    for idx, value in enumerate([baseline_dice, consensus_dice]):
        ax.text(idx, value + 0.015, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "dice_bar.png", dpi=180)
    plt.close(fig)

    deltas = np.asarray([float(row["delta_dice"]) for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(deltas, bins=30, color="#64748b", edgecolor="white")
    ax.axvline(0.0, color="#111827", linewidth=1.2)
    ax.set_xlabel("Consensus Dice Delta vs Baseline")
    ax.set_ylabel("Samples")
    ax.set_title("Per-Sample Consensus Delta")
    fig.tight_layout()
    fig.savefig(out_dir / "delta_hist.png", dpi=180)
    plt.close(fig)

    if max_visual_samples <= 0:
        return
    order = np.argsort(deltas)
    selected_indices = (order[-max_visual_samples // 2 :][::-1].tolist() + order[: max_visual_samples // 2 + 1].tolist())[:max_visual_samples]
    cols = ["baseline prob", "baseline", "consensus prob", "consensus", "ground truth"]
    fig, axes = plt.subplots(len(selected_indices), len(cols), figsize=(2.3 * len(cols), 2.15 * len(selected_indices)))
    if len(selected_indices) == 1:
        axes = np.asarray([axes])
    for row_idx, sample_idx in enumerate(selected_indices):
        images = [
            test["scores"][sample_idx],
            test["scores"][sample_idx] >= baseline_threshold,
            consensus_scores[sample_idx],
            consensus_scores[sample_idx] >= selected_threshold,
            test["masks"][sample_idx],
        ]
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="viridis" if col_idx in {0, 2} else "gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=9)
            if col_idx == 0:
                row = rows[sample_idx]
                ax.set_ylabel(
                    f"{test['names'][sample_idx]}\n"
                    f"B {float(row['baseline_dice']):.3f} C {float(row['consensus_dice']):.3f}",
                    fontsize=8,
                )
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_contact_sheet.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    best = sorted(rows, key=lambda row: float(row["delta_dice"]), reverse=True)[:5]
    worst = sorted(rows, key=lambda row: float(row["delta_dice"]))[:5]
    lines = [
        "# Multi-Trajectory Shape Consensus",
        "",
        "This run assumes repeated trajectories for each base phantom are available.",
        "Validation selects the probability aggregation rule and threshold; test labels are only used for reporting.",
        "",
        "## Test Dice",
        "",
        f"- Baseline: {summary['baseline']['dice']:.6f}",
        f"- Consensus: {summary['selected_test_metrics']['dice']:.6f} ({summary['delta_vs_baseline']['dice']:+.6f})",
        f"- Selected aggregation: {summary['selected']['aggregation']}",
        f"- Selected threshold: {summary['selected']['val_threshold']}",
        "",
        "## Largest Improvements",
        "",
    ]
    lines.extend(format_rows(best))
    lines.extend(["", "## Largest Regressions", ""])
    lines.extend(format_rows(worst))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_rows(rows: Sequence[dict[str, Any]]) -> list[str]:
    return [
        f"- {row['sample']}: baseline={float(row['baseline_dice']):.4f}, "
        f"consensus={float(row['consensus_dice']):.4f}, delta={float(row['delta_dice']):+.4f}"
        for row in rows
    ]


def read_baseline_threshold(path: Path) -> float:
    summary = json.loads(path.read_text(encoding="utf-8"))
    return float(summary.get("val_selected_threshold", {}).get("threshold", 0.5))


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
