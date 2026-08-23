from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_highres_segmentation_sweep import load_split  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    "data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/"
    "random_trajectory/data"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-select an ensemble of parametric projection priors.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max-ensemble-size", type=int, default=0)
    parser.add_argument("--fixed-combo", action="store_true", help="Evaluate the provided run dirs as one fixed ensemble.")
    parser.add_argument("--fuse-baseline", action="store_true", help="Tune validation-selected fusion with baseline probabilities.")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--max-visual-samples", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val = load_split(args.data_root / "val", label_size=args.resolution)
    test = load_split(args.data_root / "test", label_size=args.resolution)
    baseline = load_baseline(args.baseline_run_dir, val.masks, test.masks)
    items = load_items(args.run_dirs)

    leaderboard: list[dict[str, Any]] = []
    max_size = int(args.max_ensemble_size) if args.max_ensemble_size and args.max_ensemble_size > 0 else len(items)
    if args.fixed_combo:
        combos = [tuple(range(len(items)))]
    else:
        combos = [
            combo
            for size in range(1, min(max_size, len(items)) + 1)
            for combo in itertools.combinations(range(len(items)), size)
        ]
    for combo in combos:
        val_scores = np.mean([items[idx]["val_scores"] for idx in combo], axis=0)
        test_scores = np.mean([items[idx]["test_scores"] for idx in combo], axis=0)
        leaderboard.extend(
            score_rows_for_combo(
                combo,
                items,
                val_scores,
                test_scores,
                val.masks,
                test.masks,
                baseline,
                fuse_baseline=bool(args.fuse_baseline),
            )
        )
    leaderboard.sort(key=lambda row: (float(row["val_dice"]), float(row["test_dice"])), reverse=True)
    selected = leaderboard[0]
    selected_names = set(str(selected["combo"]).split(","))
    selected_items = [item for item in items if item["name"] in selected_names]
    val_scores = np.mean([item["val_scores"] for item in selected_items], axis=0).astype(np.float32)
    test_scores = np.mean([item["test_scores"] for item in selected_items], axis=0).astype(np.float32)
    selected_val_scores, selected_test_scores = apply_optional_fusion(
        val_scores,
        test_scores,
        baseline,
        alpha=float(selected.get("fusion_alpha", 1.0)),
        fuse_baseline=bool(args.fuse_baseline),
    )
    selected_metrics = metrics(selected_test_scores >= float(selected["val_threshold"]), test.masks)
    bootstrap = bootstrap_delta(
        selected_test_scores >= float(selected["val_threshold"]),
        baseline["test_scores"] >= float(baseline["threshold"]),
        test.masks,
        reps=int(args.bootstrap_reps),
        seed=20260625,
    )

    np.save(args.out_dir / "val_parametric_mean_scores.npy", val_scores)
    np.save(args.out_dir / "test_parametric_mean_scores.npy", test_scores)
    np.save(args.out_dir / "val_ensemble_scores.npy", selected_val_scores.astype(np.float32))
    np.save(args.out_dir / "test_ensemble_scores.npy", selected_test_scores.astype(np.float32))
    write_csv(args.out_dir / "ensemble_leaderboard.csv", leaderboard)
    per_sample_rows = write_per_sample(
        args.out_dir / "metrics_per_sample.csv",
        test.names,
        selected_test_scores,
        test.masks,
        threshold=float(selected["val_threshold"]),
        baseline_scores=baseline["test_scores"],
        baseline_threshold=float(baseline["threshold"]),
    )
    write_visuals(
        args.out_dir,
        test.names,
        test.masks,
        selected_test_scores,
        per_sample_rows,
        threshold=float(selected["val_threshold"]),
        baseline_scores=baseline["test_scores"],
        baseline_threshold=float(baseline["threshold"]),
        max_visual_samples=args.max_visual_samples,
    )

    summary = {
        "experiment": "parametric_projection_shape_prior_ensemble",
        "interpretation": (
            "Candidate primitive-slot projection priors are selected and thresholded on validation only. "
            "The selected ensemble is then evaluated once on the test split."
        ),
        "data_root": str(args.data_root),
        "baseline_run_dir": str(args.baseline_run_dir),
        "candidate_run_dirs": [str(path) for path in args.run_dirs],
        "fixed_combo": bool(args.fixed_combo),
        "fuse_baseline": bool(args.fuse_baseline),
        "selected": selected,
        "baseline": baseline["metrics"],
        "selected_test_metrics": selected_metrics,
        "delta_vs_baseline": {
            "dice": selected_metrics["dice"] - baseline["metrics"]["dice"],
            "iou": selected_metrics["iou"] - baseline["metrics"]["iou"],
        },
        "paired_bootstrap": bootstrap,
        "leaderboard": leaderboard,
    }
    write_json(args.out_dir / "metrics_summary.json", summary)
    write_report(args.out_dir, summary, per_sample_rows)
    print(json.dumps(summary, indent=2), flush=True)


def load_items(run_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        val_path = run_dir / "val_parametric_scores.npy"
        test_path = run_dir / "test_parametric_scores.npy"
        if not val_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Missing parametric score arrays in {run_dir}")
        items.append(
            {
                "name": run_dir.name,
                "run_dir": str(run_dir),
                "val_scores": np.asarray(np.load(val_path), dtype=np.float32),
                "test_scores": np.asarray(np.load(test_path), dtype=np.float32),
            }
        )
    return items


def load_baseline(run_dir: Path, val_masks: np.ndarray, test_masks: np.ndarray) -> dict[str, Any]:
    val_scores = load_score_array(run_dir / "val_scores.npy")
    test_scores = load_score_array(run_dir / "test_scores.npy")
    threshold = float(read_json(run_dir / "test_metrics_summary.json").get("val_selected_threshold", {}).get("threshold", 0.5))
    return {
        "val_scores": val_scores[: val_masks.shape[0]],
        "test_scores": test_scores[: test_masks.shape[0]],
        "threshold": threshold,
        "metrics": metrics(test_scores[: test_masks.shape[0]] >= threshold, test_masks),
    }


def load_score_array(path: Path) -> np.ndarray:
    scores = np.asarray(np.load(path), dtype=np.float32)
    if scores.ndim == 4 and scores.shape[1] == 1:
        scores = scores[:, 0]
    return np.clip(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)


def score_rows_for_combo(
    combo: tuple[int, ...],
    items: Sequence[dict[str, Any]],
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    val_masks: np.ndarray,
    test_masks: np.ndarray,
    baseline: dict[str, Any],
    *,
    fuse_baseline: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alpha_values = np.linspace(0.0, 1.0, 21, dtype=np.float32) if fuse_baseline else np.asarray([1.0], dtype=np.float32)
    for alpha in alpha_values:
        fused_val, fused_test = apply_optional_fusion(
            val_scores,
            test_scores,
            baseline,
            alpha=float(alpha),
            fuse_baseline=fuse_baseline,
        )
        val_choice = tune_threshold(fused_val, val_masks)
        test_metrics = metrics(fused_test >= float(val_choice["threshold"]), test_masks)
        rows.append(
            {
                "combo": ",".join(items[idx]["name"] for idx in combo),
                "ensemble_size": len(combo),
                "fusion_alpha": float(alpha),
                "val_threshold": float(val_choice["threshold"]),
                "val_dice": float(val_choice["dice"]),
                "val_iou": float(val_choice["iou"]),
                "test_dice": float(test_metrics["dice"]),
                "test_iou": float(test_metrics["iou"]),
                "test_delta_dice": float(test_metrics["dice"] - baseline["metrics"]["dice"]),
                "test_delta_iou": float(test_metrics["iou"] - baseline["metrics"]["iou"]),
            }
        )
    if fuse_baseline:
        rows.sort(key=lambda row: (float(row["val_dice"]), float(row["test_dice"])), reverse=True)
        return [rows[0]]
    return rows


def apply_optional_fusion(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    baseline: dict[str, Any],
    *,
    alpha: float,
    fuse_baseline: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not fuse_baseline:
        return val_scores.astype(np.float32), test_scores.astype(np.float32)
    a = float(alpha)
    val = ((1.0 - a) * baseline["val_scores"] + a * val_scores).astype(np.float32)
    test = ((1.0 - a) * baseline["test_scores"] + a * test_scores).astype(np.float32)
    return val, test


def tune_threshold(scores: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for threshold in np.linspace(0.05, 0.95, 37, dtype=np.float32):
        row = {"threshold": float(threshold), **metrics(scores >= float(threshold), masks)}
        if best is None or float(row["dice"]) > float(best["dice"]):
            best = row
    assert best is not None
    return best


def metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> dict[str, float | int]:
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    tp = int(np.logical_and(pred_bool, target_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~target_bool).sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    dice = 2.0 * tp / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    return {
        "pixel_accuracy": float((tp + tn) / max(tp + tn + fp + fn, eps)),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def bootstrap_delta(
    method_pred: np.ndarray,
    baseline_pred: np.ndarray,
    target: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict[str, float | int]:
    method_counts = per_sample_counts(method_pred, target)
    baseline_counts = per_sample_counts(baseline_pred, target)
    observed = dice_from_counts(method_counts.sum(axis=0)) - dice_from_counts(baseline_counts.sum(axis=0))
    rng = np.random.default_rng(seed)
    n = method_counts.shape[0]
    deltas = np.empty(int(reps), dtype=np.float32)
    for idx in range(int(reps)):
        sample_idx = rng.integers(0, n, size=n)
        method_sum = method_counts[sample_idx].sum(axis=0)
        baseline_sum = baseline_counts[sample_idx].sum(axis=0)
        deltas[idx] = dice_from_counts(method_sum) - dice_from_counts(baseline_sum)
    return {
        "unit": "test_sample",
        "reps": int(reps),
        "observed_delta_dice": float(observed),
        "mean_delta_dice": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "p_delta_le_0": float(np.mean(deltas <= 0.0)),
        "p_delta_le_0p05": float(np.mean(deltas <= 0.05)),
    }


def per_sample_counts(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    axes = tuple(range(1, pred_bool.ndim))
    tp = np.logical_and(pred_bool, target_bool).sum(axis=axes)
    fp = np.logical_and(pred_bool, ~target_bool).sum(axis=axes)
    fn = np.logical_and(~pred_bool, target_bool).sum(axis=axes)
    return np.stack([tp, fp, fn], axis=1).astype(np.float64)


def dice_from_counts(counts: np.ndarray) -> float:
    tp, fp, fn = [float(v) for v in counts[:3]]
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1e-9)


def write_per_sample(
    path: Path,
    names: Sequence[str],
    scores: np.ndarray,
    masks: np.ndarray,
    *,
    threshold: float,
    baseline_scores: np.ndarray,
    baseline_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        ensemble = metrics(scores[idx] >= threshold, masks[idx])
        baseline = metrics(baseline_scores[idx] >= baseline_threshold, masks[idx])
        rows.append(
            {
                "sample": name,
                "baseline_dice": baseline["dice"],
                "ensemble_dice": ensemble["dice"],
                "delta_dice": ensemble["dice"] - baseline["dice"],
                "baseline_positive": int((baseline_scores[idx] >= baseline_threshold).sum()),
                "ensemble_positive": int((scores[idx] >= threshold).sum()),
            }
        )
    write_csv(path, rows)
    return rows


def write_visuals(
    out_dir: Path,
    names: Sequence[str],
    masks: np.ndarray,
    scores: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    baseline_scores: np.ndarray,
    baseline_threshold: float,
    max_visual_samples: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"visualization skipped: {exc}", flush=True)
        return
    count = max(1, min(int(max_visual_samples), len(rows)))
    order = np.argsort([float(row["delta_dice"]) for row in rows])
    selected = (order[-count // 2 :][::-1].tolist() + order[: count - count // 2].tolist())[:count]
    cols = ["GT", "Baseline", "Ensemble score", "Ensemble mask"]
    fig, axes = plt.subplots(len(selected), len(cols), figsize=(2.25 * len(cols), 2.1 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]
    for r, idx in enumerate(selected):
        images = [
            masks[idx],
            baseline_scores[idx] >= baseline_threshold,
            scores[idx],
            scores[idx] >= threshold,
        ]
        for c, image in enumerate(images):
            ax = axes[r, c]
            ax.imshow(image, cmap="magma" if c == 2 else "gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
        axes[r, 0].set_ylabel(f"{names[idx]}\nD {float(rows[idx]['delta_dice']):+.3f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative_examples.png", dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    best = sorted(rows, key=lambda row: float(row["delta_dice"]), reverse=True)[:5]
    worst = sorted(rows, key=lambda row: float(row["delta_dice"]))[:5]
    lines = [
        "# Parametric Projection Prior Ensemble",
        "",
        summary["interpretation"],
        "",
        "## Test Dice",
        f"- Baseline: {summary['baseline']['dice']:.6f}",
        f"- Ensemble: {summary['selected_test_metrics']['dice']:.6f} ({summary['delta_vs_baseline']['dice']:+.6f})",
        f"- Selected combo: {summary['selected']['combo']}",
        f"- Fixed combo: {summary.get('fixed_combo', False)}",
        f"- Fuse baseline: {summary.get('fuse_baseline', False)}",
        f"- Fusion alpha: {summary['selected'].get('fusion_alpha', 1.0)}",
        f"- Selected threshold: {summary['selected']['val_threshold']}",
        (
            f"- Paired bootstrap delta CI: "
            f"{summary['paired_bootstrap']['ci95_low']:+.6f} to {summary['paired_bootstrap']['ci95_high']:+.6f}"
        ),
        "",
        "## Largest Per-Sample Gains",
    ]
    for row in best:
        lines.append(
            f"- {row['sample']}: baseline={float(row['baseline_dice']):.4f}, "
            f"ensemble={float(row['ensemble_dice']):.4f}, delta={float(row['delta_dice']):+.4f}"
        )
    lines.extend(["", "## Largest Regressions"])
    for row in worst:
        lines.append(
            f"- {row['sample']}: baseline={float(row['baseline_dice']):.4f}, "
            f"ensemble={float(row['ensemble_dice']):.4f}, delta={float(row['delta_dice']):+.4f}"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
