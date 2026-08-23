from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize palpation WM / JEPA / VISReg experiment evidence.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("WM/reports"))
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(full_context_rows(root))
    rows.extend(sparse_context_rows(root))
    rows.extend(sparse_context_repeat_ci_rows(root))
    rows.extend(volume_projection_rows(root))
    rows.extend(jepa_rows(root))
    write_csv(out_dir / "wm_summary_table.csv", rows)
    write_report(out_dir / "wm_current_results.md", rows)
    print(f"wrote {out_dir / 'wm_current_results.md'}")
    print(f"wrote {out_dir / 'wm_summary_table.csv'}")


def full_context_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_path = root / "runs/nonlinear_trajectory_20x_repeats10_seed20260618/highres_wrench_temporal_variants/leaderboard_with_baselines.csv"
    fusion_path = root / "runs/wrench_world_fusion_20260622/leaderboard.csv"
    baseline = find_row(read_csv(baseline_path), "method", "r128_wrench_temporal_cnn32_unet")
    baseline_dice = as_float(baseline.get("test_val_selected_dice")) if baseline else None
    if baseline:
        rows.append(row("full_context_2d", "temporal_cnn32_unet", baseline_dice, "baseline", baseline_path))
    for item in read_csv(fusion_path):
        dice = as_float(item.get("test_val_selected_dice"))
        rows.append(
            row(
                "full_context_2d",
                item.get("method", ""),
                dice,
                f"delta_vs_temporal_cnn32={format_delta(dice, baseline_dice)} alpha_world={item.get('alpha_world', '')}",
                fusion_path,
            )
        )
    return rows


def sparse_context_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    world_paths = [
        root / "runs/wrench_temporal_cnn_world_20260622/world_model_metrics.csv",
        root / "WM/runs/sparse_baseline_core_20260701/world_model_metrics.csv",
    ]
    sparse_by_ratio: dict[float, dict[str, float]] = {}
    for path in world_paths:
        for item in read_csv(path):
            if item.get("world_prediction_type") != "sparse_context_segmentation":
                continue
            ratio = as_float(item.get("sparse_context_ratio"))
            dice = as_float(item.get("sparse_context_fixed_dice"))
            if ratio is None or dice is None:
                continue
            method = item.get("method", "")
            sparse_by_ratio.setdefault(ratio, {})[method] = dice
            rows.append(row("sparse_context_2d", f"{method}@{ratio:g}", dice, "fixed_threshold=0.5", path))
    for ratio, values in sorted(sparse_by_ratio.items()):
        baseline = values.get("r128_wrench_temporal_cnn_sparse_unet")
        for method, dice in sorted(values.items()):
            if method == "r128_wrench_temporal_cnn_sparse_unet" or baseline is None:
                continue
            rows.append(
                row(
                    "sparse_context_delta",
                    f"{method}@{ratio:g} vs sparse_unet",
                    dice - baseline,
                    f"world={dice:.6f} baseline={baseline:.6f}",
                    "computed",
                )
            )
    return rows


def sparse_context_repeat_ci_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = root / "WM/reports/sparse_context_repeats/sparse_context_delta_ci.csv"
    for item in read_csv(path):
        value = as_float(item.get("mean_aggregate_dice_delta"))
        lo = as_float(item.get("sample_bootstrap_ci_low"))
        hi = as_float(item.get("sample_bootstrap_ci_high"))
        note = (
            f"sample_mean_delta={as_float(item.get('mean_sample_dice_delta')):.6f} "
            f"sample_bootstrap_95ci=[{lo:.6f},{hi:.6f}] seeds={item.get('num_seeds', '')}"
            if value is not None and lo is not None and hi is not None
            else "missing repeated-eval CI"
        )
        rows.append(
            row(
                "sparse_context_repeated_ci",
                f"{item.get('method', '')}@{item.get('ratio', '')} vs {item.get('baseline', '')}",
                value,
                note,
                path,
            )
        )
    return rows


def volume_projection_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [
        root / "runs/wrench_3d_volume_depth_aware_train_20260622/leaderboard.csv",
        root / "runs/wrench_3d_volume_depth_aware_fusion_20260622/leaderboard.csv",
    ]:
        for item in read_csv(path):
            dice = as_float(item.get("test_val_selected_projection_dice"))
            method = item.get("method", "")
            if dice is not None:
                rows.append(row("3d_to_2d_projection", method, dice, "projection Dice from predicted volume", path))
    return rows


def jepa_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [
        root / "WM/runs/jepa_visreg_core_20260701/leaderboard.csv",
    ]:
        for item in read_csv(path):
            method = item.get("method", "")
            if "jepa" not in method and "visreg" not in method:
                continue
            dice = as_float(item.get("test_val_selected_dice"))
            rows.append(row("jepa_visreg_2d", method, dice, f"heldout_latent_mse={item.get('heldout_latent_mse', '')}", path))
    return rows


def row(task: str, method: str, value: float | None, note: str, source: Path | str) -> dict[str, Any]:
    return {
        "task": task,
        "method": method,
        "value": "" if value is None else f"{value:.6f}",
        "note": note,
        "source": str(source),
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# WM / JEPA / VISReg Palpation Results",
        "",
        f"Generated: {now}",
        "",
        "## Summary Table",
        "",
        "| Task | Method | Value | Note |",
        "|---|---:|---:|---|",
    ]
    for item in rows:
        lines.append(f"| {item['task']} | `{item['method']}` | {item['value']} | {item['note']} |")
    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            "- Full-context 2D WM fusion is useful but modest; it should be treated as supporting evidence, not the main 5-point claim.",
            "- The main WM task should be sparse-context palpation: same object and split, fewer observed probe locations, full-mask prediction.",
            "- The sparse-context delta rows are the key decision gate: keep the WM contribution only if the action-conditioned world objective beats the no-world sparse baseline by at least 0.05 Dice on validation-selected test metrics.",
            "- The 3D-to-2D projection rows support the mechanistic story that latent 3D occupancy contains useful segmentation signal, even when full voxel Dice remains harder than projected Dice.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["task", "method", "value", "note", "source"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str] | None:
    for item in rows:
        if item.get(key) == value:
            return item
    return None


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "n/a"
    return f"{value - baseline:+.6f}"


if __name__ == "__main__":
    main()
