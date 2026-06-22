from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from palpation_sim.native_data import load_phantom_scan_material_lumps
from run_highres_segmentation_sweep import (
    MODEL_RESIZE_POLICY,
    STIFFNESS_INPUT_POLICY,
    highres_mask_for_scan_area,
)
from run_segmentation_accuracy_sweep import FEATURE_NAMES, _load_displacement_hwt, _load_fz_hwt, _load_nonlinearity_map, extract_mechanical_features


DEFAULT_SWEEP_DIR = Path("runs/highres_segmentation_sweep_128_800_80_80")
DEFAULT_PACKAGE_DIR = Path("runs/dataset_packages/palpation_random_shapes_20x_800train_80val_80test_20260616")
DEFAULT_OUT_DIR_NAME = "test_render"
ROW_GROUPS = (
    ("fz", "unet"),
    ("stiffness", "unet"),
    ("fz", "shallow_cnn"),
    ("stiffness", "shallow_cnn"),
    ("fz", "mlp"),
    ("stiffness", "mlp"),
)


@dataclass(frozen=True)
class MethodInfo:
    method: str
    resolution: int
    input_name: str
    model: str
    val_best_dice: float
    test_dice: float
    threshold: float
    path: Path


@dataclass(frozen=True)
class SampleCase:
    index: int
    name: str
    dice: float
    category: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Render held-out test predictions for the high-res segmentation sweep.")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--primary-method", type=str, default="r128_fz_unet")
    parser.add_argument("--cases-per-band", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=16)
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    package_dir = args.package_dir
    out_dir = args.out_dir or sweep_dir / DEFAULT_OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = read_leaderboard(sweep_dir / "leaderboard.csv")
    methods = {row.method: row for row in leaderboard}
    if args.primary_method not in methods:
        raise SystemExit(f"Primary method not found in leaderboard: {args.primary_method}")

    test_files = sorted((package_dir / "data" / "test").glob("*.npz"))
    if not test_files:
        raise FileNotFoundError(f"No test .npz files found under {package_dir / 'data' / 'test'}")

    primary = methods[args.primary_method]
    primary_metrics = read_sample_metrics(primary.path / "test_metrics_per_sample.csv")
    cases = choose_cases(primary_metrics, args.cases_per_band)
    write_case_selection(out_dir / "case_selection.csv", cases)

    resolutions = sorted({int(row.resolution) for row in leaderboard})
    canonical_size = int(max(resolutions))
    gt_masks = load_gt_masks(test_files, canonical_size)
    peak_fz = load_peak_fz_maps(test_files)
    stiffness_maps = load_stiffness_maps(test_files)

    primary_scores = np.load(primary.path / "test_scores.npy")
    render_top_methods_chart(leaderboard, out_dir / "top_test_methods.png", top_n=args.top_n)
    render_unet_resolution_chart(leaderboard, out_dir / "unet_resolution_test.png", resolutions=resolutions)
    render_primary_case_grid(
        out_dir / "primary_worst_cases.png",
        "Worst held-out cases",
        primary,
        cases_by_category(cases, "worst"),
        test_files,
        gt_masks,
        peak_fz,
        stiffness_maps,
        primary_scores,
    )
    render_primary_case_grid(
        out_dir / "primary_median_cases.png",
        "Median held-out cases",
        primary,
        cases_by_category(cases, "median"),
        test_files,
        gt_masks,
        peak_fz,
        stiffness_maps,
        primary_scores,
    )
    render_primary_case_grid(
        out_dir / "primary_best_cases.png",
        "Best held-out cases",
        primary,
        cases_by_category(cases, "best"),
        test_files,
        gt_masks,
        peak_fz,
        stiffness_maps,
        primary_scores,
    )

    representative = [
        cases_by_category(cases, "worst")[0],
        cases_by_category(cases, "median")[len(cases_by_category(cases, "median")) // 2],
        cases_by_category(cases, "best")[0],
    ]
    method_scores = load_method_scores(leaderboard)
    method_metrics = load_all_sample_metrics(leaderboard)
    for case in representative:
        render_all_methods_sample_grid(
            out_dir / f"all_methods_{case.category}_{Path(case.name).stem}.png",
            case,
            methods,
            method_scores,
            method_metrics,
            gt_masks[case.index],
            resolutions,
        )

    write_html_report(out_dir / "index.html", sweep_dir, package_dir, leaderboard, primary, cases, representative)
    print(f"rendered test report: {out_dir / 'index.html'}")


def read_leaderboard(path: Path) -> list[MethodInfo]:
    rows: list[MethodInfo] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                MethodInfo(
                    method=row["method"],
                    resolution=int(row["resolution"]),
                    input_name=row["input"],
                    model=row["model"],
                    val_best_dice=float(row["val_best_dice"]),
                    test_dice=float(row["test_val_selected_dice"]),
                    threshold=float(row["test_val_selected_threshold"]),
                    path=Path(row["path"]),
                )
            )
    rows.sort(key=lambda item: item.test_dice, reverse=True)
    return rows


def read_sample_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for index, row in enumerate(csv.DictReader(f)):
            rows.append(
                {
                    "index": index,
                    "sample": row["sample"],
                    "fixed_dice": float(row["fixed_dice"]),
                    "val_selected_dice": float(row["val_selected_dice"]),
                    "gt_positive": int(row["gt_positive"]),
                }
            )
    return rows


def choose_cases(rows: list[dict[str, Any]], per_band: int) -> list[SampleCase]:
    ordered = sorted(rows, key=lambda row: row["val_selected_dice"])
    per_band = max(1, min(int(per_band), len(ordered) // 3))
    worst = ordered[:per_band]
    center = len(ordered) // 2
    half = per_band // 2
    start = max(0, center - half)
    median = ordered[start : start + per_band]
    best = list(reversed(ordered[-per_band:]))
    cases: list[SampleCase] = []
    for category, group in (("worst", worst), ("median", median), ("best", best)):
        for row in group:
            cases.append(SampleCase(int(row["index"]), str(row["sample"]), float(row["val_selected_dice"]), category))
    return cases


def cases_by_category(cases: list[SampleCase], category: str) -> list[SampleCase]:
    return [case for case in cases if case.category == category]


def write_case_selection(path: Path, cases: list[SampleCase]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "index", "sample", "primary_val_selected_dice"])
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "category": case.category,
                    "index": case.index,
                    "sample": case.name,
                    "primary_val_selected_dice": f"{case.dice:.8f}",
                }
            )


def load_gt_masks(test_files: list[Path], label_size: int) -> np.ndarray:
    masks: list[np.ndarray] = []
    for path in test_files:
        phantom, scan, _material, lumps, _metadata = load_phantom_scan_material_lumps(
            sample_path=path,
            metadata_path=path.with_name(f"{path.stem}_gt.json"),
        )
        masks.append(highres_mask_for_scan_area(scan, phantom, lumps, label_size).astype(np.uint8))
    return np.stack(masks)


def load_peak_fz_maps(test_files: list[Path]) -> np.ndarray:
    maps: list[np.ndarray] = []
    for path in test_files:
        with np.load(path, allow_pickle=False) as sample:
            fz_hwt = _load_fz_hwt(sample)
        maps.append(np.nanmax(np.abs(fz_hwt), axis=-1).astype(np.float32))
    return np.stack(maps)


def load_stiffness_maps(test_files: list[Path]) -> np.ndarray:
    maps: list[np.ndarray] = []
    stiffness_idx = FEATURE_NAMES.index("equivalent_stiffness")
    for path in test_files:
        with np.load(path, allow_pickle=False) as sample:
            fz_hwt = _load_fz_hwt(sample)
            z_hwt = _load_displacement_hwt(sample, fz_hwt.shape)
            nonlinearity = _load_nonlinearity_map(sample, fz_hwt.shape[:2])
        features = extract_mechanical_features(z_hwt, fz_hwt, nonlinearity)
        maps.append(np.nan_to_num(features[stiffness_idx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    return np.stack(maps).astype(np.float32)


def load_method_scores(leaderboard: list[MethodInfo]) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for method in leaderboard:
        scores_path = method.path / "test_scores.npy"
        if scores_path.exists():
            scores[method.method] = np.load(scores_path)
    return scores


def load_all_sample_metrics(leaderboard: list[MethodInfo]) -> dict[str, dict[str, dict[str, float]]]:
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for method in leaderboard:
        metrics_path = method.path / "test_metrics_per_sample.csv"
        if not metrics_path.exists():
            continue
        rows: dict[str, dict[str, float]] = {}
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[row["sample"]] = {
                    "val_selected_dice": float(row["val_selected_dice"]),
                    "fixed_dice": float(row["fixed_dice"]),
                }
        all_metrics[method.method] = rows
    return all_metrics


def render_top_methods_chart(leaderboard: list[MethodInfo], path: Path, *, top_n: int) -> None:
    top = leaderboard[:top_n]
    labels = [item.method.replace("_", "\n") for item in reversed(top)]
    values = [item.test_dice for item in reversed(top)]
    colors = [method_color(item) for item in reversed(top)]
    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 0.38 * len(top))))
    ax.barh(range(len(top)), values, color=colors, edgecolor="#2b2b2b", linewidth=0.5)
    ax.set_yticks(range(len(top)), labels, fontsize=8)
    ax.set_xlim(max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.025))
    ax.set_xlabel("Test Dice using validation-selected threshold")
    ax.set_title("Held-out test Dice leaderboard")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    for i, value in enumerate(values):
        ax.text(value + 0.003, i, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_unet_resolution_chart(leaderboard: list[MethodInfo], path: Path, *, resolutions: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    plotted_values: list[float] = []
    for input_name, color in (("fz", "#1f77b4"), ("stiffness", "#2ca02c")):
        rows = sorted(
            [row for row in leaderboard if row.model == "unet" and row.input_name == input_name],
            key=lambda row: row.resolution,
        )
        if not rows:
            continue
        plotted_values.extend(row.test_dice for row in rows)
        ax.plot(
            [row.resolution for row in rows],
            [row.test_dice for row in rows],
            marker="o",
            linewidth=2.0,
            color=color,
            label=input_name,
        )
        for row in rows:
            ax.text(row.resolution, row.test_dice + 0.002, f"{row.test_dice:.3f}", ha="center", fontsize=8)
    ax.set_xticks(resolutions)
    if plotted_values:
        ymin = max(0.0, min(plotted_values) - 0.025)
        ymax = min(1.0, max(plotted_values) + 0.025)
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Output label resolution")
    ax.set_ylabel("Test Dice")
    ax.set_title("U-Net test Dice by output resolution")
    ax.grid(color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_primary_case_grid(
    path: Path,
    title: str,
    method: MethodInfo,
    cases: list[SampleCase],
    test_files: list[Path],
    gt_masks: np.ndarray,
    peak_fz: np.ndarray,
    stiffness_maps: np.ndarray,
    scores: np.ndarray,
) -> None:
    n_rows = len(cases)
    columns = [f"stiffness{stiffness_maps.shape[-1]}", "peak |Fz|", "GT", "probability", f"pred @ {method.threshold:.2f}", "TP / FP / FN"]
    fig, axes = plt.subplots(n_rows, len(columns), figsize=(15.8, max(3.2, 2.05 * n_rows)), squeeze=False)
    fig.suptitle(f"{title}: {method.method} on test split", fontsize=13)
    for row_index, case in enumerate(cases):
        gt = gt_masks[case.index]
        score = scores[case.index]
        pred = score >= method.threshold
        panels = [
            (stiffness_maps[case.index], "magma", None),
            (peak_fz[case.index], "magma", None),
            (gt, "gray_r", (0, 1)),
            (score, "viridis", (0, 1)),
            (pred.astype(float), "gray_r", (0, 1)),
            (error_overlay(pred, gt), None, None),
        ]
        for col_index, (image, cmap, clim) in enumerate(panels):
            ax = axes[row_index, col_index]
            if cmap is None:
                ax.imshow(image, interpolation="nearest")
            else:
                ax.imshow(image, cmap=cmap, interpolation="nearest", vmin=None if clim is None else clim[0], vmax=None if clim is None else clim[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(columns[col_index], fontsize=9)
            if col_index == 0:
                ax.set_ylabel(
                    f"{Path(test_files[case.index]).stem}\nDice {case.dice:.3f}",
                    rotation=0,
                    labelpad=44,
                    va="center",
                    ha="right",
                    fontsize=8,
                )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_all_methods_sample_grid(
    path: Path,
    case: SampleCase,
    methods: dict[str, MethodInfo],
    method_scores: dict[str, np.ndarray],
    method_metrics: dict[str, dict[str, dict[str, float]]],
    gt_mask: np.ndarray,
    resolutions: list[int],
) -> None:
    columns = ["GT"] + [str(resolution) for resolution in resolutions]
    fig, axes = plt.subplots(len(ROW_GROUPS), len(columns), figsize=(max(6.8, 2.35 * len(columns)), 12.2), squeeze=False)
    fig.suptitle(f"All model outputs for {Path(case.name).stem}: {case.category}, primary Dice {case.dice:.3f}", fontsize=13)
    for row_index, (input_name, model_name) in enumerate(ROW_GROUPS):
        for col_index, column in enumerate(columns):
            ax = axes[row_index, col_index]
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(column, fontsize=9)
            if col_index == 0:
                ax.imshow(gt_mask, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
                ax.set_ylabel(f"{input_name}\n{model_name}", rotation=0, labelpad=40, va="center", ha="right", fontsize=8)
                continue
            resolution = int(column)
            method_name = f"r{resolution}_{input_name}_{model_name}"
            method = methods.get(method_name)
            scores = method_scores.get(method_name)
            if method is None or scores is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue
            pred = scores[case.index] >= method.threshold
            dice = method_metrics.get(method_name, {}).get(case.name, {}).get("val_selected_dice", np.nan)
            ax.imshow(pred.astype(float), cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(f"{method_name}\nDice {dice:.3f}", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def error_overlay(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    image = np.full((*gt_bool.shape, 3), 0.95, dtype=np.float32)
    image[gt_bool & pred_bool] = np.array([0.12, 0.55, 0.28], dtype=np.float32)
    image[~gt_bool & pred_bool] = np.array([0.85, 0.22, 0.18], dtype=np.float32)
    image[gt_bool & ~pred_bool] = np.array([0.12, 0.32, 0.85], dtype=np.float32)
    return image


def method_color(method: MethodInfo) -> str:
    if method.model == "unet" and method.input_name == "fz":
        return "#4477aa"
    if method.model == "unet":
        return "#228833"
    if method.model == "shallow_cnn" and method.input_name == "fz":
        return "#cc6677"
    if method.model == "shallow_cnn":
        return "#aa3377"
    if method.input_name == "fz":
        return "#ee9944"
    return "#bbbbbb"


def write_html_report(
    path: Path,
    sweep_dir: Path,
    package_dir: Path,
    leaderboard: list[MethodInfo],
    primary: MethodInfo,
    cases: list[SampleCase],
    representative: list[SampleCase],
) -> None:
    top_rows = "\n".join(
        "<tr>"
        f"<td>{i}</td>"
        f"<td>{html.escape(row.method)}</td>"
        f"<td>{row.resolution}</td>"
        f"<td>{html.escape(row.input_name)}</td>"
        f"<td>{html.escape(row.model)}</td>"
        f"<td>{row.val_best_dice:.4f}</td>"
        f"<td>{row.test_dice:.4f}</td>"
        f"<td>{row.threshold:.2f}</td>"
        "</tr>"
        for i, row in enumerate(leaderboard, start=1)
    )
    case_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(case.category)}</td>"
        f"<td>{case.index}</td>"
        f"<td>{html.escape(case.name)}</td>"
        f"<td>{case.dice:.4f}</td>"
        "</tr>"
        for case in cases
    )
    representative_imgs = "\n".join(
        f'<figure><img src="all_methods_{case.category}_{Path(case.name).stem}.png" alt="all methods {html.escape(case.name)}">'
        f"<figcaption>All methods: {html.escape(case.category)} case {html.escape(case.name)}</figcaption></figure>"
        for case in representative
    )
    payload = {
        "sweep_dir": str(sweep_dir),
        "package_dir": str(package_dir),
        "primary_method": primary.method,
        "primary_threshold": primary.threshold,
    }
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>High-res segmentation test render</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f6f4;
      color: #1e2428;
    }}
    body {{
      margin: 0;
      padding: 32px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 32px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    p, li {{
      line-height: 1.55;
    }}
    code {{
      background: #e8e6df;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    figure {{
      margin: 18px 0 28px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid #d3d0c8;
      background: white;
    }}
    figcaption {{
      margin-top: 7px;
      color: #565e64;
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #dedbd3;
      padding: 7px 8px;
      text-align: left;
    }}
    th {{
      background: #ece9e2;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .table-scroll {{
      max-height: 560px;
      overflow: auto;
      border: 1px solid #d3d0c8;
      background: white;
    }}
    .meta {{
      color: #565e64;
      margin: 0 0 18px;
    }}
  </style>
</head>
<body>
<main>
  <h1>High-res segmentation test render</h1>
  <p class="meta">Primary method <code>{html.escape(primary.method)}</code>, validation-selected threshold <code>{primary.threshold:.2f}</code>. Test split size is 80 samples.</p>

  <h2>Summary</h2>
  <p>Stiffness-map methods use the native 20x20 equivalent-stiffness map as input. {html.escape(STIFFNESS_INPUT_POLICY)} {html.escape(MODEL_RESIZE_POLICY)}</p>
  <figure><img src="top_test_methods.png" alt="top test methods"><figcaption>Top methods ranked by held-out test Dice.</figcaption></figure>
  <figure><img src="unet_resolution_test.png" alt="unet resolution chart"><figcaption>U-Net test Dice across output label resolutions.</figcaption></figure>

  <h2>Primary Method Cases</h2>
  <figure><img src="primary_worst_cases.png" alt="primary worst cases"><figcaption>Lowest Dice cases for the primary model. Error colors: green TP, red FP, blue FN.</figcaption></figure>
  <figure><img src="primary_median_cases.png" alt="primary median cases"><figcaption>Middle Dice cases for the primary model.</figcaption></figure>
  <figure><img src="primary_best_cases.png" alt="primary best cases"><figcaption>Highest Dice cases for the primary model.</figcaption></figure>

  <h2>All-Method Sample Grids</h2>
  {representative_imgs}

  <h2>Leaderboard</h2>
  <div class="table-scroll">
  <table>
    <thead><tr><th>Rank</th><th>Method</th><th>Resolution</th><th>Input</th><th>Model</th><th>Val Dice</th><th>Test Dice</th><th>Threshold</th></tr></thead>
    <tbody>{top_rows}</tbody>
  </table>
  </div>

  <h2>Rendered Cases</h2>
  <div class="table-scroll">
  <table>
    <thead><tr><th>Band</th><th>Index</th><th>Sample</th><th>Primary Dice</th></tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
  </div>

  <script type="application/json" id="render-config">{html.escape(json.dumps(payload, indent=2))}</script>
</main>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
