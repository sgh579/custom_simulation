from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation U-Net and save visual outputs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/validation_unet_eval"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", action="store_true", help="Also evaluate a threshold sweep.")
    parser.add_argument("--sweep-min", type=float, default=0.1)
    parser.add_argument("--sweep-max", type=float, default=0.9)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    _load_dependencies()

    device = _resolve_device(args.device)
    dataset = PalpationProcessDataset(args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ValidationUNet(
        in_channels=int(checkpoint["in_channels"]),
        base_channels=int(checkpoint.get("base_channels", 24)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    global_counts = _empty_counts()
    sweep_thresholds = _make_thresholds(args.sweep_min, args.sweep_max, args.sweep_step) if args.sweep_thresholds else []
    sweep_counts = {threshold: _empty_counts() for threshold in sweep_thresholds}
    visuals = []

    for idx, path in enumerate(dataset.files):
        features, target = dataset[idx]
        with torch.no_grad():
            logits = model(features.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        gt = target[0].numpy().astype(np.uint8)
        pred = (prob >= args.threshold).astype(np.uint8)

        counts = _counts(pred, gt)
        _add_counts(global_counts, counts)
        metrics = _metrics_from_counts(counts)
        rows.append({"sample": path.name, **metrics})
        for threshold in sweep_thresholds:
            _add_counts(sweep_counts[threshold], _counts((prob >= threshold).astype(np.uint8), gt))

        if idx < args.max_images:
            stem = path.stem
            baseline_stiffness = _load_equivalent_stiffness_map(path, gt.shape)
            np.save(args.out_dir / f"{stem}_prob.npy", prob.astype(np.float32))
            np.save(args.out_dir / f"{stem}_pred.npy", pred)
            np.save(args.out_dir / f"{stem}_baseline_stiffness.npy", baseline_stiffness.astype(np.float32))
            _save_stiffness_figure(args.out_dir / f"{stem}_baseline_stiffness.png", baseline_stiffness, path.name)
            _save_sample_figure(
                args.out_dir / f"{stem}_comparison.png",
                prob,
                pred,
                gt,
                baseline_stiffness,
                path.name,
                metrics,
            )
            visuals.append((path.name, prob, pred, gt, baseline_stiffness, metrics))

    summary = _metrics_from_counts(global_counts)
    summary["num_samples"] = len(dataset)
    summary["threshold"] = args.threshold
    summary["checkpoint"] = str(args.checkpoint)
    summary["data_dir"] = str(args.data_dir)
    summary["baseline_stiffness"] = "Equivalent stiffness map k=(F_peak-F_start)/(disp_peak-disp_start) is saved for visualized samples."
    summary["gt_note"] = "Metrics and visualizations use the scan-grid mask, which is the target used for training loss."

    with (args.out_dir / "metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (args.out_dir / "metrics_per_sample.csv").open("w", newline="") as f:
        fieldnames = ["sample", "pixel_accuracy", "precision", "recall", "dice", "iou", "tp", "tn", "fp", "fn"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if sweep_counts:
        sweep_rows = []
        for threshold, counts in sweep_counts.items():
            sweep_rows.append({"threshold": threshold, **_metrics_from_counts(counts)})
        best_row = max(sweep_rows, key=lambda row: float(row["dice"]))
        with (args.out_dir / "threshold_sweep.json").open("w") as f:
            json.dump({"best": best_row, "thresholds": sweep_rows}, f, indent=2)
        summary["threshold_sweep_best"] = best_row
        with (args.out_dir / "metrics_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

    if visuals:
        _save_contact_sheet(args.out_dir / "prediction_contact_sheet.png", visuals)

    print(json.dumps(summary, indent=2))


def _load_dependencies() -> None:
    global PalpationProcessDataset, ValidationUNet, np, plt, torch
    try:
        import matplotlib.pyplot as plt_mod
        import numpy as np_mod
        import torch as torch_mod

        from palpation_sim.dataset import PalpationProcessDataset as dataset_cls
        from palpation_sim.models import ValidationUNet as model_cls
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch/Numpy/Matplotlib dependencies are required for evaluation. Use: "
            "/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/evaluate_validation_unet.py ..."
        ) from exc

    np = np_mod
    plt = plt_mod
    torch = torch_mod
    PalpationProcessDataset = dataset_cls
    ValidationUNet = model_cls


def _resolve_device(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def _make_thresholds(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise SystemExit("--sweep-step must be positive.")
    lo = float(min(start, stop))
    hi = float(max(start, stop))
    count = int(np.floor((hi - lo) / step + 1e-9)) + 1
    values = [round(lo + i * step, 6) for i in range(count)]
    if not values or values[-1] < hi - 1e-9:
        values.append(round(hi, 6))
    return values


def _counts(pred, gt) -> dict[str, int]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    return {
        "tp": int(np.logical_and(pred_bool, gt_bool).sum()),
        "tn": int(np.logical_and(~pred_bool, ~gt_bool).sum()),
        "fp": int(np.logical_and(pred_bool, ~gt_bool).sum()),
        "fn": int(np.logical_and(~pred_bool, gt_bool).sum()),
    }


def _add_counts(total: dict[str, int], new: dict[str, int]) -> None:
    for key in total:
        total[key] += new[key]


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    eps = 1e-8
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    dice = 2.0 * tp / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, eps)
    return {
        "pixel_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _load_equivalent_stiffness_map(path: Path, fallback_shape: tuple[int, ...]):
    with np.load(path) as sample:
        if "presses" in sample:
            return _equivalent_stiffness_map(sample["presses"])
        if "indentation_depth" in sample and "fz" in sample:
            presses = np.stack([sample["indentation_depth"], sample["fz"]], axis=-1)
            return _equivalent_stiffness_map(presses)
    return np.zeros(fallback_shape, dtype=np.float32)


def _equivalent_stiffness_map(presses):
    presses = np.asarray(presses, dtype=np.float32)
    if presses.ndim != 4 or presses.shape[-1] < 2:
        raise ValueError(f"Expected presses shape [H, W, T, 2], got {presses.shape}")
    h, w, _, _ = presses.shape
    stiffness = np.zeros((h, w), dtype=np.float32)
    for row in range(h):
        for col in range(w):
            stiffness[row, col] = _equivalent_stiffness(presses[row, col, :, 0], presses[row, col, :, 1])
    return np.nan_to_num(stiffness, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _equivalent_stiffness(displacement, force) -> float:
    z_raw = np.asarray(displacement, dtype=np.float32)
    f_raw = np.asarray(force, dtype=np.float32)
    valid = np.isfinite(z_raw) & np.isfinite(f_raw)
    if int(valid.sum()) < 2:
        return 0.0
    z = z_raw[valid] - np.float32(z_raw[valid][0])
    f = f_raw[valid] - np.float32(f_raw[valid][0])
    if abs(float(np.nanmin(z))) > abs(float(np.nanmax(z))):
        z = -z
    if abs(float(np.nanmin(f))) > abs(float(np.nanmax(f))):
        f = -f

    peak_idx = int(np.nanargmax(z))
    loading_z = z[: peak_idx + 1]
    loading_f = f[: peak_idx + 1]
    if loading_z.size < 2:
        loading_z = z
        loading_f = f

    peak_idx = int(np.nanargmax(loading_z))
    displacement_delta = float(loading_z[peak_idx] - loading_z[0])
    if abs(displacement_delta) < 1e-9:
        return 0.0
    force_delta = float(loading_f[peak_idx] - loading_f[0])
    return max(force_delta / displacement_delta, 0.0)


def _stiffness_limits(stiffness) -> tuple[float, float]:
    finite = np.asarray(stiffness, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2.0, 98.0])
    lo = float(max(lo, 0.0))
    hi = float(hi)
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.1, 1.0)
    return lo, hi


def _save_sample_figure(
    path: Path,
    prob,
    pred,
    gt,
    baseline_stiffness,
    title: str,
    metrics: dict[str, float | int],
) -> None:
    ncols = 4
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.0), constrained_layout=True)
    panels = [("GT", gt, "gray", 0.0, 1.0)]
    panels.extend([("Prob", prob, "viridis", 0.0, 1.0), ("Pred", pred, "gray", 0.0, 1.0)])
    for ax, (name, data, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    k_vmin, k_vmax = _stiffness_limits(baseline_stiffness)
    image = axes[-1].imshow(baseline_stiffness, cmap="magma", vmin=k_vmin, vmax=k_vmax, interpolation="nearest")
    axes[-1].set_title("Baseline k")
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])
    fig.colorbar(image, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle(f"{title}  Dice={metrics['dice']:.3f} IoU={metrics['iou']:.3f} Acc={metrics['pixel_accuracy']:.3f}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_stiffness_figure(path: Path, stiffness, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.2), constrained_layout=True)
    k_vmin, k_vmax = _stiffness_limits(stiffness)
    image = ax.imshow(stiffness, cmap="magma", vmin=k_vmin, vmax=k_vmax, interpolation="nearest")
    ax.set_title(f"{title}\nequivalent stiffness")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_contact_sheet(path: Path, visuals) -> None:
    n = len(visuals)
    ncols = 4
    fig, axes = plt.subplots(n, ncols, figsize=(2.8 * ncols, max(2.0, 2.25 * n)), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    for row, (name, prob, pred, gt, baseline_stiffness, metrics) in enumerate(visuals):
        k_vmin, k_vmax = _stiffness_limits(baseline_stiffness)
        panels = [(gt, "GT", "gray", 0.0, 1.0)]
        panels.extend(
            [
                (prob, "Prob", "viridis", 0.0, 1.0),
                (pred, "Pred", "gray", 0.0, 1.0),
                (baseline_stiffness, "Baseline k", "magma", k_vmin, k_vmax),
            ]
        )
        for col, (data, panel_name, cmap, vmin, vmax) in enumerate(panels):
            axes[row, col].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            title = panel_name if col > 0 else f"{name}\nDice={metrics['dice']:.3f}"
            axes[row, col].set_title(title, fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
