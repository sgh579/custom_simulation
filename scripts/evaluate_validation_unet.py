from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.workflow import require_runtime_environment, resolve_required_torch_cuda_device, with_run_date_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation U-Net and save visual outputs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/validation_unet_eval"))
    parser.add_argument(
        "--exact-out-dir",
        action="store_true",
        help="Use --out-dir exactly instead of adding a date prefix under runs/.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", action="store_true", help="Also evaluate a threshold sweep.")
    parser.add_argument("--sweep-min", type=float, default=0.1)
    parser.add_argument("--sweep-max", type=float, default=0.9)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument(
        "--no-analytic-phantom-3d",
        action="store_true",
        help="Do not write per-sample analytic 3D phantom/lump geometry PNGs.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Required CUDA device, e.g. cuda or cuda:0.")
    args = parser.parse_args()
    require_runtime_environment()
    args.out_dir = with_run_date_prefix(args.out_dir, enabled=not args.exact_out_dir)

    _load_dependencies()

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    input_mode = str(checkpoint.get("input_mode", "features"))
    dataset = PalpationProcessDataset(args.data_dir, input_mode=input_mode)
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
            baseline_kmeans = _kmeans_high_stiffness_mask(baseline_stiffness)
            np.save(args.out_dir / f"{stem}_prob.npy", prob.astype(np.float32))
            np.save(args.out_dir / f"{stem}_pred.npy", pred)
            np.save(args.out_dir / f"{stem}_baseline_stiffness.npy", baseline_stiffness.astype(np.float32))
            np.save(args.out_dir / f"{stem}_baseline_kmeans_pred.npy", baseline_kmeans)
            _save_stiffness_figure(args.out_dir / f"{stem}_baseline_stiffness.png", baseline_stiffness, path.name)
            if not args.no_analytic_phantom_3d:
                _save_analytic_phantom_figure(args.out_dir / f"{stem}_analytic_phantom_3d.png", path)
            _save_sample_figure(
                args.out_dir / f"{stem}_comparison.png",
                prob,
                pred,
                gt,
                baseline_stiffness,
                baseline_kmeans,
                path.name,
                metrics,
            )
            visuals.append((path.name, prob, pred, gt, baseline_stiffness, baseline_kmeans, metrics))

    summary = _metrics_from_counts(global_counts)
    summary["num_samples"] = len(dataset)
    summary["threshold"] = args.threshold
    summary["checkpoint"] = str(args.checkpoint)
    summary["data_dir"] = str(args.data_dir)
    summary["input_mode"] = input_mode
    summary["input_description"] = str(checkpoint.get("input_description", input_mode))
    summary["baseline_stiffness"] = (
        "Equivalent stiffness map k=(F_peak-F_start)/(disp_peak-disp_start) and its two-cluster "
        "high-stiffness mask are saved for visualized samples."
    )
    summary["analytic_phantom_3d"] = (
        "Per-sample analytic phantom/lump distribution PNGs are saved for visualized samples unless "
        "--no-analytic-phantom-3d is passed."
    )
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
        import matplotlib
        import numpy as np_mod
        import torch as torch_mod

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod

        from palpation_sim.dataset import PalpationProcessDataset as dataset_cls
        from palpation_sim.models import ValidationUNet as model_cls
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch/Numpy/Matplotlib dependencies are required for evaluation. "
            "Run this script from the conda environment 'palpation'."
        ) from exc

    np = np_mod
    plt = plt_mod
    torch = torch_mod
    PalpationProcessDataset = dataset_cls
    ValidationUNet = model_cls


def _resolve_device(name: str):
    return resolve_required_torch_cuda_device(torch, name)


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


def _kmeans_high_stiffness_mask(stiffness) -> np.ndarray:
    values = np.asarray(stiffness, dtype=np.float32)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    result = np.zeros(values.shape, dtype=np.uint8)
    if finite_values.size == 0:
        return result

    c0, c1 = np.percentile(finite_values, [25.0, 75.0]).astype(np.float32)
    if abs(float(c1 - c0)) < 1e-6:
        c0 = float(np.nanmin(finite_values))
        c1 = float(np.nanmax(finite_values))
    if abs(float(c1 - c0)) < 1e-6:
        return result

    labels = np.zeros(finite_values.shape, dtype=np.uint8)
    for _ in range(30):
        d0 = np.abs(finite_values - c0)
        d1 = np.abs(finite_values - c1)
        new_labels = (d1 < d0).astype(np.uint8)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        if np.any(labels == 0):
            c0 = float(np.mean(finite_values[labels == 0]))
        if np.any(labels == 1):
            c1 = float(np.mean(finite_values[labels == 1]))

    high_label = 1 if c1 >= c0 else 0
    result[finite_mask] = (labels == high_label).astype(np.uint8)
    return result


def _save_sample_figure(
    path: Path,
    prob,
    pred,
    gt,
    baseline_stiffness,
    baseline_kmeans,
    title: str,
    metrics: dict[str, float | int],
) -> None:
    ncols = 5
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.0), constrained_layout=True)
    panels = [("GT", gt, "gray", 0.0, 1.0)]
    panels.extend([("Prob", prob, "viridis", 0.0, 1.0), ("Pred", pred, "gray", 0.0, 1.0)])
    for ax, (name, data, cmap, vmin, vmax) in zip(axes[:3], panels):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    k_vmin, k_vmax = _stiffness_limits(baseline_stiffness)
    image = axes[3].imshow(baseline_stiffness, cmap="magma", vmin=k_vmin, vmax=k_vmax, interpolation="nearest")
    axes[3].set_title("Stiffness map")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(image, ax=axes[3], fraction=0.046, pad=0.04)
    axes[4].imshow(baseline_kmeans, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[4].set_title("K-means mask")
    axes[4].set_xticks([])
    axes[4].set_yticks([])
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


def _save_analytic_phantom_figure(path: Path, sample_path: Path) -> None:
    phantom, lumps = _load_analytic_geometry(sample_path)
    sx = float(phantom.get("size_x", 0.18))
    sy = float(phantom.get("size_y", 0.18))
    height = float(phantom.get("height", 0.08))
    fig = plt.figure(figsize=(6.6, 5.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    _draw_phantom_box(ax, sx, sy, height)
    colors = ["#d94f3d", "#1687d9", "#eba21a", "#3fa662", "#9367c7", "#d56a9f"]
    for idx, lump in enumerate(lumps):
        color = colors[idx % len(colors)]
        _draw_lump_surface(ax, lump, color=color)
        center = tuple(float(v) for v in lump["center"])
        ax.text(
            center[0],
            center[1],
            center[2],
            f"{idx} {lump.get('shape', '')}\n{float(lump.get('stiffness_multiplier', 0.0)):.0f}x",
            color="#111111",
            fontsize=8,
            ha="center",
            va="center",
        )

    ax.set_title(f"{sample_path.stem}: analytic phantom/lump geometry", fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-46)
    _set_equal_3d(ax, (-0.5 * sx, 0.5 * sx), (-0.5 * sy, 0.5 * sy), (0.0, height))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_analytic_geometry(sample_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    with np.load(sample_path) as sample:
        phantom = json.loads(_sample_string(sample["phantom_json"])) if "phantom_json" in sample else {}
        lumps = json.loads(_sample_string(sample["lumps_json"])) if "lumps_json" in sample else []
    return phantom, list(lumps)


def _sample_string(value) -> str:
    item = np.asarray(value).reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _draw_phantom_box(ax, sx: float, sy: float, height: float) -> None:
    x0, x1 = -0.5 * sx, 0.5 * sx
    y0, y1 = -0.5 * sy, 0.5 * sy
    z0, z1 = 0.0, height
    corners = {
        "000": (x0, y0, z0),
        "100": (x1, y0, z0),
        "110": (x1, y1, z0),
        "010": (x0, y1, z0),
        "001": (x0, y0, z1),
        "101": (x1, y0, z1),
        "111": (x1, y1, z1),
        "011": (x0, y1, z1),
    }
    edges = [
        ("000", "100"),
        ("100", "110"),
        ("110", "010"),
        ("010", "000"),
        ("001", "101"),
        ("101", "111"),
        ("111", "011"),
        ("011", "001"),
        ("000", "001"),
        ("100", "101"),
        ("110", "111"),
        ("010", "011"),
    ]
    for a, b in edges:
        xs, ys, zs = zip(corners[a], corners[b])
        ax.plot(xs, ys, zs, color="#6d8796", linewidth=1.0, alpha=0.82)
    xx, yy = np.meshgrid([x0, x1], [y0, y1])
    zz = np.full_like(xx, z1, dtype=float)
    ax.plot_surface(xx, yy, zz, color="#8fc7e8", alpha=0.08, linewidth=0.0, shade=False)


def _draw_lump_surface(ax, lump: dict[str, object], *, color: str) -> None:
    shape = str(lump.get("shape", ""))
    center = np.asarray(lump.get("center", (0.0, 0.0, 0.0)), dtype=float)
    radii = np.maximum(np.asarray(lump.get("radii", (0.01, 0.01, 0.01)), dtype=float), 1.0e-9)
    yaw = float(lump.get("yaw", 0.0))
    if shape in {"sphere", "ellipsoid"}:
        x, y, z = _ellipsoid_surface(center, radii, yaw)
        _surface(ax, x, y, z, color)
    elif shape == "cylinder":
        for x, y, z in _cylinder_surfaces(center, radii, yaw):
            _surface(ax, x, y, z, color)
    elif shape == "box":
        _draw_box_lump(ax, center, radii, yaw, color)
    elif shape == "capsule":
        for x, y, z in _capsule_surfaces(center, radii, yaw):
            _surface(ax, x, y, z, color)
    else:
        x, y, z = _ellipsoid_surface(center, radii, yaw)
        _surface(ax, x, y, z, color)


def _ellipsoid_surface(center: np.ndarray, radii: np.ndarray, yaw: float, *, n_u: int = 36, n_v: int = 18):
    u = np.linspace(0.0, 2.0 * math.pi, n_u)
    v = np.linspace(0.0, math.pi, n_v)
    uu, vv = np.meshgrid(u, v)
    local = np.stack(
        [
            radii[0] * np.cos(uu) * np.sin(vv),
            radii[1] * np.sin(uu) * np.sin(vv),
            radii[2] * np.cos(vv),
        ],
        axis=-1,
    )
    world = _rotate_translate(local, center, yaw)
    return world[..., 0], world[..., 1], world[..., 2]


def _cylinder_surfaces(center: np.ndarray, radii: np.ndarray, yaw: float):
    theta = np.linspace(0.0, 2.0 * math.pi, 40)
    zz = np.linspace(-radii[2], radii[2], 12)
    tt, z_grid = np.meshgrid(theta, zz)
    side = np.stack([radii[0] * np.cos(tt), radii[1] * np.sin(tt), z_grid], axis=-1)
    top_r = np.linspace(0.0, 1.0, 10)
    tt_cap, rr = np.meshgrid(theta, top_r)
    top = np.stack([radii[0] * rr * np.cos(tt_cap), radii[1] * rr * np.sin(tt_cap), np.full_like(rr, radii[2])], axis=-1)
    bottom = np.stack([top[..., 0], top[..., 1], np.full_like(rr, -radii[2])], axis=-1)
    return [_surface_xyz(surface, center, yaw) for surface in (side, top, bottom)]


def _capsule_surfaces(center: np.ndarray, radii: np.ndarray, yaw: float):
    radius = float(radii[0])
    half_axis = float(radii[2])
    theta = np.linspace(0.0, 2.0 * math.pi, 36)
    z_body = np.linspace(-half_axis, half_axis, 10)
    tt, zz = np.meshgrid(theta, z_body)
    body = np.stack([radius * np.cos(tt), radius * np.sin(tt), zz], axis=-1)
    u = np.linspace(0.0, 2.0 * math.pi, 36)
    phi_top = np.linspace(0.0, 0.5 * math.pi, 12)
    uu, pp = np.meshgrid(u, phi_top)
    top = np.stack(
        [radius * np.cos(uu) * np.cos(pp), radius * np.sin(uu) * np.cos(pp), half_axis + radius * np.sin(pp)],
        axis=-1,
    )
    bottom = np.stack([top[..., 0], top[..., 1], -top[..., 2]], axis=-1)
    return [_surface_xyz(surface, center, yaw) for surface in (body, top, bottom)]


def _draw_box_lump(ax, center: np.ndarray, radii: np.ndarray, yaw: float, color: str) -> None:
    xs = [-radii[0], radii[0]]
    ys = [-radii[1], radii[1]]
    zs = [-radii[2], radii[2]]
    faces = [
        np.asarray([[[xs[0], y, z] for y in ys] for z in zs], dtype=float),
        np.asarray([[[xs[1], y, z] for y in ys] for z in zs], dtype=float),
        np.asarray([[[x, ys[0], z] for x in xs] for z in zs], dtype=float),
        np.asarray([[[x, ys[1], z] for x in xs] for z in zs], dtype=float),
        np.asarray([[[x, y, zs[0]] for x in xs] for y in ys], dtype=float),
        np.asarray([[[x, y, zs[1]] for x in xs] for y in ys], dtype=float),
    ]
    for face in faces:
        x, y, z = _surface_xyz(face, center, yaw)
        _surface(ax, x, y, z, color)


def _surface_xyz(local: np.ndarray, center: np.ndarray, yaw: float):
    world = _rotate_translate(local, center, yaw)
    return world[..., 0], world[..., 1], world[..., 2]


def _rotate_translate(local: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    world = np.array(local, copy=True, dtype=float)
    x = local[..., 0]
    y = local[..., 1]
    world[..., 0] = c * x - s * y + center[0]
    world[..., 1] = s * x + c * y + center[1]
    world[..., 2] = local[..., 2] + center[2]
    return world


def _surface(ax, x, y, z, color: str) -> None:
    ax.plot_surface(x, y, z, color=color, alpha=0.66, linewidth=0.2, edgecolor="#1f1f1f", shade=True)


def _set_equal_3d(ax, xlim: tuple[float, float], ylim: tuple[float, float], zlim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    except AttributeError:
        pass


def _save_contact_sheet(path: Path, visuals) -> None:
    n = len(visuals)
    ncols = 5
    fig, axes = plt.subplots(n, ncols, figsize=(2.8 * ncols, max(2.0, 2.25 * n)), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    for row, (name, prob, pred, gt, baseline_stiffness, baseline_kmeans, metrics) in enumerate(visuals):
        k_vmin, k_vmax = _stiffness_limits(baseline_stiffness)
        panels = [(gt, "GT", "gray", 0.0, 1.0)]
        panels.extend(
            [
                (prob, "Prob", "viridis", 0.0, 1.0),
                (pred, "Pred", "gray", 0.0, 1.0),
                (baseline_stiffness, "Stiffness map", "magma", k_vmin, k_vmax),
                (baseline_kmeans, "K-means mask", "gray", 0.0, 1.0),
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
