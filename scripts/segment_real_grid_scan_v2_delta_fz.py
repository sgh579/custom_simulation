from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_highres_segmentation_sweep import UpsampleLogits  # noqa: E402
from run_segmentation_accuracy_sweep import ValidationUNet, predict_neural, preload_subtract_fz  # noqa: E402
from palpation_sim.workflow import require_runtime_environment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment a real 20x20 UR grid scan with the V2 delta-Fz U-Net.")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--source-train-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-steps", type=int, default=40)
    parser.add_argument(
        "--source-depth-steps",
        type=int,
        default=0,
        help="0 uses --input-steps; otherwise use this many leading V2 source depth channels for normalization.",
    )
    parser.add_argument("--v2-max-depth-m", type=float, default=0.036)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--model-label", type=str, default="V2 delta-Fz model")
    args = parser.parse_args()

    require_runtime_environment()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_summary = read_json(args.source_metrics)
    threshold = float(source_summary.get("threshold_sweep_best", {}).get("threshold", 0.8))

    print("loading V2 train normalization statistics", flush=True)
    source_steps = int(args.source_depth_steps) if int(args.source_depth_steps) > 0 else int(args.input_steps)
    source_train = load_v2_fz_split(args.source_train_dir, depth_steps=source_steps)
    source_delta = preload_subtract_fz(source_train)
    source_mean = np.nanmean(source_delta, axis=(0, 2, 3), keepdims=True)
    source_std = np.nanstd(source_delta, axis=(0, 2, 3), keepdims=True)
    if source_mean.shape[1] != int(args.input_steps):
        raise ValueError(
            f"source normalization has {source_mean.shape[1]} channels but model input uses {int(args.input_steps)}"
        )

    print("loading real grid scan", flush=True)
    real = load_real_grid_scan(args.scan_dir)
    physical_name = f"physical{args.v2_max_depth_m * 1000.0:.1f}mm_hold".replace(".", "p")
    variants = {
        physical_name: resample_real_response(real, args.input_steps, policy="physical_hold", max_depth_m=args.v2_max_depth_m),
        "stretch10": resample_real_response(real, args.input_steps, policy="stretch", max_depth_m=args.v2_max_depth_m),
    }

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = UpsampleLogits(ValidationUNet(int(args.input_steps), base_channels=24), (int(args.resolution), int(args.resolution)))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    rows = []
    for name, response_hwt in variants.items():
        print(f"predicting {name}", flush=True)
        x = np.moveaxis(response_hwt, -1, 0)[None].astype(np.float32)
        x = preload_subtract_fz(x)
        x_norm = (x - source_mean) / (source_std + np.float32(1e-6))
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        score = predict_neural(model, x_norm, device, batch_size=int(args.batch_size))[0]
        mask = score >= threshold
        variant_dir = args.out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        np.save(variant_dir / "response_hwt.npy", response_hwt.astype(np.float32))
        np.save(variant_dir / "input_delta_fz_chw.npy", x[0].astype(np.float32))
        np.save(variant_dir / "score_128x128.npy", score.astype(np.float32))
        np.save(variant_dir / "mask_128x128_threshold_source.npy", mask.astype(np.uint8))
        render_variant(
            variant_dir / "segmentation_overview.png",
            variant=name,
            model_label=args.model_label,
            scan_label=args.scan_dir.name,
            response_hwt=response_hwt,
            max_depth_m=real["max_depth_m"],
            score=score,
            mask=mask,
            threshold=threshold,
        )
        rows.append(
            {
                "variant": name,
                "threshold": threshold,
                "score_min": float(np.nanmin(score)),
                "score_median": float(np.nanmedian(score)),
                "score_max": float(np.nanmax(score)),
                "mask_positive_pixels": int(mask.sum()),
                "mask_positive_fraction": float(mask.mean()),
                "path": str(variant_dir),
            }
        )

    write_csv(args.out_dir / "leaderboard.csv", rows, list(rows[0].keys()))
    write_json(
        args.out_dir / "summary.json",
        {
            "scan_dir": str(args.scan_dir),
            "source_train_dir": str(args.source_train_dir),
            "checkpoint": str(args.checkpoint),
            "source_metrics": str(args.source_metrics),
            "source_threshold": threshold,
            "input_steps": int(args.input_steps),
            "source_depth_steps": int(source_steps),
            "v2_max_depth_m": float(args.v2_max_depth_m),
            "model_label": args.model_label,
            "real_grid": {
                "rows": int(real["rows"]),
                "cols": int(real["cols"]),
                "valid_points": int(real["valid_points"]),
                "max_depth_m_min": float(np.nanmin(real["max_depth_m"])),
                "max_depth_m_median": float(np.nanmedian(real["max_depth_m"])),
                "max_depth_m_max": float(np.nanmax(real["max_depth_m"])),
                "final_response_n_median": float(np.nanmedian(real["final_response_n"])),
                "final_response_n_max": float(np.nanmax(real["final_response_n"])),
            },
            "variants": rows,
        },
    )
    print(f"wrote {args.out_dir}", flush=True)


def load_v2_fz_split(split_dir: Path, *, depth_steps: int) -> np.ndarray:
    items = []
    for path in sorted(split_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as sample:
            fz = np.asarray(sample["fz"], dtype=np.float32)
            if fz.ndim != 3:
                raise ValueError(f"{path}: expected fz[H,W,T], got {fz.shape}")
            if int(depth_steps) <= 0 or int(depth_steps) > fz.shape[-1]:
                raise ValueError(f"{path}: invalid depth_steps={int(depth_steps)} for fz shape {fz.shape}")
            fz = fz[..., : int(depth_steps)]
            items.append(np.moveaxis(fz, -1, 0))
    if not items:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")
    return np.stack(items).astype(np.float32)


def load_real_grid_scan(scan_dir: Path) -> dict[str, Any]:
    spec = read_json(scan_dir / "grid_scan_spec.json")
    rows = int(spec["rows"])
    cols = int(spec["cols"])
    response_curves: list[list[list[tuple[np.ndarray, np.ndarray] | None]]] = [[ [None for _ in range(cols)] for _ in range(rows) ]][0]
    max_depth = np.full((rows, cols), np.nan, dtype=np.float32)
    final_response = np.full((rows, cols), np.nan, dtype=np.float32)
    valid_points = 0
    with (scan_dir / "press_index.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("valid_for_analysis", "")).lower() != "true":
                continue
            r = int(row["grid_row"])
            c = int(row["grid_col"])
            press_npz = scan_dir / row["press_npz"]
            press_json = scan_dir / row["press_json"]
            depth, response = load_press_response(press_npz, press_json)
            if depth.size < 2:
                continue
            response_curves[r][c] = (depth, response)
            max_depth[r, c] = float(np.nanmax(depth))
            final_response[r, c] = float(response[np.nanargmax(depth)])
            valid_points += 1
    if valid_points != rows * cols:
        raise ValueError(f"Expected full grid {rows * cols}, got {valid_points} valid points")
    return {
        "rows": rows,
        "cols": cols,
        "curves": response_curves,
        "max_depth_m": max_depth,
        "final_response_n": final_response,
        "valid_points": valid_points,
    }


def load_press_response(press_npz: Path, press_json: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(press_npz, allow_pickle=False) as sample:
        depth = np.asarray(sample["depth"], dtype=np.float32)
        fz = np.asarray(sample["fz"], dtype=np.float32)
    meta = read_json(press_json)
    ranges = [item for item in meta.get("sample_phase_ranges", []) if item.get("phase") == "press"]
    if ranges:
        start = int(ranges[0]["start_row"])
        end = int(ranges[0]["end_row_exclusive"])
        depth = depth[start:end]
        fz = fz[start:end]
    finite = np.isfinite(depth) & np.isfinite(fz)
    depth = depth[finite]
    fz = fz[finite]
    if depth.size == 0:
        return depth.astype(np.float32), fz.astype(np.float32)
    dmin = float(np.nanmin(depth))
    dmax = float(np.nanmax(depth))
    shallow = depth <= dmin + 0.02 * max(dmax - dmin, 1.0e-9)
    baseline = float(np.nanmedian(fz[shallow])) if np.any(shallow) else float(fz[0])
    response = np.maximum(baseline - fz, 0.0).astype(np.float32)
    order = np.argsort(depth)
    depth = depth[order].astype(np.float32)
    response = response[order].astype(np.float32)
    unique_depth, unique_idx = np.unique(depth, return_index=True)
    return unique_depth.astype(np.float32), response[unique_idx].astype(np.float32)


def resample_real_response(real: dict[str, Any], steps: int, *, policy: str, max_depth_m: float) -> np.ndarray:
    rows = int(real["rows"])
    cols = int(real["cols"])
    out = np.zeros((rows, cols, int(steps)), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            curve = real["curves"][r][c]
            if curve is None:
                continue
            depth, response = curve
            if policy == "physical_hold":
                target_depth = np.linspace(0.0, float(max_depth_m), int(steps), dtype=np.float32)
                out[r, c] = np.interp(target_depth, depth, response, left=0.0, right=float(response[-1])).astype(np.float32)
            elif policy == "stretch":
                target_depth = np.linspace(0.0, float(depth[-1]), int(steps), dtype=np.float32)
                out[r, c] = np.interp(target_depth, depth, response, left=0.0, right=float(response[-1])).astype(np.float32)
            else:
                raise ValueError(f"unknown policy: {policy}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def render_variant(
    path: Path,
    *,
    variant: str,
    model_label: str,
    scan_label: str,
    response_hwt: np.ndarray,
    max_depth_m: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> None:
    final_response = response_hwt[..., -1]
    max_response = np.max(response_hwt, axis=-1)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = [
        (final_response, "final response input (N)", "viridis"),
        (max_response, "max response input (N)", "viridis"),
        (max_depth_m * 1000.0, "measured max depth (mm)", "magma"),
        (score, f"{model_label} score", "magma"),
        (mask.astype(np.float32), f"mask @ {threshold:g}", "gray"),
        (score * mask.astype(np.float32), "score inside mask", "magma"),
    ]
    for ax, (image, title, cmap) in zip(axes.reshape(-1), panels):
        im = ax.imshow(image, origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("grid col / +x")
        ax.set_ylabel("grid row / +y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Real grid scan {scan_label}, {model_label}, {variant}", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(data), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


if __name__ == "__main__":
    main()
