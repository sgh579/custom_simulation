from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.dataset import PalpationProcessDataset
from palpation_sim.workflow import require_runtime_environment, resolve_required_torch_cuda_device, with_run_date_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a flat MLP baseline for palpation segmentation.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with training .npz files.")
    parser.add_argument("--val-dir", type=Path, default=None, help="Optional directory with validation .npz files.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/validation_mlp"))
    parser.add_argument(
        "--exact-out-dir",
        action="store_true",
        help="Use --out-dir exactly instead of adding a date prefix under runs/.",
    )
    parser.add_argument("--input-mode", choices=["fz", "presses", "features", "auto"], default="fz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--architecture",
        choices=["flatten", "pixel"],
        default="flatten",
        help="flatten predicts the whole mask from a flattened map; pixel applies one MLP to each scan point.",
    )
    parser.add_argument("--hidden-dims", type=str, default="512,512")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep-min", type=float, default=0.1)
    parser.add_argument("--sweep-max", type=float, default=0.9)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument("--max-saved-predictions", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    args = parser.parse_args()

    require_runtime_environment()
    args.out_dir = with_run_date_prefix(args.out_dir, enabled=not args.exact_out_dir)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    _seed_everything(args.seed)

    if args.val_dir is None:
        train_files, val_files = _split_files(args.data_dir, args.val_fraction, args.seed)
    else:
        train_files = sorted(args.data_dir.glob("*.npz"))
        val_files = sorted(args.val_dir.glob("*.npz"))
    if not train_files:
        raise FileNotFoundError(f"No training .npz files found in {args.data_dir}")
    if not val_files:
        raise FileNotFoundError("MLP baseline needs validation files to report results.")

    train_ds = PalpationProcessDataset(train_files, input_mode=args.input_mode)
    val_ds = PalpationProcessDataset(val_files, input_mode=args.input_mode)
    sample_features, sample_target = train_ds[0]
    input_shape = tuple(int(dim) for dim in sample_features.shape)
    target_shape = tuple(int(dim) for dim in sample_target.shape)
    if len(target_shape) != 3 or target_shape[0] != 1:
        raise ValueError(f"Expected target shape [1, H, W], got {target_shape}")

    device = resolve_required_torch_cuda_device(torch, args.device)
    model = _build_model(args.architecture, input_shape, target_shape, hidden_dims, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "data_dir": str(args.data_dir),
        "val_dir": str(args.val_dir) if args.val_dir is not None else None,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "feature_shape_chw": list(input_shape),
        "target_shape_chw": list(target_shape),
        "input_dim": int(math.prod(input_shape)),
        "output_dim": int(math.prod(target_shape)),
        "architecture": _architecture_name(args.architecture),
        "architecture_mode": args.architecture,
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "input_mode": args.input_mode,
        "input_description": _input_description(args.input_mode),
        "device": str(device),
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "threshold": float(args.threshold),
        "sweep_min": float(args.sweep_min),
        "sweep_max": float(args.sweep_max),
        "sweep_step": float(args.sweep_step),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
    }
    _write_json(args.out_dir / "run_config.json", run_config)

    best_score = -1.0
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int | str]] = []
    start_time = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(model, train_loader, optimizer, device)
        val_stats = _run_epoch(model, val_loader, None, device)
        elapsed_seconds = time.monotonic() - start_time
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_dice": train_stats["dice"],
            "train_iou": train_stats["iou"],
            "val_loss": val_stats["loss"],
            "val_dice": val_stats["dice"],
            "val_iou": val_stats["iou"],
            "elapsed_seconds": elapsed_seconds,
        }
        history_rows.append(row)
        _write_history(args.out_dir / "history.csv", history_rows)

        score = val_stats["dice"]
        checkpoint = {
            "model_state": model.state_dict(),
            "architecture": _architecture_name(args.architecture),
            "architecture_mode": args.architecture,
            "input_shape_chw": input_shape,
            "target_shape_chw": target_shape,
            "hidden_dims": hidden_dims,
            "dropout": float(args.dropout),
            "input_mode": args.input_mode,
            "input_description": _input_description(args.input_mode),
            "run_config": run_config,
        }
        torch.save(checkpoint, args.out_dir / "last.pt")
        improved = score > best_score + float(args.early_stop_min_delta)
        if score > best_score:
            best_score = score
            torch.save(checkpoint, args.out_dir / "best.pt")
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d} elapsed_min={elapsed_seconds / 60.0:.2f} "
            f"train_loss={train_stats['loss']:.4f} train_dice={train_stats['dice']:.4f} "
            f"train_iou={train_stats['iou']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_dice={val_stats['dice']:.4f} val_iou={val_stats['iou']:.4f}"
        )
        if epochs_without_improvement >= max(int(args.early_stop_patience), 1):
            print(f"early stopping after epoch {epoch:03d}: no val Dice improvement for {epochs_without_improvement} epochs")
            break

    best_checkpoint = torch.load(args.out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    metrics = _evaluate(
        model,
        val_ds,
        device,
        args.out_dir,
        threshold=float(args.threshold),
        sweep_min=float(args.sweep_min),
        sweep_max=float(args.sweep_max),
        sweep_step=float(args.sweep_step),
        max_saved_predictions=int(args.max_saved_predictions),
    )
    metrics.update(
        {
            "checkpoint": str(args.out_dir / "best.pt"),
            "history": str(args.out_dir / "history.csv"),
            "run_config": str(args.out_dir / "run_config.json"),
            "input_mode": args.input_mode,
            "input_description": _input_description(args.input_mode),
            "architecture": _architecture_name(args.architecture),
            "architecture_mode": args.architecture,
            "hidden_dims": hidden_dims,
        }
    )
    _write_json(args.out_dir / "metrics_summary.json", metrics)
    print(json.dumps(metrics, indent=2))


class FlattenSegmentationMLP(nn.Module):
    """Flatten [C, H, W] inputs and predict a full [1, H, W] mask."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
        hidden_dims: list[int],
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = int(math.prod(input_shape))
        output_dim = int(math.prod(target_shape))
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(p=float(dropout)),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.target_shape = tuple(target_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x.reshape(x.shape[0], -1))
        return logits.reshape(x.shape[0], *self.target_shape)


class PixelSegmentationMLP(nn.Module):
    """Apply the same MLP independently to each scan point's channel vector."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
        hidden_dims: list[int],
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        channels, height, width = input_shape
        if target_shape != (1, height, width):
            raise ValueError(f"Expected target shape {(1, height, width)}, got {target_shape}")
        layers: list[nn.Module] = []
        previous_dim = int(channels)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(p=float(dropout)),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.net = nn.Sequential(*layers)
        self.height = int(height)
        self.width = int(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if height != self.height or width != self.width:
            raise ValueError(f"Expected spatial shape {(self.height, self.width)}, got {(height, width)}")
        point_features = x.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
        logits = self.net(point_features)
        return logits.reshape(batch, height, width, 1).permute(0, 3, 1, 2)


def _build_model(
    architecture: str,
    input_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    hidden_dims: list[int],
    *,
    dropout: float,
) -> nn.Module:
    if architecture == "flatten":
        return FlattenSegmentationMLP(input_shape, target_shape, hidden_dims, dropout=dropout)
    if architecture == "pixel":
        return PixelSegmentationMLP(input_shape, target_shape, hidden_dims, dropout=dropout)
    raise ValueError(f"Unknown architecture: {architecture}")


def _architecture_name(architecture: str) -> str:
    if architecture == "flatten":
        return "FlattenSegmentationMLP"
    if architecture == "pixel":
        return "PixelSegmentationMLP"
    return architecture


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    bce = nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0}
    count = 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = bce(logits, targets) + _dice_loss(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        metrics = _segmentation_metrics(logits.detach(), targets)
        batch_size = int(features.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["iou"] += metrics["iou"] * batch_size
        totals["dice"] += metrics["dice"] * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= 0.5).float()
        dims = tuple(range(1, preds.ndim))
        intersection = torch.sum(preds * targets, dim=dims)
        union = torch.sum((preds + targets) > 0, dim=dims).float()
        pred_sum = torch.sum(preds, dim=dims)
        target_sum = torch.sum(targets, dim=dims)
        iou = ((intersection + eps) / (union + eps)).mean()
        dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    return {"iou": float(iou.cpu()), "dice": float(dice.cpu())}


def _evaluate(
    model: nn.Module,
    dataset: Any,
    device: torch.device,
    out_dir: Path,
    *,
    threshold: float,
    sweep_min: float,
    sweep_max: float,
    sweep_step: float,
    max_saved_predictions: int,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, float | int | str]] = []
    global_counts = _empty_counts()
    sweep_thresholds = _make_thresholds(sweep_min, sweep_max, sweep_step)
    sweep_counts = {value: _empty_counts() for value in sweep_thresholds}
    with torch.no_grad():
        for idx, path in enumerate(dataset.files):
            features, target = dataset[idx]
            logits = model(features.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            gt = target[0].numpy().astype(np.uint8)
            pred = (prob >= threshold).astype(np.uint8)

            counts = _counts(pred, gt)
            _add_counts(global_counts, counts)
            rows.append({"sample": path.name, **_metrics_from_counts(counts)})
            for sweep_threshold in sweep_thresholds:
                _add_counts(sweep_counts[sweep_threshold], _counts((prob >= sweep_threshold).astype(np.uint8), gt))

            if idx < max_saved_predictions:
                stem = path.stem
                np.save(out_dir / f"{stem}_prob.npy", prob.astype(np.float32))
                np.save(out_dir / f"{stem}_pred.npy", pred.astype(np.uint8))

    _write_per_sample(out_dir / "metrics_per_sample.csv", rows)
    sweep_rows = [{"threshold": value, **_metrics_from_counts(counts)} for value, counts in sweep_counts.items()]
    best_sweep = max(sweep_rows, key=lambda row: float(row["dice"])) if sweep_rows else None
    _write_json(out_dir / "threshold_sweep.json", {"best": best_sweep, "thresholds": sweep_rows})

    summary = _metrics_from_counts(global_counts)
    summary.update(
        {
            "num_samples": len(dataset),
            "threshold": float(threshold),
            "data_dir": str(Path(dataset.files[0]).parent) if dataset.files else "",
            "threshold_sweep_best": best_sweep,
        }
    )
    return summary


def _counts(pred: Any, gt: Any) -> dict[str, int]:
    pred_bool = np.asarray(pred).astype(bool)
    gt_bool = np.asarray(gt).astype(bool)
    return {
        "tp": int(np.logical_and(pred_bool, gt_bool).sum()),
        "tn": int(np.logical_and(~pred_bool, ~gt_bool).sum()),
        "fp": int(np.logical_and(pred_bool, ~gt_bool).sum()),
        "fn": int(np.logical_and(~pred_bool, gt_bool).sum()),
    }


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


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


def _split_files(data_dir: Path, val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    val_count = max(1, int(round(len(files) * val_fraction))) if len(files) > 1 else 0
    return files[val_count:], files[:val_count]


def _parse_hidden_dims(value: str) -> list[int]:
    dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not dims:
        raise SystemExit("--hidden-dims must contain at least one integer.")
    if any(dim <= 0 for dim in dims):
        raise SystemExit("--hidden-dims must be positive integers.")
    return dims


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _input_description(input_mode: str) -> str:
    if input_mode in {"fz", "auto"}:
        return "raw Fz trajectory channels [T, H, W]"
    if input_mode == "presses":
        return "raw indentation and Fz press channels [2*T, H, W]"
    if input_mode == "features":
        return "engineered mechanical feature maps [C, H, W]"
    return input_mode


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2)


def _write_history(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_per_sample(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample", "pixel_accuracy", "precision", "recall", "dice", "iou", "tp", "tn", "fp", "fn"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value

if __name__ == "__main__":
    main()
