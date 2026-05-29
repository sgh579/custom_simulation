from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
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


def split_files(data_dir: Path, val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    val_count = max(1, int(round(len(files) * val_fraction))) if len(files) > 1 else 0
    return files[val_count:], files[:val_count]


def run_epoch(
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
        features = features.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = bce(logits, targets) + dice_loss(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        metrics = segmentation_metrics(logits.detach(), targets)
        batch_size = int(features.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["iou"] += metrics["iou"] * batch_size
        totals["dice"] += metrics["dice"] * batch_size
        count += batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train validation U-Net on palpation process data.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with training .npz files.")
    parser.add_argument("--val-dir", type=Path, default=None, help="Optional directory with validation .npz files.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/validation_unet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Optional wall-clock training budget. Training stops after the first completed epoch past this budget.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="Minimum completed epochs before honoring --max-minutes.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop after this many epochs without validation/train score improvement.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum score improvement required to reset early-stop patience.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    _load_ml_dependencies()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.val_dir is None:
        train_files, val_files = split_files(args.data_dir, args.val_fraction, args.seed)
    else:
        train_files = sorted(args.data_dir.glob("*.npz"))
        val_files = sorted(args.val_dir.glob("*.npz"))

    train_ds = PalpationProcessDataset(train_files)
    val_ds = PalpationProcessDataset(val_files) if val_files else None
    sample_features, _ = train_ds[0]
    device = resolve_device(args.device)

    model = ValidationUNet(sample_features.shape[0], base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_ds is not None
        else None
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "data_dir": str(args.data_dir),
        "val_dir": str(args.val_dir) if args.val_dir is not None else None,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds is not None else 0,
        "feature_shape_chw": list(sample_features.shape),
        "device": str(device),
        "epochs_requested": args.epochs,
        "max_minutes": args.max_minutes,
        "min_epochs": args.min_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_channels": args.base_channels,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "feature_names": list(FEATURE_NAMES),
    }
    with (args.out_dir / "run_config.json").open("w") as config_file:
        json.dump(run_config, config_file, indent=2)

    best_dice = -1.0
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int | str]] = []
    start_time = time.monotonic()
    max_seconds = None if args.max_minutes is None else max(float(args.max_minutes), 0.0) * 60.0
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device)
        val_stats = run_epoch(model, val_loader, None, device) if val_loader is not None else None
        elapsed_seconds = time.monotonic() - start_time

        message = (
            f"epoch {epoch:03d} "
            f"elapsed_min={elapsed_seconds / 60.0:.2f} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_dice={train_stats['dice']:.4f} "
            f"train_iou={train_stats['iou']:.4f}"
        )
        if val_stats is not None:
            message += (
                f" val_loss={val_stats['loss']:.4f} "
                f"val_dice={val_stats['dice']:.4f} "
                f"val_iou={val_stats['iou']:.4f}"
            )
        print(message)

        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_dice": train_stats["dice"],
            "train_iou": train_stats["iou"],
            "val_loss": "",
            "val_dice": "",
            "val_iou": "",
            "elapsed_seconds": elapsed_seconds,
        }
        if val_stats is not None:
            row.update({"val_loss": val_stats["loss"], "val_dice": val_stats["dice"], "val_iou": val_stats["iou"]})
        history_rows.append(row)

        with (args.out_dir / "history.csv").open("w", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history_rows)

        score = val_stats["dice"] if val_stats is not None else train_stats["dice"]
        checkpoint = {
            "model_state": model.state_dict(),
            "in_channels": int(sample_features.shape[0]),
            "base_channels": args.base_channels,
            "feature_names": FEATURE_NAMES,
        }
        torch.save(checkpoint, args.out_dir / "last.pt")
        previous_best = best_dice
        improved = score > previous_best + float(args.early_stop_min_delta)
        if score > previous_best:
            best_dice = score
            torch.save(checkpoint, args.out_dir / "best.pt")
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if max_seconds is not None and epoch >= max(args.min_epochs, 1) and elapsed_seconds >= max_seconds:
            print(
                f"stopping after epoch {epoch:03d}: "
                f"elapsed_min={elapsed_seconds / 60.0:.2f} reached max_minutes={args.max_minutes:.2f}"
            )
            break
        if (
            args.early_stop_patience is not None
            and epoch >= max(args.min_epochs, 1)
            and epochs_without_improvement >= max(int(args.early_stop_patience), 1)
        ):
            print(
                f"early stopping after epoch {epoch:03d}: "
                f"no score improvement for {epochs_without_improvement} epochs"
            )
            break

def _load_ml_dependencies() -> None:
    global DataLoader, FEATURE_NAMES, PalpationProcessDataset, ValidationUNet, nn, torch
    try:
        import torch as torch_mod
        from torch import nn as nn_mod
        from torch.utils.data import DataLoader as data_loader_cls

        from palpation_sim.dataset import PalpationProcessDataset as dataset_cls
        from palpation_sim.features import FEATURE_NAMES as feature_names
        from palpation_sim.models import ValidationUNet as model_cls
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch/Numpy dependencies are required for training. Install them with: "
            "python -m pip install -r requirements-ml.txt"
        ) from exc

    torch = torch_mod
    nn = nn_mod
    DataLoader = data_loader_cls
    PalpationProcessDataset = dataset_cls
    ValidationUNet = model_cls
    FEATURE_NAMES = feature_names


if __name__ == "__main__":
    main()
