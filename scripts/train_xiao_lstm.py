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

from palpation_sim.workflow import require_runtime_environment, resolve_required_torch_cuda_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Xiao et al. 2020-style LSTM depth classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with training .npz files.")
    parser.add_argument("--val-dir", type=Path, default=None, help="Optional directory with validation .npz files.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/xiao_lstm"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-decay", type=float, default=0.2)
    parser.add_argument("--lr-decay-epochs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=70)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--normalize", choices=["none", "sample", "dataset"], default="dataset")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help="Required CUDA device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Optional wall-clock training budget. Stops after the first completed epoch past this budget.",
    )
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    args = parser.parse_args()
    require_runtime_environment()

    _load_ml_dependencies()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.val_dir is None:
        train_files, val_files = split_files(args.data_dir, args.val_fraction, args.seed)
    else:
        train_files = sorted(args.data_dir.glob("*.npz"))
        val_files = sorted(args.val_dir.glob("*.npz"))
    if not train_files:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir}")

    train_ds = XiaoDepthSequenceDataset(train_files, sequence_length=args.sequence_length, normalize=args.normalize)
    val_ds = (
        XiaoDepthSequenceDataset(
            val_files,
            sequence_length=args.sequence_length,
            normalize=args.normalize,
            sequence_mean=train_ds.sequence_mean,
            sequence_std=train_ds.sequence_std,
        )
        if val_files
        else None
    )
    sample_sequence, _ = train_ds[0]
    device = resolve_device(args.device)
    model = Xiao2020DepthLSTM(
        input_size=int(sample_sequence.shape[-1]),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(train_ds.class_depths_mm),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(int(args.lr_decay_epochs), 1),
        gamma=float(args.lr_decay),
    )
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
        "sequence_shape": list(sample_sequence.shape),
        "class_depths_mm": list(train_ds.class_depths_mm),
        "device": str(device),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "lr_decay_epochs": args.lr_decay_epochs,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "sequence_length": args.sequence_length,
        "normalize": args.normalize,
        "sequence_mean": None if train_ds.sequence_mean is None else train_ds.sequence_mean.reshape(-1).tolist(),
        "sequence_std": None if train_ds.sequence_std is None else train_ds.sequence_std.reshape(-1).tolist(),
        "seed": args.seed,
    }
    with (args.out_dir / "run_config.json").open("w") as config_file:
        json.dump(run_config, config_file, indent=2)

    best_score = -1.0
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int | str]] = []
    start_time = time.monotonic()
    max_seconds = None if args.max_minutes is None else max(float(args.max_minutes), 0.0) * 60.0
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device)
        val_stats = run_epoch(model, val_loader, None, device) if val_loader is not None else None
        scheduler.step()
        elapsed_seconds = time.monotonic() - start_time
        current_lr = float(optimizer.param_groups[0]["lr"])

        message = (
            f"epoch {epoch:03d} "
            f"elapsed_min={elapsed_seconds / 60.0:.2f} "
            f"lr={current_lr:.6g} "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy']:.4f}"
        )
        if val_stats is not None:
            message += f" val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f}"
        print(message)

        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": "",
            "val_accuracy": "",
            "elapsed_seconds": elapsed_seconds,
        }
        if val_stats is not None:
            row.update({"val_loss": val_stats["loss"], "val_accuracy": val_stats["accuracy"]})
        history_rows.append(row)
        with (args.out_dir / "history.csv").open("w", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history_rows)

        score = val_stats["accuracy"] if val_stats is not None else train_stats["accuracy"]
        checkpoint = {
            "model_state": model.state_dict(),
            "input_size": int(sample_sequence.shape[-1]),
            "hidden_size": int(args.hidden_size),
            "num_layers": int(args.num_layers),
            "num_classes": len(train_ds.class_depths_mm),
            "class_depths_mm": list(train_ds.class_depths_mm),
            "sequence_length": int(args.sequence_length),
            "normalize": args.normalize,
            "sequence_mean": None if train_ds.sequence_mean is None else train_ds.sequence_mean.reshape(-1).tolist(),
            "sequence_std": None if train_ds.sequence_std is None else train_ds.sequence_std.reshape(-1).tolist(),
            "dropout": float(args.dropout),
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
                f"no accuracy improvement for {epochs_without_improvement} epochs"
            )
            break


def split_files(data_dir: Path, val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    val_count = max(1, int(round(len(files) * val_fraction))) if len(files) > 1 else 0
    return files[val_count:], files[:val_count]


def run_epoch(
    model,
    loader,
    optimizer,
    device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(sequences)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        preds = torch.argmax(logits.detach(), dim=1)
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((preds == targets).sum().detach().cpu())
        total_count += batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def resolve_device(name: str):
    return resolve_required_torch_cuda_device(torch, name)


def _load_ml_dependencies() -> None:
    global DataLoader, Xiao2020DepthLSTM, XiaoDepthSequenceDataset, nn, torch
    try:
        import torch as torch_mod
        from torch import nn as nn_mod
        from torch.utils.data import DataLoader as data_loader_cls

        from palpation_sim.dataset import XiaoDepthSequenceDataset as dataset_cls
        from palpation_sim.models import Xiao2020DepthLSTM as model_cls
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch/Numpy dependencies are required for training. Create the conda env from "
            "environment.yml, activate it, then rerun this script."
        ) from exc

    torch = torch_mod
    nn = nn_mod
    DataLoader = data_loader_cls
    XiaoDepthSequenceDataset = dataset_cls
    Xiao2020DepthLSTM = model_cls


if __name__ == "__main__":
    main()
