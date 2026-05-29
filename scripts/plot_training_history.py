from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training history curves.")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _load_dependencies()

    rows = list(csv.DictReader(args.history.open()))
    if not rows:
        raise SystemExit(f"No rows in {args.history}")

    epoch = np.asarray([int(row["epoch"]) for row in rows], dtype=np.int32)
    train_loss = _column(rows, "train_loss")
    val_loss = _column(rows, "val_loss")
    train_dice = _column(rows, "train_dice")
    val_dice = _column(rows, "val_dice")
    train_iou = _column(rows, "train_iou")
    val_iou = _column(rows, "val_iou")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    _plot_pair(axes[0], epoch, train_loss, val_loss, "Loss", "loss")
    _plot_pair(axes[1], epoch, train_dice, val_dice, "Dice / F1", "score")
    _plot_pair(axes[2], epoch, train_iou, val_iou, "IoU", "score")
    fig.suptitle(args.history.parent.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


def _column(rows, name: str):
    values = []
    for row in rows:
        raw = row.get(name, "")
        values.append(float(raw) if raw != "" else np.nan)
    return np.asarray(values, dtype=np.float32)


def _plot_pair(ax, epoch, train, val, title, ylabel):
    ax.plot(epoch, train, label="train", linewidth=1.8)
    if not np.isnan(val).all():
        ax.plot(epoch, val, label="val", linewidth=1.8)
        best_idx = int(np.nanargmax(val)) if "Dice" in title or "IoU" in title else int(np.nanargmin(val))
        ax.scatter([epoch[best_idx]], [val[best_idx]], s=28, zorder=3)
        ax.annotate(
            f"best e{int(epoch[best_idx])}",
            xy=(epoch[best_idx], val[best_idx]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.legend()


def _load_dependencies() -> None:
    global np, plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
        import numpy as np_mod
    except ModuleNotFoundError as exc:
        raise SystemExit("Need numpy and matplotlib in the active environment.") from exc

    np = np_mod
    plt = plt_mod


if __name__ == "__main__":
    main()
