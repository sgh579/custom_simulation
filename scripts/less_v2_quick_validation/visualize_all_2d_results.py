#!/usr/bin/env python3
"""Render all eight test-set 2D decoding results in one 2x4 figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((args.result_dir / "metrics.json").read_text(encoding="utf-8"))
    arrays = np.load(args.result_dir / "test_predictions.npz")
    probabilities = np.asarray(arrays["probabilities_2d"], dtype=np.float32)
    targets = np.asarray(arrays["targets_2d"], dtype=np.float32) >= 0.5
    threshold = float(metrics["thresholds_selected_on_validation"]["2d"])
    sample_ids = [Path(value).stem for value in manifest["selected_samples"]["test"]]
    if probabilities.shape != (8, 20, 20) or targets.shape != probabilities.shape:
        raise ValueError(f"expected eight 20x20 results, got {probabilities.shape} and {targets.shape}")

    colors = ["#f5f5f5", "#2ca02c", "#1f77b4", "#d62728"]
    cmap = ListedColormap(colors)
    figure, axes = plt.subplots(2, 4, figsize=(12.8, 6.9), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        prediction = probabilities[index] >= threshold
        truth = targets[index]
        tp = int((prediction & truth).sum())
        fp = int((prediction & ~truth).sum())
        fn = int((~prediction & truth).sum())
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        classes = np.zeros((20, 20), dtype=np.uint8)
        classes[prediction & truth] = 1
        classes[~prediction & truth] = 2
        classes[prediction & ~truth] = 3
        axis.imshow(classes, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axis.set_title(f"{sample_ids[index]}\nDice {dice:.3f} · IoU {iou:.3f}", fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#777777")

    legend = [
        Patch(facecolor=colors[1], label="TP: prediction ∩ GT"),
        Patch(facecolor=colors[2], label="FN: missed GT"),
        Patch(facecolor=colors[3], label="FP: extra prediction"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    experiment_title = str(manifest.get("kind", "Synthetic palpation V2"))
    figure.suptitle(
        f"{experiment_title} — all 8 test samples, 2D decoded masks (threshold {threshold:.3f})",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
