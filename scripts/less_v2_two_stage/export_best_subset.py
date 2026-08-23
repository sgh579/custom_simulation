#!/usr/bin/env python3
"""Evaluate stopped high-resolution LESS checkpoints and export the matched eight views."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from train_less_v2_two_stage import (
    Config,
    LocalParticleRepresentation,
    ParticleTCNN2D,
    ParticleTCNN3D,
    binary_metrics,
    build_metric_neighbourhood,
    choose_threshold,
    collect_decoder_predictions,
    extract_representations,
    load_and_validate_xy,
    load_split_at_resolution,
    normalize_force_curves,
    tensor_sha256,
)


MATCHED_TEST_IDS = (
    "sample_0016",
    "sample_0201",
    "sample_0373",
    "sample_0386",
    "sample_0461",
    "sample_0464",
    "sample_0495",
    "sample_0608",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(run_dir / "run_manifest.json")
    config = config_from_manifest(source_manifest["config"])
    data_root = Path(config.data_root)
    val_paths = [data_root / "val" / name for name in source_manifest["selected_samples"]["val"]]
    test_paths = [data_root / "test" / f"{sample_id}.npz" for sample_id in MATCHED_TEST_IDS]
    require_files(val_paths + test_paths)

    print("loading 800 validation targets and matched eight test targets", flush=True)
    raw_val = load_split_at_resolution(
        val_paths,
        volume_depth=config.volume_depth,
        output_size=config.output_size,
    )
    raw_test = load_split_at_resolution(
        test_paths,
        volume_depth=config.volume_depth,
        output_size=config.output_size,
    )
    normalization = source_manifest["normalization"]
    curves = {
        "val": normalize_force_curves(raw_val[0], normalization),
        "test": normalize_force_curves(raw_test[0], normalization),
    }
    xy = load_and_validate_xy({"val": val_paths, "test": test_paths})
    neighbourhood = build_metric_neighbourhood(xy, config.particle_radius_m)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    representation = LocalParticleRepresentation(
        curve_steps=curves["val"].shape[-1],
        embedding_dim=config.embedding_dim,
        representation_dim=config.representation_dim,
        neighbour_indices=torch.from_numpy(neighbourhood["indices"]),
        neighbour_mask=torch.from_numpy(neighbourhood["mask"]),
        relative_xy=torch.from_numpy(neighbourhood["relative_xy"]),
        centre_slots=torch.from_numpy(neighbourhood["centre_slots"]),
    ).to(device)
    stage1 = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    representation.load_state_dict(stage1["model"])
    representation.eval()
    for parameter in representation.parameters():
        parameter.requires_grad = False
    representations = {
        split: extract_representations(representation, values, config, device)
        for split, values in curves.items()
    }
    representation_hash = tensor_sha256(representation.state_dict())
    del representation, curves
    if device.type == "cuda":
        torch.cuda.empty_cache()

    decoder_2d = ParticleTCNN2D(
        config.representation_dim,
        config.decoder_channels,
        patch_size=config.patch_size,
        output_size=config.output_size,
    ).to(device)
    checkpoint_2d = torch.load(
        run_dir / "stage2_decoder_2d" / "best_decoder_2d.pt",
        map_location=device,
        weights_only=False,
    )
    verify_representation_hash(checkpoint_2d, representation_hash, "2d")
    decoder_2d.load_state_dict(checkpoint_2d["model"])
    probabilities_2d = collect_decoder_predictions(
        decoder_2d,
        representations,
        device,
        config,
    )
    del decoder_2d
    if device.type == "cuda":
        torch.cuda.empty_cache()

    decoder_3d = ParticleTCNN3D(
        config.representation_dim,
        config.decoder_channels,
        volume_depth=config.volume_depth,
        patch_size=config.patch_size,
        output_size=config.output_size,
    ).to(device)
    checkpoint_3d = torch.load(
        run_dir / "stage2_decoder_3d" / "best_decoder_3d.pt",
        map_location=device,
        weights_only=False,
    )
    verify_representation_hash(checkpoint_3d, representation_hash, "3d")
    decoder_3d.load_state_dict(checkpoint_3d["model"])
    probabilities_3d = collect_decoder_predictions(
        decoder_3d,
        representations,
        device,
        config,
    )

    thresholds = {
        "2d": choose_threshold(probabilities_2d["val"], raw_val[2]),
        "3d": choose_threshold(probabilities_3d["val"], raw_val[3]),
    }
    test_metrics = {
        "fixed_0.5": {
            "2d": binary_metrics(probabilities_2d["test"], raw_test[2], 0.5),
            "3d": binary_metrics(probabilities_3d["test"], raw_test[3], 0.5),
        },
        "val_selected_threshold": {
            "2d": binary_metrics(probabilities_2d["test"], raw_test[2], thresholds["2d"]),
            "3d": binary_metrics(probabilities_3d["test"], raw_test[3], thresholds["3d"]),
        },
    }
    val_metrics = {
        "val_selected_threshold": {
            "2d": binary_metrics(probabilities_2d["val"], raw_val[2], thresholds["2d"]),
            "3d": binary_metrics(probabilities_3d["val"], raw_val[3], thresholds["3d"]),
        }
    }

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        probabilities_2d=probabilities_2d["test"].astype(np.float32),
        probabilities_3d=probabilities_3d["test"].astype(np.float32),
        targets_2d=raw_test[2].astype(np.float32),
        targets_3d=raw_test[3].astype(np.float32),
    )
    completed_2d = history_length(run_dir / "stage2_decoder_2d" / "history.csv")
    completed_3d = history_length(run_dir / "stage2_decoder_3d" / "history.csv")
    summary = {
        "status": "manually stopped after validation overfitting; evaluated from best checkpoints",
        "best_epoch": int(checkpoint_3d["epoch"]),
        "best_epochs": {
            "stage1_representation": int(stage1["epoch"]),
            "stage2_decoder_2d": int(checkpoint_2d["epoch"]),
            "stage2_decoder_3d": int(checkpoint_3d["epoch"]),
        },
        "completed_epochs": {
            "stage1_representation": int(stage1["config"]["pretrain_epochs"]),
            "stage2_decoder_2d": completed_2d,
            "stage2_decoder_3d": completed_3d,
        },
        "thresholds_selected_on_validation": thresholds,
        "metrics": {"val": val_metrics, "test": test_metrics},
        "selection_contract": "minimum validation loss checkpoint; thresholds selected on all 800 validation samples",
        "representation_sha256": representation_hash,
    }
    write_json(output_dir / "metrics.json", summary)

    curated_manifest = copy.deepcopy(source_manifest)
    curated_manifest["kind"] = (
        "full-corpus high-resolution two-stage LESS V2 — matched 8-sample visualization subset"
    )
    curated_manifest["selected_samples"]["test"] = [path.name for path in test_paths]
    curated_manifest["curation"] = {
        "rule": "same eight test sample IDs as the prior full LESS visualization",
        "sample_ids": list(MATCHED_TEST_IDS),
        "source_test_size": int(config.test_samples),
    }
    curated_manifest["checkpoint_selection"] = {
        "stage1": int(stage1["epoch"]),
        "stage2_2d": int(checkpoint_2d["epoch"]),
        "stage2_3d": int(checkpoint_3d["epoch"]),
        "stage2_3d_training_stopped_after_epoch": completed_3d,
    }
    write_json(output_dir / "run_manifest.json", curated_manifest)

    p2 = probabilities_2d["test"]
    p3 = probabilities_3d["test"]
    y2 = raw_test[2]
    y3 = raw_test[3]
    ids = list(MATCHED_TEST_IDS)
    per_sample = write_per_sample_metrics(output_dir, ids, p2, p3, y2, y3, thresholds)
    make_all_2d_figure(output_dir / "all_8_test_2d_results.png", ids, p2, y2, thresholds["2d"])
    selected, roles = representative_indices(p3, y3, thresholds["3d"])
    make_2d_figure(
        output_dir / "decoded_examples_2d.png",
        ids,
        p2,
        y2,
        thresholds["2d"],
        selected,
        roles,
    )
    make_3d_figure(
        output_dir / "decoded_examples_3d.png",
        ids,
        p3,
        y3,
        thresholds["3d"],
        selected,
        roles,
    )
    make_player_bundle(
        output_dir / "3d_player",
        ids,
        p3,
        y3,
        thresholds["3d"],
        int(checkpoint_3d["epoch"]),
        str(run_dir),
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "thresholds": thresholds,
                "best_epochs": summary["best_epochs"],
                "completed_epochs": summary["completed_epochs"],
                "matched_eight_metrics": test_metrics["val_selected_threshold"],
                "samples": per_sample,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def config_from_manifest(values: dict[str, Any]) -> Config:
    allowed = {field.name for field in fields(Config)}
    return Config(**{key: value for key, value in values.items() if key in allowed})


def verify_representation_hash(checkpoint: dict[str, Any], expected: str, task: str) -> None:
    actual = checkpoint.get("frozen_representation_sha256")
    if actual != expected:
        raise RuntimeError(f"{task} checkpoint representation hash mismatch: {actual} != {expected}")


def representative_indices(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> tuple[list[int], list[str]]:
    scores = [single_metrics(probabilities[index], targets[index], threshold)[0] for index in range(len(targets))]
    order = np.argsort(scores)
    return [int(order[-1]), int(order[len(order) // 2]), int(order[0])], [
        "strong",
        "typical",
        "challenging",
    ]


def single_metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> tuple[float, float]:
    prediction = probability >= threshold
    truth = target >= 0.5
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp + fn, 1)


def classification_rgb(probability: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    prediction = probability >= threshold
    truth = target >= 0.5
    rgb = np.ones((*truth.shape, 3), dtype=np.float32)
    rgb[prediction & truth] = (0.18, 0.65, 0.22)
    rgb[prediction & ~truth] = (0.88, 0.14, 0.14)
    rgb[~prediction & truth] = (0.12, 0.47, 0.71)
    return rgb


def make_all_2d_figure(
    path: Path,
    ids: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 7.8))
    for axis, sample_id, probability, target in zip(axes.flat, ids, probabilities, targets):
        dice, iou = single_metrics(probability, target, threshold)
        axis.imshow(classification_rgb(probability, target, threshold), interpolation="nearest")
        axis.set_title(f"{sample_id}\nDice {dice:.3f} · IoU {iou:.3f}", fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"High-resolution LESS V2 — matched eight test samples, 2D decoded masks "
        f"(128×128, validation threshold {threshold:.3f})",
        fontsize=15,
        y=0.98,
    )
    figure.subplots_adjust(left=0.025, right=0.99, bottom=0.09, top=0.88, wspace=0.10, hspace=0.24)
    figure.text(
        0.5,
        0.025,
        "green: TP    blue: FN    red: FP",
        ha="center",
        fontsize=11,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_2d_figure(
    path: Path,
    ids: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    selected: list[int],
    roles: list[str],
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(9.5, 9.3), constrained_layout=True)
    for row, (index, role) in enumerate(zip(selected, roles)):
        dice, iou = single_metrics(probabilities[index], targets[index], threshold)
        axes[row, 0].imshow(targets[index] >= 0.5, cmap="Blues", vmin=0, vmax=1)
        axes[row, 1].imshow(probabilities[index], cmap="magma", vmin=0, vmax=1)
        axes[row, 2].imshow(classification_rgb(probabilities[index], targets[index], threshold))
        axes[row, 0].set_ylabel(
            f"{role}\n{ids[index]}\nDice {dice:.3f} · IoU {iou:.3f}",
            fontsize=10,
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("2D ground truth")
    axes[0, 1].set_title("2D probability")
    axes[0, 2].set_title(f"threshold {threshold:.3f}\nTP / FP / FN")
    figure.suptitle("High-resolution LESS V2: representative 2D decoded test examples", fontsize=14)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_3d_figure(
    path: Path,
    ids: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    selected: list[int],
    roles: list[str],
) -> None:
    figure = plt.figure(figsize=(12.5, 10.6), constrained_layout=True)
    for row, (index, role) in enumerate(zip(selected, roles)):
        truth = targets[index] >= 0.5
        prediction = probabilities[index] >= threshold
        dice, iou = single_metrics(probabilities[index], truth, threshold)
        display_truth = downsample_xy(truth, 4)
        display_prediction = downsample_xy(prediction, 4)
        for col, (volume, color, title) in enumerate(
            (
                (display_truth, "tab:blue", "3D ground truth"),
                (display_prediction, "tab:orange", "3D prediction"),
            )
        ):
            axis = figure.add_subplot(3, 4, row * 4 + col + 1, projection="3d")
            axis.voxels(volume.transpose(2, 1, 0), facecolors=color, edgecolor="none", alpha=0.75)
            style_3d_axis(axis, volume.shape)
            if row == 0:
                axis.set_title(title)
        axis = figure.add_subplot(3, 4, row * 4 + 3, projection="3d")
        overlap = display_truth & display_prediction
        misses = display_truth & ~display_prediction
        extras = display_prediction & ~display_truth
        axis.voxels(overlap.transpose(2, 1, 0), facecolors="tab:green", edgecolor="none", alpha=0.8)
        axis.voxels(misses.transpose(2, 1, 0), facecolors="tab:blue", edgecolor="none", alpha=0.65)
        axis.voxels(extras.transpose(2, 1, 0), facecolors="tab:red", edgecolor="none", alpha=0.45)
        style_3d_axis(axis, display_truth.shape)
        if row == 0:
            axis.set_title("overlap: TP/FN/FP")
        depth = int(np.argmax(truth.sum(axis=(1, 2))))
        classification = np.zeros_like(truth[depth], dtype=np.uint8)
        classification[truth[depth] & prediction[depth]] = 1
        classification[truth[depth] & ~prediction[depth]] = 2
        classification[~truth[depth] & prediction[depth]] = 3
        axis_2d = figure.add_subplot(3, 4, row * 4 + 4)
        axis_2d.imshow(
            classification,
            cmap=ListedColormap(["white", "tab:green", "tab:blue", "tab:red"]),
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        axis_2d.set_xticks([])
        axis_2d.set_yticks([])
        axis_2d.set_ylabel(f"{role}\n{ids[index]}\nDice {dice:.3f} · IoU {iou:.3f}")
        if row == 0:
            axis_2d.set_title("most occupied GT depth slice")
    figure.suptitle(
        f"High-resolution LESS V2: representative 3D decoded test examples "
        f"(threshold {threshold:.3f}; 3D display pooled 128→32)",
        fontsize=14,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def downsample_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    depth, height, width = volume.shape
    if height % factor or width % factor:
        raise ValueError(f"cannot pool {volume.shape} by {factor}")
    return volume.reshape(depth, height // factor, factor, width // factor, factor).max(axis=(2, 4))


def style_3d_axis(axis: plt.Axes, shape: tuple[int, ...]) -> None:
    depth, height, width = shape
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_zlim(0, depth)
    axis.set_box_aspect((width, height, depth))
    axis.view_init(elev=25, azim=-55)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])


def write_per_sample_metrics(
    output_dir: Path,
    ids: list[str],
    p2: np.ndarray,
    p3: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for index, sample_id in enumerate(ids):
        dice_2d, iou_2d = single_metrics(p2[index], y2[index], thresholds["2d"])
        dice_3d, iou_3d = single_metrics(p3[index], y3[index], thresholds["3d"])
        rows.append(
            {
                "sample_id": sample_id,
                "dice_2d": dice_2d,
                "iou_2d": iou_2d,
                "dice_3d": dice_3d,
                "iou_3d": iou_3d,
            }
        )
    with (output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_player_bundle(
    output_dir: Path,
    ids: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    best_epoch: int,
    source_run: str,
) -> None:
    player_input = output_dir / "player_input"
    player_input.mkdir(parents=True, exist_ok=True)
    records = []
    for index, sample_id in enumerate(ids):
        score = probabilities[index].astype(np.float32)
        gt = (targets[index] >= 0.5).astype(np.uint8)
        pred = (score >= threshold).astype(np.uint8)
        filename = f"{index:02d}_{sample_id}_3d_prediction.npz"
        np.savez_compressed(
            player_input / filename,
            score=score,
            pred=pred,
            gt=gt,
            threshold=np.asarray(threshold, dtype=np.float32),
            sample_id=np.asarray(sample_id),
            bounds_mm=np.asarray([-75.0, 75.0, -75.0, 75.0, 0.0, 80.0], dtype=np.float32),
        )
        dice, iou = single_metrics(score, gt, threshold)
        records.append(
            {
                "index": index,
                "sample_id": sample_id,
                "file": f"player_input/{filename}",
                "shape_d_h_w": list(gt.shape),
                "metrics": {"dice": dice, "iou": iou},
            }
        )
    write_json(
        output_dir / "player_manifest.json",
        {
            "format": "RSS synthetic 3D prediction package for PySide6/PyVistaQt",
            "required_npz_fields": ["score", "pred", "gt", "threshold"],
            "array_order": "D,H,W = depth,y,x",
            "coordinate_contract": {
                "x_mm": [-75.0, 75.0],
                "y_mm": [-75.0, 75.0],
                "z_mm_from_bottom": [0.0, 80.0],
            },
            "threshold_3d": threshold,
            "samples": records,
            "source_run": source_run,
            "source_checkpoint_epoch": best_epoch,
        },
    )


def history_length(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} source files; first={missing[0]}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
