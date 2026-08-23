#!/usr/bin/env python3
"""Quick LESS-style 2D/3D validation on a deterministic subset of synthetic V2.

This is an intentionally small integration experiment, not a reproduction of
the released LESS paper.  It preserves the architectural ideas needed for the
check: a shared force-location encoder, a shared recurrent encoder at every
spatial particle, local shared interaction, and shared local patch decoders
whose logits are stitched into global 2D and 3D predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    data_root: str
    output_dir: str
    train_samples: int
    val_samples: int
    test_samples: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    embedding_dim: int
    hidden_dim: int
    volume_depth: int
    patch_size: int
    seed: int
    num_workers: int
    amp: bool


class LessStyleV2(nn.Module):
    """A compact local-particle LESS adaptation for one-taxel V2 curves."""

    def __init__(
        self,
        *,
        embedding_dim: int = 48,
        hidden_dim: int = 64,
        volume_depth: int = 16,
        patch_size: int = 3,
        height: int = 20,
        width: int = 20,
    ) -> None:
        super().__init__()
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        self.height = height
        self.width = width
        self.volume_depth = volume_depth
        self.patch_size = patch_size

        # LESS-style FLE: location/depth and force are embedded separately,
        # then added in a common latent space before the shared GRU.
        self.location_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim), nn.LayerNorm(embedding_dim)
        )
        self.force_encoder = nn.Linear(1, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # A fixed-radius 3x3 local receptive field.  All particles share these
        # weights, preserving the central local/compositional LESS bias.
        self.local_interaction = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        patch_area = patch_size * patch_size
        self.decoder_2d = nn.Linear(hidden_dim, patch_area)
        self.decoder_3d = nn.Linear(hidden_dim, volume_depth * patch_area)
        self.register_buffer("overlap_count", self._overlap_count(), persistent=False)

    def _overlap_count(self) -> torch.Tensor:
        k = self.patch_size
        n = self.height * self.width
        ones = torch.ones(1, k * k, n)
        count = F.fold(
            ones,
            output_size=(self.height, self.width),
            kernel_size=k,
            padding=k // 2,
        )
        return count.clamp_min(1.0)

    def _stitch(self, patch_logits: torch.Tensor) -> torch.Tensor:
        # patch_logits: [B, N, K*K]
        return F.fold(
            patch_logits.transpose(1, 2),
            output_size=(self.height, self.width),
            kernel_size=self.patch_size,
            padding=self.patch_size // 2,
        ) / self.overlap_count

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sequences: [B,H,W,T,4] = x,y,depth,normalized delta-Fz
        b, h, w, t, d = sequences.shape
        if (h, w, d) != (self.height, self.width, 4):
            raise ValueError(f"unexpected input shape {tuple(sequences.shape)}")
        flat = sequences.reshape(b * h * w, t, d)
        encoded = self.location_encoder(flat[..., :3]) + self.force_encoder(flat[..., 3:4])
        encoded = F.gelu(encoded)
        _, state = self.gru(encoded)
        particles = state[-1].reshape(b, h, w, -1).permute(0, 3, 1, 2)
        particles = particles + self.local_interaction(particles)
        particles = particles.permute(0, 2, 3, 1).reshape(b, h * w, -1)

        logits_2d = self._stitch(self.decoder_2d(particles))[:, 0]
        patch_area = self.patch_size * self.patch_size
        patches_3d = self.decoder_3d(particles).reshape(
            b, h * w, self.volume_depth, patch_area
        )
        folded = self._stitch(
            patches_3d.permute(0, 2, 1, 3).reshape(
                b * self.volume_depth, h * w, patch_area
            )
        )[:, 0]
        logits_3d = folded.reshape(b, self.volume_depth, h, w)
        return logits_2d, logits_3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--volume-depth", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9400)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        data_root=str(args.data_root.resolve()),
        output_dir=str(args.output_dir.resolve()),
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        volume_depth=args.volume_depth,
        patch_size=args.patch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    print("loading deterministic V2 subset", flush=True)
    selected = {
        "train": select_paths(Path(config.data_root) / "train", config.train_samples, config.seed + 1),
        "val": select_paths(Path(config.data_root) / "val", config.val_samples, config.seed + 2),
        "test": select_paths(Path(config.data_root) / "test", config.test_samples, config.seed + 3),
    }
    raw = {
        split: load_split(paths, volume_depth=config.volume_depth)
        for split, paths in selected.items()
    }
    normalization = fit_normalization(raw["train"][0])
    arrays = {
        split: (
            make_sequences(values[0], values[1], normalization),
            values[2],
            values[3],
        )
        for split, values in raw.items()
    }

    qa = {
        split: {
            "samples": len(selected[split]),
            "projection_consistency_mean": float(raw[split][4].mean()),
            "projection_consistency_min": float(raw[split][4].min()),
            "positive_fraction_2d": float(raw[split][2].mean()),
            "positive_fraction_3d": float(raw[split][3].mean()),
        }
        for split in ("train", "val", "test")
    }
    manifest = {
        "kind": "LESS-style quick integration validation; not released-LESS reproduction",
        "config": asdict(config),
        "normalization": normalization,
        "selected_samples": {
            split: [path.name for path in paths] for split, paths in selected.items()
        },
        "data_qa": qa,
        "input_semantics": "log1p(max(Fz(t)-Fz(0),0)); one force taxel; full 40-step fixed-depth sequence",
        "target_2d": "provided V2 full-phantom xy projection mask",
        "target_3d": "analytic inclusion occupancy at 16x20x20 voxel centers reconstructed from V2 lump geometry",
        "architecture": {
            "fle": "shared additive location/depth and force encoders",
            "temporal": "shared GRU per spatial particle",
            "locality": "shared 3x3 local interaction",
            "decoders": "shared 3x3 particle patches stitched by overlap averaging for 2D and 3D",
        },
        "environment": environment_snapshot(),
    }
    write_json(out / "run_manifest.json", manifest)

    train_loader = make_loader(arrays["train"], config, shuffle=True)
    val_loader = make_loader(arrays["val"], config, shuffle=False)
    test_loader = make_loader(arrays["test"], config, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LessStyleV2(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        volume_depth=config.volume_depth,
        patch_size=config.patch_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} parameters={parameter_count:,} "
        f"train/val/test={config.train_samples}/{config.val_samples}/{config.test_samples}",
        flush=True,
    )

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_result = run_epoch(model, train_loader, device, optimizer, amp=config.amp)
        val_result = run_epoch(model, val_loader, device, None, amp=config.amp)
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            **{f"train_{key}": value for key, value in train_result.items()},
            **{f"val_{key}": value for key, value in val_result.items()},
        }
        history.append(row)
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "normalization": normalization,
                    "val": val_result,
                },
                out / "best.pt",
            )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == config.epochs:
            elapsed = time.perf_counter() - start
            print(
                f"epoch={epoch:04d}/{config.epochs} "
                f"train_loss={train_result['loss']:.4f} val_loss={val_result['loss']:.4f} "
                f"val_dice2d={val_result['dice_2d']:.4f} "
                f"val_dice3d={val_result['dice_3d']:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            write_history(out / "history.csv", history)

    torch.save(
        {
            "epoch": config.epochs,
            "model": model.state_dict(),
            "config": asdict(config),
            "normalization": normalization,
        },
        out / "final.pt",
    )
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    val_predictions = collect_predictions(model, val_loader, device, amp=config.amp)
    thresholds = {
        "2d": choose_threshold(val_predictions[0], val_predictions[2]),
        "3d": choose_threshold(val_predictions[1], val_predictions[3]),
    }
    reports: dict[str, Any] = {}
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        predictions = collect_predictions(model, loader, device, amp=config.amp)
        reports[split] = {
            "fixed_0.5": {
                "2d": binary_metrics(predictions[0], predictions[2], 0.5),
                "3d": binary_metrics(predictions[1], predictions[3], 0.5),
            },
            "val_selected_threshold": {
                "2d": binary_metrics(predictions[0], predictions[2], thresholds["2d"]),
                "3d": binary_metrics(predictions[1], predictions[3], thresholds["3d"]),
            },
        }
        if split == "test":
            np.savez_compressed(
                out / "test_predictions.npz",
                probabilities_2d=predictions[0],
                probabilities_3d=predictions[1],
                targets_2d=predictions[2],
                targets_3d=predictions[3],
            )

    elapsed = time.perf_counter() - start
    convergence = convergence_summary(history, best_epoch)
    summary = {
        "completed_epochs": config.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        "parameter_count": parameter_count,
        "thresholds_selected_on_validation": thresholds,
        "convergence": convergence,
        "metrics": reports,
    }
    write_json(out / "metrics.json", summary)
    write_history(out / "history.csv", history)
    print("RESULT " + json.dumps(summary, sort_keys=True), flush=True)


def select_paths(split_dir: Path, count: int, seed: int) -> list[Path]:
    paths = sorted(split_dir.glob("*.npz"))
    if len(paths) < count:
        raise ValueError(f"{split_dir} has {len(paths)} samples, requested {count}")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(paths), size=count, replace=False))
    return [paths[int(index)] for index in indices]


def load_split(
    paths: Iterable[Path], *, volume_depth: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fz_values: list[np.ndarray] = []
    depth_values: list[np.ndarray] = []
    masks_2d: list[np.ndarray] = []
    masks_3d: list[np.ndarray] = []
    projection_consistency: list[float] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            fz = np.asarray(data["fz"], dtype=np.float32)
            depth = np.asarray(data["indentation_depth"], dtype=np.float32)
            mask_2d = np.asarray(data["mask"], dtype=np.float32)
            label_xy = np.asarray(data["label_xy"], dtype=np.float32)
            lumps = json.loads(str(np.asarray(data["lumps_json"]).item()))
            phantom = json.loads(str(np.asarray(data["phantom_json"]).item()))
        if fz.shape != (20, 20, 40) or depth.shape != fz.shape or mask_2d.shape != (20, 20):
            raise ValueError(f"unexpected V2 arrays in {path}: {fz.shape}, {depth.shape}, {mask_2d.shape}")
        mask_3d, projected = make_geometry_targets(
            label_xy, lumps, float(phantom["height"]), volume_depth
        )
        fz_values.append(fz)
        depth_values.append(depth)
        masks_2d.append(mask_2d)
        masks_3d.append(mask_3d)
        projection_consistency.append(float((projected == (mask_2d > 0.5)).mean()))
    return (
        np.stack(fz_values),
        np.stack(depth_values),
        np.stack(masks_2d),
        np.stack(masks_3d),
        np.asarray(projection_consistency, dtype=np.float32),
    )


def make_geometry_targets(
    label_xy: np.ndarray,
    lumps: list[dict[str, Any]],
    phantom_height: float,
    volume_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = label_xy.shape[:2]
    z_edges = np.linspace(0.0, phantom_height, volume_depth + 1, dtype=np.float32)
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    xy = np.broadcast_to(label_xy[None], (volume_depth, h, w, 2))
    z = np.broadcast_to(z_centres[:, None, None, None], (volume_depth, h, w, 1))
    points_3d = np.concatenate([xy, z], axis=-1)
    volume = np.zeros((volume_depth, h, w), dtype=bool)
    projection = np.zeros((h, w), dtype=bool)
    for lump in lumps:
        volume |= lump_membership(points_3d, lump, project_xy=False)
        projection |= lump_membership(label_xy, lump, project_xy=True)
    return volume.astype(np.float32), projection


def lump_membership(points: np.ndarray, lump: dict[str, Any], *, project_xy: bool) -> np.ndarray:
    centre = np.asarray(lump["center"], dtype=np.float32)
    radii = np.asarray(lump["radii"], dtype=np.float32)
    rel = np.asarray(points, dtype=np.float32) - centre[: points.shape[-1]]
    yaw = -float(lump.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    x = rel[..., 0].copy()
    y = rel[..., 1].copy()
    rel[..., 0] = c * x - s * y
    rel[..., 1] = s * x + c * y
    shape = str(lump["shape"])
    if project_xy:
        rel_eval, radii_eval = rel[..., :2], radii[:2]
    else:
        rel_eval, radii_eval = rel, radii
    if shape in {"sphere", "ellipsoid"}:
        return np.sum((rel_eval / radii_eval) ** 2, axis=-1) <= 1.0
    if shape == "box":
        return np.all(np.abs(rel_eval) <= radii_eval, axis=-1)
    if shape == "cylinder":
        radial = (rel[..., 0] / radii[0]) ** 2 + (rel[..., 1] / radii[1]) ** 2
        if project_xy:
            return radial <= 1.0
        return (radial <= 1.0) & (np.abs(rel[..., 2]) <= radii[2])
    if shape == "capsule":
        if project_xy:
            return np.sum((rel_eval / radii_eval) ** 2, axis=-1) <= 1.0
        radius = float(radii[0])
        closest_z = np.clip(rel[..., 2], -float(radii[2]), float(radii[2]))
        return rel[..., 0] ** 2 + rel[..., 1] ** 2 + (rel[..., 2] - closest_z) ** 2 <= radius**2
    raise ValueError(f"unsupported lump shape {shape!r}")


def fit_normalization(fz: np.ndarray) -> dict[str, float]:
    delta = np.maximum(fz - fz[..., :1], 0.0)
    transformed = np.log1p(delta)
    return {
        "force_log_mean": float(transformed.mean()),
        "force_log_std": float(max(transformed.std(), 1e-6)),
        "xy_scale": 0.075,
        "depth_scale": 0.036,
    }


def make_sequences(
    fz: np.ndarray, depth: np.ndarray, normalization: dict[str, float]
) -> np.ndarray:
    b, h, w, t = fz.shape
    delta = np.maximum(fz - fz[..., :1], 0.0)
    force = (np.log1p(delta) - normalization["force_log_mean"]) / normalization["force_log_std"]
    xs = np.linspace(-normalization["xy_scale"], normalization["xy_scale"], w, dtype=np.float32)
    ys = np.linspace(-normalization["xy_scale"], normalization["xy_scale"], h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    x = np.broadcast_to((xv / normalization["xy_scale"])[None, ..., None], (b, h, w, t))
    y = np.broadcast_to((yv / normalization["xy_scale"])[None, ..., None], (b, h, w, t))
    z = depth / normalization["depth_scale"]
    return np.stack([x, y, z, force], axis=-1).astype(np.float32)


def make_loader(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray], config: Config, *, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(*(torch.from_numpy(value) for value in arrays))
    generator = torch.Generator().manual_seed(config.seed + (11 if shuffle else 17))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=generator,
    )


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    dims = tuple(range(1, targets.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum().clamp_min(1.0)
    negatives = torch.tensor(targets.numel(), device=targets.device) - positives
    positive_weight = (negatives / positives).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
    return 0.5 * bce + 0.5 * dice_loss(logits, targets)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "loss_2d": 0.0, "loss_3d": 0.0, "dice_2d": 0.0, "dice_3d": 0.0}
    samples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for sequences, targets_2d, targets_3d in loader:
            sequences = sequences.to(device, non_blocking=True)
            targets_2d = targets_2d.to(device, non_blocking=True)
            targets_3d = targets_3d.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                logits_2d, logits_3d = model(sequences)
                loss_2d = combined_loss(logits_2d, targets_2d)
                loss_3d = combined_loss(logits_3d, targets_3d)
                loss = 0.5 * loss_2d + 0.5 * loss_3d
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            batch = int(sequences.shape[0])
            totals["loss"] += float(loss.detach()) * batch
            totals["loss_2d"] += float(loss_2d.detach()) * batch
            totals["loss_3d"] += float(loss_3d.detach()) * batch
            totals["dice_2d"] += batch_mean_dice(logits_2d.detach(), targets_2d) * batch
            totals["dice_3d"] += batch_mean_dice(logits_3d.detach(), targets_3d) * batch
            samples += batch
    return {key: value / max(samples, 1) for key, value in totals.items()}


def batch_mean_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    predictions = logits.sigmoid() >= threshold
    truth = targets >= 0.5
    dims = tuple(range(1, targets.ndim))
    intersection = (predictions & truth).sum(dim=dims).float()
    denominator = predictions.sum(dim=dims).float() + truth.sum(dim=dims).float()
    dice = torch.where(denominator > 0, 2.0 * intersection / denominator, torch.ones_like(denominator))
    return float(dice.mean())


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device, *, amp: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    p2, p3, y2, y3 = [], [], [], []
    with torch.no_grad():
        for sequences, targets_2d, targets_3d in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                logits_2d, logits_3d = model(sequences.to(device, non_blocking=True))
            p2.append(logits_2d.sigmoid().float().cpu().numpy())
            p3.append(logits_3d.sigmoid().float().cpu().numpy())
            y2.append(targets_2d.numpy())
            y3.append(targets_3d.numpy())
    return tuple(np.concatenate(values) for values in (p2, p3, y2, y3))  # type: ignore[return-value]


def choose_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    candidates = np.linspace(0.1, 0.9, 33)
    scores = [binary_metrics(probabilities, targets, float(value))["dice_mean"] for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def binary_metrics(probabilities: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probabilities >= threshold
    truth = targets >= 0.5
    axes = tuple(range(1, targets.ndim))
    tp = (prediction & truth).sum(axis=axes).astype(np.float64)
    fp = (prediction & ~truth).sum(axis=axes).astype(np.float64)
    fn = (~prediction & truth).sum(axis=axes).astype(np.float64)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.ones_like(tp), where=(2 * tp + fp + fn) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.ones_like(tp), where=(tp + fp + fn) > 0)
    micro_tp, micro_fp, micro_fn = float(tp.sum()), float(fp.sum()), float(fn.sum())
    return {
        "threshold": threshold,
        "dice_mean": float(dice.mean()),
        "dice_std": float(dice.std()),
        "iou_mean": float(iou.mean()),
        "iou_std": float(iou.std()),
        "dice_micro": float(2 * micro_tp / max(2 * micro_tp + micro_fp + micro_fn, 1.0)),
        "iou_micro": float(micro_tp / max(micro_tp + micro_fp + micro_fn, 1.0)),
        "predicted_positive_fraction": float(prediction.mean()),
        "target_positive_fraction": float(truth.mean()),
    }


def convergence_summary(history: list[dict[str, float | int]], best_epoch: int) -> dict[str, Any]:
    train = np.asarray([float(row["train_loss"]) for row in history])
    val = np.asarray([float(row["val_loss"]) for row in history])
    window = min(50, max(5, len(history) // 10))
    relative_train_drop = float((train[0] - train[-1]) / max(abs(train[0]), 1e-8))
    tail_range = float((val[-window:].max() - val[-window:].min()) / max(abs(val[-window:].mean()), 1e-8))
    return {
        "relative_train_loss_drop": relative_train_drop,
        "validation_tail_window": window,
        "validation_tail_relative_range": tail_range,
        "best_epoch": best_epoch,
        "loss_decreased": bool(relative_train_drop > 0.5),
        "tail_stable": bool(tail_range < 0.1),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def environment_snapshot() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_head_custom_simulation": command_output(
            ["git", "-C", "/home/guoheng/custom_simulation", "rev-parse", "HEAD"]
        ),
    }


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
