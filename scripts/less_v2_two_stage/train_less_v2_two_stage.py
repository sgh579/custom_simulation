#!/usr/bin/env python3
"""Two-stage LESS port for the fixed-grid synthetic palpation V2 dataset.

Stage 1 learns a local particle representation without segmentation labels by
holding out one force curve in each particle neighbourhood and reconstructing
it from the remaining curves.  Stage 2 freezes that representation and trains
independent particle-wise transposed-convolution decoders for 2D and 3D masks.
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
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Reuse only the audited V2 file/target/metric utilities.  No model or training
# component is imported from the superseded quick smoke test.
QUICK_UTILS = Path(__file__).resolve().parents[1] / "less_v2_quick_validation"
sys.path.insert(0, str(QUICK_UTILS))
from train_less_v2_quick import (  # noqa: E402
    binary_metrics,
    choose_threshold,
    combined_loss,
    fit_normalization,
    load_split as load_native_split,
    make_geometry_targets,
    select_paths,
)


@dataclass(frozen=True)
class Config:
    data_root: str
    output_dir: str
    train_samples: int = 32
    val_samples: int = 8
    test_samples: int = 8
    pretrain_epochs: int = 1000
    decoder_epochs: int = 1000
    batch_size: int = 8
    decoder_batch_size: int = 0
    pretrain_learning_rate: float = 2e-3
    decoder_learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    embedding_dim: int = 128
    representation_dim: int = 128
    decoder_channels: int = 64
    volume_depth: int = 16
    output_size: int = 20
    patch_size: int = 5
    particle_radius_m: float = 0.0136
    seed: int = 9400
    num_workers: int = 0
    amp: bool = True
    log_every: int = 10


class LocalParticleRepresentation(nn.Module):
    """LESS particle encoder with metric-radius observation assignment."""

    def __init__(
        self,
        *,
        curve_steps: int,
        embedding_dim: int,
        representation_dim: int,
        neighbour_indices: torch.Tensor,
        neighbour_mask: torch.Tensor,
        relative_xy: torch.Tensor,
        centre_slots: torch.Tensor,
    ) -> None:
        super().__init__()
        self.curve_steps = curve_steps
        self.embedding_dim = embedding_dim
        self.representation_dim = representation_dim
        self.force_encoder = nn.Sequential(
            nn.Linear(curve_steps, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.location_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.gru_cell = nn.GRUCell(embedding_dim, representation_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.force_predictor = nn.Sequential(
            nn.Linear(representation_dim + embedding_dim, 2 * representation_dim),
            nn.GELU(),
            nn.Linear(2 * representation_dim, curve_steps),
        )
        self.register_buffer("neighbour_indices", neighbour_indices.long())
        self.register_buffer("neighbour_mask", neighbour_mask.bool())
        self.register_buffer("relative_xy", relative_xy.float())
        self.register_buffer("centre_slots", centre_slots.long())

    @property
    def num_particles(self) -> int:
        return int(self.neighbour_indices.shape[0])

    @property
    def max_neighbours(self) -> int:
        return int(self.neighbour_indices.shape[1])

    def gathered_curves(self, curves: torch.Tensor) -> torch.Tensor:
        # curves [B,N,T] -> [B,P,K,T]
        return curves[:, self.neighbour_indices]

    def encode(self, curves: torch.Tensor, visible_mask: torch.Tensor | None = None) -> torch.Tensor:
        gathered = self.gathered_curves(curves)
        b, p, k, _ = gathered.shape
        valid = self.neighbour_mask.unsqueeze(0).expand(b, -1, -1)
        if visible_mask is not None:
            valid = valid & visible_mask
        locations = self.relative_xy.unsqueeze(0).expand(b, -1, -1, -1)
        embedded = F.gelu(self.force_encoder(gathered) + self.location_encoder(locations))
        embedded = embedded.reshape(b * p, k, self.embedding_dim)
        valid = valid.reshape(b * p, k)
        hidden = embedded.new_zeros(b * p, self.representation_dim)
        for slot in range(k):
            proposed = self.gru_cell(embedded[:, slot], hidden)
            hidden = torch.where(valid[:, slot : slot + 1], proposed, hidden)
        return hidden.reshape(b, p, self.representation_dim)

    def reconstruct_one_local_curve(
        self, curves: torch.Tensor, *, random_target: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = curves.shape[0]
        valid = self.neighbour_mask.unsqueeze(0).expand(b, -1, -1)
        if random_target:
            scores = torch.rand(valid.shape, device=curves.device).masked_fill(~valid, -1.0)
            target_slots = scores.argmax(dim=-1)
        else:
            target_slots = self.centre_slots.unsqueeze(0).expand(b, -1)

        visible = valid.clone()
        visible.scatter_(2, target_slots.unsqueeze(-1), False)
        representation = self.encode(curves, visible)
        gathered = self.gathered_curves(curves)
        gather_curve = target_slots[..., None, None].expand(-1, -1, 1, self.curve_steps)
        target_curves = gathered.gather(2, gather_curve).squeeze(2)
        locations = self.relative_xy.unsqueeze(0).expand(b, -1, -1, -1)
        gather_xy = target_slots[..., None, None].expand(-1, -1, 1, 2)
        target_xy = locations.gather(2, gather_xy).squeeze(2)
        prediction = self.force_predictor(
            torch.cat((representation, self.query_encoder(target_xy)), dim=-1)
        )
        return prediction, target_curves


class ParticleTCNN2D(nn.Module):
    """Decode every frozen particle to a local patch and add it to the output canvas."""

    def __init__(
        self,
        representation_dim: int,
        channels: int,
        patch_size: int = 5,
        output_size: int = 20,
        particle_grid_size: int = 20,
    ) -> None:
        super().__init__()
        validate_decoder_geometry(output_size, patch_size)
        self.patch_size = patch_size
        self.output_size = output_size
        self.particle_grid_size = particle_grid_size
        self.channels = channels
        self.project = nn.Linear(representation_dim, channels)
        self.patch_decoder = make_patch_decoder_2d(channels, patch_size)
        placement_indices, placement_valid = build_patch_placement(
            particle_height=particle_grid_size,
            particle_width=particle_grid_size,
            output_height=output_size,
            output_width=output_size,
            patch_size=patch_size,
        )
        # Placement is fixed by geometry and does not belong in decoder checkpoints.
        self.register_buffer("placement_indices", placement_indices, persistent=False)
        self.register_buffer("placement_valid", placement_valid, persistent=False)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        b, n, _ = representations.shape
        latent = self.project(representations).reshape(b * n, self.channels, 1, 1)
        patches = self.patch_decoder(latent)
        expected = (1, self.patch_size, self.patch_size)
        if tuple(patches.shape[1:]) != expected:
            raise RuntimeError(f"2D patch decoder produced {tuple(patches.shape[1:])}, expected {expected}")
        patches = patches.reshape(b, n, 1, self.patch_size**2).permute(0, 2, 1, 3)
        return stitch_patches(
            patches,
            placement_indices=self.placement_indices,
            placement_valid=self.placement_valid,
            height=self.output_size,
            width=self.output_size,
        )[:, 0]


class ParticleTCNN3D(nn.Module):
    """Decode every frozen particle to a local 3D patch and add it to the canvas."""

    def __init__(
        self,
        representation_dim: int,
        channels: int,
        volume_depth: int = 16,
        patch_size: int = 5,
        output_size: int = 20,
        particle_grid_size: int = 20,
    ) -> None:
        super().__init__()
        validate_decoder_geometry(output_size, patch_size)
        if volume_depth != 16:
            raise ValueError("the V2 3D decoder currently requires volume_depth=16")
        self.patch_size = patch_size
        self.output_size = output_size
        self.particle_grid_size = particle_grid_size
        self.volume_depth = volume_depth
        self.channels = channels
        self.project = nn.Linear(representation_dim, channels)
        self.patch_decoder = make_patch_decoder_3d(channels, patch_size, volume_depth)
        placement_indices, placement_valid = build_patch_placement(
            particle_height=particle_grid_size,
            particle_width=particle_grid_size,
            output_height=output_size,
            output_width=output_size,
            patch_size=patch_size,
        )
        self.register_buffer("placement_indices", placement_indices, persistent=False)
        self.register_buffer("placement_valid", placement_valid, persistent=False)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        b, n, _ = representations.shape
        latent = self.project(representations).reshape(b * n, self.channels, 1, 1, 1)
        patches = self.patch_decoder(latent)[:, 0]
        expected = (self.volume_depth, self.patch_size, self.patch_size)
        if tuple(patches.shape[1:]) != expected:
            raise RuntimeError(f"3D patch decoder produced {tuple(patches.shape[1:])}, expected {expected}")
        patches = patches.reshape(b, n, self.volume_depth, self.patch_size**2)
        return stitch_patches(
            patches.permute(0, 2, 1, 3),
            placement_indices=self.placement_indices,
            placement_valid=self.placement_valid,
            height=self.output_size,
            width=self.output_size,
        )


def validate_decoder_geometry(output_size: int, patch_size: int) -> None:
    supported = {(20, 5), (128, 32)}
    if (output_size, patch_size) not in supported:
        raise ValueError(
            f"unsupported output/patch geometry {(output_size, patch_size)}; "
            f"supported geometries are {sorted(supported)}"
        )


def group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise ValueError(f"cannot construct GroupNorm for {channels} channels")


def make_patch_decoder_2d(channels: int, patch_size: int) -> nn.Sequential:
    if patch_size == 5:
        # Keep the original low-resolution architecture checkpoint-compatible.
        return nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=3),
            group_norm(channels),
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels // 2, kernel_size=3),
            group_norm(channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
        )
    if patch_size != 32:
        raise ValueError(f"unsupported 2D patch_size={patch_size}")
    c2, c3, c4 = max(channels // 2, 8), max(channels // 4, 8), max(channels // 8, 8)
    return nn.Sequential(
        nn.ConvTranspose2d(channels, channels, kernel_size=4),  # 1 -> 4
        group_norm(channels),
        nn.GELU(),
        nn.ConvTranspose2d(channels, c2, kernel_size=4, stride=2, padding=1),  # 4 -> 8
        group_norm(c2),
        nn.GELU(),
        nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),  # 8 -> 16
        group_norm(c3),
        nn.GELU(),
        nn.ConvTranspose2d(c3, c4, kernel_size=4, stride=2, padding=1),  # 16 -> 32
        group_norm(c4),
        nn.GELU(),
        nn.Conv2d(c4, 1, kernel_size=3, padding=1),
    )


def make_patch_decoder_3d(channels: int, patch_size: int, volume_depth: int) -> nn.Sequential:
    if patch_size == 5:
        return nn.Sequential(
            nn.ConvTranspose3d(channels, channels, kernel_size=(4, 3, 3)),
            group_norm(channels),
            nn.GELU(),
            nn.ConvTranspose3d(
                channels,
                channels // 2,
                kernel_size=(4, 3, 3),
                stride=(2, 1, 1),
                padding=(1, 0, 0),
            ),
            group_norm(channels // 2),
            nn.GELU(),
            nn.ConvTranspose3d(
                channels // 2,
                channels // 4,
                kernel_size=(4, 1, 1),
                stride=(2, 1, 1),
                padding=(1, 0, 0),
            ),
            group_norm(channels // 4),
            nn.GELU(),
            nn.Conv3d(channels // 4, 1, kernel_size=3, padding=1),
        )
    if patch_size != 32 or volume_depth != 16:
        raise ValueError(
            f"unsupported 3D patch geometry depth={volume_depth}, patch_size={patch_size}"
        )
    c2, c3, c4 = max(channels // 2, 8), max(channels // 4, 8), max(channels // 8, 8)
    return nn.Sequential(
        nn.ConvTranspose3d(channels, channels, kernel_size=4),  # 1^3 -> 4^3
        group_norm(channels),
        nn.GELU(),
        nn.ConvTranspose3d(channels, c2, kernel_size=4, stride=2, padding=1),  # 4 -> 8
        group_norm(c2),
        nn.GELU(),
        nn.ConvTranspose3d(c2, c3, kernel_size=4, stride=2, padding=1),  # 8 -> 16
        group_norm(c3),
        nn.GELU(),
        nn.ConvTranspose3d(
            c3,
            c4,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        ),  # 16x16x16 -> 16x32x32
        group_norm(c4),
        nn.GELU(),
        nn.Conv3d(c4, 1, kernel_size=3, padding=1),
    )


def build_patch_placement(
    *,
    particle_height: int,
    particle_width: int,
    output_height: int,
    output_width: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map row-major particles to their nearest output-grid centres."""
    centre_rows = torch.linspace(0, output_height - 1, particle_height).round().long()
    centre_cols = torch.linspace(0, output_width - 1, particle_width).round().long()
    centre_y, centre_x = torch.meshgrid(centre_rows, centre_cols, indexing="ij")
    offsets = torch.arange(patch_size, dtype=torch.long) - patch_size // 2
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    rows = centre_y.reshape(-1, 1, 1) + offset_y
    cols = centre_x.reshape(-1, 1, 1) + offset_x
    valid = (rows >= 0) & (rows < output_height) & (cols >= 0) & (cols < output_width)
    indices = rows.clamp(0, output_height - 1) * output_width + cols.clamp(0, output_width - 1)
    return indices.reshape(particle_height * particle_width, -1), valid.reshape(
        particle_height * particle_width, -1
    )


def stitch_patches(
    patches: torch.Tensor,
    *,
    placement_indices: torch.Tensor,
    placement_valid: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    # Original LESS combines particle patches by addition.  Do not average away
    # overlap: overlap count is part of the learned downstream prediction.
    if patches.ndim != 4:
        raise ValueError(f"expected patches [B,C,N,P], got {tuple(patches.shape)}")
    b, channels, particles, patch_area = patches.shape
    if tuple(placement_indices.shape) != (particles, patch_area):
        raise ValueError(
            f"placement shape {tuple(placement_indices.shape)} does not match "
            f"patch tensor {(particles, patch_area)}"
        )
    valid = placement_valid.to(dtype=patches.dtype).reshape(1, 1, particles, patch_area)
    source = (patches * valid).reshape(b, channels, particles * patch_area)
    indices = placement_indices.reshape(1, 1, particles * patch_area).expand(
        b, channels, -1
    )
    canvas = patches.new_zeros((b, channels, height * width))
    canvas.scatter_add_(2, indices, source)
    return canvas.reshape(b, channels, height, width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=8)
    parser.add_argument("--pretrain-epochs", type=int, default=1000)
    parser.add_argument("--decoder-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--decoder-batch-size",
        type=int,
        default=0,
        help="Decoder batch size; 0 reuses --batch-size.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=None,
        help="Reuse a completed best_representation.pt and skip Stage 1 training.",
    )
    parser.add_argument("--pretrain-learning-rate", type=float, default=2e-3)
    parser.add_argument("--decoder-learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--representation-dim", type=int, default=128)
    parser.add_argument("--decoder-channels", type=int, default=64)
    parser.add_argument(
        "--output-size",
        type=int,
        choices=(20, 128),
        default=20,
        help="2D/3D XY output size. 128 uses 32x32 local patches; default preserves 20x20/5x5.",
    )
    parser.add_argument("--particle-radius-m", type=float, default=0.0136)
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
        pretrain_epochs=args.pretrain_epochs,
        decoder_epochs=args.decoder_epochs,
        batch_size=args.batch_size,
        decoder_batch_size=args.decoder_batch_size,
        pretrain_learning_rate=args.pretrain_learning_rate,
        decoder_learning_rate=args.decoder_learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        representation_dim=args.representation_dim,
        decoder_channels=args.decoder_channels,
        output_size=args.output_size,
        patch_size=args.output_size // 4,
        particle_radius_m=args.particle_radius_m,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        log_every=args.log_every,
    )
    validate_config(config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage1_representation").mkdir(exist_ok=True)
    (out / "stage2_decoder_2d").mkdir(exist_ok=True)
    (out / "stage2_decoder_3d").mkdir(exist_ok=True)
    set_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("loading deterministic V2 subset", flush=True)
    selected = {
        "train": select_paths(Path(config.data_root) / "train", config.train_samples, config.seed + 1),
        "val": select_paths(Path(config.data_root) / "val", config.val_samples, config.seed + 2),
        "test": select_paths(Path(config.data_root) / "test", config.test_samples, config.seed + 3),
    }
    raw = {
        split: load_split_at_resolution(
            paths,
            volume_depth=config.volume_depth,
            output_size=config.output_size,
        )
        for split, paths in selected.items()
    }
    normalization = fit_normalization(raw["train"][0])
    curves = {
        split: normalize_force_curves(values[0], normalization)
        for split, values in raw.items()
    }
    xy = load_and_validate_xy(selected)
    neighbourhood = build_metric_neighbourhood(xy, config.particle_radius_m)
    loaders = {
        split: make_curve_loader(curves[split], config, shuffle=split == "train")
        for split in ("train", "val", "test")
    }

    model = LocalParticleRepresentation(
        curve_steps=curves["train"].shape[-1],
        embedding_dim=config.embedding_dim,
        representation_dim=config.representation_dim,
        neighbour_indices=torch.from_numpy(neighbourhood["indices"]),
        neighbour_mask=torch.from_numpy(neighbourhood["mask"]),
        relative_xy=torch.from_numpy(neighbourhood["relative_xy"]),
        centre_slots=torch.from_numpy(neighbourhood["centre_slots"]),
    ).to(device)
    manifest = make_manifest(config, normalization, selected, raw, xy, neighbourhood, model)
    write_json(out / "run_manifest.json", manifest)

    started = time.perf_counter()
    if args.stage1_checkpoint is None:
        stage1 = train_representation(model, loaders, device, config, out / "stage1_representation")
        stage1_checkpoint = out / "stage1_representation" / "best_representation.pt"
    else:
        stage1_checkpoint = args.stage1_checkpoint.resolve()
        if not stage1_checkpoint.is_file():
            raise FileNotFoundError(stage1_checkpoint)
        reused = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
        if reused.get("stage") != "self_supervised_local_force_reconstruction":
            raise ValueError(f"not a Stage 1 representation checkpoint: {stage1_checkpoint}")
        reused_config = reused.get("config", {})
        required_match = {
            "data_root": config.data_root,
            "train_samples": config.train_samples,
            "val_samples": config.val_samples,
            "test_samples": config.test_samples,
            "embedding_dim": config.embedding_dim,
            "representation_dim": config.representation_dim,
            "particle_radius_m": config.particle_radius_m,
        }
        mismatches = {
            key: {"checkpoint": reused_config.get(key), "current": expected}
            for key, expected in required_match.items()
            if reused_config.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Stage 1 checkpoint contract mismatch: {mismatches}")
        stage1 = {
            "best_epoch": int(reused["epoch"]),
            "best_val_force_reconstruction_mse": float(reused["val"]["loss"]),
            "reused_completed_checkpoint": True,
            "source_checkpoint": str(stage1_checkpoint),
            "source_checkpoint_sha256": file_sha256(stage1_checkpoint),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        print(
            f"stage1 reuse epoch={stage1['best_epoch']} "
            f"val_mse={stage1['best_val_force_reconstruction_mse']:.8f} "
            f"checkpoint={stage1_checkpoint}",
            flush=True,
        )
    checkpoint = torch.load(stage1_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("representation model was not frozen before decoder training")

    representation_arrays = {
        split: extract_representations(model, curves[split], config, device)
        for split in ("train", "val", "test")
    }
    representation_sha256 = tensor_sha256(model.state_dict())

    decoder_2d = ParticleTCNN2D(
        config.representation_dim,
        config.decoder_channels,
        patch_size=config.patch_size,
        output_size=config.output_size,
    ).to(device)
    stage2_2d = train_decoder(
        decoder_2d,
        task="2d",
        representations=representation_arrays,
        targets={split: raw[split][2] for split in raw},
        device=device,
        config=config,
        stage_dir=out / "stage2_decoder_2d",
        frozen_representation_sha256=representation_sha256,
    )

    decoder_3d = ParticleTCNN3D(
        config.representation_dim,
        config.decoder_channels,
        volume_depth=config.volume_depth,
        patch_size=config.patch_size,
        output_size=config.output_size,
    ).to(device)
    stage2_3d = train_decoder(
        decoder_3d,
        task="3d",
        representations=representation_arrays,
        targets={split: raw[split][3] for split in raw},
        device=device,
        config=config,
        stage_dir=out / "stage2_decoder_3d",
        frozen_representation_sha256=representation_sha256,
    )

    probabilities_2d = collect_decoder_predictions(
        decoder_2d, representation_arrays, device, config
    )
    probabilities_3d = collect_decoder_predictions(
        decoder_3d, representation_arrays, device, config
    )
    thresholds = {
        "2d": choose_threshold(probabilities_2d["val"], raw["val"][2]),
        "3d": choose_threshold(probabilities_3d["val"], raw["val"][3]),
    }
    reports: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        reports[split] = {
            "fixed_0.5": {
                "2d": binary_metrics(probabilities_2d[split], raw[split][2], 0.5),
                "3d": binary_metrics(probabilities_3d[split], raw[split][3], 0.5),
            },
            "val_selected_threshold": {
                "2d": binary_metrics(probabilities_2d[split], raw[split][2], thresholds["2d"]),
                "3d": binary_metrics(probabilities_3d[split], raw[split][3], thresholds["3d"]),
            },
        }

    np.savez_compressed(
        out / "test_predictions.npz",
        probabilities_2d=probabilities_2d["test"],
        probabilities_3d=probabilities_3d["test"],
        targets_2d=raw["test"][2],
        targets_3d=raw["test"][3],
    )
    summary = {
        "protocol": "two-stage LESS port: self-supervised local force reconstruction, then frozen task decoders",
        "completed_epochs": {
            "stage1_representation": config.pretrain_epochs,
            "stage2_decoder_2d": config.decoder_epochs,
            "stage2_decoder_3d": config.decoder_epochs,
        },
        "best_epoch": stage2_3d["best_epoch"],
        "best_epochs": {
            "stage1_representation": stage1["best_epoch"],
            "stage2_decoder_2d": stage2_2d["best_epoch"],
            "stage2_decoder_3d": stage2_3d["best_epoch"],
        },
        "stage1": stage1,
        "stage2_2d": stage2_2d,
        "stage2_3d": stage2_3d,
        "representation_frozen_for_stage2": True,
        "representation_sha256_used_by_both_decoders": representation_sha256,
        "thresholds_selected_on_validation": thresholds,
        "metrics": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "metrics.json", summary)
    print("RESULT " + json.dumps(summary, sort_keys=True), flush=True)


def normalize_force_curves(fz: np.ndarray, normalization: dict[str, float]) -> np.ndarray:
    delta_fz = np.maximum(fz - fz[..., :1], 0.0)
    transformed = np.log1p(delta_fz)
    normalized = (
        transformed - normalization["force_log_mean"]
    ) / normalization["force_log_std"]
    b, h, w, t = normalized.shape
    return normalized.reshape(b, h * w, t).astype(np.float32)


def validate_config(config: Config) -> None:
    validate_decoder_geometry(config.output_size, config.patch_size)
    if config.volume_depth != 16:
        raise ValueError("volume_depth must remain 16 for the matched V2 3D target")
    if config.decoder_channels < 8:
        raise ValueError("decoder_channels must be at least 8")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.decoder_batch_size < 0:
        raise ValueError("decoder_batch_size must be non-negative")


def load_split_at_resolution(
    paths: Iterable[Path],
    *,
    volume_depth: int,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load native curves and rasterize labels on the requested scan-area grid."""
    paths = list(paths)
    native = load_native_split(paths, volume_depth=volume_depth)
    if output_size == native[2].shape[-1]:
        return native

    masks_2d: list[np.ndarray] = []
    masks_3d: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            scan_xy = np.asarray(data["xy"], dtype=np.float32)
            lumps = json.loads(str(np.asarray(data["lumps_json"]).item()))
            phantom = json.loads(str(np.asarray(data["phantom_json"]).item()))
        target_xy = scan_area_label_xy(scan_xy, output_size)
        mask_3d, mask_2d = make_geometry_targets(
            target_xy,
            lumps,
            float(phantom["height"]),
            volume_depth,
        )
        masks_2d.append(mask_2d.astype(np.float32))
        masks_3d.append(mask_3d.astype(np.float32))
    return (
        native[0],
        native[1],
        np.stack(masks_2d),
        np.stack(masks_3d),
        native[4],
    )


def scan_area_label_xy(native_scan_xy: np.ndarray, output_size: int) -> np.ndarray:
    """Match the Grid U-Net analytic scan-area label-grid convention."""
    if native_scan_xy.shape != (20, 20, 2):
        raise ValueError(f"unexpected native scan xy shape: {native_scan_xy.shape}")
    expected_x = np.broadcast_to(native_scan_xy[0:1, :, 0], (20, 20))
    expected_y = np.broadcast_to(native_scan_xy[:, 0:1, 1], (20, 20))
    if not np.allclose(native_scan_xy[..., 0], expected_x, rtol=0.0, atol=1e-7):
        raise ValueError("scan x coordinates are not a rectilinear grid")
    if not np.allclose(native_scan_xy[..., 1], expected_y, rtol=0.0, atol=1e-7):
        raise ValueError("scan y coordinates are not a rectilinear grid")
    xs = np.linspace(
        float(native_scan_xy[0, 0, 0]),
        float(native_scan_xy[0, -1, 0]),
        output_size,
        dtype=np.float32,
    )
    ys = np.linspace(
        float(native_scan_xy[0, 0, 1]),
        float(native_scan_xy[-1, 0, 1]),
        output_size,
        dtype=np.float32,
    )
    xv, yv = np.meshgrid(xs, ys)
    return np.stack((xv, yv), axis=-1).astype(np.float32)


def load_and_validate_xy(selected: dict[str, list[Path]]) -> np.ndarray:
    reference: np.ndarray | None = None
    for split, paths in selected.items():
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                current = np.asarray(data["xy"], dtype=np.float32)
            if current.shape != (20, 20, 2):
                raise ValueError(f"unexpected xy grid in {path}: {current.shape}")
            if reference is None:
                reference = current
            elif not np.allclose(current, reference, rtol=0.0, atol=1e-7):
                raise ValueError(f"{split}/{path.name} does not share the fixed V2 xy grid")
    if reference is None:
        raise ValueError("no selected samples")
    return reference


def build_metric_neighbourhood(xy: np.ndarray, radius_m: float) -> dict[str, np.ndarray | float | int]:
    flat = xy.reshape(-1, 2).astype(np.float32)
    distances = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    neighbours = [np.flatnonzero(distances[index] <= radius_m + 1e-8) for index in range(len(flat))]
    for index, values in enumerate(neighbours):
        if index not in values:
            raise RuntimeError(f"particle {index} is missing its centre observation")
        order = np.lexsort((values, distances[index, values]))
        neighbours[index] = values[order]
    maximum = max(len(values) for values in neighbours)
    indices = np.zeros((len(flat), maximum), dtype=np.int64)
    mask = np.zeros((len(flat), maximum), dtype=bool)
    relative = np.zeros((len(flat), maximum, 2), dtype=np.float32)
    centre_slots = np.zeros(len(flat), dtype=np.int64)
    for particle, values in enumerate(neighbours):
        count = len(values)
        indices[particle, :count] = values
        mask[particle, :count] = True
        relative[particle, :count] = (flat[values] - flat[particle]) / radius_m
        centre_slots[particle] = int(np.flatnonzero(values == particle)[0])
    step_x = float(np.median(np.diff(np.unique(flat[:, 0]))))
    step_y = float(np.median(np.diff(np.unique(flat[:, 1]))))
    return {
        "indices": indices,
        "mask": mask,
        "relative_xy": relative,
        "centre_slots": centre_slots,
        "num_particles": len(flat),
        "max_neighbours": maximum,
        "min_neighbours": min(len(values) for values in neighbours),
        "mean_neighbours": float(np.mean([len(values) for values in neighbours])),
        "radius_m": radius_m,
        "grid_step_x_m": step_x,
        "grid_step_y_m": step_y,
    }


def make_curve_loader(
    curves: np.ndarray,
    config: Config,
    *,
    shuffle: bool,
) -> DataLoader:
    # Stage 1 is label-free; keeping high-resolution masks out of this loader
    # avoids duplicating several gigabytes of 3D targets in worker state.
    dataset = TensorDataset(torch.from_numpy(curves))
    generator = torch.Generator().manual_seed(config.seed + (11 if shuffle else 17))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=generator,
    )


def train_representation(
    model: LocalParticleRepresentation,
    loaders: dict[str, DataLoader],
    device: torch.device,
    config: Config,
    stage_dir: Path,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.pretrain_learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.pretrain_epochs,
        eta_min=config.pretrain_learning_rate * 0.01,
    )
    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, config.pretrain_epochs + 1):
        train_values = reconstruction_epoch(model, loaders["train"], device, optimizer, config.amp)
        val_values = reconstruction_epoch(model, loaders["val"], device, None, config.amp)
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            **{f"train_{key}": value for key, value in train_values.items()},
            **{f"val_{key}": value for key, value in val_values.items()},
        }
        history.append(row)
        if val_values["loss"] < best_loss:
            best_loss = val_values["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "stage": "self_supervised_local_force_reconstruction",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "val": val_values,
                },
                stage_dir / "best_representation.pt",
            )
        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.pretrain_epochs:
            print(
                f"stage1 epoch={epoch:04d}/{config.pretrain_epochs} "
                f"train_mse={train_values['loss']:.5f} val_mse={val_values['loss']:.5f} "
                f"val_rmse={val_values['rmse']:.5f}",
                flush=True,
            )
            write_history(stage_dir / "history.csv", history)
    write_history(stage_dir / "history.csv", history)
    return {
        "best_epoch": best_epoch,
        "best_val_force_reconstruction_mse": best_loss,
        "initial_train_force_reconstruction_mse": float(history[0]["train_loss"]),
        "final_train_force_reconstruction_mse": float(history[-1]["train_loss"]),
        "relative_train_loss_drop": relative_drop(history, "train_loss"),
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def reconstruction_epoch(
    model: LocalParticleRepresentation,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_squared = 0.0
    total_absolute = 0.0
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for (curves,) in loader:
            curves = curves.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                prediction, target = model.reconstruct_one_local_curve(
                    curves, random_target=training
                )
                loss = F.mse_loss(prediction, target)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            error = prediction.detach().float() - target.detach().float()
            total_squared += float((error**2).sum())
            total_absolute += float(error.abs().sum())
            count += error.numel()
    mse = total_squared / max(count, 1)
    return {"loss": mse, "rmse": math.sqrt(mse), "mae": total_absolute / max(count, 1)}


def extract_representations(
    model: LocalParticleRepresentation,
    curves: np.ndarray,
    config: Config,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(curves)), batch_size=config.batch_size, shuffle=False
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.amp and device.type == "cuda",
            ):
                representation = model.encode(batch.to(device, non_blocking=True))
            outputs.append(representation.float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def train_decoder(
    model: nn.Module,
    *,
    task: str,
    representations: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    device: torch.device,
    config: Config,
    stage_dir: Path,
    frozen_representation_sha256: str,
) -> dict[str, Any]:
    # Keep only the train/validation tensors resident. Test labels are not used
    # for fitting and high-resolution 3D labels are large.
    tensors = {
        split: (
            torch.from_numpy(representations[split]).to(device),
            torch.from_numpy(targets[split]).to(device),
        )
        for split in ("train", "val")
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.decoder_learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.decoder_epochs,
        eta_min=config.decoder_learning_rate * 0.01,
    )
    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, config.decoder_epochs + 1):
        train_values = decoder_tensor_epoch(
            model,
            *tensors["train"],
            optimizer=optimizer,
            amp=config.amp,
            batch_size=effective_decoder_batch_size(config),
            shuffle=True,
        )
        val_values = decoder_tensor_epoch(
            model,
            *tensors["val"],
            optimizer=None,
            amp=config.amp,
            batch_size=effective_decoder_batch_size(config),
            shuffle=False,
        )
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            **{f"train_{key}": value for key, value in train_values.items()},
            **{f"val_{key}": value for key, value in val_values.items()},
        }
        history.append(row)
        if val_values["loss"] < best_loss:
            best_loss = val_values["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "stage": f"frozen_representation_decoder_{task}",
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(config),
                    "frozen_representation_sha256": frozen_representation_sha256,
                    "val": val_values,
                },
                stage_dir / f"best_decoder_{task}.pt",
            )
        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.decoder_epochs:
            print(
                f"stage2-{task} epoch={epoch:04d}/{config.decoder_epochs} "
                f"train_loss={train_values['loss']:.5f} val_loss={val_values['loss']:.5f} "
                f"val_dice@0.5={val_values['dice']:.4f}",
                flush=True,
            )
            write_history(stage_dir / "history.csv", history)
    write_history(stage_dir / "history.csv", history)
    checkpoint = torch.load(
        stage_dir / f"best_decoder_{task}.pt", map_location=device, weights_only=False
    )
    if checkpoint["frozen_representation_sha256"] != frozen_representation_sha256:
        raise RuntimeError("decoder checkpoint representation hash mismatch")
    model.load_state_dict(checkpoint["model"])
    result = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "initial_train_loss": float(history[0]["train_loss"]),
        "final_train_loss": float(history[-1]["train_loss"]),
        "relative_train_loss_drop": relative_drop(history, "train_loss"),
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "frozen_representation_sha256": frozen_representation_sha256,
    }
    del tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def make_representation_loader(
    representations: np.ndarray,
    targets: np.ndarray,
    config: Config,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + (31 if shuffle else 37))
    return DataLoader(
        TensorDataset(torch.from_numpy(representations), torch.from_numpy(targets)),
        batch_size=effective_decoder_batch_size(config),
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        generator=generator,
    )


def decoder_tensor_epoch(
    model: nn.Module,
    representations: torch.Tensor,
    targets: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer | None,
    amp: bool,
    batch_size: int,
    shuffle: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    dice_total = 0.0
    samples = 0
    device = representations.device
    order = (
        torch.randperm(representations.shape[0], device=device)
        if shuffle
        else torch.arange(representations.shape[0], device=device)
    )
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_representations = representations.index_select(0, indices)
            batch_targets = targets.index_select(0, indices)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                logits = model(batch_representations)
                loss = combined_loss(logits, batch_targets)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            batch = int(batch_targets.shape[0])
            loss_total += float(loss.detach()) * batch
            dice_total += batch_dice(logits.detach(), batch_targets) * batch
            samples += batch
    return {"loss": loss_total / max(samples, 1), "dice": dice_total / max(samples, 1)}


def collect_decoder_predictions(
    model: nn.Module,
    representations: dict[str, np.ndarray],
    device: torch.device,
    config: Config,
) -> dict[str, np.ndarray]:
    model.eval()
    result: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for split, values in representations.items():
            tensor = torch.from_numpy(values).to(device)
            outputs: list[np.ndarray] = []
            batch_size = effective_decoder_batch_size(config)
            for start in range(0, tensor.shape[0], batch_size):
                batch = tensor[start : start + batch_size]
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=config.amp and device.type == "cuda",
                ):
                    logits = model(batch)
                outputs.append(logits.sigmoid().float().cpu().numpy())
            result[split] = np.concatenate(outputs)
            del tensor
    return result


def effective_decoder_batch_size(config: Config) -> int:
    return config.decoder_batch_size or config.batch_size


def batch_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    prediction = logits.sigmoid() >= threshold
    truth = targets >= 0.5
    dims = tuple(range(1, targets.ndim))
    intersection = (prediction & truth).sum(dim=dims).float()
    denominator = prediction.sum(dim=dims).float() + truth.sum(dim=dims).float()
    values = torch.where(
        denominator > 0, 2.0 * intersection / denominator, torch.ones_like(denominator)
    )
    return float(values.mean())


def relative_drop(history: list[dict[str, float | int]], key: str) -> float:
    first = float(history[0][key])
    last = float(history[-1][key])
    return float((first - last) / max(abs(first), 1e-8))


def make_manifest(
    config: Config,
    normalization: dict[str, float],
    selected: dict[str, list[Path]],
    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    xy: np.ndarray,
    neighbourhood: dict[str, np.ndarray | float | int],
    model: nn.Module,
) -> dict[str, Any]:
    return {
        "kind": "two-stage LESS port for synthetic palpation V2",
        "config": asdict(config),
        "training_contract": {
            "stage1": "self-supervised held-out local force-curve reconstruction; no 2D/3D labels",
            "stage2_2d": "load best stage1 representation, freeze it, train only particle TCNN 2D decoder",
            "stage2_3d": "load the same frozen representation, train only particle TCNN 3D decoder",
            "joint_end_to_end_training": False,
        },
        "input_semantics": "preload-subtracted delta Fz, log1p, train-set normalization; one 40-step curve per scan location",
        "target_contract": {
            "label_policy": "analytic lump projection/occupancy on scan-area grid centres",
            "xy_extent_source": "native 20x20 palpation xy coordinates, not full-phantom label_xy",
            "two_dimensional_shape": [config.output_size, config.output_size],
            "three_dimensional_shape": [
                config.volume_depth,
                config.output_size,
                config.output_size,
            ],
            "xy_grid_matches_grid_unet_baseline": config.output_size == 128,
        },
        "normalization": normalization,
        "selected_samples": {
            split: [path.name for path in paths] for split, paths in selected.items()
        },
        "data_qa": {
            split: {
                "samples": len(selected[split]),
                "native_full_phantom_projection_consistency_mean": float(raw[split][4].mean()),
                "native_full_phantom_projection_consistency_min": float(raw[split][4].min()),
                "positive_fraction_2d": float(raw[split][2].mean()),
                "positive_fraction_3d": float(raw[split][3].mean()),
            }
            for split in ("train", "val", "test")
        },
        "particles": {
            "distribution": "fixed particles at the shared V2 physical XY scan coordinates",
            "grid_shape": list(xy.shape[:2]),
            "num_particles": int(neighbourhood["num_particles"]),
            "x_range_m": [float(xy[..., 0].min()), float(xy[..., 0].max())],
            "y_range_m": [float(xy[..., 1].min()), float(xy[..., 1].max())],
            "grid_step_x_m": float(neighbourhood["grid_step_x_m"]),
            "grid_step_y_m": float(neighbourhood["grid_step_y_m"]),
            "assignment": "Euclidean cdist in unnormalised physical XY coordinates",
            "max_particle_distance_m": float(neighbourhood["radius_m"]),
            "min_neighbours": int(neighbourhood["min_neighbours"]),
            "mean_neighbours": float(neighbourhood["mean_neighbours"]),
            "max_neighbours": int(neighbourhood["max_neighbours"]),
        },
        "decoder": {
            "type": "independent per-particle transposed-convolution patch prediction",
            "patch_size_pixels": config.patch_size,
            "output_size_pixels": config.output_size,
            "patch_fraction_of_canvas": config.patch_size / config.output_size,
            "combination": "additive overlap, matching released LESS default",
            "particle_to_canvas_placement": "nearest output-grid centre in physical scan order",
            "two_dimensional": f"{config.patch_size}x{config.patch_size} patch",
            "three_dimensional": (
                f"{config.volume_depth}x{config.patch_size}x{config.patch_size} patch"
            ),
        },
        "parameter_count_stage1": sum(parameter.numel() for parameter in model.parameters()),
        "environment": environment_snapshot(),
    }


def tensor_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode("utf-8"))
        digest.update(state_dict[name].detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": sys.version,
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    main()
