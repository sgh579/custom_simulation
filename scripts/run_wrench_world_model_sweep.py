from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from palpation_sim.native_data import load_phantom_scan_material_lumps
from palpation_sim.workflow import require_runtime_environment
from run_highres_segmentation_sweep import highres_mask_for_scan_area
from run_segmentation_accuracy_sweep import (
    _method_done,
    _read_json,
    _seed_everything,
    _write_csv,
    _write_json,
    counts_from_prediction,
    dice_loss,
    evaluate_scores,
    metrics_from_counts,
    write_score_method,
)
from run_highres_segmentation_sweep import write_test_metrics


DEFAULT_PACKAGE_DIR = Path("data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/random_trajectory")
DEFAULT_OUT_DIR = Path("runs/wrench_world_model_sweep_20260622")
DEFAULT_BASELINE_LEADERBOARD = Path(
    "runs/nonlinear_trajectory_20x_repeats10_seed20260618/highres_wrench_temporal_variants/leaderboard_with_baselines.csv"
)
WRENCH_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
DEFAULT_VARIANTS = (
    "wrench_vit_unet",
    "wrench_jepa_vit_unet",
    "wrench_lejepa_vit_unet",
    "wrench_visreg_vit_unet",
    "wrench_temporal_cnn_sparse_unet",
    "wrench_action_world_perceiver",
    "wrench_world_with_curve_decoder",
    "wrench_temporal_cnn_world_unet",
    "wrench_temporal_cnn_world_curve_unet",
)
ALLOWED_VARIANTS = DEFAULT_VARIANTS + ("wrench_world_with_mask_diffusion_aux",)


@dataclass(frozen=True)
class WrenchSplit:
    names: list[str]
    wrench_raw: np.ndarray
    wrench_x: np.ndarray
    action_raw: np.ndarray
    action_x: np.ndarray
    world_x: np.ndarray
    xy: np.ndarray
    masks: np.ndarray


@dataclass(frozen=True)
class NormalizationStats:
    wrench_mean: np.ndarray
    wrench_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


class SegmentationTensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y[:, None].astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class TensorOnlyDataset(Dataset):
    def __init__(self, x: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 6D wrench JEPA / world-model segmentation methods.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline-leaderboard", type=Path, default=DEFAULT_BASELINE_LEADERBOARD)
    parser.add_argument("--resolutions", type=str, default="128")
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrain-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--pretrain-patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=0, help="0 picks a conservative 5090-sized default.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--temporal-depth", type=int, default=2)
    parser.add_argument("--spatial-depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--decoder-base-channels", type=int, default=32)
    parser.add_argument("--perceiver-latents", type=int, default=128)
    parser.add_argument("--perceiver-depth", type=int, default=6)
    parser.add_argument("--mask-ratios", type=str, default="0.5,0.75")
    parser.add_argument("--world-target-ratio", type=float, default=0.35)
    parser.add_argument("--seg-context-min", type=float, default=0.25)
    parser.add_argument("--seg-context-max", type=float, default=1.0)
    parser.add_argument("--sparse-context-ratios", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--curve-aux-weight", type=float, default=0.1)
    parser.add_argument("--mask-aux-weight", type=float, default=0.05)
    parser.add_argument("--jepa-ema", type=float, default=0.996)
    parser.add_argument("--lejepa-reg-weight", type=float, default=0.05)
    parser.add_argument("--visreg-projections", type=int, default=64)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    require_runtime_environment()
    _seed_everything(args.seed)
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.pretrain_epochs = min(args.pretrain_epochs, 2)
        args.patience = min(args.patience, 2)
        args.pretrain_patience = min(args.pretrain_patience, 2)
        args.dim = min(args.dim, 64)
        args.temporal_depth = min(args.temporal_depth, 1)
        args.spatial_depth = min(args.spatial_depth, 2)
        args.heads = min(args.heads, 4)
        args.decoder_base_channels = min(args.decoder_base_channels, 16)
        args.perceiver_latents = min(args.perceiver_latents, 16)
        args.perceiver_depth = min(args.perceiver_depth, 2)

    args.batch_size = resolve_batch_size(args.batch_size, smoke=args.smoke)
    variants = parse_variants(args.variants)
    resolutions = parse_ints(args.resolutions)
    mask_ratios = parse_floats(args.mask_ratios)
    sparse_ratios = parse_floats(args.sparse_context_ratios)
    if not (0.0 < float(args.seg_context_min) <= float(args.seg_context_max) <= 1.0):
        raise ValueError("--seg-context-min/max must satisfy 0 < min <= max <= 1")
    if not (0.0 < float(args.world_target_ratio) < 1.0):
        raise ValueError("--world-target-ratio must be in (0, 1)")
    device = resolve_device(args.device)

    run_config = {
        "package_dir": str(args.package_dir),
        "out_dir": str(args.out_dir),
        "baseline_leaderboard": str(args.baseline_leaderboard),
        "variants": variants,
        "resolutions": resolutions,
        "input_contract": "probe_wrench[H,W,T,6] flattened as [t0_Fx,t0_Fy,...]",
        "action_contract": "xy + trajectory_xy_offset + indentation_depth + probe_pose",
        "wrench_channels": WRENCH_NAMES,
        "epochs": int(args.epochs),
        "pretrain_epochs": int(args.pretrain_epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "pretrain_lr": float(args.pretrain_lr),
        "dim": int(args.dim),
        "temporal_depth": int(args.temporal_depth),
        "spatial_depth": int(args.spatial_depth),
        "heads": int(args.heads),
        "perceiver_latents": int(args.perceiver_latents),
        "perceiver_depth": int(args.perceiver_depth),
        "mask_ratios": mask_ratios,
        "world_target_ratio": float(args.world_target_ratio),
        "seg_context_min": float(args.seg_context_min),
        "seg_context_max": float(args.seg_context_max),
        "sparse_context_ratios": sparse_ratios,
        "jepa_ema": float(args.jepa_ema),
        "lejepa_reg_weight": float(args.lejepa_reg_weight),
        "visreg_projections": int(args.visreg_projections),
        "amp": bool(not args.no_amp and device.type == "cuda"),
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
    }
    _write_json(args.out_dir / "run_config.json", run_config)

    data_root = args.package_dir / "data"
    for resolution in resolutions:
        max_train = 8 if args.smoke else None
        max_eval = 4 if args.smoke else None
        print(f"loading r{resolution} train split: {data_root / 'train'}")
        train_raw = load_world_split(data_root / "train", label_size=resolution, max_samples=max_train)
        print(f"loading r{resolution} val split: {data_root / 'val'}")
        val_raw = load_world_split(data_root / "val", label_size=resolution, max_samples=max_eval)
        test_raw = None
        if not args.no_test:
            print(f"loading r{resolution} test split: {data_root / 'test'}")
            test_raw = load_world_split(data_root / "test", label_size=resolution, max_samples=max_eval)

        stats = fit_normalization(train_raw)
        train = apply_normalization(train_raw, stats)
        val = apply_normalization(val_raw, stats)
        test = apply_normalization(test_raw, stats) if test_raw is not None else None

        for variant in variants:
            if variant == "wrench_vit_unet":
                run_vit_method(variant, resolution, train, val, test, run_config, args, device, sparse_ratios)
            elif variant in {"wrench_jepa_vit_unet", "wrench_lejepa_vit_unet", "wrench_visreg_vit_unet"}:
                for mask_ratio in mask_ratios:
                    run_jepa_method(variant, resolution, mask_ratio, train, val, test, run_config, args, device, sparse_ratios)
            elif variant in {
                "wrench_temporal_cnn_sparse_unet",
                "wrench_action_world_perceiver",
                "wrench_world_with_curve_decoder",
                "wrench_world_with_mask_diffusion_aux",
                "wrench_temporal_cnn_world_unet",
                "wrench_temporal_cnn_world_curve_unet",
            }:
                run_world_method(variant, resolution, train, val, test, run_config, args, device, sparse_ratios)
            else:
                raise ValueError(f"Unhandled variant: {variant}")
            write_leaderboards(args.out_dir, baseline_path=args.baseline_leaderboard)

    write_leaderboards(args.out_dir, baseline_path=args.baseline_leaderboard)
    print(f"wrench world-model sweep complete: {args.out_dir}")


def load_world_split(split_dir: Path, *, label_size: int, max_samples: int | None) -> dict[str, Any]:
    files = sorted(split_dir.glob("*.npz"))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")

    names: list[str] = []
    wrench_items: list[np.ndarray] = []
    action_items: list[np.ndarray] = []
    xy_items: list[np.ndarray] = []
    mask_items: list[np.ndarray] = []
    for path in files:
        with np.load(path, allow_pickle=False) as sample:
            wrench = np.asarray(sample["probe_wrench"], dtype=np.float32)
            if wrench.ndim != 4 or wrench.shape[-1] != 6:
                raise ValueError(f"{path}: expected probe_wrench[H,W,T,6], got {wrench.shape}")
            if not np.isfinite(wrench).all():
                raise ValueError(f"{path}: probe_wrench contains non-finite values")
            h, w, t, _channels = wrench.shape
            xy = np.asarray(sample["xy"], dtype=np.float32)
            trajectory_xy = np.asarray(sample["trajectory_xy_offset"], dtype=np.float32)
            indentation = np.asarray(sample["indentation_depth"], dtype=np.float32)
            pose = np.asarray(sample["probe_pose"], dtype=np.float32)
            if xy.shape != (h, w, 2):
                raise ValueError(f"{path}: expected xy shape {(h, w, 2)}, got {xy.shape}")
            if trajectory_xy.shape != (h, w, t, 2):
                raise ValueError(f"{path}: expected trajectory_xy_offset shape {(h, w, t, 2)}, got {trajectory_xy.shape}")
            if indentation.shape != (h, w, t):
                raise ValueError(f"{path}: expected indentation_depth shape {(h, w, t)}, got {indentation.shape}")
            if pose.shape != (h, w, t, 7):
                raise ValueError(f"{path}: expected probe_pose shape {(h, w, t, 7)}, got {pose.shape}")
            action = np.concatenate(
                [
                    xy,
                    trajectory_xy.reshape(h, w, t * 2),
                    indentation.reshape(h, w, t),
                    pose.reshape(h, w, t * 7),
                ],
                axis=-1,
            )
        phantom, scan, _material, lumps, _metadata = load_phantom_scan_material_lumps(
            sample_path=path,
            metadata_path=path.with_name(f"{path.stem}_gt.json"),
        )
        mask = highres_mask_for_scan_area(scan, phantom, lumps, label_size)
        names.append(path.name)
        wrench_items.append(wrench.astype(np.float32))
        action_items.append(np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        xy_items.append(xy.astype(np.float32))
        mask_items.append(mask.astype(np.uint8))

    return {
        "names": names,
        "wrench_raw": np.stack(wrench_items).astype(np.float32),
        "action_raw": np.stack(action_items).astype(np.float32),
        "xy": np.stack(xy_items).astype(np.float32),
        "masks": np.stack(mask_items).astype(np.uint8),
    }


def fit_normalization(raw: dict[str, Any]) -> NormalizationStats:
    wrench = raw["wrench_raw"].astype(np.float32)
    action = raw["action_raw"].astype(np.float32)
    wrench_mean = np.mean(wrench, axis=(0, 1, 2), keepdims=True).astype(np.float32)
    wrench_std = np.std(wrench, axis=(0, 1, 2), keepdims=True).astype(np.float32)
    action_mean = np.mean(action, axis=(0, 1, 2), keepdims=True).astype(np.float32)
    action_std = np.std(action, axis=(0, 1, 2), keepdims=True).astype(np.float32)
    return NormalizationStats(
        wrench_mean=wrench_mean,
        wrench_std=np.maximum(wrench_std, np.float32(1e-6)),
        action_mean=action_mean,
        action_std=np.maximum(action_std, np.float32(1e-6)),
    )


def apply_normalization(raw: dict[str, Any] | None, stats: NormalizationStats) -> WrenchSplit | None:
    if raw is None:
        return None
    wrench_raw = raw["wrench_raw"].astype(np.float32)
    action_raw = raw["action_raw"].astype(np.float32)
    wrench_norm = (wrench_raw - stats.wrench_mean) / stats.wrench_std
    action_norm = (action_raw - stats.action_mean) / stats.action_std
    wrench_x = wrench_to_chw(wrench_norm)
    action_x = action_norm.transpose(0, 3, 1, 2).astype(np.float32)
    world_x = np.concatenate([wrench_x, action_x], axis=1).astype(np.float32)
    return WrenchSplit(
        names=list(raw["names"]),
        wrench_raw=wrench_raw,
        wrench_x=wrench_x,
        action_raw=action_raw,
        action_x=action_x,
        world_x=world_x,
        xy=raw["xy"].astype(np.float32),
        masks=raw["masks"].astype(np.uint8),
    )


def wrench_to_chw(wrench_nhwtc: np.ndarray) -> np.ndarray:
    n, h, w, t, c = wrench_nhwtc.shape
    return wrench_nhwtc.transpose(0, 3, 4, 1, 2).reshape(n, t * c, h, w).astype(np.float32)


class WrenchTokenBackbone(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        dim: int,
        temporal_depth: int,
        spatial_depth: int,
        heads: int,
        grid_hw: tuple[int, int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if curve_channels % len(WRENCH_NAMES) != 0:
            raise ValueError(f"curve_channels={curve_channels} must be divisible by {len(WRENCH_NAMES)}")
        self.curve_channels = int(curve_channels)
        self.steps = int(curve_channels) // len(WRENCH_NAMES)
        self.dim = int(dim)
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.temporal_in = nn.Linear(len(WRENCH_NAMES), dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.steps, dim))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_depth)
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.grid_hw[0] * self.grid_hw[1], dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=spatial_depth)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def curve_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        curve = x[:, : self.curve_channels].reshape(b, self.steps, len(WRENCH_NAMES), h, w)
        curve = curve.permute(0, 3, 4, 1, 2).reshape(b * h * w, self.steps, len(WRENCH_NAMES))
        temporal = self.temporal_in(curve) + self.temporal_pos[:, : self.steps]
        temporal = self.temporal_encoder(temporal)
        tokens = temporal.mean(dim=1).reshape(b, h * w, self.dim)
        return tokens

    def forward(self, x: torch.Tensor, *, target_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, _c, h, w = x.shape
        tokens = self.curve_to_tokens(x)
        if tokens.shape[1] != self.spatial_pos.shape[1]:
            raise ValueError(f"Expected {self.spatial_pos.shape[1]} scan tokens, got {tokens.shape[1]}")
        if target_mask is not None:
            mask = target_mask.to(dtype=torch.bool, device=x.device).unsqueeze(-1)
            tokens = torch.where(mask, self.mask_token.to(dtype=tokens.dtype), tokens)
        tokens = tokens + self.spatial_pos
        return self.norm(self.spatial_encoder(tokens))


class WrenchViTUNet(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        output_shape: tuple[int, int],
        dim: int,
        temporal_depth: int,
        spatial_depth: int,
        heads: int,
        decoder_base_channels: int,
        grid_hw: tuple[int, int],
    ) -> None:
        super().__init__()
        self.backbone = WrenchTokenBackbone(
            curve_channels=curve_channels,
            dim=dim,
            temporal_depth=temporal_depth,
            spatial_depth=spatial_depth,
            heads=heads,
            grid_hw=grid_hw,
        )
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.decoder = SmallUNet(dim, base_channels=decoder_base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(x)
        b = tokens.shape[0]
        h, w = self.grid_hw
        feature_map = tokens.transpose(1, 2).reshape(b, -1, h, w)
        logits = self.decoder(feature_map)
        return resize_logits(logits, self.output_shape)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        b = int(base_channels)
        self.inc = conv_block(in_channels, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), conv_block(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), conv_block(b * 2, b * 4))
        self.up2 = conv_block(b * 4 + b * 2, b * 2)
        self.up1 = conv_block(b * 2 + b, b)
        self.outc = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x2, x], dim=1))
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x1, x], dim=1))
        return self.outc(x)


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.GELU(),
    )


class WrenchJEPAPretrainer(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        dim: int,
        temporal_depth: int,
        spatial_depth: int,
        heads: int,
        grid_hw: tuple[int, int],
        ema: float,
        reg_type: str,
        reg_weight: float,
        visreg_projections: int,
    ) -> None:
        super().__init__()
        self.online = WrenchTokenBackbone(
            curve_channels=curve_channels,
            dim=dim,
            temporal_depth=temporal_depth,
            spatial_depth=spatial_depth,
            heads=heads,
            grid_hw=grid_hw,
        )
        self.target = copy.deepcopy(self.online)
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.predictor = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ema = float(ema)
        self.reg_type = str(reg_type)
        self.reg_weight = float(reg_weight)
        self.visreg_projections = int(visreg_projections)

    def forward_loss(self, x: torch.Tensor, *, mask_ratio: float) -> tuple[torch.Tensor, dict[str, float]]:
        b, _c, h, w = x.shape
        target_mask = torch.rand((b, h * w), device=x.device) < float(mask_ratio)
        target_mask = ensure_any_masked_and_context(target_mask)
        online_tokens = self.online(x, target_mask=target_mask)
        with torch.no_grad():
            target_tokens = self.target(x)
        pred = self.predictor(online_tokens)
        masked = target_mask.unsqueeze(-1)
        latent_loss = F.mse_loss(pred[masked.expand_as(pred)], target_tokens[masked.expand_as(target_tokens)])
        if self.reg_type == "sigreg":
            reg_loss = sigreg_loss(online_tokens)
        elif self.reg_type == "visreg":
            reg_loss = visreg_loss(online_tokens, num_projections=self.visreg_projections)
        elif self.reg_type == "none":
            reg_loss = torch.zeros((), device=x.device)
        else:
            raise ValueError(f"Unknown JEPA regularizer: {self.reg_type}")
        loss = latent_loss + self.reg_weight * reg_loss
        return loss, {
            "latent_mse": float(latent_loss.detach().cpu()),
            "sigreg": float(reg_loss.detach().cpu()),
        }

    @torch.no_grad()
    def update_target(self) -> None:
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.mul_(self.ema).add_(online_param.data, alpha=1.0 - self.ema)


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, latents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        query = self.cross_norm(latents)
        key_value = self.token_norm(tokens)
        latents = latents + self.cross(query, key_value, key_value, need_weights=False)[0]
        latents = latents + self.self_attn(self.self_norm(latents), self.self_norm(latents), self.self_norm(latents), need_weights=False)[0]
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class ActionWorldPerceiver(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        action_channels: int,
        output_shape: tuple[int, int],
        dim: int,
        heads: int,
        depth: int,
        latent_slots: int,
        decoder_base_channels: int,
        grid_hw: tuple[int, int],
        target_ratio: float,
        curve_aux: bool,
        mask_diffusion_aux: bool,
    ) -> None:
        super().__init__()
        self.curve_channels = int(curve_channels)
        self.action_channels = int(action_channels)
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.target_ratio = float(target_ratio)
        self.curve_aux = bool(curve_aux)
        self.mask_diffusion_aux = bool(mask_diffusion_aux)
        self.token_proj = nn.Sequential(
            nn.Linear(self.curve_channels + self.action_channels + 1, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.action_query = nn.Sequential(nn.Linear(self.action_channels, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))
        self.latents = nn.Parameter(torch.zeros(1, int(latent_slots), dim))
        self.blocks = nn.ModuleList([PerceiverBlock(dim, heads) for _ in range(int(depth))])
        self.query_cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.query_norm = nn.LayerNorm(dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.query_ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.curve_decoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, self.curve_channels))
        self.mask_decoder = SmallUNet(dim, base_channels=decoder_base_channels)
        self.denoise_head = nn.Sequential(
            nn.Conv2d(dim + 1, min(64, dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(min(64, dim), 1, kernel_size=1),
        )
        nn.init.trunc_normal_(self.latents, std=0.02)

    def split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        curve = x[:, : self.curve_channels]
        action = x[:, self.curve_channels : self.curve_channels + self.action_channels]
        b, _c, h, w = x.shape
        curve_tokens = curve.permute(0, 2, 3, 1).reshape(b, h * w, self.curve_channels)
        action_tokens = action.permute(0, 2, 3, 1).reshape(b, h * w, self.action_channels)
        return curve_tokens, action_tokens

    def encode_latents(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        curve_tokens, action_tokens = self.split_input(x)
        b, n, _ = curve_tokens.shape
        if context_keep_mask is None:
            keep = torch.ones((b, n), dtype=torch.bool, device=x.device)
        else:
            keep = context_keep_mask.to(device=x.device, dtype=torch.bool)
        observed_curve = torch.where(keep.unsqueeze(-1), curve_tokens, torch.zeros_like(curve_tokens))
        observed_flag = keep.to(dtype=x.dtype).unsqueeze(-1)
        tokens = self.token_proj(torch.cat([action_tokens, observed_curve, observed_flag], dim=-1))
        latents = self.latents.expand(b, -1, -1)
        for block in self.blocks:
            latents = block(latents, tokens)
        return latents, action_tokens

    def query_tokens(self, latents: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        query = self.action_query(action_tokens)
        query = query + self.query_cross(self.query_norm(query), self.latent_norm(latents), self.latent_norm(latents), need_weights=False)[0]
        query = query + self.query_ffn(query)
        return query

    def token_feature_map(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        latents, action_tokens = self.encode_latents(x, context_keep_mask=context_keep_mask)
        tokens = self.query_tokens(latents, action_tokens)
        b = tokens.shape[0]
        h, w = self.grid_hw
        return tokens.transpose(1, 2).reshape(b, -1, h, w)

    def forward(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        feature_map = self.token_feature_map(x, context_keep_mask=context_keep_mask)
        logits = self.mask_decoder(feature_map)
        return resize_logits(logits, self.output_shape)

    def predict_curves(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor) -> torch.Tensor:
        latents, action_tokens = self.encode_latents(x, context_keep_mask=context_keep_mask)
        tokens = self.query_tokens(latents, action_tokens)
        return self.curve_decoder(tokens)

    def auxiliary_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        losses: list[torch.Tensor] = []
        stats: dict[str, float] = {}
        curve_tokens, _action_tokens = self.split_input(x)
        b, n, _ = curve_tokens.shape
        target_mask = torch.rand((b, n), device=x.device) < self.target_ratio
        target_mask = ensure_any_masked_and_context(target_mask)
        context_keep = ~target_mask
        pred = self.predict_curves(x, context_keep_mask=context_keep)
        curve_loss = F.smooth_l1_loss(pred[target_mask], curve_tokens[target_mask])
        if self.curve_aux:
            losses.append(curve_loss)
        stats["curve_aux_huber"] = float(curve_loss.detach().cpu())
        if self.mask_diffusion_aux and y is not None:
            features = self.token_feature_map(x)
            y_low = F.interpolate(y, size=self.grid_hw, mode="bilinear", align_corners=False)
            sigma = torch.rand((y.shape[0], 1, 1, 1), device=y.device) * 0.5 + 0.05
            noisy = torch.clamp(y_low + sigma * torch.randn_like(y_low), 0.0, 1.0)
            denoised = self.denoise_head(torch.cat([features, noisy], dim=1))
            mask_loss = F.mse_loss(torch.sigmoid(denoised), y_low)
            losses.append(mask_loss)
            stats["mask_denoise_mse"] = float(mask_loss.detach().cpu())
        if not losses:
            return torch.zeros((), device=x.device), stats
        return sum(losses), stats


class TemporalCNNWorldUNet(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        action_channels: int,
        output_shape: tuple[int, int],
        embed_channels: int,
        hidden_channels: int,
        decoder_base_channels: int,
        grid_hw: tuple[int, int],
        target_ratio: float,
        curve_aux: bool,
    ) -> None:
        super().__init__()
        if int(curve_channels) % len(WRENCH_NAMES) != 0:
            raise ValueError(f"curve_channels={curve_channels} must be divisible by {len(WRENCH_NAMES)}")
        self.curve_channels = int(curve_channels)
        self.action_channels = int(action_channels)
        self.temporal_feature_size = len(WRENCH_NAMES)
        self.steps = self.curve_channels // self.temporal_feature_size
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.target_ratio = float(target_ratio)
        self.curve_aux = bool(curve_aux)
        hidden = int(hidden_channels)
        embed = int(embed_channels)
        self.temporal = nn.Sequential(
            nn.Conv1d(self.temporal_feature_size, hidden, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, embed, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, embed), num_channels=embed),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.action_proj = nn.Sequential(
            nn.Conv2d(self.action_channels, embed, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, embed), num_channels=embed),
            nn.GELU(),
        )
        self.flag_proj = nn.Conv2d(1, embed, kernel_size=1)
        self.context_mixer = nn.Sequential(
            conv_block(embed, embed),
            nn.Conv2d(embed, embed, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(num_groups=min(8, embed), num_channels=embed),
            nn.GELU(),
            conv_block(embed, embed),
        )
        self.mask_decoder = SmallUNet(embed, base_channels=decoder_base_channels)
        self.curve_decoder = nn.Sequential(
            nn.Conv2d(embed, embed, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed, self.curve_channels, kernel_size=1),
        )

    def split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        curve = x[:, : self.curve_channels]
        action = x[:, self.curve_channels : self.curve_channels + self.action_channels]
        b, _c, h, w = x.shape
        curve_tokens = curve.permute(0, 2, 3, 1).reshape(b, h * w, self.curve_channels)
        action_tokens = action.permute(0, 2, 3, 1).reshape(b, h * w, self.action_channels)
        return curve_tokens, action_tokens

    def encode_feature_map(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, _c, h, w = x.shape
        curve = x[:, : self.curve_channels]
        action = x[:, self.curve_channels : self.curve_channels + self.action_channels]
        if context_keep_mask is None:
            keep = torch.ones((b, h * w), dtype=torch.bool, device=x.device)
        else:
            keep = context_keep_mask.to(device=x.device, dtype=torch.bool)
        keep_map = keep.reshape(b, 1, h, w).to(dtype=x.dtype)
        observed_curve = curve * keep_map
        seq = observed_curve.reshape(b, self.steps, self.temporal_feature_size, h, w)
        seq = seq.permute(0, 3, 4, 2, 1).reshape(b * h * w, self.temporal_feature_size, self.steps)
        encoded = self.temporal(seq).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        feature = encoded + self.action_proj(action) + self.flag_proj(keep_map)
        return self.context_mixer(feature)

    def forward(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        feature = self.encode_feature_map(x, context_keep_mask=context_keep_mask)
        return resize_logits(self.mask_decoder(feature), self.output_shape)

    def predict_curves(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor) -> torch.Tensor:
        feature = self.encode_feature_map(x, context_keep_mask=context_keep_mask)
        pred = self.curve_decoder(feature)
        b, _c, h, w = pred.shape
        return pred.permute(0, 2, 3, 1).reshape(b, h * w, self.curve_channels)

    def auxiliary_loss(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        curve_tokens, _action_tokens = self.split_input(x)
        b, n, _ = curve_tokens.shape
        target_mask = torch.rand((b, n), device=x.device) < self.target_ratio
        target_mask = ensure_any_masked_and_context(target_mask)
        pred = self.predict_curves(x, context_keep_mask=~target_mask)
        curve_loss = F.smooth_l1_loss(pred[target_mask], curve_tokens[target_mask])
        if not self.curve_aux:
            return torch.zeros((), device=x.device), {"curve_aux_huber": float(curve_loss.detach().cpu())}
        return curve_loss, {"curve_aux_huber": float(curve_loss.detach().cpu())}


def run_vit_method(
    variant: str,
    resolution: int,
    train: WrenchSplit,
    val: WrenchSplit,
    test: WrenchSplit | None,
    run_config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    sparse_ratios: list[float],
) -> None:
    method = f"r{resolution}_{variant}"
    model = make_vit_model(train.wrench_x, resolution, args)
    context = method_context(run_config, variant, method, resolution, train, val, test, input_name="probe_wrench", model_name="vit_unet")
    train_segmentation_method(method, model, train.wrench_x, train.masks, val.wrench_x, val.masks, val.names, test.wrench_x if test else None, test.masks if test else None, test.names if test else None, args.out_dir, context, args, device)
    write_no_world_metrics(args.out_dir / method, method, reason="supervised backbone has no action-conditioned predictor")
    write_sparse_segmentation_metrics(args.out_dir / method, method, model, test, device, args, sparse_ratios, supports_sparse=False)


def run_jepa_method(
    variant: str,
    resolution: int,
    mask_ratio: float,
    train: WrenchSplit,
    val: WrenchSplit,
    test: WrenchSplit | None,
    run_config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    sparse_ratios: list[float],
) -> None:
    flavor = {
        "wrench_jepa_vit_unet": "jepa",
        "wrench_lejepa_vit_unet": "lejepa",
        "wrench_visreg_vit_unet": "visreg",
    }[variant]
    reg_type = {"jepa": "none", "lejepa": "sigreg", "visreg": "visreg"}[flavor]
    ratio_tag = f"m{int(round(mask_ratio * 100)):02d}"
    method = f"r{resolution}_{variant}_{ratio_tag}"
    pretrain_dir = args.out_dir / f"pretrain_r{resolution}_{flavor}_{ratio_tag}"
    pretrainer = make_jepa_pretrainer(train.wrench_x, args, reg_type=reg_type).to(device)
    pretrain_summary = train_jepa_pretrainer(
        pretrainer,
        train.wrench_x,
        val.wrench_x,
        pretrain_dir,
        mask_ratio=mask_ratio,
        args=args,
        device=device,
        config={
            **run_config,
            "pretrain_flavor": flavor,
            "pretrain_reg_type": reg_type,
            "mask_ratio": float(mask_ratio),
            "resolution": int(resolution),
        },
    )
    model = make_vit_model(train.wrench_x, resolution, args)
    model.backbone.load_state_dict(pretrainer.online.state_dict())
    context = method_context(run_config, variant, method, resolution, train, val, test, input_name="probe_wrench", model_name=f"{flavor}_vit_unet")
    context.update(
        {
            "pretrain_dir": str(pretrain_dir),
            "pretrain_flavor": flavor,
            "pretrain_reg_type": reg_type,
            "mask_ratio": float(mask_ratio),
            "pretrain_best_val_loss": pretrain_summary.get("best_val_loss"),
            "pretrain_epochs_completed": pretrain_summary.get("epochs_completed"),
        }
    )
    train_segmentation_method(method, model, train.wrench_x, train.masks, val.wrench_x, val.masks, val.names, test.wrench_x if test else None, test.masks if test else None, test.names if test else None, args.out_dir, context, args, device)
    jepa_metrics = evaluate_jepa_world_metrics(pretrainer, test.wrench_x if test else val.wrench_x, mask_ratio=mask_ratio, device=device, args=args)
    write_method_world_metrics(args.out_dir / method, method, jepa_metrics)
    write_sparse_segmentation_metrics(args.out_dir / method, method, model, test, device, args, sparse_ratios, supports_sparse=False)


def run_world_method(
    variant: str,
    resolution: int,
    train: WrenchSplit,
    val: WrenchSplit,
    test: WrenchSplit | None,
    run_config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    sparse_ratios: list[float],
) -> None:
    method = f"r{resolution}_{variant}"
    curve_aux = variant in {"wrench_world_with_curve_decoder", "wrench_world_with_mask_diffusion_aux"}
    mask_aux = variant == "wrench_world_with_mask_diffusion_aux"
    if variant == "wrench_temporal_cnn_sparse_unet":
        curve_aux = False
        mask_aux = False
        model = make_temporal_cnn_world_model(train.world_x, train.wrench_x.shape[1], resolution, args, curve_aux=False)
        pretrain_dir = None
        pretrain_summary = {"best_val_loss": "", "epochs_completed": 0}
    elif variant in {"wrench_temporal_cnn_world_unet", "wrench_temporal_cnn_world_curve_unet"}:
        curve_aux = variant == "wrench_temporal_cnn_world_curve_unet"
        mask_aux = False
        model = make_temporal_cnn_world_model(train.world_x, train.wrench_x.shape[1], resolution, args, curve_aux=curve_aux)
        pretrain_dir = args.out_dir / f"pretrain_r{resolution}_{variant}"
        pretrain_summary = train_action_world_pretrainer(
            model,
            train.world_x,
            val.world_x,
            pretrain_dir,
            args=args,
            device=device,
            config={**run_config, "variant": variant, "resolution": int(resolution)},
        )
    else:
        model = make_world_model(train.world_x, train.wrench_x.shape[1], resolution, args, curve_aux=curve_aux, mask_aux=mask_aux)
        pretrain_dir = args.out_dir / f"pretrain_r{resolution}_{variant}"
        pretrain_summary = train_action_world_pretrainer(
            model,
            train.world_x,
            val.world_x,
            pretrain_dir,
            args=args,
            device=device,
            config={**run_config, "variant": variant, "resolution": int(resolution)},
        )
    model_name = "temporal_cnn_action_world_unet" if isinstance(model, TemporalCNNWorldUNet) else "action_world_perceiver"
    context = method_context(run_config, variant, method, resolution, train, val, test, input_name="probe_wrench_plus_action", model_name=model_name)
    context.update(
        {
            "pretrain_dir": str(pretrain_dir) if pretrain_dir is not None else "",
            "pretrain_best_val_curve_huber": pretrain_summary.get("best_val_loss"),
            "pretrain_epochs_completed": pretrain_summary.get("epochs_completed"),
            "curve_aux_weight": float(args.curve_aux_weight if curve_aux else 0.0),
            "mask_aux_weight": float(args.mask_aux_weight if mask_aux else 0.0),
        }
    )
    aux_weight = float(args.curve_aux_weight if curve_aux else 0.0)
    mask_aux_weight = float(args.mask_aux_weight if mask_aux else 0.0)
    train_segmentation_method(
        method,
        model,
        train.world_x,
        train.masks,
        val.world_x,
        val.masks,
        val.names,
        test.world_x if test else None,
        test.masks if test else None,
        test.names if test else None,
        args.out_dir,
        context,
        args,
        device,
        aux_weight=aux_weight,
        mask_aux_weight=mask_aux_weight,
    )
    world_metrics = evaluate_action_world_metrics(model, test if test is not None else val, device=device, args=args)
    write_method_world_metrics(args.out_dir / method, method, world_metrics)
    write_sparse_segmentation_metrics(args.out_dir / method, method, model, test, device, args, sparse_ratios, supports_sparse=True)


def make_vit_model(x: np.ndarray, resolution: int, args: argparse.Namespace) -> WrenchViTUNet:
    return WrenchViTUNet(
        curve_channels=int(x.shape[1]),
        output_shape=(resolution, resolution),
        dim=int(args.dim),
        temporal_depth=int(args.temporal_depth),
        spatial_depth=int(args.spatial_depth),
        heads=int(args.heads),
        decoder_base_channels=int(args.decoder_base_channels),
        grid_hw=(int(x.shape[-2]), int(x.shape[-1])),
    )


def make_jepa_pretrainer(x: np.ndarray, args: argparse.Namespace, *, reg_type: str) -> WrenchJEPAPretrainer:
    return WrenchJEPAPretrainer(
        curve_channels=int(x.shape[1]),
        dim=int(args.dim),
        temporal_depth=int(args.temporal_depth),
        spatial_depth=int(args.spatial_depth),
        heads=int(args.heads),
        grid_hw=(int(x.shape[-2]), int(x.shape[-1])),
        ema=float(args.jepa_ema),
        reg_type=reg_type,
        reg_weight=float(args.lejepa_reg_weight),
        visreg_projections=int(args.visreg_projections),
    )


def make_world_model(
    x: np.ndarray,
    curve_channels: int,
    resolution: int,
    args: argparse.Namespace,
    *,
    curve_aux: bool,
    mask_aux: bool,
) -> ActionWorldPerceiver:
    curve_channels = int(curve_channels)
    action_channels = int(x.shape[1]) - curve_channels
    return ActionWorldPerceiver(
        curve_channels=curve_channels,
        action_channels=action_channels,
        output_shape=(resolution, resolution),
        dim=int(args.dim),
        heads=int(args.heads),
        depth=int(args.perceiver_depth),
        latent_slots=int(args.perceiver_latents),
        decoder_base_channels=int(args.decoder_base_channels),
        grid_hw=(int(x.shape[-2]), int(x.shape[-1])),
        target_ratio=float(args.world_target_ratio),
        curve_aux=curve_aux,
        mask_diffusion_aux=mask_aux,
    )


def make_temporal_cnn_world_model(
    x: np.ndarray,
    curve_channels: int,
    resolution: int,
    args: argparse.Namespace,
    *,
    curve_aux: bool,
) -> TemporalCNNWorldUNet:
    curve_channels = int(curve_channels)
    action_channels = int(x.shape[1]) - curve_channels
    embed = min(int(args.dim), 96)
    hidden = max(64, embed)
    return TemporalCNNWorldUNet(
        curve_channels=curve_channels,
        action_channels=action_channels,
        output_shape=(resolution, resolution),
        embed_channels=embed,
        hidden_channels=hidden,
        decoder_base_channels=int(args.decoder_base_channels),
        grid_hw=(int(x.shape[-2]), int(x.shape[-1])),
        target_ratio=float(args.world_target_ratio),
        curve_aux=curve_aux,
    )


def train_segmentation_method(
    method: str,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_names: list[str],
    x_test: np.ndarray | None,
    y_test: np.ndarray | None,
    test_names: list[str] | None,
    out_dir: Path,
    context: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    *,
    aux_weight: float = 0.0,
    mask_aux_weight: float = 0.0,
) -> None:
    method_dir = out_dir / method
    if _method_done(method_dir, args.force):
        print(f"{method}: existing metrics found, loading checkpoint")
        checkpoint = torch.load(method_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        method_dir.mkdir(parents=True, exist_ok=True)
        _write_json(method_dir / "config.json", context)
        model = model.to(device)
        train_loader = DataLoader(SegmentationTensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(SegmentationTensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
        pos = max(float(y_train.sum()), 1.0)
        neg = max(float(y_train.size - y_train.sum()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp(args, device))
        best_dice = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without = 0
        rows: list[dict[str, Any]] = []
        start = time.perf_counter()
        for epoch in range(1, int(args.epochs) + 1):
            train_stats = run_segmentation_epoch(
                model,
                train_loader,
                optimizer,
                device,
                pos_weight=pos_weight,
                scaler=scaler,
                args=args,
                aux_weight=aux_weight,
                mask_aux_weight=mask_aux_weight,
            )
            val_stats = run_segmentation_epoch(
                model,
                val_loader,
                None,
                device,
                pos_weight=pos_weight,
                scaler=None,
                args=args,
                aux_weight=0.0,
                mask_aux_weight=0.0,
            )
            elapsed = time.perf_counter() - start
            row = {"epoch": epoch, **prefix_keys(train_stats, "train_"), **prefix_keys(val_stats, "val_"), "elapsed_seconds": elapsed}
            rows.append(row)
            _write_csv(method_dir / "history.csv", rows, list(row.keys()))
            if val_stats["dice"] > best_dice + 1e-4:
                best_dice = val_stats["dice"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save(
                    {
                        "model_state": best_state,
                        "method": method,
                        "group": "wrench_world_model",
                        "context": context,
                        "input_channels": int(x_train.shape[1]),
                    },
                    method_dir / "best.pt",
                )
                epochs_without = 0
            else:
                epochs_without += 1
            print(
                f"{method} epoch {epoch:03d} train_dice={train_stats['dice']:.4f} "
                f"val_dice={val_stats['dice']:.4f} loss={val_stats['loss']:.4f} elapsed_min={elapsed / 60.0:.2f}"
            )
            if epochs_without >= int(args.patience):
                break
        if best_state is not None:
            model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
        context = {**context, "epochs_requested": int(args.epochs), "epochs_completed": len(rows), "best_fixed_val_dice": best_dice}
        val_scores = predict_scores(model, x_val, device=device, batch_size=args.batch_size)
        write_score_method(out_dir, method, "wrench_world_model", val_scores, y_val, val_names, config=context, fixed_threshold=0.5, force=True)

    if x_test is not None and y_test is not None and test_names is not None:
        model = model.to(device)
        test_scores = predict_scores(model, x_test, device=device, batch_size=args.batch_size)
        val_summary = _read_json(method_dir / "metrics_summary.json")
        val_best_threshold = float(val_summary.get("threshold_sweep_best", {}).get("threshold", 0.5))
        write_test_metrics(
            method_dir,
            test_scores,
            y_test,
            test_names,
            fixed_threshold=0.5,
            val_best_threshold=val_best_threshold,
            context=context,
            force=True,
        )


def run_segmentation_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    pos_weight: torch.Tensor,
    scaler: torch.cuda.amp.GradScaler | None,
    args: argparse.Namespace,
    aux_weight: float,
    mask_aux_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "seg_loss": 0.0, "aux_loss": 0.0, "dice": 0.0, "iou": 0.0}
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=use_amp(args, device)):
            forward_kwargs: dict[str, Any] = {}
            if training and supports_context_mask(model):
                ratio = random.uniform(float(args.seg_context_min), float(args.seg_context_max))
                b, _c, h, w = x.shape
                forward_kwargs["context_keep_mask"] = random_context_keep_mask(b, h * w, ratio, x.device)
            logits = model(x, **forward_kwargs) if forward_kwargs else model(x)
            seg_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight) + dice_loss(logits, y)
            aux_loss = torch.zeros((), device=device)
            if (aux_weight > 0.0 or mask_aux_weight > 0.0) and hasattr(model, "auxiliary_loss"):
                raw_aux, _aux_stats = model.auxiliary_loss(x, y)
                aux_loss = raw_aux * float(aux_weight + mask_aux_weight)
            loss = seg_loss + aux_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        metrics = torch_segmentation_metrics(logits.detach(), y)
        batch = int(x.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["seg_loss"] += float(seg_loss.detach().cpu()) * batch
        totals["aux_loss"] += float(aux_loss.detach().cpu()) * batch
        totals["dice"] += metrics["dice"] * batch
        totals["iou"] += metrics["iou"] * batch
        count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_jepa_pretrainer(
    model: WrenchJEPAPretrainer,
    x_train: np.ndarray,
    x_val: np.ndarray,
    out_dir: Path,
    *,
    mask_ratio: float,
    args: argparse.Namespace,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    summary_path = out_dir / "pretrain_summary.json"
    if summary_path.exists() and not args.force:
        checkpoint = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"loaded JEPA pretrain checkpoint: {out_dir / 'best.pt'}")
        return _read_json(summary_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", config)
    train_loader = DataLoader(TensorOnlyDataset(x_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorOnlyDataset(x_val), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.pretrain_lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp(args, device))
    best_val = float("inf")
    epochs_without = 0
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    model = model.to(device)
    for epoch in range(1, int(args.pretrain_epochs) + 1):
        train_stats = run_jepa_epoch(model, train_loader, optimizer, device, mask_ratio=mask_ratio, scaler=scaler, args=args)
        val_stats = run_jepa_epoch(model, val_loader, None, device, mask_ratio=mask_ratio, scaler=None, args=args)
        elapsed = time.perf_counter() - start
        row = {"epoch": epoch, **prefix_keys(train_stats, "train_"), **prefix_keys(val_stats, "val_"), "elapsed_seconds": elapsed}
        rows.append(row)
        _write_csv(out_dir / "history.csv", rows, list(row.keys()))
        if val_stats["loss"] < best_val - 1e-5:
            best_val = val_stats["loss"]
            torch.save({"model_state": model.state_dict(), "config": config}, out_dir / "best.pt")
            epochs_without = 0
        else:
            epochs_without += 1
        print(f"pretrain {out_dir.name} epoch {epoch:03d} train_loss={train_stats['loss']:.5f} val_loss={val_stats['loss']:.5f}")
        if epochs_without >= int(args.pretrain_patience):
            break
    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    summary = {
        "best_val_loss": best_val,
        "epochs_requested": int(args.pretrain_epochs),
        "epochs_completed": len(rows),
        "mask_ratio": float(mask_ratio),
        "config": config,
    }
    _write_json(summary_path, summary)
    return summary


def run_jepa_epoch(
    model: WrenchJEPAPretrainer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    mask_ratio: float,
    scaler: torch.cuda.amp.GradScaler | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "latent_mse": 0.0, "sigreg": 0.0}
    count = 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=use_amp(args, device)):
            loss, stats = model.forward_loss(x, mask_ratio=mask_ratio)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            model.update_target()
        batch = int(x.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["latent_mse"] += float(stats["latent_mse"]) * batch
        totals["sigreg"] += float(stats["sigreg"]) * batch
        count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_action_world_pretrainer(
    model: ActionWorldPerceiver,
    x_train: np.ndarray,
    x_val: np.ndarray,
    out_dir: Path,
    *,
    args: argparse.Namespace,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    summary_path = out_dir / "pretrain_summary.json"
    if summary_path.exists() and not args.force:
        checkpoint = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"loaded action-world pretrain checkpoint: {out_dir / 'best.pt'}")
        return _read_json(summary_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", config)
    train_loader = DataLoader(TensorOnlyDataset(x_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorOnlyDataset(x_val), batch_size=args.batch_size, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.pretrain_lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp(args, device))
    best_val = float("inf")
    epochs_without = 0
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, int(args.pretrain_epochs) + 1):
        train_stats = run_action_world_epoch(model, train_loader, optimizer, device, scaler=scaler, args=args)
        val_stats = run_action_world_epoch(model, val_loader, None, device, scaler=None, args=args)
        elapsed = time.perf_counter() - start
        row = {"epoch": epoch, **prefix_keys(train_stats, "train_"), **prefix_keys(val_stats, "val_"), "elapsed_seconds": elapsed}
        rows.append(row)
        _write_csv(out_dir / "history.csv", rows, list(row.keys()))
        if val_stats["loss"] < best_val - 1e-5:
            best_val = val_stats["loss"]
            torch.save({"model_state": model.state_dict(), "config": config}, out_dir / "best.pt")
            epochs_without = 0
        else:
            epochs_without += 1
        print(f"pretrain {out_dir.name} epoch {epoch:03d} train_huber={train_stats['loss']:.5f} val_huber={val_stats['loss']:.5f}")
        if epochs_without >= int(args.pretrain_patience):
            break
    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    summary = {"best_val_loss": best_val, "epochs_requested": int(args.pretrain_epochs), "epochs_completed": len(rows), "config": config}
    _write_json(summary_path, summary)
    return summary


def run_action_world_epoch(
    model: ActionWorldPerceiver,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    scaler: torch.cuda.amp.GradScaler | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0}
    count = 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=use_amp(args, device)):
            curve_tokens, _action_tokens = model.split_input(x)
            b, n, _ = curve_tokens.shape
            target_mask = torch.rand((b, n), device=device) < float(args.world_target_ratio)
            target_mask = ensure_any_masked_and_context(target_mask)
            pred = model.predict_curves(x, context_keep_mask=~target_mask)
            loss = F.smooth_l1_loss(pred[target_mask], curve_tokens[target_mask])
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        batch = int(x.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch
        count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    context_ratio: float | None = None,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        kwargs: dict[str, Any] = {}
        if context_ratio is not None and supports_context_mask(model):
            b, _c, h, w = batch.shape
            kwargs["context_keep_mask"] = random_context_keep_mask(b, h * w, context_ratio, batch.device)
        logits = model(batch, **kwargs) if kwargs else model(batch)
        probs.append(torch.sigmoid(logits)[:, 0].cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


@torch.no_grad()
def evaluate_jepa_world_metrics(
    model: WrenchJEPAPretrainer,
    x: np.ndarray,
    *,
    mask_ratio: float,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    model.eval()
    loader = DataLoader(TensorOnlyDataset(x), batch_size=args.batch_size, shuffle=False)
    total_loss = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device)
        loss, _stats = model.forward_loss(batch, mask_ratio=mask_ratio)
        total_loss += float(loss.detach().cpu()) * int(batch.shape[0])
        count += int(batch.shape[0])
    return [
        {
            "split": "test_or_val",
            "supports_world_prediction": True,
            "world_prediction_type": "masked_latent_jepa",
            "mask_ratio": float(mask_ratio),
            "heldout_latent_mse": total_loss / max(count, 1),
            "heldout_curve_huber": "",
            "heldout_curve_mse": "",
            **{f"{name}_mse": "" for name in WRENCH_NAMES},
            "sparse_context_ratio": "",
            "sparse_context_fixed_dice": "",
        }
    ]


@torch.no_grad()
def evaluate_action_world_metrics(
    model: ActionWorldPerceiver,
    split: WrenchSplit,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    model.eval()
    loader = DataLoader(torch.from_numpy(split.world_x.astype(np.float32)), batch_size=args.batch_size, shuffle=False)
    sums = {"huber": 0.0, "mse": 0.0}
    channel_sums = np.zeros(len(WRENCH_NAMES), dtype=np.float64)
    count = 0
    channel_count = 0
    for batch in loader:
        batch = batch.to(device)
        curve_tokens, _action_tokens = model.split_input(batch)
        b, n, c = curve_tokens.shape
        target_mask = torch.rand((b, n), device=device) < float(args.world_target_ratio)
        target_mask = ensure_any_masked_and_context(target_mask)
        pred = model.predict_curves(batch, context_keep_mask=~target_mask)
        diff = pred[target_mask] - curve_tokens[target_mask]
        huber = F.smooth_l1_loss(pred[target_mask], curve_tokens[target_mask], reduction="sum")
        mse = torch.sum(diff.square())
        items = int(diff.numel())
        sums["huber"] += float(huber.cpu())
        sums["mse"] += float(mse.cpu())
        steps = int(model.curve_channels) // len(WRENCH_NAMES)
        curve_diff = diff.reshape(-1, steps, len(WRENCH_NAMES))
        channel_sums += curve_diff.square().sum(dim=(0, 1)).cpu().numpy()
        channel_count += int(curve_diff.shape[0] * curve_diff.shape[1])
        count += items
    denom = max(count, 1)
    channel_denom = max(channel_count, 1)
    return [
        {
            "split": "test_or_val",
            "supports_world_prediction": True,
            "world_prediction_type": "action_conditioned_curve",
            "mask_ratio": float(args.world_target_ratio),
            "heldout_latent_mse": "",
            "heldout_curve_huber": sums["huber"] / denom,
            "heldout_curve_mse": sums["mse"] / denom,
            **{f"{name}_mse": float(channel_sums[idx] / channel_denom) for idx, name in enumerate(WRENCH_NAMES)},
            "sparse_context_ratio": "",
            "sparse_context_fixed_dice": "",
        }
    ]


def write_sparse_segmentation_metrics(
    method_dir: Path,
    method: str,
    model: nn.Module,
    test: WrenchSplit | None,
    device: torch.device,
    args: argparse.Namespace,
    sparse_ratios: list[float],
    *,
    supports_sparse: bool,
) -> None:
    if test is None:
        return
    rows = read_method_world_metrics(method_dir, method)
    for ratio in sparse_ratios:
        row = {
            "split": "test",
            "supports_world_prediction": supports_sparse,
            "world_prediction_type": "sparse_context_segmentation" if supports_sparse else "not_supported",
            "mask_ratio": "",
            "heldout_latent_mse": "",
            "heldout_curve_huber": "",
            "heldout_curve_mse": "",
            **{f"{name}_mse": "" for name in WRENCH_NAMES},
            "sparse_context_ratio": float(ratio),
            "sparse_context_fixed_dice": "",
        }
        if supports_sparse:
            x_eval = test.world_x if supports_context_mask(model) else test.wrench_x
            scores = predict_scores(model, x_eval, device=device, batch_size=args.batch_size, context_ratio=ratio)
            fixed = metrics_from_counts(counts_from_prediction(scores >= 0.5, test.masks), threshold=0.5)
            row["sparse_context_fixed_dice"] = fixed["dice"]
        rows.append(row)
    write_method_world_metrics(method_dir, method, rows)


def write_no_world_metrics(method_dir: Path, method: str, *, reason: str) -> None:
    rows = [
        {
            "split": "test_or_val",
            "supports_world_prediction": False,
            "world_prediction_type": reason,
            "mask_ratio": "",
            "heldout_latent_mse": "",
            "heldout_curve_huber": "",
            "heldout_curve_mse": "",
            **{f"{name}_mse": "" for name in WRENCH_NAMES},
            "sparse_context_ratio": "",
            "sparse_context_fixed_dice": "",
        }
    ]
    write_method_world_metrics(method_dir, method, rows)


def write_method_world_metrics(method_dir: Path, method: str, rows: list[dict[str, Any]]) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    ready_rows = [{"method": method, **row} for row in rows]
    fieldnames = world_metric_fieldnames()
    _write_csv(method_dir / "world_metrics.csv", ready_rows, fieldnames)
    _write_json(method_dir / "world_metrics.json", ready_rows)


def read_method_world_metrics(method_dir: Path, method: str) -> list[dict[str, Any]]:
    path = method_dir / "world_metrics.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [{key: value for key, value in row.items() if key != "method"} for row in rows]


def write_leaderboards(out_dir: Path, *, baseline_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    if baseline_path.exists():
        baseline_rows = read_csv_dicts(baseline_path)
        for row in baseline_rows[:3]:
            rows.append(
                {
                    "source": "baseline",
                    "method": row.get("method", ""),
                    "variant": row.get("variant", ""),
                    "resolution": row.get("resolution", ""),
                    "input": row.get("input", ""),
                    "model": row.get("model", ""),
                    "val_best_dice": row.get("val_best_dice", ""),
                    "val_best_threshold": row.get("val_best_threshold", ""),
                    "test_fixed_dice": row.get("test_fixed_dice", ""),
                    "test_val_selected_dice": row.get("test_val_selected_dice", ""),
                    "test_val_selected_threshold": row.get("test_val_selected_threshold", ""),
                    "test_oracle_dice": row.get("test_oracle_dice", ""),
                    "heldout_latent_mse": "",
                    "heldout_curve_mse": "",
                    "path": row.get("path", ""),
                }
            )
    for metrics_path in sorted(out_dir.glob("r*/metrics_summary.json")):
        summary = _read_json(metrics_path)
        config = summary.get("config", {})
        test_summary = _read_json(metrics_path.parent / "test_metrics_summary.json")
        fixed = summary.get("fixed_threshold", {})
        best = summary.get("threshold_sweep_best", {})
        test_fixed = test_summary.get("fixed_threshold", {})
        test_val = test_summary.get("val_selected_threshold", {})
        test_oracle = test_summary.get("test_oracle_threshold", {})
        world_rows = _read_json(metrics_path.parent / "world_metrics.json")
        first_world = world_rows[0] if world_rows else {}
        rows.append(
            {
                "source": "world_sweep",
                "method": summary.get("method", metrics_path.parent.name),
                "variant": config.get("variant", ""),
                "resolution": config.get("resolution", ""),
                "input": config.get("input", ""),
                "model": config.get("model", ""),
                "val_best_dice": best.get("dice", ""),
                "val_best_threshold": best.get("threshold", ""),
                "test_fixed_dice": test_fixed.get("dice", ""),
                "test_val_selected_dice": test_val.get("dice", ""),
                "test_val_selected_threshold": test_val.get("threshold", ""),
                "test_oracle_dice": test_oracle.get("dice", ""),
                "heldout_latent_mse": first_world.get("heldout_latent_mse", ""),
                "heldout_curve_mse": first_world.get("heldout_curve_mse", ""),
                "path": str(metrics_path.parent),
            }
        )
    rows.sort(key=lambda row: safe_float(row.get("test_val_selected_dice")), reverse=True)
    if rows:
        _write_csv(out_dir / "leaderboard.csv", rows, list(rows[0].keys()))

    world_rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("r*/world_metrics.json")):
        world_rows.extend(_read_json(path))
    if world_rows:
        _write_csv(out_dir / "world_model_metrics.csv", world_rows, world_metric_fieldnames())


def method_context(
    run_config: dict[str, Any],
    variant: str,
    method: str,
    resolution: int,
    train: WrenchSplit,
    val: WrenchSplit,
    test: WrenchSplit | None,
    *,
    input_name: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        **run_config,
        "method": method,
        "variant": variant,
        "resolution": int(resolution),
        "input": input_name,
        "model": model_name,
        "train_samples": len(train.names),
        "val_samples": len(val.names),
        "test_samples": len(test.names) if test is not None else 0,
        "wrench_input_shape_chw": list(train.wrench_x.shape[1:]),
        "world_input_shape_chw": list(train.world_x.shape[1:]),
        "target_shape_hw": [int(resolution), int(resolution)],
    }


def torch_segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum((preds + targets) > 0, dim=dims).float()
    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)
    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    iou = ((intersection + eps) / (union + eps)).mean()
    return {"dice": float(dice.cpu()), "iou": float(iou.cpu())}


def sigreg_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    mean_loss = flat.mean(dim=0).square().mean()
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + eps)
    std_loss = (std - 1.0).square().mean()
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = flat.T @ flat / max(flat.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.square().mean()
    return mean_loss + std_loss + cov_loss


def visreg_loss(z: torch.Tensor, *, num_projections: int = 64, eps: float = 1e-4) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    mean = flat.mean(dim=0, keepdim=True)
    std = torch.sqrt(flat.var(dim=0, unbiased=False, keepdim=True) + eps)
    mean_loss = mean.square().mean()
    var_loss = F.relu(1.0 - std).square().mean()
    normalized = (flat - mean) / std.clamp_min(eps)
    projections = max(1, int(num_projections))
    directions = torch.randn((normalized.shape[-1], projections), dtype=normalized.dtype, device=normalized.device)
    directions = F.normalize(directions, dim=0)
    projected = normalized @ directions
    target = torch.randn_like(projected)
    sw_loss = (torch.sort(projected, dim=0).values - torch.sort(target, dim=0).values).square().mean()
    return mean_loss + var_loss + sw_loss


def ensure_any_masked_and_context(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected mask [B,N], got {tuple(mask.shape)}")
    out = mask.clone()
    n = out.shape[1]
    all_false = ~out.any(dim=1)
    if all_false.any():
        out[all_false, 0] = True
    all_true = out.all(dim=1)
    if all_true.any() and n > 1:
        out[all_true, -1] = False
    return out


def random_context_keep_mask(batch: int, tokens: int, ratio: float, device: torch.device) -> torch.Tensor:
    keep = torch.rand((batch, tokens), device=device) < float(ratio)
    return ~ensure_any_masked_and_context(~keep)


def resize_logits(logits: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == output_shape:
        return logits
    return F.interpolate(logits, size=output_shape, mode="bilinear", align_corners=False)


def prefix_keys(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in ALLOWED_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; allowed: {ALLOWED_VARIANTS}")
    return variants


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def resolve_batch_size(requested: int, *, smoke: bool) -> int:
    if requested > 0:
        return int(requested)
    if smoke:
        return 4
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gb >= 30:
            return 24
    return 16


def use_amp(args: argparse.Namespace, device: torch.device) -> bool:
    return bool(not args.no_amp and device.type == "cuda")


def supports_context_mask(model: nn.Module) -> bool:
    return isinstance(model, (ActionWorldPerceiver, TemporalCNNWorldUNet))


def safe_float(value: Any) -> float:
    try:
        if value == "":
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def world_metric_fieldnames() -> list[str]:
    return [
        "method",
        "split",
        "supports_world_prediction",
        "world_prediction_type",
        "mask_ratio",
        "heldout_latent_mse",
        "heldout_curve_huber",
        "heldout_curve_mse",
        *[f"{name}_mse" for name in WRENCH_NAMES],
        "sparse_context_ratio",
        "sparse_context_fixed_dice",
    ]


if __name__ == "__main__":
    main()
