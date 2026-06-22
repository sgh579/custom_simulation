from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from palpation_sim.models import ValidationUNet
from palpation_sim.workflow import require_runtime_environment
from run_highres_segmentation_sweep import (
    DEFAULT_PACKAGE_DIR,
    FEATURE_NAMES,
    SplitData,
    UpsampleLogits,
    displacement_position_scale,
    limited_trajectory_sincos_inputs,
    load_split,
    normalize_temporal_feature_fz_channels,
    normalize_train_val_test,
    write_test_metrics,
)
from run_segmentation_accuracy_sweep import (
    _method_done,
    _seed_everything,
    normalize_maps,
    predict_neural,
    train_neural_method,
)


DEFAULT_OUT_DIR = Path("runs/highres_fz_temporal_variant_sweep_800_80_80")
BASELINE_SWEEP_DIR = Path("runs/highres_segmentation_sweep_800_80_80")
DEFAULT_RESOLUTIONS = (64,)
DEFAULT_VARIANTS = (
    "fz_stats_unet",
    "fz_stats_stiffness_unet",
    "fz_delta_unet",
    "fz_temporal_cnn16_unet",
    "fz_temporal_cnn32_unet",
    "fz_temporal_cnn32_stiffness_unet",
    "fz_unet_aug_focal",
    "fz_features_unet",
    "fz_features_aug_focal_unet",
    "fz_features_wide_unet",
    "fz_stats_features_unet",
    "fz_residual_se_unet",
    "fz_temporal_gru32_unet",
    "fz_temporal_attention32_unet",
    "fz_temporal_multiscale32_unet",
    "fz_temporal_multiscale32_features_unet",
    "limited_sincos_unet",
    "limited_sincos_temporal_cnn16_unet",
    "limited_sincos_temporal_cnn32_unet",
    "limited_sincos_temporal_gru32_unet",
    "limited_sincos_temporal_attention32_unet",
    "limited_sincos_temporal_multiscale32_unet",
)
TEMPORAL_STAT_NAMES = [
    "fz_initial",
    "fz_final",
    "fz_max",
    "fz_mean",
    "fz_std",
    "fz_integral",
    "fz_t25",
    "fz_t50",
    "fz_t75",
    "fz_slope_0_25",
    "fz_slope_25_50",
    "fz_slope_50_75",
    "fz_slope_75_100",
    "fz_max_delta",
    "fz_mean_delta",
    "fz_argmax_time",
    "fz_contact_time",
    "fz_contact_to_peak",
]


@dataclass(frozen=True)
class VariantBatch:
    method: str
    input_name: str
    model_name: str
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    model: nn.Module
    context_extra: dict[str, Any]
    augment: bool = False
    focal: bool = False


def _curve_to_conv_sequence(x: torch.Tensor, curve_channels: int, temporal_feature_size: int) -> tuple[torch.Tensor, int, int, int, int]:
    b, c, h, w = x.shape
    feature_size = int(temporal_feature_size)
    if int(curve_channels) % feature_size != 0:
        raise ValueError(f"curve_channels={curve_channels} must be divisible by temporal_feature_size={feature_size}")
    steps = int(curve_channels) // feature_size
    curve = x[:, :curve_channels].reshape(b, steps, feature_size, h, w)
    seq = curve.permute(0, 3, 4, 2, 1).reshape(b * h * w, feature_size, steps)
    return seq, b, h, w, steps


def _curve_to_rnn_sequence(x: torch.Tensor, curve_channels: int, temporal_feature_size: int) -> tuple[torch.Tensor, int, int, int, int]:
    b, c, h, w = x.shape
    feature_size = int(temporal_feature_size)
    if int(curve_channels) % feature_size != 0:
        raise ValueError(f"curve_channels={curve_channels} must be divisible by temporal_feature_size={feature_size}")
    steps = int(curve_channels) // feature_size
    curve = x[:, :curve_channels].reshape(b, steps, feature_size, h, w)
    seq = curve.permute(0, 3, 4, 1, 2).reshape(b * h * w, steps, feature_size)
    return seq, b, h, w, steps


class SqueezeExcite2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, int(channels) // int(reduction))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(channels), hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, int(channels), kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm2d(out_channels),
            SqueezeExcite2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class ResidualSEUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        b = int(base_channels)
        self.inc = ResidualSEBlock(in_channels, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), ResidualSEBlock(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), ResidualSEBlock(b * 2, b * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), ResidualSEBlock(b * 4, b * 8))
        self.up3 = ResidualSEBlock(b * 8 + b * 4, b * 4)
        self.up2 = ResidualSEBlock(b * 4 + b * 2, b * 2)
        self.up1 = ResidualSEBlock(b * 2 + b, b)
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x3, x], dim=1))
        x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x2, x], dim=1))
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x1, x], dim=1))
        return self.outc(x)


class PointwiseTemporalConvUNet(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        extra_channels: int,
        embed_channels: int,
        output_shape: tuple[int, int],
        temporal_feature_size: int = 1,
        spatial_base_channels: int = 24,
    ) -> None:
        super().__init__()
        self.curve_channels = int(curve_channels)
        self.extra_channels = int(extra_channels)
        self.temporal_feature_size = int(temporal_feature_size)
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        hidden = max(16, int(embed_channels))
        self.temporal = nn.Sequential(
            nn.Conv1d(self.temporal_feature_size, hidden, kernel_size=7, padding=3, bias=False),
            _norm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=False),
            _norm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, int(embed_channels), kernel_size=3, padding=1, bias=False),
            _norm1d(int(embed_channels)),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.spatial = ValidationUNet(int(embed_channels) + self.extra_channels, base_channels=spatial_base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, b, h, w, _steps = _curve_to_conv_sequence(x, self.curve_channels, self.temporal_feature_size)
        encoded = self.temporal(seq).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        if self.extra_channels:
            encoded = torch.cat([encoded, x[:, self.curve_channels : self.curve_channels + self.extra_channels]], dim=1)
        logits = self.spatial(encoded)
        if logits.shape[-2:] != self.output_shape:
            logits = F.interpolate(logits, size=self.output_shape, mode="bilinear", align_corners=False)
        return logits


class PointwiseGRUUNet(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        extra_channels: int,
        embed_channels: int,
        output_shape: tuple[int, int],
        temporal_feature_size: int = 1,
        spatial_base_channels: int = 24,
    ) -> None:
        super().__init__()
        self.curve_channels = int(curve_channels)
        self.extra_channels = int(extra_channels)
        self.temporal_feature_size = int(temporal_feature_size)
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.gru = nn.GRU(input_size=self.temporal_feature_size, hidden_size=int(embed_channels), num_layers=1, batch_first=True)
        self.norm = _norm2d(int(embed_channels))
        self.spatial = ValidationUNet(int(embed_channels) + self.extra_channels, base_channels=spatial_base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, b, h, w, _steps = _curve_to_rnn_sequence(x, self.curve_channels, self.temporal_feature_size)
        _out, hidden = self.gru(seq)
        encoded = hidden[-1].reshape(b, h, w, -1).permute(0, 3, 1, 2)
        encoded = self.norm(encoded)
        if self.extra_channels:
            encoded = torch.cat([encoded, x[:, self.curve_channels : self.curve_channels + self.extra_channels]], dim=1)
        logits = self.spatial(encoded)
        if logits.shape[-2:] != self.output_shape:
            logits = F.interpolate(logits, size=self.output_shape, mode="bilinear", align_corners=False)
        return logits


class PointwiseTemporalAttentionUNet(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        extra_channels: int,
        embed_channels: int,
        output_shape: tuple[int, int],
        temporal_feature_size: int = 1,
        spatial_base_channels: int = 24,
    ) -> None:
        super().__init__()
        self.curve_channels = int(curve_channels)
        self.extra_channels = int(extra_channels)
        self.temporal_feature_size = int(temporal_feature_size)
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.temporal = nn.Sequential(
            nn.Conv1d(self.temporal_feature_size, int(embed_channels), kernel_size=7, padding=3, bias=False),
            _norm1d(int(embed_channels)),
            nn.GELU(),
            nn.Conv1d(int(embed_channels), int(embed_channels), kernel_size=5, padding=2, bias=False),
            _norm1d(int(embed_channels)),
            nn.GELU(),
        )
        hidden = max(8, int(embed_channels) // 2)
        self.attention = nn.Sequential(nn.Linear(int(embed_channels), hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.norm = _norm2d(int(embed_channels))
        self.spatial = ValidationUNet(int(embed_channels) + self.extra_channels, base_channels=spatial_base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, b, h, w, _steps = _curve_to_conv_sequence(x, self.curve_channels, self.temporal_feature_size)
        features = self.temporal(seq).transpose(1, 2)
        weights = torch.softmax(self.attention(features), dim=1)
        encoded = torch.sum(weights * features, dim=1).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        encoded = self.norm(encoded)
        if self.extra_channels:
            encoded = torch.cat([encoded, x[:, self.curve_channels : self.curve_channels + self.extra_channels]], dim=1)
        logits = self.spatial(encoded)
        if logits.shape[-2:] != self.output_shape:
            logits = F.interpolate(logits, size=self.output_shape, mode="bilinear", align_corners=False)
        return logits


class PointwiseMultiScaleTemporalConvUNet(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        extra_channels: int,
        embed_channels: int,
        output_shape: tuple[int, int],
        temporal_feature_size: int = 1,
        spatial_base_channels: int = 24,
    ) -> None:
        super().__init__()
        self.curve_channels = int(curve_channels)
        self.extra_channels = int(extra_channels)
        self.temporal_feature_size = int(temporal_feature_size)
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        branch_channels = max(8, int(embed_channels) // 2)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.temporal_feature_size, branch_channels, kernel_size=kernel, padding=kernel // 2, bias=False),
                    _norm1d(branch_channels),
                    nn.GELU(),
                    nn.Conv1d(branch_channels, branch_channels, kernel_size=kernel, padding=kernel // 2, bias=False),
                    _norm1d(branch_channels),
                    nn.GELU(),
                )
                for kernel in (3, 5, 9)
            ]
        )
        self.project = nn.Sequential(
            nn.Linear(3 * 2 * branch_channels, int(embed_channels)),
            nn.LayerNorm(int(embed_channels)),
            nn.GELU(),
        )
        self.norm = _norm2d(int(embed_channels))
        self.spatial = ValidationUNet(int(embed_channels) + self.extra_channels, base_channels=spatial_base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, b, h, w, _steps = _curve_to_conv_sequence(x, self.curve_channels, self.temporal_feature_size)
        pooled = []
        for branch in self.branches:
            features = branch(seq)
            pooled.extend([features.mean(dim=-1), features.amax(dim=-1)])
        encoded = self.project(torch.cat(pooled, dim=1)).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        encoded = self.norm(encoded)
        if self.extra_channels:
            encoded = torch.cat([encoded, x[:, self.curve_channels : self.curve_channels + self.extra_channels]], dim=1)
        logits = self.spatial(encoded)
        if logits.shape[-2:] != self.output_shape:
            logits = F.interpolate(logits, size=self.output_shape, mode="bilinear", align_corners=False)
        return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Try temporal raw-Fz variants for high-resolution mask segmentation.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline-sweep-dir", type=Path, default=BASELINE_SWEEP_DIR)
    parser.add_argument("--resolutions", type=str, default=",".join(str(value) for value in DEFAULT_RESOLUTIONS))
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fz-normalize", choices=["none", "sample", "dataset"], default="dataset")
    parser.add_argument("--feature-normalize", choices=["none", "sample", "dataset"], default="dataset")
    parser.add_argument("--stiffness-normalize", choices=["none", "sample", "dataset"], default="dataset")
    parser.add_argument("--limited-trajectory-seed", type=int, default=None)
    parser.add_argument("--trajectory-input-steps", type=int, default=0, help="0 keeps the native number of trajectory samples.")
    parser.add_argument("--positional-embedding-dim", type=int, default=8, help="Even sinusoidal displacement embedding dimension.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    require_runtime_environment()
    _seed_everything(args.seed)
    random.seed(args.seed)
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
    limited_trajectory_seed = int(args.seed if args.limited_trajectory_seed is None else args.limited_trajectory_seed)

    resolutions = parse_int_list(args.resolutions)
    variants = parse_variant_list(args.variants)
    data_root = args.package_dir / "data"
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "package_dir": str(args.package_dir),
        "baseline_sweep_dir": str(args.baseline_sweep_dir),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "test_dir": str(test_dir),
        "resolutions": resolutions,
        "variants": variants,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "fz_normalize": args.fz_normalize,
        "feature_normalize": args.feature_normalize,
        "stiffness_normalize": args.stiffness_normalize,
        "limited_trajectory_seed": limited_trajectory_seed,
        "trajectory_input_steps": int(args.trajectory_input_steps),
        "positional_embedding_dim": int(args.positional_embedding_dim),
        "limited_trajectory_policy": "Random endpoint per scan point, prefix resampled to fixed input length, with Fz plus sinusoidal displacement embedding.",
        "temporal_stat_names": TEMPORAL_STAT_NAMES,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
    }
    write_json(args.out_dir / "run_config.json", run_config)

    for resolution in resolutions:
        max_train = 8 if args.smoke else None
        max_eval = 4 if args.smoke else None
        print(f"loading r{resolution} train split: {train_dir}")
        train = load_split(train_dir, label_size=resolution, max_samples=max_train)
        print(f"loading r{resolution} val split: {val_dir}")
        val = load_split(val_dir, label_size=resolution, max_samples=max_eval)
        print(f"loading r{resolution} test split: {test_dir}")
        test = load_split(test_dir, label_size=resolution, max_samples=max_eval)

        for variant in variants:
            batch = build_variant(variant, resolution, train, val, test, args)
            context = {
                **run_config,
                "resolution": int(resolution),
                "output_shape": [int(resolution), int(resolution)],
                "input": batch.input_name,
                "model": batch.model_name,
                "variant": variant,
                "method": batch.method,
                "train_samples": len(train.names),
                "val_samples": len(val.names),
                "test_samples": len(test.names),
                "input_shape_chw": list(batch.x_train.shape[1:]),
                "target_shape_hw": [int(resolution), int(resolution)],
                **batch.context_extra,
            }
            run_variant(batch, train, val, test, args.out_dir, context, args)
            write_leaderboard(args.out_dir, baseline_dir=args.baseline_sweep_dir)

    write_leaderboard(args.out_dir, baseline_dir=args.baseline_sweep_dir)
    print(f"temporal variant sweep complete: {args.out_dir}")


def build_variant(
    variant: str,
    resolution: int,
    train: SplitData,
    val: SplitData,
    test: SplitData,
    args: argparse.Namespace,
) -> VariantBatch:
    if variant == "fz_stats_unet":
        x_train, x_val, x_test = normalize_train_val_test(
            fz_temporal_stats(train.fz),
            fz_temporal_stats(val.fz),
            fz_temporal_stats(test.fz),
            mode=args.feature_normalize,
        )
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz_temporal_stats",
            model_name="unet",
            x_train=x_train,
            x_val=x_val,
            x_test=require_array(x_test),
            model=UpsampleLogits(ValidationUNet(x_train.shape[1], base_channels=24), (resolution, resolution)),
            context_extra={"feature_names": TEMPORAL_STAT_NAMES},
        )
    if variant == "fz_stats_stiffness_unet":
        stats_train, stats_val, stats_test = normalize_train_val_test(
            fz_temporal_stats(train.fz),
            fz_temporal_stats(val.fz),
            fz_temporal_stats(test.fz),
            mode=args.feature_normalize,
        )
        stiff_train, stiff_val, stiff_test = normalized_stiffness(train, val, test, mode=args.stiffness_normalize)
        x_train = np.concatenate([stats_train, stiff_train], axis=1)
        x_val = np.concatenate([stats_val, stiff_val], axis=1)
        x_test = np.concatenate([require_array(stats_test), require_array(stiff_test)], axis=1)
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz_temporal_stats_plus_stiffness",
            model_name="unet",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            model=UpsampleLogits(ValidationUNet(x_train.shape[1], base_channels=24), (resolution, resolution)),
            context_extra={"feature_names": TEMPORAL_STAT_NAMES + ["equivalent_stiffness"]},
        )
    if variant == "fz_delta_unet":
        x_train, x_val, x_test = normalize_train_val_test(
            fz_plus_delta(train.fz),
            fz_plus_delta(val.fz),
            fz_plus_delta(test.fz),
            mode=args.fz_normalize,
        )
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz_plus_delta",
            model_name="unet",
            x_train=x_train,
            x_val=x_val,
            x_test=require_array(x_test),
            model=UpsampleLogits(ValidationUNet(x_train.shape[1], base_channels=24), (resolution, resolution)),
            context_extra={"curve_channels": int(train.fz.shape[1]), "delta_channels": int(train.fz.shape[1])},
        )
    if variant == "fz_unet_aug_focal":
        x_train, x_val, x_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz",
            model_name="unet_aug_focal",
            x_train=x_train,
            x_val=x_val,
            x_test=require_array(x_test),
            model=UpsampleLogits(ValidationUNet(x_train.shape[1], base_channels=24), (resolution, resolution)),
            context_extra={"augmentation": "d4", "loss": "focal_bce_dice"},
            augment=True,
            focal=True,
        )
    if variant == "fz_features_unet":
        x_train, x_val, x_test = normalized_fz_features(train, val, test, args)
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz_plus_mechanical_features",
            model_name="unet",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            model=UpsampleLogits(ValidationUNet(x_train.shape[1], base_channels=24), (resolution, resolution)),
            context_extra={"feature_names": FEATURE_NAMES},
        )
    if variant == "fz_features_aug_focal_unet":
        x_train, x_val, x_test = normalized_fz_features(train, val, test, args)
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz_plus_mechanical_features",
            model_name="unet_aug_focal",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            model=UpsampleLogits(ValidationUNet(x_train.shape[1], base_channels=24), (resolution, resolution)),
            context_extra={"feature_names": FEATURE_NAMES, "augmentation": "d4", "loss": "focal_bce_dice"},
            augment=True,
            focal=True,
        )
    if variant == "fz_features_wide_unet":
        x_train, x_val, x_test = normalized_fz_features(train, val, test, args)
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz_plus_mechanical_features",
            model_name="wide_unet_base32",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            model=UpsampleLogits(ValidationUNet(x_train.shape[1], base_channels=32), (resolution, resolution)),
            context_extra={"feature_names": FEATURE_NAMES, "base_channels": 32},
        )
    if variant == "fz_stats_features_unet":
        stats_train, stats_val, stats_test = normalize_train_val_test(
            fz_temporal_stats(train.fz),
            fz_temporal_stats(val.fz),
            fz_temporal_stats(test.fz),
            mode=args.feature_normalize,
        )
        feat_train, feat_val, feat_test = normalized_features(train, val, test, mode=args.feature_normalize)
        x_train = np.concatenate([stats_train, feat_train], axis=1)
        x_val = np.concatenate([stats_val, feat_val], axis=1)
        x_test = np.concatenate([require_array(stats_test), feat_test], axis=1)
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz_temporal_stats_plus_mechanical_features",
            model_name="unet",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            model=UpsampleLogits(ValidationUNet(x_train.shape[1], base_channels=24), (resolution, resolution)),
            context_extra={"feature_names": TEMPORAL_STAT_NAMES + FEATURE_NAMES},
        )
    if variant == "fz_residual_se_unet":
        x_train, x_val, x_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz",
            model_name="residual_se_unet",
            x_train=x_train,
            x_val=x_val,
            x_test=require_array(x_test),
            model=UpsampleLogits(ResidualSEUNet(x_train.shape[1], base_channels=24), (resolution, resolution)),
            context_extra={"attention": "squeeze_excite", "residual_blocks": True},
        )
    if variant == "fz_temporal_cnn16_unet":
        x_train, x_val, x_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
        return temporal_cnn_batch(variant, resolution, x_train, x_val, require_array(x_test), embed_channels=16, extra_channels=0)
    if variant == "fz_temporal_cnn32_unet":
        x_train, x_val, x_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
        return temporal_cnn_batch(variant, resolution, x_train, x_val, require_array(x_test), embed_channels=32, extra_channels=0)
    if variant == "fz_temporal_cnn32_stiffness_unet":
        fz_train, fz_val, fz_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
        stiff_train, stiff_val, stiff_test = normalized_stiffness(train, val, test, mode=args.stiffness_normalize)
        x_train = np.concatenate([fz_train, stiff_train], axis=1)
        x_val = np.concatenate([fz_val, stiff_val], axis=1)
        x_test = np.concatenate([require_array(fz_test), require_array(stiff_test)], axis=1)
        return temporal_cnn_batch(variant, resolution, x_train, x_val, x_test, embed_channels=32, extra_channels=1)
    if variant == "fz_temporal_gru32_unet":
        x_train, x_val, x_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
        return temporal_encoder_batch(variant, "temporal_gru32_unet", "gru", resolution, x_train, x_val, require_array(x_test), embed_channels=32, extra_channels=0)
    if variant == "fz_temporal_attention32_unet":
        x_train, x_val, x_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
        return temporal_encoder_batch(
            variant,
            "temporal_attention32_unet",
            "attention",
            resolution,
            x_train,
            x_val,
            require_array(x_test),
            embed_channels=32,
            extra_channels=0,
        )
    if variant == "fz_temporal_multiscale32_unet":
        x_train, x_val, x_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
        return temporal_encoder_batch(
            variant,
            "temporal_multiscale32_unet",
            "multiscale",
            resolution,
            x_train,
            x_val,
            require_array(x_test),
            embed_channels=32,
            extra_channels=0,
        )
    if variant == "fz_temporal_multiscale32_features_unet":
        fz_train, fz_val, fz_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
        feat_train, feat_val, feat_test = normalized_features(train, val, test, mode=args.feature_normalize)
        x_train = np.concatenate([fz_train, feat_train], axis=1)
        x_val = np.concatenate([fz_val, feat_val], axis=1)
        x_test = np.concatenate([require_array(fz_test), feat_test], axis=1)
        return temporal_encoder_batch(
            variant,
            "temporal_multiscale32_features_unet",
            "multiscale",
            resolution,
            x_train,
            x_val,
            x_test,
            embed_channels=32,
            extra_channels=int(feat_train.shape[1]),
            input_name="fz_plus_mechanical_features",
            context_extra={"feature_names": FEATURE_NAMES},
        )
    if variant == "limited_sincos_unet":
        x_train, x_val, x_test, sincos_context = limited_sincos_data(train, val, test, args)
        return VariantBatch(
            method=f"r{resolution}_{variant}",
            input_name="fz_limited_sincos",
            model_name="unet",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            model=UpsampleLogits(ValidationUNet(x_train.shape[1], base_channels=24), (resolution, resolution)),
            context_extra=sincos_context,
        )
    if variant == "limited_sincos_temporal_cnn16_unet":
        x_train, x_val, x_test, sincos_context = limited_sincos_data(train, val, test, args)
        return temporal_cnn_batch(
            variant,
            resolution,
            x_train,
            x_val,
            x_test,
            embed_channels=16,
            extra_channels=0,
            temporal_feature_size=sincos_context["temporal_feature_size"],
            input_name="fz_limited_sincos",
            context_extra=sincos_context,
        )
    if variant == "limited_sincos_temporal_cnn32_unet":
        x_train, x_val, x_test, sincos_context = limited_sincos_data(train, val, test, args)
        return temporal_cnn_batch(
            variant,
            resolution,
            x_train,
            x_val,
            x_test,
            embed_channels=32,
            extra_channels=0,
            temporal_feature_size=sincos_context["temporal_feature_size"],
            input_name="fz_limited_sincos",
            context_extra=sincos_context,
        )
    if variant == "limited_sincos_temporal_gru32_unet":
        x_train, x_val, x_test, sincos_context = limited_sincos_data(train, val, test, args)
        return temporal_encoder_batch(
            variant,
            "limited_sincos_temporal_gru32_unet",
            "gru",
            resolution,
            x_train,
            x_val,
            x_test,
            embed_channels=32,
            extra_channels=0,
            temporal_feature_size=sincos_context["temporal_feature_size"],
            input_name="fz_limited_sincos",
            context_extra=sincos_context,
        )
    if variant == "limited_sincos_temporal_attention32_unet":
        x_train, x_val, x_test, sincos_context = limited_sincos_data(train, val, test, args)
        return temporal_encoder_batch(
            variant,
            "limited_sincos_temporal_attention32_unet",
            "attention",
            resolution,
            x_train,
            x_val,
            x_test,
            embed_channels=32,
            extra_channels=0,
            temporal_feature_size=sincos_context["temporal_feature_size"],
            input_name="fz_limited_sincos",
            context_extra=sincos_context,
        )
    if variant == "limited_sincos_temporal_multiscale32_unet":
        x_train, x_val, x_test, sincos_context = limited_sincos_data(train, val, test, args)
        return temporal_encoder_batch(
            variant,
            "limited_sincos_temporal_multiscale32_unet",
            "multiscale",
            resolution,
            x_train,
            x_val,
            x_test,
            embed_channels=32,
            extra_channels=0,
            temporal_feature_size=sincos_context["temporal_feature_size"],
            input_name="fz_limited_sincos",
            context_extra=sincos_context,
        )
    raise ValueError(f"Unknown variant: {variant}")


def temporal_cnn_batch(
    variant: str,
    resolution: int,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    embed_channels: int,
    extra_channels: int,
    temporal_feature_size: int = 1,
    input_name: str | None = None,
    context_extra: dict[str, Any] | None = None,
) -> VariantBatch:
    curve_channels = int(x_train.shape[1] - extra_channels)
    default_input_name = "fz" if extra_channels == 0 else "fz_plus_stiffness"
    return VariantBatch(
        method=f"r{resolution}_{variant}",
        input_name=input_name or default_input_name,
        model_name=f"temporal_cnn{embed_channels}_unet" if extra_channels == 0 else f"temporal_cnn{embed_channels}_stiffness_unet",
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        model=PointwiseTemporalConvUNet(
            curve_channels=curve_channels,
            extra_channels=extra_channels,
            embed_channels=embed_channels,
            output_shape=(resolution, resolution),
            temporal_feature_size=temporal_feature_size,
        ),
        context_extra={
            "curve_channels": curve_channels,
            "extra_channels": int(extra_channels),
            "embed_channels": int(embed_channels),
            "temporal_feature_size": int(temporal_feature_size),
            **(context_extra or {}),
        },
    )


def temporal_encoder_batch(
    variant: str,
    model_name: str,
    encoder_type: str,
    resolution: int,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    embed_channels: int,
    extra_channels: int,
    temporal_feature_size: int = 1,
    input_name: str = "fz",
    context_extra: dict[str, Any] | None = None,
) -> VariantBatch:
    curve_channels = int(x_train.shape[1] - extra_channels)
    if encoder_type == "gru":
        model = PointwiseGRUUNet(
            curve_channels=curve_channels,
            extra_channels=extra_channels,
            embed_channels=embed_channels,
            output_shape=(resolution, resolution),
            temporal_feature_size=temporal_feature_size,
        )
    elif encoder_type == "attention":
        model = PointwiseTemporalAttentionUNet(
            curve_channels=curve_channels,
            extra_channels=extra_channels,
            embed_channels=embed_channels,
            output_shape=(resolution, resolution),
            temporal_feature_size=temporal_feature_size,
        )
    elif encoder_type == "multiscale":
        model = PointwiseMultiScaleTemporalConvUNet(
            curve_channels=curve_channels,
            extra_channels=extra_channels,
            embed_channels=embed_channels,
            output_shape=(resolution, resolution),
            temporal_feature_size=temporal_feature_size,
        )
    else:
        raise ValueError(f"Unknown temporal encoder type: {encoder_type}")
    return VariantBatch(
        method=f"r{resolution}_{variant}",
        input_name=input_name,
        model_name=model_name,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        model=model,
        context_extra={
            "curve_channels": curve_channels,
            "extra_channels": int(extra_channels),
            "embed_channels": int(embed_channels),
            "temporal_feature_size": int(temporal_feature_size),
            "temporal_encoder": encoder_type,
            **(context_extra or {}),
        },
    )


def run_variant(
    batch: VariantBatch,
    train: SplitData,
    val: SplitData,
    test: SplitData,
    out_dir: Path,
    context: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    method_dir = out_dir / batch.method
    already_done = _method_done(method_dir, args.force)
    if not already_done:
        started = time.perf_counter()
        train_neural_method(
            batch.method,
            "fz_temporal_variant",
            batch.model,
            batch.x_train,
            train.masks,
            batch.x_val,
            val.masks,
            val.names,
            out_dir,
            context={**context, "started_at_unix": started},
            args=args,
            augment=batch.augment,
            focal=batch.focal,
        )
    else:
        print(f"{batch.method}: existing metrics found, loading checkpoint")
        checkpoint = torch.load(method_dir / "best.pt", map_location="cpu")
        batch.model.load_state_dict(checkpoint["model_state"])

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = batch.model.to(device)
    test_scores = predict_neural(model, batch.x_test, device, batch_size=args.batch_size)
    val_summary = read_json(method_dir / "metrics_summary.json")
    val_best_threshold = float(val_summary.get("threshold_sweep_best", {}).get("threshold", 0.5))
    write_test_metrics(
        method_dir,
        test_scores,
        test.masks,
        test.names,
        fixed_threshold=0.5,
        val_best_threshold=val_best_threshold,
        context=context,
        force=True,
    )


def limited_sincos_data(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    seed = int(args.seed if args.limited_trajectory_seed is None else args.limited_trajectory_seed)
    input_steps = int(args.trajectory_input_steps) if int(args.trajectory_input_steps) > 0 else int(train.fz.shape[1])
    embedding_dim = int(args.positional_embedding_dim)
    position_scale = displacement_position_scale(train.z)
    train_x = limited_trajectory_sincos_inputs(
        train,
        split_name="train",
        input_steps=input_steps,
        seed=seed,
        embedding_dim=embedding_dim,
        position_scale=position_scale,
    )
    val_x = limited_trajectory_sincos_inputs(
        val,
        split_name="val",
        input_steps=input_steps,
        seed=seed,
        embedding_dim=embedding_dim,
        position_scale=position_scale,
    )
    test_x = limited_trajectory_sincos_inputs(
        test,
        split_name="test",
        input_steps=input_steps,
        seed=seed,
        embedding_dim=embedding_dim,
        position_scale=position_scale,
    )
    temporal_feature_size = 1 + embedding_dim
    train_x, val_x, test_x_norm = normalize_temporal_feature_fz_channels(
        train_x,
        val_x,
        test_x,
        temporal_feature_size=temporal_feature_size,
        mode=args.fz_normalize,
    )
    context = {
        "limited_trajectory_seed": seed,
        "trajectory_input_steps": input_steps,
        "position_encoding": "sinusoidal_displacement",
        "positional_embedding_dim": embedding_dim,
        "position_scale_from_train": position_scale,
        "temporal_feature_size": temporal_feature_size,
        "feature_layout": "per time step: [normalized_fz, sin/cos displacement embedding]",
        "fz_normalize_only": args.fz_normalize,
    }
    return train_x, val_x, require_array(test_x_norm), context


def fz_temporal_stats(fz: np.ndarray) -> np.ndarray:
    f = np.nan_to_num(fz.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    t = f.shape[1]
    i25 = max(0, min(t - 1, round(0.25 * (t - 1))))
    i50 = max(0, min(t - 1, round(0.50 * (t - 1))))
    i75 = max(0, min(t - 1, round(0.75 * (t - 1))))
    delta = np.diff(f, axis=1, prepend=f[:, :1])
    f_max = np.max(f, axis=1)
    max_safe = np.maximum(f_max, np.float32(1e-6))
    above_contact = f >= (0.05 * max_safe[:, None])
    contact_idx = np.argmax(above_contact, axis=1).astype(np.float32)
    has_contact = np.any(above_contact, axis=1)
    contact_time = np.where(has_contact, contact_idx / max(t - 1, 1), 1.0).astype(np.float32)
    argmax_idx = np.argmax(f, axis=1).astype(np.float32)
    argmax_time = argmax_idx / max(t - 1, 1)
    contact_to_peak = np.maximum(argmax_time - contact_time, 0.0).astype(np.float32)
    features = [
        f[:, 0],
        f[:, -1],
        f_max,
        (0.5 * f[:, 0] + np.sum(f[:, 1:-1], axis=1) + 0.5 * f[:, -1]) / max(t - 1, 1),
        np.std(f, axis=1),
        np.mean(f, axis=1),
        f[:, i25],
        f[:, i50],
        f[:, i75],
        (f[:, i25] - f[:, 0]) / max(i25, 1),
        (f[:, i50] - f[:, i25]) / max(i50 - i25, 1),
        (f[:, i75] - f[:, i50]) / max(i75 - i50, 1),
        (f[:, -1] - f[:, i75]) / max((t - 1) - i75, 1),
        np.max(delta, axis=1),
        np.mean(delta, axis=1),
        argmax_time.astype(np.float32),
        contact_time,
        contact_to_peak,
    ]
    return np.stack(features, axis=1).astype(np.float32)


def fz_plus_delta(fz: np.ndarray) -> np.ndarray:
    f = np.nan_to_num(fz.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    delta = np.diff(f, axis=1, prepend=f[:, :1]).astype(np.float32)
    return np.concatenate([f, delta], axis=1).astype(np.float32)


def normalized_stiffness(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = FEATURE_NAMES.index("equivalent_stiffness")
    train_x = train.features[:, idx][:, None]
    val_x = val.features[:, idx][:, None]
    test_x = test.features[:, idx][:, None]
    train_norm, val_norm = normalize_maps(train_x, val_x, mode=mode)
    if mode == "none":
        test_norm = np.nan_to_num(test_x.astype(np.float32))
    elif mode == "sample":
        test_norm = normalize_maps(train_x[:1], test_x, mode="sample")[1]
    elif mode == "dataset":
        mean = np.nanmean(train_x.astype(np.float32), axis=(0, 2, 3), keepdims=True)
        std = np.nanstd(train_x.astype(np.float32), axis=(0, 2, 3), keepdims=True)
        test_norm = np.nan_to_num((test_x.astype(np.float32) - mean) / (std + np.float32(1e-6)), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError(f"Unknown stiffness normalize mode: {mode}")
    return train_norm.astype(np.float32), val_norm.astype(np.float32), test_norm.astype(np.float32)


def normalized_features(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_norm, val_norm = normalize_maps(train.features, val.features, mode=mode)
    if mode == "none":
        test_norm = np.nan_to_num(test.features.astype(np.float32))
    elif mode == "sample":
        test_norm = normalize_maps(train.features[:1], test.features, mode="sample")[1]
    elif mode == "dataset":
        mean = np.nanmean(train.features.astype(np.float32), axis=(0, 2, 3), keepdims=True)
        std = np.nanstd(train.features.astype(np.float32), axis=(0, 2, 3), keepdims=True)
        test_norm = np.nan_to_num((test.features.astype(np.float32) - mean) / (std + np.float32(1e-6)), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError(f"Unknown feature normalize mode: {mode}")
    return train_norm.astype(np.float32), val_norm.astype(np.float32), test_norm.astype(np.float32)


def normalized_fz_features(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fz_train, fz_val, fz_test = normalize_train_val_test(train.fz, val.fz, test.fz, mode=args.fz_normalize)
    feat_train, feat_val, feat_test = normalized_features(train, val, test, mode=args.feature_normalize)
    return (
        np.concatenate([fz_train, feat_train], axis=1),
        np.concatenate([fz_val, feat_val], axis=1),
        np.concatenate([require_array(fz_test), feat_test], axis=1),
    )


def write_leaderboard(out_dir: Path, *, baseline_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out_dir.glob("*/metrics_summary.json")):
        summary = read_json(metrics_path)
        config = summary.get("config", {})
        fixed = summary.get("fixed_threshold", {})
        best = summary.get("threshold_sweep_best", {})
        test_summary = read_json(metrics_path.parent / "test_metrics_summary.json")
        test_fixed = test_summary.get("fixed_threshold", {})
        test_val = test_summary.get("val_selected_threshold", {})
        test_oracle = test_summary.get("test_oracle_threshold", {})
        rows.append(
            {
                "source": "variant",
                "method": summary.get("method", metrics_path.parent.name),
                "resolution": config.get("resolution", ""),
                "input": config.get("input", ""),
                "model": config.get("model", ""),
                "variant": config.get("variant", ""),
                "val_fixed_dice": fixed.get("dice", ""),
                "val_best_dice": best.get("dice", ""),
                "val_best_threshold": best.get("threshold", ""),
                "test_fixed_dice": test_fixed.get("dice", ""),
                "test_val_selected_dice": test_val.get("dice", ""),
                "test_val_selected_threshold": test_val.get("threshold", ""),
                "test_oracle_dice": test_oracle.get("dice", ""),
                "path": str(metrics_path.parent),
            }
        )
    baseline_rows = read_baseline_rows(baseline_dir)
    rows.extend(baseline_rows)
    rows.sort(key=lambda row: float(row["test_val_selected_dice"] or -1.0), reverse=True)
    if rows:
        write_csv(out_dir / "leaderboard_with_baselines.csv", rows, list(rows[0].keys()))
        write_csv(out_dir / "leaderboard.csv", [row for row in rows if row["source"] == "variant"], list(rows[0].keys()))
    write_report(out_dir / "report.md", rows)


def read_baseline_rows(baseline_dir: Path) -> list[dict[str, Any]]:
    path = baseline_dir / "leaderboard.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "source": "baseline",
                    "method": row["method"],
                    "resolution": row["resolution"],
                    "input": row["input"],
                    "model": row["model"],
                    "variant": "",
                    "val_fixed_dice": row["val_fixed_dice"],
                    "val_best_dice": row["val_best_dice"],
                    "val_best_threshold": row["val_best_threshold"],
                    "test_fixed_dice": row["test_fixed_dice"],
                    "test_val_selected_dice": row["test_val_selected_dice"],
                    "test_val_selected_threshold": row["test_val_selected_threshold"],
                    "test_oracle_dice": row["test_oracle_dice"],
                    "path": row["path"],
                }
            )
    return rows


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# High-Resolution Raw-Fz Temporal Variant Sweep", ""]
    if rows:
        variant_rows = [row for row in rows if row["source"] == "variant"]
        lines.append(f"- Variant methods completed: {len(variant_rows)}")
        lines.append(f"- Best row by test Dice: `{rows[0]['method']}` ({rows[0]['source']}) Dice={fmt(rows[0]['test_val_selected_dice'])}")
        lines.append("")
        lines.append("| rank | source | method | input | model | val best Dice | test val-selected Dice | threshold |")
        lines.append("|---:|---|---|---|---|---:|---:|---:|")
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {row['source']} | `{row['method']}` | {row['input']} | {row['model']} | "
                f"{fmt(row['val_best_dice'])} | {fmt(row['test_val_selected_dice'])} | {fmt(row['test_val_selected_threshold'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("Expected at least one integer resolution.")
    if any(value <= 0 for value in values):
        raise SystemExit("Resolutions must be positive.")
    return values


def parse_variant_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = set(DEFAULT_VARIANTS)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; allowed: {sorted(allowed)}")
    if not values:
        raise SystemExit("Expected at least one variant.")
    return values


def _norm1d(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


def _norm2d(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


def require_array(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        raise ValueError("Expected an array, got None.")
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
