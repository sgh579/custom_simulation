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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from palpation_sim.config import PhantomConfig, ScanConfig
from palpation_sim.native_data import load_phantom_scan_material_lumps
from palpation_sim.phantom import LumpSpec, lumps_membership
from palpation_sim.workflow import require_runtime_environment
from run_segmentation_accuracy_sweep import _read_json, _seed_everything, _write_csv, _write_json


DEFAULT_PACKAGE_DIR = Path("data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/random_trajectory")
DEFAULT_OUT_DIR = Path("runs/wrench_3d_volume_prediction_20260622")
DEFAULT_VOLUME_SHAPE = (32, 64, 64)
DEFAULT_VARIANTS = ("wrench_temporal_cnn_volume",)
ALLOWED_VARIANTS = (
    "wrench_temporal_cnn_volume",
    "wrench_temporal_cnn_world_volume",
    "wrench_temporal_cnn_world_curve_volume",
)
WRENCH_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
ACTION_NAMES = [
    "x",
    "y",
    "traj_dx_mean",
    "traj_dy_mean",
    "traj_dx_std",
    "traj_dy_std",
    "traj_dx_final",
    "traj_dy_final",
    "indent_mean",
    "indent_max",
    "indent_final",
    "pose_final_px",
    "pose_final_py",
    "pose_final_pz",
    "pose_final_qx",
    "pose_final_qy",
    "pose_final_qz",
    "pose_final_qw",
]


@dataclass(frozen=True)
class RawVolumeSplit:
    names: list[str]
    wrench: np.ndarray
    action: np.ndarray
    volumes: np.ndarray


@dataclass(frozen=True)
class VolumeSplit:
    names: list[str]
    x: np.ndarray
    volumes: np.ndarray


@dataclass(frozen=True)
class NormStats:
    wrench_mean: np.ndarray
    wrench_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray


class VolumeDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class TemporalCNNVolumeNet(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        action_channels: int,
        volume_shape: tuple[int, int, int],
        embed_channels: int,
        hidden_channels: int,
        decoder_base_channels: int,
    ) -> None:
        super().__init__()
        if int(curve_channels) % len(WRENCH_NAMES) != 0:
            raise ValueError(f"curve_channels={curve_channels} must be divisible by {len(WRENCH_NAMES)}")
        self.curve_channels = int(curve_channels)
        self.action_channels = int(action_channels)
        self.steps = self.curve_channels // len(WRENCH_NAMES)
        self.volume_shape = tuple(int(v) for v in volume_shape)
        embed = int(embed_channels)
        hidden = int(hidden_channels)
        self.temporal = nn.Sequential(
            nn.Conv1d(len(WRENCH_NAMES), hidden, kernel_size=7, padding=3, bias=False),
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
        self.mixer = nn.Sequential(
            conv_block(embed * 2, embed),
            nn.Conv2d(embed, embed, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(num_groups=min(8, embed), num_channels=embed),
            nn.GELU(),
            conv_block(embed, embed),
        )
        depth, _height, _width = self.volume_shape
        self.decoder = SliceUNet2D(embed, out_channels=depth, base_channels=int(decoder_base_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        curve = x[:, : self.curve_channels]
        action = x[:, self.curve_channels : self.curve_channels + self.action_channels]
        b, _c, h, w = x.shape
        seq = curve.reshape(b, self.steps, len(WRENCH_NAMES), h, w)
        seq = seq.permute(0, 3, 4, 2, 1).reshape(b * h * w, len(WRENCH_NAMES), self.steps)
        encoded = self.temporal(seq).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        features = self.mixer(torch.cat([encoded, self.action_proj(action)], dim=1))
        logits = self.decoder(features)
        _depth, out_h, out_w = self.volume_shape
        if logits.shape[-2:] != (out_h, out_w):
            logits = F.interpolate(logits, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return logits


class TemporalCNNWorldVolumeNet(nn.Module):
    def __init__(
        self,
        *,
        curve_channels: int,
        action_channels: int,
        volume_shape: tuple[int, int, int],
        embed_channels: int,
        hidden_channels: int,
        decoder_base_channels: int,
        world_target_ratio: float,
        curve_aux: bool,
    ) -> None:
        super().__init__()
        if int(curve_channels) % len(WRENCH_NAMES) != 0:
            raise ValueError(f"curve_channels={curve_channels} must be divisible by {len(WRENCH_NAMES)}")
        self.curve_channels = int(curve_channels)
        self.action_channels = int(action_channels)
        self.steps = self.curve_channels // len(WRENCH_NAMES)
        self.volume_shape = tuple(int(v) for v in volume_shape)
        self.world_target_ratio = float(world_target_ratio)
        self.curve_aux = bool(curve_aux)
        embed = int(embed_channels)
        hidden = int(hidden_channels)
        self.temporal = nn.Sequential(
            nn.Conv1d(len(WRENCH_NAMES), hidden, kernel_size=7, padding=3, bias=False),
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
        self.mixer = nn.Sequential(
            conv_block(embed * 2 + embed, embed),
            nn.Conv2d(embed, embed, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(num_groups=min(8, embed), num_channels=embed),
            nn.GELU(),
            conv_block(embed, embed),
        )
        depth, _height, _width = self.volume_shape
        self.decoder = SliceUNet2D(embed, out_channels=depth, base_channels=int(decoder_base_channels))
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

    def encode_features(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        curve = x[:, : self.curve_channels]
        action = x[:, self.curve_channels : self.curve_channels + self.action_channels]
        b, _c, h, w = x.shape
        if context_keep_mask is None:
            keep_map = torch.ones((b, 1, h, w), dtype=x.dtype, device=x.device)
        else:
            keep = context_keep_mask.to(device=x.device, dtype=torch.bool)
            keep_map = keep.reshape(b, 1, h, w).to(dtype=x.dtype)
        observed_curve = curve * keep_map
        seq = observed_curve.reshape(b, self.steps, len(WRENCH_NAMES), h, w)
        seq = seq.permute(0, 3, 4, 2, 1).reshape(b * h * w, len(WRENCH_NAMES), self.steps)
        encoded = self.temporal(seq).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        features = torch.cat([encoded, self.action_proj(action), self.flag_proj(keep_map)], dim=1)
        return self.mixer(features)

    def forward(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encode_features(x, context_keep_mask=context_keep_mask)
        logits = self.decoder(features)
        _depth, out_h, out_w = self.volume_shape
        if logits.shape[-2:] != (out_h, out_w):
            logits = F.interpolate(logits, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return logits

    def predict_curves(self, x: torch.Tensor, *, context_keep_mask: torch.Tensor) -> torch.Tensor:
        features = self.encode_features(x, context_keep_mask=context_keep_mask)
        pred = self.curve_decoder(features)
        b, _c, h, w = pred.shape
        return pred.permute(0, 2, 3, 1).reshape(b, h * w, self.curve_channels)

    def auxiliary_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        curve_tokens, _action_tokens = self.split_input(x)
        b, n, _ = curve_tokens.shape
        target_mask = torch.rand((b, n), device=x.device) < self.world_target_ratio
        target_mask = ensure_any_masked_and_context(target_mask)
        pred = self.predict_curves(x, context_keep_mask=~target_mask)
        curve_loss = F.smooth_l1_loss(pred[target_mask], curve_tokens[target_mask])
        if not self.curve_aux:
            return torch.zeros((), device=x.device), {"curve_huber": float(curve_loss.detach().cpu())}
        return curve_loss, {"curve_huber": float(curve_loss.detach().cpu())}


class SliceUNet2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int) -> None:
        super().__init__()
        b = int(base_channels)
        self.inc = conv_block(in_channels, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), conv_block(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), conv_block(b * 2, b * 4))
        self.up2 = conv_block(b * 4 + b * 2, b * 2)
        self.up1 = conv_block(b * 2 + b, b)
        self.outc = nn.Conv2d(b, int(out_channels), kernel_size=1)

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
        nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, int(out_channels)), num_channels=int(out_channels)),
        nn.GELU(),
        nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, int(out_channels)), num_channels=int(out_channels)),
        nn.GELU(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict scan-aligned 3D lump occupancy volumes from 6D wrench palpation.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--volume-shape", type=str, default=",".join(str(v) for v in DEFAULT_VOLUME_SHAPE), help="D,H,W depth-slice volume shape.")
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS), help="Comma list of 3D variants.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--embed-channels", type=int, default=96)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--decoder-base-channels", type=int, default=32)
    parser.add_argument("--projection-loss-weight", type=float, default=0.25)
    parser.add_argument("--curve-aux-weight", type=float, default=0.1)
    parser.add_argument("--depth-aware-loss-weight", type=float, default=0.0)
    parser.add_argument("--depth-loss-shallow-weight", type=float, default=1.35)
    parser.add_argument("--depth-loss-mid-weight", type=float, default=1.0)
    parser.add_argument("--depth-loss-deep-weight", type=float, default=0.65)
    parser.add_argument("--deep-fp-loss-weight", type=float, default=0.35)
    parser.add_argument("--checkpoint-metric", choices=("dice", "depth_utility"), default="dice")
    parser.add_argument("--depth-utility-shallow-weight", type=float, default=0.45)
    parser.add_argument("--depth-utility-mid-weight", type=float, default=0.35)
    parser.add_argument("--depth-utility-deep-weight", type=float, default=0.20)
    parser.add_argument("--depth-utility-deep-fp-penalty", type=float, default=1.0)
    parser.add_argument("--seg-context-min", type=float, default=0.5)
    parser.add_argument("--seg-context-max", type=float, default=1.0)
    parser.add_argument("--world-target-ratio", type=float, default=0.35)
    parser.add_argument("--sparse-context-ratios", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--pos-weight-cap", type=float, default=50.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    require_runtime_environment()
    _seed_everything(args.seed)
    random.seed(args.seed)
    volume_shape = parse_volume_shape(args.volume_shape)
    variants = parse_variants(args.variants)
    sparse_context_ratios = parse_floats(args.sparse_context_ratios)
    if not (0.0 < float(args.seg_context_min) <= float(args.seg_context_max) <= 1.0):
        raise ValueError("--seg-context-min/max must satisfy 0 < min <= max <= 1")
    if not (0.0 < float(args.world_target_ratio) < 1.0):
        raise ValueError("--world-target-ratio must be in (0, 1)")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.batch_size = min(args.batch_size, 4)
        args.embed_channels = min(args.embed_channels, 48)
        args.hidden_channels = min(args.hidden_channels, 48)
        args.decoder_base_channels = min(args.decoder_base_channels, 16)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.package_dir / "data"
    max_train = 8 if args.smoke else None
    max_eval = 4 if args.smoke else None
    print(f"loading 3D train split: {data_root / 'train'} volume_shape={volume_shape}")
    raw_train = load_raw_volume_split(data_root / "train", volume_shape=volume_shape, max_samples=max_train)
    print(f"loading 3D val split: {data_root / 'val'} volume_shape={volume_shape}")
    raw_val = load_raw_volume_split(data_root / "val", volume_shape=volume_shape, max_samples=max_eval)
    raw_test = None
    if not args.no_test:
        print(f"loading 3D test split: {data_root / 'test'} volume_shape={volume_shape}")
        raw_test = load_raw_volume_split(data_root / "test", volume_shape=volume_shape, max_samples=max_eval)

    stats = fit_norm_stats(raw_train)
    train = apply_norm(raw_train, stats)
    val = apply_norm(raw_val, stats)
    test = apply_norm(raw_test, stats) if raw_test is not None else None
    run_config = {
        "package_dir": str(args.package_dir),
        "out_dir": str(args.out_dir),
        "representation": "scan_aligned_depth_slice_occupancy",
        "variants": variants,
        "volume_shape_dhw": list(volume_shape),
        "volume_axis_order": "depth_from_top,y,x",
        "volume_xy_extent": "same x/y range as ScanConfig.x_values/y_values",
        "volume_z_extent": "full phantom height, depth 0 at top surface",
        "input": "probe_wrench_plus_compact_action",
        "input_contract": "probe_wrench[H,W,T,6] with compact action channels",
        "wrench_channels": WRENCH_NAMES,
        "action_channels": ACTION_NAMES,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "projection_loss_weight": float(args.projection_loss_weight),
        "curve_aux_weight": float(args.curve_aux_weight),
        "depth_aware_loss_weight": float(args.depth_aware_loss_weight),
        "depth_loss_shallow_weight": float(args.depth_loss_shallow_weight),
        "depth_loss_mid_weight": float(args.depth_loss_mid_weight),
        "depth_loss_deep_weight": float(args.depth_loss_deep_weight),
        "deep_fp_loss_weight": float(args.deep_fp_loss_weight),
        "checkpoint_metric": args.checkpoint_metric,
        "depth_utility_weights": {
            "shallow": float(args.depth_utility_shallow_weight),
            "mid": float(args.depth_utility_mid_weight),
            "deep": float(args.depth_utility_deep_weight),
            "deep_fp_penalty": float(args.depth_utility_deep_fp_penalty),
        },
        "seg_context_min": float(args.seg_context_min),
        "seg_context_max": float(args.seg_context_max),
        "world_target_ratio": float(args.world_target_ratio),
        "sparse_context_ratios": sparse_context_ratios,
        "pos_weight_cap": float(args.pos_weight_cap),
        "amp": bool(use_amp(args, resolve_device(args.device))),
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "train_samples": len(train.names),
        "val_samples": len(val.names),
        "test_samples": len(test.names) if test is not None else 0,
        "input_shape_chw": list(train.x.shape[1:]),
    }
    _write_json(args.out_dir / "run_config.json", run_config)
    curve_channels = raw_train.wrench.shape[3] * len(WRENCH_NAMES)
    action_channels = raw_train.action.shape[1]
    for variant in variants:
        method = f"v{volume_shape[0]}x{volume_shape[1]}x{volume_shape[2]}_{variant}"
        model = build_model(
            variant,
            curve_channels=curve_channels,
            action_channels=action_channels,
            volume_shape=volume_shape,
            args=args,
        )
        train_method(
            method,
            model,
            train,
            val,
            test,
            args,
            context={**run_config, "method": method, "variant": variant, "model": model_name_for_variant(variant)},
        )
        write_leaderboard(args.out_dir)
    print(f"3D volume prediction complete: {args.out_dir}")


def load_raw_volume_split(split_dir: Path, *, volume_shape: tuple[int, int, int], max_samples: int | None) -> RawVolumeSplit:
    files = sorted(split_dir.glob("*.npz"))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")
    names: list[str] = []
    wrench_items: list[np.ndarray] = []
    action_items: list[np.ndarray] = []
    volume_items: list[np.ndarray] = []
    for path in files:
        with np.load(path, allow_pickle=False) as sample:
            wrench = np.asarray(sample["probe_wrench"], dtype=np.float32)
            if wrench.ndim != 4 or wrench.shape[-1] != len(WRENCH_NAMES):
                raise ValueError(f"{path}: expected probe_wrench[H,W,T,6], got {wrench.shape}")
            action = compact_action_channels(sample, wrench.shape)
        phantom, scan, _material, lumps, _metadata = load_phantom_scan_material_lumps(
            sample_path=path,
            metadata_path=path.with_name(f"{path.stem}_gt.json"),
        )
        volume = volume_label(scan, phantom, lumps, volume_shape)
        names.append(path.name)
        wrench_items.append(np.nan_to_num(wrench, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        action_items.append(action.astype(np.float32))
        volume_items.append(volume.astype(np.uint8))
    return RawVolumeSplit(
        names=names,
        wrench=np.stack(wrench_items).astype(np.float32),
        action=np.stack(action_items).astype(np.float32),
        volumes=np.stack(volume_items).astype(np.uint8),
    )


def build_model(
    variant: str,
    *,
    curve_channels: int,
    action_channels: int,
    volume_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> nn.Module:
    if variant == "wrench_temporal_cnn_volume":
        return TemporalCNNVolumeNet(
            curve_channels=curve_channels,
            action_channels=action_channels,
            volume_shape=volume_shape,
            embed_channels=int(args.embed_channels),
            hidden_channels=int(args.hidden_channels),
            decoder_base_channels=int(args.decoder_base_channels),
        )
    if variant in {"wrench_temporal_cnn_world_volume", "wrench_temporal_cnn_world_curve_volume"}:
        return TemporalCNNWorldVolumeNet(
            curve_channels=curve_channels,
            action_channels=action_channels,
            volume_shape=volume_shape,
            embed_channels=int(args.embed_channels),
            hidden_channels=int(args.hidden_channels),
            decoder_base_channels=int(args.decoder_base_channels),
            world_target_ratio=float(args.world_target_ratio),
            curve_aux=variant == "wrench_temporal_cnn_world_curve_volume",
        )
    raise ValueError(f"Unhandled variant: {variant}")


def model_name_for_variant(variant: str) -> str:
    if variant == "wrench_temporal_cnn_volume":
        return "temporal_cnn_depth_slice_volume"
    if variant in {"wrench_temporal_cnn_world_volume", "wrench_temporal_cnn_world_curve_volume"}:
        return "temporal_cnn_world_depth_slice_volume"
    return variant


def compact_action_channels(sample: np.lib.npyio.NpzFile, wrench_shape: tuple[int, int, int, int]) -> np.ndarray:
    h, w, t, _c = wrench_shape
    xy = np.asarray(sample["xy"], dtype=np.float32)
    trajectory_xy = np.asarray(sample["trajectory_xy_offset"], dtype=np.float32)
    indentation = np.asarray(sample["indentation_depth"], dtype=np.float32)
    pose = np.asarray(sample["probe_pose"], dtype=np.float32)
    if xy.shape != (h, w, 2):
        raise ValueError(f"Expected xy shape {(h, w, 2)}, got {xy.shape}")
    if trajectory_xy.shape != (h, w, t, 2):
        raise ValueError(f"Expected trajectory_xy_offset shape {(h, w, t, 2)}, got {trajectory_xy.shape}")
    if indentation.shape != (h, w, t):
        raise ValueError(f"Expected indentation_depth shape {(h, w, t)}, got {indentation.shape}")
    if pose.shape != (h, w, t, 7):
        raise ValueError(f"Expected probe_pose shape {(h, w, t, 7)}, got {pose.shape}")
    channels = [
        xy[..., 0],
        xy[..., 1],
        trajectory_xy[..., 0].mean(axis=2),
        trajectory_xy[..., 1].mean(axis=2),
        trajectory_xy[..., 0].std(axis=2),
        trajectory_xy[..., 1].std(axis=2),
        trajectory_xy[..., -1, 0],
        trajectory_xy[..., -1, 1],
        indentation.mean(axis=2),
        indentation.max(axis=2),
        indentation[..., -1],
    ]
    channels.extend([pose[..., -1, idx] for idx in range(7)])
    action = np.stack(channels, axis=0)
    if action.shape[0] != len(ACTION_NAMES):
        raise AssertionError(f"Expected {len(ACTION_NAMES)} action channels, got {action.shape[0]}")
    return np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def volume_label(
    scan: ScanConfig,
    phantom: PhantomConfig,
    lumps: Sequence[LumpSpec],
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    depth_count, height, width = volume_shape
    scan_x = np.asarray(scan.x_values(phantom), dtype=np.float32)
    scan_y = np.asarray(scan.y_values(phantom), dtype=np.float32)
    xs = voxel_centers(float(scan_x[0]), float(scan_x[-1]), width)
    ys = voxel_centers(float(scan_y[0]), float(scan_y[-1]), height)
    depths = voxel_centers(0.0, float(phantom.height), depth_count)
    zv = float(phantom.height) - depths[:, None, None]
    yv = ys[None, :, None]
    xv = xs[None, None, :]
    points = np.empty((depth_count, height, width, 3), dtype=np.float32)
    points[..., 0] = xv
    points[..., 1] = yv
    points[..., 2] = zv
    return lumps_membership(points, lumps, project_xy=False).astype(np.uint8)


def voxel_centers(lo: float, hi: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([0.5 * (lo + hi)], dtype=np.float32)
    edges = np.linspace(float(lo), float(hi), int(count) + 1, dtype=np.float32)
    return (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)


def fit_norm_stats(raw: RawVolumeSplit) -> NormStats:
    wrench_mean = raw.wrench.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    wrench_std = raw.wrench.std(axis=(0, 1, 2), keepdims=True).astype(np.float32)
    action_mean = raw.action.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    action_std = raw.action.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    return NormStats(
        wrench_mean=wrench_mean,
        wrench_std=np.maximum(wrench_std, np.float32(1e-6)),
        action_mean=action_mean,
        action_std=np.maximum(action_std, np.float32(1e-6)),
    )


def apply_norm(raw: RawVolumeSplit | None, stats: NormStats) -> VolumeSplit | None:
    if raw is None:
        return None
    wrench = (raw.wrench.astype(np.float32) - stats.wrench_mean) / stats.wrench_std
    action = (raw.action.astype(np.float32) - stats.action_mean) / stats.action_std
    n, h, w, t, c = wrench.shape
    wrench_x = wrench.transpose(0, 3, 4, 1, 2).reshape(n, t * c, h, w).astype(np.float32)
    x = np.concatenate([wrench_x, action.astype(np.float32)], axis=1)
    return VolumeSplit(names=list(raw.names), x=x.astype(np.float32), volumes=raw.volumes.astype(np.uint8))


def train_method(
    method: str,
    model: nn.Module,
    train: VolumeSplit,
    val: VolumeSplit,
    test: VolumeSplit | None,
    args: argparse.Namespace,
    *,
    context: dict[str, Any],
) -> None:
    method_dir = args.out_dir / method
    if method_dir.joinpath("metrics_summary.json").exists() and not args.force:
        print(f"{method}: existing metrics found")
        return
    method_dir.mkdir(parents=True, exist_ok=True)
    _write_json(method_dir / "config.json", context)
    device = resolve_device(args.device)
    model = model.to(device)
    train_loader = DataLoader(VolumeDataset(train.x, train.volumes), batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(VolumeDataset(val.x, val.volumes), batch_size=int(args.batch_size), shuffle=False)
    pos = max(float(train.volumes.sum()), 1.0)
    neg = max(float(train.volumes.size - train.volumes.sum()), 1.0)
    pos_weight_value = min(neg / pos, float(args.pos_weight_cap))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp(args, device))
    best_score = -1.0
    best_dice = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without = 0
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight, scaler=scaler, args=args)
        val_stats = run_epoch(model, val_loader, None, device, pos_weight=pos_weight, scaler=None, args=args)
        elapsed = time.perf_counter() - start
        row = {"epoch": epoch, **prefix_keys(train_stats, "train_"), **prefix_keys(val_stats, "val_"), "elapsed_seconds": elapsed}
        rows.append(row)
        _write_csv(method_dir / "history.csv", rows, list(row.keys()))
        checkpoint_value = checkpoint_score(val_stats, args)
        if checkpoint_value > best_score + 1e-4:
            best_score = checkpoint_value
            best_dice = val_stats["dice"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model_state": best_state, "method": method, "context": context}, method_dir / "best.pt")
            epochs_without = 0
        else:
            epochs_without += 1
        print(
            f"{method} epoch {epoch:03d} train_dice={train_stats['dice']:.4f} "
            f"val_dice={val_stats['dice']:.4f} val_proj_dice={val_stats['projection_dice']:.4f} "
            f"val_depth_utility={val_stats['depth_utility']:.4f} "
            f"loss={val_stats['loss']:.4f} elapsed_min={elapsed / 60.0:.2f}"
        )
        if epochs_without >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    val_scores = predict_scores(model, val.x, device=device, batch_size=int(args.batch_size))
    write_volume_metrics(
        method_dir,
        method,
        val_scores,
        val.volumes,
        val.names,
        context={
            **context,
            "epochs_completed": len(rows),
            "best_fixed_val_dice": best_dice,
            "best_checkpoint_score": best_score,
            "pos_weight": pos_weight_value,
        },
        fixed_threshold=0.5,
        split="val",
    )
    if test is not None:
        val_summary = _read_json(method_dir / "metrics_summary.json")
        threshold = float(val_summary.get("threshold_sweep_best", {}).get("threshold", 0.5))
        test_scores = predict_scores(model, test.x, device=device, batch_size=int(args.batch_size))
        write_volume_metrics(
            method_dir,
            method,
            test_scores,
            test.volumes,
            test.names,
            context={**context, "val_selected_threshold": threshold},
            fixed_threshold=threshold,
            split="test",
        )
        if supports_context_mask(model):
            world_rows = evaluate_world_volume_metrics(
                model,
                test,
                device=device,
                batch_size=int(args.batch_size),
                sparse_ratios=parse_floats(args.sparse_context_ratios),
                threshold=threshold,
                args=args,
            )
            write_world_metrics(method_dir, method, world_rows)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    pos_weight: torch.Tensor,
    scaler: torch.amp.GradScaler | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "projection_loss": 0.0,
        "depth_loss": 0.0,
        "dice": 0.0,
        "projection_dice": 0.0,
        "shallow_dice": 0.0,
        "mid_dice": 0.0,
        "deep_dice": 0.0,
        "deep_false_positive_rate": 0.0,
        "depth_utility": 0.0,
    }
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=use_amp(args, device)):
            forward_kwargs: dict[str, Any] = {}
            if training and supports_context_mask(model):
                ratio = random.uniform(float(args.seg_context_min), float(args.seg_context_max))
                b, _channels, h, w = x.shape
                forward_kwargs["context_keep_mask"] = random_context_keep_mask(b, h * w, ratio, x.device)
            logits = model(x, **forward_kwargs) if forward_kwargs else model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            dloss = volume_dice_loss(logits, y)
            ploss = projection_loss(logits, y)
            depth_loss = torch.zeros((), device=device)
            if float(args.depth_aware_loss_weight) > 0.0:
                depth_loss = float(args.depth_aware_loss_weight) * depth_aware_volume_loss(logits, y, pos_weight, args)
            aux_loss = torch.zeros((), device=device)
            if training and float(args.curve_aux_weight) > 0.0 and hasattr(model, "auxiliary_loss"):
                raw_aux, _aux_stats = model.auxiliary_loss(x)
                aux_loss = float(args.curve_aux_weight) * raw_aux
            loss = bce + dloss + float(args.projection_loss_weight) * ploss + depth_loss + aux_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        metrics = torch_volume_metrics(logits.detach(), y, args=args)
        batch = int(x.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["bce"] += float(bce.detach().cpu()) * batch
        totals["dice_loss"] += float(dloss.detach().cpu()) * batch
        totals["projection_loss"] += float(ploss.detach().cpu()) * batch
        totals["depth_loss"] += float(depth_loss.detach().cpu()) * batch
        for key, value in metrics.items():
            if key in totals:
                totals[key] += float(value) * batch
        count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def volume_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    return 1.0 - ((2.0 * intersection + eps) / (denom + eps)).mean()


def depth_aware_volume_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
    eps: float = 1e-6,
) -> torch.Tensor:
    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        logits_f = logits.float()
        targets_f = targets.float()
        depth_count = int(logits_f.shape[1])
        depth_weights = torch.ones(depth_count, dtype=logits_f.dtype, device=logits_f.device)
        group_weights = {
            "shallow": float(args.depth_loss_shallow_weight),
            "mid": float(args.depth_loss_mid_weight),
            "deep": float(args.depth_loss_deep_weight),
        }
        for name, start, end in depth_group_bounds(depth_count):
            depth_weights[start:end] = group_weights.get(name, 1.0)
        weight_map = depth_weights.view(1, depth_count, 1, 1)
        bce_map = F.binary_cross_entropy_with_logits(
            logits_f,
            targets_f,
            pos_weight=pos_weight.float(),
            reduction="none",
        )
        weighted_bce = torch.sum(bce_map * weight_map) / torch.clamp(torch.sum(torch.ones_like(bce_map) * weight_map), min=1.0)

        probs = torch.sigmoid(logits_f)
        dice_terms: list[torch.Tensor] = []
        dice_weights: list[float] = []
        for name, start, end in depth_group_bounds(depth_count):
            if end <= start:
                continue
            p = probs[:, start:end]
            t = targets_f[:, start:end]
            intersection = torch.sum(p * t, dim=(1, 2, 3))
            denom = torch.sum(p, dim=(1, 2, 3)) + torch.sum(t, dim=(1, 2, 3))
            dice_terms.append(1.0 - ((2.0 * intersection + eps) / (denom + eps)).mean())
            dice_weights.append(group_weights.get(name, 1.0))
        if dice_terms:
            dice_weight_tensor = torch.tensor(dice_weights, dtype=logits_f.dtype, device=logits_f.device)
            group_dice_loss = torch.sum(torch.stack(dice_terms) * dice_weight_tensor) / torch.clamp(dice_weight_tensor.sum(), min=eps)
        else:
            group_dice_loss = volume_dice_loss(logits_f, targets_f, eps=eps)

        deep_bounds = [bounds for bounds in depth_group_bounds(depth_count) if bounds[0] == "deep"]
        deep_fp_loss = torch.zeros((), dtype=logits_f.dtype, device=logits_f.device)
        if deep_bounds:
            _name, start, end = deep_bounds[0]
            deep_neg = 1.0 - targets_f[:, start:end]
            deep_fp_loss = torch.sum(probs[:, start:end] * deep_neg) / torch.clamp(deep_neg.sum(), min=1.0)
        return 0.55 * weighted_bce + 0.45 * group_dice_loss + float(args.deep_fp_loss_weight) * deep_fp_loss


def projection_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        probs = torch.sigmoid(logits.float())
        target_float = targets.float()
        pred_proj = 1.0 - torch.prod(torch.clamp(1.0 - probs, min=eps, max=1.0), dim=1)
        target_proj = target_float.amax(dim=1)
        bce = F.binary_cross_entropy(torch.clamp(pred_proj, eps, 1.0 - eps), target_proj)
        intersection = torch.sum(pred_proj * target_proj, dim=(1, 2))
        denom = torch.sum(pred_proj, dim=(1, 2)) + torch.sum(target_proj, dim=(1, 2))
        dice = 1.0 - ((2.0 * intersection + eps) / (denom + eps)).mean()
        return bce + dice


def torch_volume_metrics(logits: torch.Tensor, targets: torch.Tensor, *, args: argparse.Namespace, eps: float = 1e-6) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum((preds + targets) > 0, dim=dims).float()
    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)
    dice = ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean()
    proj_pred = preds.amax(dim=1)
    proj_target = targets.amax(dim=1)
    proj_intersection = torch.sum(proj_pred * proj_target, dim=(1, 2))
    proj_sum = torch.sum(proj_pred, dim=(1, 2)) + torch.sum(proj_target, dim=(1, 2))
    proj_dice = ((2.0 * proj_intersection + eps) / (proj_sum + eps)).mean()
    depth_metrics = torch_depth_metrics(preds, targets, args=args, eps=eps)
    return {
        "dice": float(dice.cpu()),
        "iou": float(((intersection + eps) / (union + eps)).mean().cpu()),
        "projection_dice": float(proj_dice.cpu()),
        **depth_metrics,
    }


def torch_depth_metrics(preds: torch.Tensor, targets: torch.Tensor, *, args: argparse.Namespace, eps: float = 1e-6) -> dict[str, float]:
    depth_count = int(preds.shape[1])
    values: dict[str, float] = {}
    for name, start, end in depth_group_bounds(depth_count):
        if end <= start:
            values[f"{name}_dice"] = 0.0
            continue
        p = preds[:, start:end].bool()
        t = targets[:, start:end].bool()
        tp = torch.logical_and(p, t).sum().float()
        fp = torch.logical_and(p, ~t).sum().float()
        fn = torch.logical_and(~p, t).sum().float()
        tn = torch.logical_and(~p, ~t).sum().float()
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        values[f"{name}_dice"] = float(dice.cpu())
        if name == "deep":
            deep_fp = fp / torch.clamp(fp + tn, min=1.0)
            values["deep_false_positive_rate"] = float(deep_fp.cpu())
    shallow = values.get("shallow_dice", 0.0)
    mid = values.get("mid_dice", 0.0)
    deep = values.get("deep_dice", 0.0)
    deep_fp = values.get("deep_false_positive_rate", 0.0)
    values["depth_utility"] = float(
        float(args.depth_utility_shallow_weight) * shallow
        + float(args.depth_utility_mid_weight) * mid
        + float(args.depth_utility_deep_weight) * deep
        - float(args.depth_utility_deep_fp_penalty) * deep_fp
    )
    return values


def checkpoint_score(stats: dict[str, float], args: argparse.Namespace) -> float:
    if args.checkpoint_metric == "depth_utility":
        return float(stats.get("depth_utility", float("-inf")))
    return float(stats.get("dice", float("-inf")))


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
    outputs: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        kwargs: dict[str, Any] = {}
        if context_ratio is not None and supports_context_mask(model):
            b, _channels, h, w = batch.shape
            kwargs["context_keep_mask"] = random_context_keep_mask(b, h * w, context_ratio, batch.device)
        logits = model(batch, **kwargs) if kwargs else model(batch)
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


@torch.no_grad()
def evaluate_world_volume_metrics(
    model: nn.Module,
    split: VolumeSplit,
    *,
    device: torch.device,
    batch_size: int,
    sparse_ratios: list[float],
    threshold: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if hasattr(model, "predict_curves") and hasattr(model, "split_input"):
        loader = DataLoader(torch.from_numpy(split.x.astype(np.float32)), batch_size=batch_size, shuffle=False)
        huber_sum = 0.0
        mse_sum = 0.0
        channel_sums = np.zeros(len(WRENCH_NAMES), dtype=np.float64)
        element_count = 0
        channel_count = 0
        model.eval()
        for batch in loader:
            batch = batch.to(device)
            curve_tokens, _action_tokens = model.split_input(batch)
            b, n, _channels = curve_tokens.shape
            target_mask = torch.rand((b, n), device=device) < float(args.world_target_ratio)
            target_mask = ensure_any_masked_and_context(target_mask)
            pred = model.predict_curves(batch, context_keep_mask=~target_mask)
            diff = pred[target_mask] - curve_tokens[target_mask]
            huber_sum += float(F.smooth_l1_loss(pred[target_mask], curve_tokens[target_mask], reduction="sum").cpu())
            mse_sum += float(diff.square().sum().cpu())
            steps = int(model.curve_channels) // len(WRENCH_NAMES)
            curve_diff = diff.reshape(-1, steps, len(WRENCH_NAMES))
            channel_sums += curve_diff.square().sum(dim=(0, 1)).cpu().numpy()
            channel_count += int(curve_diff.shape[0] * curve_diff.shape[1])
            element_count += int(diff.numel())
        denom = max(element_count, 1)
        rows.append(
            {
                "split": "test",
                "metric_type": "heldout_curve_prediction",
                "context_ratio": "",
                "world_target_ratio": float(args.world_target_ratio),
                "heldout_curve_huber": huber_sum / denom,
                "heldout_curve_mse": mse_sum / denom,
                **{f"{name}_mse": float(channel_sums[idx] / max(channel_count, 1)) for idx, name in enumerate(WRENCH_NAMES)},
                "sparse_voxel_dice": "",
                "sparse_projection_dice": "",
            }
        )
    for ratio in sparse_ratios:
        scores = predict_scores(model, split.x, device=device, batch_size=batch_size, context_ratio=ratio)
        result = evaluate_volume_scores(scores, split.volumes, fixed_threshold=float(threshold))
        fixed = result["fixed_threshold"]
        rows.append(
            {
                "split": "test",
                "metric_type": "sparse_context_volume",
                "context_ratio": float(ratio),
                "world_target_ratio": "",
                "heldout_curve_huber": "",
                "heldout_curve_mse": "",
                **{f"{name}_mse": "" for name in WRENCH_NAMES},
                "sparse_voxel_dice": fixed.get("dice", ""),
                "sparse_projection_dice": fixed.get("projection_dice", ""),
            }
        )
    return rows


def write_world_metrics(method_dir: Path, method: str, rows: list[dict[str, Any]]) -> None:
    ready_rows = [{"method": method, **row} for row in rows]
    _write_csv(method_dir / "world_metrics.csv", ready_rows, world_metric_fieldnames())
    _write_json(method_dir / "world_metrics.json", ready_rows)


def world_metric_fieldnames() -> list[str]:
    return [
        "method",
        "split",
        "metric_type",
        "context_ratio",
        "world_target_ratio",
        "heldout_curve_huber",
        "heldout_curve_mse",
        *[f"{name}_mse" for name in WRENCH_NAMES],
        "sparse_voxel_dice",
        "sparse_projection_dice",
    ]


def write_volume_metrics(
    method_dir: Path,
    method: str,
    scores: np.ndarray,
    volumes: np.ndarray,
    names: list[str],
    *,
    context: dict[str, Any],
    fixed_threshold: float,
    split: str,
) -> None:
    result = evaluate_volume_scores(scores, volumes, fixed_threshold=fixed_threshold)
    prefix = "" if split == "val" else "test_"
    np.save(method_dir / f"{prefix}scores.npy", scores.astype(np.float32))
    per_rows = []
    best_threshold = float(result["threshold_sweep_best"]["threshold"])
    fixed_threshold_value = float(result["fixed_threshold"]["threshold"])
    for idx, name in enumerate(names):
        fixed = metrics_from_counts(volume_counts(scores[idx] >= fixed_threshold_value, volumes[idx]), threshold=fixed_threshold_value)
        best = metrics_from_counts(volume_counts(scores[idx] >= best_threshold, volumes[idx]), threshold=best_threshold)
        proj_fixed = projection_metrics(scores[idx] >= fixed_threshold_value, volumes[idx])
        per_rows.append(
            {
                "sample": name,
                "fixed_dice": fixed["dice"],
                "fixed_iou": fixed["iou"],
                "best_dice": best["dice"],
                "best_iou": best["iou"],
                "projection_fixed_dice": proj_fixed["projection_dice"],
                "gt_positive_voxels": int(volumes[idx].sum()),
            }
        )
    _write_csv(method_dir / f"{prefix}metrics_per_sample.csv", per_rows, list(per_rows[0].keys()))
    depth_rows = depth_metric_rows(
        scores,
        volumes,
        fixed_threshold=fixed_threshold_value,
        best_threshold=best_threshold,
        split=split,
        method=method,
    )
    _write_csv(method_dir / f"{prefix}depth_metrics.csv", depth_rows, depth_metric_fieldnames())
    summary = {"method": method, "group": "wrench_3d_volume", **result, "num_samples": int(volumes.shape[0]), "config": context}
    summary["fixed_depth_groups"] = depth_group_metric_rows(scores >= fixed_threshold_value, volumes, threshold=fixed_threshold_value)
    summary["best_depth_groups"] = depth_group_metric_rows(scores >= best_threshold, volumes, threshold=best_threshold)
    summary["fixed_depth_summary"] = summarize_depth_groups(summary["fixed_depth_groups"])
    summary["best_depth_summary"] = summarize_depth_groups(summary["best_depth_groups"])
    _write_json(method_dir / f"{prefix}metrics_summary.json", summary)
    if split == "val":
        _write_json(method_dir / "threshold_sweep.json", {"thresholds": result["threshold_rows"], "best": result["threshold_sweep_best"]})
        _write_json(method_dir / "run_config.json", context)


def evaluate_volume_scores(scores: np.ndarray, volumes: np.ndarray, *, fixed_threshold: float) -> dict[str, Any]:
    thresholds = probability_thresholds()
    threshold_rows = [metrics_from_counts(volume_counts(scores >= threshold, volumes), threshold=float(threshold)) for threshold in thresholds]
    fixed = metrics_from_counts(volume_counts(scores >= fixed_threshold, volumes), threshold=float(fixed_threshold))
    best = max(threshold_rows, key=lambda row: float(row["dice"]))
    fixed_proj = projection_metrics(scores >= fixed_threshold, volumes)
    best_proj = projection_metrics(scores >= float(best["threshold"]), volumes)
    return {
        "fixed_threshold": {**fixed, **fixed_proj},
        "threshold_sweep_best": {**best, **best_proj},
        "threshold_rows": threshold_rows,
    }


def depth_metric_rows(
    scores: np.ndarray,
    volumes: np.ndarray,
    *,
    fixed_threshold: float,
    best_threshold: float,
    split: str,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold_kind, threshold in (("fixed", fixed_threshold), ("best", best_threshold)):
        pred = scores >= float(threshold)
        for row in depth_group_metric_rows(pred, volumes, threshold=float(threshold)):
            rows.append({"method": method, "split": split, "threshold_kind": threshold_kind, **row})
        for row in depth_slice_metric_rows(pred, volumes, threshold=float(threshold)):
            rows.append({"method": method, "split": split, "threshold_kind": threshold_kind, **row})
    return rows


def depth_group_metric_rows(pred: np.ndarray, target: np.ndarray, *, threshold: float) -> list[dict[str, Any]]:
    pred_bool = np.asarray(pred, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    depth_count = depth_axis_size(pred_bool)
    rows: list[dict[str, Any]] = []
    for name, start, end in depth_group_bounds(depth_count):
        if end <= start:
            continue
        counts = volume_counts(depth_slice(pred_bool, start, end), depth_slice(target_bool, start, end))
        rows.append(
            {
                "depth_scope": "group",
                "depth_name": name,
                "depth_index_start": int(start),
                "depth_index_end": int(end),
                "depth_fraction_start": float(start / max(depth_count, 1)),
                "depth_fraction_end": float(end / max(depth_count, 1)),
                **metrics_from_counts(counts, threshold=threshold),
            }
        )
    return rows


def depth_slice_metric_rows(pred: np.ndarray, target: np.ndarray, *, threshold: float) -> list[dict[str, Any]]:
    pred_bool = np.asarray(pred, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    depth_count = depth_axis_size(pred_bool)
    rows: list[dict[str, Any]] = []
    for index in range(depth_count):
        counts = volume_counts(depth_slice(pred_bool, index, index + 1), depth_slice(target_bool, index, index + 1))
        rows.append(
            {
                "depth_scope": "slice",
                "depth_name": f"z{index:02d}",
                "depth_index_start": int(index),
                "depth_index_end": int(index + 1),
                "depth_fraction_start": float(index / max(depth_count, 1)),
                "depth_fraction_end": float((index + 1) / max(depth_count, 1)),
                **metrics_from_counts(counts, threshold=threshold),
            }
        )
    return rows


def summarize_depth_groups(rows: list[dict[str, Any]]) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {}
    metric_names = (
        "dice",
        "iou",
        "precision",
        "recall",
        "predicted_positive_rate",
        "target_positive_rate",
        "false_positive_rate",
        "overprediction_rate",
        "fp_per_gt_positive",
        "pred_to_gt_positive_ratio",
    )
    for row in rows:
        name = str(row.get("depth_name", ""))
        if not name:
            continue
        for metric_name in metric_names:
            summary[f"{name}_{metric_name}"] = row.get(metric_name, "")
    shallow = safe_float(summary.get("shallow_dice", ""))
    deep_fp = safe_float(summary.get("deep_false_positive_rate", ""))
    if np.isfinite(shallow) and np.isfinite(deep_fp):
        summary["shallow_dice_minus_deep_fp_rate"] = float(shallow - deep_fp)
    return summary


def depth_metric_fieldnames() -> list[str]:
    return [
        "method",
        "split",
        "threshold_kind",
        "depth_scope",
        "depth_name",
        "depth_index_start",
        "depth_index_end",
        "depth_fraction_start",
        "depth_fraction_end",
        "threshold",
        "voxel_accuracy",
        "precision",
        "recall",
        "dice",
        "iou",
        "predicted_positive_rate",
        "target_positive_rate",
        "false_positive_rate",
        "false_negative_rate",
        "overprediction_rate",
        "underprediction_rate",
        "fp_per_gt_positive",
        "pred_to_gt_positive_ratio",
        "tp",
        "tn",
        "fp",
        "fn",
    ]


def depth_axis_size(array: np.ndarray) -> int:
    if array.ndim == 4:
        return int(array.shape[1])
    if array.ndim == 3:
        return int(array.shape[0])
    raise ValueError(f"Expected volume array with shape [N,D,H,W] or [D,H,W], got {array.shape}")


def depth_slice(array: np.ndarray, start: int, end: int) -> np.ndarray:
    if array.ndim == 4:
        return array[:, start:end]
    if array.ndim == 3:
        return array[start:end]
    raise ValueError(f"Expected volume array with shape [N,D,H,W] or [D,H,W], got {array.shape}")


def depth_group_bounds(depth_count: int) -> list[tuple[str, int, int]]:
    if depth_count <= 0:
        return []
    first = max(1, int(round(depth_count / 3.0)))
    second = max(first + 1, int(round(2.0 * depth_count / 3.0))) if depth_count >= 3 else depth_count
    second = min(second, depth_count)
    return [
        ("shallow", 0, min(first, depth_count)),
        ("mid", min(first, depth_count), min(second, depth_count)),
        ("deep", min(second, depth_count), depth_count),
    ]


def volume_counts(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_bool = np.asarray(pred, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(pred_bool, target_bool).sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~target_bool).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: dict[str, int], *, threshold: float) -> dict[str, float | int]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    total = max(tp + fp + fn + tn, 1)
    target_positive = tp + fn
    pred_positive = tp + fp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / total
    predicted_positive_rate = pred_positive / total
    target_positive_rate = target_positive / total
    false_positive_rate = fp / max(fp + tn, 1)
    false_negative_rate = fn / max(fn + tp, 1)
    return {
        "threshold": float(threshold),
        "voxel_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "predicted_positive_rate": float(predicted_positive_rate),
        "target_positive_rate": float(target_positive_rate),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "overprediction_rate": float(max(predicted_positive_rate - target_positive_rate, 0.0)),
        "underprediction_rate": float(max(target_positive_rate - predicted_positive_rate, 0.0)),
        "fp_per_gt_positive": float(fp / max(target_positive, 1)),
        "pred_to_gt_positive_ratio": float(pred_positive / max(target_positive, 1)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def projection_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_proj = np.asarray(pred, dtype=bool).max(axis=1 if pred.ndim == 4 else 0)
    target_proj = np.asarray(target, dtype=bool).max(axis=1 if target.ndim == 4 else 0)
    counts = volume_counts(pred_proj, target_proj)
    dice = 2 * counts["tp"] / max(2 * counts["tp"] + counts["fp"] + counts["fn"], 1)
    iou = counts["tp"] / max(counts["tp"] + counts["fp"] + counts["fn"], 1)
    return {"projection_dice": float(dice), "projection_iou": float(iou)}


def probability_thresholds() -> np.ndarray:
    return np.asarray([round(value, 6) for value in np.arange(0.05, 0.951, 0.05)], dtype=np.float32)


def write_leaderboard(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out_dir.glob("*/metrics_summary.json")):
        summary = _read_json(metrics_path)
        test_summary = _read_json(metrics_path.parent / "test_metrics_summary.json")
        config = summary.get("config", {})
        fixed = summary.get("fixed_threshold", {})
        best = summary.get("threshold_sweep_best", {})
        test_fixed = test_summary.get("fixed_threshold", {})
        test_best = test_summary.get("threshold_sweep_best", {})
        rows.append(
            {
                "method": summary.get("method", metrics_path.parent.name),
                "representation": config.get("representation", ""),
                "volume_shape_dhw": "x".join(str(v) for v in config.get("volume_shape_dhw", [])),
                "val_fixed_dice": fixed.get("dice", ""),
                "val_best_dice": best.get("dice", ""),
                "val_best_threshold": best.get("threshold", ""),
                "val_projection_dice": best.get("projection_dice", ""),
                "test_val_selected_dice": test_fixed.get("dice", ""),
                "test_val_selected_projection_dice": test_fixed.get("projection_dice", ""),
                "test_oracle_dice": test_best.get("dice", ""),
                **depth_leaderboard_values(summary, test_summary),
                "path": str(metrics_path.parent),
            }
        )
    rows.sort(key=lambda row: safe_float(row.get("test_val_selected_dice")), reverse=True)
    if rows:
        _write_csv(out_dir / "leaderboard.csv", rows, list(rows[0].keys()))
    world_rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*/world_metrics.json")):
        world_rows.extend(_read_json(path))
    if world_rows:
        _write_csv(out_dir / "world_model_metrics.csv", world_rows, world_metric_fieldnames())


def parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one variant is required")
    unknown = [item for item in variants if item not in ALLOWED_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; allowed: {ALLOWED_VARIANTS}")
    return variants


def parse_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return values


def parse_volume_shape(raw: str) -> tuple[int, int, int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--volume-shape must be D,H,W")
    if any(value <= 0 for value in values):
        raise ValueError("--volume-shape values must be positive")
    return int(values[0]), int(values[1]), int(values[2])


def prefix_keys(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def depth_leaderboard_values(summary: dict[str, Any], test_summary: dict[str, Any]) -> dict[str, Any]:
    val_depth = summary.get("best_depth_summary", {})
    test_depth = test_summary.get("fixed_depth_summary", {})
    return {
        "val_best_shallow_dice": val_depth.get("shallow_dice", ""),
        "val_best_mid_dice": val_depth.get("mid_dice", ""),
        "val_best_deep_dice": val_depth.get("deep_dice", ""),
        "val_best_deep_pred_positive_rate": val_depth.get("deep_predicted_positive_rate", ""),
        "val_best_deep_false_positive_rate": val_depth.get("deep_false_positive_rate", ""),
        "val_best_deep_fp_per_gt_positive": val_depth.get("deep_fp_per_gt_positive", ""),
        "test_shallow_dice": test_depth.get("shallow_dice", ""),
        "test_mid_dice": test_depth.get("mid_dice", ""),
        "test_deep_dice": test_depth.get("deep_dice", ""),
        "test_deep_pred_positive_rate": test_depth.get("deep_predicted_positive_rate", ""),
        "test_deep_false_positive_rate": test_depth.get("deep_false_positive_rate", ""),
        "test_deep_fp_per_gt_positive": test_depth.get("deep_fp_per_gt_positive", ""),
    }


def supports_context_mask(model: nn.Module) -> bool:
    return isinstance(model, TemporalCNNWorldVolumeNet)


def random_context_keep_mask(batch: int, tokens: int, ratio: float, device: torch.device) -> torch.Tensor:
    keep = torch.rand((batch, tokens), device=device) < float(ratio)
    return ~ensure_any_masked_and_context(~keep)


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


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def use_amp(args: argparse.Namespace, device: torch.device) -> bool:
    return bool(not args.no_amp and device.type == "cuda")


def safe_float(value: Any) -> float:
    try:
        if value == "":
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


if __name__ == "__main__":
    main()
