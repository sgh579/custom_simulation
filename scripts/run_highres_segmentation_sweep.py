from __future__ import annotations

import argparse
import csv
import hashlib
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

from palpation_sim.config import PhantomConfig, ScanConfig
from palpation_sim.native_data import load_phantom_scan_material_lumps
from palpation_sim.phantom import LumpSpec, lumps_membership
from palpation_sim.workflow import require_runtime_environment
from run_segmentation_accuracy_sweep import (
    FEATURE_NAMES,
    ShallowCNN,
    ValidationUNet,
    _load_displacement_hwt,
    _load_fz_hwt,
    _load_nonlinearity_map,
    _load_sklearn,
    _method_done,
    _seed_everything,
    counts_from_prediction,
    evaluate_scores,
    extract_mechanical_features,
    metrics_from_counts,
    normalize_maps,
    predict_neural,
    train_neural_method,
    write_score_method,
)


DEFAULT_PACKAGE_DIR = Path("runs/dataset_packages/palpation_random_shapes_20x_800train_80val_80test_20260616")
DEFAULT_OUT_DIR = Path("runs/highres_segmentation_sweep_128_800_80_80")
DEFAULT_RESOLUTIONS = (128,)
DEFAULT_INPUTS = ("stiffness", "fz")
ALLOWED_INPUTS = (
    "stiffness",
    "stiffness_random_pair",
    "fz",
    "fz_limited_resampled",
    "press_limited_resampled",
    "fz_limited_sincos",
)
DEFAULT_MODELS = ("unet", "mlp", "shallow_cnn")
ALLOWED_MODELS = ("unet", "mlp", "shallow_cnn", "kmeans")
STIFFNESS_INPUT_POLICY = (
    "Equivalent-stiffness maps are kept on the native 20x20 scan grid with no preprocessing resize, "
    "smoothing, morphology, or label-aware processing. Any spatial resizing needed for a high-resolution "
    "target is handled inside the model path."
)
STIFFNESS_RANDOM_PAIR_POLICY = (
    "Random-pair stiffness maps use each press curve's first sample and one reproducible random later "
    "sample from index 1..T-1 per scan point; stiffness is max((Fj-F0)/(Zj-Z0), 0)."
)
LIMITED_TRAJECTORY_POLICY = (
    "Limited-trajectory raw-curve inputs choose one reproducible random endpoint from index 1..T-1 "
    "per scan point, then linearly resample the prefix 0..endpoint to the requested input length. "
    "Displacement and Fz are resampled on the same fractional source positions."
)
MODEL_RESIZE_POLICY = (
    "When a model emits logits at a different spatial size from the requested target, UpsampleLogits "
    "resizes logits inside the PyTorch model with torch.nn.functional.interpolate(mode='bilinear', "
    "align_corners=False)."
)


@dataclass
class SplitData:
    names: list[str]
    fz: np.ndarray
    z: np.ndarray
    masks: np.ndarray
    features: np.ndarray
    random_pair_stiffness: np.ndarray


class UpsampleLogits(nn.Module):
    def __init__(self, base: nn.Module, output_shape: tuple[int, int]) -> None:
        super().__init__()
        self.base = base
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        if logits.shape[-2:] != self.output_shape:
            logits = F.interpolate(logits, size=self.output_shape, mode="bilinear", align_corners=False)
        return logits


class GlobalSegmentationMLP(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int],
        hidden_dims: Sequence[int],
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = int(np.prod(input_shape))
        output_dim = int(output_shape[0] * output_shape[1])
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, int(hidden)), nn.LayerNorm(int(hidden)), nn.GELU(), nn.Dropout(dropout)])
            previous = int(hidden)
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x.reshape(x.shape[0], -1))
        return logits.reshape(x.shape[0], 1, *self.output_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 20x20-input models against analytic high-resolution labels.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resolutions", type=str, default=",".join(str(value) for value in DEFAULT_RESOLUTIONS))
    parser.add_argument(
        "--inputs",
        type=str,
        default=",".join(DEFAULT_INPUTS),
        help="Comma list: stiffness,stiffness_random_pair,fz,fz_limited_resampled,press_limited_resampled,fz_limited_sincos",
    )
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="Comma list: unet,mlp,shallow_cnn,kmeans")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stiffness-random-seed", type=int, default=None)
    parser.add_argument("--limited-trajectory-seed", type=int, default=None)
    parser.add_argument("--trajectory-input-steps", type=int, default=0, help="0 keeps the native number of trajectory samples.")
    parser.add_argument("--positional-embedding-dim", type=int, default=8, help="Even sinusoidal displacement embedding dimension.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fz-normalize", choices=["none", "sample", "dataset"], default="dataset")
    parser.add_argument("--stiffness-normalize", choices=["none", "sample", "dataset"], default="dataset")
    parser.add_argument("--mlp-hidden-dims", type=str, default="1024,512")
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--no-test", action="store_true", help="Skip held-out test evaluation.")
    parser.add_argument("--smoke", action="store_true", help="Use tiny splits and short training for validation.")
    args = parser.parse_args()

    require_runtime_environment()
    _seed_everything(args.seed)
    random.seed(args.seed)
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)

    resolutions = _parse_int_list(args.resolutions)
    inputs = _parse_choice_list(args.inputs, ALLOWED_INPUTS, "inputs")
    models = _parse_choice_list(args.models, ALLOWED_MODELS, "models")
    hidden_dims = _parse_int_list(args.mlp_hidden_dims)
    stiffness_random_seed = int(args.seed if args.stiffness_random_seed is None else args.stiffness_random_seed)
    limited_trajectory_seed = int(args.seed if args.limited_trajectory_seed is None else args.limited_trajectory_seed)
    data_root = args.package_dir / "data"
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "package_dir": str(args.package_dir),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "test_dir": str(test_dir),
        "resolutions": resolutions,
        "inputs": inputs,
        "models": models,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "fz_normalize": args.fz_normalize,
        "stiffness_normalize": args.stiffness_normalize,
        "label_policy": "analytic lump xy projection on scan-area grid centers",
        "stiffness_input_policy": STIFFNESS_INPUT_POLICY,
        "stiffness_random_pair_policy": STIFFNESS_RANDOM_PAIR_POLICY,
        "stiffness_random_seed": stiffness_random_seed,
        "limited_trajectory_policy": LIMITED_TRAJECTORY_POLICY,
        "limited_trajectory_seed": limited_trajectory_seed,
        "trajectory_input_steps": int(args.trajectory_input_steps),
        "positional_embedding_dim": int(args.positional_embedding_dim),
        "model_resize_policy": MODEL_RESIZE_POLICY,
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
    }
    _write_json(args.out_dir / "run_config.json", run_config)

    for resolution in resolutions:
        max_train = 8 if args.smoke else None
        max_eval = 4 if args.smoke else None
        print(f"loading r{resolution} train split: {train_dir}")
        train = load_split(train_dir, label_size=resolution, max_samples=max_train, random_pair_seed=stiffness_random_seed)
        print(f"loading r{resolution} val split: {val_dir}")
        val = load_split(val_dir, label_size=resolution, max_samples=max_eval, random_pair_seed=stiffness_random_seed)
        test = None
        if not args.no_test:
            print(f"loading r{resolution} test split: {test_dir}")
            test = load_split(test_dir, label_size=resolution, max_samples=max_eval, random_pair_seed=stiffness_random_seed)

        x_by_input = build_inputs(train, val, test, args, limited_trajectory_seed=limited_trajectory_seed)
        for input_name in inputs:
            x_train, x_val, x_test = x_by_input[input_name]
            for model_name in models:
                method = f"r{resolution}_{input_name}_{model_name}"
                context = {
                    **run_config,
                    "resolution": int(resolution),
                    "output_shape": [int(resolution), int(resolution)],
                    "input": input_name,
                    "model": model_name,
                    "train_samples": len(train.names),
                    "val_samples": len(val.names),
                    "test_samples": len(test.names) if test is not None else 0,
                    "input_shape_chw": list(x_train.shape[1:]),
                    "target_shape_hw": [int(resolution), int(resolution)],
                    "feature_names": FEATURE_NAMES,
                    "stiffness_input_shape_hw": list(train.features.shape[-2:]),
                }
                if model_name == "kmeans":
                    run_kmeans_method(
                        method,
                        x_train,
                        train.masks,
                        x_val,
                        val.masks,
                        val.names,
                        x_test,
                        test.masks if test is not None else None,
                        test.names if test is not None else None,
                        args.out_dir,
                        context,
                        args,
                    )
                    write_leaderboard(args.out_dir)
                    continue
                model = build_model(model_name, input_shape=tuple(x_train.shape[1:]), output_shape=(resolution, resolution), args=args, hidden_dims=hidden_dims)
                run_method(
                    method,
                    model,
                    x_train,
                    train.masks,
                    x_val,
                    val.masks,
                    val.names,
                    x_test,
                    test.masks if test is not None else None,
                    test.names if test is not None else None,
                    args.out_dir,
                    context,
                    args,
                )
                write_leaderboard(args.out_dir)

    write_leaderboard(args.out_dir)
    print(f"high-res sweep complete: {args.out_dir}")


def load_split(
    split_dir: Path,
    *,
    label_size: int,
    max_samples: int | None = None,
    random_pair_seed: int = 7,
) -> SplitData:
    files = sorted(split_dir.glob("*.npz"))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")

    names: list[str] = []
    fz_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []
    feature_list: list[np.ndarray] = []
    random_pair_stiffness_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    for path in files:
        with np.load(path, allow_pickle=False) as sample:
            fz_hwt = _load_fz_hwt(sample)
            z_hwt = _load_displacement_hwt(sample, fz_hwt.shape)
            nonlinearity = _load_nonlinearity_map(sample, fz_hwt.shape[:2])
        features = extract_mechanical_features(z_hwt, fz_hwt, nonlinearity)
        sample_seed = stable_sample_seed(random_pair_seed, split_dir.name, path.name)
        random_pair_stiffness = random_pair_stiffness_map(z_hwt, fz_hwt, seed=sample_seed)
        metadata_path = path.with_name(f"{path.stem}_gt.json")
        phantom, scan, _material, lumps, _metadata = load_phantom_scan_material_lumps(sample_path=path, metadata_path=metadata_path)
        mask = highres_mask_for_scan_area(scan, phantom, lumps, label_size)
        names.append(path.name)
        fz_list.append(np.moveaxis(fz_hwt, -1, 0).astype(np.float32))
        z_list.append(np.moveaxis(z_hwt, -1, 0).astype(np.float32))
        feature_list.append(features)
        random_pair_stiffness_list.append(random_pair_stiffness.astype(np.float32))
        mask_list.append(mask.astype(np.uint8))

    return SplitData(
        names=names,
        fz=np.stack(fz_list).astype(np.float32),
        z=np.stack(z_list).astype(np.float32),
        masks=np.stack(mask_list).astype(np.uint8),
        features=np.stack(feature_list).astype(np.float32),
        random_pair_stiffness=np.stack(random_pair_stiffness_list).astype(np.float32),
    )


def stable_sample_seed(base_seed: int, split_name: str, sample_name: str) -> int:
    payload = f"{int(base_seed)}|{split_name}|{sample_name}".encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def random_pair_stiffness_map(z_raw: np.ndarray, f_raw: np.ndarray, *, seed: int) -> np.ndarray:
    z, f = make_compression_positive_hwt(z_raw, f_raw)
    h, w, t = z.shape
    if t < 2:
        return np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(seed)
    endpoint_idx = rng.integers(1, t, size=(h, w), endpoint=False)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    z0 = z[..., 0]
    f0 = f[..., 0]
    z1 = z[rows, cols, endpoint_idx]
    f1 = f[rows, cols, endpoint_idx]
    dz = z1 - z0
    df = f1 - f0
    stiffness = np.divide(df, dz, out=np.zeros_like(df, dtype=np.float32), where=np.abs(dz) >= 1e-9)
    stiffness = np.maximum(stiffness, 0.0)
    return np.nan_to_num(stiffness, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def make_compression_positive_hwt(z_raw: np.ndarray, f_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = z_raw.astype(np.float32) - z_raw[..., :1].astype(np.float32)
    f = f_raw.astype(np.float32) - f_raw[..., :1].astype(np.float32)
    z = np.where(np.abs(np.nanmin(z, axis=-1, keepdims=True)) > np.abs(np.nanmax(z, axis=-1, keepdims=True)), -z, z)
    f = np.where(np.abs(np.nanmin(f, axis=-1, keepdims=True)) > np.abs(np.nanmax(f, axis=-1, keepdims=True)), -f, f)
    return z.astype(np.float32), f.astype(np.float32)


def highres_mask_for_scan_area(
    scan: ScanConfig,
    phantom: PhantomConfig,
    lumps: Sequence[LumpSpec],
    label_size: int,
) -> np.ndarray:
    x_values = np.asarray(scan.x_values(phantom), dtype=np.float32)
    y_values = np.asarray(scan.y_values(phantom), dtype=np.float32)
    xs = np.linspace(float(x_values[0]), float(x_values[-1]), int(label_size), dtype=np.float32)
    ys = np.linspace(float(y_values[0]), float(y_values[-1]), int(label_size), dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    points = np.stack([xv, yv, np.zeros_like(xv)], axis=-1)
    return lumps_membership(points, lumps, project_xy=True).astype(np.uint8)


def build_inputs(
    train: SplitData,
    val: SplitData,
    test: SplitData | None,
    args: argparse.Namespace,
    *,
    limited_trajectory_seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    train_stiffness = train.features[:, FEATURE_NAMES.index("equivalent_stiffness")][:, None]
    val_stiffness = val.features[:, FEATURE_NAMES.index("equivalent_stiffness")][:, None]
    test_stiffness = test.features[:, FEATURE_NAMES.index("equivalent_stiffness")][:, None] if test is not None else None
    train_random_pair = train.random_pair_stiffness[:, None]
    val_random_pair = val.random_pair_stiffness[:, None]
    test_random_pair = test.random_pair_stiffness[:, None] if test is not None else None
    stiffness_train, stiffness_val, stiffness_test = normalize_train_val_test(
        train_stiffness,
        val_stiffness,
        test_stiffness,
        mode=args.stiffness_normalize,
    )
    random_pair_train, random_pair_val, random_pair_test = normalize_train_val_test(
        train_random_pair,
        val_random_pair,
        test_random_pair,
        mode=args.stiffness_normalize,
    )
    fz_train, fz_val, fz_test = normalize_train_val_test(train.fz, val.fz, test.fz if test is not None else None, mode=args.fz_normalize)
    input_steps = int(args.trajectory_input_steps) if int(args.trajectory_input_steps) > 0 else int(train.fz.shape[1])
    limited_train_fz, limited_train_press = limited_trajectory_inputs(train, split_name="train", input_steps=input_steps, seed=limited_trajectory_seed)
    limited_val_fz, limited_val_press = limited_trajectory_inputs(val, split_name="val", input_steps=input_steps, seed=limited_trajectory_seed)
    limited_test_fz = None
    limited_test_press = None
    if test is not None:
        limited_test_fz, limited_test_press = limited_trajectory_inputs(test, split_name="test", input_steps=input_steps, seed=limited_trajectory_seed)
    position_scale = displacement_position_scale(train.z)
    limited_sincos_train = limited_trajectory_sincos_inputs(
        train,
        split_name="train",
        input_steps=input_steps,
        seed=limited_trajectory_seed,
        embedding_dim=int(args.positional_embedding_dim),
        position_scale=position_scale,
    )
    limited_sincos_val = limited_trajectory_sincos_inputs(
        val,
        split_name="val",
        input_steps=input_steps,
        seed=limited_trajectory_seed,
        embedding_dim=int(args.positional_embedding_dim),
        position_scale=position_scale,
    )
    limited_sincos_test = None
    if test is not None:
        limited_sincos_test = limited_trajectory_sincos_inputs(
            test,
            split_name="test",
            input_steps=input_steps,
            seed=limited_trajectory_seed,
            embedding_dim=int(args.positional_embedding_dim),
            position_scale=position_scale,
        )
    limited_fz_train, limited_fz_val, limited_fz_test = normalize_train_val_test(
        limited_train_fz,
        limited_val_fz,
        limited_test_fz,
        mode=args.fz_normalize,
    )
    limited_press_train, limited_press_val, limited_press_test = normalize_train_val_test(
        limited_train_press,
        limited_val_press,
        limited_test_press,
        mode=args.fz_normalize,
    )
    limited_sincos_train, limited_sincos_val, limited_sincos_test = normalize_temporal_feature_fz_channels(
        limited_sincos_train,
        limited_sincos_val,
        limited_sincos_test,
        temporal_feature_size=1 + int(args.positional_embedding_dim),
        mode=args.fz_normalize,
    )
    return {
        "stiffness": (stiffness_train, stiffness_val, stiffness_test),
        "stiffness_random_pair": (random_pair_train, random_pair_val, random_pair_test),
        "fz": (fz_train, fz_val, fz_test),
        "fz_limited_resampled": (limited_fz_train, limited_fz_val, limited_fz_test),
        "press_limited_resampled": (limited_press_train, limited_press_val, limited_press_test),
        "fz_limited_sincos": (limited_sincos_train, limited_sincos_val, limited_sincos_test),
    }


def limited_trajectory_inputs(split: SplitData, *, split_name: str, input_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    fz_items: list[np.ndarray] = []
    press_items: list[np.ndarray] = []
    for idx, sample_name in enumerate(split.names):
        sample_seed = stable_sample_seed(seed, split_name, sample_name)
        z_resampled, fz_resampled = limited_resample_press(split.z[idx], split.fz[idx], input_steps=input_steps, seed=sample_seed)
        fz_items.append(fz_resampled)
        press_items.append(interleave_press_channels(z_resampled, fz_resampled))
    return np.stack(fz_items).astype(np.float32), np.stack(press_items).astype(np.float32)


def limited_resample_press(z_cthw: np.ndarray, fz_cthw: np.ndarray, *, input_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z_cthw, dtype=np.float32)
    fz = np.asarray(fz_cthw, dtype=np.float32)
    if z.shape != fz.shape or z.ndim != 3:
        raise ValueError(f"Expected z/fz shapes [T,H,W] to match, got {z.shape} and {fz.shape}")
    source_steps, h, w = z.shape
    if source_steps < 2:
        return np.repeat(z[:1], input_steps, axis=0), np.repeat(fz[:1], input_steps, axis=0)
    rng = np.random.default_rng(seed)
    endpoints = rng.integers(1, source_steps, size=(h, w), endpoint=False).astype(np.float32)
    fractions = np.linspace(0.0, 1.0, int(input_steps), dtype=np.float32)
    source_positions = endpoints[None, :, :] * fractions[:, None, None]
    lo = np.floor(source_positions).astype(np.int64)
    hi = np.minimum(lo + 1, source_steps - 1)
    alpha = (source_positions - lo.astype(np.float32)).astype(np.float32)
    rows = np.arange(h)[None, :, None]
    cols = np.arange(w)[None, None, :]
    z_lo = z[lo, rows, cols]
    z_hi = z[hi, rows, cols]
    f_lo = fz[lo, rows, cols]
    f_hi = fz[hi, rows, cols]
    z_out = z_lo + alpha * (z_hi - z_lo)
    f_out = f_lo + alpha * (f_hi - f_lo)
    return (
        np.nan_to_num(z_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num(f_out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    )


def interleave_press_channels(z: np.ndarray, fz: np.ndarray) -> np.ndarray:
    stacked = np.stack([z, fz], axis=1)
    return stacked.reshape(z.shape[0] * 2, z.shape[1], z.shape[2]).astype(np.float32)


def limited_trajectory_sincos_inputs(
    split: SplitData,
    *,
    split_name: str,
    input_steps: int,
    seed: int,
    embedding_dim: int,
    position_scale: float,
) -> np.ndarray:
    items: list[np.ndarray] = []
    for idx, sample_name in enumerate(split.names):
        sample_seed = stable_sample_seed(seed, split_name, sample_name)
        z_resampled, fz_resampled = limited_resample_press(split.z[idx], split.fz[idx], input_steps=input_steps, seed=sample_seed)
        items.append(interleave_fz_sincos_channels(z_resampled, fz_resampled, embedding_dim=embedding_dim, position_scale=position_scale))
    return np.stack(items).astype(np.float32)


def interleave_fz_sincos_channels(
    z: np.ndarray,
    fz: np.ndarray,
    *,
    embedding_dim: int,
    position_scale: float,
) -> np.ndarray:
    embedding = sinusoidal_displacement_embedding(z, embedding_dim=embedding_dim, position_scale=position_scale)
    stacked = np.concatenate([fz[:, None], embedding], axis=1)
    return stacked.reshape(z.shape[0] * (1 + int(embedding_dim)), z.shape[1], z.shape[2]).astype(np.float32)


def sinusoidal_displacement_embedding(z: np.ndarray, *, embedding_dim: int, position_scale: float) -> np.ndarray:
    if embedding_dim <= 0 or embedding_dim % 2 != 0:
        raise ValueError("embedding_dim must be a positive even integer")
    z = np.asarray(z, dtype=np.float32)
    scale = max(float(position_scale), 1e-12)
    positions = z / np.float32(scale)
    num_bands = embedding_dim // 2
    div_term = np.exp(np.arange(0, embedding_dim, 2, dtype=np.float32) * (-np.log(10000.0) / float(embedding_dim)))
    angles = positions[:, None] * div_term.reshape(1, num_bands, 1, 1)
    out = np.empty((z.shape[0], embedding_dim, z.shape[1], z.shape[2]), dtype=np.float32)
    out[:, 0::2] = np.sin(angles)
    out[:, 1::2] = np.cos(angles)
    return out


def displacement_position_scale(z: np.ndarray) -> float:
    value = float(np.nanmax(np.abs(np.asarray(z, dtype=np.float32))))
    return value if np.isfinite(value) and value > 1e-12 else 1.0


def normalize_temporal_feature_fz_channels(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray | None,
    *,
    temporal_feature_size: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    feature_size = int(temporal_feature_size)
    if feature_size <= 0 or train_x.shape[1] % feature_size != 0:
        raise ValueError(f"Invalid temporal_feature_size={feature_size} for input shape {train_x.shape}")
    train_out = train_x.astype(np.float32, copy=True)
    val_out = val_x.astype(np.float32, copy=True)
    test_out = test_x.astype(np.float32, copy=True) if test_x is not None else None
    fz_idx = np.arange(0, train_x.shape[1], feature_size)
    fz_train, fz_val, fz_test = normalize_train_val_test(
        train_x[:, fz_idx],
        val_x[:, fz_idx],
        test_x[:, fz_idx] if test_x is not None else None,
        mode=mode,
    )
    train_out[:, fz_idx] = fz_train
    val_out[:, fz_idx] = fz_val
    if test_out is not None and fz_test is not None:
        test_out[:, fz_idx] = fz_test
    return train_out, val_out, test_out


def normalize_train_val_test(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray | None,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    train_norm, val_norm = normalize_maps(train_x, val_x, mode=mode)
    if test_x is None:
        return train_norm, val_norm, None
    if mode == "none":
        test_norm = np.nan_to_num(test_x.astype(np.float32))
    elif mode == "sample":
        _dummy_train, test_norm = normalize_maps(train_x[:1], test_x, mode="sample")
    elif mode == "dataset":
        mean = np.nanmean(train_x.astype(np.float32), axis=(0, 2, 3), keepdims=True)
        std = np.nanstd(train_x.astype(np.float32), axis=(0, 2, 3), keepdims=True)
        test_norm = (test_x.astype(np.float32) - mean) / (std + np.float32(1e-6))
        test_norm = np.nan_to_num(test_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")
    return train_norm, val_norm, test_norm


def build_model(
    model_name: str,
    *,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int],
    args: argparse.Namespace,
    hidden_dims: Sequence[int],
) -> nn.Module:
    if model_name == "unet":
        return UpsampleLogits(ValidationUNet(input_shape[0], base_channels=24), output_shape)
    if model_name == "shallow_cnn":
        return UpsampleLogits(ShallowCNN(input_shape[0], base_channels=32), output_shape)
    if model_name == "mlp":
        return GlobalSegmentationMLP(input_shape, output_shape, hidden_dims, dropout=float(args.mlp_dropout))
    raise ValueError(f"Unknown model: {model_name}")


def run_method(
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
) -> None:
    method_dir = out_dir / method
    already_done = _method_done(method_dir, args.force)
    if not already_done:
        started = time.perf_counter()
        train_neural_method(
            method,
            "highres",
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            val_names,
            out_dir,
            context={**context, "started_at_unix": started},
            args=args,
            augment=False,
            focal=False,
        )
    else:
        print(f"{method}: existing metrics found, loading checkpoint")
        checkpoint = torch.load(method_dir / "best.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])

    if x_test is None or y_test is None or test_names is None:
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = model.to(device)
    test_scores = predict_neural(model, x_test, device, batch_size=args.batch_size)
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


def run_kmeans_method(
    method: str,
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
) -> None:
    if x_train.shape[1] != 1:
        raise ValueError(f"{method} requires a single-channel stiffness map, got input shape {x_train.shape[1:]}")
    method_dir = out_dir / method
    if _method_done(method_dir, args.force):
        print(f"{method}: existing metrics found")
    else:
        started = time.perf_counter()
        sklearn = _load_sklearn()
        kmeans = sklearn["KMeans"](n_clusters=2, n_init=10, random_state=args.seed)
        train_scores_native = x_train[:, 0]
        kmeans.fit(train_scores_native.reshape(-1, 1))
        centers = sorted(float(value) for value in kmeans.cluster_centers_.reshape(-1))
        fixed_threshold = float(sum(centers) / 2.0)
        val_scores = resize_score_maps(x_val[:, 0], y_val.shape[-2:])
        write_score_method(
            out_dir,
            method,
            "highres_kmeans",
            val_scores,
            y_val,
            val_names,
            config={
                **context,
                "selection": "two-cluster KMeans fitted on train split stiffness values",
                "centers": centers,
                "score_resize_policy": "bilinear interpolation from native scan grid to target mask grid before thresholding",
                "elapsed_seconds": time.perf_counter() - started,
            },
            fixed_threshold=fixed_threshold,
            force=True,
        )
    if x_test is None or y_test is None or test_names is None:
        return
    val_summary = _read_json(method_dir / "metrics_summary.json")
    fixed_threshold = float(val_summary.get("fixed_threshold", {}).get("threshold", 0.5))
    val_best_threshold = float(val_summary.get("threshold_sweep_best", {}).get("threshold", fixed_threshold))
    test_scores = resize_score_maps(x_test[:, 0], y_test.shape[-2:])
    write_test_metrics(
        method_dir,
        test_scores,
        y_test,
        test_names,
        fixed_threshold=fixed_threshold,
        val_best_threshold=val_best_threshold,
        context=context,
        force=True,
    )


def resize_score_maps(scores: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape[-2:] == output_shape:
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tensor = torch.from_numpy(scores[:, None].astype(np.float32))
    resized = F.interpolate(tensor, size=output_shape, mode="bilinear", align_corners=False)
    return np.nan_to_num(resized[:, 0].numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def write_test_metrics(
    method_dir: Path,
    scores: np.ndarray,
    masks: np.ndarray,
    names: list[str],
    *,
    fixed_threshold: float,
    val_best_threshold: float,
    context: dict[str, Any],
    force: bool,
) -> None:
    summary_path = method_dir / "test_metrics_summary.json"
    if summary_path.exists() and not force:
        return
    fixed = metrics_from_counts(counts_from_prediction(scores >= fixed_threshold, masks), threshold=fixed_threshold)
    val_selected = metrics_from_counts(counts_from_prediction(scores >= val_best_threshold, masks), threshold=val_best_threshold)
    oracle = evaluate_scores(scores, masks, fixed_threshold=0.5).best
    per_sample_rows = []
    for idx, name in enumerate(names):
        fixed_row = metrics_from_counts(counts_from_prediction(scores[idx] >= fixed_threshold, masks[idx]), threshold=fixed_threshold)
        val_row = metrics_from_counts(counts_from_prediction(scores[idx] >= val_best_threshold, masks[idx]), threshold=val_best_threshold)
        per_sample_rows.append(
            {
                "sample": name,
                "fixed_dice": fixed_row["dice"],
                "fixed_iou": fixed_row["iou"],
                "val_selected_dice": val_row["dice"],
                "val_selected_iou": val_row["iou"],
                "gt_positive": int(masks[idx].sum()),
            }
        )
    np.save(method_dir / "test_scores.npy", np.asarray(scores, dtype=np.float32))
    _write_csv(method_dir / "test_metrics_per_sample.csv", per_sample_rows, list(per_sample_rows[0].keys()))
    _write_json(
        summary_path,
        {
            "fixed_threshold": fixed,
            "val_selected_threshold": val_selected,
            "test_oracle_threshold": oracle,
            "num_samples": int(masks.shape[0]),
            "config": context,
        },
    )


def write_leaderboard(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out_dir.glob("*/metrics_summary.json")):
        summary = _read_json(metrics_path)
        config = summary.get("config", {})
        fixed = summary.get("fixed_threshold", {})
        best = summary.get("threshold_sweep_best", {})
        test_summary = _read_json(metrics_path.parent / "test_metrics_summary.json")
        test_fixed = test_summary.get("fixed_threshold", {})
        test_val = test_summary.get("val_selected_threshold", {})
        test_oracle = test_summary.get("test_oracle_threshold", {})
        rows.append(
            {
                "method": summary.get("method", metrics_path.parent.name),
                "resolution": config.get("resolution", ""),
                "input": config.get("input", ""),
                "model": config.get("model", ""),
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
    rows.sort(key=lambda row: float(row["val_best_dice"] or -1.0), reverse=True)
    if rows:
        _write_csv(out_dir / "leaderboard.csv", rows, list(rows[0].keys()))
    run_config = _read_json(out_dir / "run_config.json")
    lines = ["# High-Resolution Segmentation Sweep", ""]
    if run_config:
        lines.append(f"- Package source: `{run_config.get('package_dir', '')}`")
        lines.append(f"- Requested resolutions: `{run_config.get('resolutions', [])}`")
        lines.append(f"- Stiffness input: {run_config.get('stiffness_input_policy', STIFFNESS_INPUT_POLICY)}")
        lines.append(f"- Random-pair stiffness: {run_config.get('stiffness_random_pair_policy', STIFFNESS_RANDOM_PAIR_POLICY)}")
        lines.append(f"- Limited trajectory: {run_config.get('limited_trajectory_policy', LIMITED_TRAJECTORY_POLICY)}")
        lines.append(f"- Model resize: {run_config.get('model_resize_policy', MODEL_RESIZE_POLICY)}")
        lines.append("")
    if rows:
        lines.append(f"- Methods completed: {len(rows)}")
        lines.append(f"- Best validation method: `{rows[0]['method']}` Dice={_fmt(rows[0]['val_best_dice'])}")
        lines.append("")
        lines.append("| rank | method | resolution | input | model | val best Dice | test val-selected Dice |")
        lines.append("|---:|---|---:|---|---|---:|---:|")
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | `{row['method']}` | {row['resolution']} | {row['input']} | {row['model']} | "
                f"{_fmt(row['val_best_dice'])} | {_fmt(row['test_val_selected_dice'])} |"
            )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_packaging_notes(out_dir, rows, run_config)


def write_packaging_notes(out_dir: Path, rows: list[dict[str, Any]], run_config: dict[str, Any]) -> None:
    resolutions = sorted({str(row.get("resolution", "")) for row in rows if row.get("resolution", "")})
    inputs = sorted({str(row.get("input", "")) for row in rows if row.get("input", "")})
    models = sorted({str(row.get("model", "")) for row in rows if row.get("model", "")})
    lines = [
        "# High-Resolution Result Package Notes",
        "",
        "## Scope",
        "",
        f"- Source dataset package: `{run_config.get('package_dir', '')}`",
        f"- Result directory: `{out_dir}`",
        f"- Target label resolutions in this run: {', '.join(resolutions) if resolutions else 'pending'}",
        f"- Compared inputs: {', '.join(inputs) if inputs else 'pending'}",
        f"- Compared models: {', '.join(models) if models else 'pending'}",
        "",
        "## Stiffness Map Handling",
        "",
        "- Native equivalent-stiffness maps are computed on the original 20x20 scan grid from the force/displacement curves.",
        "- The random-pair variant replaces the full-curve equivalent stiffness with one first-sample-to-random-later-sample slope per scan point.",
        "- The limited-trajectory raw-curve variant crops each scan-point trajectory at a random endpoint and resamples Fz/displacement together.",
        f"- Input policy: {run_config.get('stiffness_input_policy', STIFFNESS_INPUT_POLICY)}",
        f"- Random-pair policy: {run_config.get('stiffness_random_pair_policy', STIFFNESS_RANDOM_PAIR_POLICY)}",
        f"- Limited-trajectory policy: {run_config.get('limited_trajectory_policy', LIMITED_TRAJECTORY_POLICY)}",
        f"- Internal resize policy: {run_config.get('model_resize_policy', MODEL_RESIZE_POLICY)}",
        "- For the requested 128x128 run, stiffness-map models receive a raw 20x20 stiffness input map.",
        "- Raw-Fz models also keep the native 20x20 spatial scan grid and predict the requested output resolution through the model path.",
        "",
        "## Files To Include When Packaging",
        "",
        "- `leaderboard.csv`: validation/test metrics for each method.",
        "- `report.md`: compact Markdown summary.",
        "- `run_config.json`: exact CLI/runtime configuration.",
        "- `*/metrics_summary.json`, `*/test_metrics_summary.json`, `*/history.csv`: per-method training and evaluation records.",
        "- `test_render/`: generated held-out test figures and HTML report, if rendered.",
        "",
    ]
    (out_dir / "PACKAGING_NOTES.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise SystemExit("Expected at least one integer.")
    if any(value <= 0 for value in values):
        raise SystemExit("Integer list values must be positive.")
    return values


def _parse_choice_list(raw: str, allowed: Sequence[str], label: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown {label}: {unknown}; allowed: {list(allowed)}")
    if not values:
        raise SystemExit(f"Expected at least one {label} value.")
    return values


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
