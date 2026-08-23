from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import warnings
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

from palpation_sim.models import ValidationUNet
from palpation_sim.workflow import require_runtime_environment


DEFAULT_DATA_DIR = Path("data/palpation_newton_random_shapes_20x_scan20_combined_seed11_seed23_seed37_seed53")
DEFAULT_OUT_DIR = Path("runs/segmentation_accuracy_sweep_20x_4seed")
BASELINE_UNET_METRICS = Path(
    "runs/validation_unet_random_shapes_20x_scan20_combined_seed11_seed23_seed37_seed53/eval_combined/metrics_summary.json"
)
METHOD_GROUPS = (
    "sanity",
    "stiffness",
    "stiffness_patch_mlp",
    "stiffness_spatial",
    "rimon_representation",
    "classifiers",
    "sequence",
    "spatial",
    "unet",
)
GROUP_ALIASES = {"stiffness_mlp": "stiffness_patch_mlp"}
FEATURE_NAMES = [
    "f_max",
    "equivalent_stiffness",
    "global_stiffness",
    "early_stiffness",
    "late_stiffness",
    "loading_work",
    "force_z25",
    "force_z50",
    "force_z75",
    "nonlinearity_ratio",
]
METHOD_MEANINGS = {
    "unet_fz_features_aug_focal": "U-Net on raw Fz plus mechanical feature maps with D4 augmentation and focal+BCE+Dice loss.",
    "unet_fz_norm_none": "U-Net on raw Fz channels without input normalization.",
    "unet_fz_features_dataset_norm": "U-Net on raw Fz plus mechanical feature maps with dataset normalization.",
    "unet_fz_norm_dataset": "U-Net on raw Fz channels with dataset normalization.",
    "unet_delta_fz_norm_dataset": "U-Net on preload-subtracted Fz-Fz[0] channels with dataset normalization.",
    "unet_stiffness": "U-Net on the single equivalent-stiffness map.",
    "temporal_transformer_spatial_head": "Per-point temporal Transformer embeddings followed by a shallow CNN spatial head.",
    "shallow_cnn_fz": "Shallow 2D CNN on the raw Fz channel map.",
    "spatial_token_transformer_fz": "Transformer over 20x20 spatial scan tokens using raw Fz features.",
    "spatial_token_transformer_fz_features": "Transformer over 20x20 spatial scan tokens using raw Fz plus mechanical features.",
    "shallow_cnn_stiffness": "Shallow 2D CNN on the equivalent-stiffness map.",
    "unet_stiffness_aug_focal": "U-Net on the equivalent-stiffness map with D4 augmentation and focal+BCE+Dice loss.",
    "patchwise_mlp_stiffness_p5": "Sliding-window MLP using a 5x5 local equivalent-stiffness patch.",
    "stiffness_val_oracle_per_sample_threshold": "Oracle per-sample threshold on equivalent stiffness selected using val labels.",
    "patchwise_mlp_stiffness_p3": "Sliding-window MLP using a 3x3 local equivalent-stiffness patch.",
    "patch_mlp_fz_p3": "Sliding-window MLP using a 3x3 local raw-Fz patch.",
    "patch_mlp_fz_p5": "Sliding-window MLP using a 5x5 local raw-Fz patch.",
    "pixel_mlp_rerun": "Pointwise MLP using each scan point's raw Fz curve only.",
    "random_forest_mechanical": "Random forest on per-point mechanical feature vectors.",
    "random_forest_mechanical_xy": "Random forest on mechanical features plus normalized x,y coordinates.",
    "hist_gradient_boosting_mechanical": "HistGradientBoosting classifier on per-point mechanical features.",
    "point_transformer_dataset": "Pointwise temporal Transformer on raw Fz with dataset normalization.",
    "point_transformer_sample": "Pointwise temporal Transformer on raw Fz with sample normalization.",
    "hist_gradient_boosting_mechanical_xy": "HistGradientBoosting classifier on mechanical features plus normalized x,y coordinates.",
    "point_gru": "Pointwise GRU on each raw Fz time series.",
    "point_1d_cnn": "Pointwise 1D CNN along each raw Fz time series.",
    "logistic_mechanical": "Logistic regression on per-point mechanical features.",
    "stiffness_morphology_val_oracle": "Oracle morphology postprocessing selected on val stiffness maps.",
    "stiffness_global_train_threshold": "Global equivalent-stiffness threshold selected on the train split.",
    "stiffness_gmm_train": "Two-component Gaussian mixture threshold fitted on train stiffness values.",
    "stiffness_kmeans_train": "Two-cluster KMeans threshold fitted on train stiffness values.",
    "stiffness_otsu_train": "Otsu threshold fitted on train stiffness values.",
    "stiffness_val_oracle_global_threshold": "Oracle global equivalent-stiffness threshold selected using val labels.",
    "logistic_mechanical_xy": "Logistic regression on mechanical features plus normalized x,y coordinates.",
    "pointwise_mlp_stiffness": "Pointwise MLP using each scan point's equivalent stiffness only.",
    "stiffness_morphology_train_selected": "Morphology postprocessing selected on train stiffness maps and applied to val.",
    "rimon_gru_reconstruction_decoder": "Rimon-style FLE+GRU force-reconstruction representation with frozen transposed-conv mask decoder.",
    "stiffness_template_shape_prior": "Template/shape prior built from thresholded equivalent-stiffness components.",
    "sanity_all_foreground": "Sanity check that predicts every pixel as foreground.",
    "sanity_train_foreground_prior": "Sanity check using a constant foreground prior from the train positive rate.",
    "sanity_all_background": "Sanity check that predicts every pixel as background.",
}


@dataclass
class SplitData:
    names: list[str]
    fz: np.ndarray  # [N, T, H, W]
    masks: np.ndarray  # [N, H, W]
    features: np.ndarray  # [N, F, H, W]
    xy: np.ndarray  # [N, 2, H, W]


@dataclass
class EvalResult:
    fixed: dict[str, float | int]
    best: dict[str, float | int]
    threshold_rows: list[dict[str, float | int]]
    per_sample_rows: list[dict[str, float | int | str]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 2D segmentation baseline and improvement sweep.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--groups", type=str, default=",".join(METHOD_GROUPS))
    parser.add_argument("--force", action="store_true", help="Rerun methods even if metrics_summary.json exists.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true", help="Use very small settings to validate artifacts.")
    args = parser.parse_args()

    require_runtime_environment()
    _seed_everything(args.seed)
    train_dir = args.train_dir or args.data_dir / "train"
    val_dir = args.val_dir or args.data_dir / "val"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = _parse_groups(args.groups)
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)

    print(f"loading train split: {train_dir}")
    train = load_split(train_dir, max_samples=8 if args.smoke else None)
    print(f"loading val split: {val_dir}")
    val = load_split(val_dir, max_samples=4 if args.smoke else None)
    context = {
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "train_samples": len(train.names),
        "val_samples": len(val.names),
        "shape": list(val.masks.shape[-2:]),
        "feature_names": FEATURE_NAMES,
        "seed": args.seed,
        "smoke": args.smoke,
    }
    _write_json(args.out_dir / "run_config.json", context)

    if "sanity" in groups:
        run_sanity_baselines(train, val, args.out_dir, context, force=args.force)
    if "stiffness" in groups:
        run_stiffness_baselines(train, val, args.out_dir, context, force=args.force)
    if "stiffness_patch_mlp" in groups:
        run_stiffness_patch_mlp_baselines(train, val, args.out_dir, context, args)
    if "stiffness_spatial" in groups:
        run_stiffness_spatial_baselines(train, val, args.out_dir, context, args)
    if "rimon_representation" in groups:
        run_rimon_representation_baselines(train, val, args.out_dir, context, args)
    if "classifiers" in groups:
        run_classifier_baselines(train, val, args.out_dir, context, force=args.force, seed=args.seed)
    if "sequence" in groups:
        run_sequence_baselines(train, val, args.out_dir, context, args)
    if "spatial" in groups:
        run_spatial_baselines(train, val, args.out_dir, context, args)
    if "unet" in groups:
        run_unet_improvements(train, val, args.out_dir, context, args)

    write_leaderboard(args.out_dir)
    print(f"sweep complete: {args.out_dir}")


def load_split(split_dir: Path, *, max_samples: int | None = None) -> SplitData:
    files = sorted(split_dir.glob("*.npz"))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")
    names: list[str] = []
    fz_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    feature_list: list[np.ndarray] = []
    xy_list: list[np.ndarray] = []
    for path in files:
        with np.load(path) as sample:
            fz_hwt = _load_fz_hwt(sample)
            z_hwt = _load_displacement_hwt(sample, fz_hwt.shape)
            mask = np.asarray(sample["mask"], dtype=np.float32)
            xy = _load_xy_chw(sample, mask.shape)
            nonlinearity = _load_nonlinearity_map(sample, mask.shape)
        features = extract_mechanical_features(z_hwt, fz_hwt, nonlinearity)
        names.append(path.name)
        fz_list.append(np.moveaxis(fz_hwt, -1, 0).astype(np.float32))
        mask_list.append((mask > 0.5).astype(np.uint8))
        feature_list.append(features)
        xy_list.append(xy)
    return SplitData(
        names=names,
        fz=np.stack(fz_list).astype(np.float32),
        masks=np.stack(mask_list).astype(np.uint8),
        features=np.stack(feature_list).astype(np.float32),
        xy=np.stack(xy_list).astype(np.float32),
    )


def _load_fz_hwt(sample: np.lib.npyio.NpzFile) -> np.ndarray:
    if "fz" in sample:
        fz = np.asarray(sample["fz"], dtype=np.float32)
    elif "presses" in sample:
        presses = np.asarray(sample["presses"], dtype=np.float32)
        fz = presses[..., 1]
    else:
        raise KeyError("sample must contain 'fz' or 'presses'")
    if fz.ndim != 3:
        raise ValueError(f"Expected fz shape [H, W, T], got {fz.shape}")
    return fz.astype(np.float32)


def _load_displacement_hwt(sample: np.lib.npyio.NpzFile, shape: tuple[int, int, int]) -> np.ndarray:
    if "presses" in sample:
        z = np.asarray(sample["presses"], dtype=np.float32)[..., 0]
    elif "indentation_depth" in sample:
        z = np.asarray(sample["indentation_depth"], dtype=np.float32)
    else:
        z = np.linspace(0.0, 1.0, shape[-1], dtype=np.float32)
    if z.ndim == 1:
        z = np.broadcast_to(z.reshape(1, 1, -1), shape)
    if z.shape != shape:
        raise ValueError(f"Expected displacement shape {shape}, got {z.shape}")
    return z.astype(np.float32)


def _load_xy_chw(sample: np.lib.npyio.NpzFile, mask_shape: tuple[int, int]) -> np.ndarray:
    if "xy" in sample:
        xy_hwc = np.asarray(sample["xy"], dtype=np.float32)
        if xy_hwc.shape[:2] == mask_shape and xy_hwc.shape[-1] >= 2:
            xy = np.moveaxis(xy_hwc[..., :2], -1, 0)
        else:
            xy = _default_xy(mask_shape)
    else:
        xy = _default_xy(mask_shape)
    out = xy.astype(np.float32)
    for idx in range(2):
        channel = out[idx]
        span = float(channel.max() - channel.min())
        out[idx] = 0.0 if span < 1e-12 else 2.0 * (channel - channel.min()) / span - 1.0
    return out


def _default_xy(mask_shape: tuple[int, int]) -> np.ndarray:
    h, w = mask_shape
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    return np.stack([x, y], axis=0)


def _load_nonlinearity_map(sample: np.lib.npyio.NpzFile, mask_shape: tuple[int, int]) -> np.ndarray:
    if "nonlinearity_ratio" not in sample:
        return np.zeros(mask_shape, dtype=np.float32)
    value = np.asarray(sample["nonlinearity_ratio"], dtype=np.float32)
    if value.shape == mask_shape:
        return value.astype(np.float32)
    if value.size == 1:
        return np.full(mask_shape, float(value.reshape(())), dtype=np.float32)
    return np.zeros(mask_shape, dtype=np.float32)


def extract_mechanical_features(z_raw: np.ndarray, f_raw: np.ndarray, nonlinearity: np.ndarray) -> np.ndarray:
    z, f = _make_compression_positive(z_raw, f_raw)
    h, w, t = z.shape
    z2 = z.reshape(-1, t)
    f2 = f.reshape(-1, t)
    features = np.zeros((z2.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = np.max(f2, axis=1)
    dz = z2[:, -1] - z2[:, 0]
    df = f2[:, -1] - f2[:, 0]
    features[:, 1] = np.divide(df, dz, out=np.zeros_like(df), where=np.abs(dz) >= 1e-9)
    features[:, 1] = np.maximum(features[:, 1], 0.0)
    features[:, 2] = _slope_many(z2, f2)
    features[:, 3] = _segment_slope_many(z2, f2, 0.10, 0.40)
    features[:, 4] = _segment_slope_many(z2, f2, 0.60, 0.90)
    features[:, 5] = _trapz_many(f2, z2)
    features[:, 6] = _interp_force_many(z2, f2, 0.25)
    features[:, 7] = _interp_force_many(z2, f2, 0.50)
    features[:, 8] = _interp_force_many(z2, f2, 0.75)
    features[:, 9] = nonlinearity.reshape(-1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.moveaxis(features.reshape(h, w, len(FEATURE_NAMES)), -1, 0).astype(np.float32)


def _make_compression_positive(z_raw: np.ndarray, f_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = z_raw.astype(np.float32) - z_raw[..., :1].astype(np.float32)
    f = f_raw.astype(np.float32) - f_raw[..., :1].astype(np.float32)
    z = np.where(np.abs(np.nanmin(z, axis=-1, keepdims=True)) > np.abs(np.nanmax(z, axis=-1, keepdims=True)), -z, z)
    f = np.where(np.abs(np.nanmin(f, axis=-1, keepdims=True)) > np.abs(np.nanmax(f, axis=-1, keepdims=True)), -f, f)
    return z.astype(np.float32), f.astype(np.float32)


def _slope_many(z: np.ndarray, f: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        weights = np.ones_like(z, dtype=np.float32)
    else:
        weights = mask.astype(np.float32)
    counts = np.sum(weights, axis=1, keepdims=True)
    valid = counts >= 2.0
    counts = np.maximum(counts, 1.0)
    z_mean = np.sum(z * weights, axis=1, keepdims=True) / counts
    f_mean = np.sum(f * weights, axis=1, keepdims=True) / counts
    z_centered = (z - z_mean) * weights
    denom = np.sum(z_centered * z_centered, axis=1)
    numer = np.sum(z_centered * (f - f_mean) * weights, axis=1)
    out = np.divide(numer, denom, out=np.zeros_like(numer, dtype=np.float32), where=(denom >= 1e-12) & valid[:, 0])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _segment_slope_many(z: np.ndarray, f: np.ndarray, low: float, high: float) -> np.ndarray:
    z_min = np.min(z, axis=1, keepdims=True)
    z_max = np.max(z, axis=1, keepdims=True)
    span = z_max - z_min
    keep = (span >= 1e-12) & (z >= z_min + low * span) & (z <= z_min + high * span)
    return _slope_many(z, f, keep)


def _trapz_many(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sum(0.5 * (y[:, 1:] + y[:, :-1]) * (x[:, 1:] - x[:, :-1]), axis=1).astype(np.float32)


def _interp_force_many(z: np.ndarray, f: np.ndarray, fraction: float) -> np.ndarray:
    z_min = np.min(z, axis=1)
    z_max = np.max(z, axis=1)
    span = z_max - z_min
    target = z_min + fraction * span
    idx_hi = np.sum(z < target[:, None], axis=1)
    idx_hi = np.clip(idx_hi, 1, z.shape[1] - 1)
    idx_lo = idx_hi - 1
    rows = np.arange(z.shape[0])
    z0 = z[rows, idx_lo]
    z1 = z[rows, idx_hi]
    f0 = f[rows, idx_lo]
    f1 = f[rows, idx_hi]
    denom = z1 - z0
    alpha = np.divide(target - z0, denom, out=np.zeros_like(target), where=np.abs(denom) > 1e-12)
    out = f0 + alpha * (f1 - f0)
    return np.where(span >= 1e-12, out, np.max(f, axis=1)).astype(np.float32)


def run_sanity_baselines(train: SplitData, val: SplitData, out_dir: Path, context: dict[str, Any], *, force: bool) -> None:
    foreground_rate = float(train.masks.mean())
    methods = {
        "sanity_all_background": np.zeros_like(val.masks, dtype=np.float32),
        "sanity_all_foreground": np.ones_like(val.masks, dtype=np.float32),
        "sanity_train_foreground_prior": np.full_like(val.masks, foreground_rate, dtype=np.float32),
    }
    for method, scores in methods.items():
        write_score_method(
            out_dir,
            method,
            "sanity",
            scores,
            val.masks,
            val.names,
            config={**context, "foreground_rate": foreground_rate},
            fixed_threshold=0.5,
            force=force,
        )


def run_stiffness_baselines(train: SplitData, val: SplitData, out_dir: Path, context: dict[str, Any], *, force: bool) -> None:
    sklearn = _load_sklearn()
    filters, morphology, measure, draw = _load_skimage()
    train_k = _stiffness(train)
    val_k = _stiffness(val)
    train_t = best_threshold(train_k, train.masks)["threshold"]
    val_t = best_threshold(val_k, val.masks)["threshold"]
    write_score_method(
        out_dir,
        "stiffness_global_train_threshold",
        "stiffness",
        val_k,
        val.masks,
        val.names,
        config={**context, "feature": "equivalent_stiffness", "selection": "best global threshold on train"},
        fixed_threshold=float(train_t),
        force=force,
    )
    write_score_method(
        out_dir,
        "stiffness_val_oracle_global_threshold",
        "stiffness_oracle",
        val_k,
        val.masks,
        val.names,
        config={**context, "feature": "equivalent_stiffness", "selection": "best global threshold on val", "oracle": True},
        fixed_threshold=float(val_t),
        force=force,
    )
    write_binary_method(
        out_dir,
        "stiffness_val_oracle_per_sample_threshold",
        "stiffness_oracle",
        per_sample_threshold_predictions(val_k, val.masks),
        val.masks,
        val.names,
        config={**context, "feature": "equivalent_stiffness", "oracle": True},
        force=force,
    )

    otsu_t = float(filters.threshold_otsu(train_k.reshape(-1)))
    write_score_method(
        out_dir,
        "stiffness_otsu_train",
        "stiffness",
        val_k,
        val.masks,
        val.names,
        config={**context, "feature": "equivalent_stiffness", "selection": "Otsu threshold on train"},
        fixed_threshold=otsu_t,
        force=force,
    )

    kmeans = sklearn["KMeans"](n_clusters=2, n_init=10, random_state=7)
    kmeans.fit(train_k.reshape(-1, 1))
    centers = sorted(float(value) for value in kmeans.cluster_centers_.reshape(-1))
    write_score_method(
        out_dir,
        "stiffness_kmeans_train",
        "stiffness",
        val_k,
        val.masks,
        val.names,
        config={**context, "feature": "equivalent_stiffness", "centers": centers},
        fixed_threshold=float(sum(centers) / 2.0),
        force=force,
    )

    gmm = sklearn["GaussianMixture"](n_components=2, random_state=7)
    gmm.fit(train_k.reshape(-1, 1))
    means = sorted(float(value) for value in gmm.means_.reshape(-1))
    write_score_method(
        out_dir,
        "stiffness_gmm_train",
        "stiffness",
        val_k,
        val.masks,
        val.names,
        config={**context, "feature": "equivalent_stiffness", "means": means},
        fixed_threshold=float(sum(means) / 2.0),
        force=force,
    )

    fair_pred, fair_config = fit_morphology_grid(train_k, train.masks, val_k, morphology, oracle=False)
    write_binary_method(
        out_dir,
        "stiffness_morphology_train_selected",
        "stiffness",
        fair_pred,
        val.masks,
        val.names,
        config={**context, **fair_config},
        force=force,
    )
    oracle_pred, oracle_config = fit_morphology_grid(val_k, val.masks, val_k, morphology, oracle=True)
    write_binary_method(
        out_dir,
        "stiffness_morphology_val_oracle",
        "stiffness_oracle",
        oracle_pred,
        val.masks,
        val.names,
        config={**context, **oracle_config, "oracle": True},
        force=force,
    )

    template_pred, template_config = template_shape_prior(val_k, fixed_threshold=float(train_t), morphology=morphology, measure=measure, draw=draw)
    write_binary_method(
        out_dir,
        "stiffness_template_shape_prior",
        "template",
        template_pred,
        val.masks,
        val.names,
        config={**context, **template_config},
        force=force,
    )


def run_stiffness_patch_mlp_baselines(
    train: SplitData,
    val: SplitData,
    out_dir: Path,
    context: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    x_train, x_val = normalized_stiffness_maps(train, val)
    common_context = {**context, "input": "equivalent_stiffness", "normalize_mode": "dataset"}
    train_neural_method(
        "pointwise_mlp_stiffness",
        "stiffness_patch_mlp",
        PointMLP(x_train.shape[1], hidden_dims=(128, 64)),
        x_train,
        train.masks,
        x_val,
        val.masks,
        val.names,
        out_dir,
        context=common_context,
        args=args,
    )
    for patch_size in (3, 5):
        train_neural_method(
            f"patchwise_mlp_stiffness_p{patch_size}",
            "stiffness_patch_mlp",
            PatchMLP(x_train.shape[1], patch_size=patch_size, hidden_dims=(512, 128)),
            x_train,
            train.masks,
            x_val,
            val.masks,
            val.names,
            out_dir,
            context={**common_context, "patch_size": patch_size},
            args=args,
        )


def run_stiffness_spatial_baselines(
    train: SplitData,
    val: SplitData,
    out_dir: Path,
    context: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    x_train, x_val = normalized_stiffness_maps(train, val)
    common_context = {**context, "input": "equivalent_stiffness", "normalize_mode": "dataset"}
    train_neural_method(
        "shallow_cnn_stiffness",
        "stiffness_spatial",
        ShallowCNN(x_train.shape[1], base_channels=32),
        x_train,
        train.masks,
        x_val,
        val.masks,
        val.names,
        out_dir,
        context={**common_context, "model": "shallow_2d_cnn"},
        args=args,
    )
    train_neural_method(
        "unet_stiffness",
        "stiffness_spatial",
        ValidationUNet(x_train.shape[1], base_channels=24),
        x_train,
        train.masks,
        x_val,
        val.masks,
        val.names,
        out_dir,
        context={**common_context, "model": "validation_unet"},
        args=args,
        augment=False,
    )
    train_neural_method(
        "unet_stiffness_aug_focal",
        "stiffness_spatial",
        ValidationUNet(x_train.shape[1], base_channels=24),
        x_train,
        train.masks,
        x_val,
        val.masks,
        val.names,
        out_dir,
        context={**common_context, "model": "validation_unet", "augmentation": "d4", "loss": "focal_bce_dice"},
        args=args,
        augment=True,
        focal=True,
    )


def run_rimon_representation_baselines(
    train: SplitData,
    val: SplitData,
    out_dir: Path,
    context: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    method = "rimon_gru_reconstruction_decoder"
    method_dir = out_dir / method
    if _method_done(method_dir, args.force):
        return

    train_forces, val_forces, train_locations, val_locations, norm_stats = rimon_sequence_data(train, val)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    rep_config = {
        "paper": "Rimon et al. 2025 Toward Artificial Palpation",
        "adaptation": "FLE+GRU force reconstruction pretraining on scan-point force curves, frozen representation with transposed-conv mask decoder",
        "force_size": int(train_forces.shape[-1]),
        "locations_size": int(train_locations.shape[-1]),
        "sequence_length": int(train_forces.shape[1]),
        "representation_size": 512,
        "force_location_embed_dim": 128,
        "decoder_input_embed_dim": 512,
        "force_predictor_hidden": 1024,
        "input_num_random_samples": 64,
        "reconstruction_num_random_samples": 64,
        "pretrain_epochs_requested": int(args.epochs),
        "pretrain_patience": int(args.patience),
        "normalize_mode": "dataset_force_channel",
        "force_normalization_mean": norm_stats["mean"],
        "force_normalization_std": norm_stats["std"],
        "trajectory_permutation_augmentation": True,
    }
    pretrain_dir = out_dir / "rimon_gru_reconstruction_pretrain"
    encoder = RimonReconstructionModel(
        force_size=rep_config["force_size"],
        locations_size=rep_config["locations_size"],
        representation_size=rep_config["representation_size"],
        force_location_embed_dim=rep_config["force_location_embed_dim"],
        decoder_input_embed_dim=rep_config["decoder_input_embed_dim"],
        force_predictor_hidden=rep_config["force_predictor_hidden"],
        input_num_random_samples=rep_config["input_num_random_samples"],
        reconstruction_num_random_samples=rep_config["reconstruction_num_random_samples"],
    ).to(device)
    pretrain_ckpt = pretrain_dir / "best.pt"
    if pretrain_ckpt.exists() and not args.force:
        checkpoint = torch.load(pretrain_ckpt, map_location=device)
        encoder.load_state_dict(checkpoint["model_state"])
        pretrain_summary = _read_json(pretrain_dir / "pretrain_summary.json")
        print(f"loaded Rimon-style pretrain checkpoint: {pretrain_ckpt}")
    else:
        pretrain_summary = train_rimon_reconstruction_model(
            encoder,
            train_forces,
            train_locations,
            val_forces,
            val_locations,
            pretrain_dir,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            config={**context, **rep_config},
        )

    train_repr = encode_rimon_representations(encoder, train_forces, train_locations, device=device, batch_size=args.batch_size)
    val_repr = encode_rimon_representations(encoder, val_forces, val_locations, device=device, batch_size=args.batch_size)
    train_repr_map = train_repr[:, :, None, None]
    val_repr_map = val_repr[:, :, None, None]
    train_neural_method(
        method,
        "rimon_representation",
        RepresentationToMaskDecoder(rep_config["representation_size"], output_shape=train.masks.shape[-2:], base_channels=64),
        train_repr_map,
        train.masks,
        val_repr_map,
        val.masks,
        val.names,
        out_dir,
        context={
            **context,
            **rep_config,
            "pretrain_dir": str(pretrain_dir),
            "pretrain_checkpoint": str(pretrain_ckpt),
            "pretrain_best_val_mse": pretrain_summary.get("best_val_mse"),
            "pretrain_epochs_completed": pretrain_summary.get("epochs_completed"),
            "input": "rimon_frozen_gru_representation",
            "downstream_head": "transposed_conv_mask_decoder",
        },
        args=args,
    )


def run_classifier_baselines(
    train: SplitData,
    val: SplitData,
    out_dir: Path,
    context: dict[str, Any],
    *,
    force: bool,
    seed: int,
) -> None:
    sklearn = _load_sklearn()
    feature_sets = {
        "mechanical": (train.features, val.features, FEATURE_NAMES),
        "mechanical_xy": (
            np.concatenate([train.features, train.xy], axis=1),
            np.concatenate([val.features, val.xy], axis=1),
            FEATURE_NAMES + ["x_norm", "y_norm"],
        ),
    }
    for suffix, (train_feat, val_feat, names) in feature_sets.items():
        x_train = _flatten_features(train_feat)
        x_val = _flatten_features(val_feat)
        y_train = train.masks.reshape(-1).astype(np.uint8)
        models = {
            f"logistic_{suffix}": sklearn["Pipeline"](
                [
                    ("scaler", sklearn["StandardScaler"]()),
                    (
                        "model",
                        sklearn["LogisticRegression"](
                            max_iter=1000,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            f"random_forest_{suffix}": sklearn["RandomForestClassifier"](
                n_estimators=160,
                max_depth=16,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            f"hist_gradient_boosting_{suffix}": sklearn["Pipeline"](
                [
                    ("scaler", sklearn["StandardScaler"]()),
                    (
                        "model",
                        sklearn["HistGradientBoostingClassifier"](
                            max_iter=220,
                            learning_rate=0.08,
                            max_leaf_nodes=31,
                            l2_regularization=0.01,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        }
        sample_weight = _balanced_sample_weight(y_train)
        for method, model in models.items():
            method_dir = out_dir / method
            if _method_done(method_dir, force):
                continue
            started = time.perf_counter()
            if "hist_gradient_boosting" in method:
                model.fit(x_train, y_train, model__sample_weight=sample_weight)
            else:
                model.fit(x_train, y_train)
            probs = model.predict_proba(x_val)[:, 1].reshape(val.masks.shape)
            write_score_method(
                out_dir,
                method,
                "classifiers",
                probs.astype(np.float32),
                val.masks,
                val.names,
                config={**context, "features": names, "elapsed_seconds": time.perf_counter() - started},
                fixed_threshold=0.5,
                force=True,
            )


def run_sequence_baselines(train: SplitData, val: SplitData, out_dir: Path, context: dict[str, Any], args: argparse.Namespace) -> None:
    x_train, x_val = normalize_maps(train.fz, val.fz, mode="sample")
    x_train_dataset, x_val_dataset = normalize_maps(train.fz, val.fz, mode="dataset")
    feature_train_dataset, feature_val_dataset = normalize_maps(train.features, val.features, mode="dataset")
    x_train_dataset_features = np.concatenate([x_train_dataset, feature_train_dataset], axis=1)
    x_val_dataset_features = np.concatenate([x_val_dataset, feature_val_dataset], axis=1)
    train_neural_method(
        "pixel_mlp_rerun",
        "sequence",
        PointMLP(x_train.shape[1], hidden_dims=(128, 64)),
        x_train,
        train.masks,
        x_val,
        val.masks,
        val.names,
        out_dir,
        context={**context, "input": "raw_fz", "normalize_mode": "sample"},
        args=args,
    )
    train_neural_method(
        "point_transformer_sample",
        "sequence_transformer",
        PointTemporalTransformer(x_train.shape[1], d_model=48, nhead=4, num_layers=2),
        x_train,
        train.masks,
        x_val,
        val.masks,
        val.names,
        out_dir,
        context={**context, "input": "raw_fz", "normalize_mode": "sample", "temporal_model": "transformer_cls"},
        args=args,
    )
    train_neural_method(
        "point_transformer_dataset",
        "sequence_transformer",
        PointTemporalTransformer(x_train_dataset.shape[1], d_model=48, nhead=4, num_layers=2),
        x_train_dataset,
        train.masks,
        x_val_dataset,
        val.masks,
        val.names,
        out_dir,
        context={**context, "input": "raw_fz", "normalize_mode": "dataset", "temporal_model": "transformer_cls"},
        args=args,
    )
    train_neural_method(
        "temporal_transformer_spatial_head",
        "sequence_transformer",
        TemporalTransformerSpatialHead(x_train_dataset.shape[1], d_model=32, nhead=4, num_layers=2),
        x_train_dataset,
        train.masks,
        x_val_dataset,
        val.masks,
        val.names,
        out_dir,
        context={
            **context,
            "input": "raw_fz",
            "normalize_mode": "dataset",
            "temporal_model": "transformer_encoder",
            "spatial_head": "shallow_cnn",
        },
        args=args,
    )
    train_neural_method(
        "spatial_token_transformer_fz",
        "sequence_transformer",
        SpatialTokenTransformer(
            x_train_dataset.shape[1],
            spatial_shape=train.masks.shape[-2:],
            d_model=64,
            nhead=4,
            num_layers=2,
        ),
        x_train_dataset,
        train.masks,
        x_val_dataset,
        val.masks,
        val.names,
        out_dir,
        context={
            **context,
            "input": "raw_fz",
            "normalize_mode": "dataset",
            "transformer_model": "spatial_tokens",
        },
        args=args,
    )
    train_neural_method(
        "spatial_token_transformer_fz_features",
        "sequence_transformer",
        SpatialTokenTransformer(
            x_train_dataset_features.shape[1],
            spatial_shape=train.masks.shape[-2:],
            d_model=64,
            nhead=4,
            num_layers=2,
        ),
        x_train_dataset_features,
        train.masks,
        x_val_dataset_features,
        val.masks,
        val.names,
        out_dir,
        context={
            **context,
            "input": "raw_fz_plus_features",
            "normalize_mode": "dataset",
            "transformer_model": "spatial_tokens",
        },
        args=args,
    )
    train_neural_method(
        "point_1d_cnn",
        "sequence",
        PointCNN1D(x_train.shape[1]),
        x_train,
        train.masks,
        x_val,
        val.masks,
        val.names,
        out_dir,
        context={**context, "input": "raw_fz", "normalize_mode": "sample"},
        args=args,
    )
    train_neural_method(
        "point_gru",
        "sequence",
        PointGRU(),
        x_train,
        train.masks,
        x_val,
        val.masks,
        val.names,
        out_dir,
        context={**context, "input": "raw_fz", "normalize_mode": "sample"},
        args=args,
    )


def run_spatial_baselines(train: SplitData, val: SplitData, out_dir: Path, context: dict[str, Any], args: argparse.Namespace) -> None:
    x_train, x_val = normalize_maps(train.fz, val.fz, mode="sample")
    for patch_size in (3, 5):
        train_neural_method(
            f"patch_mlp_fz_p{patch_size}",
            "spatial",
            PatchMLP(x_train.shape[1], patch_size=patch_size, hidden_dims=(512, 128)),
            x_train,
            train.masks,
            x_val,
            val.masks,
            val.names,
            out_dir,
            context={**context, "input": "raw_fz", "normalize_mode": "sample", "patch_size": patch_size},
            args=args,
        )
    train_neural_method(
        "shallow_cnn_fz",
        "spatial",
        ShallowCNN(x_train.shape[1], base_channels=32),
        x_train,
        train.masks,
        x_val,
        val.masks,
        val.names,
        out_dir,
        context={**context, "input": "raw_fz", "normalize_mode": "sample"},
        args=args,
    )


def run_unet_improvements(train: SplitData, val: SplitData, out_dir: Path, context: dict[str, Any], args: argparse.Namespace) -> None:
    for normalize_mode in ("none", "dataset"):
        x_train, x_val = normalize_maps(train.fz, val.fz, mode=normalize_mode)
        train_neural_method(
            f"unet_fz_norm_{normalize_mode}",
            "unet",
            ValidationUNet(x_train.shape[1], base_channels=24),
            x_train,
            train.masks,
            x_val,
            val.masks,
            val.names,
            out_dir,
            context={**context, "input": "raw_fz", "normalize_mode": normalize_mode},
            args=args,
            augment=False,
        )

    delta_train, delta_val = normalize_maps(preload_subtract_fz(train.fz), preload_subtract_fz(val.fz), mode="dataset")
    train_neural_method(
        "unet_delta_fz_norm_dataset",
        "unet",
        ValidationUNet(delta_train.shape[1], base_channels=24),
        delta_train,
        train.masks,
        delta_val,
        val.masks,
        val.names,
        out_dir,
        context={
            **context,
            "input": "delta_fz",
            "force_preprocess": "subtract_initial_fz_per_curve",
            "normalize_mode": "dataset",
        },
        args=args,
        augment=False,
    )

    fz_train, fz_val = normalize_maps(train.fz, val.fz, mode="dataset")
    feat_train, feat_val = normalize_maps(train.features, val.features, mode="dataset")
    combo_train = np.concatenate([fz_train, feat_train], axis=1)
    combo_val = np.concatenate([fz_val, feat_val], axis=1)
    train_neural_method(
        "unet_fz_features_dataset_norm",
        "unet",
        ValidationUNet(combo_train.shape[1], base_channels=24),
        combo_train,
        train.masks,
        combo_val,
        val.masks,
        val.names,
        out_dir,
        context={**context, "input": "raw_fz_plus_features", "normalize_mode": "dataset"},
        args=args,
        augment=False,
    )
    train_neural_method(
        "unet_fz_features_aug_focal",
        "unet",
        ValidationUNet(combo_train.shape[1], base_channels=24),
        combo_train,
        train.masks,
        combo_val,
        val.masks,
        val.names,
        out_dir,
        context={**context, "input": "raw_fz_plus_features", "normalize_mode": "dataset", "augmentation": "d4", "loss": "focal_bce_dice"},
        args=args,
        augment=True,
        focal=True,
    )


def _stiffness(split: SplitData) -> np.ndarray:
    return split.features[:, FEATURE_NAMES.index("equivalent_stiffness")]


def preload_subtract_fz(fz: np.ndarray) -> np.ndarray:
    fz = np.asarray(fz, dtype=np.float32)
    if fz.ndim < 2:
        raise ValueError(f"Expected Fz with sample and depth/channel axes, got shape {fz.shape}")
    out = fz - fz[:, :1, ...]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalized_stiffness_maps(train: SplitData, val: SplitData) -> tuple[np.ndarray, np.ndarray]:
    return normalize_maps(_stiffness(train)[:, None], _stiffness(val)[:, None], mode="dataset")


def rimon_sequence_data(
    train: SplitData,
    val: SplitData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[float]]]:
    train_forces = _flatten_force_curves(train.fz)
    val_forces = _flatten_force_curves(val.fz)
    mean = np.nanmean(train_forces, axis=(0, 1), keepdims=True)
    std = np.nanstd(train_forces, axis=(0, 1), keepdims=True)
    train_forces = _apply_force_curve_normalize(train_forces, mean, std)
    val_forces = _apply_force_curve_normalize(val_forces, mean, std)
    train_locations = _flatten_locations(train.xy)
    val_locations = _flatten_locations(val.xy)
    return (
        train_forces,
        val_forces,
        train_locations,
        val_locations,
        {"mean": mean.reshape(-1).astype(float).tolist(), "std": std.reshape(-1).astype(float).tolist()},
    )


def _flatten_force_curves(fz: np.ndarray) -> np.ndarray:
    return np.moveaxis(fz, 1, -1).reshape(fz.shape[0], -1, fz.shape[1]).astype(np.float32)


def _flatten_locations(xy: np.ndarray) -> np.ndarray:
    return np.moveaxis(xy, 1, -1).reshape(xy.shape[0], -1, 1, xy.shape[1]).astype(np.float32)


def _apply_force_curve_normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (x - mean) / (std + np.float32(1e-6))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _flatten_features(features: np.ndarray) -> np.ndarray:
    return np.moveaxis(features, 1, -1).reshape(-1, features.shape[1]).astype(np.float32)


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.uint8)
    pos = max(int(y.sum()), 1)
    neg = max(int(y.size - y.sum()), 1)
    out = np.ones(y.shape, dtype=np.float32)
    out[y == 1] = y.size / (2.0 * pos)
    out[y == 0] = y.size / (2.0 * neg)
    return out


def normalize_maps(train_x: np.ndarray, val_x: np.ndarray, *, mode: str) -> tuple[np.ndarray, np.ndarray]:
    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    if mode == "none":
        return np.nan_to_num(train_x), np.nan_to_num(val_x)
    if mode == "sample":
        return _normalize_samplewise(train_x), _normalize_samplewise(val_x)
    if mode == "dataset":
        mean = np.nanmean(train_x, axis=(0, 2, 3), keepdims=True)
        std = np.nanstd(train_x, axis=(0, 2, 3), keepdims=True)
        return _apply_normalize(train_x, mean, std), _apply_normalize(val_x, mean, std)
    raise ValueError("normalize mode must be one of: none, sample, dataset")


def _normalize_samplewise(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=(2, 3), keepdims=True)
    std = np.nanstd(x, axis=(2, 3), keepdims=True)
    return _apply_normalize(x, mean, std)


def _apply_normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (x - mean) / (std + np.float32(1e-6))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class MapTensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, *, augment: bool, seed: int) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y[:, None].astype(np.float32))
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        y = self.y[idx]
        if self.augment:
            k = self.rng.randrange(4)
            if k:
                x = torch.rot90(x, k, dims=(-2, -1))
                y = torch.rot90(y, k, dims=(-2, -1))
            if self.rng.random() < 0.5:
                x = torch.flip(x, dims=(-1,))
                y = torch.flip(y, dims=(-1,))
            if self.rng.random() < 0.5:
                x = torch.flip(x, dims=(-2,))
                y = torch.flip(y, dims=(-2,))
        return x, y


class RimonSequenceDataset(Dataset):
    def __init__(self, forces: np.ndarray, locations: np.ndarray, *, augment_permutation: bool, seed: int) -> None:
        self.forces = torch.from_numpy(forces.astype(np.float32))
        self.locations = torch.from_numpy(locations.astype(np.float32))
        self.augment_permutation = augment_permutation
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return int(self.forces.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        forces = self.forces[idx]
        locations = self.locations[idx]
        if self.augment_permutation:
            perm = torch.randperm(forces.shape[0], generator=self.generator)
            forces = forces[perm]
            locations = locations[perm]
        return forces, locations


class RimonFrequencyEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, max_freq_log2: int = 4) -> None:
        super().__init__()
        if output_dim % (input_dim * 2) != 0:
            raise ValueError(f"output_dim={output_dim} must be divisible by input_dim*2={input_dim * 2}")
        self.input_dim = int(input_dim)
        self.num_freqs = int(output_dim // (input_dim * 2))
        self.register_buffer("freq_bands", 2.0 ** torch.linspace(0, max_freq_log2, self.num_freqs), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for freq in self.freq_bands:
            scaled = x * freq
            pieces.extend([torch.sin(scaled), torch.cos(scaled)])
        return torch.cat(pieces, dim=-1)


class RimonVectorLocationEncoder(nn.Module):
    def __init__(self, vector_size: int, locations_size: int, output_dim: int) -> None:
        super().__init__()
        self.vector_encoder = nn.Linear(vector_size, output_dim)
        self.location_encoder = RimonFrequencyEncoder(locations_size, output_dim, max_freq_log2=4)

    def forward(self, vectors: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        if locations.ndim == vectors.ndim + 1:
            locations = locations.mean(dim=-2)
        return self.vector_encoder(vectors) + self.location_encoder(locations)


class RimonReconstructionModel(nn.Module):
    def __init__(
        self,
        *,
        force_size: int,
        locations_size: int,
        representation_size: int,
        force_location_embed_dim: int,
        decoder_input_embed_dim: int,
        force_predictor_hidden: int,
        input_num_random_samples: int,
        reconstruction_num_random_samples: int,
    ) -> None:
        super().__init__()
        self.representation_size = int(representation_size)
        self.input_num_random_samples = int(input_num_random_samples)
        self.reconstruction_num_random_samples = int(reconstruction_num_random_samples)
        self.force_location_encoder = RimonVectorLocationEncoder(force_size, locations_size, force_location_embed_dim)
        self.encoder = nn.GRU(force_location_embed_dim, representation_size, batch_first=True)
        self.latent_location_encoder = RimonVectorLocationEncoder(representation_size, locations_size, decoder_input_embed_dim)
        self.force_predictor = nn.Sequential(
            nn.Linear(decoder_input_embed_dim, force_predictor_hidden),
            nn.ReLU(),
            nn.Linear(force_predictor_hidden, force_predictor_hidden // 2),
            nn.ReLU(),
            nn.Linear(force_predictor_hidden // 2, force_size),
        )

    def encode(self, forces: torch.Tensor, locations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, steps, force_size = forces.shape
        loc_dim = locations.shape[-1]
        combined = self.force_location_encoder(forces.reshape(b * steps, force_size), locations.reshape(b * steps, 1, loc_dim))
        combined = combined.reshape(b, steps, -1)
        outputs, hidden = self.encoder(combined)
        return outputs, hidden[-1]

    def reconstruction_loss(self, forces: torch.Tensor, locations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representations, _ = self.encode(forces, locations)
        predicted, target = self.sampled_reconstruction(representations, forces, locations)
        return F.mse_loss(predicted, target), predicted

    def sampled_reconstruction(
        self,
        representations: torch.Tensor,
        forces: torch.Tensor,
        locations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, steps, _ = representations.shape
        k_in = min(self.input_num_random_samples, steps)
        k_out = min(self.reconstruction_num_random_samples, steps)
        device = representations.device
        input_steps = torch.randint(0, steps, (b, k_in), device=device)
        reconstruction_steps = torch.randint(0, steps, (b, k_in, k_out), device=device)
        batch_indices = torch.arange(b, device=device)
        source = representations[batch_indices[:, None], input_steps]
        source = source[:, :, None, :].expand(-1, -1, k_out, -1)
        target_locations = locations[batch_indices[:, None, None], reconstruction_steps]
        target_forces = forces[batch_indices[:, None, None], reconstruction_steps]
        flat_features = self.latent_location_encoder(
            source.reshape(-1, source.shape[-1]),
            target_locations.reshape(-1, 1, target_locations.shape[-1]),
        )
        predicted = self.force_predictor(flat_features).reshape(b, k_in, k_out, -1)
        return predicted, target_forces


class RepresentationToMaskDecoder(nn.Module):
    def __init__(self, representation_size: int, *, output_shape: tuple[int, int], base_channels: int = 64) -> None:
        super().__init__()
        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(representation_size, base_channels * 5 * 5),
            nn.GELU(),
            nn.Unflatten(1, (base_channels, 5, 5)),
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, base_channels // 4),
            nn.GELU(),
            nn.Conv2d(base_channels // 4, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if logits.shape[-2:] != self.output_shape:
            logits = F.interpolate(logits, size=self.output_shape, mode="bilinear", align_corners=False)
        return logits


class PointMLP(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_channels
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1)])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        points = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        logits = self.net(points)
        return logits.reshape(b, h, w, 1).permute(0, 3, 1, 2)


class PointCNN1D(nn.Module):
    def __init__(self, in_steps: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        points = x.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        emb = self.net(points).squeeze(-1)
        logits = self.out(emb)
        return logits.reshape(b, h, w, 1).permute(0, 3, 1, 2)


class PointGRU(nn.Module):
    def __init__(self, hidden_size: int = 48) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b * h * w, c, 1)
        _, hidden = self.gru(seq)
        logits = self.out(hidden[-1])
        return logits.reshape(b, h, w, 1).permute(0, 3, 1, 2)


class PointTemporalTransformer(nn.Module):
    def __init__(
        self,
        in_steps: int,
        *,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TemporalTransformerEncoder(
            in_steps,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_cls_token=True,
        )
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        emb = self.encoder(x)
        logits = self.out(emb)
        return logits.reshape(b, h, w, 1).permute(0, 3, 1, 2)


class TemporalTransformerSpatialHead(nn.Module):
    def __init__(
        self,
        in_steps: int,
        *,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TemporalTransformerEncoder(
            in_steps,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_cls_token=True,
        )
        self.head = nn.Sequential(
            nn.Conv2d(d_model, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        emb = self.encoder(x)
        emb_map = emb.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return self.head(emb_map)


class SpatialTokenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        spatial_shape: tuple[int, int],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
        self.token_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.spatial_shape[0] * self.spatial_shape[1], d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        tokens = self.token_proj(tokens)
        tokens = tokens + self._positional_tokens(h, w)
        encoded = self.transformer(tokens)
        logits = self.out(encoded)
        return logits.reshape(b, h, w, 1).permute(0, 3, 1, 2)

    def _positional_tokens(self, h: int, w: int) -> torch.Tensor:
        if (h, w) == self.spatial_shape:
            return self.pos_embed
        base_h, base_w = self.spatial_shape
        pos = self.pos_embed.reshape(1, base_h, base_w, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_steps: int,
        *,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        use_cls_token: bool,
    ) -> None:
        super().__init__()
        self.in_steps = int(in_steps)
        self.use_cls_token = use_cls_token
        self.input_proj = nn.Linear(1, d_model)
        total_tokens = self.in_steps + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.in_steps:
            raise ValueError(f"Expected {self.in_steps} time steps, got {c}")
        seq = x.permute(0, 2, 3, 1).reshape(b * h * w, c, 1)
        tokens = self.input_proj(seq)
        if self.use_cls_token:
            assert self.cls_token is not None
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)
        if self.use_cls_token:
            return encoded[:, 0]
        return encoded.mean(dim=1)


class PatchMLP(nn.Module):
    def __init__(self, in_channels: int, *, patch_size: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        input_dim = in_channels * self.patch_size * self.patch_size
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1)])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        pad = self.patch_size // 2
        patches = F.unfold(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=self.patch_size)
        patches = patches.transpose(1, 2).reshape(b * h * w, -1)
        logits = self.net(patches)
        return logits.reshape(b, h, w, 1).permute(0, 3, 1, 2)


class ShallowCNN(nn.Module):
    def __init__(self, in_channels: int, *, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, base_channels // 2),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_rimon_reconstruction_model(
    model: RimonReconstructionModel,
    train_forces: np.ndarray,
    train_locations: np.ndarray,
    val_forces: np.ndarray,
    val_locations: np.ndarray,
    out_dir: Path,
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = RimonSequenceDataset(train_forces, train_locations, augment_permutation=True, seed=seed)
    val_ds = RimonSequenceDataset(val_forces, val_locations, augment_permutation=False, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without = 0
    rows: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = _run_rimon_reconstruction_epoch(model, train_loader, optimizer, device)
        val_loss = _run_rimon_reconstruction_epoch(model, val_loader, None, device)
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        _write_csv(out_dir / "history.csv", rows, list(row.keys()))
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "config": config,
                    "best_val_mse": best_val,
                    "epoch": epoch,
                },
                out_dir / "best.pt",
            )
            epochs_without = 0
        else:
            epochs_without += 1
        print(f"rimon_gru_pretrain epoch {epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f} elapsed_min={elapsed / 60.0:.2f}")
        if epochs_without >= patience:
            break
    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    summary = {
        "best_val_mse": float(best_val),
        "epochs_requested": int(epochs),
        "epochs_completed": len(rows),
        "checkpoint": str(out_dir / "best.pt"),
        "config": config,
    }
    _write_json(out_dir / "run_config.json", config)
    _write_json(out_dir / "pretrain_summary.json", summary)
    return summary


def _run_rimon_reconstruction_epoch(
    model: RimonReconstructionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for forces, locations in loader:
        forces = forces.to(device)
        locations = locations.to(device)
        with torch.set_grad_enabled(training):
            loss, _pred = model.reconstruction_loss(forces, locations)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch = int(forces.shape[0])
        total += float(loss.detach().cpu()) * batch
        count += batch
    return total / max(count, 1)


def encode_rimon_representations(
    model: RimonReconstructionModel,
    forces: np.ndarray,
    locations: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(RimonSequenceDataset(forces, locations, augment_permutation=False, seed=0), batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for force_batch, location_batch in loader:
            _all_reps, final_rep = model.encode(force_batch.to(device), location_batch.to(device))
            outputs.append(final_rep.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def train_neural_method(
    method: str,
    group: str,
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_names: list[str],
    out_dir: Path,
    *,
    context: dict[str, Any],
    args: argparse.Namespace,
    augment: bool = False,
    focal: bool = False,
) -> None:
    method_dir = out_dir / method
    if _method_done(method_dir, args.force):
        return
    method_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = model.to(device)
    train_ds = MapTensorDataset(x_train, y_train, augment=augment, seed=args.seed)
    val_ds = MapTensorDataset(x_val, y_val, augment=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(y_train.size - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_dice = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without = 0
    rows: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_stats = _run_nn_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight, focal=focal)
        val_stats = _run_nn_epoch(model, val_loader, None, device, pos_weight=pos_weight, focal=focal)
        elapsed = time.perf_counter() - start
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_dice": train_stats["dice"],
            "train_iou": train_stats["iou"],
            "val_loss": val_stats["loss"],
            "val_dice": val_stats["dice"],
            "val_iou": val_stats["iou"],
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        _write_csv(method_dir / "history.csv", rows, list(row.keys()))
        if val_stats["dice"] > best_dice + 1e-4:
            best_dice = val_stats["dice"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "method": method,
                    "group": group,
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
            f"val_dice={val_stats['dice']:.4f} elapsed_min={elapsed / 60.0:.2f}"
        )
        if epochs_without >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    probs = predict_neural(model, x_val, device, batch_size=args.batch_size)
    write_score_method(
        out_dir,
        method,
        group,
        probs,
        y_val,
        val_names,
        config={
            **context,
            "epochs_requested": args.epochs,
            "epochs_completed": len(rows),
            "best_fixed_val_dice": best_dice,
            "augment": augment,
            "focal": focal,
            "checkpoint": str(method_dir / "best.pt"),
        },
        fixed_threshold=0.5,
        force=True,
    )


def _run_nn_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    pos_weight: torch.Tensor,
    focal: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            if focal:
                loss = focal_bce_loss(logits, y, pos_weight=pos_weight) + dice_loss(logits, y)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight) + dice_loss(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        metrics = torch_segmentation_metrics(logits.detach(), y)
        batch = int(x.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["dice"] += metrics["dice"] * batch
        totals["iou"] += metrics["iou"] * batch
        count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor, *, pos_weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    return ((1.0 - pt).pow(gamma) * bce).mean()


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


def predict_neural(model: nn.Module, x: np.ndarray, device: torch.device, *, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            probs.append(torch.sigmoid(logits)[:, 0].cpu().numpy())
    return np.concatenate(probs, axis=0).astype(np.float32)


def best_threshold(scores: np.ndarray, masks: np.ndarray, *, max_thresholds: int = 240) -> dict[str, float | int]:
    thresholds = score_thresholds(scores, max_thresholds=max_thresholds)
    rows = [metrics_from_counts(counts_from_prediction(scores >= threshold, masks), threshold=float(threshold)) for threshold in thresholds]
    return max(rows, key=lambda row: float(row["dice"]))


def score_thresholds(scores: np.ndarray, *, max_thresholds: int = 240) -> np.ndarray:
    scores_1d = np.asarray(scores, dtype=np.float64).reshape(-1)
    unique = np.unique(scores_1d)
    if unique.size <= max_thresholds:
        return unique.astype(np.float64)
    quantiles = np.linspace(0.0, 1.0, max_thresholds)
    return np.unique(np.quantile(scores_1d, quantiles)).astype(np.float64)


def probability_thresholds() -> np.ndarray:
    return np.asarray([round(value, 6) for value in np.arange(0.05, 0.951, 0.05)], dtype=np.float64)


def evaluate_scores(
    scores: np.ndarray,
    masks: np.ndarray,
    *,
    fixed_threshold: float,
    thresholds: np.ndarray | None = None,
) -> EvalResult:
    scores = np.asarray(scores, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.uint8)
    if thresholds is None:
        if float(np.nanmin(scores)) >= -1e-6 and float(np.nanmax(scores)) <= 1.0 + 1e-6:
            thresholds = probability_thresholds()
        else:
            thresholds = score_thresholds(scores)
    threshold_rows = [
        metrics_from_counts(counts_from_prediction(scores >= float(threshold), masks), threshold=float(threshold))
        for threshold in thresholds
    ]
    fixed = metrics_from_counts(counts_from_prediction(scores >= fixed_threshold, masks), threshold=float(fixed_threshold))
    best = max(threshold_rows, key=lambda row: float(row["dice"]))
    per_sample_rows = []
    best_threshold_value = float(best["threshold"])
    fixed_pred = scores >= fixed_threshold
    best_pred = scores >= best_threshold_value
    for idx in range(scores.shape[0]):
        fixed_row = metrics_from_counts(counts_from_prediction(fixed_pred[idx], masks[idx]), threshold=float(fixed_threshold))
        best_row = metrics_from_counts(counts_from_prediction(best_pred[idx], masks[idx]), threshold=best_threshold_value)
        per_sample_rows.append(
            {
                "sample": idx,
                "fixed_dice": fixed_row["dice"],
                "fixed_iou": fixed_row["iou"],
                "best_dice": best_row["dice"],
                "best_iou": best_row["iou"],
                "gt_positive": int(masks[idx].sum()),
            }
        )
    return EvalResult(fixed=fixed, best=best, threshold_rows=threshold_rows, per_sample_rows=per_sample_rows)


def write_score_method(
    out_dir: Path,
    method: str,
    group: str,
    scores: np.ndarray,
    masks: np.ndarray,
    names: list[str],
    *,
    config: dict[str, Any],
    fixed_threshold: float,
    force: bool,
) -> None:
    method_dir = out_dir / method
    if _method_done(method_dir, force):
        return
    method_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_scores(scores, masks, fixed_threshold=fixed_threshold)
    per_sample_rows = []
    for row, name in zip(result.per_sample_rows, names):
        per_sample_rows.append({**row, "sample": name})
    summary = {
        "method": method,
        "group": group,
        "fixed_threshold": result.fixed,
        "threshold_sweep_best": result.best,
        "num_samples": int(masks.shape[0]),
        "config": config,
    }
    np.save(method_dir / "val_scores.npy", np.asarray(scores, dtype=np.float32))
    _write_json(method_dir / "run_config.json", config)
    _write_json(method_dir / "metrics_summary.json", summary)
    _write_json(method_dir / "threshold_sweep.json", {"best": result.best, "thresholds": result.threshold_rows})
    _write_csv(method_dir / "metrics_per_sample.csv", per_sample_rows, list(per_sample_rows[0].keys()))


def write_binary_method(
    out_dir: Path,
    method: str,
    group: str,
    pred: np.ndarray,
    masks: np.ndarray,
    names: list[str],
    *,
    config: dict[str, Any],
    force: bool,
) -> None:
    write_score_method(
        out_dir,
        method,
        group,
        pred.astype(np.float32),
        masks,
        names,
        config={**config, "binary_prediction": True},
        fixed_threshold=0.5,
        force=force,
    )


def per_sample_threshold_predictions(scores: np.ndarray, masks: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(masks, dtype=np.uint8)
    for idx in range(scores.shape[0]):
        threshold = float(best_threshold(scores[idx : idx + 1], masks[idx : idx + 1])["threshold"])
        pred[idx] = scores[idx] >= threshold
    return pred


def fit_morphology_grid(
    select_scores: np.ndarray,
    select_masks: np.ndarray,
    apply_scores: np.ndarray,
    morphology: Any,
    *,
    oracle: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    base_thresholds = [float(best_threshold(select_scores, select_masks)["threshold"])]
    base_thresholds.extend(float(value) for value in np.quantile(select_scores.reshape(-1), [0.80, 0.85, 0.90, 0.92, 0.95]))
    operations = ["none", "closing", "opening", "closing_opening"]
    radii = [1, 2]
    min_sizes = [0, 4, 8, 16]
    fill_holes = [False, True]
    best: tuple[float, dict[str, Any]] | None = None
    for threshold in sorted(set(base_thresholds)):
        binary = select_scores >= threshold
        for operation in operations:
            for radius in radii:
                for min_size in min_sizes:
                    for fill in fill_holes:
                        pred = apply_morphology(binary, morphology, operation=operation, radius=radius, min_size=min_size, fill_holes=fill)
                        dice = float(metrics_from_counts(counts_from_prediction(pred, select_masks))["dice"])
                        cfg = {
                            "threshold": threshold,
                            "operation": operation,
                            "radius": radius,
                            "min_size": min_size,
                            "fill_holes": fill,
                            "selection": "val oracle morphology grid" if oracle else "train morphology grid",
                        }
                        if best is None or dice > best[0]:
                            best = (dice, cfg)
    assert best is not None
    cfg = best[1]
    applied = apply_morphology(
        apply_scores >= float(cfg["threshold"]),
        morphology,
        operation=str(cfg["operation"]),
        radius=int(cfg["radius"]),
        min_size=int(cfg["min_size"]),
        fill_holes=bool(cfg["fill_holes"]),
    )
    return applied.astype(np.uint8), {**cfg, "selection_dice": best[0]}


def apply_morphology(
    binary: np.ndarray,
    morphology: Any,
    *,
    operation: str,
    radius: int,
    min_size: int,
    fill_holes: bool,
) -> np.ndarray:
    out = np.zeros_like(binary, dtype=bool)
    footprint = morphology.disk(radius)
    for idx in range(binary.shape[0]):
        item = binary[idx].astype(bool)
        if operation == "closing":
            item = morphology.closing(item, footprint)
        elif operation == "opening":
            item = morphology.opening(item, footprint)
        elif operation == "closing_opening":
            item = morphology.opening(morphology.closing(item, footprint), footprint)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            if min_size > 0:
                item = morphology.remove_small_objects(item, min_size=min_size)
            if fill_holes:
                item = morphology.remove_small_holes(item, area_threshold=max(min_size, 4))
        out[idx] = item
    return out


def template_shape_prior(
    scores: np.ndarray,
    *,
    fixed_threshold: float,
    morphology: Any,
    measure: Any,
    draw: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = apply_morphology(scores >= fixed_threshold, morphology, operation="closing_opening", radius=1, min_size=4, fill_holes=True)
    pred = np.zeros_like(base, dtype=np.uint8)
    chosen_counts: dict[str, int] = {"ellipse": 0, "rectangle": 0, "hull": 0}
    for idx in range(base.shape[0]):
        labels = measure.label(base[idx])
        for region in measure.regionprops(labels):
            if region.area < 3:
                continue
            component = labels == region.label
            candidates = _template_candidates(component, draw, morphology)
            name, candidate = max(candidates, key=lambda item: _binary_iou(item[1], component))
            chosen_counts[name] += 1
            pred[idx] |= candidate.astype(np.uint8)
    return pred, {"source": "equivalent_stiffness", "threshold": fixed_threshold, "templates": chosen_counts}


def _template_candidates(component: np.ndarray, draw: Any, morphology: Any) -> list[tuple[str, np.ndarray]]:
    rows, cols = np.where(component)
    h, w = component.shape
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())
    rect = np.zeros_like(component, dtype=bool)
    rect[min_r : max_r + 1, min_c : max_c + 1] = True
    rr, cc = draw.ellipse(
        (min_r + max_r) / 2.0,
        (min_c + max_c) / 2.0,
        max((max_r - min_r + 1) / 2.0, 1.0),
        max((max_c - min_c + 1) / 2.0, 1.0),
        shape=(h, w),
    )
    ellipse = np.zeros_like(component, dtype=bool)
    ellipse[rr, cc] = True
    hull = morphology.convex_hull_image(component)
    return [("ellipse", ellipse), ("rectangle", rect), ("hull", hull)]


def _binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / union)


def counts_from_prediction(pred: np.ndarray, target: np.ndarray) -> dict[str, int]:
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    return {
        "tp": int(np.logical_and(pred_bool, target_bool).sum()),
        "tn": int(np.logical_and(~pred_bool, ~target_bool).sum()),
        "fp": int(np.logical_and(pred_bool, ~target_bool).sum()),
        "fn": int(np.logical_and(~pred_bool, target_bool).sum()),
    }


def metrics_from_counts(counts: dict[str, int], *, threshold: float | None = None) -> dict[str, float | int]:
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    eps = 1e-8
    row: dict[str, float | int] = {
        "pixel_accuracy": float((tp + tn) / max(tp + tn + fp + fn, eps)),
        "precision": float(tp / max(tp + fp, eps)),
        "recall": float(tp / max(tp + fn, eps)),
        "dice": float(2.0 * tp / max(2 * tp + fp + fn, eps)),
        "iou": float(tp / max(tp + fp + fn, eps)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if threshold is not None:
        row = {"threshold": float(threshold), **row}
    return row


def write_leaderboard(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(out_dir.glob("*/metrics_summary.json")):
        summary = _read_json(metrics_path)
        fixed = summary.get("fixed_threshold", {})
        best = summary.get("threshold_sweep_best", {})
        method = summary.get("method", metrics_path.parent.name)
        rows.append(
            {
                "method": method,
                "group": summary.get("group", ""),
                "meaning": method_meaning(str(method)),
                "fixed_dice": fixed.get("dice"),
                "fixed_iou": fixed.get("iou"),
                "fixed_precision": fixed.get("precision"),
                "fixed_recall": fixed.get("recall"),
                "fixed_threshold": fixed.get("threshold"),
                "best_dice": best.get("dice"),
                "best_iou": best.get("iou"),
                "best_precision": best.get("precision"),
                "best_recall": best.get("recall"),
                "best_threshold": best.get("threshold"),
                "path": str(metrics_path.parent),
            }
        )
    rows.sort(key=lambda row: float(row["best_dice"] or -1.0), reverse=True)
    if rows:
        _write_csv(out_dir / "leaderboard.csv", rows, list(rows[0].keys()))
    baseline = _read_json(BASELINE_UNET_METRICS) if BASELINE_UNET_METRICS.exists() else {}
    lines = ["# Segmentation Accuracy Sweep", ""]
    if baseline:
        best = baseline.get("threshold_sweep_best", {})
        lines.append(
            f"- Existing U-Net reference: Dice={best.get('dice', baseline.get('dice'))}, "
            f"IoU={best.get('iou', baseline.get('iou'))}, threshold={best.get('threshold', baseline.get('threshold'))}"
        )
    if rows:
        lines.append(f"- Best sweep method: `{rows[0]['method']}` Dice={rows[0]['best_dice']} IoU={rows[0]['best_iou']}")
        lines.append("")
        lines.append("| rank | method | group | meaning | best Dice | best IoU | fixed Dice |")
        lines.append("|---:|---|---|---|---:|---:|---:|")
        for rank, row in enumerate(rows[:20], start=1):
            lines.append(
                f"| {rank} | `{row['method']}` | {row['group']} | {row['meaning']} | "
                f"{_fmt(row['best_dice'])} | {_fmt(row['best_iou'])} | {_fmt(row['fixed_dice'])} |"
            )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def method_meaning(method: str) -> str:
    return METHOD_MEANINGS.get(method, "")


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _method_done(method_dir: Path, force: bool) -> bool:
    return method_dir.joinpath("metrics_summary.json").exists() and not force


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _parse_groups(value: str) -> set[str]:
    groups = {GROUP_ALIASES.get(part.strip(), part.strip()) for part in value.split(",") if part.strip()}
    unknown = sorted(groups - set(METHOD_GROUPS))
    if unknown:
        raise SystemExit(f"Unknown group(s): {unknown}; choices are {METHOD_GROUPS}")
    return groups


def _load_sklearn() -> dict[str, Any]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.mixture import GaussianMixture
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit("scikit-learn is required. Install/update the palpation env from environment.yml.") from exc
    return {
        "KMeans": KMeans,
        "GaussianMixture": GaussianMixture,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def _load_skimage() -> tuple[Any, Any, Any, Any]:
    try:
        from skimage import draw, filters, measure, morphology
    except ModuleNotFoundError as exc:
        raise SystemExit("scikit-image is required in the palpation env.") from exc
    return filters, morphology, measure, draw


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
