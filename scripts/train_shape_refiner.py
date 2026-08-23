from __future__ import annotations

import argparse
import csv
import json
import math
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

from palpation_sim.workflow import require_runtime_environment  # noqa: E402
from run_highres_fz_temporal_variant_sweep import build_variant as build_temporal_variant  # noqa: E402
from run_highres_segmentation_sweep import build_inputs as build_sweep_inputs  # noqa: E402
from run_highres_segmentation_sweep import build_model as build_sweep_model  # noqa: E402
from run_highres_segmentation_sweep import load_split  # noqa: E402
from run_segmentation_accuracy_sweep import predict_neural  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "runs/nonlinear_trajectory_20x_repeats10_seed20260618/"
    "highres_fz_temporal_variants/r128_fz_features_aug_focal_unet"
)
DEFAULT_DATA_ROOT = Path(
    "data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/"
    "random_trajectory/data"
)
DEFAULT_OUT_DIR = Path("runs/shape_refiner_r128_fz_features_aug_focal_20260625")


@dataclass
class SplitPayload:
    names: list[str]
    scores: np.ndarray
    masks: np.ndarray


class ShapeRefinerDataset(Dataset):
    def __init__(
        self,
        scores: np.ndarray,
        masks: np.ndarray,
        *,
        baseline_threshold: float,
        augment: bool,
        seed: int,
    ) -> None:
        self.scores = scores.astype(np.float32)
        self.masks = masks.astype(np.float32)
        self.baseline_threshold = float(baseline_threshold)
        self.augment = bool(augment)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score = self.scores[idx]
        mask = self.masks[idx]
        cond = condition_array(score[None], self.baseline_threshold)[0]
        target = mask[None]
        base_logit = logit_np(score)[None]
        if self.augment:
            code = self.rng.randrange(8)
            cond = d4_transform_chw(cond, code)
            target = d4_transform_chw(target, code)
            base_logit = d4_transform_chw(base_logit, code)
        return (
            torch.from_numpy(cond.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(base_logit.astype(np.float32)),
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShapeRefinerUNet(nn.Module):
    def __init__(self, in_channels: int, *, base_channels: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        b = int(base_channels)
        self.residual_scale = float(residual_scale)
        self.inc = ConvBlock(in_channels, b, dropout)
        self.down1 = ConvBlock(b, b * 2, dropout)
        self.down2 = ConvBlock(b * 2, b * 4, dropout)
        self.down3 = ConvBlock(b * 4, b * 4, dropout)
        self.mid = ConvBlock(b * 4, b * 4, dropout)
        self.up2 = ConvBlock(b * 8, b * 2, dropout)
        self.up1 = ConvBlock(b * 4, b, dropout)
        self.up0 = ConvBlock(b * 2, b, dropout)
        self.out = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, base_logit: torch.Tensor | None = None, *, mode: str = "residual") -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.down1(F.avg_pool2d(x0, 2))
        x2 = self.down2(F.avg_pool2d(x1, 2))
        x3 = self.down3(F.avg_pool2d(x2, 2))
        h = self.mid(x3)
        h = F.interpolate(h, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up2(torch.cat([h, x2], dim=1))
        h = F.interpolate(h, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up1(torch.cat([h, x1], dim=1))
        h = F.interpolate(h, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up0(torch.cat([h, x0], dim=1))
        raw = self.out(h)
        if mode == "direct":
            return raw
        if base_logit is None:
            raise ValueError("base_logit is required for residual mode")
        return base_logit + self.residual_scale * torch.tanh(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised probability-to-shape refiner for high-res masks.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--variant", type=str, default="fz_features_aug_focal_unet")
    parser.add_argument(
        "--train-score-path",
        type=Path,
        default=None,
        help="Optional precomputed train probability map cache. Overrides model recomputation.",
    )
    parser.add_argument("--baseline-threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", choices=["residual", "direct"], default="residual")
    parser.add_argument(
        "--group-aggregation",
        choices=["none", "mean", "median", "logit_mean", "trimmed_mean", "p60", "p70"],
        default="none",
        help="Aggregate probability maps across samples sharing base_phantom_index before training/evaluation.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=4.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--force-score-cache", action="store_true")
    parser.add_argument("--max-visual-samples", type=int, default=10)
    args = parser.parse_args()

    require_runtime_environment()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    baseline_threshold = args.baseline_threshold or read_baseline_threshold(args.run_dir / "test_metrics_summary.json")
    started = time.perf_counter()

    print(f"device: {device}", flush=True)
    print("loading score/mask payloads", flush=True)
    train, val, test = load_payloads(args, device=device)
    sample_input = condition_array(train.scores[:1], baseline_threshold)
    model = ShapeRefinerUNet(
        sample_input.shape[1],
        base_channels=args.base_channels,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)

    train_loader = make_loader(train, args, baseline_threshold, augment=args.augment)
    val_loader = make_loader(val, args, baseline_threshold, augment=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([negative_positive_ratio(train.masks)], dtype=torch.float32, device=device)

    best_dice = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without = 0
    history: list[dict[str, float | int]] = []
    print("training shape refiner", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight, args=args)
        val_stats = run_epoch(model, val_loader, None, device, pos_weight=pos_weight, args=args)
        elapsed = time.perf_counter() - started
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
        history.append(row)
        write_csv(args.out_dir / "history.csv", history)
        print(
            f"epoch {epoch:03d} train_loss={train_stats['loss']:.4f} train_dice={train_stats['dice']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_dice={val_stats['dice']:.4f} elapsed_min={elapsed / 60.0:.2f}",
            flush=True,
        )
        if val_stats["dice"] > best_dice + 1e-4:
            best_dice = val_stats["dice"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "config": jsonable_config(args),
                    "baseline_threshold": baseline_threshold,
                    "best_val_dice": best_dice,
                    "input_channels": int(sample_input.shape[1]),
                },
                args.out_dir / "best.pt",
            )
            epochs_without = 0
        else:
            epochs_without += 1
        if epochs_without >= args.patience:
            print(f"early stop after epoch {epoch:03d}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    print("predicting validation/test refined probabilities", flush=True)
    val_refined = predict_refiner(model, val, baseline_threshold, args, device)
    test_refined = predict_refiner(model, test, baseline_threshold, args, device)
    np.save(args.out_dir / "val_refined_scores.npy", val_refined.astype(np.float32))
    np.save(args.out_dir / "test_refined_scores.npy", test_refined.astype(np.float32))

    refiner_threshold = tune_threshold(val_refined, val.masks)
    fusion_choice = tune_fusion(val.scores, val_refined, val.masks)
    test_baseline = metrics(test.scores >= baseline_threshold, test.masks)
    test_refiner = metrics(test_refined >= float(refiner_threshold["threshold"]), test.masks)
    fused_test_scores = fuse_scores(test.scores, test_refined, float(fusion_choice["alpha"]))
    test_fusion = metrics(fused_test_scores >= float(fusion_choice["threshold"]), test.masks)
    test_fusion_oracle = tune_fusion(test.scores, test_refined, test.masks)

    val_baseline = metrics(val.scores >= baseline_threshold, val.masks)
    val_refiner_metrics = metrics(val_refined >= float(refiner_threshold["threshold"]), val.masks)
    val_fusion = metrics(
        fuse_scores(val.scores, val_refined, float(fusion_choice["alpha"])) >= float(fusion_choice["threshold"]),
        val.masks,
    )
    per_sample_rows = write_per_sample(
        args.out_dir / "metrics_per_sample.csv",
        test.names,
        test.scores,
        test_refined,
        fused_test_scores,
        test.masks,
        baseline_threshold=baseline_threshold,
        refiner_threshold=float(refiner_threshold["threshold"]),
        fusion_threshold=float(fusion_choice["threshold"]),
    )
    summary = {
        "experiment": "supervised_shape_refiner",
        "interpretation": (
            "Supervised probability-to-mask shape refiner. Validation chooses refined-score threshold "
            "and probability/refiner fusion; test labels are only used for reporting."
        ),
        "run_dir": str(args.run_dir),
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "baseline_threshold": baseline_threshold,
        "train_samples": len(train.names),
        "val_samples": len(val.names),
        "test_samples": len(test.names),
        "config": jsonable_config(args),
        "best_val_dice": best_dice,
        "refiner_threshold_selection": refiner_threshold,
        "fusion_selection": fusion_choice,
        "val_metrics": {
            "baseline": val_baseline,
            "refiner": val_refiner_metrics,
            "fusion": val_fusion,
        },
        "test_metrics": {
            "baseline": test_baseline,
            "refiner": test_refiner,
            "fusion_val_selected": test_fusion,
            "fusion_test_oracle": test_fusion_oracle,
        },
        "delta_vs_baseline": {
            "refiner_dice": test_refiner["dice"] - test_baseline["dice"],
            "fusion_val_selected_dice": test_fusion["dice"] - test_baseline["dice"],
            "fusion_test_oracle_dice": test_fusion_oracle["dice"] - test_baseline["dice"],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.out_dir / "metrics_summary.json", summary)
    write_report(args.out_dir / "REPORT.md", summary, per_sample_rows)
    write_visuals(
        args.out_dir,
        test,
        test_refined,
        fused_test_scores,
        per_sample_rows,
        baseline_threshold,
        refiner_threshold=float(refiner_threshold["threshold"]),
        fusion_threshold=float(fusion_choice["threshold"]),
        max_visual_samples=args.max_visual_samples,
    )
    print(json.dumps(summary["test_metrics"], indent=2), flush=True)
    print(f"shape refiner complete: {args.out_dir}", flush=True)


def load_payloads(args: argparse.Namespace, *, device: torch.device) -> tuple[SplitPayload, SplitPayload, SplitPayload]:
    train_split = load_split(args.data_root / "train", label_size=args.resolution, max_samples=positive_or_none(args.max_train_samples))
    val_split = load_split(args.data_root / "val", label_size=args.resolution, max_samples=positive_or_none(args.max_eval_samples))
    test_split = load_split(args.data_root / "test", label_size=args.resolution, max_samples=positive_or_none(args.max_eval_samples))
    train_scores = load_or_compute_train_scores(args, train_split, val_split, test_split, device=device)
    val_scores = load_scores(args.run_dir / "val_scores.npy", max_samples=args.max_eval_samples)
    test_scores = load_scores(args.run_dir / "test_scores.npy", max_samples=args.max_eval_samples)
    if args.group_aggregation != "none":
        train_scores = aggregate_by_base_phantom(args.data_root / "train", train_split.names, train_scores, args.group_aggregation)
        val_scores = aggregate_by_base_phantom(args.data_root / "val", val_split.names, val_scores, args.group_aggregation)
        test_scores = aggregate_by_base_phantom(args.data_root / "test", test_split.names, test_scores, args.group_aggregation)
    return (
        SplitPayload(train_split.names, train_scores, train_split.masks.astype(np.uint8)),
        SplitPayload(val_split.names, val_scores, val_split.masks.astype(np.uint8)),
        SplitPayload(test_split.names, test_scores, test_split.masks.astype(np.uint8)),
    )


def aggregate_by_base_phantom(data_dir: Path, names: Sequence[str], scores: np.ndarray, mode: str) -> np.ndarray:
    ids: list[int] = []
    for name in names:
        with np.load(data_dir / name) as sample:
            ids.append(int(np.asarray(sample["base_phantom_index"]).reshape(())))
    id_array = np.asarray(ids, dtype=np.int64)
    output = np.empty_like(scores)
    for base_id in sorted(set(ids)):
        idx = np.flatnonzero(id_array == int(base_id))
        stack = scores[idx]
        if mode == "mean":
            aggregate = stack.mean(axis=0)
        elif mode == "median":
            aggregate = np.median(stack, axis=0)
        elif mode == "logit_mean":
            aggregate = sigmoid_np(logit_np(stack).mean(axis=0))
        elif mode == "trimmed_mean":
            ordered = np.sort(stack, axis=0)
            aggregate = ordered[1:-1].mean(axis=0) if ordered.shape[0] > 2 else ordered.mean(axis=0)
        elif mode == "p60":
            aggregate = np.quantile(stack, 0.60, axis=0)
        elif mode == "p70":
            aggregate = np.quantile(stack, 0.70, axis=0)
        else:
            raise ValueError(f"Unsupported group aggregation: {mode}")
        output[idx] = aggregate.astype(np.float32)
    return output.astype(np.float32)


def load_or_compute_train_scores(args: argparse.Namespace, train_split: Any, val_split: Any, test_split: Any, *, device: torch.device) -> np.ndarray:
    if args.train_score_path is not None:
        print(f"loading explicit train scores: {args.train_score_path}", flush=True)
        return load_scores(args.train_score_path, max_samples=args.max_train_samples)

    cache_dir = args.out_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"train_scores_{args.variant}_r{args.resolution}.npy"
    if cache_path.exists() and not args.force_score_cache:
        return load_scores(cache_path, max_samples=args.max_train_samples)

    print("computing train U-Net probability cache", flush=True)
    run_config = read_json(args.run_dir / "run_config.json")
    variant_args = argparse.Namespace(
        fz_normalize=run_config.get("fz_normalize", "dataset"),
        feature_normalize=run_config.get("feature_normalize", "dataset"),
        stiffness_normalize=run_config.get("stiffness_normalize", "dataset"),
        seed=int(run_config.get("seed", args.seed)),
        limited_trajectory_seed=run_config.get("limited_trajectory_seed", None),
        trajectory_input_steps=int(run_config.get("trajectory_input_steps", 0)),
        positional_embedding_dim=int(run_config.get("positional_embedding_dim", 8)),
    )
    batch = build_train_score_batch(args.variant, args.resolution, train_split, val_split, test_split, variant_args)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu")
    batch.model.load_state_dict(checkpoint["model_state"])
    scores = predict_neural(batch.model.to(device), batch.x_train, device, batch_size=args.batch_size)
    np.save(cache_path, scores.astype(np.float32))
    return scores.astype(np.float32)


def build_train_score_batch(
    variant: str,
    resolution: int,
    train_split: Any,
    val_split: Any,
    test_split: Any,
    variant_args: argparse.Namespace,
) -> Any:
    try:
        return build_temporal_variant(variant, resolution, train_split, val_split, test_split, variant_args)
    except ValueError as exc:
        if "Unknown variant" not in str(exc):
            raise

    input_name, model_name = parse_segmentation_sweep_variant(variant)
    inputs = build_sweep_inputs(
        train_split,
        val_split,
        test_split,
        variant_args,
        limited_trajectory_seed=int(variant_args.limited_trajectory_seed or variant_args.seed),
    )
    if input_name not in inputs:
        raise ValueError(f"Unsupported segmentation-sweep input in variant {variant!r}: {input_name!r}")
    x_train, x_val, x_test = inputs[input_name]
    output_shape = tuple(int(v) for v in train_split.masks.shape[-2:])
    model = build_sweep_model(
        model_name,
        input_shape=tuple(int(v) for v in x_train.shape[1:]),
        output_shape=output_shape,
        args=variant_args,
        hidden_dims=(512, 512),
    )
    return argparse.Namespace(model=model, x_train=x_train, x_val=x_val, x_test=x_test)


def parse_segmentation_sweep_variant(variant: str) -> tuple[str, str]:
    for model_name in ("shallow_cnn", "unet", "mlp"):
        suffix = f"_{model_name}"
        if variant.endswith(suffix):
            return variant[: -len(suffix)], model_name
    raise ValueError(f"Cannot parse segmentation-sweep variant name: {variant}")


def make_loader(split: SplitPayload, args: argparse.Namespace, baseline_threshold: float, *, augment: bool) -> DataLoader:
    ds = ShapeRefinerDataset(split.scores, split.masks, baseline_threshold=baseline_threshold, augment=augment, seed=args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=augment, generator=generator)


def condition_array(scores: np.ndarray, baseline_threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    hard = (scores >= float(baseline_threshold)).astype(np.float32)
    margin = np.clip((scores - float(baseline_threshold)) / max(float(baseline_threshold), 1e-6), -1.0, 1.0)
    logit = logit_np(scores) / 8.0
    h, w = scores.shape[-2:]
    yy, xx = np.meshgrid(np.linspace(-1, 1, h, dtype=np.float32), np.linspace(-1, 1, w, dtype=np.float32), indexing="ij")
    xx = np.broadcast_to(xx, scores.shape)
    yy = np.broadcast_to(yy, scores.shape)
    return np.stack([scores, hard, margin, logit, xx, yy], axis=1).astype(np.float32)


def logit_np(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(scores, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(scores / (1.0 - scores)).astype(np.float32)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float32)))).astype(np.float32)


def d4_transform_chw(array: np.ndarray, code: int) -> np.ndarray:
    out = np.asarray(array)
    if code & 1:
        out = np.flip(out, axis=-1)
    if code & 2:
        out = np.flip(out, axis=-2)
    if code & 4:
        out = np.swapaxes(out, -1, -2)
    return np.ascontiguousarray(out)


def run_epoch(
    model: ShapeRefinerUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    pos_weight: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    count = 0
    for cond, target, base_logit in loader:
        cond = cond.to(device)
        target = target.to(device)
        base_logit = base_logit.to(device)
        with torch.set_grad_enabled(training):
            logits = model(cond, base_logit, mode=args.mode)
            loss = refiner_loss(
                logits,
                target,
                pos_weight=pos_weight,
                focal_gamma=args.focal_gamma,
                dice_weight=args.dice_weight,
                boundary_weight=args.boundary_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch = int(target.shape[0])
        row = torch_metrics(logits.detach(), target)
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["dice"] += row["dice"] * batch
        totals["iou"] += row["iou"] * batch
        count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def refiner_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
    focal_gamma: float,
    dice_weight: float,
    boundary_weight: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, probs, 1.0 - probs)
    focal = ((1.0 - pt).pow(float(focal_gamma)) * bce).mean()
    dice = dice_loss_from_probs(probs, target)
    if boundary_weight <= 0:
        boundary = torch.tensor(0.0, device=logits.device)
    else:
        boundary = F.l1_loss(avg_gradient(probs), avg_gradient(target))
    return focal + float(dice_weight) * dice + float(boundary_weight) * boundary


def dice_loss_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * target, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def avg_gradient(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(torch.abs(x[..., :, 1:] - x[..., :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(x[..., 1:, :] - x[..., :-1, :]), (0, 0, 0, 1))
    return 0.5 * (dx + dy)


def torch_metrics(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    dims = tuple(range(1, pred.ndim))
    inter = torch.sum(pred * target, dim=dims)
    union = torch.sum((pred + target) > 0, dim=dims).float()
    pred_sum = torch.sum(pred, dim=dims)
    target_sum = torch.sum(target, dim=dims)
    dice = ((2 * inter + eps) / (pred_sum + target_sum + eps)).mean()
    iou = ((inter + eps) / (union + eps)).mean()
    return {"dice": float(dice.cpu()), "iou": float(iou.cpu())}


def predict_refiner(
    model: ShapeRefinerUNet,
    split: SplitPayload,
    baseline_threshold: float,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    cond = torch.from_numpy(condition_array(split.scores, baseline_threshold))
    base = torch.from_numpy(logit_np(split.scores)[:, None])
    loader = DataLoader(torch.utils.data.TensorDataset(cond, base), batch_size=args.batch_size, shuffle=False)
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for cond_batch, base_batch in loader:
            logits = model(cond_batch.to(device), base_batch.to(device), mode=args.mode)
            outs.append(torch.sigmoid(logits)[:, 0].cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def tune_threshold(scores: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for threshold in threshold_grid():
        row = {"threshold": float(threshold), **metrics(scores >= float(threshold), masks)}
        if best is None or float(row["dice"]) > float(best["dice"]):
            best = row
    assert best is not None
    return best


def tune_fusion(base_scores: np.ndarray, refined_scores: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for alpha in np.linspace(0.0, 1.0, 21):
        fused = fuse_scores(base_scores, refined_scores, float(alpha))
        for threshold in threshold_grid():
            row = {"alpha": float(alpha), "threshold": float(threshold), **metrics(fused >= float(threshold), masks)}
            if best is None or float(row["dice"]) > float(best["dice"]):
                best = row
    assert best is not None
    return best


def threshold_grid() -> np.ndarray:
    return np.round(np.linspace(0.05, 0.95, 37), 4)


def fuse_scores(base_scores: np.ndarray, refined_scores: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - float(alpha)) * base_scores + float(alpha) * refined_scores).astype(np.float32)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred_bool = np.asarray(pred).astype(bool)
    gt_bool = np.asarray(gt).astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    eps = 1e-8
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    dice = 2.0 * tp / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    pixel_accuracy = (tp + tn) / max(tp + tn + fp + fn, eps)
    return {
        "pixel_accuracy": float(pixel_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_per_sample(
    path: Path,
    names: Sequence[str],
    base_scores: np.ndarray,
    refined_scores: np.ndarray,
    fused_scores: np.ndarray,
    masks: np.ndarray,
    *,
    baseline_threshold: float,
    refiner_threshold: float,
    fusion_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        baseline = metrics(base_scores[idx] >= baseline_threshold, masks[idx])
        refined = metrics(refined_scores[idx] >= refiner_threshold, masks[idx])
        fusion = metrics(fused_scores[idx] >= fusion_threshold, masks[idx])
        rows.append(
            {
                "sample": name,
                "baseline_dice": baseline["dice"],
                "refiner_dice": refined["dice"],
                "fusion_dice": fusion["dice"],
                "refiner_delta_dice": refined["dice"] - baseline["dice"],
                "fusion_delta_dice": fusion["dice"] - baseline["dice"],
                "baseline_iou": baseline["iou"],
                "refiner_iou": refined["iou"],
                "fusion_iou": fusion["iou"],
                "gt_positive": int(masks[idx].sum()),
                "baseline_positive": int((base_scores[idx] >= baseline_threshold).sum()),
                "refiner_positive": int((refined_scores[idx] >= refiner_threshold).sum()),
                "fusion_positive": int((fused_scores[idx] >= fusion_threshold).sum()),
            }
        )
    write_csv(path, rows)
    return rows


def write_visuals(
    out_dir: Path,
    test: SplitPayload,
    refined_scores: np.ndarray,
    fused_scores: np.ndarray,
    rows: Sequence[dict[str, Any]],
    baseline_threshold: float,
    *,
    refiner_threshold: float,
    fusion_threshold: float,
    max_visual_samples: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["baseline", "refiner", "fusion"]
    values = [
        metrics(test.scores >= baseline_threshold, test.masks)["dice"],
        metrics(refined_scores >= refiner_threshold, test.masks)["dice"],
        metrics(fused_scores >= fusion_threshold, test.masks)["dice"],
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(labels, values, color=["#3b82f6", "#10b981", "#f97316"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice")
    ax.set_title("Supervised Shape Refiner")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.015, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_dir / "dice_bar.png", dpi=180)
    plt.close(fig)

    deltas = np.asarray([float(row["fusion_delta_dice"]) for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.hist(deltas, bins=30, color="#64748b", edgecolor="white")
    ax.axvline(0.0, color="#111827", linewidth=1.2)
    ax.set_xlabel("Fusion Dice Delta vs Baseline")
    ax.set_ylabel("Samples")
    ax.set_title("Shape Refiner Fusion Delta")
    fig.tight_layout()
    fig.savefig(out_dir / "fusion_delta_hist.png", dpi=180)
    plt.close(fig)

    selected = select_visual_rows(rows, max_visual_samples)
    if not selected:
        return
    cols = ["probability", "baseline", "refiner", "fusion", "ground truth"]
    fig, axes = plt.subplots(len(selected), len(cols), figsize=(2.3 * len(cols), 2.15 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_idx, sample_idx in enumerate(selected):
        images = [
            test.scores[sample_idx],
            test.scores[sample_idx] >= baseline_threshold,
            refined_scores[sample_idx],
            fused_scores[sample_idx],
            test.masks[sample_idx],
        ]
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="viridis" if col_idx in {0, 2, 3} else "gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=9)
            if col_idx == 0:
                row = rows[sample_idx]
                ax.set_ylabel(
                    f"{test.names[sample_idx]}\n"
                    f"B {float(row['baseline_dice']):.3f} F {float(row['fusion_dice']):.3f}",
                    fontsize=8,
                )
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_contact_sheet.png", dpi=180)
    plt.close(fig)


def select_visual_rows(rows: Sequence[dict[str, Any]], max_rows: int) -> list[int]:
    if max_rows <= 0:
        return []
    order = np.argsort([float(row["fusion_delta_dice"]) for row in rows])
    picks = order[-max_rows // 2 :][::-1].tolist() + order[: max_rows // 2 + 1].tolist()
    unique: list[int] = []
    for idx in picks:
        if int(idx) not in unique:
            unique.append(int(idx))
    return unique[:max_rows]


def write_report(path: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    test = summary["test_metrics"]
    delta = summary["delta_vs_baseline"]
    best = sorted(rows, key=lambda row: float(row["fusion_delta_dice"]), reverse=True)[:5]
    worst = sorted(rows, key=lambda row: float(row["fusion_delta_dice"]))[:5]
    lines = [
        "# Supervised Shape Refiner",
        "",
        "This run trains a probability-to-mask shape refiner on baseline U-Net probability maps.",
        "Validation selects the refined-score threshold and probability/refiner fusion; test labels are only used for reporting.",
        "",
        "## Test Dice",
        "",
        f"- Baseline: {test['baseline']['dice']:.6f}",
        f"- Refiner only: {test['refiner']['dice']:.6f} ({delta['refiner_dice']:+.6f})",
        f"- Fusion, validation-selected: {test['fusion_val_selected']['dice']:.6f} ({delta['fusion_val_selected_dice']:+.6f})",
        f"- Fusion, test oracle: {test['fusion_test_oracle']['dice']:.6f} ({delta['fusion_test_oracle_dice']:+.6f})",
        "",
        "## Validation Selection",
        "",
        f"- Refiner threshold: {summary['refiner_threshold_selection']['threshold']}",
        f"- Fusion alpha: {summary['fusion_selection']['alpha']}",
        f"- Fusion threshold: {summary['fusion_selection']['threshold']}",
        f"- Fusion validation Dice: {summary['fusion_selection']['dice']:.6f}",
        "",
        "## Largest Fusion Improvements",
        "",
    ]
    lines.extend(format_rows(best))
    lines.extend(["", "## Largest Fusion Regressions", ""])
    lines.extend(format_rows(worst))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_rows(rows: Sequence[dict[str, Any]]) -> list[str]:
    return [
        f"- {row['sample']}: baseline={float(row['baseline_dice']):.4f}, "
        f"fusion={float(row['fusion_dice']):.4f}, delta={float(row['fusion_delta_dice']):+.4f}"
        for row in rows
    ]


def load_scores(path: Path, *, max_samples: int) -> np.ndarray:
    scores = np.asarray(np.load(path), dtype=np.float32)
    if scores.ndim == 4 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if max_samples and max_samples > 0:
        scores = scores[:max_samples]
    return np.clip(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)


def negative_positive_ratio(masks: np.ndarray) -> float:
    pos = max(float(masks.sum()), 1.0)
    neg = max(float(masks.size - masks.sum()), 1.0)
    return neg / pos


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, int(channels))
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def read_baseline_threshold(path: Path) -> float:
    summary = read_json(path)
    return float(summary.get("val_selected_threshold", {}).get("threshold", 0.5))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def positive_or_none(value: int) -> int | None:
    return int(value) if int(value) > 0 else None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
