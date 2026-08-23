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
from run_highres_segmentation_sweep import build_inputs, load_split  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    "data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/"
    "random_trajectory/data"
)


@dataclass
class SplitPayload:
    names: list[str]
    x: np.ndarray
    masks: np.ndarray
    slots: np.ndarray


class ProjectionSlotDataset(Dataset):
    def __init__(self, split: SplitPayload, *, augment: bool, seed: int) -> None:
        self.x = split.x.astype(np.float32)
        self.masks = split.masks.astype(np.float32)
        self.slots = split.slots.astype(np.float32)
        self.augment = bool(augment)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        mask = self.masks[idx][None]
        slots = self.slots[idx]
        if self.augment:
            code = self.rng.randrange(8)
            x = d4_transform_chw(x, code)
            mask = d4_transform_chw(mask, code)
            slots = transform_slots(slots, code)
        return torch.from_numpy(x.copy()), torch.from_numpy(mask.copy()), torch.from_numpy(slots.copy())


class ParametricProjectionNet(nn.Module):
    def __init__(self, in_channels: int, *, max_slots: int, base_channels: int, dropout: float) -> None:
        super().__init__()
        self.max_slots = int(max_slots)
        c = int(base_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(c), c),
            nn.SiLU(),
            ConvBlock(c),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(c * 2), c * 2),
            nn.SiLU(),
            ConvBlock(c * 2),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(c * 4), c * 4),
            nn.SiLU(),
            ConvBlock(c * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(c * 4, c * 4),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(c * 4, self.max_slots * 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.encoder(x))
        return raw.view(x.shape[0], self.max_slots, 7)


class ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a parametric primitive-slot projection prior for low-information palpation segmentation."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--input-mode", type=str, default="stiffness_random_pair")
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max-slots", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--slot-loss-weight", type=float, default=1.5)
    parser.add_argument("--object-loss-weight", type=float, default=0.5)
    parser.add_argument("--sharpness", type=float, default=28.0)
    parser.add_argument("--min-radius", type=float, default=0.015)
    parser.add_argument("--max-radius", type=float, default=0.24)
    parser.add_argument("--fz-normalize", type=str, default=None)
    parser.add_argument("--stiffness-normalize", type=str, default=None)
    parser.add_argument("--trajectory-input-steps", type=int, default=None)
    parser.add_argument("--limited-trajectory-seed", type=int, default=None)
    parser.add_argument("--positional-embedding-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-visual-samples", type=int, default=8)
    args = parser.parse_args()

    require_runtime_environment()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    started = time.perf_counter()

    if args.baseline_run_dir is not None:
        hydrate_config_from_baseline(args, args.baseline_run_dir / "run_config.json")
    train, val, test = load_payloads(args)
    model = ParametricProjectionNet(
        int(train.x.shape[1]),
        max_slots=args.max_slots,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    train_loader = make_loader(train, args, augment=True)
    val_loader = make_loader(val, args, augment=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    grid = make_grid(args.resolution, device)

    history: list[dict[str, float | int]] = []
    best_dice = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without = 0
    print(f"device: {device}", flush=True)
    print(f"input_mode={args.input_mode} train={len(train.names)} val={len(val.names)} test={len(test.names)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, grid, args, optimizer=optimizer, device=device)
        val_stats = run_epoch(model, val_loader, grid, args, optimizer=None, device=device)
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        write_csv(args.out_dir / "history.csv", history)
        print(
            f"epoch {epoch:03d} train_loss={train_stats['loss']:.4f} train_dice={train_stats['dice']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_dice={val_stats['dice']:.4f} elapsed_min={elapsed/60.0:.2f}",
            flush=True,
        )
        if val_stats["dice"] > best_dice + 1e-4:
            best_dice = val_stats["dice"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "config": jsonable_config(args),
                    "best_val_dice_0p5": best_dice,
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

    val_scores, val_slots = predict_scores(model, val, grid, args, device)
    test_scores, test_slots = predict_scores(model, test, grid, args, device)
    np.save(args.out_dir / "val_parametric_scores.npy", val_scores.astype(np.float32))
    np.save(args.out_dir / "test_parametric_scores.npy", test_scores.astype(np.float32))
    np.save(args.out_dir / "test_predicted_slots.npy", test_slots.astype(np.float32))

    threshold_choice = tune_threshold(val_scores, val.masks)
    test_parametric = metrics(test_scores >= float(threshold_choice["threshold"]), test.masks)
    test_oracle = tune_threshold(test_scores, test.masks)
    val_parametric = metrics(val_scores >= float(threshold_choice["threshold"]), val.masks)
    baseline = load_baseline_metrics(args, test.masks)
    per_sample_rows = write_per_sample(
        args.out_dir / "metrics_per_sample.csv",
        test.names,
        test_scores,
        test.masks,
        threshold=float(threshold_choice["threshold"]),
        baseline_scores=baseline.get("scores"),
        baseline_threshold=baseline.get("threshold"),
    )
    write_visuals(
        args.out_dir,
        test,
        test_scores,
        per_sample_rows,
        threshold=float(threshold_choice["threshold"]),
        baseline_scores=baseline.get("scores"),
        baseline_threshold=baseline.get("threshold"),
        max_visual_samples=args.max_visual_samples,
    )

    summary = {
        "experiment": "parametric_projection_shape_prior",
        "interpretation": (
            "The model predicts low-dimensional primitive slots from palpation input and projects them to a 2D mask. "
            "This makes the generated-shape prior part of the model rather than a post-processing step."
        ),
        "data_root": str(args.data_root),
        "input_mode": args.input_mode,
        "baseline_run_dir": str(args.baseline_run_dir) if args.baseline_run_dir else None,
        "train_samples": len(train.names),
        "val_samples": len(val.names),
        "test_samples": len(test.names),
        "config": jsonable_config(args),
        "threshold_selection": threshold_choice,
        "val_metrics": {"parametric_projection": val_parametric},
        "test_metrics": {
            "baseline_val_selected": baseline.get("metrics"),
            "parametric_projection_val_selected": test_parametric,
            "parametric_projection_test_oracle": test_oracle,
        },
        "delta_vs_baseline": (
            {"dice": test_parametric["dice"] - baseline["metrics"]["dice"], "iou": test_parametric["iou"] - baseline["metrics"]["iou"]}
            if baseline.get("metrics") is not None
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.out_dir / "metrics_summary.json", summary)
    write_report(args.out_dir, summary, per_sample_rows)
    print(json.dumps(summary["test_metrics"], indent=2), flush=True)
    print(f"parametric projection prior complete: {args.out_dir}", flush=True)


def hydrate_config_from_baseline(args: argparse.Namespace, config_path: Path) -> None:
    if not config_path.exists():
        return
    config = read_json(config_path)
    if args.fz_normalize is None:
        args.fz_normalize = str(config.get("fz_normalize", "dataset"))
    if args.stiffness_normalize is None:
        args.stiffness_normalize = str(config.get("stiffness_normalize", "dataset"))
    if args.trajectory_input_steps is None:
        args.trajectory_input_steps = int(config.get("trajectory_input_steps", 0))
    if args.limited_trajectory_seed is None:
        args.limited_trajectory_seed = int(config.get("limited_trajectory_seed", config.get("seed", args.seed)))
    if args.positional_embedding_dim is None:
        args.positional_embedding_dim = int(config.get("positional_embedding_dim", 8))


def load_payloads(args: argparse.Namespace) -> tuple[SplitPayload, SplitPayload, SplitPayload]:
    split_args = argparse.Namespace(
        fz_normalize=args.fz_normalize or "dataset",
        stiffness_normalize=args.stiffness_normalize or "dataset",
        trajectory_input_steps=int(args.trajectory_input_steps if args.trajectory_input_steps is not None else 20),
        positional_embedding_dim=int(args.positional_embedding_dim if args.positional_embedding_dim is not None else 8),
    )
    random_pair_seed = int(args.seed)
    if args.baseline_run_dir is not None:
        config = read_json(args.baseline_run_dir / "run_config.json")
        random_pair_seed = int(config.get("stiffness_random_seed", config.get("seed", args.seed)))
    train_raw = load_split(
        args.data_root / "train",
        label_size=args.resolution,
        max_samples=positive_or_none(args.max_train_samples),
        random_pair_seed=random_pair_seed,
    )
    val_raw = load_split(
        args.data_root / "val",
        label_size=args.resolution,
        max_samples=positive_or_none(args.max_eval_samples),
        random_pair_seed=random_pair_seed,
    )
    test_raw = load_split(
        args.data_root / "test",
        label_size=args.resolution,
        max_samples=positive_or_none(args.max_eval_samples),
        random_pair_seed=random_pair_seed,
    )
    inputs = build_inputs(
        train_raw,
        val_raw,
        test_raw,
        split_args,
        limited_trajectory_seed=int(args.limited_trajectory_seed if args.limited_trajectory_seed is not None else args.seed),
    )
    if args.input_mode not in inputs:
        raise ValueError(f"Unknown input mode {args.input_mode!r}; available: {sorted(inputs)}")
    train_x, val_x, test_x = inputs[args.input_mode]
    return (
        SplitPayload(train_raw.names, train_x, train_raw.masks.astype(np.uint8), load_slots(args.data_root / "train", train_raw.names, args.max_slots)),
        SplitPayload(val_raw.names, val_x, val_raw.masks.astype(np.uint8), load_slots(args.data_root / "val", val_raw.names, args.max_slots)),
        SplitPayload(test_raw.names, test_x, test_raw.masks.astype(np.uint8), load_slots(args.data_root / "test", test_raw.names, args.max_slots)),
    )


def load_slots(data_dir: Path, names: Sequence[str], max_slots: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for name in names:
        with np.load(data_dir / name, allow_pickle=False) as sample:
            lumps = json.loads(str(sample["lumps_json"].item()))
            xy = np.asarray(sample["xy"], dtype=np.float32)
        x_min, x_max = float(np.nanmin(xy[..., 0])), float(np.nanmax(xy[..., 0]))
        y_min, y_max = float(np.nanmin(xy[..., 1])), float(np.nanmax(xy[..., 1]))
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        slots = np.zeros((int(max_slots), 7), dtype=np.float32)
        parsed: list[tuple[float, np.ndarray]] = []
        for lump in lumps:
            center = lump["center"]
            radii = lump["radii"]
            cx = np.clip((float(center[0]) - x_min) / x_span, 0.0, 1.0)
            cy = np.clip((float(center[1]) - y_min) / y_span, 0.0, 1.0)
            rx = np.clip(float(radii[0]) / x_span, 0.005, 0.45)
            ry = np.clip(float(radii[1]) / y_span, 0.005, 0.45)
            yaw = float(lump.get("yaw", 0.0))
            slot = np.asarray([1.0, cx, cy, rx, ry, math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            parsed.append((cx + 0.01 * cy, slot))
        for idx, (_key, slot) in enumerate(sorted(parsed, key=lambda item: item[0])[: int(max_slots)]):
            slots[idx] = slot
        rows.append(slots)
    return np.stack(rows).astype(np.float32)


def make_loader(split: SplitPayload, args: argparse.Namespace, *, augment: bool) -> DataLoader:
    ds = ProjectionSlotDataset(split, augment=augment, seed=args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=augment, generator=generator)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    grid: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mask_loss": 0.0, "dice_loss": 0.0, "slot_loss": 0.0, "object_loss": 0.0, "dice": 0.0}
    count = 0
    for x, mask, target_slots in loader:
        x = x.to(device)
        mask = mask.to(device)
        target_slots = target_slots.to(device)
        with torch.set_grad_enabled(training):
            raw_slots = model(x)
            pred = decode_slots(raw_slots, args)
            mask_prob = render_slots(pred, grid, sharpness=float(args.sharpness))
            losses = compute_losses(mask_prob, raw_slots, pred, target_slots, mask, args)
            loss = losses["loss"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch = int(x.shape[0])
        count += batch
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu()) * batch
        totals["dice"] += float(batch_dice(mask_prob >= 0.5, mask >= 0.5).detach().cpu()) * batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def compute_losses(
    mask_prob: torch.Tensor,
    raw_slots: torch.Tensor,
    pred: dict[str, torch.Tensor],
    target_slots: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    eps = 1e-6
    mask_prob = mask_prob.clamp(eps, 1.0 - eps)
    mask_loss = F.binary_cross_entropy(mask_prob, mask)
    dice_loss = soft_dice_loss(mask_prob, mask)
    target_obj = target_slots[..., 0]
    object_loss = F.binary_cross_entropy_with_logits(raw_slots[..., 0], target_obj)
    active = target_obj > 0.5
    if active.any():
        target_params = target_slots[..., 1:]
        pred_params = torch.cat([pred["center"], pred["radii"], pred["angle"]], dim=-1)
        slot_loss = F.smooth_l1_loss(pred_params[active], target_params[active])
    else:
        slot_loss = raw_slots.sum() * 0.0
    total = (
        float(args.mask_loss_weight) * mask_loss
        + float(args.dice_loss_weight) * dice_loss
        + float(args.slot_loss_weight) * slot_loss
        + float(args.object_loss_weight) * object_loss
    )
    return {
        "loss": total,
        "mask_loss": mask_loss,
        "dice_loss": dice_loss,
        "slot_loss": slot_loss,
        "object_loss": object_loss,
    }


def decode_slots(raw_slots: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    obj_logit = raw_slots[..., 0]
    center = torch.sigmoid(raw_slots[..., 1:3])
    radius_range = float(args.max_radius) - float(args.min_radius)
    radii = float(args.min_radius) + radius_range * torch.sigmoid(raw_slots[..., 3:5])
    angle_raw = raw_slots[..., 5:7]
    angle = F.normalize(angle_raw, dim=-1, eps=1e-6)
    return {"object_logit": obj_logit, "object_prob": torch.sigmoid(obj_logit), "center": center, "radii": radii, "angle": angle}


def render_slots(pred: dict[str, torch.Tensor], grid: tuple[torch.Tensor, torch.Tensor], *, sharpness: float) -> torch.Tensor:
    yy, xx = grid
    cx = pred["center"][..., 0][..., None, None]
    cy = pred["center"][..., 1][..., None, None]
    rx = pred["radii"][..., 0][..., None, None].clamp_min(1e-4)
    ry = pred["radii"][..., 1][..., None, None].clamp_min(1e-4)
    cos = pred["angle"][..., 0][..., None, None]
    sin = pred["angle"][..., 1][..., None, None]
    dx = xx[None, None] - cx
    dy = yy[None, None] - cy
    x_rot = cos * dx + sin * dy
    y_rot = -sin * dx + cos * dy
    level = (x_rot / rx) ** 2 + (y_rot / ry) ** 2
    primitive = torch.sigmoid((1.0 - level) * float(sharpness))
    primitive = primitive * pred["object_prob"][..., None, None]
    union = 1.0 - torch.prod(1.0 - primitive.clamp(0.0, 1.0), dim=1, keepdim=True)
    return union.clamp(0.0, 1.0)


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    split: SplitPayload,
    grid: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(ProjectionSlotDataset(split, augment=False, seed=args.seed), batch_size=args.batch_size, shuffle=False)
    scores: list[np.ndarray] = []
    slots: list[np.ndarray] = []
    for x, _mask, _target_slots in loader:
        raw = model(x.to(device))
        pred = decode_slots(raw, args)
        mask_prob = render_slots(pred, grid, sharpness=float(args.sharpness))
        scores.append(mask_prob[:, 0].detach().cpu().numpy().astype(np.float32))
        slot_array = torch.cat(
            [
                pred["object_prob"][..., None],
                pred["center"],
                pred["radii"],
                pred["angle"],
            ],
            dim=-1,
        )
        slots.append(slot_array.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(scores, axis=0), np.concatenate(slots, axis=0)


def load_baseline_metrics(args: argparse.Namespace, masks: np.ndarray) -> dict[str, Any]:
    if args.baseline_run_dir is None:
        return {"metrics": None, "scores": None, "threshold": None}
    score_path = args.baseline_run_dir / "test_scores.npy"
    summary_path = args.baseline_run_dir / "test_metrics_summary.json"
    threshold = float(read_json(summary_path).get("val_selected_threshold", {}).get("threshold", 0.5))
    scores = np.asarray(np.load(score_path), dtype=np.float32)
    if scores.ndim == 4 and scores.shape[1] == 1:
        scores = scores[:, 0]
    scores = scores[: masks.shape[0]]
    return {"metrics": metrics(scores >= threshold, masks), "scores": scores, "threshold": threshold}


def tune_threshold(scores: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for threshold in np.linspace(0.05, 0.95, 37, dtype=np.float32):
        row = {"threshold": float(threshold), **metrics(scores >= float(threshold), masks)}
        if best is None or float(row["dice"]) > float(best["dice"]):
            best = row
    assert best is not None
    return best


def metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> dict[str, float | int]:
    pred_bool = np.asarray(pred).astype(bool)
    target_bool = np.asarray(target).astype(bool)
    tp = int(np.logical_and(pred_bool, target_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~target_bool).sum())
    fp = int(np.logical_and(pred_bool, ~target_bool).sum())
    fn = int(np.logical_and(~pred_bool, target_bool).sum())
    dice = 2.0 * tp / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    return {
        "pixel_accuracy": float((tp + tn) / max(tp + tn + fp + fn, eps)),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def soft_dice_loss(scores: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    reduce_dims = tuple(range(1, scores.ndim))
    inter = (scores * target).sum(dim=reduce_dims)
    denom = scores.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def batch_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()
    reduce_dims = tuple(range(1, pred.ndim))
    inter = (pred_f * target_f).sum(dim=reduce_dims)
    denom = pred_f.sum(dim=reduce_dims) + target_f.sum(dim=reduce_dims)
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def make_grid(resolution: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(0.0, 1.0, int(resolution), device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return yy, xx


def transform_slots(slots: np.ndarray, code: int) -> np.ndarray:
    # Keep slot supervision stable for flips/rotations in normalized [0, 1] image coordinates.
    out = slots.copy()
    active = out[:, 0] > 0.5
    if not active.any():
        return out
    cx = out[:, 1].copy()
    cy = out[:, 2].copy()
    cos = out[:, 5].copy()
    sin = out[:, 6].copy()
    if code & 1:
        out[:, 1] = 1.0 - cx
        out[:, 5] = -cos
    if code & 2:
        out[:, 2] = 1.0 - cy
        out[:, 6] = -sin
    if code & 4:
        out[:, 1] = cy
        out[:, 2] = cx
        out[:, 3], out[:, 4] = out[:, 4].copy(), out[:, 3].copy()
        out[:, 5] = sin
        out[:, 6] = cos
    return out


def d4_transform_chw(array: np.ndarray, code: int) -> np.ndarray:
    out = np.asarray(array)
    if code & 1:
        out = np.flip(out, axis=-1)
    if code & 2:
        out = np.flip(out, axis=-2)
    if code & 4:
        out = np.swapaxes(out, -1, -2)
    return np.ascontiguousarray(out)


def write_per_sample(
    path: Path,
    names: Sequence[str],
    scores: np.ndarray,
    masks: np.ndarray,
    *,
    threshold: float,
    baseline_scores: np.ndarray | None,
    baseline_threshold: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        row: dict[str, Any] = {"sample": name, "parametric_dice": metrics(scores[idx] >= threshold, masks[idx])["dice"]}
        if baseline_scores is not None and baseline_threshold is not None:
            base = metrics(baseline_scores[idx] >= baseline_threshold, masks[idx])
            row["baseline_dice"] = base["dice"]
            row["delta_dice"] = float(row["parametric_dice"]) - float(base["dice"])
        rows.append(row)
    write_csv(path, rows)
    return rows


def write_visuals(
    out_dir: Path,
    split: SplitPayload,
    scores: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    baseline_scores: np.ndarray | None,
    baseline_threshold: float | None,
    max_visual_samples: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"visualization skipped: {exc}", flush=True)
        return

    count = max(1, min(int(max_visual_samples), len(rows)))
    if rows and "delta_dice" in rows[0]:
        order = np.argsort([float(row["delta_dice"]) for row in rows])
        selected = (order[-count // 2 :][::-1].tolist() + order[: count - count // 2].tolist())[:count]
    else:
        selected = list(range(count))
    cols = ["GT", "Parametric score", "Parametric mask"]
    if baseline_scores is not None:
        cols.insert(1, "Baseline mask")
    fig, axes = plt.subplots(len(selected), len(cols), figsize=(2.3 * len(cols), 2.1 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]
    for r, idx in enumerate(selected):
        images: list[tuple[str, np.ndarray]] = [("GT", split.masks[idx])]
        if baseline_scores is not None and baseline_threshold is not None:
            images.append(("Baseline mask", baseline_scores[idx] >= baseline_threshold))
        images.extend([("Parametric score", scores[idx]), ("Parametric mask", scores[idx] >= threshold)])
        for c, (title, image) in enumerate(images):
            ax = axes[r, c]
            ax.imshow(image, cmap="magma" if "score" in title.lower() else "gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
        label = split.names[idx]
        if "delta_dice" in rows[idx]:
            label += f"\nD {float(rows[idx]['delta_dice']):+.3f}"
        axes[r, 0].set_ylabel(label, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative_examples.png", dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    baseline = summary["test_metrics"].get("baseline_val_selected")
    parametric = summary["test_metrics"]["parametric_projection_val_selected"]
    lines = [
        "# Parametric Projection Shape Prior",
        "",
        summary["interpretation"],
        "",
        "## Test Dice",
    ]
    if baseline is not None:
        delta = summary["delta_vs_baseline"]["dice"]
        lines.append(f"- Baseline: {baseline['dice']:.6f}")
        lines.append(f"- Parametric projection: {parametric['dice']:.6f} ({delta:+.6f})")
    else:
        lines.append(f"- Parametric projection: {parametric['dice']:.6f}")
    lines.extend(
        [
            f"- Selected threshold: {summary['threshold_selection']['threshold']}",
            "",
            "## Largest Per-Sample Gains",
        ]
    )
    if rows and "delta_dice" in rows[0]:
        best = sorted(rows, key=lambda row: float(row["delta_dice"]), reverse=True)[:5]
        worst = sorted(rows, key=lambda row: float(row["delta_dice"]))[:5]
        for row in best:
            lines.append(
                f"- {row['sample']}: baseline={float(row['baseline_dice']):.4f}, "
                f"parametric={float(row['parametric_dice']):.4f}, delta={float(row['delta_dice']):+.4f}"
            )
        lines.extend(["", "## Largest Regressions"])
        for row in worst:
            lines.append(
                f"- {row['sample']}: baseline={float(row['baseline_dice']):.4f}, "
                f"parametric={float(row['parametric_dice']):.4f}, delta={float(row['delta_dice']):+.4f}"
            )
    write_text(out_dir / "REPORT.md", "\n".join(lines) + "\n")


def positive_or_none(value: int) -> int | None:
    return int(value) if value and value > 0 else None


def group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while channels % groups != 0:
        groups -= 1
    return groups


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
