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
from torch.utils.data import DataLoader, TensorDataset

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
DEFAULT_OUT_DIR = Path(
    "runs/conditional_shape_diffusion_r128_fz_features_aug_focal_20260625"
)


@dataclass
class SplitPayload:
    names: list[str]
    scores: np.ndarray
    masks: np.ndarray


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * freq[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = group_norm(out_channels)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = F.silu(self.norm1(h))
        h = self.conv2(h)
        h = F.silu(self.norm2(h))
        return h


class ConditionalDenoiseUNet(nn.Module):
    def __init__(self, in_channels: int, *, base_channels: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        b = int(base_channels)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.inc = TimeConvBlock(in_channels, b, time_dim)
        self.down1 = TimeConvBlock(b, b * 2, time_dim)
        self.down2 = TimeConvBlock(b * 2, b * 4, time_dim)
        self.down3 = TimeConvBlock(b * 4, b * 4, time_dim)
        self.mid = TimeConvBlock(b * 4, b * 4, time_dim)
        self.up2 = TimeConvBlock(b * 8, b * 2, time_dim)
        self.up1 = TimeConvBlock(b * 4, b, time_dim)
        self.up0 = TimeConvBlock(b * 2, b, time_dim)
        self.out = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        x0 = self.inc(x, time_emb)
        x1 = self.down1(F.avg_pool2d(x0, kernel_size=2), time_emb)
        x2 = self.down2(F.avg_pool2d(x1, kernel_size=2), time_emb)
        x3 = self.down3(F.avg_pool2d(x2, kernel_size=2), time_emb)
        xm = self.mid(x3, time_emb)
        h = F.interpolate(xm, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up2(torch.cat([h, x2], dim=1), time_emb)
        h = F.interpolate(h, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up1(torch.cat([h, x1], dim=1), time_emb)
        h = F.interpolate(h, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up0(torch.cat([h, x0], dim=1), time_emb)
        return self.out(h)


class DiffusionSchedule:
    def __init__(self, steps: int, device: torch.device) -> None:
        self.steps = int(steps)
        betas = cosine_beta_schedule(self.steps).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conditional DDPM shape prior over 128x128 segmentation masks.")
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
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--sample-init", choices=["condition", "noise"], default="condition")
    parser.add_argument("--x0-loss-weight", type=float, default=0.15)
    parser.add_argument("--num-samples", type=int, default=1, help="Number of DDIM samples averaged at evaluation.")
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
    print(f"device: {device}", flush=True)

    baseline_threshold = args.baseline_threshold or read_baseline_threshold(args.run_dir / "test_metrics_summary.json")
    started = time.perf_counter()

    print("loading score/mask payloads", flush=True)
    train, val, test = load_payloads(args, device=device, baseline_threshold=baseline_threshold)
    condition_channels = 2
    model = ConditionalDenoiseUNet(condition_channels + 1, base_channels=args.base_channels).to(device)
    schedule = DiffusionSchedule(args.diffusion_steps, device)

    train_loader = make_loader(train, baseline_threshold, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val, baseline_threshold, args.batch_size, shuffle=False, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("training conditional diffusion prior", flush=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without = 0
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            schedule,
            optimizer=optimizer,
            device=device,
            x0_loss_weight=args.x0_loss_weight,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            schedule,
            optimizer=None,
            device=device,
            x0_loss_weight=args.x0_loss_weight,
        )
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_noise_mse": train_stats["noise_mse"],
            "train_x0_mse": train_stats["x0_mse"],
            "val_loss": val_stats["loss"],
            "val_noise_mse": val_stats["noise_mse"],
            "val_x0_mse": val_stats["x0_mse"],
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        write_csv(args.out_dir / "history.csv", history)
        print(
            f"epoch {epoch:03d} train_loss={train_stats['loss']:.5f} "
            f"val_loss={val_stats['loss']:.5f} elapsed_min={elapsed/60.0:.2f}",
            flush=True,
        )
        if val_stats["loss"] < best_loss - 1e-4:
            best_loss = val_stats["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "config": jsonable_config(args),
                    "baseline_threshold": baseline_threshold,
                    "condition_channels": condition_channels,
                    "best_val_loss": best_loss,
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

    print("sampling validation and test shape priors", flush=True)
    val_prior_scores = sample_split(model, val, baseline_threshold, schedule, args, device)
    test_prior_scores = sample_split(model, test, baseline_threshold, schedule, args, device)
    np.save(args.out_dir / "val_shape_prior_scores.npy", val_prior_scores.astype(np.float32))
    np.save(args.out_dir / "test_shape_prior_scores.npy", test_prior_scores.astype(np.float32))

    prior_threshold = tune_threshold(val_prior_scores, val.masks)
    fusion_choice = tune_fusion(val.scores, val_prior_scores, val.masks)
    test_baseline = metrics(test.scores >= baseline_threshold, test.masks)
    test_prior = metrics(test_prior_scores >= float(prior_threshold["threshold"]), test.masks)
    fused_test_scores = fuse_scores(test.scores, test_prior_scores, float(fusion_choice["alpha"]))
    test_fusion = metrics(fused_test_scores >= float(fusion_choice["threshold"]), test.masks)
    test_fusion_oracle = tune_fusion(test.scores, test_prior_scores, test.masks)

    val_baseline = metrics(val.scores >= baseline_threshold, val.masks)
    val_prior = metrics(val_prior_scores >= float(prior_threshold["threshold"]), val.masks)
    val_fusion = metrics(
        fuse_scores(val.scores, val_prior_scores, float(fusion_choice["alpha"])) >= float(fusion_choice["threshold"]),
        val.masks,
    )
    per_sample_rows = write_per_sample(
        args.out_dir / "metrics_per_sample.csv",
        test.names,
        test.scores,
        test_prior_scores,
        fused_test_scores,
        test.masks,
        baseline_threshold=baseline_threshold,
        prior_threshold=float(prior_threshold["threshold"]),
        fusion_threshold=float(fusion_choice["threshold"]),
    )

    summary = {
        "experiment": "conditional_diffusion_shape_prior",
        "interpretation": (
            "Conditional DDPM trained on train-set masks with existing U-Net probabilities as condition. "
            "Validation chooses prior threshold and probability/prior fusion; test labels are used only for reporting."
        ),
        "run_dir": str(args.run_dir),
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "baseline_threshold": baseline_threshold,
        "train_samples": len(train.names),
        "val_samples": len(val.names),
        "test_samples": len(test.names),
        "config": jsonable_config(args),
        "best_val_loss": best_loss,
        "prior_threshold_selection": prior_threshold,
        "fusion_selection": fusion_choice,
        "val_metrics": {
            "baseline": val_baseline,
            "shape_prior": val_prior,
            "fusion": val_fusion,
        },
        "test_metrics": {
            "baseline": test_baseline,
            "shape_prior": test_prior,
            "fusion_val_selected": test_fusion,
            "fusion_test_oracle": test_fusion_oracle,
        },
        "delta_vs_baseline": {
            "shape_prior_dice": test_prior["dice"] - test_baseline["dice"],
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
        test_prior_scores,
        fused_test_scores,
        per_sample_rows,
        baseline_threshold,
        prior_threshold=float(prior_threshold["threshold"]),
        fusion_threshold=float(fusion_choice["threshold"]),
        args=args,
    )
    print(json.dumps(summary["test_metrics"], indent=2), flush=True)
    print(f"conditional diffusion prior complete: {args.out_dir}", flush=True)


def load_payloads(args: argparse.Namespace, *, device: torch.device, baseline_threshold: float) -> tuple[SplitPayload, SplitPayload, SplitPayload]:
    del baseline_threshold
    train_split = load_split(args.data_root / "train", label_size=args.resolution, max_samples=positive_or_none(args.max_train_samples))
    val_split = load_split(args.data_root / "val", label_size=args.resolution, max_samples=positive_or_none(args.max_eval_samples))
    test_split = load_split(args.data_root / "test", label_size=args.resolution, max_samples=positive_or_none(args.max_eval_samples))
    train_scores = load_or_compute_train_scores(args, train_split, val_split, test_split, device=device)
    val_scores = load_scores(args.run_dir / "val_scores.npy", max_samples=args.max_eval_samples)
    test_scores = load_scores(args.run_dir / "test_scores.npy", max_samples=args.max_eval_samples)
    return (
        SplitPayload(train_split.names, train_scores, train_split.masks.astype(np.uint8)),
        SplitPayload(val_split.names, val_scores, val_split.masks.astype(np.uint8)),
        SplitPayload(test_split.names, test_scores, test_split.masks.astype(np.uint8)),
    )


def load_or_compute_train_scores(
    args: argparse.Namespace,
    train_split: Any,
    val_split: Any,
    test_split: Any,
    *,
    device: torch.device,
) -> np.ndarray:
    if args.train_score_path is not None:
        print(f"loading explicit train scores: {args.train_score_path}", flush=True)
        return load_scores(args.train_score_path, max_samples=args.max_train_samples)

    cache_dir = args.out_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"train_scores_{args.variant}_r{args.resolution}.npy"
    if cache_path.exists() and not args.force_score_cache:
        print(f"loading cached train scores: {cache_path}", flush=True)
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


def load_scores(path: Path, *, max_samples: int) -> np.ndarray:
    scores = np.asarray(np.load(path), dtype=np.float32)
    if scores.ndim == 4 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if scores.ndim != 3:
        raise ValueError(f"Expected scores [N,H,W], got {scores.shape} from {path}")
    scores = np.clip(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if max_samples and max_samples > 0:
        scores = scores[:max_samples]
    return scores.astype(np.float32)


def make_loader(
    split: SplitPayload,
    baseline_threshold: float,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    cond = condition_array(split.scores, baseline_threshold)
    masks = split.masks[:, None].astype(np.float32)
    generator = torch.Generator()
    generator.manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(cond), torch.from_numpy(masks))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def condition_array(scores: np.ndarray, baseline_threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    hard = (scores >= float(baseline_threshold)).astype(np.float32)
    return np.stack([scores, hard], axis=1).astype(np.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    x0_loss_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "noise_mse": 0.0, "x0_mse": 0.0}
    count = 0
    for cond, mask in loader:
        cond = cond.to(device)
        x0 = (mask.to(device) * 2.0) - 1.0
        batch = int(x0.shape[0])
        t = torch.randint(0, schedule.steps, (batch,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        sqrt_ab = schedule.sqrt_alpha_bars[t].view(batch, 1, 1, 1)
        sqrt_om = schedule.sqrt_one_minus_alpha_bars[t].view(batch, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_om * noise
        model_input = torch.cat([x_t, cond], dim=1)
        with torch.set_grad_enabled(training):
            eps_pred = model(model_input, t)
            noise_mse = F.mse_loss(eps_pred, noise)
            x0_pred = (x_t - sqrt_om * eps_pred) / torch.clamp(sqrt_ab, min=1e-6)
            x0_mse = F.mse_loss(torch.clamp(x0_pred, -1.5, 1.5), x0)
            loss = noise_mse + float(x0_loss_weight) * x0_mse
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["noise_mse"] += float(noise_mse.detach().cpu()) * batch
        totals["x0_mse"] += float(x0_mse.detach().cpu()) * batch
        count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def sample_split(
    model: nn.Module,
    split: SplitPayload,
    baseline_threshold: float,
    schedule: DiffusionSchedule,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    cond_np = condition_array(split.scores, baseline_threshold)
    loader = DataLoader(torch.from_numpy(cond_np), batch_size=args.batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1009)
    model.eval()
    with torch.no_grad():
        for cond in loader:
            cond = cond.to(device)
            sample_sum = torch.zeros((cond.shape[0], 1, cond.shape[-2], cond.shape[-1]), device=device)
            for _ in range(max(int(args.num_samples), 1)):
                x = initial_sample_noise(cond, schedule, args, generator)
                x0 = ddim_sample(model, x, cond, schedule, sample_steps=args.sample_steps)
                sample_sum += torch.clamp((x0 + 1.0) * 0.5, 0.0, 1.0)
            sample = sample_sum / float(max(int(args.num_samples), 1))
            outputs.append(sample[:, 0].cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def initial_sample_noise(
    cond: torch.Tensor,
    schedule: DiffusionSchedule,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn((cond.shape[0], 1, cond.shape[-2], cond.shape[-1]), device=cond.device, generator=generator)
    if args.sample_init == "noise":
        return noise
    x0_cond = (cond[:, 1:2] * 2.0) - 1.0
    t = schedule.steps - 1
    return schedule.sqrt_alpha_bars[t] * x0_cond + schedule.sqrt_one_minus_alpha_bars[t] * noise


def ddim_sample(
    model: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor,
    schedule: DiffusionSchedule,
    *,
    sample_steps: int,
) -> torch.Tensor:
    step_count = max(2, min(int(sample_steps), schedule.steps))
    timesteps = torch.linspace(schedule.steps - 1, 0, step_count, device=x.device).round().long()
    for idx, t_scalar in enumerate(timesteps):
        t = torch.full((x.shape[0],), int(t_scalar.item()), device=x.device, dtype=torch.long)
        eps = model(torch.cat([x, cond], dim=1), t)
        ab_t = schedule.alpha_bars[t].view(x.shape[0], 1, 1, 1)
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_om_t = torch.sqrt(1.0 - ab_t)
        x0 = (x - sqrt_om_t * eps) / torch.clamp(sqrt_ab_t, min=1e-6)
        x0 = torch.clamp(x0, -1.5, 1.5)
        if idx == len(timesteps) - 1:
            x = x0
        else:
            next_t = timesteps[idx + 1]
            ab_next = schedule.alpha_bars[next_t].view(1, 1, 1, 1)
            x = torch.sqrt(ab_next) * x0 + torch.sqrt(1.0 - ab_next) * eps
    return x


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float32) / float(steps)
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, min=1e-5, max=0.02)


def tune_threshold(scores: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for threshold in threshold_grid():
        row = {"threshold": float(threshold), **metrics(scores >= float(threshold), masks)}
        if best is None or float(row["dice"]) > float(best["dice"]):
            best = row
    assert best is not None
    return best


def tune_fusion(base_scores: np.ndarray, prior_scores: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for alpha in np.linspace(0.0, 1.0, 21):
        fused = fuse_scores(base_scores, prior_scores, float(alpha))
        for threshold in threshold_grid():
            row = {"alpha": float(alpha), "threshold": float(threshold), **metrics(fused >= float(threshold), masks)}
            if best is None or float(row["dice"]) > float(best["dice"]):
                best = row
    assert best is not None
    return best


def threshold_grid() -> np.ndarray:
    return np.round(np.linspace(0.05, 0.95, 37), 4)


def fuse_scores(base_scores: np.ndarray, prior_scores: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - float(alpha)) * base_scores + float(alpha) * prior_scores).astype(np.float32)


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
    prior_scores: np.ndarray,
    fused_scores: np.ndarray,
    masks: np.ndarray,
    *,
    baseline_threshold: float,
    prior_threshold: float,
    fusion_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        baseline = metrics(base_scores[idx] >= baseline_threshold, masks[idx])
        prior = metrics(prior_scores[idx] >= prior_threshold, masks[idx])
        fusion = metrics(fused_scores[idx] >= fusion_threshold, masks[idx])
        rows.append(
            {
                "sample": name,
                "baseline_dice": baseline["dice"],
                "shape_prior_dice": prior["dice"],
                "fusion_dice": fusion["dice"],
                "shape_prior_delta_dice": prior["dice"] - baseline["dice"],
                "fusion_delta_dice": fusion["dice"] - baseline["dice"],
                "baseline_iou": baseline["iou"],
                "shape_prior_iou": prior["iou"],
                "fusion_iou": fusion["iou"],
                "gt_positive": int(masks[idx].sum()),
                "baseline_positive": int((base_scores[idx] >= baseline_threshold).sum()),
                "shape_prior_positive": int((prior_scores[idx] >= prior_threshold).sum()),
                "fusion_positive": int((fused_scores[idx] >= fusion_threshold).sum()),
            }
        )
    write_csv(path, rows)
    return rows


def write_visuals(
    out_dir: Path,
    test: SplitPayload,
    prior_scores: np.ndarray,
    fused_scores: np.ndarray,
    rows: Sequence[dict[str, Any]],
    baseline_threshold: float,
    *,
    prior_threshold: float,
    fusion_threshold: float,
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["baseline", "shape prior", "fusion"]
    values = [
        metrics(test.scores >= baseline_threshold, test.masks)["dice"],
        metrics(prior_scores >= float(prior_threshold), test.masks)["dice"],
        metrics(fused_scores >= float(fusion_threshold), test.masks)["dice"],
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(labels, values, color=["#3b82f6", "#10b981", "#f97316"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice")
    ax.set_title("Conditional Diffusion Shape Prior")
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
    ax.set_title("Conditional Diffusion Fusion Delta")
    fig.tight_layout()
    fig.savefig(out_dir / "fusion_delta_hist.png", dpi=180)
    plt.close(fig)

    selected = select_visual_rows(rows, args.max_visual_samples)
    if not selected:
        return
    cols = ["probability", "baseline", "shape prior", "fusion", "ground truth"]
    fig, axes = plt.subplots(len(selected), len(cols), figsize=(2.3 * len(cols), 2.15 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_idx, sample_idx in enumerate(selected):
        images = [
            test.scores[sample_idx],
            test.scores[sample_idx] >= baseline_threshold,
            prior_scores[sample_idx],
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
        "# Conditional Diffusion Shape Prior",
        "",
        "This run trains a conditional DDPM shape prior from U-Net probability maps to high-resolution inclusion masks.",
        "Validation selects the shape-prior threshold and the probability/prior fusion parameters; test labels are used only for reporting.",
        "",
        "## Test Dice",
        "",
        f"- Baseline: {test['baseline']['dice']:.6f}",
        f"- Shape prior only: {test['shape_prior']['dice']:.6f} ({delta['shape_prior_dice']:+.6f})",
        f"- Fusion, validation-selected: {test['fusion_val_selected']['dice']:.6f} ({delta['fusion_val_selected_dice']:+.6f})",
        f"- Fusion, test oracle: {test['fusion_test_oracle']['dice']:.6f} ({delta['fusion_test_oracle_dice']:+.6f})",
        "",
        "## Validation Selection",
        "",
        f"- Prior threshold: {summary['prior_threshold_selection']['threshold']}",
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


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
