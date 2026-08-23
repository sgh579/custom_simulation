#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0


def robust_minmax_per_sample(
    value: np.ndarray,
    *,
    low_percentile: float = LOW_PERCENTILE,
    high_percentile: float = HIGH_PERCENTILE,
) -> np.ndarray:
    """Scale each complete sample to [0, 1] after label-free percentile clipping."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"Expected a batch dimension plus features, got {array.shape}")
    flat = array.reshape(array.shape[0], -1)
    low = np.nanpercentile(flat, low_percentile, axis=1).astype(np.float32)
    high = np.nanpercentile(flat, high_percentile, axis=1).astype(np.float32)
    shape = (-1,) + (1,) * (array.ndim - 1)
    low = low.reshape(shape)
    high = high.reshape(shape)
    scale = np.maximum(high - low, np.float32(1.0e-6))
    normalized = (np.clip(array, low, high) - low) / scale
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def curve_response_robust_minmax_per_sample(fz: np.ndarray) -> np.ndarray:
    curves = np.asarray(fz, dtype=np.float32)
    response = np.maximum(curves - curves[:, :1], np.float32(0.0))
    return robust_minmax_per_sample(response)


def stiffness_robust_minmax_per_sample(stiffness: np.ndarray) -> np.ndarray:
    return robust_minmax_per_sample(np.maximum(np.asarray(stiffness, dtype=np.float32), 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Nets with per-sample robust min-max inputs.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--inputs", default="curve,stiffness")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--depth-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "scripts"))

    from run_highres_segmentation_sweep import (  # noqa: PLC0415
        UpsampleLogits,
        load_split,
        write_leaderboard,
        write_test_metrics,
    )
    from run_segmentation_accuracy_sweep import (  # noqa: PLC0415
        FEATURE_NAMES,
        ValidationUNet,
        _seed_everything,
        predict_neural,
        train_neural_method,
    )

    inputs = [item.strip() for item in str(args.inputs).split(",") if item.strip()]
    unknown = sorted(set(inputs) - {"curve", "stiffness"})
    if unknown:
        raise SystemExit(f"Unknown inputs: {unknown}")

    _seed_everything(int(args.seed))
    data_root = args.package_dir.expanduser().resolve() / "data"
    train = load_split(data_root / "train", label_size=args.resolution, depth_steps=args.depth_steps)
    val = load_split(data_root / "val", label_size=args.resolution, depth_steps=args.depth_steps)
    test = load_split(data_root / "test", label_size=args.resolution, depth_steps=args.depth_steps)
    stiffness_index = FEATURE_NAMES.index("equivalent_stiffness")
    x_by_input = {
        "curve": (
            curve_response_robust_minmax_per_sample(train.fz),
            curve_response_robust_minmax_per_sample(val.fz),
            curve_response_robust_minmax_per_sample(test.fz),
        ),
        "stiffness": (
            stiffness_robust_minmax_per_sample(train.features[:, stiffness_index][:, None]),
            stiffness_robust_minmax_per_sample(val.features[:, stiffness_index][:, None]),
            stiffness_robust_minmax_per_sample(test.features[:, stiffness_index][:, None]),
        ),
    }
    run_config = {
        "schema_version": 1,
        "normalization": (
            "per sample over all input elements: clip to q01/q99, then "
            "(x-q01)/(q99-q01) to [0,1]"
        ),
        "curve_preprocessing": "response=max(fz-fz_at_first_depth,0), then robust min-max",
        "stiffness_preprocessing": "max(equivalent_stiffness_N_per_m,0), then robust min-max",
        "fit_on_target_labels": False,
        "package_dir": str(args.package_dir.expanduser().resolve()),
        "resolution": int(args.resolution),
        "depth_steps": int(args.depth_steps),
        "inputs": inputs,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "seed": int(args.seed),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "run_config.json", run_config)

    for input_name in inputs:
        x_train, x_val, x_test = x_by_input[input_name]
        method = f"r{args.resolution}_{input_name}_unet_robust_minmax_per_sample"
        model = UpsampleLogits(
            ValidationUNet(int(x_train.shape[1]), base_channels=24),
            (int(args.resolution), int(args.resolution)),
        )
        context = {
            **run_config,
            "method": method,
            "input": input_name,
            "model": "UpsampleLogits(ValidationUNet, base_channels=24)",
            "input_shape_chw": list(x_train.shape[1:]),
            "output_shape": [int(args.resolution), int(args.resolution)],
            "train_samples": len(train.names),
            "val_samples": len(val.names),
            "test_samples": len(test.names),
            "started_at_unix": time.time(),
        }
        train_neural_method(
            method,
            "robust_minmax_per_sample",
            model,
            x_train,
            train.masks,
            x_val,
            val.masks,
            val.names,
            args.out_dir,
            context=context,
            args=args,
            augment=False,
            focal=False,
        )
        method_dir = args.out_dir / method
        metrics = json.loads((method_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        threshold = float(metrics["threshold_sweep_best"]["threshold"])
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(method_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device).eval()
        test_scores = predict_neural(model, x_test, device, batch_size=int(args.batch_size))
        write_test_metrics(
            method_dir,
            test_scores,
            test.masks,
            test.names,
            fixed_threshold=0.5,
            val_best_threshold=threshold,
            context=context,
            force=True,
        )
        _write_json(method_dir / "run_config.json", context)
        write_leaderboard(args.out_dir)

    write_leaderboard(args.out_dir)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
