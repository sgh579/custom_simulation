#!/usr/bin/env python3
"""Evaluate a high-resolution grid U-Net with checkpoint-locked source stats."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from palpation_sim.models import ValidationUNet  # noqa: E402
from palpation_sim.workflow import require_runtime_environment  # noqa: E402
from run_highres_segmentation_sweep import (  # noqa: E402
    UpsampleLogits,
    load_split,
    write_test_metrics,
)
from run_highres_per_sample_norm_unet import (  # noqa: E402
    curve_force_response_scale_per_sample,
)
from run_segmentation_accuracy_sweep import predict_neural  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resample_channels(x: np.ndarray, channels: int) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    current = int(value.shape[1])
    if current == int(channels):
        return value
    source = np.linspace(0.0, 1.0, current, dtype=np.float32)
    target = np.linspace(0.0, 1.0, int(channels), dtype=np.float32)
    moved = np.moveaxis(value, 1, -1).reshape(-1, current)
    output = np.empty((moved.shape[0], int(channels)), dtype=np.float32)
    for index, curve in enumerate(moved):
        output[index] = np.interp(target, source, curve).astype(np.float32)
    shape = (value.shape[0], value.shape[2], value.shape[3], int(channels))
    return np.moveaxis(output.reshape(shape), -1, 1)


def apply_checkpoint_normalization(x: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    if contract.get("schema_version") != 1:
        raise RuntimeError("Checkpoint has no supported normalization contract.")
    preprocessing = contract.get("preprocessing")
    if preprocessing == "response=max(fz-fz_at_first_depth,0)":
        return curve_force_response_scale_per_sample(x)
    if preprocessing != "raw_fz":
        raise RuntimeError(f"Unsupported checkpoint preprocessing: {preprocessing!r}.")
    mode = contract.get("mode")
    value = np.asarray(x, dtype=np.float32)
    if mode == "dataset":
        mean = np.asarray(contract.get("mean"), dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.asarray(contract.get("std"), dtype=np.float32).reshape(1, -1, 1, 1)
        if mean.shape[1] != value.shape[1] or std.shape != mean.shape:
            raise RuntimeError(
                f"Normalization channels {mean.shape[1]} do not match input {value.shape[1]}."
            )
        epsilon = np.float32(contract.get("epsilon", 1e-6))
        value = (value - mean) / (std + epsilon)
    elif mode == "sample":
        mean = np.nanmean(value, axis=(2, 3), keepdims=True)
        std = np.nanstd(value, axis=(2, 3), keepdims=True)
        value = (value - mean) / (std + np.float32(contract.get("epsilon", 1e-6)))
    elif mode != "none":
        raise RuntimeError(f"Unsupported checkpoint normalization mode: {mode!r}.")
    return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def summarize(value: np.ndarray) -> dict[str, float | list[int]]:
    array = np.asarray(value, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-metrics", type=Path, required=True)
    parser.add_argument("--target-test-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    require_runtime_environment()
    if args.out_dir.exists():
        raise RuntimeError(f"Refusing to reuse output directory: {args.out_dir}.")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    context = saved.get("context", {})
    contract = context.get("input_normalization")
    if not isinstance(contract, dict):
        raise RuntimeError(
            "Checkpoint does not contain an input normalization contract; retrain with the fixed pipeline."
        )
    binary_contract = context.get("binary_output_contract", {})
    if binary_contract.get("inference_probability_link") != "sigmoid":
        raise RuntimeError("Checkpoint binary output contract is missing or incompatible.")
    channels = int(contract.get("channels", contract.get("depth_steps", 0)))
    if channels <= 0:
        raise RuntimeError("Checkpoint normalization contract does not declare its input channels.")
    target = load_split(args.target_test_dir, label_size=args.resolution)
    raw_resampled = resample_channels(target.fz, channels)
    normalized = apply_checkpoint_normalization(raw_resampled, contract)

    model = UpsampleLogits(
        ValidationUNet(channels, base_channels=24),
        (args.resolution, args.resolution),
    )
    model.load_state_dict(saved["model_state"])
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    probabilities = predict_neural(
        model.to(device), normalized, device, batch_size=args.batch_size
    )
    if float(probabilities.min()) < 0.0 or float(probabilities.max()) > 1.0:
        raise RuntimeError("Inference output is not a sigmoid probability map.")
    source_metrics = json.loads(args.source_metrics.read_text(encoding="utf-8"))
    threshold = float(source_metrics["threshold_sweep_best"]["threshold"])
    args.out_dir.mkdir(parents=True)
    np.save(args.out_dir / "probabilities.npy", probabilities.astype(np.float32))
    write_test_metrics(
        args.out_dir,
        probabilities,
        target.masks,
        target.names,
        fixed_threshold=0.5,
        val_best_threshold=threshold,
        context={
            "checkpoint": str(args.checkpoint),
            "source_metrics": str(args.source_metrics),
            "target_test_dir": str(args.target_test_dir),
            "target_original_channels": int(target.fz.shape[1]),
            "target_evaluated_channels": channels,
            "resampling": (
                "none"
                if int(target.fz.shape[1]) == channels
                else "linear_in_normalized_depth_fraction"
            ),
            "normalization": (
                "checkpoint_locked_per_sample_contract"
                if contract.get("mode") == "per_sample"
                else "checkpoint_locked_source_train_statistics"
            ),
            "probability_link": "sigmoid",
            "input_before_normalization": summarize(raw_resampled),
            "input_after_normalization": summarize(normalized),
        },
        force=True,
    )
    summary_path = args.out_dir / "test_metrics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["normalization_contract"] = contract
    summary["binary_output_contract"] = binary_contract
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
