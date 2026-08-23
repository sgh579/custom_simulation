from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_highres_segmentation_sweep import (  # noqa: E402
    UpsampleLogits,
    load_split,
    write_leaderboard,
    write_test_metrics,
)
from run_segmentation_accuracy_sweep import (  # noqa: E402
    FEATURE_NAMES,
    ValidationUNet,
    _seed_everything,
    predict_neural,
    train_neural_method,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train high-res U-Nets with per-sample normalized V1-style inputs.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=str, default="curve,stiffness")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--depth-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _seed_everything(int(args.seed))
    inputs = [item.strip() for item in args.inputs.split(",") if item.strip()]
    allowed = {"curve", "stiffness"}
    unknown = sorted(set(inputs) - allowed)
    if unknown:
        raise SystemExit(f"Unknown inputs: {unknown}; allowed: {sorted(allowed)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.package_dir / "data"
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    run_config = {
        "schema_version": 1,
        "package_dir": str(args.package_dir),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "test_dir": str(test_dir),
        "resolution": int(args.resolution),
        "depth_steps": int(args.depth_steps),
        "inputs": inputs,
        "models": ["unet"],
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "curve_preprocessing": "response=max(fz-fz_at_first_depth,0); x=response/sample_p95(abs(response))",
        "stiffness_preprocessing": "x=(log1p(max(equivalent_stiffness_N_per_m,0))-sample_median)/(sample_q75-sample_q25)",
        "label_policy": "analytic lump xy projection on scan-area grid centers",
    }
    write_json(args.out_dir / "run_config.json", run_config)

    print(f"loading train split: {train_dir}", flush=True)
    train = load_split(train_dir, label_size=int(args.resolution), depth_steps=int(args.depth_steps))
    print(f"loading val split: {val_dir}", flush=True)
    val = load_split(val_dir, label_size=int(args.resolution), depth_steps=int(args.depth_steps))
    print(f"loading test split: {test_dir}", flush=True)
    test = load_split(test_dir, label_size=int(args.resolution), depth_steps=int(args.depth_steps))

    x_by_input = {
        "curve": (
            curve_force_response_scale_per_sample(train.fz),
            curve_force_response_scale_per_sample(val.fz),
            curve_force_response_scale_per_sample(test.fz),
        ),
        "stiffness": (
            stiffness_log_robust_per_sample(train.features[:, FEATURE_NAMES.index("equivalent_stiffness")][:, None]),
            stiffness_log_robust_per_sample(val.features[:, FEATURE_NAMES.index("equivalent_stiffness")][:, None]),
            stiffness_log_robust_per_sample(test.features[:, FEATURE_NAMES.index("equivalent_stiffness")][:, None]),
        ),
    }

    for input_name in inputs:
        x_train, x_val, x_test = x_by_input[input_name]
        method = method_name(int(args.resolution), input_name)
        model = UpsampleLogits(ValidationUNet(int(x_train.shape[1]), base_channels=24), (int(args.resolution), int(args.resolution)))
        context = {
            **run_config,
            "method": method,
            "input": input_name,
            "model": "UpsampleLogits(ValidationUNet, output_shape=(128,128))",
            "output_shape": [int(args.resolution), int(args.resolution)],
            "input_shape_chw": list(x_train.shape[1:]),
            "train_samples": len(train.names),
            "val_samples": len(val.names),
            "test_samples": len(test.names),
            "preprocessing_summary": {
                "train": summarize_input(input_name, x_train),
                "val": summarize_input(input_name, x_val),
                "test": summarize_input(input_name, x_test),
            },
            "input_normalization": input_normalization_contract(
                input_name,
                source_train_dir=train_dir,
                depth_steps=int(args.depth_steps),
            ),
            "binary_output_contract": {
                "model_output": "one inclusion logit per output pixel",
                "training_probability_link": "sigmoid inside BCEWithLogits and Dice loss",
                "inference_probability_link": "sigmoid",
                "softmax_applicable": False,
            },
            "started_at_unix": time.time(),
        }
        train_neural_method(
            method,
            "highres_per_sample_norm",
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
        threshold = float(json.loads((method_dir / "metrics_summary.json").read_text())["threshold_sweep_best"]["threshold"])
        device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
        checkpoint = torch.load(method_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
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
        write_json(method_dir / "run_config.json", context)
        write_leaderboard(args.out_dir)

    write_leaderboard(args.out_dir)
    print(f"complete: {args.out_dir}", flush=True)


def method_name(resolution: int, input_name: str) -> str:
    if input_name == "curve":
        return f"r{resolution}_unet_curve_force_response_scale_per_sample"
    if input_name == "stiffness":
        return f"r{resolution}_unet_stiffness_log_robust_per_sample"
    raise ValueError(input_name)


def input_normalization_contract(
    input_name: str,
    *,
    source_train_dir: Path,
    depth_steps: int,
) -> dict[str, Any]:
    common = {
        "schema_version": 1,
        "mode": "per_sample",
        "source_train_dir": str(source_train_dir),
        "depth_steps": int(depth_steps),
        "fit_on_target_labels": False,
    }
    if input_name == "curve":
        return {
            **common,
            "preprocessing": "response=max(fz-fz_at_first_depth,0)",
            "scale": "p95(abs(response)) over all channels and spatial points in each sample",
            "application": "response / max(sample_p95, 1e-6)",
        }
    if input_name == "stiffness":
        return {
            **common,
            "preprocessing": "log1p(max(equivalent_stiffness_N_per_m,0))",
            "centre": "sample_median",
            "scale": "sample_q75_minus_q25",
            "application": "(x - sample_median) / max(sample_iqr, 1e-6)",
        }
    raise ValueError(input_name)


def curve_force_response_scale_per_sample(fz: np.ndarray) -> np.ndarray:
    x = np.asarray(fz, dtype=np.float32)
    response = np.maximum(x - x[:, :1], 0.0)
    flat = np.abs(response).reshape(response.shape[0], -1)
    scale = np.nanpercentile(flat, 95.0, axis=1).astype(np.float32)
    scale = np.maximum(scale, np.float32(1.0e-6)).reshape(-1, 1, 1, 1)
    out = response / scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def stiffness_log_robust_per_sample(stiffness: np.ndarray) -> np.ndarray:
    x = np.log1p(np.maximum(np.asarray(stiffness, dtype=np.float32), 0.0))
    flat = x.reshape(x.shape[0], -1)
    median = np.nanmedian(flat, axis=1).astype(np.float32).reshape(-1, 1, 1, 1)
    q25 = np.nanpercentile(flat, 25.0, axis=1).astype(np.float32)
    q75 = np.nanpercentile(flat, 75.0, axis=1).astype(np.float32)
    scale = np.maximum(q75 - q25, np.float32(1.0e-6)).reshape(-1, 1, 1, 1)
    out = (x - median) / scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def summarize_input(input_name: str, x: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    return {
        "input": input_name,
        "shape": list(x.shape),
        "min": float(np.nanmin(flat)),
        "p05": float(np.nanpercentile(flat, 5.0)),
        "median": float(np.nanmedian(flat)),
        "mean": float(np.nanmean(flat)),
        "p95": float(np.nanpercentile(flat, 95.0)),
        "max": float(np.nanmax(flat)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
