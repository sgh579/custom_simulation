from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.workflow import REQUIRED_NEWTON_DEVICE, disk_usage_metadata, json_ready


PHASES = ("prepare", "generate", "validate_dataset", "train_unet", "evaluate_unet", "report")
DEFAULT_DASHBOARD = Path("runs/random_shape_20x_pipeline/dashboard.json")
DEFAULT_DATA_DIR = Path("data/palpation_newton_random_shapes_20x_scan20_v1")
DEFAULT_TRAIN_OUT_DIR = Path("runs/validation_unet_random_shapes_20x_scan20_v1")
ALLOWED_SHAPES = ("sphere", "ellipsoid", "box", "cylinder", "capsule")


class DatasetValidationError(RuntimeError):
    pass


def main() -> None:
    args = _parse_args()
    dashboard_path = args.dashboard
    dashboard = _load_or_create_dashboard(dashboard_path, args)
    _mark_phase(dashboard, "prepare", "complete", {"dashboard": str(dashboard_path)})
    _write_dashboard(dashboard_path, dashboard)
    if args.stop_after == "prepare":
        print(f"prepared dashboard: {dashboard_path}")
        return

    try:
        _run_generate_phase(dashboard_path, dashboard)
        if args.stop_after == "generate":
            return
        _run_validate_phase(dashboard_path, dashboard)
        if args.stop_after == "validate_dataset":
            return
        _run_train_phase(dashboard_path, dashboard, force=args.force_train)
        if args.stop_after == "train_unet":
            return
        _run_evaluate_phase(dashboard_path, dashboard, force=args.force_eval)
        if args.stop_after == "evaluate_unet":
            return
        _run_report_phase(dashboard_path, dashboard)
    except Exception as exc:
        dashboard["status"] = "train_incomplete" if dashboard.get("phase") == "train_unet" else "failed"
        dashboard["failure"] = {"phase": dashboard.get("phase"), "error": str(exc)}
        _write_dashboard(dashboard_path, dashboard)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the random-shape 20x Newton/VBD dataset and U-Net pipeline.")
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--backend", choices=["newton", "analytic"], default="newton")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-out-dir", type=Path, default=DEFAULT_TRAIN_OUT_DIR)
    parser.add_argument("--eval-out-dir", type=Path, default=None)
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--training-seed", type=int, default=7)
    parser.add_argument("--stop-after", choices=PHASES, default=None)
    parser.add_argument("--force-train", action="store_true", help="Rerun training even when best.pt exists.")
    parser.add_argument("--force-eval", action="store_true", help="Rerun evaluation even when metrics exist.")
    return parser.parse_args()


def _load_or_create_dashboard(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            dashboard = json.load(f)
        if int(dashboard.get("schema_version", 0)) != 1:
            raise RuntimeError(f"Unsupported dashboard schema at {path}: {dashboard.get('schema_version')}")
        return dashboard

    eval_out_dir = args.eval_out_dir or args.train_out_dir / "eval"
    recipe = {
        "backend": args.backend,
        "num_train": int(args.num_train),
        "num_val": int(args.num_val),
        "seed": int(args.seed),
        "grid_h": 20,
        "grid_w": 20,
        "press_steps": 20,
        "substeps_per_depth": 3,
        "vbd_iterations": 10,
        "cells_x": 48,
        "cells_y": 48,
        "cells_z": 16,
        "lumps_min": 1,
        "lumps_max": 4,
        "lump_shapes": list(ALLOWED_SHAPES),
        "lump_stiffness_multiplier": 20.0,
        "device": REQUIRED_NEWTON_DEVICE,
    }
    paths = {
        "data_dir": str(args.data_dir),
        "train_dir": str(args.data_dir / "train"),
        "val_dir": str(args.data_dir / "val"),
        "train_out_dir": str(args.train_out_dir),
        "eval_out_dir": str(eval_out_dir),
        "dashboard": str(path),
        "logs_dir": str(path.parent / "logs"),
        "report": str(path.parent / "performance_report.md"),
    }
    commands = _build_commands(recipe, paths, training_seed=int(args.training_seed))
    return {
        "schema_version": 1,
        "status": "created",
        "created_at_local": _now(),
        "updated_at_local": _now(),
        "phase": "prepare",
        "recipe": recipe,
        "paths": paths,
        "commands": commands,
        "phases": {phase: {"status": "pending"} for phase in PHASES},
        "progress": {},
    }


def _build_commands(recipe: dict[str, Any], paths: dict[str, str], *, training_seed: int) -> dict[str, Any]:
    generate = _conda_python_cmd() + [
        "scripts/generate_palpation_dataset.py",
        "--backend",
        str(recipe["backend"]),
        "--out-dir",
        paths["data_dir"],
        "--num-train",
        str(recipe["num_train"]),
        "--num-val",
        str(recipe["num_val"]),
        "--seed",
        str(recipe["seed"]),
        "--grid-h",
        str(recipe["grid_h"]),
        "--grid-w",
        str(recipe["grid_w"]),
        "--press-steps",
        str(recipe["press_steps"]),
        "--substeps-per-depth",
        str(recipe["substeps_per_depth"]),
        "--vbd-iterations",
        str(recipe["vbd_iterations"]),
        "--cells-x",
        str(recipe["cells_x"]),
        "--cells-y",
        str(recipe["cells_y"]),
        "--cells-z",
        str(recipe["cells_z"]),
        "--lumps-min",
        str(recipe["lumps_min"]),
        "--lumps-max",
        str(recipe["lumps_max"]),
        "--lump-shapes",
        ",".join(recipe["lump_shapes"]),
        "--lump-stiffness-multiplier",
        str(recipe["lump_stiffness_multiplier"]),
        "--no-save-phantom-3d",
        "--no-save-press-records",
        "--no-save-scan-animation",
        "--device",
        str(recipe["device"]),
    ]
    train = _conda_python_cmd() + [
        "scripts/train_validation_unet.py",
        "--data-dir",
        paths["train_dir"],
        "--val-dir",
        paths["val_dir"],
        "--out-dir",
        paths["train_out_dir"],
        "--exact-out-dir",
        "--input-mode",
        "fz",
        "--epochs",
        "50",
        "--early-stop-patience",
        "10",
        "--batch-size",
        "16",
        "--base-channels",
        "24",
        "--seed",
        str(training_seed),
        "--device",
        str(recipe["device"]),
    ]
    evaluate = _conda_python_cmd() + [
        "scripts/evaluate_validation_unet.py",
        "--checkpoint",
        str(Path(paths["train_out_dir"]) / "best.pt"),
        "--data-dir",
        paths["val_dir"],
        "--out-dir",
        paths["eval_out_dir"],
        "--exact-out-dir",
        "--threshold",
        "0.5",
        "--sweep-thresholds",
        "--max-images",
        "20",
        "--device",
        str(recipe["device"]),
    ]
    return {
        "generate_formal": generate,
        "generate_resume": generate + ["--resume"],
        "train_unet": train,
        "evaluate_unet": evaluate,
    }


def _run_generate_phase(path: Path, dashboard: dict[str, Any]) -> None:
    _mark_phase(dashboard, "generate", "running")
    _write_dashboard(path, dashboard)
    try:
        validation = _validate_dataset(dashboard, strict=True)
        _mark_phase(dashboard, "generate", "complete", {"skipped": True, "reason": "dataset already valid"})
        dashboard["validation"] = validation
        _write_dashboard(path, dashboard)
        return
    except DatasetValidationError:
        pass

    log_path = Path(dashboard["paths"]["logs_dir"]) / "generate.log"
    started = time.perf_counter()
    _run_logged(dashboard["commands"]["generate_resume"], log_path)
    progress = _count_samples(Path(dashboard["paths"]["data_dir"]))
    _mark_phase(
        dashboard,
        "generate",
        "complete",
        {"elapsed_seconds": time.perf_counter() - started, "log": str(log_path), "progress": progress},
    )
    dashboard["progress"]["dataset"] = progress
    _write_dashboard(path, dashboard)


def _run_validate_phase(path: Path, dashboard: dict[str, Any]) -> None:
    _mark_phase(dashboard, "validate_dataset", "running")
    _write_dashboard(path, dashboard)
    validation = _validate_dataset(dashboard, strict=True)
    dashboard["validation"] = validation
    dashboard["progress"]["dataset"] = validation["counts"]
    _mark_phase(dashboard, "validate_dataset", "complete", validation)
    _write_dashboard(path, dashboard)


def _run_train_phase(path: Path, dashboard: dict[str, Any], *, force: bool) -> None:
    train_out_dir = Path(dashboard["paths"]["train_out_dir"])
    best_path = train_out_dir / "best.pt"
    history_path = train_out_dir / "history.csv"
    if best_path.exists() and history_path.exists() and not force:
        summary = _training_summary(train_out_dir)
        dashboard["training"] = {"status": "complete", "skipped": True, "summary": summary}
        _mark_phase(dashboard, "train_unet", "complete", dashboard["training"])
        _write_dashboard(path, dashboard)
        return

    _mark_phase(dashboard, "train_unet", "running")
    _write_dashboard(path, dashboard)
    log_path = Path(dashboard["paths"]["logs_dir"]) / "train_unet.log"
    started = time.perf_counter()
    try:
        _run_logged(dashboard["commands"]["train_unet"], log_path)
    except Exception as exc:
        dashboard["training"] = {"status": "incomplete", "log": str(log_path), "error": str(exc)}
        _mark_phase(dashboard, "train_unet", "incomplete", dashboard["training"])
        _write_dashboard(path, dashboard)
        raise
    summary = _training_summary(train_out_dir)
    dashboard["training"] = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "log": str(log_path),
        "summary": summary,
    }
    _mark_phase(dashboard, "train_unet", "complete", dashboard["training"])
    _write_dashboard(path, dashboard)


def _run_evaluate_phase(path: Path, dashboard: dict[str, Any], *, force: bool) -> None:
    eval_out_dir = Path(dashboard["paths"]["eval_out_dir"])
    metrics_path = eval_out_dir / "metrics_summary.json"
    sweep_path = eval_out_dir / "threshold_sweep.json"
    if metrics_path.exists() and sweep_path.exists() and not force:
        summary = _evaluation_summary(eval_out_dir)
        dashboard["evaluation"] = {"status": "complete", "skipped": True, "summary": summary}
        _mark_phase(dashboard, "evaluate_unet", "complete", dashboard["evaluation"])
        _write_dashboard(path, dashboard)
        return

    _mark_phase(dashboard, "evaluate_unet", "running")
    _write_dashboard(path, dashboard)
    log_path = Path(dashboard["paths"]["logs_dir"]) / "evaluate_unet.log"
    started = time.perf_counter()
    _run_logged(dashboard["commands"]["evaluate_unet"], log_path)
    summary = _evaluation_summary(eval_out_dir)
    dashboard["evaluation"] = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "log": str(log_path),
        "summary": summary,
    }
    _mark_phase(dashboard, "evaluate_unet", "complete", dashboard["evaluation"])
    _write_dashboard(path, dashboard)


def _run_report_phase(path: Path, dashboard: dict[str, Any]) -> None:
    _mark_phase(dashboard, "report", "running")
    _write_dashboard(path, dashboard)
    report = {
        "dataset": dashboard.get("validation", {}),
        "training": dashboard.get("training", {}).get("summary", {}),
        "evaluation": dashboard.get("evaluation", {}).get("summary", {}),
        "disk_usage": disk_usage_metadata([dashboard["paths"]["data_dir"], dashboard["paths"]["train_out_dir"]]),
    }
    dashboard["report"] = report
    dashboard["status"] = "complete"
    _write_report_markdown(Path(dashboard["paths"]["report"]), dashboard)
    _mark_phase(dashboard, "report", "complete", {"report": dashboard["paths"]["report"]})
    _write_dashboard(path, dashboard)
    print(f"pipeline complete: {path}")
    print(f"performance report: {dashboard['paths']['report']}")


def _validate_dataset(dashboard: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    recipe = dashboard["recipe"]
    data_dir = Path(dashboard["paths"]["data_dir"])
    expected = {"train": int(recipe["num_train"]), "val": int(recipe["num_val"])}
    expected_fz_shape = (int(recipe["grid_h"]), int(recipe["grid_w"]), int(recipe["press_steps"]))
    expected_presses_shape = (*expected_fz_shape, 2)
    expected_multiplier = float(recipe["lump_stiffness_multiplier"])
    failures: list[str] = []
    counts: dict[str, Any] = {}
    lump_hist: dict[str, int] = {}
    shape_hist: dict[str, int] = {}
    coverage_values: list[float] = []

    for split, count in expected.items():
        split_dir = data_dir / split
        files = sorted(split_dir.glob("sample_*.npz"))
        counts[split] = {"expected": count, "found": len(files)}
        if len(files) != count:
            failures.append(f"{split}: expected {count} sample files, found {len(files)}")
        for idx in range(count):
            sample_path = split_dir / f"sample_{idx:04d}.npz"
            gt_path = split_dir / f"sample_{idx:04d}_gt.json"
            if not sample_path.exists():
                failures.append(f"missing {sample_path}")
                continue
            if not gt_path.exists():
                failures.append(f"missing {gt_path}")
            try:
                with np.load(sample_path) as sample:
                    _check_array(sample, "fz", expected_fz_shape)
                    _check_array(sample, "presses", expected_presses_shape)
                    mask = _check_array(sample, "mask", expected_fz_shape[:2])
                    if int(np.count_nonzero(mask > 0.5)) <= 0:
                        failures.append(f"{sample_path}: mask has no positive cells")
                    coverage_values.append(float(np.count_nonzero(mask > 0.5) / max(mask.size, 1)))
                    num_lumps = int(np.asarray(sample["num_lumps"]).reshape(()))
                    if num_lumps < int(recipe["lumps_min"]) or num_lumps > int(recipe["lumps_max"]):
                        failures.append(f"{sample_path}: num_lumps={num_lumps}")
                    lump_hist[str(num_lumps)] = lump_hist.get(str(num_lumps), 0) + 1
                    lumps = json.loads(_npz_string(sample["lumps_json"]))
                    if len(lumps) != num_lumps:
                        failures.append(f"{sample_path}: lumps_json length {len(lumps)} != num_lumps {num_lumps}")
                    for lump in lumps:
                        shape = str(lump.get("shape", ""))
                        if shape not in ALLOWED_SHAPES:
                            failures.append(f"{sample_path}: unsupported shape {shape}")
                        shape_hist[shape] = shape_hist.get(shape, 0) + 1
                        multiplier = float(lump.get("stiffness_multiplier", np.nan))
                        if abs(multiplier - expected_multiplier) > 1.0e-6:
                            failures.append(f"{sample_path}: stiffness_multiplier={multiplier}")
                    if recipe["backend"] == "newton":
                        _check_member(sample, "tet_lump_mask")
                        _check_member(sample, "tet_lump_id")
            except Exception as exc:
                failures.append(f"{sample_path}: {exc}")

    summary: dict[str, Any] = {
        "status": "complete" if not failures else "failed",
        "counts": counts,
        "expected_fz_shape": list(expected_fz_shape),
        "expected_presses_shape": list(expected_presses_shape),
        "lump_count_histogram": lump_hist,
        "shape_histogram": shape_hist,
        "mask_coverage": _coverage_summary(coverage_values),
        "failures": failures[:50],
        "failure_count": len(failures),
    }
    if failures and strict:
        raise DatasetValidationError(f"dataset validation failed with {len(failures)} issue(s): {failures[:5]}")
    return summary


def _check_array(sample: np.lib.npyio.NpzFile, key: str, shape: tuple[int, ...]) -> np.ndarray:
    _check_member(sample, key)
    value = np.asarray(sample[key])
    if value.shape != shape:
        raise ValueError(f"{key} shape {value.shape} != {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def _check_member(sample: np.lib.npyio.NpzFile, key: str) -> None:
    if key not in sample.files:
        raise KeyError(key)


def _npz_string(value: Any) -> str:
    item = np.asarray(value).reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _coverage_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"count": int(arr.size), "min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}


def _count_samples(data_dir: Path) -> dict[str, int]:
    return {
        "train": len(list((data_dir / "train").glob("sample_*.npz"))),
        "val": len(list((data_dir / "val").glob("sample_*.npz"))),
    }


def _training_summary(train_out_dir: Path) -> dict[str, Any]:
    history_path = train_out_dir / "history.csv"
    rows = _read_csv(history_path)
    best_row = None
    if rows:
        metric = "val_dice" if rows[0].get("val_dice") not in ("", None) else "train_dice"
        best_row = max(rows, key=lambda row: _float(row.get(metric), default=-1.0))
    return {
        "out_dir": str(train_out_dir),
        "best_checkpoint": str(train_out_dir / "best.pt"),
        "last_checkpoint": str(train_out_dir / "last.pt"),
        "history": str(history_path),
        "epochs_completed": len(rows),
        "best_epoch": _json_ready_row(best_row) if best_row is not None else None,
    }


def _evaluation_summary(eval_out_dir: Path) -> dict[str, Any]:
    metrics = _read_json(eval_out_dir / "metrics_summary.json")
    sweep = _read_json(eval_out_dir / "threshold_sweep.json")
    return {
        "out_dir": str(eval_out_dir),
        "metrics_summary": metrics,
        "threshold_sweep_best": sweep.get("best") if isinstance(sweep, dict) else None,
        "metrics_summary_path": str(eval_out_dir / "metrics_summary.json"),
        "metrics_per_sample_path": str(eval_out_dir / "metrics_per_sample.csv"),
        "threshold_sweep_path": str(eval_out_dir / "threshold_sweep.json"),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _json_ready_row(row: dict[str, str] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            out[key] = value
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_report_markdown(path: Path, dashboard: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validation = dashboard.get("validation", {})
    training = dashboard.get("training", {}).get("summary", {})
    evaluation = dashboard.get("evaluation", {}).get("summary", {})
    metrics = evaluation.get("metrics_summary", {}) if isinstance(evaluation, dict) else {}
    sweep = evaluation.get("threshold_sweep_best") if isinstance(evaluation, dict) else None
    lines = [
        "# Random Shape 20x Pipeline Performance",
        "",
        f"- Dashboard: `{dashboard['paths']['dashboard']}`",
        f"- Dataset: `{dashboard['paths']['data_dir']}`",
        f"- Train samples: `{validation.get('counts', {}).get('train', {}).get('found', 0)}`",
        f"- Val samples: `{validation.get('counts', {}).get('val', {}).get('found', 0)}`",
        f"- Best training epoch: `{training.get('best_epoch')}`",
        f"- Eval Dice: `{metrics.get('dice')}`",
        f"- Eval IoU: `{metrics.get('iou')}`",
        f"- Eval precision: `{metrics.get('precision')}`",
        f"- Eval recall: `{metrics.get('recall')}`",
        f"- Threshold sweep best: `{sweep}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        header = f"\n$ {' '.join(cmd)}\nstarted_at_local={_now()}\n".encode("utf-8")
        log.write(header)
        log.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        footer = f"finished_at_local={_now()}\nreturncode={proc.returncode}\n".encode("utf-8")
        log.write(footer)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with returncode {proc.returncode}; see {log_path}")


def _mark_phase(dashboard: dict[str, Any], phase: str, status: str, details: dict[str, Any] | None = None) -> None:
    dashboard["phase"] = phase
    if status == "running":
        dashboard["status"] = "running"
    elif phase == "prepare" and status == "complete" and dashboard.get("status") in {"created", "running"}:
        dashboard["status"] = "prepared"
    entry = {"status": status, "updated_at_local": _now()}
    if details:
        entry.update(json_ready(details))
    dashboard.setdefault("phases", {})[phase] = entry


def _write_dashboard(path: Path, dashboard: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dashboard["updated_at_local"] = _now()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(json_ready(dashboard), indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _conda_python_cmd() -> list[str]:
    return ["conda", "run", "-n", "palpation", "python"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
