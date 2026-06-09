from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


OLD_DATA_DIR = Path("data/palpation_newton_random_shapes_20x_scan20_v1")
NEW_DATA_DIR = Path("data/palpation_newton_random_shapes_20x_scan20_seed23")
NEW_GENERATION_DASHBOARD = Path("runs/random_shape_20x_pipeline_seed23/dashboard.json")
COMBINED_DATA_DIR = Path("data/palpation_newton_random_shapes_20x_scan20_combined_seed11_seed23")
OUT_DIR = Path("runs/validation_unet_random_shapes_20x_scan20_combined_seed11_seed23")
SCALING_DASHBOARD = Path("runs/random_shape_20x_scaling_seed11_seed23/dashboard.json")


def main() -> None:
    args = _parse_args()
    dashboard = _load_or_create_dashboard(args.dashboard, args)
    _write_dashboard(args.dashboard, dashboard)
    try:
        _wait_for_new_data(args, dashboard)
        _assemble_combined_dataset(args, dashboard)
        _train_combined(args, dashboard)
        _evaluate(args, dashboard, "combined", args.combined_data_dir / "val", args.out_dir / "eval_combined", max_images=40)
        _evaluate(args, dashboard, "old_val", args.old_data_dir / "val", args.out_dir / "eval_old", max_images=20)
        _evaluate(args, dashboard, "new_val", args.new_data_dir / "val", args.out_dir / "eval_new", max_images=20)
        _write_report(args, dashboard)
        dashboard["status"] = "complete"
        dashboard["phase"] = "complete"
        _write_dashboard(args.dashboard, dashboard)
    except Exception as exc:
        dashboard["status"] = "failed"
        dashboard["failure"] = {"phase": dashboard.get("phase"), "error": str(exc)}
        _write_dashboard(args.dashboard, dashboard)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on old+new random-shape 20x datasets and report scaling.")
    parser.add_argument("--dashboard", type=Path, default=SCALING_DASHBOARD)
    parser.add_argument("--old-data-dir", type=Path, default=OLD_DATA_DIR)
    parser.add_argument("--new-data-dir", type=Path, default=NEW_DATA_DIR)
    parser.add_argument("--new-generation-dashboard", type=Path, default=NEW_GENERATION_DASHBOARD)
    parser.add_argument("--combined-data-dir", type=Path, default=COMBINED_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--reference-metrics",
        type=Path,
        default=Path("runs/validation_unet_random_shapes_20x_scan20_v1/eval/metrics_summary.json"),
    )
    parser.add_argument("--reference-label", type=str, default="Baseline Old-Only Reference")
    parser.add_argument("--expected-old-train", type=int, default=200)
    parser.add_argument("--expected-old-val", type=int, default=20)
    parser.add_argument("--expected-new-train", type=int, default=200)
    parser.add_argument("--expected-new-val", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    return parser.parse_args()


def _load_or_create_dashboard(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "schema_version": 1,
        "status": "created",
        "phase": "created",
        "created_at_local": _now(),
        "updated_at_local": _now(),
        "paths": {
            "old_data_dir": str(args.old_data_dir),
            "new_data_dir": str(args.new_data_dir),
            "new_generation_dashboard": str(args.new_generation_dashboard),
            "combined_data_dir": str(args.combined_data_dir),
            "out_dir": str(args.out_dir),
            "logs_dir": str(path.parent / "logs"),
            "report": str(path.parent / "scaling_report.md"),
            "reference_metrics": str(args.reference_metrics),
        },
        "expected_counts": {
            "old_train": int(args.expected_old_train),
            "old_val": int(args.expected_old_val),
            "new_train": int(args.expected_new_train),
            "new_val": int(args.expected_new_val),
            "combined_train": int(args.expected_old_train + args.expected_new_train),
            "combined_val": int(args.expected_old_val + args.expected_new_val),
        },
        "phases": {},
    }


def _wait_for_new_data(args: argparse.Namespace, dashboard: dict[str, Any]) -> None:
    _mark(dashboard, "wait_for_new_data", "running")
    while True:
        old_counts = _dataset_counts(args.old_data_dir)
        new_counts = _dataset_counts(args.new_data_dir)
        generation = _read_json(args.new_generation_dashboard)
        dashboard["source_counts"] = {"old": old_counts, "new": new_counts}
        dashboard["new_generation_status"] = {
            "status": generation.get("status"),
            "phase": generation.get("phase"),
            "validate_dataset": generation.get("phases", {}).get("validate_dataset"),
            "failure": generation.get("failure"),
        }
        _write_dashboard(args.dashboard, dashboard)
        if generation.get("failure"):
            raise RuntimeError(f"new generation failed: {generation['failure']}")
        if (
            old_counts["train"] == args.expected_old_train
            and old_counts["val"] == args.expected_old_val
            and new_counts["train"] == args.expected_new_train
            and new_counts["val"] == args.expected_new_val
            and generation.get("phases", {}).get("validate_dataset", {}).get("status") == "complete"
        ):
            _mark(dashboard, "wait_for_new_data", "complete")
            return
        time.sleep(max(float(args.poll_seconds), 5.0))


def _assemble_combined_dataset(args: argparse.Namespace, dashboard: dict[str, Any]) -> None:
    _mark(dashboard, "assemble_combined_dataset", "running")
    for split in ("train", "val"):
        split_dir = args.combined_data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in split_dir.glob("*.npz"):
            if path.is_symlink() or path.is_file():
                path.unlink()
        _link_npz_files(args.old_data_dir / split, split_dir, "old")
        _link_npz_files(args.new_data_dir / split, split_dir, "new")
    counts = _dataset_counts(args.combined_data_dir)
    expected = dashboard["expected_counts"]
    if counts["train"] != expected["combined_train"] or counts["val"] != expected["combined_val"]:
        raise RuntimeError(f"combined counts mismatch: {counts} expected {expected}")
    dashboard["combined_counts"] = counts
    _mark(dashboard, "assemble_combined_dataset", "complete")
    _write_dashboard(args.dashboard, dashboard)


def _train_combined(args: argparse.Namespace, dashboard: dict[str, Any]) -> None:
    best = args.out_dir / "best.pt"
    history = args.out_dir / "history.csv"
    if best.exists() and history.exists() and not args.force_train:
        dashboard["training"] = {"status": "complete", "skipped": True, "summary": _training_summary(args.out_dir)}
        _mark(dashboard, "train_combined", "complete")
        _write_dashboard(args.dashboard, dashboard)
        return
    _mark(dashboard, "train_combined", "running")
    log_path = Path(dashboard["paths"]["logs_dir"]) / "train_combined.log"
    cmd = _conda_python_cmd() + [
        "scripts/train_validation_unet.py",
        "--data-dir",
        str(args.combined_data_dir / "train"),
        "--val-dir",
        str(args.combined_data_dir / "val"),
        "--out-dir",
        str(args.out_dir),
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
        "7",
        "--device",
        "cuda:0",
    ]
    started = time.perf_counter()
    _run_logged(cmd, log_path)
    dashboard["training"] = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "log": str(log_path),
        "summary": _training_summary(args.out_dir),
    }
    _mark(dashboard, "train_combined", "complete")
    _write_dashboard(args.dashboard, dashboard)


def _evaluate(
    args: argparse.Namespace,
    dashboard: dict[str, Any],
    label: str,
    data_dir: Path,
    out_dir: Path,
    *,
    max_images: int,
) -> None:
    metrics_path = out_dir / "metrics_summary.json"
    if metrics_path.exists() and not args.force_eval:
        dashboard.setdefault("evaluations", {})[label] = _evaluation_summary(out_dir)
        _mark(dashboard, f"evaluate_{label}", "complete")
        _write_dashboard(args.dashboard, dashboard)
        return
    _mark(dashboard, f"evaluate_{label}", "running")
    log_path = Path(dashboard["paths"]["logs_dir"]) / f"evaluate_{label}.log"
    cmd = _conda_python_cmd() + [
        "scripts/evaluate_validation_unet.py",
        "--checkpoint",
        str(args.out_dir / "best.pt"),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--exact-out-dir",
        "--threshold",
        "0.5",
        "--sweep-thresholds",
        "--max-images",
        str(max_images),
        "--device",
        "cuda:0",
    ]
    _run_logged(cmd, log_path)
    dashboard.setdefault("evaluations", {})[label] = _evaluation_summary(out_dir)
    _mark(dashboard, f"evaluate_{label}", "complete")
    _write_dashboard(args.dashboard, dashboard)


def _write_report(args: argparse.Namespace, dashboard: dict[str, Any]) -> None:
    _mark(dashboard, "report", "running")
    reference = _read_json(args.reference_metrics)
    dashboard["reference_eval"] = {"label": args.reference_label, "metrics": reference}
    report_path = Path(dashboard["paths"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Combined Scaling Report",
        "",
        f"- Old data: `{args.old_data_dir}`",
        f"- New data: `{args.new_data_dir}`",
        f"- Combined data: `{args.combined_data_dir}`",
        f"- Combined model: `{args.out_dir / 'best.pt'}`",
        "",
        f"## {args.reference_label}",
        _metric_line(reference),
        "",
        "## Combined Model Evaluations",
    ]
    for label, summary in dashboard.get("evaluations", {}).items():
        lines.append(f"- `{label}`: {_metric_line(summary.get('metrics_summary', {}))}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dashboard["report"] = {"path": str(report_path)}
    _mark(dashboard, "report", "complete")
    _write_dashboard(args.dashboard, dashboard)


def _metric_line(metrics: dict[str, Any]) -> str:
    sweep = metrics.get("threshold_sweep_best") or {}
    return (
        f"thr0.5 Dice={float(metrics.get('dice', 0.0)):.4f}, IoU={float(metrics.get('iou', 0.0)):.4f}, "
        f"Precision={float(metrics.get('precision', 0.0)):.4f}, Recall={float(metrics.get('recall', 0.0)):.4f}; "
        f"best sweep threshold={sweep.get('threshold')}, Dice={float(sweep.get('dice', 0.0)):.4f}, "
        f"IoU={float(sweep.get('iou', 0.0)):.4f}"
    )


def _link_npz_files(src_dir: Path, dst_dir: Path, prefix: str) -> None:
    for src in sorted(src_dir.glob("*.npz")):
        dst = dst_dir / f"{prefix}_{src.name}"
        dst.symlink_to(src.resolve())


def _dataset_counts(root: Path) -> dict[str, int]:
    return {
        "train": len(list((root / "train").glob("*.npz"))),
        "val": len(list((root / "val").glob("*.npz"))),
    }


def _training_summary(out_dir: Path) -> dict[str, Any]:
    history = out_dir / "history.csv"
    rows = []
    if history.exists():
        import csv

        with history.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    best_row = None
    if rows:
        best_row = max(rows, key=lambda row: _float(row.get("val_dice"), -1.0))
    return {
        "out_dir": str(out_dir),
        "best_checkpoint": str(out_dir / "best.pt"),
        "history": str(history),
        "epochs_completed": len(rows),
        "best_epoch": _row_to_json(best_row),
    }


def _evaluation_summary(out_dir: Path) -> dict[str, Any]:
    return {
        "out_dir": str(out_dir),
        "metrics_summary": _read_json(out_dir / "metrics_summary.json"),
        "threshold_sweep": _read_json(out_dir / "threshold_sweep.json"),
    }


def _row_to_json(row: dict[str, str] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out: dict[str, Any] = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write(f"\n$ {' '.join(cmd)}\nstarted_at_local={_now()}\n".encode("utf-8"))
        log.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"finished_at_local={_now()}\nreturncode={proc.returncode}\n".encode("utf-8"))
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with returncode {proc.returncode}; see {log_path}")


def _mark(dashboard: dict[str, Any], phase: str, status: str) -> None:
    dashboard["phase"] = phase
    dashboard["status"] = "running" if status == "running" else dashboard.get("status", "running")
    dashboard.setdefault("phases", {})[phase] = {"status": status, "updated_at_local": _now()}


def _write_dashboard(path: Path, dashboard: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dashboard["updated_at_local"] = _now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    tmp.replace(path)


def _conda_python_cmd() -> list[str]:
    return ["conda", "run", "-n", "palpation", "python"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
