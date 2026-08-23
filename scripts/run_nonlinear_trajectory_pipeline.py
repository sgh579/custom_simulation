from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate nonlinear-trajectory palpation data and run segmentation sweeps.")
    parser.add_argument("--package-dir", type=Path, default=Path("data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/random_trajectory"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/nonlinear_trajectory_20x_repeats10_seed20260618"))
    parser.add_argument("--num-train-phantoms", type=int, default=100)
    parser.add_argument("--num-val-phantoms", type=int, default=40)
    parser.add_argument("--num-test-phantoms", type=int, default=40)
    parser.add_argument("--trajectory-repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--backend", choices=["newton", "analytic"], default="newton")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resolutions", type=str, default="128")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trajectory-input-steps", type=int, default=0)
    parser.add_argument("--positional-embedding-dim", type=int, default=8)
    parser.add_argument("--monitor-interval", type=float, default=30.0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.package_dir = args.package_dir.with_name(args.package_dir.name + "_smoke")
        args.run_dir = args.run_dir.with_name(args.run_dir.name + "_smoke")
        args.num_train_phantoms = min(args.num_train_phantoms, 1)
        args.num_val_phantoms = min(args.num_val_phantoms, 1)
        args.num_test_phantoms = min(args.num_test_phantoms, 1)
        args.trajectory_repeats = min(args.trajectory_repeats, 2)
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.package_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dashboard = Dashboard(args.run_dir / "dashboard.json", args)
    dashboard.update(status="running", phase="init")

    try:
        if not args.skip_generation:
            run_generation(args, dashboard, logs_dir, data_dir)
        validate_counts(args, dashboard, data_dir)
        if not args.skip_training:
            run_training(args, dashboard, logs_dir)
        dashboard.update(status="complete", phase="complete", completed_at=utc_now(), metrics=collect_metrics(args))
    except Exception as exc:
        dashboard.update(status="failed", phase=dashboard.state.get("phase", "unknown"), error=repr(exc), updated_at=utc_now())
        raise


def run_generation(args: argparse.Namespace, dashboard: "Dashboard", logs_dir: Path, data_dir: Path) -> None:
    expected = expected_counts(args)
    dashboard.update(
        phase="generation",
        expected_samples=expected,
        completed_samples=count_samples(data_dir),
        generation_started_at=utc_now(),
    )
    processes: list[tuple[int, subprocess.Popen[bytes], Any, Path, list[str]]] = []
    for worker_index in range(args.worker_count):
        cmd = generation_command(args, data_dir, worker_index)
        log_path = logs_dir / f"generation_worker_{worker_index:02d}.log"
        log_file = log_path.open("ab")
        log_file.write(("\n\n=== generation worker started " + utc_now() + " ===\n").encode("utf-8"))
        log_file.write((" ".join(cmd) + "\n").encode("utf-8"))
        log_file.flush()
        process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((worker_index, process, log_file, log_path, cmd))

    while processes:
        time.sleep(max(float(args.monitor_interval), 1.0))
        still_running: list[tuple[int, subprocess.Popen[bytes], Any, Path, list[str]]] = []
        workers = []
        for worker_index, process, log_file, log_path, cmd in processes:
            returncode = process.poll()
            workers.append(
                {
                    "worker_index": worker_index,
                    "returncode": returncode,
                    "log": str(log_path),
                    "command": cmd,
                }
            )
            if returncode is None:
                still_running.append((worker_index, process, log_file, log_path, cmd))
            else:
                log_file.write((f"=== generation worker finished {utc_now()} rc={returncode} ===\n").encode("utf-8"))
                log_file.close()
                if returncode != 0:
                    dashboard.update(
                        phase="generation",
                        completed_samples=count_samples(data_dir),
                        workers=workers,
                        updated_at=utc_now(),
                    )
                    raise RuntimeError(f"generation worker {worker_index} failed with return code {returncode}; see {log_path}")
        processes = still_running
        dashboard.update(
            phase="generation",
            completed_samples=count_samples(data_dir),
            workers=workers,
            updated_at=utc_now(),
        )

    dashboard.update(phase="generation_complete", completed_samples=count_samples(data_dir), generation_finished_at=utc_now())


def run_training(args: argparse.Namespace, dashboard: "Dashboard", logs_dir: Path) -> None:
    baseline_dir = args.run_dir / "highres_wrench_segmentation_sweep"
    temporal_dir = args.run_dir / "highres_wrench_temporal_variants"
    wrench_cmd = [
        sys.executable,
        "scripts/run_highres_segmentation_sweep.py",
        "--package-dir",
        str(args.package_dir),
        "--out-dir",
        str(baseline_dir),
        "--resolutions",
        args.resolutions,
        "--inputs",
        "wrench",
        "--models",
        "mlp,shallow_cnn,unet",
        "--seed",
        str(args.seed),
        "--stiffness-random-seed",
        str(args.seed + 17),
        "--limited-trajectory-seed",
        str(args.seed + 31),
        "--trajectory-input-steps",
        str(args.trajectory_input_steps),
        "--positional-embedding-dim",
        str(args.positional_embedding_dim),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]
    wrench_variants = ",".join(
        [
            "wrench_unet_aug_focal",
            "wrench_temporal_cnn16_unet",
            "wrench_temporal_cnn32_unet",
            "wrench_temporal_gru32_unet",
            "wrench_temporal_attention32_unet",
            "wrench_temporal_multiscale32_unet",
        ]
    )
    temporal_cmd = [
        sys.executable,
        "scripts/run_highres_fz_temporal_variant_sweep.py",
        "--package-dir",
        str(args.package_dir),
        "--out-dir",
        str(temporal_dir),
        "--baseline-sweep-dir",
        str(baseline_dir),
        "--resolutions",
        args.resolutions,
        "--variants",
        wrench_variants,
        "--seed",
        str(args.seed),
        "--limited-trajectory-seed",
        str(args.seed + 31),
        "--trajectory-input-steps",
        str(args.trajectory_input_steps),
        "--positional-embedding-dim",
        str(args.positional_embedding_dim),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]
    if args.smoke:
        wrench_cmd.append("--smoke")
        temporal_cmd.append("--smoke")

    for phase, cmd in (
        ("train_wrench_methods", wrench_cmd),
        ("train_wrench_temporal_methods", temporal_cmd),
    ):
        dashboard.update(phase=phase, command=cmd, completed_samples=count_samples(args.package_dir / "data"), updated_at=utc_now())
        run_logged(cmd, logs_dir / f"{phase}.log")
        dashboard.update(phase=f"{phase}_complete", metrics=collect_metrics(args), updated_at=utc_now())


def generation_command(args: argparse.Namespace, data_dir: Path, worker_index: int) -> list[str]:
    return [
        sys.executable,
        "scripts/generate_nonlinear_trajectory_dataset.py",
        "--backend",
        args.backend,
        "--out-dir",
        str(data_dir),
        "--num-train-phantoms",
        str(args.num_train_phantoms),
        "--num-val-phantoms",
        str(args.num_val_phantoms),
        "--num-test-phantoms",
        str(args.num_test_phantoms),
        "--trajectory-repeats",
        str(args.trajectory_repeats),
        "--seed",
        str(args.seed),
        "--resume",
        "--worker-count",
        str(args.worker_count),
        "--worker-index",
        str(worker_index),
        "--grid-h",
        "20",
        "--grid-w",
        "20",
        "--press-steps",
        "20",
        "--max-indentation",
        "0.018",
        "--substeps-per-depth",
        "3",
        "--vbd-iterations",
        "10",
        "--cells-x",
        "48",
        "--cells-y",
        "48",
        "--cells-z",
        "16",
        "--lumps-min",
        "1",
        "--lumps-max",
        "4",
        "--lump-shapes",
        "sphere,ellipsoid,box,cylinder,capsule",
        "--lump-stiffness-multiplier",
        "20.0",
        "--no-save-phantom-3d",
        "--no-save-press-records",
        "--no-save-scan-animation",
        "--device",
        args.device,
    ]


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        log_file.write(("\n\n=== command started " + utc_now() + " ===\n").encode("utf-8"))
        log_file.write((" ".join(cmd) + "\n").encode("utf-8"))
        log_file.flush()
        process = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        returncode = process.wait()
        log_file.write((f"=== command finished {utc_now()} rc={returncode} ===\n").encode("utf-8"))
        if returncode != 0:
            raise RuntimeError(f"command failed with return code {returncode}; see {log_path}")


def validate_counts(args: argparse.Namespace, dashboard: "Dashboard", data_dir: Path) -> None:
    expected = expected_counts(args)
    counts = count_samples(data_dir)
    dashboard.update(phase="validate_dataset", expected_samples=expected, completed_samples=counts, updated_at=utc_now())
    missing = {split: int(expected[split] - counts.get(split, 0)) for split in expected if counts.get(split, 0) != expected[split]}
    if missing:
        raise RuntimeError(f"dataset incomplete: {missing}")


def expected_counts(args: argparse.Namespace) -> dict[str, int]:
    return {
        "train": int(args.num_train_phantoms) * int(args.trajectory_repeats),
        "val": int(args.num_val_phantoms) * int(args.trajectory_repeats),
        "test": int(args.num_test_phantoms) * int(args.trajectory_repeats),
    }


def count_samples(data_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            counts[split] = 0
        else:
            counts[split] = sum(1 for path in split_dir.glob("*.npz") if not path.name.startswith("."))
    return counts


def collect_metrics(args: argparse.Namespace) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    baseline_path = args.run_dir / "highres_wrench_segmentation_sweep" / "leaderboard.csv"
    temporal_path = args.run_dir / "highres_wrench_temporal_variants" / "leaderboard_with_baselines.csv"
    if baseline_path.exists():
        metrics["wrench_baseline_leaderboard"] = str(baseline_path)
        metrics["best_wrench_baseline"] = best_row(baseline_path)
    if temporal_path.exists():
        metrics["wrench_temporal_leaderboard_with_baselines"] = str(temporal_path)
        metrics["best_wrench_temporal_or_baseline"] = best_row(temporal_path)
    return metrics


def best_row(path: Path) -> dict[str, str] | None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("test_val_selected_dice") or row.get("val_best_dice") or 0.0))


class Dashboard:
    def __init__(self, path: Path, args: argparse.Namespace) -> None:
        self.path = path
        self.state: dict[str, Any] = {}
        if path.exists():
            try:
                self.state.update(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        self.state.update({
            "status": "starting",
            "phase": "init",
            "started_at": self.state.get("started_at", utc_now()),
            "updated_at": utc_now(),
            "project_root": str(PROJECT_ROOT),
            "package_dir": str(args.package_dir),
            "run_dir": str(args.run_dir),
            "config": jsonable(vars(args)),
        })

    def update(self, **kwargs: Any) -> None:
        self.state.update(kwargs)
        self.state["updated_at"] = utc_now()
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
