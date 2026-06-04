from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "perf_curve_center_sphere"
CENTER_SCRIPT = PROJECT_ROOT / "scripts" / "run_center_sphere_newton.py"
H03_SECONDS = 3306.4033203125
H03_WORK_UNITS = 15 * 15 * 96 * 8 * 16
SECONDS_PER_WORK_UNIT = H03_SECONDS / H03_WORK_UNITS
DEFAULT_GPU_RESERVE_MIB = 5120
DEFAULT_WORKER_GPU_MIB = 900
DEFAULT_RAM_RESERVE_MIB = 5120
DEFAULT_WORKER_RAM_MIB = 2800
DEFAULT_LABELS = ("h05_4h_mesh96_g20_t120_s12_i24",)


@dataclass(frozen=True)
class Target:
    label: str
    target_serial_hours: float
    grid: int
    press_steps: int
    substeps: int
    vbd_iterations: int
    cells_x: int = 96
    cells_y: int = 96
    cells_z: int = 32

    @property
    def work_units(self) -> int:
        return int(self.grid * self.grid * self.press_steps * self.substeps * self.vbd_iterations)

    @property
    def estimated_serial_seconds(self) -> float:
        return float(self.work_units * SECONDS_PER_WORK_UNIT)


TARGETS: tuple[Target, ...] = (
    Target("h04_2h_mesh96_g17_t112_s10_i20", 2.0, 17, 112, 10, 20),
    Target("h05_4h_mesh96_g19_t120_s12_i24", 4.0, 19, 120, 12, 24),
    Target("h05_4h_mesh96_g20_t120_s12_i24", 4.0, 20, 120, 12, 24),
    Target("h06_6h_mesh96_g19_t128_s14_i28", 6.0, 19, 128, 14, 28),
    Target("h07_8h_mesh96_g21_t144_s14_i28", 8.0, 21, 144, 14, 28),
    Target("h08_11h_mesh96_g23_t152_s14_i30", 11.0, 23, 152, 14, 30),
    Target("h09_16h_mesh96_g25_t160_s16_i30", 16.0, 25, 160, 16, 30),
    Target("h10_22h_mesh96_g27_t176_s16_i32", 22.0, 27, 176, 16, 32),
    Target("h11_31h_mesh96_g29_t192_s18_i32", 31.0, 29, 192, 18, 32),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the high-precision centered-sphere Neo-Hookean/VBD sweep with row-level parallelism."
    )
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent row workers per target.")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional subset of target labels to run. Defaults to the h05 production parameter set.",
    )
    parser.add_argument("--gpu-reserve-mib", type=int, default=DEFAULT_GPU_RESERVE_MIB)
    parser.add_argument("--worker-gpu-mib", type=int, default=DEFAULT_WORKER_GPU_MIB)
    parser.add_argument("--no-gpu-budget", action="store_true")
    parser.add_argument("--ram-reserve-mib", type=int, default=DEFAULT_RAM_RESERVE_MIB)
    parser.add_argument("--worker-ram-mib", type=int, default=DEFAULT_WORKER_RAM_MIB)
    parser.add_argument("--no-ram-budget", action="store_true")
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.5)
    parser.add_argument("--plan-only", action="store_true", help="Print the planned targets without launching jobs.")
    parser.add_argument("--keep-going", action="store_true", help="Continue to later targets if one target fails.")
    args = parser.parse_args()

    targets = _select_targets(args.labels)
    requested_jobs = max(args.jobs, 1)
    gpu_budget = _gpu_budget(args.gpu_reserve_mib, args.worker_gpu_mib, disabled=args.no_gpu_budget)
    ram_budget = _ram_budget(args.ram_reserve_mib, args.worker_ram_mib, disabled=args.no_ram_budget)
    job_cap = _job_cap(requested_jobs, gpu_budget)
    plan = [_target_plan(target, _effective_jobs(target, job_cap), requested_jobs) for target in targets]
    print(
        json.dumps(
            {
                "requested_jobs": requested_jobs,
                "job_cap_after_gpu_budget": job_cap,
                "gpu_budget": gpu_budget,
                "ram_budget": ram_budget,
                "targets": plan,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.plan_only:
        return

    args.run_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    for target in targets:
        try:
            _run_target(
                target,
                args.run_root,
                job_cap,
                requested_jobs,
                ram_budget,
                max(float(args.launch_stagger_seconds), 0.0),
            )
        except RuntimeError as exc:
            failure = {"label": target.label, "error": str(exc)}
            failures.append(failure)
            print(json.dumps({"target_failed": failure}, ensure_ascii=True), flush=True)
            if not args.keep_going:
                raise

    if failures:
        raise SystemExit(f"{len(failures)} target(s) failed; see logs above")


def _select_targets(labels: Sequence[str] | None) -> list[Target]:
    if not labels:
        labels = DEFAULT_LABELS
    by_label = {target.label: target for target in TARGETS}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise SystemExit(f"Unknown target label(s): {', '.join(missing)}")
    return [by_label[label] for label in labels]


def _run_target(
    target: Target,
    run_root: Path,
    job_cap: int,
    requested_jobs: int,
    ram_budget: dict[str, object],
    launch_stagger_seconds: float,
) -> None:
    started = time.perf_counter()
    out_dir = run_root / f"{_timestamp()}-{target.label}"
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    jobs = _effective_jobs(target, job_cap)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at_local": datetime.now().isoformat(timespec="seconds"),
        "target": _target_plan(target, jobs, requested_jobs),
        "out_dir": str(out_dir),
        "logs_dir": str(logs_dir),
    }
    _write_json(out_dir / "parallel_run_summary.json", manifest)

    print(f"[{target.label}] initializing {out_dir}", flush=True)
    init_cmd = _base_cmd(target, out_dir) + ["--init-only", "--no-features"]
    _run_logged(init_cmd, logs_dir / "init.log")

    partitions = _row_partitions(target.grid, jobs)
    manifest["row_partitions"] = [{"row_start": start, "row_end": end} for start, end in partitions]
    _write_json(out_dir / "parallel_run_summary.json", manifest)

    worker_failures: list[dict[str, object]] = []
    active: list[subprocess.Popen[bytes]] = []
    worker_logs: dict[int, object] = {}
    worker_meta: dict[int, dict[str, object]] = {}
    for idx, (start, end) in enumerate(partitions):
        _wait_for_ram_budget(ram_budget, target.label, idx)
        log_path = logs_dir / f"worker_{idx:02d}_rows_{start:03d}_{end:03d}.log"
        cmd = _base_cmd(target, out_dir) + [
            "--resume",
            "--no-assemble",
            "--no-features",
            "--row-start",
            str(start),
            "--row-end",
            str(end),
            "--row-chunk-size",
            "1",
        ]
        log = log_path.open("ab")
        log.write(_log_header(cmd).encode("utf-8"))
        log.flush()
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        active.append(proc)
        worker_logs[proc.pid] = log
        worker_meta[proc.pid] = {
            "worker_index": idx,
            "row_start": start,
            "row_end": end,
            "pid": proc.pid,
            "log": str(log_path),
            "command": cmd,
        }
        print(f"[{target.label}] worker {idx:02d} pid={proc.pid} rows {start}:{end}", flush=True)
        if launch_stagger_seconds > 0.0 and idx + 1 < len(partitions):
            time.sleep(launch_stagger_seconds)

    try:
        for proc in active:
            rc = proc.wait()
            log = worker_logs.pop(proc.pid)
            log.close()
            meta = worker_meta[proc.pid]
            meta["returncode"] = rc
            if rc != 0:
                worker_failures.append(meta)
    finally:
        for log in worker_logs.values():
            log.close()

    if worker_failures:
        manifest["status"] = "worker_failed"
        manifest["worker_failures"] = worker_failures
        manifest["finished_at_local"] = datetime.now().isoformat(timespec="seconds")
        manifest["parallel_elapsed_seconds"] = float(time.perf_counter() - started)
        _write_json(out_dir / "parallel_run_summary.json", manifest)
        raise RuntimeError(f"{target.label} had {len(worker_failures)} failed worker(s)")

    print(f"[{target.label}] assembling", flush=True)
    assemble_cmd = _base_cmd(target, out_dir) + ["--resume", "--assemble-only", "--no-features"]
    _run_logged(assemble_cmd, logs_dir / "assemble.log")

    manifest["status"] = "complete"
    manifest["finished_at_local"] = datetime.now().isoformat(timespec="seconds")
    manifest["parallel_elapsed_seconds"] = float(time.perf_counter() - started)
    manifest["outputs"] = {
        "npz": str(out_dir / "center_sphere_newton_sample.npz"),
        "metadata": str(out_dir / "metadata.json"),
        "summary": str(out_dir / "summary.json"),
        "curve_summary": str(out_dir / "curve_summary.csv"),
    }
    _copy_summary_metrics(out_dir, manifest)
    _write_json(out_dir / "parallel_run_summary.json", manifest)
    print(f"[{target.label}] complete in {manifest['parallel_elapsed_seconds']:.1f}s", flush=True)


def _base_cmd(target: Target, out_dir: Path) -> list[str]:
    return _python_cmd() + [
        str(CENTER_SCRIPT),
        "--out-dir",
        str(out_dir),
        "--cells-x",
        str(target.cells_x),
        "--cells-y",
        str(target.cells_y),
        "--cells-z",
        str(target.cells_z),
        "--grid-h",
        str(target.grid),
        "--grid-w",
        str(target.grid),
        "--press-steps",
        str(target.press_steps),
        "--substeps",
        str(target.substeps),
        "--vbd-iterations",
        str(target.vbd_iterations),
    ]


def _python_cmd() -> list[str]:
    executable = Path(sys.executable)
    if os.environ.get("CONDA_DEFAULT_ENV") == "palpation" or "palpation" in executable.parts:
        return [str(executable)]
    conda = shutil.which("conda") or "/opt/anaconda3/bin/conda"
    return [conda, "run", "--no-capture-output", "-n", "palpation", "python"]


def _run_logged(cmd: Sequence[str], log_path: Path) -> None:
    with log_path.open("ab") as log:
        log.write(_log_header(cmd).encode("utf-8"))
        log.flush()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with rc={result.returncode}; log={log_path}")


def _log_header(cmd: Sequence[str]) -> str:
    return (
        f"\n--- started {datetime.now().isoformat(timespec='seconds')} ---\n"
        f"cwd: {PROJECT_ROOT}\n"
        f"cmd: {' '.join(str(part) for part in cmd)}\n\n"
    )


def _row_partitions(rows: int, jobs: int) -> list[tuple[int, int]]:
    jobs = max(1, min(int(jobs), int(rows)))
    base = rows // jobs
    extra = rows % jobs
    partitions: list[tuple[int, int]] = []
    start = 0
    for idx in range(jobs):
        width = base + (1 if idx < extra else 0)
        end = start + width
        if start < end:
            partitions.append((start, end))
        start = end
    return partitions


def _target_plan(target: Target, jobs: int, requested_jobs: int) -> dict[str, object]:
    return {
        **asdict(target),
        "requested_jobs": requested_jobs,
        "active_jobs": jobs,
        "work_units": target.work_units,
        "estimated_serial_seconds_from_h03": target.estimated_serial_seconds,
        "estimated_serial_hours_from_h03": target.estimated_serial_seconds / 3600.0,
        "estimated_parallel_hours_from_h03": target.estimated_serial_seconds / 3600.0 / max(jobs, 1),
    }


def _effective_jobs(target: Target, job_cap: int) -> int:
    return max(1, min(int(job_cap), int(target.grid)))


def _job_cap(requested_jobs: int, *budgets: dict[str, object]) -> int:
    cap = max(1, int(requested_jobs))
    for budget in budgets:
        max_workers = budget.get("max_worker_jobs")
        if isinstance(max_workers, int) and max_workers > 0:
            cap = min(cap, max_workers)
    return max(1, cap)


def _gpu_budget(reserve_mib: int, worker_mib: int, *, disabled: bool) -> dict[str, object]:
    reserve_mib = max(int(reserve_mib), 0)
    worker_mib = max(int(worker_mib), 1)
    budget: dict[str, object] = {
        "enabled": not disabled,
        "reserve_mib": reserve_mib,
        "worker_gpu_mib_assumption": worker_mib,
    }
    if disabled:
        return budget

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        row = result.stdout.strip().splitlines()[0]
        total_raw, used_raw = [part.strip() for part in row.split(",", maxsplit=1)]
        total_mib = int(float(total_raw))
        baseline_used_mib = int(float(used_raw))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        budget["query_error"] = str(exc)
        return budget

    available_for_workers_mib = max(total_mib - reserve_mib - baseline_used_mib, 0)
    budget.update(
        {
            "total_mib": total_mib,
            "baseline_used_mib": baseline_used_mib,
            "available_for_workers_mib": available_for_workers_mib,
            "max_worker_jobs": max(1, available_for_workers_mib // worker_mib),
        }
    )
    return budget


def _ram_budget(reserve_mib: int, worker_mib: int, *, disabled: bool) -> dict[str, object]:
    reserve_mib = max(int(reserve_mib), 0)
    worker_mib = max(int(worker_mib), 1)
    budget: dict[str, object] = {
        "enabled": not disabled,
        "reserve_mib": reserve_mib,
        "worker_ram_mib_assumption": worker_mib,
    }
    if disabled:
        return budget

    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        values: dict[str, int] = {}
        for line in meminfo.splitlines():
            key, rest = line.split(":", maxsplit=1)
            raw_value = rest.strip().split()[0]
            values[key] = int(raw_value) // 1024
        total_mib = values["MemTotal"]
        available_mib = values["MemAvailable"]
    except (OSError, KeyError, ValueError) as exc:
        budget["query_error"] = str(exc)
        return budget

    available_for_workers_mib = max(available_mib - reserve_mib, 0)
    budget.update(
        {
            "total_mib": total_mib,
            "available_mib": available_mib,
            "available_for_workers_mib": available_for_workers_mib,
            "max_worker_jobs": max(1, available_for_workers_mib // worker_mib),
        }
    )
    return budget


def _wait_for_ram_budget(ram_budget: dict[str, object], label: str, worker_index: int) -> None:
    if not ram_budget.get("enabled", False):
        return
    reserve_mib = int(ram_budget.get("reserve_mib", 0))
    worker_mib = int(ram_budget.get("worker_ram_mib_assumption", 1))
    required_available_mib = reserve_mib + worker_mib
    last_notice = 0.0
    while True:
        available_mib = _mem_available_mib()
        if available_mib is None or available_mib >= required_available_mib:
            return
        now = time.time()
        if now - last_notice >= 30.0:
            print(
                f"[{label}] waiting for RAM before worker {worker_index:02d}: "
                f"available={available_mib}MiB required={required_available_mib}MiB",
                flush=True,
            )
            last_notice = now
        time.sleep(5.0)


def _mem_available_mib() -> int | None:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        return None
    return None


def _copy_summary_metrics(out_dir: Path, manifest: dict[str, object]) -> None:
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    keys = [
        "curve_shape",
        "negative_step_fraction",
        "monotone_curve_fraction",
        "median_convex_fraction",
        "peak_force_median_n",
        "peak_force_max_n",
        "chunk_count",
        "total_chunk_elapsed_seconds",
    ]
    manifest["assembled_summary_metrics"] = {key: summary.get(key) for key in keys}


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    main()
