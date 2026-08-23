from __future__ import annotations

import re
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


CONDA_ENV_NAME = "palpation"
DEFAULT_NEWTON_ROOT = Path(os.environ.get("PALPATION_NEWTON_ROOT", "/home/guoheng/newton")).expanduser()
REQUIRED_NEWTON_DEVICE = "cuda:0"
SIMULATOR_ENTRYPOINT = "palpation_sim.newton_vbd.NewtonVBDPalpationSimulator"
DATA_CONTRACT_VERSION = 2
SAMPLE_METADATA_SCHEMA_VERSION = 3
DATASET_METADATA_SCHEMA_VERSION = 2
DATA_CONTRACT_PATH = "docs/data_contract.md"
METADATA_TEMPLATE_PATH = "docs/metadata_template.json"
RUN_DIR_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
RUN_DIR_TIMESTAMP_DESCRIPTION = "yyyymmdd-hhmmss"
_RUN_DIR_PREFIX_RE = re.compile(r"^(\d{8}-\d{6})(?:-|$)")


def require_runtime_environment(*, require_newton: bool = False, newton_root: str | Path | None = None) -> None:
    """Fail early when a workflow script is launched outside the pinned runtime."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    executable = Path(sys.executable)
    executable_looks_right = any(part == CONDA_ENV_NAME for part in executable.parts)
    if conda_env != CONDA_ENV_NAME and not executable_looks_right:
        raise RuntimeError(
            "This workflow is pinned to the conda environment 'palpation'. "
            "Run `conda activate palpation` first, or use `conda run -n palpation python ...`."
        )

    if require_newton:
        root = Path(newton_root) if newton_root is not None else DEFAULT_NEWTON_ROOT
        root = root.expanduser()
        if root != DEFAULT_NEWTON_ROOT:
            raise RuntimeError(f"This workflow is pinned to Newton root {DEFAULT_NEWTON_ROOT}; got {root}")
        if not root.exists():
            raise RuntimeError(f"Newton root not found: {root}")


def runtime_metadata(*, newton_root: str | Path | None = None, device: str | None = None) -> dict[str, object]:
    root = Path(newton_root) if newton_root is not None else DEFAULT_NEWTON_ROOT
    return {
        "python": {
            "environment_manager": "conda",
            "environment_name": CONDA_ENV_NAME,
            "executable": sys.executable,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "newton": {
            "root": str(root.expanduser()),
            "simulator_entrypoint": SIMULATOR_ENTRYPOINT,
        },
        "device": {
            "required": REQUIRED_NEWTON_DEVICE,
            "requested": device or REQUIRED_NEWTON_DEVICE,
        },
    }


def metadata_contract() -> dict[str, object]:
    return {
        "version": DATA_CONTRACT_VERSION,
        "definition": DATA_CONTRACT_PATH,
        "template": METADATA_TEMPLATE_PATH,
    }


def run_date_prefix() -> str:
    return datetime.now().strftime(RUN_DIR_TIMESTAMP_FORMAT)


def with_run_date_prefix(out_dir: str | Path, *, enabled: bool = True) -> Path:
    """Prefix new output directories under runs/ with yyyymmdd-hhmmss."""
    path = Path(out_dir)
    if not enabled or _RUN_DIR_PREFIX_RE.match(path.name):
        return path
    if "runs" not in path.parts:
        return path
    if path.name == "runs":
        return path / run_date_prefix()
    return path.with_name(f"{run_date_prefix()}-{path.name}")


def run_output_metadata(out_dir: str | Path) -> dict[str, object]:
    path = Path(out_dir)
    prefixed_directory = None
    date_prefix = None
    for part in reversed(path.parts):
        match = _RUN_DIR_PREFIX_RE.match(part)
        if match:
            prefixed_directory = part
            date_prefix = match.group(1)
            break
    return {
        "output_dir": str(path),
        "directory_name": path.name,
        "date_prefix": date_prefix,
        "date_prefixed_directory": prefixed_directory,
        "date_prefix_format": RUN_DIR_TIMESTAMP_DESCRIPTION,
    }


def disk_usage_metadata(root: str | Path | list[str | Path | None] | tuple[str | Path | None, ...]) -> dict[str, object]:
    """Return file-byte and allocated-byte usage for a saved output root."""
    roots = [Path(root)] if isinstance(root, (str, Path)) else [Path(path) for path in root if path is not None]
    files: list[tuple[Path, Path]] = []
    for root_path in roots:
        if not root_path.exists():
            continue
        if root_path.is_file():
            files.append((root_path, root_path))
        else:
            files.extend((root_path, path) for path in sorted(root_path.rglob("*")) if path.is_file())

    total_bytes = 0
    total_allocated_bytes = 0
    entries: dict[str, object] = {}
    for root_path, path in files:
        stat = path.stat()
        size_bytes = int(stat.st_size)
        allocated_bytes = int(getattr(stat, "st_blocks", 0)) * 512
        if allocated_bytes <= 0:
            allocated_bytes = size_bytes
        total_bytes += size_bytes
        total_allocated_bytes += allocated_bytes
        if len(roots) == 1:
            name = path.name if root_path.is_file() else str(path.relative_to(root_path))
        else:
            name = str(path)
        entries[name] = {
            "bytes": size_bytes,
            "allocated_bytes": allocated_bytes,
        }

    return {
        "root": str(roots[0]) if len(roots) == 1 else None,
        "roots": [str(path) for path in roots],
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_allocated_bytes": total_allocated_bytes,
        "files": entries,
    }


class ResourceMonitor:
    """Track wall time, nvidia-smi memory.used, and final storage usage."""

    def __init__(self, *, device: str | None = None, poll_interval_seconds: float = 0.2) -> None:
        self.device = device
        self.poll_interval_seconds = max(float(poll_interval_seconds), 0.05)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_perf = 0.0
        self._finished_perf: float | None = None
        self._started_at = ""
        self._finished_at = ""
        self._before_bytes: int | None = None
        self._after_bytes: int | None = None
        self._peak_bytes: int | None = None
        self._samples = 0
        self._query_error: str | None = None

    def start(self) -> "ResourceMonitor":
        self._started_perf = time.perf_counter()
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._before_bytes = self._query_gpu_memory_bytes()
        self._peak_bytes = self._before_bytes
        if self._before_bytes is not None:
            self._thread = threading.Thread(target=self._poll_gpu_memory, daemon=True)
            self._thread.start()
        return self

    def elapsed_seconds(self) -> float:
        end = self._finished_perf if self._finished_perf is not None else time.perf_counter()
        return float(end - self._started_perf)

    def finish(self, *, storage_root: str | Path | None = None) -> dict[str, object]:
        if self._finished_perf is None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=max(1.0, self.poll_interval_seconds * 2.0))
            self._after_bytes = self._query_gpu_memory_bytes()
            if self._after_bytes is not None:
                self._peak_bytes = self._max_optional(self._peak_bytes, self._after_bytes)
            self._finished_perf = time.perf_counter()
            self._finished_at = datetime.now().isoformat(timespec="seconds")

        metadata: dict[str, object] = {
            "elapsed_seconds": self.elapsed_seconds(),
            "started_at_local": self._started_at,
            "finished_at_local": self._finished_at,
            "gpu_memory": self._gpu_memory_metadata(),
        }
        if storage_root is not None:
            metadata["disk_usage"] = disk_usage_metadata(storage_root)
        return metadata

    def _poll_gpu_memory(self) -> None:
        while not self._stop.wait(self.poll_interval_seconds):
            used = self._query_gpu_memory_bytes()
            if used is None:
                continue
            self._samples += 1
            self._peak_bytes = self._max_optional(self._peak_bytes, used)

    def _gpu_memory_metadata(self) -> dict[str, object]:
        available = self._before_bytes is not None or self._after_bytes is not None or self._peak_bytes is not None
        peak_delta = None
        if self._before_bytes is not None and self._peak_bytes is not None:
            peak_delta = max(int(self._peak_bytes) - int(self._before_bytes), 0)
        return {
            "available": available,
            "device": self.device,
            "query_method": "nvidia-smi memory.used polling",
            "unit": "bytes",
            "before_used_bytes": self._before_bytes,
            "peak_observed_used_bytes": self._peak_bytes,
            "after_used_bytes": self._after_bytes,
            "peak_delta_from_start_bytes": peak_delta,
            "poll_interval_seconds": self.poll_interval_seconds,
            "sample_count": self._samples,
            "query_error": self._query_error,
            "note": "memory.used is full-device usage reported by nvidia-smi and can include other processes.",
        }

    def _query_gpu_memory_bytes(self) -> int | None:
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        index = _cuda_device_index(self.device)
        if index is not None:
            cmd.insert(1, f"--id={index}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2.0)
        except (OSError, subprocess.SubprocessError) as exc:
            self._query_error = str(exc)
            return None
        raw = result.stdout.strip().splitlines()
        if not raw:
            self._query_error = "nvidia-smi returned no memory.used rows"
            return None
        try:
            mib = int(float(raw[0].strip()))
        except ValueError:
            self._query_error = f"could not parse nvidia-smi memory.used row: {raw[0]!r}"
            return None
        return mib * 1024 * 1024

    @staticmethod
    def _max_optional(left: int | None, right: int | None) -> int | None:
        if left is None:
            return right
        if right is None:
            return left
        return max(left, right)


def _cuda_device_index(device: str | None) -> int | None:
    if device is None:
        return None
    text = str(device).strip().lower()
    if not text.startswith("cuda:"):
        return None
    try:
        return int(text.split(":", 1)[1])
    except ValueError:
        return None


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def resolve_required_torch_cuda_device(torch_module: Any, requested: str | None = "cuda") -> Any:
    device_name = "cuda" if requested is None or str(requested).strip() == "" else str(requested).strip()
    if device_name.lower() == "auto":
        raise RuntimeError("Device 'auto' is disabled. Use a CUDA device such as 'cuda' or 'cuda:0'.")
    if device_name == "cuda":
        device_name = REQUIRED_NEWTON_DEVICE
    if device_name != REQUIRED_NEWTON_DEVICE:
        raise RuntimeError(f"This workflow is pinned to CUDA device '{REQUIRED_NEWTON_DEVICE}'; got '{device_name}'.")
    if not device_name.startswith("cuda"):
        raise RuntimeError(f"This workflow requires a CUDA GPU device; got '{device_name}'.")
    if not torch_module.cuda.is_available():
        raise RuntimeError(f"CUDA is not available, but this workflow requires '{device_name}'.")
    if ":" in device_name:
        try:
            index = int(device_name.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDA device name: {device_name}") from exc
        if index >= int(torch_module.cuda.device_count()):
            raise RuntimeError(
                f"Requested CUDA device '{device_name}' is not visible. "
                f"torch.cuda.device_count()={torch_module.cuda.device_count()}"
            )
    return torch_module.device(device_name)
